"""
model.py — MSEGraphLanguageModel orchestrator.

Two inference modes, one set of graphs -- there is no separate
"experience" data source anymore:
  strict  — InferenceEngine(E, B, R, mode="strict")
            candidates gated to literal Edge/Bridge successors,
            two-stage lineage-vote pipeline (default)
  open    — InferenceEngine(E, B, R, mode="open", vocab=<all tokens>)
            candidates are the ENTIRE vocabulary every step, scored
            by IVM's Q1/Q2 scores (see ivm.py) plus noise.py's noise-cancellation
            score, filtered by the Q3/Q4 legality checks

Both engines are built automatically as soon as the model is trained
or loaded -- there is no separate "build Open Mode" step anymore.
"""

import json
import os

from tokenizer import split_sentences, normalize, CharWordTokenizer
from graph import EdgeMatrix, BridgeMatrix, RelationshipMatrix
from inference import InferenceEngine
from config import (TokenizerConfig, GenerationConfig, CTMConfig, InterpretConfig,
                    NoiseConfig, PAD, UNK, BOS, EOS)


class MSEGraphLanguageModel:

    def __init__(self, vocab_size=TokenizerConfig.DEFAULT_VOCAB_SIZE):
        self.tokenizer   = CharWordTokenizer(vocab_size=vocab_size)
        self.edges       = EdgeMatrix()
        self.bridges     = BridgeMatrix()
        self.rels        = RelationshipMatrix()
        self._strict     = None   # InferenceEngine, mode="strict"
        self._open       = None   # InferenceEngine, mode="open" -- auto-built alongside _strict
        self.ctm         = None   # ContextTriggerMatrix — None until built
        self.ivm         = None   # ImportanceVoteMatrix — Strict Mode legacy tie-break, opt-in
        self.open_ctm    = None   # ImportanceVoteMatrix — Open Mode PRIMARY mechanism, auto-built
        self.noise_index = None  # NoiseCancellationIndex — see noise.py
        self.token_vocab = None  # TokenVocabularyMatrix — permanent per-token vocab rows, see noise.py

    # ==== training ==============================================================

    def train(self, corpus):
        """
        Segments each sentence exactly ONCE and reuses that same word
        list for both vocabulary counting and encoding -- segment()
        (tokenizer.py's punctuation-isolating regex pass) was
        previously paid twice per sentence: once inside
        tokenizer.train()'s own word-frequency count, and again inside
        encode_for_training() right below it, re-segmenting the exact
        same text. Profiling on a real corpus showed segment()'s regex
        substitutions as the single largest remaining cost after noise's
        eager build was fixed (see config.NoiseConfig.EAGER_BUILD) --
        this halves that cost outright rather than tuning it.
        """
        from collections import Counter
        from tokenizer import segment

        sentences = split_sentences(corpus)
        segmented = [segment(s) for s in sentences]
        word_freq = Counter()
        for words in segmented:
            for w in words:
                word_freq[w] += 1
        self.tokenizer._train_from_word_freq(word_freq)
        seqs = [self.tokenizer.encode_for_training_words(words) for words in segmented]
        self._build_graphs(seqs)

    def train_from_file(self, path):
        """
        Delegates to train() on the file's full contents. The old
        separate streamed-word-count pass here bought nothing in
        practice: this method already had to read the whole file into
        memory anyway for the encoding pass right after it, so
        streaming just the FIRST pass never actually avoided holding
        the full text in memory -- it just paid segment() twice on it
        instead of once (see train()'s docstring). Genuinely streamed,
        never-fully-materialized training across many files is still
        available -- see train_corpus.py, which uses
        tokenizer.stream_word_freq() directly for exactly that case.
        """
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        self.train(text)

    def train_incremental(self, corpus, extend_vocab=False, target_vocab_size=None):
        """
        Add a new corpus to an ALREADY-TRAINED model, without discarding
        what it already knows.

        By default the tokenizer stays frozen: new text is encoded with
        the existing vocabulary (falling back to <UNK> for genuinely
        novel characters, same as encoding any out-of-vocabulary text),
        so every previously-built triple stays valid unchanged. Pass
        extend_vocab=True (with target_vocab_size > current vocab_size)
        to also grow the vocabulary from this new corpus first -- see
        CharWordTokenizer.extend_vocab for what that does and doesn't
        guarantee.

        The Edge, Bridge, and Relationship matrices are then rebuilt
        from the UNION of old + new structural facts (not just old
        facts plus new ones appended side by side): clusters are
        recomputed from scratch over the merged triple set, because a
        cluster can only form once you know about ALL the triples that
        might belong to it -- e.g. two sentences that each independently
        arrived in a different training call ("the cat sat on the mat"
        now, "the dog sat on the carpet" later) only form a real
        bridge-axis cluster once both exist together. Existing
        relationship_ids (one per unique training sentence) are
        preserved for any sentence whose exact content isn't a repeat
        of something in the new corpus (only their triple_id
        references are remapped for the merged Bridge Matrix's new
        row order); if a NEW sentence is an exact literal repeat of
        an EXISTING one, it now collapses onto that existing
        relationship_id instead of getting a new one -- the same
        dedup-by-content rule RelationshipMatrix.build() applies to a
        from-scratch train(), now applied consistently across an
        incremental merge too (see graph.py's RelationshipMatrix
        docstring). Genuinely new sentences get new relationship_ids
        continuing after the last existing one. rel_count is merged
        accordingly: a sentence that already occurred N times before
        this call and appears M more times in the new corpus ends up
        with rel_count = N + M under the SAME relationship_id, not a
        second one. One known limitation: a pre-existing
        relationship_id whose sentence was too short to have any
        triples at all (< 3 tokens) can't be reconstructed from the
        Bridge Matrix alone (there's nothing there to reconstruct
        from); every such degenerate sentence collapses onto one
        shared relationship_id across a merge, even if their original
        literal contents actually differed -- harmless in practice,
        since a triple-less sentence casts no votes and drives no
        evidence anywhere else in this codebase, but worth knowing if
        you're inspecting rel_count directly on a corpus with very
        short sentences.

        Open Mode (self._open/self.open_ctm) is automatically rebuilt
        from the merged graphs -- no separate call needed. The Context
        Trigger Matrix and Strict Mode's legacy Importance Vote Matrix
        (self.ctm/self.ivm) are NOT auto-rebuilt (both are opt-in
        add-ons, not part of the default pipeline) -- they're
        invalidated instead, since both were derived from the
        pre-merge structure; call build_context_triggers()/
        build_importance_votes() again if you were using either.

        Returns a summary dict of before/after counts.
        """
        if self._strict is None:
            raise RuntimeError(
                "train_incremental() requires an already-trained or "
                "loaded model. Call train()/train_from_file()/load() first.")

        before = self.stats()

        from tokenizer import segment
        sentences = split_sentences(corpus)
        segmented = [segment(s) for s in sentences]

        added_vocab = 0
        if extend_vocab:
            if not target_vocab_size:
                raise ValueError(
                    "extend_vocab=True requires target_vocab_size > current vocab_size")
            from collections import Counter
            word_freq = Counter()
            for words in segmented:
                for w in words:
                    word_freq[w] += 1
            added_vocab = self.tokenizer.extend_vocab_from_word_freq(word_freq, target_vocab_size)

        new_seqs = [self.tokenizer.encode_for_training_words(words) for words in segmented]
        had_ctm = self.ctm is not None
        self._merge_graphs(new_seqs)

        after = self.stats()
        return {
            "sentences_added": len(new_seqs),
            "vocab_added": added_vocab,
            "before": before,
            "after": after,
            "ctm_invalidated": had_ctm,
        }

    def _merge_graphs(self, new_seqs):
        """
        Extend Edge/Bridge/Relationship matrices with the structural
        facts in new_seqs.

        Edge Matrix: full recompute every call, same as always -- see
        graph.py's EdgeMatrix docstring for why this was never the
        expensive part (its cost tracks unique bigrams, which is
        bounded by vocab size, not corpus size).

        Bridge and Relationship: INCREMENTAL merge
        (BridgeMatrix.merge_incremental / RelationshipMatrix.
        merge_incremental in graph.py) rather than a full recompute.
        Previously both were rebuilt from scratch over the union of
        old + new triples/sentences on every call -- correct, but it
        meant batch N of a batched corpus (train_corpus.py) cost
        roughly proportional to the TOTAL corpus seen through batch N,
        not just batch N's own size, so N batches cost O(N *
        corpus_size_so_far) in total rather than O(corpus_size). The
        incremental versions only reprocess clustering for groups a
        genuinely new triple actually touches, and only reconstruct
        relationship content once per loaded model (cached after
        that) instead of on every merge -- see each method's own
        docstring in graph.py for the correctness argument. Output is
        provably identical to the old full-recompute path (see
        test_bridge_merge_incremental_matches_full_rebuild() /
        test_relationship_merge_incremental_matches_full_rebuild() in
        test.py); only the cost profile changed.
        """
        from array import array
        from collections import defaultdict

        vsz = self.tokenizer.vocab_size_actual

        # ── Edge Matrix: union of (src, dst) pairs, counts preserved ────
        # Counts must be genuinely merged, not dropped: an incrementally-
        # trained model's bigram frequencies feed Strict Mode's tie-break
        # and ivm.py's bigram_frequencies() (Open Mode's own tie-break
        # cascade) -- an empty/wrong count array here crashes the first
        # real frequency() lookup after a merge, not just returns 0.
        old_counts = {(s, d): c for s, d, c in
                       zip(self.edges.src, self.edges.dst, self.edges.count)}
        new_counts = defaultdict(int)
        for seq in new_seqs:
            for i in range(len(seq) - 1):
                new_counts[(seq[i], seq[i + 1])] += 1
        merged_edges = sorted(set(old_counts) | set(new_counts), key=lambda p: p[0])

        em = EdgeMatrix()
        em.src = array("i", [p[0] for p in merged_edges])
        em.dst = array("i", [p[1] for p in merged_edges])
        em.count = array("i", [old_counts.get(p, 0) + new_counts.get(p, 0)
                                for p in merged_edges])
        em.index = array("i", [0] * (vsz + 1))
        for s in em.src:
            em.index[s + 1] += 1
        for i in range(1, len(em.index)):
            em.index[i] += em.index[i - 1]
        em._vocab_size = vsz

        # ====Relationship Matrix: cache this relationship set's raw
        #    content BEFORE Bridge's row order changes underneath it
        #    (see RelationshipMatrix._ensure_raw_cache's docstring --
        #    a no-op after the first merge on any given model) =======
        self.rels._ensure_raw_cache(self.bridges)

        # ==== Bridge Matrix: incremental merge -- only clusters actually
        #    touched by a new triple are reprocessed (see
        #    BridgeMatrix.merge_incremental's docstring in graph.py) =====
        new_triples = set()
        for seq in new_seqs:
            for i in range(len(seq) - 2):
                source, bridge_tok, target = seq[i], seq[i + 1], seq[i + 2]
                new_triples.add((source, target, bridge_tok))
        self.bridges.merge_incremental(new_triples, vsz)

        # ========= Relationship Matrix: extend from the cache above, against
        #    Bridge's now-updated (post-merge) triple_id numbering ========
        self.rels.merge_incremental(new_seqs, self.bridges)

        self.edges = em
        # noise.py's index/vocabulary matrix feed noise (see
        # _ensure_noise_layer()), so they must be dropped BEFORE the open
        # engine is rebuilt below -- ImportanceVoteMatrix.build() copies
        # whatever model.token_vocab currently is, and a pre-merge one
        # (with its cached per-token rows) would otherwise keep scoring
        # noise from the old graphs.
        self.noise_index = None
        self.token_vocab = None
        self._drop_evidence_cache()
        self._strict = InferenceEngine(self.edges, self.bridges, self.rels, mode="strict")
        self._rebuild_open_engine()

        # The Context Trigger Matrix and Strict Mode's legacy Importance
        # Vote Matrix are opt-in add-ons, not part of the default
        # pipeline -- invalidated (not auto-rebuilt) since both were
        # built from the pre-merge triple/relationship structure.
        self.ctm = None
        self.ivm = None

    def encode(self, text):
        return self.tokenizer.encode(text)

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    # ====== unified-matrix style views (analyse.py / chat.py) =================
    # Thin read-only views over the Edge/Bridge/Relationship matrices, in the
    # flat shape analyse.py expects.

    def token_name(self, token_id):
        return self.tokenizer.id_to_token.get(token_id, token_id)

    token_label = token_name

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size_actual

    def successors(self, token):
        return self.edges.successors(token)

    def triples_from_source(self, token):
        return self.bridges.triples_from_source(token)

    def triple_out_degree(self):
        """One entry per trained triple: its source token (count them for out-degree)."""
        return list(self.bridges.source)

    def clusters_of_token(self, token):
        return list(self.bridges.t_index.get(token, []))

    def cluster_ids(self):
        return sorted({c for c in self.bridges.cluster_id if c != 0})

    def cluster_axis(self, cluster_id):
        return self.bridges.cluster_axis(cluster_id)

    def count(self, rel_id):
        return self.rels.count(rel_id)

    @property
    def rel_count(self):
        return self.rels.rel_count

    def triples_for_relationship(self, rel_id):
        """[(source, target, bridge)] for one training sentence."""
        b = self.bridges
        return [(b.source[i], b.target[i], b.bridge[i])
                for i in self.rels.triples_for_relationship(rel_id)]

    def bigram_rows(self):
        """[(source, target, count)] -- every deduplicated bigram."""
        e = self.edges
        return [(e.src[i], e.dst[i], e.count[i]) for i in range(len(e.src))]

    def triple_rows(self):
        """[(source, bridge, target, cluster_id)] -- every deduplicated triple (0 = no cluster)."""
        b = self.bridges
        return [(b.source[i], b.bridge[i], b.target[i], b.cluster_id[i])
                for i in range(len(b.source))]

    def tile_rows(self, relationship_id=None):
        """
        How each unique sentence is stored as ordered non-overlapping adjacent
        pairs: (relationship_id, position, first, second). An odd-length
        sentence ends with an overlapping pair; a one-token sentence is a
        single token (second = None).
        """
        rows = []
        ids = range(self.n_rels) if relationship_id is None else [relationship_id]
        for rid in ids:
            seq = self.sequence(rid)
            n = len(seq)
            if n == 1:
                rows.append((rid, 0, seq[0], None))
                continue
            pos = 0
            for i in range(0, n - 1, 2):
                rows.append((rid, pos, seq[i], seq[i + 1]))
                pos += 1
            if n > 1 and n % 2 == 1:
                rows.append((rid, pos, seq[n - 2], seq[n - 1]))
        return rows

    @property
    def t_cell(self):
        return self.tile_rows()

    def bigram_info(self, prev_word, curr_word, mode="strict"):
        prev_id = self.tokenizer.encode(prev_word)[-1]
        curr_id = self.tokenizer.encode(curr_word)[-1]
        engine = self._engine(mode)
        witnesses = self.bigram_provenance().get((prev_id, curr_id), ())
        return {
            "prev": prev_word, "curr": curr_word,
            "training": self.edges.frequency(prev_id, curr_id),
            "total": engine._bigram_frequency(prev_id, curr_id),
            "witness_sentences": len(witnesses),
        }

    # ========== Open Mode controls =============================================================

    def set_legality_top_k(self, k=None):
        """How many of the top-scored tokens Q3/Q4 may work on (int, or None/0/'all' = every
        candidate). Sticks across retraining. Returns the stored value (None = all)."""
        try:
            from ivm import parse_top_k
        except ImportError:
            raise RuntimeError("this ivm.py has no legality top-K support (no parse_top_k) -- update ivm.py")
        value = parse_top_k(k)
        self._legality_override = (value,)
        if self.open_ctm is not None and hasattr(self.open_ctm, "set_legality_top_k"):
            self.open_ctm.set_legality_top_k(value)
        return value

    def cache_control(self, action="status"):
        """'on' | 'off' | 'status' for ivm's Q1 cache -> {"enabled", "token_rows", "entries"}."""
        if self.open_ctm is None:
            raise RuntimeError("Open Mode isn't ready (model not trained/loaded) -- no cache to toggle.")
        if action not in ("on", "off", "status", "", None):
            raise ValueError("cache action must be on, off, or status")
        if action == "on":
            self.open_ctm.enable_cache(self.all_candidate_tokens())
        elif action == "off":
            self.open_ctm.disable_cache()
        tc = self.open_ctm._token_cache
        return {"enabled": bool(self.open_ctm._use_cache), "token_rows": len(tc),
                "entries": sum(len(r) for r in tc.values())}

    @staticmethod
    def score_layers():
        """(result key, column label) for each layer of open_mode_candidate_scores(), in print order."""
        return [("q1", "Q1 co-occ %"), ("q2", "Q2 bigram %"), ("q3", "Q3 legal"), ("q4", "Q4 agree"),
                ("ivm", "ivm"), ("noise", "noise"), ("score", "score"), ("final", "final")]

    # ======== evidence API for ivm.py / noise.py ======================================
    # Both scorers read the training data only through these methods, all
    # derived from the Edge/Bridge/Relationship matrices. Caches are dropped
    # whenever the graphs change (see _drop_evidence_cache()).

    def _drop_evidence_cache(self):
        self._evidence = {}

    def _ev(self, key, builder):
        ev = self.__dict__.setdefault("_evidence", {})
        if key not in ev:
            ev[key] = builder()
        return ev[key]

    @property
    def n_rels(self):
        """Number of distinct training sentences (relationship ids)."""
        return self.rels._n_rels

    def _sequences(self):
        def build():
            self.rels._ensure_raw_cache(self.bridges)
            return [tuple(s) for s in self.rels._unique_seqs]
        return self._ev("sequences", build)

    def sequence(self, rel_id):
        """Token-id sequence of one training sentence ((): out of range)."""
        seqs = self._sequences()
        return seqs[rel_id] if 0 <= rel_id < len(seqs) else ()

    def token_relationships(self, min_len=3):
        """{token: {rel_id}} -- every sentence (of at least `min_len` tokens) each token appears in."""
        def build():
            out = {}
            for rid, seq in enumerate(self._sequences()):
                if len(seq) < min_len:
                    continue
                for t in set(seq):
                    out.setdefault(t, set()).add(rid)
            return out
        return {t: set(r) for t, r in self._ev(("token_rels", min_len), build).items()}

    def co_row(self, token):
        """{other_token: number of sentences shared with `token`} (never `token` itself)."""
        def build():
            rows = {}
            for rid, seq in enumerate(self._sequences()):
                if len(seq) < 3:
                    continue
                toks = set(seq)
                for a in toks:
                    row = rows.setdefault(a, {})
                    for b in toks:
                        if a != b:
                            row[b] = row.get(b, 0) + 1
            return rows
        return self._ev("co_rows", build).get(token, {})

    def bigram_counts(self):
        """{(a, b): times b literally followed a in training}."""
        e = self.edges
        return {(e.src[i], e.dst[i]): e.count[i] for i in range(len(e.src))}

    def adjacent_pairs(self):
        """{(a, b)} -- every directed legal bigram."""
        e = self.edges
        return {(e.src[i], e.dst[i]) for i in range(len(e.src))}

    def bigram_provenance(self, include_final=True):
        """{(a, b): {rel_id}} -- the training sentences each bigram occurred in."""
        out = {}
        for rid, seq in enumerate(self._sequences()):
            last = len(seq) - 1 if include_final else len(seq) - 2
            for i in range(last):
                out.setdefault((seq[i], seq[i + 1]), set()).add(rid)
        return out

    def triples(self):
        """Iterate (source, target, bridge) for every consecutive trained triple."""
        b = self.bridges
        return [(b.source[i], b.target[i], b.bridge[i]) for i in range(len(b.source))]

    def _ensure_noise_layer(self):
        """
        Make sure noise.py's TokenVocabularyMatrix (self.token_vocab) exists --
        called at the top of every Open Mode inference entry point below.
        Lazy: nothing is built at train()/train_incremental()/load() time,
        and the matrix itself only computes a token's row the first time
        that token is asked for. ivm.py knows nothing about noise; this is
        the only place the two scorers meet (see noise_final_scores()).
        """
        if self.token_vocab is None:
            self.build_token_vocab()

    def noise_final_scores(self, context_tokens):
        """
        noise.py's FINAL score for every candidate it has evidence for,
        summed over the context tokens and scaled by
        config.NoiseConfig.INFERENCE_WEIGHT. {token_id: score}. Reserved
        tokens (PAD/UNK/BOS/EOS) are skipped as context. This is what
        select(..., extra_scores=...) adds to ivm's own score.
        """
        self._ensure_noise_layer()
        ctx = sorted(t for t in (context_tokens or ()) if t not in (PAD, UNK, BOS, EOS))
        combined = self.token_vocab.combine_final_score(ctx)
        w = NoiseConfig.INFERENCE_WEIGHT
        return {c: w * v for c, v in combined.items()}

    def _rebuild_open_engine(self):
        """
        (Re)build self._open and self.open_ctm from the model's
        current graphs. Open Mode has no separate "build" step -- it's
        always available as soon as the model is trained, and stays in
        sync with the graphs automatically whenever they change (fresh
        train, incremental merge, or load).
        """
        from ivm import ImportanceVoteMatrix
        self._open = InferenceEngine(
            self.edges, self.bridges, self.rels,
            mode="open", vocab=self.all_candidate_tokens(),
        )
        self.open_ctm = ImportanceVoteMatrix.build(self)
        ov = getattr(self, '_legality_override', None)
        if ov is not None and hasattr(self.open_ctm, "set_legality_top_k"):
            self.open_ctm.set_legality_top_k(ov[0])

    def _build_graphs(self, seqs):
        vsz = self.tokenizer.vocab_size_actual
        self.edges.build(seqs, vsz)
        self.bridges.build(seqs, vsz)
        self.rels.build(seqs, self.bridges)
        # Same reason as _merge_graphs(): noise's data source is derived
        # from the graphs just rebuilt, so any earlier one is stale.
        self.noise_index = None
        self.token_vocab = None
        self._drop_evidence_cache()
        self._strict = InferenceEngine(self.edges, self.bridges, self.rels, mode="strict")
        self._rebuild_open_engine()

    def build_context_triggers(self, mode="strict", min_support=CTMConfig.MIN_SUPPORT):
        """
        Build and cache a Context Trigger Matrix -- per-cluster,
        per-member trigger signatures from whole-sentence co-occurrence
        (see ctm.py). This does not change generate()'s behavior by
        itself; pass use_context_triggers=True to generate() to opt in.
        Returns the built ContextTriggerMatrix (also stored on self.ctm).
        """
        from ctm import ContextTriggerMatrix
        self.ctm = ContextTriggerMatrix.build(self, mode=mode, min_support=min_support)
        return self.ctm

    def has_context_triggers(self):
        return self.ctm is not None

    def build_importance_votes(self, mode="strict"):
        """
        Build and cache an Importance Vote Matrix -- vote weights are
        distinct-training-sentence co-occurrence counts between every
        pair of tokens (see ivm.py). This does not change generate()'s
        behavior by itself; pass use_importance_votes=True to
        generate() to opt in. Returns the built ImportanceVoteMatrix
        (also stored on self.ivm).

        Unlike Context Triggers, opting into importance voting also
        changes WHICH mechanism resolves a Stage 1 tie: it replaces
        Stage 2's lineage tie-break rather than merely adding a step
        before the random fallback -- see ivm.py / inference.py.
        """
        from ivm import ImportanceVoteMatrix
        self.ivm = ImportanceVoteMatrix.build(self, mode=mode)
        return self.ivm

    def has_importance_votes(self):
        return self.ivm is not None

    def build_noise_index(self, token_rels=None):
        """
        Build and cache a NoiseCancellationIndex (see noise.py) --
        cross-sentence gated voting, no Bridge-Matrix clustering
        needed: for one anchor token sitting in one home sentence,
        every OTHER training sentence that doesn't also contain the
        anchor votes on which of the home sentence's other words are
        worth attention (lacking a word = voting for it; already
        having it = staying quiet). Read-only analysis, same as
        build_context_triggers() -- doesn't change generate()'s
        behavior by itself (noise's data source is the token_vocab built
        on top of it -- see build_token_vocab()). Returns the built
        index (also stored on self.noise_index). `token_rels` is an
        optional precomputed token -> {relationship_id} map (what
        ivm.token_to_relationships() returns) to reuse instead of
        recomputing it.
        """
        from noise import NoiseCancellationIndex
        self.noise_index = NoiseCancellationIndex.build(self, token_rels=token_rels)
        return self.noise_index

    def noise_scores(self, anchor_word, home_relationship_id=None):
        """
        Thin wrapper -- see noise.noise_scores() for the full
        contract (auto-builds self.noise_index on first use; requires
        home_relationship_id if anchor_word appears in more than one
        training sentence -- see noise.find_homes() to list them).
        """
        from noise import noise_scores as _noise_scores
        return _noise_scores(self, anchor_word, home_relationship_id=home_relationship_id)

    def focus_scores(self, anchor_word):
        """
        Global, cross-sentence version of noise_scores() -- see
        noise.focus_scores()'s docstring. Sums the per-home
        noise-cancellation table across EVERY training sentence
        anchor_word appears in, so the result is a property of the
        TOKEN ("wherever this appears, focus on these"), not of one
        sentence.
        """
        from noise import focus_scores as _focus_scores
        return _focus_scores(self, anchor_word)

    def build_token_vocab(self):
        """
        Build and cache a TokenVocabularyMatrix (see noise.py) --
        every token's PERMANENT (for this model's lifetime; not
        persisted, same as ctm/ivm/noise_index) vocabulary row,
        precomputed once by summing focus_scores() for every token
        that has one. After this call, row()/combine() queries never
        touch a literal training sentence again -- the sentence was
        only ever the training mechanism. This matrix is also Open
        Mode's noise data source: each row value is the three-stage
        AVERAGE score, and generate(mode="open") sums the context
        tokens' rows into noise (see ivm.py's _noise_vote()). You never
        need to call this for that -- _ensure_noise_layer() does it on
        first Open Mode use. Returns the built matrix (also stored on
        self.token_vocab).
        """
        from noise import TokenVocabularyMatrix
        self.token_vocab = TokenVocabularyMatrix.build(self)
        return self.token_vocab

    def token_row(self, word):
        """Thin wrapper -- see noise.vocabulary_row()'s docstring."""
        from noise import vocabulary_row
        return vocabulary_row(self, word)

    def combine_context(self, words):
        """Thin wrapper -- see noise.combine_context()'s docstring."""
        from noise import combine_context as _combine_context
        return _combine_context(self, words)

    def noise_why(self, anchor_word, home_relationship_id, candidate_word):
        """
        The independent follow-up to noise_scores()/focus_scores()/
        token_row(): which specific relationship_ids voted yes for
        candidate_word, given this anchor+home? See noise.why()'s
        docstring -- never computed as part of the bulk/bulk-summing
        paths, only on demand for one candidate at a time.
        """
        from noise import why as _why
        return _why(self, anchor_word, home_relationship_id, candidate_word)

    def noise_homes(self, word):
        """Thin wrapper -- see noise.find_homes()'s docstring (every training sentence containing `word`)."""
        from noise import find_homes
        if self.noise_index is None:
            self.build_noise_index()
        return find_homes(self, word)

    def noise_scores_detailed(self, anchor_word, home=None):
        """
        Thin wrapper -- see noise.noise_scores_detailed()'s docstring:
        one anchor+home's noise-cancellation scores, broken down by
        the three underlying stages (no combined average). Raises
        ValueError if anchor_word has multiple candidate homes and
        `home` wasn't given (see noise_homes() to list them).
        """
        from noise import noise_scores_detailed as _noise_scores_detailed
        if self.noise_index is None:
            self.build_noise_index()
        return _noise_scores_detailed(self, anchor_word, home_relationship_id=home)

    def focus_scores_detailed(self, anchor_word, weight_mode="count"):
        """
        Thin wrapper -- see noise.focus_scores_detailed()'s docstring:
        stage1/stage2/stage3 (unchanged) plus stage4
        (shared-knowledge attention; weight_mode="count"/"flat"), per
        candidate, for one anchor word.
        """
        from noise import focus_scores_detailed as _focus_scores_detailed
        if self.noise_index is None:
            self.build_noise_index()
        return _focus_scores_detailed(self, anchor_word, weight_mode=weight_mode)

    def token_row_detailed(self, word):
        """Thin wrapper -- see noise.vocabulary_row_detailed()'s docstring (permanent, precomputed, broken down by stage)."""
        from noise import vocabulary_row_detailed
        if self.token_vocab is None:
            self.build_token_vocab()
        return vocabulary_row_detailed(self, word)

    def combine_detailed(self, words, weight_mode="count"):
        """
        Thin wrapper -- see noise.combine_context_detailed()'s
        docstring: RAW combine, each named word's own
        focus_scores_detailed() row summed element-wise across words.
        """
        from noise import combine_context_detailed
        if self.noise_index is None:
            self.build_noise_index()
        return combine_context_detailed(self, words, weight_mode=weight_mode)

    def combine_normalized(self, words, weight_mode="count"):
        """
        Thin wrapper -- see noise.combine_context_normalized()'s
        docstring: the named words' RAW rows summed element-wise, THEN
        normalized once (combine_detailed() before the normalizing step).
        """
        from noise import combine_context_normalized
        if self.noise_index is None:
            self.build_noise_index()
        return combine_context_normalized(self, words, weight_mode=weight_mode)

    def normalize_scores(self, scores):
        """
        Thin wrapper -- see noise.normalize_row()'s docstring: turn a
        raw {candidate: {"stage1",...,"stage4"}} dict (from
        focus_scores_detailed()/combine_detailed()/combine_normalized())
        into percentages, scoped to the candidates in `scores` itself.
        """
        from noise import normalize_row
        return normalize_row(scores)

    def context_trigger_rows(self, min_support=CTMConfig.MIN_SUPPORT, mode="strict"):
        """
        Thin wrapper -- see ctm.build_context_trigger_matrix()'s
        docstring. Analysis/display only: does NOT build or cache
        self.ctm (call build_context_triggers() for that, needed
        before generate(..., use_context_triggers=True)).
        """
        from ctm import build_context_trigger_matrix
        return build_context_trigger_matrix(self, min_support=min_support, mode=mode)

    # ========== generate ===================================================================

    def bigram_witness_sentences(self, prev_id, curr_id):
        """
        {rel_id, ...} of literal training sentences that contained the
        exact consecutive bigram (prev_id -> curr_id) -- the evidence
        set V5's bigram-witness vote checks context tokens against
        (see ivm.py's bigram_relationships()/_bigram_witness_vote()).

        Mode-independent: bigram_relationships() is built ONLY from
        the literal Bridge + Relationship Matrix regardless of mode
        (Experience triples were never one real sentence, so there's
        nothing to record for them here either), so self.ivm and
        self.open_ctm always carry the identical index -- this uses
        whichever is already built to avoid recomputing it, and falls
        back to computing it directly from the graphs (still
        literal-only) if neither has been built yet.
        """
        ivm = self.open_ctm or self.ivm
        if ivm is not None:
            return set(ivm._bigram_rels.get((prev_id, curr_id), set()))
        from ivm import bigram_relationships
        return set(bigram_relationships(self).get((prev_id, curr_id), set()))

    def all_candidate_tokens(self):
        """
        Every vocabulary token id EXCEPT <PAD>/<UNK>/<BOS> -- Open
        Mode's candidate universe (see self._open.vocab / _rebuild_open_engine()),
        computed once whenever the graphs (re)build and reused from
        there. <EOS> is deliberately kept IN this set (not excluded as
        reserved) so the model can still choose to end generation on
        IVM's own evidence (Q1-Q4, see ivm.py) rather than only ever
        stopping via a successors gate -- Open Mode has none anymore.
        """
        return sorted(t for t in self.tokenizer.token_to_id.values()
                      if t not in (PAD, UNK, BOS))

    def generate(self, prompt, max_tokens=GenerationConfig.MAX_TOKENS, mode="strict",
                 use_context_triggers=False, use_importance_votes=False):
        engine = self._engine(mode)
        if mode == "open":
            self._ensure_noise_layer()   # noise's data source -- see above
            # Open Mode's CTM/IVM weighted voting is the PRIMARY
            # candidate-selection mechanism -- always wired in, auto-
            # built alongside self._open as soon as the model is
            # trained (see _rebuild_open_engine()). The
            # use_context_triggers/use_importance_votes flags below
            # are Strict-Mode-only legacy opt-ins and have no effect
            # here; see inference.py's Open Mode fork.
            #
            # Candidates for every step are the ENTIRE vocabulary
            # (self._open.vocab, set once at construction -- see
            # all_candidate_tokens() above), not gated by whether a
            # bigram was ever literally observed -- and neither is the
            # PROMPT itself: unlike Strict Mode, engine.generate() skips
            # its prompt-bigram legality check entirely for Open Mode
            # (see inference.py), so any prompt is accepted here, even
            # one containing a transition the model has genuinely never
            # seen. IVM's ten vote layers (V1-noise) score the full
            # candidate set directly, every step.
            ids, trace = engine.generate(
                self.tokenizer.encode(prompt), max_tokens=max_tokens,
                importance_votes=self.open_ctm,
                extra_scores_fn=self.noise_final_scores)
            return self.tokenizer.decode(ids), ids, trace
        triggers = self.ctm if (use_context_triggers and self.ctm) else None
        votes = self.ivm if (use_importance_votes and self.ivm) else None
        ids, trace = engine.generate(
            self.tokenizer.encode(prompt), max_tokens=max_tokens,
            context_triggers=triggers, importance_votes=votes)
        return self.tokenizer.decode(ids), ids, trace

    def explain_step(self, previous_text, current_text, mode="strict"):
        engine   = self._engine(mode)
        prev_ids = self.tokenizer.encode(previous_text) if previous_text else []
        curr_ids = self.tokenizer.encode(current_text)
        prev     = prev_ids[-1] if prev_ids else None
        curr     = curr_ids[-1]
        if mode == "open":
            self._ensure_noise_layer()
            token, trace = engine.step(prev, curr, importance_votes=self.open_ctm,
                                       context_tokens=set(prev_ids + curr_ids),
                                       extra_scores_fn=self.noise_final_scores)
        else:
            token, trace = engine.step(prev, curr)
        return self.tokenizer.decode([token]), trace

    def open_mode_candidate_scores(self, prompt, full_vocab_candidates=None):
        """
        Full CTM/IVM score breakdown for the NEXT token given `prompt`,
        under Open Mode's primary selection mechanism (see ivm.py's
        score_candidates()/select()). Requires open_ctm to be built
        (i.e. Open Mode itself must be available). Returns None if
        Open Mode isn't ready. Decoded to surface strings for display.

        "scores" is the combined score candidates are ranked by:
        ivm's Q1 + Q2 ("ivm_scores") plus noise.py's final score
        ("noise_scores"). Q3/Q4 are legality checks applied to that
        ranking when the winner is picked ("legality" says which applied;
        "final_scores" zeroes the ineligible). Each piece is exposed
        separately so it stays auditable.
        "winner" and
        "tie_break_stage" report what select() actually returned and
        which stage of the cascade decided it (score /
        bigram_frequency / global_frequency / lowest_token_id) -- see
        ivm.py's select(). "cache_used" reports whether open_ctm's
        opt-in sparse per-token cache (V1/V2/V3/V4/V6 -- see
        ivm.py's build_cache()/enable_cache()) actually served this
        breakdown, or whether it was computed live -- off by default,
        call model.open_ctm.enable_cache(model.all_candidate_tokens())
        to turn it on.

        Candidates are always the entire vocabulary (self._open.vocab)
        -- there is no narrower option anymore, since Open Mode is no
        longer gated by literal successors at all. `full_vocab_candidates`
        is accepted-but-ignored for call-site compatibility with older
        callers; a deprecation note, not a live toggle.
        """
        if self.open_ctm is None or self._open is None:
            return None
        self._ensure_noise_layer()
        ids = self.tokenizer.encode(prompt)
        current = ids[-1]
        previous = ids[-2] if len(ids) >= 2 else None
        candidates = self._open.vocab
        context_tokens = set(ids)
        extra = self.noise_final_scores(context_tokens)
        winner, trace = self.open_ctm.select(candidates, context_tokens, current=current,
                                             previous=previous, extra_scores=extra)
        dec = self._dec_tok
        d = lambda m: {dec(c): v for c, v in m.items()}

        return {
            "candidates": [dec(c) for c in candidates],
            "context_tokens": [dec(c) for c in sorted(context_tokens)],
            "q1": d(trace["q1"]),
            "q2": d(trace["q2"]),
            "q3": d(trace["q3"]),
            "q4": d(trace["q4"]),
            "q1_raw": d(trace["q1_raw"]),
            "q2_raw": d(trace["q2_raw"]),
            "ivm": d(trace["scores"]),                    # Q1 + Q2
            "noise": d({c: extra.get(c, 0) for c in candidates}),
            "score": d(trace["total_scores"]),            # ivm + noise: what candidates are ranked by
            "scores": d(trace["total_scores"]),           # same as "score" (ranking key)
            "final": d(trace["final_scores"]),            # score if eligible to win, else 0
            "noise_weight": NoiseConfig.INFERENCE_WEIGHT,
            "legality": trace["legality"],                # "q3+q4" | "q3" | "none" (none -> <EOS>)
            "q4_active": trace.get("q4_active", True),
            "legality_top_k": trace.get("legality_top_k"),
            "window_size": trace.get("window_size", len(candidates)),
            "winner": dec(winner) if winner is not None else None,
            "tie_break_stage": trace["tie_break_stage"],
            "cache_used": trace.get("cache_used", False),
        }

    def infer_shared_role(self, words, mode="strict"):
        ids = []
        for w in words:
            enc = [t for t in self.tokenizer.encode(w) if t != 2]
            if enc:
                ids.append(enc[-1])
        engine  = self._engine(mode)
        results = engine.infer_shared_role(ids)
        decoded = []
        for token, axis, ev in results:
            label = (self.tokenizer.id_to_token.get(token, str(token))
                     if token in (0, 1, 2, 3)
                     else self.tokenizer.decode([token]))
            decoded.append((label, axis, ev))
        return decoded

    def _dec_tok(self, t):
        tok = self.tokenizer
        return tok.id_to_token.get(t, t) if t in (0, 1, 2, 3) else tok.decode([t])

    def _decode_interpretation(self, result):
        """Decode a raw interpret.py result (token ids) to surface strings."""
        if result is None:
            return None
        dec = self._dec_tok
        return {
            "cluster_id": result["cluster_id"],
            "axis": result["axis"],
            "members": [dec(m) for m in result["members"]],
            "candidates": [
                {**c,
                 "interpreter_token": dec(c["interpreter_token"]),
                 "via_bridge_token": dec(c["via_bridge_token"]),
                 "members_covered": [dec(m) for m in c["members_covered"]],
                 "shared_role_overlap": c["shared_role_overlap"]}
                for c in result["candidates"]
            ],
        }

    def _decode_ci_row(self, row):
        dec = self._dec_tok
        return {
            **row,
            "members": [dec(m) for m in row["members"]],
            "interpreter_token": dec(row["interpreter_token"]),
            "via_bridge_token": dec(row["via_bridge_token"]),
            "members_covered": [dec(m) for m in row["members_covered"]],
        }

    def interpret_cluster(self, cluster_id, top_n=InterpretConfig.TOP_N, mode="strict"):
        """
        Propose a human-readable interpreter token for one cluster_id,
        with evidence gathered from Edge, Bridge, and Relationship
        matrices. See interpret.py for the method and its honesty
        caveats (coverage is a plain fraction, not a calibrated
        confidence score; evidence_mask entries are correlated signals,
        not independent votes to be summed into a percentage).
        Returns None if cluster_id is unknown / degenerate.
        """
        from interpret import interpret_cluster as _interpret_cluster
        result = _interpret_cluster(self, cluster_id, top_n=top_n, mode=mode)
        return self._decode_interpretation(result)

    def interpret_all_clusters(self, min_coverage=InterpretConfig.MIN_COVERAGE, max_per_cluster=InterpretConfig.MAX_PER_CLUSTER, mode="strict"):
        """
        Every candidate that clears min_coverage for each cluster (up to
        max_per_cluster, best first) -- not just one label per cluster.
        See interpret.py.
        """
        from interpret import interpret_all_clusters as _interpret_all
        raw = _interpret_all(self, min_coverage=min_coverage,
                              max_per_cluster=max_per_cluster, mode=mode)
        return [self._decode_interpretation(r) for r in raw]

    def build_interpreter_matrix(self, min_coverage=InterpretConfig.MIN_COVERAGE, min_signals=InterpretConfig.MIN_SIGNALS,
                                  max_per_cluster=None, mode="strict"):
        """
        The filtered Cluster Interpreter Matrix: one row per (cluster_id,
        interpreter) pair whose candidate clears both min_coverage and
        min_signals -- every qualifying interpreter is kept, so a cluster
        can carry more than one label at once (see
        interpret.build_interpreter_matrix for exact semantics).
        Returned rows are decoded to surface strings, ready to print or
        json-dump for persistence.
        """
        from interpret import build_interpreter_matrix as _build_ci
        raw = _build_ci(self, min_coverage=min_coverage, min_signals=min_signals,
                         max_per_cluster=max_per_cluster, mode=mode)
        return [self._decode_ci_row(r) for r in raw]

    def _decode_zero_group(self, row):
        dec = self._dec_tok
        return {
            **row,
            "members": [dec(m) for m in row["members"]],
            "interpreter_token": dec(row["interpreter_token"]),
            "via_bridge_token": dec(row["via_bridge_token"]),
        }

    def discover_zero_cluster_groups(self, min_group_size=InterpretConfig.MIN_GROUP_SIZE, mode="strict"):
        """
        Mine cluster_id==0 for source-axis groups the standard dual-axis
        rule never assigns a cluster_id to at all (fix bridge+target,
        source varies -- a rule BridgeMatrix.build doesn't implement).
        See interpret.discover_zero_cluster_groups for exact semantics
        and its caveats (larger candidate space than the regular CI
        matrix -- eyeball results, don't trust min_group_size alone).
        """
        from interpret import discover_zero_cluster_groups as _discover
        raw = _discover(self, min_group_size=min_group_size, mode=mode)
        return [self._decode_zero_group(r) for r in raw]

    def token_similarity(self, word_a, word_b, mode="strict"):
        """
        `mode` is accepted for call-site compatibility but has no
        effect -- there is no longer a separate Experience Bridge
        Matrix, so "strict" and "open" see identical cluster overlap.
        """
        tok = self.tokenizer
        def resolve(w):
            enc = [t for t in tok.encode(w) if t != 2]
            return enc[-1] if enc else None
        ta, tb = resolve(word_a), resolve(word_b)
        if ta is None or tb is None:
            return {"word_a": word_a, "word_b": word_b,
                    "similarity": 0, "shared_clusters": []}
        sa = set(self.bridges.t_index.get(ta, []))
        sb = set(self.bridges.t_index.get(tb, []))
        shared = sorted(sa & sb)
        return {"word_a": word_a, "word_b": word_b,
                "similarity": len(shared), "shared_clusters": shared}

    def _engine(self, mode):
        if mode not in ("strict", "open"):
            raise ValueError(f"Unknown mode: {mode!r} (expected 'strict' or 'open')")
        if mode == "open":
            if self._open is None:
                raise RuntimeError("Model has not been trained or loaded.")
            return self._open
        if self._strict is None:
            raise RuntimeError("Model has not been trained or loaded.")
        return self._strict

    # ===== stats ====================================================================

    def stats(self):
        return {
            "vocab_size":        self.tokenizer.vocab_size_actual,
            "edges":             len(self.edges.src),
            "bridges":           len(self.bridges.source),
            "clustered_bridges": sum(1 for c in self.bridges.cluster_id if c != 0),
            "clusters":          len(set(c for c in self.bridges.cluster_id if c != 0)),
            "relationships":     self.rels._n_rels,
            "relationship_rows": len(self.rels.r_triple),
            "relationship_occurrences": sum(self.rels.rel_count),
        }

    # ======= save / load =================================================

    def save(self, folder):
        os.makedirs(folder, exist_ok=True)
        self.tokenizer.save(os.path.join(folder, "vocabulary.json"))
        with open(os.path.join(folder, "edges.json"),         "w") as f: json.dump(self.edges.to_dict(),   f)
        with open(os.path.join(folder, "bridges.json"),       "w") as f: json.dump(self.bridges.to_dict(), f)
        with open(os.path.join(folder, "relationships.json"), "w") as f: json.dump(self.rels.to_dict(),    f)
        with open(os.path.join(folder, "meta.json"),          "w") as f:
            json.dump({"vocab_size_config": self.tokenizer.vocab_size,
                       "stats": self.stats()}, f, indent=2)

    @classmethod
    def load(cls, folder):
        m = cls()
        m.tokenizer = CharWordTokenizer.load(os.path.join(folder, "vocabulary.json"))
        with open(os.path.join(folder, "edges.json"))         as f: m.edges   = EdgeMatrix.from_dict(json.load(f))
        with open(os.path.join(folder, "bridges.json"))       as f: m.bridges = BridgeMatrix.from_dict(json.load(f))
        with open(os.path.join(folder, "relationships.json")) as f: m.rels    = RelationshipMatrix.from_dict(json.load(f))
        m._strict = InferenceEngine(m.edges, m.bridges, m.rels, mode="strict")
        m._rebuild_open_engine()
        return m
