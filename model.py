"""
model.py — MSEGraphLanguageModel orchestrator.

Two inference modes, one set of graphs -- there is no separate
"experience" data source anymore:
  strict  — InferenceEngine(E, B, R, mode="strict")
            candidates gated to literal Edge/Bridge successors,
            two-stage lineage-vote pipeline (default)
  open    — InferenceEngine(E, B, R, mode="open", vocab=<all tokens>)
            candidates are the ENTIRE vocabulary every step, scored
            by IVM's V1-V6 weighted voting (see ivm.py)

Both engines are built automatically as soon as the model is trained
or loaded -- there is no separate "build Open Mode" step anymore.
"""

import json
import os

from tokenizer import BPETokenizer, split_sentences
from graph import EdgeMatrix, BridgeMatrix, RelationshipMatrix
from inference import InferenceEngine


class MSEGraphLanguageModel:

    def __init__(self, vocab_size=2000):
        self.tokenizer   = BPETokenizer(vocab_size=vocab_size)
        self.edges       = EdgeMatrix()
        self.bridges     = BridgeMatrix()
        self.rels        = RelationshipMatrix()
        self._strict     = None   # InferenceEngine, mode="strict"
        self._open       = None   # InferenceEngine, mode="open" -- auto-built alongside _strict
        self.ctm         = None   # ContextTriggerMatrix — None until built
        self.ivm         = None   # ImportanceVoteMatrix — Strict Mode legacy tie-break, opt-in
        self.open_ctm    = None   # ImportanceVoteMatrix — Open Mode PRIMARY mechanism, auto-built

    # ─── training ─────────────────────────────────────────────────────────

    def train(self, corpus):
        self.tokenizer.train(corpus)
        seqs = [self.tokenizer.encode_for_training(s)
                for s in split_sentences(corpus)]
        self._build_graphs(seqs)

    def train_from_file(self, path):
        self.tokenizer.train_from_file(path)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        seqs = [self.tokenizer.encode_for_training(s)
                for s in split_sentences(text)]
        self._build_graphs(seqs)

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
        BPETokenizer.extend_vocab for what that does and doesn't
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
        relationship_ids (one per originally-trained sentence) are
        preserved and only remapped to whatever row position their
        triple ends up at after the merge; new sentences get new
        relationship_ids continuing after the last existing one.

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

        added_vocab = 0
        if extend_vocab:
            if not target_vocab_size:
                raise ValueError(
                    "extend_vocab=True requires target_vocab_size > current vocab_size")
            added_vocab = self.tokenizer.extend_vocab(corpus, target_vocab_size)

        new_seqs = [self.tokenizer.encode_for_training(s)
                    for s in split_sentences(corpus)]
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
        Rebuild Edge/Bridge/Relationship matrices as the union of the
        model's current matrices and the structural facts in new_seqs.
        See train_incremental's docstring for why this is a full
        recompute rather than an append.
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

        # ── Bridge Matrix: union of (source,target,bridge) triples,
        #    cluster_id recomputed from scratch over the merged set ────
        old_triple_content = list(zip(self.bridges.source, self.bridges.target,
                                       self.bridges.bridge))
        merged_triples = set(old_triple_content)
        for seq in new_seqs:
            for i in range(len(seq) - 2):
                source, bridge_tok, target = seq[i], seq[i + 1], seq[i + 2]
                merged_triples.add((source, target, bridge_tok))
        merged_triples = sorted(merged_triples, key=lambda t: t[0])

        n = len(merged_triples)
        cluster_id = [0] * n
        groups_by_st = defaultdict(list)
        groups_by_sb = defaultdict(list)
        for idx, (s, t, b) in enumerate(merged_triples):
            groups_by_st[(s, t)].append(idx)
            groups_by_sb[(s, b)].append(idx)
        next_cluster = 1
        for _key, idxs in groups_by_st.items():
            if len(idxs) > 1:
                for i in idxs:
                    cluster_id[i] = next_cluster
                next_cluster += 1
        for _key, idxs in groups_by_sb.items():
            if len(idxs) > 1 and all(cluster_id[i] == 0 for i in idxs):
                for i in idxs:
                    cluster_id[i] = next_cluster
                next_cluster += 1

        bm = BridgeMatrix()
        bm.source = array("i", [t[0] for t in merged_triples])
        bm.target = array("i", [t[1] for t in merged_triples])
        bm.bridge = array("i", [t[2] for t in merged_triples])
        bm.cluster_id = array("i", cluster_id)
        bm.index = array("i", [0] * (vsz + 1))
        for s in bm.source:
            bm.index[s + 1] += 1
        for i in range(1, len(bm.index)):
            bm.index[i] += bm.index[i - 1]
        bm._vocab_size = vsz
        t_index = defaultdict(set)
        for s, t, b, c in zip(bm.source, bm.target, bm.bridge, bm.cluster_id):
            if c != 0:
                t_index[b].add(c)
                t_index[t].add(c)
        bm.t_index = {k: sorted(v) for k, v in t_index.items()}

        # ── Relationship Matrix: remap existing rows to their new
        #    triple_id (content unchanged, row position may have moved),
        #    then append new sentences with fresh relationship_ids ─────
        new_triple_to_id = {trip: idx for idx, trip in enumerate(merged_triples)}
        old_n_rels = self.rels._n_rels

        merged_rows = []
        for old_tid, rel_id in zip(self.rels.r_triple, self.rels.r_rel):
            content = old_triple_content[old_tid]
            merged_rows.append((new_triple_to_id[content], rel_id))

        for local_idx, seq in enumerate(new_seqs):
            rel_id = old_n_rels + local_idx
            for i in range(len(seq) - 2):
                source, bridge_tok, target = seq[i], seq[i + 1], seq[i + 2]
                tid = new_triple_to_id.get((source, target, bridge_tok))
                if tid is not None:
                    merged_rows.append((tid, rel_id))

        merged_rows.sort(key=lambda r: r[1])
        rm = RelationshipMatrix()
        rm._n_rels = old_n_rels + len(new_seqs)
        rm.r_triple = array("i", [r[0] for r in merged_rows])
        rm.r_rel = array("i", [r[1] for r in merged_rows])
        rm.index = array("i", [0] * (rm._n_rels + 1))
        for r in rm.r_rel:
            rm.index[r + 1] += 1
        for i in range(1, len(rm.index)):
            rm.index[i] += rm.index[i - 1]
        rm._by_triple_rel = None
        rm._by_triple_index = None

        self.edges, self.bridges, self.rels = em, bm, rm
        self._strict = InferenceEngine(self.edges, self.bridges, self.rels, mode="strict")
        self._rebuild_open_engine()

        # The Context Trigger Matrix and Strict Mode's legacy Importance
        # Vote Matrix are opt-in add-ons, not part of the default
        # pipeline -- invalidated (not auto-rebuilt) since both were
        # built from the pre-merge triple/relationship structure.
        self.ctm = None
        self.ivm = None

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

    def _build_graphs(self, seqs):
        vsz = self.tokenizer.vocab_size_actual
        self.edges.build(seqs, vsz)
        self.bridges.build(seqs, vsz)
        self.rels.build(seqs, self.bridges)
        self._strict = InferenceEngine(self.edges, self.bridges, self.rels, mode="strict")
        self._rebuild_open_engine()

    def build_context_triggers(self, mode="strict", min_support=1):
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

    # ─── generate ─────────────────────────────────────────────────────────

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
        IVM's own evidence (V1-V6, see ivm.py) rather than only ever
        stopping via a successors gate -- Open Mode has none anymore.
        """
        from tokenizer import PAD, UNK, BOS
        return sorted(t for t in self.tokenizer.token_to_id.values()
                      if t not in (PAD, UNK, BOS))

    def generate(self, prompt, max_tokens=40, mode="strict",
                 use_context_triggers=False, use_importance_votes=False):
        engine = self._engine(mode)
        if mode == "open":
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
            # bigram was ever literally observed. IVM's V1-V6 vote
            # layers score the full set directly, every step.
            ids, trace = engine.generate(
                self.tokenizer.encode(prompt), max_tokens=max_tokens,
                importance_votes=self.open_ctm)
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
            token, trace = engine.step(prev, curr, importance_votes=self.open_ctm)
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

        "scores" is the FINAL combined score: V1 ("important_vote") +
        V2 ("influence_vote") + V3 ("context_vote") + V4
        ("context_influence_vote") + V5 ("bigram_witness_vote") + V6
        ("adjacency_vote") -- six independent vote layers, summed,
        each exposed separately so all six stay independently
        auditable. "winner" and "tie_break_stage" report what
        select() actually returned and which stage of the cascade
        decided it (score / bigram_frequency / global_frequency /
        lowest_token_id) -- see ivm.py's select().

        Candidates are always the entire vocabulary (self._open.vocab)
        -- there is no narrower option anymore, since Open Mode is no
        longer gated by literal successors at all. `full_vocab_candidates`
        is accepted-but-ignored for call-site compatibility with older
        callers; a deprecation note, not a live toggle.
        """
        if self.open_ctm is None or self._open is None:
            return None
        ids = self.tokenizer.encode(prompt)
        current = ids[-1]
        candidates = self._open.vocab
        context_tokens = set(ids)
        trace = self.open_ctm.score_candidates(candidates, context_tokens, current=current)
        winner, _ = self.open_ctm.select(candidates, context_tokens, current=current)
        dec = self._dec_tok

        max_score = max(trace["scores"].values()) if trace["scores"] else None
        tied_on_score = [c for c, s in trace["scores"].items() if s == max_score]
        if len(tied_on_score) <= 1:
            tie_break_stage = "score"
        else:
            bigram_freqs = {c: self.open_ctm._bigram_frequency(current, c) for c in tied_on_score}
            if len(set(bigram_freqs.values())) > 1:
                tie_break_stage = "bigram_frequency"
            else:
                global_freqs = {c: self.open_ctm._global_frequency(c) for c in tied_on_score}
                tie_break_stage = "global_frequency" if len(set(global_freqs.values())) > 1 else "lowest_token_id"

        return {
            "candidates": [dec(c) for c in candidates],
            "important_tokens": [dec(t) for t in trace["important_tokens"]],
            "influence": {dec(t): v for t, v in trace["influence"].items()},
            "knows": {dec(t): {dec(c): s for c, s in k.items()}
                      for t, k in trace["knows"].items()},
            "important_vote": {dec(c): v for c, v in trace["important_vote"].items()},
            "influence_vote": {dec(c): v for c, v in trace["influence_vote"].items()},
            "context_vote": {dec(c): v for c, v in trace["context_vote"].items()},
            "context_influence_vote": {dec(c): v for c, v in trace["context_influence_vote"].items()},
            "bigram_witness_vote": {dec(c): v for c, v in trace["bigram_witness_vote"].items()},
            "adjacency_vote": {dec(c): v for c, v in trace["adjacency_vote"].items()},
            "scores": {dec(c): v for c, v in trace["scores"].items()},
            "winner": dec(winner) if winner is not None else None,
            "tie_break_stage": tie_break_stage,
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

    def interpret_cluster(self, cluster_id, top_n=5, mode="strict"):
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

    def interpret_all_clusters(self, min_coverage=0.5, max_per_cluster=3, mode="strict"):
        """
        Every candidate that clears min_coverage for each cluster (up to
        max_per_cluster, best first) -- not just one label per cluster.
        See interpret.py.
        """
        from interpret import interpret_all_clusters as _interpret_all
        raw = _interpret_all(self, min_coverage=min_coverage,
                              max_per_cluster=max_per_cluster, mode=mode)
        return [self._decode_interpretation(r) for r in raw]

    def build_interpreter_matrix(self, min_coverage=0.5, min_signals=2,
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

    def discover_zero_cluster_groups(self, min_group_size=2, mode="strict"):
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

    # ─── stats ────────────────────────────────────────────────────────────

    def stats(self):
        return {
            "vocab_size":        self.tokenizer.vocab_size_actual,
            "edges":             len(self.edges.src),
            "bridges":           len(self.bridges.source),
            "clustered_bridges": sum(1 for c in self.bridges.cluster_id if c != 0),
            "clusters":          len(set(c for c in self.bridges.cluster_id if c != 0)),
            "relationships":     self.rels._n_rels,
            "relationship_rows": len(self.rels.r_triple),
        }

    # ─── save / load ──────────────────────────────────────────────────────

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
        m.tokenizer = BPETokenizer.load(os.path.join(folder, "vocabulary.json"))
        with open(os.path.join(folder, "edges.json"))         as f: m.edges   = EdgeMatrix.from_dict(json.load(f))
        with open(os.path.join(folder, "bridges.json"))       as f: m.bridges = BridgeMatrix.from_dict(json.load(f))
        with open(os.path.join(folder, "relationships.json")) as f: m.rels    = RelationshipMatrix.from_dict(json.load(f))
        m._strict = InferenceEngine(m.edges, m.bridges, m.rels, mode="strict")
        m._rebuild_open_engine()
        return m
