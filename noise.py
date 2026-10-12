"""
noise.py — noise-cancellation token scoring for MSE-GLM.

THE GOAL. Decide which tokens should pay attention to which, without letting common tokens
make important ones fire for meaningless reasons. No clustering, no stopword list -- just
sentence membership, widened to global knowledge in different ways (the "stages").

THE RULE: ATTENTION IS SYMMETRIC, AND COMMON TOKENS NEITHER GIVE NOR RECEIVE IT.
A token may only receive attention from a token it also gives attention to. Take

    the cat sat on the mat.
    the dog sat on the carpet.

"the", "sat" and "on" are COMMON to the two sentences, so they neither give nor receive
attention. What is left over -- the distinctive tokens -- attends to each other:

    * stages 1-3: distinctive tokens of the SAME sentence attend to each other
      (sentence 1: "cat" <-> "mat"; sentence 2: "dog" <-> "carpet"), one vote each;
    * stage 4:    distinctive tokens of TWO different sentences attend to each other.

Each token's row is how it distributes its attention over the other tokens, e.g. CAT gives
"mat" 1 and everything else 0; THE gives nothing to anybody.

STAGES 1-3 (per-PAIR votes). A candidate is any other token in the anchor's sentence. A
voter casts one vote for the PAIR (anchor, candidate) only if it knows NEITHER token -- a
voter that already knows one of them treats it as common. What "knows" means is what the
three stages differ in:

    stage 1 (literal):    a voter SENTENCE knows T iff T is literally one of its tokens.
    stage 2 (sentence-    a voter SENTENCE knows T iff ANY of its tokens has EVER shared a
      global):            sentence with T, anywhere in the corpus.
    stage 3 (token-       every DISTINCT token of the corpus votes (deduplicated, so a token
      global, deduped):   living in 300 sentences still counts once) iff it has never
                          co-occurred with either the anchor or the candidate.

In closed form, with rels(t) = the sentences holding t, knowing(t) = the sentences that know
t, reach(t) = every token that has ever shared a sentence with t (t included):

    stage1(a, c) = n_rels - |rels(a)    | rels(c)|
    stage2(a, c) = n_rels - |knowing(a) | knowing(c)|
    stage3(a, c) = |V_all| - |reach(a)  | reach(c)|

Each is symmetric in (a, c), and is zero for a pair as soon as one of the two tokens is
known everywhere ("the" scores 0 against everything). A pair is scored ONCE however many
sentences it shares (a voter that already voted for a pair does not vote again), so
nothing is ever summed across homes. The home sentences hold both tokens, so they sit
inside every union above and never vote. The three stages are separate RAW counts on
different scales (stage 3 counts tokens, not sentences) and are never added to each other
before normalization.

(Worked on the corpus above, stage 1: "cat"->"mat" = 1, because sentence 2 contains neither;
"cat"->"sat" = 0, because sentence 2 contains "sat".)

STAGE 4 (shared-knowledge attention; see shared_attention_scores()): for every OTHER
sentence, the tokens NOT common between it and one of the anchor's homes attend to each
other with weight = the number of tokens the two sentences share. Computed in closed form
(_shared_attention_fast()); the literal pair-by-pair algorithm is kept as
_shared_attention_literal().

COMBINING CONTEXT TOKENS AND NORMALIZING. Element-wise addition happens on RAW scores:

    1. for each context token, take its row of raw stage1..stage4 scores;
    2. add the rows together element-wise, target by target, stage by stage;
    3. per stage, normalize the SUM: total = sum over every target, factor = 100 / total,
       percentage = raw * factor (a target with raw 0 stays 0; a stage with total 0 stays 0);
    4. final(T) = (stage1% + stage2% + stage3% + stage4%) / 4.

Worked example, context "cat carpet": cat's row gives mat 1; carpet's row gives dog 1; the
sum is mat 1, dog 1; the total is 2; 100 / 2 = 50, so the context is about mat = 50% and
dog = 50% (on stage 1). A stage with no evidence contributes 0, it is not skipped, so the
final is always an average over four. This final is THE noise score generation uses (see
TokenVocabularyMatrix.combine_final_score() and model.noise_final_scores()). One word on its
own is the same thing with a single row.

MIXING WEIGHT: noise has no internal weight. model.py scales the final by
config.NoiseConfig.INFERENCE_WEIGHT when it adds it to ivm's score.

Punctuation tokens are NOT specially excluded; a sentence-final "." lands at 0 for the same
reason "the" does. Structural markers (PAD/UNK/BOS/EOS) ARE excluded outright -- they are
not tokens of the training text.

PERFORMANCE: stages 1-3 are bitmask/set arithmetic per pair (pair_stages(); a token's
knowing-mask is cached), stage 4 is closed-form from corpus-wide co-occurrence counts
(stage4_tables(); built once), and a token's raw row is cached the first time it is asked
for. Nothing is built at train time.
"""

from config import RESERVED, EOS, WORD_BOUND

_STRUCTURAL = RESERVED | {EOS}

# Number of set bits in a non-negative int. int.bit_count() needs Python
# 3.10+; the fallback keeps this module's "Python 3 only" promise.
_popcount = int.bit_count if hasattr(int, "bit_count") else (lambda x: bin(x).count("1"))


class NoiseCancellationIndex:
    """
    Precompute-once, query-many, same role as every other analysis
    index in this codebase: reuses the model's token_relationships
    directly for the gate (not reimplemented), and lazily caches each
    relationship's token set, each token's global reach, each token's
    knowing_rids set (stage 2), and each home's distinct-token pool
    (stage 3) on first use.

    It produces RAW stage counts only (stage1/stage2/stage3); turning
    them into the final score is normalize_row()'s job, because that
    needs a whole query's worth of candidates to normalize against.
    """

    def __init__(self, model, token_rels=None):
        self.model = model
        # token -> {relationship_id, ...}. `token_rels` lets a caller that already
        # holds this exact map hand it over instead of paying for a second
        # full pass over every triple; it is only ever READ here.
        self.token_rels = token_rels if token_rels is not None else model.token_relationships(min_len=3)
        self._rel_mask_cache = {}       # token -> int bitmask over relationship ids (see _rel_mask())
        self._knowing_mask_cache = {}   # token -> int bitmask of the sentences that know it -- see knowing_mask()
        self._rel_tokens_cache = {}
        self._global_reach_cache = {}   # token -> every token it has ever shared a sentence with
        self._knowing_rids_cache = {}   # token -> every relationship_id that knows it (stage 2's cache)
        self._other_vocab_cache = {}    # home_rid -> frozenset of distinct tokens across every OTHER relationship (stage 3's voter pool)
        self._other_vocab_multi_cache = {}   # frozenset(home_rids) -> same, generalized to a SET of excluded homes (stage 3's cross-home "vote once" pool)
        self._vocab_all_cache = None    # every distinct token across the WHOLE corpus, computed once (see _vocab_all())
        self._stage4_cache = None       # whole-corpus inverted index + co-occurrence counts for stage 4 (see stage4_tables())
        self._n_rels = model.n_rels
        self._all_rids = frozenset(range(self._n_rels))

    @classmethod
    def build(cls, model, token_rels=None):
        return cls(model, token_rels=token_rels)

    # ==== per-PAIR stage counts (the symmetric attention rule) ==============
    #
    # A voter votes "these two tokens should attend to each other" for the pair
    # (anchor, candidate) only if it knows NEITHER of them. A voter that knows the anchor, or
    # the candidate, treats that token as COMMON -- common tokens neither give nor receive
    # attention, so a token can only receive attention from a token it also gives attention to.
    # "Knows" is what the three stages differ in (literal / sentence-global / token-global):
    #     stage1(a, c) = n_rels - |rels(a) | rels(c)|
    #     stage2(a, c) = n_rels - |knowing_rids(a) | knowing_rids(c)|
    #     stage3(a, c) = |V_all| - |reach(a) | reach(c)|
    # Each is symmetric in (a, c). The home sentence(s) of the pair hold both tokens, so they are
    # inside every union above and never vote; for stage 3 the tokens that live only in those
    # homes are inside reach(a) & reach(c) too, so the "outside the home" voter pool of the old
    # per-home formulation collapses to |V_all| exactly as before.

    def _rel_mask(self, token):
        """Bitmask (Python int) with bit r set for every relationship r the
        token appears in -- lets 'union of many tokens' relationship sets'
        be a handful of big-int ORs instead of millions of set inserts."""
        cached = self._rel_mask_cache.get(token)
        if cached is None:
            rels = self.token_rels.get(token)
            if not rels:
                cached = 0
            else:
                buf = bytearray((self._n_rels >> 3) + 1)
                for r in rels:
                    buf[r >> 3] |= 1 << (r & 7)
                cached = int.from_bytes(buf, "little")
            self._rel_mask_cache[token] = cached
        return cached

    def knowing_mask(self, token):
        """
        Bitmask over relationship ids of every sentence that KNOWS `token` (== _knowing_rids(),
        as an int): the sentences holding the token itself or any token that has ever shared a
        sentence with it. Candidate-alone, so cached; the cache is capped (a mask is n_rels bits)
        and simply refilled when it overflows.
        """
        cached = self._knowing_mask_cache.get(token)
        if cached is not None:
            return cached
        reach = self.global_reach(token)
        order = reach
        if len(reach) > 64:
            order = sorted(reach, key=lambda u: -len(self.token_rels.get(u) or ()))
        full = (1 << self._n_rels) - 1
        knowing = 0
        for u in order:
            knowing |= self._rel_mask(u)
            if knowing == full:       # every sentence already knows it -- nothing left to add
                break
        cap = max(64, (64 << 20) // ((self._n_rels >> 3) + 1))
        if len(self._knowing_mask_cache) >= cap:
            self._knowing_mask_cache.clear()
        self._knowing_mask_cache[token] = knowing
        return knowing

    def pair_stages(self, anchor, cand):
        """
        (stage1, stage2, stage3) RAW counts for the attention pair (anchor, cand) -- see the
        block comment above. Symmetric: pair_stages(a, c) == pair_stages(c, a). Both tokens must
        be tokens of the corpus; a token with no sentences simply knows nothing.
        """
        n = self._n_rels
        ra = self.token_rels.get(anchor) or frozenset()
        rc = self.token_rels.get(cand) or frozenset()
        stage1 = n - (len(ra) + len(rc) - len(ra & rc))
        stage2 = n - _popcount(self.knowing_mask(anchor) | self.knowing_mask(cand))
        ga, gc = self.global_reach(anchor), self.global_reach(cand)
        stage3 = len(self._vocab_all()) - (len(ga) + len(gc) - len(ga & gc))
        return stage1, stage2, stage3

    def stage4_tables(self):
        """
        Whole-corpus tables the closed-form stage 4 (see _shared_attention_fast()) needs, built
        ONCE on first use and cached on the index:
            invsz[t]   number of sentences whose token set (minus <WORD_BOUND>) holds t
            inv[t]     the set of those sentences
            co[t][c]   number of sentences holding BOTH t and c (t != c)
        Built from tokens_for() over every sentence, so they describe exactly the token sets the
        literal algorithm iterates. Cost is one pass over the sentences, O(sum of len^2).
        """
        if self._stage4_cache is None:
            inv, co = {}, {}
            tokens_for = self.tokens_for
            for rid in range(self._n_rels):
                toks = tokens_for(rid) - {WORD_BOUND}
                for t in toks:
                    inv.setdefault(t, set()).add(rid)
                    row = co.get(t)
                    if row is None:
                        row = co[t] = {}
                    for c in toks:
                        if c != t:
                            row[c] = row.get(c, 0) + 1
            self._stage4_cache = {"inv": inv, "co": co,
                                  "invsz": {t: len(r) for t, r in inv.items()}}
        return self._stage4_cache

    def tokens_for(self, rid):
        cached = self._rel_tokens_cache.get(rid)
        if cached is None:
            cached = set(self.model.sequence(rid)) - _STRUCTURAL
            self._rel_tokens_cache[rid] = cached
        return cached

    def global_reach(self, token):
        """
        Every token that has EVER shared a training sentence with
        `token`, anywhere in the corpus -- token's global
        co-occurrence neighborhood, the building block both
        _knowing_rids() (stage 2) and stage 3's per-candidate check
        are built from. Cached like tokens_for(), since the same
        token's neighborhood gets reused every time it's scored as a
        candidate again.

            global_reach(T) = union_{rel in token_rels[T]} tokens_for(rel)

        t is in this set iff t and T co-occur in at least one
        relationship_id -- exactly "does t know T". token_rels[T] is
        reused directly (not recomputed), same convention as
        everywhere else in this class.
        """
        cached = self._global_reach_cache.get(token)
        if cached is None:
            cached = set()
            for rel in self.token_rels.get(token, ()):
                cached |= self.tokens_for(rel)
            self._global_reach_cache[token] = cached
        return cached

    def _knowing_rids(self, token):
        """
        STAGE 2's optimization (see module docstring): every
        relationship_id that KNOWS `token` -- directly (token itself
        sits there) or through a neighbor (some token in
        global_reach(token) sits there). Depends only on `token`
        itself, never on which anchor or home is asking, so it's
        computed once per token and cached forever after -- every
        later counts_detailed()/voters_for() call that scores this same token
        as a candidate reuses this set instead of rebuilding the
        union from scratch.

            knowing_rids(T) = union_{t in global_reach(T)} token_rels[t]

        T's own home sentence(s) are always inside this set (T's own
        relationship_id(s) are among the ones unioned in, since T is
        trivially a member of its own global_reach), so a specific
        home_rid is always correctly excludable via the same
        -1-if-member trick counts_detailed() uses below -- never double
        counted, never missed.
        """
        cached = self._knowing_rids_cache.get(token)
        if cached is None:
            merged = set()
            for t in self.global_reach(token):
                merged |= self.token_rels.get(t, set())
            cached = frozenset(merged)
            self._knowing_rids_cache[token] = cached
        return cached

    def _vocab_all(self):
        """
        Every distinct token across the WHOLE corpus -- every
        relationship included, home or not -- computed once, ever,
        and reused by _other_vocab() for every home. This is what
        turns _other_vocab() into an O(|home's own tokens|) operation
        per NEW home instead of O(n_rels): see _other_vocab()'s
        docstring for the exact reformulation this enables.
        """
        if self._vocab_all_cache is None:
            merged = set()
            for rid in self._all_rids:
                merged |= self.tokens_for(rid)
            self._vocab_all_cache = frozenset(merged)
        return self._vocab_all_cache

    def _other_vocab(self, home_rid):
        """
        STAGE 3's fixed voter POOL for one home_rid: the set of every
        DISTINCT token that appears in some relationship OTHER than
        home_rid -- not a per-occurrence count. A token sitting in
        300 other sentences is still exactly one voter here; it casts
        one verdict about itself, once, regardless of how many
        sentences it happens to occupy (this is the dedup fix from
        this module's earlier three-column era -- without it, a
        common token could inflate/deflate a candidate's stage-3
        count purely by recurring often, unrelated to any actual
        signal about that candidate).

        OPTIMIZATION: the original implementation unioned every OTHER
        relationship's tokens from scratch, per home -- O(n_rels)
        EVERY time a NEW home_rid showed up, which is exactly what
        made TokenVocabularyMatrix.combine()/row() slow for any token
        with many homes (a frequent word visits many homes, each one
        re-paying that full O(n_rels) union). Instead, this starts
        from _vocab_all() (every token in the whole corpus, computed
        ONCE, ever) and removes only whichever of home_rid's OWN
        tokens are EXCLUSIVE to it -- appear in no relationship other
        than home_rid:

            other_vocab(home_rid) = vocab_all() - {t in tokens_for(home_rid) : token_rels[t] == {home_rid}}

        This is an EXACT reformulation, not an approximation: a token
        only needs removing from the whole-corpus set if home_rid is
        the ONLY place it appears anywhere -- any token that shows up
        in even one other relationship stays in the union regardless
        of whether it also happens to sit in home_rid too. Cost per
        NEW home_rid is O(|tokens_for(home_rid)|) -- typically a
        handful of words -- not O(n_rels). Cached per home_rid, since
        every candidate scored against the same home reuses the same
        pool.
        """
        cached = self._other_vocab_cache.get(home_rid)
        if cached is None:
            exclusive = {t for t in self.tokens_for(home_rid)
                         if self.token_rels.get(t) == {home_rid}}
            cached = self._vocab_all() - exclusive
            self._other_vocab_cache[home_rid] = cached
        return cached

    def _other_vocab_multi(self, home_rids):
        """
        Generalizes _other_vocab() from ONE excluded home to a SET of
        excluded homes -- STAGE 3's "vote once per (anchor, token)
        pair" voter pool for cross-home accumulation (see module
        docstring's "VOTE ONCE PER PAIR"). When a candidate T recurs
        across several of the anchor's homes, the correct pool for T's
        ONE combined stage3 count treats ALL of those homes as a
        single internal region at once -- not one home at a time,
        summed -- and removes only whichever tokens are EXCLUSIVE to
        that whole region (appear nowhere else in the corpus):

            other_vocab_multi(H) = vocab_all() - {t : token_rels[t] <= H}

        Every token that shows up in even ONE relationship outside H
        stays in the pool and is counted exactly ONCE, no matter how
        many homes in H it happens to also sit in -- that's the "vote
        once" fix. With |H| == 1 this is identical to _other_vocab()
        (a token's relationship set is a subset of a single-element
        set only if it equals that set exactly), so this is a strict
        generalization, not a different rule.

        Cached per frozenset(H): the same recurrence pattern (the same
        set of homes a candidate appears in) is often shared by more
        than one candidate, so this is reused across candidates with
        identical T_homes, not recomputed per candidate.
        """
        home_rids = frozenset(home_rids)
        cached = self._other_vocab_multi_cache.get(home_rids)
        if cached is None:
            exclusive = set()
            for h in home_rids:
                for t in self.tokens_for(h):
                    if self.token_rels.get(t, set()) <= home_rids:
                        exclusive.add(t)
            cached = self._vocab_all() - exclusive
            self._other_vocab_multi_cache[home_rids] = cached
        return cached

    def homes(self, anchor):
        """Every relationship_id whose sentence contains `anchor` -- candidate 'home' sentences to score against."""
        return sorted(self.token_rels.get(anchor, set()))

    def counts_detailed(self, anchor, home_rid):
        """
        The three RAW stage counts for every candidate in one (anchor,
        home) pair -- the single source of truth for that arithmetic
        (score() and noise_scores_detailed() both build on it). There is
        no combined number: the stages are on different scales and are
        only made comparable by normalize_row().

        A candidate is any other token of the home sentence. A voter counts for the pair
        (anchor, candidate) only if it knows NEITHER token (see pair_stages()), so common
        tokens neither give nor receive attention and the pair scores the same from either side:

            stage1 = n_rels - |token_rels[anchor] | token_rels[T]|
            stage2 = n_rels - |knowing_rids(anchor) | knowing_rids(T)|
            stage3 = |V_all| - |global_reach(anchor) | global_reach(T)|

            {"anchor":.., "home":.., "candidates": {...}, "total_voters":..,
             "scores": {token: {"stage1": n, "stage2": n, "stage3": n}}}
        """
        home_tokens = self.tokens_for(home_rid) - {anchor}
        scores = {}
        for cand in home_tokens:
            stage1, stage2, stage3 = self.pair_stages(anchor, cand)
            scores[cand] = {"stage1": stage1, "stage2": stage2, "stage3": stage3}
        return {"anchor": anchor, "home": home_rid, "candidates": home_tokens,
                "total_voters": self._n_rels - 1, "scores": scores}

    def voters_for(self, anchor, home_rid, candidate):
        """
        THE independent "which specific sentences voted yes for this
        one candidate" question -- on demand only, same fast-count/
        slow-evidence split (a cheap count for scoring, the specific evidence only on demand).
        STAGE 2 ONLY: this reports stage 2's specific relationship-id
        voters, not a receipt for all three stages. Stage 1's voters would be a different (smaller-or-
        equal) set of the same relationship-id type; stage 3's voters
        are individual TOKENS, not relationship ids at all, so there
        is no single list that could represent all three stages'
        receipts at once -- stage 2 is reported here because it's the
        widest of the two relationship-level rules. Built from the
        cached knowing masks of the anchor and the candidate (see knowing_mask()): a sentence
        is a voter only if it knows neither of them.
        """
        # a sentence votes only if it knows NEITHER the anchor nor the candidate
        knowing = self.knowing_mask(anchor) | self.knowing_mask(candidate)
        return [r for r in range(self._n_rels) if r != home_rid and not (knowing >> r) & 1]

    def score(self, anchor, home_rid):
        """
        Full table INCLUDING per-candidate voter lists -- for callers
        that want receipts immediately for ONE (anchor, home) pair
        (e.g. noise_scores()). Built from counts_detailed() (the three
        raw stage counts) plus one voters_for() call per candidate --
        materializing K full lists of up to N items each is inherently
        more expensive than the plain counts, but that cost is paid
        only for a single explicit query, never inside a bulk
        operation.

        NOTE: "voters" is STAGE 2 ONLY (see voters_for()), so
        len(voters) equals stage2's count only up to the home sentence's
        own exclusion -- it is a receipt for one stage, not for all
        three. A real, documented limit: stage 3's voters are tokens,
        not relationship ids.

            {"anchor": anchor, "home": home_rid,
             "candidates": {token, ...},
             "voters": [relationship_id, ...],     # every OTHER relationship_id, unconditionally
             "scores": {token: {"stage1": n, "stage2": n, "stage3": n,
                                "voters": [relationship_id, ...]}}}
        """
        raw = self.counts_detailed(anchor, home_rid)
        scores = {cand: {**info, "voters": self.voters_for(anchor, home_rid, cand)}
                  for cand, info in raw["scores"].items()}
        return {"anchor": anchor, "home": home_rid, "candidates": raw["candidates"],
                "voters": sorted(self._all_rids - {home_rid}), "scores": scores}


# ====== word-level convenience =======================================================

def _resolve(model, word):
    """word -> token id, same convention used throughout this codebase's analysis layer."""
    ids = [t for t in model.encode(word) if t != 2]  # 2 == BOS
    return ids[-1] if ids else None


def find_homes(model, anchor_word):
    """
    Every relationship_id whose training sentence contains
    anchor_word, decoded for display -- use this to pick a
    home_relationship_id for noise_scores() when the anchor word
    appears in more than one training sentence. Auto-builds/caches
    model.noise_index on first use.
    """
    if model.noise_index is None:
        model.build_noise_index()
    anchor = _resolve(model, anchor_word)
    if anchor is None:
        return []
    return [{"relationship_id": rid, "sentence": model.decode(model.sequence(rid))}
            for rid in model.noise_index.homes(anchor)]


def noise_scores(model, anchor_word, home_relationship_id=None):
    """
    THE word-level entry point for noise.py -- per candidate, the three
    raw stage counts plus stage 2's voter receipts, for one anchor+home.

    home_relationship_id is REQUIRED if anchor_word appears in more
    than one training sentence -- never silently picked for you; use
    find_homes() to list the options first. If anchor_word appears in
    exactly one training sentence, home_relationship_id can be
    omitted and that one sentence is used.

    Returns None if anchor_word was never trained. Raises ValueError
    if home_relationship_id is required but missing, or doesn't
    actually contain anchor_word. Otherwise:

        {"anchor": word, "home": relationship_id, "home_sentence": str,
         "voter_pool": [relationship_id, ...],  # every OTHER relationship_id -- the fixed voter roster
         "scores": {token: {"stage1": n, "stage2": n, "stage3": n,
                            "voters": [relationship_id, ...]}, ...}}

    Everything is decoded to words, not token ids.
    """
    if model.noise_index is None:
        model.build_noise_index()
    idx = model.noise_index
    anchor = _resolve(model, anchor_word)
    if anchor is None:
        return None

    homes = idx.homes(anchor)
    if not homes:
        return None
    if home_relationship_id is None:
        if len(homes) > 1:
            raise ValueError(
                f"'{anchor_word}' appears in {len(homes)} training sentences "
                f"({homes}) -- pass home_relationship_id to pick one; see find_homes()."
            )
        home_relationship_id = homes[0]
    elif home_relationship_id not in homes:
        raise ValueError(f"relationship {home_relationship_id} doesn't contain '{anchor_word}'; "
                          f"it appears in {homes}")

    full = idx.score(anchor, home_relationship_id)   # counts + voter receipts

    dec = lambda t: model.decode([t])
    scores = {dec(cand): dict(info) for cand, info in full["scores"].items()}
    return {
        "anchor": anchor_word,
        "home": home_relationship_id,
        "home_sentence": model.decode(model.sequence(home_relationship_id)),
        "voter_pool": full["voters"],
        "scores": scores,
    }


def noise_scores_detailed(model, anchor_word, home_relationship_id=None):
    """
    Same anchor+home resolution rules as noise_scores(), but without
    the voter receipts: each candidate's three raw stage values (see
    NoiseCancellationIndex.counts_detailed()) -- what analyse.py's
    noise-scores table displays as separate columns.

        {"anchor": word, "home": relationship_id, "home_sentence": str,
         "scores": {token: {"stage1": n, "stage2": n, "stage3": n}, ...}}
    """
    if model.noise_index is None:
        model.build_noise_index()
    idx = model.noise_index
    anchor = _resolve(model, anchor_word)
    if anchor is None:
        return None

    homes = idx.homes(anchor)
    if not homes:
        return None
    if home_relationship_id is None:
        if len(homes) > 1:
            raise ValueError(
                f"'{anchor_word}' appears in {len(homes)} training sentences "
                f"({homes}) -- pass home_relationship_id to pick one; see find_homes()."
            )
        home_relationship_id = homes[0]
    elif home_relationship_id not in homes:
        raise ValueError(f"relationship {home_relationship_id} doesn't contain '{anchor_word}'; "
                          f"it appears in {homes}")

    detailed = idx.counts_detailed(anchor, home_relationship_id)
    dec = lambda t: model.decode([t])
    return {
        "anchor": anchor_word,
        "home": home_relationship_id,
        "home_sentence": model.decode(model.sequence(home_relationship_id)),
        "scores": {dec(cand): dict(info) for cand, info in detailed["scores"].items()},
    }


def focus_scores(model, anchor_word, weight_mode="count"):
    """
    Global, cross-sentence attention profile for anchor_word, reduced to
    ONE number per candidate: its FINAL score (the average of the four
    normalized stages -- see module docstring's "THE FINAL SCORE").
    Finds every candidate across EVERY training sentence the anchor
    appears in, scores each ONCE ("VOTE ONCE PER PAIR"), and normalizes
    scoped to this one anchor's candidates, exactly what focus-scores
    shows by default.

    Returns None if anchor_word was never trained. Otherwise:
        {"anchor": word, "homes": [relationship_id, ...],
         "scores": {token: {"total": n}}}
    "total" is what you sort by to answer "which tokens should I focus
    on most when I see this anchor, anywhere?" -- a percentage-scale
    number (see focus_scores_detailed() for the four stages behind it).
    """
    detailed = focus_scores_detailed(model, anchor_word, weight_mode=weight_mode)
    if detailed is None:
        return None
    normalized = normalize_row(detailed["scores"])
    return {"anchor": anchor_word, "homes": detailed["homes"],
            "scores": {w: {"total": info["final"]} for w, info in normalized.items()}}


def _stage4_by_id(model, idx, anchor, homes, weight_mode="count"):
    """{token_id: stage-4 weight} for one anchor. The default (count) comes straight from the
    closed form; weight_mode="flat" goes through the literal algorithm (word-keyed) and is
    mapped back to ids."""
    if weight_mode == "count":
        return {c: w for c, (w, _p) in _shared_attention_fast(idx, anchor, homes).items()}
    literal = _shared_attention_literal(model, model.decode([anchor]), weight_mode=weight_mode)
    out = {}
    for word, info in (literal["scores"] if literal else {}).items():
        tid = _resolve(model, word)
        if tid is not None:
            out[tid] = info["weight"]
    return out


def _focus_rows(model, anchor, weight_mode="count"):
    """
    The RAW stage1..stage4 row of one anchor TOKEN ID, keyed by candidate token id -- the single
    place the four stages are assembled (focus_scores_detailed() and TokenVocabularyMatrix both
    wrap it). Returns None if the anchor has no home sentence, else (homes, rows) with
    rows = {candidate_id: {"stage1": n, "stage2": n, "stage3": n, "stage4": n}}.

    Stages 1-3 are scored for every other token of the anchor's own sentences, as a PAIR rule
    (see pair_stages()); stage 4 may add tokens that never share a sentence with the anchor.
    """
    if model.noise_index is None:
        model.build_noise_index()
    idx = model.noise_index
    homes = idx.homes(anchor)
    if not homes:
        return None
    candidates = set()
    for home in homes:
        candidates |= idx.tokens_for(home)
    candidates.discard(anchor)
    rows = {}
    for cand in candidates:
        stage1, stage2, stage3 = idx.pair_stages(anchor, cand)
        rows[cand] = {"stage1": stage1, "stage2": stage2, "stage3": stage3, "stage4": 0}
    for cand, weight in _stage4_by_id(model, idx, anchor, homes, weight_mode).items():
        entry = rows.get(cand)
        if entry is None:
            entry = rows[cand] = {"stage1": 0, "stage2": 0, "stage3": 0, "stage4": 0}
        entry["stage4"] = weight
    return homes, rows


def focus_scores_detailed(model, anchor_word, weight_mode="count"):
    """
    Global, cross-sentence attention profile for anchor_word, with all
    four stages broken out as separate RAW numbers -- what analyse.py's
    focus-scores --raw table displays as columns. Raw stages are on
    different scales and are never added to each other; normalize_row() turns
    them into percentages and the final score.

    stage1/stage2/stage3: "who attends to whom inside the same sentence" -- the symmetric
    pair rule in the module docstring (a voter counts for the pair (anchor, token) only if it
    knows neither). Scored only for tokens that share at least one of the anchor's own sentences.

    stage4: shared-knowledge attention -- see shared_attention_scores().
    For every OTHER sentence (whether or not it shares a home with the
    anchor), the tokens NOT common between that sentence and one of the
    anchor's homes attend to each other with weight = number of common
    tokens (weight_mode="count", default) or a flat 1 per pair
    (weight_mode="flat"); common tokens score 0. Points accumulate across
    every sentence pair a token pair appears in (agg="sum"). This is an
    independent rule, not a stage1-3 vote.

    A token can carry stage1-3 only (never reached by stage4, e.g. it's
    always a common token when paired with the anchor's homes), stage4
    only (never a home-mate of the anchor), or both. Missing stages are 0.

    Returns None if anchor_word was never trained. Otherwise:
        {"anchor": word, "homes": [relationship_id, ...],
         "scores": {token: {"stage1": n, "stage2": n, "stage3": n,
                             "stage4": n}, ...}}
    """
    anchor = _resolve(model, anchor_word)
    if anchor is None:
        return None
    got = _focus_rows(model, anchor, weight_mode=weight_mode)
    if got is None:
        return None
    homes, rows = got
    dec = lambda t: model.decode([t])
    return {"anchor": anchor_word, "homes": homes,
            "scores": {dec(c): info for c, info in rows.items()}}


def pair_pool_scores_detailed(model, rid_x, rid_y):
    """
    Extends focus_scores()'s "score every candidate across every home
    an anchor lives in" to a candidate pool drawn from TWO SPECIFIC
    sentences instead of one anchor's homes: every token in EITHER
    rid_x or rid_y is a candidate, scored with BOTH rid_x and rid_y
    excluded from the voter pool -- not just whichever one the
    candidate happens to literally live in, the way a plain single-
    home score would only exclude that one home.

    This is the whole extension: the same three stage formulas
    NoiseCancellationIndex already uses everywhere else
    (token_rels/_knowing_rids/_other_vocab_multi/global_reach,
    untouched), just handed an explicit two-relationship region
    instead of one anchor token's own home set. Stage 1 was validated
    against a hand-worked multi-sentence example (see
    test_pair_pool_scores() in test_pair_attention.py); stage 2/3 are
    the same formulas focus_scores_detailed() already uses for its own
    multi-home region.

    Returns the three RAW stage counts per candidate -- no combined
    number (the stages are on different scales and are not addable):
        {"rid_x": int, "rid_y": int, "pool": [token, ...],
         "scores": {token: {"stage1": n, "stage2": n, "stage3": n}}}
    """
    if model.noise_index is None:
        model.build_noise_index()
    idx = model.noise_index
    homes_set = frozenset({rid_x, rid_y})
    pool = idx.tokens_for(rid_x) | idx.tokens_for(rid_y)
    other_vocab = idx._other_vocab_multi(homes_set)

    dec = lambda t: model.decode([t])
    scores = {}
    for cand in pool:
        cand_rels = idx.token_rels.get(cand, set())
        stage1 = idx._n_rels - len(cand_rels | homes_set)
        knowing_rids = idx._knowing_rids(cand)
        stage2 = idx._n_rels - len(knowing_rids | homes_set)
        stage3 = len(other_vocab) - len(other_vocab & idx.global_reach(cand))
        scores[dec(cand)] = {"stage1": stage1, "stage2": stage2, "stage3": stage3}
    return {"rid_x": rid_x, "rid_y": rid_y, "pool": [dec(c) for c in pool], "scores": scores}


def pair_pool_stage1(model, rid_x, rid_y):
    """
    STAGE 1 ONLY view of pair_pool_scores_detailed() -- same pool
    (tokens_for(rid_x) | tokens_for(rid_y)) and same voter exclusion
    (every relationship except rid_x and rid_y), but keeps just the
    literal-co-occurrence vote instead of also showing stage 2/3. This
    is the piece being validated against hand-worked multi-sentence
    examples before the |shared(rid_x, rid_y)| weighting step is
    added on top of it -- see pair_pool_scores_detailed()'s docstring for the
    full three-stage mechanism this isolates one piece of.

        stage1(T) = n_rels - |token_rels[T] | {rid_x, rid_y}|

    A candidate T's count is exactly "how many relationships OTHER
    than rid_x/rid_y do not literally contain T" -- both rid_x and
    rid_y are excluded from the voter pool unconditionally, even for
    a candidate that isn't literally one of their own tokens (the
    `| homes_set` union, not a per-candidate membership check), so
    the two sentences forming the pair never vote on their own pair's
    candidates.

    Returns {"rid_x": int, "rid_y": int, "pool": [token, ...],
             "scores": {token: n}}
    """
    if model.noise_index is None:
        model.build_noise_index()
    idx = model.noise_index
    homes_set = frozenset({rid_x, rid_y})
    pool = idx.tokens_for(rid_x) | idx.tokens_for(rid_y)
    dec = lambda t: model.decode([t])
    scores = {}
    for cand in pool:
        cand_rels = idx.token_rels.get(cand, set())
        stage1 = idx._n_rels - len(cand_rels | homes_set)
        scores[dec(cand)] = stage1
    return {"rid_x": rid_x, "rid_y": rid_y, "pool": [dec(c) for c in pool], "scores": scores}


def shared_attention_scores(model, anchor_word, agg="sum", weight_mode="count"):
    """
    Stage 4 -- shared-knowledge attention (full description in _shared_attention_literal()).

    The default (agg="sum", weight_mode="count") -- the only form inference ever uses -- is
    computed in closed form from corpus-wide co-occurrence counts instead of by walking every
    (home, other sentence) pair, so its cost no longer grows with (number of sentences) x (how
    often the anchor occurs). The result is IDENTICAL to the literal pair-by-pair algorithm,
    which stays as _shared_attention_literal() -- used for agg="max" / weight_mode="flat" and as
    the correctness reference the fast path is checked against.
    """
    if agg not in ("max", "sum"):
        raise ValueError("agg must be 'max' or 'sum'")
    if weight_mode not in ("count", "flat"):
        raise ValueError("weight_mode must be 'count' or 'flat'")
    if agg != "sum" or weight_mode != "count":
        return _shared_attention_literal(model, anchor_word, agg=agg, weight_mode=weight_mode)
    if model.noise_index is None:
        model.build_noise_index()
    idx = model.noise_index
    anchor = _resolve(model, anchor_word)
    if anchor is None:
        return None
    homes = idx.homes(anchor)
    if not homes:
        return None
    fast = _shared_attention_fast(idx, anchor, homes)
    dec = lambda t: model.decode([t])
    return {"anchor": anchor_word, "homes": homes,
            "scores": {dec(c): {"weight": w, "pairs": p} for c, (w, p) in fast.items()}}


def _shared_attention_fast(idx, anchor, homes):
    """
    Closed form of the literal stage-4 loop for agg="sum", weight_mode="count".
    Returns {token_id: (weight, pairs)} for every token with pairs > 0.

    Notation: tx' = a home's tokens minus the anchor; R' = sentences NOT containing the anchor;
    ov(x, y) = |tx' & ty|; S_x = sum over y in R' of ov(x, y); co(t, c) = sentences holding both;
    m_t = homes holding t; h(t, c) = homes holding both; g_t / gc(t, c) = the same counts over ALL
    sentences containing the anchor. For a candidate c (c != anchor), summing the literal rule
    "a sentence pair adds |common| to c when c is in exactly one of the two sentences" over every
    home x and y in R' gives
        weight(c) = sum_{x has c} S_x + sum_t m_t*co_R(t,c) - 2 * sum_t h(t,c)*co_R(t,c)
        pairs(c)  = |R'|*m_c + (|homes| - m_c) * (|inv c| - g_c)
    with co_R(t, c) = co(t, c) - gc(t, c) (pairs counted in sentences without the anchor).
    Sentences that contain the anchor but are NOT homes (sentences too short to hold a triple)
    are still "other sentences" in the literal loop, so they are added pair by pair at the end.
    """
    tb = idx.stage4_tables()
    inv, co, invsz = tb["inv"], tb["co"], tb["invsz"]
    tokens_for = idx.tokens_for
    anchor_rels = idx.token_rels.get(anchor, set())
    A = inv.get(anchor, set())                  # every sentence holding the anchor
    Z = [r for r in A if r not in anchor_rels]  # ...that the literal loop still treats as 'other'
    n_plain = idx._n_rels - len(A)              # |R'|
    if idx._n_rels - len(anchor_rels) == 0:
        return {}
    drop = {WORD_BOUND, anchor}
    hx = [tuple(tokens_for(x) - drop) for x in homes]
    homes_set = set(homes)
    ax = hx + [tuple(tokens_for(z) - drop) for z in A if z not in homes_set]   # all of A
    n_homes = len(hx)

    m = {}
    for xs in hx:
        for t in xs:
            m[t] = m.get(t, 0) + 1
    g = {}
    for zs in ax:
        for t in zs:
            g[t] = g.get(t, 0) + 1

    K = (max(invsz) + 1) if invsz else 1
    weight1, part1, part2, part3 = {}, {}, {}, {}
    hc = {}                                     # (t*K + c) -> h(t, c), incl. t == c
    for xs in hx:
        sx = 0
        for t in xs:
            sx += invsz[t] - g[t]
        for t in xs:
            weight1[t] = weight1.get(t, 0) + sx
            base = t * K
            for c in xs:
                key = base + c
                hc[key] = hc.get(key, 0) + 1
    # sum_t m_t * co(t, c), over every t that sits in a home
    for t, mt in m.items():
        part1[t] = part1.get(t, 0) + mt * invsz[t]            # co(c, c) = |inv c|
        for c, v in co[t].items():
            if c != anchor:
                part1[c] = part1.get(c, 0) + mt * v
    # sum_t m_t * gc(t, c), over every sentence holding the anchor
    for zs in ax:
        s_m = 0
        for t in zs:
            s_m += m.get(t, 0)
        if s_m:
            for c in zs:
                part2[c] = part2.get(c, 0) + s_m
    # gc(t, c): the same pair counts, over every sentence holding the anchor
    if len(ax) == n_homes:
        gc = hc
    else:
        gc = dict(hc)
        for zs in ax[n_homes:]:
            for t in zs:
                base = t * K
                for c in zs:
                    key = base + c
                    gc[key] = gc.get(key, 0) + 1
    for key, h in hc.items():
        t, c = divmod(key, K)
        cotc = invsz[t] if t == c else co[t].get(c, 0)
        part3[c] = part3.get(c, 0) + h * (cotc - gc[key])

    out = {}
    for c, size in invsz.items():
        if c == anchor:
            continue
        mc = m.get(c, 0)
        pairs = n_plain * mc + (n_homes - mc) * (size - g.get(c, 0))
        if pairs <= 0:
            continue
        out[c] = [weight1.get(c, 0) + part1.get(c, 0) - part2.get(c, 0) - 2 * part3.get(c, 0),
                  pairs]

    # Sentences holding the anchor that are not homes: literal pair-by-pair contribution.
    for z in Z:
        tz = tokens_for(z) - {WORD_BOUND}
        for x in homes:
            tx = tokens_for(x) - {WORD_BOUND}
            w = len(tx & tz)
            left = (tx ^ tz) - {anchor}
            for c in tx | tz:
                if c == anchor:
                    continue
                entry = out.setdefault(c, [0, 0])
                entry[1] += 1
                if c in left:
                    entry[0] += w
    return {c: (w, p) for c, (w, p) in out.items() if p > 0}


def _shared_attention_literal(model, anchor_word, agg="sum", weight_mode="count"):
    """
    Stage 4 -- "shared-knowledge attention", no voting, independent of
    stage1/2/3.

    For two sentences x and y (token SETS):
        common = tokens_for(x) & tokens_for(y)     what they know together
        left   = tokens_for(x) ^ tokens_for(y)     everything NOT common,
                                                   from BOTH sentences
    Every token in `left` attends to every other token in `left` -- across
    the two sentences AND within each one -- with weight |common|. Common
    tokens get attention 0.

        "the cat sat on the mat"  /  "the boy sit on mat"
        common = {the, on, mat}  -> weight 3
        left   = {cat, sat, boy, sit}
        focus "boy" -> cat 3, sat 3, sit 3, the 0, on 0, mat 0

    No stage 1/2/3, no voters -- the old "who doesn't know this word"
    voting rewarded rare, unrelated tokens.

    Scoring for one anchor: every sentence pair (x, y) where x is a home of
    the anchor and y is any other sentence. If the anchor is itself common
    to x and y it is subtracted (zero attention), so that pair is skipped.
    The sentences are the training environment; the points accumulate on
    the token pair. Every sentence pair that puts anchor and candidate in
    its "not common" side adds |common| to that pair's score, so a token
    pair that keeps meeting in different sentence pairs keeps strengthening:
        S1 "the cat sat on the mat"   S2 "the boy sit on mat"
        S3 "the boy ran on the road because he wanted to be fit"
        cat<->sat : 3 from (S1,S2) + 2 from (S1,S3) = 5
    `agg="sum"` (default) adds every pair's weight; `agg="max"` keeps only
    the strongest single pair (no accumulation).

    weight_mode controls what a single sentence pair's weight IS, before
    agg combines it across pairs:
        "count" (default) -- weight = |common| (the rule above)
        "flat"            -- weight = 1 for every pair that has at least
                              one common token to score against (0 stays
                              0 for common tokens themselves, same as
                              always); a pair with NO common tokens still
                              contributes 0, since there's nothing to
                              anchor "not common" against
    "flat" isolates the effect of scaling by |common| -- every sentence
    pair a token pair meets in counts equally, instead of pairs that
    share more context counting for more.

    Returns None if anchor_word was never trained. Otherwise:
        {"anchor": word, "homes": [...],
         "scores": {token: {"weight": w, "pairs": n_sentence_pairs}}}
    (common tokens appear with weight 0; <WORD_BOUND> is ignored.)
    """
    if agg not in ("max", "sum"):
        raise ValueError("agg must be 'max' or 'sum'")
    if weight_mode not in ("count", "flat"):
        raise ValueError("weight_mode must be 'count' or 'flat'")
    if model.noise_index is None:
        model.build_noise_index()
    idx = model.noise_index
    anchor = _resolve(model, anchor_word)
    if anchor is None:
        return None
    homes = idx.homes(anchor)
    if not homes:
        return None
    anchor_rels = idx.token_rels.get(anchor, set())
    dec = lambda t: model.decode([t])

    # <WORD_BOUND> is a structural marker present in every sentence, not a
    # token from the text: counting it would add a constant +1 to every
    # weight and list a blank row. Everything else (punctuation included)
    # counts as a token.
    def _toks(rid):
        return idx.tokens_for(rid) - {WORD_BOUND}

    weight, pairs = {}, {}
    for x in homes:
        tx = _toks(x)
        for y in idx._all_rids - anchor_rels:   # anchor not common => anchor in left
            ty = _toks(y)
            common = tx & ty
            w = len(common) if weight_mode == "count" else (1 if common else 0)
            left = (tx ^ ty) - {anchor}
            for cand in tx | ty:
                if cand == anchor:
                    continue
                cw = w if cand in left else 0     # common tokens get 0
                pairs[cand] = pairs.get(cand, 0) + 1
                if agg == "sum":
                    weight[cand] = weight.get(cand, 0) + cw
                else:
                    weight[cand] = max(weight.get(cand, 0), cw)

    scores = {dec(c): {"weight": weight[c], "pairs": pairs[c]} for c in weight}
    return {"anchor": anchor_word, "homes": homes, "scores": scores}


def normalize_row(scores):
    """
    Turn one query's raw {candidate: {"stage1","stage2","stage3","stage4"}}
    dict (from focus_scores_detailed() or combine_context_detailed())
    into percentages, SCOPED TO THIS QUERY (one word's row, or the element-wise SUM of several
    words' raw rows -- never rows that were already normalized): for stage1/stage2/stage3/
    stage4 independently,
        total   = sum of that stage's raw value across every candidate
                  in `scores` (not the whole vocabulary)
        factor  = 100 / total
        normalized = raw * factor
    A candidate's normalized value is 0 only when its own raw value was
    0 -- a positive raw value always produces a positive normalized
    value, since factor is a single fixed positive number for that
    stage. If every candidate is 0 on a stage, that stage's total is 0
    and everyone stays 0 (no divide-by-zero).

    final -- THE noise score: what generation actually uses (see
    TokenVocabularyMatrix.combine_final_score() / model.noise_final_scores()) --
    is the AVERAGE OF ALL FOUR normalized stages:

        final = (stage1% + stage2% + stage3% + stage4%) / 4

    All four are on the same percentage scale after normalization (that
    is the whole reason normalization exists: the raw counts' scales
    differ by construction, so raw stages are never averaged). A stage
    a candidate has no evidence for contributes 0, it is not skipped, so
    a candidate scored by only some stages is averaged over four, not
    over however many it happens to have.

    Returns a new dict, same shape as `scores`, values replaced by
    percentages (0-100, not fractions), plus "final". There is no
    stage1-3 "average" field.
    """
    keys = ("stage1", "stage2", "stage3", "stage4")
    totals = {k: sum(info.get(k, 0) for info in scores.values()) for k in keys}
    out = {}
    for cand, info in scores.items():
        row = {k: (info.get(k, 0) / totals[k] * 100) if totals[k] else 0.0
               for k in keys}
        row["final"] = sum(row[k] for k in keys) / 4
        out[cand] = row
    return out


class TokenVocabularyMatrix:
    """
    The cache of accumulated token knowledge: each token's RAW stage row --
    {target: (stage1, stage2, stage3, stage4)} -- built from _focus_rows(). Not persisted to
    disk; same convention as ivm.py / this module's own NoiseCancellationIndex.

    Rows are RAW on purpose: when several context tokens are combined their rows are added
    element-wise as raw scores and only the SUM is normalized (see combine_final_score()), so
    a row is never normalized before it is combined.

    LAZY, not eager: a row is computed the FIRST time that specific token is actually asked
    for (via row(), combine() or combine_final_score()), then cached forever after -- never
    for every token in the corpus up front. A context only ever names a handful of tokens, so
    an eager pass would be pure wasted work on a corpus of any real size.
    """

    def __init__(self, model):
        self.model = model
        self._raw_rows = {}    # token -> {target: (s1, s2, s3, s4)}, filled in lazily, one token at a time

    @classmethod
    def build(cls, model):
        """
        Just wraps the model reference -- no eager full-vocabulary pass (see the class
        docstring's "LAZY, not eager"). Cheap and instant regardless of corpus size; the real
        cost only shows up later, per token, the first time that token is asked for.
        """
        return cls(model)

    def _index(self):
        m = self.model
        return m.noise_index if m.noise_index is not None else m.build_noise_index()

    def _raw_row(self, token):
        """
        Pay-once-per-token cache of ONE token's RAW attention row: {target_id: (stage1, stage2,
        stage3, stage4)} for every target it has any evidence for ({} if it has no home).
        This is the ONLY thing noise at generation time ever computes per context token; once
        cached, every later use of the same token is a plain dict lookup.
        """
        cached = self._raw_rows.get(token)
        if cached is None:
            got = _focus_rows(self.model, token)
            cached = {}
            if got is not None:
                for cand, info in got[1].items():
                    cached[cand] = (info["stage1"], info["stage2"], info["stage3"], info["stage4"])
            self._raw_rows[token] = cached
        return cached

    def combine_raw(self, tokens):
        """
        Element-wise sum of the named tokens' RAW rows -- "cat gives mat 1, carpet gives dog 1
        -> mat 1, dog 1" -- as {target: {"stage1": n, ..., "stage4": n}}. A token with no row
        contributes nothing, not an error; a repeated token contributes once per occurrence.
        """
        summed = {}
        for t in tokens:
            for cand, (s1, s2, s3, s4) in self._raw_row(t).items():
                entry = summed.get(cand)
                if entry is None:
                    summed[cand] = {"stage1": s1, "stage2": s2, "stage3": s3, "stage4": s4}
                else:
                    entry["stage1"] += s1
                    entry["stage2"] += s2
                    entry["stage3"] += s3
                    entry["stage4"] += s4
        return summed

    def combine_final_score(self, tokens):
        """
        THE noise score for a context: add the named tokens' raw rows element-wise
        (combine_raw()), THEN normalize the sum per stage (100 / that stage's total across every
        target) and average the four stages -- normalize_row()'s "final". Percentages are scoped
        to this context, so the final scores add up to at most 100. Order of `tokens` doesn't
        matter. Returns {target: score}.
        """
        summed = self.combine_raw(tokens)
        if not summed:
            return {}
        return {cand: info["final"] for cand, info in normalize_row(summed).items()}

    combine = combine_final_score

    def row(self, token):
        """The {target: final score} row for ONE token (its own raw row, normalized) -- computed on first request. {} if the token has no homes."""
        return self.combine_final_score([token])

    def row_detailed(self, token):
        """
        Same target set as the stage 1-3 part of row(), but each value is the RAW
        {"stage1", "stage2", "stage3"} breakdown instead of the final score -- what
        analyse.py's token-row table displays. Same "score each pair once" rule as
        focus_scores_detailed(). NOT cached: a display-only, on-demand view.
        """
        idx = self._index()
        candidates = set()
        for home in idx.homes(token):
            candidates |= idx.tokens_for(home)
        candidates.discard(token)
        row = {}
        for cand in candidates:
            stage1, stage2, stage3 = idx.pair_stages(token, cand)
            row[cand] = {"stage1": stage1, "stage2": stage2, "stage3": stage3}
        return row

    def combine_detailed(self, tokens):
        """
        Same as combine_raw(), but only the stage1/stage2/stage3 breakdown, summed across the
        named tokens -- see row_detailed() (not cached, computed fresh each call).
        """
        combined = {}
        for t in tokens:
            for cand, info in self.row_detailed(t).items():
                entry = combined.setdefault(cand, {"stage1": 0, "stage2": 0, "stage3": 0})
                entry["stage1"] += info["stage1"]
                entry["stage2"] += info["stage2"]
                entry["stage3"] += info["stage3"]
        return combined


# ===== word-level convenience =================================================

def vocabulary_row(model, word):
    """
    Word-level convenience for TokenVocabularyMatrix.row() -- the
    "who should I pay attention to?" table for one token (final scores),
    e.g. vocabulary_row(model, "cat"). Auto-builds/caches
    model.token_vocab on first use. Returns None if word was never
    trained (a trained word with no eligible homes returns an empty
    dict, an honest "nothing accumulated", not None -- see row()).
    """
    if model.token_vocab is None:
        model.build_token_vocab()
    anchor = _resolve(model, word)
    if anchor is None:
        return None
    dec = lambda t: model.decode([t])
    raw = model.token_vocab.row(anchor)
    return {dec(t): s for t, s in raw.items()}


def combine_context(model, words):
    """
    Word-level convenience for TokenVocabularyMatrix.combine() -- e.g.
    combine_context(model, ["cat", "sat", "the"]) sums those three
    tokens' final-score rows into one combined score per target word.
    This is the inference-time step: a context activates several tokens,
    and each brings its own row -- computed the first time that specific
    token is asked for, cached after (see TokenVocabularyMatrix's "LAZY,
    not eager" docstring) -- only the named tokens are ever computed,
    not the rest of the vocabulary. Words never trained are silently
    skipped (contribute nothing), not an error -- same as an
    individual row() lookup for one.

    Returns {word: score}.
    """
    if model.token_vocab is None:
        model.build_token_vocab()
    tokens = [t for t in (_resolve(model, w) for w in words) if t is not None]
    combined = model.token_vocab.combine(tokens)
    dec = lambda t: model.decode([t])
    return {dec(t): s for t, s in combined.items()}


def vocabulary_row_detailed(model, word):
    """
    Word-level convenience for TokenVocabularyMatrix.row_detailed() --
    same target set as vocabulary_row(), but each value is the raw
    stage1/stage2/stage3 breakdown instead of the final score -- what
    analyse.py's token-row table displays as separate columns. Returns
    None if word was never trained.
    """
    if model.token_vocab is None:
        model.build_token_vocab()
    anchor = _resolve(model, word)
    if anchor is None:
        return None
    dec = lambda t: model.decode([t])
    raw = model.token_vocab.row_detailed(anchor)
    return {dec(t): dict(info) for t, info in raw.items()}


def combine_context_detailed(model, words, weight_mode="count"):
    """
    RAW combine: for each named word, take its own focus_scores_detailed()
    row -- the SAME per-anchor stage1/stage2/stage3/stage4 table the
    single-word focus-scores command shows, not a separately maintained
    structure -- and add the rows together element-wise, target by
    target: "stage1 compute scores for all tokens 'cat' pays attention
    to, and also for all tokens 'sat' pays attention to, then add them."

    All four stages are summed the same way. There is no combined
    number on the raw rows: the stages are on different scales, so
    normalize_row() (or combine_context_normalized()) is what produces a
    final.

    combine_context_normalized() is this followed by normalize_row(): the raw rows are
    added first and only the SUM is turned into percentages.

    Returns {word: {"stage1": n, "stage2": n, "stage3": n, "stage4": n}}.
    """
    if model.noise_index is None:
        model.build_noise_index()
    combined = {}
    for w in words:
        row = focus_scores_detailed(model, w, weight_mode=weight_mode)
        if row is None:
            continue
        for cand, info in row["scores"].items():
            entry = combined.setdefault(
                cand, {"stage1": 0, "stage2": 0, "stage3": 0, "stage4": 0})
            entry["stage1"] += info["stage1"]
            entry["stage2"] += info["stage2"]
            entry["stage3"] += info["stage3"]
            entry["stage4"] += info["stage4"]

    return combined


def combine_context_normalized(model, words, weight_mode="count"):
    """
    NORMALIZED combine: add the named words' RAW rows together element-wise
    (combine_context_detailed()), THEN normalize the sum -- per stage, 100 / that stage's total
    across every target -- and average the four stages into "final". This is exactly what
    generation does (TokenVocabularyMatrix.combine_final_score()): element-wise addition
    happens on raw scores, normalization once on the sum.

    Example, "cat carpet": cat's row gives mat 1, carpet's gives dog 1; the sum is mat 1, dog 1;
    the total is 2 and 100 / 2 = 50, so the context is about mat = 50% and dog = 50%.

    Returns {word: {"stage1": n, "stage2": n, "stage3": n, "stage4": n, "final": n}}
    (percentages scoped to this context; each stage adds up to 100 over the targets).
    """
    return normalize_row(combine_context_detailed(model, words, weight_mode=weight_mode))


def why(model, anchor_word, home_relationship_id, candidate_word):
    """
    THE independent follow-up question for noise.py: which SPECIFIC other sentences voted yes for
    candidate_word, given this anchor+home? STAGE 2 ONLY -- see
    NoiseCancellationIndex.voters_for()'s docstring for why only
    stage 2 has a materializable relationship-id voter list (stage 1's
    would be a subset of the same type; stage 3's voters are tokens,
    not relationships, so there's no single list that could stand in
    for all three stages). Built
    from the cached knowing_rids(candidate) set (see
    NoiseCancellationIndex._knowing_rids()), so this is a set-difference
    against cached state, not a fresh corpus scan. Never computed as
    part of focus_scores()/token_row()/build_token_vocab()'s bulk
    paths (those all work from the plain stage counts and never
    materialize a voter list) -- only here, on demand, for one
    candidate at a time. This is what turns a number on a permanent
    row back into a concrete, inspectable claim: "cat earned 21 points
    for 'it' -- show me which 3 sentences those points came from" is
    three separate why() calls (one per home), not something baked
    into every token_row() call.

    Returns None if anchor_word/candidate_word was never trained, or
    raises ValueError if home_relationship_id doesn't actually contain
    anchor_word.
    """
    if model.noise_index is None:
        model.build_noise_index()
    idx = model.noise_index
    anchor = _resolve(model, anchor_word)
    candidate = _resolve(model, candidate_word)
    if anchor is None or candidate is None:
        return None
    if home_relationship_id not in idx.homes(anchor):
        raise ValueError(f"relationship {home_relationship_id} doesn't contain '{anchor_word}'; "
                          f"it appears in {idx.homes(anchor)}")
    voters = idx.voters_for(anchor, home_relationship_id, candidate)
    return {"anchor": anchor_word, "home": home_relationship_id, "candidate": candidate_word,
            "score": len(voters), "voters": voters}
