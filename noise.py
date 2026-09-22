"""
noise.py — noise-cancellation token scoring for MSE-GLM.

Given one ANCHOR token sitting in one HOME training sentence, this
scores every OTHER token in that home sentence by how many OTHER
training sentences think it's worth paying attention to. It needs no
Bridge-Matrix clustering at all (unlike products.py) -- just plain
sentence membership, widened to GLOBAL knowledge in two different
ways (see "THE RULE" below).

THE RULE, exactly: THE AVERAGE OF THREE SEPARATE VOTE RULES.

  Every OTHER training sentence votes -- unconditionally, whether or
  not it happens to also contain the anchor. There is no gate. For
  every candidate token in the home sentence (every token except the
  anchor itself), a voter votes "yes, pay attention to this" if it
  does NOT already KNOW that token; it abstains on whichever
  candidates it already knows. What "knows" means differs across the
  three rules being averaged:

    stage 1 (literal):    the voter SENTENCE knows T iff T is
                           literally one of its own tokens.
    stage 2 (sentence-     the voter SENTENCE knows T iff ANY of its
      global):             own tokens has EVER shared a sentence with
                           T, anywhere in the corpus -- not just
                           whether T sits literally in the voter's
                           own text.
    stage 3 (token-        every DISTINCT token that appears anywhere
      global, deduped):    outside the home sentence casts its own
                           vote based on whether IT has ever
                           co-occurred with T anywhere -- deduplicated
                           across sentences, so a token that happens
                           to live in 300 other sentences still only
                           counts once (see _other_vocab()).

  score(T) = vote_weight * (stage1(T) + stage2(T) + stage3(T)) / 3

  A literal, UNWEIGHTED arithmetic mean of the three raw counts, per
  request -- NOT normalized to a common scale first. Stage 1 and
  stage 2 share a ceiling of (n_rels - 1) OTHER SENTENCES; stage 3's
  ceiling is the number of DISTINCT TOKENS outside the home sentence
  (_other_vocab()), which is typically much larger than the sentence
  count on any corpus with more than a handful of words per sentence.
  That means stage 3 usually dominates the averaged number in raw
  magnitude -- a real, known property of averaging three quantities
  on different scales, not a bug, but worth knowing before reading
  too much into the combined number. If a normalized (0-1 per stage,
  then averaged) blend is ever wanted instead, that's a different,
  explicitly-requested formula, not this one.

  A token every voter already knows -- on all three counts -- gets
  zero on all three, so it averages to zero too: "the" lands at 0
  with no stopword list, purely because nothing ever qualifies to
  vote against it on any of the three rules.

  A sentence that also happens to contain the anchor is NOT thrown
  out of the voter pool -- it still legitimately may not know
  everything else in the home sentence, and still votes for whichever
  of those it doesn't know.

This is closer to IDF (inverse document frequency) than to
products.py's cluster-contrast mechanism -- no clustering, no
structural axis, no gate, just "how many OTHER sentences/tokens don't
already know this word, anywhere in their own co-occurrence reach".

HISTORY, briefly, for anyone reading old commits/docs: this module
went through three shapes. Originally three parallel stages, kept as
three separate score columns, never summed. Then collapsed down to
stage 2 alone (stage 3 had a real recurrence-counting bug, and once
fixed, its corrected numbers looked like a token-grained restatement
of stage 1 with none of stage 2's actual signal -- see any older copy
of this docstring's former "WHY GLOBAL, NOT LITERAL" section for the
full reasoning at the time). Now: back to all three, averaged into
one number per request, rather than either kept fully separate or
fully collapsed to one. Stage 3's dedup fix from the three-column era
carries forward unchanged into this version.

GLOBAL (CROSS-SENTENCE) ACCUMULATION: everything above scores one
anchor token against ONE home sentence -- "sentence knowledge". If
the anchor token appears in several training sentences, each home
produces its OWN table (different candidates, since candidates come
from that home's own words). focus_scores() below turns this into
"token knowledge": it finds every candidate across EVERY sentence the
anchor appears in, and scores each candidate EXACTLY ONCE -- see "VOTE
ONCE PER PAIR (ALL THREE STAGES)" immediately below. There used to be
a "stage 2 is different, it sums across homes" carve-out documented
here; that turned out to be WRONG (see that section for the actual
derivation and the correction) -- stage 2 has the identical issue
stage 1 does, for the identical algebraic reason. All three stages are
now scored once per candidate; none of them are summed across homes.

VOTE ONCE PER PAIR (ALL THREE STAGES): the rule, in one line -- "a
voter that already voted for this anchor->token pair doesn't get to
vote again just because the same pair shows up in another sentence;
it already knows this connection."

  STAGE 1: stage1(T) -- (n_rels - 1) - |token_rels[T] - {home_rid}| --
  looks like it depends on home_rid, but for any LEGITIMATE candidate
  T (T is always one of home_rid's own tokens, by definition of
  "candidate"), home_rid is ALWAYS a member of token_rels[T]. That
  collapses the formula to a CONSTANT that doesn't actually depend on
  which home is being scored at all:

      stage1(T) = (n_rels - 1) - (|token_rels[T]| - 1) = n_rels - |token_rels[T]|

  STAGE 2: SAME DERIVATION, same conclusion -- this codebase's own
  documentation originally claimed otherwise, and that claim was
  checked and found wrong, so the actual reasoning is recorded here
  rather than just the corrected number. knowing_rids(T) -- union_{t
  in global_reach(T)} token_rels[t] -- always contains T itself as one
  of the tokens unioned over, because T is trivially a member of its
  own global_reach(T) (T's own home_rid is in token_rels[T], and T is
  a member of tokens_for(home_rid), so T is in the union that defines
  global_reach(T)). That means knowing_rids(T) always contains
  token_rels[T] as a subset, which always contains home_rid for a
  legitimate candidate -- exactly stage 1's argument, one layer
  removed:

      stage2(T) = (n_rels - 1) - (|knowing_rids(T)| - 1) = n_rels - |knowing_rids(T)|

  STAGE 3: less perfectly constant than stage 1/2 (other_vocab(home_rid)
  really does change slightly per home -- see _other_vocab()), but the
  SAME duplication happens for the same reason: almost every token in
  other_vocab(home) is NOT exclusive to that one home, so it shows up
  again in the next home's other_vocab() too. Fix: generalize
  _other_vocab(home_rid) -- ONE excluded home -- to
  _other_vocab_multi(T_homes) -- ALL of the homes where T is a
  candidate for this anchor, treated as a single excluded region at
  once:

      other_vocab_multi(T_homes) = vocab_all() - {t : token_rels[t] <= T_homes}
      stage3(T) = |other_vocab_multi(T_homes)| - |other_vocab_multi(T_homes) & global_reach(T)|

  Only tokens EXCLUSIVE to that whole region (appear nowhere else in
  the corpus) get removed; every other token remains a legitimate
  voter and is counted exactly ONCE against T. With |T_homes| == 1
  this is identical to the single-home _other_vocab() result.

  So if candidate T recurs across several of the anchor's homes,
  naively summing any of these three per home added the exact same (or
  near-exact, for stage 3) number to itself once per recurrence --
  every extra "vote" was the same fact, restated, not new evidence.
  Fix, uniformly: every stage is computed ONCE per (anchor, candidate)
  pair, never summed per home. Since stage 1 and stage 2 are pure
  constants and stage 3 is computed directly from T_homes rather than
  per home and combined, there is no per-home value left to sum at
  all -- "more homes this pair recurs in" no longer inflates any
  stage's number, for any of the three.

With ALL THREE stages now per-(anchor, candidate) constants, single-
home counts()/counts_detailed() needed NO change -- a single (anchor,
home) query was always correct on its own; the bug only ever existed
in how MULTIPLE such snapshots for the same recurring candidate got
combined. That combining now happens in focus_scores()/
focus_scores_detailed()/TokenVocabularyMatrix._row_for()/row_detailed()
-- the four cross-home accumulators -- which no longer call
counts_detailed() per home at all: they just discover which candidates
exist (and, for stage 3, which homes each one recurs in) by walking
each home's own token set directly, then score each candidate once.

OPTIMIZATION -- O(1) per candidate, for ALL THREE stages: a candidate
T's stage-2 "knowing_rids" set and stage-3's own relationship to
_other_vocab(home_rid) both depend only on T itself (knowing_rids) or
only on home_rid (_other_vocab) -- never on the (anchor, home, T)
triple as a whole. Both are cached the first time they're needed and
reused forever after (see _knowing_rids()/_other_vocab()); stage 1
was always O(1) (a single token_rels[T] membership check, no caching
needed). Reintroducing three rules here does NOT reintroduce three
times the asymptotic cost of the single-rule version -- just three
O(1) lookups added together per candidate instead of one.

    stage1(T) = (n_rels - 1) - |token_rels[T] - {home_rid}|
    stage2(T) = (n_rels - 1) - |knowing_rids(T) - {home_rid}|
    stage3(T) = |other_vocab(home_rid)| - |other_vocab(home_rid) & global_reach(T)|

Only the FIRST time a given T is ever scored (stage 2) or a given
home_rid is ever scored against (stage 3) costs real work; every
subsequent query is a cache hit -- and focus_scores()/
TokenVocabularyMatrix.build() are exactly the callers that repeat
those queries most, once per home the anchor has.

VOTE WEIGHT: the final AVERAGED score is scaled by
config.NoiseConfig.VOTE_WEIGHT (default 1.0, a no-op) -- see that
class's docstring for why it's there (a lever for combining this
module's output with another weighted system later, e.g. IVM's V1-V9
layers). Applied once, to the average, not separately to each stage
before averaging. Set once per NoiseCancellationIndex at build() time,
same convention as ivm.py's own per-layer weights -- never re-passed
on individual counts()/score() calls.

Punctuation tokens are NOT specially excluded -- they aren't given any
different treatment from ordinary words. A sentence-final "." lands
at 0 for the same reason "the" does: every voter has one too, so
nothing ever qualifies to vote for it on any of the three stages.
Structural markers (PAD/UNK/BOS/EOS) ARE excluded outright, since
those aren't tokens from the training text at all -- there's no
"vote" to be had over whether a sentence contains its own
end-of-sequence marker.
"""

from config import RESERVED, EOS, NoiseConfig
from importance import sequence_for_relationship
from ivm import token_to_relationships

_STRUCTURAL = RESERVED | {EOS}

# Number of set bits in a non-negative int. int.bit_count() needs Python
# 3.10+; the fallback keeps this module's "Python 3 only" promise.
_popcount = int.bit_count if hasattr(int, "bit_count") else (lambda x: bin(x).count("1"))


class NoiseCancellationIndex:
    """
    Precompute-once, query-many, same role as every other analysis
    index in this codebase: reuses ivm.py's token_to_relationships
    directly for the gate (not reimplemented), and lazily caches each
    relationship's token set, each token's global reach, each token's
    knowing_rids set (stage 2), and each home's distinct-token pool
    (stage 3) on first use.

    vote_weight (default config.NoiseConfig.VOTE_WEIGHT) scales the
    final averaged score counts() produces -- see config.py's
    NoiseConfig docstring. Like every other weight in this codebase,
    it's a per-instance default set at build() time, not something
    re-passed on every individual counts()/score() call.
    """

    def __init__(self, model, vote_weight=NoiseConfig.VOTE_WEIGHT, token_rels=None):
        self.model = model
        self.vote_weight = vote_weight
        # token -> {relationship_id, ...}. `token_rels` lets a caller that already
        # holds this exact map (ImportanceVoteMatrix does -- see
        # ivm.attach_noise_layer()) hand it over instead of paying for a second
        # full pass over every triple; it is only ever READ here.
        self.token_rels = token_rels if token_rels is not None else token_to_relationships(model)
        self._rel_mask_cache = {}       # token -> int bitmask over relationship ids (see _rel_mask())
        self._stage_cache = {}          # candidate -> (stage1, stage2, stage3) -- see candidate_stages()
        self._average_cache = {}        # candidate -> final averaged score, or None if unscorable -- see candidate_average()
        self._rel_tokens_cache = {}
        self._global_reach_cache = {}   # token -> every token it has ever shared a sentence with
        self._knowing_rids_cache = {}   # token -> every relationship_id that knows it (stage 2's cache)
        self._other_vocab_cache = {}    # home_rid -> frozenset of distinct tokens across every OTHER relationship (stage 3's voter pool)
        self._other_vocab_multi_cache = {}   # frozenset(home_rids) -> same, generalized to a SET of excluded homes (stage 3's cross-home "vote once" pool)
        self._vocab_all_cache = None    # every distinct token across the WHOLE corpus, computed once (see _vocab_all())
        self._n_rels = model.rels._n_rels
        self._all_rids = frozenset(range(self._n_rels))

    @classmethod
    def build(cls, model, vote_weight=NoiseConfig.VOTE_WEIGHT, token_rels=None):
        return cls(model, vote_weight=vote_weight, token_rels=token_rels)

    # ── fast per-candidate scoring (what V10 / TokenVocabularyMatrix use) ──
    #
    # Working through TokenVocabularyMatrix's cross-home "vote once"
    # formula shows a candidate's averaged score does NOT depend on which
    # anchor token it is being scored against:
    #     stage1 = n_rels - |rels(c)|
    #     stage2 = n_rels - |rels sharing any token with c|
    #     stage3 = |other_vocab| - |other_vocab & reach(c)|
    # and stage3's pool other_vocab = V_all - excl (excl = the tokens that
    # live ONLY in the anchor/candidate shared homes) obeys excl <= reach(c)
    # -- every such token sits in a sentence that also holds c -- so the
    # excl terms cancel and stage3 = |V_all| - |reach(c)|. Everything is a
    # property of the candidate alone. That lets one cheap per-candidate
    # computation replace the old per-(anchor, candidate) work, which built
    # a fresh vocabulary-sized frozenset for nearly every pair (O(V^2)
    # memory, and minutes of CPU on a real model). noise_scores()/
    # counts_detailed() (the per-home analysis views) are untouched;
    # TokenVocabularyMatrix._row_for_reference() keeps the literal
    # per-pair algorithm, and the test suite checks the two agree.

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

    def candidate_stages(self, cand):
        """
        (stage1, stage2, stage3) integer counts for one candidate token,
        cached after the first call (three ints per candidate -- nothing
        vocabulary-sized is ever stored). See the block comment above for
        why these need no anchor. Cost of a first call is roughly one
        pass over the sentences `cand` appears in, plus a few big-int ORs.
        """
        cached = self._stage_cache.get(cand)
        if cached is not None:
            return cached
        rels = self.token_rels.get(cand) or ()
        tokens_for = self.tokens_for
        reach = set()                       # every token cand has ever shared a sentence with
        for r in rels:
            reach |= tokens_for(r)
        stage1 = self._n_rels - len(rels)
        # rels that contain at least one token of `reach` (== the old
        # _knowing_rids(cand)), via bitmask ORs. Visit the most widespread
        # tokens first and stop the moment every relationship is covered --
        # for a common candidate that is usually within a few dozen ORs.
        order = reach
        if len(reach) > 64:
            order = sorted(reach, key=lambda u: -len(self.token_rels.get(u) or ()))
        full = (1 << self._n_rels) - 1
        knowing = 0
        for u in order:
            knowing |= self._rel_mask(u)
            if knowing == full:
                break
        stage2 = self._n_rels - _popcount(knowing)
        stage3 = len(self._vocab_all()) - len(reach)
        cached = (stage1, stage2, stage3)
        self._stage_cache[cand] = cached
        return cached

    def candidate_average(self, cand):
        """
        The three-stage AVERAGE noise-cancellation score of `cand`
        (vote_weight * (stage1 + stage2 + stage3) / 3 -- the same
        expression counts_detailed() uses), or None for a token that
        can't be scored: structural markers (never in any home's word
        set) and tokens that appear in no relationship.
        """
        try:
            return self._average_cache[cand]
        except KeyError:
            pass
        if cand in _STRUCTURAL or not self.token_rels.get(cand):
            value = None
        else:
            s1, s2, s3 = self.candidate_stages(cand)
            value = self.vote_weight * (s1 + s2 + s3) / 3
        self._average_cache[cand] = value
        return value

    def tokens_for(self, rid):
        cached = self._rel_tokens_cache.get(rid)
        if cached is None:
            cached = set(sequence_for_relationship(self.model, rid)) - _STRUCTURAL
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
        later counts()/voters_for() call that scores this same token
        as a candidate reuses this set instead of rebuilding the
        union from scratch.

            knowing_rids(T) = union_{t in global_reach(T)} token_rels[t]

        T's own home sentence(s) are always inside this set (T's own
        relationship_id(s) are among the ones unioned in, since T is
        trivially a member of its own global_reach), so a specific
        home_rid is always correctly excludable via the same
        -1-if-member trick counts() uses below -- never double
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
        Same computation as counts(), but keeps each candidate's
        THREE raw stage values alongside the final average, instead
        of collapsing straight to the blended number -- for callers
        that want to show the breakdown (e.g. analyse.py's
        noise-scores/focus-scores/token-row/combine columns), not
        just the one number. counts() is a thin wrapper around this
        that throws the breakdown away and keeps only "average" -- a
        single source of truth for the actual per-candidate
        arithmetic, not two copies of the same three set operations.

        O(1) per candidate for all three stages (see module
        docstring's "OPTIMIZATION"):

            stage1(T) = (n_rels - 1) - |token_rels[T] - {home_rid}|
            stage2(T) = (n_rels - 1) - |knowing_rids(T) - {home_rid}|
            stage3(T) = |other_vocab(home_rid)| - |other_vocab(home_rid) & global_reach(T)|
            average(T) = vote_weight * (stage1(T) + stage2(T) + stage3(T)) / 3

        stage1/stage2/stage3 are the RAW, UNWEIGHTED counts;
        vote_weight (config.NoiseConfig.VOTE_WEIGHT by default) is
        applied ONCE, only to "average" -- see module docstring's
        "VOTE WEIGHT".

            {"anchor":.., "home":.., "candidates": {...}, "total_voters":..,
             "scores": {token: {"stage1": n, "stage2": n, "stage3": n, "average": n}}}
        """
        home_tokens = self.tokens_for(home_rid) - {anchor}
        total_voters = self._n_rels - 1
        other_vocab = self._other_vocab(home_rid)
        total_token_voters = len(other_vocab)
        scores = {}
        for cand in home_tokens:
            cand_rels = self.token_rels.get(cand, set())
            stage1 = total_voters - (len(cand_rels) - (1 if home_rid in cand_rels else 0))

            knowing_rids = self._knowing_rids(cand)
            stage2 = total_voters - (len(knowing_rids) - (1 if home_rid in knowing_rids else 0))

            knowing_tokens = other_vocab & self.global_reach(cand)
            stage3 = total_token_voters - len(knowing_tokens)

            average = self.vote_weight * (stage1 + stage2 + stage3) / 3
            scores[cand] = {"stage1": stage1, "stage2": stage2, "stage3": stage3, "average": average}
        return {"anchor": anchor, "home": home_rid, "candidates": home_tokens,
                "total_voters": total_voters, "scores": scores}

    def counts(self, anchor, home_rid):
        """
        THE noise-cancellation counts for one (anchor, home) pair --
        the plain-number fast path every BULK consumer (focus_scores()/
        TokenVocabularyMatrix.build()/row()/combine()) uses: just the
        final averaged score per candidate, none of counts_detailed()'s
        per-stage breakdown -- see that method for the full three-stage
        numbers and the exact formula, and module docstring's "THE
        RULE"/"OPTIMIZATION" for what's being computed and why it's
        O(1) per candidate.
        """
        detailed = self.counts_detailed(anchor, home_rid)
        return {"anchor": detailed["anchor"], "home": detailed["home"],
                "candidates": detailed["candidates"], "total_voters": detailed["total_voters"],
                "scores": {cand: info["average"] for cand, info in detailed["scores"].items()}}

    def voters_for(self, anchor, home_rid, candidate):
        """
        THE independent "which specific sentences voted yes for this
        one candidate" question -- on demand only, same fast-count/
        slow-evidence split as products.py's vote_fn/evidence_fn.
        STAGE 2 ONLY: this reports stage 2's specific relationship-id
        voters, not a receipt for the three-way averaged score counts()
        returns. Stage 1's voters would be a different (smaller-or-
        equal) set of the same relationship-id type; stage 3's voters
        are individual TOKENS, not relationship ids at all, so there
        is no single list that could represent all three stages'
        receipts at once -- stage 2 is reported here because it's the
        widest of the two relationship-level rules. Built from the
        cached knowing_rids(candidate) set (see _knowing_rids()), so a
        repeat query for the same candidate against a different home
        is a cache hit, not a re-derivation from scratch.
        """
        knowing_rids = self._knowing_rids(candidate)
        return sorted((self._all_rids - {home_rid}) - knowing_rids)

    def score(self, anchor, home_rid):
        """
        Full table INCLUDING per-candidate voter lists -- for callers
        that want receipts immediately for ONE (anchor, home) pair
        (e.g. noise_scores()/the CLI's noise-scores command). Built
        from counts() (the fast path) plus one voters_for() call per
        candidate -- materializing K full lists of up to N items each
        is inherently more expensive than counts()'s plain numbers,
        but that cost is now paid only for a single explicit query,
        never inside a bulk operation like focus_scores()/
        TokenVocabularyMatrix.build(), which use counts() directly and
        never call this method.

        NOTE: "score" is the three-stage AVERAGE (see counts());
        "voters" is STAGE 2 ONLY (see voters_for()) -- the two numbers
        are related but len(voters) will not generally equal "score",
        since "score" blends in stage 1 and stage 3 too. This is a
        real, documented mismatch between the summary number and the
        one stage that has a materializable relationship-id voter
        list, not a bug.

            {"anchor": anchor, "home": home_rid,
             "candidates": {token, ...},
             "voters": [relationship_id, ...],     # every OTHER relationship_id, unconditionally
             "scores": {token: {"score": n, "voters": [relationship_id, ...]}}}
        """
        raw = self.counts(anchor, home_rid)
        scores = {cand: {"score": count, "voters": self.voters_for(anchor, home_rid, cand)}
                  for cand, count in raw["scores"].items()}
        return {"anchor": anchor, "home": home_rid, "candidates": raw["candidates"],
                "voters": sorted(self._all_rids - {home_rid}), "scores": scores}


# ─── word-level convenience ────────────────────────────────────────────────

def _resolve(model, word):
    """word -> token id, same convention used throughout this codebase's analysis layer."""
    ids = [t for t in model.tokenizer.encode(word) if t != 2]  # 2 == BOS
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
    return [{"relationship_id": rid, "sentence": model.tokenizer.decode(sequence_for_relationship(model, rid))}
            for rid in model.noise_index.homes(anchor)]


def noise_scores(model, anchor_word, home_relationship_id=None):
    """
    THE word-level entry point for noise.py -- one score per
    candidate, plus per-candidate voter receipts, for one anchor+home.

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
         "scores": {token: {"score": n, "voters": [relationship_id, ...]}, ...}}

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

    dec = lambda t: model.tokenizer.decode([t])
    scores = {
        dec(cand): {"score": info["score"], "voters": info["voters"]}
        for cand, info in full["scores"].items()
    }
    return {
        "anchor": anchor_word,
        "home": home_relationship_id,
        "home_sentence": model.tokenizer.decode(sequence_for_relationship(model, home_relationship_id)),
        "voter_pool": full["voters"],
        "scores": scores,
    }


def noise_scores_detailed(model, anchor_word, home_relationship_id=None):
    """
    Same anchor+home resolution rules as noise_scores(), but keeps
    each candidate's three raw stage values (see
    NoiseCancellationIndex.counts_detailed()) alongside the blended
    average, instead of only the average -- what analyse.py's
    noise-scores table actually displays as separate columns.

        {"anchor": word, "home": relationship_id, "home_sentence": str,
         "scores": {token: {"stage1": n, "stage2": n, "stage3": n, "average": n}, ...}}
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
    dec = lambda t: model.tokenizer.decode([t])
    return {
        "anchor": anchor_word,
        "home": home_relationship_id,
        "home_sentence": model.tokenizer.decode(sequence_for_relationship(model, home_relationship_id)),
        "scores": {dec(cand): dict(info) for cand, info in detailed["scores"].items()},
    }


def focus_scores(model, anchor_word):
    """
    Global, cross-sentence attention profile for anchor_word -- see
    module docstring's "GLOBAL (CROSS-SENTENCE) ACCUMULATION" section.
    Finds every candidate token across EVERY training sentence the
    anchor appears in, then scores each ONCE -- see module docstring's
    "VOTE ONCE PER PAIR (ALL THREE STAGES)": all three stages turned
    out to be per-(anchor, candidate) constants (stage 1/2 provably,
    stage 3 once correctly generalized over every home that pair
    recurs in), so there is nothing left to sum across homes for any
    of them -- "more homes" no longer means "bigger score" for any
    stage, and doesn't need to.

    Returns None if anchor_word was never trained. Otherwise:
        {"anchor": word, "homes": [relationship_id, ...],
         "scores": {token: {"total": n}}}
    "total" is what you sort by to answer "which tokens should I
    focus on most when I see this anchor, anywhere?" No more
    "by_home" breakdown -- there is only one value per candidate now,
    not one per home to break down (see focus_scores_detailed() for
    the three-stage breakdown of that one value).
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

    t_homes = {}
    for home in homes:
        for cand in idx.tokens_for(home) - {anchor}:
            t_homes.setdefault(cand, set()).add(home)

    dec = lambda t: model.tokenizer.decode([t])
    scores = {}
    for cand, homes_set in t_homes.items():
        cand_rels = idx.token_rels.get(cand, set())
        stage1 = idx._n_rels - len(cand_rels)
        knowing_rids = idx._knowing_rids(cand)
        stage2 = idx._n_rels - len(knowing_rids)
        other_vocab = idx._other_vocab_multi(homes_set)
        stage3 = len(other_vocab) - len(other_vocab & idx.global_reach(cand))
        total = idx.vote_weight * (stage1 + stage2 + stage3) / 3
        scores[dec(cand)] = {"total": total}
    return {
        "anchor": anchor_word,
        "homes": homes,
        "scores": scores,
    }


def focus_scores_detailed(model, anchor_word):
    """
    Same anchor/homes resolution as focus_scores(), same "score each
    candidate once" logic, but keeps stage1/stage2/stage3 as separate
    numbers instead of collapsing to "total" -- what analyse.py's
    focus-scores table actually displays as separate columns. See
    module docstring's "VOTE ONCE PER PAIR (ALL THREE STAGES)" -- all
    three stages are per-(anchor, candidate) constants once correctly
    computed, so this is not a sum over homes for any of them, just
    one clean evaluation per candidate.

        {"anchor": word, "homes": [relationship_id, ...],
         "scores": {token: {"stage1": n, "stage2": n, "stage3": n, "average": n}, ...}}
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

    t_homes = {}
    for home in homes:
        for cand in idx.tokens_for(home) - {anchor}:
            t_homes.setdefault(cand, set()).add(home)

    dec = lambda t: model.tokenizer.decode([t])
    scores = {}
    for cand, homes_set in t_homes.items():
        cand_rels = idx.token_rels.get(cand, set())
        stage1 = idx._n_rels - len(cand_rels)
        knowing_rids = idx._knowing_rids(cand)
        stage2 = idx._n_rels - len(knowing_rids)
        other_vocab = idx._other_vocab_multi(homes_set)
        stage3 = len(other_vocab) - len(other_vocab & idx.global_reach(cand))
        average = idx.vote_weight * (stage1 + stage2 + stage3) / 3
        scores[dec(cand)] = {"stage1": stage1, "stage2": stage2, "stage3": stage3, "average": average}
    return {
        "anchor": anchor_word,
        "homes": homes,
        "scores": scores,
    }


class TokenVocabularyMatrix:
    """
    THE permanent-for-this-session (not persisted to disk; same
    convention as ctm.py/ivm.py/this module's own
    NoiseCancellationIndex) cache of accumulated token knowledge:
    each token's vocabulary-score row -- {target: score} -- summed
    from counts() over every home sentence that token appears in.

    LAZY, not eager: a row is computed the FIRST time that specific
    token is actually asked for (via row() or as one of combine()'s
    inputs), then cached forever after -- never for every token in
    the corpus up front. This used to be an eager build() that walked
    every trained token's every home in one pass, whether or not
    anything ever asked about most of them; combine_context() only
    ever needs the handful of words actually named in a context, so
    that eager pass was pure wasted work on a corpus of any real size
    -- exactly why "combining tokens" was slow. Same "pay once, reuse
    forever" shape as NoiseCancellationIndex._knowing_rids(), just
    applied per requested token instead of per requested candidate.

    After a token's row is computed once, every later row()/combine()
    call touches ONLY the cached dict for that token -- no repeat call
    to sequence_for_relationship, no re-walk of any literal sentence.
    """

    def __init__(self, model):
        self.model = model
        self._rows = {}   # token -> {target: score}, filled in lazily, one token at a time

    @classmethod
    def build(cls, model):
        """
        Just wraps the model reference -- no eager full-vocabulary
        pass (see class docstring's "LAZY, not eager"). Cheap and
        instant regardless of corpus size; the real cost only shows up
        later, per token, the first time that specific token is
        actually asked for.
        """
        return cls(model)

    def _index(self):
        m = self.model
        return m.noise_index if m.noise_index is not None else m.build_noise_index()

    def candidate_average(self, cand):
        """Anchor-independent averaged score for one candidate (None if
        unscorable) -- see NoiseCancellationIndex.candidate_average()."""
        return self._index().candidate_average(cand)

    def weighted_counts(self, counts):
        """
        {candidate: candidate_average(candidate) * n} for a {candidate: n}
        map, skipping unscorable candidates. With n = the number of context
        tokens whose row contains the candidate, this IS combine() over
        that context -- every row entry for a candidate carries the same
        anchor-independent value, so summing rows is multiplying by a
        count. ivm.py's V10 feeds it V3's raw context counts, which
        avoids materializing any per-token row (each up to vocabulary
        size) at inference time.
        """
        idx = self._index()
        average = idx._average_cache
        out = {}
        for c, n in counts.items():
            try:
                g = average[c]
            except KeyError:
                g = idx.candidate_average(c)
            if g is not None:
                out[c] = g * n
        return out

    def _row_for(self, token):
        """
        The single per-token cache lookup/fill that row() and
        combine() both go through. Each candidate is scored ONCE (see
        module docstring's "VOTE ONCE PER PAIR (ALL THREE STAGES)") --
        no per-home summing for any of the three stages -- cached
        forever after under this exact token id. Values come from
        NoiseCancellationIndex.candidate_average() (the per-candidate
        closed form -- see the comment block in that class).
        """
        cached = self._rows.get(token)
        if cached is not None:
            return cached
        idx = self._index()
        neighbours = set()               # every token sharing a home sentence with `token`
        for home in idx.homes(token):
            neighbours |= idx.tokens_for(home)
        neighbours.discard(token)
        row = {cand: idx.candidate_average(cand) for cand in neighbours}
        self._rows[token] = row
        return row

    def _row_for_reference(self, token):
        """
        The original literal per-(anchor, candidate) algorithm, kept ONLY
        as the ground truth the fast _row_for() is tested against
        (test.py). Slow and memory-hungry on any real corpus -- its
        stage-3 pool is a fresh vocabulary-sized set per pair -- and
        never cached or used at runtime.
        """
        idx = self._index()
        t_homes = {}
        for home in idx.homes(token):
            for cand in idx.tokens_for(home) - {token}:
                t_homes.setdefault(cand, set()).add(home)
        row = {}
        for cand, homes_set in t_homes.items():
            cand_rels = idx.token_rels.get(cand, set())
            stage1 = idx._n_rels - len(cand_rels)
            knowing_rids = idx._knowing_rids(cand)
            stage2 = idx._n_rels - len(knowing_rids)
            other_vocab = idx._other_vocab_multi(homes_set)
            stage3 = len(other_vocab) - len(other_vocab & idx.global_reach(cand))
            row[cand] = idx.vote_weight * (stage1 + stage2 + stage3) / 3
        return row

    def row(self, token):
        """The {target: score} row for one token -- computed on first request, cached after. {} if the token has no homes."""
        return dict(self._row_for(token))

    def combine(self, tokens):
        """
        Sum several active tokens' rows into one combined score per
        target token -- the "cat contributes 7 for mat, sat
        contributes 8 for mat -> combined mat = 15" step. Each named
        token's row is computed (or reused from cache) via _row_for(),
        so this call only ever pays for the tokens actually named
        here, never the rest of the vocabulary. A token with no row
        (never trained, or trained but never had a home to learn a
        row from) contributes nothing, not an error. Order of
        `tokens` doesn't matter -- this is a plain sum, same as every
        other accumulation in this module. (This sums DIFFERENT
        tokens' independent rows together -- unrelated to, and not
        affected by, the "vote once per pair" rule inside _row_for(),
        which is about a single token's OWN multiple homes, not about
        combining several different tokens.)

        Returns {target: score}.
        """
        combined = {}
        for t in tokens:
            for cand, score in self._row_for(t).items():
                combined[cand] = combined.get(cand, 0) + score
        return combined

    def row_detailed(self, token):
        """
        Same target set as row(), but each value is the full
        {"stage1", "stage2", "stage3", "average"} breakdown instead of
        just the combined average -- what analyse.py's token-row/
        combine breakdown columns display. Same "score each candidate
        once" logic as _row_for() -- see module docstring's "VOTE
        ONCE PER PAIR (ALL THREE STAGES)". NOT cached alongside row()'s
        plain-number cache: this is a display-only, on-demand view,
        computed fresh each call, since the row()/combine() hot path
        (real callers, not the CLI) never needs the breakdown by
        default and shouldn't pay to keep a second cache warm for it.
        """
        idx = self.model.noise_index if self.model.noise_index is not None else self.model.build_noise_index()
        t_homes = {}
        for home in idx.homes(token):
            for cand in idx.tokens_for(home) - {token}:
                t_homes.setdefault(cand, set()).add(home)
        row = {}
        for cand, homes_set in t_homes.items():
            cand_rels = idx.token_rels.get(cand, set())
            stage1 = idx._n_rels - len(cand_rels)
            knowing_rids = idx._knowing_rids(cand)
            stage2 = idx._n_rels - len(knowing_rids)
            other_vocab = idx._other_vocab_multi(homes_set)
            stage3 = len(other_vocab) - len(other_vocab & idx.global_reach(cand))
            average = idx.vote_weight * (stage1 + stage2 + stage3) / 3
            row[cand] = {"stage1": stage1, "stage2": stage2, "stage3": stage3, "average": average}
        return row

    def combine_detailed(self, tokens):
        """
        Same as combine(), but each target's value is the full
        stage1/stage2/stage3/average breakdown, summed across the
        named tokens -- see row_detailed() (not cached, computed
        fresh each call, same reasoning).
        """
        combined = {}
        for t in tokens:
            for cand, info in self.row_detailed(t).items():
                entry = combined.setdefault(cand, {"stage1": 0, "stage2": 0, "stage3": 0, "average": 0})
                entry["stage1"] += info["stage1"]
                entry["stage2"] += info["stage2"]
                entry["stage3"] += info["stage3"]
                entry["average"] += info["average"]
        return combined


# ─── word-level convenience ────────────────────────────────────────────────

def vocabulary_row(model, word):
    """
    Word-level convenience for TokenVocabularyMatrix.row() -- the
    permanent "who should I pay attention to?" table for one token,
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
    dec = lambda t: model.tokenizer.decode([t])
    raw = model.token_vocab.row(anchor)
    return {dec(t): s for t, s in raw.items()}


def combine_context(model, words):
    """
    Word-level convenience for TokenVocabularyMatrix.combine() -- e.g.
    combine_context(model, ["cat", "sat", "the"]) sums those three
    tokens' rows into one combined score per target word. This is the
    inference-time step: a context activates several tokens, and each
    brings its own row -- computed the first time that specific token
    is asked for, cached after (see TokenVocabularyMatrix's "LAZY, not
    eager" docstring) -- only the named tokens are ever computed, not
    the rest of the vocabulary. Words never trained are silently
    skipped (contribute nothing), not an error -- same as an
    individual row() lookup for one.

    Returns {word: score}.
    """
    if model.token_vocab is None:
        model.build_token_vocab()
    tokens = [t for t in (_resolve(model, w) for w in words) if t is not None]
    combined = model.token_vocab.combine(tokens)
    dec = lambda t: model.tokenizer.decode([t])
    return {dec(t): s for t, s in combined.items()}


def vocabulary_row_detailed(model, word):
    """
    Word-level convenience for TokenVocabularyMatrix.row_detailed() --
    same target set as vocabulary_row(), but each value is the full
    stage1/stage2/stage3/average breakdown instead of just the summed
    average -- what analyse.py's token-row table displays as separate
    columns. Returns None if word was never trained.
    """
    if model.token_vocab is None:
        model.build_token_vocab()
    anchor = _resolve(model, word)
    if anchor is None:
        return None
    dec = lambda t: model.tokenizer.decode([t])
    raw = model.token_vocab.row_detailed(anchor)
    return {dec(t): dict(info) for t, info in raw.items()}


def combine_context_detailed(model, words):
    """
    Word-level convenience for TokenVocabularyMatrix.combine_detailed()
    -- same as combine_context(), but each value is the full
    stage1/stage2/stage3/average breakdown instead of just the summed
    average -- what analyse.py's combine table displays as separate
    columns.

    Returns {word: {"stage1": n, "stage2": n, "stage3": n, "average": n}}.
    """
    if model.token_vocab is None:
        model.build_token_vocab()
    tokens = [t for t in (_resolve(model, w) for w in words) if t is not None]
    combined = model.token_vocab.combine_detailed(tokens)
    dec = lambda t: model.tokenizer.decode([t])
    return {dec(t): dict(info) for t, info in combined.items()}


def why(model, anchor_word, home_relationship_id, candidate_word):
    """
    THE independent follow-up question for noise.py, same role as
    products.why(): which SPECIFIC other sentences voted yes for
    candidate_word, given this anchor+home? STAGE 2 ONLY -- see
    NoiseCancellationIndex.voters_for()'s docstring for why only
    stage 2 has a materializable relationship-id voter list (stage 1's
    would be a subset of the same type; stage 3's voters are tokens,
    not relationships, so there's no single list that could stand in
    for all three stages counts() actually averages together). Built
    from the cached knowing_rids(candidate) set (see
    NoiseCancellationIndex._knowing_rids()), so this is a set-difference
    against cached state, not a fresh corpus scan. Never computed as
    part of focus_scores()/token_row()/build_token_vocab()'s bulk
    paths (those all use the fast counts()-only path and never
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
