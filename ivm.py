"""
ivm.py — Importance Vote Matrix (IVM) for MSE-GLM.

Two entry points, two different roles:

  resolve_tie(tied_candidates, context_tokens)
      LEGACY Strict Mode tie-break. Only ever called on a set that
      something else (Stage 1) has already narrowed to a tie. Formula:
      score(C) = Σ influence(t) for every important t that knows C.
      Opt-in only -- see inference.py's Strict Mode path.

  select(candidates, context_tokens, current=None) / score_candidates(...)
      PRIMARY Open Mode mechanism. Called on the FULL legal candidate
      set every step, not just on ties -- this is what decides Open
      Mode generation now, replacing Stage 2's lineage tie-break
      entirely (see inference.py). SIX INDEPENDENT vote layers,
      summed -- not one formula where a factor multiplies another.
      An important token is ALSO a context token (I ⊆ P), so it casts
      up to six votes: V1 and V2 because it's important, PLUS its
      own V3, V4, V5, and V6 votes as an ordinary context member --
      it never loses those for being important, V1/V2 are additive
      bonuses:

        V1(C) = important_weight × Σ_{t∈I} 1[knowledge(t, C) > 0]
            Important tokens (I = cluster members present in context)
            cast their own vote, independently -- RAW/binary, one
            vote per important token that knows C at all, NOT scaled
            by how many relationships they co-occurred in (that
            magnitude enters through V4 instead, for every context
            token, not just important ones -- see below). Scaled down
            (default weight 0.1). Deliberately SMALL and SUPPORTING,
            not a competing independent channel -- see below.

        V2(C) = influence_weight × Σ_{t∈I, knows C} influence(t)
            Influence casts ITS OWN independent vote too -- not a
            multiplier on V1. influence(t) (how many distinct
            candidates t knows -- breadth) contributes its own small
            term (default weight 0.1), once per important token that
            knows C. Breadth (how many DIFFERENT candidates a token
            knows) is a different quantity from "how many times it
            knows THIS candidate" -- V4 covers the latter, for
            everyone, not just important tokens.

        V3(C) = context_weight × Σ_{t∈P} 1[knowledge(t,C) > 0]
            EVERY token in context P (not just important members --
            this is the one place non-important tokens get a say)
            casts one full vote (default weight 1.0) for any
            candidate it co-occurred with at least once. Binary, not
            weighted by evidence count: a token either attests C or
            it doesn't.

        V4(C) = context_influence_weight × Σ_{t∈P} knowledge(t, C)
            "Context token influence" -- one step up from V3's plain
            presence check: EVERY context token P also votes
            proportional to HOW MANY TIMES it co-occurred with C
            specifically (the same knowledge(t,C) quantity V1 used to
            use before it was made binary). Weighted very small
            (default 0.01 -- smaller even than V1/V2's 0.1) so it can
            only ever refine what V3 already decided, never compete
            with it.

        V5(C) = bigram_witness_weight × Σ_{t∈P} 1[t was ever in a
                                                     training sentence
                                                     containing the
                                                     literal bigram
                                                     (current, C)]
            "Bigram witness" vote -- a narrower, STRONGER claim than
            V3. V3 asks "did t ever co-occur with C, anywhere, in any
            sentence" -- V5 asks the sharper question "was t present
            in one of the SPECIFIC sentences where current->C
            actually occurred as a consecutive pair." Every context
            token P casts one full binary vote per candidate: either
            it was there when that exact transition happened ("I know
            that connection") or it wasn't ("I have never seen that
            connection"), regardless of how much unrelated evidence
            it shares with C otherwise. Default weight 1.0 -- UNLIKE
            V1/V2/V4, V5 is deliberately peer-weighted with V3, not
            kept small beneath it: it is independently strong,
            specific evidence in its own right (which sentences
            witnessed THIS bigram), not a structural nudge riding on
            top of a broader signal. See bigram_relationships() for
            how the per-bigram witnessing sentence sets are built --
            literal Bridge + Relationship Matrix only, same honesty
            constraint as _token_rels (see notes below).

        score(C) = V1(C) + V2(C) + V3(C) + V4(C) + V5(C) + V6(C)

      important_weight, influence_weight, and context_influence_weight
      all default strictly SMALLER than context_weight (0.1, 0.1, 0.01
      vs 1.0) BY DESIGN: important tokens (and raw evidence magnitude)
      must not be able to override what context presence already
      decided, only nudge a tie context itself left open. V1/V2/V4 are
      a bonus riding on top of V3, not a competing signal of equal
      standing -- an important token gets counted up to four times for
      related evidence (once via V3 and V4 like everyone else, twice
      more small because it's important), but those small votes still
      can't outweigh a real V3 majority. If you change these weights,
      that guarantee is a property of the ratios (important_weight,
      influence_weight, context_influence_weight < context_weight),
      not automatic -- verify it still holds. V5 and V6 are the two
      layers deliberately exempted from that "stay small" rule -- see
      above (V5) and below (V6).

        V6(C) = adjacency_weight × Σ_{t∈P}
                    1[t was ever DIRECTLY, immediately followed by C
                      in training -- t->C specifically, not C->t]
            "Adjacency" vote -- a different, WEAKER-in-one-sense,
            STRONGER-in-another question than both V3 and V5. Every
            context token P asks C: "were you ever the very next
            token right after me, anywhere in training?" (see
            adjacent_pairs()) -- strictly DIRECTIONAL, FROM t TO C
            only; C having been immediately followed by t does NOT
            count. Weaker than V5: the adjacency doesn't have to be
            the SPECIFIC current->C transition being scored right
            now -- any sentence counts, as long as the direction is
            t->C. Stronger than V3: mere co-occurrence in the same
            training sentence is NOT enough -- t must have actually
            been immediately followed by C at some point, not just
            present in the same sentence (in either order). Default
            weight 1.0 -- PEER-weighted with V3/V5, same reasoning:
            independently strong, bigram-level evidence, not a
            structural nudge. Literal Edge Matrix only, same honesty
            constraint as every other layer (see notes below).

      This deliberately reopens the door rule 24 shut ("unclustered
      tokens contribute 0 votes") -- V3/V4 are a considered exception,
      not an oversight: every context token gets a full, equal,
      independent say (V3) plus a tiny magnitude-sensitive refinement
      (V4), on top of (not instead of) the structural importance-
      weighted signal in V1/V2.

      Tie-break cascade when score itself doesn't discriminate (see
      select()): bigram frequency (how often `candidate` literally
      followed `current`), then global frequency (candidate's raw
      corpus-wide count -- the one signal deliberately excluded from
      the primary formula, used here ONLY as an absolute last resort),
      then lowest token id. Always deterministic. A None winner means
      candidates itself was empty; the caller (inference.py) still
      needs a deterministic fallback for that degenerate case, never
      randomness.

An alternative to Stage 2's lineage tie-break (inference.py), NOT an
alternative to CTM -- see ctm.py's module docstring for the
distinction. Stage 2 asks "does the exact triple (previous, current,
C) exist with a matching relationship_id?" -- a single previous
token, checked for an exact match. This module asks a broader
question instead: "which of the currently tied candidates do the
IMPORTANT tokens already generated so far collectively point to?" --
every clustered token in the whole context gets a vote, not just the
immediately preceding one, and a vote is driven by co-occurrence, not
exact-triple match.

1. Candidates are whatever Stage 1 already produced (successors from
   E, +EE in open mode, narrowed by bridge/lineage authority) -- IVM
   does not generate its own candidate list.

2. Important tokens -- corrected definition
   -----------------------------------------
   A token is important only if it ever occupies the VARYING
   (member) role of some cluster -- not merely "any role in any
   clustered triple." This is NOT the same test as bridges.t_index:
   t_index is populated from BOTH the bridge and target roles of
   every clustered triple regardless of which one is the axis that
   actually varies, so it also flags the FIXED half of a cluster
   (the trigger) as if it were a member. On "the boy sat on the
   chair" (bridge-axis cluster {cat,dog,boy} fixed on (the, sat);
   bridge-axis cluster {sat,ran} fixed on (boy, on)):
       - "boy" is the varying bridge of the first cluster -> a true
         member -> important.
       - "sat" is the varying bridge of the second cluster -> a true
         member -> important.
       - "the" and "on" only ever appear as the FIXED half of a
         cluster (the trigger), never as the varying member of any
         cluster -- t_index still lists them (they sit in a
         clustered triple's bridge/target slot), but they are not
         important under this stricter test, and must not be. An
         earlier version of this module used t_index directly and
         let "the"/"on" cast votes; on real generation traces this
         let common structural words swing a tie purely by sentence
         frequency, drowning out the two tokens that actually carry
         information (boy, sat). This version fixes that by reusing
         importance.py's own axis-aware member/trigger split
         (_trigger_for_triple) instead of t_index.
   This is computed once per model (over every triple, both roles,
   using the SAME axis logic importance.py already uses to tag
   per-occurrence importance) and cached as a flat set of token ids
   -- see important_member_tokens(). Only V1/V2 restrict themselves
   to this set; V3 deliberately does not (see above).

3. "Knows" -- an important token t knows candidate C if they ever
   co-occur in the same training sentence (same reuse of ctm.py's
   token_to_relationships as before): support(t, C) =
   |rels(t) ∩ rels(C)| > 0. That count IS the V1 vote itself now
   (not a multiplier); influence enters only through V2, as its own
   independent term.

4. Influence(t) = number of distinct candidates t knows (breadth).

5. resolve_tie's legacy formula (unchanged): score(C) = Σ influence(t),
   once per candidate a token knows, regardless of evidence count.
   A remaining tie means "no opinion, return None." This is a
   completely separate formula from V1/V2/V3 above -- see the module
   docstring's opening split.

Honesty notes
--------------
  - Vote weights (support / "knows") come ONLY from the literal
    training Relationship Matrix -- via token_to_relationships,
    reused unmodified from ctm.py. There is no longer a separate
    Experience data source: "strict" and "open" mode see identical
    evidence for every layer, including V5's bigram-witness sets
    (bigram_relationships()) and V6's adjacency index
    (adjacent_pairs()), both built ONLY from the literal
    Bridge/Edge + Relationship Matrix.
  - `important_member_tokens()`'s `mode` parameter is likewise
    accepted for call-site compatibility but has no effect anymore --
    "important" (V1/V2 eligibility) is cluster membership from the
    (only) Bridge Matrix, the same set in both modes.
  - A token never votes for itself if it's also a tied candidate --
    counting rels(t) ∩ rels(t) as "evidence" would just be the
    token's own sentence-frequency inflating its own candidacy, not
    co-occurrence with anything. Applies to all three vote layers.
  - Reserved tokens (<PAD>/<UNK>/<BOS>/<EOS>) never count as
    important and never vote in any layer, same exclusion ctm.py
    applies and for the same reason.
  - resolve_tie never changes what inference.py decides UNLESS
    explicitly opted into (Strict Mode only). select()/
    score_candidates() are unconditional for an Open Mode engine --
    see inference.py's module docstring.
"""

from collections import Counter, defaultdict

from ctm import RESERVED, token_to_relationships
from importance import _trigger_for_triple


def important_member_tokens(model, mode="strict"):
    """
    Every token id that ever occupies the VARYING (member) role of
    some cluster -- see the module docstring for why this differs
    from bridges.t_index. Computed once by scanning every triple and
    reusing importance.py's own axis-aware split (the same logic
    that decides, per occurrence, whether a triple's bridge or its
    target is the "important" one).

    `mode` is accepted for call-site compatibility but has no effect
    -- there is no longer a separate Experience Bridge Matrix.
    """
    tokens = set()
    b = model.bridges
    for tid in range(len(b.source)):
        info = _trigger_for_triple(b, tid)
        if info:
            _axis, _trigger, tok = info
            tokens.add(tok)
    tokens -= RESERVED
    return tokens


def bigram_frequencies(model, mode="strict"):
    """
    {(src, dst): count} from EdgeMatrix -- literal consecutive-pair
    counts, precomputed once so select()'s tie-break cascade doesn't
    need to touch model.edges directly at query time. See
    EdgeMatrix.frequency() for what these counts mean.

    `mode` is accepted for call-site compatibility but has no effect
    -- there is no longer a separate Experience Edge Matrix.
    """
    freqs = {}
    edges = model.edges
    for i in range(len(edges.src)):
        # edges.src/edges.dst are already deduplicated one row per
        # unique pair (see EdgeMatrix's docstring), so no accumulation
        # is needed here -- each key is set exactly once.
        freqs[(edges.src[i], edges.dst[i])] = edges.count[i]
    return freqs


def bigram_relationships(model):
    """
    {(source, candidate): {rel_id, ...}} -- for every literal training
    triple (source=source, bridge=candidate, target=anything), the
    set of relationship_ids (training sentences) that triple belongs
    to. This is the literal evidence V5's bigram-witness vote checks
    context tokens against: "which sentences actually contained the
    consecutive pair source->candidate."

    Built ONLY from the Bridge + Relationship Matrix -- the same
    literal-only reasoning as token_to_relationships (ctm.py): a
    "this bigram happened in this sentence" fact only exists for
    bigrams that were actually observed together in one sentence.
    """
    out = defaultdict(set)
    b = model.bridges
    for tid in range(len(b.source)):
        rel_ids = model.rels.relationships_for_triple(tid)
        if rel_ids:
            out[(b.source[tid], b.bridge[tid])].update(rel_ids)
    return dict(out)


def adjacent_pairs(model):
    """
    {(from_token, to_token), ...} -- every DIRECTED literal training
    bigram: from_token immediately followed by to_token, at least
    once. This is the evidence V6's adjacency vote checks context
    tokens against: a context token t votes for candidate C iff
    (t, C) is in this set -- i.e. t was literally, immediately
    followed by C somewhere in training. Directional: FROM the
    context token TO the candidate ONLY -- (C, t) being in this set
    does NOT imply (t, C) is, and the reverse observation does not
    count as a vote.

    Deliberately a different, WEAKER question than V5's bigram-witness
    evidence (bigram_relationships() above): V6 only asks "was t ever
    immediately followed by candidate C, in ANY sentence" -- it
    doesn't care whether that adjacency has anything to do with the
    specific transition current->C being scored right now (that's
    V5's job), and it doesn't require a shared sentence beyond the
    adjacency itself (V3/V4 ask a broader, even weaker question
    still: merely sharing a sentence at all, not necessarily standing
    next to each other in it, in either order).

    Built ONLY from the literal Edge Matrix -- every training bigram
    that was ever actually observed, in either mode.
    """
    e = model.edges
    return {(a, b) for a, b in zip(e.src, e.dst)}


class ImportanceVoteMatrix:
    """
    Precomputed, once-built support for importance voting: which
    tokens count as important members under a mode, the shared
    token -> {relationship_id, ...} index votes are counted from, and
    the bigram-frequency index the tie-break cascade uses. Same
    precompute-once-query-many pattern as ContextTriggerMatrix.
    """

    def __init__(self):
        self._token_rels = {}   # token -> set(rel_id), from ctm.token_to_relationships
        self._important = set()  # token ids eligible to cast V1/V2 votes under this mode
        self._important_weight = 0.1  # V1's per-token weight -- see score_candidates()
        self._influence_weight = 0.09  # V2's per-token weight -- see score_candidates()
        self._context_weight = 1.0    # V3's per-token weight -- see score_candidates()
        self._context_influence_weight = 0.01  # V4's per-token weight -- see score_candidates()
        self._bigram_witness_weight = 1.0  # V5's per-token weight -- see score_candidates()
        self._adjacency_weight = 1.0  # V6's per-token weight -- see score_candidates()
        self._bigram_freq = {}  # (token, candidate) -> count, from bigram_frequencies()
        self._bigram_rels = {}  # (token, candidate) -> set(rel_id), from bigram_relationships()
        self._adjacent = set()  # {frozenset({a,b}), ...}, from adjacent_pairs()

    @classmethod
    def build(cls, model, mode="strict", important_weight=0.1,
              influence_weight=0.09, context_weight=1.0,
              context_influence_weight=0.01, bigram_witness_weight=1.0,
              adjacency_weight=1.0):
        ivm = cls()
        ivm._token_rels = token_to_relationships(model)
        ivm._important = important_member_tokens(model, mode=mode)
        ivm._important_weight = important_weight
        ivm._influence_weight = influence_weight
        ivm._context_weight = context_weight
        ivm._context_influence_weight = context_influence_weight
        ivm._bigram_witness_weight = bigram_witness_weight
        ivm._adjacency_weight = adjacency_weight
        ivm._bigram_freq = bigram_frequencies(model, mode=mode)
        ivm._bigram_rels = bigram_relationships(model)
        ivm._adjacent = adjacent_pairs(model)
        return ivm

    # ── queries ──────────────────────────────────────────────────────────

    def important_tokens_in(self, context_tokens):
        """Sorted subset of context_tokens that count as important."""
        return sorted(t for t in context_tokens
                      if t in self._important and t not in RESERVED)

    def _knows_rels(self, important_token, candidates):
        """
        Like _knows, but returns the actual set of relationship ids
        shared with each candidate, not just the count -- exposed for
        tracing/auditing. A token never votes for itself -- see
        honesty notes.
        """
        rels_t = self._token_rels.get(important_token)
        if not rels_t:
            return {}
        out = {}
        for c in candidates:
            if c in RESERVED or c == important_token:
                continue
            rels_c = self._token_rels.get(c)
            if not rels_c:
                continue
            shared = rels_t & rels_c
            if shared:
                out[c] = shared
        return out

    def _knows(self, important_token, candidates):
        """{candidate: support} for one important token, support > 0 only."""
        return {c: len(rels) for c, rels in self._knows_rels(important_token, candidates).items()}

    def _context_vote(self, context_tokens, candidates):
        """
        V3: EVERY token in context -- important or not -- casts a
        full, un-scaled vote of 1.0 for any candidate it co-occurred
        with at least once. Binary: a token either attests a
        candidate or it doesn't; evidence count does not scale this
        vote (V1 is also binary now -- see score_candidates -- and V4
        is where evidence count/magnitude enters, at a much smaller
        weight). Same exclusions as every other vote layer: reserved
        tokens never vote, and a token never votes for itself.
        """
        votes = Counter()
        for t in context_tokens or []:
            if t in RESERVED:
                continue
            rels_t = self._token_rels.get(t)
            if not rels_t:
                continue
            for c in candidates:
                if c in RESERVED or c == t:
                    continue
                rels_c = self._token_rels.get(c)
                if rels_c and (rels_t & rels_c):
                    votes[c] += 1.0
        return dict(votes)

    def _context_influence_vote(self, context_tokens, candidates):
        """
        V4: "context token influence" -- every context token P (not
        just important members) ALSO casts a magnitude-sensitive
        vote: not just "do I know this candidate at all" (that's
        V3's binary job) but "how many times have I co-occurred with
        THIS SPECIFIC candidate" -- the raw evidence count, same
        knowledge(t, C) quantity V1 used to use before it was made
        binary (see score_candidates). Weighted very small (default
        0.01, smaller even than V1/V2's 0.1) so it can only ever
        refine a decision V3 already made, never compete with it --
        same "small vote riding on top of a big one" principle as
        V1/V2 relative to V3, just for the WHOLE context instead of
        only the important subset. Same exclusions as every other
        layer: reserved tokens never vote, a token never votes for
        itself.
        """
        votes = Counter()
        for t in context_tokens or []:
            if t in RESERVED:
                continue
            rels_t = self._token_rels.get(t)
            if not rels_t:
                continue
            for c in candidates:
                if c in RESERVED or c == t:
                    continue
                rels_c = self._token_rels.get(c)
                if not rels_c:
                    continue
                n = len(rels_t & rels_c)
                if n:
                    votes[c] += self._context_influence_weight * n
        return dict(votes)

    def _bigram_witness_vote(self, current, context_tokens, candidates):
        """
        V5: for each candidate C, look up the literal set of training
        sentences (relationship_ids) where the exact bigram
        (current, C) occurred (self._bigram_rels -- see
        bigram_relationships()). Every token in context then casts one
        full binary vote for C if it was ALSO present in at least one
        of those specific witnessing sentences -- "I have seen
        current->C happen, and I was there." A token that shares
        plenty of other sentences with C, but never one containing
        this exact bigram, casts no V5 vote for it ("I have never
        seen that connection") -- that is the whole point of this
        layer being distinct from V3's broader "co-occurred anywhere"
        question. Returns {} if `current` is None (nothing to look up)
        or the candidate was never literally seen as this bigram at
        all (no entry in self._bigram_rels). Same exclusions as every
        other layer: reserved tokens never vote, a token never votes
        for a candidate it IS.
        """
        votes = Counter()
        if current is None:
            return dict(votes)
        for c in candidates:
            if c in RESERVED:
                continue
            witness_rels = self._bigram_rels.get((current, c))
            if not witness_rels:
                continue
            for t in context_tokens or []:
                if t in RESERVED or t == c:
                    continue
                rels_t = self._token_rels.get(t)
                if rels_t and (rels_t & witness_rels):
                    votes[c] += 1.0
        return dict(votes)

    def _adjacency_vote(self, context_tokens, candidates):
        """
        V6: for each candidate C, every context token t asks a single,
        strictly DIRECTIONAL bigram-level question -- "was I (t) ever
        immediately followed by you (C), anywhere in training?"
        (self._adjacent -- see adjacent_pairs()). Full vote (1.0) if
        the literal bigram (t -> C) exists, 0 if not. FROM t TO C
        only: t being immediately followed by C is what counts here;
        C being immediately followed by t does NOT vote for C (it
        would instead be evidence for t, if t were ever scored as a
        candidate itself). Note what this does NOT require, to keep
        it distinct from the other layers:
          - unlike V5, it doesn't matter whether this (t -> C)
            adjacency has anything to do with the specific
            current->C transition being scored right now -- t voting
            here only means t was ONCE literally, immediately
            followed by C, in some sentence, possibly unrelated to
            the current step;
          - unlike V3/V4, mere co-occurrence in the same training
            sentence is NOT enough -- t must have been immediately
            followed by C, not just present in the same sentence.
        Same exclusions as every other layer: reserved tokens never
        vote, a token never votes for a candidate it IS.
        """
        votes = Counter()
        for c in candidates:
            if c in RESERVED:
                continue
            for t in context_tokens or []:
                if t in RESERVED or t == c:
                    continue
                if (t, c) in self._adjacent:
                    votes[c] += 1.0
        return dict(votes)

    def votes(self, candidates, context_tokens):
        """
        Full working, exposed for inspection/tracing (not just the
        winner) -- this is the log that makes the mechanism auditable
        rather than a black box.

        Returns:
            {"important_tokens": [...], "knows": {t: {C: support}},
             "influence": {t: int}, "totals": {C: int}}
        knows/influence are keyed by every important token present in
        context that knows at least one candidate; totals[C] = sum of
        influence(t) over every t that knows C (not sum of support --
        see rule 5 in the module docstring).
        """
        candidates = sorted(set(candidates))
        important = self.important_tokens_in(context_tokens or [])

        knows = {}
        for t in important:
            k = self._knows(t, candidates)
            if k:
                knows[t] = k
        influence = {t: len(k) for t, k in knows.items()}

        totals = Counter()
        for t, k in knows.items():
            for c in k:
                totals[c] += influence[t]

        return {"important_tokens": important, "knows": knows,
                "influence": influence, "totals": dict(totals)}

    def resolve_tie(self, candidates, context_tokens):
        """
        Try to break a tie among `candidates` (raw token ids) using
        importance voting. Returns the winning token id, or None if
        no important tokens are present in context, none of them
        know any tied candidate, or the vote total itself remains
        tied -- every None case means "fall back to the caller's
        normal tie-break," same contract as
        ContextTriggerMatrix.resolve_tie.

        This is the LEGACY tie-break entry point (flat influence-sum,
        only ever called on an already-narrowed tied set). Strict
        Mode's optional Stage-2-skip still uses this. Open Mode's
        primary selection uses score_candidates()/select() below
        instead -- a different formula (influence × knowledge, not
        just influence) evaluated over the FULL legal candidate set,
        not just a tie.
        """
        if not context_tokens:
            return None
        totals = self.votes(candidates, context_tokens)["totals"]
        if not totals:
            return None
        max_votes = max(totals.values())
        top = sorted(c for c, v in totals.items() if v == max_votes)
        return top[0] if len(top) == 1 else None

    def score_candidates(self, candidates, context_tokens, current=None):
        """
        Open Mode's primary scoring pass -- evaluates EVERY candidate
        given, not just a pre-existing tie. SIX independent vote
        layers, summed (see the module docstring for the full
        rationale). An important token is ALSO a context token
        (I ⊆ P), so it votes in every layer it qualifies for -- V3,
        V4, and V6 as a plain context member, PLUS V1 and V2 because
        it's important. Non-important context tokens cast V3/V4/V6.

            V1(C) = important_weight × Σ_{t∈I} 1[knowledge(t, C) > 0]
                Important tokens vote independently, RAW/binary --
                one vote per important token that knows C at all, NOT
                scaled by how many relationships they co-occurred in.
                Scaled down (default weight 0.1). Deliberately a
                SMALL, supporting vote, not an independent channel
                that can compete with V3 -- see the module docstring.

            V2(C) = influence_weight × Σ_{t∈I, knows C} influence(t)
                Influence casts its OWN small vote too -- not a
                multiplier on V1 -- worth influence_weight (default
                0.1) per important token that knows C, scaled by that
                token's breadth (how many distinct candidates it
                knows -- this is a DIFFERENT quantity from "how many
                times it knows THIS candidate," which V4 covers).

            V3(C) = context_weight × Σ_{t∈P} 1[knowledge(t, C) > 0]
                EVERY context token P (not just important members)
                casts one full vote (default weight 1.0 -- strictly
                larger than V1's important_weight) for any candidate
                it co-occurred with at least once. Binary per token,
                not scaled by evidence count.

            V4(C) = context_influence_weight × Σ_{t∈P} knowledge(t, C)
                "Context token influence" -- the one place evidence
                MAGNITUDE (not just presence) enters: every context
                token P, important or not, also votes proportional to
                HOW MANY TIMES it co-occurred with C specifically.
                Weighted very small (default 0.01 -- smaller even
                than V1/V2) so it only ever refines what V3 already
                decided, never competes with it.

            V5(C) = bigram_witness_weight × Σ_{t∈P}
                        1[t was ever in a training sentence
                          containing the literal bigram (current, C)]
                "Bigram witness" vote -- requires `current` (the
                token this step is generating from); returns all
                zeros if current is None. Every context token P casts
                one full binary vote per candidate: not "did I ever
                co-occur with C" (V3's question) but the narrower,
                stronger "was I present in one of the specific
                sentences where current->C actually happened as a
                consecutive pair." Default weight 1.0 -- PEER-weighted
                with V3, deliberately NOT kept small like V1/V2/V4:
                see the module docstring for why this layer is exempt
                from the "stay beneath V3" rule the others follow.

            V6(C) = adjacency_weight × Σ_{t∈P}
                        1[t was ever DIRECTLY, immediately followed
                          by C in training -- t->C, not C->t]
                "Adjacency" vote -- every context token P asks C a
                single, strictly DIRECTIONAL bigram-level question:
                "were you ever the very next token right after me?"
                (see adjacent_pairs()) -- FROM t TO C only, the
                reverse direction does not count. Unlike V3/V4 (mere
                co-occurrence in the same sentence is enough), V6
                requires actual t->C consecutiveness; unlike V5, that
                adjacency doesn't have to be the specific current->C
                transition being scored -- it can be from any
                sentence, as long as the direction is t->C. Default
                weight 1.0 -- also PEER-weighted with V3/V5, same
                reasoning: independently strong, specific (bigram-
                level, not sentence-level) evidence, not a structural
                nudge.

            score(C) = V1(C) + V2(C) + V3(C) + V4(C) + V5(C) + V6(C)

        By construction (important_weight, influence_weight, and
        context_influence_weight all smaller than context_weight),
        no amount of important-token or magnitude support can outvote
        a genuine context-vote (V3) lead -- V1+V2+V4 can only ever
        nudge a decision V3 left open, never override one it already
        made. Confirm this holds for your own weights if you change
        them: it's a property of the ratios, not guaranteed
        automatically. V5 and V6 are the two layers NOT bound by that
        rule -- both are independently strong, specific evidence, and
        by default can each outweigh V3 on their own (weight 1.0,
        same as V3).

        Returns a full trace dict, not just scores, so this stays
        auditable: {"important_tokens": [...], "knows": {...},
        "influence": {...}, "important_vote": {C: V1},
        "influence_vote": {C: V2}, "context_vote": {C: V3},
        "context_influence_vote": {C: V4}, "bigram_witness_vote":
        {C: V5}, "adjacency_vote": {C: V6},
        "scores": {C: V1+V2+V3+V4+V5+V6}}.
        "scores" is the one that actually drives select(); the six
        components are exposed separately (already weighted, so they
        sum directly to "scores") so each stays independently
        auditable. Every candidate in `candidates` gets a "scores"
        entry (0 if no layer voted for it); the per-layer dicts only
        have entries where that layer actually cast a vote.
        """
        candidates = sorted(set(candidates))
        context_tokens = context_tokens or []
        important = self.important_tokens_in(context_tokens)

        knows_rels = {}
        for t in important:
            kr = self._knows_rels(t, candidates)
            if kr:
                knows_rels[t] = kr
        knows = {t: {c: len(rels) for c, rels in kr.items()}
                 for t, kr in knows_rels.items()}
        influence = {t: len(k) for t, k in knows.items()}

        # V1 -- important tokens vote independently, RAW/binary, scaled small
        important_vote = Counter()
        for t, k in knows.items():
            for c in k:
                important_vote[c] += self._important_weight * 1

        # V2 -- influence casts its OWN vote, scaled, per token that knows C
        influence_vote = Counter()
        for t, k in knows.items():
            for c in k:
                influence_vote[c] += self._influence_weight * influence[t]

        # V3 -- every context token, important or not, casts a full binary vote
        raw_context_vote = self._context_vote(context_tokens, candidates)
        context_vote = {c: self._context_weight * v for c, v in raw_context_vote.items()}

        # V4 -- every context token also casts a tiny magnitude-sensitive vote
        context_influence_vote = self._context_influence_vote(context_tokens, candidates)

        # V5 -- every context token casts a full vote iff it literally
        # witnessed the exact (current, C) bigram in a training sentence
        raw_bigram_witness_vote = self._bigram_witness_vote(current, context_tokens, candidates)
        bigram_witness_vote = {c: self._bigram_witness_weight * v
                                for c, v in raw_bigram_witness_vote.items()}

        # V6 -- every context token casts a full vote iff it was ever
        # literally, immediately followed by C in training (t->C only)
        raw_adjacency_vote = self._adjacency_vote(context_tokens, candidates)
        adjacency_vote = {c: self._adjacency_weight * v
                           for c, v in raw_adjacency_vote.items()}

        final_scores = {
            c: important_vote.get(c, 0) + influence_vote.get(c, 0)
               + context_vote.get(c, 0) + context_influence_vote.get(c, 0)
               + bigram_witness_vote.get(c, 0) + adjacency_vote.get(c, 0)
            for c in candidates
        }

        return {"important_tokens": important, "knows": knows,
                "influence": influence,
                "important_vote": dict(important_vote),
                "influence_vote": dict(influence_vote),
                "context_vote": context_vote,
                "context_influence_vote": context_influence_vote,
                "bigram_witness_vote": bigram_witness_vote,
                "adjacency_vote": adjacency_vote,
                "scores": final_scores}

    def _bigram_frequency(self, token, candidate):
        """How many times `candidate` literally followed `token` (+ experience-implied, in open mode)."""
        return self._bigram_freq.get((token, candidate), 0)

    def _global_frequency(self, candidate):
        """Candidate's raw corpus-wide relationship count -- LAST-RESORT tie-break only, see select()."""
        return len(self._token_rels.get(candidate, set()))

    def select(self, candidates, context_tokens, current=None):
        """
        Deterministic winner among `candidates` by the combined score
        (V1 + V2 + V3 + V4 + V5, see score_candidates), with a
        three-stage deterministic tie-break cascade when the combined
        score itself doesn't discriminate:

            1. score (V1+V2+V3+V4+V5) -- primary, structural + contextual
            2. bigram frequency     -- how often `candidate` literally
                                        followed `current` (requires
                                        `current`; skipped if not given)
            3. global frequency     -- candidate's raw corpus-wide
                                        frequency (last resort only)
            4. lowest token id      -- final tie-break, always available

        `current` is also forwarded into score_candidates() itself now
        (not just used here for the tie-break), since V5 needs it to
        look up which bigram is being scored -- passing it through
        once is enough; V5 degrades to all-zero votes if omitted.

        Each stage only ever narrows a tie the stage before it left
        open; it never overrides a decision an earlier stage already
        made. Returns (winner_or_None, trace_dict). winner is None
        only when `candidates` itself is empty.
        """
        trace = self.score_candidates(candidates, context_tokens, current=current)
        scores = trace["scores"]
        if not scores:
            return None, trace
        max_score = max(scores.values())
        top = sorted(c for c, s in scores.items() if s == max_score)

        if len(top) > 1 and current is not None:
            bigram_freqs = {c: self._bigram_frequency(current, c) for c in top}
            max_bf = max(bigram_freqs.values())
            top = sorted(c for c in top if bigram_freqs[c] == max_bf)

        if len(top) > 1:
            global_freqs = {c: self._global_frequency(c) for c in top}
            max_gf = max(global_freqs.values())
            top = sorted(c for c in top if global_freqs[c] == max_gf)

        return top[0], trace

    def to_dict(self):
        return {
            "token_rels": {str(t): sorted(r) for t, r in self._token_rels.items()},
            "important": sorted(self._important),
            "important_weight": self._important_weight,
            "influence_weight": self._influence_weight,
            "context_weight": self._context_weight,
            "context_influence_weight": self._context_influence_weight,
            "bigram_witness_weight": self._bigram_witness_weight,
            "adjacency_weight": self._adjacency_weight,
            "bigram_freq": {f"{t}:{c}": n for (t, c), n in self._bigram_freq.items()},
            "bigram_rels": {f"{t}:{c}": sorted(r) for (t, c), r in self._bigram_rels.items()},
            "adjacent": [list(pair) for pair in self._adjacent],
        }

    @classmethod
    def from_dict(cls, d):
        ivm = cls()
        ivm._token_rels = {int(t): set(r) for t, r in d["token_rels"].items()}
        ivm._important = set(d["important"])
        ivm._important_weight = d.get("important_weight", 0.1)
        ivm._influence_weight = d.get("influence_weight", d.get("freq_weight", 0.1))
        ivm._context_weight = d.get("context_weight", 1.0)
        ivm._context_influence_weight = d.get("context_influence_weight", 0.01)
        ivm._bigram_witness_weight = d.get("bigram_witness_weight", 1.0)
        ivm._adjacency_weight = d.get("adjacency_weight", 1.0)
        ivm._bigram_freq = {}
        for key, n in d.get("bigram_freq", {}).items():
            t, c = key.split(":")
            ivm._bigram_freq[(int(t), int(c))] = n
        ivm._bigram_rels = {}
        for key, r in d.get("bigram_rels", {}).items():
            t, c = key.split(":")
            ivm._bigram_rels[(int(t), int(c))] = set(r)
        ivm._adjacent = {tuple(pair) for pair in d.get("adjacent", [])}
        return ivm
