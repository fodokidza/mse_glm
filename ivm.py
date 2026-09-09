"""
ivm.py — Importance Vote Matrix (IVM) for MSE-GLM.

Two entry points, two different roles:

  resolve_tie(tied_candidates, context_tokens)
      LEGACY Strict Mode tie-break. Only ever called on a set that
      something else (Stage 1) has already narrowed to a tie. Formula:
      score(C) = Σ influence(t) for every important t that knows C.
      Opt-in only -- see inference.py's Strict Mode path.

  select(candidates, context_tokens, current=None, previous=None) / score_candidates(...)
      PRIMARY Open Mode mechanism. Called on the FULL legal candidate
      set every step, not just on ties -- this is what decides Open
      Mode generation now, replacing Stage 2's lineage tie-break
      entirely (see inference.py). NINE INDEPENDENT vote layers,
      summed -- not one formula where a factor multiplies another.
      An important token is ALSO a context token (I ⊆ P), so it casts
      up to six of the nine votes: V1 and V2 because it's important,
      PLUS its own V3, V4, V5, and V6 votes as an ordinary context
      member -- it never loses those for being important, V1/V2 are
      additive bonuses. V7 and V8 are different from all six -- neither
      is "every context token votes," each is a single fixed-pair (V7)
      or fixed-triple (V8) check on `previous`/`current` specifically
      (see below). V9 is different again -- it's the only layer that
      requires EVERY context token to agree, not just one (V3) or a
      fixed pair (V7/V8).

  The gate (co-occurrence index, `_co_occurring` -- see build())
      V1-V6 and V9 all reduce, ultimately, to the same underlying
      question for a given (context token t, candidate C) pair: "did t
      and C ever appear in the same training sentence at all?" A
      candidate C that no context token has EVER co-occurred with is
      therefore guaranteed a zero vote from every one of those six
      layers -- not approximately, exactly, by construction (see
      score_candidates()'s docstring for the proof). `_co_occurring`
      precomputes, once per model, exactly which OTHER tokens each
      token has ever shared a training sentence with -- the same
      reverse-index trick build_cache() already used for its sparse
      cache, generalized to the WHOLE vocabulary (not just one
      candidate set) and computed once so both the live scoring path
      and build_cache() itself can reuse it instead of each
      recomputing their own version. score_candidates()'s live path
      uses it to skip straight to a zero score for any candidate no
      context token has ever heard of, instead of testing it against
      every layer in turn.

  build_cache(candidates) / enable_cache(candidates=None) / disable_cache()
      Opt-in sparse per-token cache for V1/V2/V3/V4/V6 -- each is a sum
      over context tokens of a (t, c)-only contribution, so it's
      precomputable once and reused every step instead of recomputed
      from scratch. V5/V7/V8 stay live always (see build_cache()'s own
      comment for why). Stores only nonzero (t, c) entries, never a
      dense |vocab| x |vocab| table. Off by default; NOT serialized by
      to_dict/from_dict (fully re-derivable from state that is). See
      score_candidates()'s "cache_used" trace field and
      benchmark_cache.py to compare cached vs. live directly.

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

        score(C) = V1(C) + V2(C) + V3(C) + V4(C) + V5(C) + V6(C) + V7(C) + V8(C)

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

        V7(C) = prev_current_weight × 1[rels(previous) ∩ rels(current)
                                          ∩ rels(C) ≠ ∅]
            "Previous+current co-occurrence" vote -- UNLIKE every
            other layer, this is NOT "every context token P casts a
            vote"; it's a single fixed-pair question about the two
            most recent tokens specifically: "was there ever ONE
            training sentence that mentioned `previous`, `current`,
            AND C together?" One flat binary vote per candidate, not
            summed over context. Broader than V5 in one sense: V5
            asks about the literal consecutive bigram (current, C) as
            its own triple; V7 only asks whether all three of
            previous, current, and C simply shared a sentence, no
            adjacency required between any pair. Narrower than V3 in
            another: V3 only needs ONE token to share a sentence with
            C; V7 needs BOTH previous and current to share a sentence
            with C and with each other, all at once. Default weight
            1.0 -- PEER-weighted with V3/V5/V6, same reasoning:
            specific, literal, sentence-level evidence, not a
            structural nudge. No vote for C if `previous` or `current`
            is None or reserved, or if C is `previous` or `current`
            itself. Literal Relationship Matrix only (via
            token_to_relationships, same source as every other layer).

        V8(C) = triple_weight × 1[(previous, current, C) was ever a
                                    literal, consecutive Bridge Matrix
                                    triple in training]
            "Triple witness" vote -- among the eight OTHER layers, the
            single STRICTEST, most literal question (V9 is stricter
            still, but strict in a different DIMENSION -- unanimity
            across an arbitrary-size context, not positional
            exactness): not "did these three
            tokens ever share a sentence" (V7), not "was C ever
            immediately preceded by current, in some sentence, for
            some reason" (V5's bigram_relationships key is (source,
            bridge), i.e. current->C as the first two slots of SOME
            triple) -- V8 asks whether `previous`, `current`, and C
            were ever literally seen back-to-back-to-back as ONE
            EXACT trained triple: source=previous, bridge=current,
            target=C. This is precisely the same literal-triple test
            Strict Mode's Stage 2 already uses to decide legality
            (see inference.py) -- V8 exposes it as a vote instead of a
            hard filter, so Open Mode (whose candidates span the
            whole vocabulary, not just literal successors) can reward
            a candidate for being the EXACT literal continuation
            without that candidate being the ONLY thing allowed to
            win. One flat binary vote per candidate, not summed over
            context -- same shape as V7, but a three-way EXACT MATCH
            instead of a three-way SHARED-SENTENCE check. No vote for
            C if `previous` or `current` is None or reserved, or if C
            is `previous` or `current` itself. Literal Bridge Matrix
            only (see literal_triples()) -- every triple there already
            occurred at least once in training, same honesty
            guarantee as every other layer. Default weight 2.0 --
            PEER-weighted with V5/V6/V7 (independently strong, literal
            evidence, not a structural nudge), set a notch above them
            since an exact triple match is strictly more specific
            evidence than any of V5/V6/V7 individually.

        V9(C) = whole_context_weight × 1[every non-reserved token
                                          t ∈ P, t ≠ C, satisfies
                                          knowledge(t, C) > 0]
            "Whole context" / UNANIMOUS vote -- the odd one out among
            the nine: every other layer either sums a per-token
            contribution over context (V1-V6) or checks one FIXED pair
            (V7) or triple (V8); V9 instead asks whether the WHOLE of
            context agrees at once. V3 already asks "does at least one
            context token know C" -- V9 asks the strictly harder
            question "do ALL of them" (excluding C itself, if C
            happens to already be a context token -- same "a token
            never has to vouch for itself" rule as every other layer,
            just checked once per required token here instead of
            skipped inside a sum). One flat binary vote per candidate,
            not scaled by context size or evidence count -- unanimity
            either holds or it doesn't. No vote for ANY candidate if
            context has no qualifying (non-reserved, not-C) tokens at
            all -- unanimity among zero voters proves nothing, so an
            empty or fully-reserved context casts no V9 votes for
            anyone, same as V3 casts no votes from an empty context
            either. Default weight 2.5 -- see config.py's
            WHOLE_CONTEXT_WEIGHT for why it's set a notch above V8.

        score(C) = V1(C) + V2(C) + V3(C) + V4(C) + V5(C) + V6(C) + V7(C) + V8(C) + V9(C)

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
from config import IVMConfig


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


def co_occurrence_index(token_rels, adjacent=None):
    """
    {token: {other_token: shared_relationship_count}} for EVERY pair
    of (non-reserved) tokens that ever appear together in at least one
    training relationship (sentence) -- the whole-vocabulary
    co-occurrence graph underneath V1/V3/V4/V6/V9, and the gate that
    lets score_candidates() skip a candidate straight to a 0 score
    once no context token has ever heard of it (see the module
    docstring's "The gate" section for the proof of why that's always
    safe for V1-V6/V9, and score_candidates()'s docstring for the one
    extra condition V7/V8 need).

    Built by inverting token_rels (relationship_id -> its member
    tokens) ONCE and, for each relationship, only visiting the tokens
    actually IN it -- the same technique
    ImportanceVoteMatrix.build_cache() used to use internally (see its
    docstring for the full O(vocab^2)-vs-O(corpus co-occurrence) cost
    argument), just generalized here to the ENTIRE vocabulary instead
    of one candidate set, and computed exactly once per model build so
    build_cache() (now just a filter over this) and score_candidates()
    (the live gate) both reuse the same underlying work instead of
    each paying for their own O(sum of sentence-length^2) pass.

    If `adjacent` (a set of directed (t, c) pairs -- see
    adjacent_pairs()) is given, also backfills any adjacency-only pair
    NOT already present with a shared count of 0, for any t that has
    AT LEAST ONE relationship-based entry already (token_rels.get(t)
    is truthy) -- the same condition the live scoring path and the
    cache-building code both already applied before this function
    existed (a token with literally zero known relationships is
    skipped by both, regardless of any adjacency it might have; this
    backfill deliberately preserves that existing behavior rather than
    widening it). This covers a real, if rare, edge case: two tokens
    that were literally consecutive in a training sentence too SHORT
    (fewer than 3 tokens) to have produced a Bridge Matrix triple --
    and therefore a relationship_id -- at all (see importance.py's
    note on degenerate short sentences). Such a pair has adjacency
    (V6's evidence) but zero relationship-level co-occurrence
    (V1/V3/V4/V9's evidence) for that SPECIFIC candidate -- V6 can
    still vote for it on its own terms regardless of what this index
    says, but WITHOUT this backfill the gate would wrongly treat that
    candidate as "no context token has ever heard of it" and skip it,
    silently dropping the V6 vote it's actually entitled to. The
    backfilled entry's shared count is 0, so it contributes nothing to
    V1/V3/V4/V9 (still correctly zero for them) -- it only keeps the
    candidate visible to the gate so V6 itself still gets a chance to
    fire.

    Reserved tokens never appear as a key or a value, and a token
    never "co-occurs" with itself -- same exclusions every other
    layer in this module applies.
    """
    rel_to_tokens = defaultdict(set)
    for tok, rels in token_rels.items():
        if tok in RESERVED or not rels:
            continue
        for r in rels:
            rel_to_tokens[r].add(tok)

    co_occurring = defaultdict(Counter)
    for toks in rel_to_tokens.values():
        toks = list(toks)
        for i, ti in enumerate(toks):
            row = co_occurring[ti]
            for j, tj in enumerate(toks):
                if i != j:
                    row[tj] += 1

    result = {t: dict(row) for t, row in co_occurring.items()}

    if adjacent:
        for (t, c) in adjacent:
            if t in RESERVED or c in RESERVED or c == t:
                continue
            if not token_rels.get(t):
                continue
            row = result.setdefault(t, {})
            if c not in row:
                row[c] = 0

    return result


def literal_triples(model):
    """
    {(source, bridge, target), ...} -- every literal, deduplicated
    Bridge Matrix triple: three tokens that were seen consecutively,
    in exactly that order, at least once in training. This is the
    evidence V8's triple-witness vote checks (previous, current, C)
    against: an exact match here means previous->current->C happened
    as one literal trained continuation, not merely that the three
    tokens shared a sentence somewhere (that weaker question is V7's
    job, via token_to_relationships/rels intersection).

    Built ONLY from the literal Bridge Matrix -- every triple there
    already occurred at least once in training (BridgeMatrix.build()
    only ever records triples it actually saw), so no additional
    relationship-existence check is needed here, unlike
    bigram_relationships() (which layers relationship_ids on top of
    the Bridge Matrix for V5's narrower per-sentence question). Same
    literal-only reasoning as every other helper in this module.
    """
    b = model.bridges
    return {(b.source[i], b.bridge[i], b.target[i]) for i in range(len(b.source))}


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
        self._important_weight = IVMConfig.IMPORTANT_WEIGHT  # V1's per-token weight -- see score_candidates()
        self._influence_weight = IVMConfig.INFLUENCE_WEIGHT  # V2's per-token weight -- see score_candidates()
        self._context_weight = IVMConfig.CONTEXT_WEIGHT    # V3's per-token weight -- see score_candidates()
        self._context_influence_weight = IVMConfig.CONTEXT_INFLUENCE_WEIGHT  # V4's per-token weight -- see score_candidates()
        self._bigram_witness_weight = IVMConfig.BIGRAM_WITNESS_WEIGHT  # V5's per-token weight -- see score_candidates()
        self._adjacency_weight = IVMConfig.ADJACENCY_WEIGHT  # V6's per-token weight -- see score_candidates()
        self._prev_current_weight = IVMConfig.PREV_CURRENT_WEIGHT  # V7's flat weight -- see score_candidates()
        self._triple_weight = IVMConfig.TRIPLE_WEIGHT  # V8's flat weight -- see score_candidates()
        self._whole_context_weight = IVMConfig.WHOLE_CONTEXT_WEIGHT  # V9's flat weight -- see score_candidates()
        self._bigram_freq = {}  # (token, candidate) -> count, from bigram_frequencies()
        self._bigram_rels = {}  # (token, candidate) -> set(rel_id), from bigram_relationships()
        self._adjacent = set()  # {(from_token, to_token), ...}, from adjacent_pairs() -- directed
        self._triples = set()   # {(source, bridge, target), ...}, from literal_triples()
        # {token: {other_token: shared_relationship_count}} -- the
        # whole-vocabulary co-occurrence gate/reverse-index, built once
        # in build()/from_dict() (never serialized -- fully re-derivable
        # from _token_rels, same "derived, not stored" treatment as
        # _token_cache below). Powers three things: (1) build_cache()'s
        # sparse cache construction, (2) score_candidates()'s live gate
        # (skip a candidate straight to a 0 score if no context token
        # has ever co-occurred with it), (3) V9's unanimous-agreement
        # check. See the module docstring's "The gate" section.
        self._co_occurring = {}
        # Sparse per-token score cache for V1/V2/V3/V4/V6 -- see build_cache()
        # and the "cache" branch of score_candidates(). NOT serialized by
        # to_dict/from_dict (it's fully re-derivable from _token_rels,
        # _important, _adjacent, and the weights above, all of which
        # already are serialized) -- call build_cache()/enable_cache()
        # again after from_dict() if you want it back.
        self._token_cache = {}      # {t: {c: (shared_count, adjacent_flag)}}
        self._cache_candidates = None  # frozenset the cache was built for, or None
        self._use_cache = False     # toggle -- see enable_cache()/disable_cache()

    @classmethod
    def build(cls, model, mode="strict", important_weight=IVMConfig.IMPORTANT_WEIGHT,
              influence_weight=IVMConfig.INFLUENCE_WEIGHT, context_weight=IVMConfig.CONTEXT_WEIGHT,
              context_influence_weight=IVMConfig.CONTEXT_INFLUENCE_WEIGHT,
              bigram_witness_weight=IVMConfig.BIGRAM_WITNESS_WEIGHT,
              adjacency_weight=IVMConfig.ADJACENCY_WEIGHT,
              prev_current_weight=IVMConfig.PREV_CURRENT_WEIGHT,
              triple_weight=IVMConfig.TRIPLE_WEIGHT,
              whole_context_weight=IVMConfig.WHOLE_CONTEXT_WEIGHT, use_cache=False):
        ivm = cls()
        ivm._token_rels = token_to_relationships(model)
        ivm._important = important_member_tokens(model, mode=mode)
        ivm._important_weight = important_weight
        ivm._influence_weight = influence_weight
        ivm._context_weight = context_weight
        ivm._context_influence_weight = context_influence_weight
        ivm._bigram_witness_weight = bigram_witness_weight
        ivm._adjacency_weight = adjacency_weight
        ivm._prev_current_weight = prev_current_weight
        ivm._triple_weight = triple_weight
        ivm._whole_context_weight = whole_context_weight
        ivm._bigram_freq = bigram_frequencies(model, mode=mode)
        ivm._bigram_rels = bigram_relationships(model)
        ivm._adjacent = adjacent_pairs(model)
        ivm._triples = literal_triples(model)
        # Whole-vocabulary co-occurrence gate/reverse-index -- built
        # once here (from the _token_rels just computed above) and
        # reused by build_cache() below AND by score_candidates()'s
        # live path, instead of each recomputing its own version.
        ivm._co_occurring = co_occurrence_index(ivm._token_rels, adjacent=ivm._adjacent)
        if use_cache:
            # Open Mode always scores the FULL vocabulary as candidates
            # (see model.all_candidate_tokens()/model.py's
            # _rebuild_open_engine) -- that's the one candidate set this
            # cache is actually built and validated against.
            ivm.enable_cache(model.all_candidate_tokens())
        return ivm

    # ── sparse per-token score cache (V1/V2/V3/V4/V6 only) ─────────────────
    #
    # V1, V2, V3, V4, and V6 are each a SUM OVER CONTEXT TOKENS of a
    # per-(t, c) contribution that depends only on t, c, and static
    # per-instance state (_token_rels, _important, _adjacent, weights) --
    # never on which OTHER tokens happen to be in context this step, and
    # never on `current`/`previous`. That means each (t, c) contribution
    # can be computed once and reused for every future step, instead of
    # being recomputed from scratch (three separate O(|context| x
    # |candidates|) loops -- _context_vote, _context_influence_vote,
    # _adjacency_vote -- plus _knows_rels' own O(|important present| x
    # |candidates|) loop) on every single call.
    #
    # V5, V7, V8, and V9 are deliberately NOT part of this cache: V5's
    # vote for (t, c) also depends on `current` (a different `current`
    # means a different witness set for the same t/c); V7/V8 aren't a
    # per-token contribution at all -- each is a single fixed-pair
    # (V7) or fixed-triple (V8) check on (previous, current), not a
    # sum over context tokens; V9 is a conjunction ACROSS every
    # context token for one candidate, not a sum of independent
    # per-(t, c) terms, so it can't be decomposed into per-token rows
    # the way V1-V4/V6 can either. All four are still fast, though --
    # V5/V7/V8 are cheap sparse set/tuple lookups (over _bigram_rels /
    # _token_rels / _triples), and V9 reuses this same instance's
    # _co_occurring gate (see score_candidates()) -- so they're
    # computed live whether or not the V1-V4/V6 cache is enabled.
    #
    # Only ONE raw quantity actually varies per (t, c) pair here:
    # shared = len(rels(t) & rels(c)). V1 (a token knows C at all) and V3
    # (a context token co-occurred with C at all) are both just
    # "shared > 0"; V4 IS shared, scaled small; V2's influence(t) is the
    # count of candidates t shares any relationship with (== the number
    # of cache[t] entries with shared > 0); V6 is a separate binary flag
    # (adjacency does not require sharing multiple relationships, just
    # one -- and in practice adjacency implies shared > 0 anyway, since
    # two literally-consecutive tokens are necessarily in the same
    # sentence). Storing the two raw numbers (not the five pre-weighted
    # votes) also means changing a weight (important_weight, ...) after
    # the cache is built does NOT stale it -- the weights are re-applied
    # at score time, only the underlying evidence is cached.

    def build_cache(self, candidates):
        """
        Build the sparse per-token cache for exactly `candidates` (an
        iterable of token ids -- for Open Mode this should be
        model.all_candidate_tokens(), the same full-vocabulary set
        score_candidates()/select() are actually called with every
        step). Does NOT enable the cache by itself -- see enable_cache().

        self._token_cache ends up as {t: {c: (shared, adjacent)}}, sparse:
        a token t only gets an entry at all if _token_rels has one for it
        (reserved/never-seen tokens are skipped entirely), and within
        that, a candidate c only gets an entry if shared > 0 or the pair
        is adjacent -- exactly the same "only nonzero rows" sparsity the
        rest of this module already uses for _bigram_rels/_adjacent, not
        a dense |vocab| x |vocab| table.

        ALGORITHM NOTE: shared(t, c) = |rels(t) & rels(c)| is exactly
        the number of relationships (training sentences) containing
        BOTH t and c. An old version tested every (t, c) pair in a
        full |tokens_with_relationships| x |candidates| double loop,
        computing that intersection from scratch each time -- an
        O(vocab^2) cost that dominated model build time for anything
        past a small vocabulary (empirically ~4x slower per ~2x
        vocabulary growth: quadratic, not linear). A later version
        fixed that here directly, by inverting relationship -> member
        tokens ONCE and only visiting tokens actually co-occurring
        together -- O(sum of sentence-length^2 across the corpus)
        instead of O(vocab^2). That inversion is now
        co_occurrence_index(), computed ONCE per model in build() (and
        from_dict()) as self._co_occurring, since score_candidates()'s
        live gate needs the exact same whole-vocabulary co-occurrence
        graph -- this method is now just a FILTER over that shared
        structure down to `candidates`, not a separate O(sum of
        sentence-length^2) pass of its own every time build_cache() is
        called (e.g. re-enabling the cache for a different candidate
        set after incremental training no longer re-pays that cost).
        """
        candidates = frozenset(candidates)

        cache = {}
        for t, row in self._co_occurring.items():
            out_row = {}
            for c, shared in row.items():
                # co_occurrence_index() already excludes RESERVED and
                # self-pairs (t, t) by construction, and already
                # backfills adjacency-only (shared=0) pairs -- only the
                # candidate-set restriction is this method's own job.
                if c not in candidates:
                    continue
                adjacent = 1 if (t, c) in self._adjacent else 0
                out_row[c] = (shared, adjacent)
            if out_row:
                cache[t] = out_row

        self._token_cache = cache
        self._cache_candidates = candidates

    def enable_cache(self, candidates=None):
        """
        Turn the cache ON. If `candidates` is given, (re)builds it first
        (via build_cache) -- pass this whenever the model's graphs may
        have changed since the cache was last built (fresh train,
        incremental merge) or the very first time you enable it. If
        `candidates` is omitted, reuses whatever was built last time --
        raises ValueError if nothing has been built yet, rather than
        silently scoring against a stale or nonexistent cache.
        """
        if candidates is not None:
            self.build_cache(candidates)
        elif self._cache_candidates is None:
            raise ValueError("enable_cache() called with no `candidates` and "
                              "no prior build_cache() call -- nothing to enable")
        self._use_cache = True

    def disable_cache(self):
        """
        Turn the cache OFF -- score_candidates() falls back to its live
        computation. The built cache data itself is left in place (not
        cleared), so a later enable_cache() with no arguments is instant.
        This is the toggle for comparing cached vs. live scoring/timing
        without rebuilding anything in between.
        """
        self._use_cache = False

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

    def _prev_current_vote(self, previous, current, candidates):
        """
        V7: for each candidate C, cast one full binary vote iff
        `previous`, `current`, AND C all appear together in at least
        one training sentence -- rels(previous) ∩ rels(current) ∩
        rels(C) ≠ ∅. UNLIKE every other layer, this is not "every
        context token t votes" -- it's a single fixed-pair check on
        the two most recent tokens specifically: "have I ever seen a
        sentence that mentioned both of my last two tokens and you?"
        One flat vote per candidate, not summed over context.

        Different question from V5 (bigram_relationships(), the
        literal (current, C) triple regardless of `previous`) and
        from V6 (adjacent_pairs(), direct t->C adjacency for every
        context token) -- V7 needs no adjacency between ANY of the
        three tokens, only that all three shared one sentence.

        Returns {} if `previous` or `current` is None or reserved, or
        if the two of them never even share a sentence with each
        other (nothing left to check candidates against). A candidate
        never votes via being `previous` or `current` itself, same
        "a token never votes for itself" rule as every other layer.
        """
        votes = Counter()
        if previous is None or current is None:
            return dict(votes)
        if previous in RESERVED or current in RESERVED:
            return dict(votes)
        rels_prev = self._token_rels.get(previous)
        rels_curr = self._token_rels.get(current)
        if not rels_prev or not rels_curr:
            return dict(votes)
        shared_pc = rels_prev & rels_curr
        if not shared_pc:
            return dict(votes)
        for c in candidates:
            if c in RESERVED or c == previous or c == current:
                continue
            rels_c = self._token_rels.get(c)
            if rels_c and (shared_pc & rels_c):
                votes[c] += 1.0
        return dict(votes)

    def _triple_vote(self, previous, current, candidates):
        """
        V8: for each candidate C, cast one full binary vote iff
        (previous, current, C) was ever literally a single, consecutive
        Bridge Matrix triple in training -- source=previous,
        bridge=current, target=C, all three in that exact order,
        checked against self._triples (see literal_triples()). Same
        fixed-pair SHAPE as V7 (one flat vote per candidate, not
        summed over context), but a strictly EXACT-MATCH question
        instead of V7's "did all three merely share a sentence"
        question -- (previous, current, C) can share a sentence
        without ever having occurred as this literal triple (V7 fires,
        V8 doesn't), but it can never occur as this literal triple
        without also sharing a sentence (V8 firing implies V7 would
        too, for the same candidate).

        Returns {} if `previous` or `current` is None or reserved. A
        candidate never votes via being `previous` or `current`
        itself, same "a token never votes for itself" rule as every
        other layer.
        """
        votes = Counter()
        if previous is None or current is None:
            return dict(votes)
        if previous in RESERVED or current in RESERVED:
            return dict(votes)
        for c in candidates:
            if c in RESERVED or c == previous or c == current:
                continue
            if (previous, current, c) in self._triples:
                votes[c] += 1.0
        return dict(votes)

    def _whole_context_vote(self, context_tokens, candidates):
        """
        V9: for each candidate C, cast one full binary vote iff EVERY
        non-reserved token currently in context (other than C itself,
        if C happens to already be a context token) has co-occurred
        with C in at least one training relationship. V3 asks "does
        at least one context token know C" -- this asks the strictly
        harder "do ALL of them" -- unanimous CONSENSUS rather than
        mere presence of evidence.

        Uses self._co_occurring (the same whole-vocabulary
        co-occurrence gate score_candidates()'s live path uses -- see
        co_occurrence_index()) so each "does t know C" check is an
        O(1)-average dict lookup against a precomputed row, checking
        the stored shared-relationship COUNT is > 0 -- not just key
        membership, since that same structure also carries adjacency-
        only backfill entries (shared=0) that must NOT count as
        "knowledge" here even though they're valid gate members for
        V6's sake elsewhere.

        A qualifying context token that equals C itself is excluded
        from the requirement, not from eligibility -- same "a token
        never has to vouch for itself" rule every other layer applies
        to voting, just checked once per required token here instead
        of skipped inside a sum. If that leaves NO qualifying tokens
        at all for a given C (e.g. context is empty, entirely
        reserved, or entirely equal to C itself), no vote is cast for
        C -- unanimity among zero voters is not evidence, the same
        reasoning V3 already applies to an empty context.
        """
        votes = Counter()
        required_all = [t for t in (context_tokens or []) if t not in RESERVED]
        if not required_all:
            return dict(votes)
        for c in candidates:
            if c in RESERVED:
                continue
            required = [t for t in required_all if t != c]
            if not required:
                continue
            if all(self._co_occurring.get(t, {}).get(c, 0) > 0 for t in required):
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

    def score_candidates(self, candidates, context_tokens, current=None, previous=None):
        """
        Open Mode's primary scoring pass -- evaluates EVERY candidate
        given, not just a pre-existing tie. NINE independent vote
        layers, summed (see the module docstring for the full
        rationale). An important token is ALSO a context token
        (I ⊆ P), so it votes in every context-token layer it
        qualifies for -- V3, V4, and V6 as a plain context member,
        PLUS V1 and V2 because it's important. Non-important context
        tokens cast V3/V4/V6. V7 and V8 are separate from all of
        that -- neither iterates over context at all, only `previous`
        and `current` specifically (see below and _prev_current_vote).
        V9 is separate again -- it's the only layer that requires
        EVERY context token to agree, not just one or a fixed pair.

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

            V7(C) = prev_current_weight × 1[rels(previous) ∩
                        rels(current) ∩ rels(C) ≠ ∅]
                "Previous+current co-occurrence" vote -- requires
                `previous` (this step's actual preceding token, a
                fixed value, NOT "every important token" or "every
                context token"); returns all zeros if `previous` or
                `current` is None. A single flat check per candidate,
                not summed over context: "was there ever one training
                sentence that mentioned previous, current, AND C, all
                together?" No adjacency required between any pair --
                broader than V5 in that one sense -- but it also
                requires BOTH previous and current (not just one
                token) to share a sentence with C, so it's narrower
                than V3. Default weight 1.0 -- PEER-weighted with
                V3/V5/V6: independently strong, specific, sentence-
                level evidence, not a structural nudge.

            V8(C) = triple_weight × 1[(previous, current, C) was ever
                        a literal, consecutive Bridge Matrix triple]
                "Triple witness" vote -- the strictest of all eight:
                not "did these three share a sentence" (V7), but "was
                source=previous, bridge=current, target=C ever the
                EXACT trained triple." Same fixed-pair shape as V7
                (one flat vote per candidate, not summed over
                context; all zeros if `previous` or `current` is
                None), but an exact three-way match instead of a
                three-way shared-sentence check -- the same literal
                test Strict Mode's Stage 2 already uses for legality,
                exposed here as a vote instead of a hard filter.
                Default weight 2.0 -- a notch above V5/V6/V7's peer
                weight, since exact-triple evidence is strictly more
                specific than any of theirs individually.

            V9(C) = whole_context_weight × 1[every non-reserved token
                        t ∈ P, t ≠ C, satisfies knowledge(t, C) > 0]
                "Whole context" / UNANIMOUS vote -- V3 asks "does at
                least one context token know C"; this asks the
                strictly harder "do ALL of them" (excluding C itself
                from the requirement if it's already a context
                token -- same self-exclusion rule as every other
                layer). One flat binary vote, not scaled by context
                size or evidence magnitude -- unanimity either holds
                or it doesn't. No vote for ANY candidate if context
                has no qualifying tokens at all (empty, fully
                reserved, or entirely equal to C) -- unanimity among
                zero voters proves nothing. Default weight 2.5 -- see
                config.py's WHOLE_CONTEXT_WEIGHT for why it sits a
                notch above V8.

            score(C) = V1(C) + V2(C) + V3(C) + V4(C) + V5(C) + V6(C) + V7(C) + V8(C) + V9(C)

        By construction (important_weight, influence_weight, and
        context_influence_weight all smaller than context_weight),
        no amount of important-token or magnitude support can outvote
        a genuine context-vote (V3) lead -- V1+V2+V4 can only ever
        nudge a decision V3 left open, never override one it already
        made. Confirm this holds for your own weights if you change
        them: it's a property of the ratios, not guaranteed
        automatically. V5, V6, V7, V8, and V9 are the five layers NOT
        bound by that rule -- each is independently strong, specific
        evidence, and by default can each outweigh V3 on its own
        (weight >= 1.0, same as or greater than V3).

        THE GATE: V1, V2, V3, V4, V6, and V9 all reduce to sums or
        conjunctions of "did t and C ever co-occur" (self._co_occurring
        -- see co_occurrence_index()) -- so if NO context token has
        EVER co-occurred with a given candidate C, every one of those
        six layers is guaranteed 0 for C, by construction, not by
        approximation:
          - V1/V2 only consider important tokens, a SUBSET of context.
          - V3/V4 sum over exactly the tokens the gate checks.
          - V6 requires t immediately followed by C in training, which
            implies t and C shared that sentence -- a STRICTER
            condition than mere co-occurrence, so V6-eligible implies
            gate-eligible.
          - V9 requires ALL context tokens to co-occur with C, which
            trivially implies at least one does.
        V5 sums over context tokens too (checking a stronger,
        bigram-specific condition), so the same argument applies to it
        UNCONDITIONALLY. V7 and V8 check `previous`/`current` directly
        rather than summing over context, so the gate only covers them
        when `previous` and `current` are THEMSELVES members of
        `context_tokens` -- true for every real generation call (see
        inference.py/model.py, which always build context_tokens as
        the full token sequence including both), but NOT guaranteed
        for an arbitrary caller. This method checks that condition at
        runtime before applying the gate to V7/V8, and falls back to
        scoring them against the full candidate list (never skipping
        work that might change the answer) when it doesn't hold --
        e.g. a caller (see test.py) that deliberately narrows
        context_tokens to isolate one layer in a unit test.

        Returns a full trace dict, not just scores, so this stays
        auditable: {"important_tokens": [...], "knows": {...},
        "influence": {...}, "important_vote": {C: V1},
        "influence_vote": {C: V2}, "context_vote": {C: V3},
        "context_influence_vote": {C: V4}, "bigram_witness_vote":
        {C: V5}, "adjacency_vote": {C: V6}, "prev_current_vote":
        {C: V7}, "triple_vote": {C: V8}, "whole_context_vote": {C: V9},
        "scores": {C: V1+V2+...+V9}, "cache_used": bool}.
        "scores" is the one that actually drives select(); the nine
        components are exposed separately (already weighted, so they
        sum directly to "scores") so each stays independently
        auditable. Every candidate in `candidates` gets a "scores"
        entry (0 if no layer voted for it); the per-layer dicts only
        have entries where that layer actually cast a vote.

        If self._use_cache is on AND `candidates` (as a set) is exactly
        the set build_cache()/enable_cache() was last built for, V1,
        V2, V3, V4, and V6 are read from the precomputed per-token
        cache instead of recomputed from _token_rels/_important/
        _adjacent -- same formulas, same weights, identical numeric
        result, just without redoing the O(|context| x |candidates|)
        work every step (see build_cache()'s comment for why those five
        layers -- and only those five -- are cacheable this way). When
        the cache is off (or doesn't match), the same five layers are
        computed live via the gate above instead of the old dense
        O(|context| x |candidates|) loop: for each context token, only
        the candidates it's KNOWN to co-occur with (self._co_occurring
        row) are ever visited, so the loop's real cost tracks actual
        co-occurrence in the corpus, not vocabulary size. V5, V7, V8,
        and V9 are always computed live either way, each restricted to
        the gate-derived candidate set per the paragraph above.
        "cache_used" in the returned trace reports which path actually
        ran, so cached vs. live runs stay easy to compare/verify
        against each other. A candidate-set mismatch (e.g. Strict Mode
        calling this with a narrower successor set) silently falls
        back to the live path -- never a wrong answer, just not the
        fast one.
        """
        candidates = sorted(set(candidates))
        candidates_set = set(candidates)
        context_tokens = context_tokens or []
        context_set = set(context_tokens)
        important = self.important_tokens_in(context_tokens)
        use_cache = (self._use_cache and self._cache_candidates is not None
                     and self._cache_candidates == frozenset(candidates))

        if use_cache:
            knows = {}
            for t in important:
                row = self._token_cache.get(t)
                if not row:
                    continue
                k = {c: shared for c, (shared, _adj) in row.items() if shared}
                if k:
                    knows[t] = k
            influence = {t: len(k) for t, k in knows.items()}

            important_vote = Counter()
            influence_vote = Counter()
            for t, k in knows.items():
                for c in k:
                    important_vote[c] += self._important_weight * 1
                    influence_vote[c] += self._influence_weight * influence[t]

            raw_context_vote = Counter()
            raw_adjacency_vote = Counter()
            context_influence_vote = Counter()
            for t in context_tokens:
                if t in RESERVED:
                    continue
                row = self._token_cache.get(t)
                if not row:
                    continue
                for c, (shared, adjacent) in row.items():
                    if shared:
                        raw_context_vote[c] += 1.0
                        context_influence_vote[c] += self._context_influence_weight * shared
                    if adjacent:
                        raw_adjacency_vote[c] += 1.0
            context_vote = {c: self._context_weight * v for c, v in raw_context_vote.items()}
            context_influence_vote = dict(context_influence_vote)
            adjacency_vote = {c: self._adjacency_weight * v for c, v in raw_adjacency_vote.items()}
        else:
            # SINGLE merged pass for V1/V2/V3/V4/V6 -- the previous version
            # called _knows_rels() (for V1/V2, important tokens only), then
            # _context_vote(), _context_influence_vote(), and
            # _adjacency_vote() (for V3/V4/V6, every context token), each as
            # its OWN independent O(|context| x |candidates|) loop -- so an
            # important token that's also an ordinary context member (the
            # common case: I ⊆ P) had `shared = len(rels_t & rels_c)`
            # recomputed from scratch up to three separate times per
            # candidate (once in _knows_rels, again in _context_vote, again
            # in _context_influence_vote), on top of a fourth separate scan
            # for adjacency. A later version merged those four scans into
            # one loop over (t, c), still tested against every candidate in
            # `candidates` regardless of whether t and c had ever
            # co-occurred. This version goes one step further: instead of
            # `for c in candidates: rels_c = ...; shared = len(rels_t &
            # rels_c)`, it looks up self._co_occurring[t] -- precomputed
            # once per model in build() (see co_occurrence_index()) -- and
            # only visits the candidates t is ALREADY known to co-occur
            # (or be directly adjacent to) with. For a candidate t has never
            # heard of, this skips the work entirely rather than computing
            # an intersection that was always going to come back empty --
            # real cost now tracks actual co-occurrence in the corpus, not
            # |candidates| (typically the full vocabulary in Open Mode).
            # Same formulas, same weights, identical numeric output to the
            # previous versions (test.py's cached-vs-live parity checks
            # cover this, and co_occurrence_index()'s adjacency backfill is
            # what keeps V6 exact even for the rare degenerate-short-
            # sentence case -- see its docstring).
            knows = {}
            influence = {}
            raw_context_vote = Counter()
            raw_adjacency_vote = Counter()
            context_influence_vote = Counter()
            important_set = self._important

            for t in context_tokens:
                if t in RESERVED:
                    continue
                rels_t = self._token_rels.get(t)
                if not rels_t:
                    continue
                is_important = t in important_set
                t_knows = {} if is_important else None
                # GATE: only visit candidates t has ever actually
                # co-occurred (or, per co_occurrence_index()'s
                # backfill, been directly adjacent to) with, instead
                # of testing every candidate in the full set. Same
                # `shared`/`adjacent` values the old per-candidate
                # intersection would have computed -- see
                # co_occurrence_index()'s docstring for why this is
                # exactly equivalent, not an approximation.
                row = self._co_occurring.get(t)
                if not row:
                    continue
                for c, shared in row.items():
                    if c not in candidates_set:
                        continue
                    if shared:
                        raw_context_vote[c] += 1.0
                        context_influence_vote[c] += self._context_influence_weight * shared
                        if is_important:
                            t_knows[c] = shared
                    if (t, c) in self._adjacent:
                        raw_adjacency_vote[c] += 1.0
                if is_important and t_knows:
                    knows[t] = t_knows
                    influence[t] = len(t_knows)

            # V1 -- important tokens vote independently, RAW/binary, scaled small
            # V2 -- influence casts its OWN vote, scaled, per token that knows C
            important_vote = Counter()
            influence_vote = Counter()
            for t, k in knows.items():
                inf = influence[t]
                for c in k:
                    important_vote[c] += self._important_weight * 1
                    influence_vote[c] += self._influence_weight * inf

            # V3 -- every context token, important or not, casts a full binary vote
            context_vote = {c: self._context_weight * v for c, v in raw_context_vote.items()}

            # V4 -- already accumulated above (context_influence_vote)

            # V6 -- every context token casts a full vote iff it was ever
            # literally, immediately followed by C in training (t->C only)
            adjacency_vote = {c: self._adjacency_weight * v
                               for c, v in raw_adjacency_vote.items()}
            context_influence_vote = dict(context_influence_vote)
            important_vote = dict(important_vote)
            influence_vote = dict(influence_vote)

        # GATE for V5/V7/V8/V9: any candidate that raised a V3 or V6
        # vote above is a candidate at least one context token has
        # co-occurred (or, via the adjacency backfill, been directly
        # adjacent) with. V5 and V9 are BOTH sums/conjunctions over
        # context tokens themselves, so restricting them to this set is
        # always exact (see the module docstring's "THE GATE" section
        # for the proof) -- no candidate outside it could ever have
        # scored nonzero on V5 or V9 anyway. V7/V8 check `previous`/
        # `current` directly rather than summing over context, so the
        # same restriction is only valid for them when both are
        # themselves members of context_tokens (true for every real
        # generation call -- see model.py/inference.py -- but not
        # guaranteed for an arbitrary caller, e.g. some of test.py's
        # isolated-layer checks); otherwise fall back to the full
        # candidate list for V7/V8 specifically, so nothing is ever
        # silently under-scored.
        gate_candidates = (set(raw_context_vote) | set(raw_adjacency_vote)) & candidates_set

        # V5 -- every context token casts a full vote iff it literally
        # witnessed the exact (current, C) bigram in a training sentence.
        # Always live (see build_cache()'s comment for why) -- depends on
        # `current`, not just t/c, so it can't be a per-token cache entry.
        raw_bigram_witness_vote = self._bigram_witness_vote(current, context_tokens, gate_candidates)
        bigram_witness_vote = {c: self._bigram_witness_weight * v
                                for c, v in raw_bigram_witness_vote.items()}

        v78_gate_safe = ((previous is None or previous in context_set)
                          and (current is None or current in context_set))
        v78_candidates = gate_candidates if v78_gate_safe else candidates_set

        # V7 -- a single fixed-pair check: did previous, current, and C
        # ever all three share one training sentence together
        raw_prev_current_vote = self._prev_current_vote(previous, current, v78_candidates)
        prev_current_vote = {c: self._prev_current_weight * v
                              for c, v in raw_prev_current_vote.items()}

        # V8 -- a single fixed-triple check: was (previous, current, C)
        # ever literally one consecutive trained Bridge Matrix triple
        raw_triple_vote = self._triple_vote(previous, current, v78_candidates)
        triple_vote = {c: self._triple_weight * v
                       for c, v in raw_triple_vote.items()}

        # V9 -- unanimous vote: every non-reserved, non-self context
        # token has to know C, not just one (V3's question). Gated the
        # same unconditional way as V5 -- see the comment above.
        raw_whole_context_vote = self._whole_context_vote(context_tokens, gate_candidates)
        whole_context_vote = {c: self._whole_context_weight * v
                               for c, v in raw_whole_context_vote.items()}

        final_scores = {
            c: important_vote.get(c, 0) + influence_vote.get(c, 0)
               + context_vote.get(c, 0) + context_influence_vote.get(c, 0)
               + bigram_witness_vote.get(c, 0) + adjacency_vote.get(c, 0)
               + prev_current_vote.get(c, 0) + triple_vote.get(c, 0)
               + whole_context_vote.get(c, 0)
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
                "prev_current_vote": prev_current_vote,
                "triple_vote": triple_vote,
                "whole_context_vote": whole_context_vote,
                "scores": final_scores,
                "cache_used": use_cache}

    def _bigram_frequency(self, token, candidate):
        """How many times `candidate` literally followed `token` (+ experience-implied, in open mode)."""
        return self._bigram_freq.get((token, candidate), 0)

    def _global_frequency(self, candidate):
        """Candidate's raw corpus-wide relationship count -- LAST-RESORT tie-break only, see select()."""
        return len(self._token_rels.get(candidate, set()))

    def select(self, candidates, context_tokens, current=None, previous=None):
        """
        Deterministic winner among `candidates` by the combined score
        (V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9, see
        score_candidates), with a three-stage deterministic tie-break
        cascade when the combined score itself doesn't discriminate:

            1. score (V1..V9)       -- primary, structural + contextual
            2. bigram frequency     -- how often `candidate` literally
                                        followed `current` (requires
                                        `current`; skipped if not given)
            3. global frequency     -- candidate's raw corpus-wide
                                        frequency (last resort only)
            4. lowest token id      -- final tie-break, always available

        `current` and `previous` are also forwarded into
        score_candidates() itself now (not just used here for the
        tie-break), since V5 needs `current` to look up which bigram
        is being scored and V7/V8 each need BOTH `previous` and
        `current` to check their respective three-way conditions --
        passing them through once is enough; V5 degrades to all-zero
        votes if `current` is omitted, V7/V8 degrade to all-zero votes
        if either is omitted.

        Each stage only ever narrows a tie the stage before it left
        open; it never overrides a decision an earlier stage already
        made. Returns (winner_or_None, trace_dict). winner is None
        only when `candidates` itself is empty.
        """
        trace = self.score_candidates(candidates, context_tokens, current=current, previous=previous)
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
            "prev_current_weight": self._prev_current_weight,
            "triple_weight": self._triple_weight,
            "whole_context_weight": self._whole_context_weight,
            "bigram_freq": {f"{t}:{c}": n for (t, c), n in self._bigram_freq.items()},
            "bigram_rels": {f"{t}:{c}": sorted(r) for (t, c), r in self._bigram_rels.items()},
            "adjacent": [list(pair) for pair in self._adjacent],
            "triples": [list(tri) for tri in self._triples],
        }

    @classmethod
    def from_dict(cls, d):
        ivm = cls()
        ivm._token_rels = {int(t): set(r) for t, r in d["token_rels"].items()}
        ivm._important = set(d["important"])
        ivm._important_weight = d.get("important_weight", IVMConfig.IMPORTANT_WEIGHT)
        ivm._influence_weight = d.get("influence_weight", d.get("freq_weight", IVMConfig.INFLUENCE_WEIGHT))
        ivm._context_weight = d.get("context_weight", IVMConfig.CONTEXT_WEIGHT)
        ivm._context_influence_weight = d.get("context_influence_weight", IVMConfig.CONTEXT_INFLUENCE_WEIGHT)
        ivm._bigram_witness_weight = d.get("bigram_witness_weight", IVMConfig.BIGRAM_WITNESS_WEIGHT)
        ivm._adjacency_weight = d.get("adjacency_weight", IVMConfig.ADJACENCY_WEIGHT)
        ivm._prev_current_weight = d.get("prev_current_weight", IVMConfig.PREV_CURRENT_WEIGHT)
        ivm._triple_weight = d.get("triple_weight", IVMConfig.TRIPLE_WEIGHT)
        ivm._whole_context_weight = d.get("whole_context_weight", IVMConfig.WHOLE_CONTEXT_WEIGHT)
        ivm._bigram_freq = {}
        for key, n in d.get("bigram_freq", {}).items():
            t, c = key.split(":")
            ivm._bigram_freq[(int(t), int(c))] = n
        ivm._bigram_rels = {}
        for key, r in d.get("bigram_rels", {}).items():
            t, c = key.split(":")
            ivm._bigram_rels[(int(t), int(c))] = set(r)
        ivm._adjacent = {tuple(pair) for pair in d.get("adjacent", [])}
        ivm._triples = {tuple(tri) for tri in d.get("triples", [])}
        # Derived, not stored -- same "re-derive on load" treatment as
        # everything else here that's fully computable from data
        # already in this dict (see build()'s matching comment).
        ivm._co_occurring = co_occurrence_index(ivm._token_rels, adjacent=ivm._adjacent)
        return ivm
