"""
ctm.py — Context Trigger Matrix (CTM) for MSE-GLM.

The Bridge Matrix already answers "which tokens can legally occupy the
same structural slot?" (cat/dog/pig are interchangeable subjects of
"sat"). It has no answer for "which one SHOULD occupy it, given this
particular context?" -- that's a different question, and this module
answers it using only structure already in the trained matrices: no
new training, no embeddings, no probabilities.

Core idea
---------
For every member of a cluster, look at every training sentence
(relationship_id) that supports that member's place in the cluster,
and collect every OTHER token that appears in those sentences. That's
the member's trigger signature -- the whole-sentence company it keeps,
not just its immediate 3-token window (importance.py's trigger_matrix
covers that narrower, local case; this is the broader one). A trigger
token's support for a member is the number of DISTINCT sentences in
which they co-occurred.

At generation time, when the graph is genuinely torn between two or
more members of the same cluster (a real tie, not a case with a
unique legal answer), score each tied member by summing the support
of whichever of its known triggers are present in the context
generated so far. The candidate with the highest score wins. If CTM
itself doesn't discriminate (every candidate scores 0, or several tie
for the top score), generation falls back to the existing random
tie-break exactly as before -- this is a NEW, EARLIER disambiguation
step inserted before the coin flip, not a replacement for it.

Honesty notes
--------------
  - Strict mode is bounded to the literally-seen training corpus, and
    always will be: trigger evidence (token_to_relationships and the
    sentences it points to) is built ONLY from the strict Relationship
    Matrix, in both strict AND open mode, because that's the only
    place real surrounding context exists -- Experience Matrix triples
    are inferred by cluster substitution, never observed in an actual
    sentence, so there's no real context to mine for them even if you
    wanted to. This does not change between modes; it's a property of
    where context comes from, not a mode setting.
  - What DOES change in open mode is which CLUSTERS are eligible for
    disambiguation in the first place: strict mode only recognizes
    cluster_ids from the (literal) Bridge Matrix; open mode also
    recognizes cluster_ids from the Experience Bridge Matrix (see
    _cluster_axis_any / _all_cluster_ids). A cluster that only exists
    through inference -- e.g. two tokens that were never literally
    seen sharing a slot, only inferred to via cluster substitution --
    is invisible to CTM in strict mode and can be disambiguated in
    open mode, using triggers built entirely from real sentences that
    happen to mention its (real, literally-occurring) member tokens
    elsewhere. Open mode widens which candidates CTM can adjudicate
    between; it never widens what counts as evidence.
  - Reserved tokens (<PAD>/<UNK>/<BOS>/<EOS>) are excluded from
    trigger signatures. Including them would inflate every member's
    score roughly equally (they appear in nearly every sentence) while
    adding zero discriminative value -- pure noise against the whole
    point of this mechanism. This is a deliberate deviation from a
    literal reading of "collect all surrounding tokens."
  - Very common tokens (e.g. "the", "on") will still end up as
    triggers for many unrelated members, for the same underlying
    reason -- they just aren't filtered by a fixed exclusion list the
    way reserved tokens are, because "common" is corpus-dependent and
    a fixed stopword list isn't something this codebase has committed
    to elsewhere. On a corpus with many short, structurally similar
    sentences, expect low-information triggers to accumulate support
    fast; `min_support` is a floor, not a quality guarantee, same
    caveat as every other threshold in this project's analysis layer.
  - Support counts distinct SENTENCES a trigger/member pair
    co-occurred in, not distinct triples -- if a member's cluster
    membership is attested by several triples that all happen to come
    from one sentence, that sentence's other tokens are only counted
    once each, not once per triple.
  - This module never changes what inference.py decides UNLESS it is
    explicitly opted into (a ContextTriggerMatrix instance must be
    passed to InferenceEngine.step()/generate() by the caller). Every
    existing call site that doesn't pass one gets IDENTICAL behavior
    to before this module existed.
"""

from collections import Counter, defaultdict

RESERVED = {0, 1, 2, 3}  # <PAD>, <UNK>, <BOS>, <EOS>


def token_to_relationships(model):
    """
    Every relationship_id (training sentence) each token appears in,
    ANYWHERE in that sentence -- not just where it happens to sit in
    one particular cluster's defining triple. Derived once by scanning
    every triple's source/bridge/target roles; a token only fails to
    appear here if it never occurs in any sentence of length >= 3
    (degenerate short sentences have no triples at all, see
    importance.sequence_for_relationship).

    This is the "retrieve every relationship containing that token"
    step from the proposal, taken literally and globally rather than
    scoped to one cluster -- computing it once and sharing it across
    every cluster's signature build (see ContextTriggerMatrix.build)
    is both more correct and far cheaper than rebuilding an equivalent
    per-cluster index for every cluster in the model.
    """
    token_rels = defaultdict(set)
    b = model.bridges
    for tid in range(len(b.source)):
        rel_ids = model.rels.relationships_for_triple(tid)
        if not rel_ids:
            continue
        for tok in (b.source[tid], b.bridge[tid], b.target[tid]):
            token_rels[tok].update(rel_ids)
    return token_rels


def _cluster_axis_any(model, cluster_id, mode):
    """
    Look up a cluster's axis/members, checking the strict Bridge Matrix
    first and, only in Open Mode, falling back to the Experience Bridge
    Matrix. Experience cluster_ids are assigned starting at
    max(strict_cluster_id) + 1 (see experience.py), so the two id
    spaces never collide -- a cluster_id unambiguously belongs to one
    matrix or the other, never both.

    This is the "open mode can use everything else" half of this
    module's design: which CLUSTERS are eligible for disambiguation
    widens in Open Mode. It does NOT mean trigger evidence itself
    stops being bounded to the literally-seen corpus -- see
    token_to_relationships, which never changes behavior based on mode
    for exactly that reason.
    """
    axis, triples = model.bridges.cluster_axis(cluster_id)
    if triples:
        return axis, triples
    if mode == "open" and model.exp_bridges:
        axis, triples = model.exp_bridges.cluster_axis(cluster_id)
        if triples:
            return axis, triples
    return None, []


def build_context_triggers(model, cluster_id, mode="strict", token_rels=None):
    """
    Build per-member trigger signatures for one cluster. For each
    member, gathers every sentence that mentions it ANYWHERE (via
    token_rels), and counts every other token that co-occurs with it
    across those sentences.

    In strict mode, only cluster_ids from the (literal) Bridge Matrix
    are recognized. In open mode, cluster_ids from the Experience
    Bridge Matrix are ALSO recognized (see _cluster_axis_any) -- but
    the trigger evidence itself (token_rels) is always built from the
    literal training corpus only, in both modes: inferred Experience
    triples were never part of an actual sentence, so there is no real
    surrounding context to mine for them. Widening `mode` to "open"
    means more clusters become eligible for CTM disambiguation, not
    that fabricated context gets invented for members that only exist
    through inference.

    Pass a precomputed token_rels (from token_to_relationships) when
    building signatures for many clusters in one pass -- see
    ContextTriggerMatrix.build. Building it fresh here is only for
    convenience when inspecting a single cluster on its own.

    Returns None if cluster_id is unknown/degenerate. Otherwise:
        {"cluster_id", "axis", "members": [token_id, ...],
         "signatures": {member_token: {trigger_token: support_count}}}
    """
    from importance import sequence_for_relationship

    axis, triples = _cluster_axis_any(model, cluster_id, mode)
    if not triples:
        return None
    members = sorted(set(br for _, _, br in triples)) if axis == "bridge" \
        else sorted(set(t for _, t, _ in triples))

    if token_rels is None:
        token_rels = token_to_relationships(model)

    signatures = {}
    for member in members:
        rel_ids = token_rels.get(member, set())
        trigger_counts = Counter()
        for rid in rel_ids:
            seq = sequence_for_relationship(model, rid)
            distinct_triggers = {t for t in seq if t != member and t not in RESERVED}
            for trig in distinct_triggers:
                trigger_counts[trig] += 1
        signatures[member] = dict(trigger_counts)

    return {"cluster_id": cluster_id, "axis": axis, "members": members,
            "signatures": signatures}


def _all_cluster_ids(model, mode):
    """Every non-zero cluster_id eligible under this mode -- strict
    Bridge Matrix always; Experience Bridge Matrix too in open mode."""
    seen = set(model.bridges.cluster_id)
    seen.discard(0)
    if mode == "open" and model.exp_bridges:
        seen.update(c for c in model.exp_bridges.cluster_id if c != 0)
    return seen


def build_context_trigger_matrix(model, min_support=1, mode="strict"):
    """
    Flat table across every non-zero cluster eligible under `mode`
    (strict clusters always; Experience clusters too when mode="open"),
    matching the proposal's schema: one row per (trigger_token,
    cluster_id, member_token) with its support count. Sorted by
    support descending.
    """
    seen = _all_cluster_ids(model, mode)
    token_rels = token_to_relationships(model)

    rows = []
    for cid in sorted(seen):
        sig = build_context_triggers(model, cid, mode=mode, token_rels=token_rels)
        if not sig:
            continue
        for member, triggers in sig["signatures"].items():
            for trig, support in triggers.items():
                if support >= min_support:
                    rows.append({
                        "trigger_token": trig,
                        "cluster_id": cid,
                        "member_token": member,
                        "support": support,
                    })
    rows.sort(key=lambda r: (-r["support"], r["cluster_id"]))
    return rows


class ContextTriggerMatrix:
    """
    Precomputed trigger signatures for every cluster in a model, built
    once and reused across many scoring calls (building fresh per call
    would rescan the whole graph every time -- this is the
    precompute-once-query-many pattern the rest of the codebase
    already uses for Edge/Bridge/Relationship).
    """

    def __init__(self):
        self._sigs = {}       # cluster_id -> {member: {trigger: support}}
        self._axis = {}       # cluster_id -> "bridge"|"target"

    @classmethod
    def build(cls, model, mode="strict", min_support=1):
        ctm = cls()
        seen = _all_cluster_ids(model, mode)
        token_rels = token_to_relationships(model)
        for cid in sorted(seen):
            sig = build_context_triggers(model, cid, mode=mode, token_rels=token_rels)
            if not sig:
                continue
            filtered = {
                member: {t: s for t, s in triggers.items() if s >= min_support}
                for member, triggers in sig["signatures"].items()
            }
            ctm._sigs[cid] = filtered
            ctm._axis[cid] = sig["axis"]
        return ctm

    def score_members(self, cluster_id, context_tokens):
        """
        {member_token: score} for every member of cluster_id, where
        score = sum of support for every trigger present in
        context_tokens. Returns {} if cluster_id has no signature.
        """
        sig = self._sigs.get(cluster_id)
        if not sig:
            return {}
        ctx = set(context_tokens)
        return {
            member: sum(support for trig, support in triggers.items() if trig in ctx)
            for member, triggers in sig.items()
        }

    def select(self, cluster_id, context_tokens):
        """
        Highest-scoring member(s) for cluster_id given context_tokens.
        Returns None if no signature or every score is 0 (no signal).
        Otherwise {"top_members": [...], "score": int, "all_scores": {...}}
        -- top_members has more than one entry exactly when CTM itself
        is tied and can't discriminate further.
        """
        scores = self.score_members(cluster_id, context_tokens)
        if not scores:
            return None
        max_score = max(scores.values())
        if max_score <= 0:
            return None
        top = sorted(m for m, s in scores.items() if s == max_score)
        return {"top_members": top, "score": max_score, "all_scores": scores}

    def resolve_tie(self, candidates, bridges, exp_bridges, context_tokens):
        """
        Given a set of tied generation candidates (raw token ids), find
        a cluster_id they all share (via t_index) and use it to pick a
        unique winner. Returns the winning token id, or None if the
        candidates share no cluster, have no signature, or CTM itself
        can't discriminate -- callers should fall back to their normal
        tie-break in every None case.
        """
        if not context_tokens:
            return None
        candidates = sorted(candidates)
        cluster_sets = []
        for c in candidates:
            s = set(bridges.t_index.get(c, []))
            if exp_bridges:
                s |= set(exp_bridges.t_index.get(c, []))
            cluster_sets.append(s)
        if not cluster_sets or any(not s for s in cluster_sets):
            return None
        shared = set.intersection(*cluster_sets)
        for cid in sorted(shared):
            result = self.select(cid, context_tokens)
            if not result:
                continue
            relevant = [m for m in result["top_members"] if m in candidates]
            if len(relevant) == 1:
                return relevant[0]
        return None

    def to_dict(self):
        return {
            "sigs": {str(cid): {str(m): {str(t): s for t, s in triggers.items()}
                                 for m, triggers in sig.items()}
                     for cid, sig in self._sigs.items()},
            "axis": {str(cid): axis for cid, axis in self._axis.items()},
        }

    @classmethod
    def from_dict(cls, d):
        ctm = cls()
        ctm._sigs = {int(cid): {int(m): {int(t): s for t, s in triggers.items()}
                                 for m, triggers in sig.items()}
                     for cid, sig in d["sigs"].items()}
        ctm._axis = {int(cid): axis for cid, axis in d["axis"].items()}
        return ctm
