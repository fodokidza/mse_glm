"""
inference.py — Single deterministic inference engine for MSE-GLM.

Strict mode: InferenceEngine(E, B, R, mode="strict")
Open mode:   InferenceEngine(E, B, R, mode="open", vocab=<all token ids>)

Both modes share the SAME literal Edge/Bridge/Relationship matrices --
there is no separate "experience" data source anymore. What
distinguishes them is entirely how step() picks candidates and how it
decides among them; `self.mode` (set at construction) is checked at
the very top of step() and completely forks the two modes' logic —
see below.

STRICT MODE — two-stage lineage-vote pipeline, now with a deterministic
bigram-frequency tie-break in place of the old random fallback:
    Stage 1  current token authority  (vote for candidates it knows as its
                                       own outgoing bridge, filtered by
                                       active_rels lineage)
    Stage 2  previous token authority (exact triple match, lineage-narrowed
                                       to rel_ids shared by previous+current)
    Every genuine tie that lineage itself can't resolve now falls through
    to _bigram_tie_break(): whichever candidate literally followed
    `current` most often in training wins (EdgeMatrix.frequency), not an
    arbitrary random.choice. A residual tie in bigram frequency (including
    "neither was ever literally seen after current," both 0) resolves to
    the lowest token id -- still fully deterministic. No randomness
    anywhere in Strict Mode's decision path anymore.
    Optional opt-in add-ons (both no-ops unless passed), tried BEFORE the
    bigram-frequency fallback:
      context_triggers  — ContextTriggerMatrix (ctm.py), an extra
                           disambiguation step.
      importance_votes  — ImportanceVoteMatrix (ivm.py) in its LEGACY
                           tie-break role (resolve_tie): skips Stage 2
                           entirely and substitutes importance voting in
                           its place. Opt-in only; strict mode's default
                           behavior (no importance_votes passed) is
                           otherwise unchanged from before either add-on
                           existed.
    Candidates are gated by _successors(current) — only tokens literally
    seen as the next token after `current` somewhere in training.

OPEN MODE — no lineage tie-breaking and no successor gating at all, by
design (not merely skipped on a tie: Stage 1/2 are never reached for an
Open Mode engine, and _successors() is never consulted for candidates
either). Candidates are `self.vocab` — the ENTIRE vocabulary, precomputed
once at construction (see model.py's all_candidate_tokens()) — not
narrowed by whether a bigram was ever literally observed. EVERY
candidate is scored — via `importance_votes.select()` (ivm.py's
PRIMARY, not legacy, API) — using cluster-membership-derived
"important" context tokens plus seven other independent vote layers
(V1-V8, see ivm.py), and the highest scorer wins. This is deterministic
by construction (ties broken by lowest token id inside select()); if no
importance_votes object is supplied, or `self.vocab` is empty, the
fallback is ALSO deterministic — lowest token id in `self.vocab` — never
random.choice. See ivm.py's module docstring for the full formula and
rationale.
"""

from collections import Counter
from tokenizer import EOS
from config import GenerationConfig

class InferenceEngine:
    """
    STRICT MODE: two-stage deterministic pipeline driven by
    relationship-id lineage rather than vote counting.

    At every step, active_rels is the set of training relationship_ids the
    generation path has been consistent with so far (seeded from the prompt
    itself — see generate()).

    Stage 1 — Current token authority.
        Collect legal successors from E. For each candidate C, check
        whether a bridge triple (source=current, bridge=C) exists with a
        rel_id in active_rels — "I only vote for whom I know as my bridge,
        with a matching relationship_id." If active_rels is empty (start of
        generation with no lineage yet), any candidate with a bridge triple
        passes. Unique winner → done. Tie + no previous token → resolved by
        bigram frequency (_bigram_tie_break: whichever candidate literally
        followed `current` most often in training). Tie + previous exists
        → Stage 2.

    Stage 2 — Previous token authority.
        First narrow active_rels to the rel_ids shared by the exact
        (previous, current) pair, if any exist — this keeps the lineage
        tied to what previous and current were actually trained on
        together. Then, among the Stage 1 survivors, vote for candidate C
        if the exact triple (source=previous, bridge=current, target=C)
        exists with a rel_id in the (possibly narrowed) active_rels.
        Unique winner → done. Tie → resolved by bigram frequency, same as
        Stage 1. No survivors at all → fall back to bigram frequency among
        the Stage 1 survivors.

    Termination: if Stage 1 finds no legal successors at all (bigram check
    fails), emit <EOS> immediately.

    active_rels narrows by intersection at every successful step (never
    replaced outright) so a shared triple cannot widen a lineage that an
    earlier, more specific step already established.

    OPEN MODE has its own, entirely separate path — see _open_mode_step().
    """

    def __init__(self, edge, bridge, rel, mode="strict", vocab=None):
        self.edges   = edge
        self.bridges = bridge
        self.rels    = rel
        self.mode    = mode   # "strict" or "open" -- forks step()'s whole logic
        self.vocab   = vocab or []   # Open Mode only: full candidate pool, see model.all_candidate_tokens()


    # ── helpers ───────────────────────────────────────────────────────────

    def _successors(self, token):
        """All legal next tokens from E. Sorted for determinism. Strict Mode only --
        Open Mode uses self.vocab directly and never calls this for candidates
        (still used for generate()'s prompt-bigram validation in both modes)."""
        return sorted(self.edges.successors(token))

    def _bridge_rels_from_source(self, current, candidate):
        """
        Stage 1 signal: rel_ids of triples where source=current AND bridge=candidate.
        These are the sequences current→candidate→? that current was trained on.
        "Current knows candidate as its outgoing bridge."
        """
        out = set()
        if current + 1 < len(self.bridges.index):
            start, end = self.bridges.index[current], self.bridges.index[current + 1]
            for i in range(start, end):
                if self.bridges.bridge[i] == candidate and self.rels:
                    out.update(self.rels.relationships_for_triple(i))
        return out

    def _exact_triple_rels(self, previous, current, target):
        """
        Stage 2 signal: rel_ids of the exact triple source=previous, bridge=current,
        target=target. If target is None, returns all rel_ids for source=previous,
        bridge=current (any target). Used both in Stage 2 voting and prompt seeding.

        Kept for Strict Mode and general relationship bookkeeping (also used
        by generate()'s prompt-seeding, which still runs in both modes).
        Open Mode's primary candidate selection does NOT use this function.
        """
        out = set()
        if previous is None:
            return out
        if previous + 1 < len(self.bridges.index):
            start, end = self.bridges.index[previous], self.bridges.index[previous + 1]
            for i in range(start, end):
                if self.bridges.bridge[i] == current:
                    if target is None or self.bridges.target[i] == target:
                        if self.rels:
                            out.update(self.rels.relationships_for_triple(i))
        return out

    def _rel_ids(self, previous, current, target):
        """Alias for _exact_triple_rels — used by prompt seeding in generate()."""
        return self._exact_triple_rels(previous, current, target)

    def _narrow(self, active_rels, new_rels):
        if not active_rels:
            return new_rels
        narrowed = active_rels & new_rels
        return narrowed if narrowed else new_rels

    def _bigram_frequency(self, token, candidate):
        """
        How many times `candidate` literally followed `token` as a
        consecutive pair across training sequences (EdgeMatrix.frequency).
        """
        return self.edges.frequency(token, candidate)

    def _bigram_tie_break(self, current, tied_candidates):
        """
        STRICT MODE deterministic tie-break, replacing random.choice
        wherever Stage 1/2 lineage genuinely can't discriminate: among
        `tied_candidates`, pick whichever was literally seen most
        often as the immediate next token after `current` (bigram
        frequency -- see EdgeMatrix.frequency). If bigram frequency
        also ties (including the "never literally seen, both 0" case),
        fall back to the lowest token id -- still fully deterministic,
        just with no further evidence to discriminate on. Never
        randomness, matching the same "no randomness in the core
        mechanism" principle Open Mode's primary selection follows.
        """
        tied_candidates = sorted(tied_candidates)
        freqs = {c: self._bigram_frequency(current, c) for c in tied_candidates}
        max_freq = max(freqs.values())
        top = sorted(c for c, f in freqs.items() if f == max_freq)
        return top[0]

    def _ctm_resolve(self, tied_candidates, context_tokens, context_triggers):
        """
        Strict Mode only. Try to break a genuine tie using a Context
        Trigger Matrix, if one was passed for this call. Returns a
        winning token id, or None if context_triggers is None,
        context_tokens is empty, the candidates share no cluster, or
        CTM itself can't discriminate -- every None case means "fall
        back to the existing random tie-break," unchanged from before
        this existed.
        """
        if context_triggers is None:
            return None
        return context_triggers.resolve_tie(
            tied_candidates, self.bridges, context_tokens)

    def _ivm_resolve(self, tied_candidates, context_tokens, importance_votes):
        """
        Strict Mode only (legacy tie-break role). Try to break a
        genuine tie using an Importance Vote Matrix's resolve_tie(),
        if one was passed for this call. Returns a winning token id,
        or None if importance_votes is None, context_tokens is
        empty, no important tokens are present, none of them know
        any tied candidate, or the vote itself remains tied -- every
        None case means "fall through to the next resolution step,"
        same contract as _ctm_resolve.
        """
        if importance_votes is None:
            return None
        return importance_votes.resolve_tie(tied_candidates, context_tokens)

    def _open_mode_step(self, previous, current, candidates, context_tokens, importance_votes, active_rels):
        """
        OPEN MODE primary candidate selection. `candidates` is
        `self.vocab` — the ENTIRE vocabulary, not gated by whether a
        bigram was ever literally observed — every one of them is
        scored here, this is not a tie-break inserted after some
        other mechanism narrows things down. See ivm.py's
        select()/score_candidates() for the eight-layer weighted-
        voting formula (V1-V8; V1/V2/V4 deliberately weighted small
        so they can only nudge a tie V3 left open, never override it;
        V5/V6/V7/V8 peer-weighted with V3, V8 a notch above) and for
        the bigram-frequency →
        global-frequency → lowest-token-id tie-break cascade `current`
        feeds into. `previous` is forwarded too now, purely for V7
        (the previous+current co-occurrence vote) and V8 (the triple
        witness vote) -- it plays no role
        in the tie-break cascade itself, only `current` does.

        Deterministic, always: select() itself resolves any residual
        tie all the way down to lowest token id internally, so this
        method only ever returns None from importance_votes.select()
        when `candidates` itself is empty (can't happen here — step()
        already checked self.vocab is nonempty). The lowest-id
        fallback below only exists for that degenerate case, or when
        no CTM/IVM object was supplied at all — never random.choice,
        matching the "no randomness in the core mechanism" rule.
        Temperature/sampling, if ever wanted, belongs as an external
        layer on top of these scores, not inside this method.

        active_rels is passed straight through unchanged (Open Mode
        doesn't use it for candidate selection) — kept only as
        relationship bookkeeping/trace context, per the spec's note
        that _exact_triple_rels() and lineage data may still be
        useful for that purpose even though they no longer gate
        Open Mode's decision.
        """
        if importance_votes is not None:
            winner, trace = importance_votes.select(candidates, context_tokens,
                                                      current=current, previous=previous)
            if winner is not None:
                return winner, {"stage": 1, "rule": "ctm_weighted_vote",
                                 "chosen": winner, "active_rels": active_rels,
                                 "candidates": candidates, "scores": trace["scores"],
                                 "important_vote": trace.get("important_vote", {}),
                                 "influence_vote": trace.get("influence_vote", {}),
                                 "context_vote": trace.get("context_vote", {})}
        # No CTM/IVM object supplied at all, or `candidates` was
        # somehow empty (can't happen here — _successors already
        # checked nonempty before this method is called). Deterministic
        # fallback — lowest legal token id — never randomness.
        winner = candidates[0]
        return winner, {"stage": 1, "rule": "ctm_unavailable_deterministic_fallback",
                         "chosen": winner, "active_rels": active_rels,
                         "candidates": candidates}

    # ── main step ─────────────────────────────────────────────────────────

    def step(self, previous, current, active_rels=None, context_tokens=None,
              context_triggers=None, importance_votes=None):
        active_rels = active_rels or set()

        # ── OPEN MODE: candidates are ALWAYS the full vocabulary ───────────
        # No successor gating at all -- self.vocab (set once at construction,
        # see model.all_candidate_tokens()) is the entire candidate universe
        # every step, scored by IVM's V1-V8 weighted voting (ivm.py). Not an
        # opt-in: any engine constructed with mode="open" behaves this way
        # unconditionally. Strict Mode engines never reach this branch.
        if self.mode == "open":
            candidates = self.vocab
            if not candidates:
                return EOS, {"stage": 4, "rule": "termination_empty_vocabulary",
                             "chosen": EOS, "active_rels": active_rels}
            return self._open_mode_step(previous, current, candidates, context_tokens,
                                         importance_votes, active_rels)

        # ── STRICT MODE ────────────────────────────────────────────────────
        # ── Collect legal successors (bigram check) ───────────────────────
        succs = self._successors(current)
        if not succs:
            return EOS, {"stage": 4, "rule": "termination_no_successors",
                         "chosen": EOS, "active_rels": active_rels}

        # ── Stage 1: current token authority ─────────────────────────────
        # Vote for candidate C if triple (source=current, bridge=C) has
        # rel_id ∈ active_rels. "I only vote for whom I know as my bridge
        # and whose relationship_id matches mine."
        s1_pass = {}   # C → matching_rel_ids
        for C in succs:
            rels = self._bridge_rels_from_source(current, C)
            if active_rels:
                matched = rels & active_rels
                if matched:
                    s1_pass[C] = matched
            else:
                # No lineage yet — all candidates with any bridge triple pass
                if rels:
                    s1_pass[C] = rels

        # If no candidate has a bridge triple at all, all successors are equally
        # valid (the graph is sparse at this point) — let Stage 2 decide
        if not s1_pass:
            s1_pass = {C: set() for C in succs}

        if len(s1_pass) == 1:
            token = next(iter(s1_pass))
            new_rels = self._narrow(active_rels, s1_pass[token])
            return token, {"stage": 1, "rule": "bridge_lineage_unique",
                           "chosen": token, "active_rels": new_rels,
                           "candidates": list(s1_pass)}

        # Tie in Stage 1
        if importance_votes is not None:
            # Strict Mode legacy behavior: passing an ImportanceVoteMatrix
            # (its resolve_tie() role) SKIPS Stage 2's lineage tie-break
            # entirely and substitutes importance voting in its place —
            # unchanged, opt-in-only, from before Open Mode had its own
            # dedicated primary mechanism.
            winner = self._ivm_resolve(list(s1_pass), context_tokens, importance_votes)
            rule = "importance_vote_resolved"
            if winner is None:
                winner = self._ctm_resolve(list(s1_pass), context_tokens, context_triggers)
                rule = "context_trigger_resolved"
            if winner is None:
                winner = self._bigram_tie_break(current, s1_pass)
                rule = "importance_vote_bigram_frequency"
            new_rels = self._narrow(active_rels, s1_pass[winner])
            return winner, {"stage": 1, "rule": rule,
                             "chosen": winner, "active_rels": new_rels,
                             "candidates": list(s1_pass)}

        if previous is None:
            # No previous context — try Context Trigger resolution first,
            # then fall back to bigram frequency among tied candidates
            # (deterministic: whichever was literally seen most often
            # right after `current`; never random.choice).
            winner = self._ctm_resolve(list(s1_pass), context_tokens, context_triggers)
            token = winner if winner is not None else self._bigram_tie_break(current, s1_pass)
            new_rels = self._narrow(active_rels, s1_pass[token])
            rule = "context_trigger_resolved" if winner is not None else "no_previous_bigram_frequency"
            return token, {"stage": 1, "rule": rule,
                           "chosen": token, "active_rels": new_rels,
                           "candidates": list(s1_pass)}

        # ── Stage 2: previous token exact match with lineage ──────────────
        # First: check if previous and current share a relationship_id in
        # active_rels — narrow active_rels to only the shared ones
        prev_curr_rels = self._exact_triple_rels(previous, current, None)
        if active_rels and prev_curr_rels:
            narrowed = active_rels & prev_curr_rels
            if narrowed:
                active_rels = narrowed   # stay on the shared lineage

        # Vote for candidate C if exact triple (source=previous, bridge=current,
        # target=C) exists with rel_id ∈ active_rels
        s2_pass = {}
        for C in s1_pass:   # only consider Stage 1 survivors
            rels = self._exact_triple_rels(previous, current, C)
            if active_rels:
                matched = rels & active_rels
                if matched:
                    s2_pass[C] = matched
            else:
                if rels:
                    s2_pass[C] = rels

        if s2_pass:
            if len(s2_pass) == 1:
                token = next(iter(s2_pass))
                new_rels = self._narrow(active_rels, s2_pass[token])
                return token, {"stage": 2, "rule": "exact_match_unique",
                               "chosen": token, "active_rels": new_rels,
                               "candidates": list(s2_pass)}
            # Still tied — try Context Trigger resolution first, then
            # fall back to bigram frequency (deterministic, never random).
            winner = self._ctm_resolve(list(s2_pass), context_tokens, context_triggers)
            token = winner if winner is not None else self._bigram_tie_break(current, s2_pass)
            new_rels = self._narrow(active_rels, s2_pass[token])
            rule = "context_trigger_resolved" if winner is not None else "bigram_frequency"
            return token, {"stage": 2, "rule": rule,
                           "chosen": token, "active_rels": new_rels,
                           "candidates": list(s2_pass)}

        # Stage 2 found nothing — try Context Trigger resolution among
        # the Stage 1 survivors first, then fall back to bigram frequency.
        winner = self._ctm_resolve(list(s1_pass), context_tokens, context_triggers)
        token = winner if winner is not None else self._bigram_tie_break(current, s1_pass)
        new_rels = self._narrow(active_rels, s1_pass[token])
        rule = "context_trigger_resolved" if winner is not None else "s2_empty_bigram_frequency"
        return token, {"stage": 1, "rule": rule,
                       "chosen": token, "active_rels": new_rels,
                       "candidates": list(s1_pass)}

    def generate(self, prompt_ids, max_tokens=GenerationConfig.MAX_TOKENS, context_triggers=None,
                 importance_votes=None):
        ids = list(prompt_ids)

        # ── validate every adjacent bigram in the prompt ──────────────
        # If any (curr→next) pair does not exist in E, the prompt
        # contains a transition the model has never seen. Return
        # immediately — cannot legally continue. Applied in BOTH modes:
        # Open Mode's per-step candidates are the full vocabulary, but
        # the PROMPT itself must still start from real, literally-seen
        # transitions -- otherwise there's nothing for lineage/IVM to
        # ground the first step in.
        def _has_edge(a, b):
            return b in self.edges.successors(a)

        # ── validate prompt bigrams (skip BOS→first_token) ──────────
        # BOS is prepended artificially by encode() — it is not a real
        # training token with known successors. Any real token is a valid
        # generation starting point. Only validate pairs between real tokens.
        for i in range(1, len(ids) - 1):
            curr, nxt = ids[i], ids[i + 1]
            if not _has_edge(curr, nxt):
                return ids, [{"stage": 0,
                              "rule": "illegal_prompt_bigram",
                              "chosen": EOS,
                              "detail": f"bigram ({curr}->{nxt}) not in edge matrix",
                              "active_rels": set()}]

        # ── seed active_rels from prompt triples ──────────────────────
        active_rels = set()
        for i in range(len(ids) - 2):
            rels = self._rel_ids(ids[i], ids[i+1], ids[i+2])
            if rels:
                active_rels = self._narrow(active_rels, rels)
            else:
                active_rels = set()   # lineage broken — reset, don't keep stale lock
        # ─────────────────────────────────────────────────────────────
        trace    = []
        previous = ids[-2] if len(ids) >= 2 else None
        current  = ids[-1]
        for _ in range(max_tokens):
            context_tokens = set(ids)
            token, step_trace = self.step(previous, current, active_rels,
                                           context_tokens=context_tokens,
                                           context_triggers=context_triggers,
                                           importance_votes=importance_votes)
            trace.append(step_trace)
            if token == EOS:
                break
            ids.append(token)
            active_rels = set(step_trace.get("active_rels", active_rels)) or active_rels
            previous, current = current, token
        return ids, trace

    def infer_shared_role(self, tokens):
        """
        Cluster-based shared-role query over the training Bridge Matrix.
        Returns ranked (predicted_token, axis, evidence) list.
        """
        t_index_combined = {}
        for tok_id, cids in self.bridges.t_index.items():
            t_index_combined.setdefault(tok_id, set()).update(cids)

        cluster_sets = [t_index_combined.get(t, set()) for t in tokens]
        if not cluster_sets or any(not s for s in cluster_sets):
            return []
        shared = set.intersection(*cluster_sets)
        if not shared:
            return []

        results = []
        for cid in sorted(shared):
            axis, members = self.bridges.cluster_axis(cid)
            if not members:
                continue
            if axis == "bridge":
                _, target, _ = members[0]
                overlap = sum(1 for cs in cluster_sets if cid in cs)
                results.append((target, "bridge_axis", {
                    "cluster_id": cid, "overlap": overlap, "source": "training"}))
            elif axis == "target":
                _, _, bridge = members[0]
                overlap = sum(1 for cs in cluster_sets if cid in cs)
                results.append((bridge, "target_axis", {
                    "cluster_id": cid, "overlap": overlap, "source": "training"}))

        results.sort(key=lambda r: r[2]["overlap"], reverse=True)
        return results
