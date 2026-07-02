"""
inference.py — Single deterministic inference engine for MSE-GLM.

InferenceEngine — two-stage lineage-vote pipeline:
    Stage 1  current token authority  (vote for candidates it knows as its
                                       own outgoing bridge, filtered by
                                       active_rels lineage)
    Stage 2  previous token authority (exact triple match, lineage-narrowed
                                       to rel_ids shared by previous+current)

Strict mode: InferenceEngine(E, B, R)          — training data only
Open mode:   InferenceEngine(E+EE, B+EB, R+ER) — training + experience

Passing exp_edge/exp_bridge/exp_rel=None degrades gracefully to strict.
"""

from collections import Counter
from tokenizer import EOS

class InferenceEngine:
    """
    Two-stage deterministic pipeline driven by relationship-id lineage
    rather than vote counting.

    At every step, active_rels is the set of training relationship_ids the
    generation path has been consistent with so far (seeded from the prompt
    itself — see generate()).

    Stage 1 — Current token authority.
        Collect legal successors from E (+EE). For each candidate C, check
        whether a bridge triple (source=current, bridge=C) exists with a
        rel_id in active_rels — "I only vote for whom I know as my bridge,
        with a matching relationship_id." If active_rels is empty (start of
        generation with no lineage yet), any candidate with a bridge triple
        passes. Unique winner → done. Tie + no previous token → pick the
        first (all tied candidates are equally legitimate). Tie + previous
        exists → Stage 2.

    Stage 2 — Previous token authority.
        First narrow active_rels to the rel_ids shared by the exact
        (previous, current) pair, if any exist — this keeps the lineage
        tied to what previous and current were actually trained on
        together. Then, among the Stage 1 survivors, vote for candidate C
        if the exact triple (source=previous, bridge=current, target=C)
        exists with a rel_id in the (possibly narrowed) active_rels.
        Unique winner → done. Tie → pick the first (genuinely ambiguous;
        all tied candidates are valid continuations). No survivors at all
        → fall back to the first Stage 1 candidate.

    Termination: if Stage 1 finds no legal successors at all (bigram check
    fails), emit <EOS> immediately.

    active_rels narrows by intersection at every successful step (never
    replaced outright) so a shared triple cannot widen a lineage that an
    earlier, more specific step already established.
    """

    def __init__(self, edge, bridge, rel, exp_edge=None, exp_bridge=None, exp_rel=None):
        self.edges      = edge
        self.bridges    = bridge
        self.rels       = rel
        self.exp_edges  = exp_edge
        self.exp_bridge = exp_bridge
        self.exp_rel    = exp_rel


    # ── helpers ───────────────────────────────────────────────────────────

    def _successors(self, token):
        """All legal next tokens from E (and EE in open mode). Sorted for determinism."""
        s = set(self.edges.successors(token))
        if self.exp_edges:
            s.update(self.exp_edges.successors(token))
        return sorted(s)

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
        if self.exp_bridge and current + 1 < len(self.exp_bridge.index):
            start, end = self.exp_bridge.index[current], self.exp_bridge.index[current + 1]
            for i in range(start, end):
                if self.exp_bridge.bridge[i] == candidate and self.exp_rel:
                    out.update(self.exp_rel.relationships_for_exp_triple(i))
                    out.update(self.exp_rel.training_rels_for_exp_triple(i))
        return out

    def _exact_triple_rels(self, previous, current, target):
        """
        Stage 2 signal: rel_ids of the exact triple source=previous, bridge=current,
        target=target. If target is None, returns all rel_ids for source=previous,
        bridge=current (any target). Used both in Stage 2 voting and prompt seeding.
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
        if self.exp_bridge and previous + 1 < len(self.exp_bridge.index):
            start, end = self.exp_bridge.index[previous], self.exp_bridge.index[previous + 1]
            for i in range(start, end):
                if self.exp_bridge.bridge[i] == current:
                    if target is None or self.exp_bridge.target[i] == target:
                        if self.exp_rel:
                            out.update(self.exp_rel.relationships_for_exp_triple(i))
                            out.update(self.exp_rel.training_rels_for_exp_triple(i))
        return out

    def _rel_ids(self, previous, current, target):
        """Alias for _exact_triple_rels — used by prompt seeding in generate()."""
        return self._exact_triple_rels(previous, current, target)

    def _narrow(self, active_rels, new_rels):
        if not active_rels:
            return new_rels
        narrowed = active_rels & new_rels
        return narrowed if narrowed else new_rels

    # ── main step ─────────────────────────────────────────────────────────

    def step(self, previous, current, active_rels=None):
        active_rels = active_rels or set()

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
        if previous is None:
            # No previous context — all tied candidates are equally valid
            token = next(iter(s1_pass))   # storage/sorted order
            new_rels = self._narrow(active_rels, s1_pass[token])
            return token, {"stage": 1, "rule": "no_previous_all_valid",
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
            # Still tied — all are genuinely valid continuations
            token = next(iter(s2_pass))
            new_rels = self._narrow(active_rels, s2_pass[token])
            return token, {"stage": 2, "rule": "all_valid_first",
                           "chosen": token, "active_rels": new_rels,
                           "candidates": list(s2_pass)}

        # Stage 2 found nothing — fall back to Stage 1 first candidate
        token = next(iter(s1_pass))
        new_rels = self._narrow(active_rels, s1_pass[token])
        return token, {"stage": 1, "rule": "s2_empty_s1_first",
                       "chosen": token, "active_rels": new_rels,
                       "candidates": list(s1_pass)}

    def generate(self, prompt_ids, max_tokens=40):
        ids         = list(prompt_ids)
        # ── seed active_rels from prompt tokens ───────────────────────────
        # Walk every consecutive triple in the prompt and narrow active_rels
        # exactly as we would during generation, so the lineage established
        # by the prompt carries into the first generated token.
        active_rels = set()
        for i in range(len(ids) - 2):
            rels = self._rel_ids(ids[i], ids[i+1], ids[i+2])
            if rels:
                active_rels = self._narrow(active_rels, rels)
            else:
                active_rels = set()   # lineage broken — reset, don't keep stale lock
        # ─────────────────────────────────────────────────────────────
        trace       = []
        previous    = ids[-2] if len(ids) >= 2 else None
        current     = ids[-1]
        for _ in range(max_tokens):
            token, step_trace = self.step(previous, current, active_rels)
            trace.append(step_trace)
            if token == EOS:
                break
            ids.append(token)
            active_rels = set(step_trace.get("active_rels", active_rels)) or active_rels
            previous, current = current, token
        return ids, trace

    def infer_shared_role(self, tokens):
        """
        Cluster-based shared-role query across both training B and ExpB.
        Returns ranked (predicted_token, axis, evidence) list.
        """
        t_index_combined = {}
        for tok_id, cids in self.bridges.t_index.items():
            t_index_combined.setdefault(tok_id, set()).update(cids)
        if self.exp_bridge:
            for tok_id, cids in self.exp_bridge.t_index.items():
                t_index_combined.setdefault(tok_id, set()).update(cids)

        cluster_sets = [t_index_combined.get(t, set()) for t in tokens]
        if not cluster_sets or any(not s for s in cluster_sets):
            return []
        shared = set.intersection(*cluster_sets)
        if not shared:
            return []

        results = []
        for cid in sorted(shared):
            # check training B first, then ExpB
            for matrix, label in [(self.bridges, "training"), (self.exp_bridge, "experience")]:
                if matrix is None:
                    continue
                axis, members = matrix.cluster_axis(cid)
                if not members:
                    continue
                if axis == "bridge":
                    _, target, _ = members[0]
                    overlap = sum(1 for cs in cluster_sets if cid in cs)
                    results.append((target, "bridge_axis", {
                        "cluster_id": cid, "overlap": overlap, "source": label}))
                elif axis == "target":
                    _, _, bridge = members[0]
                    overlap = sum(1 for cs in cluster_sets if cid in cs)
                    results.append((bridge, "target_axis", {
                        "cluster_id": cid, "overlap": overlap, "source": label}))
                break

        results.sort(key=lambda r: r[2]["overlap"], reverse=True)
        return results
