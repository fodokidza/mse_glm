"""
inference.py — Deterministic four-stage inference pipeline for MSE-GLM,
extended (v2.1) with relationship-lineage tie-breaking and the
infer_shared_role() cluster-based prediction mode.
"""

from collections import Counter
from tokenizer import EOS


class InferenceEngine:
    def __init__(self, edge_matrix, bridge_matrix, relationship_matrix=None):
        self.edges = edge_matrix
        self.bridges = bridge_matrix
        self.rels = relationship_matrix

    # ----------------------------------------------------------- triple ids
    def _triple_id(self, source, target, bridge):
        # linear-ish lookup restricted to this source's CSR slice
        if source < 0 or source + 1 >= len(self.bridges.index):
            return None
        start, end = self.bridges.index[source], self.bridges.index[source + 1]
        for idx in range(start, end):
            if self.bridges.target[idx] == target and self.bridges.bridge[idx] == bridge:
                return idx
        return None

    # --------------------------------------------------------------- step
    def step(self, previous, current, active_rels=None):
        """
        Returns (next_token, trace) where trace is a dict describing the
        stage and rule that produced the selection (for explain_step()).
        """
        active_rels = active_rels or set()

        # Stage 1 — exact bridge match
        if previous is not None:
            candidates = []  # (target, triple_id)
            if previous + 1 < len(self.bridges.index):
                start, end = self.bridges.index[previous], self.bridges.index[previous + 1]
                for idx in range(start, end):
                    if self.bridges.bridge[idx] == current:
                        candidates.append((self.bridges.target[idx], idx))

            if candidates:
                if len(candidates) == 1 or self.rels is None:
                    chosen = candidates[0]
                    rule = "single_match" if len(candidates) == 1 else "storage_order_no_R"
                else:
                    lineage_matches = []
                    for tgt, tid in candidates:
                        rel_ids = set(self.rels.relationships_for_triple(tid))
                        if rel_ids & active_rels:
                            lineage_matches.append((tgt, tid))
                    if lineage_matches:
                        chosen = lineage_matches[0]
                        rule = "lineage_match"
                    else:
                        chosen = candidates[0]
                        rule = "storage_order_fallback"
                token, triple_id = chosen
                candidate_rels = set(self.rels.relationships_for_triple(triple_id)) if self.rels else set()
                # Narrow the running lineage: intersect with what the path has matched so
                # far, rather than replacing it. A triple shared by several training
                # sequences (e.g. a common sat->on->the) must not erase the narrowing a
                # prior, more specific step already established. Only reset to the new
                # triple's full lineage if the path has genuinely diverged (no overlap).
                if active_rels:
                    narrowed = active_rels & candidate_rels
                    new_rels = narrowed if narrowed else candidate_rels
                else:
                    new_rels = candidate_rels
                return token, {
                    "stage": 1, "rule": rule, "candidates": candidates,
                    "chosen": token, "active_rels": new_rels,
                }

        # Stage 2 — bridge voting
        if previous is not None and previous + 1 < len(self.bridges.index):
            start, end = self.bridges.index[previous], self.bridges.index[previous + 1]
            votes = Counter()
            for idx in range(start, end):
                tgt, brg = self.bridges.target[idx], self.bridges.bridge[idx]
                votes[tgt] += 2.0 if brg == current else 1.0
            if votes:
                token = max(votes.items(), key=lambda kv: kv[1])[0]
                return token, {"stage": 2, "rule": "bridge_voting", "votes": dict(votes),
                                "chosen": token, "active_rels": active_rels}

        # Stage 3 — bigram voting
        successors = self.edges.successors(current)
        if successors:
            votes = Counter(successors)
            token = max(votes.items(), key=lambda kv: kv[1])[0]
            return token, {"stage": 3, "rule": "bigram_voting", "votes": dict(votes),
                            "chosen": token, "active_rels": active_rels}

        # Stage 4 — termination
        return EOS, {"stage": 4, "rule": "termination", "chosen": EOS, "active_rels": active_rels}

    # ------------------------------------------------------------ generate
    def generate(self, prompt_ids, max_tokens=40):
        ids = list(prompt_ids)
        active_rels = set()
        trace = []
        previous = ids[-2] if len(ids) >= 2 else None
        current = ids[-1]
        for _ in range(max_tokens):
            token, step_trace = self.step(previous, current, active_rels)
            trace.append(step_trace)
            if token == EOS:
                break
            ids.append(token)
            active_rels = set(step_trace.get("active_rels", active_rels)) or active_rels
            previous, current = current, token
        return ids, trace

    # ------------------------------------------------------- shared role
    def infer_shared_role(self, tokens):
        """
        Unordered multi-token query (SDD v2.1 section 3.5). Returns a
        ranked list of (predicted_token, axis, evidence) or [] if no
        shared cluster exists across all prompt tokens.
        """
        t_index = self.bridges.t_index
        cluster_sets = [set(t_index.get(t, [])) for t in tokens]
        if not cluster_sets or any(not s for s in cluster_sets):
            return []
        shared = set.intersection(*cluster_sets)
        if not shared:
            return []

        results = []
        for cid in sorted(shared):
            axis, members = self.bridges.cluster_axis(cid)
            if axis == "bridge":
                # source+target fixed, bridge varies -> predict target
                _, target, _ = members[0]
                overlap = sum(1 for cs in cluster_sets if cid in cs)
                results.append((target, "bridge_axis", {"cluster_id": cid, "overlap": overlap}))
            elif axis == "target":
                # source+bridge fixed, target varies -> predict bridge
                _, _, bridge = members[0]
                overlap = sum(1 for cs in cluster_sets if cid in cs)
                results.append((bridge, "target_axis", {"cluster_id": cid, "overlap": overlap}))
        results.sort(key=lambda r: r[2]["overlap"], reverse=True)
        return results
