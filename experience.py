"""
experience.py — Experience Matrix builder for MSE-GLM Open Mode.

Runs a structural inference pass over the trained Bridge Matrix,
using cluster membership to derive new (source, target, bridge)
triples that were never observed in training but are structurally
justified by substitutability.

Two expansion rules:
  Rule 1 — Bridge axis: if X shares a cluster with bridge member B
            in slot (S, T), and B also bridges another slot (S2, T2),
            then X can bridge (S2, T2) too.

  Rule 2 — Target axis: if X shares a cluster with target member T
            in slot (S, B), and T also targets another slot (S2, B2),
            then X can target (S2, B2) too.

Three output matrices, saved alongside training artefacts:
  experience_edges.json
  experience_bridges.json
  experience_relationships.json
"""

import json
from array import array
from collections import defaultdict


# ─── Experience Edge Matrix ───────────────────────────────────────────────────

class ExperienceEdgeMatrix:
    """
    New bigrams implied by experience triples — same schema as EdgeMatrix.
    source → bridge  and  bridge → target  pairs not already in E.
    """

    def __init__(self):
        self.src   = array("i")
        self.dst   = array("i")
        self.index = array("i")
        self._vocab_size = 0

    def build_from_pairs(self, pairs, vocab_size):
        self._vocab_size = vocab_size
        self.src   = array("i", [p[0] for p in pairs])
        self.dst   = array("i", [p[1] for p in pairs])
        self.index = array("i", [0] * (vocab_size + 1))
        for s in self.src:
            self.index[s + 1] += 1
        for i in range(1, len(self.index)):
            self.index[i] += self.index[i - 1]

    def successors(self, token):
        if token < 0 or token + 1 >= len(self.index):
            return []
        start, end = self.index[token], self.index[token + 1]
        return list(self.dst[start:end])

    def to_dict(self):
        return {"src": list(self.src), "dst": list(self.dst),
                "index": list(self.index), "vocab_size": self._vocab_size}

    @classmethod
    def from_dict(cls, d):
        m = cls()
        m.src   = array("i", d["src"])
        m.dst   = array("i", d["dst"])
        m.index = array("i", d["index"])
        m._vocab_size = d["vocab_size"]
        return m


# ─── Experience Bridge Matrix ─────────────────────────────────────────────────

class ExperienceBridgeMatrix:
    """
    Inferred (source, target, bridge) triples with dual-axis cluster_ids
    starting from max(training_cluster_id) + 1.
    Same CSR schema as BridgeMatrix; kept separate so Strict Mode never
    touches experience data.
    """

    def __init__(self):
        self.source     = array("i")
        self.target     = array("i")
        self.bridge     = array("i")
        self.cluster_id = array("i")
        self.index      = array("i")
        self._vocab_size = 0
        self.t_index    = {}   # token → sorted non-zero cluster_ids

    def build_from_triples(self, triples_with_cluster, vocab_size):
        """
        triples_with_cluster: list of (source, target, bridge, cluster_id)
        Already sorted by source.
        """
        self._vocab_size = vocab_size
        self.source     = array("i", [t[0] for t in triples_with_cluster])
        self.target     = array("i", [t[1] for t in triples_with_cluster])
        self.bridge     = array("i", [t[2] for t in triples_with_cluster])
        self.cluster_id = array("i", [t[3] for t in triples_with_cluster])

        self.index = array("i", [0] * (vocab_size + 1))
        for s in self.source:
            self.index[s + 1] += 1
        for i in range(1, len(self.index)):
            self.index[i] += self.index[i - 1]

        ti = defaultdict(set)
        for s, t, b, c in zip(self.source, self.target, self.bridge, self.cluster_id):
            if c != 0:
                ti[b].add(c)
                ti[t].add(c)
        self.t_index = {k: sorted(v) for k, v in ti.items()}

    def triples_from_source(self, source):
        if source < 0 or source + 1 >= len(self.index):
            return []
        start, end = self.index[source], self.index[source + 1]
        return list(zip(self.target[start:end],
                        self.bridge[start:end],
                        self.cluster_id[start:end]))

    def cluster_axis(self, cluster_id):
        members = [(s, t, b) for s, t, b, c in
                   zip(self.source, self.target, self.bridge, self.cluster_id)
                   if c == cluster_id]
        if len(members) < 2:
            return None, members
        s0, t0, b0 = members[0]
        if all(t == t0 for _, t, _ in members):
            return "bridge", members
        if all(b == b0 for _, _, b in members):
            return "target", members
        return None, members

    def to_dict(self):
        return {
            "source":     list(self.source),
            "target":     list(self.target),
            "bridge":     list(self.bridge),
            "cluster_id": list(self.cluster_id),
            "index":      list(self.index),
            "vocab_size": self._vocab_size,
            "t_index":    {str(k): v for k, v in self.t_index.items()},
        }

    @classmethod
    def from_dict(cls, d):
        m = cls()
        m.source     = array("i", d["source"])
        m.target     = array("i", d["target"])
        m.bridge     = array("i", d["bridge"])
        m.cluster_id = array("i", d["cluster_id"])
        m.index      = array("i", d["index"])
        m._vocab_size = d["vocab_size"]
        m.t_index    = {int(k): v for k, v in d["t_index"].items()}
        return m


# ─── Experience Relationship Matrix ──────────────────────────────────────────

class ExperienceRelationshipMatrix:
    """
    (exp_triple_id, exp_rel_id) — same schema as RelationshipMatrix.
    exp_rel_ids start from n_training_rels.

    parent_map: exp_rel_id → frozenset of training rel_ids that
    justified this experience relationship.  Used by Open Mode Stage 3
    to resolve lineage overlap against active_rels (which holds
    training rel_ids).
    """

    def __init__(self):
        self.r_triple        = array("i")
        self.r_rel           = array("i")
        self.index           = array("i")
        self._n_exp_rels     = 0
        self._n_training_rels = 0
        self.parent_map      = {}   # exp_rel_id → set of training rel_ids
        self._by_triple_rel   = None
        self._by_triple_index = None

    def _ensure_triple_index(self):
        """See RelationshipMatrix._ensure_triple_index — same fix, same reason:
        avoids an O(total experience R rows) scan on every Open-Mode candidate."""
        if self._by_triple_index is not None:
            return
        n_triples = (max(self.r_triple) + 1) if len(self.r_triple) else 0
        pairs = sorted(range(len(self.r_triple)), key=lambda i: self.r_triple[i])
        sorted_rel = array("i", [self.r_rel[i] for i in pairs])
        sorted_tid = array("i", [self.r_triple[i] for i in pairs])
        offsets = array("i", [0] * (n_triples + 1))
        for tid in sorted_tid:
            offsets[tid + 1] += 1
        for i in range(1, len(offsets)):
            offsets[i] += offsets[i - 1]
        self._by_triple_rel = sorted_rel
        self._by_triple_index = offsets

    def build_from_rows(self, rows, n_exp_rels, n_training_rels, parent_map):
        rows_sorted = sorted(rows, key=lambda r: r[1])
        self._n_exp_rels     = n_exp_rels
        self._n_training_rels = n_training_rels
        self.parent_map      = {k: set(v) for k, v in parent_map.items()}

        self.r_triple = array("i", [r[0] for r in rows_sorted])
        self.r_rel    = array("i", [r[1] for r in rows_sorted])

        # CSR index keyed on (exp_rel_id - n_training_rels)
        self.index = array("i", [0] * (n_exp_rels + 1))
        for r in self.r_rel:
            local = r - n_training_rels
            if 0 <= local < n_exp_rels:
                self.index[local + 1] += 1
        for i in range(1, len(self.index)):
            self.index[i] += self.index[i - 1]
        self._by_triple_rel = None
        self._by_triple_index = None

    def relationships_for_exp_triple(self, exp_triple_id):
        self._ensure_triple_index()
        offsets = self._by_triple_index
        if exp_triple_id < 0 or exp_triple_id + 1 >= len(offsets):
            return []
        start, end = offsets[exp_triple_id], offsets[exp_triple_id + 1]
        return list(self._by_triple_rel[start:end])

    def training_rels_for_exp_triple(self, exp_triple_id):
        """All training rel_ids reachable via parent_map for this exp triple."""
        out = set()
        for exp_rel_id in self.relationships_for_exp_triple(exp_triple_id):
            out.update(self.parent_map.get(exp_rel_id, set()))
        return out

    def to_dict(self):
        return {
            "r_triple":          list(self.r_triple),
            "r_rel":             list(self.r_rel),
            "index":             list(self.index),
            "n_exp_rels":        self._n_exp_rels,
            "n_training_rels":   self._n_training_rels,
            "parent_map":        {str(k): list(v) for k, v in self.parent_map.items()},
        }

    @classmethod
    def from_dict(cls, d):
        m = cls()
        m.r_triple         = array("i", d["r_triple"])
        m.r_rel            = array("i", d["r_rel"])
        m.index            = array("i", d["index"])
        m._n_exp_rels      = d["n_exp_rels"]
        m._n_training_rels = d["n_training_rels"]
        m.parent_map       = {int(k): set(v) for k, v in d["parent_map"].items()}
        return m


# ─── Experience Builder ───────────────────────────────────────────────────────

class ExperienceBuilder:
    """
    Derives Experience Matrices from a trained MSEGraphLanguageModel.
    Call build(model) once after training; results are saved to the
    model folder so Open Mode has zero inference-time overhead.
    """

    def build(self, model):
        B = model.bridges
        R = model.rels
        E = model.edges
        vocab_size = model.tokenizer.vocab_size_actual

        # ── Step 1: parse cluster info from B ─────────────────────────────
        groups = defaultdict(list)
        for s, t, b, c in zip(B.source, B.target, B.bridge, B.cluster_id):
            if c != 0:
                groups[c].append((s, t, b))

        cluster_info = {}  # cid → (axis, members_list, fixed_key)
        for cid, members in groups.items():
            axis, _ = B.cluster_axis(cid)
            if axis == "bridge":
                s0, t0, _ = members[0]
                cluster_info[cid] = ("bridge", [b for _, _, b in members], (s0, t0))
            elif axis == "target":
                s0, _, b0 = members[0]
                cluster_info[cid] = ("target", [t for _, t, _ in members], (s0, b0))

        # ── Step 2: reverse maps (all triples, not just clustered) ─────────
        triple_to_id    = {}
        token_as_bridge = defaultdict(list)  # bridge_token → [(s, t, triple_id)]
        token_as_target = defaultdict(list)  # target_token → [(s, b, triple_id)]

        for idx, (s, t, b) in enumerate(zip(B.source, B.target, B.bridge)):
            triple_to_id[(s, t, b)] = idx
            token_as_bridge[b].append((s, t, idx))
            token_as_target[t].append((s, b, idx))

        existing = set(triple_to_id.keys())
        existing_edges = set(zip(E.src, E.dst))

        exp_seen   = set()
        exp_items  = []  # (s, t, b, source_triple_id, justifying_cid, rule)

        # ── Rule 1: bridge axis expansion ─────────────────────────────────
        for cid, (axis, members, (s0, t0)) in cluster_info.items():
            if axis != "bridge":
                continue
            for mi in members:
                for s2, t2, src_tid in token_as_bridge[mi]:
                    if s2 == s0 and t2 == t0:
                        continue
                    for mj in members:
                        if mj == mi:
                            continue
                        entry = (s2, t2, mj)
                        if entry not in existing and entry not in exp_seen:
                            exp_seen.add(entry)
                            exp_items.append((s2, t2, mj, src_tid, cid, "rule1"))

        # ── Rule 2: target axis expansion ─────────────────────────────────
        for cid, (axis, members, (s0, b0)) in cluster_info.items():
            if axis != "target":
                continue
            for ti in members:
                for s2, b2, src_tid in token_as_target[ti]:
                    if s2 == s0 and b2 == b0:
                        continue
                    for tj in members:
                        if tj == ti:
                            continue
                        entry = (s2, tj, b2)
                        if entry not in existing and entry not in exp_seen:
                            exp_seen.add(entry)
                            exp_items.append((s2, tj, b2, src_tid, cid, "rule2"))

        # Sort by source for CSR
        exp_items.sort(key=lambda x: x[0])

        # ── Step 3: assign cluster_ids to experience triples ──────────────
        max_cluster = max(B.cluster_id) if len(B.cluster_id) > 0 else 0
        exp_cluster_id = [0] * len(exp_items)

        exp_gst = defaultdict(list)
        exp_gsb = defaultdict(list)
        for idx, (s, t, b, *_) in enumerate(exp_items):
            exp_gst[(s, t)].append(idx)
            exp_gsb[(s, b)].append(idx)

        nc = max_cluster + 1
        for idxs in exp_gst.values():
            if len(idxs) > 1:
                for i in idxs:
                    exp_cluster_id[i] = nc
                nc += 1
        for idxs in exp_gsb.values():
            if len(idxs) > 1 and all(exp_cluster_id[i] == 0 for i in idxs):
                for i in idxs:
                    exp_cluster_id[i] = nc
                nc += 1

        # ── Step 4: experience edge matrix ────────────────────────────────
        new_edges = set()
        for s, t, b, *_ in exp_items:
            if (s, b) not in existing_edges:
                new_edges.add((s, b))
            if (b, t) not in existing_edges:
                new_edges.add((b, t))

        exp_edge = ExperienceEdgeMatrix()
        exp_edge.build_from_pairs(sorted(new_edges), vocab_size)

        # ── Step 5: experience bridge matrix ──────────────────────────────
        exp_bridge = ExperienceBridgeMatrix()
        exp_bridge.build_from_triples(
            [(s, t, b, c) for (s, t, b, *_), c in zip(exp_items, exp_cluster_id)],
            vocab_size
        )

        # ── Step 6: experience relationship matrix ────────────────────────
        n_training = R._n_rels
        justification_to_exp_rel = {}
        parent_map = {}
        exp_rel_rows = []
        next_exp_rel = n_training

        for exp_tid, (s, t, b, src_tid, just_cid, rule) in enumerate(exp_items):
            key = (just_cid, src_tid)
            if key not in justification_to_exp_rel:
                justification_to_exp_rel[key] = next_exp_rel
                parent_map[next_exp_rel] = set(R.relationships_for_triple(src_tid))
                next_exp_rel += 1
            exp_rel_rows.append((exp_tid, justification_to_exp_rel[key]))

        exp_rel = ExperienceRelationshipMatrix()
        exp_rel.build_from_rows(
            exp_rel_rows,
            n_exp_rels=next_exp_rel - n_training,
            n_training_rels=n_training,
            parent_map=parent_map,
        )

        return exp_edge, exp_bridge, exp_rel

    def summary(self, exp_edge, exp_bridge, exp_rel):
        return {
            "exp_edges":           len(exp_edge.src),
            "exp_bridges":         len(exp_bridge.source),
            "exp_clustered":       sum(1 for c in exp_bridge.cluster_id if c != 0),
            "exp_clusters":        len(set(c for c in exp_bridge.cluster_id if c != 0)),
            "exp_relationships":   exp_rel._n_exp_rels,
            "exp_rel_rows":        len(exp_rel.r_triple),
        }
