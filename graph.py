"""
graph.py — Edge Matrix, Bridge Matrix (with dual-axis cluster_id), and
Relationship Matrix for MSE-GLM.

All structures are array-backed (array('i')) and CSR-indexed for O(1)
successor / membership lookup, per SDD v2.0 section 6 and the v2.1
addendum sections 2-3.
"""

from array import array
from collections import defaultdict


class EdgeMatrix:
    """Deduplicated bigram edge list, CSR-indexed by source token."""

    def __init__(self):
        self.src = array("i")
        self.dst = array("i")
        self.index = array("i")  # size vocab+1
        self._vocab_size = 0

    def build(self, sequences, vocab_size: int):
        self._vocab_size = vocab_size
        seen = set()
        pairs = []
        for seq in sequences:
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                if pair not in seen:
                    seen.add(pair)
                    pairs.append(pair)
        pairs.sort(key=lambda p: p[0])

        self.src = array("i", [p[0] for p in pairs])
        self.dst = array("i", [p[1] for p in pairs])
        self.index = array("i", [0] * (vocab_size + 1))
        for s in self.src:
            self.index[s + 1] += 1
        for i in range(1, len(self.index)):
            self.index[i] += self.index[i - 1]

    def successors(self, token: int):
        if token < 0 or token + 1 >= len(self.index):
            return []
        start, end = self.index[token], self.index[token + 1]
        return list(self.dst[start:end])

    def to_dict(self):
        return {
            "src": list(self.src), "dst": list(self.dst),
            "index": list(self.index), "vocab_size": self._vocab_size,
        }

    @classmethod
    def from_dict(cls, d):
        m = cls()
        m.src = array("i", d["src"])
        m.dst = array("i", d["dst"])
        m.index = array("i", d["index"])
        m._vocab_size = d["vocab_size"]
        return m


class BridgeMatrix:
    """
    Deduplicated (source, target, bridge) triples, CSR-indexed by source.

    cluster_id is assigned via the dual-axis rule (SDD v2.1 section 3):
      - bridge axis: triples sharing (source, target) get a shared cluster_id
        (groups interchangeable bridge tokens)
      - target axis: triples sharing (source, bridge), not already grouped
        on the bridge axis, get a shared cluster_id
        (groups interchangeable target tokens)
      - otherwise cluster_id = 0
    """

    def __init__(self):
        self.source = array("i")
        self.target = array("i")
        self.bridge = array("i")
        self.cluster_id = array("i")
        self.index = array("i")  # CSR offsets keyed on source
        self._vocab_size = 0
        # token -> sorted list of non-zero cluster_ids (built alongside)
        self.t_index = {}

    def build(self, sequences, vocab_size: int):
        self._vocab_size = vocab_size
        seen = set()
        triples = []
        for seq in sequences:
            for i in range(len(seq) - 2):
                source, bridge, target = seq[i], seq[i + 1], seq[i + 2]
                entry = (source, target, bridge)
                if entry not in seen:
                    seen.add(entry)
                    triples.append(entry)
        triples.sort(key=lambda t: t[0])

        n = len(triples)
        cluster_id = [0] * n

        groups_by_st = defaultdict(list)
        groups_by_sb = defaultdict(list)
        for idx, (s, t, b) in enumerate(triples):
            groups_by_st[(s, t)].append(idx)
            groups_by_sb[(s, b)].append(idx)

        next_cluster = 1
        for key, idxs in groups_by_st.items():
            if len(idxs) > 1:
                for i in idxs:
                    cluster_id[i] = next_cluster
                next_cluster += 1
        for key, idxs in groups_by_sb.items():
            if len(idxs) > 1 and all(cluster_id[i] == 0 for i in idxs):
                for i in idxs:
                    cluster_id[i] = next_cluster
                next_cluster += 1

        self.source = array("i", [t[0] for t in triples])
        self.target = array("i", [t[1] for t in triples])
        self.bridge = array("i", [t[2] for t in triples])
        self.cluster_id = array("i", cluster_id)

        self.index = array("i", [0] * (vocab_size + 1))
        for s in self.source:
            self.index[s + 1] += 1
        for i in range(1, len(self.index)):
            self.index[i] += self.index[i - 1]

        t_index = defaultdict(set)
        for s, t, b, c in zip(self.source, self.target, self.bridge, self.cluster_id):
            if c != 0:
                t_index[b].add(c)
                t_index[t].add(c)
        self.t_index = {k: sorted(v) for k, v in t_index.items()}

    def triples_from_source(self, source: int):
        if source < 0 or source + 1 >= len(self.index):
            return []
        start, end = self.index[source], self.index[source + 1]
        return list(zip(self.target[start:end], self.bridge[start:end],
                         self.cluster_id[start:end]))

    def cluster_axis(self, cluster_id: int):
        """Return ('bridge'|'target', list of triple tuples) for a cluster_id."""
        members = [(s, t, b) for s, t, b, c in
                   zip(self.source, self.target, self.bridge, self.cluster_id)
                   if c == cluster_id]
        if len(members) < 2:
            return None, members
        s0, t0, b0 = members[0]
        if all(t == t0 for _, t, _ in members):
            return "bridge", members  # source+target fixed, bridge varies
        if all(b == b0 for _, _, b in members):
            return "target", members  # source+bridge fixed, target varies
        return None, members

    def to_dict(self):
        return {
            "source": list(self.source), "target": list(self.target),
            "bridge": list(self.bridge), "cluster_id": list(self.cluster_id),
            "index": list(self.index), "vocab_size": self._vocab_size,
            "t_index": {str(k): v for k, v in self.t_index.items()},
        }

    @classmethod
    def from_dict(cls, d):
        m = cls()
        m.source = array("i", d["source"])
        m.target = array("i", d["target"])
        m.bridge = array("i", d["bridge"])
        m.cluster_id = array("i", d["cluster_id"])
        m.index = array("i", d["index"])
        m._vocab_size = d["vocab_size"]
        m.t_index = {int(k): v for k, v in d["t_index"].items()}
        return m


class RelationshipMatrix:
    """
    R matrix — stores only (triple_id, relationship_id). triple_id is a
    foreign key into BridgeMatrix's row order; no triple content is
    duplicated here. Many-to-many: a triple_id may appear under several
    relationship_ids if shared across training sequences.
    """

    def __init__(self):
        self.r_triple = array("i")
        self.r_rel = array("i")
        self.index = array("i")  # CSR offsets keyed on relationship_id
        self._n_rels = 0

    def build(self, sequences, bridge: BridgeMatrix):
        # rebuild triple_id lookup matching BridgeMatrix's dedup + sort order
        triple_to_id = {}
        for idx, (s, t, b) in enumerate(zip(bridge.source, bridge.target, bridge.bridge)):
            triple_to_id[(s, t, b)] = idx

        rows = []  # (triple_id, rel_id)
        for rel_id, seq in enumerate(sequences):
            for i in range(len(seq) - 2):
                source, bridge_tok, target = seq[i], seq[i + 1], seq[i + 2]
                entry = (source, target, bridge_tok)
                triple_id = triple_to_id.get(entry)
                if triple_id is not None:
                    rows.append((triple_id, rel_id))

        rows.sort(key=lambda r: r[1])
        self._n_rels = len(sequences)
        self.r_triple = array("i", [r[0] for r in rows])
        self.r_rel = array("i", [r[1] for r in rows])

        self.index = array("i", [0] * (self._n_rels + 1))
        for r in self.r_rel:
            self.index[r + 1] += 1
        for i in range(1, len(self.index)):
            self.index[i] += self.index[i - 1]

    def triples_for_relationship(self, rel_id: int):
        if rel_id < 0 or rel_id + 1 >= len(self.index):
            return []
        start, end = self.index[rel_id], self.index[rel_id + 1]
        return list(self.r_triple[start:end])

    def relationships_for_triple(self, triple_id: int):
        return [rel for tid, rel in zip(self.r_triple, self.r_rel) if tid == triple_id]

    def to_dict(self):
        return {
            "r_triple": list(self.r_triple), "r_rel": list(self.r_rel),
            "index": list(self.index), "n_rels": self._n_rels,
        }

    @classmethod
    def from_dict(cls, d):
        m = cls()
        m.r_triple = array("i", d["r_triple"])
        m.r_rel = array("i", d["r_rel"])
        m.index = array("i", d["index"])
        m._n_rels = d["n_rels"]
        return m
