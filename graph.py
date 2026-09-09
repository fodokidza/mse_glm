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
    """
    Deduplicated bigram edge list, CSR-indexed by source token. Also
    tracks `count` -- how many times each (src, dst) pair literally
    occurred consecutively across training sequences ("bigram
    frequency") -- kept parallel to src/dst, one count per unique
    pair (not per occurrence): the src/dst rows themselves stay
    deduplicated for legality lookups (successors()); count is purely
    additional weight for tie-breaking (see frequency()).
    """

    def __init__(self):
        self.src = array("i")
        self.dst = array("i")
        self.count = array("i")
        self.index = array("i")  # size vocab+1
        self._vocab_size = 0

    def build(self, sequences, vocab_size: int):
        self._vocab_size = vocab_size
        counts = {}
        order = []
        for seq in sequences:
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                if pair not in counts:
                    counts[pair] = 0
                    order.append(pair)
                counts[pair] += 1
        order.sort(key=lambda p: p[0])

        self.src = array("i", [p[0] for p in order])
        self.dst = array("i", [p[1] for p in order])
        self.count = array("i", [counts[p] for p in order])
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

    def frequency(self, token: int, candidate: int):
        """
        Bigram frequency: how many times `candidate` literally
        followed `token` consecutively across training sequences.
        0 if the pair never occurred (including illegal pairs).
        """
        if token < 0 or token + 1 >= len(self.index):
            return 0
        start, end = self.index[token], self.index[token + 1]
        for i in range(start, end):
            if self.dst[i] == candidate:
                return self.count[i]
        return 0

    def to_dict(self):
        return {
            "src": list(self.src), "dst": list(self.dst),
            "count": list(self.count),
            "index": list(self.index), "vocab_size": self._vocab_size,
        }

    @classmethod
    def from_dict(cls, d):
        m = cls()
        m.src = array("i", d["src"])
        m.dst = array("i", d["dst"])
        # Older saved models were written before bigram counts existed;
        # default every pair to a count of 1 so frequency() degrades to
        # "did this pair ever occur" rather than crashing on load.
        m.count = array("i", d.get("count", [1] * len(d["src"])))
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
        # cluster_id -> [triple_idx, ...], lazily built on first
        # cluster_axis() call -- see _ensure_cluster_index(). Always
        # None right after __init__/build()/from_dict(), so it can
        # never go stale silently: every path that (re)populates
        # cluster_id constructs a fresh BridgeMatrix (or otherwise
        # leaves this None), and the next cluster_axis() call rebuilds
        # it from whatever cluster_id currently holds.
        self._cluster_members = None
        self._cluster_axis_cache = None  # cluster_id -> (axis, members), same lifetime

    def _ensure_cluster_index(self):
        """
        Build (once) a cluster_id -> [triple_idx, ...] index, so
        cluster_axis() below can look up one cluster's members without
        scanning every triple in the graph. cluster_axis() used to do
        exactly that full scan on EVERY call -- fine for one call, but
        it's invoked once per CLUSTERED TRIPLE from
        importance.py's _trigger_for_triple() (via
        ivm.py's important_member_tokens(), rebuilt automatically by
        every train()/train_incremental() call) -- O(triples) work,
        repeated once per clustered triple, made the whole training
        pipeline effectively O(triples^2). This mirrors
        RelationshipMatrix._ensure_triple_index()'s same fix for the
        same class of bug (see that method's comment).
        """
        if self._cluster_members is not None:
            return
        idx = defaultdict(list)
        for i, c in enumerate(self.cluster_id):
            if c:
                idx[c].append(i)
        self._cluster_members = idx
        self._cluster_axis_cache = {}

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
        self._cluster_members = None    # stale after repopulating cluster_id --
        self._cluster_axis_cache = None  # rebuilt lazily on next cluster_axis() call

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
        self._ensure_cluster_index()
        cached = self._cluster_axis_cache.get(cluster_id)
        if cached is not None:
            return cached
        idxs = self._cluster_members.get(cluster_id, [])
        members = [(self.source[i], self.target[i], self.bridge[i]) for i in idxs]
        if len(members) < 2:
            result = (None, members)
        else:
            s0, t0, b0 = members[0]
            if all(t == t0 for _, t, _ in members):
                result = ("bridge", members)  # source+target fixed, bridge varies
            elif all(b == b0 for _, _, b in members):
                result = ("target", members)  # source+bridge fixed, target varies
            else:
                result = (None, members)
        self._cluster_axis_cache[cluster_id] = result
        return result

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

    Deduplicated by literal sentence content, same principle
    EdgeMatrix already applies to bigrams and BridgeMatrix already
    applies to triples: two training sentences with the IDENTICAL
    token sequence get exactly one relationship_id, not two -- kept
    parallel to that dedup is `rel_count` -- how many times each
    unique sentence literally occurred across training ("sentence
    frequency"), one count per unique sequence (not per occurrence).
    This mirrors EdgeMatrix.count exactly: the dedup keeps every
    relationship_id-based lookup (triples_for_relationship,
    relationships_for_triple, sequence_for_relationship in
    importance.py) counting genuinely DISTINCT sentences, while
    rel_count is purely additional weight for anything that cares how
    often a given sentence was actually seen -- nothing in this file
    reads it; it exists for callers like model.stats() and analyse.py.

    Before this dedup existed, a corpus with the same sentence
    repeated N times produced N distinct relationship_ids for
    identical content -- inflating "how many distinct sentences does
    this evidence span" claims elsewhere (ctm.py's context-trigger
    support counts, importance.py's trigger_matrix distinct_sequences)
    even though Edge/Bridge already correctly deduplicated the
    bigrams/triples those same N repeats produced. That inconsistency
    is exactly what this dedup closes.
    """

    def __init__(self):
        self.r_triple = array("i")
        self.r_rel = array("i")
        self.index = array("i")  # CSR offsets keyed on relationship_id
        self.rel_count = array("i")  # count[rel_id] = how many literal
                                      # training occurrences shared this
                                      # exact sentence content
        self._n_rels = 0
        self._by_triple_rel = None    # lazily-built CSR keyed on triple_id
        self._by_triple_index = None  # (rel array, offset array)

    def _ensure_triple_index(self):
        """
        Build a secondary CSR index keyed on triple_id, on first use.
        relationships_for_triple was previously an O(total R rows) linear
        scan called from every candidate in every inference step; that made
        per-step cost scale with total corpus size rather than local degree.
        This index makes it O(occurrences for that triple), same as every
        other lookup in the system. Built lazily so callers that never
        query by triple_id (e.g. simple training/persistence round-trips)
        pay nothing for it, and it is invalidated automatically whenever
        build()/from_dict() repopulate r_triple/r_rel.
        """
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

    def build(self, sequences, bridge: BridgeMatrix):
        # rebuild triple_id lookup matching BridgeMatrix's dedup + sort order
        triple_to_id = {}
        for idx, (s, t, b) in enumerate(zip(bridge.source, bridge.target, bridge.bridge)):
            triple_to_id[(s, t, b)] = idx

        # Dedup sequences by exact content, first-occurrence order (same
        # "stable, deterministic, order-of-first-appearance" rule
        # EdgeMatrix/BridgeMatrix already use for their own dedup) --
        # a sentence with zero triples (< 3 tokens) still gets exactly
        # one relationship_id, it just never appears in any row below.
        seq_to_rel_id = {}
        unique_seqs = []
        counts = []
        for seq in sequences:
            key = tuple(seq)
            rel_id = seq_to_rel_id.get(key)
            if rel_id is None:
                rel_id = len(unique_seqs)
                seq_to_rel_id[key] = rel_id
                unique_seqs.append(seq)
                counts.append(0)
            counts[rel_id] += 1

        rows = []  # (triple_id, rel_id)
        for rel_id, seq in enumerate(unique_seqs):
            for i in range(len(seq) - 2):
                source, bridge_tok, target = seq[i], seq[i + 1], seq[i + 2]
                entry = (source, target, bridge_tok)
                triple_id = triple_to_id.get(entry)
                if triple_id is not None:
                    rows.append((triple_id, rel_id))

        rows.sort(key=lambda r: r[1])
        self._n_rels = len(unique_seqs)
        self.rel_count = array("i", counts)
        self.r_triple = array("i", [r[0] for r in rows])
        self.r_rel = array("i", [r[1] for r in rows])

        self.index = array("i", [0] * (self._n_rels + 1))
        for r in self.r_rel:
            self.index[r + 1] += 1
        for i in range(1, len(self.index)):
            self.index[i] += self.index[i - 1]
        self._by_triple_rel = None
        self._by_triple_index = None

    def count(self, rel_id: int):
        """
        How many literal training occurrences shared this exact
        sentence content -- 0 for an out-of-range rel_id, same
        "never raise, just report no evidence" convention as
        EdgeMatrix.frequency(). 1 for every relationship_id that
        wasn't a repeat of another sentence, same as any freshly-built
        model before this field existed would imply.
        """
        if rel_id < 0 or rel_id >= len(self.rel_count):
            return 0
        return self.rel_count[rel_id]

    def triples_for_relationship(self, rel_id: int):
        if rel_id < 0 or rel_id + 1 >= len(self.index):
            return []
        start, end = self.index[rel_id], self.index[rel_id + 1]
        return list(self.r_triple[start:end])

    def relationships_for_triple(self, triple_id: int):
        self._ensure_triple_index()
        offsets = self._by_triple_index
        if triple_id < 0 or triple_id + 1 >= len(offsets):
            return []
        start, end = offsets[triple_id], offsets[triple_id + 1]
        return list(self._by_triple_rel[start:end])

    def to_dict(self):
        return {
            "r_triple": list(self.r_triple), "r_rel": list(self.r_rel),
            "index": list(self.index), "n_rels": self._n_rels,
            "rel_count": list(self.rel_count),
        }

    @classmethod
    def from_dict(cls, d):
        m = cls()
        m.r_triple = array("i", d["r_triple"])
        m.r_rel = array("i", d["r_rel"])
        m.index = array("i", d["index"])
        m._n_rels = d["n_rels"]
        # Older saved models were written before rel_count existed;
        # default every relationship_id to a count of 1 so count()
        # degrades to "this sentence was seen (at least once)" rather
        # than crashing on load -- same fallback EdgeMatrix.from_dict
        # already uses for its own .count.
        m.rel_count = array("i", d.get("rel_count", [1] * m._n_rels))
        return m
