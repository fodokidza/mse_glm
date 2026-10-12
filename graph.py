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

        # Persistent, content-keyed clustering state -- lets
        # merge_incremental() (below) extend the graph without redoing
        # the full dual-axis clustering pass over every triple on
        # every merge. None until first populated (by build() or, for
        # a matrix that came from from_dict(), lazily on first
        # merge_incremental() call); never touched by anything else,
        # so it can't silently go stale.
        #   _cluster_of : {(source,target,bridge): cluster_id}
        #   _axis_of    : {(source,target,bridge): "bridge"|"target"}
        #                 (only present for entries with cluster_id != 0;
        #                 records WHICH axis a triple currently owes its
        #                 cluster to, so a later merge can tell "won via
        #                 the dominant bridge axis, never revert" apart
        #                 from "via the target axis, revertible if the
        #                 bridge axis later reclaims a groupmate" -- see
        #                 merge_incremental()'s docstring)
        #   _groups_by_st / _groups_by_sb : same grouping build() uses,
        #                 kept keyed by triple CONTENT (not row index,
        #                 which shifts every merge) so it stays valid
        #                 across merges without remapping.
        self._cluster_of = None
        self._axis_of = None
        self._groups_by_st = None
        self._groups_by_sb = None
        self._next_cluster = 1

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

        # A from-scratch build already computed groups_by_st/groups_by_sb
        # and next_cluster above -- reuse them directly (content-keyed,
        # so a later merge_incremental() can extend them) instead of
        # paying _rebuild_persistent_state()'s equivalent extra scan.
        self._cluster_of = dict(zip(triples, cluster_id))
        self._axis_of = {}
        for key, idxs in groups_by_st.items():
            if len(idxs) > 1:
                for i in idxs:
                    self._axis_of[triples[i]] = "bridge"
        # Anything else with a nonzero cluster_id can only have gotten
        # it from the target/SB pass (those are the only two ways a
        # triple's cluster_id becomes nonzero) -- NOT
        # `all(cluster_id[i] == 0 for i in idxs)` re-evaluated here,
        # since cluster_id was already mutated by that same SB pass
        # above and would make this condition vacuously false for
        # every genuinely SB-clustered group.
        for i, c in enumerate(cluster_id):
            if c != 0 and triples[i] not in self._axis_of:
                self._axis_of[triples[i]] = "target"
        self._groups_by_st = defaultdict(set)
        self._groups_by_sb = defaultdict(set)
        for (s, t, b) in triples:
            self._groups_by_st[(s, t)].add((s, t, b))
            self._groups_by_sb[(s, b)].add((s, t, b))
        self._next_cluster = next_cluster

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

    def _rebuild_persistent_state(self):
        """
        (Re)derive _cluster_of/_axis_of/_groups_by_st/_groups_by_sb/
        _next_cluster from the current source/target/bridge/cluster_id
        arrays. Only needed for a matrix that reached this point via
        from_dict() (a fresh load never populates this state) -- a
        matrix that came from build() already has it, and every
        merge_incremental() call keeps it up to date itself. Paid at
        most once per loaded model (the first merge_incremental() call
        after a load), not once per merge.
        """
        self._cluster_of = {}
        self._groups_by_st = defaultdict(set)
        self._groups_by_sb = defaultdict(set)
        max_cluster = 0
        clustered = []
        for s, t, b, c in zip(self.source, self.target, self.bridge, self.cluster_id):
            entry = (s, t, b)
            self._cluster_of[entry] = c
            self._groups_by_st[(s, t)].add(entry)
            self._groups_by_sb[(s, b)].add(entry)
            if c:
                max_cluster = max(max_cluster, c)
                clustered.append((entry, c))
        self._axis_of = {}
        seen_clusters = set()
        for entry, c in clustered:
            if c in seen_clusters:
                continue
            seen_clusters.add(c)
            axis, members = self.cluster_axis(c)
            for m in members:
                self._axis_of[m] = axis
        self._next_cluster = max_cluster + 1

    def merge_incremental(self, new_triples, vocab_size: int):
        """
        Extend the graph with new_triples (an iterable of (source,
        target, bridge) tuples) WITHOUT rerunning the dual-axis
        clustering pass over every triple seen so far -- only the
        (source,target)/(source,bridge) groups actually touched by a
        genuinely new triple are reprocessed. Replaces build()'s
        "recompute everything from the merged set" approach for the
        incremental-training/large-corpus path (see model.py's
        _merge_graphs), which is what made batch N of a batched corpus
        train cost roughly proportional to the TOTAL corpus seen so
        far rather than just batch N's own size.

        Why touching only the affected groups is enough: a group's
        cluster assignment is a pure function of that group's OWN
        current members (see build()'s docstring for the dual-axis
        rule) -- nothing else. A (source,target)/(source,bridge) group
        that gains no new member this call has the exact same members
        it had before, so its assignment is provably unchanged and
        doesn't need reprocessing.

        One real subtlety this handles: the bridge axis (source,target)
        is dominant over the target axis (source,bridge) -- a
        from-scratch build() always runs the bridge-axis pass over the
        WHOLE merged set before the target-axis pass even starts, so a
        triple that only qualifies for a target-axis cluster in an
        EARLIER merge can later be "reclaimed" by its bridge axis once
        a new triple makes that (source,target) group qualify too --
        and reclaiming it can retroactively disqualify the rest of its
        old target-axis group (build()'s target-axis condition is
        `all cluster_id == 0`, which now fails for that whole group).
        This is handled below by tracking, for every entry, WHICH axis
        it currently owes its cluster_id to (_axis_of), and cascading
        any bridge-axis reclaim into a re-check of the reclaimed
        entry's target-axis group -- reverting that group's other
        members back to unclustered if the group no longer qualifies,
        exactly matching what a full recompute would produce. This
        cascade goes at most one hop: bridge-axis qualification depends
        only on group SIZE, never on cluster_id, so reverting a
        target-axis group can never itself trigger a new bridge-axis
        claim elsewhere.
        """
        self._vocab_size = vocab_size
        if self._cluster_of is None:
            self._rebuild_persistent_state()

        new_entries = []
        touched_st = set()
        for s, t, b in new_triples:
            entry = (s, t, b)
            if entry in self._cluster_of:
                continue  # already known (exact dedup, same rule as build())
            self._cluster_of[entry] = 0
            self._groups_by_st[(s, t)].add(entry)
            self._groups_by_sb[(s, b)].add(entry)
            new_entries.append(entry)
            touched_st.add((s, t))

        # ── Bridge axis (dominant): unconditional for any touched group
        #    that now qualifies, exactly like build()'s first pass ────
        reclaimed_sb_keys = set()
        for key in touched_st:
            members = self._groups_by_st[key]
            if len(members) <= 1:
                continue
            existing = {self._cluster_of[m] for m in members
                        if self._axis_of.get(m) == "bridge"}
            cid = min(existing) if existing else self._next_cluster
            if not existing:
                self._next_cluster += 1
            for m in members:
                if self._axis_of.get(m) == "target":
                    # was target-axis clustered before -- bridge axis
                    # now claims it, so its old target-axis group needs
                    # re-checking (it may no longer qualify at all).
                    reclaimed_sb_keys.add((m[0], m[2]))
                self._cluster_of[m] = cid
                self._axis_of[m] = "bridge"

        # ── Target axis: groups literally touched by a new triple,
        #    plus groups invalidated by a bridge-axis reclaim above ───
        touched_sb = {(s, b) for (s, t, b) in new_entries} | reclaimed_sb_keys
        for key in touched_sb:
            members = self._groups_by_sb[key]
            if len(members) <= 1:
                continue
            if any(self._axis_of.get(m) == "bridge" for m in members):
                # disqualified -- a from-scratch rebuild's `all(...==0)`
                # check would fail for this whole group too; revert any
                # member that was previously clustered ONLY via this axis.
                for m in members:
                    if self._axis_of.get(m) == "target":
                        self._cluster_of[m] = 0
                        del self._axis_of[m]
                continue
            existing = {self._cluster_of[m] for m in members
                        if self._axis_of.get(m) == "target"}
            cid = min(existing) if existing else self._next_cluster
            if not existing:
                self._next_cluster += 1
            for m in members:
                self._cluster_of[m] = cid
                self._axis_of[m] = "target"

        self._flatten_arrays(vocab_size)

    def _flatten_arrays(self, vocab_size: int):
        """
        Rebuild the public source/target/bridge/cluster_id/index/
        t_index arrays from the current _cluster_of state. A single
        O(total triples) layout pass (sort + CSR offsets) -- no
        clustering logic runs here, so this is the same order of cost
        EdgeMatrix already pays every merge for its own array layout.
        """
        triples = sorted(self._cluster_of.items(), key=lambda kv: kv[0][0])
        self.source = array("i", [e[0] for e, _ in triples])
        self.target = array("i", [e[1] for e, _ in triples])
        self.bridge = array("i", [e[2] for e, _ in triples])
        self.cluster_id = array("i", [c for _, c in triples])
        self._vocab_size = vocab_size

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

        self._cluster_members = None
        self._cluster_axis_cache = None

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

        # Persisted raw-content cache -- {sequence tuple: rel_id} and
        # rel_id -> sequence (list) -- kept in sync by build() and
        # merge_incremental() so a later merge never has to reconstruct
        # an existing relationship's sequence by walking the Bridge
        # Matrix (see importance.sequence_for_relationship). None until
        # populated; a matrix that came from from_dict() has to pay for
        # one reconstruction pass on its first merge (see
        # _ensure_raw_cache()) since a save/load round trip doesn't
        # persist raw sequences, only the (triple_id, rel_id) rows.
        self._seq_to_rel_id = None
        self._unique_seqs = None

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

        # build() already computed these locally -- keep them so a
        # later merge_incremental() never needs to reconstruct anything.
        self._seq_to_rel_id = seq_to_rel_id
        self._unique_seqs = unique_seqs

    def _reconstruct_sequence(self, rel_id: int, bridge: "BridgeMatrix"):
        """
        Chain rel_id's triples (via `bridge`) back into its ordered
        token sequence. Same logic as importance.sequence_for_relationship
        (kept here too, decoupled from needing a full `model` object,
        so _ensure_raw_cache() below can call it directly) -- see that
        function's docstring for the full explanation. Returns a tuple;
        () if rel_id has no triples (sentence shorter than 3 tokens).
        """
        triple_ids = self.triples_for_relationship(rel_id)
        if not triple_ids:
            return ()
        t0 = triple_ids[0]
        seq = [bridge.source[t0], bridge.bridge[t0], bridge.target[t0]]
        for tid in triple_ids[1:]:
            if bridge.source[tid] != seq[-2] or bridge.bridge[tid] != seq[-1]:
                raise ValueError(
                    f"relationship {rel_id}: triple {tid} doesn't chain from "
                    f"the previous triple -- Relationship/Bridge matrices are "
                    f"out of sync")
            seq.append(bridge.target[tid])
        return tuple(seq)

    def _ensure_raw_cache(self, bridge: "BridgeMatrix"):
        """
        Populate _seq_to_rel_id/_unique_seqs from the CURRENT (pre-merge)
        triples/relationships, if not already populated. Must be called
        with the bridge matrix as it stood when self.r_triple's
        triple_id values were assigned -- i.e. BEFORE
        BridgeMatrix.merge_incremental() rebuilds bridge's row order
        for the merge in progress (see model.py's _merge_graphs).

        A no-op after the first call on a given matrix -- build() and
        merge_incremental() both leave this cache populated -- so the
        reconstruction this replaces (previously paid on EVERY batch in
        train_corpus.py's merge loop) now happens at most ONCE per
        loaded model, only if that model came from from_dict().
        """
        if self._seq_to_rel_id is not None and self._unique_seqs is not None:
            return
        self._seq_to_rel_id = {}
        self._unique_seqs = []
        for rid in range(self._n_rels):
            seq = self._reconstruct_sequence(rid, bridge)
            self._seq_to_rel_id[seq] = rid
            self._unique_seqs.append(list(seq))

    def merge_incremental(self, new_seqs, bridge: "BridgeMatrix"):
        """
        Extend the matrix with new_seqs. Requires _ensure_raw_cache()
        to already have been called (with the PRE-merge bridge matrix,
        before its row order changed) -- this method itself never
        walks the Bridge Matrix to reconstruct anything, only to
        translate triple CONTENT into `bridge`'s (post-merge, already
        rebuilt) triple_id numbering. That translation is the one part
        that must still touch every relationship row, not just the new
        ones: BridgeMatrix's row order can shift on every merge (a
        triple's position depends on the full current triple set, see
        its own merge_incremental()), so a triple_id recorded before
        this merge may no longer point at the same content afterwards.
        It's a flat dict build + one pass over already-known token
        sequences though, not a graph-walk reconstruction -- cheap
        relative to what it replaces.
        """
        for seq in new_seqs:
            key = tuple(seq)
            rid = self._seq_to_rel_id.get(key)
            if rid is None:
                rid = len(self._unique_seqs)
                self._seq_to_rel_id[key] = rid
                self._unique_seqs.append(list(seq))
                self.rel_count.append(1)
            else:
                self.rel_count[rid] += 1

        self._n_rels = len(self._unique_seqs)

        triple_to_id = {}
        for idx, (s, t, b) in enumerate(zip(bridge.source, bridge.target, bridge.bridge)):
            triple_to_id[(s, t, b)] = idx

        rows = []
        for rel_id, seq in enumerate(self._unique_seqs):
            for i in range(len(seq) - 2):
                source, bridge_tok, target = seq[i], seq[i + 1], seq[i + 2]
                triple_id = triple_to_id.get((source, target, bridge_tok))
                if triple_id is not None:
                    rows.append((triple_id, rel_id))
        rows.sort(key=lambda r: r[1])

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
