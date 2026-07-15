"""
interpret.py — Cluster Interpreter (CI) layer for MSE-GLM.

A read-only analysis pass, entirely separate from inference (same
category as analyse.py's other methods — it never touches generate()
or step()). Given a dual-axis cluster already discovered in the Bridge
Matrix (SDD §4.3.3), this proposes a human-readable interpreter token
for it — e.g. labelling the cluster {cat, dog, pig} as "animal" — by
combining evidence from all three matrix families already trained:
Edge, Bridge, and Relationship. No embeddings, no external dictionary,
no learned weights.

Evidence sources actually used (v2)
------------------------------------
A cluster's members already share one structural slot: the bridge or
target position between some fixed source/other-endpoint pair (the
"bridge axis" / "target axis" — how the cluster was originally formed).
Everything below is *additional* evidence gathered about that already-
formed cluster, not a re-derivation of it.

1. Bridge Matrix, source axis (primary signal).
   Fix (bridge, target), let source vary: if cluster members are each
   the *source* of a triple ending in the same (bridge, target) pair —
   e.g. cat/dog/pig are each the source of an "is animal" triple — that
   target token is a candidate interpreter. `coverage` = fraction of
   cluster members that reach it this way. This is the mechanism that
   actually finds the label; everything else corroborates or filters it.

2. Edge Matrix, direct adjacency (real, if narrow, corroboration).
   A Bridge triple (source, bridge, target) only guarantees the bigrams
   source->bridge and bridge->target exist -- NOT source->target
   directly. So checking whether the corpus *also* contains a direct
   source->target bigram somewhere (skipping the bridge word entirely,
   e.g. a stray "cat animal" alongside "cat is animal") is genuinely
   separate information, not a restatement of #1. It's a strict, rarely
   firing check for that reason -- when it does fire, the corpus states
   the same fact two different ways.

3. Bridge Matrix, shared-role check (via t_index).
   Independent of the specific (bridge, target) pair used above: does
   the candidate interpreter token *itself* turn up as an interchangeable
   member of some other dual-axis cluster alongside one of these members?
   (t_index[token] = the non-zero cluster_ids that token participates in
   as a bridge or target anywhere in the graph.) If "animal" and "cat"
   share a cluster_id somewhere else entirely, that's a structural
   family resemblance found by a completely different route.

4. Relationship Matrix, robustness (not "relatedness between distinct
   contexts" -- the training corpus's rel_id is one per training
   sentence, so this checks something more modest and honest: how many
   distinct training sentences assert the specific fact behind a
   candidate, via RelationshipMatrix.relationships_for_triple). A
   candidate backed by one sentence could be a fluke; backed by several
   is more durable.

Honesty notes (do not remove when editing this file)
-----------------------------------------------------
  - This *surfaces* a label already latent in the training corpus. If
    the corpus has no categorical/hypernym-shaped sentence for a
    cluster's members, no candidate will be found -- that is the
    correct outcome, not a bug.
  - `coverage` is a plain fraction, not a calibrated probability.
  - Signals 1-4 are correlated to different degrees (they are all read
    off the same underlying training sequences) -- `evidence_mask`
    reports which checks passed for transparency, but the count of
    passing checks is not a confidence percentage and must not be
    presented as one.
  - High coverage does not imply semantic relevance: grammatical
    continuations (prepositions, articles) can win on coverage exactly
    as easily as a real category word, because this method doesn't know
    English grammar -- it only knows graph structure. Signal 3
    (shared-role) and signal 4 (relationship robustness) are the main
    tools for filtering those out; `build_interpreter_matrix`'s
    `min_signals` argument uses them for that. Always sanity-check the
    top-N candidates for a cluster, not just its rank-1 pick.
"""

from collections import defaultdict


def find_cluster_members(bridge_matrix, cluster_id):
    """
    Return (axis, sorted distinct member token ids) for a cluster_id.
    axis is "bridge" or "target"; (None, []) if the cluster_id is unknown
    or degenerate (fewer than two members).
    """
    axis, triples = bridge_matrix.cluster_axis(cluster_id)
    if not triples:
        return None, []
    if axis == "bridge":
        members = sorted(set(br for _, _, br in triples))
    elif axis == "target":
        members = sorted(set(t for _, t, _ in triples))
    else:
        return None, []
    return axis, members


def _rows_with_triple_id(bridge_matrix, source):
    """Like BridgeMatrix.triples_from_source, but also yields each row's
    CSR position (its triple_id, as used by RelationshipMatrix)."""
    idx = bridge_matrix.index
    if source < 0 or source + 1 >= len(idx):
        return []
    start, end = idx[source], idx[source + 1]
    return [
        (bridge_matrix.target[i], bridge_matrix.bridge[i], bridge_matrix.cluster_id[i], i)
        for i in range(start, end)
    ]


def _has_direct_edge(model, exp_edges, a, b):
    if b in model.edges.successors(a):
        return True
    if exp_edges and b in exp_edges.successors(a):
        return True
    return False


def _shared_role_overlap(bridges, exp_bridges, this_cluster_id, members, candidate):
    """Non-zero cluster_ids the candidate token shares with any member,
    other than the cluster currently being interpreted."""
    member_clusters = set()
    for m in members:
        member_clusters.update(bridges.t_index.get(m, []))
        if exp_bridges:
            member_clusters.update(exp_bridges.t_index.get(m, []))
    member_clusters.discard(this_cluster_id)

    candidate_clusters = set(bridges.t_index.get(candidate, []))
    if exp_bridges:
        candidate_clusters.update(exp_bridges.t_index.get(candidate, []))
    candidate_clusters.discard(this_cluster_id)

    return sorted(member_clusters & candidate_clusters)


def interpret_cluster(model, cluster_id, top_n=5, mode="strict"):
    """
    Propose interpreter token(s) for one cluster_id, with evidence
    gathered from Edge, Bridge, and Relationship matrices.

    Returns None if the cluster_id is unknown / has fewer than 2 members.
    Otherwise a dict:
        {
          "cluster_id": int, "axis": "bridge"|"target",
          "members": [token_id, ...],
          "candidates": [
             {"interpreter_token": id, "via_bridge_token": id,
              "coverage": float, "members_covered": [id, ...],
              "members_total": int,
              "edge_corroborated": bool,
              "shared_role_overlap": [cluster_id, ...],
              "relationship_ids": [rel_id, ...],
              "evidence_mask": ["bridge_source_axis", ...]},
             ...
          ]   # sorted by (coverage desc, #evidence signals desc)
        }

    Operates on raw token ids -- see model.interpret_cluster() for the
    decoded, human-readable wrapper.
    """
    bridges = model.bridges
    axis, members = find_cluster_members(bridges, cluster_id)
    if not members:
        return None

    exp_bridges = model.exp_bridges if (mode == "open" and model.exp_bridges) else None
    exp_edges   = model.exp_edges   if (mode == "open" and model.exp_edges)   else None
    exp_rels    = model.exp_rels    if (mode == "open" and model.exp_rels)    else None

    # bt_hits[(bridge, target)] -> {member_token: [(triple_id, "train"|"exp"), ...]}
    bt_hits = defaultdict(lambda: defaultdict(list))
    for src in members:
        for target, bridge, _cid, tid in _rows_with_triple_id(bridges, src):
            if target in members or bridge in members:
                continue  # skip self-referential slots -- not an external label
            bt_hits[(bridge, target)][src].append((tid, "train"))
        if exp_bridges:
            for target, bridge, _cid, tid in _rows_with_triple_id(exp_bridges, src):
                if target in members or bridge in members:
                    continue
                bt_hits[(bridge, target)][src].append((tid, "exp"))

    candidates = []
    for (bridge, target), per_member in bt_hits.items():
        covered = sorted(per_member.keys())
        coverage = len(covered) / len(members)

        edge_corroborated = all(_has_direct_edge(model, exp_edges, m, target) for m in covered)

        shared_role_overlap = _shared_role_overlap(
            bridges, exp_bridges, cluster_id, members, target)

        rel_ids = set()
        for m in covered:
            for tid, src_kind in per_member[m]:
                if src_kind == "train":
                    rel_ids.update(model.rels.relationships_for_triple(tid))
                elif exp_rels:
                    rel_ids.update(exp_rels.training_rels_for_exp_triple(tid))

        evidence_mask = ["bridge_source_axis"]
        if edge_corroborated:
            evidence_mask.append("edge_adjacency")
        if shared_role_overlap:
            evidence_mask.append("shared_role")
        if len(rel_ids) > 1:
            evidence_mask.append("relationship_robustness")

        candidates.append({
            "interpreter_token": target,
            "via_bridge_token": bridge,
            "coverage": round(coverage, 3),
            "members_covered": covered,
            "members_total": len(members),
            "edge_corroborated": edge_corroborated,
            "shared_role_overlap": shared_role_overlap,
            "relationship_ids": sorted(rel_ids),
            "evidence_mask": evidence_mask,
        })

    candidates.sort(key=lambda c: (-c["coverage"], -len(c["evidence_mask"])))
    return {
        "cluster_id": cluster_id,
        "axis": axis,
        "members": members,
        "candidates": candidates[:top_n],
    }


def interpret_all_clusters(model, min_coverage=0.5, max_per_cluster=3, mode="strict"):
    """
    Scan every non-zero cluster_id and return every candidate that clears
    `min_coverage` for it (up to `max_per_cluster`, sorted best-first),
    NOT just the single top pick. A cluster is one set of interchangeable
    tokens, but nothing says only one label can be true of that set at
    once -- {cat, dog, pig} can legitimately be "animal" AND {cat, dog}
    within it can separately support "pet" if the corpus says so. Pass
    max_per_cluster=None for no cap.

    Cheap for typical corpora: reuses the same CSR slices every other
    analysis method already relies on; nothing new is persisted here.
    """
    seen = set(model.bridges.cluster_id)
    seen.discard(0)
    if mode == "open" and model.exp_bridges:
        seen.update(c for c in model.exp_bridges.cluster_id if c != 0)

    scan_n = max_per_cluster * 4 if max_per_cluster else 50

    results = []
    for cid in sorted(seen):
        r = interpret_cluster(model, cid, top_n=scan_n, mode=mode)
        if not r:
            continue
        kept = [c for c in r["candidates"] if c["coverage"] >= min_coverage]
        if max_per_cluster:
            kept = kept[:max_per_cluster]
        if kept:
            results.append({**r, "candidates": kept})
    results.sort(key=lambda r: -r["candidates"][0]["coverage"])
    return results


def build_interpreter_matrix(model, min_coverage=0.5, min_signals=2,
                              max_per_cluster=None, mode="strict"):
    """
    The filtered "final" CI Matrix: one row per (cluster_id, interpreter)
    pair that clears BOTH coverage and min_signals -- every qualifying
    interpreter is kept, not just the single best one per cluster. A
    cluster is free to carry several simultaneous labels (different
    semantic granularities, different tag dimensions like part-of-speech
    vs. category) as long as each independently earns its place. Pass
    max_per_cluster to cap how many labels a single cluster can keep
    (best coverage first); default None keeps all that qualify.

    This is the CI Matrix itself, in the original proposal's spirit
    (interpreter_token_id, cluster_id, + a transparent evidence record)
    but with coverage/evidence_mask in place of a fabricated single
    confidence float, and without artificially forcing one label per
    cluster. Rows are plain dicts -- write them out with
    `analyse.py interpreter-matrix --json > interpreter_matrix.json` to
    persist, the same way every other artifact in this project is saved.

    Returns a list of rows sorted by (coverage desc, #signals desc):
        {"cluster_id", "axis", "members", "interpreter_token",
         "via_bridge_token", "coverage", "members_covered",
         "members_total", "evidence_mask", "relationship_ids",
         "shared_role_overlap"}
    """
    seen = set(model.bridges.cluster_id)
    seen.discard(0)
    if mode == "open" and model.exp_bridges:
        seen.update(c for c in model.exp_bridges.cluster_id if c != 0)

    rows = []
    for cid in sorted(seen):
        r = interpret_cluster(model, cid, top_n=50, mode=mode)
        if not r or not r["candidates"]:
            continue
        qualifying = [
            c for c in r["candidates"]
            if c["coverage"] >= min_coverage and len(c["evidence_mask"]) >= min_signals
        ]
        if max_per_cluster:
            qualifying = qualifying[:max_per_cluster]
        for c in qualifying:
            rows.append({
                "cluster_id": r["cluster_id"],
                "axis": r["axis"],
                "members": r["members"],
                "interpreter_token": c["interpreter_token"],
                "via_bridge_token": c["via_bridge_token"],
                "coverage": c["coverage"],
                "members_covered": c["members_covered"],
                "members_total": c["members_total"],
                "evidence_mask": c["evidence_mask"],
                "relationship_ids": c["relationship_ids"],
                "shared_role_overlap": c["shared_role_overlap"],
            })

    rows.sort(key=lambda r: (-r["coverage"], -len(r["evidence_mask"])))
    return rows


def discover_zero_cluster_groups(model, min_group_size=2, mode="strict"):
    """
    Mine cluster_id==0 -- the standard dual-axis rule's "unclustered"
    bucket -- for source-axis groups the current architecture never
    assigns a cluster_id to at all.

    Why this finds something new (not just re-deriving what
    interpret_cluster already sees): BridgeMatrix.build only implements
    two clustering rules, both keyed on a FIXED SOURCE --
      - bridge axis:  fix (source, target), bridge varies
      - target axis:  fix (source, bridge), target varies
    There is no third rule for "fix (bridge, target), source varies".
    So a set of tokens that are each individually the source of one
    "X is animal"-shaped triple, but never co-occur in any OTHER shared
    context (no shared verb-cluster, nothing), gets no cluster_id at
    all -- cluster_id stays 0 for each of them individually -- and is
    completely invisible to cluster_report() / interpret_all_clusters()
    / build_interpreter_matrix(), all of which only ever look at
    cluster_id != 0. This function is exactly that missing third rule,
    applied only to the rows the first two rules left at 0.

    Returns a list of dicts (note: NOT the same schema as
    build_interpreter_matrix rows -- these never had a real cluster_id,
    so one isn't fabricated here):
        {"interpreter_token", "via_bridge_token", "members",
         "member_count", "edge_corroborated", "shared_role_overlap",
         "relationship_ids", "evidence_mask"}
    Sorted by (member_count desc, #evidence signals desc).

    Caution: unlike interpret_cluster/build_interpreter_matrix, this
    doesn't start from an already-known cluster's membership -- it scans
    the WHOLE unclustered bucket, so the candidate space is much larger.
    On a bigger, noisier corpus expect more coincidental groupings (two
    unrelated sources that happen to share one throwaway bigram-adjacent
    pattern). `min_group_size` is a floor, not a guarantee of semantic
    relevance -- eyeball the results, same as everywhere else in this
    module.
    """
    bridges = model.bridges
    exp_bridges = model.exp_bridges if (mode == "open" and model.exp_bridges) else None
    exp_edges   = model.exp_edges   if (mode == "open" and model.exp_edges)   else None
    exp_rels    = model.exp_rels    if (mode == "open" and model.exp_rels)    else None

    # groups[(bridge, target)] -> {source: [(triple_id, "train"|"exp"), ...]}
    groups = defaultdict(lambda: defaultdict(list))
    for i in range(len(bridges.source)):
        if bridges.cluster_id[i] != 0:
            continue
        s, t, br = bridges.source[i], bridges.target[i], bridges.bridge[i]
        groups[(br, t)][s].append((i, "train"))
    if exp_bridges:
        for i in range(len(exp_bridges.source)):
            if exp_bridges.cluster_id[i] != 0:
                continue
            s, t, br = exp_bridges.source[i], exp_bridges.target[i], exp_bridges.bridge[i]
            groups[(br, t)][s].append((i, "exp"))

    results = []
    for (bridge, target), per_source in groups.items():
        if bridge in per_source or target in per_source:
            continue  # skip self-referential -- not an external label
        members = sorted(per_source.keys())
        if len(members) < min_group_size:
            continue

        edge_corroborated = all(_has_direct_edge(model, exp_edges, m, target) for m in members)

        # No pre-existing cluster_id to exclude here -- these members
        # never had one, so nothing to discard from the overlap check.
        member_clusters = set()
        for m in members:
            member_clusters.update(bridges.t_index.get(m, []))
            if exp_bridges:
                member_clusters.update(exp_bridges.t_index.get(m, []))
        candidate_clusters = set(bridges.t_index.get(target, []))
        if exp_bridges:
            candidate_clusters.update(exp_bridges.t_index.get(target, []))
        shared_role_overlap = sorted(member_clusters & candidate_clusters)

        rel_ids = set()
        for m in members:
            for tid, src_kind in per_source[m]:
                if src_kind == "train":
                    rel_ids.update(model.rels.relationships_for_triple(tid))
                elif exp_rels:
                    rel_ids.update(exp_rels.training_rels_for_exp_triple(tid))

        evidence_mask = ["zero_cluster_source_axis"]
        if edge_corroborated:
            evidence_mask.append("edge_adjacency")
        if shared_role_overlap:
            evidence_mask.append("shared_role")
        if len(rel_ids) > 1:
            evidence_mask.append("relationship_robustness")

        results.append({
            "interpreter_token": target,
            "via_bridge_token": bridge,
            "members": members,
            "member_count": len(members),
            "edge_corroborated": edge_corroborated,
            "shared_role_overlap": shared_role_overlap,
            "relationship_ids": sorted(rel_ids),
            "evidence_mask": evidence_mask,
        })

    results.sort(key=lambda r: (-r["member_count"], -len(r["evidence_mask"])))
    return results
