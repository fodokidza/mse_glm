"""
importance.py — Token Importance / Trigger analysis for MSE-GLM.

A read-only analysis pass, same category as analyse.py and
interpret.py -- never touches generate()/step(), only reads matrices
that already exist.

What this actually adds, and what it doesn't
----------------------------------------------
Three genuinely new capabilities:

  1. sequence_for_relationship -- reconstruct the literal, ordered
     training sentence for a relationship_id. Nothing in the codebase
     did this before; RelationshipMatrix only stores which triple_ids
     belong to a rel_id, not the sequence itself. This rebuilds it by
     chaining consecutive triples (each overlaps the next by two
     tokens), verifying the overlap holds at every step rather than
     assuming it.

  2. important_tokens_in_sequence -- for one reconstructed sequence,
     tag every position whose triple has a non-zero cluster_id. A
     token is "important" in a sequence exactly when ITS OWN
     occurrence there is one of the pieces of evidence that put its
     triple in a cluster -- not merely "this token happens to be
     clustered somewhere else in the graph" (that's a different,
     weaker claim; t_index answers that one, this answers a stricter,
     per-occurrence one). Each important token is tagged with its
     "trigger" -- the fixed part of the triple that made it important:
     (source, target) for a bridge-axis cluster, (source, bridge) for
     a target-axis cluster.

One capability that mostly CONFIRMS existing structure rather than
adding new inferential power, and is presented that way on purpose:

  3. trigger_matrix -- group every clustered triple in the WHOLE graph
     by its trigger, across every relationship_id, and report which
     triggers activate different tokens in genuinely different
     sequences. Read this as a verification/reporting tool: any
     cluster with 2+ members whose triples came from 2+ different
     training sentences ALREADY satisfies "the same trigger makes
     different tokens important in different sequences" purely by how
     clustering works (a cluster only forms because 2+ triples shared
     a fixed axis; if those triples came from different sentences, the
     generalization already happened at training time). Building this
     table surfaces and quantifies that, it doesn't discover a new
     mechanism.

Also included: an explicit, read-only bridge to what generation
already does with this structure -- expected_importance() tells you
what inference.py's own Stage 2 (source, bridge) -> target lookup
already implies about the next token, without changing what
inference.py actually decides. It's a named window onto existing
behavior, not a new inference mechanism, and is documented as such so
it isn't mistaken for one.

Scope note: sequence reconstruction only makes sense for Strict Mode
(literally-observed sentences). Experience Matrix triples (Open Mode)
were never part of one literal sentence -- they're inferred by
cluster substitution -- so they have no sequence to reconstruct.
Everything in this module operates on the training Relationship
Matrix only.
"""


def sequence_for_relationship(model, rel_id):
    """
    Reconstruct the ordered token sequence for one training sentence
    (relationship_id), by chaining its triples: triple k covers
    positions (k, k+1, k+2), so triple k+1's target is always the one
    new token beyond triple k. Verifies that overlap at every step
    instead of assuming it, since a break would mean the Relationship
    Matrix and Bridge Matrix have drifted out of sync with each other.

    Returns the token-id sequence, or [] if rel_id has no triples
    (e.g. a sentence shorter than 3 tokens has none, by construction).
    Raises ValueError if the chain doesn't hold together (should never
    happen on a model built by this codebase's own train()/
    train_incremental() -- signals a corrupted or hand-edited model).
    """
    b = model.bridges
    triple_ids = model.rels.triples_for_relationship(rel_id)
    if not triple_ids:
        return []

    t0 = triple_ids[0]
    seq = [b.source[t0], b.bridge[t0], b.target[t0]]
    for tid in triple_ids[1:]:
        if b.source[tid] != seq[-2] or b.bridge[tid] != seq[-1]:
            raise ValueError(
                f"relationship {rel_id}: triple {tid} doesn't chain from the "
                f"previous triple -- Relationship/Bridge matrices are out of sync")
        seq.append(b.target[tid])
    return seq


def _trigger_for_triple(bridges, triple_id):
    """(axis, trigger_tuple, important_token) for one clustered triple,
    or None if the triple isn't clustered (cluster_id == 0)."""
    cid = bridges.cluster_id[triple_id]
    if cid == 0:
        return None
    axis, _triples = bridges.cluster_axis(cid)
    s, t, br = bridges.source[triple_id], bridges.target[triple_id], bridges.bridge[triple_id]
    if axis == "bridge":
        return axis, (s, t), br
    else:  # axis == "target"
        return axis, (s, br), t


def important_tokens_in_sequence(model, rel_id):
    """
    For one training sentence, return every position whose triple is
    clustered, tagged with what made it important.

    Returns:
        {"rel_id": int, "sequence": [token_id, ...],
         "important": [
             {"position": int, "token": token_id, "axis": "bridge"|"target",
              "cluster_id": int, "trigger": (a, b)},
             ...
         ]}
    "position" indexes into "sequence" (0-based). trigger is a tuple of
    raw token ids -- decode both with model.tokenizer for display.
    """
    b = model.bridges
    seq = sequence_for_relationship(model, rel_id)
    triple_ids = model.rels.triples_for_relationship(rel_id)

    important = []
    for i, tid in enumerate(triple_ids):
        info = _trigger_for_triple(b, tid)
        if info is None:
            continue
        axis, trigger, token = info
        # the important token sits 1 position ahead of triple i's start
        # (triple i covers seq[i:i+3]; bridge is seq[i+1], target is seq[i+2])
        position = i + 1 if axis == "bridge" else i + 2
        important.append({
            "position": position,
            "token": token,
            "axis": axis,
            "cluster_id": b.cluster_id[tid],
            "trigger": trigger,
        })
    return {"rel_id": rel_id, "sequence": seq, "important": important}


def trigger_matrix(model, min_sequences=2):
    """
    Group every clustered triple in the graph by its trigger (axis +
    fixed token pair), across every relationship_id that instantiates
    it, and report which triggers activate genuinely different tokens
    in genuinely different sequences.

    This is a verification/reporting tool, not a new inference
    mechanism -- see the module docstring. Any row with
    distinct_sequences >= 2 already existed implicitly the moment its
    cluster formed from triples in different training sentences; this
    just makes it explicit and countable.

    Returns a list of rows, sorted by (distinct_sequences desc,
    distinct_tokens desc), each:
        {"axis", "trigger": (a, b), "cluster_id",
         "activations": [(rel_id, token), ...],
         "distinct_sequences": int, "distinct_tokens": int}
    filtered to distinct_sequences >= min_sequences.
    """
    b = model.bridges
    rels = model.rels

    by_trigger = {}  # (axis, trigger) -> {"cluster_id": cid, "activations": [(rel_id, token)]}
    for tid in range(len(b.source)):
        info = _trigger_for_triple(b, tid)
        if info is None:
            continue
        axis, trigger, token = info
        key = (axis, trigger)
        entry = by_trigger.setdefault(key, {"cluster_id": b.cluster_id[tid], "activations": []})
        for rel_id in rels.relationships_for_triple(tid):
            entry["activations"].append((rel_id, token))

    rows = []
    for (axis, trigger), entry in by_trigger.items():
        distinct_sequences = len(set(rid for rid, _tok in entry["activations"]))
        distinct_tokens = len(set(tok for _rid, tok in entry["activations"]))
        if distinct_sequences < min_sequences:
            continue
        rows.append({
            "axis": axis,
            "trigger": trigger,
            "cluster_id": entry["cluster_id"],
            "activations": sorted(entry["activations"]),
            "distinct_sequences": distinct_sequences,
            "distinct_tokens": distinct_tokens,
        })

    rows.sort(key=lambda r: (-r["distinct_sequences"], -r["distinct_tokens"]))
    return rows


def expected_importance(model, prev_token, current_token, mode="strict"):
    """
    Given the last two tokens of a generation-in-progress, report what
    the graph's OWN structure already implies about the next token --
    without running or changing generation. This is exactly the
    (source=prev, bridge=current) -> target lookup inference.py's
    Stage 2 already performs; naming and exposing it here is for
    explainability, not a new prediction mechanism. If inference.py's
    actual behavior ever needs to change, that's a change to
    inference.py, made deliberately and tested there -- not something
    this function should quietly influence.

    Returns None if (prev_token, current_token) isn't a known
    target-axis trigger. Otherwise:
        {"cluster_id": int, "expected_members": [token_id, ...]}
    """
    bridges = model.bridges
    exp_bridges = model.exp_bridges if (mode == "open" and model.exp_bridges) else None

    for target, bridge_tok, cid in bridges.triples_from_source(prev_token):
        if bridge_tok == current_token and cid != 0:
            axis, triples = bridges.cluster_axis(cid)
            if axis == "target":
                members = sorted(set(t for _, t, _ in triples))
                return {"cluster_id": cid, "expected_members": members}
    if exp_bridges:
        for target, bridge_tok, cid in exp_bridges.triples_from_source(prev_token):
            if bridge_tok == current_token and cid != 0:
                axis, triples = exp_bridges.cluster_axis(cid)
                if axis == "target":
                    members = sorted(set(t for _, t, _ in triples))
                    return {"cluster_id": cid, "expected_members": members}
    return None
