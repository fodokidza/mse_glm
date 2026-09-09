"""
test.py — Full regression suite for MSE-GLM v2.1 + Open Mode.
56 original checks + experience + open mode checks.
Usage:  python3 test.py
"""

import os, random, shutil, sys, tempfile
from model import MSEGraphLanguageModel
from analyse import CorpusAnalyser, Analyser
from tokenizer import normalize, split_sentences
from train_corpus import discover_txt_files, train_from_folder
from graph import RelationshipMatrix

PASS = FAIL = 0

def check(name, cond, detail=""):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  [PASS] {name}")
    else:    FAIL += 1; print(f"  [FAIL] {name}  {detail}")

def section(t): print(f"\n=== {t} ===")

# ── corpus ────────────────────────────────────────────────────────────────────
CORPUS = """
the cat sat on the mat.
the dog sat on the carpet.
the boy sat on the mat.
the cat ran on the road.
the dog ran on the road.
the girl ran on the road.
a bird flew over the lake.
a bird flew over the hill.
a plane flew over the lake.
the fish swam in the pond.
the fish swam in the river.
the duck swam in the pond.
"""

# ── experience corpus: ONLY boy ran — cat/dog inferred via experience ─────────
# cat and dog must NOT have "ran" in training so experience builder creates them
CORPUS_EXP = """
the cat sat on the mat.
the dog sat on the carpet.
the boy sat on the chair.
the boy ran on the road.
"""

def main():
    # ── Original 56 checks (v2.1) ────────────────────────────────────────────
    model = MSEGraphLanguageModel(vocab_size=300)
    model.train(CORPUS)

    section("Tokenizer round-trip")
    for phrase in ["the cat sat on the mat", "a bird flew over the lake", "gibberish zzz"]:
        ids = model.tokenizer.encode(phrase)
        check(f"encode '{phrase}'", isinstance(ids, list) and len(ids) > 0)
        check(f"BOS prepended '{phrase}'", ids[0] == 2)
        check(f"no EOS in prompt '{phrase}'", 3 not in ids)
    train_ids = model.tokenizer.encode_for_training("the cat sat on the mat")
    check("encode_for_training appends EOS", train_ids[-1] == 3)
    check("normalize", normalize("The Cat... SAT!!") == "the cat . . . sat ! !", normalize("The Cat... SAT!!"))
    check("sentence split", split_sentences("One. Two! Three") == ["One .", "Two !", "Three"], split_sentences("One. Two! Three"))

    section("Graph construction")
    s = model.stats()
    check("vocab built",         s["vocab_size"] > 10)
    check("edges built",         s["edges"] > 0)
    check("bridges built",       s["bridges"] > 0)
    check("relationships match", s["relationships"] == len(split_sentences(CORPUS)), s)
    check("some clustered",      s["clustered_bridges"] > 0)
    check("some unclustered",    s["clustered_bridges"] < s["bridges"])

    section("Relationship Matrix schema")
    r = model.rels
    check("R two-column schema",        len(r.r_triple) == len(r.r_rel))
    check("R has shared triple",        any(len(r.relationships_for_triple(t)) > 1 for t in set(r.r_triple)))

    section("Lineage tie-breaking (regression)")
    lineage_checks = [
        ("the cat",    ["mat","road"]),   # sat or ran — both valid
        ("the dog",    ["carpet","road"]),
        ("the boy",    ["mat"]),
        ("the cat ran",["road"]),
        ("the dog ran",["road"]),
        ("the fish",   ["pond","river"]),
        ("the duck",   ["pond"]),
    ]
    for prompt, valid in lineage_checks:
        text, _, _ = model.generate(prompt, max_tokens=12)
        check(f"'{prompt}' → one of {valid}", any(v in text for v in valid), f"got '{text}'")
    text, _, _ = model.generate("a bird flew over", max_tokens=12)
    check("bird lands on lake or hill", ("lake" in text) or ("hill" in text), text)

    section("Determinism")
    # ALL prompts are now deterministic in Strict Mode -- including ones
    # that are genuinely structurally ambiguous (multiple candidates tie
    # with no lineage signal to prefer one). Before this change, such
    # ties fell to random.choice and varied run to run; now they fall to
    # _bigram_tie_break (inference.py) -- whichever candidate was
    # literally seen most often as the next token, or lowest token id if
    # that ties too. Still a real behavior change worth naming: an
    # "ambiguous" prompt no longer means "the output will vary," it means
    # "lineage alone didn't decide it -- bigram frequency did."
    for det_prompt in ["the boy", "the girl", "the duck", "the cat ran", "the dog sat",
                        "the cat", "the dog", "the fish", "a bird"]:
        runs = {model.generate(det_prompt, max_tokens=12)[0] for _ in range(5)}
        check(f"'{det_prompt}' deterministic (bigram tie-break, not random)", len(runs) == 1, runs)

    section("explain_step()")
    next_tok, tr = model.explain_step("the", "dog")
    check("explain returns stage", "stage" in tr)
    check("explain (the,dog) returns a token", next_tok is not None)

    section("infer_shared_role()")
    r1 = model.infer_shared_role(["cat","dog"])
    check("cat+dog share cluster",      len(r1) > 0)
    check("cat+dog → sat",              any(t=="sat" for t,_,_ in r1))
    r2 = model.infer_shared_role(["bird","plane"])
    check("bird+plane share cluster",   len(r2) > 0)
    r3 = model.infer_shared_role(["fish","duck"])
    check("fish+duck share cluster",    len(r3) > 0)
    r4 = model.infer_shared_role(["lake","hill"])
    check("lake+hill target-axis",      any(ax=="target_axis" for _,ax,_ in r4))
    r5 = model.infer_shared_role(["sat","flew"])
    check("sat+flew no shared cluster", r5 == [], r5)
    r5b = model.infer_shared_role(["cat","fish"])
    check("cat+fish share BOS subject cluster", len(r5b) > 0)

    section("Analyser")
    ca = CorpusAnalyser(CORPUS)
    cs = ca.stats()
    check("corpus sentences", cs["sentences"] == len(split_sentences(CORPUS)))
    check("corpus words",     cs["words"] > 20)
    a = Analyser(model)
    check("topology hubs",     len(a.topology()["hub_tokens"]) > 0)
    cl = a.cluster_report()
    check("cluster report",    len(cl) > 0)
    check("cluster axis label", all(c["axis"] in ("bridge","target") for c in cl))
    rr = a.relationship_report()
    check("shared triple count", rr["shared_triple_count"] >= 1)
    pt = a.per_token_report("cat")
    check("per-token cat", pt is not None and len(pt["edge_successors"]) > 0)
    _, gt = a.generation_trace("the dog", max_tokens=10)
    check("generation trace", len(gt) > 0 and all("stage" in s for s in gt))

    section("Cluster Interpreter (CI)")
    # Separate corpus: cat/dog/pig share a bridge-axis cluster from the
    # "sat" sentences AND are each the source of a 3-token "X is animal"
    # triple. Interpreter discovery requires the label to fit inside a
    # single (source, bridge, target) window, so the hypernym sentence is
    # deliberately phrased "X is animal" (3 tokens), not "X is an animal"
    # (4 tokens) which falls outside any single triple's reach.
    CORPUS_CI = """
the cat sat on the mat.
the dog sat on the carpet.
the boy sat on the mat.
cat is animal.
dog is animal.
pig is animal.
the boy ran on the road.
the pig ran on the road.
"""
    m_ci = MSEGraphLanguageModel(vocab_size=200)
    m_ci.train(CORPUS_CI)
    ci_report = m_ci.interpret_all_clusters(min_coverage=0.3)
    check("interpret_all_clusters returns results", len(ci_report) > 0, ci_report)
    animal_hit = [r for r in ci_report
                  if set(r["members"]) == {"cat", "dog", "pig"}]
    check("cat/dog/pig cluster found", len(animal_hit) == 1, ci_report)
    if animal_hit:
        top = animal_hit[0]["candidates"][0]
        check("cat/dog/pig interpreted as 'animal'",
              top["interpreter_token"] == "animal", top)
        check("cat/dog/pig coverage is full", top["coverage"] == 1.0, top)
        check("bridge_source_axis always present in evidence_mask",
              "bridge_source_axis" in top["evidence_mask"], top)
        check("relationship_ids reflects >=1 supporting training sentence",
              len(top["relationship_ids"]) >= 1, top)
        check("relationship_robustness flagged since >1 sentence assert it",
              "relationship_robustness" in top["evidence_mask"], top)
    # Unknown cluster_id must return None, not raise
    check("unknown cluster_id returns None", m_ci.interpret_cluster(99999) is None)
    # A cluster with no matching categorical statement should surface
    # nothing (or a low/partial-coverage candidate), not a fabricated label
    sat_only = MSEGraphLanguageModel(vocab_size=100)
    sat_only.train("the cat sat on the mat.\nthe dog sat on the carpet.\n")
    empty_report = sat_only.interpret_all_clusters(min_coverage=0.99)
    check("no spurious full-coverage interpreter without categorical data",
          all(r["candidates"][0]["coverage"] < 1.0
              or r["candidates"][0]["interpreter_token"] not in ("animal",)
              for r in empty_report), empty_report)

    # Known, documented limitation: on a corpus this small, relationship_ids
    # counts a distinct rel_id per training SENTENCE, so grammatical
    # continuations shared by >=2 sentences ("on" via "sat"/"ran") clear
    # relationship_robustness exactly as easily as a real categorical fact
    # ("animal" via "is") does, and shared_role_overlap stays empty
    # everywhere because no token happens to double up across clusters in
    # such a tiny vocabulary. So min_signals=2 does NOT yet separate the
    # semantic hit from the syntactic ones at this scale -- this test
    # documents that honestly rather than asserting a discrimination that
    # doesn't actually hold yet, so a future change to the scoring is
    # forced to update this test consciously instead of silently.
    ci_matrix = m_ci.build_interpreter_matrix(min_coverage=0.3, min_signals=2)
    interpreters_found = {row["interpreter_token"] for row in ci_matrix}
    check("interpreter_matrix still includes the real 'animal' hit at min_signals=2",
          "animal" in interpreters_found, ci_matrix)
    check("interpreter_matrix (documented limitation) does not yet exclude "
          "grammatical 'on' hit at min_signals=2",
          "on" in interpreters_found, ci_matrix)
    check("every row's evidence_mask meets the requested min_signals floor",
          all(len(row["evidence_mask"]) >= 2 for row in ci_matrix), ci_matrix)

    # A cluster must be able to carry more than one qualifying label at
    # once -- e.g. {cat, dog, pig} = "animal" (full coverage) AND the
    # {cat, dog} subset within it also supports "pet" (partial coverage,
    # since pig is never called a pet). Both must survive filtering
    # independently rather than the second being discarded just because
    # the first has higher coverage.
    m_multi = MSEGraphLanguageModel(vocab_size=200)
    m_multi.train("""
the cat sat on the mat.
the dog sat on the carpet.
the boy sat on the mat.
cat is animal.
dog is animal.
pig is animal.
cat is pet.
dog is pet.
the boy ran on the road.
the pig ran on the road.
""")
    multi_matrix = m_multi.build_interpreter_matrix(min_coverage=0.5, min_signals=2)
    cdp_rows = [row for row in multi_matrix if set(row["members"]) == {"cat", "dog", "pig"}]
    cdp_labels = {row["interpreter_token"] for row in cdp_rows}
    check("cat/dog/pig cluster keeps 'animal' label", "animal" in cdp_labels, cdp_rows)
    check("cat/dog/pig cluster ALSO keeps 'pet' label (not discarded for lower coverage)",
          "pet" in cdp_labels, cdp_rows)
    pet_row = next(row for row in cdp_rows if row["interpreter_token"] == "pet")
    check("'pet' label correctly covers only cat+dog, not pig",
          set(pet_row["members_covered"]) == {"cat", "dog"}, pet_row)

    section("Zero-cluster mining (3rd axis)")
    # cat/dog/pig deliberately never share a verb OR a lead-in word, so
    # they get NO cluster_id at all under the standard dual-axis rule
    # (which only clusters by fixed source) -- they must be invisible
    # to cluster_report() and interpret_all_clusters(), and ONLY
    # recoverable by mining cluster_id==0 directly.
    CORPUS_ZERO = """
the cat slept on a mat.
the dog barked at a car.
the pig rolled in a puddle.
well cat is animal.
sure dog is animal.
hey pig is animal.
"""
    m_zero = MSEGraphLanguageModel(vocab_size=200)
    m_zero.train(CORPUS_ZERO)
    a_zero = Analyser(m_zero)

    standard_clusters = a_zero.cluster_report(top_n=50)
    check("cat/dog/pig never share a standard dual-axis cluster",
          not any(set(c["members"]) >= {"cat", "dog", "pig"} for c in standard_clusters),
          standard_clusters)
    check("regular interpret_all_clusters never finds the animal grouping either",
          not any("animal" in [cand["interpreter_token"] for cand in r["candidates"]]
                  for r in m_zero.interpret_all_clusters(min_coverage=0.1)))

    zero_groups = m_zero.discover_zero_cluster_groups(min_group_size=2)
    check("zero-cluster mining returns results", len(zero_groups) > 0, zero_groups)
    animal_group = [g for g in zero_groups if g["interpreter_token"] == "animal"]
    check("zero-cluster mining recovers cat/dog/pig -> animal",
          len(animal_group) == 1 and set(animal_group[0]["members"]) == {"cat", "dog", "pig"},
          zero_groups)
    if animal_group:
        check("recovered group's evidence_mask starts from zero_cluster_source_axis",
              animal_group[0]["evidence_mask"][0] == "zero_cluster_source_axis",
              animal_group[0])
    # Self-referential groups (bridge or target equal to a member) must
    # never be proposed as their own label.
    check("no self-referential interpreter proposed",
          all(g["interpreter_token"] not in g["members"] for g in zero_groups), zero_groups)

    section("Token Importance / Trigger analysis (importance.py)")

    from importance import (sequence_for_relationship, important_tokens_in_sequence,
                             trigger_matrix, expected_importance)

    CORPUS_IMP = """
the cat sat on the mat.
the dog sat on the carpet.
the pig sat on the rug.
"""
    m_imp = MSEGraphLanguageModel(vocab_size=200)
    m_imp.train(CORPUS_IMP)
    tok_imp = m_imp.tokenizer
    def dec_imp(t): return tok_imp.id_to_token.get(t, t) if t in (0, 1, 2, 3) else tok_imp.decode([t])

    # Sequence reconstruction must exactly match the original sentence.
    seq0 = sequence_for_relationship(m_imp, 0)
    check("sequence_for_relationship reconstructs rel_id=0 exactly",
          [dec_imp(t) for t in seq0] ==
          ["<BOS>", "the", "cat", "sat", "on", "the", "mat", ".", "<EOS>"], seq0)

    # A rel_id beyond range must return [] rather than raise.
    check("sequence_for_relationship on out-of-range rel_id returns []",
          sequence_for_relationship(m_imp, 9999) == [])

    # cat/dog/pig must each be tagged important in their own sentence,
    # via BOTH the axes they actually participate in (bridge-axis "the
    # ___ sat" and target-axis "<BOS> the ___").
    imp0 = important_tokens_in_sequence(m_imp, 0)
    cat_tags = [it for it in imp0["important"] if dec_imp(it["token"]) == "cat"]
    check("'cat' is tagged important in its own sentence",
          len(cat_tags) > 0, imp0)
    check("'cat' is tagged important via both bridge and target axis",
          {t["axis"] for t in cat_tags} == {"bridge", "target"}, cat_tags)
    check("'cat's bridge-axis trigger is ('the', 'sat')",
          any(tuple(dec_imp(x) for x in t["trigger"]) == ("the", "sat")
              for t in cat_tags if t["axis"] == "bridge"), cat_tags)

    # The trigger matrix must recover that ('the','sat') and
    # ('<BOS>','the') each generalize across all 3 sentences, activating
    # 3 distinct tokens (cat/dog/pig), confirming what cluster formation
    # already guarantees for a multi-sentence cluster.
    triggers = trigger_matrix(m_imp, min_sequences=2)
    decoded_triggers = {tuple(dec_imp(x) for x in row["trigger"]): row for row in triggers}
    check("trigger_matrix finds ('the','sat') generalizing across sentences",
          ("the", "sat") in decoded_triggers, decoded_triggers)
    if ("the", "sat") in decoded_triggers:
        row = decoded_triggers[("the", "sat")]
        check("('the','sat') spans all 3 sentences with 3 distinct tokens",
              row["distinct_sequences"] == 3 and row["distinct_tokens"] == 3, row)
        activated = {dec_imp(tok) for _rid, tok in row["activations"]}
        check("('the','sat') activates exactly {cat, dog, pig}",
              activated == {"cat", "dog", "pig"}, row)

    # min_sequences filter must actually filter.
    check("trigger_matrix respects min_sequences (none span 4 sentences)",
          trigger_matrix(m_imp, min_sequences=4) == [])

    # expected_importance must match what Stage 2 generation would
    # itself use: (<BOS>, the) should expect {cat, dog, pig} next.
    bos_id = 2
    the_id = tok_imp.token_to_id["the"]
    sat_id = tok_imp.token_to_id["sat"]
    result = expected_importance(m_imp, bos_id, the_id)
    check("expected_importance(<BOS>, the) predicts cat/dog/pig",
          result is not None and
          {dec_imp(t) for t in result["expected_members"]} == {"cat", "dog", "pig"}, result)
    check("expected_importance returns None for a non-trigger pair",
          expected_importance(m_imp, sat_id, the_id) is None)

    section("Context Trigger Matrix (ctm.py)")

    from ctm import ContextTriggerMatrix, build_context_trigger_matrix, token_to_relationships

    # Each animal gets its own distinct surrounding vocabulary so the
    # signatures should cleanly discriminate them, matching the
    # proposal's own worked example (farm -> pig, park -> dog, mice -> cat).
    CORPUS_CTM = """
the cat sat on the mat.
the cat likes mice.
a kitten is like a cat.
the dog sat on the carpet.
the dog likes the park.
a puppy is like a dog.
the pig sat on the rug.
the pig likes the farm.
a piglet is like a pig.
"""
    m_ctm = MSEGraphLanguageModel(vocab_size=400)
    m_ctm.train(CORPUS_CTM)
    tok_ctm = m_ctm.tokenizer
    def enc_ctm(w): return tok_ctm.token_to_id[w]
    def dec_ctm(t): return tok_ctm.id_to_token.get(t, t) if t in (0, 1, 2, 3) else tok_ctm.decode([t])

    a_ctm = Analyser(m_ctm)
    cluster_sat = next(c["cluster_id"] for c in a_ctm.cluster_report(top_n=50)
                        if set(c["members"]) == {"cat", "dog", "pig"} and c["axis"] == "bridge")

    ctm = m_ctm.build_context_triggers()
    check("build_context_triggers caches onto model.ctm", m_ctm.ctm is ctm)
    check("has_context_triggers true after building", m_ctm.has_context_triggers())

    # Reserved tokens must never appear as a trigger anywhere.
    all_triggers = set()
    for sig in ctm._sigs.values():
        for triggers in sig.values():
            all_triggers.update(triggers.keys())
    check("no reserved tokens (<PAD>/<UNK>/<BOS>/<EOS>) appear as triggers",
          all_triggers.isdisjoint({0, 1, 2, 3}), all_triggers)

    # Signatures must be built from EVERY sentence mentioning a member --
    # not just the sentences that happen to instantiate this specific
    # cluster's own triples. "mice"/"kitten" only appear in sentences that
    # have nothing to do with the "the ___ sat" cluster's own triples, so
    # if they show up in cat's signature for THIS cluster, the broader
    # (correct) scoping is confirmed.
    cat_sig = ctm._sigs[cluster_sat][enc_ctm("cat")]
    check("cat's trigger signature includes tokens from OTHER sentences (mice, kitten)",
          enc_ctm("mice") in cat_sig and enc_ctm("kitten") in cat_sig, cat_sig)

    farm_ctx  = {enc_ctm(w) for w in ("farm", "barn") if w in tok_ctm.token_to_id}
    park_ctx  = {enc_ctm("park")}
    mice_ctx  = {enc_ctm("mice"), enc_ctm("kitten")}

    r_farm = ctm.select(cluster_sat, {enc_ctm("farm")})
    r_park = ctm.select(cluster_sat, park_ctx)
    r_mice = ctm.select(cluster_sat, mice_ctx)
    check("'farm' context selects pig", r_farm is not None and r_farm["top_members"] == [enc_ctm("pig")], r_farm)
    check("'park' context selects dog", r_park is not None and r_park["top_members"] == [enc_ctm("dog")], r_park)
    check("'mice'/'kitten' context selects cat", r_mice is not None and r_mice["top_members"] == [enc_ctm("cat")], r_mice)

    check("select() with empty context returns None (no signal)",
          ctm.select(cluster_sat, set()) is None)
    check("select() with unknown cluster_id returns {} scores / None",
          ctm.select(999999, {enc_ctm("farm")}) is None)

    # JSON round-trip.
    ctm_dict = ctm.to_dict()
    import json as _json
    _json.dumps(ctm_dict)  # must not raise -- proves it's actually JSON-safe
    restored_ctm = ContextTriggerMatrix.from_dict(ctm_dict)
    check("ContextTriggerMatrix round-trips through to_dict/from_dict",
          restored_ctm.score_members(cluster_sat, {enc_ctm("farm")}) ==
          ctm.score_members(cluster_sat, {enc_ctm("farm")}))

    # ── Inference-level integration ──────────────────────────────────
    # Default behavior (no context_triggers passed) must be BIT-FOR-BIT
    # unchanged -- this is checked throughout the rest of this suite
    # implicitly (all existing tests never pass context_triggers), but
    # asserted explicitly here too as a direct regression guard.
    engine_ctm = m_ctm._engine("strict")
    bos_id = 2
    the_id = enc_ctm("the")

    without_ctm = {engine_ctm.step(bos_id, the_id, active_rels=set())[0] for _ in range(30)}
    check("without context_triggers, tie-break is now fully deterministic "
          "(bigram frequency, not random -- see inference.py)",
          len(without_ctm) == 1, without_ctm)

    with_farm = [engine_ctm.step(bos_id, the_id, active_rels=set(),
                                  context_tokens={enc_ctm("farm")},
                                  context_triggers=ctm)[0] for _ in range(15)]
    check("with CTM + farm context, step() deterministically picks pig every time",
          set(with_farm) == {enc_ctm("pig")}, with_farm)

    _, trace_ctm = engine_ctm.step(bos_id, the_id, active_rels=set(),
                                    context_tokens={enc_ctm("farm")}, context_triggers=ctm)
    check("trace labels CTM-resolved steps distinctly ('context_trigger_resolved')",
          trace_ctm["rule"] == "context_trigger_resolved", trace_ctm)

    # generate()-level integration via use_context_triggers flag.
    _, ids_off, trace_off = m_ctm.generate("the", max_tokens=1, use_context_triggers=False)
    check("generate(use_context_triggers=False) never emits the CTM rule tag",
          all(t.get("rule") != "context_trigger_resolved" for t in trace_off), trace_off)

    # ── train_incremental must invalidate ctm, same as Experience Matrices ──
    m_ctm2 = MSEGraphLanguageModel(vocab_size=200)
    m_ctm2.train("the cat sat on the mat. the dog sat on the carpet.")
    m_ctm2.build_context_triggers()
    check("has_context_triggers true before incremental training", m_ctm2.has_context_triggers())
    incr_summary = m_ctm2.train_incremental("the pig sat on the rug.",
                                             extend_vocab=True, target_vocab_size=250)
    check("has_context_triggers false after incremental training (invalidated)",
          not m_ctm2.has_context_triggers())
    check("train_incremental summary reports ctm_invalidated=True",
          incr_summary["ctm_invalidated"])

    # token_to_relationships sanity: every token in a >=3-token sentence
    # must be covered by at least one relationship_id.
    trmap = token_to_relationships(m_ctm)
    seq0 = sequence_for_relationship(m_ctm, 0)
    check("token_to_relationships covers every token of a real sentence",
          all(t in trmap for t in seq0 if t not in (0, 1, 2, 3)), seq0)

    section("Save/load round-trip")

    tmp = tempfile.mkdtemp(prefix="mse_test_")
    try:
        model.save(tmp)
        m2 = MSEGraphLanguageModel.load(tmp)
        check("reloaded stats match", m2.stats() == model.stats())
        # Reload test: same graph → same candidate sets (even if random picks differ)
        t1_stats = m2.stats()
        check("reloaded model has same stats", t1_stats == model.stats())
        # Unambiguous prompts must be identical across reload
        for p in ["the boy", "the girl", "the cat ran", "the dog sat"]:
            t1 = model.generate(p, max_tokens=12)[0]
            t2 = m2.generate(p, max_tokens=12)[0]
            check(f"reload '{p}' deterministic", t1 == t2, f"'{t1}' vs '{t2}'")
        check("reload shared-role", model.infer_shared_role(["cat","dog"]) ==
              m2.infer_shared_role(["cat","dog"]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    section("Incremental training (train_incremental)")

    # Re-feeding the identical sentence must not duplicate Edge/Bridge
    # structure -- both are always deduplicated -- and now Relationship
    # structure is deduplicated the same way: relationship_ids stay
    # unique per DISTINCT sentence content, with rel_count tracking how
    # many literal occurrences share that content (see graph.py's
    # RelationshipMatrix docstring).
    m_incr = MSEGraphLanguageModel(vocab_size=200)
    m_incr.train("the cat sat on the mat.")
    before_dup = m_incr.stats()
    m_incr.train_incremental("the cat sat on the mat.")
    after_dup = m_incr.stats()
    check("re-feeding identical sentence: no new edges",
          after_dup["edges"] == before_dup["edges"], (before_dup, after_dup))
    check("re-feeding identical sentence: no new bridges",
          after_dup["bridges"] == before_dup["bridges"], (before_dup, after_dup))
    check("re-feeding identical sentence: relationships do NOT grow -- "
          "same content collapses onto the same relationship_id",
          after_dup["relationships"] == before_dup["relationships"], (before_dup, after_dup))
    check("re-feeding identical sentence: relationship_occurrences DOES "
          "grow by 1 -- the repeat is still counted, just not as a new "
          "relationship_id",
          after_dup["relationship_occurrences"] == before_dup["relationship_occurrences"] + 1,
          (before_dup, after_dup))
    check("re-feeding identical sentence: the repeat's rel_count is "
          "actually 2 on the one relationship_id it shares",
          m_incr.rels.count(0) == 2, list(m_incr.rels.rel_count))

    # The core case this feature exists for: a cluster that can only form
    # once BOTH increments are present, because it depends on tokens from
    # two separate training calls sharing a structural slot.
    m_incr2 = MSEGraphLanguageModel(vocab_size=200)
    m_incr2.train("the cat sat on the mat.")
    a_incr2 = Analyser(m_incr2)
    check("no cat/dog cluster before second increment exists",
          not any(set(c["members"]) >= {"cat", "dog"} for c in a_incr2.cluster_report(top_n=50)))
    m_incr2.train_incremental("the dog sat on the carpet.",
                               extend_vocab=True, target_vocab_size=200)
    clusters_after = a_incr2.cluster_report(top_n=50)
    check("cat/dog cluster forms once both increments are present",
          any(set(c["members"]) >= {"cat", "dog"} for c in clusters_after), clusters_after)

    # extend_vocab must never change what an already-known word encodes
    # to -- old triples depend on that id staying stable.
    old_cat_ids = m_incr2.tokenizer.encode("cat")
    check("extend_vocab preserves existing token ids for known words",
          old_cat_ids == [2] + [t for t in old_cat_ids if t != 2])  # sanity: still valid ids
    m_before_ids = list(m_incr2.tokenizer.encode("cat sat"))
    m_incr2.tokenizer.extend_vocab("brand new unseen vocabulary words here", 250)
    check("extend_vocab doesn't change encoding of words it already knew",
          m_incr2.tokenizer.encode("cat sat") == m_before_ids, m_before_ids)

    # Old facts must still generate correctly after a merge, and the
    # merged model must still be able to generate the NEW fact too.
    m_incr3 = MSEGraphLanguageModel(vocab_size=200)
    m_incr3.train("the cat sat on the mat.")
    m_incr3.train_incremental("the dog sat on the carpet.",
                               extend_vocab=True, target_vocab_size=200)
    check("old fact still generates correctly after merge",
          m_incr3.generate("the cat", max_tokens=6)[0] == "the cat sat on the mat.")
    check("new fact generates correctly after merge",
          m_incr3.generate("the dog", max_tokens=6)[0] == "the dog sat on the carpet.")

    # Open Mode has no separate build step anymore -- it must be
    # automatically rebuilt (not left stale, not invalidated) after
    # incremental training, in sync with the merged graphs.
    m_incr4 = MSEGraphLanguageModel(vocab_size=200)
    m_incr4.train("the cat sat on the mat. the dog sat on the carpet.")
    open_ctm_before = m_incr4.open_ctm
    check("Open Mode auto-built right after training",
          m_incr4._open is not None and open_ctm_before is not None)
    summary = m_incr4.train_incremental("the pig sat on the rug.",
                                         extend_vocab=True, target_vocab_size=250)
    check("Open Mode still available after incremental training (auto-rebuilt)",
          m_incr4._open is not None and m_incr4.open_ctm is not None)
    check("open_ctm is a FRESH object after the merge, not the stale pre-merge one",
          m_incr4.open_ctm is not open_ctm_before)
    check("the new fact is reachable in Open Mode after the merge",
          "rug" in [m_incr4.tokenizer.decode([t]) for t in m_incr4.all_candidate_tokens()])
    check("summary no longer reports a stale experience_invalidated key",
          "experience_invalidated" not in summary, summary)

    # train_incremental on a never-trained model must fail clearly rather
    # than silently doing the wrong thing.
    m_untrained = MSEGraphLanguageModel(vocab_size=200)
    try:
        m_untrained.train_incremental("anything")
        check("train_incremental on untrained model raises", False)
    except RuntimeError:
        check("train_incremental on untrained model raises RuntimeError", True)

    # Save/load round-trip after incremental training uses the exact same
    # persistence format -- no special-casing required.
    tmp2 = tempfile.mkdtemp(prefix="mse_test_incr_")
    try:
        m_incr3.save(tmp2)
        reloaded_incr = MSEGraphLanguageModel.load(tmp2)
        check("incremental model reloads with matching stats",
              reloaded_incr.stats() == m_incr3.stats())
        check("incremental model reloads with matching generation",
              reloaded_incr.generate("the cat", max_tokens=6)[0] ==
              m_incr3.generate("the cat", max_tokens=6)[0])
    finally:
        shutil.rmtree(tmp2, ignore_errors=True)

    section("Large-corpus pipeline (train_corpus.py)")

    corpus_dir = tempfile.mkdtemp(prefix="mse_test_corpus_")
    out_dir1   = tempfile.mkdtemp(prefix="mse_test_out1_")
    out_dir3   = tempfile.mkdtemp(prefix="mse_test_out3_")
    out_norec  = tempfile.mkdtemp(prefix="mse_test_outnr_")
    try:
        os.makedirs(os.path.join(corpus_dir, "subdir"), exist_ok=True)
        with open(os.path.join(corpus_dir, "a_cat.txt"), "w") as f:
            f.write("the cat sat on the mat.\n")
        with open(os.path.join(corpus_dir, "b_dog.txt"), "w") as f:
            f.write("the dog sat on the carpet.\n")
        with open(os.path.join(corpus_dir, "subdir", "c_pig.txt"), "w") as f:
            f.write("the pig sat on the rug.\n")
        with open(os.path.join(corpus_dir, "ignored.md"), "w") as f:
            f.write("this is not a txt file and must be skipped.\n")

        found_recursive = discover_txt_files(corpus_dir, recursive=True)
        check("discover_txt_files finds all 3 .txt files recursively",
              len(found_recursive) == 3, found_recursive)
        check("discover_txt_files skips non-.txt files",
              all(p.endswith(".txt") for p in found_recursive), found_recursive)
        check("discover_txt_files returns sorted (deterministic) order",
              found_recursive == sorted(found_recursive), found_recursive)

        found_top = discover_txt_files(corpus_dir, recursive=False)
        check("discover_txt_files (non-recursive) excludes subdir file",
              len(found_top) == 2, found_top)

        # Full pipeline, batch_size=1: shared vocabulary across files
        # (no UNK collisions) and a cluster that only exists because
        # all three files' facts were merged together.
        m1 = train_from_folder(corpus_dir, out_dir1, vocab_size=200,
                                batch_size=1, recursive=True, quiet=True)
        a1 = Analyser(m1)
        clusters1 = a1.cluster_report(top_n=20)
        check("pipeline merges all 3 files into one cat/dog/pig cluster",
              any(set(c["members"]) >= {"cat", "dog", "pig"} for c in clusters1),
              clusters1)
        check("pipeline gives every animal a clean (non-UNK) token",
              all(1 not in m1.tokenizer.encode(w) for w in ("cat", "dog", "pig")))
        check("pipeline generation correct for each file's fact",
              m1.generate("the cat", max_tokens=6)[0] == "the cat sat on the mat." and
              m1.generate("the dog", max_tokens=6)[0] == "the dog sat on the carpet." and
              m1.generate("the pig", max_tokens=6)[0] == "the pig sat on the rug.")

        # batch_size is a memory/speed knob only -- final structure must
        # be identical regardless of how files are grouped into batches.
        m3 = train_from_folder(corpus_dir, out_dir3, vocab_size=200,
                                batch_size=3, recursive=True, quiet=True)
        check("batch_size=1 vs batch_size=3 produce identical final stats",
              m1.stats() == m3.stats(), (m1.stats(), m3.stats()))

        # Non-recursive run must not see the subdir file's facts at all.
        m_norec = train_from_folder(corpus_dir, out_norec, vocab_size=200,
                                     batch_size=1, recursive=False, quiet=True)
        a_norec = Analyser(m_norec)
        check("non-recursive pipeline never learns the subdir fact",
              not any(set(c["members"]) >= {"cat", "dog", "pig"}
                      for c in a_norec.cluster_report(top_n=20)))

        # Saved output must reload identically.
        reloaded = MSEGraphLanguageModel.load(out_dir1)
        check("pipeline output reloads with matching stats",
              reloaded.stats() == m1.stats())
        check("pipeline output reloads with matching generation",
              reloaded.generate("the cat", max_tokens=6)[0] ==
              m1.generate("the cat", max_tokens=6)[0])

        # Empty / nonexistent folder must fail clearly, not silently.
        empty_dir = tempfile.mkdtemp(prefix="mse_test_empty_")
        try:
            try:
                train_from_folder(empty_dir, tempfile.mkdtemp(), vocab_size=200, quiet=True)
                check("train_from_folder on empty dir raises", False)
            except FileNotFoundError:
                check("train_from_folder on empty dir raises FileNotFoundError", True)
        finally:
            shutil.rmtree(empty_dir, ignore_errors=True)
    finally:
        shutil.rmtree(corpus_dir, ignore_errors=True)
        shutil.rmtree(out_dir1, ignore_errors=True)
        shutil.rmtree(out_dir3, ignore_errors=True)
        shutil.rmtree(out_norec, ignore_errors=True)

    # ── Open Mode (auto-built, vocabulary-as-candidates) ────────────────────
    section("Open Mode is auto-built as soon as the model is trained")
    m_open = MSEGraphLanguageModel(vocab_size=300)
    m_open.train(CORPUS_EXP)
    check("self._open is ready immediately after train() -- no separate build step",
          m_open._open is not None)
    check("self.open_ctm is ready immediately after train() -- no separate build step",
          m_open.open_ctm is not None)
    check("model has no experience-related attributes left at all",
          not hasattr(m_open, "exp_edges") and not hasattr(m_open, "exp_bridges")
          and not hasattr(m_open, "exp_rels"))

    section("Open Mode candidates are the ENTIRE vocabulary, not gated by successors")
    tok = m_open.tokenizer
    cat_id = tok.token_to_id.get("cat")
    dog_id = tok.token_to_id.get("dog")
    ran_id = tok.token_to_id.get("ran")
    all_ids = m_open.all_candidate_tokens()
    check("all_candidate_tokens() matches self._open.vocab exactly",
          all_ids == m_open._open.vocab, (len(all_ids), len(m_open._open.vocab)))
    if cat_id and ran_id:
        legal_strict_succs = m_open._strict._successors(cat_id)
        check("'ran' was never a literal successor of 'cat' in training "
              "(only 'sat' was) -- yet it's still a valid Open Mode candidate",
              ran_id not in legal_strict_succs and ran_id in all_ids,
              (legal_strict_succs, ran_id in all_ids))

    section("Open Mode generation")
    # Open mode should be able to generate "the cat ran" -- a transition
    # never literally observed after "cat" (only "cat sat" was trained),
    # reachable now because candidates are the whole vocabulary, scored
    # by IVM rather than gated by literal successors.
    strict_cat = m_open.generate("the cat", max_tokens=12, mode="strict")[0]
    check("strict cat stays on training path", "sat" in strict_cat, strict_cat)
    open_cat = m_open.generate("the cat", max_tokens=12, mode="open")[0]
    check("open cat does not crash", isinstance(open_cat, str) and len(open_cat) > 0)
    open_dog = m_open.generate("the dog", max_tokens=12, mode="open")[0]
    check("open dog does not crash", isinstance(open_dog, str) and len(open_dog) > 0)
    for prompt in ["the cat", "the dog", "the boy"]:
        text, _, trace = m_open.generate(prompt, max_tokens=12, mode="open")
        stages = [t["stage"] for t in trace]
        check(f"open '{prompt}' valid stages", all(s in (1, 2, 3, 4) for s in stages))

    section("Open Mode infer_shared_role")
    r_open = m_open.infer_shared_role(["cat", "dog"], mode="open")
    check("open shared-role cat+dog non-empty", len(r_open) > 0, r_open)
    sources = [ev.get("source", "") for _, _, ev in r_open]
    check("infer_shared_role's evidence is always literal training now "
          "(no separate experience source exists anymore)",
          all(s == "training" for s in sources), sources)

    section("Open Mode save/load round-trip -- no separate experience files")
    tmp2 = tempfile.mkdtemp(prefix="mse_open_test_")
    try:
        m_open.save(tmp2)
        saved_files = set(os.listdir(tmp2))
        check("no experience_*.json files are written anymore",
              not any(f.startswith("experience_") for f in saved_files), saved_files)
        m_reload = MSEGraphLanguageModel.load(tmp2)
        check("reloaded model has Open Mode auto-available (no build step needed)",
              m_reload._open is not None and m_reload.open_ctm is not None)
        for p in ["the cat", "the dog"]:
            t_orig   = m_open.generate(p,    max_tokens=12, mode="open")[0]
            t_reload = m_reload.generate(p,  max_tokens=12, mode="open")[0]
            check(f"open reload '{p}' matches", t_orig == t_reload, f"'{t_orig}' vs '{t_reload}'")
    finally:
        shutil.rmtree(tmp2, ignore_errors=True)

    section("Open Mode determinism")
    runs = {m_open.generate("the dog", max_tokens=12, mode="open")[0] for _ in range(5)}
    check("open mode 5 runs identical", len(runs)==1, runs)

    section("Open Mode: CTM weighted voting is primary (not a tie-break)")
    check("open_ctm auto-built alongside training, no separate call needed",
          m_open.open_ctm is not None)
    _, _, trace_open = m_open.generate("the dog", max_tokens=12, mode="open")
    check("every open-mode step is CTM-scored or a deterministic fallback "
          "(never Stage 2 lineage rules)",
          all(t["rule"] in ("ctm_weighted_vote", "ctm_unavailable_deterministic_fallback",
                             "termination_empty_vocabulary")
              for t in trace_open), trace_open)
    check("open-mode steps never carry a Stage-2-only rule tag",
          all(t.get("rule") not in ("exact_match_unique", "all_valid_random",
                                     "s2_empty_s1_random")
              for t in trace_open), trace_open)

    # score_candidates()/select() evaluate the FULL legal candidate set,
    # not just a pre-existing tie -- verify against a hand-checked example.
    tok = m_open.tokenizer
    def enc(w):
        e = [t for t in tok.encode(w) if t != 2]
        return e[-1]
    info = m_open.open_mode_candidate_scores("the boy sat on the")
    check("open_mode_candidate_scores exposes the full auditable trace",
          set(info.keys()) == {"candidates", "important_tokens", "influence",
                                "knows", "important_vote", "influence_vote",
                                "context_vote", "context_influence_vote",
                                "bigram_witness_vote", "adjacency_vote",
                                "prev_current_vote", "triple_vote",
                                "scores", "winner", "tie_break_stage",
                                "cache_used"}, info)
    check("open_mode_candidate_scores always covers the entire vocabulary now",
          len(info["candidates"]) == len(m_open.all_candidate_tokens()), info)
    # Now scoring the ENTIRE vocabulary (not just legal successors), so
    # many low-scoring subword/junk tokens are in the mix too -- but the
    # genuine training-grounded candidate should still win on V3/V5/V6/V7/V8
    # evidence alone.
    check("chair is still the top-scoring candidate in open mode",
          max(info["scores"], key=info["scores"].get) == "chair", info["scores"])
    check("scores equal the sum of the eight independent vote layers",
          all(abs(info["scores"][c] -
                  (info["important_vote"].get(c, 0) + info["influence_vote"].get(c, 0)
                   + info["context_vote"].get(c, 0) + info["context_influence_vote"].get(c, 0)
                   + info["bigram_witness_vote"].get(c, 0) + info["adjacency_vote"].get(c, 0)
                   + info["prev_current_vote"].get(c, 0) + info["triple_vote"].get(c, 0))) < 1e-9
              for c in info["candidates"]),
          info)

    section("Open Mode: four independent vote layers (V1 + V2 + V3 + V4)")
    # Built directly against the class so every number is exact and
    # traceable: I = {A, B} (important), P = {A, B, N} (full context,
    # N is NOT a cluster member). Candidates X, Y.
    #   A knows X with evidence 2, knows Y with evidence 1 -> influence(A)=2
    #   B knows X with evidence 1 only (doesn't know Y)     -> influence(B)=1
    #   N (unimportant) co-occurs with Y only, not X -- V3/V4 are the ONLY
    #     layers N can contribute to, and it does, for Y only.
    from ivm import ImportanceVoteMatrix
    ivm3 = ImportanceVoteMatrix()
    # This worked example's comments use clean 0.1 weights throughout for
    # readability. _important_weight already defaults to 0.1, but
    # _influence_weight's actual default is 0.09 (see ivm.py's __init__) --
    # pin it here so the hand-computed numbers in the checks below match
    # exactly, without touching the real production default.
    # _important_weight and _context_influence_weight are left at their
    # real production defaults (IVMConfig.IMPORTANT_WEIGHT = 0.4,
    # IVMConfig.CONTEXT_INFLUENCE_WEIGHT = 0.0) -- the checks below are
    # computed against those actual values, not the old 0.1/0.01
    # illustration. Only _influence_weight is pinned here, since
    # IVMConfig.INFLUENCE_WEIGHT (0.0) would zero out V2 entirely and
    # this worked example wants to demonstrate a nonzero V2.
    ivm3._influence_weight = 0.1
    ivm3._important = {"A", "B"}
    ivm3._token_rels = {
        "A": {1, 2, 3},
        "B": {50},
        "N": {60},
        "X": {1, 2, 50, 51},
        "Y": {3, 60},
    }
    trace2 = ivm3.score_candidates(["X", "Y"], {"A", "B", "N"})
    check("V1 (important_vote) is RAW/BINARY, NOT scaled by evidence count -- "
          "one vote (x0.4, IVMConfig.IMPORTANT_WEIGHT) per important token "
          "that knows C at all: X=0.4*(A knows + B knows)=0.4*2=0.8, "
          "Y=0.4*(A knows only)=0.4*1=0.4 -- note this does NOT scale with "
          "A's evidence count of 2 for X",
          abs(trace2["important_vote"]["X"] - 0.8) < 1e-9 and
          abs(trace2["important_vote"]["Y"] - 0.4) < 1e-9, trace2["important_vote"])
    check("V2 (influence_vote) = 0.1 x sum of influence(t) for t that knows C: "
          "X=0.1*(2+1)=0.3, Y=0.1*2=0.2",
          abs(trace2["influence_vote"]["X"] - 0.3) < 1e-9 and
          abs(trace2["influence_vote"]["Y"] - 0.2) < 1e-9, trace2["influence_vote"])
    check("V3 (context_vote) counts EVERY context token including N, "
          "full weight 1.0 each -- strictly bigger than V1/V2's 0.1: "
          "X gets A+B=2 votes, Y gets A+N=2 votes",
          trace2["context_vote"] == {"X": 2.0, "Y": 2.0}, trace2["context_vote"])
    check("V4 (context_influence_vote) IS scaled by raw evidence count, but "
          "IVMConfig.CONTEXT_INFLUENCE_WEIGHT is currently 0.0 -- so despite "
          "nonzero evidence (knowledge(A,X)=2, knowledge(B,X)=1, "
          "knowledge(A,Y)=1, knowledge(N,Y)=1) this layer contributes "
          "nothing to either candidate right now: X=0.0*(2+1+0)=0.0, "
          "Y=0.0*(1+0+1)=0.0",
          abs(trace2["context_influence_vote"]["X"] - 0.0) < 1e-9 and
          abs(trace2["context_influence_vote"]["Y"] - 0.0) < 1e-9,
          trace2["context_influence_vote"])
    check("N (not important) contributed to Y's score only via V3/V4 -- proof "
          "V3/V4 genuinely include non-important tokens, unlike V1/V2",
          "N" not in trace2["knows"] and trace2["context_vote"]["Y"] > trace2["important_vote"]["Y"] - 1,
          trace2)
    check("final score = V1+V2+V3+V4 exactly: X=0.8+0.3+2+0.0=3.1, Y=0.4+0.2+2+0.0=2.6",
          abs(trace2["scores"]["X"] - 3.1) < 1e-9 and
          abs(trace2["scores"]["Y"] - 2.6) < 1e-9, trace2["scores"])
    check("select() picks X, the higher combined score", 
          ivm3.select(["X", "Y"], {"A", "B", "N"})[0] == "X", trace2)
    check("V1+V2+V4 alone for X (1.1) could never have outvoted V3's "
          "contribution to Y (2.0) -- proof important tokens/magnitude can't "
          "override context presence by design",
          (trace2["important_vote"]["X"] + trace2["influence_vote"]["X"]
           + trace2["context_influence_vote"]["X"]) < trace2["context_vote"]["Y"],
          trace2)

    section("Open Mode: tie-break cascade (bigram frequency -> global frequency -> lowest id)")
    # A genuine score tie, engineered directly: two candidates with
    # identical V1/V2/V3 (X2 and Y2 both known by the same important
    # token with equal evidence, and equal context support).
    ivm4 = ImportanceVoteMatrix()
    ivm4._important = {"A"}
    ivm4._token_rels = {
        "A": {1, 2},
        "X2": {1, 100},
        "Y2": {2, 200},
    }
    trace4 = ivm4.score_candidates(["X2", "Y2"], {"A"})
    check("scores tie by construction (symmetric evidence)",
          trace4["scores"]["X2"] == trace4["scores"]["Y2"], trace4["scores"])
    # No bigram data at all -- cascade should skip straight to global
    # frequency: X2 has 2 relationships, Y2 has 2 -- still tied -- falls
    # to lowest token id.
    winner4a, _ = ivm4.select(["X2", "Y2"], {"A"}, current="the")
    check("with no bigram evidence and tied global frequency, falls to "
          "lowest token id", winner4a == min("X2", "Y2"), winner4a)

    # Now inject a real bigram-frequency gap in Y2's favor and confirm it
    # decides the tie WITHOUT changing the primary score at all.
    ivm4._bigram_freq = {("the", "X2"): 1, ("the", "Y2"): 5}
    winner4b, _ = ivm4.select(["X2", "Y2"], {"A"}, current="the")
    check("bigram frequency breaks the tie in favor of Y2 (5 > 1), "
          "even though the primary score never changed",
          winner4b == "Y2", winner4b)

    # Now make bigram frequency ALSO tie, but give X2 more global
    # (corpus-wide) frequency -- should decide it at the LAST stage.
    ivm4._bigram_freq = {("the", "X2"): 3, ("the", "Y2"): 3}
    ivm4._token_rels["X2"] = {1, 100, 101, 102}   # 4 relationships total
    winner4c, _ = ivm4.select(["X2", "Y2"], {"A"}, current="the")
    check("bigram frequency ties too -- global frequency breaks it "
          "(X2 now has 4 relationships vs Y2's 2)",
          winner4c == "X2", winner4c)

    section("Strict Mode: bigram frequency replaces random tie-break")
    ms = MSEGraphLanguageModel(vocab_size=150)
    ms.train("""
the cat sat on the bench.
the cat sat on the bench.
the cat sat on the mat.
""")
    tok_s = ms.tokenizer
    def encs(w):
        e = [t for t in tok_s.encode(w) if t != 2]
        return e[-1]
    the, bench, mat = encs("the"), encs("bench"), encs("mat")
    check("EdgeMatrix tracks real bigram counts, not just legality: "
          "the->bench seen twice, the->mat seen once",
          ms.edges.frequency(the, bench) == 2 and ms.edges.frequency(the, mat) == 1,
          (ms.edges.frequency(the, bench), ms.edges.frequency(the, mat)))
    engine_s = ms._strict
    check("_bigram_tie_break deterministically prefers the more frequent bigram",
          engine_s._bigram_tie_break(the, [bench, mat]) == bench,
          engine_s._bigram_tie_break(the, [bench, mat]))
    check("_bigram_tie_break is fully deterministic across repeated calls",
          len({engine_s._bigram_tie_break(the, [bench, mat]) for _ in range(10)}) == 1,
          None)
    # inference.py no longer imports random at all -- confirms every
    # random.choice call site was actually replaced, not just some.
    import inference as inference_module
    check("inference.py contains no 'random' import (no randomness left in Strict Mode)",
          "random" not in dir(inference_module) and not hasattr(inference_module, "random"),
          dir(inference_module))

    section("Summary")
    print(f"  {PASS} passed, {FAIL} failed")
    if FAIL: sys.exit(1)


def test_prompt_seeding_and_mode_boundaries():
    """
    Documents the strict bigram-validation boundary and deterministic
    tie-break behaviour.

    Key findings:
    - BOTH modes reject any prompt containing a bigram not in E --
      there is no longer a separate Experience Edge Matrix to widen
      what counts as a legal prompt in Open Mode. The prompt itself
      must still start from a literally-observed transition; only
      what happens AFTER the prompt differs between modes (Open
      Mode's per-step candidates are the whole vocabulary, not gated
      by successors at all).
    - Genuine ties resolve deterministically (bigram frequency, then
      global frequency, then lowest token id), never randomly.
    """
    section("Prompt seeding and mode boundaries")

    CORPUS_BOUNDARY = """
the cat sat on the mat.
the dog sat on the carpet.
the boy sat on the mat.
the boy ran on the road.
"""
    mb = MSEGraphLanguageModel(vocab_size=150)
    mb.train(CORPUS_BOUNDARY)

    # Case 1: short prompts. Strict Mode keeps its lineage-driven, exact
    # per-prompt-thread outputs unchanged.
    for prompt, valid in [("the cat",["mat"]), ("the dog",["carpet"]),
                           ("the boy",["mat","road"])]:
        text, _, _ = mb.generate(prompt, max_tokens=12, mode="strict")
        check(f"case1 '{prompt}' strict → one of {valid}",
              any(v in text for v in valid), f"got '{text}'")

    # Open Mode no longer uses lineage at all (see inference.py), so its
    # output is whichever candidate the V1+V2+V3 weighted vote scores
    # highest across the WHOLE corpus, not whichever fact this specific
    # prompt's thread pointed to. Historically (under an earlier formula
    # that multiplied influence x evidence into one term) this produced
    # 'mat' instead of the correct 'carpet' here, because 'sat' (broad
    # influence, knows 5 candidates) could multiply up its modest evidence
    # for the more-common 'mat' and swamp 'dog's narrow, correct evidence
    # for 'carpet'. Splitting influence into its OWN independent, flatly-
    # weighted vote (V2) fixes that: 'dog' -- narrow but specific -- still
    # gets a first-class V1 vote for 'carpet' that influence can no longer
    # multiply away, and 'carpet' correctly outscores 'mat'
    # (model.open_mode_candidate_scores('the dog sat on the') shows
    # carpet=6.6 vs mat=5.5). Still fully deterministic throughout.
    text, _, trace = mb.generate("the dog", max_tokens=12, mode="open")
    check("case1 'the dog' open → deterministic CTM-scored result ('carpet')",
          "carpet" in text, f"got '{text}'")
    check("case1 'the dog' open never falls back to randomness",
          all(t["rule"] != "all_valid_random" for t in trace), trace)

    text, _, _ = mb.generate("the boy", max_tokens=12, mode="open")
    check("case1 'the boy' open → one of ['mat', 'road']",
          any(v in text for v in ["mat", "road"]), f"got '{text}'")

    # Case 2: 'the cat ran' — cat->ran not in training E, in EITHER mode
    # now (no separate Experience Edge Matrix to widen prompt legality
    # in Open Mode anymore). The PROMPT itself must still start from a
    # literally-observed transition in both modes; only what happens
    # AFTER the prompt differs (Open Mode's per-step candidates are the
    # whole vocabulary, not gated by successors -- see inference.py).
    for prompt in ["the cat ran", "the dog ran"]:
        ts, _, trace_s = mb.generate(prompt, max_tokens=10, mode="strict")
        check(f"case2 '{prompt}' strict → illegal_prompt_bigram",
              trace_s[0].get("rule") == "illegal_prompt_bigram",
              f"rule={trace_s[0].get('rule')} out='{ts}'")
        to, _, trace_o = mb.generate(prompt, max_tokens=10, mode="open")
        check(f"case2 '{prompt}' open ALSO → illegal_prompt_bigram (same "
              "literal Edge Matrix gates the prompt in both modes now)",
              trace_o[0].get("rule") == "illegal_prompt_bigram",
              f"rule={trace_o[0].get('rule')} out='{to}'")

    # Case 3: 'the cat ran on the' — also illegal in BOTH modes, same
    # bad bigram, same reasoning as case 2.
    for prompt in ["the cat ran on the", "the dog ran on the"]:
        ts, _, trace_s = mb.generate(prompt, max_tokens=4, mode="strict")
        check(f"case3 '{prompt}' strict → illegal_prompt_bigram",
              trace_s[0].get("rule") == "illegal_prompt_bigram",
              f"rule={trace_s[0].get('rule')} out='{ts}'")
        to, _, trace_o = mb.generate(prompt, max_tokens=4, mode="open")
        check(f"case3 '{prompt}' open ALSO → illegal_prompt_bigram",
              trace_o[0].get("rule") == "illegal_prompt_bigram",
              f"rule={trace_o[0].get('rule')} out='{to}'")

    # Case 4: legal unambiguous prompt resolves correctly in strict
    text, _, trace = mb.generate("the cat sat", max_tokens=10, mode="strict")
    check("legal prompt 'the cat sat' → on",
          "on" in text, f"got '{text}'")
    check("first step uses stage 1 or 2",
          trace[0]["stage"] in (1, 2), f"stage was {trace[0]['stage']}")

    # Case 5: "the boy" has two structurally valid paths in CORPUS_BOUNDARY
    # (sat/ran). Previously resolved by random.choice (varied across runs);
    # now resolved deterministically by bigram frequency -- see
    # inference.py's _bigram_tie_break. Same prompt, same output every time.
    boy_runs = {mb.generate("the boy", max_tokens=10, mode="strict")[0]
                for _ in range(30)}
    check("'the boy' deterministic across runs (bigram tie-break, not random)",
          len(boy_runs) == 1, boy_runs)


def test_importance_vote_matrix():
    """
    Validates ivm.py against the exact worked example from the spec:
    corpus = CORPUS_EXP, prompt = "the boy sat on the ___", tied
    candidates = {mat, carpet, chair, road}. Expected winner: chair
    (sat, influence 3, is out-voted into agreement with boy on chair;
    boy, influence 2, only reaches chair/road).

    Also checks that IVM is a true no-op when not passed, and that
    build/save/load round-trips its vote weights.
    """
    section("Importance Vote Matrix (IVM)")

    mi = MSEGraphLanguageModel(vocab_size=150)
    mi.train(CORPUS_EXP)
    ivm = mi.build_importance_votes(mode="strict")
    check("build_importance_votes returns an IVM", ivm is not None)
    check("has_importance_votes() true after build", mi.has_importance_votes())

    tok = mi.tokenizer
    def enc(w):
        e = [t for t in tok.encode(w) if t != 2]
        return e[-1]
    def dec(t): return tok.decode([t])

    boy, sat, the, on = enc("boy"), enc("sat"), enc("the"), enc("on")
    mat, carpet, chair, road = enc("mat"), enc("carpet"), enc("chair"), enc("road")
    candidates = [mat, carpet, chair, road]
    context_tokens = [the, boy, sat, on, the]

    result = ivm.votes(candidates, context_tokens)
    check("only sat and boy are important (the/on excluded)",
          set(result["important_tokens"]) == {sat, boy},
          [dec(t) for t in result["important_tokens"]])
    check("sat has higher influence than boy",
          result["influence"].get(sat, 0) > result["influence"].get(boy, 0),
          result["influence"])
    check("chair scores boy's influence + sat's influence (2+3=5)",
          result["totals"].get(chair) == 5,
          {dec(c): v for c, v in result["totals"].items()})
    check("chair has the highest vote total",
          max(result["totals"], key=result["totals"].get) == chair,
          {dec(c): v for c, v in result["totals"].items()})

    winner = ivm.resolve_tie(candidates, context_tokens)
    check("resolve_tie picks chair", winner == chair, dec(winner) if winner is not None else None)

    # A token never votes for itself if it's also a tied candidate
    self_vote = ivm.votes([boy, chair], [the, boy, sat, on, the])
    check("important token excludes itself from its own known-candidates",
          boy not in self_vote["knows"].get(boy, {}),
          self_vote["knows"].get(boy, {}))

    # No-op contract: passing nothing changes nothing
    text_default, _, trace_default = mi.generate("the boy sat on the", max_tokens=1)
    text_no_ivm, _, trace_no_ivm = mi.generate("the boy sat on the", max_tokens=1,
                                                use_importance_votes=False)
    check("use_importance_votes=False matches default (no-op contract)",
          text_default == text_no_ivm, f"'{text_default}' vs '{text_no_ivm}'")

    # Passing an IVM changes the resolution rule used on a genuine tie
    _, _, trace_ivm = mi.generate("the boy sat on the", max_tokens=1,
                                   use_importance_votes=True)
    check("importance voting produces a distinct rule when it fires",
          trace_ivm[-1]["rule"] in ("importance_vote_resolved",
                                     "context_trigger_resolved",
                                     "importance_vote_no_signal_random")
          or trace_ivm[-1]["stage"] != 1,
          trace_ivm[-1])

    # Save/load round-trip
    d = ivm.to_dict()
    from ivm import ImportanceVoteMatrix
    ivm2 = ImportanceVoteMatrix.from_dict(d)
    winner2 = ivm2.resolve_tie(candidates, context_tokens)
    check("IVM to_dict/from_dict round-trip preserves the winner",
          winner2 == chair, dec(winner2) if winner2 is not None else None)


def test_train_py_cli_path_matches_model_api():
    """
    train.py's `main()`/`train_with_display()` build the Edge/Bridge/
    Relationship matrices by hand (for the live progress display),
    completely independently of graph.py's own EdgeMatrix.build()/
    model.py's _build_graphs()/_merge_graphs(). This is a real,
    previously-uncovered gap: test.py never exercised this code path
    at all before this function existed -- every other test in this
    suite trains through model.train()/model.train_incremental(),
    which never goes anywhere near train.py's hand-rolled construction.

    That gap hid a real bug: train.py's hand-rolled EdgeMatrix never
    set `.count` at all (stayed an empty array while `.src`/`.dst` had
    real entries), so every model trained via `python3 train.py ...`
    -- the exact command the README's own quickstart uses -- crashed
    the moment anything touched EdgeMatrix.frequency() on a real edge
    (Strict Mode's tie-break, /bigram, Open Mode's V6 adjacency vote,
    ivm.py's bigram_frequencies()). This was only caught by hand,
    running the literal reported command, not by this suite -- fixed
    here so it can't silently regress again.
    """
    section("train.py's CLI path (train_with_display) matches the model.train() API")

    import train as train_module

    m1 = MSEGraphLanguageModel(vocab_size=200)
    summary = train_module.train_with_display(
        m1, corpus_text=CORPUS_EXP, vocab_size=200,
        display=train_module.Display(quiet=True),
    )

    check("train_with_display returns without raising",
          isinstance(summary, dict), summary)
    check("EdgeMatrix.count is populated (same length as src/dst, all real "
          "counts) -- the exact bug this test exists to catch",
          len(m1.edges.count) == len(m1.edges.src) and len(m1.edges.count) > 0
          and all(c > 0 for c in m1.edges.count),
          (len(m1.edges.src), list(m1.edges.count)))

    m2 = MSEGraphLanguageModel(vocab_size=200)
    m2.train(CORPUS_EXP)
    check("train.py's EdgeMatrix.count matches model.train()'s for every pair",
          sorted(zip(m1.edges.src, m1.edges.dst, m1.edges.count)) ==
          sorted(zip(m2.edges.src, m2.edges.dst, m2.edges.count)),
          (list(zip(m1.edges.src, m1.edges.dst, m1.edges.count)),
           list(zip(m2.edges.src, m2.edges.dst, m2.edges.count))))

    # Same class of check, for the Relationship Matrix's dedup-by-content
    # + rel_count (train.py hand-rolls this matrix too, for its live
    # display -- see graph.py's RelationshipMatrix docstring for what
    # "dedup" means here). Compare via reconstructed sentence CONTENT
    # (sequence_for_relationship), not raw rel_id numbering -- the two
    # paths aren't guaranteed to assign the same numeric rel_id to the
    # same sentence, only the same SET of (content, count) pairs.
    from importance import sequence_for_relationship
    def content_counts(m):
        return sorted(
            (tuple(sequence_for_relationship(m, rid)), m.rels.count(rid))
            for rid in range(m.rels._n_rels)
        )
    check("train.py's RelationshipMatrix has rel_count populated (same "
          "length as _n_rels, all real counts) -- the same class of bug "
          "the EdgeMatrix.count check above exists to catch",
          len(m1.rels.rel_count) == m1.rels._n_rels and m1.rels._n_rels > 0
          and all(c > 0 for c in m1.rels.rel_count),
          list(m1.rels.rel_count))
    check("train.py's RelationshipMatrix dedup matches model.train()'s "
          "for every unique sentence (content, rel_count) pair",
          content_counts(m1) == content_counts(m2),
          (content_counts(m1), content_counts(m2)))
    check("train.py's RelationshipMatrix has no duplicate relationship_ids "
          "for identical sentence content -- CORPUS_EXP has 4 distinct "
          "sentences, so _n_rels must be exactly 4, not more",
          m1.rels._n_rels == 4, m1.rels._n_rels)

    # The actual reported failure: frequency() on a real edge must not crash.
    the_id = m1.tokenizer.token_to_id.get("the")
    cat_id = m1.tokenizer.token_to_id.get("cat")
    if the_id is not None and cat_id is not None:
        freq = m1.edges.frequency(the_id, cat_id)
        check("EdgeMatrix.frequency() on a real trained-via-train.py edge "
              "doesn't crash and returns a real count",
              freq > 0, freq)

    # Save/load round-trip through train.py's own JSON-writing path,
    # then exercise every count-dependent feature end to end.
    tmp = tempfile.mkdtemp(prefix="mse_trainpy_test_")
    try:
        train_module.train_with_display(
            MSEGraphLanguageModel(vocab_size=200), corpus_text=CORPUS_EXP,
            vocab_size=200, display=train_module.Display(quiet=True),
            out_path=tmp,
        )
        m_loaded = MSEGraphLanguageModel.load(tmp)
        check("count survives the save/load round-trip too",
              len(m_loaded.edges.count) == len(m_loaded.edges.src)
              and all(c > 0 for c in m_loaded.edges.count),
              list(m_loaded.edges.count))

        text, ids, trace = m_loaded.generate("the cat sat", mode="strict")
        check("generation from a reloaded train.py model doesn't crash",
              isinstance(text, str) and len(text) > 0, text)

        info = m_loaded.open_mode_candidate_scores("the cat sat on the")
        check("open_mode_candidate_scores doesn't crash on a train.py-trained "
              "model (this is exactly where V6's adjacency vote and the "
              "bigram-frequency tie-break touch EdgeMatrix.count)",
              info is not None and "scores" in info, info)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_bigram_witness_vote_and_vocab_candidates():
    """
    V5 (bigram witness vote) and V6 (adjacency vote), plus Open Mode's
    vocabulary-as-candidates architecture -- exercised directly
    against ivm.py / inference.py / model.py so the numbers are
    hand-checkable.
    """
    section("V5: bigram-witness vote")

    from ivm import ImportanceVoteMatrix, bigram_relationships

    m = MSEGraphLanguageModel(vocab_size=150)
    m.train(CORPUS_EXP)
    tok = m.tokenizer
    def enc(w):
        e = [t for t in tok.encode(w) if t != 2]
        return e[-1]
    def dec(t): return tok.decode([t])

    cat, dog, boy, sat, the, on = (enc("cat"), enc("dog"), enc("boy"),
                                    enc("sat"), enc("the"), enc("on"))
    mat, carpet, chair, road = enc("mat"), enc("carpet"), enc("chair"), enc("road")

    # CORPUS_EXP: "the cat sat on the mat.", "the dog sat on the carpet.",
    # "the boy sat on the chair.", "the boy ran on the road." -- the
    # literal CONSECUTIVE bigram (the, mat) -- "the" immediately followed
    # by "mat" -- is witnessed ONLY by "the cat sat on the mat" (rel_id
    # for that one sentence). "cat" is subject-specific to that sentence,
    # so it's a clean witness; "sat"/"on" appear in ALL three "X sat on
    # the Y" sentences and would trivially witness every one of them --
    # deliberately avoided here as a control.
    ivm = ImportanceVoteMatrix.build(m, mode="strict")
    br = bigram_relationships(m)
    check("bigram_relationships records a literal (the,mat) witness set",
          (the, mat) in br and len(br[(the, mat)]) > 0, br.get((the, mat)))

    v5 = ivm._bigram_witness_vote(the, [cat, dog], [mat, carpet])
    check("cat witnessed the->mat (its own sentence) -> full vote for mat",
          v5.get(mat, 0) == 1.0, v5)
    check("dog witnessed the->carpet (its own sentence) -> full vote for carpet",
          v5.get(carpet, 0) == 1.0, v5)
    check("cat never witnessed the->carpet (different sentence) -- no cross-vote",
          v5.get(carpet, 0) == 1.0 and (br[(the, carpet)] & (ivm._token_rels.get(cat) or set())) == set(),
          (v5, br.get((the, carpet)), ivm._token_rels.get(cat)))

    v5_none = ivm._bigram_witness_vote(None, [cat, dog], [mat])
    check("V5 is all-zero when current is None (documented degrade)",
          v5_none == {}, v5_none)

    v5_unseen = ivm._bigram_witness_vote(boy, [cat, dog], [road])
    check("no vote for a bigram nobody ever literally witnessed",
          v5_unseen.get(road, 0) == 0, v5_unseen)

    # score_candidates()/select() now require `current` to activate V5;
    # omitting it should reproduce the old V1-V4-only behavior IF V6 also
    # happens to contribute 0 here -- which it does for this specific
    # (context, candidates) pair: cat/dog are never literally ADJACENT to
    # mat/carpet/chair/road (they're 3 tokens apart: "cat sat on the mat"),
    # even though they share a sentence. This is NOT a general guarantee
    # that omitting `current` zeroes V6 too -- V6 doesn't depend on
    # `current` at all, see the dedicated V6 section below for that.
    trace_with = ivm.score_candidates([mat, carpet, chair, road],
                                       [cat, dog], current=the)
    trace_without = ivm.score_candidates([mat, carpet, chair, road],
                                          [cat, dog])
    check("adjacency_vote key is present even when current is omitted "
          "(V6 doesn't depend on current) -- happens to be all-zero here "
          "only because cat/dog are never adjacent to these candidates",
          "adjacency_vote" in trace_without and
          all(v == 0 for v in trace_without["adjacency_vote"].values()),
          trace_without.get("adjacency_vote"))
    check("bigram_witness_vote key present and non-empty when current is given",
          any(v > 0 for v in trace_with["bigram_witness_vote"].values()), trace_with)
    check("omitting current reproduces the old 4-layer score exactly",
          all(abs(trace_without["scores"][c] -
                  (trace_without["important_vote"].get(c, 0)
                   + trace_without["influence_vote"].get(c, 0)
                   + trace_without["context_vote"].get(c, 0)
                   + trace_without["context_influence_vote"].get(c, 0))) < 1e-9
              for c in [mat, carpet, chair, road]),
          trace_without)
    check("supplying current can only raise or hold a candidate's score, never lower it",
          all(trace_with["scores"][c] >= trace_without["scores"][c] - 1e-9
              for c in [mat, carpet, chair, road]),
          (trace_with["scores"], trace_without["scores"]))

    # to_dict/from_dict must round-trip the new fields
    ivm_rt = ImportanceVoteMatrix.from_dict(ivm.to_dict())
    v5_rt = ivm_rt._bigram_witness_vote(the, [cat, dog], [mat, carpet])
    check("bigram_rels/bigram_witness_weight survive to_dict/from_dict",
          v5_rt == v5 and abs(ivm_rt._bigram_witness_weight - ivm._bigram_witness_weight) < 1e-9,
          v5_rt)

    section("V6: adjacency vote")

    from ivm import adjacent_pairs

    ap = adjacent_pairs(m)
    # "the cat sat on the mat" -> literal DIRECTED consecutive pairs:
    # (the,cat), (cat,sat), (sat,on), (on,the), (the,mat). NOT adjacent:
    # cat->mat (3 tokens apart, never consecutive in either direction).
    check("adjacent_pairs() records literal DIRECTED consecutive pairs "
          "(from_token, to_token) from training",
          (the, cat) in ap and (cat, sat) in ap and
          (sat, on) in ap and (on, the) in ap and
          (the, mat) in ap,
          ap)
    check("adjacent_pairs() does NOT record cat->mat or mat->cat -- same "
          "sentence, never literally consecutive in either direction",
          (cat, mat) not in ap and (mat, cat) not in ap, ap)

    # Directionality is the whole point of V6: "the cat" occurs (the->cat),
    # but "cat the" never does (cat->the) -- only the observed direction
    # is recorded, and only that direction votes.
    check("adjacent_pairs() is directional: (the,cat) recorded because "
          "'the cat' occurs, but (cat,the) is NOT recorded -- 'cat the' "
          "never occurs",
          (the, cat) in ap and (cat, the) not in ap,
          (m.edges.frequency(the, cat), m.edges.frequency(cat, the)))

    v6_reverse = ivm._adjacency_vote([cat], [the])
    check("directionality in _adjacency_vote(): context token 'cat' was "
          "NEVER immediately followed by 'the' -> no vote for 'the'",
          v6_reverse.get(the, 0) == 0, v6_reverse)
    v6_forward = ivm._adjacency_vote([the], [cat])
    check("directionality in _adjacency_vote(): context token 'the' WAS "
          "immediately followed by 'cat' -> full vote for 'cat'",
          v6_forward.get(cat, 0) == 1.0, v6_forward)

    v6 = ivm._adjacency_vote([cat, on], [sat, mat, road])
    check("cat was immediately followed by 'sat' -> full vote for 'sat'",
          v6.get(sat, 0) == 1.0, v6)
    check("no vote for 'mat' -- neither cat nor 'on' was ever immediately "
          "followed by 'mat' ('on' is always followed by 'the', not 'mat' "
          "directly -- 'on the mat', not 'on mat')",
          v6.get(mat, 0) == 0, v6)
    check("no vote for 'road' -- same reasoning, no direct t->road edge",
          v6.get(road, 0) == 0, v6)

    v6_empty_ctm = ImportanceVoteMatrix()  # bare object, self._adjacent = set()
    v6_empty = v6_empty_ctm._adjacency_vote([cat, sat], [the, mat])
    check("a bare (unbuilt) ImportanceVoteMatrix has no adjacency evidence "
          "at all -- _adjacent defaults to an empty set, so V6 is all-zero, "
          "never an error",
          v6_empty == {}, v6_empty)

    v6_self = ivm._adjacency_vote([the], [the])
    check("a token never votes for a candidate it IS, even if trivially "
          "'immediately followed by itself' would otherwise apply",
          v6_self == {}, v6_self)

    # score_candidates() integration: V6 doesn't require `current` at all
    # (unlike V5), so it should fire identically whether or not current is
    # supplied, and combine additively with the other five layers.
    trace_v6 = ivm.score_candidates([sat, mat, road], [cat, on])
    check("adjacency_vote is populated in the full trace without needing current",
          trace_v6["adjacency_vote"].get(sat, 0) == ivm._adjacency_weight * 1.0,
          trace_v6["adjacency_vote"])
    check("V6's contribution is included in the final combined score",
          abs(trace_v6["scores"][sat] -
              (trace_v6["important_vote"].get(sat, 0) + trace_v6["influence_vote"].get(sat, 0)
               + trace_v6["context_vote"].get(sat, 0) + trace_v6["context_influence_vote"].get(sat, 0)
               + trace_v6["bigram_witness_vote"].get(sat, 0) + trace_v6["adjacency_vote"].get(sat, 0))) < 1e-9,
          trace_v6["scores"])

    # to_dict/from_dict must round-trip V6's index and weight too
    ivm_rt2 = ImportanceVoteMatrix.from_dict(ivm.to_dict())
    check("adjacent pairs survive to_dict/from_dict (including direction)",
          ivm_rt2._adjacent == ivm._adjacent, len(ivm_rt2._adjacent))
    check("adjacency_weight survives to_dict/from_dict",
          abs(ivm_rt2._adjacency_weight - ivm._adjacency_weight) < 1e-9,
          ivm_rt2._adjacency_weight)
    v6_rt = ivm_rt2._adjacency_vote([cat, on], [sat, mat, road])
    check("_adjacency_vote gives identical results after a round-trip",
          v6_rt == v6, (v6_rt, v6))

    section("Open Mode: vocabulary is ALWAYS the candidate source now")

    m2 = MSEGraphLanguageModel(vocab_size=150)
    m2.train(CORPUS_EXP)

    ids = m2.tokenizer.encode("the boy sat on the")
    current2 = ids[-1]
    legal = set(m2._strict._successors(current2))
    full_pool = set(m2.all_candidate_tokens())
    check("all_candidate_tokens() is a strict superset of literal successors",
          legal < full_pool, (len(legal), len(full_pool)))
    check("all_candidate_tokens() excludes PAD/UNK/BOS",
          not ({0, 1, 2} & full_pool), full_pool & {0, 1, 2})
    check("self._open.vocab IS all_candidate_tokens() -- no separate toggle exists anymore",
          m2._open.vocab == m2.all_candidate_tokens())

    text_open, ids_open, trace_open = m2.generate(
        "the boy sat on the", max_tokens=6, mode="open")
    check("open mode generation completes without crashing",
          isinstance(text_open, str))
    check("every open-mode step's candidates are the ENTIRE vocabulary, not "
          "just legal successors -- no opt-in needed, this is just what "
          "Open Mode does now",
          all(set(t.get("candidates", [])) == full_pool
              for t in trace_open if t["stage"] != 4),
          [len(t.get("candidates", [])) for t in trace_open])

    runs = {m2.generate("the boy sat on the", max_tokens=6, mode="open")[0]
            for _ in range(5)}
    check("open mode generation is fully deterministic",
          len(runs) == 1, runs)

    info_open = m2.open_mode_candidate_scores("the boy sat on the")
    check("open_mode_candidate_scores always scores the entire vocabulary",
          len(info_open["candidates"]) == len(full_pool),
          (len(info_open["candidates"]), len(full_pool)))

    # Strict Mode is completely unaffected -- it never had vocabulary-wide
    # candidates and still doesn't; its own successors-gated Stage 1/2
    # pipeline is untouched by any of this.
    strict_ids = m2.tokenizer.encode("the boy sat on the")
    tok_strict, trace_strict = m2._strict.step(strict_ids[-2], strict_ids[-1])
    check("Strict Mode's own step() still exists and works normally",
          isinstance(tok_strict, int), m2.tokenizer.decode([tok_strict]))
    check("Strict Mode's candidates are still gated to legal successors, "
          "never the whole vocabulary",
          set(trace_strict.get("candidates", legal)) <= legal or "candidates" not in trace_strict,
          trace_strict)

    section("model._engine(): invalid mode string must raise, never silently fall back to Strict")
    # Regression: _engine() used to check ONLY `if mode == "open"`, so any
    # other string (a typo, an unvalidated API field) silently fell through
    # to Strict Mode instead of erroring -- found via server.py's /bigram
    # endpoint accepting an unchecked `mode` field from a JSON request body.
    raised = False
    try:
        m2._engine("bogus")
    except ValueError:
        raised = True
    except Exception:
        pass
    check("_engine('bogus') raises ValueError instead of silently returning Strict",
          raised, "no exception raised")
    check("_engine('strict')/_engine('open') still work normally",
          m2._engine("strict") is m2._strict and m2._engine("open") is m2._open)


def test_prev_current_co_occurrence_vote():
    """
    V7 (previous+current co-occurrence vote) -- exercised directly
    against ivm.py / inference.py / model.py, same hand-checkable
    style as the V5/V6 section above.
    """
    section("V7: previous+current co-occurrence vote")

    from ivm import ImportanceVoteMatrix

    m = MSEGraphLanguageModel(vocab_size=150)
    m.train(CORPUS_EXP)
    tok = m.tokenizer
    def enc(w):
        e = [t for t in tok.encode(w) if t != 2]
        return e[-1]

    cat, dog, boy, sat, ran, the, on = (enc("cat"), enc("dog"), enc("boy"),
                                          enc("sat"), enc("ran"), enc("the"), enc("on"))
    mat, carpet, chair, road = enc("mat"), enc("carpet"), enc("chair"), enc("road")

    # CORPUS_EXP: rel 1 "the cat sat on the mat.", rel 2 "the dog sat on
    # the carpet.", rel 3 "the boy sat on the chair.", rel 4 "the boy ran
    # on the road." -- "sat" only occurs in rels {1,2,3} (NOT 4, which has
    # "ran" instead), so (sat, on) is a fixed pair whose shared sentences
    # are exactly {1,2,3} -- a clean way to demonstrate V7 votes for
    # mat/carpet/chair but not road, without relying on adjacency at all.
    ivm = ImportanceVoteMatrix.build(m, mode="strict")

    v7 = ivm._prev_current_vote(sat, on, [mat, carpet, chair, road])
    check("previous='sat' current='on' -> full vote for 'mat' (rel 1, "
          "shared by sat/on/mat)",
          v7.get(mat, 0) == 1.0, v7)
    check("previous='sat' current='on' -> full vote for 'carpet' (rel 2)",
          v7.get(carpet, 0) == 1.0, v7)
    check("previous='sat' current='on' -> full vote for 'chair' (rel 3)",
          v7.get(chair, 0) == 1.0, v7)
    check("previous='sat' current='on' -> NO vote for 'road' -- 'sat' "
          "never occurs in rel 4 at all ('ran' does instead), so sat/on "
          "never share a sentence with road even though on/road do",
          v7.get(road, 0) == 0, v7)

    # Narrower than V3: 'boy' co-occurs with 'chair' (rel 3) on its own,
    # but the specific PAIR (boy, ran) only ever shares a sentence with
    # 'road' (rel 4), never with 'chair' -- V7 should reflect the pair,
    # not just boy's own broader co-occurrence.
    v7b = ivm._prev_current_vote(boy, ran, [chair, road])
    check("previous='boy' current='ran' -> vote for 'road' (rel 4, the "
          "only sentence containing both boy and ran)",
          v7b.get(road, 0) == 1.0, v7b)
    check("previous='boy' current='ran' -> NO vote for 'chair', even "
          "though 'boy' alone co-occurs with 'chair' in rel 3 -- 'ran' "
          "does not, so the pair never shares a sentence with chair "
          "(this is what makes V7 narrower than V3's single-token test)",
          v7b.get(chair, 0) == 0, v7b)

    v7_none1 = ivm._prev_current_vote(None, on, [mat])
    v7_none2 = ivm._prev_current_vote(sat, None, [mat])
    check("V7 is all-zero when previous is None (documented degrade)",
          v7_none1 == {}, v7_none1)
    check("V7 is all-zero when current is None (documented degrade)",
          v7_none2 == {}, v7_none2)

    v7_self = ivm._prev_current_vote(sat, on, [sat, on, mat])
    check("a candidate never votes via being previous or current itself "
          "-- only 'mat' should appear, not 'sat' or 'on'",
          v7_self == {mat: 1.0}, v7_self)

    v7_empty_ctm = ImportanceVoteMatrix()  # bare object, no token_rels built
    v7_empty = v7_empty_ctm._prev_current_vote(sat, on, [mat])
    check("a bare (unbuilt) ImportanceVoteMatrix has no relationship "
          "evidence at all -- V7 is all-zero, never an error",
          v7_empty == {}, v7_empty)

    # score_candidates()/select() integration -- V7 requires BOTH
    # `previous` and `current`; combines additively with V1-V6.
    trace_v7 = ivm.score_candidates([mat, carpet, chair, road], [cat, dog],
                                     current=on, previous=sat)
    check("prev_current_vote key is present and matches _prev_current_vote, "
          "scaled by prev_current_weight",
          trace_v7["prev_current_vote"].get(mat, 0) == ivm._prev_current_weight * 1.0,
          trace_v7["prev_current_vote"])
    check("V7's contribution is included in the final combined score",
          abs(trace_v7["scores"][mat] -
              (trace_v7["important_vote"].get(mat, 0) + trace_v7["influence_vote"].get(mat, 0)
               + trace_v7["context_vote"].get(mat, 0) + trace_v7["context_influence_vote"].get(mat, 0)
               + trace_v7["bigram_witness_vote"].get(mat, 0) + trace_v7["adjacency_vote"].get(mat, 0)
               + trace_v7["prev_current_vote"].get(mat, 0))) < 1e-9,
          trace_v7["scores"])

    trace_no_prev = ivm.score_candidates([mat, carpet, chair, road], [cat, dog], current=on)
    check("omitting `previous` zeroes V7 specifically, without affecting "
          "that omission being visible as an empty prev_current_vote key",
          trace_no_prev.get("prev_current_vote", {}).get(mat, 0) == 0,
          trace_no_prev.get("prev_current_vote"))

    winner7, _ = ivm.select([mat, carpet, chair, road], [cat, dog],
                             current=on, previous=sat)
    check("select() runs end-to-end with `previous` supplied and returns "
          "a deterministic winner",
          winner7 is not None, winner7)

    # to_dict/from_dict must round-trip prev_current_weight (V7 itself
    # needs no separate precomputed index -- it reuses _token_rels,
    # which is already covered by the existing round-trip tests).
    ivm_rt = ImportanceVoteMatrix.from_dict(ivm.to_dict())
    check("prev_current_weight survives to_dict/from_dict",
          abs(ivm_rt._prev_current_weight - ivm._prev_current_weight) < 1e-9,
          ivm_rt._prev_current_weight)
    v7_rt = ivm_rt._prev_current_vote(sat, on, [mat, carpet, chair, road])
    check("_prev_current_vote gives identical results after a round-trip",
          v7_rt == v7, (v7_rt, v7))

    # inference.py / model.py integration: Open Mode's real step()/
    # open_mode_candidate_scores() must thread `previous` all the way
    # through to V7 without crashing.
    info = m.open_mode_candidate_scores("the cat sat on the")
    check("open_mode_candidate_scores() exposes prev_current_vote in "
          "its public breakdown",
          "prev_current_vote" in info, info.keys())

    token, trace_step = m._open.step(sat, on, importance_votes=ivm)
    check("Open Mode's step() accepts `previous` and returns a token "
          "without error (V7 now wired through inference.py)",
          isinstance(token, int), trace_step)


def test_triple_witness_vote():
    """
    V8 (triple witness vote) -- the strictest of the eight layers:
    unlike V7 (did previous/current/C ever merely share a sentence),
    V8 asks whether (previous, current, C) was ever literally ONE
    consecutive trained Bridge Matrix triple, in that exact order.
    Uses the same CORPUS_EXP / enc() setup as the V7 section above so
    the two can be directly compared on the SAME (previous, current)
    pairs.
    """
    section("V8: triple witness vote")

    from ivm import ImportanceVoteMatrix

    m = MSEGraphLanguageModel(vocab_size=150)
    m.train(CORPUS_EXP)
    tok = m.tokenizer
    def enc(w):
        e = [t for t in tok.encode(w) if t != 2]
        return e[-1]

    cat, dog, boy, sat, ran, the, on = (enc("cat"), enc("dog"), enc("boy"),
                                          enc("sat"), enc("ran"), enc("the"), enc("on"))
    mat, carpet, chair, road = enc("mat"), enc("carpet"), enc("chair"), enc("road")

    # CORPUS_EXP: rel 1 "the cat sat on the mat.", rel 2 "the dog sat on
    # the carpet.", rel 3 "the boy sat on the chair.", rel 4 "the boy ran
    # on the road." -- the literal triple immediately following (sat, on)
    # is ALWAYS (sat, on, the) in every one of rels 1-3 (the word right
    # after "on" is always "the"), never (sat, on, mat/carpet/chair)
    # directly -- those nouns are the NEXT triple's target, one step
    # further out. This is exactly the case V7's own test section
    # demonstrates a vote for (mat/carpet/chair all share a sentence
    # with sat+on) -- V8 must NOT vote for any of them here, proving
    # V8 is strictly narrower than V7 on the identical pair.
    ivm = ImportanceVoteMatrix.build(m, mode="strict")

    v8 = ivm._triple_vote(sat, on, [mat, carpet, chair, road, the])
    check("previous='sat' current='on' -> NO vote for 'mat', even though "
          "V7 votes for it (sat/on/mat share rel 1) -- the literal triple "
          "after (sat, on) is (sat, on, the), not (sat, on, mat)",
          v8.get(mat, 0) == 0, v8)
    check("previous='sat' current='on' -> NO vote for 'carpet' either "
          "(same reasoning, rel 2)",
          v8.get(carpet, 0) == 0, v8)
    check("previous='sat' current='on' -> NO vote for 'chair' either "
          "(same reasoning, rel 3)",
          v8.get(chair, 0) == 0, v8)
    check("previous='sat' current='on' -> NO vote for 'road' -- 'sat' "
          "never occurs in rel 4 at all",
          v8.get(road, 0) == 0, v8)
    check("previous='sat' current='on' -> full vote for 'the' -- "
          "(sat, on, the) IS the literal trained triple in rels 1, 2, "
          "AND 3 all at once",
          v8.get(the, 0) == 1.0, v8)

    # Now the pair one step further along: (on, the) is immediately
    # followed by mat/carpet/chair/road in rels 1/2/3/4 respectively --
    # every one of them IS the literal triple here, unlike the (sat, on)
    # pair above.
    v8b = ivm._triple_vote(on, the, [mat, carpet, chair, road, sat])
    check("previous='on' current='the' -> full vote for 'mat' -- "
          "(on, the, mat) is the literal triple in rel 1",
          v8b.get(mat, 0) == 1.0, v8b)
    check("previous='on' current='the' -> full vote for 'carpet' (rel 2)",
          v8b.get(carpet, 0) == 1.0, v8b)
    check("previous='on' current='the' -> full vote for 'chair' (rel 3)",
          v8b.get(chair, 0) == 1.0, v8b)
    check("previous='on' current='the' -> full vote for 'road' (rel 4)",
          v8b.get(road, 0) == 1.0, v8b)
    check("previous='on' current='the' -> NO vote for 'sat' -- 'sat' "
          "was never the token right after (on, the) in any sentence",
          v8b.get(sat, 0) == 0, v8b)

    # Direct V7-vs-V8 contrast on the identical (previous, current) pair
    # used in the V7 section: (boy, ran) shares a sentence (rel 4) with
    # 'road' -- V7 votes for it -- but the literal triple right after
    # (boy, ran) is (boy, ran, on), not (boy, ran, road) -- V8 must not.
    v8c = ivm._triple_vote(boy, ran, [on, road, chair])
    check("previous='boy' current='ran' -> full vote for 'on' -- "
          "(boy, ran, on) is the literal triple in rel 4",
          v8c.get(on, 0) == 1.0, v8c)
    check("previous='boy' current='ran' -> NO vote for 'road', unlike V7 "
          "-- 'road' shares rel 4 with boy+ran but is not the literal "
          "very-next token after (boy, ran)",
          v8c.get(road, 0) == 0, v8c)
    check("previous='boy' current='ran' -> NO vote for 'chair' -- never "
          "part of the same sentence as 'ran' at all",
          v8c.get(chair, 0) == 0, v8c)

    v8_none1 = ivm._triple_vote(None, on, [mat])
    v8_none2 = ivm._triple_vote(sat, None, [mat])
    check("V8 is all-zero when previous is None (documented degrade)",
          v8_none1 == {}, v8_none1)
    check("V8 is all-zero when current is None (documented degrade)",
          v8_none2 == {}, v8_none2)

    v8_self = ivm._triple_vote(on, the, [on, the, mat])
    check("a candidate never votes via being previous or current itself "
          "-- only 'mat' should appear, not 'on' or 'the'",
          v8_self == {mat: 1.0}, v8_self)

    v8_empty_ivm = ImportanceVoteMatrix()  # bare object, no triples built
    v8_empty = v8_empty_ivm._triple_vote(on, the, [mat])
    check("a bare (unbuilt) ImportanceVoteMatrix has no triple evidence "
          "at all -- V8 is all-zero, never an error",
          v8_empty == {}, v8_empty)

    # score_candidates()/select() integration -- V8 requires BOTH
    # `previous` and `current`; combines additively with V1-V7.
    trace_v8 = ivm.score_candidates([mat, carpet, chair, road], [cat, dog],
                                     current=the, previous=on)
    check("triple_vote key is present and matches _triple_vote, scaled "
          "by triple_weight",
          trace_v8["triple_vote"].get(mat, 0) == ivm._triple_weight * 1.0,
          trace_v8["triple_vote"])
    check("V8's contribution is included in the final combined score",
          abs(trace_v8["scores"][mat] -
              (trace_v8["important_vote"].get(mat, 0) + trace_v8["influence_vote"].get(mat, 0)
               + trace_v8["context_vote"].get(mat, 0) + trace_v8["context_influence_vote"].get(mat, 0)
               + trace_v8["bigram_witness_vote"].get(mat, 0) + trace_v8["adjacency_vote"].get(mat, 0)
               + trace_v8["prev_current_vote"].get(mat, 0) + trace_v8["triple_vote"].get(mat, 0))) < 1e-9,
          trace_v8["scores"])

    trace_no_prev8 = ivm.score_candidates([mat, carpet, chair, road], [cat, dog], current=the)
    check("omitting `previous` zeroes V8 specifically, without affecting "
          "that omission being visible as an empty triple_vote key",
          trace_no_prev8.get("triple_vote", {}).get(mat, 0) == 0,
          trace_no_prev8.get("triple_vote"))

    winner8, _ = ivm.select([mat, carpet, chair, road], [cat, dog],
                             current=the, previous=on)
    check("select() runs end-to-end with V8 evidence present and returns "
          "a deterministic winner",
          winner8 is not None, winner8)

    # to_dict/from_dict must round-trip triple_weight AND the precomputed
    # _triples set itself (unlike V7, V8 needs its own dedicated index --
    # it does NOT reuse _token_rels).
    ivm_rt = ImportanceVoteMatrix.from_dict(ivm.to_dict())
    check("triple_weight survives to_dict/from_dict",
          abs(ivm_rt._triple_weight - ivm._triple_weight) < 1e-9,
          ivm_rt._triple_weight)
    v8_rt = ivm_rt._triple_vote(on, the, [mat, carpet, chair, road, sat])
    check("_triple_vote gives identical results after a round-trip",
          v8_rt == v8b, (v8_rt, v8b))

    # inference.py / model.py integration: Open Mode's real step()/
    # open_mode_candidate_scores() must thread `previous` all the way
    # through to V8 without crashing.
    info = m.open_mode_candidate_scores("the cat sat on the")
    check("open_mode_candidate_scores() exposes triple_vote in its "
          "public breakdown",
          "triple_vote" in info, info.keys())

    token8, trace_step8 = m._open.step(on, the, importance_votes=ivm)
    check("Open Mode's step() accepts `previous` and returns a token "
          "without error (V8 now wired through inference.py)",
          isinstance(token8, int), trace_step8)


def test_sparse_token_score_cache():
    """
    Sparse per-token score cache for V1/V2/V3/V4/V6 (see build_cache()/
    enable_cache()/disable_cache() and the "cache" branch of
    score_candidates() in ivm.py). V5, V7, and V8 are always live -- this
    section exists to prove the cache is a pure optimization: identical
    numbers whether it's on or off, never a shortcut that changes the
    answer, plus the toggle and fallback behavior around it.
    """
    section("Sparse per-token score cache (V1/V2/V3/V4/V6, opt-in)")

    from ivm import ImportanceVoteMatrix
    import random as _random

    m = MSEGraphLanguageModel(vocab_size=200)
    m.train(CORPUS)
    candidates = m.all_candidate_tokens()

    ivm_live = ImportanceVoteMatrix.build(m, mode="strict", use_cache=False)
    ivm_cached = ImportanceVoteMatrix.build(m, mode="strict", use_cache=True)

    check("use_cache=False (default) leaves the cache empty and off",
          ivm_live._use_cache is False and ivm_live._token_cache == {},
          (ivm_live._use_cache, len(ivm_live._token_cache)))
    check("use_cache=True builds a nonempty cache and turns it on",
          ivm_cached._use_cache is True and len(ivm_cached._token_cache) > 0,
          (ivm_cached._use_cache, len(ivm_cached._token_cache)))
    check("cache is built for exactly the full-vocabulary candidate set "
          "(model.all_candidate_tokens()) -- the one set Open Mode "
          "actually calls score_candidates() with",
          ivm_cached._cache_candidates == frozenset(candidates),
          (len(ivm_cached._cache_candidates), len(candidates)))

    # Cache is SPARSE -- only nonzero (token, candidate) rows/entries are
    # stored, never a dense |vocab| x |vocab| table.
    dense_size = len(candidates) * len(candidates)
    sparse_size = sum(len(row) for row in ivm_cached._token_cache.values())
    check("sparse cache stores far fewer (t,c) entries than a dense "
          "|vocab| x |vocab| table would",
          0 < sparse_size < dense_size, (sparse_size, dense_size))

    # Correctness: cached and live paths must agree on EVERY score, across
    # many random contexts -- not just one hand-picked example.
    _random.seed(0)
    mismatches = 0
    trials = 0
    for _ in range(60):
        ctx = set(_random.sample(candidates, k=min(5, len(candidates))))
        current = _random.choice(list(ctx))
        previous = _random.choice(list(ctx))
        trace_live = ivm_live.score_candidates(candidates, ctx, current=current, previous=previous)
        trace_cached = ivm_cached.score_candidates(candidates, ctx, current=current, previous=previous)
        for c in candidates:
            trials += 1
            if abs(trace_live["scores"][c] - trace_cached["scores"][c]) > 1e-9:
                mismatches += 1
    check("cached and live score_candidates() agree exactly across many "
          f"random contexts ({trials} candidate-scores compared)",
          mismatches == 0, mismatches)

    # cache_used trace field reports which path actually ran
    ctx = set(candidates[:4])
    trace_c = ivm_cached.score_candidates(candidates, ctx, current=candidates[0])
    trace_l = ivm_live.score_candidates(candidates, ctx, current=candidates[0])
    check("trace exposes cache_used=True when the cache path ran",
          trace_c["cache_used"] is True, trace_c["cache_used"])
    check("trace exposes cache_used=False when the live path ran",
          trace_l["cache_used"] is False, trace_l["cache_used"])

    # Candidate-set mismatch -- e.g. a narrower successor set -- must fall
    # back to the live path silently, never use a stale/wrong cache.
    narrower = candidates[:3]
    trace_mismatch = ivm_cached.score_candidates(narrower, set(narrower), current=narrower[0])
    check("a candidate set different from the one the cache was built "
          "for falls back to the live path (never a wrong answer)",
          trace_mismatch["cache_used"] is False, trace_mismatch["cache_used"])

    # disable_cache()/enable_cache() toggle without losing the build
    ivm_cached.disable_cache()
    trace_off = ivm_cached.score_candidates(candidates, ctx, current=candidates[0])
    check("disable_cache() turns the cache off",
          trace_off["cache_used"] is False, trace_off["cache_used"])
    ivm_cached.enable_cache()  # no candidates -- reuse the prior build
    trace_on = ivm_cached.score_candidates(candidates, ctx, current=candidates[0])
    check("enable_cache() with no arguments reuses the previous build "
          "instantly (no rebuild needed)",
          trace_on["cache_used"] is True, trace_on["cache_used"])

    fresh = ImportanceVoteMatrix()
    raised = False
    try:
        fresh.enable_cache()
    except ValueError:
        raised = True
    check("enable_cache() with nothing ever built and no candidates "
          "given raises, rather than silently scoring against an "
          "empty/nonexistent cache", raised)

    # Weight changes are re-applied at score time, not baked into the
    # cache -- changing a weight after build_cache() must NOT stale it.
    ivm_w = ImportanceVoteMatrix.build(m, mode="strict", use_cache=True)
    before = ivm_w.score_candidates(candidates, ctx, current=candidates[0])["scores"]
    ivm_w._context_weight = 5.0
    after = ivm_w.score_candidates(candidates, ctx, current=candidates[0])["scores"]
    check("changing a weight after the cache is built still changes the "
          "score -- the cache stores raw evidence, not pre-weighted "
          "votes, so it can't go stale when a weight changes",
          any(abs(before[c] - after[c]) > 1e-9 for c in candidates),
          (before, after))

    # to_dict/from_dict: the cache itself is NOT serialized (fully
    # re-derivable from state that IS serialized) -- restored objects
    # always start with the cache off, even if the original had it on.
    d = ivm_cached.to_dict()
    check("to_dict() does not serialize the cache itself",
          "token_cache" not in d and "cache_candidates" not in d, sorted(d.keys()))
    ivm_rt = ImportanceVoteMatrix.from_dict(d)
    check("from_dict() always restores with the cache off, even though "
          "the original had it enabled -- must be rebuilt explicitly",
          ivm_rt._use_cache is False and ivm_rt._cache_candidates is None,
          (ivm_rt._use_cache, ivm_rt._cache_candidates))

    # select() and inference.py/model.py integration
    winner_c, _ = ivm_cached.select(candidates, ctx, current=candidates[0])
    winner_l, _ = ivm_live.select(candidates, ctx, current=candidates[0])
    check("select() returns the same winner whether the cache is on or off",
          winner_c == winner_l, (winner_c, winner_l))

    m2 = MSEGraphLanguageModel(vocab_size=200)
    m2.train(CORPUS)
    m2.open_ctm.enable_cache(m2.all_candidate_tokens())
    info = m2.open_mode_candidate_scores("the cat sat on the")
    check("open_mode_candidate_scores() works end-to-end with the cache "
          "enabled on the model's own open_ctm",
          info is not None and "scores" in info, info)
    check("open_mode_candidate_scores() surfaces cache_used=True in its "
          "public breakdown when the model's open_ctm has caching on",
          info["cache_used"] is True, info["cache_used"])


def test_relationship_matrix_dedup():
    """
    RelationshipMatrix.build() now deduplicates by literal sentence
    content, the same principle EdgeMatrix already applies to bigrams
    and BridgeMatrix already applies to triples -- see graph.py's
    RelationshipMatrix docstring. A corpus with an exact duplicate
    sentence should produce ONE relationship_id for it (rel_count=2),
    not two relationship_ids.
    """
    section("Relationship Matrix: dedup by sentence content")

    from importance import sequence_for_relationship

    # "the cat sat on the mat." appears twice, verbatim -- everything
    # else is distinct.
    corpus = ("the cat sat on the mat.\n"
              "the dog sat on the carpet.\n"
              "the cat sat on the mat.\n"
              "the boy ran on the road.\n")
    m = MSEGraphLanguageModel(vocab_size=200)
    m.train(corpus)

    check("3 distinct sentences -> exactly 3 relationship_ids, not 4 "
          "(the raw sentence count)",
          m.rels._n_rels == 3, m.rels._n_rels)
    check("total occurrences across all relationship_ids still equals "
          "the raw sentence count (4) -- dedup shrinks _n_rels, never "
          "loses the fact that a repeat happened",
          sum(m.rels.rel_count) == 4, list(m.rels.rel_count))

    counts_by_content = {
        tuple(sequence_for_relationship(m, rid)): m.rels.count(rid)
        for rid in range(m.rels._n_rels)
    }
    the_cat = m.tokenizer.encode_for_training("the cat sat on the mat.")
    the_dog = m.tokenizer.encode_for_training("the dog sat on the carpet.")
    the_boy = m.tokenizer.encode_for_training("the boy ran on the road.")
    check("the repeated sentence's relationship_id has rel_count == 2",
          counts_by_content.get(tuple(the_cat)) == 2, counts_by_content)
    check("a non-repeated sentence's relationship_id has rel_count == 1",
          counts_by_content.get(tuple(the_dog)) == 1, counts_by_content)
    check("every relationship_id's rel_count matches count(rel_id)",
          all(m.rels.count(rid) == m.rels.rel_count[rid]
              for rid in range(m.rels._n_rels)))
    check("count() on an out-of-range rel_id returns 0, never raises",
          m.rels.count(9999) == 0 and m.rels.count(-1) == 0)

    check("model.stats()['relationships'] reports the UNIQUE count (3)",
          m.stats()["relationships"] == 3, m.stats())
    check("model.stats()['relationship_occurrences'] reports the RAW "
          "total (4) -- the two numbers now genuinely differ when a "
          "corpus has repeats, instead of relationships silently "
          "inflating to match raw count",
          m.stats()["relationship_occurrences"] == 4, m.stats())

    # to_dict/from_dict round-trip must preserve rel_count exactly.
    m2_rels = RelationshipMatrix.from_dict(m.rels.to_dict())
    check("rel_count survives to_dict/from_dict exactly",
          list(m2_rels.rel_count) == list(m.rels.rel_count),
          (list(m2_rels.rel_count), list(m.rels.rel_count)))

    # Backward compatibility: a dict saved before rel_count existed
    # (no "rel_count" key at all) must still load, defaulting every
    # count to 1 -- same fallback EdgeMatrix.from_dict already uses.
    old_style = m.rels.to_dict()
    del old_style["rel_count"]
    m3_rels = RelationshipMatrix.from_dict(old_style)
    check("from_dict on a pre-rel_count save defaults every count to 1, "
          "never crashes",
          list(m3_rels.rel_count) == [1] * m3_rels._n_rels,
          list(m3_rels.rel_count))

    # train_incremental: re-feeding an EXISTING sentence verbatim must
    # collapse onto its existing relationship_id, not mint a new one --
    # see train_incremental's own docstring for this exact guarantee.
    m4 = MSEGraphLanguageModel(vocab_size=200)
    m4.train("the cat sat on the mat.\nthe dog sat on the carpet.\n")
    before = m4.stats()
    m4.train_incremental("the cat sat on the mat.\nthe boy ran on the road.\n")
    after = m4.stats()
    check("train_incremental: one genuinely NEW sentence -> "
          "relationships grows by exactly 1, not 2",
          after["relationships"] == before["relationships"] + 1,
          (before, after))
    check("train_incremental: the repeated sentence's occurrence count "
          "grows even though its relationship_id didn't change",
          after["relationship_occurrences"] == before["relationship_occurrences"] + 2,
          (before, after))
    the_cat_m4 = m4.tokenizer.encode_for_training("the cat sat on the mat.")
    cat_rel_id = next(rid for rid in range(m4.rels._n_rels)
                       if tuple(sequence_for_relationship(m4, rid)) == tuple(the_cat_m4))
    check("the repeated sentence's rel_count is now 2 after the merge",
          m4.rels.count(cat_rel_id) == 2, m4.rels.count(cat_rel_id))

    # analyse.py surfaces the same numbers
    a = Analyser(m)
    rr = a.relationship_report()
    check("Analyser.relationship_report()'s total_relationships matches "
          "the unique count",
          rr["total_relationships"] == 3, rr)
    check("Analyser.relationship_report()'s total_occurrences matches "
          "the raw total",
          rr["total_occurrences"] == 4, rr)
    the_cat_rid = next(rid for rid in range(m.rels._n_rels)
                        if tuple(sequence_for_relationship(m, rid)) == tuple(the_cat))
    rd = a.relationship_detail(the_cat_rid)
    check("Analyser.relationship_detail() reports occurrences==2 for the "
          "repeated sentence",
          rd["occurrences"] == 2, rd)


def test_cluster_axis_indexed_lookup():
    """
    BridgeMatrix.cluster_axis() used to scan EVERY triple in the graph
    on every call -- fine in isolation, but importance.py's
    _trigger_for_triple() calls it once per CLUSTERED triple (via
    ivm.py's important_member_tokens(), which every train()/
    train_incremental() call rebuilds automatically), which made the
    whole training pipeline effectively O(triples^2). Fixed with a
    lazy cluster_id -> [triple_idx] index + a memoized per-cluster
    result (see graph.py's _ensure_cluster_index()). This section
    proves the fix changed nothing observable except speed: same
    results as the old brute-force scan, for every real cluster_id in
    a real trained model, plus the specific behaviors the fix
    introduces (index built lazily, invalidated on rebuild).
    """
    section("BridgeMatrix.cluster_axis() indexed lookup (was O(triples) per call)")

    m = MSEGraphLanguageModel(vocab_size=400)
    m.train(CORPUS)
    b = m.bridges

    def brute_force_cluster_axis(cid):
        members = [(s, t, br) for s, t, br, c in
                   zip(b.source, b.target, b.bridge, b.cluster_id) if c == cid]
        if len(members) < 2:
            return None, members
        s0, t0, _b0 = members[0]
        if all(t == t0 for _, t, _ in members):
            return "bridge", members
        if all(br == members[0][2] for _, _, br in members):
            return "target", members
        return None, members

    check("cluster_axis() is already populated right after train() -- "
          "open_ctm's own build() (important_member_tokens(), auto-run by "
          "every train()/train_incremental()) queries every clustered "
          "triple's cluster_axis() internally, which is exactly the call "
          "pattern the O(triples^2) bug came from",
          b._cluster_members is not None, b._cluster_members)

    real_cluster_ids = sorted(set(c for c in b.cluster_id if c))
    check("this model actually has clustered triples to test against",
          len(real_cluster_ids) > 0, len(real_cluster_ids))

    mismatches = 0
    for cid in real_cluster_ids:
        fast = b.cluster_axis(cid)
        slow = brute_force_cluster_axis(cid)
        if fast != slow:
            mismatches += 1
    check(f"indexed cluster_axis() matches the old brute-force scan exactly "
          f"for every real cluster_id ({len(real_cluster_ids)} checked)",
          mismatches == 0, mismatches)

    check("cluster_id=0 (never a real cluster) returns (None, []), matching "
          "interpret.py's documented \"unknown cluster_id\" contract",
          b.cluster_axis(0) == (None, []), b.cluster_axis(0))
    check("a real cluster_id is memoized after its first lookup",
          real_cluster_ids[0] in b._cluster_axis_cache,
          b._cluster_axis_cache.get(real_cluster_ids[0]))

    # Laziness itself is only observable one level down, on a BridgeMatrix
    # used directly (nothing has queried cluster_axis() yet) -- the full
    # model pipeline above always triggers it internally via open_ctm.
    from graph import BridgeMatrix
    from tokenizer import BPETokenizer
    tok = BPETokenizer(vocab_size=400)
    tok.train(CORPUS)
    from tokenizer import split_sentences
    seqs = [tok.encode_for_training(s) for s in split_sentences(CORPUS)]
    bm = BridgeMatrix()
    bm.build(seqs, tok.vocab_size_actual)
    check("a BridgeMatrix used directly (no IVM/model wrapping it) starts "
          "with its cluster index unbuilt -- genuinely lazy, not eager",
          bm._cluster_members is None, bm._cluster_members)
    some_cid = next(c for c in bm.cluster_id if c)
    bm.cluster_axis(some_cid)
    check("first cluster_axis() call on it builds the index",
          bm._cluster_members is not None)
    bm.build(seqs, tok.vocab_size_actual)  # rebuild -- e.g. retrained in place
    check("calling build() again invalidates the index/cache -- no stale "
          "answers left over from the previous build",
          bm._cluster_members is None, bm._cluster_members)

    # train_incremental() constructs a brand-new BridgeMatrix for the
    # merged graph (see model.py's _merge_graphs) -- its cluster index
    # should reflect ONLY the merged clustering, never anything left
    # over from the pre-merge BridgeMatrix instance it replaced.
    m2 = MSEGraphLanguageModel(vocab_size=400)
    m2.train(CORPUS)
    pre_merge_bridges = m2.bridges
    m2.train_incremental("the pig sat on the log. the pig sat on the rug.")
    check("train_incremental() replaces bridges with a genuinely new object",
          m2.bridges is not pre_merge_bridges)
    post_merge_ids = set(c for c in m2.bridges.cluster_id if c)
    indexed_ids = set(m2.bridges._cluster_members.keys()) if m2.bridges._cluster_members else set()
    check("the merged model's cluster index (already populated by its own "
          "open_ctm rebuild) covers exactly the post-merge cluster ids, "
          "nothing stale from before the merge",
          indexed_ids == post_merge_ids, (indexed_ids, post_merge_ids))


if __name__ == "__main__":
    main()
    test_prompt_seeding_and_mode_boundaries()
    test_importance_vote_matrix()
    test_train_py_cli_path_matches_model_api()
    test_bigram_witness_vote_and_vocab_candidates()
    test_prev_current_co_occurrence_vote()
    test_triple_witness_vote()
    test_sparse_token_score_cache()
    test_relationship_matrix_dedup()
    test_cluster_axis_indexed_lookup()
    print(f"\n{PASS} passed, {FAIL} failed (grand total)")
    if FAIL: sys.exit(1)

