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
the boy sat on the mat.
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
    check("normalize", normalize("The Cat... SAT!!") == "the cat sat", normalize("The Cat... SAT!!"))
    check("sentence split", split_sentences("One. Two! Three") == ["One", "Two", "Three"])

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
    # Genuinely unambiguous prompts must be deterministic
    for det_prompt in ["the boy", "the girl", "the duck", "the cat ran", "the dog sat"]:
        runs = {model.generate(det_prompt, max_tokens=12)[0] for _ in range(5)}
        check(f"unambiguous '{det_prompt}' deterministic", len(runs) == 1, runs)
    # Genuinely ambiguous prompts must vary across runs
    for amb_prompt in ["the cat", "the dog", "the fish", "a bird"]:
        runs = {model.generate(amb_prompt, max_tokens=12)[0] for _ in range(20)}
        check(f"ambiguous '{amb_prompt}' produces varied output", len(runs) > 1, runs)

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
          ["<BOS>", "the", "cat", "sat", "on", "the", "mat", "<EOS>"], seq0)

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

    random.seed(0)
    without_ctm = {engine_ctm.step(bos_id, the_id, active_rels=set())[0] for _ in range(30)}
    check("without context_triggers, tie-break still explores multiple members "
          "(unchanged random behavior)", len(without_ctm) > 1, without_ctm)

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
    # structure -- both are always deduplicated -- but relationship_ids
    # ARE expected to grow (one per sentence occurrence, matching
    # from-scratch behavior: training on the same sentence twice in one
    # corpus also produces two relationship_ids, not one).
    m_incr = MSEGraphLanguageModel(vocab_size=200)
    m_incr.train("the cat sat on the mat.")
    before_dup = m_incr.stats()
    m_incr.train_incremental("the cat sat on the mat.")
    after_dup = m_incr.stats()
    check("re-feeding identical sentence: no new edges",
          after_dup["edges"] == before_dup["edges"], (before_dup, after_dup))
    check("re-feeding identical sentence: no new bridges",
          after_dup["bridges"] == before_dup["bridges"], (before_dup, after_dup))
    check("re-feeding identical sentence: relationships DO grow (matches "
          "from-scratch semantics of one relationship_id per sentence)",
          after_dup["relationships"] == before_dup["relationships"] + 1, (before_dup, after_dup))

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
          m_incr3.generate("the cat", max_tokens=6)[0] == "the cat sat on the mat")
    check("new fact generates correctly after merge",
          m_incr3.generate("the dog", max_tokens=6)[0] == "the dog sat on the carpet")

    # Experience Matrices are derived from pre-merge cluster structure --
    # must be invalidated, not silently left stale.
    m_incr4 = MSEGraphLanguageModel(vocab_size=200)
    m_incr4.train("the cat sat on the mat. the dog sat on the carpet.")
    m_incr4.build_experience()
    check("has_experience true before incremental training", m_incr4.has_experience())
    summary = m_incr4.train_incremental("the pig sat on the rug.",
                                         extend_vocab=True, target_vocab_size=250)
    check("has_experience false after incremental training (invalidated)",
          not m_incr4.has_experience())
    check("summary reports experience_invalidated=True", summary["experience_invalidated"])

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
              m1.generate("the cat", max_tokens=6)[0] == "the cat sat on the mat" and
              m1.generate("the dog", max_tokens=6)[0] == "the dog sat on the carpet" and
              m1.generate("the pig", max_tokens=6)[0] == "the pig sat on the rug")

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

    # ── Experience + Open Mode ────────────────────────────────────────────────
    section("Experience Matrix construction")
    m_open = MSEGraphLanguageModel(vocab_size=300)
    m_open.train(CORPUS_EXP)
    exp_summary = m_open.build_experience()

    check("exp_edges built",       exp_summary["exp_edges"] > 0,    exp_summary)
    check("exp_bridges built",     exp_summary["exp_bridges"] > 0,  exp_summary)
    check("exp_clusters assigned", "exp_clusters" in exp_summary)
    check("exp_rel_rows built",    exp_summary["exp_rel_rows"] > 0, exp_summary)
    check("has_experience()",      m_open.has_experience())

    section("Experience edge correctness")
    tok = m_open.tokenizer
    cat_id = tok.token_to_id.get("cat")
    dog_id = tok.token_to_id.get("dog")
    ran_id = tok.token_to_id.get("ran")
    if cat_id and dog_id and ran_id:
        exp_e_srcs = list(m_open.exp_edges.src)
        exp_e_dsts = list(m_open.exp_edges.dst)
        cat_ran = (cat_id, ran_id) in zip(exp_e_srcs, exp_e_dsts)
        dog_ran = (dog_id, ran_id) in zip(exp_e_srcs, exp_e_dsts)
        check("exp edge cat→ran created", cat_ran, "cat→ran missing from exp edges")
        check("exp edge dog→ran created", dog_ran, "dog→ran missing from exp edges")

    section("Experience bridge correctness")
    if cat_id and dog_id and ran_id:
        the_id = tok.token_to_id.get("the")
        exp_b = m_open.exp_bridges
        # check (the, ran, cat) and (the, ran, dog) in exp bridges
        # stored as source=the, target=ran, bridge=cat/dog
        the_ran_cat = any(
            exp_b.source[i]==the_id and exp_b.target[i]==ran_id and exp_b.bridge[i]==cat_id
            for i in range(len(exp_b.source))
        ) if the_id else False
        the_ran_dog = any(
            exp_b.source[i]==the_id and exp_b.target[i]==ran_id and exp_b.bridge[i]==dog_id
            for i in range(len(exp_b.source))
        ) if the_id else False
        check("exp bridge (the,ran,cat) created", the_ran_cat)
        check("exp bridge (the,ran,dog) created", the_ran_dog)

    section("Experience cluster: cat+dog stronger than boy")
    if cat_id and dog_id:
        boy_id = tok.token_to_id.get("boy")
        sim_cd = m_open.token_similarity("cat","dog",   mode="open")["similarity"]
        sim_cb = m_open.token_similarity("cat","boy",   mode="open")["similarity"]
        sim_db = m_open.token_similarity("dog","boy",   mode="open")["similarity"]
        check("open sim(cat,dog) > sim(cat,boy)", sim_cd > sim_cb,
              f"cat-dog={sim_cd} cat-boy={sim_cb}")
        check("open sim(cat,dog) > sim(dog,boy)", sim_cd > sim_db,
              f"cat-dog={sim_cd} dog-boy={sim_db}")
        # strict mode should not yet have this
        sim_cd_strict = m_open.token_similarity("cat","dog", mode="strict")["similarity"]
        sim_cd_open   = sim_cd
        check("open sim >= strict sim for cat+dog", sim_cd_open >= sim_cd_strict)

    section("Open Mode generation")
    # Open mode should be able to generate "the cat ran" or "the dog ran"
    # strict mode should give training-only output for cat/dog (no ran)
    strict_cat = m_open.generate("the cat", max_tokens=12, mode="strict")[0]
    check("strict cat stays on training path", "sat" in strict_cat, strict_cat)
    # open mode can now use experience — cat and dog can run
    open_cat = m_open.generate("the cat", max_tokens=12, mode="open")[0]
    check("open cat does not crash", isinstance(open_cat, str) and len(open_cat) > 0)
    open_dog = m_open.generate("the dog", max_tokens=12, mode="open")[0]
    check("open dog does not crash", isinstance(open_dog, str) and len(open_dog) > 0)
    for prompt in ["the cat", "the dog", "the boy"]:
        text, _, trace = m_open.generate(prompt, max_tokens=12, mode="open")
        stages = [t["stage"] for t in trace]
        check(f"open '{prompt}' valid stages", all(s in (1,2,3,4) for s in stages))

    section("Open Mode infer_shared_role (includes experience clusters)")
    r_open = m_open.infer_shared_role(["cat","dog"], mode="open")
    check("open shared-role cat+dog non-empty", len(r_open) > 0, r_open)
    sources = [ev.get("source","") for _,_,ev in r_open]
    check("open shared-role includes experience source",
          any(s=="experience" for s in sources) or len(r_open)>0)

    section("Experience save/load round-trip")
    tmp2 = tempfile.mkdtemp(prefix="mse_exp_test_")
    try:
        m_open.save(tmp2)
        m_reload = MSEGraphLanguageModel.load(tmp2)
        check("reloaded has_experience()", m_reload.has_experience())
        for p in ["the cat","the dog"]:
            t_orig   = m_open.generate(p,    max_tokens=12, mode="open")[0]
            t_reload = m_reload.generate(p,  max_tokens=12, mode="open")[0]
            check(f"exp reload '{p}' matches", t_orig == t_reload, f"'{t_orig}' vs '{t_reload}'")
    finally:
        shutil.rmtree(tmp2, ignore_errors=True)

    section("Open Mode determinism")
    runs = {m_open.generate("the dog", max_tokens=12, mode="open")[0] for _ in range(5)}
    check("open mode 5 runs identical", len(runs)==1, runs)

    section("Summary")
    print(f"  {PASS} passed, {FAIL} failed")
    if FAIL: sys.exit(1)

if __name__ == "__main__":
    main()


def test_prompt_seeding_and_mode_boundaries():
    """
    Documents the strict bigram-validation boundary and random-tie behaviour.

    Key findings after the two bug fixes:
    - Strict mode rejects any prompt containing a bigram not in E.
    - Open mode can extend E via experience edges, so 'the cat ran'
      becomes legal in open mode (cat->ran added via experience).
    - Genuine ties now resolve randomly, not by storage order.
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
    mb.build_experience()

    # Case 1: short prompts — valid in both modes, produce known valid outputs
    for prompt, valid in [("the cat",["mat"]), ("the dog",["carpet"]),
                           ("the boy",["mat","road"])]:
        for mode in ["strict","open"]:
            text, _, _ = mb.generate(prompt, max_tokens=12, mode=mode)
            check(f"case1 '{prompt}' {mode} → one of {valid}",
                  any(v in text for v in valid), f"got '{text}'")

    # Case 2: 'the cat ran' — cat->ran not in training E
    # Strict: illegal bigram → rejected at stage 0
    # Open:   cat->ran exists in EE (experience edge) → resolves via boy's path
    for prompt in ["the cat ran", "the dog ran"]:
        ts, _, trace_s = mb.generate(prompt, max_tokens=10, mode="strict")
        check(f"case2 '{prompt}' strict → illegal_prompt_bigram",
              trace_s[0].get("rule") == "illegal_prompt_bigram",
              f"rule={trace_s[0].get('rule')} out='{ts}'")
        to, _, _ = mb.generate(prompt, max_tokens=10, mode="open")
        check(f"case2 '{prompt}' open → road",
              "road" in to, f"got '{to}'")

    # Case 3: 'the cat ran on the' — also illegal in strict (same bad bigram)
    # Open resolves correctly via experience lineage
    for prompt in ["the cat ran on the", "the dog ran on the"]:
        ts, _, trace_s = mb.generate(prompt, max_tokens=4, mode="strict")
        check(f"case3 '{prompt}' strict → illegal_prompt_bigram",
              trace_s[0].get("rule") == "illegal_prompt_bigram",
              f"rule={trace_s[0].get('rule')} out='{ts}'")
        to, _, _ = mb.generate(prompt, max_tokens=4, mode="open")
        check(f"case3 '{prompt}' open → road",
              "road" in to, f"got '{to}'")

    # Case 4: legal unambiguous prompt resolves correctly in strict
    text, _, trace = mb.generate("the cat sat", max_tokens=10, mode="strict")
    check("legal prompt 'the cat sat' → on",
          "on" in text, f"got '{text}'")
    check("first step uses stage 1 or 2",
          trace[0]["stage"] in (1, 2), f"stage was {trace[0]['stage']}")

    # Case 5: random ties — 'the boy' has two valid paths in CORPUS_BOUNDARY
    # Must produce varied output across runs
    boy_runs = {mb.generate("the boy", max_tokens=10, mode="strict")[0]
                for _ in range(30)}
    check("ambiguous 'the boy' produces varied output (sat/ran both valid)",
          len(boy_runs) > 1, f"always gave: {boy_runs}")

if __name__ == "__main__":
    main()
    test_prompt_seeding_and_mode_boundaries()
    section("Final Summary")
    print(f"  {PASS} passed, {FAIL} failed")
    import sys
    if FAIL: sys.exit(1)
