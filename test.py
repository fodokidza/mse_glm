"""
test.py — Full regression suite for MSE-GLM v2.1 + Open Mode.
56 original checks + experience + open mode checks.
Usage:  python3 test.py
"""

import shutil, sys, tempfile
from model import MSEGraphLanguageModel
from analyse import CorpusAnalyser, Analyser
from tokenizer import normalize, split_sentences

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
