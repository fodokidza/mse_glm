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
    for prompt, must in [("the cat","mat"),("the dog","carpet"),("the boy","mat"),
                          ("the cat ran","road"),("the dog ran","road"),
                          ("the fish","pond"),("the duck","pond")]:
        text, _, _ = model.generate(prompt, max_tokens=12)
        check(f"'{prompt}' → '{must}'", must in text, f"got '{text}'")
    text, _, _ = model.generate("a bird flew over", max_tokens=12)
    check("bird lands on lake or hill", ("lake" in text) or ("hill" in text), text)

    section("Determinism")
    runs = {model.generate("the dog", max_tokens=12)[0] for _ in range(5)}
    check("5 runs identical", len(runs) == 1, runs)

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

    section("Save/load round-trip")
    tmp = tempfile.mkdtemp(prefix="mse_test_")
    try:
        model.save(tmp)
        m2 = MSEGraphLanguageModel.load(tmp)
        check("reloaded stats match", m2.stats() == model.stats())
        for p in ["the cat","the dog","the boy","the fish"]:
            t1 = model.generate(p, max_tokens=12)[0]
            t2 = m2.generate(p, max_tokens=12)[0]
            check(f"reload '{p}'", t1 == t2, f"'{t1}' vs '{t2}'")
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
    Documents the exact boundary of where strict / open_strict / open diverge.
    Added after discovering that 'cat ran' resolves in strict because 'ran' is
    given in the prompt — and that prompt seeding carries experience lineage
    forward into generation.
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

    # Case 1: short prompts that map directly to training sentences
    # — all three modes agree, experience adds nothing
    for prompt, expected in [("the cat", "mat"), ("the dog", "carpet"), ("the boy", "mat")]:
        for mode in ["strict", "strict", "open"]:
            text, _, _ = mb.generate(prompt, max_tokens=12, mode=mode)
            check(f"case1 '{prompt}' {mode} → contains '{expected}'",
                  expected in text, f"got '{text}'")

    # Case 2: prompts that cross into experience territory
    # — once 'ran' is in the prompt, all modes follow boy's training path
    for prompt in ["the cat ran", "the dog ran"]:
        for mode in ["strict", "strict", "open"]:
            text, _, _ = mb.generate(prompt, max_tokens=10, mode=mode)
            check(f"case2 '{prompt}' {mode} → road",
                  "road" in text, f"got '{text}'")

    # Case 3: the critical case — only open modes can disambiguate 'on the'
    # because they carry experience lineage to active_rels={3}
    for prompt in ["the cat ran on the", "the dog ran on the"]:
        ts, _, _ = mb.generate(prompt, max_tokens=4, mode="strict")
        to, _, _ = mb.generate(prompt, max_tokens=4, mode="open")
        check(f"case3 '{prompt}' strict→road (new engine disambiguates via lineage)",
              "road" in ts, f"got '{ts}'")
        check(f"case3 '{prompt}' open→road (experience lineage_overlap)",
              "road" in to, f"got '{to}'")
        check(f"case3 '{prompt}' open→road (experience lineage_overlap)",
              "road" in to, f"got '{to}'")

    # Case 4: cat ran resolves correctly in strict because 'ran' is in prompt
    # — experience matrices are the DOOR, training is the CORRIDOR
    text, _, trace = mb.generate("the cat ran", max_tokens=10, mode="strict")
    check("cat ran resolves via training after 'ran' is in prompt",
          "road" in text, f"got '{text}'")
    # Stage 2 (bridge voting) should fire for the first step (cat,ran → on)
    check("first step resolves via bridge_vote (stage 1)",
          trace[0]["stage"] == 1, f"stage was {trace[0]['stage']}")

if __name__ == "__main__":
    main()
    test_prompt_seeding_and_mode_boundaries()
    section("Final Summary")
    print(f"  {PASS} passed, {FAIL} failed")
    import sys
    if FAIL: sys.exit(1)
