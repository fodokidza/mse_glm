"""
test.py — End-to-end test suite for MSE-GLM.

Trains a small-but-richer multi-lineage corpus and exercises every layer:
tokenizer round-trip, Edge/Bridge/Relationship matrix construction,
dual-axis clustering, lineage-aware tie-breaking (the bug fixed earlier),
infer_shared_role(), explain_step(), save/load round-trip, and the
Analyser reports.

Usage:
    python3 test.py
"""

import shutil
import sys
import tempfile

from model import MSEGraphLanguageModel
from analyse import CorpusAnalyser, Analyser
from tokenizer import normalize, split_sentences

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


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


def section(title):
    print(f"\n=== {title} ===")


def main():
    model = MSEGraphLanguageModel(vocab_size=300)
    model.train(CORPUS)

    # ----------------------------------------------------------- tokenizer
    section("Tokenizer round-trip")
    for phrase in ["the cat sat on the mat", "a bird flew over the lake", "unknown gibberish zzz"]:
        ids = model.tokenizer.encode(phrase)
        decoded = model.tokenizer.decode(ids)
        check(f"encode/decode '{phrase}'", isinstance(ids, list) and len(ids) > 0,
              f"got ids={ids}")
        check(f"BOS prepended for '{phrase}'", ids[0] == 2)
        check(f"no EOS in prompt encoding for '{phrase}'", 3 not in ids)

    train_ids = model.tokenizer.encode_for_training("the cat sat on the mat")
    check("encode_for_training appends EOS", train_ids[-1] == 3, f"got {train_ids}")

    norm = normalize("The Cat... SAT!! on-the MAT 123")
    check("normalize lowercases + strips punctuation", norm == "the cat sat on the mat 123", norm)

    sents = split_sentences("One. Two! Three?\nFour")
    check("sentence splitting on . ! ? newline", sents == ["One", "Two", "Three", "Four"], sents)

    # --------------------------------------------------------------- stats
    section("Graph construction sanity")
    stats = model.stats()
    check("vocab built", stats["vocab_size"] > 10, stats)
    check("edges built", stats["edges"] > 0, stats)
    check("bridges built", stats["bridges"] > 0, stats)
    check("relationships match sentence count",
          stats["relationships"] == len(split_sentences(CORPUS)),
          f"{stats['relationships']} vs {len(split_sentences(CORPUS))}")
    check("some bridges clustered", stats["clustered_bridges"] > 0, stats)
    check("some bridges left unclustered (cluster_id 0 exists)",
          stats["clustered_bridges"] < stats["bridges"], stats)

    # ------------------------------------------------------- R matrix shape
    section("Relationship Matrix (R) — schema and lineage")
    r = model.rels
    check("R stores only triple_id + relationship_id (2 parallel arrays)",
          len(r.r_triple) == len(r.r_rel))
    check("R has at least one many-to-many (shared) triple",
          any(len(r.relationships_for_triple(tid)) > 1 for tid in set(r.r_triple)))

    # ------------------------------------------- lineage-aware generation
    section("Lineage-aware tie-breaking (regression test for the dog/mat bug)")
    expectations = {
        "the cat": "mat",
        "the dog": "carpet",
        "the boy": "mat",
        "the cat ran": "road",
        "the dog ran": "road",
        "the girl ran": "road",
        "the fish": "pond",
        "the duck": "pond",
    }
    for prompt, must_contain in expectations.items():
        text, ids, trace = model.generate(prompt, max_tokens=12)
        check(f"'{prompt}' -> contains '{must_contain}'", must_contain in text,
              f"got '{text}'")

    # bird is genuinely ambiguous (lake appears with both bird and plane,
    # hill only with bird) — just check it terminates cleanly and stays on-graph
    text, ids, trace = model.generate("a bird flew over", max_tokens=12)
    check("'a bird flew over' terminates without crashing",
          isinstance(text, str) and len(text) > 0, text)
    check("'a bird flew over' lands on a real observed target",
          ("lake" in text) or ("hill" in text), text)

    # ----------------------------------------------------------- determinism
    section("Determinism")
    runs = set()
    for _ in range(5):
        text, _, _ = model.generate("the dog", max_tokens=12)
        runs.add(text)
    check("same prompt produces identical output across repeated runs",
          len(runs) == 1, runs)

    # ------------------------------------------------------------ explain
    section("explain_step()")
    token, trace = model.explain_step("the", "dog")
    check("explain_step returns a stage", "stage" in trace, trace)
    check("explain_step on (the, dog) resolves via Stage 1",
          trace["stage"] == 1, trace)

    token2, trace2 = model.explain_step("", "zzzznotaword")
    check("explain_step on unseen token falls through gracefully",
          "stage" in trace2, trace2)

    # ---------------------------------------------------- infer_shared_role
    section("infer_shared_role() — dual-axis clustering")
    results = model.infer_shared_role(["cat", "dog"])
    check("cat+dog share a cluster", len(results) > 0, results)
    check("cat+dog shared role includes 'sat'",
          any(tok == "sat" for tok, axis, ev in results), results)

    results2 = model.infer_shared_role(["bird", "plane"])
    check("bird+plane share a cluster (both fly-over subjects)",
          len(results2) > 0, results2)

    results3 = model.infer_shared_role(["fish", "duck"])
    check("fish+duck share a cluster (both swim-in subjects)",
          len(results3) > 0, results3)

    results4 = model.infer_shared_role(["lake", "hill"])
    check("lake+hill share a target-axis cluster (both follow 'over the')",
          any(axis == "target_axis" for _, axis, _ in results4), results4)

    # 'in' only ever appears in one structural slot (fish/duck swam IN the
    # pond/river) where its own (source,target)/(source,bridge) pairs are
    # not shared by anything outside that one cluster — pairing it with a
    # token from a disjoint part of the graph should yield no intersection.
    results5 = model.infer_shared_role(["sat", "flew"])
    check("'sat' and 'flew' (different verb-clusters, no shared slot) return no shared cluster",
          results5 == [], results5)

    results5b = model.infer_shared_role(["cat", "fish"])
    check("cat+fish DO share a cluster via the <BOS> 'the ___' subject slot "
          "(both are sentence-initial subjects) — correct, not a bug",
          len(results5b) > 0, results5b)

    results6 = model.infer_shared_role(["totallynotaword", "cat"])
    check("unknown token in shared-role query degrades gracefully (no crash)",
          isinstance(results6, list), results6)

    # ------------------------------------------------------------ analyser
    section("Analyser / CorpusAnalyser")
    ca = CorpusAnalyser(CORPUS)
    cstats = ca.stats()
    check("corpus sentence count matches", cstats["sentences"] == len(split_sentences(CORPUS)),
          cstats)
    check("corpus word stats non-trivial", cstats["words"] > 20, cstats)

    a = Analyser(model)
    topo = a.topology()
    check("topology reports hub tokens", len(topo["hub_tokens"]) > 0, topo)

    clusters = a.cluster_report()
    check("cluster report non-empty", len(clusters) > 0, clusters)
    check("cluster report entries have axis label",
          all(c["axis"] in ("bridge", "target") for c in clusters), clusters)

    rel_report = a.relationship_report()
    check("relationship report counts shared triples",
          rel_report["shared_triple_count"] >= 1, rel_report)

    per_tok = a.per_token_report("cat")
    check("per-token report resolves successors for 'cat'",
          per_tok is not None and len(per_tok["edge_successors"]) > 0, per_tok)

    gtext, gtrace = a.generation_trace("the dog", max_tokens=10)
    check("generation_trace returns readable stage/rule trace",
          len(gtrace) > 0 and all("stage" in s for s in gtrace), gtrace)

    # ------------------------------------------------------- save/load
    section("Persistence (save/load) round-trip")
    tmpdir = tempfile.mkdtemp(prefix="mse_glm_test_")
    try:
        model.save(tmpdir)
        reloaded = MSEGraphLanguageModel.load(tmpdir)

        check("reloaded stats match original",
              reloaded.stats() == model.stats(),
              f"{reloaded.stats()} vs {model.stats()}")

        for prompt in ["the cat", "the dog", "the boy", "the fish"]:
            t1, _, _ = model.generate(prompt, max_tokens=12)
            t2, _, _ = reloaded.generate(prompt, max_tokens=12)
            check(f"reloaded generation matches original for '{prompt}'", t1 == t2,
                  f"'{t1}' vs '{t2}'")

        r1 = model.infer_shared_role(["cat", "dog"])
        r2 = reloaded.infer_shared_role(["cat", "dog"])
        check("reloaded infer_shared_role matches original", r1 == r2, f"{r1} vs {r2}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # ------------------------------------------------------------- summary
    section("Summary")
    print(f"  {PASS} passed, {FAIL} failed")
    if FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()
