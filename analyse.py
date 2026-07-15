"""
analyse.py — Analysis layer for MSE-GLM.

Two library classes:
    CorpusAnalyser  — raw-text statistics, no trained model required.
    Analyser        — graph-level statistics on a trained MSEGraphLanguageModel:
                       topology, dual-axis cluster reports, Relationship Matrix
                       reports, per-token reports, token similarity, and full
                       step-by-step generation traces.

Plus a CLI (`python3 analyse.py ...`) so the model can be interrogated freely
from the command line without writing one-off scripts — stats, topology,
clusters, a specific cluster, a specific token, similarity between two
tokens, a generation trace, or a single combined report, optionally
exported to JSON.
"""

import argparse
import json
import sys
from collections import Counter

from model import MSEGraphLanguageModel
from tokenizer import normalize, split_sentences


# =============================================================================
# Library
# =============================================================================

class CorpusAnalyser:
    """Statistics over raw corpus text — no trained model required."""

    def __init__(self, corpus: str):
        self.corpus = corpus

    def stats(self, top_n: int = 10) -> dict:
        sentences = split_sentences(self.corpus)
        words = []
        for s in sentences:
            words.extend(normalize(s).split(" "))
        words = [w for w in words if w]
        return {
            "sentences": len(sentences),
            "words": len(words),
            "unique_words": len(set(words)),
            "avg_sentence_len": round((len(words) / len(sentences)), 2) if sentences else 0,
            "top_words": Counter(words).most_common(top_n),
        }


class Analyser:
    """Graph-level statistics on a trained MSEGraphLanguageModel."""

    def __init__(self, model: MSEGraphLanguageModel):
        self.model = model

    # ------------------------------------------------------------- topology
    def topology(self, top_n: int = 10) -> dict:
        b = self.model.bridges
        tok = self.model.tokenizer
        out_degree = Counter(b.source)
        hubs = out_degree.most_common(top_n)
        vocab_ids = range(tok.vocab_size_actual)
        dead_ends = [t for t in vocab_ids if t not in out_degree]
        return {
            "hub_tokens": [(tok.id_to_token.get(t, t), c) for t, c in hubs],
            "dead_end_count": len(dead_ends),
            "dead_end_tokens": [tok.id_to_token.get(t, t) for t in dead_ends[:top_n]],
        }

    # -------------------------------------------------------------- clusters
    def cluster_report(self, top_n: int = 10, axis: str = None) -> list:
        """
        List dual-axis clusters (cluster_id != 0). `axis` optionally filters
        to 'bridge' or 'target' clusters only.
        """
        b = self.model.bridges
        tok = self.model.tokenizer
        groups = {}
        for s, t, br, c in zip(b.source, b.target, b.bridge, b.cluster_id):
            if c == 0:
                continue
            groups.setdefault(c, []).append((s, t, br))

        report = []
        for cid, members in groups.items():
            cluster_axis, _ = b.cluster_axis(cid)
            if axis and cluster_axis != axis:
                continue
            s0, t0, _ = members[0]
            if cluster_axis == "bridge":
                varying = sorted(set(tok.id_to_token.get(br, br) for _, _, br in members))
                fixed = f"{tok.id_to_token.get(s0, s0)} -> ___ -> {tok.id_to_token.get(t0, t0)}"
            elif cluster_axis == "target":
                varying = sorted(set(tok.id_to_token.get(t, t) for _, t, _ in members))
                fixed = f"{tok.id_to_token.get(s0, s0)} -> {tok.id_to_token.get(members[0][2], members[0][2])} -> ___"
            else:
                varying, fixed = [], "?"
            report.append({
                "cluster_id": cid, "axis": cluster_axis, "slot": fixed,
                "members": varying, "size": len(varying),
            })
        report.sort(key=lambda r: r["size"], reverse=True)
        return report[:top_n]

    def cluster_detail(self, cluster_id: int) -> dict:
        """Full detail for a single cluster_id, including raw triples."""
        b = self.model.bridges
        tok = self.model.tokenizer
        axis, members = b.cluster_axis(cluster_id)
        decoded = [
            (tok.id_to_token.get(s, s), tok.id_to_token.get(t, t), tok.id_to_token.get(br, br))
            for s, t, br in members
        ]
        return {"cluster_id": cluster_id, "axis": axis, "triples": decoded}

    # --------------------------------------------------------- relationships
    def relationship_report(self) -> dict:
        r = self.model.rels
        shared = [tid for tid in set(r.r_triple) if len(r.relationships_for_triple(tid)) > 1]
        return {
            "total_relationships": r._n_rels,
            "total_rows": len(r.r_triple),
            "unique_triples_referenced": len(set(r.r_triple)),
            "shared_triple_count": len(shared),
        }

    def relationship_detail(self, relationship_id: int) -> dict:
        """Every triple belonging to a single training sequence (sentence)."""
        b = self.model.bridges
        tok = self.model.tokenizer
        triple_ids = self.model.rels.triples_for_relationship(relationship_id)
        triples = []
        for tid in triple_ids:
            s, t, br = b.source[tid], b.target[tid], b.bridge[tid]
            triples.append((tok.id_to_token.get(s, s), tok.id_to_token.get(t, t),
                             tok.id_to_token.get(br, br)))
        return {"relationship_id": relationship_id, "triples": triples}

    # ---------------------------------------------------------- per-token
    def per_token_report(self, word: str) -> dict:
        tok = self.model.tokenizer
        enc = [t for t in tok.encode(word) if t != 2]
        if not enc:
            return None
        token = enc[-1]
        out_edges = self.model.edges.successors(token)
        triples = self.model.bridges.triples_from_source(token)
        clusters = self.model.bridges.t_index.get(token, [])
        return {
            "token": tok.id_to_token.get(token, token),
            "edge_successors": [tok.id_to_token.get(t, t) for t in out_edges],
            "bridge_triples_as_source": len(triples),
            "cluster_memberships": clusters,
        }

    def token_similarity(self, word_a: str, word_b: str) -> dict:
        """
        Relatedness between two tokens = |T_index[a] ∩ T_index[b]|, per SDD
        v2.1 §10. Returns the shared clusters and a plain similarity count.
        """
        tok = self.model.tokenizer
        t_index = self.model.bridges.t_index

        def resolve(word):
            enc = [t for t in tok.encode(word) if t != 2]
            return enc[-1] if enc else None

        ta, tb = resolve(word_a), resolve(word_b)
        if ta is None or tb is None:
            return {"word_a": word_a, "word_b": word_b, "similarity": 0, "shared_clusters": []}
        sa, sb = set(t_index.get(ta, [])), set(t_index.get(tb, []))
        shared = sorted(sa & sb)
        return {
            "word_a": word_a, "word_b": word_b,
            "similarity": len(shared), "shared_clusters": shared,
        }

    # ------------------------------------------------------- interpretation
    def cluster_interpretation(self, cluster_id: int, top_n: int = 5, mode: str = "strict") -> dict:
        """
        Propose a human-readable label for a cluster (e.g. {cat, dog, pig}
        -> "animal"), derived purely from structure already in the trained
        matrices. Returns None if the cluster_id doesn't exist or nothing
        was found. `coverage` is a plain fraction, not a confidence score —
        see interpret.py's module docstring before presenting this to a
        user as a certainty figure.
        """
        return self.model.interpret_cluster(cluster_id, top_n=top_n, mode=mode)

    def interpretation_report(self, min_coverage: float = 0.5, max_per_cluster: int = 3,
                               mode: str = "strict") -> list:
        """Every candidate clearing min_coverage per cluster (not just one)."""
        return self.model.interpret_all_clusters(
            min_coverage=min_coverage, max_per_cluster=max_per_cluster, mode=mode)

    def interpreter_matrix(self, min_coverage: float = 0.5, min_signals: int = 2,
                            max_per_cluster: int = None, mode: str = "strict") -> list:
        """
        The filtered Cluster Interpreter Matrix -- coverage AND a minimum
        number of corroborating evidence_mask entries required. A cluster
        can carry multiple qualifying labels at once (e.g. both "animal"
        and "pet"); this is what should actually get persisted as "the"
        interpreter matrix. interpretation_report()/interpret_cluster()
        are the wider, unfiltered views used to sanity-check candidates
        by eye.
        """
        return self.model.build_interpreter_matrix(
            min_coverage=min_coverage, min_signals=min_signals,
            max_per_cluster=max_per_cluster, mode=mode)

    def zero_cluster_groups(self, min_group_size: int = 2, mode: str = "strict") -> list:
        """
        Mine cluster_id==0 for the "third axis" the standard dual-axis
        rule never implements (fix bridge+target, source varies). Finds
        groups the regular cluster_report()/interpreter_matrix() never
        see at all -- see model.discover_zero_cluster_groups for caveats
        (bigger candidate space, more prone to coincidental groupings).
        """
        return self.model.discover_zero_cluster_groups(
            min_group_size=min_group_size, mode=mode)

    # -------------------------------------------------------------- traces
    def generation_trace(self, prompt: str, max_tokens: int = 20):
        text, ids, trace = self.model.generate(prompt, max_tokens=max_tokens)
        tok = self.model.tokenizer
        readable = []
        for step in trace:
            chosen = step["chosen"]
            readable.append({
                "stage": step["stage"], "rule": step.get("rule"),
                "chosen_token": tok.id_to_token.get(chosen, chosen),
                "active_rels": sorted(step.get("active_rels", [])) if step.get("active_rels") else [],
            })
        return text, readable

    # ----------------------------------------------------------- full report
    def full_report(self, top_n: int = 10) -> dict:
        return {
            "stats": self.model.stats(),
            "topology": self.topology(top_n=top_n),
            "clusters": self.cluster_report(top_n=top_n),
            "relationships": self.relationship_report(),
        }


# =============================================================================
# Plain-text rendering helpers (no external deps)
# =============================================================================

def _print_kv(d: dict):
    width = max((len(str(k)) for k in d), default=0)
    for k, v in d.items():
        print(f"  {str(k).ljust(width)} : {v}")


def _print_table(rows, headers):
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    line = "  " + "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(line)
    print("  " + "-" * (len(line) - 2))
    for row in rows:
        print("  " + "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)))


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyse an MSE-GLM corpus or trained model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", help="Path to a saved model folder (required for most commands)")
    parser.add_argument("--json", help="Write the result as JSON to this path instead of printing")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("corpus", help="Raw-text statistics — no trained model required")
    p.add_argument("--text", help="Inline corpus text")
    p.add_argument("--file", help="Path to a corpus text file")
    p.add_argument("--top", type=int, default=10)

    sub.add_parser("stats", help="Model-level structure counts (vocab/edges/bridges/clusters/relationships)")

    p = sub.add_parser("topology", help="Hub tokens and dead ends")
    p.add_argument("--top", type=int, default=10)

    p = sub.add_parser("clusters", help="Dual-axis cluster report")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--axis", choices=["bridge", "target"], default=None)

    p = sub.add_parser("cluster", help="Full detail for one cluster_id")
    p.add_argument("cluster_id", type=int)

    sub.add_parser("relationships", help="Relationship Matrix summary")

    p = sub.add_parser("relationship", help="Full detail for one relationship_id (training sentence)")
    p.add_argument("relationship_id", type=int)

    p = sub.add_parser("token", help="Per-token report: successors, bridge triples, clusters")
    p.add_argument("word")

    p = sub.add_parser("similarity", help="Cluster-overlap similarity between two tokens")
    p.add_argument("word_a")
    p.add_argument("word_b")

    p = sub.add_parser("shared", help="infer_shared_role() across two or more tokens")
    p.add_argument("words", nargs="+")

    p = sub.add_parser("interpret", help="Propose a human-readable label for one cluster_id")
    p.add_argument("cluster_id", type=int)
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--mode", choices=["strict", "open"], default="strict")

    p = sub.add_parser("interpretations", help="Every qualifying label per cluster (unfiltered)")
    p.add_argument("--min-coverage", type=float, default=0.5)
    p.add_argument("--max-per-cluster", type=int, default=3)
    p.add_argument("--mode", choices=["strict", "open"], default="strict")

    p = sub.add_parser("interpreter-matrix",
                        help="Filtered CI Matrix (coverage + corroborating evidence required); "
                             "a cluster may carry multiple labels. "
                             "Pipe with --json > interpreter_matrix.json to persist it.")
    p.add_argument("--min-coverage", type=float, default=0.5)
    p.add_argument("--min-signals", type=int, default=2)
    p.add_argument("--max-per-cluster", type=int, default=None)
    p.add_argument("--mode", choices=["strict", "open"], default="strict")

    p = sub.add_parser("zero-cluster",
                        help="Mine cluster_id==0 for groups the standard dual-axis rule "
                             "never assigns a cluster_id to (fix bridge+target, source varies)")
    p.add_argument("--min-group-size", type=int, default=2)
    p.add_argument("--mode", choices=["strict", "open"], default="strict")

    p = sub.add_parser("trace", help="Step-by-step generation trace for a prompt")
    p.add_argument("prompt")
    p.add_argument("--max-tokens", type=int, default=20)

    p = sub.add_parser("report", help="Combined stats + topology + clusters + relationships")
    p.add_argument("--top", type=int, default=10)

    args = parser.parse_args()

    if args.command == "corpus":
        if not args.text and not args.file:
            print("Provide --text or --file", file=sys.stderr)
            sys.exit(1)
        text = args.text or open(args.file, "r", encoding="utf-8", errors="ignore").read()
        result = CorpusAnalyser(text).stats(top_n=args.top)
        _emit(result, args.json, lambda r: (
            _print_kv({k: v for k, v in r.items() if k != "top_words"}),
            print("  top_words:"),
            _print_table(r["top_words"], ["word", "count"]),
        ))
        return

    if not args.model:
        print("This command requires --model <folder>", file=sys.stderr)
        sys.exit(1)

    model = MSEGraphLanguageModel.load(args.model)
    analyser = Analyser(model)

    if args.command == "stats":
        result = model.stats()
        _emit(result, args.json, _print_kv)

    elif args.command == "topology":
        result = analyser.topology(top_n=args.top)
        _emit(result, args.json, lambda r: (
            print("  hub tokens (highest out-degree):"),
            _print_table(r["hub_tokens"], ["token", "out_degree"]),
            print(f"  dead-end token count: {r['dead_end_count']}"),
        ))

    elif args.command == "clusters":
        result = analyser.cluster_report(top_n=args.top, axis=args.axis)
        _emit(result, args.json, lambda r: _print_table(
            [(c["cluster_id"], c["axis"], c["slot"], ", ".join(c["members"])) for c in r],
            ["cluster_id", "axis", "slot", "members"],
        ))

    elif args.command == "cluster":
        result = analyser.cluster_detail(args.cluster_id)
        _emit(result, args.json, lambda r: (
            print(f"  cluster_id: {r['cluster_id']}   axis: {r['axis']}"),
            _print_table(r["triples"], ["source", "target", "bridge"]),
        ))

    elif args.command == "relationships":
        result = analyser.relationship_report()
        _emit(result, args.json, _print_kv)

    elif args.command == "relationship":
        result = analyser.relationship_detail(args.relationship_id)
        _emit(result, args.json, lambda r: (
            print(f"  relationship_id: {r['relationship_id']}"),
            _print_table(r["triples"], ["source", "target", "bridge"]),
        ))

    elif args.command == "token":
        result = analyser.per_token_report(args.word)
        if result is None:
            print(f"'{args.word}' not found in vocabulary", file=sys.stderr)
            sys.exit(1)
        _emit(result, args.json, _print_kv)

    elif args.command == "similarity":
        result = analyser.token_similarity(args.word_a, args.word_b)
        _emit(result, args.json, _print_kv)

    elif args.command == "shared":
        results = model.infer_shared_role(args.words)
        result = [{"predicted": tok, "axis": axis, **ev} for tok, axis, ev in results]
        if not result:
            print("  no shared cluster found across those tokens")
        _emit(result, args.json, lambda r: _print_table(
            [(x["predicted"], x["axis"], x["cluster_id"], x["overlap"]) for x in r],
            ["predicted", "axis", "cluster_id", "overlap"],
        ) if r else None)

    elif args.command == "interpret":
        result = analyser.cluster_interpretation(args.cluster_id, top_n=args.top, mode=args.mode)
        if result is None:
            print(f"cluster_id {args.cluster_id} not found", file=sys.stderr)
            sys.exit(1)
        if not result["candidates"]:
            print(f"  cluster {args.cluster_id} ({result['axis']} axis, "
                  f"members: {', '.join(result['members'])}) — no interpreter found "
                  f"(corpus has no matching categorical statement for these members)")
        else:
            _emit(result, args.json, lambda r: (
                print(f"  cluster_id: {r['cluster_id']}   axis: {r['axis']}   "
                      f"members: {', '.join(r['members'])}"),
                _print_table(
                    [(c["interpreter_token"], c["coverage"], ",".join(c["evidence_mask"]),
                      ", ".join(c["members_covered"])) for c in r["candidates"]],
                    ["interpreter", "coverage", "evidence_mask", "members_covered"],
                ),
            ))

    elif args.command == "interpretations":
        result = analyser.interpretation_report(min_coverage=args.min_coverage,
                                                 max_per_cluster=args.max_per_cluster,
                                                 mode=args.mode)
        if not result:
            print("  no clusters cleared the coverage threshold")
        _emit(result, args.json, lambda r: _print_table(
            [(c["cluster_id"], cand["interpreter_token"], cand["coverage"],
              ",".join(cand["evidence_mask"]), ", ".join(cand["members_covered"]))
             for c in r for cand in c["candidates"]],
            ["cluster_id", "interpreter", "coverage", "evidence_mask", "members_covered"],
        ) if r else None)

    elif args.command == "interpreter-matrix":
        result = analyser.interpreter_matrix(min_coverage=args.min_coverage,
                                              min_signals=args.min_signals,
                                              max_per_cluster=args.max_per_cluster,
                                              mode=args.mode)
        if not result:
            print("  no clusters cleared both the coverage and evidence thresholds")
        _emit(result, args.json, lambda r: _print_table(
            [(row["cluster_id"], row["interpreter_token"], row["coverage"],
              ",".join(row["evidence_mask"]), ", ".join(row["members_covered"])) for row in r],
            ["cluster_id", "interpreter", "coverage", "evidence_mask", "members_covered"],
        ) if r else None)

    elif args.command == "zero-cluster":
        result = analyser.zero_cluster_groups(min_group_size=args.min_group_size, mode=args.mode)
        if not result:
            print("  no groups found in the unclustered bucket at this min_group_size")
        _emit(result, args.json, lambda r: _print_table(
            [(row["interpreter_token"], row["member_count"], ",".join(row["evidence_mask"]),
              ", ".join(row["members"])) for row in r],
            ["interpreter", "member_count", "evidence_mask", "members"],
        ) if r else None)

    elif args.command == "trace":
        text, trace = analyser.generation_trace(args.prompt, max_tokens=args.max_tokens)
        result = {"output": text, "trace": trace}
        _emit(result, args.json, lambda r: (
            print(f"  output: {r['output']}"),
            _print_table(
                [(s["stage"], s["rule"], s["chosen_token"], s["active_rels"]) for s in r["trace"]],
                ["stage", "rule", "chosen_token", "active_rels"],
            ),
        ))

    elif args.command == "report":
        result = analyser.full_report(top_n=args.top)
        _emit(result, args.json, lambda r: (
            print("== stats =="), _print_kv(r["stats"]),
            print("\n== topology =="),
            _print_table(r["topology"]["hub_tokens"], ["token", "out_degree"]),
            print("\n== clusters =="),
            _print_table(
                [(c["cluster_id"], c["axis"], c["slot"], ", ".join(c["members"])) for c in r["clusters"]],
                ["cluster_id", "axis", "slot", "members"],
            ),
            print("\n== relationships =="), _print_kv(r["relationships"]),
        ))


def _emit(result, json_path, printer):
    if json_path:
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Wrote {json_path}")
    else:
        printer(result)


if __name__ == "__main__":
    main()
