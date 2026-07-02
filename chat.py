"""
chat.py — Interactive REPL for MSE-GLM (Strict + Open Mode).

Commands:
  <text>                    generate continuation (current mode)
  /mode strict              switch to Strict Mode (v2.1, training data only)
  /mode open                switch to Open Mode  (training + Experience Matrices)
  /explain <prev> | <curr>  explain a single inference step
  /shared <tok1> <tok2> … infer_shared_role() across token set
  /similarity <a> <b>       cluster-set similarity between two tokens
  /stats                    model stats (includes experience if loaded)
  /clusters                 top dual-axis cluster groups (training)
  /exp                      experience matrix summary
  /quit
"""

import argparse
from model import MSEGraphLanguageModel
from analyse import Analyser


def main():
    parser = argparse.ArgumentParser(description="Chat with a trained MSE-GLM model")
    parser.add_argument("--model",      required=True, help="Saved model folder")
    parser.add_argument("--max-tokens", type=int, default=30)
    parser.add_argument("--mode",       choices=["strict","open"], default="strict")
    args = parser.parse_args()

    print(f"\nLoading model from {args.model} …")
    model = MSEGraphLanguageModel.load(args.model)

    mode = args.mode
    if mode == "open" and not model.has_experience():
        print("  No experience matrices found. Building now…")
        import build_experience
        summary = build_experience.build(args.model, quiet=True)
        model.load_experience(args.model)
        print(f"  Experience built: {summary}")

    analyser = Analyser(model)
    print(f"  Stats: {model.stats()}")
    print(f"  Mode:  {mode.upper()}")
    print("\n  Commands: /mode strict|open  /explain  /shared  /similarity  /stats  /clusters  /exp  /quit\n")

    while True:
        try:
            line = input(f"[{mode}] you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue

        if line in ("/quit", "/exit"):
            break

        # ── /mode ─────────────────────────────────────────────────────────
        if line.startswith("/mode"):
            parts = line.split()
            if len(parts) < 2 or parts[1] not in ("strict", "open"):
                print("  Usage: /mode strict|open")
                continue
            mode = parts[1]
            if mode == "open" and not model.has_experience():
                print("  Building experience matrices…")
                import build_experience
                summary = build_experience.build(args.model, quiet=True)
                model.load_experience(args.model)
                print(f"  Done: {summary}")
            print(f"  Switched to {mode.upper()} mode.")
            continue

        # ── /stats ────────────────────────────────────────────────────────
        if line == "/stats":
            for k, v in model.stats().items():
                print(f"  {k}: {v}")
            continue

        # ── /exp ──────────────────────────────────────────────────────────
        if line == "/exp":
            if not model.has_experience():
                print("  No experience matrices. Run /mode open to build them.")
            else:
                from experience import ExperienceBuilder
                s = ExperienceBuilder().summary(model.exp_edges, model.exp_bridges, model.exp_rels)
                for k, v in s.items():
                    print(f"  {k}: {v}")
            continue

        # ── /clusters ─────────────────────────────────────────────────────
        if line == "/clusters":
            for c in analyser.cluster_report(top_n=10):
                print(f"  [{c['cluster_id']}] {c['axis']:6s}  {c['slot']:30s}  {', '.join(c['members'])}")
            continue

        # ── /similarity ───────────────────────────────────────────────────
        if line.startswith("/similarity "):
            parts = line.split()[1:]
            if len(parts) != 2:
                print("  Usage: /similarity <word_a> <word_b>")
                continue
            result = model.token_similarity(parts[0], parts[1], mode=mode)
            print(f"  similarity({parts[0]}, {parts[1]}) = {result['similarity']}"
                  f"  shared_clusters={result['shared_clusters']}")
            continue

        # ── /shared ───────────────────────────────────────────────────────
        if line.startswith("/shared "):
            tokens = line[len("/shared "):].split()
            results = model.infer_shared_role(tokens, mode=mode)
            if not results:
                print("  no shared cluster found")
            else:
                for tok, axis, ev in results:
                    print(f"  {' '.join(tokens)} -> {tok}  [{axis}  {ev}]")
            continue

        # ── /explain ──────────────────────────────────────────────────────
        if line.startswith("/explain "):
            payload = line[len("/explain "):]
            prev, curr = ("", payload) if "|" not in payload else \
                         [p.strip() for p in payload.split("|", 1)]
            token, trace = model.explain_step(prev, curr, mode=mode)
            print(f"  next='{token}'  {trace}")
            continue

        # ── generate ──────────────────────────────────────────────────────
        try:
            text, ids, trace = model.generate(line, max_tokens=args.max_tokens, mode=mode)
            print(f"  model> {text}")
            stages = [f"s{t['stage']}/{t.get('rule','')[:8]}" for t in trace]
            print(f"  trace> {' · '.join(stages)}")
        except RuntimeError as e:
            print(f"  error: {e}")


if __name__ == "__main__":
    main()
