"""
chat.py — Interactive REPL to load a trained MSE-GLM model and chat /
test / analyse it from the command line.

Usage:
    python3 chat.py --model runs/demo

Commands inside the REPL:
    <any text>                 generate a continuation
    /explain <prev> | <curr>   explain a single inference step
    /shared <tok1> <tok2> ...  infer_shared_role() over a token set
    /stats                     show model stats
    /clusters                  show top cluster groups
    /quit                      exit
"""

import argparse

from model import MSEGraphLanguageModel
from analyse import Analyser


def main():
    parser = argparse.ArgumentParser(description="Chat with a trained MSE-GLM model")
    parser.add_argument("--model", required=True, help="Folder of a saved model")
    parser.add_argument("--max-tokens", type=int, default=30)
    args = parser.parse_args()

    print(f"Loading model from {args.model} ...")
    model = MSEGraphLanguageModel.load(args.model)
    analyser = Analyser(model)
    print("Loaded. Stats:", model.stats())
    print("Type text to generate, or /explain, /shared, /stats, /clusters, /quit\n")

    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line in ("/quit", "/exit"):
            break
        if line == "/stats":
            print(model.stats())
            continue
        if line == "/clusters":
            for c in analyser.cluster_report():
                print(c)
            continue
        if line.startswith("/shared "):
            tokens = line[len("/shared "):].split()
            results = model.infer_shared_role(tokens)
            if not results:
                print("model> no shared cluster found across those tokens")
            else:
                for tok, axis, evidence in results:
                    print(f"model> {' '.join(tokens)} -> {tok}  [{axis}, {evidence}]")
            continue
        if line.startswith("/explain "):
            payload = line[len("/explain "):]
            if "|" in payload:
                prev, curr = [p.strip() for p in payload.split("|", 1)]
            else:
                prev, curr = "", payload.strip()
            token, trace = model.explain_step(prev, curr)
            print(f"model> next='{token}'  trace={trace}")
            continue

        text, ids, trace = model.generate(line, max_tokens=args.max_tokens)
        print(f"model> {text}")


if __name__ == "__main__":
    main()
