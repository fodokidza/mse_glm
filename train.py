"""
train.py — Training pipeline for MSE-GLM.

Usage:
    python3 train.py --corpus path/to/corpus.txt --out runs/my_model --vocab-size 500
    python3 train.py --text "the cat sat on the mat. the dog sat on the carpet." --out runs/demo
"""

import argparse
import os
import sys

from model import MSEGraphLanguageModel


def main():
    parser = argparse.ArgumentParser(description="Train an MSE Graph Language Model")
    parser.add_argument("--corpus", help="Path to a text file to train from (streamed)")
    parser.add_argument("--text", help="Inline corpus text (small corpora / demos)")
    parser.add_argument("--out", required=True, help="Output folder for the trained model")
    parser.add_argument("--vocab-size", type=int, default=1000)
    args = parser.parse_args()

    if not args.corpus and not args.text:
        print("Provide --corpus <file> or --text <string>", file=sys.stderr)
        sys.exit(1)

    model = MSEGraphLanguageModel(vocab_size=args.vocab_size)

    if args.corpus:
        print(f"Training from file: {args.corpus}")
        model.train_from_file(args.corpus)
    else:
        print("Training from inline text")
        model.train(args.text)

    model.save(args.out)

    stats = model.stats()
    print(f"\nSaved model to: {os.path.abspath(args.out)}")
    print("Stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
