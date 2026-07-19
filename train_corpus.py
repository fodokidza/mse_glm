"""
train_corpus.py — Large-corpus training pipeline for MSE-GLM.

Trains from many .txt files in a folder instead of a single string or
file. Two passes, each bounded in memory:

  Pass 1 (vocabulary):  stream word frequencies across every file --
                         only a Counter of (word -> count) and the
                         current read chunk are ever in memory, never
                         a whole file's text and never the whole
                         corpus's. One tokenizer is trained from the
                         combined counts, so every file shares one
                         vocabulary (no frozen-vocab UNK-collision
                         risk between files -- see train_incremental's
                         docs for what that risk looks like).

  Pass 2 (graph):        process files in batches (--batch-size files
                         at a time). The first batch calls the model's
                         normal from-scratch graph build; every batch
                         after that is merged in via train_incremental's
                         machinery (full recompute over the union of
                         old + new triples, clusters re-derived from
                         scratch -- same reasoning as train_incremental:
                         a cluster can only be discovered once every
                         triple that belongs to it is known).

Usage:
    python3 train_corpus.py --corpus-dir data/ --out runs/big_model --vocab-size 8000
    python3 train_corpus.py --corpus-dir data/ --out runs/big_model --batch-size 5
    python3 train_corpus.py --corpus-dir data/ --out runs/big_model --no-recursive --quiet

Known limitations (read before pointing this at something huge):
  - Pass 2 reads each file's full text into memory at once (not chunked
    within a file). This pipeline assumes a large CORPUS split across
    many reasonably-sized FILES, which is the scenario it was built
    for -- it does not chunk within a single giant file. If one file
    is itself enormous, either pre-split it or expect that file's
    memory footprint to be paid in one go.
  - Pass 2's per-batch merge is a full recompute over everything seen
    so far, same cost profile as train_incremental. Total cost across
    N batches is closer to O(N * corpus_size_so_far) than O(corpus_size)
    -- roughly quadratic in the number of batches for a corpus of fixed
    total size. --batch-size trades memory for speed: fewer, bigger
    batches means fewer full-recompute passes. Start higher than 1 for
    genuinely large corpora; 1 file at a time is the safest default,
    not the fastest one.
"""

import argparse
import os
import sys
import time
from collections import Counter

from tokenizer import BPETokenizer, split_sentences, stream_word_freq
from model import MSEGraphLanguageModel


def discover_txt_files(folder, recursive=True):
    """Sorted list of .txt file paths under folder (deterministic order,
    so re-running the same folder produces the same training order)."""
    paths = []
    if recursive:
        for root, _dirs, files in os.walk(folder):
            for fn in files:
                if fn.lower().endswith(".txt"):
                    paths.append(os.path.join(root, fn))
    else:
        for fn in os.listdir(folder):
            full = os.path.join(folder, fn)
            if os.path.isfile(full) and fn.lower().endswith(".txt"):
                paths.append(full)
    return sorted(paths)


def _dim(t):   return f"\033[2m{t}\033[0m"
def _teal(t):  return f"\033[36m{t}\033[0m"
def _green(t): return f"\033[32m{t}\033[0m"
def _bold(t):  return f"\033[1m{t}\033[0m"
def _amber(t): return f"\033[33m{t}\033[0m"


def train_from_folder(folder, out_path, vocab_size=2000, batch_size=1,
                       recursive=True, quiet=False):
    """
    Run the full two-pass pipeline and save the resulting model to
    out_path. Returns the trained MSEGraphLanguageModel.
    """
    files = discover_txt_files(folder, recursive=recursive)
    if not files:
        raise FileNotFoundError(f"No .txt files found under {folder!r}")

    def log(msg):
        if not quiet:
            print(msg)

    log(f"  {_bold('MSE Graph Language Model')}  {_dim('─  Large-corpus training')}")
    log(f"  {_dim('corpus_dir')}   {_teal(folder)}")
    log(f"  {_dim('files found')}  {_teal(str(len(files)))}"
        f"  {_dim('(recursive)' if recursive else '(top-level only)')}")
    log(f"  {_dim('vocab_size')}   {_teal(str(vocab_size))}")
    log(f"  {_dim('batch_size')}   {_teal(str(batch_size))}  files per merge step")
    log("")

    t0 = time.time()

    # ── Pass 1: vocabulary from streamed word frequencies ────────────
    log(f"  {_bold('Pass 1/2')}  building shared vocabulary…")
    word_freq = Counter()
    total_sentences_p1 = 0
    for i, path in enumerate(files, 1):
        n = stream_word_freq(path, word_freq)
        total_sentences_p1 += n
        if not quiet:
            print(f"\r    [{i}/{len(files)}] {os.path.basename(path)}"
                  f"  {_dim(f'({len(word_freq):,} distinct words so far)')}"
                  + " " * 10, end="", flush=True)
    if not quiet:
        print()

    tok = BPETokenizer(vocab_size=vocab_size)
    tok._train_from_word_freq(word_freq)
    log(f"  {_green('✓')}  vocabulary: {tok.vocab_size_actual:,} tokens"
        f"  ({len(tok.merges):,} merges)  {_dim(f'{time.time()-t0:.2f}s')}")
    log("")

    # ── Pass 2: graph, in batches, reusing incremental-training merge ─
    log(f"  {_bold('Pass 2/2')}  building graph in batches of {batch_size} file(s)…")
    model = MSEGraphLanguageModel(vocab_size=vocab_size)
    model.tokenizer = tok

    batches = [files[i:i + batch_size] for i in range(0, len(files), batch_size)]
    t1 = time.time()
    for bi, batch in enumerate(batches, 1):
        sentences = []
        for path in batch:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            sentences.extend(split_sentences(text))
        seqs = [tok.encode_for_training(s) for s in sentences]

        if bi == 1:
            model._build_graphs(seqs)
        else:
            model._merge_graphs(seqs)

        if not quiet:
            s = model.stats()
            print(f"\r    [batch {bi}/{len(batches)}]"
                  f"  edges {s['edges']:,}  bridges {s['bridges']:,}"
                  f"  clusters {s['clusters']:,}  rels {s['relationships']:,}"
                  f"  {_dim(f'{time.time()-t1:.1f}s')}" + " " * 6, end="", flush=True)
    if not quiet:
        print()

    log(f"  {_green('✓')}  graph built  {_dim(f'{time.time()-t1:.2f}s')}")
    log("")

    model.save(out_path)

    stats = model.stats()
    log(f"  {_green('─' * 60)}")
    log(f"  {_green('✓')}  {_bold('Training complete')}")
    log("")
    rows = [
        ("Output folder",     os.path.abspath(out_path)),
        ("Files processed",   f"{len(files):,}"),
        ("Vocabulary",        f"{stats['vocab_size']:,} tokens"),
        ("Edge Matrix",       f"{stats['edges']:,} unique bigrams"),
        ("Bridge Matrix",     f"{stats['bridges']:,} unique triples"),
        ("Clustered triples", f"{stats['clustered_bridges']:,}  ({stats['clusters']} clusters)"),
        ("Relationship rows", f"{stats['relationship_rows']:,}  ({stats['relationships']} sentences)"),
        ("Total time",        f"{time.time()-t0:.2f}s"),
    ]
    w = max(len(k) for k, _ in rows)
    for k, v in rows:
        log(f"  {_dim(k.ljust(w))}  {_teal(v)}")
    log("")

    return model


def main():
    p = argparse.ArgumentParser(
        description="Train an MSE Graph Language Model from a folder of .txt files")
    p.add_argument("--corpus-dir",  required=True, help="Folder containing .txt files")
    p.add_argument("--out",         required=True, help="Output folder")
    p.add_argument("--vocab-size",  type=int, default=2000)
    p.add_argument("--batch-size",  type=int, default=1,
                    help="Files merged into the graph per step (default 1 -- "
                         "safest memory profile; raise for fewer, faster merge passes)")
    p.add_argument("--no-recursive", action="store_true",
                    help="Only scan the top level of --corpus-dir, not subfolders")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    if args.batch_size < 1:
        print("--batch-size must be >= 1", file=sys.stderr)
        sys.exit(1)

    try:
        train_from_folder(
            args.corpus_dir, args.out,
            vocab_size=args.vocab_size,
            batch_size=args.batch_size,
            recursive=not args.no_recursive,
            quiet=args.quiet,
        )
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
