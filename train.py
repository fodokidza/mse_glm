"""
train.py  --  the ONE training script for MSE-GLM.

One command, any source, fresh or incremental:

    SOURCE   --corpus PATH    a text file, OR a folder of .txt files (recursive)
             --text "..."     an inline string

    fresh    python3 train.py --corpus corpus.txt   --out runs/model
             python3 train.py --corpus data/        --out runs/model --vocab-size 8000
             python3 train.py --text "the cat sat on the mat." --out runs/demo

    add to an existing model (incremental -- same SOURCE options)
             python3 train.py --corpus new.txt  --continue-from runs/model
             python3 train.py --corpus more/    --continue-from runs/model --out runs/model2
             python3 train.py --corpus more/    --continue-from runs/model \\
                              --extend-vocab --target-vocab-size 3000

    big folders: --batch-size N  files merged per step (default from config.TrainCorpusConfig);
                 --no-recursive  only the top level of a folder;  --quiet  no live display.

This file is only a front end: argument parsing and the progress output (the plain pass-1 / pass-2 style train_corpus.py used).  EVERY
training decision (file vs folder, vocabulary pass, batching, merging, building
the inference engines, saving) lives in model.py -- see MSEGraphLanguageModel's
train_from_source() / train_incremental_source() / train() / train_incremental().
Change training there and this script needs no edit.
"""

import argparse
import os
import sys
import time

from config import TokenizerConfig, TrainCorpusConfig


# ============= terminal helpers =================================================================

def _dim(t):   return f"\033[2m{t}\033[0m"
def _teal(t):  return f"\033[36m{t}\033[0m"
def _green(t): return f"\033[32m{t}\033[0m"
def _bold(t):  return f"\033[1m{t}\033[0m"
def _amber(t): return f"\033[33m{t}\033[0m"


def _preview(text, n=72):
    return text[:n].replace("\n", " ") + ("…" if len(text) > n else "")


def _table(rows):
    w = max(len(k) for k, _ in rows)
    for k, v in rows:
        print(f"  {_dim(k.ljust(w))}  {_teal(v)}")
    print()


# ================ Fresh training ==============================================================

def train_fresh(args):
    """Fresh training from --corpus (file or folder) or --text. All work happens in model.py."""
    from model import MSEGraphLanguageModel, discover_corpus_files

    quiet = args.quiet
    def log(msg=""):
        if not quiet:
            print(msg)

    model = MSEGraphLanguageModel(vocab_size=args.vocab_size)
    files = []
    if args.corpus:
        files = discover_corpus_files(args.corpus, recursive=not args.no_recursive)

    log(f"  {_bold('MSE Graph Language Model')}  {_dim('─  Training')}")
    if args.corpus:
        kind = "folder" if os.path.isdir(args.corpus) else "file"
        log(f"  {_dim('corpus')}       {_teal(args.corpus)}  {_dim(f'({kind})')}")
        note = "(top-level only)" if args.no_recursive and kind == "folder" else ""
        log(f"  {_dim('files found')}  {_teal(str(len(files)))}" + (f"  {_dim(note)}" if note else ""))
        if len(files) > 1:
            log(f"  {_dim('batch_size')}   {_teal(str(args.batch_size))}  files per merge step")
    else:
        log(f"  {_dim('corpus')}       {_teal(_preview(args.text))}")
    log(f"  {_dim('vocab_size')}   {_teal(str(args.vocab_size))}")
    log()

    t0 = time.time()
    state = {"t_graph": None}

    def progress(stage, done, total, info):
        if quiet:
            return
        if stage == "vocab":
            if done == 1:
                print(f"  {_bold('Pass 1/2')}  building shared vocabulary…")
            words = f"({info['distinct_words']:,} distinct words so far)"
            print(f"\r    [{done}/{total}] {os.path.basename(info['path'])}  {_dim(words)}" + " " * 10,
                  end="", flush=True)
            if done == total:
                print()
                print(f"  {_green('✓')}  vocabulary pass done  {_dim(f'{time.time() - t0:.2f}s')}")
                print()
        elif stage == "graph":
            if done == 1:
                state["t_graph"] = time.time()
                plural = f" in batches of {args.batch_size} file(s)" if total > 1 else ""
                print(f"  {_bold('Pass 2/2')}  building graph{plural}…")
            s = info["stats"]
            elapsed = time.time() - state["t_graph"]
            print(f"\r    [batch {done}/{total}]"
                  f"  edges {s['edges']:,}  bridges {s['bridges']:,}"
                  f"  clusters {s['clusters']:,}  rels {s['relationships']:,}"
                  f"  {_dim(f'{elapsed:.1f}s')}" + " " * 6, end="", flush=True)
            if done == total:
                print()
                print(f"  {_green('✓')}  graph built  {_dim(f'{elapsed:.2f}s')}")
                print()

    if args.corpus:
        model.train_from_source(args.corpus, batch_size=args.batch_size,
                                recursive=not args.no_recursive, progress=progress)
    else:
        log(f"  {_bold('Training')}  tokenizing and building the matrix from inline text…")
        model.train(args.text)
        log(f"  {_green('✓')}  trained  {_dim(f'{time.time() - t0:.2f}s')}")
        log()

    model.save(args.out)

    stats = model.stats()
    log(f"  {_green('─' * 60)}")
    log(f"  {_green('✓')}  {_bold('Training complete')}")
    log()
    if not quiet:
        rows = [("Output folder", os.path.abspath(args.out))]
        if args.corpus:
            rows.append(("Files processed", f"{len(files):,}"))
        rows += [
            ("Vocabulary",        f"{stats['vocab_size']:,} tokens"),
            ("Slot fillers",      f"{stats['bridges']:,} unique triples"),
            ("Slot groups",       f"{stats['clusters']:,} groups  ·  {stats['clustered_bridges']:,} triples grouped"),
            ("Total time",        f"{time.time() - t0:.2f}s"),
        ]
        _table(rows)
    return model


# ======= Incremental training =================================================

def train_continue(args):
    """Add --corpus (file or folder) or --text to an already-trained model."""
    from model import MSEGraphLanguageModel

    quiet = args.quiet
    def log(msg=""):
        if not quiet:
            print(msg)

    out_path = args.out or args.continue_from
    log(f"  {_bold('MSE Graph Language Model')}  {_dim('─  Continuing training')}")
    log(f"  {_dim('loading')}      {_teal(os.path.abspath(args.continue_from))}")
    model = MSEGraphLanguageModel.load(args.continue_from)
    if args.corpus:
        kind = "folder" if os.path.isdir(args.corpus) else "file"
        log(f"  {_dim('new corpus')}   {_teal(args.corpus)}  {_dim(f'({kind})')}")
        log(f"  {_dim('batch_size')}   {_teal(str(args.batch_size))}  files per merge step")
    else:
        log(f"  {_dim('new corpus')}   {_teal(_preview(args.text))}")
    log(f"  {_dim('output')}       {_teal(os.path.abspath(out_path))}")
    if args.extend_vocab:
        log(f"  {_dim('extend_vocab')} {_teal('yes')}  {_dim(f'(target vocab_size {args.target_vocab_size})')}")
    log()

    t0 = time.time()

    def progress(stage, done, total, info):
        if quiet:
            return
        if stage == "vocab":
            print(f"\r    vocabulary [{done}/{total}] {os.path.basename(info['path'])}"
                  f"  {_dim('(' + format(info['distinct_words'], ',') + ' distinct words so far)')}" + " " * 10,
                  end="\n" if done == total else "", flush=True)
        else:
            s = info["stats"]
            print(f"\r    [batch {done}/{total}]"
                  f"  edges {s['edges']:,}  bridges {s['bridges']:,}"
                  f"  clusters {s['clusters']:,}  rels {s['relationships']:,}"
                  f"  {_dim(f'{time.time() - t0:.1f}s')}" + " " * 6,
                  end="\n" if done == total else "", flush=True)

    kw = dict(extend_vocab=args.extend_vocab, target_vocab_size=args.target_vocab_size)
    if args.corpus:
        summary = model.train_incremental_source(
            args.corpus, batch_size=args.batch_size, recursive=not args.no_recursive,
            progress=progress, **kw)
    else:
        summary = model.train_incremental(args.text, **kw)
    model.save(out_path)

    before, after = summary["before"], summary["after"]
    log()
    log(f"  {_green('─' * 60)}")
    log(f"  {_green('✓')}  {_bold('Saved to')} {os.path.abspath(out_path)}")
    log()
    if not quiet:
        rows = [
            ("Sentences added",  f"{summary['sentences_added']:,}"),
            ("Vocabulary",       f"{before['vocab_size']:,} → {after['vocab_size']:,}  (+{summary['vocab_added']:,} tokens)"),
            ("Adjacent pairs",   f"{before['edges']:,} → {after['edges']:,} unique bigrams"),
            ("Slot fillers",     f"{before['bridges']:,} → {after['bridges']:,} unique triples"),
            ("Slot groups",      f"{before['clusters']:,} → {after['clusters']:,} groups"
                                 f"  ({before['clustered_bridges']:,} → {after['clustered_bridges']:,} triples grouped)"),
            ("Sentence tiles",   f"{before['tile_rows']:,} → {after['tile_rows']:,}"
                                 f"  ({before['relationships']:,} → {after['relationships']:,} unique sentences)"),
        ]
        rows.append(("Open Mode", "auto-rebuilt from the merged matrix"))
        rows.append(("Total time", f"{time.time() - t0:.2f}s"))
        _table(rows)
    return model


# ==== CLI ===============================================================

def main():
    p = argparse.ArgumentParser(
        description="Train an MSE Graph Language Model from a file, a folder of .txt files, "
                    "or inline text -- fresh, or added to an existing model with --continue-from.")
    p.add_argument("--corpus", "--corpus-dir", dest="corpus", metavar="PATH",
                   help="A text file OR a folder of .txt files (folders are scanned recursively)")
    p.add_argument("--text",        help="Inline corpus string")
    p.add_argument("--out",         help="Output folder (required for fresh training; "
                                         "defaults to --continue-from's folder otherwise)")
    p.add_argument("--vocab-size",  type=int, default=TokenizerConfig.TRAIN_CLI_DEFAULT_VOCAB_SIZE)
    p.add_argument("--batch-size",  type=int, default=TrainCorpusConfig.DEFAULT_BATCH_SIZE,
                   help="Folder sources: files merged into the graph per step (fewer, bigger "
                        "batches = fewer merge passes, more memory per step)")
    p.add_argument("--no-recursive", action="store_true",
                   help="Folder sources: only scan the top level, not subfolders")
    p.add_argument("--quiet",       action="store_true")
    p.add_argument("--continue-from", metavar="FOLDER",
                   help="Add the corpus to an already-trained model instead of training a fresh "
                        "one. Loads FOLDER, merges, saves to --out (or back to FOLDER).")
    p.add_argument("--extend-vocab", action="store_true",
                   help="With --continue-from: also grow the vocabulary from the new corpus "
                        "(requires --target-vocab-size). Default: vocabulary stays frozen.")
    p.add_argument("--target-vocab-size", type=int, default=None,
                   help="New vocab_size ceiling when --extend-vocab is set")
    args = p.parse_args()

    def die(msg):
        print(msg, file=sys.stderr); sys.exit(1)

    if bool(args.corpus) == bool(args.text):
        die("Provide exactly one of --corpus <file-or-folder> or --text <string>")
    if args.batch_size < 1:
        die("--batch-size must be >= 1")
    if args.extend_vocab and not args.target_vocab_size:
        die("--extend-vocab requires --target-vocab-size")
    if not args.continue_from and not args.out:
        die("Provide --out <folder> (or use --continue-from to update a model in place)")
    if args.corpus and not os.path.exists(args.corpus):
        die(f"corpus not found: {args.corpus}")

    try:
        (train_continue if args.continue_from else train_fresh)(args)
    except (FileNotFoundError, ValueError) as e:
        die(str(e))


if __name__ == "__main__":
    main()
