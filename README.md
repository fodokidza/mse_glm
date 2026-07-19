# MSE-GLM — Command Reference

**Matrix-Structured Edge — Graph Language Model.** Deterministic,
zero-weight, explainable. No embeddings, no gradient descent — every
output traces back to specific rows in specific matrices built from
training text.

Repo: `fodokidza/mse_glm` · 
Author: Clifford Chivhanga

Requires **Python 3** only — zero external dependencies (standard
library: `array`, `collections`, `json`, `re`, `os`, `random`, `time`,
`argparse`, `shutil`, `tempfile`). No `pip install` needed.

---

## Contents

- [File overview](#file-overview)
- [1. Train from scratch](#1-train-from-scratch)
- [2. Continue training on new data](#2-continue-training-on-new-data)
- [3. Train from a folder of .txt files](#3-train-from-a-folder-of-txt-files)
- [4. Build Experience Matrices (Open Mode)](#4-build-experience-matrices-open-mode)
- [5. Chat / generate interactively](#5-chat--generate-interactively)
- [6. Analyse a trained model](#6-analyse-a-trained-model)
- [7. Cluster Interpreter Matrix (naming clusters)](#7-cluster-interpreter-matrix-naming-clusters)
- [8. Context Trigger Matrix (contextual disambiguation)](#8-context-trigger-matrix-contextual-disambiguation)
- [9. Token Importance / Trigger analysis (Python API only)](#9-token-importance--trigger-analysis-python-api-only)
- [10. Analyse raw text with no trained model](#10-analyse-raw-text-with-no-trained-model)
- [11. Run the test suite](#11-run-the-test-suite)
- [Typical end-to-end session](#typical-end-to-end-session)
- [Notes and gotchas](#notes-and-gotchas)

---

## File overview

| File                  | What it's for                                                          |
|-----------------------|--------------------------------------------------------------------------|
| `tokenizer.py`        | From-scratch BPE tokenizer                                              |
| `graph.py`            | Edge / Bridge / Relationship matrices + dual-axis clustering            |
| `model.py`            | `MSEGraphLanguageModel` — orchestrates everything, save/load            |
| `inference.py`        | Two-stage lineage-vote generation engine                                |
| `experience.py`       | Cluster-substitution rules that derive Open Mode's Experience Matrices  |
| `interpret.py`        | Cluster Interpreter Matrix — names clusters, mines the zero-cluster bucket |
| `importance.py`       | Sequence reconstruction + per-triple importance/trigger tagging (Python API only) |
| `ctm.py`               | Context Trigger Matrix — contextual disambiguation among cluster members |
| `analyse.py`          | Read-only analysis CLI + `Analyser`/`CorpusAnalyser` Python API         |
| `train.py`            | Fresh training + `--continue-from` incremental training, with live display |
| `train_corpus.py`     | Large-corpus pipeline — train from a folder of `.txt` files             |
| `build_experience.py` | CLI to build/save Experience Matrices for an already-trained model      |
| `chat.py`             | Interactive REPL over a trained model                                  |
| `test.py`             | Full regression suite (all features, 340+ checks)                       |

---

## 1. Train from scratch

```bash
# Inline text
python3 train.py --text "the cat sat on the mat. the dog sat on the carpet." \
    --out runs/demo --vocab-size 1000

# From a file (streamed in chunks, doesn't load the whole file into memory for tokenizing)
python3 train.py --corpus corpus.txt --out runs/model --vocab-size 2000

# Quiet mode (no live animated display, just a final summary)
python3 train.py --corpus corpus.txt --out runs/model --quiet
```

| Flag             | Meaning                                              |
|------------------|-------------------------------------------------------|
| `--text`         | Inline corpus string (use this OR `--corpus`)         |
| `--corpus`       | Path to a text file (streamed)                        |
| `--out`          | Output folder — **required** for fresh training       |
| `--vocab-size`   | Target BPE vocabulary size (default 1000)             |
| `--quiet`        | Skip the live animated display                        |

---

## 2. Continue training on new data

Adds a new corpus to an already-trained model **without discarding
what it already knows**. Clusters are recomputed over the union of
old + new facts (a cluster can only form once every triple that
belongs to it exists). Experience Matrices AND the Context Trigger
Matrix are both invalidated automatically if present (both are
derived from the pre-merge structure).

```bash
# Frozen vocabulary (default, safest) -- reuses the existing tokenizer as-is
python3 train.py --continue-from runs/model --text "the pig sat on the rug." \
    --out runs/model

# Grow the vocabulary too (needed if the new text has substantially new words --
# otherwise unseen characters collapse onto the same <UNK> id and can create
# false structural matches between unrelated new words)
python3 train.py --continue-from runs/model --corpus new_data.txt \
    --extend-vocab --target-vocab-size 3000
```

| Flag                    | Meaning                                                        |
|-------------------------|-------------------------------------------------------------------|
| `--continue-from FOLDER`| Load this model and merge new data into it                      |
| `--out`                 | Where to save (omit to save back into `--continue-from`'s folder) |
| `--extend-vocab`        | Also grow the vocabulary from the new corpus (requires `--target-vocab-size`) |
| `--target-vocab-size`   | New vocab ceiling; must exceed the loaded model's current vocab size |

Python API equivalent:
```python
model = MSEGraphLanguageModel.load("runs/model")
summary = model.train_incremental(new_text, extend_vocab=True, target_vocab_size=3000)
# summary["experience_invalidated"] / summary["ctm_invalidated"] tell you what to rebuild
model.save("runs/model")
```

---

## 3. Train from a folder of .txt files

For a large corpus split across many files. Two bounded-memory passes:
builds one shared vocabulary by streaming word frequencies across every
file first, then builds the graph in batches, reusing the same merge
machinery as `--continue-from` above.

```bash
python3 train_corpus.py --corpus-dir data/ --out runs/big_model --vocab-size 8000

# Merge 5 files per step instead of 1 (fewer full-recompute passes, more memory per step)
python3 train_corpus.py --corpus-dir data/ --out runs/big_model --batch-size 5

# Only scan the top level of the folder, not subfolders
python3 train_corpus.py --corpus-dir data/ --out runs/big_model --no-recursive

python3 train_corpus.py --corpus-dir data/ --out runs/big_model --quiet
```

| Flag             | Meaning                                                              |
|------------------|----------------------------------------------------------------------|
| `--corpus-dir`   | Folder containing `.txt` files — **required**                       |
| `--out`          | Output folder — **required**                                        |
| `--vocab-size`   | Target BPE vocabulary size (default 2000)                            |
| `--batch-size`   | Files merged into the graph per step (default 1 — safest memory profile; raise for fewer, faster merge passes on large corpora) |
| `--no-recursive` | Only scan the top level, skip subfolders                             |
| `--quiet`        | Suppress progress output                                             |

> Files are discovered recursively by default, sorted for a
> deterministic training order. Reads each file's full text at once
> (doesn't chunk within a single file) — this pipeline assumes a
> corpus split across many reasonably-sized files, not one giant file.

---

## 4. Build Experience Matrices (Open Mode)

Derives inferred (never-literally-seen) triples from cluster
substitutability, enabling `--mode open` in generation/analysis.

```bash
python3 build_experience.py --model runs/model

python3 build_experience.py --model runs/model --dry-run   # preview without writing files
python3 build_experience.py --model runs/model --quiet
```

---

## 5. Chat / generate interactively

```bash
python3 chat.py --model runs/model
python3 chat.py --model runs/model --mode open --max-tokens 40
```

REPL commands once inside:

| Command                     | What it does                                      |
|------------------------------|----------------------------------------------------|
| `<any text>`                 | Generate a continuation in the current mode        |
| `/mode strict` / `/mode open`| Switch modes (builds Experience Matrices on first use if needed) |
| `/explain <prev> \| <curr>`  | Explain one inference step                         |
| `/shared <tok1> <tok2> ...`  | `infer_shared_role()` across a token set           |
| `/similarity <a> <b>`        | Cluster-overlap similarity between two tokens      |
| `/stats`                     | Model stats (includes experience counts if built)  |
| `/clusters`                  | Top dual-axis cluster groups                       |
| `/exp`                       | Experience matrix summary                          |
| `/quit`                      | Exit                                                |

> `chat.py` does not currently expose Context Trigger Matrix
> disambiguation (`use_context_triggers`) — that's available through
> the Python API (`model.generate(..., use_context_triggers=True)`)
> and could be wired into the REPL as a `/ctm` toggle if useful.

---

## 6. Analyse a trained model

All subcommands need `--model FOLDER`. Add `--json PATH` (**before**
the subcommand) to write the result as JSON to a file instead of
printing it.

```bash
python3 analyse.py --model runs/model <subcommand> [options]

# Write to JSON instead of printing -- note --json comes BEFORE the subcommand
python3 analyse.py --model runs/model --json out.json clusters
```

| Subcommand      | Example                                              | What it shows |
|------------------|--------------------------------------------------------|---------------|
| `stats`          | `analyse.py --model runs/model stats`                  | vocab/edges/bridges/clusters/relationship counts |
| `topology`       | `... topology --top 10`                                | hub tokens (highest out-degree), dead-end count |
| `clusters`       | `... clusters --top 20 --axis bridge`                   | dual-axis cluster report |
| `cluster`        | `... cluster 3`                                         | full detail for one cluster_id |
| `relationships`  | `... relationships`                                     | Relationship Matrix summary |
| `relationship`   | `... relationship 2`                                    | full detail for one training sentence |
| `token`          | `... token cat`                                         | successors, bridge triples, clusters for one token |
| `similarity`     | `... similarity cat dog`                                | cluster-overlap similarity between two tokens |
| `shared`         | `... shared cat dog pig`                                | `infer_shared_role()` across 2+ tokens |
| `trace`          | `... trace "the cat" --max-tokens 10`                   | step-by-step generation trace (stage/rule/lineage per token) |
| `report`         | `... report --top 10`                                   | combined stats + topology + clusters + relationships |

---

## 7. Cluster Interpreter Matrix (naming clusters)

```bash
# All candidates for one cluster, unfiltered, ranked best-first
python3 analyse.py --model runs/model interpret 1 --top 10 --mode open

# Every candidate clearing --min-coverage, up to --max-per-cluster per cluster
python3 analyse.py --model runs/model interpretations --min-coverage 0.5 --max-per-cluster 3

# The filtered CI Matrix: coverage AND evidence_mask size both required
# (a cluster may carry more than one qualifying label at once)
python3 analyse.py --model runs/model interpreter-matrix --min-coverage 0.5 --min-signals 2

# Persist it
python3 analyse.py --model runs/model --json runs/model/interpreter_matrix.json \
    interpreter-matrix --min-coverage 0.5 --min-signals 2

# Mine cluster_id==0 for groups the standard dual-axis rule never assigns
# an id to at all (fix bridge+target, source varies)
python3 analyse.py --model runs/model zero-cluster --min-group-size 2
```

`--mode open` is available on `interpret`, `interpretations`,
`interpreter-matrix`, and `zero-cluster` — it additionally folds in
the Experience Matrices (requires Experience Matrices to already
exist; see section 4).

---

## 8. Context Trigger Matrix (contextual disambiguation)

The Bridge Matrix answers "which tokens can occupy this slot?"
(cat/dog/pig are interchangeable). The Context Trigger Matrix answers
"given this surrounding context, which one SHOULD?" — built from
whole-sentence co-occurrence, no new training.

```bash
# Inspect the flat trigger table (analysis only -- see below for actually
# enabling this in generation)
python3 analyse.py --model runs/model context-triggers --min-support 1
python3 analyse.py --model runs/model context-triggers --min-support 3 --mode open
```

**Building and using it for generation is Python-API only:**

```python
from model import MSEGraphLanguageModel

model = MSEGraphLanguageModel.load("runs/model")
model.build_context_triggers()          # builds + caches model.ctm

text, ids, trace = model.generate(
    "the farmer fed the", max_tokens=5, use_context_triggers=True)
```

Without `use_context_triggers=True` (the default), generation is
byte-for-byte identical to before this feature existed — it's strictly
opt-in. When enabled, it only ever activates at a genuine tie between
members of the same cluster (never overrides a unique answer), and
falls straight back to the existing random tie-break if the context
gives it no signal. Look for `"rule": "context_trigger_resolved"` in
the trace to see exactly when it fired.

```python
model.has_context_triggers()   # bool
```

---

## 9. Token Importance / Trigger analysis (Python API only)

Not wired into the CLI — import directly:

```python
from importance import (sequence_for_relationship, important_tokens_in_sequence,
                         trigger_matrix, expected_importance)

seq = sequence_for_relationship(model, rel_id=0)          # reconstruct a training sentence
tagged = important_tokens_in_sequence(model, rel_id=0)     # which tokens in it are "important", and why
triggers = trigger_matrix(model, min_sequences=2)          # local (immediate-window) trigger generalization
expected_importance(model, prev_token_id, current_token_id)  # what Stage 2 already implies comes next
```

See `importance.py`'s module docstring for the distinction between
this (immediate 2-3 token window) and the Context Trigger Matrix in
section 8 (whole-sentence window) — they're deliberately different
granularities of the same underlying idea.

---

## 10. Analyse raw text with no trained model

```bash
python3 analyse.py corpus --text "the cat sat on the mat." --top 10
python3 analyse.py corpus --file corpus.txt --top 10
```

The only subcommand that doesn't need `--model` — just word/sentence
statistics over raw text.

---

## 11. Run the test suite

```bash
python3 test.py
```

No flags. Runs the full regression suite (tokenizer, graph
construction, generation determinism, Experience Matrix / Open Mode,
Cluster Interpreter, zero-cluster mining, Token Importance analysis,
Context Trigger Matrix, incremental training, large-corpus pipeline,
save/load round-trips) and prints a final `N passed, 0 failed` summary.

---

## Typical end-to-end session

```bash
# 1. Train on a folder of files
python3 train_corpus.py --corpus-dir data/ --out runs/model --vocab-size 4000

# 2. Look at what it learned
python3 analyse.py --model runs/model clusters --top 20
python3 analyse.py --model runs/model interpreter-matrix --min-coverage 0.5 --min-signals 2
python3 analyse.py --model runs/model context-triggers --min-support 2

# 3. Build Open Mode + Context Trigger Matrix
python3 build_experience.py --model runs/model
python3 -c "from model import MSEGraphLanguageModel as M; m=M.load('runs/model'); \
    m.build_context_triggers(); print('ok')"
# (CTM isn't persisted by save()/load() -- rebuild it each session, or
#  add your own caching around ContextTriggerMatrix.to_dict()/from_dict())

# 4. Chat with it
python3 chat.py --model runs/model --mode open

# 5. Add more data later, in place
python3 train.py --continue-from runs/model --corpus more_data.txt \
    --extend-vocab --target-vocab-size 6000
# Experience Matrices AND the Context Trigger Matrix are now invalidated:
python3 build_experience.py --model runs/model
# (rebuild CTM in your own script/session too, per step 3)
```

---

## Notes and gotchas

- **`--json` is a path, not a flag**, and belongs on the *main*
  `analyse.py` parser — it must come before the subcommand:
  `analyse.py --model X --json out.json clusters`, not
  `analyse.py --model X clusters --json out.json`.
- **Incremental training invalidates both Experience Matrices AND the
  Context Trigger Matrix.** If you use `--mode open` or
  `use_context_triggers=True` anywhere, rebuild both after any
  `--continue-from` or `train_corpus.py` run.
  `train_incremental()`'s return value tells you which were affected:
  `summary["experience_invalidated"]` / `summary["ctm_invalidated"]`.
- **The Context Trigger Matrix is not persisted by `save()`/`load()`.**
  It's cheap enough to rebuild per session
  (`model.build_context_triggers()`) but if you want it cached to
  disk, `ContextTriggerMatrix.to_dict()`/`from_dict()` are
  JSON-safe — wire your own save/load around them if needed.
- **`use_context_triggers=True` only ever changes behavior at a
  genuine tie.** It never overrides a unique, structurally-determined
  answer, and it falls back to the exact pre-existing random
  tie-break whenever it has no signal. Default is `False` everywhere.
- **Frozen vocabulary is the default** for incremental/continued
  training. It's safe for corpora similar to what the model already
  knows; pass `--extend-vocab --target-vocab-size N` when the new
  text introduces substantially new vocabulary, or unseen characters
  will collapse onto the same `<UNK>` id and can create false
  structural matches between unrelated words.
- **`--batch-size` in `train_corpus.py` is a memory/speed knob, not a
  correctness one** — the final model is identical regardless of
  batch size (verified in `test.py`). Raise it for large corpora to
  reduce the number of full-recompute merge passes.
- **Common tokens will show up as Context Trigger Matrix noise.**
  `"the"`, `"on"`, `"is"` etc. accumulate support across nearly every
  cluster member on a small/repetitive corpus, since there's no
  stopword filtering — only reserved tokens
  (`<PAD>/<UNK>/<BOS>/<EOS>`) are excluded. Use `--min-support` to
  cut noise, and don't treat a high-support common-word trigger as
  meaningful without checking what else fired for that member.
- **Cluster Interpreter caveats** (coverage isn't a probability,
  evidence signals are correlated not independent, zero-cluster
  mining has a much larger candidate space than the regular matrix)
  are documented in depth in `interpret.py`'s module docstring.
