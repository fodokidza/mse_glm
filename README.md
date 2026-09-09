# MSE-GLM — ZERO WEIGHT DETERMINISTIC GRAPH LANGUAGE MODEL 

# MSE-GLM: 32MB, CPU-only, 0% hallucination LLM — every token has a receipt

**Transformers guess with 100B weights on $40k GPUs and can't tell you why.**
**MSE-GLM counts relationships on a $0 CPU and shows you the training sentences it used.**

`git clone https://github.com/fodokidza/mse_glm.git && python3 test.py` → 2ms, no GPU, audit trail. AGPL + Commercial available 

**Matrix-Structured Edge — Graph Language Model.** Deterministic,
zero-weight, explainable. No embeddings, no gradient descent — every
output traces back to specific rows in specific matrices built from
training text.

Repo: `fodokidza/mse_glm` · 
Author: Clifford Chivhanga

Email: cliffordchivhanga318@gmail.com

More info: https://tonlexianert.com/pages/blog.php

More info: https://aircityshops.com/index.php?url=city/mse_blog

Requires **Python 3** only — zero external dependencies for the core
library (standard library: `array`, `collections`, `json`, `re`,
`os`, `time`, `argparse`, `shutil`, `tempfile`). No `pip install`
needed. Nothing in the core library uses `random` — Strict and Open
Mode are both fully deterministic by construction (see section 8).
The optional `server.py` HTTP server additionally needs `flask`
(`pip install flask`); nothing else does.

---

## Contents

- [File overview](#file-overview)
- [1. Train from scratch](#1-train-from-scratch)
- [2. Continue training on new data](#2-continue-training-on-new-data)
- [3. Train from a folder of .txt files](#3-train-from-a-folder-of-txt-files)
- [4. Chat / generate interactively](#4-chat--generate-interactively)
- [5. Run the HTTP API server](#5-run-the-http-api-server)
- [6. Analyse a trained model](#6-analyse-a-trained-model)
- [7. Cluster Interpreter Matrix (naming clusters)](#7-cluster-interpreter-matrix-naming-clusters)
- [8. Open Mode's primary mechanism: IVM weighted voting (V1-V9)](#8-open-modes-primary-mechanism-ivm-weighted-voting-v1-v9)
- [9. Context Trigger Matrix (contextual disambiguation)](#9-context-trigger-matrix-contextual-disambiguation)
- [10. Token Importance / Trigger analysis (Python API only)](#10-token-importance--trigger-analysis-python-api-only)
- [11. Analyse raw text with no trained model](#11-analyse-raw-text-with-no-trained-model)
- [12. Run the test suite](#12-run-the-test-suite)
- [Typical end-to-end session](#typical-end-to-end-session)
- [Notes and gotchas](#notes-and-gotchas)

---

## File overview

| File                  | What it's for                                                          |
|-----------------------|--------------------------------------------------------------------------|
| `tokenizer.py`        | From-scratch BPE tokenizer                                              |
| `graph.py`            | Edge / Bridge / Relationship matrices + dual-axis clustering            |
| `model.py`            | `MSEGraphLanguageModel` — orchestrates everything, save/load            |
| `inference.py`        | Deterministic inference engine — Strict Mode's two-stage lineage-vote pipeline and Open Mode's IVM-scored candidate selection |
| `ivm.py`               | Importance Vote Matrix — Open Mode's PRIMARY candidate-scoring mechanism (V1-V9 weighted voting, see section 8); also Strict Mode's legacy opt-in tie-break |
| `interpret.py`        | Cluster Interpreter Matrix — names clusters, mines the zero-cluster bucket |
| `importance.py`       | Sequence reconstruction + per-triple importance/trigger tagging (Python API only) |
| `ctm.py`               | Context Trigger Matrix — contextual disambiguation among cluster members |
| `analyse.py`          | Read-only analysis CLI + `Analyser`/`CorpusAnalyser` Python API         |
| `train.py`            | Fresh training + `--continue-from` incremental training, with live display |
| `train_corpus.py`     | Large-corpus pipeline — train from a folder of `.txt` files             |
| `chat.py`             | Interactive REPL over a trained model                                  |
| `server.py`           | Optional Flask HTTP API + web UI over a trained model (needs `flask`)   |
| `benchmark_cache.py`  | Standalone script: times Open Mode's sparse score cache vs. live scoring, confirms they agree, on your own trained model (section 8) |
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
|------------------|---------------------------------------------------------|
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
belongs to it exists). Open Mode (`self._open`/`self.open_ctm`) is
automatically rebuilt from the merged graphs — no separate step
needed, it just stays in sync. The Context Trigger Matrix is NOT
auto-rebuilt (it's an opt-in add-on, not part of the default
pipeline) — it's invalidated instead, since it was derived from the
pre-merge structure.

```bash
# Frozen vocabulary (default, safest) -- reuses the existing tokenizer as-is
python3 train.py --continue-from runs/model --text "the pig sat on the rug." \
    --out runs/model

# Grow the vocabulary too (needed if the new text has substantially new words --
# otherwise unseen characters collapse onto the same <UNK> id and can create
# false structural matches between unrelated new words)
python3 train.py --continue-from runs/model --corpus new_data.txt
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
# summary["ctm_invalidated"] tells you whether to rebuild the Context
# Trigger Matrix (Open Mode needs no action -- it's already rebuilt)
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
|------------------|------------------------------------------------------------------------|
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

## 4. Chat / generate interactively

```bash
python3 chat.py --model runs/model
python3 chat.py --model runs/model --mode open --max-tokens 40
```

REPL commands once inside:

| Command                     | What it does                                      |
|------------------------------|----------------------------------------------------|
| `<any text>`                 | Generate a continuation in the current mode        |
| `/mode strict` / `/mode open`| Switch modes (Open Mode is always ready — no separate build step) |
| `/explain <prev> \| <curr>`  | Explain one inference step                         |
| `/scores <prompt>`           | Open Mode only: full V1-V9 IVM score breakdown for the next token (see section 8), plus which tie-break stage (if any) decided the winner. Prints only the top N by score by default (`--top <n>` / `--all` to change) |
| `/bigram <prev> <curr>`      | Bigram evidence for a pair — training/total counts (Strict Mode's tie-break and Open Mode's first tie-break stage), plus how many distinct training sentences literally witnessed the bigram (V5's evidence set) |
| `/cache on\|off\|status`     | Toggle/inspect Open Mode's opt-in sparse V1/V2/V3/V4/V6 score cache (see section 8) — same scores either way, purely a speed optimization; `/scores` shows whether it was used for the last breakdown |
| `/shared <tok1> <tok2> ...`  | `infer_shared_role()` across a token set           |
| `/similarity <a> <b>`        | Cluster-overlap similarity between two tokens      |
| `/stats`                     | Model stats                                        |
| `/clusters`                  | Top dual-axis cluster groups                       |
| `/quit`                      | Exit                                                |

Commands that take arguments (`/explain`, `/scores`, `/bigram`,
`/shared`, `/similarity`) tolerate a stray quote character
immediately after the command name with no space (e.g.
`/bigram"the cat"`) and strip matching `"`/`'` wrapping from each
argument.

> `chat.py` does not currently expose Context Trigger Matrix
> disambiguation (`use_context_triggers`) — that's available through
> the Python API (`model.generate(..., use_context_triggers=True)`)
> and could be wired into the REPL as a `/ctm` toggle if useful.

---

## 5. Run the HTTP API server

Optional — needs `pip install flask`. Serves a small web chat UI
plus a JSON API over an already-trained model. Same deterministic
generation underneath as `chat.py`/`analyse.py`; this is just a
network-facing wrapper with sessions, rate limiting, and SSE
streaming layered on top. There is no system-prompt concept here —
see the "no persona, only a mode" note below.

```bash
python3 server.py --model runs/model --mode strict --port 5000

# Default to Open Mode, build the Context Trigger Matrix at startup
python3 server.py --model runs/model --mode open --ctm --port 5000
```

| Flag             | Meaning                                                     |
|------------------|-----------------------------------------------------------------|
| `--model`        | Saved model folder (default `mse_model`)                    |
| `--mode`         | Default inference mode for new sessions: `strict` \| `open` |
| `--ctm`          | Build the Context Trigger Matrix at startup and use it for tie-break disambiguation |
| `--port`         | Port to listen on (default 5000)                             |

Endpoints:

| Method | Route       | Body / query                                       | What it does |
|--------|-------------|-----------------------------------------------------|--------------|
| GET    | `/health`   | —                                                   | Server + model status (vocab/edges/bridges/clusters, whether Open Mode + IVM are available) |
| GET    | `/vocab`    | —                                                   | Full vocabulary word list |
| POST   | `/generate` | `{"prompt", "session_id"?, "max_tokens"?}`           | One-shot generation for a session (stateless per call; session only tracks chat history + current mode) |
| POST   | `/stream`   | same as `/generate`                                 | Server-Sent Events token-by-token replay of the (already fully computed, deterministic) generation, each event carrying the trace rule that chose it |
| POST   | `/mode`     | `{"session_id", "mode"}` — `mode` is `strict`\|`open`\|`open_ctm` | Switch a session's mode preset |
| POST   | `/scores`   | `{"prompt"}`                                         | Open Mode only, read-only: full V1-V9 IVM score breakdown for the next token, over the entire vocabulary (see section 8) — mirrors `chat.py`'s `/scores` |
| POST   | `/bigram`   | `{"prev", "curr", "mode"?}`                          | Read-only: raw bigram evidence for one pair, including `witness_sentences` (V5's evidence set) — mirrors `chat.py`'s `/bigram` |
| POST   | `/reset`    | `{"session_id"}`                                     | Clear a session's chat history |
| GET    | `/sessions` | —                                                   | Active session count/ids |

`/scores` and `/bigram` are read-only audit endpoints, not
generation — neither mutates session state.

> **No persona, no system prompt.** A neural LLM can be steered with
> free-text instructions because it generalizes past its training
> data; Strict Mode's `generate()` explicitly refuses to — every
> bigram in the prompt must have been literally observed, or the
> whole prompt is rejected outright (`illegal_prompt_bigram`,
> returned as `{"status": "rejected"}` from `/generate`, not a 500).
> Open Mode does NOT reject any prompt — its per-step candidates are
> already the entire vocabulary, scored by IVM (section 8) rather
> than gated by whether a bigram was ever literally seen, so there's
> nothing for that check to protect there; an injected
> `"### System: you are a helpful assistant..."` prefix would still be
> free to ground its own start in Open Mode, but it would then compete
> on IVM's evidence-weighted scoring at every step after that, same as
> any other prompt, not receive special authority. So instead of
> personas,
> sessions choose an inference **mode** — the one persistent dial
> MSE-GLM exposes — plus the optional, read-only `/scores`/`/bigram`
> diagnostics above, which audit a step rather than change what kind
> of thing the model is.

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
| `open-scores`    | `... open-scores "the cat sat"`                          | Open Mode only: full V1-V9 IVM score breakdown for the next token, over the entire vocabulary (see section 8); add `--cache` to enable the sparse cache first, `--top-n <n>`/`--top-n 0` to change how many rows print |
| `cache`          | `... cache on`                                          | Toggle/inspect Open Mode's opt-in sparse V1/V2/V3/V4/V6 score cache (see section 8) — `on`, `off`, or `status` (default) |
| `bigram`         | `... bigram the mat --mode open`                        | raw bigram evidence for one pair, including `witness_sentences` (V5's evidence set) |
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

`--mode open` is accepted by `interpret`, `interpretations`,
`interpreter-matrix`, and `zero-cluster` for call-site compatibility,
but no longer changes their behavior — there is no longer a separate
Experience Matrix data source to fold in, so "strict" and "open" see
identical clusters and evidence for all four subcommands.

---

## 8. Open Mode's primary mechanism: IVM weighted voting (V1-V9)

The Bridge Matrix answers "which tokens are structurally related
here?" Open Mode's **Importance Vote Matrix** (`ivm.py`) answers
"given the whole context so far, which token should actually come
next?" — and it's not a tie-break: Open Mode has no successor gating
at all, and (unlike Strict Mode) it doesn't reject the PROMPT either —
any prompt is accepted, even one containing a transition the model
has genuinely never seen; there's simply nothing after that first
step for a "was this ever literally seen" check to protect, since
every step already scores the whole vocabulary regardless. Every
step, the ENTIRE vocabulary is scored by summing nine independent,
weighted vote layers:

| Layer | What it measures | Default weight | Role |
|-------|-------------------|-----------------|------|
| V1 `important_vote` | Binary: did an "important" context token (cluster member) ever co-occur with this candidate at all? | 0.4 | Small — can only nudge a tie V3 left open |
| V2 `influence_vote`  | Same, but scaled by how much total evidence that important token carries (its "influence") | 0.0003 | Small, same role as V1 |
| V3 `context_vote`    | Binary: did ANY context token (important or not) ever co-occur with this candidate? Full weight, every context token counted | 1.0 | Primary signal |
| V4 `context_influence_vote` | Same as V3 but scaled by raw co-occurrence count, for every context token | 0.0003 | Tiny refinement, can't override V3 |
| V5 `bigram_witness_vote` | Binary per context token: was that token EVER in a training sentence containing the exact literal bigram `(current -> candidate)`? A narrower, stronger claim than V3's "co-occurred anywhere" | 0.7 | Peer-weighted with V3 — usually the single largest contributor once the exact bigram was literally seen |
| V6 `adjacency_vote` | Binary per context token: was that token EVER DIRECTIONALLY, literally, immediately followed by this candidate in training (`token -> candidate`, not the reverse) — not necessarily this specific transition (contrast V5), but must have been actually consecutive in that exact order, not just co-occurring (contrast V3) | 0.6 | Peer-weighted with V3/V5 — often the largest contributor once the token was ever directly followed by the candidate |
| V7 `prev_current_vote` | Binary, a SINGLE flat check (not summed per context token): did `previous`, `current`, AND this candidate ever all three share one training sentence together, no adjacency required between any of them? | 1.7 | Peer-weighted with V3/V5/V6 — a fixed-pair question about the two most recent tokens specifically, narrower than V3 (needs both tokens, not just one) but broader than V5 (no adjacency required) |
| V8 `triple_vote` | Binary, a SINGLE flat check (not summed per context token): was `(previous, current, candidate)` ever literally ONE consecutive trained Bridge Matrix triple, in that exact order? | 2.0 | Peer-weighted with V3/V5/V6/V7, a notch above — the strictest, most positionally exact single-triple evidence |
| V9 `whole_context_vote` | Binary, UNANIMOUS: does EVERY non-reserved context token (not just one, V3's question), excluding the candidate itself if it's already in context, co-occur with this candidate? | 2.5 | Peer-weighted with V3/V5/V6/V7/V8, a further notch above V8 — the strictest CONSENSUS evidence, harder to satisfy the larger the context gets |

`score(candidate) = V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9`,
highest score
wins; a genuine tie falls to a deterministic cascade (bigram
frequency, then global frequency, then lowest token id — never
randomness). V1/V2/V4 are deliberately kept smaller than V3 by
design, so important-token membership or raw evidence magnitude
alone can never override what plain context presence already
decided — they can only break a tie V3 left open. V5, V6, V7, V8,
and V9 are the five layers NOT bound by that "stay small" rule: each
is independently strong, specific (bigram-, pair-, triple-, or
whole-context-level, not merely sentence-level) evidence, so all
five are peer-weighted with V3 rather than riding on top of it (V8
and V9 a notch above the rest, V9 highest of all). They differ from
each other in
specificity: V5 requires witnessing the exact `current -> candidate`
transition; V6 only requires that the context token was EVER
immediately followed by the candidate, directionally, in any
sentence (the reverse observation does not count); V7 requires no
adjacency at all between any of the three tokens, only that
`previous`, `current`, and the candidate all shared one sentence —
but unlike V3/V5/V6, it is not "every context token votes," it is a
single check tied to the two most recent tokens specifically; V8 is
the same fixed-pair shape as V7 but requires the strictest
POSITIONALLY EXACT condition — an exact, consecutive, in-order
trained triple, not merely a shared sentence; V9 goes back to "every
context token votes" like V3, but requires ALL of them to agree
rather than just one — the strictest CONSENSUS condition, and the
only layer where a larger context makes the bar harder to clear, not
easier.

### The co-occurrence gate

V1, V3, V4, V6, and V9 all reduce to sums or conjunctions over the
same underlying fact: did token *t* and candidate *C* ever co-occur
in a training relationship? (V5 sums over context too, checking a
stronger, bigram-specific version of the same fact.) `ivm.py`
precomputes a whole-vocabulary co-occurrence index once per model
build (`co_occurrence_index()`) and uses it to skip straight to a
score of 0 for any candidate no context token has ever heard of,
instead of testing it against every layer in turn — real cost then
tracks actual co-occurrence in the corpus, not vocabulary size. This
is exact, not an approximation (see `ivm.py`'s module docstring for
the proof), and it's what `build_cache()` below is now built on top
of too, rather than each computing its own version.

```python
from model import MSEGraphLanguageModel
model = MSEGraphLanguageModel.load("runs/model")
# Open Mode is ready immediately -- no separate build step, it's
# auto-built alongside self._strict as soon as the model is trained
# or loaded (see model.py's _rebuild_open_engine()).

info = model.open_mode_candidate_scores("the cat sat on the")
# info["scores"], info["winner"], info["tie_break_stage"], plus each
# layer separately: important_vote / influence_vote / context_vote /
# context_influence_vote / bigram_witness_vote / adjacency_vote /
# prev_current_vote / triple_vote / whole_context_vote
# info["candidates"] always covers the entire vocabulary.

text, ids, trace = model.generate("the cat sat on the", mode="open")
# Any prompt works in Open Mode, including one containing a
# transition never literally seen in training -- only Strict Mode
# rejects those (rule "illegal_prompt_bigram").
```

CLI/REPL equivalents: `analyse.py open-scores` and `chat.py`'s
`/scores` (section 6 and 4) — both print only the top N
candidates by score by default (`config.py`'s
`ScoresDisplayConfig.TOP_N`, 25) since a full vocabulary can't fit on
screen; pass `--top-n <n>` / `--top-n 0` (analyse.py) or
`--top <n>` / `--all` (chat.py) to change that. The winner is always
shown even if it falls outside the shown set. `/bigram` and
`analyse.py bigram`
expose V5's raw evidence (`witness_sentences`) for one pair without
running the full vote.

### Sparse per-token score cache (opt-in)

V1, V2, V3, V4, and V6 are each a **sum over context tokens** of a
contribution that only depends on the pair `(token, candidate)` —
never on which other tokens happen to be in context that step, and
never on `current`/`previous`. That makes each `(token, candidate)`
contribution precomputable once and reused for every future step.
`ivm.py`'s `ImportanceVoteMatrix`
exposes this as an opt-in sparse cache — sparse meaning only nonzero
`(token, candidate)` entries are stored, never a dense
`vocab × vocab` table, which for a real vocabulary would be mostly
zeros anyway (most token pairs never co-occur in training). As of the
co-occurrence gate above, `build_cache()` is now just a filter over
the same precomputed index the live path uses, rather than its own
separate pass — building the cache and skipping ungated candidates
live are the same underlying optimization, applied in two different
places.

V5, V7, V8, and V9 are deliberately **not** part of this cache and are
always
computed live: V5's vote also depends on `current` (a different
`current` means a different witness set for the same pair); V7/V8
aren't a per-token contribution at all — each is a single fixed-pair
(V7) or fixed-triple (V8) check on `(previous, current)`; V9 is a
conjunction ACROSS every context token for one candidate, not a sum
of independent per-`(token, candidate)` terms. All four are still
fast, though — V5/V7/V8 are cheap sparse
lookups, and V9 reuses the same co-occurrence gate as the cache
itself, so all four run live whether or not the V1-V4/V6 cache is
enabled.

```python
model = MSEGraphLanguageModel.load("runs/model")

# Off by default -- identical scores either way, this is purely a
# speed optimization. Build it for the exact candidate set Open Mode
# actually uses every step (the full vocabulary):
model.open_ctm.enable_cache(model.all_candidate_tokens())

info = model.open_mode_candidate_scores("the cat sat on the")
info["cache_used"]   # True -- this breakdown was served from the cache

model.open_ctm.disable_cache()   # back to live scoring; built cache
                                  # data is kept, so enabling again
                                  # later (with no args) is instant
```

Passing a `candidates` set the cache *wasn't* built for (e.g. Strict
Mode's narrower successor set) falls back to the live path
automatically — never a wrong answer, just not the fast one; the
`"cache_used"` field in the trace always reports which path actually
ran. The cache stores raw evidence counts, not pre-weighted votes,
so changing a weight (`important_weight`, `context_weight`, ...)
after building it does **not** stale it. It's also not persisted by
`to_dict()`/`from_dict()` — it's fully re-derivable from state that
already is, so a restored `ImportanceVoteMatrix` always starts with
caching off; call `enable_cache()` again if you want it back.

CLI/REPL equivalents: `analyse.py cache [on|off|status]` and
`analyse.py open-scores --cache`; `chat.py`'s `/cache on|off|status`.
Run `python3 benchmark_cache.py` to measure the actual speedup on
your own trained model (confirms cached and live scores match
exactly, then times both). Since the co-occurrence gate now speeds
up the LIVE path too, the cache's own marginal benefit over live
scoring is smaller than it used to be, and on a small vocabulary the
cache can even come out slightly slower than gated live scoring (its
per-step bookkeeping overhead no longer has as much dead weight to
save) — the gate is what does most of the heavy lifting now; the
cache is worth measuring on your own model and vocabulary size rather
than assumed.

`ivm.py`'s `ImportanceVoteMatrix` also has a Strict Mode LEGACY role
(`resolve_tie()`) — opt-in only, substituting importance voting for
Strict Mode's Stage 2 lineage tie-break when explicitly passed to
`generate(..., use_importance_votes=True)`. That's a different
object/build than Open Mode's `model.open_ctm` (which is auto-built
alongside training/loading); Strict Mode's is built separately via
`model.build_importance_votes()` and is `None` unless you call it.

---

## 9. Context Trigger Matrix (contextual disambiguation)

The Bridge Matrix answers "which tokens can occupy this slot?"
(cat/dog/pig are interchangeable). The Context Trigger Matrix answers
"given this surrounding context, which one SHOULD?" — built from
whole-sentence co-occurrence, no new training. This is a **Strict
Mode** opt-in disambiguation aid (see section 8 for Open Mode's
actual primary mechanism, which doesn't use this at all).

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
falls back to the same deterministic bigram-frequency tie-break
Strict Mode always uses if the context gives it no signal — never
randomness; see section 8's table and `inference.py`'s
`_bigram_tie_break`. Look for `"rule": "context_trigger_resolved"` in
the trace to see exactly when it fired.

```python
model.has_context_triggers()   # bool
```

---

## 10. Token Importance / Trigger analysis (Python API only)

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
section 9 (whole-sentence window) — they're deliberately different
granularities of the same underlying idea.

---

## 11. Analyse raw text with no trained model

```bash
python3 analyse.py corpus --text "the cat sat on the mat." --top 10
python3 analyse.py corpus --file corpus.txt --top 10
```

The only subcommand that doesn't need `--model` — just word/sentence
statistics over raw text.

---

## 12. Run the test suite

```bash
python3 test.py
```

No flags. Runs the full regression suite (tokenizer, graph
construction, generation determinism, Open Mode's vocabulary-as-
candidates architecture and any-prompt-accepted behavior, IVM
weighted voting incl. V5 bigram-witness, V6 adjacency, V7
prev+current co-occurrence, V8 triple witness, and V9 whole-context
unanimous vote, the co-occurrence gate, the sparse per-token
score cache, Cluster Interpreter, zero-cluster mining, Token
Importance analysis, Context Trigger Matrix, incremental training,
large-corpus pipeline, save/load round-trips) and prints a final
`N passed, 0 failed` summary.

---

## Typical end-to-end session

```bash
# 1. Train on a folder of files
python3 train_corpus.py --corpus-dir data/ --out runs/model --vocab-size 4000

# 2. Look at what it learned
python3 analyse.py --model runs/model clusters --top 20
python3 analyse.py --model runs/model interpreter-matrix --min-coverage 0.5 --min-signals 2
python3 analyse.py --model runs/model context-triggers --min-support 2

# 3. Open Mode is already ready -- no build step. If you want the
#    (optional, opt-in) Context Trigger Matrix for Strict Mode too:
python3 -c "from model import MSEGraphLanguageModel as M; m=M.load('runs/model'); \
    m.build_context_triggers(); print('ok')"
# (CTM isn't persisted by save()/load() -- rebuild it each session, or
#  add your own caching around ContextTriggerMatrix.to_dict()/from_dict())

# 4. Chat with it, or serve it over HTTP
python3 chat.py --model runs/model --mode open
python3 server.py --model runs/model --mode open --port 5000   # needs flask

# 5. Audit why it picked what it picked
python3 analyse.py --model runs/model open-scores "the farmer fed the"
python3 analyse.py --model runs/model bigram the pig --mode open

# 5b. Speed up repeated Open Mode scoring/generation over the same
#     vocabulary (opt-in, identical scores either way):
python3 analyse.py --model runs/model cache on
python3 benchmark_cache.py   # measure it yourself on your own model

# 6. Add more data later, in place
python3 train.py --continue-from runs/model --corpus more_data.txt \
    --extend-vocab --target-vocab-size 6000
# Open Mode is automatically rebuilt from the merged graphs -- no
# action needed. Only the Context Trigger Matrix is invalidated:
# (rebuild CTM in your own script/session too, per step 3, if you use it)
```

---

## Notes and gotchas

- **`train()`/`train_incremental()` used to be effectively O(triples²)**
  because `BridgeMatrix.cluster_axis()` rescanned every triple in the
  graph on every call, and `open_ctm`'s automatic rebuild (part of
  every train/incremental call) calls it once per clustered triple.
  Fixed by giving `cluster_axis()` a lazy `cluster_id -> [triple_idx]`
  index plus a per-cluster memoized result (same fix already applied
  to `RelationshipMatrix.relationships_for_triple()` — see that
  method's comment in `graph.py`), so it's now O(triples) overall
  instead of O(triples) *per clustered triple*. On this repo's own
  benchmark corpus that took training from ~67s down to ~0.36s at a
  few thousand sentences, and it now scales roughly linearly with
  corpus size instead of quadratically — the larger your corpus, the
  more this mattered. No behavior changed; `test.py`'s
  `test_cluster_axis_indexed_lookup()` checks the indexed version
  against the old brute-force definition directly.
- **`--json` is a path, not a flag**, and belongs on the *main*
  `analyse.py` parser — it must come before the subcommand:
  `analyse.py --model X --json out.json clusters`, not
  `analyse.py --model X clusters --json out.json`.
- **Incremental training auto-rebuilds Open Mode, but invalidates the
  Context Trigger Matrix.** Open Mode (`self._open`/`self.open_ctm`)
  is automatically kept in sync with the merged graphs after any
  `--continue-from` or `train_corpus.py` run — no action needed. If
  you use `use_context_triggers=True` anywhere, rebuild the CTM
  after merging; `train_incremental()`'s return value tells you
  whether it was affected: `summary["ctm_invalidated"]`.
- **The Relationship Matrix deduplicates by literal sentence content**,
  the same principle the Edge Matrix already applies to bigrams and
  the Bridge Matrix already applies to triples (see `graph.py`'s
  `RelationshipMatrix` docstring). Two training sentences with the
  IDENTICAL token sequence get exactly one `relationship_id`, not two;
  `rel_count` (queryable via `model.rels.count(rel_id)`) tracks how
  many literal occurrences share that content. `model.stats()`
  reflects this split: `"relationships"` is the unique-sentence count,
  `"relationship_occurrences"` is the raw total — on a corpus with no
  repeated sentences the two numbers are identical, but they diverge
  the moment any sentence repeats verbatim. This also means
  `ctm.py`'s trigger "support" counts and `importance.py`'s
  `trigger_matrix()`'s `distinct_sequences` now count distinct
  sentence CONTENT, not distinct raw occurrences — a sentence
  repeated N times in the corpus contributes support/distinctness 1,
  not N. `train_incremental()`'s merge honors this too: re-feeding an
  exact repeat of an already-known sentence collapses onto its
  existing `relationship_id` (incrementing its `rel_count`) instead
  of minting a new one — see that method's docstring for the one
  known limitation (a pre-existing sentence too short to have any
  triples at all, under 3 tokens, can't be reconstructed from the
  Bridge Matrix during a merge and may collapse with other such
  short sentences; harmless in practice since a triple-less sentence
  casts no votes anywhere in this codebase).
- **The Context Trigger Matrix is not persisted by `save()`/`load()`.**
  It's cheap enough to rebuild per session
  (`model.build_context_triggers()`) but if you want it cached to
  disk, `ContextTriggerMatrix.to_dict()`/`from_dict()` are
  JSON-safe — wire your own save/load around them if needed.
- **The sparse per-token score cache is off by default and not
  persisted by `save()`/`load()` either.** It's fully re-derivable
  from state that already is, so a freshly loaded model's `open_ctm`
  always starts uncached — call
  `model.open_ctm.enable_cache(model.all_candidate_tokens())` (or
  `analyse.py cache on` / `chat.py`'s `/cache on`) once per session
  if you want it. It only covers V1/V2/V3/V4/V6 (see section 8); V5,
  V7, and V8 are always computed live, cached or not.
- **`use_context_triggers=True` only ever changes behavior at a
  genuine tie.** It never overrides a unique, structurally-determined
  answer, and it falls back to the deterministic bigram-frequency
  tie-break whenever it has no signal. Default is `False` everywhere.
- **There is no randomness anywhere in generation, in either mode.**
  Every tie — Strict Mode's lineage pipeline, Open Mode's IVM
  scoring, the Context Trigger Matrix add-on — resolves through a
  deterministic cascade (see section 8), down to lowest-token-id as
  the final, always-available fallback. `random` is not imported
  anywhere in the core library.
- **`model.open_ctm` (Open Mode's IVM, the primary mechanism) and
  `model.ivm` (Strict Mode's legacy opt-in tie-break) are different
  objects, built separately.** `open_ctm` is auto-built alongside
  training/loading (see `model.py`'s `_rebuild_open_engine()`);
  `ivm` is `None` unless you explicitly call
  `model.build_importance_votes()`. Both are `ImportanceVoteMatrix`
  instances from `ivm.py`, and since there is no longer a separate
  Experience data source at all, they're built from identical
  evidence either way — bigram-witness evidence (V5, `_bigram_rels`)
  and adjacency evidence (V6, `_adjacent`) are the same for both.
- **Open Mode has no successor gating at all anymore.** Candidates
  for every generation step are the entire vocabulary
  (`model.all_candidate_tokens()` / `self._open.vocab`), scored by
  IVM's six weighted vote layers — not narrowed by whether a bigram
  was ever literally observed. There is no toggle for this; it's
  simply how Open Mode works now. The PROMPT itself still has to
  start from a literal training bigram, in both modes (same
  Edge-Matrix-based check either way).
- **`model._engine(mode)` validates `mode` strictly** — only
  `"strict"` or `"open"` are accepted; anything else raises
  `ValueError` rather than silently falling back to Strict Mode.
  Matters most for API callers that pass an unchecked mode string
  (e.g. `server.py`'s `/bigram` `mode` field).
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
- **`server.py` is a development server** (Flask's built-in `app.run`,
  not a production WSGI server) with a simple in-memory rate limiter
  and in-memory sessions — fine for local use or a demo, not
  hardened for production traffic. Put a real WSGI server (gunicorn,
  etc.) in front of it for anything public-facing.
