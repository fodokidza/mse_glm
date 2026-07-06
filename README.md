# MSE Graph Language Model (MSE-GLM)

**Deterministic · Explainable · Zero learned weights · CPU only**

A language model with no neural network. Language is represented as a
token-transition graph. Every generation decision is traceable back to the
exact rule that produced it. No gradients, no GPU, no black box.

```
$ python3 train.py --corpus corpus.txt --out runs/model

  MSE Graph Language Model  ─  Training
  corpus   corpus.txt  (14,823 bytes)
  vocab    1000  target tokens
  output   /home/user/runs/model

  Phase  Tokenizer (BPE)       [████████████████████████████████]  100%  step 756/756   0.31s
  Phase  Edge Matrix (E)       [████████████████████████████████]  100%  620 unique bigrams  0.08s
  Phase  Bridge Matrix (B)     [████████████████████████████████]  100%  891 unique triples  0.11s
  Phase  Cluster Assignment    [████████████████████████████████]  100%  14 clusters  0.04s
  Phase  Relationship Matrix   [████████████████████████████████]  100%  1,243 rows  0.06s
  Phase  Saving Model          [████████████████████████████████]  100%

  ✓  Training complete
  Vocabulary        1,000 tokens
  Edge Matrix         620 unique bigrams
  Bridge Matrix       891 unique triples
  Clustered triples   412  (14 clusters)
  Relationship rows 1,243  (38 sentences)
  Total time          0.64s
```

```
$ python3 chat.py --model runs/model --mode open
you> the dog
model> the dog sat on the carpet

you> /shared cat dog
model> cat dog -> sat  [bridge_axis  {'cluster_id': 1, 'overlap': 2, 'source': 'training'}]

you> /explain the | dog
model> next='sat'  {'stage': 1, 'rule': 'bridge_lineage_unique', 'chosen': 12, 'active_rels': {1}, 'candidates': [12]}
```

**Status:** Implemented and tested — full regression suite passing, covering
the core Strict-Mode pipeline plus the Experience Matrix / Open Mode subsystem.

---

## What this is

- **Zero learned weights** — a BPE tokenizer feeds a deduplicated bigram/trigram graph. No floats, no backprop, no framework.
- **Fully deterministic** — same input, same output, every time. The only randomness is a uniform tie-break among candidates already judged equally valid; it never changes *which* candidates are valid.
- **Fully explainable** — every generated token traces back to the exact pipeline stage, rule, and candidate set that produced it, including tie-breaks.
- **CPU only** — array-backed, CSR-indexed storage. Runs on a Raspberry Pi.
- **Distributional similarity without embeddings** — dual-axis cluster assignment identifies interchangeable tokens structurally, with no embedding model.
- **Two inference modes** — **Strict Mode** (training data only) and **Open Mode** (training data plus structurally-inferred Experience Matrices), sharing one engine.

## Best use cases 

Best suited to **grammar-constrained generation** (SQL, JSON, config files, autocomplete, assembly), **embedded / low-resource environments**, and settings where **auditability and reproducibility** matter more than fluency.

---

## Architecture

```
Corpus
  │
  ▼
Tokenizer (BPE)           from-scratch, streaming, no RAM limit
  │
  ▼
Sentence Splitting         on  .  !  ?  \n
  │
  ├──────────────────────────────────────────────────────────┐
  ▼                         ▼                                ▼
Edge Matrix (E)        Bridge Matrix (B)          Relationship Matrix (R)
deduplicated bigrams   deduplicated triples        (triple_id, rel_id) only
CSR-indexed by source  + dual-axis cluster_id      no triple content
                       + T_index                   duplicated
  │
  │   (optional, offline — one command: build_experience.py)
  ▼
Experience Builder
  Rule 1 — bridge-axis structural expansion
  Rule 2 — target-axis structural expansion
  → Experience Edge / Bridge / Relationship Matrices (EE, EB, ER)
  │
  ▼
Inference Engine                 (Strict: E,B,R only · Open: + EE,EB,ER)
  Stage 1 — Current-token authority   (bridge-lineage vote via active_rels)
  Stage 2 — Previous-token authority  (exact-triple lineage vote)
  Termination                          (emit <EOS> when no successors)
  +  infer_shared_role()               (unordered cluster-based query)
  │
  ▼
MSEGraphLanguageModel
generate · explain_step · infer_shared_role · token_similarity
train / build_experience / load_experience / save / load / stats
```

| File | Role |
|---|---|
| `tokenizer.py` | BPE tokenizer — special tokens, normalization, in-memory or streamed training |
| `graph.py` | `EdgeMatrix`, `BridgeMatrix` (dual-axis `cluster_id` + `T_index`), `RelationshipMatrix` |
| `experience.py` | `ExperienceEdgeMatrix` / `ExperienceBridgeMatrix` / `ExperienceRelationshipMatrix` + `ExperienceBuilder` — derives Open Mode's structurally-inferred triples |
| `build_experience.py` | Standalone CLI that builds Experience Matrices for a saved model, with a live progress display |
| `inference.py` | Two-stage deterministic pipeline (current-token authority, then previous-token authority), lineage-aware tie-break via `active_rels`, `infer_shared_role()` |
| `model.py` | `MSEGraphLanguageModel` — orchestrates everything, `train / generate / explain / infer_shared_role / token_similarity / save / load` |
| `analyse.py` | Library + 12-subcommand CLI for corpus stats, topology, clusters, relationships, traces |
| `train.py` | CLI training pipeline with live phase-by-phase progress display |
| `chat.py` | Interactive REPL — generation, `/mode`, `/explain`, `/shared`, `/similarity`, `/stats`, `/clusters`, `/exp` |
| `test.py` | Automated regression suite — Strict Mode core (56 original checks) plus full Experience Matrix / Open Mode coverage |

---

## Quickstart

**No pip installs required.** Python 3.8+ standard library only.

```bash
git clone https://github.com/fodokidza/mse-glm.git
cd mse-glm

# train from a file (streamed — not bounded by RAM)
python3 train.py --corpus path/to/corpus.txt --out runs/model --vocab-size 2000

# or from inline text
python3 train.py --text "the cat sat on the mat. the dog sat on the carpet." \
                 --out runs/demo --vocab-size 200

# (optional) derive Experience Matrices for Open Mode
python3 build_experience.py --model runs/model

# chat with it — strict (training data only) or open (+ experience)
python3 chat.py --model runs/model --mode strict
python3 chat.py --model runs/model --mode open

# run the tests
python3 test.py
```

### Python API

```python
from model import MSEGraphLanguageModel

m = MSEGraphLanguageModel.load("runs/model")

# generate (mode="strict" by default, or "open" if experience matrices are loaded)
text, ids, trace = m.generate("the dog", max_tokens=20)
print(text)   # "the dog sat on the carpet"

# explain every step
for step in trace:
    print(step["stage"], step["rule"], step["chosen"])

# cluster-based shared-role inference
print(m.infer_shared_role(["cat", "dog"]))
# [('sat', 'bridge_axis', {'cluster_id': 1, 'overlap': 2, 'source': 'training'}), ...]

# build Experience Matrices in-process and switch to Open Mode
m.build_experience(folder="runs/model")
text, ids, trace = m.generate("the dog", max_tokens=20, mode="open")
```

---

## Analyse

```bash
# corpus stats — no trained model needed
python3 analyse.py corpus --file mycorpus.txt

# graph stats
python3 analyse.py --model runs/model stats
python3 analyse.py --model runs/model topology
python3 analyse.py --model runs/model clusters
python3 analyse.py --model runs/model clusters --axis bridge
python3 analyse.py --model runs/model cluster 1
python3 analyse.py --model runs/model relationships
python3 analyse.py --model runs/model relationship 0
python3 analyse.py --model runs/model token cat
python3 analyse.py --model runs/model similarity cat dog
python3 analyse.py --model runs/model shared cat dog
python3 analyse.py --model runs/model trace "the dog" --max-tokens 12
python3 analyse.py --model runs/model report

# export any command as JSON
python3 analyse.py --model runs/model --json out.json report
```

---

## Chat commands

| Command | What it does |
|---|---|
| `<any text>` | Generate a continuation in the current mode |
| `/mode strict\|open` | Switch modes; switching to `open` auto-builds Experience Matrices on first use if not already present |
| `/explain <prev> \| <curr>` | Show stage, rule, candidates for one step |
| `/shared cat dog boy` | Run `infer_shared_role()` — what structural role do these share? |
| `/similarity <a> <b>` | Cluster-overlap similarity between two tokens |
| `/clusters` | Show top dual-axis cluster groups (training data) |
| `/stats` | Vocab / edge / bridge / cluster / relationship counts (plus experience counts if loaded) |
| `/exp` | Experience Matrix summary (prompts to run `/mode open` first if none loaded) |
| `/quit` | Exit |

---

## Dual-axis clustering

`cluster_id` in the Bridge Matrix groups structurally interchangeable tokens without any embedding model:

- **Bridge axis** — triples sharing `(source, target)` → same cluster. Members are interchangeable *bridges* (middle tokens).
- **Target axis** — triples sharing `(source, bridge)` → same cluster. Members are interchangeable *targets*.
- `cluster_id = 0` — triple matches no other on either axis.

```
Corpus: "the cat sat on the mat.  the dog sat on the carpet."

source  target  bridge  cluster_id
the     sat     cat     1          ← cat and dog share cluster 1
cat     on      sat     0             (same source+target, bridge varies)
sat     the     on      0
on      mat     the     2          ← mat and carpet share cluster 2
the     sat     dog     1             (same source+bridge, target varies)
dog     on      sat     0
on      carpet  the     2
```

`infer_shared_role(["cat","dog"])` intersects T_index sets → shared cluster 1 → predicted: `sat`.

---

## Open Mode: Experience Matrices

Strict Mode only ever emits transitions actually seen in training. **Open Mode**
adds a second, opt-in layer: `build_experience.py` (or `model.build_experience()`)
runs the same dual-axis clusters above through two expansion rules —

- **Rule 1 (bridge axis)** — if token X shares a cluster with an attested bridge
  B in slot `(S, T)`, and B also bridges a different slot `(S2, T2)`, then X can
  bridge `(S2, T2)` too.
- **Rule 2 (target axis)** — the same idea applied to interchangeable targets.

— and saves the results as three separate Experience Matrices
(`experience_edges.json`, `experience_bridges.json`, `experience_relationships.json`).
These are never consulted by Strict Mode; Open Mode simply unions them in.

```
Training: "the cat sat on the mat.", "the dog sat on the carpet.",
          "the boy sat on the mat.", "the boy ran on the road."
          (cat and dog are never observed with "ran")

$ python3 build_experience.py --model runs/model
  Rule 1 complete — 2 new triples
  ...

$ python3 chat.py --model runs/model --mode open
you> the cat
model> the cat ran on the road        ← inferred, never seen in training
```

Run `/exp` in `chat.py`, or `python3 analyse.py --model runs/model stats`
after building, to see Experience Matrix counts alongside training counts.

---

## Lineage-aware tie-breaking

The Relationship Matrix `R` stores `(triple_id, relationship_id)` only — no triple content duplicated. A triple shared across multiple training sentences carries one R row per sentence. At inference, `active_rels` is **narrowed by intersection** at each step (never widened back out), so the path stays locked to the most specific consistent lineage even when it passes through shared triples.

Without narrowing: `the dog → the dog sat on the mat` (wrong).
With narrowing: `the dog → the dog sat on the carpet` (correct).

This regression is covered by a dedicated automated test.

---

## Limitations

- No semantic understanding. No reasoning. Strict Mode does not generalise to unseen transitions; Open Mode generalises only as far as dual-axis clustering can justify.
- Sequential context is limited to `(previous, current)` at the triple level; lineage tracking (`active_rels`) is a global consistency signal, not a longer context window.
- Distributional clustering is slot-substitution only — does not distinguish synonyms from antonyms.
- Tokens touching universal structural positions (`<BOS>`, `<EOS>`) cluster together even when otherwise unrelated.
- Experience Matrix expansion can grow combinatorially for very large, highly-substitutable clusters — there is currently no size cap or pruning step.

---


## License

[AGPL-3.0](LICENSE) — source-available, including for network use.

If AGPL terms do not suit your use case (proprietary embedding, closed commercial service),
a commercial license is available — open an issue or contact the author.

---

## Author

**Clifford Chivhanga**
[github.com/fodokidza](https://github.com/fodokidza)
