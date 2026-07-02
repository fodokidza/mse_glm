# MSE Graph Language Model (MSE-GLM)

IS A DUAL-MODE ENGINE
1. STRICT MODE
2. OPEN MODE

**Deterministic · Explainable · Zero learned weights · CPU only**

A language model with no neural network. Language is represented as a
token-transition graph. Every generation decision is traceable back to the
exact rule that produced it. No gradients, no GPU, no black box.

Author: Clifford Chivhanga

Email: cliffordchivhanga318@gmail.com

For more in info: https://aircityshops.com/index.php?url=city/mse_blog

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
$ python3 chat.py --model runs/model
you> the dog
model> the dog sat on the carpet

you> /shared cat dog
model> cat dog -> sat  [bridge_axis, cluster_id=1, overlap=2]

you> /explain the | dog
model> next='sat'  stage=1  rule=storage_order_fallback  active_rels={1}
```

**Status:** Implemented and tested — 56 / 56 automated checks passing.

---

## What this is

- **Zero learned weights** — a BPE tokenizer feeds a deduplicated bigram/trigram graph. No floats, no backprop, no framework.
- **Fully deterministic** — same input, same output, every time. No sampling, no temperature.
- **Fully explainable** — every generated token traces back to the exact pipeline stage, rule, and candidate set that produced it, including tie-breaks.
- **CPU only** — array-backed, CSR-indexed storage. Runs on a Raspberry Pi.
- **Distributional similarity without embeddings** — dual-axis cluster assignment identifies interchangeable tokens structurally, with no embedding model.

## What this is not

Not a transformer competitor for open-domain generation, reasoning, or long-range coherence. It has no semantic understanding — only observed token adjacency and structural slot-sharing. It does not generalise to unseen transitions.

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
deduplicated bigrams   deduplicated trigrams       (triple_id, rel_id) only
CSR-indexed by source  + dual-axis cluster_id      no triple content
                       + T_index                   duplicated
  │
  ▼
Inference Engine
  Stage 1 — Exact Bridge Match  (lineage-aware tie-break via R)
  Stage 2 — Bridge Voting       (2× weight if bridge matches current)
  Stage 3 — Bigram Voting       (Edge Matrix fallback)
  Stage 4 — Termination         (emit <EOS>)
  +  infer_shared_role()        (unordered cluster-based query)
  │
  ▼
MSEGraphLanguageModel
generate · explain_step · infer_shared_role · save / load
```

| File | Role |
|---|---|
| `tokenizer.py` | BPE tokenizer — special tokens, normalization, in-memory or streamed training |
| `graph.py` | `EdgeMatrix`, `BridgeMatrix` (dual-axis `cluster_id` + `T_index`), `RelationshipMatrix` |
| `inference.py` | Four-stage deterministic pipeline, lineage-aware tie-break, `infer_shared_role()` |
| `model.py` | `MSEGraphLanguageModel` — orchestrates everything, `train / generate / explain / save / load` |
| `analyse.py` | Library + 12-subcommand CLI for corpus stats, topology, clusters, relationships, traces |
| `train.py` | CLI training pipeline with live phase-by-phase progress display |
| `chat.py` | Interactive REPL — generation, `/explain`, `/shared`, `/clusters`, `/stats` |
| `test.py` | 56 automated regression checks |

Full design rationale, worked examples, and complexity analysis: [`docs/SDD_v2_1.pdf`](docs/SDD_v2_1.pdf)

---

## Quickstart

**No pip installs required.** Python 3.8+ standard library only.

```bash
git clone https://github.com/fodokidza/mse_glm.git
cd mse_glm

# train from a file (streamed — not bounded by RAM)
python3 train.py --corpus path/to/corpus.txt --out runs/model --vocab-size 2000

# or from inline text
python3 train.py --text "the cat sat on the mat. the dog sat on the carpet." \
                 --out runs/demo --vocab-size 200

# chat with it
python3 chat.py --model runs/model

# run the tests
python3 test.py
```

### Python API

```python
from model import MSEGraphLanguageModel

m = MSEGraphLanguageModel.load("runs/model")

# generate
text, ids, trace = m.generate("the dog", max_tokens=20)
print(text)   # "the dog sat on the carpet"

# explain every step
for step in trace:
    print(step["stage"], step["rule"], step["chosen_token"])

# cluster-based shared-role inference
print(m.infer_shared_role(["cat", "dog"]))
# [('sat', 'bridge_axis', {'cluster_id': 1, 'overlap': 2}), ...]
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
| `<any text>` | Generate a continuation |
| `/explain <prev> \| <curr>` | Show stage, rule, candidates for one step |
| `/shared cat dog boy` | Run `infer_shared_role()` — what structural role do these share? |
| `/clusters` | Show top dual-axis cluster groups |
| `/stats` | Vocab / edge / bridge / cluster / relationship counts |
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

## Lineage-aware tie-breaking

The Relationship Matrix `R` stores `(triple_id, relationship_id)` only — no triple content duplicated. A triple shared across multiple training sentences carries one R row per sentence. At inference, `active_rels` is **narrowed by intersection** at each step (never replaced), so the path stays locked to the most specific consistent lineage even when it passes through shared triples.

Without narrowing: `the dog → the dog sat on the mat` (wrong).  
With narrowing: `the dog → the dog sat on the carpet` (correct).

This regression is covered by a dedicated automated test.

---


## License

[AGPL-3.0](LICENSE) — source-available, including for network use.

If AGPL terms do not suit your use case (proprietary embedding, closed commercial service),
a commercial license is available — open an issue or contact the author.

---

## Author

**Clifford Chivhanga**  
[email](cliffordchivhanga318@gmail.com)
