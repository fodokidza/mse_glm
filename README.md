# MSE Graph Language Model — v2.1 Implementation

# Author: Clifford Chivhanga

A fully deterministic, graph-based language engine that models language using discrete token transitions instead of continuous neural network weights. Built for ultra-low-resource environments, strict syntax guardrails, and 100% explainable symbolic execution.

Implements SDD v2.0 + the v2.1 addendum: deterministic graph inference,
the Relationship Matrix (lineage-aware tie-breaking), and dual-axis
Bridge Matrix clustering (`infer_shared_role()`).

## Files

| File | Role |
|---|---|
| `tokenizer.py` | BPE tokenizer, special tokens, normalization, encode/decode |
| `graph.py` | `EdgeMatrix`, `BridgeMatrix` (with dual-axis `cluster_id`), `RelationshipMatrix` |
| `inference.py` | Four-stage deterministic pipeline, lineage tie-break, `infer_shared_role()` |
| `model.py` | `MSEGraphLanguageModel` — orchestrates everything, save/load to a model folder |
| `analyse.py` | `CorpusAnalyser`, `Analyser` — stats, topology, cluster reports, traces |
| `train.py` | CLI training pipeline |
| `chat.py` | Interactive REPL to test/chat with a saved model |
| `runs/demo/` | A pre-trained example model you can load immediately |

## Train a model

```bash
# from inline text
python3 train.py --text "the cat sat on the mat. the dog sat on the carpet." --out runs/my_model --vocab-size 500

# from a file (streamed, RAM-bounded per SDD v2.0 §5.4)
python3 train.py --corpus path/to/corpus.txt --out runs/my_model --vocab-size 2000
```

This writes a self-contained folder:

```
runs/my_model/
  vocabulary.json      # BPE vocab + merge rules
  edges.json            # Edge Matrix (E)
  bridges.json          # Bridge Matrix (B) incl. cluster_id + T_index
  relationships.json    # Relationship Matrix (R)
  meta.json             # config + stats summary
```

## Chat / test a model

```bash
python3 chat.py --model runs/demo
```

Inside the REPL:
- type any text → generates a continuation
- `/explain <prev> | <curr>` → shows exactly which stage/rule produced the next token
- `/shared cat dog` → runs `infer_shared_role()` over a set of tokens
- `/stats` → vocab/edge/bridge/cluster/relationship counts
- `/clusters` → top dual-axis cluster groups
- `/quit`

## Programmatic use

```python
from model import MSEGraphLanguageModel

m = MSEGraphLanguageModel.load("runs/demo")
text, ids, trace = m.generate("the cat")
print(text)                              # "the cat sat on the mat"
print(m.infer_shared_role(["cat", "dog"]))  # [('sat', 'bridge_axis', {...}), ...]
```

## Analyse freely (analyse.py CLI)

`analyse.py` is also a standalone CLI for ad-hoc analysis of a corpus or a
trained model — no need to write a one-off script.

```bash
# corpus stats — no trained model needed
python3 analyse.py corpus --file mycorpus.txt --top 10

# everything else needs --model (placed before the subcommand)
python3 analyse.py --model runs/demo stats
python3 analyse.py --model runs/demo topology --top 10
python3 analyse.py --model runs/demo clusters --axis bridge
python3 analyse.py --model runs/demo cluster 1
python3 analyse.py --model runs/demo relationships
python3 analyse.py --model runs/demo relationship 0
python3 analyse.py --model runs/demo token cat
python3 analyse.py --model runs/demo similarity cat dog
python3 analyse.py --model runs/demo shared cat dog
python3 analyse.py --model runs/demo trace "the dog" --max-tokens 12
python3 analyse.py --model runs/demo report

# export any of the above as JSON instead of printing
python3 analyse.py --model runs/demo --json out.json report
```
