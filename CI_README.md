# Cluster Interpreter (CI) Matrix

A read-only analysis layer for MSE-GLM that proposes human-readable
labels for clusters the model already discovered — e.g. naming the
cluster `{cat, dog, pig}` as **animal** — using only structure already
present in the trained matrices. No embeddings, no external
dictionary, no learned weights, and no effect on generation.

```
cluster_id 1  ({cat, dog, pig})  ──►  "animal"   coverage 1.0
cluster_id 1  ({cat, dog})       ──►  "pet"      coverage 0.667
```

---

## Contents

- [Why this exists](#why-this-exists)
- [Where it sits in the architecture](#where-it-sits-in-the-architecture)
- [How it actually works](#how-it-actually-works)
  - [1. Bridge Matrix, source axis (primary signal)](#1-bridge-matrix-source-axis-primary-signal)
  - [2. Edge Matrix, direct adjacency](#2-edge-matrix-direct-adjacency)
  - [3. Bridge Matrix, shared-role check](#3-bridge-matrix-shared-role-check)
  - [4. Relationship Matrix, robustness](#4-relationship-matrix-robustness)
- [A cluster can carry more than one label](#a-cluster-can-carry-more-than-one-label)
- [Mining cluster_id==0 (the "3rd axis")](#mining-cluster_id0-the-3rd-axis)
- [API](#api)
  - [interpret.py (raw, token-id level)](#interpretpy-raw-token-id-level)
  - [model.py (decoded, human-readable)](#modelpy-decoded-human-readable)
- [CLI](#cli)
- [Output schema](#output-schema)
- [Worked example](#worked-example)
- [Honesty notes and known limitations](#honesty-notes-and-known-limitations)
- [Testing](#testing)
- [Files touched](#files-touched)

---

## Why this exists

MSE-GLM's Bridge Matrix already groups interchangeable tokens into
clusters (e.g. `cat`/`dog`/`pig` as interchangeable subjects before
"sat"), but a cluster_id is just a number — it carries no semantic
meaning on its own. CI adds a naming layer on top: given a cluster,
propose a real word that describes what its members have in common,
along with a transparent record of *why* that word was proposed.

This is purely additive. It doesn't change `train.py`, `generate()`,
`step()`, or any inference path — it only reads matrices that already
exist.

## Where it sits in the architecture

```
train.py     → builds Edge / Bridge / Relationship matrices (unchanged)
inference.py → generation, uses those matrices directly     (unchanged)
analyse.py   → read-only analysis layer                     (existing)
interpret.py → read-only analysis layer                     (NEW — same category as analyse.py)
```

`interpret.py` is called by `model.py` wrapper methods, which decode
token ids to strings; `analyse.py` exposes both a Python API
(`Analyser` class) and CLI subcommands on top of that.

## How it actually works

A cluster's members already share **one** structural slot — the
bridge or target position between some fixed source/other-endpoint
pair (`BridgeMatrix.cluster_axis`, unchanged, pre-existing). CI looks
for a **second**, independent slot the same members share, and
gathers corroborating evidence for it from three matrix families.

### 1. Bridge Matrix, source axis (primary signal)

Fix `(bridge, target)`, let `source` vary. If cluster members are
each the *source* of a triple ending in the same `(bridge, target)`
pair — e.g. `cat`, `dog`, `pig` are each the source of an "is animal"
triple — that target token is a candidate interpreter.

**`coverage`** = fraction of cluster members that reach the candidate
this way. This is the mechanism that actually finds the label;
everything below corroborates or filters it.

> **Window-size constraint:** a triple is exactly 3 tokens
> (`source → bridge → target`). `"cat is animal"` fits (3 tokens);
> `"cat is an animal"` does not — `"animal"` is 4 tokens from `"cat"`,
> outside any single triple's reach. Corpus phrasing matters here in a
> way it doesn't for generation.

### 2. Edge Matrix, direct adjacency

A Bridge triple `(source, bridge, target)` only guarantees the two
adjacent bigrams `source→bridge` and `bridge→target` exist — **not**
`source→target` directly. So checking whether the corpus *also*
contains a direct `source→target` bigram somewhere (skipping the
bridge word entirely, e.g. a stray `"cat animal"` alongside
`"cat is animal"`) is genuinely separate information, not a
restatement of signal 1. It's a strict, narrow check for that reason —
it rarely fires, and when it does, the corpus states the same fact two
different ways.

### 3. Bridge Matrix, shared-role check

Independent of the specific `(bridge, target)` pair used above: does
the candidate interpreter token *itself* turn up as an interchangeable
member of some **other** dual-axis cluster alongside one of these
members? (`t_index[token]` = the non-zero cluster_ids that token
participates in as a bridge or target anywhere in the graph.) If
`"animal"` and `"cat"` share a cluster_id somewhere else entirely,
that's a structural family resemblance found by a completely
different route than signal 1.

### 4. Relationship Matrix, robustness

Not "relatedness between distinct contexts" — the training corpus's
`rel_id` is one per training sentence, so this checks something more
modest and honest: how many **distinct training sentences** assert the
specific fact behind a candidate
(`RelationshipMatrix.relationships_for_triple`). A candidate backed by
one sentence could be a fluke; backed by several is more durable.

---

## A cluster can carry more than one label

Nothing requires a cluster to have exactly one interpreter. `{cat,
dog, pig}` can be `"animal"` (all three) **and**, independently, the
`{cat, dog}` subset within it can support `"pet"` (partial coverage,
since the corpus never calls `pig` a pet) — both survive filtering as
long as each independently clears the thresholds. `build_interpreter_matrix`
returns one row per **`(cluster_id, interpreter)`** pair, not one row
per cluster.

---

## Mining cluster_id==0 (the "3rd axis")

Everything above interprets clusters that `BridgeMatrix` already
assigned a real `cluster_id` to. But the standard dual-axis rule only
implements two clustering rules, both keyed on a **fixed source**:

- bridge axis: fix `(source, target)`, `bridge` varies
- target axis: fix `(source, bridge)`, `target` varies

There's no rule for **"fix `(bridge, target)`, let `source` vary."**
So a set of tokens that are each individually the source of one
`"X is animal"`-shaped triple, but never happen to co-occur in any
*other* shared context (no shared verb-cluster, nothing), gets **no
cluster_id at all** — not a bad one, none. Every one of them sits at
`cluster_id == 0` individually, and is completely invisible to
`cluster_report()`, `interpret_all_clusters()`, and
`build_interpreter_matrix()` — all three only ever look at
`cluster_id != 0`.

`discover_zero_cluster_groups` is exactly that missing third rule,
applied only to the rows the first two rules left at `0`:

```python
model.discover_zero_cluster_groups(min_group_size=2, mode="strict")
```

```bash
python3 analyse.py --model <path> zero-cluster --min-group-size 2 [--mode open] [--json]
```

**Confirmed empirically:** built a corpus where `cat`/`dog`/`pig` each
get a different verb (`slept`/`barked`/`rolled`) and a different
lead-in word before their `"is animal"` statement, specifically so
they'd share nothing under the standard rule:

```bash
python3 train.py --text "the cat slept on a mat. the dog barked at a car. \
the pig rolled in a puddle. well cat is animal. sure dog is animal. \
hey pig is animal." --out /tmp/zero_demo --vocab-size 200 --quiet

python3 analyse.py --model /tmp/zero_demo clusters
```
```
cluster_id  axis    slot                 members
---------------------------------------------------------
4           bridge  a -> ___ -> <EOS>    car, mat, puddle
1           bridge  <BOS> -> ___ -> cat  the, well
2           bridge  <BOS> -> ___ -> dog  sure, the
3           bridge  <BOS> -> ___ -> pig  hey, the
```

Confirmed — nothing groups `cat`/`dog`/`pig` together. `interpret_all_clusters`
never finds `"animal"` either, since it only scans the four cluster_ids
above. But mining `cluster_id == 0` directly:

```bash
python3 analyse.py --model /tmp/zero_demo zero-cluster --min-group-size 2
```
```
interpreter  member_count  evidence_mask                                     members
------------------------------------------------------------------------------------------
animal       3             zero_cluster_source_axis,relationship_robustness  cat, dog, pig
```

Recovered cleanly. In this corpus, 19 of 28 triples (68%) sat unused
in `cluster_id == 0` — mining it surfaced exactly one meaningful
group; the rest either don't form groups of size ≥2 at all or would
need very permissive thresholds to surface.

**Scope note — this is a targeted recovery tool, not a general quality
upgrade.** Most of what's in `cluster_id == 0` is unclustered for good
reason: one-off phrasing, low-frequency combinations, noise. Use
`zero-cluster` when you suspect a corpus has categorical statements
about tokens that never got a "real" cluster any other way, and always
check the results by eye — `min_group_size` is a floor, not a
relevance guarantee, same as `coverage` elsewhere in this feature.
It also hasn't been tested on anything larger than a 6–10 sentence toy
corpus; how noisy it gets at real scale is still an open question.

Note the output schema differs slightly from `build_interpreter_matrix`
rows: there's no `cluster_id` or `coverage` field, since these groups
never had a real cluster_id or a pre-existing member set to compute
coverage against — `member_count` and `evidence_mask` (starting from
`zero_cluster_source_axis` instead of `bridge_source_axis`) stand in
for them.

---

## API

### interpret.py (raw, token-id level)

```python
find_cluster_members(bridge_matrix, cluster_id)
# -> (axis, [member_token_id, ...])

interpret_cluster(model, cluster_id, top_n=5, mode="strict")
# -> {"cluster_id", "axis", "members", "candidates": [...]} or None

interpret_all_clusters(model, min_coverage=0.5, max_per_cluster=3, mode="strict")
# -> [ {..., "candidates": [top matches clearing min_coverage]}, ... ]

build_interpreter_matrix(model, min_coverage=0.5, min_signals=2,
                          max_per_cluster=None, mode="strict")
# -> [ {"cluster_id", "interpreter_token", "coverage", "evidence_mask", ...}, ... ]
# one row per (cluster_id, interpreter) pair that clears BOTH thresholds

discover_zero_cluster_groups(model, min_group_size=2, mode="strict")
# -> [ {"interpreter_token", "via_bridge_token", "members", "member_count",
#       "evidence_mask", ...}, ... ]
# mines cluster_id==0 for groups the standard dual-axis rule never
# assigns a cluster_id to at all (see "Mining cluster_id==0" above)
```

### model.py (decoded, human-readable)

```python
model.interpret_cluster(cluster_id, top_n=5, mode="strict")
model.interpret_all_clusters(min_coverage=0.5, max_per_cluster=3, mode="strict")
model.build_interpreter_matrix(min_coverage=0.5, min_signals=2,
                                max_per_cluster=None, mode="strict")
model.discover_zero_cluster_groups(min_group_size=2, mode="strict")
```

`mode="open"` additionally folds in the Experience matrices
(`exp_bridges` / `exp_edges` / `exp_rels`) wherever the strict-mode
matrices are consulted.

---

## CLI

```bash
# All candidates for one cluster, unfiltered, ranked best-first
python3 analyse.py --model <path> interpret <cluster_id> --top 10 [--mode open] [--json]

# Every candidate clearing --min-coverage, up to --max-per-cluster per cluster
python3 analyse.py --model <path> interpretations \
    --min-coverage 0.5 --max-per-cluster 3 [--mode open] [--json]

# The filtered CI Matrix: coverage AND evidence_mask size both required
python3 analyse.py --model <path> interpreter-matrix \
    --min-coverage 0.5 --min-signals 2 --max-per-cluster <N or omit for unbounded> \
    [--mode open] [--json]

# Mine cluster_id==0 for groups the standard rule never assigns an id to
python3 analyse.py --model <path> zero-cluster --min-group-size 2 [--mode open] [--json]
```

To persist the matrix, pipe the JSON form to a file — there's no
separate save/load subsystem, this follows the same pattern every
other artifact in the project uses:

```bash
python3 analyse.py --model <path> interpreter-matrix --min-coverage 0.5 --min-signals 2 \
    --json > <path>/interpreter_matrix.json
```

---

## Output schema

Each interpretation candidate (whether from `interpret_cluster`,
`interpret_all_clusters`, or a row of `build_interpreter_matrix`):

| field                  | type          | meaning                                                             |
|------------------------|---------------|----------------------------------------------------------------------|
| `cluster_id`           | int           | which cluster this row interprets                                    |
| `axis`                 | str           | `"bridge"` or `"target"` — how the cluster was originally formed     |
| `members` / `members_covered` | [str]   | full cluster members / just the ones this specific candidate reaches |
| `interpreter_token`    | str           | the proposed label                                                    |
| `via_bridge_token`     | str           | the bridge word connecting members to the label                      |
| `coverage`             | float         | `len(members_covered) / len(members)` — a plain fraction, **not** a calibrated probability |
| `edge_corroborated`    | bool          | signal 2 (direct adjacency) fired                                     |
| `shared_role_overlap`  | [int]         | cluster_ids the label shares with a member elsewhere (signal 3)      |
| `relationship_ids`     | [int]         | distinct training sentences asserting this fact (signal 4)           |
| `evidence_mask`        | [str]         | which of the four signals fired, e.g. `["bridge_source_axis", "relationship_robustness"]` |

---

## Worked example

```bash
python3 train.py --text "the cat sat on the mat. the dog sat on the carpet. \
the boy sat on the mat. cat is animal. dog is animal. pig is animal. \
cat is pet. dog is pet. the boy ran on the road. the pig ran on the road." \
--out runs/demo --vocab-size 200 --quiet

python3 analyse.py --model runs/demo interpreter-matrix --min-coverage 0.5 --min-signals 2
```

```
cluster_id  interpreter  coverage  evidence_mask                               members_covered
----------------------------------------------------------------------------------------------
1           animal       1.0       bridge_source_axis,relationship_robustness  cat, dog, pig
2           on           1.0       bridge_source_axis,relationship_robustness  cat, dog, boy
4           on           1.0       bridge_source_axis,relationship_robustness  boy, pig
6           the          1.0       bridge_source_axis,relationship_robustness  sat, ran
7           on           0.75      bridge_source_axis,relationship_robustness  cat, dog, boy
7           animal       0.75      bridge_source_axis,relationship_robustness  cat, dog, pig
1           on           0.667     bridge_source_axis,relationship_robustness  cat, dog
1           pet          0.667     bridge_source_axis,relationship_robustness  cat, dog
...
```

Note cluster 1 appearing three times (`animal`, `on`, `pet`) — each
row is an independently-qualifying label, not competing picks.

---

## Honesty notes and known limitations

These are also documented directly in `interpret.py`'s module
docstring — keep them there if you edit the file.

- **This surfaces labels already latent in the corpus.** If the corpus
  has no categorical/hypernym-shaped sentence for a cluster's members,
  no candidate will be found. That's the correct outcome, not a bug.
- **`coverage` is a plain fraction, not a calibrated probability.**
  Don't present it to a user as a percent-confidence figure.
- **The four signals are correlated, not independent votes.** They're
  all read off the same underlying training sequences to different
  degrees. `evidence_mask` reports which checks passed for
  transparency; the *count* of passing checks is not a confidence
  percentage and must not be treated as one.
- **High coverage ≠ semantic relevance.** Grammatical continuations
  (prepositions, articles) can win on coverage exactly as easily as a
  real category word — the method knows graph structure, not English
  grammar. In the shipped test corpus (8–10 toy sentences), requiring
  `min_signals=2` does **not** yet separate `"animal"` from a
  grammatical `"on"` cluster: `relationship_robustness` fires for
  both once each is attested in ≥2 sentences, and `shared_role_overlap`
  comes back empty everywhere because the vocabulary is too small for
  any token to double up across clusters. This is asserted directly in
  `test.py` (a test explicitly checks that `"on"` is *not yet*
  excluded) so a future scoring change has to touch that test
  consciously rather than silently changing behavior.
  **Always sanity-check the top-N candidates for a cluster by eye —
  don't trust rank-1 blindly, especially from `interpret_all_clusters`.**
- **Uncapped `max_per_cluster` can grow long tails.** Once you widen
  `min_coverage` / lower `min_signals` on a real (larger) corpus, a
  cluster with many loosely-related tokens could accumulate many
  low-signal labels. The toy corpora used for testing are too small to
  say whether your chosen thresholds hold up at scale — verify against
  real training data before trusting default thresholds in production.
- A possible future signal, not implemented: **bridge-token
  specificity** — how many unrelated `(source, target)` pairs a bridge
  token like `"is"` connects to corpus-wide, versus a narrower
  verb+preposition combo. This needs a bigger, more varied corpus to
  even test meaningfully.
- **`discover_zero_cluster_groups` has a much larger candidate space
  than everything else in this file.** It scans the *entire*
  unclustered bucket rather than starting from an already-known
  cluster's membership, so on a bigger, noisier corpus expect more
  coincidental groupings — two unrelated tokens that happen to share
  one throwaway bigram-adjacent pattern. Treat it as a targeted
  recovery tool for a specific known gap (see the dedicated section
  above), not a general-purpose second interpreter matrix to run by
  default. It's only been verified on toy corpora (6–10 sentences);
  behavior at real scale is untested.

---

## Testing

```bash
python3 test.py
```

Look for the `=== Cluster Interpreter (CI) ===` section. It covers:

- basic discovery (`cat/dog/pig` → `animal`, full coverage)
- `evidence_mask` always contains `bridge_source_axis`
- `relationship_ids` / `relationship_robustness` populate correctly
- unknown `cluster_id` returns `None` rather than raising
- no interpreter is fabricated when the corpus has no categorical data
- the documented `min_signals=2` limitation (does not yet exclude
  grammatical `"on"`)
- multi-label behavior: `{cat, dog, pig}` keeps **both** `animal` and
  `pet` as separate qualifying rows, and `pet`'s `members_covered` is
  correctly `{cat, dog}`, not the full cluster

Also look for `=== Zero-cluster mining (3rd axis) ===`. It covers:

- a corpus where `cat`/`dog`/`pig` are constructed to share nothing
  under the standard dual-axis rule, confirming they get no shared
  `cluster_id` and that `interpret_all_clusters` can't find `"animal"`
  for them
- `discover_zero_cluster_groups` recovers `{cat, dog, pig} → animal`
  from `cluster_id == 0` directly
- the recovered group's `evidence_mask` correctly starts with
  `zero_cluster_source_axis`
- no group ever proposes a member (or the bridge token) as its own
  interpreter (self-reference guard)

All existing regression checks (tokenizer, graph construction,
generation determinism, Experience Matrix, Open Mode, etc.) are
unaffected — this feature only adds new sections, it doesn't touch
any existing test.

## Files touched

| file           | change                                                                 |
|----------------|-------------------------------------------------------------------------|
| `interpret.py` | **new** — evidence gathering, matrix construction, and zero-cluster mining, all at token-id level |
| `model.py`     | new wrapper methods: `interpret_cluster`, `interpret_all_clusters`, `build_interpreter_matrix`, `discover_zero_cluster_groups` (decode token ids to strings) |
| `analyse.py`   | new `Analyser` methods + 4 new CLI subcommands: `interpret`, `interpretations`, `interpreter-matrix`, `zero-cluster` |
| `test.py`      | new `=== Cluster Interpreter (CI) ===` and `=== Zero-cluster mining (3rd axis) ===` sections |

`graph.py`, `inference.py`, `experience.py`, `tokenizer.py`,
`train.py`, `chat.py`, `build_experience.py` are **untouched**.
