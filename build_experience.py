"""
build_experience.py — Standalone Experience Matrix builder for MSE-GLM.

Operates directly on a saved model folder. Loads the training matrices,
runs Rule 1 (bridge-axis expansion) and Rule 2 (target-axis expansion),
and saves three Experience Matrix files alongside the training artefacts.
Does not require any Python import of the model object.

Usage:
    python3 build_experience.py --model runs/my_model
    python3 build_experience.py --model runs/my_model --quiet
    python3 build_experience.py --model runs/my_model --dry-run
"""

import argparse
import json
import os
import sys
import time
import shutil
from collections import defaultdict


# ─── terminal helpers (copied from train.py to keep this file self-contained) ──

def _tw():  return shutil.get_terminal_size((100, 24)).columns
def _clr(): sys.stdout.write("\r\033[K")
def _up(n): sys.stdout.write(f"\033[{n}A")
def _c(t, code): return f"\033[{code}m{t}\033[0m"
def teal(t):   return _c(t, "36")
def amber(t):  return _c(t, "33")
def green(t):  return _c(t, "32")
def dim(t):    return _c(t, "2")
def bold(t):   return _c(t, "1")
def white(t):  return _c(t, "97")
def red(t):    return _c(t, "31")
def purple(t): return _c(t, "35")
def cyan(t):   return _c(t, "96")


# ─── Live display (8 lines) ────────────────────────────────────────────────────

class Display:
    LINES = 9
    HZ    = 25

    def __init__(self, quiet=False):
        self.quiet        = quiet
        self.phase        = ""
        self.step         = 0
        self.total        = 0
        self.t0           = time.time()
        self.current      = dim("─")
        self._log         = [""] * 3
        self._lpos        = 0
        self._last        = 0.0
        self._first       = True

    def start(self):
        if self.quiet: return
        print()
        for _ in range(self.LINES): print()

    def item(self, text, stats=None, force=False):
        self.current = text
        self._log[self._lpos % 3] = text
        self._lpos += 1
        if stats: self.stats = stats
        if not self.quiet:
            now = time.time()
            if force or (now - self._last) >= 1/self.HZ:
                self._draw(); self._last = now

    def done(self):
        if not self.quiet: self._draw(final=True)

    def _draw(self, final=False):
        tw  = _tw()
        el  = time.time() - self.t0
        pct = int(min(self.step / self.total, 1.0) * 100) if self.total else 0
        bar_w   = 30
        filled  = int(bar_w * pct / 100)
        bar     = "█" * filled + "░" * (bar_w - filled)
        rule    = dim("─" * min(tw - 4, 72))
        log     = [self._log[(self._lpos - 3 + i) % 3] for i in range(3)]

        lines = [
            "",
            f"  {bold(white('MSE-GLM Experience Builder'))}",
            f"  {rule}",
            f"  {teal(self.phase):<40}  [{teal(bar)}]  {amber(f'{pct:3d}%')}  {dim(f'{self.step}/{self.total}')}",
            f"  {rule}",
            f"  {cyan('▸')} {self.current}",
            f"    {log[0]}",
            f"    {log[1]}",
            f"  {dim(f'elapsed {el:.1f}s')}",
        ]
        if not self._first: _up(self.LINES)
        self._first = False
        for l in lines: _clr(); print(l)
        sys.stdout.flush()


# ─── Core builder (standalone — reads/writes JSON directly) ────────────────────

def build(model_folder, dry_run=False, quiet=False):
    D   = Display(quiet=quiet)
    t0  = time.time()

    # ── Load saved matrices ────────────────────────────────────────────────
    print(f"\n  {bold('Experience Builder')}  {dim('─')}  {teal(os.path.abspath(model_folder))}")

    from model import MSEGraphLanguageModel
    model = MSEGraphLanguageModel.load(model_folder)
    tok   = model.tokenizer
    def dec(i): return tok.id_to_token.get(i, f"#{i}")

    B = model.bridges
    R = model.rels
    E = model.edges

    n_triples   = len(B.source)
    vocab_size  = tok.vocab_size_actual
    max_cluster = max(B.cluster_id) if n_triples > 0 else 0

    print(f"  {dim('loaded')}    vocab {teal(str(vocab_size))}  "
          f"edges {amber(str(len(E.src)))}  "
          f"bridges {purple(str(n_triples))}  "
          f"clusters {green(str(max_cluster))}  "
          f"rel_ids {red(str(R._n_rels))}")
    print()

    D.start()

    # ── Parse cluster info ─────────────────────────────────────────────────
    D.phase = "Parsing clusters…"
    D.total = n_triples
    groups  = defaultdict(list)
    for i, (s, t, b, c) in enumerate(
            zip(B.source, B.target, B.bridge, B.cluster_id)):
        if c != 0:
            groups[c].append((s, t, b))
        D.step = i
        D.item(dim(f"scanning triple {i}/{n_triples}"))

    cluster_info = {}
    for cid, members in groups.items():
        axis, _ = B.cluster_axis(cid)
        if axis == "bridge":
            s0, t0_, _ = members[0]
            cluster_info[cid] = ("bridge", [b for _,_,b in members], (s0, t0_))
        elif axis == "target":
            s0, _, b0 = members[0]
            cluster_info[cid] = ("target", [t for _,t,_ in members], (s0, b0))

    D.item(green(f"parsed {len(cluster_info)} clusters"), force=True)

    # ── Build reverse maps ─────────────────────────────────────────────────
    D.phase = "Building reverse maps…"
    triple_to_id    = {}
    token_as_bridge = defaultdict(list)
    token_as_target = defaultdict(list)
    for idx, (s, t, b) in enumerate(zip(B.source, B.target, B.bridge)):
        triple_to_id[(s, t, b)] = idx
        token_as_bridge[b].append((s, t, idx))
        token_as_target[t].append((s, b, idx))
    existing = set(triple_to_id.keys())

    D.item(green(f"reverse maps: {len(token_as_bridge)} bridge entries  "
                 f"{len(token_as_target)} target entries"), force=True)

    # ── Rule 1: Bridge axis expansion ─────────────────────────────────────
    D.phase = "Rule 1 — Bridge axis expansion"
    D.total = len(cluster_info)
    D.step  = 0
    exp_seen  = set()
    exp_items = []   # (s, t, b, source_triple_id, justifying_cid, rule)

    for cid, (axis, members, (s0, t0_)) in cluster_info.items():
        D.step += 1
        if axis != "bridge":
            D.item(dim(f"C{cid} skip (target-axis)"))
            continue
        slot_str = f"{dec(s0)}→__→{dec(t0_)}"
        new_count = 0
        for mi in members:
            for (s2, t2, src_tid) in token_as_bridge[mi]:
                if s2 == s0 and t2 == t0_: continue
                for mj in members:
                    if mj == mi: continue
                    entry = (s2, t2, mj)
                    if entry not in existing and entry not in exp_seen:
                        exp_seen.add(entry)
                        exp_items.append((s2, t2, mj, src_tid, cid, "rule1"))
                        new_count += 1
                        D.item(
                            f"{amber('R1')}  {teal(dec(s2))}→[{white(dec(mj))}]→{teal(dec(t2))}"
                            f"  {dim(f'via cluster {cid} ({slot_str})')}"
                            f"  {green('✦')}"
                        )
        if new_count == 0:
            D.item(dim(f"C{cid} [bridge]  {slot_str}  members={[dec(m) for m in members]}  no new triples"))

    D.item(green(f"Rule 1 complete — {sum(1 for _,_,_,_,_,r in exp_items if r=='rule1')} new triples"), force=True)

    # ── Rule 2: Target axis expansion ─────────────────────────────────────
    D.phase = "Rule 2 — Target axis expansion"
    D.step  = 0

    for cid, (axis, members, (s0, b0)) in cluster_info.items():
        D.step += 1
        if axis != "target":
            D.item(dim(f"C{cid} skip (bridge-axis)"))
            continue
        slot_str = f"{dec(s0)}→[{dec(b0)}]→__"
        new_count = 0
        for ti in members:
            for (s2, b2, src_tid) in token_as_target[ti]:
                if s2 == s0 and b2 == b0: continue
                for tj in members:
                    if tj == ti: continue
                    entry = (s2, tj, b2)
                    if entry not in existing and entry not in exp_seen:
                        exp_seen.add(entry)
                        exp_items.append((s2, tj, b2, src_tid, cid, "rule2"))
                        new_count += 1
                        D.item(
                            f"{purple('R2')}  {teal(dec(s2))}→[{white(dec(b2))}]→{teal(dec(tj))}"
                            f"  {dim(f'via cluster {cid} ({slot_str})')}"
                            f"  {green('✦')}"
                        )
        if new_count == 0:
            D.item(dim(f"C{cid} [target]  {slot_str}  members={[dec(m) for m in members]}  no new triples"))

    D.item(green(f"Rule 2 complete — {sum(1 for _,_,_,_,_,r in exp_items if r=='rule2')} new triples"), force=True)

    if not exp_items:
        D.done()
        print(f"\n  {amber('No new experience triples derived.')}  "
              f"Corpus may be too small or clusters too sparse.")
        return {}

    # ── Assign experience cluster_ids ──────────────────────────────────────
    D.phase = "Assigning experience cluster_ids…"
    exp_items.sort(key=lambda x: x[0])
    n_exp = len(exp_items)
    D.total = n_exp
    exp_cluster = [0] * n_exp
    exp_gst = defaultdict(list)
    exp_gsb = defaultdict(list)
    for idx, (s, t, b, *_) in enumerate(exp_items):
        exp_gst[(s, t)].append(idx)
        exp_gsb[(s, b)].append(idx)

    nc = max_cluster + 1
    for (s, t), idxs in exp_gst.items():
        if len(idxs) > 1:
            for i in idxs: exp_cluster[i] = nc
            bridges = [dec(exp_items[i][2]) for i in idxs]
            D.item(
                f"{green('EC')} {white(str(nc))} [bridge]  "
                f"{dec(s)}→__→{dec(t)}  members:{bridges}",
                force=True
            )
            nc += 1
    for (s, b), idxs in exp_gsb.items():
        if len(idxs) > 1 and all(exp_cluster[i] == 0 for i in idxs):
            for i in idxs: exp_cluster[i] = nc
            targets = [dec(exp_items[i][1]) for i in idxs]
            D.item(
                f"{green('EC')} {white(str(nc))} [target]  "
                f"{dec(s)}→[{dec(b)}]→__  members:{targets}",
                force=True
            )
            nc += 1

    # ── Build experience relationship rows ─────────────────────────────────
    D.phase = "Building experience relationships…"
    n_training      = R._n_rels
    just_to_exp_rel = {}
    parent_map      = {}
    exp_rel_rows    = []
    next_exp_rel    = n_training

    for exp_tid, (s, t, b, src_tid, just_cid, rule) in enumerate(exp_items):
        key = (just_cid, src_tid)
        if key not in just_to_exp_rel:
            just_to_exp_rel[key] = next_exp_rel
            parent_map[next_exp_rel] = set(R.relationships_for_triple(src_tid))
            next_exp_rel += 1
        exp_rel_rows.append((exp_tid, just_to_exp_rel[key]))

    D.done()

    # ── Summary ───────────────────────────────────────────────────────────
    summary = {
        "exp_edges":         0,  # computed below
        "exp_bridges":       n_exp,
        "exp_clustered":     sum(1 for c in exp_cluster if c != 0),
        "exp_clusters":      nc - max_cluster - 1,
        "exp_relationships": next_exp_rel - n_training,
        "exp_rel_rows":      len(exp_rel_rows),
        "rule1_triples":     sum(1 for *_, r in exp_items if r == "rule1"),
        "rule2_triples":     sum(1 for *_, r in exp_items if r == "rule2"),
    }

    if dry_run:
        print(f"\n  {amber('dry-run — nothing written')}")
        _print_summary(summary, time.time() - t0)
        return summary

    # ── Persist ────────────────────────────────────────────────────────────
    from array import array
    from experience import (ExperienceEdgeMatrix, ExperienceBridgeMatrix,
                             ExperienceRelationshipMatrix)

    # Experience edges: (source→bridge) and (bridge→target) not already in E
    existing_edges = set(zip(E.src, E.dst))
    new_edges = set()
    for s, t, b, *_ in exp_items:
        if (s, b) not in existing_edges: new_edges.add((s, b))
        if (b, t) not in existing_edges: new_edges.add((b, t))
    summary["exp_edges"] = len(new_edges)

    exp_edge = ExperienceEdgeMatrix()
    exp_edge.build_from_pairs(sorted(new_edges), vocab_size)

    exp_bridge = ExperienceBridgeMatrix()
    exp_bridge.build_from_triples(
        [(s, t, b, c) for (s, t, b, *_), c in zip(exp_items, exp_cluster)],
        vocab_size
    )

    exp_rel = ExperienceRelationshipMatrix()
    exp_rel.build_from_rows(
        exp_rel_rows,
        n_exp_rels=next_exp_rel - n_training,
        n_training_rels=n_training,
        parent_map=parent_map,
    )

    with open(os.path.join(model_folder, "experience_edges.json"),         "w") as f: json.dump(exp_edge.to_dict(),   f)
    with open(os.path.join(model_folder, "experience_bridges.json"),       "w") as f: json.dump(exp_bridge.to_dict(), f)
    with open(os.path.join(model_folder, "experience_relationships.json"), "w") as f: json.dump(exp_rel.to_dict(),    f)

    _print_summary(summary, time.time() - t0)
    return summary


def _print_summary(s, elapsed):
    tw = shutil.get_terminal_size((100,24)).columns
    print("  " + green("─" * min(tw-4, 70)))
    print(f"  {green('✓')}  {bold('Experience matrices saved')}")
    print()
    rows = [
        ("Rule 1 triples",  s["rule1_triples"]),
        ("Rule 2 triples",  s["rule2_triples"]),
        ("Total exp edges",   s["exp_edges"]),
        ("Total exp bridges", s["exp_bridges"]),
        ("Exp clustered",   f"{s['exp_clustered']}  ({s['exp_clusters']} clusters)"),
        ("Exp rel rows",    f"{s['exp_rel_rows']}  ({s['exp_relationships']} relationships)"),
        ("Total time",      f"{elapsed:.2f}s"),
    ]
    w = max(len(k) for k,_ in rows)
    for k, v in rows:
        print(f"  {dim(k.ljust(w))}  {teal(str(v))}")
    print()


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Build Experience Matrices for a trained MSE-GLM model.")
    p.add_argument("--model",   required=True, help="Saved model folder")
    p.add_argument("--quiet",   action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be derived without writing files")
    args = p.parse_args()

    if not os.path.isdir(args.model):
        print(f"Model folder not found: {args.model}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(os.path.join(args.model, "bridges.json")):
        print(f"No trained model found in {args.model} — run train.py first.",
              file=sys.stderr)
        sys.exit(1)

    build(args.model, dry_run=args.dry_run, quiet=args.quiet)


if __name__ == "__main__":
    main()
