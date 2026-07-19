"""
train.py  —  MSE-GLM Training Pipeline with full per-step live display.

Shows exactly what is being added at every training step:
  - Edge phase:        E  the → cat   ✦ new
  - Bridge phase:      B  the →[cat]→ sat   cluster:0
  - Cluster phase:     cluster 1  [bridge]  the→__→sat  {cat,dog,boy}
  - Rel phase:         R  triple_3  sat→on→the  → rel:2   ✦ shared

Usage:
    python3 train.py --text "the cat sat on the mat." --out runs/demo
    python3 train.py --corpus corpus.txt --out runs/model --vocab-size 2000
    python3 train.py --corpus corpus.txt --out runs/model --quiet
"""

import argparse
import os
import sys
import time
import shutil
from collections import Counter, defaultdict

# ─── terminal helpers ─────────────────────────────────────────────────────────

def _tw():  return shutil.get_terminal_size((100, 24)).columns
def _clr(): sys.stdout.write("\r\033[K")
def _up(n): sys.stdout.write(f"\033[{n}A")

def _bar(done, total, width=30, fill="█", empty="░"):
    pct = min(done / total, 1.0) if total else 1.0
    filled = int(width * pct)
    return fill * filled + empty * (width - filled), int(pct * 100)

def _c(text, code): return f"\033[{code}m{text}\033[0m"
def teal(t):   return _c(t, "36")
def amber(t):  return _c(t, "33")
def green(t):  return _c(t, "32")
def dim(t):    return _c(t, "2")
def bold(t):   return _c(t, "1")
def white(t):  return _c(t, "97")
def red(t):    return _c(t, "31")
def purple(t): return _c(t, "35")
def cyan(t):   return _c(t, "96")

PHASES = [
    ("tokenize", "Tokenizer  (BPE)    "),
    ("edges",    "Edge Matrix (E)     "),
    ("bridges",  "Bridge Matrix (B)   "),
    ("clusters", "Cluster Assignment  "),
    ("rels",     "Relationship Matrix "),
    ("save",     "Saving Model        "),
]

PHASE_COLORS = [teal, amber, purple, green, red, dim]

# ─── Display ──────────────────────────────────────────────────────────────────

class Display:
    """
    14-line live display. Redraws in place on every step.

    L0   spacer
    L1   MSE Graph Language Model  ─  Training
    L2   ─────────────────────────────────────
    L3   Phase  Edge Matrix (E)   [bar]  37%  step 7/19
    L4   ─────────────────────────────────────
    L5   ▸ E  the → cat   ✦ new          ← current item (highlighted)
    L6   ─────────────────────────────────────
    L7     E  <BOS> → the                ← recent[-3]
    L8     E  the → cat   ✦ new          ← recent[-2]
    L9     E  cat → sat   ✦ new          ← recent[-1]
    L10  spacer
    L11  vocab 40  edges 7  bridges 0  clusters 0  rels 0
    L12  elapsed 0.3s  ·  25 pair/s  ·  phase 0.1s  ·  next: Bridge Matrix
    L13  spacer
    """

    LINES       = 14
    LOG_LINES   = 3      # recent items shown (L7-L9)
    REDRAW_HZ   = 25     # max redraws per second

    def __init__(self, quiet=False):
        self.quiet        = quiet
        self.phase        = ""
        self.phase_label  = ""
        self.step         = 0
        self.total        = 0
        self.stats        = {}
        self.t0           = time.time()
        self.phase_t0     = time.time()
        self.rate_unit    = "step"
        self.next_phase   = ""
        self._first       = True
        self._last_step   = 0
        self._last_t      = time.time()
        # per-step log
        self.current_item = dim("─")
        self._log         = [""] * self.LOG_LINES  # fixed-size ring
        self._log_pos     = 0
        self._last_draw   = 0.0
        self._interval    = 1.0 / self.REDRAW_HZ

    # ── public API ────────────────────────────────────────────────────────────

    def start(self):
        if self.quiet: return
        print()
        for _ in range(self.LINES): print()
        sys.stdout.flush()

    def update(self, step=None, total=None, stats=None, phase=None,
               phase_label=None, rate_unit=None, next_phase=None):
        if step        is not None: self.step = step
        if total       is not None: self.total = total
        if phase       is not None: self.phase = phase
        if phase_label is not None: self.phase_label = phase_label
        if rate_unit   is not None: self.rate_unit = rate_unit
        if next_phase  is not None: self.next_phase = next_phase
        if stats:                   self.stats.update(stats)
        if not self.quiet:          self._maybe_redraw()

    def item(self, text, force=False):
        """Call on every training step with a human-readable description."""
        self.current_item = text
        self._log[self._log_pos % self.LOG_LINES] = text
        self._log_pos += 1
        if not self.quiet:
            now = time.time()
            if force or (now - self._last_draw) >= self._interval:
                self._redraw()
                self._last_draw = now

    def phase_done(self, label, detail=""):
        if self.quiet:
            print(f"  ✓  {label}  {detail}")
            return
        elapsed = time.time() - self.phase_t0
        pidx    = next((i for i,(k,_) in enumerate(PHASES) if k==self.phase), 0)
        col     = PHASE_COLORS[pidx % len(PHASE_COLORS)]
        bar     = "█" * 30
        done_line = (
            f"  {dim('Phase')}  {col(self.phase_label or self.phase)}"
            f"  [{green(bar)}]  {green('100%')}  "
            f"{dim(detail)}  {dim(f'{elapsed:.2f}s')}"
        )
        # overwrite phase bar line (L3) without touching the rest
        _up(self.LINES - 3)
        _clr()
        print(done_line)
        for _ in range(self.LINES - 4): print()
        sys.stdout.flush()

    def finish(self, stats, out_path):
        if not self.quiet:
            _up(1); print()
        tw = _tw()
        print("  " + green("─" * min(tw - 4, 70)))
        print(f"  {green('✓')}  {bold('Training complete')}")
        print()
        rows = [
            ("Output folder",     out_path),
            ("Vocabulary",        f"{stats.get('vocab_size',0):,} tokens"),
            ("Edge Matrix",       f"{stats.get('edges',0):,} unique bigrams"),
            ("Bridge Matrix",     f"{stats.get('bridges',0):,} unique triples"),
            ("Clustered triples", f"{stats.get('clustered_bridges',0):,}  ({stats.get('clusters',0)} clusters)"),
            ("Relationship rows", f"{stats.get('relationship_rows',0):,}  ({stats.get('relationships',0)} sentences)"),
            ("Total time",        f"{time.time() - self.t0:.2f}s"),
        ]
        w = max(len(k) for k,_ in rows)
        for k, v in rows:
            print(f"  {dim(k.ljust(w))}  {teal(v)}")
        print()

    # ── internal ──────────────────────────────────────────────────────────────

    def _maybe_redraw(self):
        now = time.time()
        if now - self._last_draw >= self._interval:
            self._redraw()
            self._last_draw = now

    def _rate(self):
        now = time.time()
        dt  = now - self._last_t
        ds  = self.step - self._last_step
        self._last_step = self.step
        self._last_t    = now
        return (ds / dt) if dt > 0.001 else 0

    def _log_lines(self):
        """Return LOG_LINES recent items in chronological order."""
        out = []
        n   = min(self._log_pos, self.LOG_LINES)
        for i in range(n):
            idx = (self._log_pos - n + i) % self.LOG_LINES
            out.append(self._log[idx])
        while len(out) < self.LOG_LINES:
            out.insert(0, "")
        return out

    def _redraw(self):
        tw           = _tw()
        bar, pct     = _bar(self.step, self.total, width=30)
        elapsed      = time.time() - self.t0
        phase_el     = time.time() - self.phase_t0
        rate         = self._rate()
        pidx         = next((i for i,(k,_) in enumerate(PHASES) if k==self.phase), 0)
        col          = PHASE_COLORS[pidx % len(PHASE_COLORS)]
        rule         = dim("─" * min(tw - 4, 72))
        step_str     = f"step {self.step:,}" + (f"/{self.total:,}" if self.total else "")
        s            = self.stats

        def sv(k, c=teal):
            return f"{dim(k)} {c(str(s.get(k,0)))}"

        rate_str  = f"{rate:,.0f} {self.rate_unit}/s" if rate > 0 else "…"
        next_str  = dim(f"next: {self.next_phase}") if self.next_phase else ""
        log_items = self._log_lines()

        lines = [
            "",
            f"  {bold(white('MSE Graph Language Model'))}  {dim('─  Training')}",
            f"  {rule}",
            (f"  {dim('Phase')}  {col(self.phase_label or self.phase)}"
             f"  [{teal(bar)}]  {amber(f'{pct:3d}%')}  {dim(step_str)}"),
            f"  {rule}",
            f"  {cyan('▸')} {self.current_item}",
            f"  {rule}",
            f"    {log_items[0]}",
            f"    {log_items[1]}",
            f"    {log_items[2]}",
            "",
            (f"  {sv('vocab')}   {sv('edges',amber)}   {sv('bridges',purple)}"
             f"   {sv('clusters',green)}   {sv('rels',red)}"),
            (f"  {dim('elapsed')} {teal(f'{elapsed:.1f}s')}"
             f"  ·  {dim(rate_str)}"
             f"  ·  {dim(f'phase {phase_el:.1f}s')}"
             f"  {next_str}"),
            "",
        ]

        if not self._first:
            _up(self.LINES)
        self._first = False

        for line in lines:
            _clr()
            print(line)
        sys.stdout.flush()


# ─── Training pipeline ────────────────────────────────────────────────────────

def train_with_display(model, corpus_text=None, corpus_file=None,
                       vocab_size=1000, display=None, out_path="runs/model"):
    from tokenizer import BPETokenizer, split_sentences, normalize
    from graph import EdgeMatrix, BridgeMatrix, RelationshipMatrix
    from inference import InferenceEngine
    from array import array

    tok = BPETokenizer(vocab_size=vocab_size)
    D   = display or Display(quiet=True)

    # ══ Phase 1: Tokenizer ════════════════════════════════════════════════════
    D.phase = "tokenize"; D.phase_label = PHASES[0][1]
    D.phase_t0 = time.time(); D.next_phase = "Edge Matrix"; D.rate_unit = "merge"
    D.update(step=0, total=vocab_size,
             stats={"vocab":4,"edges":0,"bridges":0,"clusters":0,"rels":0})

    if corpus_file:
        import re
        _ws   = re.compile(r"\s+")
        _norm = re.compile(r"[^a-z0-9\s]")
        _sp   = re.compile(r"[.!?\n]+")
        wf = Counter(); sentences = []; buf = ""
        with open(corpus_file, "r", encoding="utf-8", errors="ignore") as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk: break
                buf += chunk
                parts = _sp.split(buf); buf = parts.pop()
                for p in parts:
                    p = p.strip()
                    if not p: continue
                    sentences.append(p)
                    for w in _ws.sub(" ", _norm.sub(" ", p.lower())).strip().split():
                        if w: wf[w] += 1
        if buf.strip():
            sentences.append(buf.strip())
            for w in _ws.sub(" ", _norm.sub(" ", buf.lower())).strip().split():
                if w: wf[w] += 1
    else:
        sentences = split_sentences(corpus_text)
        wf = Counter()
        for s in sentences:
            for w in normalize(s).split():
                if w: wf[w] += 1

    # char init
    chars = set()
    for w in wf: chars.update(w)
    for c in sorted(chars):
        if c not in tok.token_to_id:
            nid = max(tok.token_to_id.values()) + 1
            tok.token_to_id[c] = nid; tok.id_to_token[nid] = c

    D.update(stats={"vocab": len(tok.token_to_id)})
    D.item(dim(f"initialized {len(chars)} base characters"))

    # BPE merges
    word_syms     = {w: list(w) for w in wf}
    total_merges  = max(vocab_size - len(tok.token_to_id), 0)
    merge_done    = 0
    while len(tok.token_to_id) < vocab_size:
        pc = Counter()
        for w, freq in wf.items():
            syms = word_syms[w]
            for i in range(len(syms)-1):
                pc[(syms[i], syms[i+1])] += freq
        if not pc: break
        (a, b), cnt = pc.most_common(1)[0]
        merged = a + b
        if merged not in tok.token_to_id:
            nid = max(tok.token_to_id.values()) + 1
            tok.token_to_id[merged] = nid; tok.id_to_token[nid] = merged
        tok.merges.append((a, b))
        for w in word_syms:
            syms = word_syms[w]; ns = []; i = 0
            while i < len(syms):
                if i < len(syms)-1 and syms[i]==a and syms[i+1]==b:
                    ns.append(merged); i += 2
                else:
                    ns.append(syms[i]); i += 1
            word_syms[w] = ns
        merge_done += 1
        D.update(step=merge_done, total=total_merges,
                 stats={"vocab": len(tok.token_to_id)})
        D.item(
            f"{amber('merge')}  {teal(repr(a))} + {teal(repr(b))}"
            f"  →  {white(repr(merged))}"
            f"  {dim(f'(freq {cnt}  vocab {len(tok.token_to_id)})')}"
        )

    D.phase_done(PHASES[0][1], f"{len(tok.token_to_id):,} tokens  ·  {len(tok.merges):,} merges")

    sequences        = [tok.encode_for_training(s) for s in sentences]
    vocab_size_actual = len(tok.token_to_id)
    def dec(i): return tok.id_to_token.get(i, f"#{i}")

    # ══ Phase 2: Edge Matrix ══════════════════════════════════════════════════
    D.phase = "edges"; D.phase_label = PHASES[1][1]
    D.phase_t0 = time.time(); D.next_phase = "Bridge Matrix"; D.rate_unit = "pair"
    total_pairs = sum(max(len(s)-1, 0) for s in sequences)
    D.update(step=0, total=total_pairs)
    D.item(dim(f"scanning {total_pairs} token pairs across {len(sequences)} sentences…"), force=True)

    seen_e = set(); edges = []; step = 0
    for seq in sequences:
        for i in range(len(seq)-1):
            s, d = seq[i], seq[i+1]
            pair = (s, d)
            is_new = pair not in seen_e
            if is_new:
                seen_e.add(pair); edges.append(pair)
            step += 1
            D.update(step=step, total=total_pairs, stats={"edges": len(edges)})
            D.item(
                f"{amber('E')}  {teal(dec(s))} → {teal(dec(d))}"
                + (f"  {green('✦ new')}  {dim(f'[{len(edges)}]')}" if is_new
                   else f"  {dim('(dup)')}")
            )

    edges.sort(key=lambda p: p[0])
    edge_matrix = EdgeMatrix()
    edge_matrix.src   = array("i", [p[0] for p in edges])
    edge_matrix.dst   = array("i", [p[1] for p in edges])
    edge_matrix.index = array("i", [0] * (vocab_size_actual + 1))
    for sv in edge_matrix.src: edge_matrix.index[sv+1] += 1
    for i in range(1, len(edge_matrix.index)): edge_matrix.index[i] += edge_matrix.index[i-1]
    edge_matrix._vocab_size = vocab_size_actual
    D.update(step=total_pairs, stats={"edges": len(edges)})
    D.phase_done(PHASES[1][1], f"{len(edges):,} unique bigrams")

    # ══ Phase 3: Bridge Matrix ════════════════════════════════════════════════
    D.phase = "bridges"; D.phase_label = PHASES[2][1]
    D.phase_t0 = time.time(); D.next_phase = "Cluster Assignment"; D.rate_unit = "triple"
    total_triples = sum(max(len(s)-2, 0) for s in sequences)
    D.update(step=0, total=total_triples)
    D.item(dim(f"scanning {total_triples} token triples…"), force=True)

    seen_b = set(); triples = []; step = 0
    for seq in sequences:
        for i in range(len(seq)-2):
            src, br, tgt = seq[i], seq[i+1], seq[i+2]
            entry = (src, tgt, br)
            is_new = entry not in seen_b
            if is_new:
                seen_b.add(entry); triples.append(entry)
            step += 1
            D.update(step=step, total=total_triples, stats={"bridges": len(triples)})
            D.item(
                f"{purple('B')}  {teal(dec(src))} →[{white(dec(br))}]→ {teal(dec(tgt))}"
                + (f"  {green('✦ new')}  {dim(f'[{len(triples)}]')}" if is_new
                   else f"  {dim('(dup)')}")
            )

    triples.sort(key=lambda t: t[0])
    n = len(triples)
    D.update(step=total_triples, stats={"bridges": n})
    D.phase_done(PHASES[2][1], f"{n:,} unique triples")

    # ══ Phase 4: Cluster Assignment ═══════════════════════════════════════════
    D.phase = "clusters"; D.phase_label = PHASES[3][1]
    D.phase_t0 = time.time(); D.next_phase = "Relationship Matrix"; D.rate_unit = "cluster"
    D.update(step=0, total=n*2)
    D.item(dim(f"grouping {n} triples by (source,target) and (source,bridge)…"), force=True)

    cluster_id = [0] * n
    gst = defaultdict(list); gsb = defaultdict(list)
    for idx, (s, t, b) in enumerate(triples):
        gst[(s, t)].append(idx)
        gsb[(s, b)].append(idx)

    nc = 1; step = 0
    # Pass 1 — bridge axis
    for (s, t), idxs in gst.items():
        step += len(idxs)
        if len(idxs) > 1:
            for i in idxs: cluster_id[i] = nc
            members = [dec(triples[i][2]) for i in idxs]
            D.update(step=step, total=n*2, stats={"clusters": nc})
            D.item(
                f"{green('C')}  cluster {white(str(nc))}  {dim('[bridge-axis]')}"
                f"  {teal(dec(s))}→__→{teal(dec(t))}"
                f"  {dim('members:')} {amber(str(members))}",
                force=True
            )
            nc += 1
        else:
            D.update(step=step, total=n*2)
            D.item(
                f"{dim('C')}  {dim(dec(s))}→__→{dim(dec(t))}"
                f"  {dim('only 1 member — cluster_id=0')}"
            )

    # Pass 2 — target axis
    for (s, b), idxs in gsb.items():
        step += len(idxs)
        if len(idxs) > 1 and all(cluster_id[i] == 0 for i in idxs):
            for i in idxs: cluster_id[i] = nc
            members = [dec(triples[i][1]) for i in idxs]
            D.update(step=step, total=n*2, stats={"clusters": nc})
            D.item(
                f"{green('C')}  cluster {white(str(nc))}  {dim('[target-axis]')}"
                f"  {teal(dec(s))}→[{teal(dec(b))}]→__"
                f"  {dim('members:')} {amber(str(members))}",
                force=True
            )
            nc += 1
        else:
            D.update(step=step, total=n*2)
            D.item(
                f"{dim('C')}  {dim(dec(s))}→[{dim(dec(b))}]→__"
                f"  {dim('(already clustered or singleton)')}"
            )

    clustered = sum(1 for c in cluster_id if c != 0)

    # pack bridge matrix
    bridge_matrix = BridgeMatrix()
    bridge_matrix.source     = array("i", [t[0] for t in triples])
    bridge_matrix.target     = array("i", [t[1] for t in triples])
    bridge_matrix.bridge     = array("i", [t[2] for t in triples])
    bridge_matrix.cluster_id = array("i", cluster_id)
    bridge_matrix.index      = array("i", [0] * (vocab_size_actual + 1))
    for sv in bridge_matrix.source: bridge_matrix.index[sv+1] += 1
    for i in range(1, len(bridge_matrix.index)): bridge_matrix.index[i] += bridge_matrix.index[i-1]
    bridge_matrix._vocab_size = vocab_size_actual
    t_index = defaultdict(set)
    for sv, tv, bv, cv in zip(bridge_matrix.source, bridge_matrix.target,
                               bridge_matrix.bridge, bridge_matrix.cluster_id):
        if cv != 0: t_index[bv].add(cv); t_index[tv].add(cv)
    bridge_matrix.t_index = {k: sorted(v) for k, v in t_index.items()}

    D.update(step=n*2, total=n*2, stats={"clusters": nc-1})
    D.phase_done(PHASES[3][1], f"{nc-1} clusters  ·  {clustered:,}/{n:,} triples clustered")

    # ══ Phase 5: Relationship Matrix ══════════════════════════════════════════
    D.phase = "rels"; D.phase_label = PHASES[4][1]
    D.phase_t0 = time.time(); D.next_phase = "Save"; D.rate_unit = "row"
    total_r = sum(max(len(s)-2, 0) for s in sequences)
    triple_to_id = {(s,t,b): i for i,(s,t,b) in enumerate(triples)}
    D.update(step=0, total=total_r)
    D.item(dim(f"linking {n} triples to {len(sequences)} sentence rel_ids…"), force=True)

    rel_matrix = RelationshipMatrix()
    rows = []; step = 0
    # count how many times each triple appears across all sentences (to detect shared)
    triple_appearances = Counter()
    for rel_id, seq in enumerate(sequences):
        for i in range(len(seq)-2):
            src, br, tgt = seq[i], seq[i+1], seq[i+2]
            tid = triple_to_id.get((src, tgt, br))
            if tid is not None: triple_appearances[tid] += 1

    for rel_id, seq in enumerate(sequences):
        for i in range(len(seq)-2):
            src, br, tgt = seq[i], seq[i+1], seq[i+2]
            tid = triple_to_id.get((src, tgt, br))
            if tid is not None:
                rows.append((tid, rel_id))
                step += 1
                appears = triple_appearances[tid]
                D.update(step=step, total=total_r, stats={"rels": len(rows)})
                flag = (f"  {red('✦ SHARED')} {dim(f'({appears} sentences)')}"
                        if appears > 1 else f"  {dim('(unique)')}")
                D.item(
                    f"{red('R')}  triple {white(str(tid))}"
                    f"  {teal(dec(src))}→[{teal(dec(br))}]→{teal(dec(tgt))}"
                    f"  {dim('→')} rel:{amber(str(rel_id))}"
                    + flag
                )

    rows.sort(key=lambda r: r[1])
    rel_matrix._n_rels  = len(sequences)
    rel_matrix.r_triple = array("i", [r[0] for r in rows])
    rel_matrix.r_rel    = array("i", [r[1] for r in rows])
    rel_matrix.index    = array("i", [0] * (len(sequences) + 1))
    for r in rel_matrix.r_rel: rel_matrix.index[r+1] += 1
    for i in range(1, len(rel_matrix.index)): rel_matrix.index[i] += rel_matrix.index[i-1]

    D.update(step=total_r, stats={"rels": len(rows)})
    D.phase_done(PHASES[4][1], f"{len(rows):,} rows  ·  {len(sequences)} sentences")

    # ══ Phase 6: Save ═════════════════════════════════════════════════════════
    D.phase = "save"; D.phase_label = PHASES[5][1]
    D.phase_t0 = time.time(); D.next_phase = ""; D.rate_unit = "file"
    D.update(step=0, total=5)

    model.tokenizer = tok
    model.edges     = edge_matrix
    model.bridges   = bridge_matrix
    model.rels      = rel_matrix
    model.engine    = InferenceEngine(edge_matrix, bridge_matrix, rel_matrix)

    import json
    os.makedirs(out_path, exist_ok=True)

    tok.save(os.path.join(out_path, "vocabulary.json"))
    D.update(step=1); D.item(dim("vocabulary.json"))
    with open(os.path.join(out_path, "edges.json"), "w") as f:
        json.dump(edge_matrix.to_dict(), f)
    D.update(step=2); D.item(dim("edges.json"))
    with open(os.path.join(out_path, "bridges.json"), "w") as f:
        json.dump(bridge_matrix.to_dict(), f)
    D.update(step=3); D.item(dim("bridges.json"))
    with open(os.path.join(out_path, "relationships.json"), "w") as f:
        json.dump(rel_matrix.to_dict(), f)
    D.update(step=4); D.item(dim("relationships.json"))
    with open(os.path.join(out_path, "meta.json"), "w") as f:
        json.dump({"vocab_size_config": vocab_size, "stats": model.stats()}, f, indent=2)
    D.update(step=5); D.item(dim("meta.json"), force=True)
    D.phase_done(PHASES[5][1], os.path.abspath(out_path))

    return model.stats()


# ─── Incremental training CLI (no live display -- just before/after) ─────────

def continue_training_cli(args):
    from model import MSEGraphLanguageModel

    print()
    print(f"  {bold('MSE Graph Language Model')}  {dim('─  Continuing training')}")
    print(f"  {dim('loading')}   {teal(os.path.abspath(args.continue_from))}")

    model = MSEGraphLanguageModel.load(args.continue_from)

    if args.corpus:
        with open(args.corpus, "r", encoding="utf-8", errors="ignore") as f:
            new_text = f.read()
        size = os.path.getsize(args.corpus)
        print(f"  {dim('new corpus')}   {teal(args.corpus)}  {dim(f'({size:,} bytes)')}")
    else:
        new_text = args.text
        preview = new_text[:72].replace("\n", " ") + ("…" if len(new_text) > 72 else "")
        print(f"  {dim('new corpus')}   {teal(preview)}")

    out_path = args.out or args.continue_from
    print(f"  {dim('output')}   {teal(os.path.abspath(out_path))}")
    if args.extend_vocab:
        print(f"  {dim('extend_vocab')}   {teal('yes')}"
              f"  {dim(f'(target vocab_size {args.target_vocab_size})')}")
    print()

    summary = model.train_incremental(
        new_text,
        extend_vocab=args.extend_vocab,
        target_vocab_size=args.target_vocab_size,
    )
    model.save(out_path)

    before, after = summary["before"], summary["after"]
    rows = [
        ("Sentences added",   str(summary["sentences_added"])),
        ("Vocabulary added",  f"{summary['vocab_added']:,} tokens"
                               f"  ({before['vocab_size']:,} → {after['vocab_size']:,})"),
        ("Edge Matrix",       f"{before['edges']:,} → {after['edges']:,} unique bigrams"),
        ("Bridge Matrix",     f"{before['bridges']:,} → {after['bridges']:,} unique triples"),
        ("Clustered triples", f"{before['clustered_bridges']:,} → {after['clustered_bridges']:,}"
                               f"  ({before['clusters']} → {after['clusters']} clusters)"),
        ("Relationship rows", f"{before['relationship_rows']:,} → {after['relationship_rows']:,}"
                               f"  ({before['relationships']} → {after['relationships']} sentences)"),
    ]
    if summary["experience_invalidated"]:
        rows.append(("Experience Matrices", amber("invalidated — rebuild with build_experience.py")))

    w = max(len(k) for k, _ in rows)
    for k, v in rows:
        print(f"  {dim(k.ljust(w))}  {teal(v)}")
    print()
    print(f"  {green('✓')}  {bold('Saved to')} {os.path.abspath(out_path)}")
    print()


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Train an MSE Graph Language Model")
    p.add_argument("--corpus",     help="Path to a text file (streamed)")
    p.add_argument("--text",       help="Inline corpus string")
    p.add_argument("--out",        help="Output folder (required for fresh training; "
                                         "defaults to --continue-from's folder otherwise)")
    p.add_argument("--vocab-size", type=int, default=1000)
    p.add_argument("--quiet",      action="store_true")
    p.add_argument("--continue-from", metavar="FOLDER",
                    help="Add this corpus to an already-trained model instead of "
                         "training a fresh one. Loads FOLDER, merges the new corpus "
                         "into it, saves to --out (or back to FOLDER if --out is omitted).")
    p.add_argument("--extend-vocab", action="store_true",
                    help="With --continue-from: also grow the vocabulary using the new "
                         "corpus (requires --target-vocab-size). Default: keep the "
                         "existing vocabulary frozen.")
    p.add_argument("--target-vocab-size", type=int, default=None,
                    help="New vocab_size ceiling when --extend-vocab is set; must be "
                         "greater than the loaded model's current vocab_size.")
    args = p.parse_args()

    if not args.corpus and not args.text:
        print("Provide --corpus <file> or --text <string>", file=sys.stderr)
        sys.exit(1)

    if args.continue_from:
        continue_training_cli(args)
        return

    if not args.out:
        print("Provide --out <folder> (or use --continue-from to update a model in place)",
              file=sys.stderr)
        sys.exit(1)

    from model import MSEGraphLanguageModel
    model = MSEGraphLanguageModel(vocab_size=args.vocab_size)

    print()
    print(f"  {bold('MSE Graph Language Model')}  {dim('─  Training')}")
    if args.corpus:
        size = os.path.getsize(args.corpus) if os.path.exists(args.corpus) else 0
        print(f"  {dim('corpus')}   {teal(args.corpus)}  {dim(f'({size:,} bytes)')}")
    else:
        preview = args.text[:72].replace("\n"," ") + ("…" if len(args.text)>72 else "")
        print(f"  {dim('corpus')}   {teal(preview)}")
    print(f"  {dim('vocab')}    {teal(str(args.vocab_size))}  target tokens")
    print(f"  {dim('output')}   {teal(os.path.abspath(args.out))}")
    print()

    D = Display(quiet=args.quiet)
    D.start()

    stats = train_with_display(
        model,
        corpus_text=args.text,
        corpus_file=args.corpus,
        vocab_size=args.vocab_size,
        display=D,
        out_path=args.out,
    )
    D.finish(stats, os.path.abspath(args.out))


if __name__ == "__main__":
    main()
