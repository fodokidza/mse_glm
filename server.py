import os
import sys
import json
import time
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ServerConfig, GenerationConfig
from flask import (
    Flask, request, jsonify,
    render_template_string,
    Response, stream_with_context
)

# ============================================================
# ARGUMENT PARSING
# ============================================================
# MSE-GLM has no checkpoints to select between (no --finetuned /
# --rlhf here — there are no learned weights of any kind). The only
# real choices are: which saved model folder to load, which
# inference mode to default to, whether to build the (optional,
# opt-in) Context Trigger Matrix for tie-break disambiguation, and
# whether to build+enable Open Mode's (also optional, opt-in) sparse
# per-token score cache.

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='mse_model',
                     help='Path to a saved MSE-GLM model folder')
parser.add_argument('--mode',  default='strict',
                     choices=['strict', 'open'],
                     help='Default inference mode for new sessions')
parser.add_argument('--ctm',   action='store_true',
                     help='Build the Context Trigger Matrix at startup '
                          'and use it for tie-break disambiguation')
parser.add_argument('--cache', action='store_true',
                     help='Build and enable Open Mode\'s sparse '
                          'V1/V2/V3/V4/V6 score cache at startup '
                          '(see ivm.py) -- same scores either way, '
                          'purely a speed optimization')
parser.add_argument('--port',  type=int, default=ServerConfig.DEFAULT_PORT)
args = parser.parse_args()

MODEL_PATH    = args.model
DEFAULT_MODE  = args.mode

# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("  MSE-GLM SERVER")
print("=" * 60)
print(f"\n  Model: {MODEL_PATH}")
print(f"  Default mode: {DEFAULT_MODE.upper()}")

if not os.path.exists(MODEL_PATH):
    print("\n\u274c No model found!")
    print("   Run: python3 train.py --corpus <file> --out "
          f"{MODEL_PATH}")
    print("   or:  python3 train_corpus.py --corpus-dir <dir> --out "
          f"{MODEL_PATH}")
    sys.exit(1)

from model import MSEGraphLanguageModel

model = MSEGraphLanguageModel.load(MODEL_PATH)

# Open Mode has no separate "build" step -- it's automatically ready
# as soon as the model is trained or loaded (see model.py's
# _rebuild_open_engine()). This check is a defensive sanity check,
# not a "was it built yet" gate the way it used to be.
OPEN_AVAILABLE = model._open is not None

USE_CTM = False
if args.ctm:
    print("  Building Context Trigger Matrix ...")
    model.build_context_triggers(mode=DEFAULT_MODE)
    USE_CTM = True
    print("  Context Trigger Matrix ready.")

if args.cache:
    if OPEN_AVAILABLE:
        print("  Building Open Mode's sparse score cache ...")
        model.open_ctm.enable_cache(model.all_candidate_tokens())
        print(f"  Cache ready ({len(model.open_ctm._token_cache)} token rows, "
              f"{sum(len(r) for r in model.open_ctm._token_cache.values())} entries).")
    else:
        print("  --cache requested but Open Mode isn't available -- skipped.")

tokenizer = model.tokenizer
vocab_words = [
    w for w in tokenizer.token_to_id.keys()
    if w not in ('<PAD>', '<UNK>', '<BOS>', '<EOS>')
]

_stats = model.stats()
print(f"  Vocabulary:    {_stats['vocab_size']:,} tokens")
print(f"  Edges:         {_stats['edges']:,}")
print(f"  Bridges:       {_stats['bridges']:,}")
print(f"  Clusters:      {_stats['clusters']:,}")
print(f"  Open Mode:     {'available' if OPEN_AVAILABLE else 'not built'}\n")

# ============================================================
# MODE PRESETS
# ============================================================
# There is no system-prompt concept here. A neural LLM can be
# steered with free-text instructions because it generalizes past
# its training data; MSE-GLM's generate() explicitly refuses to —
# every bigram in the prompt must have been literally observed
# (Strict) or structurally justified by clustering (Open), or the
# whole prompt is rejected outright (illegal_prompt_bigram). An
# injected "### System: you are a helpful assistant..." prefix
# would almost never survive that check. So instead of personas,
# sessions choose an inference MODE — the one dial MSE-GLM actually
# exposes: Strict Mode gates every step to literal training bigrams
# (a two-stage lineage vote, tie-broken deterministically); Open Mode
# has no successor gating at all -- candidates are the ENTIRE
# vocabulary every step, chosen by IVM's seven-layer weighted voting
# (see ivm.py) rather than by whether a bigram was ever literally
# observed.

MODE_PRESETS = {
    'strict':   {'mode': 'strict', 'ctm': False},
    'open':     {'mode': 'open',   'ctm': False},
    'open_ctm': {'mode': 'open',   'ctm': True},
}

# Two OPTIONAL, read-only diagnostic endpoints sit alongside the mode
# presets above -- neither mutates session state:
#   /scores, /bigram (new routes below) -- read-only audit endpoints,
#       not generation. /scores exposes the full V1-V7 weighted-vote
#       breakdown IVM used to pick the next token (see ivm.py);
#       /bigram exposes the raw evidence counts (including V5's
#       literal witness-sentence count) for one (prev, curr) pair.
# /cache is different from those two -- it's server-wide MUTATING
# state (there's only one shared model.open_ctm, not one per
# session), toggling whether V1/V2/V3/V4/V6 come from a precomputed
# cache or are recomputed live. Same scores either way; purely a
# speed optimization an operator opts into, not a session preference.

# ============================================================
# SESSION MANAGEMENT
# ============================================================
# MSE-GLM has no context window to fill with prior turns — each
# generate() call is independent and stateless by design (that's
# what keeps every output traceable to a single, specific lineage).
# Session history is therefore kept only for the chat log / export
# feature, never fed back into the model as extra prompt text.

class ConversationManager:
    def __init__(self, mode=None):
        self.history = []
        self.max_turns = 50
        self.mode = mode or DEFAULT_MODE
        self.use_ctm = USE_CTM

    def add(self, role, text):
        self.history.append((role, text))
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-self.max_turns * 2:]

    def clear(self):
        self.history = []

    def set_mode(self, key):
        preset = MODE_PRESETS.get(key)
        if preset is None:
            return False
        if preset['mode'] == 'open' and not OPEN_AVAILABLE:
            return False
        self.mode = preset['mode']
        self.use_ctm = preset['ctm'] and USE_CTM
        self.clear()
        return True


sessions = {}


def get_session(session_id):
    if session_id not in sessions:
        sessions[session_id] = ConversationManager()
    return sessions[session_id]

# ============================================================
# GENERATION
# ============================================================

class PromptRejected(Exception):
    """Raised when the prompt itself contains a transition MSE-GLM
    never observed (Strict) or never structurally justified (Open).
    This is not an error condition in the neural-LLM sense — it's
    the model correctly refusing to fabricate a continuation for
    something it has no grounds for."""
    pass


def generate_text(prompt, max_new=100, mode='strict', use_ctm=False):
    text, ids, trace = model.generate(
        prompt, max_tokens=max_new, mode=mode,
        use_context_triggers=use_ctm,
    )
    if trace and trace[0].get('rule') == 'illegal_prompt_bigram':
        raise PromptRejected(
            "That phrasing doesn't match anything MSE-GLM was "
            "trained on, so it can't ground a continuation "
            f"in {mode} mode."
        )
    return text.strip(), trace


def stream_tokens(prompt, max_new=100, mode='strict', use_ctm=False):
    """Yield (token_text, count, speed, rule) tuples.

    MSE-GLM's generation is already fully deterministic and
    near-instant (graph lookups, not floating-point inference), so
    there's no sampling loop to drive token-by-token the way a
    neural decoder needs. Instead we generate the complete,
    already-final answer once, then replay it token-by-token so the
    UI still gets a live stream — and, unlike the neural version,
    each token can honestly report *why* it was chosen (the trace
    rule), which a sampled logit never could.
    """
    text, ids, trace = model.generate(
        prompt, max_tokens=max_new, mode=mode,
        use_context_triggers=use_ctm,
    )
    if trace and trace[0].get('rule') == 'illegal_prompt_bigram':
        raise PromptRejected(
            "That phrasing doesn't match anything MSE-GLM was "
            "trained on, so it can't ground a continuation "
            f"in {mode} mode."
        )

    prompt_ids = tokenizer.encode(prompt)
    start      = time.time()
    prev_text  = tokenizer.decode(prompt_ids).strip()
    count      = 0
    for i in range(len(prompt_ids), len(ids)):
        step_trace = trace[i - len(prompt_ids)] if \
            i - len(prompt_ids) < len(trace) else {}
        rule = step_trace.get('rule', '')
        if rule == 'termination_no_successors':
            break
        cur_text = tokenizer.decode(ids[:i + 1]).strip()
        delta    = cur_text[len(prev_text):]
        prev_text = cur_text
        if not delta:
            continue
        count += 1
        yield delta, count, round(
            count / max(time.time() - start, 1e-3), 1
        ), rule

# ============================================================
# RATE LIMITING
# ============================================================

_req_counts = {}
RATE_LIMIT  = ServerConfig.RATE_LIMIT_PER_MINUTE


def check_rate_limit(ip):
    now    = time.time()
    minute = int(now / 60)
    key    = f"{ip}:{minute}"
    _req_counts[key] = _req_counts.get(key, 0) + 1
    for k in list(_req_counts):
        if int(k.split(':')[1]) < minute - 1:
            del _req_counts[k]
    return _req_counts[key] <= RATE_LIMIT

# ============================================================
# HTML
# ============================================================

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport"
      content="width=device-width,initial-scale=1.0">
<meta name="color-scheme" content="light">
<title>MSE-GLM · Deterministic, Zero-Weight Language Model</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Syne:wght@700;800&display=swap"
      rel="stylesheet">
<link rel="stylesheet"
      href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<style>
:root {
    --bg:       #fbfbfc;
    --bg2:      #f2f3f5;
    --bg3:      #e9eaed;
    --border:   #dcdfe4;
    --border2:  #c5c9d0;
    --accent:   #0090c7;
    --green:    #12875a;
    --orange:   #c96a1f;
    --text:     #1c2430;
    --text2:    #63707e;
    --text3:    #b8bfc9;
    --heading:  #0c1220;
    --watermark:rgba(12,18,32,0.05);
    --mono:     'JetBrains Mono', monospace;
    --sans:     'Syne', sans-serif;
    --r:        10px;
}

* { margin:0; padding:0; box-sizing:border-box; }

html, body {
    height:      100%;
    overflow:    hidden;
    background:  var(--bg);
    color:       var(--text);
    font-family: var(--mono);
}

/* grain */
body::after {
    content:         '';
    position:        fixed;
    inset:           0;
    background-image:
        radial-gradient(
            ellipse 80% 80% at 50% -20%,
            rgba(0,212,255,0.04) 0%,
            transparent 60%
        );
    pointer-events:  none;
    z-index:         0;
}

.app {
    display:         flex;
    height:          100vh;
    height:          100dvh;
    max-width:       960px;
    margin:          0 auto;
    flex-direction:  column;
    padding:         0 20px;
    position:        relative;
    z-index:         1;
}

/* ── Header ── */
.hdr {
    display:         flex;
    align-items:     center;
    justify-content: space-between;
    padding:         14px 0 12px;
    border-bottom:   1px solid var(--border);
    flex-shrink:     0;
}

.logo {
    display:     flex;
    align-items: center;
    gap:         10px;
}

.logo-mark {
    width:           30px;
    height:          30px;
    border:          1px solid rgba(0,212,255,0.3);
    border-radius:   7px;
    display:         flex;
    align-items:     center;
    justify-content: center;
    font-size:       0.58rem;
    color:           var(--accent);
    letter-spacing:  0.5px;
    background:      rgba(0,212,255,0.04);
    box-shadow:      0 0 14px rgba(0,212,255,0.08);
}

.logo-name {
    font-family:    var(--sans);
    font-weight:    800;
    font-size:      1rem;
    color:          var(--heading);
    letter-spacing: 3px;
}

.logo-tag {
    font-size:      0.58rem;
    color:          var(--text2);
    letter-spacing: 2px;
}

.hdr-meta {
    display:     flex;
    align-items: center;
    gap:         14px;
}

.badge {
    font-size:      0.6rem;
    letter-spacing: 2px;
    padding:        3px 9px;
    border-radius:  3px;
}

.badge-mode {
    background:  rgba(0,212,255,0.07);
    border:      1px solid rgba(0,212,255,0.18);
    color:       var(--accent);
}

.badge-cache {
    background:  rgba(0,255,136,0.06);
    border:      1px solid rgba(0,255,136,0.15);
    color:       var(--green);
}

.hdr-stat {
    font-size:   0.62rem;
    color:       var(--text2);
    letter-spacing: 1px;
}

/* ── Persona bar ── */
.persona-bar {
    display:         flex;
    align-items:     center;
    gap:             6px;
    padding:         8px 0;
    border-bottom:   1px solid var(--border);
    flex-shrink:     0;
    overflow-x:      auto;
    scrollbar-width: none;
}
.persona-bar::-webkit-scrollbar { display:none; }

.p-label {
    font-size:      0.58rem;
    color:          var(--text3);
    letter-spacing: 3px;
    white-space:    nowrap;
    flex-shrink:    0;
}

.p-btn {
    background:   transparent;
    border:       1px solid var(--border);
    color:        var(--text2);
    padding:      4px 11px;
    border-radius:20px;
    font-family:  var(--mono);
    font-size:    0.65rem;
    cursor:       pointer;
    letter-spacing:1px;
    white-space:  nowrap;
    flex-shrink:  0;
    transition:   all 0.15s;
}
.p-btn:hover { border-color:var(--border2); color:var(--text); }
.p-btn.on {
    background:   rgba(0,212,255,0.07);
    border-color: rgba(0,212,255,0.25);
    color:        var(--accent);
}

.p-sep {
    width:      1px;
    height:     14px;
    background: var(--border);
    flex-shrink:0;
}

.p-custom {
    flex:        1;
    min-width:   100px;
    background:  transparent;
    border:      none;
    color:       var(--text2);
    font-family: var(--mono);
    font-size:   0.65rem;
    outline:     none;
    padding:     4px 8px;
}
.p-custom::placeholder { color:var(--text3); }

/* ── Chat ── */
.chat {
    flex:           1;
    overflow-y:     auto;
    padding:        16px 0;
    display:        flex;
    flex-direction: column;
    gap:            0;
    scrollbar-width: thin;
    scrollbar-color: var(--border) transparent;
}
.chat::-webkit-scrollbar { width:3px; }
.chat::-webkit-scrollbar-thumb {
    background:    var(--border);
    border-radius: 2px;
}

.msg {
    display:  flex;
    gap:      12px;
    padding:  12px 0;
    animation:msgIn 0.18s ease;
}
@keyframes msgIn {
    from { opacity:0; transform:translateY(5px); }
    to   { opacity:1; transform:translateY(0); }
}

.avatar {
    width:           28px;
    height:          28px;
    border-radius:   7px;
    display:         flex;
    align-items:     center;
    justify-content: center;
    font-size:       0.58rem;
    letter-spacing:  0.5px;
    flex-shrink:     0;
    margin-top:      3px;
}
.avatar.u {
    background: rgba(255,140,66,0.08);
    border:     1px solid rgba(255,140,66,0.18);
    color:      var(--orange);
}
.avatar.a {
    background: rgba(0,212,255,0.06);
    border:     1px solid rgba(0,212,255,0.14);
    color:      var(--accent);
}
.avatar.s {
    background: rgba(0,255,136,0.05);
    border:     1px solid rgba(0,255,136,0.12);
    color:      var(--green);
}

.msg-body { flex:1; min-width:0; }

.msg-top {
    display:       flex;
    align-items:   center;
    gap:           8px;
    margin-bottom: 5px;
}

.msg-who {
    font-size:      0.62rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-weight:    500;
}
.msg-who.u { color:var(--orange); }
.msg-who.a { color:var(--accent); }
.msg-who.s { color:var(--green); }

.msg-ts {
    font-size:   0.58rem;
    color:       var(--text3);
    margin-left: auto;
}

.spd {
    font-size:      0.58rem;
    color:          rgba(0,212,255,0.35);
    letter-spacing: 1px;
}

.msg-text {
    font-size:   0.875rem;
    line-height: 1.72;
    color:       var(--text);
    word-break:  break-word;
}

.msg-text.typing::after {
    content:    '▋';
    color:      var(--accent);
    animation:  cur 0.65s infinite;
    margin-left:2px;
}
@keyframes cur {
    0%,100% { opacity:1; }
    50%      { opacity:0; }
}

/* code */
.code-wrap {
    background:    var(--bg2);
    border:        1px solid var(--border);
    border-radius: var(--r);
    margin:        10px 0;
    overflow:      hidden;
}
.code-top {
    display:         flex;
    justify-content: space-between;
    align-items:     center;
    padding:         7px 13px;
    background:      var(--bg3);
    border-bottom:   1px solid var(--border);
}
.code-lang {
    font-size:      0.6rem;
    color:          var(--accent);
    letter-spacing: 2px;
}
.cp-btn {
    background:     transparent;
    border:         1px solid var(--border);
    color:          var(--text2);
    padding:        2px 9px;
    border-radius:  3px;
    font-family:    var(--mono);
    font-size:      0.6rem;
    cursor:         pointer;
    letter-spacing: 1px;
    transition:     all 0.15s;
}
.cp-btn:hover {
    border-color: var(--accent);
    color:        var(--accent);
    transform:    none;
}
.code-wrap pre { padding:13px; margin:0; overflow-x:auto; }
.code-wrap code { font-size:0.8rem; line-height:1.6; }

/* welcome */
.welcome {
    flex:            1;
    display:         flex;
    flex-direction:  column;
    align-items:     center;
    justify-content: center;
    gap:             20px;
    padding:         40px 0;
    pointer-events:  none;
}

.wlc-title {
    font-family:    var(--sans);
    font-size:      2rem;
    font-weight:    800;
    color:          var(--watermark);
    letter-spacing: 8px;
    text-transform: uppercase;
}

.wlc-chips {
    display:    flex;
    flex-wrap:  wrap;
    gap:        8px;
    justify-content: center;
    max-width:  500px;
}

.chip {
    background:   var(--bg2);
    border:       1px solid var(--border);
    color:        var(--text2);
    padding:      6px 14px;
    border-radius:20px;
    font-size:    0.72rem;
    cursor:       pointer;
    transition:   all 0.15s;
    pointer-events: all;
}
.chip:hover {
    border-color: rgba(0,212,255,0.25);
    color:        var(--accent);
    transform:    none;
}

/* ── Token bar ── */
.tok-bar {
    display:     flex;
    align-items: center;
    gap:         16px;
    min-height:  22px;
    flex-shrink: 0;
}
.tok-stat {
    font-size:      0.6rem;
    color:          var(--text3);
    letter-spacing: 1px;
}
.tok-stat .v { color:var(--accent); }

/* ── Bottom controls ── */
.bottom {
    padding:     10px 0 14px;
    border-top:  1px solid var(--border);
    flex-shrink: 0;
}

.settings {
    display:       flex;
    align-items:   center;
    gap:           18px;
    margin-bottom: 10px;
    flex-wrap:     wrap;
}

.sl-grp {
    display:     flex;
    align-items: center;
    gap:         7px;
    font-size:   0.65rem;
    color:       var(--text2);
}

.sl-lbl { letter-spacing:1px; }

input[type=range] {
    width:        65px;
    accent-color: var(--accent);
    cursor:       pointer;
}

.sl-val {
    color:     var(--accent);
    min-width: 30px;
    font-size: 0.68rem;
}

.tog-wrap {
    display:     flex;
    align-items: center;
    gap:         7px;
    font-size:   0.65rem;
    color:       var(--text2);
    cursor:      pointer;
    letter-spacing:1px;
}

.tog {
    width:         32px;
    height:        17px;
    background:    var(--bg3);
    border:        1px solid var(--border);
    border-radius: 9px;
    position:      relative;
    transition:    all 0.2s;
}
.tog.on {
    background:   rgba(0,212,255,0.15);
    border-color: rgba(0,212,255,0.35);
}
.tog::after {
    content:       '';
    position:      absolute;
    width:         11px;
    height:        11px;
    background:    var(--text2);
    border-radius: 50%;
    top:           2px;
    left:          2px;
    transition:    all 0.2s;
}
.tog.on::after {
    left:       17px;
    background: var(--accent);
}

.input-row {
    display: flex;
    gap:     10px;
}

.inp-wrap {
    flex:          1;
    background:    var(--bg2);
    border:        1px solid var(--border);
    border-radius: var(--r);
    display:       flex;
    align-items:   flex-end;
    transition:    border-color 0.2s, box-shadow 0.2s;
}
.inp-wrap:focus-within {
    border-color: rgba(0,212,255,0.28);
    box-shadow:   0 0 0 3px rgba(0,212,255,0.04);
}

#inp {
    flex:        1;
    background:  transparent;
    border:      none;
    color:       var(--text);
    font-family: var(--mono);
    font-size:   0.875rem;
    padding:     12px 14px;
    outline:     none;
    resize:      none;
    min-height:  46px;
    max-height:  110px;
    line-height: 1.5;
}
#inp::placeholder { color:var(--text3); }

.inp-acts {
    display:     flex;
    align-items: flex-end;
    padding:     7px;
    gap:         5px;
}

.ic-btn {
    width:           30px;
    height:          30px;
    background:      transparent;
    border:          1px solid var(--border);
    border-radius:   6px;
    color:           var(--text2);
    font-size:       0.8rem;
    cursor:          pointer;
    display:         flex;
    align-items:     center;
    justify-content: center;
    transition:      all 0.15s;
}
.ic-btn:hover {
    border-color: var(--border2);
    color:        var(--text);
    transform:    none;
}

.send {
    width:           46px;
    height:          46px;
    background:      rgba(0,212,255,0.09);
    border:          1px solid rgba(0,212,255,0.22);
    border-radius:   var(--r);
    color:           var(--accent);
    font-size:       1rem;
    cursor:          pointer;
    display:         flex;
    align-items:     center;
    justify-content: center;
    flex-shrink:     0;
    transition:      all 0.15s;
}
.send:hover {
    background: rgba(0,212,255,0.16);
    box-shadow: 0 0 18px rgba(0,212,255,0.12);
    transform:  none;
}
.send:disabled {
    background:   rgba(0,212,255,0.02);
    border-color: var(--border);
    color:        var(--text3);
    cursor:       not-allowed;
}

/* ── Export toast ── */
.toast {
    position:      fixed;
    bottom:        24px;
    right:         24px;
    background:    var(--bg3);
    border:        1px solid var(--border2);
    color:         var(--text);
    padding:       10px 18px;
    border-radius: var(--r);
    font-size:     0.75rem;
    opacity:       0;
    transform:     translateY(8px);
    transition:    all 0.2s;
    z-index:       100;
    pointer-events:none;
}
.toast.show {
    opacity:   1;
    transform: translateY(0);
}
</style>
</head>
<body>
<div class="app">

  <!-- Header -->
  <header class="hdr">
    <div class="logo">
      <div class="logo-mark">MSE</div>
      <div>
        <div class="logo-name">MSE-GLM</div>
        <div class="logo-tag">ZERO WEIGHTS · FULLY TRACEABLE</div>
      </div>
    </div>
    <div class="hdr-meta">
      <span class="badge badge-mode">{{ mode }}</span>
      {% if ctm %}<span class="badge badge-cache">CTM</span>{% endif %}
      <span class="hdr-stat">{{ edges }} edges</span>
      <span class="hdr-stat">{{ bridges }} bridges</span>
      <span class="hdr-stat">{{ vocab }} tokens</span>
    </div>
  </header>

  <!-- Mode bar -->
  <div class="persona-bar">
    <span class="p-label">MODE</span>
    <button class="p-btn on" id="pb_strict"
            onclick="setMode('strict',this)">
      STRICT
    </button>
    <button class="p-btn" id="pb_open"
            onclick="setMode('open',this)">
      OPEN
    </button>
    <button class="p-btn" id="pb_open_ctm"
            onclick="setMode('open_ctm',this)">
      OPEN + CTM
    </button>
    <div class="p-sep"></div>
    <span class="p-label" style="opacity:.6">
      Strict = only text MSE-GLM literally saw. Open = adds
      structurally-inferred generalization.
    </span>
  </div>

  <!-- Chat window -->
  <div class="chat" id="chat">
    <div class="welcome" id="welcome">
      <div class="wlc-title">MSE-GLM</div>
      <div class="wlc-sub" style="color:var(--text2);font-size:12px;margin:6px 0 16px">
        Try a phrase close to your training corpus's own wording —
        Strict Mode only continues transitions it literally saw.
      </div>
      <div class="wlc-chips">
        <div class="chip" onclick="useChip(this)">the cat sat on</div>
        <div class="chip" onclick="useChip(this)">the dog ran on</div>
        <div class="chip" onclick="useChip(this)">the boy sat on</div>
      </div>
    </div>
  </div>

  <!-- Token stats -->
  <div class="tok-bar" id="tokBar"></div>

  <!-- Bottom controls -->
  <div class="bottom">
    <div class="settings">
      <div class="sl-grp">
        <span class="sl-lbl">MAX TOKENS</span>
        <input type="range" id="sTok"
               min="10" max="300" step="10"
               value="80"
               oninput="$('sTok_v').textContent=this.value">
        <span class="sl-val" id="sTok_v">80</span>
      </div>
      <div class="tog-wrap" onclick="toggleStream()">
        <div class="tog on" id="streamTog"></div>
        STREAM
      </div>
      <div class="tog-wrap" onclick="exportChat()"
           style="margin-left:auto">
        ↓ EXPORT
      </div>
    </div>

    <div class="input-row">
      <div class="inp-wrap">
        <textarea id="inp"
                  placeholder="Ask anything..."
                  rows="1"></textarea>
        <div class="inp-acts">
          <button class="ic-btn"
                  onclick="clearChat()"
                  title="Clear chat">✕</button>
        </div>
      </div>
      <button class="send" id="sendBtn"
              onclick="send()">➤</button>
    </div>
  </div>

</div>

<div class="toast" id="toast"></div>

<script>
// ── Helpers ──
const $ = id => document.getElementById(id);
const chat      = $('chat');
const welcome   = $('welcome');
const tokBar    = $('tokBar');
const sendBtn   = $('sendBtn');
const inp       = $('inp');
const appEl     = document.querySelector('.app');

// ── Keep the app (and its bottom input bar) pinned to the actual
//    visible viewport, not the layout viewport. Mobile keyboards
//    shrink the visual viewport without shrinking 100vh/100dvh
//    reliably on every browser, which is what causes an input bar
//    to end up hidden behind the keyboard or floating below it.
//    We size .app to visualViewport.height directly (with 100dvh
//    as the CSS fallback for browsers without the API) and also
//    correct for offsetTop, since some mobile browsers scroll the
//    page upward instead of resizing when the keyboard opens.
function syncViewportToKeyboard() {
    const vv = window.visualViewport;
    if (!vv || !appEl) return;
    appEl.style.height = vv.height + 'px';
    appEl.style.transform = vv.offsetTop
        ? `translateY(${vv.offsetTop}px)` : '';
    chat.scrollTop = chat.scrollHeight;
}
if (window.visualViewport) {
    window.visualViewport.addEventListener('resize', syncViewportToKeyboard);
    window.visualViewport.addEventListener('scroll', syncViewportToKeyboard);
    syncViewportToKeyboard();
}
// Also nudge on focus — the viewport resize event can lag slightly
// behind the keyboard animation on some Android browsers, so we
// re-sync a beat after focus lands and scroll the field into view.
inp.addEventListener('focus', () => {
    setTimeout(() => {
        syncViewportToKeyboard();
        inp.scrollIntoView({ block: 'end', behavior: 'smooth' });
    }, 150);
});
inp.addEventListener('blur', () => {
    setTimeout(syncViewportToKeyboard, 150);
});

let streaming   = false;
let streamOn    = true;
let currentMode = 'strict';

// Session
let sid = localStorage.getItem('mse_sid');
if (!sid) {
    sid = 'sid_' + Date.now();
    localStorage.setItem('mse_sid', sid);
}

// ── Time ──
function ts() {
    return new Date().toLocaleTimeString(
        [], { hour:'2-digit', minute:'2-digit' }
    );
}

// ── Escape HTML ──
function esc(s) {
    const d = document.createElement('div');
    d.appendChild(document.createTextNode(s));
    return d.innerHTML;
}

// ── Format message content ──
function fmt(text) {
    // Code blocks
    text = text.replace(
        /```(\w*)\n?([\s\S]*?)```/g,
        (_, lang, code) => `
<div class="code-wrap">
  <div class="code-top">
    <span class="code-lang">${lang || 'CODE'}</span>
    <button class="cp-btn"
            onclick="copyCode(this)">COPY</button>
  </div>
  <pre><code class="language-${lang || 'plaintext'}">${esc(code.trim())}</code></pre>
</div>`
    );
    // Inline code
    text = text.replace(
        /`([^`]+)`/g,
        '<code style="background:var(--bg3);padding:1px 5px;border-radius:3px;font-size:0.82em">$1</code>'
    );
    return text;
}

// ── Append message ──
function addMsg(who, name, content='', extra='') {
    if (welcome) welcome.style.display = 'none';

    const id  = 'msg_' + Date.now() + Math.random();
    const div = document.createElement('div');
    div.className = 'msg';
    div.id        = id;
    div.innerHTML = `
      <div class="avatar ${who}">${name[0]}</div>
      <div class="msg-body">
        <div class="msg-top">
          <span class="msg-who ${who}">${name}</span>
          <span class="msg-ts">${ts()}</span>
          ${extra}
        </div>
        <div class="msg-text" id="txt_${id}">
          ${content}
        </div>
      </div>`;
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
    return id;
}

function appendText(id, text) {
    const el = $('txt_' + id);
    if (el) {
        el.textContent += text;
        chat.scrollTop  = chat.scrollHeight;
    }
}

function finalizeMsg(id, fullText) {
    const el = $('txt_' + id);
    if (el) {
        el.classList.remove('typing');
        el.innerHTML = fmt(fullText);
        el.querySelectorAll('pre code').forEach(
            b => hljs.highlightElement(b)
        );
        chat.scrollTop = chat.scrollHeight;
    }
}

function removeMsg(id) {
    const el = $(id);
    if (el) el.remove();
}

// ── Token stats ──
function showStats(count, speed) {
    tokBar.innerHTML =
        `<span class="tok-stat">` +
        `Tokens <span class="v">${count}</span>` +
        `</span>` +
        `<span class="tok-stat">` +
        `Speed <span class="v">${speed}</span> tok/s` +
        `</span>`;
}
function clearStats() { tokBar.innerHTML = ''; }

// ── Loading state ──
function setLoading(v) {
    streaming       = v;
    sendBtn.disabled= v;
    inp.disabled    = v;
}

// ── Streaming send ──
async function sendStream(prompt, tok) {
    const msgId = addMsg('a', 'MSE-GLM', '');
    const el    = $('txt_' + msgId);
    el.classList.add('typing');

    let full = '';

    try {
        const res = await fetch('/stream', {
            method  : 'POST',
            headers : {'Content-Type':'application/json'},
            body    : JSON.stringify({
                prompt,
                max_tokens  : tok,
                session_id  : sid,
            })
        });

        if (!res.ok) {
            const errData = await res.json().catch(()=>({}));
            finalizeMsg(msgId, `[${errData.error || 'Server error'}]`);
            return;
        }

        const reader  = res.body.getReader();
        const decoder = new TextDecoder();
        let   buf     = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buf += decoder.decode(value, {stream:true});
            const lines = buf.split('\n');
            buf         = lines.pop();

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                try {
                    const d = JSON.parse(line.slice(6));
                    if (d.error) {
                        appendText(msgId, ` [${d.error}]`);
                        break;
                    }
                    if (d.done) break;
                    if (d.token) {
                        full += d.token;
                        appendText(msgId, d.token);
                    }
                    if (d.speed !== undefined) {
                        showStats(d.count, d.speed);
                    }
                } catch(e) {}
            }
        }
    } catch(e) {
        full = `[Error: ${e.message}]`;
    } finally {
        finalizeMsg(msgId, full || '...');
    }
}

// ── Normal send ──
async function sendNormal(prompt, tok) {
    const thinkId = addMsg(
        'a', 'MSE-GLM',
        '<span style="color:var(--text3)">tracing lineage...</span>'
    );

    try {
        const res  = await fetch('/generate', {
            method  : 'POST',
            headers : {'Content-Type':'application/json'},
            body    : JSON.stringify({
                prompt,
                max_tokens  : tok,
                session_id  : sid,
            })
        });
        const data = await res.json();
        removeMsg(thinkId);

        if (data.status === 'ok') {
            const id = addMsg('a', 'MSE-GLM');
            finalizeMsg(id, data.response);
        } else {
            addMsg('a', 'ERROR',
                esc(data.error || 'Unknown error'));
        }
    } catch(e) {
        removeMsg(thinkId);
        addMsg('a', 'ERROR', esc(e.message));
    }
}

// ── Main send ──
async function send() {
    if (streaming) return;

    const prompt = inp.value.trim();
    if (!prompt)  return;

    const tok = parseInt($('sTok').value);

    addMsg('u', 'YOU', esc(prompt));
    inp.value    = '';
    inp.style.height = '';
    setLoading(true);
    clearStats();

    try {
        if (streamOn) {
            await sendStream(prompt, tok);
        } else {
            await sendNormal(prompt, tok);
        }
    } finally {
        setLoading(false);
        inp.focus();
    }
}

// ── Mode ──
async function setMode(key, btn) {
    document.querySelectorAll('.p-btn')
        .forEach(b => b.classList.remove('on'));
    if (btn) btn.classList.add('on');

    try {
        const res = await fetch('/mode', {
            method  : 'POST',
            headers : {'Content-Type':'application/json'},
            body    : JSON.stringify({
                session_id: sid, mode: key
            })
        });
        const data = await res.json();
        if (data.status !== 'ok') {
            addMsg('s', 'SYSTEM',
                `Couldn't switch to <strong>${key.toUpperCase()}</strong>: ` +
                esc(data.error || 'unavailable'));
            return;
        }
        currentMode = key;
    } catch(e) {}

    addMsg('s', 'SYSTEM',
        `Mode: <strong>${key.toUpperCase()}</strong>. History cleared.`
    );
    chat.scrollTop = chat.scrollHeight;
}

// ── Stream toggle ──
function toggleStream() {
    streamOn = !streamOn;
    $('streamTog').classList.toggle('on', streamOn);
}

// ── Clear chat ──
async function clearChat() {
    try {
        await fetch('/reset', {
            method  : 'POST',
            headers : {'Content-Type':'application/json'},
            body    : JSON.stringify({session_id: sid})
        });
    } catch(e) {}

    // Remove all messages but keep welcome
    [...chat.children].forEach(el => {
        if (el.id !== 'welcome') el.remove();
    });
    if (welcome) welcome.style.display = '';
    clearStats();
    setLoading(false);
}

// ── Export chat ──
function exportChat() {
    const msgs = [...chat.querySelectorAll('.msg')];
    if (msgs.length === 0) {
        showToast('Nothing to export');
        return;
    }

    const lines = msgs.map(m => {
        const who  = m.querySelector('.msg-who')
                      ?.textContent || '?';
        const text = m.querySelector('.msg-text')
                      ?.textContent || '';
        return `[${who}]\n${text}\n`;
    });

    const blob = new Blob(
        [lines.join('\n---\n')],
        { type: 'text/plain' }
    );
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = `chat_${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
    showToast('Chat exported ✓');
}

// ── Copy code ──
function copyCode(btn) {
    const code = btn.closest('.code-wrap')
                    .querySelector('code')
                    .innerText;
    navigator.clipboard.writeText(code).then(() => {
        btn.textContent = 'COPIED';
        setTimeout(() => btn.textContent = 'COPY', 1800);
    });
}

// ── Chip click ──
function useChip(el) {
    inp.value = el.textContent.trim();
    inp.focus();
}

// ── Toast ──
function showToast(msg) {
    const t      = $('toast');
    t.textContent= msg;
    t.classList.add('show');
    setTimeout(() => t.classList.remove('show'), 2200);
}

// ── Auto resize textarea ──
inp.addEventListener('input', function() {
    this.style.height = '';
    this.style.height = Math.min(
        this.scrollHeight, 110
    ) + 'px';
});

inp.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        send();
    }
});

inp.focus();
</script>
</body>
</html>"""

# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


@app.route('/')
def index():
    return render_template_string(
        HTML,
        mode    = DEFAULT_MODE.upper(),
        ctm     = USE_CTM,
        edges   = f"{_stats['edges']:,}",
        bridges = f"{_stats['bridges']:,}",
        vocab   = f"{_stats['vocab_size']:,}",
    )


@app.route('/generate', methods=['POST'])
def generate():
    ip = request.remote_addr
    if not check_rate_limit(ip):
        return jsonify({'error': 'Rate limit exceeded'}), 429

    try:
        data       = request.get_json()
        user_msg   = data.get('prompt', '').strip()
        session_id = data.get('session_id', 'default')
        max_tokens = int(data.get('max_tokens', GenerationConfig.SERVER_DEFAULT_MAX_TOKENS))

        if not user_msg:
            return jsonify({'error': 'prompt required'}), 400

        session = get_session(session_id)

        response, trace = generate_text(
            user_msg, max_tokens, session.mode, session.use_ctm
        )

        session.add('human',    user_msg)
        session.add('assistant', response)

        return jsonify({
            'response': response,
            'mode'    : session.mode,
            'status'  : 'ok'
        })

    except PromptRejected as e:
        return jsonify({'error': str(e), 'status': 'rejected'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/stream', methods=['POST'])
def stream():
    ip = request.remote_addr
    if not check_rate_limit(ip):
        return jsonify({'error': 'Rate limit exceeded'}), 429

    try:
        data       = request.get_json()
        user_msg   = data.get('prompt', '').strip()
        session_id = data.get('session_id', 'default')
        max_tokens = int(data.get('max_tokens', GenerationConfig.SERVER_DEFAULT_MAX_TOKENS))

        if not user_msg:
            return jsonify({'error': 'prompt required'}), 400

        session = get_session(session_id)

        def gen():
            full = []
            try:
                for token, count, speed, rule in \
                        stream_tokens(
                            user_msg, max_tokens,
                            session.mode, session.use_ctm
                        ):
                    full.append(token)
                    payload = json.dumps({
                        'token': token,
                        'count': count,
                        'speed': speed,
                        'rule' : rule,
                    })
                    yield f"data: {payload}\n\n"

                response = ''.join(full)
                session.add('human',     user_msg)
                session.add('assistant', response)
                yield f"data: {json.dumps({'done':True})}\n\n"

            except PromptRejected as e:
                yield f"data: {json.dumps({'error':str(e)})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error':str(e)})}\n\n"

        return Response(
            stream_with_context(gen()),
            mimetype = 'text/event-stream',
            headers  = {
                'Cache-Control'    : 'no-cache',
                'X-Accel-Buffering': 'no',
            }
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/mode', methods=['POST'])
def mode_route():
    data       = request.get_json()
    session_id = data.get('session_id', 'default')
    mode_key   = data.get('mode', 'strict')
    session    = get_session(session_id)
    ok = session.set_mode(mode_key)
    if not ok:
        return jsonify({
            'status': 'error',
            'error' : 'unknown mode',
        }), 200
    return jsonify({'status': 'ok', 'mode': mode_key})


@app.route('/scores', methods=['POST'])
def scores():
    """
    Open Mode only, read-only, stateless (no session_id, no history
    mutation) -- the full V1-V7 weighted-vote breakdown IVM used (or
    would use) to pick the next token for `prompt` (see ivm.py's
    score_candidates()/select()). Mirrors chat.py's /scores REPL
    command and analyse.py's `open-scores` CLI subcommand.

    Body: {"prompt": "..."}
    Always scores the entire vocabulary -- Open Mode has no successor
    gating at all, so there is no narrower option anymore.

    "scores" is the FINAL combined score -- the sum of ALL SEVEN
    layers (important_vote/influence_vote/context_vote/
    context_influence_vote/bigram_witness_vote/adjacency_vote/
    prev_current_vote), each also returned separately so the
    breakdown stays auditable. Don't expect the first four to sum to
    "scores" on their own -- bigram_witness_vote (V5), adjacency_vote
    (V6), and prev_current_vote (V7) are usually the largest single
    contributors: V5 whenever the exact bigram was literally seen in
    training, V6 whenever the context token was ever directly,
    immediately followed by the candidate (directional -- token->
    candidate only), V7 whenever the prompt's last two tokens and the
    candidate ever all three shared one training sentence together
    (no adjacency required, but requires BOTH of the last two tokens,
    not just one).

    "cache_used" reports whether this breakdown was actually served
    from Open Mode's opt-in sparse V1/V2/V3/V4/V6 score cache (see
    ivm.py's build_cache()) or computed live -- same numbers either
    way, see /cache below to toggle it server-wide.
    """
    try:
        data = request.get_json()
        prompt = data.get('prompt', '').strip()
        if not prompt:
            return jsonify({'error': 'prompt required'}), 400
        if not OPEN_AVAILABLE:
            return jsonify({'error': 'Model has not been trained or loaded.'}), 409
        result = model.open_mode_candidate_scores(prompt)
        return jsonify({'status': 'ok', **result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/bigram', methods=['POST'])
def bigram():
    """
    Read-only, stateless -- raw bigram evidence for one (prev, curr)
    pair: literal training-count, the total Strict Mode's tie-break
    and Open Mode's first tie-break cascade stage both use (see
    inference.py's _bigram_frequency), plus "witness_sentences": how
    many distinct literal training sentences actually contained this
    bigram as a consecutive pair -- the evidence set V5's
    bigram-witness vote checks context tokens against (see ivm.py's
    bigram_relationships()). Mirrors chat.py's
    /bigram REPL command and analyse.py's `bigram` CLI subcommand.

    Body: {"prev": "the", "curr": "mat", "mode": "strict"}
    """
    try:
        data = request.get_json()
        prev_word = data.get('prev', '').strip()
        curr_word = data.get('curr', '').strip()
        mode = data.get('mode', 'strict')
        if not prev_word or not curr_word:
            return jsonify({'error': 'prev and curr required'}), 400
        engine = model._engine(mode)
        prev_id = tokenizer.encode(prev_word)[-1]
        curr_id = tokenizer.encode(curr_word)[-1]
        training = model.edges.frequency(prev_id, curr_id)
        witnesses = model.bigram_witness_sentences(prev_id, curr_id)
        return jsonify({
            'status': 'ok', 'prev': prev_word, 'curr': curr_word,
            'training': training,
            'total': engine._bigram_frequency(prev_id, curr_id),
            'witness_sentences': len(witnesses),
        })
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 409
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/cache', methods=['POST'])
def cache_route():
    """
    Toggle or inspect Open Mode's opt-in sparse per-token score cache
    (V1/V2/V3/V4/V6 only -- V5/V7 always stay live, see ivm.py's
    build_cache()). Server-wide, not per-session -- there is only one
    `model.open_ctm`, shared across every session. Read-only for
    "status"; "on"/"off" mutate server-wide state, so this is a
    deliberate operator action, not something a session switches on
    its own. Mirrors chat.py's /cache and analyse.py's `cache`
    subcommand.

    Body: {"action": "on" | "off" | "status"}  (default "status")
    Returns {"status": "ok", "enabled": bool, "token_rows": int,
             "entries": int}.
    """
    try:
        if not OPEN_AVAILABLE:
            return jsonify({'error': 'Model has not been trained or loaded.'}), 409
        data = request.get_json(silent=True) or {}
        action = data.get('action', 'status')
        if action == 'on':
            model.open_ctm.enable_cache(model.all_candidate_tokens())
        elif action == 'off':
            model.open_ctm.disable_cache()
        elif action != 'status':
            return jsonify({'error': f"unknown action {action!r} "
                                      "(expected 'on', 'off', or 'status')"}), 400
        return jsonify({
            'status'    : 'ok',
            'enabled'   : model.open_ctm._use_cache,
            'token_rows': len(model.open_ctm._token_cache),
            'entries'   : sum(len(r) for r in model.open_ctm._token_cache.values()),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/reset', methods=['POST'])
def reset():
    data       = request.get_json()
    session_id = data.get('session_id', 'default')
    if session_id in sessions:
        sessions[session_id].clear()
    return jsonify({'status': 'ok'})


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status'       : 'ok',
        'default_mode' : DEFAULT_MODE,
        'open_available': OPEN_AVAILABLE,
        'ivm_available' : model.open_ctm is not None,
        'ctm_enabled'  : USE_CTM,
        'cache_enabled': model.open_ctm._use_cache if OPEN_AVAILABLE else False,
        'vocabulary'   : _stats['vocab_size'],
        'edges'        : _stats['edges'],
        'bridges'      : _stats['bridges'],
        'clusters'     : _stats['clusters'],
        'sessions'     : len(sessions),
    })


@app.route('/vocab', methods=['GET'])
def vocab():
    return jsonify({
        'vocab_size': len(vocab_words),
        'words'     : sorted(vocab_words),
    })


@app.route('/sessions', methods=['GET'])
def session_stats():
    return jsonify({
        'total': len(sessions),
        'ids'  : list(sessions.keys()),
    })


# ============================================================
# RUN
# ============================================================

if __name__ == '__main__':
    print(f"\n{'='*60}")
    print(f"  READY")
    print(f"{'='*60}")
    print(f"\n  → http://localhost:{args.port}")
    print(f"\n  Endpoints:")
    print(f"    GET  /health")
    print(f"    GET  /vocab")
    print(f"    POST /generate")
    print(f"    POST /stream")
    print(f"    POST /mode")
    print(f"    POST /scores   (Open Mode only -- full V1-V7 breakdown for a prompt)")
    print(f"    POST /bigram   (raw bigram evidence incl. V5 witness_sentences)")
    print(f"    POST /cache    (toggle/inspect the sparse V1/V2/V3/V4/V6 score cache)")
    print(f"    POST /reset")
    print(f"\n  Flags:")
    print(f"    --model PATH  saved MSE-GLM model folder (default: mse_model)")
    print(f"    --mode  M     default inference mode: strict | open")
    print(f"    --ctm         build the Context Trigger Matrix at startup")
    print(f"    --cache       build + enable the sparse score cache at startup")
    print(f"    --port N      custom port")
    print(f"\n  CTRL+C to stop")
    print(f"{'='*60}\n")

    app.run(
        host  = '0.0.0.0',
        port  = args.port,
        debug = False,
        threaded = True,
    )
