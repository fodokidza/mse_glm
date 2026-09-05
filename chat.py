"""
chat.py — Interactive REPL for MSE-GLM (Strict + Open Mode).

Both modes are now fully deterministic -- no random.choice anywhere
in inference.py. Strict Mode's two-stage lineage pipeline is
otherwise unchanged, but every genuine tie it can't resolve now falls
to a deterministic bigram-frequency tie-break (whichever candidate
literally followed the current token most often in training) instead
of a random pick -- see inference.py's _bigram_tie_break.

Open Mode's candidates are the ENTIRE vocabulary, every step -- no
successor gating at all. Selection is CTM/IVM weighted voting over
that full candidate set -- SEVEN independent, summed vote layers:
  V1 important + V2 influence   -- both deliberately small, ride on
                                    top of V3, can only ever nudge a
                                    tie V3 left open, never override it
  V3 every context token        -- full weight, the primary signal
  V4 context-token influence    -- tiny magnitude-sensitive refinement,
                                    same "never override V3" role as V1/V2
  V5 bigram-witness vote        -- did each context token literally
                                    appear in a training sentence
                                    containing THIS exact bigram
                                    (current->candidate)? PEER-weighted
                                    with V3 (not kept small) -- this is
                                    usually the single largest
                                    contributor to a candidate's score
                                    whenever the bigram was literally
                                    seen, so don't expect V1+V2+V3+V4
                                    alone to add up to the total.
  V6 adjacency vote             -- was each context token EVER immediately
                                    followed by the candidate in training
                                    (directional: token->candidate only,
                                    not necessarily this specific
                                    transition, unlike V5)? Also
                                    PEER-weighted with V3.
  V7 prev+current co-occurrence -- a single fixed-pair check, not a
                                    per-context-token vote: did
                                    `previous`, `current`, AND the
                                    candidate ever all three share one
                                    training sentence together, no
                                    adjacency required between any of
                                    them? Also PEER-weighted with V3.
then a deterministic cascade if the score itself still ties: bigram
frequency, then global frequency, then lowest token id. See ivm.py's
module docstring for the full formula. /scores below exposes the
entire breakdown that drove the last Open Mode token -- which tokens
were important, what each layer contributed, and which stage of the
tie-break cascade (if any) actually decided it -- since an
unexplained score is not something this project wants to hand back.

V1/V2/V3/V4/V6 (not V5/V7 -- see ivm.py's build_cache()) can
optionally be served from a sparse precomputed per-token cache
instead of recomputed from scratch every step -- off by default,
same answer either way, purely a speed optimization. /cache below
toggles it on model.open_ctm and reports whether it's on.

Commands:
  <text>                    generate continuation (current mode)
  /mode strict              switch to Strict Mode (v2.1, training data only)
  /mode open                switch to Open Mode  (candidates = entire
                             vocabulary, CTM/IVM weighted voting as
                             primary mechanism)
  /explain <prev> | <curr>  explain a single inference step
  /scores <prompt>          Open Mode only: full V1-V7 score breakdown for
                             the next token, plus which tie-break stage (if
                             any) decided the winner
  /bigram <prev> <curr>     bigram frequency for a pair (used for Strict
                             Mode's tie-break and Open Mode's first
                             tie-break stage) -- also shows how many
                             distinct training sentences literally
                             witnessed the bigram (V5's evidence set)
  /cache on|off|status      toggle the sparse per-token V1/V2/V3/V4/V6
                             score cache on model.open_ctm (Open Mode
                             only) -- "on" builds it fresh for the
                             current vocabulary, "off" reverts to live
                             scoring, "status" (or bare /cache) reports
                             whether it's on and how many entries it holds
  /shared <tok1> <tok2> … infer_shared_role() across token set
  /similarity <a> <b>       cluster-set similarity between two tokens
  /stats                    model stats
  /clusters                 top dual-axis cluster groups (training)
  /quit

Commands that take arguments (/explain, /scores, /bigram, /shared,
/similarity) tolerate a stray quote character immediately after the
command name with no space (e.g. `/bigram"the cat"`) and strip
matching `"`/`'` wrapping from each argument -- see _match_command()/
_unquote() below. Without this, a command typed with quotes but no
space fails to match at all and silently falls through to being
treated as a generation prompt instead, which is confusing (it
doesn't raise an error -- it just generates from the literal command
text and reports illegal_prompt_bigram).
"""

import argparse
from model import MSEGraphLanguageModel
from analyse import Analyser
from config import GenerationConfig


def _match_command(line, cmd):
    """
    Returns the argument portion of `line` if it invokes slash-command
    `cmd` (e.g. "/bigram"), else None. Tolerant of the argument
    starting immediately after the command name with no space (a
    stray quote character, typically) as well as the usual "/cmd arg"
    form: matches whenever `line == cmd` or `line` starts with `cmd`
    followed by anything that ISN'T a word character (so "/bigram" and
    "/bigramfoo" are never confused with each other).
    """
    if line == cmd:
        return ""
    if line.startswith(cmd) and len(line) > len(cmd) and not line[len(cmd)].isalnum():
        return line[len(cmd):].strip()
    return None


def _unquote(token):
    """Strip any leading/trailing straight-quote characters from one
    parsed argument -- handles both a whole phrase quoted together
    and stray quotes left on individual words after a plain .split()."""
    return token.strip("\"'")


def main():
    parser = argparse.ArgumentParser(description="Chat with a trained MSE-GLM model")
    parser.add_argument("--model",      required=True, help="Saved model folder")
    parser.add_argument("--max-tokens", type=int, default=GenerationConfig.CHAT_CLI_DEFAULT_MAX_TOKENS)
    parser.add_argument("--mode",       choices=["strict","open"], default="strict")
    args = parser.parse_args()

    print(f"\nLoading model from {args.model} …")
    model = MSEGraphLanguageModel.load(args.model)

    mode = args.mode
    analyser = Analyser(model)
    print(f"  Stats: {model.stats()}")
    print(f"  Mode:  {mode.upper()}")
    print("  (fully deterministic -- no randomness anywhere in generation)")
    print("\n  Commands: /mode strict|open  /explain  /scores  /bigram  /cache  /shared  "
          "/similarity  /stats  /clusters  /quit\n")

    while True:
        try:
            line = input(f"[{mode}] you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue

        if line in ("/quit", "/exit"):
            break

        # ── /mode ─────────────────────────────────────────────────────────
        if line.startswith("/mode"):
            parts = line.split()
            if len(parts) < 2 or parts[1] not in ("strict", "open"):
                print("  Usage: /mode strict|open")
                continue
            mode = parts[1]
            print(f"  Switched to {mode.upper()} mode.")
            continue

        # ── /stats ────────────────────────────────────────────────────────
        if line == "/stats":
            for k, v in model.stats().items():
                print(f"  {k}: {v}")
            continue

        # ── /clusters ─────────────────────────────────────────────────────
        if line == "/clusters":
            for c in analyser.cluster_report(top_n=10):
                print(f"  [{c['cluster_id']}] {c['axis']:6s}  {c['slot']:30s}  {', '.join(c['members'])}")
            continue

        # ── /similarity ───────────────────────────────────────────────────
        arg = _match_command(line, "/similarity")
        if arg is not None:
            parts = [_unquote(p) for p in arg.split()]
            if len(parts) != 2:
                print("  Usage: /similarity <word_a> <word_b>")
                continue
            result = model.token_similarity(parts[0], parts[1], mode=mode)
            print(f"  similarity({parts[0]}, {parts[1]}) = {result['similarity']}"
                  f"  shared_clusters={result['shared_clusters']}")
            continue

        # ── /shared ───────────────────────────────────────────────────────
        arg = _match_command(line, "/shared")
        if arg is not None:
            tokens = [_unquote(p) for p in arg.split()]
            if not tokens:
                print("  Usage: /shared <tok1> <tok2> ...")
                continue
            results = model.infer_shared_role(tokens, mode=mode)
            if not results:
                print("  no shared cluster found")
            else:
                for tok, axis, ev in results:
                    print(f"  {' '.join(tokens)} -> {tok}  [{axis}  {ev}]")
            continue

        # ── /explain ──────────────────────────────────────────────────────
        arg = _match_command(line, "/explain")
        if arg is not None:
            payload = _unquote(arg)
            if not payload:
                print("  Usage: /explain <curr>  or  /explain <prev> | <curr>")
                continue
            prev, curr = ("", payload) if "|" not in payload else \
                         [_unquote(p.strip()) for p in payload.split("|", 1)]
            token, trace = model.explain_step(prev, curr, mode=mode)
            print(f"  next='{token}'  {trace}")
            continue

        # ── /scores ───────────────────────────────────────────────────────
        arg = _match_command(line, "/scores")
        if arg is not None:
            if mode != "open":
                print("  /scores only applies in Open Mode (/mode open first)")
                continue
            prompt = _unquote(arg)
            if not prompt:
                print("  Usage: /scores <prompt>")
                continue
            info = model.open_mode_candidate_scores(prompt)
            if info is None:
                print("  Open Mode isn't ready (model not trained/loaded).")
                continue
            print(f"  important tokens: {info['important_tokens']}")
            print(f"  influence:        {info['influence']}")
            print(f"  (scoring the entire vocabulary -- {len(info['scores'])} candidates)")
            print(f"  {'candidate':15s} {'V1 (important)':>15s} {'V2 (influence)':>15s} "
                  f"{'V3 (context)':>13s} {'V4 (ctx.infl.)':>15s} {'V5 (big.witness)':>17s} "
                  f"{'V6 (adjacent)':>14s} {'V7 (prev+curr)':>15s} {'score':>8s}")
            for c in sorted(info["scores"], key=info["scores"].get, reverse=True):
                v1 = info["important_vote"].get(c, 0)
                v2 = info["influence_vote"].get(c, 0)
                v3 = info["context_vote"].get(c, 0)
                v4 = info["context_influence_vote"].get(c, 0)
                v5 = info["bigram_witness_vote"].get(c, 0)
                v6 = info["adjacency_vote"].get(c, 0)
                v7 = info["prev_current_vote"].get(c, 0)
                marker = "  <- winner" if c == info["winner"] else ""
                print(f"    {c:13s} {v1:15.2f} {v2:15.2f} {v3:13.2f} {v4:15.2f} {v5:17.2f} "
                      f"{v6:14.2f} {v7:15.2f} {info['scores'][c]:8.2f}{marker}")
            print(f"  winner: {info['winner']}  (decided by: {info['tie_break_stage']}"
                  f"  ·  cache: {'on' if info['cache_used'] else 'off'})")
            continue

        # ── /bigram ───────────────────────────────────────────────────────
        arg = _match_command(line, "/bigram")
        if arg is not None:
            parts = [_unquote(p) for p in arg.split()]
            if len(parts) != 2:
                print("  Usage: /bigram <prev> <curr>")
                continue
            engine = model._engine(mode)
            prev_id = model.tokenizer.encode(parts[0])[-1]
            curr_id = model.tokenizer.encode(parts[1])[-1]
            train_count = model.edges.frequency(prev_id, curr_id)
            total = engine._bigram_frequency(prev_id, curr_id)
            witnesses = model.bigram_witness_sentences(prev_id, curr_id)
            print(f"  '{parts[0]}' -> '{parts[1]}':  training={train_count}"
                  f"  total={total}  witness_sentences={len(witnesses)}")
            print("  (training/total is the count Strict Mode's tie-break "
                  "and Open Mode's first tie-break stage use; witness_sentences is V5's "
                  "evidence set -- how many distinct training sentences literally "
                  "contained this bigram, see ivm.py)")
            continue

        # ── /cache ────────────────────────────────────────────────────────
        arg = _match_command(line, "/cache")
        if arg is not None:
            if model.open_ctm is None:
                print("  Open Mode isn't ready (model not trained/loaded) -- no cache to toggle.")
                continue
            sub = _unquote(arg).lower()
            if sub == "on":
                model.open_ctm.enable_cache(model.all_candidate_tokens())
                entries = sum(len(row) for row in model.open_ctm._token_cache.values())
                print(f"  cache: ON  ({len(model.open_ctm._token_cache)} token rows, "
                      f"{entries} (t,c) entries -- V1/V2/V3/V4/V6 only, V5/V7 stay live)")
            elif sub == "off":
                model.open_ctm.disable_cache()
                print("  cache: OFF  (back to live scoring every step)")
            elif sub in ("", "status"):
                on = model.open_ctm._use_cache
                entries = sum(len(row) for row in model.open_ctm._token_cache.values())
                print(f"  cache: {'ON' if on else 'OFF'}  "
                      f"({len(model.open_ctm._token_cache)} token rows, {entries} entries built"
                      f"{'' if model.open_ctm._token_cache else ' -- none built yet'})")
            else:
                print("  Usage: /cache on|off|status")
            continue

        # ── generate ──────────────────────────────────────────────────────
        try:
            text, ids, trace = model.generate(line, max_tokens=args.max_tokens, mode=mode)
            print(f"  model> {text}")
            stages = [f"s{t['stage']}/{t.get('rule','')[:24]}" for t in trace]
            print(f"  trace> {' · '.join(stages)}")
        except RuntimeError as e:
            print(f"  error: {e}")


if __name__ == "__main__":
    main()
