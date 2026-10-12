"""
chat.py — Interactive REPL for MSE-GLM (Strict + Open Mode).

Both modes are now fully deterministic -- no random.choice anywhere
in inference.py. Strict Mode's two-stage lineage pipeline is
otherwise unchanged, but every genuine tie it can't resolve now falls
to a deterministic bigram-frequency tie-break (whichever candidate
literally followed the current token most often in training) instead
of a random pick -- see inference.py's _bigram_tie_break.

Open Mode's candidates are the ENTIRE vocabulary, every step -- no
successor gating at all. Selection is IVM scoring plus noise.py: every candidate gets
  Q1 co-occurrence   -- how many context tokens ever shared a sentence
                         with it (a percentage of all Q1 evidence)
  Q2 witnessed bigram-- context tokens that were present in a sentence
                         where (current -> candidate) literally occurred
                         (a percentage of all Q2 evidence)
  noise              -- noise.py's noise-cancellation score, scaled by
                         NoiseConfig.INFERENCE_WEIGHT, added on top
and the ranked list is then filtered by two LEGALITY CHECKS, not scores:
  Q3  is (current -> candidate) a legal bigram?
  Q4  was (previous, current, candidate) a trained triple?
The winner is the top-scoring candidate passing Q3+Q4 (falling back to
Q3 alone, then to no filter, if nobody passes). If the score still
ties: bigram frequency, then global frequency, then lowest token id.
See ivm.py's module docstring for the full formula. /scores below
exposes the whole breakdown that drove the last Open Mode token --
what each term contributed, which legality check applied, and which
stage of the tie-break cascade (if any) decided it.

Q1 (only -- Q2/Q3/Q4 and noise stay live, see ivm.py's build_cache())
can optionally be served from a sparse precomputed per-token cache
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
  /scores <prompt>          Open Mode only: full Q1-Q4 + noise score breakdown for
                             the next token, plus which tie-break stage (if
                             any) decided the winner. Always SCORES the
                             entire vocabulary -- only DISPLAYS the top N
                             by score (config.py's ScoresDisplayConfig.TOP_N,
                             25 by default), since a big vocab can't fit on
                             screen. `/scores --top <n> <prompt>` shows a
                             different number, `/scores --all <prompt>` shows
                             every candidate. The winner is always shown even
                             if it falls outside the shown set.
  /bigram <prev> <curr>     bigram frequency for a pair (used for Strict
                             Mode's tie-break and Open Mode's first
                             tie-break stage) -- also shows how many
                             distinct training sentences literally
                             witnessed the bigram (Q2's evidence set)
  /cache on|off|status      toggle the sparse per-token Q1
                             score cache on model.open_ctm (Open Mode
                             only) -- "on" builds it fresh for the
                             current vocabulary, "off" reverts to live
                             scoring, "status" (or bare /cache) reports
                             whether it's on and how many entries it holds
  /legality <n|all>         how many of the top-scored tokens Q3/Q4 may work on
                             (default: IVMConfig.LEGALITY_TOP_K); if none of them
                             is a legal continuation, <EOS> wins
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
from config import GenerationConfig, ScoresDisplayConfig


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


def _parse_top_n_flag(arg):
    """
    Parse an optional leading `--top <n>` or `--all` flag off a
    /scores-style argument string (e.g. "/scores --top 50 the cat").
    Returns (top_n, rest):
      top_n = "all"  -- --all was given, show every candidate
      top_n = <int>  -- --top <n> was given, show that many
      top_n = None   -- no flag given, caller applies its own default
    `rest` is the remaining text (the actual prompt) with the flag
    and its value stripped off. A malformed `--top` (missing or
    non-numeric value) is left in `rest` untouched and top_n is None,
    so it surfaces as part of a normal "prompt not found" complaint
    rather than being silently swallowed.
    """
    if arg.startswith("--all"):
        return "all", arg[len("--all"):].strip()
    if arg.startswith("--top"):
        parts = arg.split(None, 2)
        if len(parts) >= 2:
            try:
                n = int(parts[1])
                rest = parts[2] if len(parts) > 2 else ""
                return n, rest
            except ValueError:
                pass
    return None, arg


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
    print("\n  Commands: /mode strict|open  /explain  /scores [--top n|--all]  /bigram  /legality  "
          "/cache  /shared  /similarity  /stats  /clusters  /quit\n")

    while True:
        try:
            line = input(f"[{mode}] you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue

        if line in ("/quit", "/exit"):
            break

        # ===== /mode ======================================================= 
        if line.startswith("/mode"):
            parts = line.split()
            if len(parts) < 2 or parts[1] not in ("strict", "open"):
                print("  Usage: /mode strict|open")
                continue
            mode = parts[1]
            print(f"  Switched to {mode.upper()} mode.")
            continue

        # =========== /stats ================================================= 
        if line == "/stats":
            for k, v in model.stats().items():
                print(f"  {k}: {v}")
            continue

        # ==== /clusters ====================================================== 
        if line == "/clusters":
            for c in analyser.cluster_report(top_n=10):
                print(f"  [{c['cluster_id']}] {c['axis']:6s}  {c['slot']:30s}  {', '.join(c['members'])}")
            continue

        # ==== /similarity ================================================== 
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

        # === /shared ========================================================
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

        # === /explain ========================================================
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

        # ==== /scores ============================================================
        arg = _match_command(line, "/scores")
        if arg is not None:
            if mode != "open":
                print("  /scores only applies in Open Mode (/mode open first)")
                continue
            top_n_flag, rest = _parse_top_n_flag(arg)
            prompt = _unquote(rest)
            if not prompt:
                print("  Usage: /scores [--top <n> | --all] <prompt>")
                continue
            info = model.open_mode_candidate_scores(prompt)
            if info is None:
                print("  Open Mode isn't ready (model not trained/loaded).")
                continue

            # Scoring always covers the entire vocabulary (that's what
            # decides the real winner/tie-break) -- top_n_flag / the
            # config default only limit what gets PRINTED below.
            ranked = sorted(info["scores"], key=info["scores"].get, reverse=True)
            total = len(ranked)
            if top_n_flag == "all":
                shown = ranked
            elif top_n_flag is not None:
                shown = ranked[:top_n_flag]
            else:
                shown = ranked[:ScoresDisplayConfig.TOP_N]
            truncated = len(shown) < total

            print(f"  context tokens: {info['context_tokens']}")
            print(f"  legality filter: {info['legality']}  "
                  f"(q3+q4 = bigram AND triple legal; q3 = bigram only; none = nothing legal -> <EOS>)"
                  + ("" if info["q4_active"] else "  [Q4 inactive: fewer than two context tokens]")
                  + ("" if info["legality_top_k"] is None else
                     f"  [Q3/Q4 worked on the top {info['legality_top_k']} (window {info['window_size']})]"))
            if truncated:
                print(f"  (scoring the entire vocabulary -- {total} candidates; "
                      f"showing top {len(shown)} by score -- "
                      f"`/scores --top <n> ...` or `/scores --all ...` to change)")
            else:
                print(f"  (scoring the entire vocabulary -- {total} candidates)")

            def _print_score_row(c, tag=""):
                q1 = info["q1"].get(c, 0)
                q2 = info["q2"].get(c, 0)
                q3 = info["q3"].get(c, 0)
                q4 = info["q4"].get(c, 0)
                nz = info["noise"].get(c, 0)
                fin = info["final"].get(c, 0)
                marker = tag if tag else ("  <- winner" if c == info["winner"] else "")
                print(f"    {c:13s} {q1:9.2f} {q2:9.2f} {int(bool(q3)):>6d} {int(bool(q4)):>6d} "
                      f"{nz:10.4f} {info['score'][c]:9.2f} {fin:9.2f}{marker}")

            print(f"  {'candidate':15s} {'Q1 (co-occ)':>9s} {'Q2 (bigram)':>9s} "
                  f"{'Q3 ok':>6s} {'Q4 ok':>6s} {'noise':>10s} {'score':>9s} {'final':>9s}")
            for c in shown:
                _print_score_row(c)
            # The winner is never allowed to silently fall off-screen --
            # if truncation cut it out of the shown set, print it separately.
            if truncated and info["winner"] not in shown:
                print("    ...")
                _print_score_row(info["winner"], tag="  <- winner (outside top N)")

            print(f"  winner: {info['winner']}  (decided by: {info['tie_break_stage']}"
                  f"  ·  cache: {'on' if info['cache_used'] else 'off'})")
            continue

        # ==== /bigram ================================================================= 
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
                  "and Open Mode's first tie-break stage use; witness_sentences is Q2's "
                  "evidence set -- how many distinct training sentences literally "
                  "contained this bigram, see ivm.py)")
            continue

        # ===== /cache ==================================================================== 
        arg = _match_command(line, "/cache")
        if arg is not None:
            sub = _unquote(arg).lower() or "status"
            if sub not in ("on", "off", "status"):
                print("  Usage: /cache on|off|status")
                continue
            try:
                r = model.cache_control(sub)
            except RuntimeError as e:
                print(f"  {e}")
                continue
            print(f"  cache: {'ON' if r['enabled'] else 'OFF'}  ({r['token_rows']} token rows, "
                  f"{r['entries']} (t,c) entries"
                  f"{'' if r['token_rows'] else ' -- none built yet'}; Q1 only, Q2/Q3/Q4 and noise stay live)")
            continue

        # === /legality ===================================================================
        arg = _match_command(line, "/legality")
        if arg is not None:
            try:
                k = model.set_legality_top_k(_unquote(arg) or None)
            except (ValueError, RuntimeError) as e:
                print(f"  {e}")
                continue
            print(f"  Q3/Q4 now work on: {'ALL scored candidates' if k is None else f'the top {k}'}"
                  f"  (if none of them is a legal continuation, <EOS> wins)")
            continue

        # ==== generate ============================================================ 
        try:
            text, ids, trace = model.generate(line, max_tokens=args.max_tokens, mode=mode)
            print(f"  model> {text}")
            stages = [f"s{t['stage']}/{t.get('rule','')[:24]}" for t in trace]
            print(f"  trace> {' · '.join(stages)}")
        except RuntimeError as e:
            print(f"  error: {e}")


if __name__ == "__main__":
    main()
