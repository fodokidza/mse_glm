"""
ivm.py — token POSITIONING for MSE-GLM: which token may come after which.

(The class keeps its old name, ImportanceVoteMatrix, so existing imports keep
working; it no longer has anything to do with "importance".)

ivm.py answers ONE kind of question about every candidate C for the next token:
"does the training data say C can follow here?"  It asks four of them, Q1-Q4,
combines the answers into ONE final score per candidate, and stops there.  It knows
nothing about noise.py (noise cancellation is a separate, independent scorer);
model.py adds the two final scores together at inference.  What each question
contributed is in the trace score_candidates() returns, and is queryable on
demand (analyse.py open-scores, chat.py /scores, server.py /scores).

THE FOUR QUESTIONS            (P = tokens in context, `previous`/`current` = the
                               last two tokens, C = the candidate)

  Q1  CO-OCCURRENCE (context-wide)
        raw1(C) = Σ_{t∈P} 1[t and C ever shared a training sentence]
        Q1(C)   = Q1_WEIGHT × 100 × raw1(C) / Σ_all raw1          (a percentage)
      Every context token casts one binary vote for each candidate it has ever
      appeared in a sentence with (either order, any distance).

  Q2  LEGAL BIGRAM, WITNESSED     (direction: current -> C only)
        raw2(C) = Σ_{t∈P} 1[t was in a sentence where the bigram
                            (current, C) actually occurred]
        Q2(C)   = Q2_WEIGHT × 100 × raw2(C) / Σ_all raw2          (a percentage)
      Votes only if (current -> C) is a legal bigram -- it was seen, in that
      order, at least once.  Given a legal bigram, each context token then votes
      if it was present in one of the sentences that bigram came from.

  score(C) = Q1(C) + Q2(C) (+ noise, added by select()).  EVERY candidate takes
  part in this, equally -- nothing is excluded from the ranking.

  Q3 and Q4 are NOT part of the score.  They are LEGALITY CHECKS applied at the
  very end, to the ranked list, just before a token is picked:

  Q3  LEGAL BIGRAM               (direction: current -> C only)
        legal3(C) = 1[(current, C) is a legal bigram]
      Has `current` ever been immediately followed by C?  (cat sat) counts;
      (sat cat) does not.

  Q4  AGREEMENT OF previous AND current   (previous, current -> C)
        legal4(C) = 1[(previous, current, C) is a trained consecutive triple]
      Do `previous` and `current`, as a pair, know C as the next token?
      "the cat sat on the mat because it was tired":  (the cat) knows -> sat;
      (the mat) knows -> because.  Order matters, and the three tokens must have
      been consecutive in one training sentence.
      Q4 only STARTS at two context tokens.  An agreement needs two tokens, so with
      only one real context token (a one-word prompt) Q4 is INACTIVE: it neither
      fires nor blocks anything, and Q3 alone decides legality.  It becomes active
      once `previous` and `current` are both real tokens (e.g. after the first
      generated token of a one-word prompt).

  PICKING (select()): rank every candidate by its total score, then take the
  highest-scored candidate that is legal.  Q3/Q4 work on the top
  LEGALITY_TOP_K scored candidates only (None = ALL of them; ties at the cut-off are
  kept): a candidate below that cut is never checked and never picked.  If the top candidate is not legal, try
  the next highest, and so on (within that window).  ("tired 34, it 33, was 32.6": tired is not a legal
  bigram after the current token, so "it" is picked.)  Legal means:
      stage 1  Q3 AND Q4          (the normal case, once Q4 is active)
      stage 2  Q3 only            (Q4 inactive -- one context token -- or NO candidate
                                   passes Q4, e.g. the previous+current pair was
                                   never seen together)
      stage 3  <EOS>              (nothing in the window is even a legal bigram: there is
                                   no legal continuation, so the model ends the text)
  The stage used is reported as trace["legality"].  Ties among the eligible
  candidates fall to the deterministic cascade (see choose()).

  The PROMPT is never checked -- Q3/Q4 judge only the CONTINUATION.  Any prompt is
  fine ("cat was on mat because"); with no usable `current`, Q3 passes for everyone,
  and with fewer than two real context tokens Q4 is inactive (see above).

NORMALIZATION (same as noise.py's normalize_row()): inference never sees raw
counts.  Each of Q1 and Q2 is turned into a percentage scoped to the query,
over ALL candidates:
    Qk%(C) = 100 × rawk(C) / Σ_{all candidates} rawk
so a question's evidence adds up to exactly 100% (a candidate with raw 0 stays 0;
if nobody has any, all stay 0).  Q1_WEIGHT / Q2_WEIGHT then scale those
percentages; they are mix weights, not count scales.

Rules shared by all four:
  - "Legal bigram" has ONE definition everywhere: a directed pair (a, b) where
    b immediately followed a at least once in training (adjacent_pairs()).
  - Reserved tokens (<PAD>/<UNK>/<BOS>) never vote and are never voted for.
  - A token never votes for itself: C == current is skipped in Q2 and Q3.
  - Q2 needs `current`; with None it casts no votes.
  - Evidence comes ONLY from the literal training sentences (model.token_relationships
    for Q1, the adjacency/witness indexes for Q2-Q4).  No Experience data source.

TIE-BREAK (choose()/select()): when the final scores tie, the winner is decided
by, in order: how often C literally followed `current` (bigram frequency), then
C's corpus-wide sentence count, then lowest token id.  Always deterministic.

MIXING WITH NOISE: select(..., extra_scores=...) accepts another scorer's final
scores and adds them before choosing, so the tie-break sees the combined total.
ivm itself never computes or inspects them.

Caching: build_cache()/enable_cache()/disable_cache() precompute Q1's per-token
rows for the full vocabulary (Q2-Q4 are already a few set lookups per step).
Same numbers either way; trace["cache_used"] reports which path ran.

Strict Mode's legacy opt-in tie-break, resolve_tie(), now ranks the tied
candidates by this same final score (unique top scorer wins, otherwise None =
"no opinion, fall through").
"""

from collections import defaultdict

from config import EOS, RESERVED, IVMConfig

# Q2's evidence is "which sentences contained the bigram (current, C)". True means every
# adjacent pair has witnesses, including a sentence's last pair (e.g. "." -> <EOS>), so
# Q2's idea of "legal bigram" is exactly the same set Q3/Q4 use.
Q2_INCLUDE_FINAL_PAIR = True

# One place that names the questions, in the order they are summed and displayed.
# (key in the trace, short column label, what it asks)
QUESTIONS = (
    ("q1", "Q1 (co-occur)",  "has each context token ever shared a sentence with C?"),
    ("q2", "Q2 (witness)",   "legal bigram current->C: how many context tokens were in its sentences?"),
    ("q3", "Q3 (legal bigram)", "is current->C a legal bigram? 1 = legal, 0 = not"),
    ("q4", "Q4 (agreement)",    "did (previous, current) ever precede C as a consecutive triple?"),
)


# ======= evidence builders (all literal, all from the model's own data) ==================

def bigram_frequencies(model, mode="strict"):
    """{(src, dst): count} -- literal consecutive-pair counts, used by the tie-break."""
    return model.bigram_counts()


def bigram_relationships(model):
    """{(source, candidate): {rel_id, ...}} -- the training sentences each bigram occurred in (Q2's evidence)."""
    return model.bigram_provenance(include_final=Q2_INCLUDE_FINAL_PAIR)


def adjacent_pairs(model):
    """{(from_token, to_token), ...} -- every DIRECTED legal bigram (Q3/Q4's evidence)."""
    return model.adjacent_pairs()


def parse_top_k(value):
    """Normalize a "how many top-scored tokens" setting: None / 0 / "all" / "" -> None (all);
    a positive int (or digit string) -> that int. Anything else raises ValueError."""
    if value is None or (isinstance(value, str) and value.strip().lower() in ("", "all", "none")):
        return None
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"legality top-k must be a positive integer or 'all', got {value!r}") from None
    if n < 0:
        raise ValueError(f"legality top-k must be a positive integer or 'all', got {value!r}")
    return n or None


def triples_by_pair(model):
    """{(a, m): {b, ...}} -- every consecutive trained triple (a, m, b), keyed by its first two
    tokens (Q4's evidence).  The knowledge store keeps a triple as (source=a, target=b, bridge=m)."""
    out = {}
    for source, target, bridge in model.triples():
        out.setdefault((source, bridge), set()).add(target)
    return out


def co_occurrence_index(token_rels, unified=None):
    """
    {token: {other_token: shared_sentence_count}} for every pair of non-reserved tokens that
    ever appear in the same training sentence (Q1's evidence). A token never co-occurs with
    itself. With `unified` it is read straight off the knowledge store; without, it is
    rebuilt by inverting token_rels.
    """
    if unified is not None:
        result = {}
        for t, rels in token_rels.items():
            if t in RESERVED or not rels:
                continue
            result[t] = {c: n for c, n in unified.co_row(t).items() if c not in RESERVED}
        return result

    rel_to_tokens = defaultdict(set)
    for tok, rels in token_rels.items():
        if tok in RESERVED or not rels:
            continue
        for r in rels:
            rel_to_tokens[r].add(tok)
    result = defaultdict(dict)
    for toks in rel_to_tokens.values():
        toks = list(toks)
        for i, ti in enumerate(toks):
            row = result[ti]
            for j, tj in enumerate(toks):
                if i != j:
                    row[tj] = row.get(tj, 0) + 1
    return dict(result)


class ImportanceVoteMatrix:
    """Precomputed evidence for Q1-Q4 plus the tie-break data. Build once, query many times."""

    def __init__(self):
        self._memos = {}        # derived lookup views (see _memo()) -- speed only, never serialized
        self._ctx_state = None  # Q1 counts for the last context seen (incremental) -- see _q1_counts()
        self._cand_memo = None  # remembered sorted/deduped candidate list -- see _prepare_candidates()
        self._token_rels = {}   # token -> {rel_id}, from model.token_relationships(min_len=3)
        self._q1_weight = IVMConfig.Q1_WEIGHT
        self._q2_weight = IVMConfig.Q2_WEIGHT
        self._legality_top_k = parse_top_k(IVMConfig.LEGALITY_TOP_K)   # None = all (see select())
        self._bigram_freq = {}  # (a, b) -> count        (tie-break)
        self._bigram_rels = {}  # (a, b) -> {rel_id}     (Q2)
        self._adjacent = set()  # {(a, b), ...} directed legal bigrams (Q3)
        self._triples = {}      # {(a, m): {b}} consecutive trained triples a m b (Q4)
        self._co_occurring = {} # token -> {other: shared_count}   (Q1)
        # Optional Q1 cache -- see build_cache(). Not serialized (re-derivable).
        self._token_cache = {}         # {t: {c: shared_count}}, shared_count > 0 only
        self._cache_candidates = None  # frozenset the cache was built for
        self._use_cache = False

    def set_legality_top_k(self, k):
        """Set how many of the top-scored tokens Q3/Q4 may work on: a positive int, or
        None / 0 / "all" for every candidate. Returns the stored value (None = all)."""
        self._legality_top_k = parse_top_k(k)
        return self._legality_top_k

    @property
    def legality_top_k(self):
        return self._legality_top_k

    @classmethod
    def build(cls, model, mode="strict", q1_weight=IVMConfig.Q1_WEIGHT, q2_weight=IVMConfig.Q2_WEIGHT,
              use_cache=False):
        ivm = cls()
        ivm._token_rels = model.token_relationships(min_len=3)
        ivm._q1_weight, ivm._q2_weight = q1_weight, q2_weight
        ivm._bigram_freq = bigram_frequencies(model, mode=mode)
        ivm._bigram_rels = bigram_relationships(model)
        ivm._adjacent = adjacent_pairs(model)
        ivm._triples = triples_by_pair(model)
        ivm._co_occurring = co_occurrence_index(ivm._token_rels, unified=model)
        if use_cache:
            ivm.enable_cache(model.all_candidate_tokens())
        return ivm

    # ======= optional Q1 cache =========================================================

    def build_cache(self, candidates):
        """Precompute, for exactly `candidates`, {t: {c: shared_count}} (shared_count > 0 only).
        Does not enable it -- see enable_cache()."""
        candidates = frozenset(candidates)
        cache = {}
        for t, row in self._co_occurring.items():
            out = {c: n for c, n in row.items() if n and c in candidates}
            if out:
                cache[t] = out
        self._token_cache = cache
        self._cache_candidates = candidates

    def enable_cache(self, candidates=None):
        """Turn the cache on, (re)building it first if `candidates` is given. Without
        `candidates` it reuses the last build, or raises ValueError if there is none."""
        if candidates is not None:
            self.build_cache(candidates)
        elif self._cache_candidates is None:
            raise ValueError("enable_cache() called with no `candidates` and "
                             "no prior build_cache() call -- nothing to enable")
        self._use_cache = True

    def disable_cache(self):
        """Back to live Q1. The built cache is kept, so enable_cache() with no arguments is instant."""
        self._use_cache = False

    # ==== derived lookup views (speed only -- never change a score) =======================

    def _memo(self, key, source, builder):
        m = self._memos.get(key)
        if m is not None and m[0] is source and m[1] == len(source):
            return m[2]
        value = builder(source)
        self._memos[key] = (source, len(source), value)
        return value

    def _adjacent_from(self):
        """{a: {b, ...}} view of the legal bigrams -- 'what has a ever been followed by'."""
        def build(adjacent):
            out = {}
            for a, b in adjacent:
                out.setdefault(a, set()).add(b)
            return out
        return self._memo("adjacent_from", self._adjacent, build)

    def _bigrams_from(self):
        """{a: {b: witness_rels}} view of Q2's evidence."""
        def build(bigram_rels):
            out = {}
            for (a, b), rels in bigram_rels.items():
                out.setdefault(a, {})[b] = rels
            return out
        return self._memo("bigrams_from", self._bigram_rels, build)

    def _rel_tokens(self):
        """{rel_id: {token, ...}} -- the inverse of self._token_rels."""
        def build(token_rels):
            out = {}
            for tok, rels in token_rels.items():
                for r in rels:
                    out.setdefault(r, set()).add(tok)
            return out
        return self._memo("rel_tokens", self._token_rels, build)

    def _prepare_candidates(self, candidates):
        """(sorted list, set) for `candidates`; the sort is remembered for the identical
        vocabulary list Open Mode passes every step."""
        if isinstance(candidates, (list, tuple)) and candidates:
            memo, n = self._cand_memo, len(candidates)
            if (memo is not None and memo[0] is candidates and memo[1] == n
                    and memo[2] == candidates[0] and memo[3] == candidates[-1]):
                return memo[4], memo[5]
            ordered = sorted(set(candidates))
            as_set = set(ordered)
            self._cand_memo = (candidates, n, candidates[0], candidates[-1], ordered, as_set)
            return ordered, as_set
        ordered = sorted(set(candidates))
        return ordered, set(ordered)

    # ============ the four questions ==================================================

    def _q1_counts(self, context_tokens, candidates_set, use_cache):
        """
        {c: how many context tokens have ever shared a sentence with c}  (Q1, unweighted).
        Three equivalent ways to get it: from the cache, incrementally for a set context
        (generation grows the context one token per step, so only NEW tokens' rows are added),
        or a plain pass for any other iterable (a list counts a repeated token each time).
        """
        token_rels = self._token_rels
        if use_cache:
            n = {}
            for t in context_tokens:
                if t in RESERVED:
                    continue
                for c in self._token_cache.get(t, ()):
                    n[c] = n.get(c, 0) + 1
            return n

        co = self._co_occurring
        if not isinstance(context_tokens, (set, frozenset)):
            n = {}
            for t in context_tokens:
                if t in RESERVED or not token_rels.get(t):
                    continue
                for c, shared in co.get(t, {}).items():
                    if shared and c in candidates_set:
                        n[c] = n.get(c, 0) + 1
            return n

        tokens = {t for t in context_tokens if t not in RESERVED}
        st = self._ctx_state
        if (st is not None and st["cands"] is candidates_set
                and st["co"] is co and st["co_len"] == len(co)
                and st["rels"] is token_rels and st["rels_len"] == len(token_rels)
                and st["tokens"] <= tokens):
            new_tokens = tokens - st["tokens"]
            if not new_tokens:
                return st["n"]
            n = dict(st["n"])           # copy, then publish: concurrent generations never see a half-update
        else:
            new_tokens, n = tokens, {}
        for t in new_tokens:
            if not token_rels.get(t):
                continue
            for c, shared in co.get(t, {}).items():
                if shared and c in candidates_set:
                    n[c] = n.get(c, 0) + 1
        self._ctx_state = {"cands": candidates_set, "co": co, "co_len": len(co),
                           "rels": token_rels, "rels_len": len(token_rels),
                           "tokens": frozenset(tokens), "n": n}
        return n

    def _q2_counts(self, current, context_tokens, candidates_set):
        """
        {c: number of context tokens that were in a sentence containing the bigram (current, c)}
        (Q2, unweighted). Only successors `current` was ever seen with can vote, so only those
        are visited. No entry (not 0) for a candidate whose bigram was never seen.
        """
        out = {}
        if current is None or current in RESERVED:
            return out
        succ = self._bigrams_from().get(current)
        if not succ:
            return out
        ctx_set = context_tokens if isinstance(context_tokens, (set, frozenset)) else None
        token_rels, rel_tokens = self._token_rels, None
        for c, witness_rels in succ.items():
            if c not in candidates_set or c in RESERVED or c == current or not witness_rels:
                continue
            if ctx_set is not None and len(witness_rels) * 4 < len(ctx_set):
                # Few witnessing sentences, many context tokens: collect the tokens in those
                # sentences once and intersect, instead of testing every context token.
                if rel_tokens is None:
                    rel_tokens = self._rel_tokens()
                pool = set()
                for r in witness_rels:
                    pool |= rel_tokens.get(r, ())
                voters = (pool & ctx_set) - RESERVED
                voters.discard(c)
                if voters:
                    out[c] = float(len(voters))
                continue
            n = 0
            for t in context_tokens or ():
                if t in RESERVED or t == c:
                    continue
                rels_t = token_rels.get(t)
                if rels_t and (rels_t & witness_rels):
                    n += 1
            if n:
                out[c] = float(n)
        return out

    def _q3_legal(self, current, candidates_set):
        """Q3: the candidates for which (current -> c) is a legal bigram. A token never follows
        itself and reserved tokens are never chosen. With no `current` there is nothing to
        check, so every candidate passes."""
        if current is None:
            return set(candidates_set)
        legal = self._adjacent_from().get(current, ())
        return {c for c in candidates_set if c in legal and c not in RESERVED and c != current}

    @staticmethod
    def _q4_active(previous, current):
        """Q4 is an agreement of TWO tokens, so it only starts once the context has two real
        (non-reserved) tokens. With one (e.g. a one-word prompt: previous is <BOS>) it is
        inactive -- it neither fires nor blocks, and Q3 alone decides."""
        return (previous is not None and previous not in RESERVED
                and current is not None and current not in RESERVED)

    def _q4_legal(self, previous, current, candidates_set):
        """Q4 (only meaningful when _q4_active): the candidates that (previous, current) know as
        the next token -- i.e. the consecutive triple (previous, current, c) was trained.
        Reserved tokens are never chosen. A (previous, current) pair never seen together passes
        nobody."""
        known = self._triples.get((previous, current), ())
        return {c for c in candidates_set if c in known and c not in RESERVED}

    # ==== scoring ========================================================================= 

    def score_candidates(self, candidates, context_tokens, current=None, previous=None):
        """
        ivm's score for EVERY candidate, all of them taking part equally: Q1 + Q2, each a
        normalized percentage (see the module docstring). Q3/Q4 are reported here as the legal
        sets but are NOT applied to the score -- select() applies them when it picks.
        Returns a trace dict:

            {"q1": {C: Q1}, "q2": {C: Q2},   # weight x percentage, candidates with votes only
             "q1_raw": {C: n}, "q2_raw": {C: n},   # raw vote counts
             "q1_pct": {C: %}, "q2_pct": {C: %},   # normalized 0-100 over ALL candidates, before the weight
             "q3": {C: 1.0}, "q4": {C: 1.0},   # candidates that pass each check (display)
             "legal3": set, "legal4": set,   # the same, as sets (legal4 is empty when Q4 is inactive)
             "q4_active": bool,   # False with fewer than two real context tokens: Q4 neither fires nor blocks
             "scores": {C: Q1+Q2},   # one entry per candidate, 0 if nothing voted
             "cache_used": bool}
        """
        candidates, candidates_set = self._prepare_candidates(candidates)
        context_tokens = context_tokens or []
        use_cache = (self._use_cache and self._cache_candidates is not None
                     and self._cache_candidates == candidates_set)

        raw1 = self._q1_counts(context_tokens, candidates_set, use_cache)
        raw2 = self._q2_counts(current, context_tokens, candidates_set)

        # Normalize Q1 and Q2 to percentages over ALL candidates (see NORMALIZATION above).
        def _pct(raw):
            total = sum(raw.values())
            return {c: 100.0 * n / total for c, n in raw.items()} if total else {}
        q1_pct, q2_pct = _pct(raw1), _pct(raw2)
        q1 = {c: self._q1_weight * p for c, p in q1_pct.items()}
        q2 = {c: self._q2_weight * p for c, p in q2_pct.items()}

        scores = dict.fromkeys(candidates, 0)
        for layer in (q1, q2):
            for c, v in layer.items():
                if c in scores:
                    scores[c] += v

        legal3 = self._q3_legal(current, candidates_set)
        q4_active = self._q4_active(previous, current)
        legal4 = self._q4_legal(previous, current, candidates_set) if q4_active else set()
        return {"q1": q1, "q2": q2,
                "q3": dict.fromkeys(legal3, 1.0), "q4": dict.fromkeys(legal4, 1.0),
                "legal3": legal3, "legal4": legal4, "q4_active": q4_active,
                "q1_raw": raw1, "q2_raw": raw2, "q1_pct": q1_pct, "q2_pct": q2_pct,
                "scores": scores, "cache_used": use_cache}

    # ==== choosing ======================================================================

    def _bigram_frequency(self, token, candidate):
        """How many times `candidate` literally followed `token` in training."""
        return self._bigram_freq.get((token, candidate), 0)

    def _global_frequency(self, candidate):
        """Candidate's corpus-wide sentence count -- last-resort tie-break only."""
        return len(self._token_rels.get(candidate, ()))

    def choose(self, scores, current=None):
        """
        Pick the winner from final `scores` ({candidate: score}), with the deterministic
        tie-break cascade. Returns (winner_or_None, stage) where stage says what decided it:
        "score" | "bigram_frequency" | "global_frequency" | "lowest_token_id".
        winner is None only if `scores` is empty.
        """
        if not scores:
            return None, "score"
        top_score = max(scores.values())
        top = sorted(c for c, s in scores.items() if s == top_score)
        if len(top) == 1:
            return top[0], "score"
        stage = "lowest_token_id"
        if current is not None:
            freqs = {c: self._bigram_frequency(current, c) for c in top}
            best = max(freqs.values())
            narrowed = sorted(c for c in top if freqs[c] == best)
            if len(narrowed) < len(top):
                stage = "bigram_frequency"
            top = narrowed
        if len(top) > 1:
            freqs = {c: self._global_frequency(c) for c in top}
            best = max(freqs.values())
            narrowed = sorted(c for c in top if freqs[c] == best)
            if len(narrowed) < len(top):
                stage = "global_frequency"
            top = narrowed
        return top[0], stage

    def select(self, candidates, context_tokens, current=None, previous=None, extra_scores=None):
        """
        Deterministic winner among `candidates`. Returns (winner_or_None, trace).

        `extra_scores` ({candidate: score}, optional) are ANOTHER scorer's final scores (model.py
        passes noise.py's); they are added to ivm's score to make the total every candidate is
        ranked by. Q3/Q4 then decide which candidates may be PICKED: the winner is the
        highest-total candidate that passes Q3 and Q4 (if the top one is not legal, the next
        highest is tried, and so on). If nobody passes Q4, Q3 alone decides; if nobody is a legal
        bigram, there is no legal continuation and the winner is <EOS>. Ties among the eligible fall to choose(). The trace is
        score_candidates()'s, plus
          "total_scores"     ivm + extra, EVERY candidate (== "scores" when there are no extras)
          "legality"         "q3+q4" | "q3" | "none" -- which check the eligible set came from
                             ("none": no legal continuation, the winner is <EOS>)
          "legality_top_k"   how many top-scored candidates Q3/Q4 worked on (None = all)
          "window_size"      how many candidates that actually was (>= top_k when scores tie)
          "eligible"         the candidates that were allowed to win (empty when "none")
          "final_scores"     total_scores for eligible candidates, 0 for the rest
          "tie_break_stage"  what decided the winner (see choose()), or
                             "no_legal_continuation_eos"
        """
        trace = self.score_candidates(candidates, context_tokens, current=current, previous=previous)
        totals = trace["scores"]
        if extra_scores:
            totals = dict(totals)
            for c, v in extra_scores.items():
                if c in totals:
                    totals[c] += v
        # Q3/Q4 only work on the top-K scored candidates (None = all). Ties at the cut-off stay in.
        k = self._legality_top_k
        window = None
        if k and k < len(totals):
            cut = sorted(totals.values(), reverse=True)[k - 1]
            window = {c for c, v in totals.items() if v >= cut}
        legal3, legal4 = trace["legal3"], trace["legal4"]
        if window is not None:
            legal3, legal4 = legal3 & window, legal4 & window
        both = (legal3 & legal4) if trace["q4_active"] else set()
        if both:
            eligible, legality = both, "q3+q4"
        elif legal3:
            eligible, legality = legal3, "q3"
        else:
            eligible, legality = set(), "none"
        if eligible:
            winner, stage = self.choose({c: totals[c] for c in eligible}, current)
        else:
            # No legal continuation after Q3/Q4: nothing may be picked, so the text ends.
            winner, stage = EOS, "no_legal_continuation_eos"
        trace["total_scores"] = totals
        trace["legality"] = legality
        trace["legality_top_k"] = k
        trace["window_size"] = len(window) if window is not None else len(totals)
        trace["eligible"] = eligible
        trace["final_scores"] = {c: (v if c in eligible else 0.0) for c, v in totals.items()}
        trace["tie_break_stage"] = stage
        return winner, trace

    def resolve_tie(self, candidates, context_tokens, current=None, previous=None):
        """
        Strict Mode's legacy opt-in tie-break: rank the tied `candidates` by ivm's final score.
        Returns the unique top scorer, or None ("no opinion -- fall through") if nobody scored
        or the top is still tied.
        """
        if not context_tokens:
            return None
        scores = self.score_candidates(candidates, context_tokens,
                                       current=current, previous=previous)["scores"]
        best = max(scores.values(), default=0)
        if best <= 0:
            return None
        top = [c for c, s in scores.items() if s == best]
        return top[0] if len(top) == 1 else None

    # ======== persistence ============================================================

    def to_dict(self):
        return {
            "token_rels": {str(t): sorted(r) for t, r in self._token_rels.items()},
            "q1_weight": self._q1_weight, "q2_weight": self._q2_weight,
            "bigram_freq": {f"{t}:{c}": n for (t, c), n in self._bigram_freq.items()},
            "bigram_rels": {f"{t}:{c}": sorted(r) for (t, c), r in self._bigram_rels.items()},
            "adjacent": [list(pair) for pair in self._adjacent],
            "triples": [[a, m, b] for (a, m), bs in self._triples.items() for b in sorted(bs)],
        }

    @classmethod
    def from_dict(cls, d, matrix=None):
        ivm = cls()
        ivm._token_rels = {int(t): set(r) for t, r in d["token_rels"].items()}
        ivm._q1_weight = d.get("q1_weight", IVMConfig.Q1_WEIGHT)
        ivm._q2_weight = d.get("q2_weight", IVMConfig.Q2_WEIGHT)
        ivm._bigram_freq = {}
        for key, n in d.get("bigram_freq", {}).items():
            t, c = key.split(":")
            ivm._bigram_freq[(int(t), int(c))] = n
        ivm._bigram_rels = {}
        for key, r in d.get("bigram_rels", {}).items():
            t, c = key.split(":")
            ivm._bigram_rels[(int(t), int(c))] = set(r)
        ivm._adjacent = {tuple(pair) for pair in d.get("adjacent", [])}
        ivm._triples = {}
        for a, m, b in d.get("triples", []):
            ivm._triples.setdefault((a, m), set()).add(b)
        ivm._co_occurring = co_occurrence_index(ivm._token_rels, unified=matrix)
        return ivm
