
"""
config.py — single source of truth for every tunable constant in
MSE-GLM: reserved token ids, IVM vote weights, default vocab/generation
sizes, Cluster Interpreter / Context Trigger thresholds, tokenizer
punctuation handling, and the HTTP server's defaults.

Nothing else in the codebase should hardcode one of these values --
import it from here instead. Two independent copies of the same
constant is exactly how train.py's hand-rolled BPE loop and
tokenizer.py's silently drifted apart before (see test.py's
test_train_py_cli_path_matches_model_api docstring); this file exists
so that class of bug can't happen again for any default value, not
just the ones that already bit us once.

Every value here is a DEFAULT -- callers can still override any of
them per-call (e.g. `ImportanceVoteMatrix.build(model,
important_weight=0.2)`); this file only fixes what a caller gets if
they don't.
"""

# ==== Reserved / special token ids ================================
# Shared by tokenizer.py (which owns encode/decode), ctm.py/ivm.py
# (which exclude these ids from every vote layer), and inference.py
# (EOS as the termination sentinel). One definition, everyone else
# imports it -- ctm.py used to keep its own separate literal
# `RESERVED = {0, 1, 2, 3}` in sync by hand.
PAD, UNK, BOS, EOS = 0, 1, 2, 3
# WORD_BOUND: tokenizer.py's CharWordTokenizer only --
# marks "a new stage-1-spelled word starts here" in the token stream,
# immediately before a single-character word or a fallback-spelled
# multi-character word (see tokenizer.py's module docstring).
# Without it, two such words sitting next to each other with nothing
# else between them (no stage-2 word, no punctuation) are
# indistinguishable from one longer word during decode() -- there is
# nothing else in a flat id stream to mark where the first one ends.
# Deliberately NOT in RESERVED, same treatment as EOS: the model needs
# to be able to select it during generation too (real graph edges into
# and out of it), or a MODEL-GENERATED fallback-spelled word would
# have this exact gluing problem with no way to fix it after the fact.
WORD_BOUND = 4
SPECIAL_TOKENS = {"<PAD>": PAD, "<UNK>": UNK, "<BOS>": BOS, "<EOS>": EOS, "<WORD_BOUND>": WORD_BOUND}
RESERVED = frozenset({PAD, UNK, BOS})


class TokenizerConfig:
    # BPETokenizer(vocab_size=...) when nothing else is specified
    # (model.py's MSEGraphLanguageModel default).
    DEFAULT_VOCAB_SIZE = 2000
    # train.py's CLI `--vocab-size` default -- kept as its own named
    # constant (not just reused from DEFAULT_VOCAB_SIZE) because the
    # CLI's documented default (README section 1) is intentionally
    # smaller than the library default; the two are allowed to differ,
    # but each is now a single named value instead of a bare literal.
    TRAIN_CLI_DEFAULT_VOCAB_SIZE = 1000
    # Bytes read per chunk in stream_word_freq()/train_from_file(),
    # so a large --corpus file is never fully loaded into memory just
    # to count word frequencies.
    STREAM_CHUNK_SIZE = 1 << 20

    # Punctuation marks normalize() preserves as their own token(s)
    # instead of silently discarding (the old behavior). Anything
    # that's neither a letter/digit, whitespace, nor in this set is
    # still stripped. An apostrophe directly between two letters
    # ("don't", "cat's") is treated as a contraction/possessive marker
    # and kept attached to its word rather than isolated -- see
    # normalize()'s own comment for the exact rule.
    PUNCTUATION = frozenset(".,!?;:()\"'-")
    # Marks that "hug" the token before them on decode() (no inserted
    # space) -- every mark here except an opening parenthesis, which
    # instead hugs the token that follows it.
    NO_SPACE_BEFORE = PUNCTUATION - {"("}


class IVMConfig:
    """Default mix weights for ivm.py's two SCORED questions, Q1 (co-occurrence)
    and Q2 (witnessed legal bigram) -- see ivm.py's module docstring. Each
    question's evidence is first normalized to a percentage over all
    candidates, so these are MIX weights, not count scales. Q3/Q4 (legality
    checks) carry no weight: they gate who may be picked, they don't score."""
    Q1_WEIGHT = 0.5
    Q2_WEIGHT = 0.05
    # How many of the top-scored candidates Q3/Q4 may work on. None = ALL of
    # them; a positive int N = only the top N (ties at the cut-off stay in).
    # If none of the window is a legal continuation the winner is <EOS>.
    LEGALITY_TOP_K = None


class NoiseConfig:
    """noise.py has no internal weight. model.py scales noise's final score by
    INFERENCE_WEIGHT when it adds it to ivm's score at inference."""
    INFERENCE_WEIGHT = 0.5

    # Which of noise.py's four stages are ACTIVE. Flip any to False to switch
    # that stage off:
    #   stage1  literal sentence membership
    #   stage2  sentence-global knowledge
    #   stage3  token-global (deduped) knowledge
    #   stage4  shared-knowledge attention
    # A disabled stage is never computed (so turning off stage4 also skips
    # building its whole-corpus tables), its raw value is 0 everywhere, and it
    # is left out of the final average: final = mean of the ENABLED stages'
    # percentages, so final stays on the same 0-100 scale whichever subset is on.
    # If every stage is disabled, noise's final score is 0 for every candidate
    # (i.e. noise contributes nothing and ivm decides alone).
    #
    # Read at call time (NoiseConfig.enabled_stages()), not copied at import.
    # Cached rows are NOT recomputed if you change this mid-session: rebuild
    # model.token_vocab / model.noise_index (or restart) after changing it.
    STAGES_ENABLED = {
        "stage1": True,
        "stage2": True,
        "stage3": True,
        "stage4": False,
    }

    @classmethod
    def enabled_stages(cls):
        """Tuple of enabled stage keys, always in stage1..stage4 order."""
        return tuple(k for k in ("stage1", "stage2", "stage3", "stage4")
                     if cls.STAGES_ENABLED.get(k, False))


class ScoresDisplayConfig:
    """
    How many candidates chat.py's `/scores <prompt>` and analyse.py's
    `open-scores <prompt>` print to the screen by default.

    Open Mode always SCORES the entire vocabulary -- that's the whole
    point of full-vocab candidates (see ivm.py/model.py's
    open_mode_candidate_scores() docstring) and this setting never
    changes that: the winner and every tie-break decision are still
    computed from every single candidate's score, exactly as before.
    This only limits what gets rendered afterward, because a vocab of
    a few thousand tokens turns into a few thousand unreadable rows on
    a terminal or REPL screen.

    The winner is always shown even when it doesn't land in the top N
    by score -- callers should never let truncation silently hide which
    token was actually picked.

    Both call sites accept a per-call override (chat.py's `/scores
    --top <n> <prompt>` / `--all`, analyse.py's `open-scores --top-n
    <n>` / `--top-n 0`); this is only the default when no override is
    given. `--json` output (analyse.py) is never truncated, since a
    piped/persisted result is exactly the case where you DO want every
    candidate.
    """
    TOP_N = 25


class CTMConfig:
    # ContextTriggerMatrix.build()'s minimum support threshold --
    # a cluster member needs at least this many witnessing
    # relationships before it counts as a real context trigger.
    MIN_SUPPORT = 1


class InterpretConfig:
    """Defaults for interpret.py's cluster-naming / zero-cluster-mining
    functions and model.py's matching wrapper methods -- see
    interpret.py's module docstring for what each one means."""
    MIN_COVERAGE = 0.5
    MIN_SIGNALS = 2
    TOP_N = 5
    MAX_PER_CLUSTER = 3
    MIN_GROUP_SIZE = 2
    # build_interpreter_matrix()'s internal per-cluster interpret_cluster()
    # call uses a wider top_n than the public default above, so the
    # matrix itself is built from more candidate evidence than a single
    # interpret_cluster() call would show by default.
    BUILD_MATRIX_TOP_N = 50


class GenerationConfig:
    # model.generate()/InferenceEngine.generate()'s default step budget.
    MAX_TOKENS = 40
    # chat.py's --max-tokens CLI default -- deliberately smaller, for
    # snappier interactive turns.
    CHAT_CLI_DEFAULT_MAX_TOKENS = 30
    # server.py's /chat HTTP endpoint default when the client omits
    # max_tokens -- deliberately larger, since a web reply reads better
    # a bit longer than a terminal chat turn.
    SERVER_DEFAULT_MAX_TOKENS = 80


class TrainCorpusConfig:
    # train.py's --batch-size default: how many files are merged
    # into the graph per train_incremental() step. 1 is the safest
    # default (smallest per-step memory footprint), not the fastest --
    # see model.train_from_source()'s docstring for the cost tradeoff.
    DEFAULT_BATCH_SIZE = 10


class ServerConfig:
    DEFAULT_PORT = 5000
    # Requests allowed per IP per rolling window in server.py's
    # in-memory rate limiter (see server.py's own comment: fine for
    # local/demo use, not a substitute for a real WSGI-level limiter).
    RATE_LIMIT_PER_MINUTE = 30
