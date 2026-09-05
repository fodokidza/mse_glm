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

# ── Reserved / special token ids ────────────────────────────────────
# Shared by tokenizer.py (which owns encode/decode), ctm.py/ivm.py
# (which exclude these ids from every vote layer), and inference.py
# (EOS as the termination sentinel). One definition, everyone else
# imports it -- ctm.py used to keep its own separate literal
# `RESERVED = {0, 1, 2, 3}` in sync by hand.
PAD, UNK, BOS, EOS = 0, 1, 2, 3
SPECIAL_TOKENS = {"<PAD>": PAD, "<UNK>": UNK, "<BOS>": BOS, "<EOS>": EOS}
RESERVED = frozenset({PAD, UNK, BOS, EOS})


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
    """Default weights for ivm.py's seven vote layers (V1-V7) -- see
    ivm.py's module docstring for what each one means. important_weight,
    influence_weight, and context_influence_weight are deliberately
    smaller than context_weight BY DESIGN (see that docstring); if you
    retune these, that ratio is a property you need to preserve
    yourself, not something enforced automatically."""
    IMPORTANT_WEIGHT = 0.4            # V1
    INFLUENCE_WEIGHT = 0.0           # V2
    CONTEXT_WEIGHT = 1.0              # V3
    CONTEXT_INFLUENCE_WEIGHT = 0.0   # V4
    BIGRAM_WITNESS_WEIGHT = 1.8       # V5
    ADJACENCY_WEIGHT = 1.0            # V6
    PREV_CURRENT_WEIGHT = 1.7         # V7


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
    # train_corpus.py's --batch-size default: how many files are merged
    # into the graph per train_incremental() step. 1 is the safest
    # default (smallest per-step memory footprint), not the fastest --
    # see train_corpus.py's own module docstring for the cost tradeoff.
    DEFAULT_BATCH_SIZE = 1


class ServerConfig:
    DEFAULT_PORT = 5000
    # Requests allowed per IP per rolling window in server.py's
    # in-memory rate limiter (see server.py's own comment: fine for
    # local/demo use, not a substitute for a real WSGI-level limiter).
    RATE_LIMIT_PER_MINUTE = 30
