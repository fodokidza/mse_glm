"""
tokenizer.py — From-scratch Byte Pair Encoding tokenizer for MSE-GLM.

Special tokens (see config.py -- the single source of truth for these
ids; re-exported here so existing `from tokenizer import EOS`-style
imports elsewhere in the codebase keep working unchanged):
    <PAD> = 0   reserved
    <UNK> = 1   unknown character fallback
    <BOS> = 2   prepended to every encoded sequence
    <EOS> = 3   appended only during training (encode_for_training)

Punctuation (see config.TokenizerConfig.PUNCTUATION) is preserved as
its own token(s) rather than stripped -- see normalize()'s docstring
for exactly how, and decode()'s for how it's put back with correct
spacing.
"""

import json
import re
from collections import Counter

from config import PAD, UNK, BOS, EOS, SPECIAL_TOKENS, TokenizerConfig

_PUNCT = TokenizerConfig.PUNCTUATION
_NO_SPACE_BEFORE = TokenizerConfig.NO_SPACE_BEFORE

_KEEP_CHARS_RE = re.compile(r"[^a-z0-9\s" + re.escape("".join(sorted(_PUNCT))) + r"]")
_ISOLATE_PUNCT_RE = re.compile("([" + re.escape("".join(sorted(_PUNCT - {"'"}))) + "])")
# An apostrophe NOT flanked by alnum on both sides is a standalone
# mark (a quote, e.g. 'hello') rather than a contraction/possessive --
# isolate only that case; leave "don't"/"cat's" attached as one word.
_LONE_APOSTROPHE_RE = re.compile(r"(?<![a-z0-9])'|'(?![a-z0-9])")
_WS_RE = re.compile(r"\s+")
# Capturing group: sentence-boundary delimiters survive split() so
# their real punctuation (.!?) can be reattached to the sentence that
# precedes them -- see _finalize_sentences(). A bare run of '\n's is a
# structural separator only and contributes no punctuation token.
_SENT_SPLIT_RE = re.compile(r"([.!?\n]+)")


def normalize(text: str) -> str:
    """
    Lowercase, drop anything that isn't a letter/digit/whitespace/
    allowed punctuation mark (TokenizerConfig.PUNCTUATION), then
    isolate each punctuation mark with surrounding spaces so it
    tokenizes as its own "word" (and, via BPE's per-character vocab
    seeding, ends up as its own atomic token) instead of being fused
    into -- or silently discarded from -- the word next to it.

    Apostrophes are the one exception: "don't"/"cat's" keep the
    apostrophe attached to the word on both sides (a contraction or
    possessive marker), while a standalone quote mark ('hello') is
    still isolated like any other punctuation.
    """
    text = text.lower()
    text = _KEEP_CHARS_RE.sub(" ", text)
    text = _ISOLATE_PUNCT_RE.sub(r" \1 ", text)
    text = _LONE_APOSTROPHE_RE.sub(" ' ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def _finalize_sentences(parts):
    """
    `parts` is the result of _SENT_SPLIT_RE.split() (capturing group,
    so delimiters survive) -- alternating body, delimiter, body, ...,
    always ending on a body (possibly empty, possibly with no matching
    delimiter after it at all). Reattaches the REAL terminal
    punctuation in each delimiter (., !, ?) to the sentence body just
    before it so it survives into the token stream instead of being
    discarded; a bare run of '\\n's contributes no punctuation of its
    own. Shared by split_sentences() and stream_word_freq() so there
    is exactly one implementation of this rule, not two kept in sync
    by hand.
    """
    out = []
    n = len(parts)
    i = 0
    while i < n:
        body = parts[i].strip()
        punct = ("".join(ch for ch in parts[i + 1] if ch in ".!?")
                 if i + 1 < n else "")
        sent = f"{body} {punct}".strip() if punct else body
        if sent:
            out.append(sent)
        i += 2
    return out


def split_sentences(text: str):
    return _finalize_sentences(_SENT_SPLIT_RE.split(text))


def stream_word_freq(path: str, word_freq: Counter,
                      chunk_size: int = TokenizerConfig.STREAM_CHUNK_SIZE) -> int:
    """
    Accumulate word frequencies from one file into an existing Counter,
    reading in fixed-size chunks so the file's full text is never held
    in memory at once (only the current chunk + a small carry-over
    buffer for a sentence split across a chunk boundary).

    Mutates `word_freq` in place so callers can share one Counter across
    many files (see train_corpus.py) without concatenating their text
    first. Returns the number of complete sentences seen in this file.
    """
    buffer = ""
    sentence_count = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            buffer += chunk
            parts = _SENT_SPLIT_RE.split(buffer)
            buffer = parts.pop()  # incomplete tail (no delimiter yet) -- carry over
            for sent in _finalize_sentences(parts):
                sentence_count += 1
                for word in normalize(sent).split(" "):
                    if word:
                        word_freq[word] += 1
    if buffer.strip():
        sentence_count += 1
        for word in normalize(buffer).split(" "):
            if word:
                word_freq[word] += 1
    return sentence_count


class BPETokenizer:
    def __init__(self, vocab_size: int = TokenizerConfig.DEFAULT_VOCAB_SIZE):
        self.vocab_size = vocab_size
        self.token_to_id = dict(SPECIAL_TOKENS)
        self.id_to_token = {v: k for k, v in SPECIAL_TOKENS.items()}
        self.merges = []  # ordered list of (a, b) -> merged_string, applied in order
        self._word_ids_cache = {}  # memoizes _ids_for_word(word) -> ids, since natural
                                    # text repeats a small set of distinct words very
                                    # often; avoids recomputing the full merge pass
                                    # for every occurrence of the same word.

    # ---------------------------------------------------------------- train
    def train(self, corpus: str):
        sentences = split_sentences(corpus)
        word_freq = Counter()
        for sent in sentences:
            for word in normalize(sent).split(" "):
                if word:
                    word_freq[word] += 1
        self._train_from_word_freq(word_freq)

    def train_from_file(self, path: str, chunk_size: int = TokenizerConfig.STREAM_CHUNK_SIZE):
        word_freq = Counter()
        stream_word_freq(path, word_freq, chunk_size)
        self._train_from_word_freq(word_freq)

    def _train_from_word_freq(self, word_freq: Counter):
        next_id = max(SPECIAL_TOKENS.values()) + 1
        # every distinct character seen becomes a base vocab entry
        chars = set()
        for w in word_freq:
            chars.update(list(w))
        for c in sorted(chars):
            if c not in self.token_to_id:
                self.token_to_id[c] = next_id
                self.id_to_token[next_id] = c
                next_id += 1

        # word -> tuple of symbols (starts as chars)
        word_symbols = {w: list(w) for w in word_freq}
        self._run_bpe_merges(word_freq, word_symbols, next_id, self.vocab_size)

    def _run_bpe_merges(self, word_freq: Counter, word_symbols: dict, next_id: int,
                         target_vocab_size: int, on_merge=None):
        """
        Repeatedly merges the most frequent adjacent symbol pair until
        self.vocab_size is reached, mutating word_symbols/self.merges/
        self.token_to_id/self.id_to_token in place.

        INCREMENTAL pair-count maintenance, not a full rescan per merge.
        The naive version recomputed pair_counts by scanning every word
        from scratch on EVERY iteration, then did a second full pass
        over every word to apply the winning merge -- O(vocab_size x
        total corpus symbols) overall, the classic naive-BPE bottleneck
        (on a 40k-word/vocab-4000 synthetic benchmark this took ~18s).
        Only one pair changes state per iteration (the one just
        merged), so instead we track, per pair, which words currently
        contain it (`pair_words`) and keep a running weighted count
        (`pair_counts`) that we adjust by exactly the words affected by
        THIS merge, rather than rescanning the whole corpus -- same
        final vocabulary/merge list for any corpus without an exact
        tie in top pair frequency (real corpora essentially never hit
        one; test.py's full suite, including tokenizer round-trips,
        passes unchanged). This mirrors the same "stop rescanning
        everything on every call" fix already applied to
        BridgeMatrix.cluster_axis()/RelationshipMatrix.
        relationships_for_triple() in graph.py.
        """
        from collections import defaultdict

        pair_counts = Counter()
        pair_words = defaultdict(set)
        for w, freq in word_freq.items():
            symbols = word_symbols[w]
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pair_counts[pair] += freq
                pair_words[pair].add(w)

        while len(self.token_to_id) < target_vocab_size:
            if not pair_counts:
                break
            (a, b), count = pair_counts.most_common(1)[0]
            merged = a + b
            if merged not in self.token_to_id:
                self.token_to_id[merged] = next_id
                self.id_to_token[next_id] = merged
                next_id += 1
            self.merges.append((a, b))
            if on_merge is not None:
                # Progress/display hook (e.g. train.py's live display) --
                # fires with the exact same (a, b, merged, count) a caller
                # hand-rolling this loop itself would have seen, so callers
                # never need their own copy of the merge loop just to
                # observe it.
                on_merge(a, b, merged, count, len(self.token_to_id))

            # Only words that actually contain (a, b) are touched --
            # every other word's pairs are untouched by this merge.
            affected = pair_words.pop((a, b), ())
            for w in affected:
                freq = word_freq[w]
                symbols = word_symbols[w]

                # Remove this word's old pair contributions.
                for i in range(len(symbols) - 1):
                    old_pair = (symbols[i], symbols[i + 1])
                    pair_counts[old_pair] -= freq
                    if pair_counts[old_pair] <= 0:
                        del pair_counts[old_pair]
                    pw = pair_words.get(old_pair)
                    if pw is not None:
                        pw.discard(w)
                        if not pw:
                            del pair_words[old_pair]

                # Apply the merge to this word only.
                new_symbols = []
                i = 0
                while i < len(symbols):
                    if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                        new_symbols.append(merged)
                        i += 2
                    else:
                        new_symbols.append(symbols[i])
                        i += 1
                word_symbols[w] = new_symbols

                # Add this word's new pair contributions.
                for i in range(len(new_symbols) - 1):
                    new_pair = (new_symbols[i], new_symbols[i + 1])
                    pair_counts[new_pair] += freq
                    pair_words[new_pair].add(w)
        return next_id

    # ------------------------------------------------------------- encode
    def _apply_merges(self, word: str):
        symbols = list(word)
        for a, b in self.merges:
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                    new_symbols.append(a + b)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        return symbols

    def _ids_for_word(self, word: str):
        cached = self._word_ids_cache.get(word)
        if cached is not None:
            return cached
        ids = []
        for sym in self._apply_merges(word):
            ids.append(self.token_to_id.get(sym, UNK))
        self._word_ids_cache[word] = ids
        return ids

    def encode(self, text: str):
        ids = [BOS]
        norm = normalize(text)
        for word in norm.split(" "):
            if word:
                ids.extend(self._ids_for_word(word))
        return ids

    def encode_for_training(self, text: str):
        ids = self.encode(text)
        ids.append(EOS)
        return ids

    def decode(self, ids):
        words = []
        current = ""
        for i in ids:
            if i in (PAD, UNK, BOS, EOS):
                if current:
                    words.append(current)
                    current = ""
                continue
            tok = self.id_to_token.get(i, "")
            if len(tok) == 1:
                current += tok
            else:
                if current:
                    words.append(current)
                    current = ""
                words.append(tok)
        if current:
            words.append(current)

        # Detokenize with punctuation-aware spacing: a "word" made up
        # entirely of punctuation marks (a single mark, or a BPE-merged
        # run like "...") hugs the word before it -- no inserted space
        # -- for every mark in TokenizerConfig.NO_SPACE_BEFORE (every
        # mark except an opening parenthesis). Turns ["cat", ".",
        # "dog"] back into "cat. dog" instead of "cat . dog".
        out = ""
        for w in words:
            if not w:
                continue
            if not out:
                out = w
            elif all(ch in _NO_SPACE_BEFORE for ch in w):
                out += w
            else:
                out += " " + w
        return out

    # ------------------------------------------------------------ persist
    @property
    def vocab_size_actual(self):
        return len(self.token_to_id)

    def extend_vocab(self, corpus: str, target_vocab_size: int):
        """
        Grow the vocabulary using a NEW corpus, without touching any
        existing token id. Every character and merge already in
        token_to_id keeps the exact same id it had before, so every
        Edge/Bridge/Relationship triple built under the old vocabulary
        stays byte-for-byte valid. Only new characters and new merges
        (learned from `corpus` alone) get appended on top.

        This is a real limitation to know about: word-frequency counts
        from whatever corpus originally trained this tokenizer are not
        retained anywhere, so the merges chosen here are picked using
        ONLY the new corpus's frequencies. This is not the same as
        retraining from scratch on the concatenation of old + new text
        -- it will not necessarily pick the same merges a joint retrain
        would -- but it never invalidates anything already learned, and
        it's the only option that doesn't require keeping the entire
        training history around forever.

        Returns the number of new vocabulary entries actually added
        (0 if target_vocab_size <= current vocab_size_actual, or if the
        new corpus has no pairs left to merge).
        """
        if target_vocab_size <= self.vocab_size_actual:
            return 0

        sentences = split_sentences(corpus)
        word_freq = Counter()
        for sent in sentences:
            for word in normalize(sent).split(" "):
                if word:
                    word_freq[word] += 1
        if not word_freq:
            return 0

        start_size = self.vocab_size_actual
        next_id = max(self.token_to_id.values()) + 1

        # New base characters this corpus introduces that the existing
        # vocabulary has never seen (e.g. digits, if the original corpus
        # never had any) -- added first, same as a from-scratch train.
        chars = set()
        for w in word_freq:
            chars.update(list(w))
        for c in sorted(chars):
            if c not in self.token_to_id and len(self.token_to_id) < target_vocab_size:
                self.token_to_id[c] = next_id
                self.id_to_token[next_id] = c
                next_id += 1

        # Re-apply every merge already learned, in the original order,
        # so this corpus's words start from the SAME symbol state a
        # from-scratch encode() would have produced -- only then do we
        # start choosing genuinely new merges on top.
        word_symbols = {w: self._apply_merges(w) for w in word_freq}
        self._run_bpe_merges(word_freq, word_symbols, next_id, target_vocab_size)

        self._word_ids_cache.clear()  # new merges can change how known words split
        return self.vocab_size_actual - start_size

    def save(self, path: str):
        data = {
            "vocab_size": self.vocab_size,
            "token_to_id": self.token_to_id,
            "merges": self.merges,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(vocab_size=data["vocab_size"])
        tok.token_to_id = data["token_to_id"]
        tok.id_to_token = {v: k for k, v in tok.token_to_id.items()}
        tok.merges = [tuple(m) for m in data["merges"]]
        return tok
