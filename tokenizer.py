"""
tokenizer.py — MSE-GLM's tokenizer: a two-stage character/word design.

This used to be two files (this one holding a from-scratch Byte Pair
Encoding tokenizer, char_tokenizer.py holding the newer char/word
design) plus a "which one is actually wired into model.py" question
to answer. There is now exactly one tokenizer, in exactly one file:
BPE has been retired for good -- not just unused, but deleted -- and
CharWordTokenizer, below, is it. BPE's iterative merge-pair scan was
the single slowest step in training: on the corpus this was measured
against, it took roughly 25 minutes. The two-pass design below (learn
every character once, then pre-define common words once, no merge
loop to iterate at all) takes about 47 seconds on the same corpus --
not a tuning difference, a different algorithm with a different cost
shape entirely.

No merge algorithm, no subword pieces. Two stages, one shared id
space, stage 2 built strictly on top of stage 1:

  STAGE 1 (CharacterVocabulary) -- learn every unique CHARACTER in the
  corpus, each getting its own id. That's the entire job: no merging,
  no frequency ranking, just "which characters exist at all".
  Case-sensitive -- 'a' and 'A' are two different characters, each
  with its own id, because the corpus can and does distinguish them.

  STAGE 2 (TokenVocabulary) -- pre-define an id for every MULTI-
  character word worth having one, so the model isn't forced to spend
  several ids spelling out "the" every single time it shows up. A
  word's stored "definition" is its RECIPE -- the sequence of stage-1
  character ids that spells it (e.g. "cat" is stored as [id('c'),
  id('a'), id('t')]), not the literal string "cat" -- a handful of
  small ints, not a copy of the word. Stage 2 is an OPTIMIZATION, not
  a requirement: every job it does, stage 1 could also do alone, just
  by spelling every word out one character at a time. Stage 2 exists
  purely so common words get a single id instead of several, and
  purely because it's fast to build (one frequency sort, no
  iteration) is what makes skipping it for rare words an easy trade
  rather than a costly one.

  STAGE 2 IS CASE-INSENSITIVE -- unlike stage 1. "The", "the", and
  "THE" are matched to and share exactly ONE stage-2 entry (see
  TokenVocabulary.build()), so a word doesn't fragment the
  pre-defined-word budget across every capitalization it happens to
  appear in -- a sentence-initial "The" is the same word as a
  mid-sentence "the", not a different one. The tradeoff: decoding a
  stage-2 id always gives back that ONE stored (lowercase) spelling,
  never the original occurrence's exact casing -- see decode()'s
  docstring. This is scoped to stage 2 alone: a word stage 2 has no
  entry for still falls back to EXACT, case-preserving spelling
  through stage 1 (see CharWordTokenizer._encode_word()), and stage 1
  itself stays fully case-sensitive throughout.

  STAGE 2 NEVER STORES A ONE-CHARACTER "WORD" -- that would just be a
  second copy of something stage 1 already has an id for. A
  single-character word (an "a", an "I", a lone "." or "!") IS its
  stage-1 character id directly; stage 2 only ever assigns new ids to
  words of two or more characters.

  ONE SHARED ID SPACE, not two separate ones: stage 1's character ids
  and stage 2's word ids are drawn from the same increasing sequence,
  right after the reserved specials (<PAD>/<UNK>/<BOS>/<EOS>/
  <WORD_BOUND>). This is what lets a fallback-spelled word (see below)
  sit directly in the token stream CharWordTokenizer.encode() hands
  back -- every id in that stream, whichever stage minted it, means
  exactly one thing to decode() and to the rest of this codebase.

TRUE ZERO VOCABULARY LOSS FOR ANY WORD BUILT FROM KNOWN CHARACTERS:
stage 2 is a lookup table of PRE-DEFINED words, capped by
vocab_size -- but a word missing from that table is not the end of
the story the way it would be for a closed word-level vocabulary,
because stage 1 can build ANY word on demand from characters it
already knows, the same way stage 2's own recipes are built. A word
stage 2 has no entry for is simply spelled out live, character by
character, through stage 1 -- see CharWordTokenizer._encode_word().
<UNK> is only ever reached now for a genuinely unseen CHARACTER (one
that never appeared anywhere in training at all) -- not for an unseen
word.

<WORD_BOUND> (see config.py) is what makes two consecutive
stage-1-spelled words unambiguous at decode() time -- inserted before
every single-character word and every fallback-spelled multi-
character word, so a flat id stream always has an explicit marker for
"a new such word starts here" instead of relying on guesswork about
where one ends and the next begins.

Sentence splitting (split_sentences()) and case-folding word
segmentation (normalize(), still used directly by analyse.py's own
word-frequency analysis, independent of this tokenizer) live in this
same file now too -- they were always generic, never specific to
whichever vocabulary-building algorithm happened to consume them, so
there was never a real reason for them to live in a separate module.
"""

import json
import re

from config import PAD, UNK, BOS, EOS, WORD_BOUND, SPECIAL_TOKENS, TokenizerConfig

_PUNCT = TokenizerConfig.PUNCTUATION
_NO_SPACE_BEFORE = TokenizerConfig.NO_SPACE_BEFORE
_WS_RE = re.compile(r"\s+")

# Capturing group: sentence-boundary delimiters survive split() so
# their real punctuation (.!?) can be reattached to the sentence that
# precedes them -- see _finalize_sentences(). A bare run of '\n's is a
# structural separator only and contributes no punctuation token.
_SENT_SPLIT_RE = re.compile(r"([.!?\n]+)")

# normalize()'s own regexes -- case-FOLDING (lowercase-only character
# class), used only by normalize() itself.
_KEEP_CHARS_RE = re.compile(r"[^a-z0-9\s" + re.escape("".join(sorted(_PUNCT))) + r"]")
_ISOLATE_PUNCT_RE = re.compile("([" + re.escape("".join(sorted(_PUNCT - {"'"}))) + "])")
_LONE_APOSTROPHE_RE = re.compile(r"(?<![a-z0-9])'|'(?![a-z0-9])")

# segment()'s own regexes -- case-PRESERVING (stage 1 is explicitly
# case-sensitive, see module docstring), otherwise the same rule.
# Separate compiled patterns from normalize()'s above (not shared)
# specifically because of that one case-sensitivity difference.
_CS_KEEP_CHARS_RE = re.compile(r"[^a-zA-Z0-9\s" + re.escape("".join(sorted(_PUNCT))) + r"]")
_CS_ISOLATE_PUNCT_RE = re.compile("([" + re.escape("".join(sorted(_PUNCT - {"'"}))) + "])")
_CS_LONE_APOSTROPHE_RE = re.compile(r"(?<![a-zA-Z0-9])'|'(?![a-zA-Z0-9])")


def normalize(text: str) -> str:
    """
    Lowercase, drop anything that isn't a letter/digit/whitespace/
    allowed punctuation mark (TokenizerConfig.PUNCTUATION), then
    isolate each punctuation mark with surrounding spaces so it
    tokenizes as its own "word" instead of being fused into -- or
    silently discarded from -- the word next to it.

    Apostrophes are the one exception: "don't"/"cat's" keep the
    apostrophe attached to the word on both sides (a contraction or
    possessive marker), while a standalone quote mark ('hello') is
    still isolated like any other punctuation.

    Case-FOLDING -- used directly by analyse.py's own word-frequency
    analysis, independent of this tokenizer. This tokenizer's own
    word segmentation is segment() below, which keeps case distinct
    (stage 1 is explicitly case-sensitive, see module docstring).
    """
    text = text.lower()
    text = _KEEP_CHARS_RE.sub(" ", text)
    text = _ISOLATE_PUNCT_RE.sub(r" \1 ", text)
    text = _LONE_APOSTROPHE_RE.sub(" ' ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def segment(text: str):
    """
    Case-PRESERVING word segmentation for stage 2 -- otherwise the
    exact same rule normalize() uses above: drop anything that isn't
    a letter/digit/whitespace/allowed punctuation mark, isolate each
    punctuation mark as its own "word" (apostrophes inside a
    contraction/possessive stay attached on both sides; a standalone
    quote mark is still isolated), then split on whitespace. Returns a
    list of words, in order, for ONE sentence (callers segment a whole
    corpus sentence-by-sentence via split_sentences() first).
    """
    text = _CS_KEEP_CHARS_RE.sub(" ", text)
    text = _CS_ISOLATE_PUNCT_RE.sub(r" \1 ", text)
    text = _CS_LONE_APOSTROPHE_RE.sub(" ' ", text)
    text = _WS_RE.sub(" ", text).strip()
    return [w for w in text.split(" ") if w]


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


def stream_word_freq(path, word_freq, chunk_size=TokenizerConfig.STREAM_CHUNK_SIZE):
    """
    Stream `path` in chunks, accumulating case-preserving word counts
    into the caller-provided `word_freq` Counter, so a caller with
    many files to combine into one shared vocabulary
    (train_corpus.py's Pass 1) can share one counter across all of
    them without ever holding a whole file's text in memory at once.
    Returns the number of sentences seen in this one file. Uses this
    module's own case-preserving segment() for word segmentation, not
    normalize()'s case-folding version.
    """
    n_sentences = 0
    buffer = ""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            buffer += chunk
            parts = _SENT_SPLIT_RE.split(buffer)
            buffer = parts.pop()
            sents = _finalize_sentences(parts)
            n_sentences += len(sents)
            for sent in sents:
                for w in segment(sent):
                    word_freq[w] += 1
    if buffer.strip():
        n_sentences += 1
        for w in segment(buffer):
            word_freq[w] += 1
    return n_sentences


class CharacterVocabulary:
    """
    STAGE 1. Learns every unique character seen across a corpus's
    words, each assigned its own id right after the reserved
    special-token ids. Deterministic assignment order (sorted by code
    point) so the same corpus always produces the same ids, run to
    run. This alone is a complete, if verbose, tokenizer -- stage 2
    (below) exists only to shortcut it for common words, never to
    replace what it can do (see module docstring).
    """

    def __init__(self):
        self.char_to_id = dict(SPECIAL_TOKENS)
        self.id_to_char = {v: k for k, v in SPECIAL_TOKENS.items()}

    def build(self, words, min_next_id=0):
        """
        Learn every character across `words` (an iterable of word
        strings) not already known. `min_next_id` is the lowest id
        this call is allowed to hand out -- pass the tokenizer's
        current GLOBAL next-free-id (across BOTH stages) on every call
        after the very first, or a new character learned here (e.g.
        during extend_vocab()) could be assigned an id stage 2 already
        gave to some word in the meantime -- this class has no
        reference to TokenVocabulary to check that itself, so the
        caller (CharWordTokenizer) has to supply it explicitly.
        """
        chars = set()
        for w in words:
            chars.update(w)
        next_id = max(self.max_id, min_next_id - 1) + 1
        for c in sorted(chars):
            if c not in self.char_to_id:
                self.char_to_id[c] = next_id
                self.id_to_char[next_id] = c
                next_id += 1

    def encode_char(self, ch):
        """One character -> its stage-1 id, or UNK if this exact character was never learned."""
        return self.char_to_id.get(ch, UNK)

    def decode_id(self, char_id):
        """One stage-1 id -> its character, or '' for a special/unknown id (nothing to render)."""
        tok = self.id_to_char.get(char_id, "")
        return tok if len(tok) == 1 else ""  # guard against a SPECIAL_TOKENS entry like "<UNK>" leaking through

    def is_char_id(self, token_id):
        """Whether `token_id` belongs to stage 1 (a single character) rather than stage 2 (a multi-char word)."""
        return token_id in self.id_to_char

    @property
    def max_id(self):
        return max(self.char_to_id.values())

    @property
    def vocab_size(self):
        return len(self.char_to_id)

    def to_dict(self):
        return {"char_to_id": self.char_to_id}

    @classmethod
    def from_dict(cls, d):
        cv = cls()
        cv.char_to_id = d["char_to_id"]
        cv.id_to_char = {v: k for k, v in cv.char_to_id.items()}
        return cv


class TokenVocabulary:
    """
    STAGE 2. Pre-defines an id for every MULTI-character word worth
    having one, built strictly ON TOP of an already-built
    CharacterVocabulary (stage 1) -- a word's stored "definition" is
    the RECIPE of stage-1 character ids that spells it, e.g. "cat" ->
    [id('c'), id('a'), id('t')]. Never stores a one-character word
    (see module docstring's "STAGE 2 NEVER STORES..." section) -- a
    single character is already stage 1's job alone.

    token_to_id/id_to_token are plain dicts, seeded with
    SPECIAL_TOKENS, so anything elsewhere in this codebase that
    reaches into a tokenizer's token_to_id/id_to_token directly
    (model.py, analyse.py, server.py, test.py all do) keeps working.
    id_to_token holds each word's RECONSTRUCTED string, derived from
    its recipe once at build time (not a second independent source of
    truth -- the recipe in id_to_chars is what a word actually IS;
    id_to_token is a cheap, derived cache so every other module gets
    its O(1) string lookup instead of re-joining a recipe on every
    access).
    """

    def __init__(self, char_vocab: CharacterVocabulary):
        self.char_vocab = char_vocab
        self.token_to_id = dict(SPECIAL_TOKENS)
        self.id_to_token = {v: k for k, v in SPECIAL_TOKENS.items()}
        self.id_to_chars = {}   # word_id -> [stage-1 char id, ...] -- the word's RECIPE (never a 1-char word)

    def build(self, word_freq):
        """
        Learn every word of 2+ characters in `word_freq` (a
        {word: count} mapping) not already known, assigning ids in
        descending-frequency order (most common word gets the lowest
        new id) -- ties broken alphabetically, so the same corpus
        always produces the same ids. Frequency-ordered assignment is
        what lets CharWordTokenizer.train()'s vocab_size cap keep the
        MOST common words pre-defined and leave only the long tail to
        stage-1 fallback spelling at encode() time (see
        CharWordTokenizer._encode_word()) -- never <UNK>, just more
        ids for that one rare word. Single-character words in
        `word_freq` are silently skipped -- they never get a stage-2
        entry (see class docstring).

        CASE-INSENSITIVE: every word is folded to lowercase before
        counting or matching, so "The"/"the"/"THE" share ONE stage-2
        entry (and one combined frequency) instead of splintering the
        pre-defined-word budget across every capitalization of what
        is semantically the same word -- a sentence-initial "The"
        would otherwise cost its own separate id purely for showing
        up capitalized. This is scoped to stage 2 only: stage 1
        (characters) stays fully case-sensitive, and a word stage 2
        has no entry for still falls back to EXACT, case-preserving
        spelling through stage 1 (see CharWordTokenizer._encode_word())
        -- the casing loss below applies only to words common enough
        to earn a pre-defined stage-2 entry, never to the long tail.

        Ids continue the ONE shared counter stage 1 and stage 2 both
        draw from -- max(every id either stage has handed out so far) + 1
        -- never restarting from stage 2's own local count alone.
        """
        folded = {}
        for w, c in word_freq.items():
            key = w.lower()
            folded[key] = folded.get(key, 0) + c
        next_id = max(self.char_vocab.max_id, max(self.token_to_id.values())) + 1
        candidates = [w for w in folded if len(w) > 1 and w not in self.token_to_id]
        new_words = sorted(candidates, key=lambda w: (-folded[w], w))
        for w in new_words:
            self.token_to_id[w] = next_id
            chars = [self.char_vocab.encode_char(c) for c in w]
            self.id_to_chars[next_id] = chars
            self.id_to_token[next_id] = "".join(self.char_vocab.decode_id(c) for c in chars)
            next_id += 1

    def decode_id(self, word_id):
        """One stage-2 id -> its word string (see build()). '' for a special/unknown id."""
        return "" if word_id in SPECIAL_TOKENS.values() else self.id_to_token.get(word_id, "")

    @property
    def vocab_size(self):
        return len(self.token_to_id)

    def to_dict(self):
        return {"token_to_id": self.token_to_id,
                "id_to_chars": {str(k): v for k, v in self.id_to_chars.items()}}

    @classmethod
    def from_dict(cls, d, char_vocab):
        tv = cls(char_vocab)
        tv.token_to_id = d["token_to_id"]
        tv.id_to_token = {v: k for k, v in tv.token_to_id.items()}
        tv.id_to_chars = {int(k): v for k, v in d["id_to_chars"].items()}
        return tv


class CharWordTokenizer:
    """
    Top-level tokenizer combining both stages -- train/train_from_file/
    encode/encode_for_training/decode/save/load/vocab_size/
    vocab_size_actual. Falls back to stage-1 characters for anything
    stage 2 doesn't have a pre-defined entry for (see _encode_word()),
    so vocabulary loss for any word built from known characters is
    zero (see module docstring).
    """

    def __init__(self, vocab_size: int = TokenizerConfig.DEFAULT_VOCAB_SIZE):
        self.vocab_size = vocab_size
        self.chars = CharacterVocabulary()
        self.words = TokenVocabulary(self.chars)

    def _next_free_id(self):
        """
        The lowest id NEITHER stage has claimed yet -- the one shared
        counter both CharacterVocabulary and TokenVocabulary must
        respect (see module docstring's "ONE SHARED ID SPACE"). Passed
        into chars.build() on every call so a character learned later
        (e.g. during extend_vocab()) can never collide with a word id
        stage 2 already handed out in the meantime.
        """
        return max(self.chars.max_id, max(self.words.token_to_id.values(), default=0)) + 1

    # ---------------------------------------------------------------- train
    def train(self, corpus: str):
        """
        Stage 1 first (every character across every word), then stage
        2 (every word of 2+ characters, most-frequent-first -- see
        TokenVocabulary.build()'s docstring), capped so stage 2 itself
        never exceeds self.vocab_size entries: a corpus with more
        distinct multi-character words than that budget only
        PRE-DEFINES the most frequent ones; the long tail isn't lost
        (see module docstring's "TRUE ZERO VOCABULARY LOSS" section)
        -- it's simply spelled out via stage 1 at encode() time
        instead of getting its own single id, costing more ids per
        occurrence but never losing the word. No merge loop anywhere
        in either stage -- see module docstring for what that costs
        in practice.
        """
        from collections import Counter

        word_freq = Counter()
        for sent in split_sentences(corpus):
            for w in segment(sent):
                word_freq[w] += 1
        self._train_from_word_freq(word_freq)

    def train_from_file(self, path: str, chunk_size: int = TokenizerConfig.STREAM_CHUNK_SIZE):
        from collections import Counter
        word_freq = Counter()
        stream_word_freq(path, word_freq, chunk_size=chunk_size)
        self._train_from_word_freq(word_freq)

    def _train_from_word_freq(self, word_freq):
        # Stage 1 must learn BOTH the exact-case characters (needed
        # for case-preserving fallback spelling of rare/unknown
        # words -- see _encode_word()) AND their lowercase-folded
        # forms (needed because stage 2 always builds a word's recipe
        # from its LOWERCASE spelling -- see TokenVocabulary.build()
        # -- even for a word the corpus only ever saw capitalized,
        # e.g. an acronym that always sits at a sentence start).
        # Without this, folding "NASA" to "nasa" for stage 2 would
        # try to spell it from a lowercase 'n' stage 1 never learned.
        self.chars.build(set(word_freq.keys()) | {w.lower() for w in word_freq},
                          min_next_id=self._next_free_id())
        # Fold to lowercase and MERGE frequencies before capping, so
        # the vocab_size budget is spent selecting among truly
        # distinct (case-insensitive) words -- not wasted on "The"
        # and "the" as if they were two different candidates, only
        # for TokenVocabulary.build() to fold them together anyway
        # (see its docstring for why stage 2 is case-insensitive at
        # all). word_freq itself is untouched; we just choose which
        # folded subset TokenVocabulary.build() pre-defines.
        multi_char = {}
        for w, c in word_freq.items():
            if len(w) > 1:
                key = w.lower()
                multi_char[key] = multi_char.get(key, 0) + c
        if len(multi_char) > self.vocab_size:
            keep = dict(sorted(multi_char.items(), key=lambda kv: (-kv[1], kv[0]))[:self.vocab_size])
        else:
            keep = multi_char
        self.words.build(keep)

    def extend_vocab_from_word_freq(self, word_freq, target_vocab_size: int):
        """
        Same as extend_vocab(), but takes an already-computed
        {word: count} mapping instead of raw corpus text -- for a
        caller that already has to segment the new corpus anyway
        (e.g. model.py's train_incremental(), which also needs the
        same segmented words to encode it right afterward), this
        skips paying for split_sentences()/segment() a second time on
        text already segmented once. extend_vocab(corpus, ...) is just
        this, fed a word_freq built from segment()ing `corpus`.
        """
        if target_vocab_size <= self.words.vocab_size - len(SPECIAL_TOKENS):
            return 0
        if not word_freq:
            return 0

        start_size = self.words.vocab_size
        self.vocab_size = target_vocab_size
        # Same dual-case reasoning as _train_from_word_freq(): stage 1
        # needs both the exact-case characters (fallback spelling) and
        # their lowercase forms (stage 2's recipes are always spelled
        # from the lowercase form -- see TokenVocabulary.build()).
        self.chars.build(set(word_freq.keys()) | {w.lower() for w in word_freq},
                          min_next_id=self._next_free_id())   # new characters only -- existing ids untouched

        # Fold to lowercase and MERGE frequencies before capping to
        # the remaining budget -- same reasoning as
        # _train_from_word_freq(): the budget should be spent on
        # truly distinct (case-insensitive) new words, not on
        # case-variant duplicates of the same one.
        remaining_budget = max(target_vocab_size - (start_size - len(SPECIAL_TOKENS)), 0)
        new_words = {}
        for w, c in word_freq.items():
            if len(w) > 1:
                key = w.lower()
                if key not in self.words.token_to_id:
                    new_words[key] = new_words.get(key, 0) + c
        if len(new_words) > remaining_budget:
            new_words = dict(sorted(new_words.items(), key=lambda kv: (-kv[1], kv[0]))[:remaining_budget])
        self.words.build(new_words)
        return self.words.vocab_size - start_size

    def extend_vocab(self, corpus: str, target_vocab_size: int):
        """
        Grow stage 2's pre-defined-word budget using a NEW corpus,
        without touching any existing token id: every character
        (stage 1) and word (stage 2) already assigned an id keeps that
        exact id, so every Edge/Bridge/Relationship triple built under
        the old vocabulary stays byte-for-byte valid. Only new
        characters and new multi-char words (most-frequent-first,
        capped to the enlarged budget) get appended on top.

        Returns the number of new stage-2 entries actually added (0 if
        target_vocab_size <= the current stage-2 budget already in
        use, or if the new corpus has no new multi-char words left to
        pre-define). A word this doesn't have room to pre-define is
        never lost either way -- see module docstring. Thin wrapper
        over extend_vocab_from_word_freq() -- see that method's
        docstring if you already have the corpus segmented.
        """
        from collections import Counter

        word_freq = Counter()
        for sent in split_sentences(corpus):
            for w in segment(sent):
                word_freq[w] += 1
        return self.extend_vocab_from_word_freq(word_freq, target_vocab_size)

    # ------------------------------------------------------------- encode
    def _encode_word(self, word):
        """
        One word -> a LIST of one or more token ids -- one id whenever
        possible, several only when stage 2 has no pre-defined entry
        for it. Never <UNK> for the word as a whole (stage 1 can
        always build it on demand):

          - length 1  -> <WORD_BOUND>, then stage 1's own id for that
                         one character, EXACT case (stage 2 never
                         stores these, so case-insensitivity doesn't
                         apply here at all).
          - pre-defined in stage 2 -> that single id, matched
                         CASE-INSENSITIVELY (word.lower() -- see
                         TokenVocabulary.build()), no boundary marker
                         needed (already atomic). Decoding this id
                         later always gives back stage 2's stored
                         lowercase form, not this occurrence's exact
                         casing -- see decode()'s docstring.
          - otherwise -> <WORD_BOUND>, then spelled out live, one id
                         per character, EXACT case, straight from
                         stage 1 ("join these characters to build that
                         word"). The ONLY way <UNK> appears here is a
                         character stage 1 itself has never seen at
                         all.

        <WORD_BOUND> is what makes two consecutive stage-1-spelled
        words (e.g. two rare words in a row, nothing between them)
        distinguishable at decode() time -- without it, "zephyr
        quixotic" and a single 14-character word spelled the same way
        look identical in a flat id stream (see module docstring).
        """
        if len(word) == 1:
            return [WORD_BOUND, self.chars.encode_char(word)]
        wid = self.words.token_to_id.get(word.lower())
        if wid is not None:
            return [wid]
        return [WORD_BOUND] + [self.chars.encode_char(c) for c in word]

    def encode_words(self, words):
        """
        Same as encode(), but takes an already-SEGMENTED list of words
        instead of raw text -- for a caller that already has to call
        segment() once anyway (e.g. model.py's train(), which needs
        each sentence's word list both to count vocabulary frequency
        AND to encode it), this skips paying for segment()'s regex
        substitutions a second time on the same text. encode(text) is
        just encode_words(segment(text)) below.
        """
        ids = [BOS]
        for w in words:
            ids.extend(self._encode_word(w))
        return ids

    def encode(self, text: str):
        return self.encode_words(segment(text))

    def encode_for_training_words(self, words):
        ids = self.encode_words(words)
        ids.append(EOS)
        return ids

    def encode_for_training(self, text: str):
        ids = self.encode(text)
        ids.append(EOS)
        return ids

    def decode(self, ids):
        """
        Reconstruct text from a token stream that may freely mix
        stage-2 word ids with stage-1 character ids. <WORD_BOUND>
        (inserted by _encode_word() before every single-character word
        and every fallback-spelled multi-character word) is what makes
        this unambiguous: it always starts a new word and never
        renders any text itself, so a run of stage-1 character ids
        between one <WORD_BOUND> and the next -- or a stage-2 word id,
        which is already atomic -- is exactly one word, never a splice
        of two. <UNK> is dropped like PAD/BOS/EOS (a genuinely unseen
        character has no text to give back), but never a boundary --
        the <WORD_BOUND> either side of it is what keeps a mid-word
        unseen character from fusing the rest of that word onto its
        neighbor; only the one unseen character itself is lost.

        A stage-2 word id always decodes to that word's ONE stored
        (lowercase) spelling -- "The", "the", and "THE" all encode to
        the same id (see TokenVocabulary.build()) and so all decode
        back to "the", regardless of which casing this particular
        occurrence actually used. A word stage 2 has no entry for
        (fallback-spelled through stage 1, one id per character) has
        no such loss -- its exact original casing round-trips exactly.
        """
        filtered = [i for i in ids if i not in (PAD, UNK, BOS, EOS)]
        words = []
        buf = []
        for i in filtered:
            if i == WORD_BOUND:
                if buf:
                    words.append("".join(buf)); buf = []
                continue
            if self.chars.is_char_id(i):
                buf.append(self.chars.decode_id(i))
            else:
                if buf:
                    words.append("".join(buf)); buf = []
                words.append(self.words.decode_id(i))
        if buf:
            words.append("".join(buf))

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
        """
        max(every id actually assigned, across BOTH stages) + 1 --
        graph.py sizes its CSR index arrays from this number and then
        indexes them with real token ids straight out of encode()'s
        output, so this MUST be an upper bound on every id that can
        appear in a sequence, not a count of stage-2's dict entries.
        Stage 1's character ids occupy real id-space too (see module
        docstring's "ONE SHARED ID SPACE") even though they're never
        counted in words.vocab_size, so leaving them out here would
        silently undersize every array indexed by a raw character id
        -- exactly the bug a naive `len(token_to_id)` had.
        """
        return max(self.chars.max_id, max(self.words.token_to_id.values(), default=0)) + 1

    # token_to_id/id_to_token are exposed as direct passthrough
    # properties (not copied) onto stage 2's own dicts, so every other
    # module that reaches into a tokenizer's token_to_id/id_to_token
    # directly (model.py, analyse.py, server.py, test.py all do) sees
    # this tokenizer's REAL, live vocabulary. Note this dict alone
    # does NOT cover every id encode() can produce -- a fallback-
    # spelled word's character ids live in self.chars.char_to_id, a
    # separate dict, precisely because those ids are stage 1's, not
    # stage 2's (see module docstring's "ONE SHARED ID SPACE").
    @property
    def token_to_id(self):
        return self.words.token_to_id

    @property
    def id_to_token(self):
        return self.words.id_to_token

    def save(self, path: str):
        data = {
            "vocab_size": self.vocab_size,
            "chars": self.chars.to_dict(),
            "words": self.words.to_dict(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(vocab_size=data["vocab_size"])
        tok.chars = CharacterVocabulary.from_dict(data["chars"])
        tok.words = TokenVocabulary.from_dict(data["words"], tok.chars)
        return tok
