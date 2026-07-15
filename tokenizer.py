"""
tokenizer.py — From-scratch Byte Pair Encoding tokenizer for MSE-GLM.

Special tokens:
    <PAD> = 0   reserved
    <UNK> = 1   unknown character fallback
    <BOS> = 2   prepended to every encoded sequence
    <EOS> = 3   appended only during training (encode_for_training)
"""

import json
import re
from collections import Counter

PAD, UNK, BOS, EOS = 0, 1, 2, 3
SPECIAL_TOKENS = {"<PAD>": PAD, "<UNK>": UNK, "<BOS>": BOS, "<EOS>": EOS}

_NORM_RE = re.compile(r"[^a-z0-9\s]")
_WS_RE = re.compile(r"\s+")
_SENT_SPLIT_RE = re.compile(r"[.!?\n]+")


def normalize(text: str) -> str:
    text = text.lower()
    text = _NORM_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def split_sentences(text: str):
    parts = _SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


class BPETokenizer:
    def __init__(self, vocab_size: int = 2000):
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

    def train_from_file(self, path: str, chunk_size: int = 1 << 20):
        word_freq = Counter()
        buffer = ""
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                buffer += chunk
                sentences = _SENT_SPLIT_RE.split(buffer)
                buffer = sentences.pop()  # keep last partial sentence for next chunk
                for sent in sentences:
                    sent = sent.strip()
                    if not sent:
                        continue
                    for word in normalize(sent).split(" "):
                        if word:
                            word_freq[word] += 1
        if buffer.strip():
            for word in normalize(buffer).split(" "):
                if word:
                    word_freq[word] += 1
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

        while len(self.token_to_id) < self.vocab_size:
            pair_counts = Counter()
            for w, freq in word_freq.items():
                symbols = word_symbols[w]
                for i in range(len(symbols) - 1):
                    pair_counts[(symbols[i], symbols[i + 1])] += freq
            if not pair_counts:
                break
            (a, b), _ = pair_counts.most_common(1)[0]
            merged = a + b
            if merged not in self.token_to_id:
                self.token_to_id[merged] = next_id
                self.id_to_token[next_id] = merged
                next_id += 1
            self.merges.append((a, b))

            for w in word_symbols:
                symbols = word_symbols[w]
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
        return " ".join(words)

    # ------------------------------------------------------------ persist
    @property
    def vocab_size_actual(self):
        return len(self.token_to_id)

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
