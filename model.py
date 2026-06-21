"""
model.py — MSEGraphLanguageModel orchestrator. Wires the tokenizer, the
Edge/Bridge/Relationship matrices, and the inference engine together, and
handles persistence to a self-contained model folder.
"""

import json
import os

from tokenizer import BPETokenizer, split_sentences
from graph import EdgeMatrix, BridgeMatrix, RelationshipMatrix
from inference import InferenceEngine


class MSEGraphLanguageModel:
    def __init__(self, vocab_size: int = 2000):
        self.tokenizer = BPETokenizer(vocab_size=vocab_size)
        self.edges = EdgeMatrix()
        self.bridges = BridgeMatrix()
        self.rels = RelationshipMatrix()
        self.engine = None
        self._sentences_cache = None

    # ----------------------------------------------------------------- train
    def train(self, corpus: str):
        self.tokenizer.train(corpus)
        sentences = split_sentences(corpus)
        sequences = [self.tokenizer.encode_for_training(s) for s in sentences]
        self._build_graphs(sequences)

    def train_from_file(self, path: str):
        self.tokenizer.train_from_file(path)
        sequences = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        for sent in split_sentences(text):
            sequences.append(self.tokenizer.encode_for_training(sent))
        self._build_graphs(sequences)

    def _build_graphs(self, sequences):
        vocab_size = self.tokenizer.vocab_size_actual
        self.edges.build(sequences, vocab_size)
        self.bridges.build(sequences, vocab_size)
        self.rels.build(sequences, self.bridges)
        self.engine = InferenceEngine(self.edges, self.bridges, self.rels)
        self._sentences_cache = sequences

    # -------------------------------------------------------------- generate
    def generate(self, prompt: str, max_tokens: int = 40):
        if self.engine is None:
            raise RuntimeError("Model has not been trained or loaded.")
        prompt_ids = self.tokenizer.encode(prompt)
        ids, trace = self.engine.generate(prompt_ids, max_tokens=max_tokens)
        return self.tokenizer.decode(ids), ids, trace

    def explain_step(self, previous_text: str, current_text: str):
        prev_ids = self.tokenizer.encode(previous_text) if previous_text else []
        curr_ids = self.tokenizer.encode(current_text)
        previous = prev_ids[-1] if prev_ids else None
        current = curr_ids[-1]
        token, trace = self.engine.step(previous, current)
        return self.tokenizer.decode([token]), trace

    def infer_shared_role(self, words):
        ids = []
        for w in words:
            enc = self.tokenizer.encode(w)
            # encode() prepends BOS; take the real token(s) after it
            real = [t for t in enc if t != 2]
            if real:
                ids.append(real[-1])
        results = self.engine.infer_shared_role(ids)
        decoded = []
        for token, axis, evidence in results:
            if token in (0, 1, 2, 3):  # PAD/UNK/BOS/EOS — show label, not blank
                label = self.tokenizer.id_to_token.get(token, str(token))
                label = f"<{label}>" if not label.startswith("<") else label
            else:
                label = self.tokenizer.decode([token])
            decoded.append((label, axis, evidence))
        return decoded

    # ----------------------------------------------------------------- stats
    def stats(self):
        return {
            "vocab_size": self.tokenizer.vocab_size_actual,
            "edges": len(self.edges.src),
            "bridges": len(self.bridges.source),
            "clustered_bridges": sum(1 for c in self.bridges.cluster_id if c != 0),
            "clusters": len(set(c for c in self.bridges.cluster_id if c != 0)),
            "relationships": self.rels._n_rels,
            "relationship_rows": len(self.rels.r_triple),
        }

    # -------------------------------------------------------------- persist
    def save(self, folder: str):
        os.makedirs(folder, exist_ok=True)
        self.tokenizer.save(os.path.join(folder, "vocabulary.json"))
        with open(os.path.join(folder, "edges.json"), "w") as f:
            json.dump(self.edges.to_dict(), f)
        with open(os.path.join(folder, "bridges.json"), "w") as f:
            json.dump(self.bridges.to_dict(), f)
        with open(os.path.join(folder, "relationships.json"), "w") as f:
            json.dump(self.rels.to_dict(), f)
        with open(os.path.join(folder, "meta.json"), "w") as f:
            json.dump({"vocab_size_config": self.tokenizer.vocab_size,
                       "stats": self.stats()}, f, indent=2)

    @classmethod
    def load(cls, folder: str):
        model = cls()
        model.tokenizer = BPETokenizer.load(os.path.join(folder, "vocabulary.json"))
        with open(os.path.join(folder, "edges.json")) as f:
            model.edges = EdgeMatrix.from_dict(json.load(f))
        with open(os.path.join(folder, "bridges.json")) as f:
            model.bridges = BridgeMatrix.from_dict(json.load(f))
        with open(os.path.join(folder, "relationships.json")) as f:
            model.rels = RelationshipMatrix.from_dict(json.load(f))
        model.engine = InferenceEngine(model.edges, model.bridges, model.rels)
        return model
