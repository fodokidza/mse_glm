"""
model.py — MSEGraphLanguageModel orchestrator.

Two inference modes, one engine:
  strict  — InferenceEngine(E, B, R)           training data only (default)
  open    — InferenceEngine(E+EE, B+EB, R+ER)  training + experience

Experience Matrices are built and saved independently by build_experience.py.
They are loaded automatically on MSEGraphLanguageModel.load() if present.
"""

import json
import os

from tokenizer import BPETokenizer, split_sentences
from graph import EdgeMatrix, BridgeMatrix, RelationshipMatrix
from inference import InferenceEngine


class MSEGraphLanguageModel:

    def __init__(self, vocab_size=2000):
        self.tokenizer   = BPETokenizer(vocab_size=vocab_size)
        self.edges       = EdgeMatrix()
        self.bridges     = BridgeMatrix()
        self.rels        = RelationshipMatrix()
        self._strict     = None   # InferenceEngine, no experience
        self._open       = None   # InferenceEngine, with experience
        # experience matrices — None until built and loaded
        self.exp_edges   = None
        self.exp_bridges = None
        self.exp_rels    = None

    # ─── training ─────────────────────────────────────────────────────────

    def train(self, corpus):
        self.tokenizer.train(corpus)
        seqs = [self.tokenizer.encode_for_training(s)
                for s in split_sentences(corpus)]
        self._build_graphs(seqs)

    def train_from_file(self, path):
        self.tokenizer.train_from_file(path)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        seqs = [self.tokenizer.encode_for_training(s)
                for s in split_sentences(text)]
        self._build_graphs(seqs)

    def _build_graphs(self, seqs):
        vsz = self.tokenizer.vocab_size_actual
        self.edges.build(seqs, vsz)
        self.bridges.build(seqs, vsz)
        self.rels.build(seqs, self.bridges)
        self._strict = InferenceEngine(self.edges, self.bridges, self.rels)
        self._open   = None   # invalidated until experience is built/loaded

    # ─── experience ───────────────────────────────────────────────────────

    def build_experience(self, folder=None):
        """
        Derive Experience Matrices from the current trained state.
        Saves to `folder` when provided. Returns a summary dict.
        Call build_experience.py instead for the standalone CLI version.
        """
        from experience import ExperienceBuilder
        builder = ExperienceBuilder()
        self.exp_edges, self.exp_bridges, self.exp_rels = builder.build(self)
        self._open = InferenceEngine(
            self.edges, self.bridges, self.rels,
            self.exp_edges, self.exp_bridges, self.exp_rels,
        )
        if folder and os.path.isdir(folder):
            self._save_experience(folder)
        return builder.summary(self.exp_edges, self.exp_bridges, self.exp_rels)

    def load_experience(self, folder):
        from experience import (ExperienceEdgeMatrix,
                                 ExperienceBridgeMatrix,
                                 ExperienceRelationshipMatrix)
        ee = os.path.join(folder, "experience_edges.json")
        eb = os.path.join(folder, "experience_bridges.json")
        er = os.path.join(folder, "experience_relationships.json")
        if not (os.path.exists(ee) and os.path.exists(eb) and os.path.exists(er)):
            return False
        with open(ee) as f: self.exp_edges   = ExperienceEdgeMatrix.from_dict(json.load(f))
        with open(eb) as f: self.exp_bridges = ExperienceBridgeMatrix.from_dict(json.load(f))
        with open(er) as f: self.exp_rels    = ExperienceRelationshipMatrix.from_dict(json.load(f))
        self._open = InferenceEngine(
            self.edges, self.bridges, self.rels,
            self.exp_edges, self.exp_bridges, self.exp_rels,
        )
        return True

    def _save_experience(self, folder):
        with open(os.path.join(folder, "experience_edges.json"),         "w") as f: json.dump(self.exp_edges.to_dict(),   f)
        with open(os.path.join(folder, "experience_bridges.json"),       "w") as f: json.dump(self.exp_bridges.to_dict(), f)
        with open(os.path.join(folder, "experience_relationships.json"), "w") as f: json.dump(self.exp_rels.to_dict(),    f)

    def has_experience(self):
        return self.exp_edges is not None

    # ─── generate ─────────────────────────────────────────────────────────

    def generate(self, prompt, max_tokens=40, mode="strict"):
        engine = self._engine(mode)
        ids, trace = engine.generate(
            self.tokenizer.encode(prompt), max_tokens=max_tokens)
        return self.tokenizer.decode(ids), ids, trace

    def explain_step(self, previous_text, current_text, mode="strict"):
        engine   = self._engine(mode)
        prev_ids = self.tokenizer.encode(previous_text) if previous_text else []
        curr_ids = self.tokenizer.encode(current_text)
        prev     = prev_ids[-1] if prev_ids else None
        curr     = curr_ids[-1]
        token, trace = engine.step(prev, curr)
        return self.tokenizer.decode([token]), trace

    def infer_shared_role(self, words, mode="strict"):
        ids = []
        for w in words:
            enc = [t for t in self.tokenizer.encode(w) if t != 2]
            if enc:
                ids.append(enc[-1])
        engine  = self._engine(mode)
        results = engine.infer_shared_role(ids)
        decoded = []
        for token, axis, ev in results:
            label = (self.tokenizer.id_to_token.get(token, str(token))
                     if token in (0, 1, 2, 3)
                     else self.tokenizer.decode([token]))
            decoded.append((label, axis, ev))
        return decoded

    def _dec_tok(self, t):
        tok = self.tokenizer
        return tok.id_to_token.get(t, t) if t in (0, 1, 2, 3) else tok.decode([t])

    def _decode_interpretation(self, result):
        """Decode a raw interpret.py result (token ids) to surface strings."""
        if result is None:
            return None
        dec = self._dec_tok
        return {
            "cluster_id": result["cluster_id"],
            "axis": result["axis"],
            "members": [dec(m) for m in result["members"]],
            "candidates": [
                {**c,
                 "interpreter_token": dec(c["interpreter_token"]),
                 "via_bridge_token": dec(c["via_bridge_token"]),
                 "members_covered": [dec(m) for m in c["members_covered"]],
                 "shared_role_overlap": c["shared_role_overlap"]}
                for c in result["candidates"]
            ],
        }

    def _decode_ci_row(self, row):
        dec = self._dec_tok
        return {
            **row,
            "members": [dec(m) for m in row["members"]],
            "interpreter_token": dec(row["interpreter_token"]),
            "via_bridge_token": dec(row["via_bridge_token"]),
            "members_covered": [dec(m) for m in row["members_covered"]],
        }

    def interpret_cluster(self, cluster_id, top_n=5, mode="strict"):
        """
        Propose a human-readable interpreter token for one cluster_id,
        with evidence gathered from Edge, Bridge, and Relationship
        matrices. See interpret.py for the method and its honesty
        caveats (coverage is a plain fraction, not a calibrated
        confidence score; evidence_mask entries are correlated signals,
        not independent votes to be summed into a percentage).
        Returns None if cluster_id is unknown / degenerate.
        """
        from interpret import interpret_cluster as _interpret_cluster
        result = _interpret_cluster(self, cluster_id, top_n=top_n, mode=mode)
        return self._decode_interpretation(result)

    def interpret_all_clusters(self, min_coverage=0.5, max_per_cluster=3, mode="strict"):
        """
        Every candidate that clears min_coverage for each cluster (up to
        max_per_cluster, best first) -- not just one label per cluster.
        See interpret.py.
        """
        from interpret import interpret_all_clusters as _interpret_all
        raw = _interpret_all(self, min_coverage=min_coverage,
                              max_per_cluster=max_per_cluster, mode=mode)
        return [self._decode_interpretation(r) for r in raw]

    def build_interpreter_matrix(self, min_coverage=0.5, min_signals=2,
                                  max_per_cluster=None, mode="strict"):
        """
        The filtered Cluster Interpreter Matrix: one row per (cluster_id,
        interpreter) pair whose candidate clears both min_coverage and
        min_signals -- every qualifying interpreter is kept, so a cluster
        can carry more than one label at once (see
        interpret.build_interpreter_matrix for exact semantics).
        Returned rows are decoded to surface strings, ready to print or
        json-dump for persistence.
        """
        from interpret import build_interpreter_matrix as _build_ci
        raw = _build_ci(self, min_coverage=min_coverage, min_signals=min_signals,
                         max_per_cluster=max_per_cluster, mode=mode)
        return [self._decode_ci_row(r) for r in raw]

    def _decode_zero_group(self, row):
        dec = self._dec_tok
        return {
            **row,
            "members": [dec(m) for m in row["members"]],
            "interpreter_token": dec(row["interpreter_token"]),
            "via_bridge_token": dec(row["via_bridge_token"]),
        }

    def discover_zero_cluster_groups(self, min_group_size=2, mode="strict"):
        """
        Mine cluster_id==0 for source-axis groups the standard dual-axis
        rule never assigns a cluster_id to at all (fix bridge+target,
        source varies -- a rule BridgeMatrix.build doesn't implement).
        See interpret.discover_zero_cluster_groups for exact semantics
        and its caveats (larger candidate space than the regular CI
        matrix -- eyeball results, don't trust min_group_size alone).
        """
        from interpret import discover_zero_cluster_groups as _discover
        raw = _discover(self, min_group_size=min_group_size, mode=mode)
        return [self._decode_zero_group(r) for r in raw]

    def token_similarity(self, word_a, word_b, mode="strict"):
        tok = self.tokenizer
        def resolve(w):
            enc = [t for t in tok.encode(w) if t != 2]
            return enc[-1] if enc else None
        ta, tb = resolve(word_a), resolve(word_b)
        if ta is None or tb is None:
            return {"word_a": word_a, "word_b": word_b,
                    "similarity": 0, "shared_clusters": []}
        sa = set(self.bridges.t_index.get(ta, []))
        sb = set(self.bridges.t_index.get(tb, []))
        if mode == "open" and self.exp_bridges:
            sa.update(self.exp_bridges.t_index.get(ta, []))
            sb.update(self.exp_bridges.t_index.get(tb, []))
        shared = sorted(sa & sb)
        return {"word_a": word_a, "word_b": word_b,
                "similarity": len(shared), "shared_clusters": shared}

    def _engine(self, mode):
        if mode == "open":
            if self._open is None:
                raise RuntimeError(
                    "Open mode requires experience matrices. "
                    "Run: python3 build_experience.py --model <folder>")
            return self._open
        if self._strict is None:
            raise RuntimeError("Model has not been trained or loaded.")
        return self._strict

    # ─── stats ────────────────────────────────────────────────────────────

    def stats(self):
        s = {
            "vocab_size":        self.tokenizer.vocab_size_actual,
            "edges":             len(self.edges.src),
            "bridges":           len(self.bridges.source),
            "clustered_bridges": sum(1 for c in self.bridges.cluster_id if c != 0),
            "clusters":          len(set(c for c in self.bridges.cluster_id if c != 0)),
            "relationships":     self.rels._n_rels,
            "relationship_rows": len(self.rels.r_triple),
        }
        if self.exp_edges is not None:
            s.update({
                "exp_edges":     len(self.exp_edges.src),
                "exp_bridges":   len(self.exp_bridges.source),
                "exp_clustered": sum(1 for c in self.exp_bridges.cluster_id if c != 0),
                "exp_clusters":  len(set(c for c in self.exp_bridges.cluster_id if c != 0)),
                "exp_rel_rows":  len(self.exp_rels.r_triple),
            })
        return s

    # ─── save / load ──────────────────────────────────────────────────────

    def save(self, folder):
        os.makedirs(folder, exist_ok=True)
        self.tokenizer.save(os.path.join(folder, "vocabulary.json"))
        with open(os.path.join(folder, "edges.json"),         "w") as f: json.dump(self.edges.to_dict(),   f)
        with open(os.path.join(folder, "bridges.json"),       "w") as f: json.dump(self.bridges.to_dict(), f)
        with open(os.path.join(folder, "relationships.json"), "w") as f: json.dump(self.rels.to_dict(),    f)
        with open(os.path.join(folder, "meta.json"),          "w") as f:
            json.dump({"vocab_size_config": self.tokenizer.vocab_size,
                       "stats": self.stats()}, f, indent=2)
        if self.exp_edges is not None:
            self._save_experience(folder)

    @classmethod
    def load(cls, folder):
        m = cls()
        m.tokenizer = BPETokenizer.load(os.path.join(folder, "vocabulary.json"))
        with open(os.path.join(folder, "edges.json"))         as f: m.edges   = EdgeMatrix.from_dict(json.load(f))
        with open(os.path.join(folder, "bridges.json"))       as f: m.bridges = BridgeMatrix.from_dict(json.load(f))
        with open(os.path.join(folder, "relationships.json")) as f: m.rels    = RelationshipMatrix.from_dict(json.load(f))
        m._strict = InferenceEngine(m.edges, m.bridges, m.rels)
        m.load_experience(folder)
        return m
