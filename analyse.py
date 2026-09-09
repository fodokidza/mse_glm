"""
analyse.py — Analysis layer for MSE-GLM.

Two library classes:
    CorpusAnalyser  — raw-text statistics, no trained model required.
    Analyser        — graph-level statistics on a trained MSEGraphLanguageModel:
                       topology, dual-axis cluster reports, Relationship Matrix
                       reports, per-token reports, token similarity, and full
                       step-by-step generation traces.

Plus a CLI (`python3 analyse.py ...`) so the model can be interrogated freely
from the command line without writing one-off scripts — stats, topology,
clusters, a specific cluster, a specific token, similarity between two
tokens, a generation trace, or a single combined report, optionally
exported to JSON.
"""

import argparse
import json
import sys
from collections import Counter

from model import MSEGraphLanguageModel
from tokenizer import normalize, split_sentences
from config import InterpretConfig, CTMConfig, ScoresDisplayConfig


# =============================================================================
# Library
# =============================================================================

class CorpusAnalyser:
    """Statistics over raw corpus text — no trained model required."""

    def __init__(self, corpus: str):
        self.corpus = corpus

    def stats(self, top_n: int = 10) -> dict:
        sentences = split_sentences(self.corpus)
        words = []
        for s in sentences:
            words.extend(normalize(s).split(" "))
        words = [w for w in words if w]
        return {
            "sentences": len(sentences),
            "words": len(words),
            "unique_words": len(set(words)),
            "avg_sentence_len": round((len(words) / len(sentences)), 2) if sentences else 0,
            "top_words": Counter(words).most_common(top_n),
        }


class Analyser:
    """Graph-level statistics on a trained MSEGraphLanguageModel."""

    def __init__(self, model: MSEGraphLanguageModel):
        self.model = model

    # ------------------------------------------------------------- topology
    def topology(self, top_n: int = 10) -> dict:
        b = self.model.bridges
        tok = self.model.tokenizer
        out_degree = Counter(b.source)
        hubs = out_degree.most_common(top_n)
        vocab_ids = range(tok.vocab_size_actual)
        dead_ends = [t for t in vocab_ids if t not in out_degree]
        return {
            "hub_tokens": [(tok.id_to_token.get(t, t), c) for t, c in hubs],
            "dead_end_count": len(dead_ends),
            "dead_end_tokens": [tok.id_to_token.get(t, t) for t in dead_ends[:top_n]],
        }

    # ---------------------------------------------------------- raw matrices
    # Unlike topology()/cluster_report()/relationship_report() above (which
    # aggregate), these three just page through a matrix's actual rows,
    # decoded to tokens -- for "what is literally stored in here" rather
    # than "what does it imply". All three matrices can be large (a
    # relationship matrix has one row per triple occurrence per training
    # sentence, so it's the biggest of the three by far -- see analyse.py's
    # own `stats` command for the real row counts before dumping one
    # unfiltered), so all three are paged (limit/offset) rather than
    # returned whole; pass --limit 0 (CLI) / limit=0 (API) for "everything",
    # e.g. right before piping to --json.

    def edge_matrix(self, source: str = None, target: str = None,
                     sort: str = "count", limit: int = 50, offset: int = 0) -> dict:
        """
        Raw EdgeMatrix rows: every deduplicated (source, dst) bigram with
        its literal training count (see EdgeMatrix.frequency()). Optionally
        filtered to a single source and/or target word. `sort`: "count"
        (default, descending) or "source" (ascending source token id, the
        matrix's native CSR order).
        """
        tok = self.model.tokenizer
        e = self.model.edges
        rows = list(zip(e.src, e.dst, e.count))

        if source is not None:
            src_id = self._resolve_token(source)
            if src_id is None:
                return {"total": 0, "offset": offset, "limit": limit, "rows": []}
            rows = [r for r in rows if r[0] == src_id]
        if target is not None:
            dst_id = self._resolve_token(target)
            if dst_id is None:
                return {"total": 0, "offset": offset, "limit": limit, "rows": []}
            rows = [r for r in rows if r[1] == dst_id]

        if sort == "count":
            rows.sort(key=lambda r: -r[2])
        else:
            rows.sort(key=lambda r: (r[0], r[1]))

        total = len(rows)
        page = rows[offset:offset + limit] if limit else rows[offset:]
        decoded = [(tok.id_to_token.get(s, s), tok.id_to_token.get(d, d), c) for s, d, c in page]
        return {"total": total, "offset": offset, "limit": limit, "rows": decoded}

    def bridge_matrix(self, source: str = None, bridge: str = None, target: str = None,
                       clustered_only: bool = False, limit: int = 50, offset: int = 0) -> dict:
        """
        Raw BridgeMatrix rows: every deduplicated (source, bridge, target)
        triple with its cluster_id (0 = unclustered -- see BridgeMatrix's
        dual-axis docstring). Optionally filtered to a single source,
        bridge, and/or target word, and/or restricted to clustered rows
        only (cluster_id != 0). Native CSR order (by source).
        """
        tok = self.model.tokenizer
        b = self.model.bridges
        rows = list(zip(b.source, b.bridge, b.target, b.cluster_id))

        for word, slot in ((source, 0), (bridge, 1), (target, 2)):
            if word is not None:
                tid = self._resolve_token(word)
                if tid is None:
                    return {"total": 0, "offset": offset, "limit": limit, "rows": []}
                rows = [r for r in rows if r[slot] == tid]
        if clustered_only:
            rows = [r for r in rows if r[3] != 0]

        total = len(rows)
        page = rows[offset:offset + limit] if limit else rows[offset:]
        decoded = [(tok.id_to_token.get(s, s), tok.id_to_token.get(br, br),
                    tok.id_to_token.get(t, t), c) for s, br, t, c in page]
        return {"total": total, "offset": offset, "limit": limit, "rows": decoded}

    def relationship_matrix(self, relationship_id: int = None, triple_id: int = None,
                             limit: int = 50, offset: int = 0) -> dict:
        """
        Raw RelationshipMatrix rows: every (triple_id, relationship_id)
        pair, decoded to the triple's actual (source, bridge, target)
        content. One row per triple OCCURRENCE per training sentence --
        NOT deduplicated the way EdgeMatrix/BridgeMatrix rows are (see
        RelationshipMatrix's docstring), so this is normally the largest
        of the three matrices and the one most worth filtering rather
        than dumping whole. Optionally filtered to one relationship_id
        (one training sentence's rows) or one triple_id (every sentence
        that triple occurs in) -- not both at once.
        """
        if relationship_id is not None and triple_id is not None:
            raise ValueError("relationship_matrix: pass relationship_id OR triple_id, not both")
        r = self.model.rels
        b = self.model.bridges
        tok = self.model.tokenizer
        rows = list(zip(r.r_triple, r.r_rel))
        if relationship_id is not None:
            rows = [row for row in rows if row[1] == relationship_id]
        elif triple_id is not None:
            rows = [row for row in rows if row[0] == triple_id]

        total = len(rows)
        page = rows[offset:offset + limit] if limit else rows[offset:]
        decoded = []
        for tid, rel_id in page:
            s, br, t = b.source[tid], b.bridge[tid], b.target[tid]
            decoded.append((rel_id, tid, tok.id_to_token.get(s, s),
                             tok.id_to_token.get(br, br), tok.id_to_token.get(t, t)))
        return {"total": total, "offset": offset, "limit": limit, "rows": decoded}

    def _resolve_token(self, word):
        """Encode a surface word to its last real token id (drops <BOS>), or
        None if it doesn't exist in the vocabulary. Shared by every raw-
        matrix filter above -- same convention as per_token_report()."""
        if word is None:
            return None
        tok = self.model.tokenizer
        enc = [t for t in tok.encode(word) if t != 2]
        return enc[-1] if enc else None

    # -------------------------------------------------------------- clusters
    def cluster_report(self, top_n: int = 10, axis: str = None) -> list:
        """
        List dual-axis clusters (cluster_id != 0). `axis` optionally filters
        to 'bridge' or 'target' clusters only.
        """
        b = self.model.bridges
        tok = self.model.tokenizer
        groups = {}
        for s, t, br, c in zip(b.source, b.target, b.bridge, b.cluster_id):
            if c == 0:
                continue
            groups.setdefault(c, []).append((s, t, br))

        report = []
        for cid, members in groups.items():
            cluster_axis, _ = b.cluster_axis(cid)
            if axis and cluster_axis != axis:
                continue
            s0, t0, _ = members[0]
            if cluster_axis == "bridge":
                varying = sorted(set(tok.id_to_token.get(br, br) for _, _, br in members))
                fixed = f"{tok.id_to_token.get(s0, s0)} -> ___ -> {tok.id_to_token.get(t0, t0)}"
            elif cluster_axis == "target":
                varying = sorted(set(tok.id_to_token.get(t, t) for _, t, _ in members))
                fixed = f"{tok.id_to_token.get(s0, s0)} -> {tok.id_to_token.get(members[0][2], members[0][2])} -> ___"
            else:
                varying, fixed = [], "?"
            report.append({
                "cluster_id": cid, "axis": cluster_axis, "slot": fixed,
                "members": varying, "size": len(varying),
            })
        report.sort(key=lambda r: r["size"], reverse=True)
        return report[:top_n]

    def cluster_detail(self, cluster_id: int) -> dict:
        """Full detail for a single cluster_id, including raw triples."""
        b = self.model.bridges
        tok = self.model.tokenizer
        axis, members = b.cluster_axis(cluster_id)
        decoded = [
            (tok.id_to_token.get(s, s), tok.id_to_token.get(t, t), tok.id_to_token.get(br, br))
            for s, t, br in members
        ]
        return {"cluster_id": cluster_id, "axis": axis, "triples": decoded}

    # --------------------------------------------------------- relationships
    def relationship_report(self) -> dict:
        r = self.model.rels
        shared = [tid for tid in set(r.r_triple) if len(r.relationships_for_triple(tid)) > 1]
        return {
            "total_relationships": r._n_rels,
            "total_occurrences": sum(r.rel_count),
            "total_rows": len(r.r_triple),
            "unique_triples_referenced": len(set(r.r_triple)),
            "shared_triple_count": len(shared),
        }

    def relationship_detail(self, relationship_id: int) -> dict:
        """Every triple belonging to a single training sequence (sentence),
        plus how many literal training occurrences shared that exact
        sentence content (see graph.py's RelationshipMatrix docstring)."""
        b = self.model.bridges
        tok = self.model.tokenizer
        triple_ids = self.model.rels.triples_for_relationship(relationship_id)
        triples = []
        for tid in triple_ids:
            s, t, br = b.source[tid], b.target[tid], b.bridge[tid]
            triples.append((tok.id_to_token.get(s, s), tok.id_to_token.get(t, t),
                             tok.id_to_token.get(br, br)))
        return {"relationship_id": relationship_id, "triples": triples,
                "occurrences": self.model.rels.count(relationship_id)}

    # ---------------------------------------------------------- per-token
    def per_token_report(self, word: str) -> dict:
        tok = self.model.tokenizer
        enc = [t for t in tok.encode(word) if t != 2]
        if not enc:
            return None
        token = enc[-1]
        out_edges = self.model.edges.successors(token)
        triples = self.model.bridges.triples_from_source(token)
        clusters = self.model.bridges.t_index.get(token, [])
        return {
            "token": tok.id_to_token.get(token, token),
            "edge_successors": [tok.id_to_token.get(t, t) for t in out_edges],
            "bridge_triples_as_source": len(triples),
            "cluster_memberships": clusters,
        }

    def token_similarity(self, word_a: str, word_b: str) -> dict:
        """
        Relatedness between two tokens = |T_index[a] ∩ T_index[b]|, per SDD
        v2.1 §10. Returns the shared clusters and a plain similarity count.
        """
        tok = self.model.tokenizer
        t_index = self.model.bridges.t_index

        def resolve(word):
            enc = [t for t in tok.encode(word) if t != 2]
            return enc[-1] if enc else None

        ta, tb = resolve(word_a), resolve(word_b)
        if ta is None or tb is None:
            return {"word_a": word_a, "word_b": word_b, "similarity": 0, "shared_clusters": []}
        sa, sb = set(t_index.get(ta, [])), set(t_index.get(tb, []))
        shared = sorted(sa & sb)
        return {
            "word_a": word_a, "word_b": word_b,
            "similarity": len(shared), "shared_clusters": shared,
        }

    # ------------------------------------------------------- interpretation
    def cluster_interpretation(self, cluster_id: int, top_n: int = InterpretConfig.TOP_N, mode: str = "strict") -> dict:
        """
        Propose a human-readable label for a cluster (e.g. {cat, dog, pig}
        -> "animal"), derived purely from structure already in the trained
        matrices. Returns None if the cluster_id doesn't exist or nothing
        was found. `coverage` is a plain fraction, not a confidence score —
        see interpret.py's module docstring before presenting this to a
        user as a certainty figure.
        """
        return self.model.interpret_cluster(cluster_id, top_n=top_n, mode=mode)

    def interpretation_report(self, min_coverage: float = InterpretConfig.MIN_COVERAGE, max_per_cluster: int = InterpretConfig.MAX_PER_CLUSTER,
                               mode: str = "strict") -> list:
        """Every candidate clearing min_coverage per cluster (not just one)."""
        return self.model.interpret_all_clusters(
            min_coverage=min_coverage, max_per_cluster=max_per_cluster, mode=mode)

    def interpreter_matrix(self, min_coverage: float = InterpretConfig.MIN_COVERAGE, min_signals: int = InterpretConfig.MIN_SIGNALS,
                            max_per_cluster: int = None, mode: str = "strict") -> list:
        """
        The filtered Cluster Interpreter Matrix -- coverage AND a minimum
        number of corroborating evidence_mask entries required. A cluster
        can carry multiple qualifying labels at once (e.g. both "animal"
        and "pet"); this is what should actually get persisted as "the"
        interpreter matrix. interpretation_report()/interpret_cluster()
        are the wider, unfiltered views used to sanity-check candidates
        by eye.
        """
        return self.model.build_interpreter_matrix(
            min_coverage=min_coverage, min_signals=min_signals,
            max_per_cluster=max_per_cluster, mode=mode)

    def zero_cluster_groups(self, min_group_size: int = InterpretConfig.MIN_GROUP_SIZE, mode: str = "strict") -> list:
        """
        Mine cluster_id==0 for the "third axis" the standard dual-axis
        rule never implements (fix bridge+target, source varies). Finds
        groups the regular cluster_report()/interpreter_matrix() never
        see at all -- see model.discover_zero_cluster_groups for caveats
        (bigger candidate space, more prone to coincidental groupings).
        """
        return self.model.discover_zero_cluster_groups(
            min_group_size=min_group_size, mode=mode)

    def context_trigger_matrix(self, min_support: int = CTMConfig.MIN_SUPPORT, mode: str = "strict") -> list:
        """
        Flat Context Trigger Matrix table: which surrounding tokens
        (from whole-sentence co-occurrence) support which cluster
        member. This is analysis/display only -- it does not build or
        cache model.ctm; call model.build_context_triggers() for that
        (needed before generate(..., use_context_triggers=True)).
        """
        from ctm import build_context_trigger_matrix
        rows = build_context_trigger_matrix(self.model, min_support=min_support, mode=mode)
        dec = self.model._dec_tok
        return [
            {**r, "trigger_token": dec(r["trigger_token"]), "member_token": dec(r["member_token"])}
            for r in rows
        ]

    # -------------------------------------------------------------- traces
    def generation_trace(self, prompt: str, max_tokens: int = 20, mode: str = "strict"):
        """
        Step-by-step trace. Both modes are now fully deterministic --
        no random.choice anywhere in inference.py.

        Strict Mode's rule names are unchanged for the lineage stages
        (e.g. 'bridge_lineage_unique', 'exact_match_unique'); any tie
        lineage itself can't resolve now carries a rule name ending in
        '..._bigram_frequency' instead of a random pick -- see
        inference.py's _bigram_tie_break.

        Open Mode's rule is almost always 'ctm_weighted_vote' (the
        V1-V9 primary selection scored the full legal candidate set,
        with its own internal bigram/global-frequency tie-break
        cascade if the score itself ties -- see ivm.py's select()).
        'ctm_unavailable_deterministic_fallback' only appears if no
        CTM/IVM object was built at all. Stage 1/2 lineage rule names
        never appear for an Open Mode trace. See inference.py's
        module docstring.
        """
        text, ids, trace = self.model.generate(prompt, max_tokens=max_tokens, mode=mode)
        tok = self.model.tokenizer
        readable = []
        for step in trace:
            chosen = step["chosen"]
            entry = {
                "stage": step["stage"], "rule": step.get("rule"),
                "chosen_token": tok.id_to_token.get(chosen, chosen),
                "active_rels": sorted(step.get("active_rels", [])) if step.get("active_rels") else [],
            }
            if "scores" in step:
                entry["scores"] = {tok.id_to_token.get(c, c) if c in (0,1,2,3) else tok.decode([c]): v
                                    for c, v in step["scores"].items()}
            readable.append(entry)
        return text, readable

    def open_mode_scores(self, prompt: str) -> dict:
        """
        Full auditable V1-V9 score breakdown for the NEXT token given
        `prompt`, under Open Mode's primary selection mechanism (see
        ivm.py's score_candidates()/select()): which tokens are
        important, their influence (breadth), what each of the NINE
        layers (important vote / influence vote / context vote /
        context-influence vote / bigram-witness vote / adjacency
        vote / prev+current co-occurrence vote / triple witness vote /
        whole-context unanimous vote) contributed per candidate, the
        final combined score, and -- critically -- which stage
        actually decided the winner ("score", "bigram_frequency",
        "global_frequency", or "lowest_token_id"). The final "score"
        is the sum of all nine layers, including bigram_witness_vote
        (V5), adjacency_vote (V6), prev_current_vote (V7), triple_vote
        (V8), and whole_context_vote (V9) -- don't add up just the
        other four expecting it to match; V5/V6/V7/V8/V9 are usually
        the largest single contributors (V5 when the exact bigram was
        literally witnessed, V6 when the context token was ever
        directly, immediately followed by the candidate --
        directional, token->candidate only -- V7 when the prompt's
        last two tokens and the candidate ever all three shared one
        training sentence together, V8 when the prompt's
        last two tokens were ever literally, immediately followed by
        the candidate as one exact trained triple, V9 when EVERY
        context token, not just one, knows the candidate).
        Requires Open Mode to be available (model trained/loaded).
        Returns None otherwise.

        Candidates are always the entire vocabulary -- Open Mode has
        no successor gating at all, so there is no narrower option.
        """
        return self.model.open_mode_candidate_scores(prompt)

    def cache_control(self, action: str = "status") -> dict:
        """
        Toggle or inspect model.open_ctm's opt-in sparse per-token
        score cache (V1/V2/V3/V4/V6 only -- V5/V7/V8/V9 always stay
        live, see ivm.py's build_cache()). `action`:
          "on"     -- (re)build the cache for the full vocabulary
                      (model.all_candidate_tokens()) and enable it
          "off"    -- disable it (built data is kept, so "on" again
                      later doesn't have to rebuild)
          "status" -- just report the current state (default)
        Returns {"enabled": bool, "token_rows": int, "entries": int}.
        Requires Open Mode to be available; raises RuntimeError
        otherwise (this is a mutating/inspection call, not a report --
        unlike the read-only methods above, silently returning None
        would hide a real usage error).
        """
        if self.model.open_ctm is None:
            raise RuntimeError("Open Mode isn't ready (model not trained/loaded)")
        ivm = self.model.open_ctm
        if action == "on":
            ivm.enable_cache(self.model.all_candidate_tokens())
        elif action == "off":
            ivm.disable_cache()
        elif action != "status":
            raise ValueError(f"cache_control: unknown action {action!r} "
                              "(expected 'on', 'off', or 'status')")
        return {
            "enabled": ivm._use_cache,
            "token_rows": len(ivm._token_cache),
            "entries": sum(len(row) for row in ivm._token_cache.values()),
        }

    def bigram_frequency(self, prev_word: str, curr_word: str, mode: str = "strict") -> dict:
        """
        Raw bigram frequency for one pair: how many times `curr_word`
        literally followed `prev_word` in training. This is the exact
        count Strict Mode's tie-break (_bigram_tie_break) and Open
        Mode's first tie-break cascade stage (select()) both use --
        see inference.py/ivm.py.

        Also reports "witness_sentences": how many literal training
        sentences actually contained this bigram as a consecutive
        pair -- this is the evidence set V5's bigram-witness vote
        checks context tokens against (see ivm.py's
        bigram_relationships()). Mode-independent -- it is NOT the
        same number as "training" above whenever the bigram occurred
        more than once inside the same sentence, or occurs across
        more than one sentence; "training" counts raw occurrences,
        "witness_sentences" counts distinct sentences.
        """
        tok = self.model.tokenizer
        prev_id = tok.encode(prev_word)[-1]
        curr_id = tok.encode(curr_word)[-1]
        engine = self.model._engine(mode)
        training = self.model.edges.frequency(prev_id, curr_id)
        witnesses = self.model.bigram_witness_sentences(prev_id, curr_id)
        return {
            "prev": prev_word, "curr": curr_word,
            "training": training,
            "total": engine._bigram_frequency(prev_id, curr_id),
            "witness_sentences": len(witnesses),
        }

    # ----------------------------------------------------------- full report
    def full_report(self, top_n: int = 10) -> dict:
        return {
            "stats": self.model.stats(),
            "topology": self.topology(top_n=top_n),
            "clusters": self.cluster_report(top_n=top_n),
            "relationships": self.relationship_report(),
        }


# =============================================================================
# Plain-text rendering helpers (no external deps)
# =============================================================================

def _print_kv(d: dict):
    width = max((len(str(k)) for k in d), default=0)
    for k, v in d.items():
        print(f"  {str(k).ljust(width)} : {v}")


def _print_table(rows, headers):
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    line = "  " + "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(line)
    print("  " + "-" * (len(line) - 2))
    for row in rows:
        print("  " + "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)))


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyse an MSE-GLM corpus or trained model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", help="Path to a saved model folder (required for most commands)")
    parser.add_argument("--json", help="Write the result as JSON to this path instead of printing")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("corpus", help="Raw-text statistics — no trained model required")
    p.add_argument("--text", help="Inline corpus text")
    p.add_argument("--file", help="Path to a corpus text file")
    p.add_argument("--top", type=int, default=10)

    sub.add_parser("stats", help="Model-level structure counts (vocab/edges/bridges/clusters/relationships)")

    p = sub.add_parser("topology", help="Hub tokens and dead ends")
    p.add_argument("--top", type=int, default=10)

    p = sub.add_parser("clusters", help="Dual-axis cluster report")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--axis", choices=["bridge", "target"], default=None)

    p = sub.add_parser("cluster", help="Full detail for one cluster_id")
    p.add_argument("cluster_id", type=int)

    sub.add_parser("relationships", help="Relationship Matrix summary")

    p = sub.add_parser("relationship", help="Full detail for one relationship_id (training sentence)")
    p.add_argument("relationship_id", type=int)

    p = sub.add_parser("edges", help="Raw EdgeMatrix rows (source, dst, count), paged")
    p.add_argument("--source", help="Filter to one source word")
    p.add_argument("--target", help="Filter to one target/dst word")
    p.add_argument("--sort", choices=["count", "source"], default="count")
    p.add_argument("--limit", type=int, default=50, help="0 = no limit")
    p.add_argument("--offset", type=int, default=0)

    p = sub.add_parser("bridges", help="Raw BridgeMatrix rows (source, bridge, target, cluster_id), paged")
    p.add_argument("--source", help="Filter to one source word")
    p.add_argument("--bridge", help="Filter to one bridge word")
    p.add_argument("--target", help="Filter to one target word")
    p.add_argument("--clustered-only", action="store_true", help="Only rows with cluster_id != 0")
    p.add_argument("--limit", type=int, default=50, help="0 = no limit")
    p.add_argument("--offset", type=int, default=0)

    p = sub.add_parser("rel-rows",
                        help="Raw RelationshipMatrix rows (relationship_id, triple_id, "
                             "decoded source/bridge/target), paged -- normally the biggest "
                             "of the three matrices; filter with --relationship or --triple")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--relationship", type=int, help="Only rows for this relationship_id")
    g.add_argument("--triple", type=int, help="Only rows for this triple_id")
    p.add_argument("--limit", type=int, default=50, help="0 = no limit")
    p.add_argument("--offset", type=int, default=0)

    p = sub.add_parser("token", help="Per-token report: successors, bridge triples, clusters")
    p.add_argument("word")

    p = sub.add_parser("similarity", help="Cluster-overlap similarity between two tokens")
    p.add_argument("word_a")
    p.add_argument("word_b")

    p = sub.add_parser("shared", help="infer_shared_role() across two or more tokens")
    p.add_argument("words", nargs="+")

    p = sub.add_parser("interpret", help="Propose a human-readable label for one cluster_id")
    p.add_argument("cluster_id", type=int)
    p.add_argument("--top", type=int, default=InterpretConfig.TOP_N)
    p.add_argument("--mode", choices=["strict", "open"], default="strict")

    p = sub.add_parser("interpretations", help="Every qualifying label per cluster (unfiltered)")
    p.add_argument("--min-coverage", type=float, default=InterpretConfig.MIN_COVERAGE)
    p.add_argument("--max-per-cluster", type=int, default=InterpretConfig.MAX_PER_CLUSTER)
    p.add_argument("--mode", choices=["strict", "open"], default="strict")

    p = sub.add_parser("interpreter-matrix",
                        help="Filtered CI Matrix (coverage + corroborating evidence required); "
                             "a cluster may carry multiple labels. "
                             "Pipe with --json > interpreter_matrix.json to persist it.")
    p.add_argument("--min-coverage", type=float, default=InterpretConfig.MIN_COVERAGE)
    p.add_argument("--min-signals", type=int, default=InterpretConfig.MIN_SIGNALS)
    p.add_argument("--max-per-cluster", type=int, default=None)
    p.add_argument("--mode", choices=["strict", "open"], default="strict")

    p = sub.add_parser("zero-cluster",
                        help="Mine cluster_id==0 for groups the standard dual-axis rule "
                             "never assigns a cluster_id to (fix bridge+target, source varies)")
    p.add_argument("--min-group-size", type=int, default=InterpretConfig.MIN_GROUP_SIZE)
    p.add_argument("--mode", choices=["strict", "open"], default="strict")

    p = sub.add_parser("context-triggers",
                        help="Context Trigger Matrix: which surrounding tokens (whole-sentence "
                             "co-occurrence) support which cluster member, for contextual "
                             "disambiguation. Display/analysis only -- does not build/cache "
                             "model.ctm (needed for generate(use_context_triggers=True)); "
                             "that's a Python-API-only call: model.build_context_triggers().")
    p.add_argument("--min-support", type=int, default=CTMConfig.MIN_SUPPORT)
    p.add_argument("--mode", choices=["strict", "open"], default="strict")

    p = sub.add_parser("trace", help="Step-by-step generation trace for a prompt")
    p.add_argument("prompt")
    p.add_argument("--max-tokens", type=int, default=20)
    p.add_argument("--mode", choices=["strict", "open"], default="strict")

    p = sub.add_parser("open-scores",
                        help="Open Mode only: full V1-V9 weighted-vote breakdown "
                             "for the next token given a prompt -- important tokens, "
                             "influence, per-layer contributions (including V5's "
                             "bigram-witness vote, V6's adjacency vote, V7's "
                             "prev+current co-occurrence vote, and V8's triple "
                             "witness vote), the final "
                             "score, and which tie-break stage (if any) decided the winner. "
                             "This is Open Mode's PRIMARY selection mechanism, not a "
                             "tie-breaker; use this to audit why it picked what it picked. "
                             "Always scores the entire vocabulary -- Open Mode has no "
                             "successor gating at all.")
    p.add_argument("prompt")
    p.add_argument("--cache", action="store_true",
                    help="Enable the sparse V1/V2/V3/V4/V6 score cache "
                         "(model.open_ctm) before scoring -- same numbers "
                         "either way, this just times/labels which path ran "
                         "(see the printed 'cache: on/off' line and the "
                         "cache_used field in --json output).")
    p.add_argument("--top-n", type=int, default=ScoresDisplayConfig.TOP_N,
                    help="How many top-scored candidates to print to the "
                         "screen (default: config.py's "
                         "ScoresDisplayConfig.TOP_N). Scoring itself always "
                         "covers the entire vocabulary -- this only limits "
                         "the printed table, since a big vocab can't fit on "
                         "screen. Pass 0 to print every candidate. The "
                         "winner is always printed even if it falls outside "
                         "the top N. Ignored with --json, which always "
                         "returns every candidate's score.")

    p = sub.add_parser("cache",
                        help="Toggle or inspect model.open_ctm's opt-in sparse "
                             "per-token score cache (V1/V2/V3/V4/V6 only -- V5/V7/V8 "
                             "always stay live). Same scores either way; this is "
                             "purely a speed optimization -- see benchmark_cache.py "
                             "to actually measure the difference.")
    p.add_argument("action", nargs="?", choices=["on", "off", "status"], default="status")

    p = sub.add_parser("bigram",
                        help="Raw bigram frequency for one (prev, curr) pair -- the "
                             "exact count used by Strict Mode's tie-break and Open "
                             "Mode's first tie-break cascade stage, plus how many "
                             "distinct training sentences witnessed it (V5's evidence "
                             "set, see inference.py/ivm.py)")
    p.add_argument("prev")
    p.add_argument("curr")
    p.add_argument("--mode", choices=["strict", "open"], default="strict")

    p = sub.add_parser("report", help="Combined stats + topology + clusters + relationships")
    p.add_argument("--top", type=int, default=10)

    args = parser.parse_args()

    if args.command == "corpus":
        if not args.text and not args.file:
            print("Provide --text or --file", file=sys.stderr)
            sys.exit(1)
        text = args.text or open(args.file, "r", encoding="utf-8", errors="ignore").read()
        result = CorpusAnalyser(text).stats(top_n=args.top)
        _emit(result, args.json, lambda r: (
            _print_kv({k: v for k, v in r.items() if k != "top_words"}),
            print("  top_words:"),
            _print_table(r["top_words"], ["word", "count"]),
        ))
        return

    if not args.model:
        print("This command requires --model <folder>", file=sys.stderr)
        sys.exit(1)

    model = MSEGraphLanguageModel.load(args.model)
    analyser = Analyser(model)

    if args.command == "stats":
        result = model.stats()
        _emit(result, args.json, _print_kv)

    elif args.command == "topology":
        result = analyser.topology(top_n=args.top)
        _emit(result, args.json, lambda r: (
            print("  hub tokens (highest out-degree):"),
            _print_table(r["hub_tokens"], ["token", "out_degree"]),
            print(f"  dead-end token count: {r['dead_end_count']}"),
        ))

    elif args.command == "clusters":
        result = analyser.cluster_report(top_n=args.top, axis=args.axis)
        _emit(result, args.json, lambda r: _print_table(
            [(c["cluster_id"], c["axis"], c["slot"], ", ".join(c["members"])) for c in r],
            ["cluster_id", "axis", "slot", "members"],
        ))

    elif args.command == "cluster":
        result = analyser.cluster_detail(args.cluster_id)
        _emit(result, args.json, lambda r: (
            print(f"  cluster_id: {r['cluster_id']}   axis: {r['axis']}"),
            _print_table(r["triples"], ["source", "target", "bridge"]),
        ))

    elif args.command == "relationships":
        result = analyser.relationship_report()
        _emit(result, args.json, _print_kv)

    elif args.command == "relationship":
        result = analyser.relationship_detail(args.relationship_id)
        _emit(result, args.json, lambda r: (
            print(f"  relationship_id: {r['relationship_id']}"),
            _print_table(r["triples"], ["source", "target", "bridge"]),
        ))

    elif args.command == "edges":
        result = analyser.edge_matrix(source=args.source, target=args.target,
                                       sort=args.sort, limit=args.limit, offset=args.offset)
        _emit(result, args.json, lambda r: (
            print(f"  {r['total']} row(s) total (showing offset={r['offset']}, "
                  f"limit={r['limit'] or 'none'})"),
            _print_table(r["rows"], ["source", "dst", "count"]),
        ))

    elif args.command == "bridges":
        result = analyser.bridge_matrix(source=args.source, bridge=args.bridge, target=args.target,
                                         clustered_only=args.clustered_only,
                                         limit=args.limit, offset=args.offset)
        _emit(result, args.json, lambda r: (
            print(f"  {r['total']} row(s) total (showing offset={r['offset']}, "
                  f"limit={r['limit'] or 'none'})"),
            _print_table(r["rows"], ["source", "bridge", "target", "cluster_id"]),
        ))

    elif args.command == "rel-rows":
        result = analyser.relationship_matrix(relationship_id=args.relationship, triple_id=args.triple,
                                               limit=args.limit, offset=args.offset)
        _emit(result, args.json, lambda r: (
            print(f"  {r['total']} row(s) total (showing offset={r['offset']}, "
                  f"limit={r['limit'] or 'none'})"),
            _print_table(r["rows"], ["relationship_id", "triple_id", "source", "bridge", "target"]),
        ))

    elif args.command == "token":
        result = analyser.per_token_report(args.word)
        if result is None:
            print(f"'{args.word}' not found in vocabulary", file=sys.stderr)
            sys.exit(1)
        _emit(result, args.json, _print_kv)

    elif args.command == "similarity":
        result = analyser.token_similarity(args.word_a, args.word_b)
        _emit(result, args.json, _print_kv)

    elif args.command == "shared":
        results = model.infer_shared_role(args.words)
        result = [{"predicted": tok, "axis": axis, **ev} for tok, axis, ev in results]
        if not result:
            print("  no shared cluster found across those tokens")
        _emit(result, args.json, lambda r: _print_table(
            [(x["predicted"], x["axis"], x["cluster_id"], x["overlap"]) for x in r],
            ["predicted", "axis", "cluster_id", "overlap"],
        ) if r else None)

    elif args.command == "interpret":
        result = analyser.cluster_interpretation(args.cluster_id, top_n=args.top, mode=args.mode)
        if result is None:
            print(f"cluster_id {args.cluster_id} not found", file=sys.stderr)
            sys.exit(1)
        if not result["candidates"]:
            print(f"  cluster {args.cluster_id} ({result['axis']} axis, "
                  f"members: {', '.join(result['members'])}) — no interpreter found "
                  f"(corpus has no matching categorical statement for these members)")
        else:
            _emit(result, args.json, lambda r: (
                print(f"  cluster_id: {r['cluster_id']}   axis: {r['axis']}   "
                      f"members: {', '.join(r['members'])}"),
                _print_table(
                    [(c["interpreter_token"], c["coverage"], ",".join(c["evidence_mask"]),
                      ", ".join(c["members_covered"])) for c in r["candidates"]],
                    ["interpreter", "coverage", "evidence_mask", "members_covered"],
                ),
            ))

    elif args.command == "interpretations":
        result = analyser.interpretation_report(min_coverage=args.min_coverage,
                                                 max_per_cluster=args.max_per_cluster,
                                                 mode=args.mode)
        if not result:
            print("  no clusters cleared the coverage threshold")
        _emit(result, args.json, lambda r: _print_table(
            [(c["cluster_id"], cand["interpreter_token"], cand["coverage"],
              ",".join(cand["evidence_mask"]), ", ".join(cand["members_covered"]))
             for c in r for cand in c["candidates"]],
            ["cluster_id", "interpreter", "coverage", "evidence_mask", "members_covered"],
        ) if r else None)

    elif args.command == "interpreter-matrix":
        result = analyser.interpreter_matrix(min_coverage=args.min_coverage,
                                              min_signals=args.min_signals,
                                              max_per_cluster=args.max_per_cluster,
                                              mode=args.mode)
        if not result:
            print("  no clusters cleared both the coverage and evidence thresholds")
        _emit(result, args.json, lambda r: _print_table(
            [(row["cluster_id"], row["interpreter_token"], row["coverage"],
              ",".join(row["evidence_mask"]), ", ".join(row["members_covered"])) for row in r],
            ["cluster_id", "interpreter", "coverage", "evidence_mask", "members_covered"],
        ) if r else None)

    elif args.command == "zero-cluster":
        result = analyser.zero_cluster_groups(min_group_size=args.min_group_size, mode=args.mode)
        if not result:
            print("  no groups found in the unclustered bucket at this min_group_size")
        _emit(result, args.json, lambda r: _print_table(
            [(row["interpreter_token"], row["member_count"], ",".join(row["evidence_mask"]),
              ", ".join(row["members"])) for row in r],
            ["interpreter", "member_count", "evidence_mask", "members"],
        ) if r else None)

    elif args.command == "context-triggers":
        result = analyser.context_trigger_matrix(min_support=args.min_support, mode=args.mode)
        if not result:
            print("  no triggers found at this min_support")
        _emit(result, args.json, lambda r: _print_table(
            [(row["trigger_token"], row["cluster_id"], row["member_token"], row["support"])
             for row in r],
            ["trigger_token", "cluster_id", "member_token", "support"],
        ) if r else None)

    elif args.command == "trace":
        text, trace = analyser.generation_trace(args.prompt, max_tokens=args.max_tokens, mode=args.mode)
        result = {"output": text, "trace": trace}
        _emit(result, args.json, lambda r: (
            print(f"  output: {r['output']}"),
            _print_table(
                [(s["stage"], s["rule"], s["chosen_token"], s["active_rels"]) for s in r["trace"]],
                ["stage", "rule", "chosen_token", "active_rels"],
            ),
        ))

    elif args.command == "open-scores":
        if args.cache:
            analyser.cache_control("on")
        result = analyser.open_mode_scores(args.prompt)
        if result is None:
            print("Open Mode isn't ready (model not trained/loaded)", file=sys.stderr)
            sys.exit(1)

        def _print_open_scores(r):
            print(f"  important tokens: {r['important_tokens']}")
            print(f"  influence:        {r['influence']}")

            # --top-n only limits what gets PRINTED -- the scores dict
            # (and therefore the winner/tie-break decision) always
            # covers the entire vocabulary. 0 (or a value >= the vocab
            # size) means print everything.
            ranked = sorted(r["scores"], key=r["scores"].get, reverse=True)
            total = len(ranked)
            top_n = args.top_n
            shown = ranked if not top_n or top_n >= total else ranked[:top_n]
            truncated = len(shown) < total
            if truncated:
                print(f"  (scoring the entire vocabulary -- {total} candidates; "
                      f"showing top {len(shown)} by score -- "
                      f"--top-n <n> or --top-n 0 to change)")

            def _row(c, tag):
                return (c, round(r["important_vote"].get(c, 0), 2),
                        round(r["influence_vote"].get(c, 0), 2),
                        round(r["context_vote"].get(c, 0), 2),
                        round(r["context_influence_vote"].get(c, 0), 2),
                        round(r["bigram_witness_vote"].get(c, 0), 2),
                        round(r["adjacency_vote"].get(c, 0), 2),
                        round(r["prev_current_vote"].get(c, 0), 2),
                        round(r["triple_vote"].get(c, 0), 2),
                        round(r["whole_context_vote"].get(c, 0), 2),
                        round(r["scores"][c], 2),
                        tag)

            rows = [_row(c, "<- winner" if c == r["winner"] else "") for c in shown]
            if truncated and r["winner"] not in shown:
                rows.append(_row(r["winner"], "<- winner (outside top N)"))
            _print_table(
                rows,
                ["candidate", "V1 (important)", "V2 (influence)", "V3 (context)",
                 "V4 (ctx.infl.)", "V5 (bigram witness)", "V6 (adjacency)",
                 "V7 (prev+curr)", "V8 (triple)", "V9 (unanimous)", "score", ""],
            )
            print(f"\n  winner: {r['winner']}  (decided by: {r['tie_break_stage']}"
                  f"  ·  cache: {'on' if r['cache_used'] else 'off'})")

        # --json always returns every candidate's score, untruncated --
        # top-n is a screen-display concern only.
        _emit(result, args.json, _print_open_scores)

    elif args.command == "cache":
        try:
            result = analyser.cache_control(args.action)
        except RuntimeError as e:
            print(e, file=sys.stderr)
            sys.exit(1)
        _emit(result, args.json, lambda r: print(
            f"  cache: {'ON' if r['enabled'] else 'OFF'}  "
            f"({r['token_rows']} token rows, {r['entries']} (t,c) entries"
            f"{' -- none built yet' if not r['token_rows'] else ''})"
        ))

    elif args.command == "bigram":
        result = analyser.bigram_frequency(args.prev, args.curr, mode=args.mode)
        _emit(result, args.json, lambda r: (
            print(f"  '{r['prev']}' -> '{r['curr']}':  training={r['training']}  "
                  f"total={r['total']}  "
                  f"witness_sentences={r['witness_sentences']}"),
            print("  (witness_sentences is the evidence V5's bigram-witness vote "
                  "checks context tokens against -- see ivm.py)"),
        ))

    elif args.command == "report":
        result = analyser.full_report(top_n=args.top)
        _emit(result, args.json, lambda r: (
            print("== stats =="), _print_kv(r["stats"]),
            print("\n== topology =="),
            _print_table(r["topology"]["hub_tokens"], ["token", "out_degree"]),
            print("\n== clusters =="),
            _print_table(
                [(c["cluster_id"], c["axis"], c["slot"], ", ".join(c["members"])) for c in r["clusters"]],
                ["cluster_id", "axis", "slot", "members"],
            ),
            print("\n== relationships =="), _print_kv(r["relationships"]),
        ))


def _emit(result, json_path, printer):
    if json_path:
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Wrote {json_path}")
    else:
        printer(result)


if __name__ == "__main__":
    main()
