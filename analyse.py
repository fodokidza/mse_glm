"""
analyse.py — Analysis layer for MSE-GLM.

Two library classes:
    CorpusAnalyser  — raw-text statistics, no trained model required.
    Analyser        — graph-level statistics on a trained MSEGraphLanguageModel:
                       topology, dual-axis cluster reports, tile table
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

from model import MSEGraphLanguageModel, normalize, split_sentences
from config import ScoresDisplayConfig


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
        out_degree = Counter(self.model.triple_out_degree())
        hubs = out_degree.most_common(top_n)
        vocab_ids = range(self.model.vocab_size)
        dead_ends = [t for t in vocab_ids if t not in out_degree]
        return {
            "hub_tokens": [(self.model.token_name(t), c) for t, c in hubs],
            "dead_end_count": len(dead_ends),
            "dead_end_tokens": [self.model.token_name(t) for t in dead_ends[:top_n]],
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
        Adjacency rows of the unified matrix: every deduplicated (source, dst)
        bigram with its literal training count. Optionally filtered to a single
        source and/or target word. `sort`: "count" (default, descending) or
        "source" (ascending source token id, the matrix's native row order).
        """
        rows = self.model.bigram_rows()

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
        decoded = [(self.model.token_name(s), self.model.token_name(d), c) for s, d, c in page]
        return {"total": total, "offset": offset, "limit": limit, "rows": decoded}

    def bridge_matrix(self, source: str = None, bridge: str = None, target: str = None,
                       clustered_only: bool = False, limit: int = 50, offset: int = 0) -> dict:
        """
        Slot-filler rows of the unified matrix: every deduplicated
        (source, bridge, target) triple with its derived cluster_id (0 = in no
        slot group; ids are computed on demand, not stored). Optionally
        filtered to a single source, bridge, and/or target word, and/or
        restricted to grouped rows only (cluster_id != 0). Native order (by source).
        """
        rows = self.model.triple_rows()

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
        decoded = [(self.model.token_name(s), self.model.token_name(br),
                    self.model.token_name(t), c) for s, br, t, c in page]
        return {"total": total, "offset": offset, "limit": limit, "rows": decoded}

    def tile_table(self, relationship_id: int = None, limit: int = 50, offset: int = 0) -> dict:
        """
        The unified matrix's tile table: how each unique training sentence is
        stored -- its ordered non-overlapping adjacent pairs (an odd-length
        sentence ends with an overlapping pair, a one-token sentence with a
        single token; the second token is then None). Rows are
        (relationship_id, position, first_token, second_token).
        Optionally filtered to one relationship_id (one sentence).
        """
        rows = self.model.tile_rows(relationship_id)
        total = len(rows)
        page = rows[offset:offset + limit] if limit else rows[offset:]
        dec = lambda t: None if t is None else self.model.token_name(t)
        decoded = [(r, pos, dec(a), dec(b)) for r, pos, a, b in page]
        return {"total": total, "offset": offset, "limit": limit, "rows": decoded}

    def _resolve_token(self, word):
        """Encode a surface word to its last real token id (drops <BOS>), or
        None if it doesn't exist in the vocabulary. Shared by every raw-
        matrix filter above -- same convention as per_token_report()."""
        if word is None:
            return None
        enc = [t for t in self.model.encode(word) if t != 2]
        return enc[-1] if enc else None

    # -------------------------------------------------------------- clusters
    def cluster_report(self, top_n: int = 10, axis: str = None) -> list:
        """
        List slot groups ("clusters"; ids are derived, not stored). `axis`
        optionally filters to 'bridge' or 'target' groups only.
        """
        u = self.model

        report = []
        for cid in u.cluster_ids():
            cluster_axis, members = u.cluster_axis(cid)
            if axis and cluster_axis != axis:
                continue
            s0, t0, _ = members[0]
            if cluster_axis == "bridge":
                varying = sorted(set(self.model.token_name(br) for _, _, br in members))
                fixed = f"{self.model.token_name(s0)} -> ___ -> {self.model.token_name(t0)}"
            elif cluster_axis == "target":
                varying = sorted(set(self.model.token_name(t) for _, t, _ in members))
                fixed = f"{self.model.token_name(s0)} -> {self.model.token_name(members[0][2])} -> ___"
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
        axis, members = self.model.cluster_axis(cluster_id)
        decoded = [
            (self.model.token_name(s), self.model.token_name(t), self.model.token_name(br))
            for s, t, br in members
        ]
        return {"cluster_id": cluster_id, "axis": axis, "triples": decoded}

    # --------------------------------------------------------- relationships
    def relationship_report(self) -> dict:
        u = self.model
        appearances = Counter()
        for rid in range(u.n_rels):
            appearances.update(set(u.triples_for_relationship(rid)))
        return {
            "total_relationships": u.n_rels,
            "total_occurrences": sum(u.rel_count),
            "tile_rows": len(u.t_cell),
            "unique_triples_referenced": len(appearances),
            "shared_triple_count": sum(1 for n in appearances.values() if n > 1),
        }

    def relationship_detail(self, relationship_id: int) -> dict:
        """Every triple of a single training sentence, plus how many literal
        training occurrences shared that exact sentence content."""
        u = self.model
        triples = [(self.model.token_name(s), self.model.token_name(t), self.model.token_name(br))
                   for s, t, br in u.triples_for_relationship(relationship_id)]
        return {"relationship_id": relationship_id, "triples": triples,
                "occurrences": u.count(relationship_id)}

    # ---------------------------------------------------------- per-token
    def per_token_report(self, word: str) -> dict:
        enc = [t for t in self.model.encode(word) if t != 2]
        if not enc:
            return None
        token = enc[-1]
        u = self.model
        out_edges = u.successors(token)
        triples = u.triples_from_source(token)
        clusters = u.clusters_of_token(token)
        return {
            "token": self.model.token_name(token),
            "edge_successors": [self.model.token_name(t) for t in out_edges],
            "bridge_triples_as_source": len(triples),
            "cluster_memberships": clusters,
        }

    def token_similarity(self, word_a: str, word_b: str) -> dict:
        """
        Relatedness between two tokens = |T_index[a] ∩ T_index[b]|, per SDD
        v2.1 §10. Returns the shared clusters and a plain similarity count.
        """
        u = self.model

        def resolve(word):
            enc = [t for t in self.model.encode(word) if t != 2]
            return enc[-1] if enc else None

        ta, tb = resolve(word_a), resolve(word_b)
        if ta is None or tb is None:
            return {"word_a": word_a, "word_b": word_b, "similarity": 0, "shared_clusters": []}
        sa, sb = set(u.clusters_of_token(ta)), set(u.clusters_of_token(tb))
        shared = sorted(sa & sb)
        return {
            "word_a": word_a, "word_b": word_b,
            "similarity": len(shared), "shared_clusters": shared,
        }

    def noise_homes(self, word: str) -> list:
        """Every training sentence containing this word -- see model.noise_homes()/noise.find_homes()."""
        return self.model.noise_homes(word)

    def noise_scores(self, anchor_word: str, home: int = None) -> dict:
        """
        Noise-cancellation scores for one anchor+home, broken down by
        the three underlying raw stages -- see
        model.noise_scores_detailed()/noise.noise_scores_detailed()'s
        docstring. Raises ValueError if anchor_word has multiple
        candidate home sentences and `home` wasn't given.
        """
        return self.model.noise_scores_detailed(anchor_word, home=home)

    def focus_scores(self, anchor_word: str, weight_mode: str = "count") -> dict:
        """
        Global, cross-sentence version of noise_scores() -- stage1,
        stage2, stage3 (the "in this sentence, who should I care about"
        scoring, only for candidates sharing a home with the anchor),
        plus stage4 (shared-knowledge attention:
        for every other sentence, the tokens NOT common with one of the
        anchor's homes attend to each other, points accumulating across
        every sentence pair; common tokens score 0). weight_mode="count"
        (default) weights each pair by |common|; "flat" gives every pair
        weight 1, isolating the effect of that scaling. These are RAW
        counts; model.normalize_scores() turns them into percentages and
        the final score (the average of the four normalized stages). See
        noise.focus_scores_detailed()'s docstring.
        """
        return self.model.focus_scores_detailed(anchor_word, weight_mode=weight_mode)

    def token_row(self, word: str) -> dict:
        """Permanent, precomputed vocabulary row for one token, broken down by stage -- see model.token_row_detailed()/noise.vocabulary_row_detailed()."""
        return self.model.token_row_detailed(word)

    def combine(self, words: list, weight_mode: str = "count") -> dict:
        """
        RAW combine: each named word's own focus_scores_detailed() row,
        summed element-wise across words, target by target (all four stages;
        no combined number on raw rows). weight_mode=
        "count" (default)/"flat" -- see noise.combine_context_detailed().
        """
        return self.model.combine_detailed(words, weight_mode=weight_mode)

    def combine_normalized(self, words: list, weight_mode: str = "count") -> dict:
        """
        NORMALIZED combine: the named words' RAW rows are summed element-wise
        and the SUM is normalized (per stage, 100 / total) -- exactly what
        generation does. See noise.combine_context_normalized().
        """
        return self.model.combine_normalized(words, weight_mode=weight_mode)

    def noise_why(self, anchor_word: str, home: int, candidate_word: str) -> dict:
        """Independent follow-up: which relationship_ids voted yes for one candidate? See model.noise_why()/noise.why()."""
        return self.model.noise_why(anchor_word, home, candidate_word)

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

        Open Mode's rule is almost always 'ivm_noise_weighted_vote' (ivm's
        Q1-Q4 plus noise's final score scored the full candidate set, with
        its own internal bigram/global-frequency tie-break cascade if the
        total itself ties -- see ivm.py's select()).
        'ivm_unavailable_deterministic_fallback' only appears if no
        ivm object was built at all. Stage 1/2 lineage rule names
        never appear for an Open Mode trace. See inference.py's
        module docstring.
        """
        text, ids, trace = self.model.generate(prompt, max_tokens=max_tokens, mode=mode)
        readable = []
        for step in trace:
            chosen = step["chosen"]
            entry = {
                "stage": step["stage"], "rule": step.get("rule"),
                "chosen_token": self.model.token_name(chosen),
                "active_rels": sorted(step.get("active_rels", [])) if step.get("active_rels") else [],
            }
            if "scores" in step:
                entry["scores"] = {self.model.token_label(c): v
                                    for c, v in step["scores"].items()}
            readable.append(entry)
        return text, readable

    def open_mode_scores(self, prompt: str) -> dict:
        """
        Full auditable score breakdown for the NEXT token given `prompt`
        under Open Mode's selection (see ivm.py and noise.py): what each
        of ivm's four questions contributed per candidate (q1 co-occurrence,
        q2 witnessed legal bigram, q3 legal-bigram check, q4 agreement check),
        ivm's final score (their sum), noise's final score (weighted, plus
        its raw scale as noise_raw), the total (ivm + noise), and --
        critically -- which stage actually decided the winner ("score",
        "bigram_frequency", "global_frequency", or "lowest_token_id").
        This is the on-demand "which question gave how many points" view;
        generation itself only uses the totals.
        Requires Open Mode to be available (model trained/loaded).
        Returns None otherwise.

        Candidates are always the entire vocabulary -- Open Mode has
        no successor gating at all, so there is no narrower option.
        """
        return self.model.open_mode_candidate_scores(prompt)

    def set_legality_top_k(self, k=None):
        """How many of the top-scored tokens Q3/Q4 may work on: a positive int, or None/0/"all"
        for every candidate -- see model.set_legality_top_k(). Returns the stored value."""
        return self.model.set_legality_top_k(k)

    def cache_control(self, action: str = "status") -> dict:
        """
        Toggle or inspect ivm's opt-in sparse per-token
        score cache (ivm's Q1 only -- Q2-Q4 always stay live, see
        ivm.py's build_cache()). `action`:
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
        return self.model.cache_control(action)

    def bigram_frequency(self, prev_word: str, curr_word: str, mode: str = "strict") -> dict:
        """
        Raw bigram frequency for one pair: how many times `curr_word`
        literally followed `prev_word` in training. This is the exact
        count Strict Mode's tie-break (_bigram_tie_break) and Open
        Mode's first tie-break cascade stage (select()) both use --
        see inference.py/ivm.py.

        Also reports "witness_sentences": how many literal training
        sentences actually contained this bigram as a consecutive
        pair -- this is the evidence set ivm's Q2
        checks context tokens against (see ivm.py's
        bigram_relationships()). Mode-independent -- it is NOT the
        same number as "training" above whenever the bigram occurred
        more than once inside the same sentence, or occurs across
        more than one sentence; "training" counts raw occurrences,
        "witness_sentences" counts distinct sentences.
        """
        return self.model.bigram_info(prev_word, curr_word, mode=mode)

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

    p = sub.add_parser("tiles",
                        help="Tile table rows (relationship_id, position, first, second): how each "
                             "training sentence is stored; filter with --relationship")
    p.add_argument("--relationship", type=int, help="Only tiles of this relationship_id")
    p.add_argument("--limit", type=int, default=50, help="0 = no limit")
    p.add_argument("--offset", type=int, default=0)

    p = sub.add_parser("token", help="Per-token report: successors, bridge triples, clusters")
    p.add_argument("word")

    p = sub.add_parser("similarity", help="Cluster-overlap similarity between two tokens")
    p.add_argument("word_a")
    p.add_argument("word_b")

    p = sub.add_parser("shared", help="infer_shared_role() across two or more tokens")
    p.add_argument("words", nargs="+")

    p = sub.add_parser("noise-homes",
                        help="noise.py: every training sentence (relationship_id) containing "
                             "this word -- use this to pick a home_relationship_id for "
                             "'noise-scores' when the word appears in more than one training "
                             "sentence.")
    p.add_argument("word")

    p = sub.add_parser("noise-scores",
                        help="Noise-cancellation scoring (noise.py): given an anchor word "
                             "sitting in one home sentence, score every OTHER word in that "
                             "sentence across three vote rules -- stage1 (literal sentence "
                             "membership), stage2 (sentence-level global co-occurrence), stage3 "
                             "(token-level global co-occurrence, deduped) -- shown as three raw "
                             "columns (they are on different scales, so they are not combined). A voter votes 'pay attention' only if it "
                             "does NOT already know the word -- 'the' lands at 0 on every stage, "
                             "with no stopword list.")
    p.add_argument("anchor_word")
    p.add_argument("--home", type=int, help="relationship_id to use as the home sentence -- "
                                              "required if anchor_word appears in more than "
                                              "one training sentence (see noise-homes)")

    p = sub.add_parser("focus-scores",
                        help="Global, cross-sentence version of noise-scores: stage1, stage2, "
                             "stage3 (for candidates sharing a home "
                             "with the anchor), plus stage4 -- shared-knowledge attention: for "
                             "every other sentence, tokens NOT common with one of the anchor's "
                             "homes attend to each other with weight = number of common tokens, "
                             "accumulating across every sentence pair (common tokens score 0). "
                             "By default the four stages are shown as percentages plus final% = "
                             "their average; --raw shows the raw counts.")
    p.add_argument("anchor_word")
    p.add_argument("--top", type=int, default=30,
                    help="Rows to print, best first by final% (by stage4 then stage3 with --raw) (0 = all). Many "
                         "candidates score 0 on stage4, so the tail can be the whole vocabulary. "
                         "Does not limit --json output.")
    p.add_argument("--stage4-weight", choices=["count", "flat"], default="count",
                    help="How each sentence pair contributes to stage4: 'count' (default) "
                         "weights it by |common tokens|; 'flat' gives every pair equal weight 1, "
                         "to see stage4's ranking without the |common| scaling.")
    p.add_argument("--raw", action="store_true",
                    help="Show raw counts instead of the default normalized percentages. "
                         "Normalized (default): every stage shown as a percentage of that "
                         "stage's total WITHIN THIS QUERY (sum across the candidates shown "
                         "here, not the whole vocabulary) -- see noise.normalize_row(). final% is the average of the four "
                         "normalized stages (stage1%..stage4%).")

    p = sub.add_parser("token-row",
                        help="Permanent, precomputed vocabulary row for one token (noise.py's "
                             "TokenVocabularyMatrix), broken down by raw stage -- "
                             "'wherever this token appears, these are the tokens it has "
                             "accumulated evidence to pay attention to'. Never re-reads any "
                             "training sentence -- built once (auto-built on first use).")
    p.add_argument("word")

    p = sub.add_parser("combine",
                        help="Each named word's own focus-scores row (stage1-3, stage4), summed "
                             "element-wise across the named words, target by target -- "
                             "'stage1 for cat' + 'stage1 for sat' (and likewise for every stage) "
                             "-- the inference-time step: which OTHER words does a context of "
                             "these active tokens have the most accumulated evidence to focus "
                             "on? --normalize switches to normalize-each-word-FIRST-then-sum "
                             "(see --normalize's own help), a different, larger result than "
                             "summing raw rows and normalizing the sum afterward.")
    p.add_argument("words", nargs="+")
    p.add_argument("--top", type=int, default=30,
                    help="Rows to print, best first by final% (by stage4 then stage3 with --raw) (0 = all). Many "
                         "candidates score 0 on stage4, so the tail can be the whole vocabulary. "
                         "Does not limit --json output.")
    p.add_argument("--stage4-weight", choices=["count", "flat"], default="count",
                    help="How each sentence pair contributes to stage4: 'count' (default) "
                         "weights it by |common tokens|; 'flat' gives every pair equal weight 1, "
                         "to see stage4's ranking without the |common| scaling.")
    p.add_argument("--raw", action="store_true",
                    help="Show raw counts instead of the default normalized combine. "
                         "Normalized (default): the named words' RAW rows are added "
                         "element-wise, then the SUM is normalized per stage (100 / that stage's "
                         "total across the targets shown); final% is the average of the four "
                         "normalized stages. --raw shows the summed raw rows.")

    p = sub.add_parser("noise-why",
                        help="Independent follow-up to noise-scores/focus-scores/token-row: "
                             "which SPECIFIC relationship_ids don't already know one candidate, "
                             "given an anchor+home? Never computed by the bulk/fast paths -- "
                             "only here, on demand.")
    p.add_argument("anchor_word")
    p.add_argument("home", type=int)
    p.add_argument("candidate_word")

    p = sub.add_parser("trace", help="Step-by-step generation trace for a prompt")
    p.add_argument("prompt")
    p.add_argument("--max-tokens", type=int, default=20)
    p.add_argument("--mode", choices=["strict", "open"], default="strict")

    p = sub.add_parser("open-scores",
                        help="Open Mode only: score breakdown for the next token given a prompt -- "
                             "what each of ivm's four questions contributed (Q1 co-occurrence, "
                             "Q2 witnessed legal bigram, Q3 legal-bigram check, Q4 agreement check), "
                             "ivm's score, noise's score, the total (ivm + noise), the final legal-only score, "
                             "and which tie-break stage (if any) decided the winner. "
                             "This is Open Mode's PRIMARY selection mechanism, not a "
                             "tie-breaker; use this to audit why it picked what it picked. "
                             "Always scores the entire vocabulary -- Open Mode has no "
                             "successor gating at all.")
    p.add_argument("prompt")
    p.add_argument("--legality-top", default=None, metavar="N|all",
                   help="How many of the top-scored tokens Q3/Q4 may work on: a number, or 'all' "
                        "(the default, IVMConfig.LEGALITY_TOP_K). If none of the top N is a legal "
                        "continuation, <EOS> wins.")
    p.add_argument("--cache", action="store_true",
                    help="Enable ivm's sparse Q1 score cache "
                         "before scoring -- same numbers "
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
                        help="Toggle or inspect ivm's opt-in sparse "
                             "per-token score cache (Q1 only -- Q2-Q4 always stay live). Same scores either way; this is "
                             "purely a speed optimization -- see benchmark_cache.py "
                             "to actually measure the difference.")
    p.add_argument("action", nargs="?", choices=["on", "off", "status"], default="status")

    p = sub.add_parser("bigram",
                        help="Raw bigram frequency for one (prev, curr) pair -- the "
                             "exact count used by Strict Mode's tie-break and Open "
                             "Mode's first tie-break cascade stage, plus how many "
                             "distinct training sentences witnessed it (Q2's evidence "
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

    elif args.command == "tiles":
        result = analyser.tile_table(relationship_id=args.relationship,
                                      limit=args.limit, offset=args.offset)
        _emit(result, args.json, lambda r: (
            print(f"  {r['total']} row(s) total (showing offset={r['offset']}, "
                  f"limit={r['limit'] or 'none'})"),
            _print_table(r["rows"], ["relationship_id", "position", "first", "second"]),
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

    elif args.command == "noise-homes":
        result = analyser.noise_homes(args.word)
        if not result:
            print(f"  '{args.word}' was never trained")
        _emit(result, args.json, lambda r: _print_table(
            [(row["relationship_id"], row["sentence"]) for row in r],
            ["relationship_id", "sentence"],
        ) if r else None)

    elif args.command == "noise-scores":
        try:
            result = analyser.noise_scores(args.anchor_word, home=args.home)
        except ValueError as e:
            print(f"  error: {e}")
            return
        if result is None:
            print(f"  '{args.anchor_word}' was never trained")
            return

        def _print_noise(r):
            print(f"  home #{r['home']}: \"{r['home_sentence']}\"")
            rows = sorted(
                ((w, info["stage1"], info["stage2"], info["stage3"])
                 for w, info in r["scores"].items()),
                key=lambda row: (-row[3], -row[2], -row[1]),
            )
            _print_table(rows, ["word", "stage1", "stage2", "stage3"])
            print("\n  (raw counts, not combined -- they are on different scales; "
                  "for stage 2's exact voters on one candidate, use noise-why)")
        _emit(result, args.json, _print_noise)

    elif args.command == "focus-scores":
        result = analyser.focus_scores(args.anchor_word, weight_mode=args.stage4_weight)
        if result is None:
            print(f"  '{args.anchor_word}' was never trained")
            return
        if not args.raw:
            result["scores"] = analyser.model.normalize_scores(result["scores"])

        def _print_focus(r):
            print(f"  homes: {r['homes']}")
            fmt = (lambda v: v) if args.raw else (lambda v: round(v, 2))
            if args.raw:
                rows = sorted(
                    ((w, fmt(info["stage1"]), fmt(info["stage2"]), fmt(info["stage3"]),
                      fmt(info["stage4"]))
                     for w, info in r["scores"].items()),
                    key=lambda row: (-row[4], -row[3], -row[2], -row[1]),
                )
                headers = ["word", "stage1", "stage2", "stage3", "stage4"]
            else:
                rows = sorted(
                    ((w, fmt(info["stage1"]), fmt(info["stage2"]), fmt(info["stage3"]),
                      fmt(info["stage4"]), fmt(info["final"]))
                     for w, info in r["scores"].items()),
                    key=lambda row: -row[5],
                )
                headers = ["word", "stage1%", "stage2%", "stage3%", "stage4%", "final%"]
            total = len(rows)
            if args.top and total > args.top:
                rows = rows[:args.top]
            _print_table(rows, headers)
            if args.top and total > args.top:
                print(f"  ... showing top {args.top} of {total} (use --top 0 for all)")
        _emit(result, args.json, _print_focus)

    elif args.command == "token-row":
        result = analyser.token_row(args.word)
        if result is None:
            print(f"  '{args.word}' was never trained")
        elif not result:
            print(f"  '{args.word}' was trained but never had a home sentence to learn a row from")

        def _print_row(r):
            if not r:
                return
            rows = sorted(
                ((w, info["stage1"], info["stage2"], info["stage3"])
                 for w, info in r.items()),
                key=lambda row: (-row[3], -row[2], -row[1]),
            )
            _print_table(rows, ["word", "stage1", "stage2", "stage3"])
        _emit(result, args.json, _print_row)

    elif args.command == "combine":
        if args.raw:
            result = analyser.combine(args.words, weight_mode=args.stage4_weight)
        else:
            result = analyser.combine_normalized(args.words, weight_mode=args.stage4_weight)
        if not result:
            print(f"  none of {args.words} were trained / had any accumulated row")

        def _print_combined(r):
            if not r:
                return
            fmt = (lambda v: v) if args.raw else (lambda v: round(v, 2))
            if args.raw:
                rows = sorted(
                    ((w, fmt(info["stage1"]), fmt(info["stage2"]), fmt(info["stage3"]),
                      fmt(info["stage4"]))
                     for w, info in r.items()),
                    key=lambda row: (-row[4], -row[3], -row[2], -row[1]),
                )
                headers = ["word", "stage1", "stage2", "stage3", "stage4"]
            else:
                rows = sorted(
                    ((w, fmt(info["stage1"]), fmt(info["stage2"]), fmt(info["stage3"]),
                      fmt(info["stage4"]), fmt(info["final"]))
                     for w, info in r.items()),
                    key=lambda row: -row[5],
                )
                headers = ["word", "stage1%", "stage2%", "stage3%", "stage4%", "final%"]
            total = len(rows)
            if args.top and total > args.top:
                rows = rows[:args.top]
            _print_table(rows, headers)
            if args.top and total > args.top:
                print(f"  ... showing top {args.top} of {total} (use --top 0 for all)")
        _emit(result, args.json, _print_combined)

    elif args.command == "noise-why":
        try:
            result = analyser.noise_why(args.anchor_word, args.home, args.candidate_word)
        except ValueError as e:
            print(f"  error: {e}")
            return
        if result is None:
            print(f"  '{args.anchor_word}' or '{args.candidate_word}' was never trained")
            return
        _emit(result, args.json, lambda r: (
            print(f"  score: {r['score']}"),
            print(f"  voters (relationship_ids): {r['voters']}"),
        ))

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
        if args.legality_top is not None:
            try:
                analyser.set_legality_top_k(args.legality_top)
            except ValueError as e:
                print(f"  {e}", file=sys.stderr)
                sys.exit(2)
        result = analyser.open_mode_scores(args.prompt)
        if result is None:
            print("Open Mode isn't ready (model not trained/loaded)", file=sys.stderr)
            sys.exit(1)

        def _print_open_scores(r):
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

            layers = analyser.model.score_layers()   # (key, label): q1..q4, ivm, noise, score, final

            def _row(c, tag):
                return (c, *[round(r[key].get(c, 0), 4) for key, _ in layers], tag)

            rows = [_row(c, "<- winner" if c == r["winner"] else "") for c in shown]
            if truncated and r["winner"] not in shown:
                rows.append(_row(r["winner"], "<- winner (outside top N)"))
            _print_table(rows, ["candidate", *[label for _, label in layers], ""])
            print(f"\n  (score = ivm + noise; noise = noise's own final score x {r['noise_weight']:g}, "
                  f"see config.NoiseConfig.INFERENCE_WEIGHT; Q3/Q4 = 1 means legal and never change a "
                  f"score; final = score if eligible to win, else 0; eligible set from: {r['legality']}"
                  + ("" if r["q4_active"] else "; Q4 inactive: fewer than two context tokens")
                  + ("" if r["legality_top_k"] is None else
                     f"; Q3/Q4 worked on the top {r['legality_top_k']} (window {r['window_size']})")
                  + ")")
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
            print("  (witness_sentences is the evidence ivm's Q2 "
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
