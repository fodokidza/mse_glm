"""
benchmark_cache.py -- compare Open Mode's IVM scoring with the sparse
per-token cache (V1/V2/V3/V4/V6) ON vs. OFF, on the same trained model,
same steps, same candidates. Prints per-call timing and confirms the
two paths agree on every score (not just that one is faster).

Usage:
    python3 benchmark_cache.py [vocab_size] [n_steps]
"""
import sys
import time
import random

from model import MSEGraphLanguageModel
from ivm import ImportanceVoteMatrix

CORPUS = """
the cat sat on the mat.
the dog sat on the carpet.
the boy sat on the chair.
the boy ran on the road.
the girl jumped over the fence.
the cat chased the dog around the yard.
a big dog barked at the small cat.
the old man walked slowly down the road.
the young girl read a book by the window.
a small bird flew over the tall tree.
the quiet cat slept on the warm mat.
the loud dog ran across the wide road.
""" * 4  # repeat so there's enough graph structure to make caching worthwhile


def run(vocab_size=400, n_steps=300):
    m = MSEGraphLanguageModel(vocab_size=vocab_size)
    m.train(CORPUS)
    candidates = m.all_candidate_tokens()

    ivm_live = ImportanceVoteMatrix.build(m, mode="strict", use_cache=False)
    ivm_cached = ImportanceVoteMatrix.build(m, mode="strict", use_cache=True)

    print(f"vocab actual size: {m.tokenizer.vocab_size_actual}")
    print(f"candidates (Open Mode full vocab): {len(candidates)}")
    print(f"cache built for {len(ivm_cached._cache_candidates)} candidates, "
          f"{len(ivm_cached._token_cache)} nonzero token rows "
          f"({sum(len(r) for r in ivm_cached._token_cache.values())} (t,c) entries stored)")

    random.seed(0)
    steps = []
    for _ in range(n_steps):
        ctx = set(random.sample(candidates, k=min(6, len(candidates))))
        current = random.choice(list(ctx))
        previous = random.choice(list(ctx))
        steps.append((ctx, current, previous))

    # correctness first -- never trust a speed number without this
    mismatches = 0
    for ctx, current, previous in steps:
        a = ivm_live.score_candidates(candidates, ctx, current=current, previous=previous)
        b = ivm_cached.score_candidates(candidates, ctx, current=current, previous=previous)
        for c in candidates:
            if abs(a["scores"][c] - b["scores"][c]) > 1e-9:
                mismatches += 1
    print(f"score mismatches between cached and live over {n_steps} steps: {mismatches}")

    t0 = time.perf_counter()
    for ctx, current, previous in steps:
        ivm_live.score_candidates(candidates, ctx, current=current, previous=previous)
    t_live = time.perf_counter() - t0

    t0 = time.perf_counter()
    for ctx, current, previous in steps:
        ivm_cached.score_candidates(candidates, ctx, current=current, previous=previous)
    t_cached = time.perf_counter() - t0

    print(f"\n{'':20s}{'total':>12s}{'per-step':>14s}")
    print(f"{'live (no cache)':20s}{t_live:12.4f}{1000*t_live/n_steps:11.4f} ms")
    print(f"{'cached':20s}{t_cached:12.4f}{1000*t_cached/n_steps:11.4f} ms")
    if t_cached > 0:
        print(f"\nspeedup: {t_live / t_cached:.2f}x")

    # build cost -- the one-time price the cache has to earn back
    t0 = time.perf_counter()
    ivm_build_only = ImportanceVoteMatrix.build(m, mode="strict", use_cache=True)
    t_build = time.perf_counter() - t0
    print(f"cache build time (one-time, amortized over all future steps): {t_build*1000:.3f} ms")
    if t_live > t_cached:
        breakeven_steps = t_build / ((t_live - t_cached) / n_steps)
        print(f"break-even after ~{breakeven_steps:.1f} steps "
              f"(fewer steps than that and the cache hasn't paid for itself yet)")


if __name__ == "__main__":
    vocab_size = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    run(vocab_size, n_steps)
