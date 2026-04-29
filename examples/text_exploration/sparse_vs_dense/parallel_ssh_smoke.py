#!/usr/bin/env python3
"""ARB-139: smoke test for Hogwild!-parallel SSH.

Compares sequential (n_workers=1) and parallel (n_workers=4) runs on a
small corpus. Outputs:
  - elapsed_train_s for each
  - SimLex Spearman for each
  - speedup ratio
  - active-bits-per-word distribution match (sanity check)

Race conditions in parallel mode mean exact bit-equality is impossible.
We expect SimLex to be within ~0.05 absolute (typical SSH variance
across seeds at this corpus size).
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.stats import spearmanr

from examples.text_exploration.sparse_vs_dense.data import (
    build_vocab,
    encode_tokens,
    load_simlex,
    load_text8,
)
from examples.text_exploration.sparse_vs_dense.evaluation import jaccard_similarity
from examples.text_exploration.sparse_vs_dense.prepare_corpus import (
    CorpusPlan,
    prepare_corpus,
)
from examples.text_exploration.sparse_vs_dense.sparse_skipgram_hebbian_modulated_baseline import (
    train_sparse_skipgram_hebbian_modulated,
)


def eval_simlex(emb, pairs):
    pred, human = [], []
    for a, b, score in pairs:
        va = emb.get(a)
        vb = emb.get(b)
        if va is None or vb is None:
            continue
        pred.append(jaccard_similarity(va, vb))
        human.append(score)
    if len(pred) < 2:
        return float("nan"), len(pred)
    rho, _ = spearmanr(pred, human)
    return float(rho), len(pred)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-tokens", type=int, default=200_000)
    p.add_argument("--n-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print(f"=== parallel SSH smoke test: n_tokens={args.n_tokens:,}, "
          f"n_workers={args.n_workers} ===")
    raw = load_text8(max_tokens=args.n_tokens)
    plan = CorpusPlan(
        chunk_size=1000,
        subsample_threshold=1e-3,
        min_count=5,
        shuffle_chunks=False,
        seed=args.seed,
    )
    prep = prepare_corpus(raw, plan=plan)
    tokens = prep.flat_tokens

    token_to_id, id_to_token = build_vocab(tokens, vocab_size=10_000)
    tids = encode_tokens(tokens, token_to_id)
    pairs = load_simlex(vocab=set(id_to_token))
    print(f"Vocab: {len(id_to_token)}, prepared tokens: {len(tids):,}, "
          f"SimLex pairs: {len(pairs)}")

    base_kwargs = dict(
        id_to_token=id_to_token,
        n_dims=1024,
        k_active=40,
        window=5,
        n_neg=5,
        lr_pos=0.05,
        lr_neg=0.05,
        modulate=True,
        single_table=True,
        sigmoid_bounded=True,
        seed=args.seed,
    )

    print("\n--- sequential (n_workers=1) ---")
    t0 = time.monotonic()
    emb_s, stats_s = train_sparse_skipgram_hebbian_modulated(
        tids, n_workers=1, **base_kwargs,
    )
    seq_time = time.monotonic() - t0
    rho_s, n_s = eval_simlex(emb_s, pairs)
    active_s = np.mean([int(v.sum()) for v in emb_s._sdrs.values()])
    print(f"  elapsed: {seq_time:.1f}s")
    print(f"  SimLex: {rho_s:+.4f} (n={n_s})")
    print(f"  mean active bits/word: {active_s:.1f}")

    print(f"\n--- parallel (n_workers={args.n_workers}) ---")
    t0 = time.monotonic()
    emb_p, stats_p = train_sparse_skipgram_hebbian_modulated(
        tids, n_workers=args.n_workers, **base_kwargs,
    )
    par_time = time.monotonic() - t0
    rho_p, n_p = eval_simlex(emb_p, pairs)
    active_p = np.mean([int(v.sum()) for v in emb_p._sdrs.values()])
    print(f"  elapsed: {par_time:.1f}s")
    print(f"  SimLex: {rho_p:+.4f} (n={n_p})")
    print(f"  mean active bits/word: {active_p:.1f}")

    speedup = seq_time / par_time if par_time > 0 else 0.0
    delta_simlex = abs(rho_p - rho_s)
    print(
        f"\nspeedup: {speedup:.2f}x  |  ΔSimLex: {delta_simlex:+.4f}  |  "
        f"Δactive_bits: {active_p - active_s:+.2f}"
    )
    print(f"parallel_mode: {stats_p['parallel_mode']}")

    if delta_simlex > 0.10:
        print(
            "WARNING: parallel SimLex diverges by >0.10 from sequential — "
            "investigate before relying on parallel results."
        )
    if speedup < 1.5:
        print(
            f"WARNING: speedup only {speedup:.2f}x with {args.n_workers} workers; "
            "expected ~3-4x. Possible JIT warmup or contention issue."
        )


if __name__ == "__main__":
    main()
