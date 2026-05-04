#!/usr/bin/env python3
"""ARB-139 plateau diagnostic: train SSH at multiple n_tokens, dump full
continuous accumulator state, plus postprocess.

Why:
SSH plateaus around SimLex 0.13 (jaccard) past 1M tokens while w2v keeps
climbing. Six possible mechanisms (saturation, lock-in, modulator collapse,
window ceiling, lateral motion, vocab cap). This script trains and dumps
the underlying continuous accumulator A_w so we can directly inspect:

  - Distribution of A_w[i] values across (word, bit) pairs
  - Saturation fraction (|sigmoid(A) - 0.5| > 0.45 ⇔ |A| > 2.2)
  - For SimLex pair words: A values at their top-k bits

Output:
  data/runs/arb139/plateau/accumulator_n{N}_seed{seed}.pkl
  data/runs/arb139/plateau/sdrs_n{N}_seed{seed}.pkl

Run: uv run python -m examples.text_exploration.sparse_vs_dense.plateau_dump
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

from examples.text_exploration.sparse_vs_dense.data import (
    build_vocab,
    encode_tokens,
    load_text8,
)
from examples.text_exploration.sparse_vs_dense.sparse_skipgram_hebbian_modulated_baseline import (
    train_sparse_skipgram_hebbian_modulated,
)


def dump_one(n_tokens: int, vocab_size: int, seed: int, out_dir: Path) -> None:
    sdr_path = out_dir / f"sdrs_n{n_tokens}_seed{seed}.pkl"
    acc_path = out_dir / f"accumulator_n{n_tokens}_seed{seed}.pkl"
    if acc_path.exists() and sdr_path.exists():
        print(f"--- skip n_tokens={n_tokens:,} (dumps present) ---")
        return

    print(f"--- training SSH @ n_tokens={n_tokens:,} seed={seed} ---")
    tokens = load_text8(max_tokens=n_tokens)
    token_to_id, id_to_token = build_vocab(tokens, vocab_size=vocab_size)
    token_ids = encode_tokens(tokens, token_to_id)

    t0 = time.monotonic()
    emb, _stats = train_sparse_skipgram_hebbian_modulated(
        token_ids,
        id_to_token=id_to_token,
        n_dims=1024,
        k_active=40,
        window=5,
        n_neg=5,
        lr_pos=0.05,
        lr_neg=0.05,
        modulate=True,
        decay=0.0,
        single_table=True,
        ema_alpha=0.0,
        sigmoid_bounded=True,
        seed=seed,
    )
    elapsed = time.monotonic() - t0

    sdrs = {w: emb.get(w) for w in emb.vocab() if emb.get(w) is not None}
    accumulator = {
        w: emb.continuous.get(w)
        for w in emb.continuous.vocab()
        if emb.continuous.get(w) is not None
    }

    with sdr_path.open("wb") as f:
        pickle.dump(sdrs, f)
    with acc_path.open("wb") as f:
        pickle.dump(accumulator, f)
    print(
        f"  trained in {elapsed:.1f}s, dumped sdrs ({sdr_path.stat().st_size // 1024} KB) "
        f"+ accumulator ({acc_path.stat().st_size // 1024} KB)"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vocab-size", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--n-tokens",
        type=int,
        nargs="+",
        default=[1_000_000, 5_000_000],
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="data/runs/arb139/plateau",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for n in args.n_tokens:
        dump_one(n, args.vocab_size, args.seed, out_dir)

    print(f"\nDone. Dumps in {out_dir}/")


if __name__ == "__main__":
    main()
