#!/usr/bin/env python3
"""Continual-learning benchmark for ARB-139 (comparison #4).

The canonical "catastrophic forgetting" probe: train on data A, evaluate
on A; then continue training on *different* data B (disjoint vocab/topic
where possible); re-evaluate on A. How much did A's performance degrade?

Protocol:
    1. Split text8 into half-A (first half) and half-B (second half).
    2. Train word2vec on A → measure SimLex-999 + capacity metrics
       restricted to words that appear in A.
    3. Continue training the *same* word2vec model on B → re-measure
       on the same A-only pairs.
    4. Mirror for T1 (word-level): train on A, re-measure, continue on B.
    5. Report: baseline_A, post_B_on_A, delta (negative = forgetting).

For both architectures we run from the same token IDs built once at
the top. Continuation is:
    - word2vec: `model.train(sentences_B, total_examples=..., epochs=1)`
    - T1:       keep the trained `region` alive and stream B's IDs through
      `region.process(...)` for another epoch.

This script does NOT try to prevent forgetting — it measures how much
happens. Sparse binary + local Hebbian is hypothesized to resist
catastrophic forgetting because representations are orthogonal and
updates are local; dense gradient-trained embeddings should drift more.

Usage:
    uv run python -m examples.text_exploration.sparse_vs_dense.continual_learning
    uv run python -m examples.text_exploration.sparse_vs_dense.continual_learning \\
        --max-tokens 2000000 --vocab-size 5000 --csv /tmp/continual.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from examples.text_exploration.sparse_vs_dense.data import (
    build_vocab,
    encode_tokens,
    load_simlex,
    load_text8,
)
from examples.text_exploration.sparse_vs_dense.evaluation import (
    evaluate_capacity,
    evaluate_simlex,
)
from examples.text_exploration.sparse_vs_dense.t1_word import (
    T1WordEmbeddings,
    _OneHotIDEncoder,
    build_t1_for_words,
)
from examples.text_exploration.sparse_vs_dense.word2vec_baseline import (
    Word2VecEmbeddings,
)


def _chunk_sentences(tokens: list[str], sent_len: int = 1000) -> list[list[str]]:
    return [tokens[i : i + sent_len] for i in range(0, len(tokens), sent_len)]


def _extract_t1_embeddings(region, id_to_token: list[str]) -> T1WordEmbeddings:
    """Context-free: reset, present each vocab word once, capture L2/3."""
    encoder = _OneHotIDEncoder(vocab_size=len(id_to_token))
    was_learning = region.learning_enabled
    region.learning_enabled = False
    sdrs: dict[str, np.ndarray] = {}
    for wid, w in enumerate(id_to_token):
        region.reset_working_memory()
        region.process(encoder.encode(wid))
        sdrs[w] = region.l23.active.copy()
    region.learning_enabled = was_learning
    return T1WordEmbeddings(sdrs)


def _wrap_w2v(model) -> Word2VecEmbeddings:
    return Word2VecEmbeddings(model.wv)


def _simlex_on(emb, simlex_pairs: list[tuple[str, str, float]], *, label: str) -> dict:
    s = evaluate_simlex(emb, simlex_pairs)
    c = evaluate_capacity(emb, seed=0)
    print(
        f"    [{label}] simlex={s['spearman']:.3f} (n={s['n_pairs']})  "
        f"cap: mean_sim={c['mean_pairwise_sim']:.3f} "
        f"coll={c['high_collision_frac']:.2f} ed={c['eff_dim']:.1f}"
    )
    return {
        "label": label,
        "simlex_spearman": s["spearman"],
        "simlex_n": s["n_pairs"],
        "cap_mean_sim": c["mean_pairwise_sim"],
        "cap_collision_frac": c["high_collision_frac"],
        "cap_eff_dim": c["eff_dim"],
    }


def run_continual(
    *,
    max_tokens: int,
    vocab_size: int,
    seed: int,
    w2v_epochs: int,
    t1_epochs: int,
    t1_cols: int,
    t1_k: int,
) -> list[dict]:
    """Full train-A → eval-A → train-B → eval-A protocol for both models."""
    from gensim.models import Word2Vec

    tokens = load_text8(max_tokens=max_tokens)
    half = len(tokens) // 2
    tokens_a = tokens[:half]
    tokens_b = tokens[half:]

    # Shared vocab from the COMBINED corpus so that A-only evaluation
    # words aren't OOV after continuing on B (and vice versa).
    token_to_id, id_to_token = build_vocab(tokens, vocab_size=vocab_size)
    vocab_set = set(id_to_token)
    # Pairs whose words actually appeared in half-A (the "knowledge"
    # we're testing retention on).
    a_word_set = set(tokens_a) & vocab_set
    simlex_a = [
        p
        for p in load_simlex(vocab=vocab_set)
        if p[0] in a_word_set and p[1] in a_word_set
    ]
    print(
        f"\n=== Continual learning: {len(tokens):,} tokens "
        f"(A={len(tokens_a):,}, B={len(tokens_b):,}) ===\n"
        f"  vocab={len(id_to_token)}  simlex_A={len(simlex_a)}"
    )

    rows: list[dict] = []

    # ------- word2vec -------
    print("\n  word2vec: training on A ...")
    t0 = time.monotonic()
    sentences_a = _chunk_sentences(tokens_a)
    model = Word2Vec(
        sentences=sentences_a,
        vector_size=100,
        window=5,
        min_count=5,
        epochs=w2v_epochs,
        workers=4,
        sg=1,
        seed=seed,
    )
    print(f"    train-A done ({time.monotonic() - t0:.1f}s)")
    after_a = _simlex_on(_wrap_w2v(model), simlex_a, label="w2v post-A")

    print("  word2vec: continuing on B ...")
    t0 = time.monotonic()
    sentences_b = _chunk_sentences(tokens_b)
    model.build_vocab(sentences_b, update=True)
    model.train(
        sentences_b,
        total_examples=len(sentences_b),
        epochs=w2v_epochs,
    )
    print(f"    train-B done ({time.monotonic() - t0:.1f}s)")
    after_b = _simlex_on(_wrap_w2v(model), simlex_a, label="w2v post-B (eval on A)")

    rows.append(
        {
            "model": "word2vec",
            "seed": seed,
            "n_tokens_a": len(tokens_a),
            "n_tokens_b": len(tokens_b),
            "simlex_a_pairs": len(simlex_a),
            "post_A_simlex": after_a["simlex_spearman"],
            "post_B_simlex": after_b["simlex_spearman"],
            "delta_simlex": after_b["simlex_spearman"] - after_a["simlex_spearman"],
            "post_A_eff_dim": after_a["cap_eff_dim"],
            "post_B_eff_dim": after_b["cap_eff_dim"],
            "post_A_collision": after_a["cap_collision_frac"],
            "post_B_collision": after_b["cap_collision_frac"],
        }
    )

    # ------- T1 sparse -------
    print("\n  t1_sparse: training on A ...")
    t0 = time.monotonic()
    token_ids_a = encode_tokens(tokens_a, token_to_id)
    token_ids_b = encode_tokens(tokens_b, token_to_id)
    encoder = _OneHotIDEncoder(vocab_size=len(id_to_token))
    region = build_t1_for_words(
        vocab_size=len(id_to_token),
        n_columns=t1_cols,
        k_columns=t1_k,
        seed=seed,
    )
    for _ in range(t1_epochs):
        for tid in token_ids_a:
            region.process(encoder.encode(tid))
    print(f"    train-A done ({time.monotonic() - t0:.1f}s)")
    t1_after_a = _simlex_on(
        _extract_t1_embeddings(region, id_to_token),
        simlex_a,
        label="t1 post-A",
    )

    print("  t1_sparse: continuing on B ...")
    t0 = time.monotonic()
    for _ in range(t1_epochs):
        for tid in token_ids_b:
            region.process(encoder.encode(tid))
    print(f"    train-B done ({time.monotonic() - t0:.1f}s)")
    t1_after_b = _simlex_on(
        _extract_t1_embeddings(region, id_to_token),
        simlex_a,
        label="t1 post-B (eval on A)",
    )

    rows.append(
        {
            "model": "t1_sparse",
            "seed": seed,
            "n_tokens_a": len(tokens_a),
            "n_tokens_b": len(tokens_b),
            "simlex_a_pairs": len(simlex_a),
            "post_A_simlex": t1_after_a["simlex_spearman"],
            "post_B_simlex": t1_after_b["simlex_spearman"],
            "delta_simlex": (
                t1_after_b["simlex_spearman"] - t1_after_a["simlex_spearman"]
            ),
            "post_A_eff_dim": t1_after_a["cap_eff_dim"],
            "post_B_eff_dim": t1_after_b["cap_eff_dim"],
            "post_A_collision": t1_after_a["cap_collision_frac"],
            "post_B_collision": t1_after_b["cap_collision_frac"],
        }
    )

    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Continual-learning benchmark (ARB-139)")
    p.add_argument("--max-tokens", type=int, default=2_000_000)
    p.add_argument("--vocab-size", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--w2v-epochs", type=int, default=3)
    p.add_argument("--t1-epochs", type=int, default=1)
    p.add_argument("--t1-cols", type=int, default=128)
    p.add_argument("--t1-k", type=int, default=8)
    p.add_argument("--csv", type=str, default=None)
    args = p.parse_args()

    rows = run_continual(
        max_tokens=args.max_tokens,
        vocab_size=args.vocab_size,
        seed=args.seed,
        w2v_epochs=args.w2v_epochs,
        t1_epochs=args.t1_epochs,
        t1_cols=args.t1_cols,
        t1_k=args.t1_k,
    )

    print("\n=== Summary ===")
    for r in rows:
        sign = "+" if r["delta_simlex"] >= 0 else ""
        print(
            f"  {r['model']:>10s}  post_A={r['post_A_simlex']:.3f}  "
            f"post_B={r['post_B_simlex']:.3f}  "
            f"Δ={sign}{r['delta_simlex']:.3f}  "
            f"(on {r['simlex_a_pairs']} A-only pairs)"
        )

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted({k for r in rows for k in r})
        with out.open("w") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
