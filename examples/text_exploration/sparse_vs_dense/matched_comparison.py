#!/usr/bin/env python3
"""ARB-139: matched-conditions head-to-head between word2vec and SSH.

Both models train on the *same* preprocessed corpus from `prepare_corpus`
(chunked + subsampled + min_count + shuffled, gensim-style defaults),
1 epoch each. Same vocab, same window, same n_neg, same seed.

For SSH: report both Jaccard-on-binary and cosine-on-raw-accumulator
SimLex (dual eval). For word2vec: report cosine SimLex (standard).

Output: data/runs/arb139/matched_comparison.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from examples.text_exploration.sparse_vs_dense.data import (
    load_analogy,
    load_simlex,
    load_text8,
)
from examples.text_exploration.sparse_vs_dense.evaluation import (
    evaluate_analogy,
    evaluate_simlex,
)
from examples.text_exploration.sparse_vs_dense.prepare_corpus import (
    CorpusPlan,
    prepare_corpus,
)
from examples.text_exploration.sparse_vs_dense.sparse_skipgram_hebbian_modulated_baseline import (
    train_sparse_skipgram_hebbian_modulated,
)
from examples.text_exploration.sparse_vs_dense.word2vec_baseline import (
    Word2VecEmbeddings,
)


def train_w2v_matched(
    chunks: list[list[str]],
    *,
    vocab: list[str],
    vector_size: int,
    window: int,
    n_neg: int,
    seed: int,
) -> tuple[Word2VecEmbeddings, dict]:
    """Train gensim Skip-gram on `chunks` for 1 epoch.

    `chunks` is a list of list-of-strings; gensim treats each as a
    sentence. We pass `min_count=1` because `prepare_corpus` already
    applied the min_count filter — we don't want gensim to filter
    twice (would create vocab mismatches with SSH).
    """
    from gensim.models import Word2Vec

    t0 = time.monotonic()
    model = Word2Vec(
        sentences=chunks,
        vector_size=vector_size,
        window=window,
        min_count=1,
        epochs=1,
        workers=1,  # match SSH's single-threaded determinism
        sg=1,
        negative=n_neg,
        sample=0,  # subsampling already done by prepare_corpus
        seed=seed,
    )
    elapsed = time.monotonic() - t0

    # Restrict to shared vocab so both models report on the same words.
    from gensim.models.keyedvectors import KeyedVectors
    present = [w for w in vocab if w in model.wv.key_to_index]
    kv = KeyedVectors(vector_size=vector_size)
    kv.add_vectors(present, [model.wv[w] for w in present])
    return Word2VecEmbeddings(kv), {
        "elapsed_s": elapsed,
        "vocab_size": len(present),
        "vector_size": vector_size,
        "epochs": 1,
    }


def run_one(*, n_tokens: int, vocab_size_hint: int, seed: int) -> list[dict]:
    print(f"\n=== matched comparison @ {n_tokens:,} tokens, seed={seed} ===")

    raw = load_text8(max_tokens=n_tokens)
    plan = CorpusPlan(
        chunk_size=1000,
        subsample_threshold=1e-3,
        min_count=5,
        shuffle_chunks=True,
        seed=seed,
    )
    prep = prepare_corpus(raw, plan=plan)
    print(
        f"  vocab={prep.stats['vocab_size']:,}  "
        f"kept={prep.stats['n_kept_tokens']:,}/{prep.stats['n_raw_tokens']:,} "
        f"({prep.stats['fraction_kept']:.0%})  "
        f"chunks={prep.stats['n_chunks']} (~{prep.stats['mean_chunk_size']:.0f} tok)"
    )

    vocab_set = set(prep.vocab)
    simlex = load_simlex(vocab=vocab_set)
    analogy = load_analogy(vocab=vocab_set)
    print(f"  simlex pairs (filtered): {len(simlex)},  analogy entries: {len(analogy)}")

    rows: list[dict] = []

    # ---- word2vec ----
    print("  --- word2vec ---")
    t0 = time.monotonic()
    w2v_emb, w2v_stats = train_w2v_matched(
        prep.chunks,
        vocab=prep.vocab,
        vector_size=100,
        window=5,
        n_neg=5,
        seed=seed,
    )
    s_w2v = evaluate_simlex(w2v_emb, simlex)
    a_w2v = evaluate_analogy(w2v_emb, analogy)
    rows.append({
        "model": "word2vec",
        "n_tokens": n_tokens,
        "seed": seed,
        "eval_method": "cosine_dense",
        "simlex_spearman": s_w2v["spearman"],
        "simlex_n": s_w2v["n_pairs"],
        "analogy_top1": a_w2v["top1"],
        "analogy_n": a_w2v["n_entries"],
        "train_s": time.monotonic() - t0,
    })
    print(
        f"    simlex(cosine)={s_w2v['spearman']:+.3f}  "
        f"analogy={a_w2v['top1']:.3f}  "
        f"({rows[-1]['train_s']:.1f}s)"
    )

    # ---- SSH (sigmoid-bounded, single-table, modulated) ----
    print("  --- SSH ---")
    t0 = time.monotonic()
    ssh_emb, ssh_stats = train_sparse_skipgram_hebbian_modulated(
        prep.flat_token_ids,
        id_to_token=prep.vocab,
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
    train_dt = time.monotonic() - t0
    # Dual eval
    s_binary = evaluate_simlex(ssh_emb, simlex)
    s_continuous = evaluate_simlex(ssh_emb.continuous, simlex)
    a_ssh = evaluate_analogy(ssh_emb, analogy)
    rows.append({
        "model": "ssh_sigmoid",
        "n_tokens": n_tokens,
        "seed": seed,
        "eval_method": "jaccard_binary",
        "simlex_spearman": s_binary["spearman"],
        "simlex_n": s_binary["n_pairs"],
        "analogy_top1": a_ssh["top1"],
        "analogy_n": a_ssh["n_entries"],
        "train_s": train_dt,
    })
    rows.append({
        "model": "ssh_sigmoid",
        "n_tokens": n_tokens,
        "seed": seed,
        "eval_method": "cosine_continuous",
        "simlex_spearman": s_continuous["spearman"],
        "simlex_n": s_continuous["n_pairs"],
        "analogy_top1": float("nan"),
        "analogy_n": 0,
        "train_s": train_dt,
    })
    print(
        f"    simlex(jaccard)={s_binary['spearman']:+.3f}  "
        f"simlex(cosine)={s_continuous['spearman']:+.3f}  "
        f"analogy={a_ssh['top1']:.3f}  ({train_dt:.1f}s)"
    )

    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--vocab-size", type=int, default=5000)  # not used directly but kept for parity
    p.add_argument(
        "--n-tokens",
        type=int,
        nargs="+",
        default=[100_000, 500_000, 1_000_000, 5_000_000],
    )
    p.add_argument(
        "--csv",
        type=str,
        default="data/runs/arb139/matched_comparison.csv",
    )
    args = p.parse_args()

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    existing: list[dict] = []
    done_keys: set[tuple[str, int, str]] = set()
    if csv_path.exists():
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                existing.append(row)
                done_keys.add((row["model"], int(row["n_tokens"]), row["eval_method"]))

    all_rows: list[dict] = list(existing)
    for n in args.n_tokens:
        # Skip if all expected rows for this n_tokens are already in CSV.
        expected = {
            ("word2vec", n, "cosine_dense"),
            ("ssh_sigmoid", n, "jaccard_binary"),
            ("ssh_sigmoid", n, "cosine_continuous"),
        }
        if expected.issubset(done_keys):
            print(f"--- skipping n_tokens={n:,} (all eval rows present) ---")
            continue
        new_rows = run_one(
            n_tokens=n,
            vocab_size_hint=args.vocab_size,
            seed=args.seed,
        )
        all_rows.extend(new_rows)
        keys = sorted({k for r in all_rows for k in r})
        with csv_path.open("w") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in all_rows:
                w.writerow(r)
        print(f"  [partial] wrote {csv_path} ({len(all_rows)} rows)")

    print(f"\nDone. Wrote {csv_path}.")


if __name__ == "__main__":
    main()
