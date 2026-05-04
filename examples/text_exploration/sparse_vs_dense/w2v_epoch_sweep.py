#!/usr/bin/env python3
"""ARB-139: word2vec epoch sweep on 5M text8.

Question: does w2v's SimLex peak then degrade with continued training?
If so, where is the peak?

Trains w2v incrementally (one epoch at a time on the same corpus) and
records SimLex Spearman after each epoch. Outputs a CSV of
(epoch, simlex_spearman, n_pairs, elapsed_s) for plotting/inspection.

Matched preprocessing to continual_learning_v2.py (chunked +
subsampling=1e-3, no shuffle, vocab=10k). Single-corpus only — this is
NOT a continual-learning experiment, just a "how does w2v evolve with
training?" probe.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from scipy.stats import spearmanr

from examples.text_exploration.sparse_vs_dense.data import (
    build_vocab,
    load_simlex,
    load_text8,
)
from examples.text_exploration.sparse_vs_dense.evaluation import cosine_similarity
from examples.text_exploration.sparse_vs_dense.prepare_corpus import (
    CorpusPlan,
    prepare_corpus,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-tokens", type=int, default=5_000_000)
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="data/runs/arb139/w2v_epoch_sweep")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"=== w2v epoch sweep: {args.n_tokens:,} tokens, {args.max_epochs} epochs ==="
    )
    print(f"Loading text8 ({args.n_tokens:,}) ...")
    raw = load_text8(max_tokens=args.n_tokens)

    print("Applying matched preprocessing (chunking + subsampling=1e-3) ...")
    plan = CorpusPlan(
        chunk_size=1000,
        subsample_threshold=1e-3,
        min_count=5,
        shuffle_chunks=False,
        seed=args.seed,
    )
    prep = prepare_corpus(raw, plan=plan)
    chunks = prep.chunks
    tokens = prep.flat_tokens
    print(f"  prepared: {len(chunks)} chunks, {len(tokens):,} tokens after subsample")

    _token_to_id, id_to_token = build_vocab(tokens, vocab_size=10_000)
    vocab_set = set(id_to_token)
    pairs = load_simlex(vocab=vocab_set)
    print(f"Vocab: {len(id_to_token)}, SimLex pairs in vocab: {len(pairs)}")

    from gensim.models import Word2Vec

    print("Initializing w2v (build_vocab only, no training yet) ...")
    model = Word2Vec(
        sentences=None,
        vector_size=100,
        window=5,
        min_count=1,
        epochs=1,
        workers=4,
        sg=1,
        seed=args.seed,
        sample=0,
    )
    model.build_vocab(chunks)

    def eval_simlex() -> tuple[float, int]:
        pred, human = [], []
        for a, b, score in pairs:
            if a not in model.wv.key_to_index or b not in model.wv.key_to_index:
                continue
            pred.append(cosine_similarity(model.wv[a], model.wv[b]))
            human.append(score)
        if len(pred) < 2:
            return float("nan"), len(pred)
        rho, _ = spearmanr(pred, human)
        return float(rho), len(pred)

    rho0, n0 = eval_simlex()
    print(f"epoch=0 (random init): simlex={rho0:+.4f} n={n0}")
    rows: list[dict] = [
        {"epoch": 0, "simlex_spearman": rho0, "n_pairs": n0, "elapsed_s": 0.0}
    ]

    cum_time = 0.0
    for ep in range(1, args.max_epochs + 1):
        t0 = time.monotonic()
        model.train(
            corpus_iterable=chunks,
            total_examples=len(chunks),
            epochs=1,
        )
        cum_time += time.monotonic() - t0
        rho, n = eval_simlex()
        rows.append(
            {
                "epoch": ep,
                "simlex_spearman": rho,
                "n_pairs": n,
                "elapsed_s": cum_time,
            }
        )
        print(f"epoch={ep:>3d}: simlex={rho:+.4f} n={n}  cum_s={cum_time:.1f}")

    csv_path = out_dir / "results.csv"
    with csv_path.open("w") as f:
        w = csv.DictWriter(
            f, fieldnames=["epoch", "simlex_spearman", "n_pairs", "elapsed_s"]
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {csv_path}")

    # Quick summary: peak epoch, peak SimLex, value at max_epochs.
    valid = [r for r in rows if r["simlex_spearman"] == r["simlex_spearman"]]
    if valid:
        peak = max(valid, key=lambda r: r["simlex_spearman"])
        last = valid[-1]
        print(
            f"\nSummary: peak={peak['simlex_spearman']:+.4f} at epoch={peak['epoch']}; "
            f"last={last['simlex_spearman']:+.4f} at epoch={last['epoch']}"
        )


if __name__ == "__main__":
    main()
