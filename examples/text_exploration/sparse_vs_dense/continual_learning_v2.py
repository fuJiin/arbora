#!/usr/bin/env python3
"""ARB-139: continual learning experiment v2.

Matched preprocessing for both word2vec and SSH (modulated, single-table,
sigmoid-bounded). Two phases:

  Phase 1: train on corpus A → snapshot embeddings + state
  Phase 2: continue training on corpus B (no rehearsal) → snapshot

Default A = first 5M tokens of text8, B = next 5M tokens. Vocab built
from A∪B upfront; both methods see identical vocab.

Metrics:
  - SimLex retention: SimLex(A-only word pairs) at end of phase 2
    vs end of phase 1
  - SimLex acquisition: SimLex(B-only pairs) at end of phase 2
  - Per-word drift: similarity(E_w_phase1, E_w_phase2) for every word
    in vocab; histogram split by which phases each word appeared in

Output:
  - data/runs/arb139/continual/results.csv (summary)
  - data/runs/arb139/continual/per_word_drift.csv (per-word)
  - data/runs/arb139/continual/embeddings_*.pkl (snapshots)
"""

from __future__ import annotations

import argparse
import csv
import pickle
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
    cosine_similarity,
    evaluate_simlex,
    jaccard_similarity,
)
from examples.text_exploration.sparse_vs_dense.prepare_corpus import (
    CorpusPlan,
    prepare_corpus,
)
from examples.text_exploration.sparse_vs_dense.sparse_skipgram_hebbian_modulated_baseline import (
    train_sparse_skipgram_hebbian_modulated,
)


def split_text8(n_per_phase: int) -> tuple[list[str], list[str]]:
    """Load text8 and split into phase-A and phase-B token lists."""
    tokens = load_text8(max_tokens=2 * n_per_phase)
    return tokens[:n_per_phase], tokens[n_per_phase:]


def words_in_corpus(tokens: list[str], vocab_set: set[str]) -> set[str]:
    """Set of vocab words that appear at least once in `tokens`."""
    return set(tokens) & vocab_set


def split_simlex_pairs(
    pairs: list[tuple[str, str, float]],
    a_words: set[str],
    b_words: set[str],
) -> dict[str, list[tuple[str, str, float]]]:
    """Partition SimLex pairs by where their words appeared during training.

    a_only:  both words appeared only in A (test of retention)
    b_only:  both words appeared only in B (test of acquisition)
    shared:  both words appeared in both
    cross:   one word in A only, the other in B only
    """
    out = {"a_only": [], "b_only": [], "shared": [], "cross": []}
    for pair in pairs:
        a, b, _ = pair
        a_in_a = a in a_words
        a_in_b = a in b_words
        b_in_a = b in a_words
        b_in_b = b in b_words
        if a_in_a and b_in_a and not a_in_b and not b_in_b:
            out["a_only"].append(pair)
        elif a_in_b and b_in_b and not a_in_a and not b_in_a:
            out["b_only"].append(pair)
        elif a_in_a and a_in_b and b_in_a and b_in_b:
            out["shared"].append(pair)
        else:
            out["cross"].append(pair)
    return out


def per_word_drift(
    emb_phase1: dict[str, np.ndarray],
    emb_phase2: dict[str, np.ndarray],
    is_sparse: bool,
) -> dict[str, float]:
    """Similarity between same word's vector before and after phase 2.

    For sparse (binary): Jaccard.
    For dense (continuous): cosine.
    """
    sim_fn = jaccard_similarity if is_sparse else cosine_similarity
    out = {}
    common = set(emb_phase1.keys()) & set(emb_phase2.keys())
    for w in common:
        out[w] = sim_fn(emb_phase1[w], emb_phase2[w])
    return out


def train_w2v_continual(
    chunks_a: list[list[str]],
    chunks_b: list[list[str]],
    seed: int,
):
    """Phase 1 train on A's chunks; build joint vocab from A∪B; phase 2 continue on B."""
    from gensim.models import Word2Vec

    print("\n  word2vec phase 1: training on A ...")
    t0 = time.monotonic()
    model = Word2Vec(
        sentences=chunks_a,
        vector_size=100,
        window=5,
        min_count=1,  # vocab is pre-determined; don't drop
        epochs=1,
        workers=4,
        sg=1,
        seed=seed,
        sample=0,  # subsampling already done by prepare_corpus
    )
    # Add B's vocab without retraining yet (so any B-only words exist as
    # randomly-initialized vectors at end of phase 1).
    model.build_vocab(chunks_b, update=True)
    elapsed_a = time.monotonic() - t0
    print(f"    phase 1 trained in {elapsed_a:.1f}s")

    snap1 = {w: model.wv[w].copy() for w in model.wv.key_to_index}

    print("  word2vec phase 2: continuing on B ...")
    t0 = time.monotonic()
    model.train(
        corpus_iterable=chunks_b,
        total_examples=len(chunks_b),
        epochs=1,
    )
    elapsed_b = time.monotonic() - t0
    print(f"    phase 2 trained in {elapsed_b:.1f}s")

    snap2 = {w: model.wv[w].copy() for w in model.wv.key_to_index}
    return snap1, snap2, {"phase1_s": elapsed_a, "phase2_s": elapsed_b}


def train_ssh_continual(
    token_ids_a: list[int],
    token_ids_b: list[int],
    id_to_token: list[str],
    seed: int,
):
    """Phase 1 train on A, phase 2 continue with same accumulator on B."""
    print("\n  SSH phase 1: training on A ...")
    t0 = time.monotonic()
    emb1, stats1 = train_sparse_skipgram_hebbian_modulated(
        token_ids_a,
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
        seed=seed,
    )
    elapsed_a = time.monotonic() - t0
    print(f"    phase 1 trained in {elapsed_a:.1f}s")

    snap1 = {w: emb1.get(w).copy() for w in emb1.vocab() if emb1.get(w) is not None}

    print("  SSH phase 2: continuing on B ...")
    t0 = time.monotonic()
    emb2, stats2 = train_sparse_skipgram_hebbian_modulated(
        token_ids_b,
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
        seed=seed + 1,  # different seed for negative-sample randomness in phase 2
        initial_A_center=emb1.A_center,
    )
    elapsed_b = time.monotonic() - t0
    print(f"    phase 2 trained in {elapsed_b:.1f}s")

    snap2 = {w: emb2.get(w).copy() for w in emb2.vocab() if emb2.get(w) is not None}
    return snap1, snap2, {"phase1_s": elapsed_a, "phase2_s": elapsed_b}


def evaluate_simlex_partition(
    snap: dict[str, np.ndarray],
    pairs: list[tuple[str, str, float]],
    is_sparse: bool,
) -> tuple[float, int]:
    """Compute SimLex Spearman on the given partition's pairs."""
    from scipy.stats import spearmanr

    sim_fn = jaccard_similarity if is_sparse else cosine_similarity
    pred, human = [], []
    for a, b, score in pairs:
        if a not in snap or b not in snap:
            continue
        pred.append(sim_fn(snap[a], snap[b]))
        human.append(score)
    if len(pred) < 2:
        return float("nan"), len(pred)
    rho, _ = spearmanr(pred, human)
    return float(rho), len(pred)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-per-phase", type=int, default=5_000_000)
    p.add_argument("--vocab-size-hint", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out-dir",
        type=str,
        default="data/runs/arb139/continual",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== continual learning at {args.n_per_phase:,} tokens per phase ===")
    print(f"Loading text8 and splitting ...")
    tokens_a_raw, tokens_b_raw = split_text8(args.n_per_phase)

    # Apply matched preprocessing to BOTH halves.
    print("Applying matched preprocessing (chunking + subsampling) ...")
    plan = CorpusPlan(
        chunk_size=1000,
        subsample_threshold=1e-3,
        min_count=5,
        shuffle_chunks=False,  # No shuffle — order matters for continual learning
        seed=args.seed,
    )
    prep_a = prepare_corpus(tokens_a_raw, plan=plan)
    prep_b = prepare_corpus(tokens_b_raw, plan=plan)
    tokens_a = prep_a.flat_tokens
    tokens_b = prep_b.flat_tokens
    chunks_a = prep_a.chunks
    chunks_b = prep_b.chunks

    # Build joint vocab from A ∪ B for both methods.
    combined_tokens = tokens_a + tokens_b
    token_to_id, id_to_token = build_vocab(combined_tokens, vocab_size=args.vocab_size_hint)
    vocab_set = set(id_to_token)
    print(f"Joint vocab size: {len(id_to_token)}")

    # Encode token streams.
    tids_a = encode_tokens(tokens_a, token_to_id)
    tids_b = encode_tokens(tokens_b, token_to_id)

    # Determine which words appeared in each phase.
    a_words = words_in_corpus(tokens_a, vocab_set)
    b_words = words_in_corpus(tokens_b, vocab_set)
    a_only_words = a_words - b_words
    b_only_words = b_words - a_words
    shared_words = a_words & b_words
    print(
        f"a_only: {len(a_only_words)}  b_only: {len(b_only_words)}  "
        f"shared: {len(shared_words)}"
    )

    # Load + partition SimLex pairs.
    pairs = load_simlex(vocab=vocab_set)
    parts = split_simlex_pairs(pairs, a_words, b_words)
    print(
        f"SimLex partitions: a_only={len(parts['a_only'])} "
        f"b_only={len(parts['b_only'])} shared={len(parts['shared'])} "
        f"cross={len(parts['cross'])} total_in_vocab={len(pairs)}"
    )

    rows: list[dict] = []
    drift_rows: list[dict] = []

    for method_name in ["word2vec", "ssh"]:
        print(f"\n--- {method_name} ---")
        if method_name == "word2vec":
            snap1, snap2, timings = train_w2v_continual(
                chunks_a, chunks_b, args.seed,
            )
            is_sparse = False
        else:
            snap1, snap2, timings = train_ssh_continual(
                tids_a, tids_b, id_to_token, args.seed,
            )
            is_sparse = True

        # Save snapshots.
        with (out_dir / f"snap1_{method_name}.pkl").open("wb") as f:
            pickle.dump(snap1, f)
        with (out_dir / f"snap2_{method_name}.pkl").open("wb") as f:
            pickle.dump(snap2, f)

        # Eval on each partition at each phase.
        for phase, snap in [("phase1", snap1), ("phase2", snap2)]:
            for part_name, part_pairs in parts.items():
                if not part_pairs:
                    continue
                rho, n = evaluate_simlex_partition(snap, part_pairs, is_sparse)
                rows.append(
                    {
                        "method": method_name,
                        "phase": phase,
                        "partition": part_name,
                        "simlex_spearman": rho,
                        "n_pairs": n,
                        "phase1_s": timings["phase1_s"],
                        "phase2_s": timings["phase2_s"],
                    }
                )
                print(
                    f"    {phase:>6s} | {part_name:>8s} | "
                    f"simlex={rho:+.4f}  n={n}"
                )

        # Per-word drift between phase 1 and phase 2.
        drift = per_word_drift(snap1, snap2, is_sparse)
        for w, sim in drift.items():
            tag = (
                "a_only"
                if w in a_only_words
                else "b_only"
                if w in b_only_words
                else "shared"
                if w in shared_words
                else "neither"
            )
            drift_rows.append(
                {"method": method_name, "word": w, "tag": tag, "phase_similarity": sim}
            )

        # Drift histogram summary.
        sims_a_only = [d["phase_similarity"] for d in drift_rows
                       if d["method"] == method_name and d["tag"] == "a_only"]
        sims_shared = [d["phase_similarity"] for d in drift_rows
                       if d["method"] == method_name and d["tag"] == "shared"]
        if sims_a_only:
            print(
                f"    drift on A-only words (n={len(sims_a_only)}): "
                f"mean={np.mean(sims_a_only):.3f}  "
                f"p10={np.percentile(sims_a_only, 10):.3f}  "
                f"p50={np.percentile(sims_a_only, 50):.3f}"
            )
        if sims_shared:
            print(
                f"    drift on shared words (n={len(sims_shared)}): "
                f"mean={np.mean(sims_shared):.3f}  "
                f"p10={np.percentile(sims_shared, 10):.3f}  "
                f"p50={np.percentile(sims_shared, 50):.3f}"
            )

    # Save CSVs.
    summary_csv = out_dir / "results.csv"
    keys = sorted({k for r in rows for k in r})
    with summary_csv.open("w") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {summary_csv}")

    drift_csv = out_dir / "per_word_drift.csv"
    keys = sorted({k for r in drift_rows for k in r})
    with drift_csv.open("w") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in drift_rows:
            w.writerow(r)
    print(f"Wrote {drift_csv}")


if __name__ == "__main__":
    main()
