#!/usr/bin/env python3
"""Sweep encoding strategies to isolate prediction learning from encoding quality.

Tests three encoding approaches:
1. CharbitEncoder (current) — character-level, space-dominated
2. CharbitEncoder with space stripping — removes leading space noise
3. RandomEncoder (control) — unique sparse binary per token, tests pure prediction

Usage: uv run experiments/scripts/sweep_encoding.py [--tokens N]
"""

import argparse
import string
import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

import step.env  # noqa: F401
from step.config import CortexConfig
from step.cortex.sensory import SensoryRegion
from step.decoders import InvertedIndexDecoder
from step.encoders.charbit import CharbitEncoder
from step.probes.diagnostics import CortexDiagnostics
from step.runner import STORY_BOUNDARY

CHARS = string.printable
CHAR_LENGTH = 8
CHAR_WIDTH = len(CHARS) + 1


@dataclass
class RunMetrics:
    overlaps: list[float] = field(default_factory=list)
    accuracies: list[float] = field(default_factory=list)
    column_accuracies: list[float] = field(default_factory=list)
    elapsed_seconds: float = 0.0


def prepare_tokens(max_tokens: int):
    """Load TinyStories tokens."""
    print("Loading dataset and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    dataset = load_dataset("roneneldan/TinyStories", split="train")

    tokens: list[tuple[int, str]] = []
    t = 0
    first = True
    for ex in dataset:
        if not first:
            tokens.append((STORY_BOUNDARY, ""))
            t += 1
            if t >= max_tokens:
                break
        first = False
        for tid in tokenizer.encode(ex["text"]):
            tokens.append((tid, tokenizer.decode([tid])))
            t += 1
            if t >= max_tokens:
                break
        if t >= max_tokens:
            break

    unique = len({tid for tid, _ in tokens if tid != STORY_BOUNDARY})
    boundaries = sum(1 for tid, _ in tokens if tid == STORY_BOUNDARY)
    print(f"  {len(tokens):,} tokens, {unique} unique, {boundaries + 1} stories\n")
    return tokens


class RandomBinaryEncoder:
    """Deterministic random sparse binary encoder for cortex.

    Each token_id maps to a fixed sparse binary vector (k-of-n bits).
    Returns numpy arrays compatible with SensoryRegion.process().
    """

    def __init__(self, n: int, k: int):
        self.n = n
        self.k = k
        self._cache: dict[int, np.ndarray] = {}

    def encode(self, token_id: int) -> np.ndarray:
        if token_id not in self._cache:
            rng = np.random.default_rng(token_id + 12345)
            vec = np.zeros(self.n, dtype=np.bool_)
            indices = rng.choice(self.n, self.k, replace=False)
            vec[indices] = True
            self._cache[token_id] = vec
        return self._cache[token_id]


def run_experiment(
    name: str,
    tokens: list[tuple[int, str]],
    encode_fn,
    input_dim: int,
    encoding_width: int,
    log_interval: int,
    show_predictions: int,
):
    """Run cortex with a given encoding function."""
    cfg = CortexConfig()
    region = SensoryRegion(
        input_dim=input_dim,
        encoding_width=encoding_width,
        n_columns=cfg.n_columns,
        n_l4=cfg.n_l4,
        n_l23=cfg.n_l23,
        k_columns=cfg.k_columns,
        voltage_decay=cfg.voltage_decay,
        eligibility_decay=cfg.eligibility_decay,
        synapse_decay=cfg.synapse_decay,
        learning_rate=cfg.learning_rate,
        max_excitability=cfg.max_excitability,
        fb_boost=cfg.fb_boost,
        ltd_rate=cfg.ltd_rate,
        burst_learning_scale=cfg.burst_learning_scale,
        seed=cfg.seed,
    )

    n_params = region.ff_weights.size + region.l23_lateral_weights.size

    diag = CortexDiagnostics(snapshot_interval=log_interval)
    decode_index = InvertedIndexDecoder()
    # Column-level inverted index
    col_index: dict[int, list[int]] = {}
    token_ids: list[int] = []
    token_id_set: set[int] = set()

    metrics = RunMetrics()
    k = region.k_columns
    start = time.monotonic()

    # Track column set ambiguity
    colset_to_tokens: dict[frozenset, set[int]] = defaultdict(set)

    prediction_log: list[tuple[str, str, str]] = []

    print(f"--- {name} ---")
    print(
        f"  input_dim={input_dim} encoding_width={encoding_width} params={n_params:,}"
    )

    for t, (token_id, token_str) in enumerate(tokens):
        if token_id == STORY_BOUNDARY:
            region.reset_working_memory()
            continue

        # Prediction
        predicted_neurons = region.get_prediction(k)
        predicted_set = frozenset(int(i) for i in predicted_neurons)

        # Decode: inverted index
        idx_predicted = decode_index.decode(predicted_set)

        # Decode: column-level
        pred_cols = np.unique(predicted_neurons // region.n_l4)
        col_scores: dict[int, float] = {}
        for col in pred_cols:
            for idx in col_index.get(int(col), ()):
                col_scores[idx] = col_scores.get(idx, 0.0) + 1.0
        col_predicted = (
            token_ids[max(col_scores, key=col_scores.__getitem__)] if col_scores else -1
        )

        # Look up strings for display
        idx_str = ""
        col_str = ""
        for tid2, tstr2 in tokens:
            if tid2 == idx_predicted and not idx_str:
                idx_str = tstr2
            if tid2 == col_predicted and not col_str:
                col_str = tstr2
            if idx_str and col_str:
                break

        # Encode and step
        encoding = encode_fn(token_id, token_str)
        active_neurons = region.process(encoding)
        active_set = frozenset(int(i) for i in active_neurons)

        diag.step(t, region)

        # Track column set ambiguity
        cols = frozenset(int(c) for c in np.nonzero(region.active_columns)[0])
        colset_to_tokens[cols].add(token_id)

        if t > 0:
            if active_set:
                overlap = len(predicted_set & active_set) / len(active_set)
            else:
                overlap = 0.0
            metrics.overlaps.append(overlap)

            accuracy = 1.0 if idx_predicted == token_id else 0.0
            metrics.accuracies.append(accuracy)

            col_acc = 1.0 if col_predicted == token_id else 0.0
            metrics.column_accuracies.append(col_acc)

            if show_predictions > 0:
                prediction_log.append((token_str, idx_str, col_str))

        # Update decode indices
        decode_index.observe(token_id, active_set)
        if token_id not in token_id_set:
            idx = len(token_ids)
            token_id_set.add(token_id)
            token_ids.append(token_id)
            for col in np.nonzero(region.active_columns)[0]:
                col_index.setdefault(int(col), []).append(idx)

        if t > 0 and t % log_interval == 0 and metrics.overlaps:
            tail_o = metrics.overlaps[-100:]
            tail_a = metrics.accuracies[-100:]
            tail_c = metrics.column_accuracies[-100:]
            elapsed = time.monotonic() - start
            print(
                f"  t={t:,} "
                f"overlap={sum(tail_o) / len(tail_o):.4f} "
                f"idx={sum(tail_a) / len(tail_a):.4f} "
                f"col={sum(tail_c) / len(tail_c):.4f} "
                f"({elapsed:.1f}s)"
            )

            if show_predictions > 0 and prediction_log:
                samples = prediction_log[-show_predictions:]
                print(f"    {'actual':>12s} | {'idx':>12s} | {'col':>12s}")
                print(f"    {'-' * 12}-+-{'-' * 12}-+-{'-' * 12}")
                for actual, ip, cp in samples:

                    def fmt(s):
                        return repr(s)[:12].ljust(12)

                    hit_i = "*" if ip == actual else " "
                    hit_c = "*" if cp == actual else " "
                    print(f"    {fmt(actual)} |{hit_i}{fmt(ip)} |{hit_c}{fmt(cp)}")
                prediction_log.clear()

    metrics.elapsed_seconds = time.monotonic() - start

    summ = diag.summary()
    snap = diag.snapshots[-1] if diag.snapshots else None

    # Column set ambiguity
    n_unique_colsets = len(colset_to_tokens)
    uniquely_identifiable = 0
    n_unique_tokens = len(token_id_set)
    token_to_colsets: dict[int, set] = defaultdict(set)
    for cs, tids in colset_to_tokens.items():
        for tid in tids:
            token_to_colsets[tid].add(cs)
    for tid in token_to_colsets:
        for cs in token_to_colsets[tid]:
            if len(colset_to_tokens[cs]) == 1:
                uniquely_identifiable += 1
                break
    max_ambiguity = max(len(tids) for tids in colset_to_tokens.values())

    tail_a = metrics.accuracies[-100:] if metrics.accuracies else []
    tail_c = metrics.column_accuracies[-100:] if metrics.column_accuracies else []
    tail_o = metrics.overlaps[-100:] if metrics.overlaps else []

    result = {
        "name": name,
        "n_params": n_params,
        "time": metrics.elapsed_seconds,
        "avg_overlap": sum(metrics.overlaps) / len(metrics.overlaps)
        if metrics.overlaps
        else 0,
        "avg_idx": sum(metrics.accuracies) / len(metrics.accuracies)
        if metrics.accuracies
        else 0,
        "avg_col": sum(metrics.column_accuracies) / len(metrics.column_accuracies)
        if metrics.column_accuracies
        else 0,
        "last_overlap": sum(tail_o) / len(tail_o) if tail_o else 0,
        "last_idx": sum(tail_a) / len(tail_a) if tail_a else 0,
        "last_col": sum(tail_c) / len(tail_c) if tail_c else 0,
        "entropy": summ["column_entropy_ratio"],
        "unique_colsets": n_unique_colsets,
        "burst_rate": summ["burst_rate"],
        "pred_sets": summ["unique_prediction_sets"],
        "n_predicted": snap.n_predicted_neurons if snap else 0,
        "fb_cosine": snap.fb_row_cosine_mean if snap else 0,
        "uniquely_id": uniquely_identifiable,
        "unique_tokens": n_unique_tokens,
        "max_ambiguity": max_ambiguity,
    }

    pct_id = uniquely_identifiable / n_unique_tokens if n_unique_tokens else 0
    print(
        f"  DONE: idx={result['avg_idx']:.1%} col={result['avg_col']:.1%} "
        f"overlap={result['avg_overlap']:.4f} ({result['time']:.1f}s)"
    )
    print(
        f"  colsets={n_unique_colsets} "
        f"uniquely_id={uniquely_identifiable}/{n_unique_tokens} "
        f"({pct_id:.1%}) max_ambig={max_ambiguity}"
    )
    print(
        f"  entropy={summ['column_entropy_ratio']:.1%} burst={summ['burst_rate']:.1%} "
        f"pred_sets={summ['unique_prediction_sets']} "
        f"predicted_n={result['n_predicted']}"
    )
    print()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=10000)
    parser.add_argument("--log-interval", type=int, default=2000)
    parser.add_argument("--show-predictions", type=int, default=5)
    args = parser.parse_args()

    tokens = prepare_tokens(args.tokens)

    charbit = CharbitEncoder(length=CHAR_LENGTH, width=CHAR_WIDTH, chars=CHARS)
    input_dim_charbit = CHAR_LENGTH * CHAR_WIDTH  # 808

    # RandomEncoder: same total dim, 8 active bits (same sparsity as charbit)
    random_n = 808
    random_k = 8
    random_enc = RandomBinaryEncoder(n=random_n, k=random_k)

    experiments = [
        (
            "charbit",
            lambda tid, tstr: charbit.encode(tstr),
            input_dim_charbit,
            CHAR_WIDTH,
        ),
        (
            "charbit-nosp",
            lambda tid, tstr: charbit.encode(tstr.lstrip(" ")),
            input_dim_charbit,
            CHAR_WIDTH,
        ),
        (
            "random-808",
            lambda tid, tstr: random_enc.encode(tid),
            random_n,
            0,  # no 2D structure
        ),
    ]

    results = []
    for name, encode_fn, input_dim, enc_width in experiments:
        result = run_experiment(
            name,
            tokens,
            encode_fn,
            input_dim,
            enc_width,
            args.log_interval,
            args.show_predictions,
        )
        results.append(result)

    # Summary table
    print("\n" + "=" * 110)
    print(
        f"{'Config':<16} {'Params':>8} {'Time':>6} "
        f"{'AvgIdx':>7} {'AvgCol':>7} {'L-Idx':>6} {'L-Col':>6} "
        f"{'ColSets':>8} {'UniqID':>7} {'MaxAmb':>7} "
        f"{'Entropy':>8} {'Burst':>6} {'PredN':>6}"
    )
    print("=" * 110)

    for r in results:
        pct = r["uniquely_id"] / r["unique_tokens"] if r["unique_tokens"] else 0
        print(
            f"{r['name']:<16} {r['n_params']:>8,} {r['time']:>5.1f}s "
            f"{r['avg_idx']:>6.1%} {r['avg_col']:>6.1%} "
            f"{r['last_idx']:>5.1%} {r['last_col']:>5.1%} "
            f"{r['unique_colsets']:>8} {pct:>6.1%} {r['max_ambiguity']:>7} "
            f"{r['entropy']:>7.1%} {r['burst_rate']:>5.1%} {r['n_predicted']:>6}"
        )

    print("=" * 110)

    best = max(results, key=lambda r: r["last_idx"])
    print(f"\nBest last-100 idx: {best['name']} ({best['last_idx']:.1%})")


if __name__ == "__main__":
    main()
