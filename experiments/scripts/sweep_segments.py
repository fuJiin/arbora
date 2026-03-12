#!/usr/bin/env python3
"""Sweep dendritic segment parameters with random encoder.

Isolates segment learning from encoding quality by using a random
encoder with near-perfect token discriminability.

Usage: uv run experiments/scripts/sweep_segments.py [--tokens N]
"""

import argparse
import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

import step.env  # noqa: F401
from step.cortex.diagnostics import CortexDiagnostics
from step.cortex.runner import STORY_BOUNDARY
from step.cortex.sensory import SensoryRegion
from step.decoders import InvertedIndexDecoder, SynapticDecoder


@dataclass
class RunMetrics:
    overlaps: list[float] = field(default_factory=list)
    accuracies: list[float] = field(default_factory=list)
    syn_accuracies: list[float] = field(default_factory=list)
    elapsed_seconds: float = 0.0


class RandomBinaryEncoder:
    """Deterministic random sparse binary encoder."""

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


def prepare_tokens(max_tokens: int):
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


def run_config(
    name: str,
    tokens: list[tuple[int, str]],
    encoder: RandomBinaryEncoder,
    log_interval: int,
    # Segment params to vary
    n_fb_segments: int = 4,
    n_lat_segments: int = 4,
    n_synapses: int = 16,
    seg_threshold: int = 4,
    perm_init: float = 0.6,
    perm_inc: float = 0.1,
    perm_dec: float = 0.05,
    perm_threshold: float = 0.5,
):
    region = SensoryRegion(
        input_dim=encoder.n,
        encoding_width=0,
        n_columns=32,
        n_l4=4,
        n_l23=4,
        k_columns=4,
        voltage_decay=0.5,
        eligibility_decay=0.95,
        synapse_decay=0.999,
        learning_rate=0.05,
        max_excitability=0.2,
        fb_boost=0.4,
        ltd_rate=0.2,
        burst_learning_scale=3.0,
        # Segment params
        n_fb_segments=n_fb_segments,
        n_lat_segments=n_lat_segments,
        n_synapses_per_segment=n_synapses,
        seg_activation_threshold=seg_threshold,
        perm_init=perm_init,
        perm_increment=perm_inc,
        perm_decrement=perm_dec,
        perm_threshold=perm_threshold,
        seed=42,
    )

    diag = CortexDiagnostics(snapshot_interval=log_interval)
    decode_index = InvertedIndexDecoder()
    syn_decoder = SynapticDecoder()
    metrics = RunMetrics()
    k = region.k_columns
    start = time.monotonic()

    colset_to_tokens: dict[frozenset, set[int]] = defaultdict(set)
    token_id_set: set[int] = set()

    print(f"--- {name} ---")

    for t, (token_id, token_str) in enumerate(tokens):
        if token_id == STORY_BOUNDARY:
            region.reset_working_memory()
            continue

        predicted_neurons = region.get_prediction(k)
        predicted_set = frozenset(int(i) for i in predicted_neurons)
        idx_predicted = decode_index.decode(predicted_set)
        syn_id, _syn_str = syn_decoder.decode_synaptic(
            predicted_neurons, region
        )

        encoding = encoder.encode(token_id)
        active_neurons = region.process(encoding)
        active_set = frozenset(int(i) for i in active_neurons)

        diag.step(t, region)

        cols = frozenset(
            int(c) for c in np.nonzero(region.active_columns)[0]
        )
        colset_to_tokens[cols].add(token_id)
        token_id_set.add(token_id)

        if t > 0:
            if active_set:
                overlap = len(predicted_set & active_set) / len(active_set)
            else:
                overlap = 0.0
            metrics.overlaps.append(overlap)
            metrics.accuracies.append(
                1.0 if idx_predicted == token_id else 0.0
            )
            metrics.syn_accuracies.append(
                1.0 if syn_id == token_id else 0.0
            )

        decode_index.observe(token_id, active_set)
        syn_decoder.observe(
            token_id, token_str, encoding, region.active_columns
        )

        if t > 0 and t % log_interval == 0 and metrics.overlaps:
            tail_s = metrics.syn_accuracies[-100:]
            tail_a = metrics.accuracies[-100:]
            elapsed = time.monotonic() - start
            bc = diag._burst_count
            pc = diag._precise_count
            print(
                f"  t={t:,} "
                f"syn={sum(tail_s)/len(tail_s):.4f} "
                f"idx={sum(tail_a)/len(tail_a):.4f} "
                f"burst={bc/(bc+pc):.1%} "
                f"fb_conn={np.mean(region.fb_seg_perm > region.perm_threshold):.1%} "
                f"({elapsed:.1f}s)"
            )

    metrics.elapsed_seconds = time.monotonic() - start

    summ = diag.summary()
    snap = diag.snapshots[-1] if diag.snapshots else None

    n_unique_colsets = len(colset_to_tokens)
    n_unique_tokens = len(token_id_set)
    uniquely_id = sum(
        1
        for tid in token_id_set
        if any(
            len(colset_to_tokens[cs]) == 1
            for cs in colset_to_tokens
            if tid in colset_to_tokens[cs]
        )
    )
    pct_id = uniquely_id / n_unique_tokens if n_unique_tokens else 0

    tail_a = metrics.accuracies[-100:] if metrics.accuracies else []
    tail_s = metrics.syn_accuracies[-100:] if metrics.syn_accuracies else []
    avg_a = (
        sum(metrics.accuracies) / len(metrics.accuracies)
        if metrics.accuracies
        else 0
    )
    avg_s = (
        sum(metrics.syn_accuracies) / len(metrics.syn_accuracies)
        if metrics.syn_accuracies
        else 0
    )

    result = {
        "name": name,
        "time": metrics.elapsed_seconds,
        "avg_idx": avg_a,
        "last_idx": sum(tail_a) / len(tail_a) if tail_a else 0,
        "avg_syn": avg_s,
        "last_syn": sum(tail_s) / len(tail_s) if tail_s else 0,
        "burst_rate": summ["burst_rate"],
        "entropy": summ["column_entropy_ratio"],
        "unique_colsets": n_unique_colsets,
        "uniquely_id_pct": pct_id,
        "fb_conn": float(
            np.mean(region.fb_seg_perm > region.perm_threshold)
        ),
        "lat_conn": float(
            np.mean(region.lat_seg_perm > region.perm_threshold)
        ),
        "fb_perm_mean": float(region.fb_seg_perm.mean()),
        "fb_perm_max": float(region.fb_seg_perm.max()),
        "n_active_fb": snap.n_active_fb_segments if snap else 0,
        "n_active_lat": snap.n_active_lat_segments if snap else 0,
        "pred_sets": summ["unique_prediction_sets"],
    }

    print(
        f"  DONE: syn={avg_s:.1%} idx={avg_a:.1%} "
        f"burst={summ['burst_rate']:.1%} "
        f"fb_conn={result['fb_conn']:.1%} "
        f"({metrics.elapsed_seconds:.1f}s)\n"
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=10000)
    parser.add_argument("--log-interval", type=int, default=2000)
    args = parser.parse_args()

    tokens = prepare_tokens(args.tokens)

    random_enc = RandomBinaryEncoder(n=808, k=8)

    configs = [
        # Baseline: current defaults
        ("baseline", {}),
        # Lower threshold (easier to fire)
        ("thresh=2", {"seg_threshold": 2}),
        ("thresh=3", {"seg_threshold": 3}),
        # Faster growth
        ("inc=0.2", {"perm_inc": 0.2}),
        ("inc=0.3", {"perm_inc": 0.3}),
        # Higher init permanence
        ("init=0.8", {"perm_init": 0.8}),
        # Combined: low thresh + fast growth
        ("t2+i0.2", {"seg_threshold": 2, "perm_inc": 0.2}),
        ("t2+i0.3", {"seg_threshold": 2, "perm_inc": 0.3}),
        # More segments
        ("8seg", {"n_fb_segments": 8, "n_lat_segments": 8}),
        # Fewer synapses (faster to fill)
        ("8syn+t2", {"n_synapses": 8, "seg_threshold": 2}),
        # Best guess combo
        (
            "combo",
            {
                "seg_threshold": 2,
                "perm_inc": 0.3,
                "perm_init": 0.8,
                "n_synapses": 8,
            },
        ),
    ]

    results = []
    for name, overrides in configs:
        result = run_config(
            name,
            tokens,
            random_enc,
            args.log_interval,
            **overrides,
        )
        results.append(result)

    # Summary table
    print("\n" + "=" * 130)
    print(
        f"{'Config':<14} {'Time':>5} "
        f"{'AvgSyn':>7} {'L-Syn':>6} "
        f"{'AvgIdx':>7} {'L-Idx':>6} "
        f"{'Burst':>6} {'FbConn':>7} {'LatConn':>8} "
        f"{'FbPerm':>7} "
        f"{'ActFb':>6} {'ActLat':>7} "
        f"{'PredSets':>9}"
    )
    print("=" * 130)

    for r in results:
        print(
            f"{r['name']:<14} {r['time']:>4.0f}s "
            f"{r['avg_syn']:>6.1%} {r['last_syn']:>5.1%} "
            f"{r['avg_idx']:>6.1%} {r['last_idx']:>5.1%} "
            f"{r['burst_rate']:>5.1%} {r['fb_conn']:>6.1%} {r['lat_conn']:>7.1%} "
            f"{r['fb_perm_mean']:>7.4f} "
            f"{r['n_active_fb']:>6} {r['n_active_lat']:>7} "
            f"{r['pred_sets']:>9}"
        )

    print("=" * 130)
    best = max(results, key=lambda r: r["last_syn"])
    print(f"\nBest last-100 syn: {best['name']} ({best['last_syn']:.1%})")


if __name__ == "__main__":
    main()
