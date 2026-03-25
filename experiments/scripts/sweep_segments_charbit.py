#!/usr/bin/env python3
"""Sweep dendritic segment parameters with CharbitEncoder on BabyLM.

Re-runs the segment parameter sweep using the canonical encoder (CharbitEncoder)
instead of the random binary encoder. CharbitEncoder has 6x richer similarity
structure, so optimal segment params may differ.

The previous sweep (random encoder) found thresh=2, perm_inc=0.2 as optimal.

Usage:
  uv run experiments/scripts/sweep_segments_charbit.py
  uv run experiments/scripts/sweep_segments_charbit.py --tokens 20000
"""

import argparse
import string
import time

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

import step.env  # noqa: F401
from step.cortex.sensory import SensoryRegion
from step.decoders import InvertedIndexDecoder, SynapticDecoder
from step.encoders.charbit import CharbitEncoder
from step.probes.diagnostics import CortexDiagnostics
from step.probes.representation import RepresentationTracker
from step.runner import STORY_BOUNDARY

CHARS = string.printable
CHAR_LENGTH = 8
CHAR_WIDTH = len(CHARS) + 1


def prepare_tokens(max_tokens: int):
    print("Loading BabyLM (10M) and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    dataset = load_dataset("nilq/babylm-10M", split="train")

    tokens: list[tuple[int, str]] = []
    t = 0
    in_doc = False
    for ex in dataset:
        text = ex.get("text", "").strip()
        if not text:
            if in_doc:
                tokens.append((STORY_BOUNDARY, ""))
                t += 1
                in_doc = False
            if t >= max_tokens:
                break
            continue
        in_doc = True
        for tid in tokenizer.encode(text):
            tokens.append((tid, tokenizer.decode([tid])))
            t += 1
            if t >= max_tokens:
                break
        if t >= max_tokens:
            break

    unique = len({tid for tid, _ in tokens if tid != STORY_BOUNDARY})
    boundaries = sum(1 for tid, _ in tokens if tid == STORY_BOUNDARY)
    print(f"  {len(tokens):,} tokens, {unique} unique, {boundaries + 1} documents\n")
    return tokens


def run_config(
    name: str,
    tokens: list[tuple[int, str]],
    encoder: CharbitEncoder,
    input_dim: int,
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
        input_dim=input_dim,
        encoding_width=CHAR_WIDTH,
        n_columns=32,
        n_l4=4,
        n_l23=4,
        k_columns=4,
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
    rep_tracker = RepresentationTracker(n_columns=32, n_l4=4)
    decode_index = InvertedIndexDecoder()
    syn_decoder = SynapticDecoder()
    syn_accuracies: list[float] = []
    k = region.k_columns
    start = time.monotonic()

    print(f"--- {name} ---")

    for t, (token_id, token_str) in enumerate(tokens):
        if token_id == STORY_BOUNDARY:
            region.reset_working_memory()
            continue

        predicted_neurons = region.get_prediction(k)
        syn_id, _ = syn_decoder.decode_synaptic(predicted_neurons, region)

        encoding = encoder.encode(token_str)
        active_neurons = region.process(encoding)
        active_set = frozenset(int(i) for i in active_neurons)

        diag.step(t, region)
        rep_tracker.observe(token_id, region.active_columns, region.l4.active)

        syn_accuracies.append(1.0 if syn_id == token_id else 0.0)

        decode_index.observe(token_id, active_set)
        syn_decoder.observe(token_id, token_str, encoding, region.active_columns)

        if t > 0 and t % log_interval == 0 and syn_accuracies:
            tail_s = syn_accuracies[-100:]
            elapsed = time.monotonic() - start
            bc = diag._burst_count
            pc = diag._precise_count
            print(
                f"  t={t:,} "
                f"syn={sum(tail_s) / len(tail_s):.4f} "
                f"burst={bc / (bc + pc):.1%} "
                f"fb_conn={np.mean(region.fb_seg_perm > region.perm_threshold):.1%} "
                f"({elapsed:.1f}s)"
            )

    elapsed = time.monotonic() - start
    summ = diag.summary()

    # Representation metrics
    rep = rep_tracker.summary(ff_weights=region.ff_weights)

    tail_s = syn_accuracies[-100:] if syn_accuracies else []

    result = {
        "name": name,
        "time": elapsed,
        "last_syn": sum(tail_s) / len(tail_s) if tail_s else 0,
        "burst_rate": summ["burst_rate"],
        # Representation metrics
        "selectivity": rep["column_selectivity_mean"],
        "ctx_disc": rep["context_discrimination"],
        "cross_cos": rep.get("ff_cross_col_cosine", 0),
        "ff_sparsity": rep.get("ff_sparsity", 0),
        # Segment health
        "fb_conn": float(np.mean(region.fb_seg_perm > region.perm_threshold)),
        "lat_conn": float(np.mean(region.lat_seg_perm > region.perm_threshold)),
        "pred_sets": summ["unique_prediction_sets"],
    }

    print(
        f"  DONE: syn={result['last_syn']:.1%} "
        f"burst={summ['burst_rate']:.1%} "
        f"sel={result['selectivity']:.3f} "
        f"ctx={result['ctx_disc']:.3f} "
        f"({elapsed:.1f}s)\n"
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=10000)
    parser.add_argument("--log-interval", type=int, default=2000)
    args = parser.parse_args()

    tokens = prepare_tokens(args.tokens)
    encoder = CharbitEncoder(length=CHAR_LENGTH, width=CHAR_WIDTH, chars=CHARS)
    input_dim = CHAR_LENGTH * CHAR_WIDTH

    configs = [
        # Current canonical: thresh=2, inc=0.2 (from random encoder sweep)
        ("canonical", {"seg_threshold": 2, "perm_inc": 0.2}),
        # Baseline defaults (thresh=4, inc=0.1)
        ("baseline", {}),
        # Threshold variations
        ("thresh=2", {"seg_threshold": 2}),
        ("thresh=3", {"seg_threshold": 3}),
        # Increment variations at thresh=2
        ("t2+i0.1", {"seg_threshold": 2, "perm_inc": 0.1}),
        ("t2+i0.15", {"seg_threshold": 2, "perm_inc": 0.15}),
        ("t2+i0.2", {"seg_threshold": 2, "perm_inc": 0.2}),
        ("t2+i0.3", {"seg_threshold": 2, "perm_inc": 0.3}),
        # Decrement variations at thresh=2, inc=0.2
        ("t2i2+d0.02", {"seg_threshold": 2, "perm_inc": 0.2, "perm_dec": 0.02}),
        ("t2i2+d0.1", {"seg_threshold": 2, "perm_inc": 0.2, "perm_dec": 0.1}),
        # Synapse count variations
        ("t2i2+8syn", {"seg_threshold": 2, "perm_inc": 0.2, "n_synapses": 8}),
        ("t2i2+24syn", {"seg_threshold": 2, "perm_inc": 0.2, "n_synapses": 24}),
        # Segment count variations
        (
            "t2i2+2seg",
            {
                "seg_threshold": 2,
                "perm_inc": 0.2,
                "n_fb_segments": 2,
                "n_lat_segments": 2,
            },
        ),
        (
            "t2i2+8seg",
            {
                "seg_threshold": 2,
                "perm_inc": 0.2,
                "n_fb_segments": 8,
                "n_lat_segments": 8,
            },
        ),
        # Init permanence
        ("t2i2+p0.5", {"seg_threshold": 2, "perm_inc": 0.2, "perm_init": 0.5}),
        ("t2i2+p0.8", {"seg_threshold": 2, "perm_inc": 0.2, "perm_init": 0.8}),
    ]

    results = []
    for name, overrides in configs:
        result = run_config(
            name, tokens, encoder, input_dim, args.log_interval, **overrides
        )
        results.append(result)

    # Summary table
    print("\n" + "=" * 140)
    print(
        f"{'Config':<14} {'Time':>5} "
        f"{'L-Syn':>6} {'Burst':>6} "
        f"{'Select':>7} {'CtxDsc':>7} {'XCos':>6} {'FFSp':>6} "
        f"{'FbCon':>6} {'LatCon':>7} {'PrdSet':>7}"
    )
    print("=" * 140)

    for r in results:
        print(
            f"{r['name']:<14} {r['time']:>4.0f}s "
            f"{r['last_syn']:>5.1%} {r['burst_rate']:>5.1%} "
            f"{r['selectivity']:>7.3f} {r['ctx_disc']:>7.3f} "
            f"{r['cross_cos']:>6.3f} {r['ff_sparsity']:>6.3f} "
            f"{r['fb_conn']:>5.1%} {r['lat_conn']:>6.1%} "
            f"{r['pred_sets']:>7}"
        )

    print("=" * 140)

    # Best by key metrics
    best_burst = min(results, key=lambda r: r["burst_rate"])
    best_ctx = max(results, key=lambda r: r["ctx_disc"])
    best_sel = min(results, key=lambda r: r["selectivity"])
    print(f"\nBest burst:  {best_burst['name']} ({best_burst['burst_rate']:.1%})")
    print(f"Best ctx disc:      {best_ctx['name']} ({best_ctx['ctx_disc']:.3f})")
    print(f"Best selectivity:   {best_sel['name']} ({best_sel['selectivity']:.3f})")


if __name__ == "__main__":
    main()
