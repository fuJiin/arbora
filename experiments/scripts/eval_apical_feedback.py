#!/usr/bin/env python3
"""A/B evaluation: hierarchy with vs without apical feedback (R2→R1).

Runs the two-region hierarchy with various prediction_gain values:
  - No apical feedback (baseline)
  - Apical feedback at each gain value

Compares R1 burst rate, context discrimination, and apical segment growth.

Usage:
  uv run experiments/scripts/eval_apical_feedback.py
  uv run experiments/scripts/eval_apical_feedback.py --tokens 10000
  uv run experiments/scripts/eval_apical_feedback.py --gain 2.0
  uv run experiments/scripts/eval_apical_feedback.py --sweep
"""

import argparse
import string
import time

from datasets import load_dataset
from transformers import AutoTokenizer

import step.env  # noqa: F401
from step.cortex.sensory import SensoryRegion
from step.cortex.surprise import SurpriseTracker
from step.data import STORY_BOUNDARY
from step.encoders.charbit import CharbitEncoder
from step.probes.diagnostics import CortexDiagnostics
from step.runner import run_hierarchy

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


def make_regions(prediction_gain: float, with_apical: bool, seed: int = 42):
    """Create R1 + R2 with optional apical feedback."""
    input_dim = CHAR_LENGTH * CHAR_WIDTH
    r1 = SensoryRegion(
        input_dim=input_dim,
        encoding_width=CHAR_WIDTH,
        n_columns=32,
        n_l4=4,
        n_l23=4,
        k_columns=4,
        prediction_gain=prediction_gain if with_apical else 1.0,
        seed=seed,
    )
    r2 = SensoryRegion(
        input_dim=r1.n_l23_total,
        encoding_width=0,
        n_columns=16,
        n_l4=4,
        n_l23=4,
        k_columns=2,
        voltage_decay=0.8,
        eligibility_decay=0.98,
        synapse_decay=0.9999,
        learning_rate=0.01,
        ltd_rate=0.4,
        seed=seed + 81,
    )
    if with_apical:
        r1.init_apical_segments(source_dim=r2.n_l23_total)
    return r1, r2


def run_one(
    label: str,
    tokens: list[tuple[int, str]],
    encoder: CharbitEncoder,
    r1: SensoryRegion,
    r2: SensoryRegion,
    log_interval: int,
) -> dict:
    """Run hierarchy and return summary dict."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")

    surprise = SurpriseTracker()
    diag1 = CortexDiagnostics(snapshot_interval=log_interval)
    diag2 = CortexDiagnostics(snapshot_interval=log_interval)

    start = time.monotonic()
    metrics = run_hierarchy(
        r1, r2, encoder, tokens,
        surprise_tracker=surprise,
        log_interval=log_interval,
        diagnostics1=diag1,
        diagnostics2=diag2,
    )
    elapsed = time.monotonic() - start

    summ1 = diag1.summary()
    summ2 = diag2.summary()
    rep1 = metrics.region1.representation
    rep2 = metrics.region2.representation
    snap1 = diag1.snapshots[-1] if diag1.snapshots else None

    # Apical health
    apical_conn = 0.0
    apical_perm_mean = 0.0
    apical_pred_cols = 0
    if snap1 and r1.has_apical:
        apical_conn = snap1.apical_seg_connected_frac
        apical_perm_mean = snap1.apical_seg_perm_mean
        apical_pred_cols = snap1.n_apical_predicted_cols

    tail_syn = metrics.region1.synaptic_accuracies[-100:]
    r1_syn = sum(tail_syn) / len(tail_syn) if tail_syn else 0

    return {
        "label": label,
        "time": elapsed,
        "r1_burst": summ1["burst_rate"],
        "r1_syn": r1_syn,
        "r1_select": rep1.get("column_selectivity_mean", 0),
        "r1_ctx_disc": rep1.get("context_discrimination", 0),
        "r1_cross_cos": rep1.get("ff_cross_col_cosine", 0),
        "r2_burst": summ2["burst_rate"],
        "r2_select": rep2.get("column_selectivity_mean", 0),
        "r2_ctx_disc": rep2.get("context_discrimination", 0),
        "apical_conn": apical_conn,
        "apical_perm": apical_perm_mean,
        "apical_pred": apical_pred_cols,
        "pred_sets": summ1["unique_prediction_sets"],
    }


def print_summary(results: list[dict]):
    """Print comparison table and delta analysis."""
    print(f"\n\n{'=' * 120}")
    print(
        f"{'Config':<22} {'Time':>5} "
        f"{'R1Brst':>7} {'R1Syn':>6} {'R1Sel':>6} "
        f"{'R1Ctx':>6} {'R1XCos':>7} "
        f"{'R2Brst':>7} {'R2Ctx':>6} "
        f"{'ApConn':>7} {'ApPrm':>6} "
        f"{'ApPrd':>6} {'PrdSet':>7}"
    )
    print("=" * 120)

    for r in results:
        print(
            f"{r['label']:<22} {r['time']:>4.0f}s "
            f"{r['r1_burst']:>6.1%} {r['r1_syn']:>5.1%} "
            f"{r['r1_select']:>6.3f} {r['r1_ctx_disc']:>6.3f} "
            f"{r['r1_cross_cos']:>7.3f} "
            f"{r['r2_burst']:>6.1%} {r['r2_ctx_disc']:>6.3f} "
            f"{r['apical_conn']:>6.1%} "
            f"{r['apical_perm']:>6.4f} "
            f"{r['apical_pred']:>6} {r['pred_sets']:>7}"
        )

    print("=" * 120)

    # Delta analysis vs baseline
    base = results[0]
    for r in results[1:]:
        d_burst = r["r1_burst"] - base["r1_burst"]
        d_ctx = r["r1_ctx_disc"] - base["r1_ctx_disc"]
        print(
            f"\n  {r['label']} vs baseline: "
            f"burst {d_burst:+.1%}  "
            f"ctx_disc {d_ctx:+.3f}  "
            f"apical_conn {r['apical_conn']:.1%}  "
            f"apical_pred {r['apical_pred']}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=5000)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument(
        "--gain", type=float, default=1.5,
        help="prediction_gain (single A/B mode)",
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="Sweep multiple gain values instead of single A/B",
    )
    args = parser.parse_args()

    tokens = prepare_tokens(args.tokens)
    encoder = CharbitEncoder(
        length=CHAR_LENGTH, width=CHAR_WIDTH, chars=CHARS,
    )

    gains = (
        [1.5, 2.0, 2.5, 3.0]
        if args.sweep
        else [args.gain]
    )

    results = []

    # Baseline: no apical feedback
    r1a, r2a = make_regions(1.0, with_apical=False)
    results.append(run_one(
        "No feedback", tokens, encoder, r1a, r2a,
        args.log_interval,
    ))

    # Each gain value
    for gain in gains:
        r1, r2 = make_regions(gain, with_apical=True)
        results.append(run_one(
            f"gain={gain}", tokens, encoder, r1, r2,
            args.log_interval,
        ))

    print_summary(results)


if __name__ == "__main__":
    main()
