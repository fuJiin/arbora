#!/usr/bin/env python3
"""Sweep S1 parameters in isolation.

Tests segment learning, neuron counts, and trace dynamics on char-level
TinyDialogues data using the real CharbitEncoder (not random encoding).

Outputs CSV + summary table for analysis.

Usage:
    uv run experiments/scripts/sweep_s1.py [--tokens N] [--confirm-tokens N]
"""

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from arbora.cortex.sensory import SensoryRegion
from arbora.decoders.dendritic import DendriticDecoder
from arbora.probes.bpc import BPCProbe
from arbora.probes.diagnostics import CortexDiagnostics
from examples.chat.data import STORY_BOUNDARY, prepare_tokens_tinydialogues

# --- Encoder (matches canonical S1) ---

DEFAULT_CHARS = (
    " abcdefghijklmnopqrstuvwxyz.?,!'\"\n-:;()0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
)


class CharbitEncoder:
    """Minimal charbit encoder matching canonical S1 config."""

    def __init__(self, length: int = 8, width: int = 101, chars: str = DEFAULT_CHARS):
        self.length = length
        self.width = width
        self.input_dim = length * width  # 808
        self.encoding_width = width
        self._char_to_col = {ch: i for i, ch in enumerate(chars)}
        self._unknown_col = width - 1

    def encode(self, char: str) -> np.ndarray:
        out = np.zeros((self.length, self.width), dtype=np.bool_)
        for i, ch in enumerate(char[: self.length]):
            col = self._char_to_col.get(ch, self._unknown_col)
            out[i, col] = True
        return out.flatten()


# --- L2/3 context discrimination tracker ---


class L23ContextTracker:
    """Track L2/3 context discrimination.

    Same token, different contexts should produce different patterns.
    """

    def __init__(self):
        self._prev_token: int | None = None
        # (prev_token, token) -> list of L2/3 neuron sets
        self._bigram_patterns: dict[tuple[int, int], list[frozenset[int]]] = (
            defaultdict(list)
        )
        # token → set of preceding tokens
        self._token_contexts: dict[int, set[int]] = defaultdict(set)

    def observe(self, token_id: int, l23_active: np.ndarray):
        neurons = frozenset(int(n) for n in np.nonzero(l23_active)[0])
        if self._prev_token is not None:
            key = (self._prev_token, token_id)
            self._bigram_patterns[key].append(neurons)
            self._token_contexts[token_id].add(self._prev_token)
        self._prev_token = token_id

    def reset(self):
        self._prev_token = None

    def discrimination(self, min_contexts: int = 3) -> float:
        """Mean Jaccard distance across tokens with enough contexts."""
        rng = np.random.default_rng(42)
        all_dists = []
        for tid, contexts in self._token_contexts.items():
            if len(contexts) < min_contexts:
                continue
            # Collect all patterns for this token across contexts
            patterns = []
            for (_prev, t), pats in self._bigram_patterns.items():
                if t == tid:
                    patterns.extend(pats)
            if len(patterns) < min_contexts:
                continue
            # Sample pairwise Jaccard distances
            n = len(patterns)
            pairs = min(50, n * (n - 1) // 2)
            for _ in range(pairs):
                i, j = rng.choice(n, 2, replace=False)
                s1, s2 = patterns[i], patterns[j]
                union = len(s1 | s2)
                if union > 0:
                    all_dists.append(1.0 - len(s1 & s2) / union)
        return float(np.mean(all_dists)) if all_dists else 0.0


# --- Effective dimensionality ---


def participation_ratio(activations: list[np.ndarray]) -> float:
    """Participation ratio of activation covariance eigenspectrum.

    PR = (Σλ)² / Σλ² where λ are eigenvalues of the covariance matrix.
    High PR = rich representation using many dimensions.
    Low PR = collapsed to few stereotyped patterns.
    """
    if len(activations) < 10:
        return 0.0
    X = np.array(activations, dtype=np.float64)
    X -= X.mean(axis=0)
    # Use SVD for numerical stability (avoids explicit covariance)
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    lambdas = s**2 / (len(activations) - 1)
    sum_l = lambdas.sum()
    sum_l2 = (lambdas**2).sum()
    if sum_l2 < 1e-12:
        return 0.0
    return float(sum_l**2 / sum_l2)


# --- Run a single config ---


@dataclass
class Result:
    name: str
    params: dict
    elapsed: float = 0.0
    # L4 KPIs
    l4_pred_recall: float = 0.0  # KPI #1: prediction recall (1 - burst_rate)
    l4_pred_precision: float = 0.0  # KPI #2: prediction precision
    l4_pop_sparseness: float = 0.0  # KPI #3: population sparseness (Treves-Rolls)
    # L2/3 KPIs
    l23_bpc: float = 0.0  # KPI #1: downstream decodability
    l23_ctx_disc: float = 0.0  # KPI #2: context discrimination
    l23_eff_dim: float = 0.0  # KPI #3: effective dimensionality
    # Supporting diagnostics
    den_pct: float = 0.0
    lat_connected_frac: float = 0.0
    lat_perm_p50: float = 0.0
    l23_connected_frac: float = 0.0
    column_entropy_ratio: float = 0.0


def run_config(
    name: str,
    tokens: list[tuple[int, str]],
    encoder: CharbitEncoder,
    log_interval: int,
    *,
    # Region params to vary
    n_columns: int = 128,
    n_l4: int = 4,
    n_l23: int = 4,
    n_l5: int = 2,
    k_columns: int = 8,
    n_l4_lat_segments: int = 4,
    n_l23_segments: int = 4,
    n_synapses_per_segment: int = 24,
    seg_activation_threshold: int = 2,
    perm_init: float = 0.6,
    perm_increment: float = 0.2,
    perm_decrement: float = 0.05,
    perm_threshold: float = 0.5,
    pre_trace_decay: float = 0.8,
    prediction_gain: float = 2.5,
    learning_rate: float = 0.05,
    ltd_rate: float = 0.05,
    synapse_decay: float = 1.0,
) -> Result:
    region = SensoryRegion(
        input_dim=encoder.input_dim,
        encoding_width=encoder.encoding_width,
        n_columns=n_columns,
        n_l4=n_l4,
        n_l23=n_l23,
        n_l5=n_l5,
        k_columns=k_columns,
        n_l4_lat_segments=n_l4_lat_segments,
        n_l23_segments=n_l23_segments,
        n_synapses_per_segment=n_synapses_per_segment,
        seg_activation_threshold=seg_activation_threshold,
        perm_init=perm_init,
        perm_increment=perm_increment,
        perm_decrement=perm_decrement,
        perm_threshold=perm_threshold,
        pre_trace_decay=pre_trace_decay,
        prediction_gain=prediction_gain,
        learning_rate=learning_rate,
        ltd_rate=ltd_rate,
        synapse_decay=synapse_decay,
        voltage_decay=0.5,
        eligibility_decay=0.95,
        max_excitability=0.2,
        fb_boost=0.4,
        burst_learning_scale=3.0,
        seed=42,
    )

    diag = CortexDiagnostics(snapshot_interval=log_interval)
    ctx_tracker = L23ContextTracker()
    # DendriticDecoder + BPCProbe on L2/3: what downstream regions would compute
    dd = DendriticDecoder(source_dim=region.n_l23_total, seed=42)
    bpc_probe = BPCProbe()
    start = time.monotonic()
    prev_l23 = np.zeros(region.n_l23_total, dtype=np.bool_)
    # Collect L2/3 activations for participation ratio (subsample for perf)
    l23_samples: list[np.ndarray] = []
    # L4 prediction precision tracking
    l4_pred_total: int = 0
    l4_pred_correct: int = 0
    # L4 population sparseness accumulator
    l4_sparseness_values: list[float] = []

    print(f"  [{name}] ", end="", flush=True)

    for t, (token_id, token_str) in enumerate(tokens):
        if token_id == STORY_BOUNDARY:
            region.reset_working_memory()
            bpc_probe.dialogue_boundary()
            ctx_tracker.reset()
            prev_l23[:] = False
            continue

        # Process
        encoding = encoder.encode(token_str)
        region.process(encoding)

        # L4 prediction precision: of neurons predicted, how many fired?
        predicted_l4 = np.nonzero(region.l4.predicted)[0]
        if len(predicted_l4) > 0:
            l4_pred_total += len(predicted_l4)
            l4_pred_correct += int((region.l4.predicted & region.l4.active).sum())

        # L4 population sparseness (Treves-Rolls)
        r = region.l4.active.astype(np.float64)
        mean_r = r.mean()
        mean_r2 = (r**2).mean()
        if mean_r2 > 0:
            l4_sparseness_values.append(mean_r**2 / mean_r2)

        # L2/3 context discrimination
        ctx_tracker.observe(token_id, region.l23.active)

        # L2/3 BPC via DendriticDecoder
        if t > 0:
            bpc_probe.step(token_id, prev_l23, dd)
        dd.observe(token_id, region.l23.active)
        prev_l23 = region.l23.active.copy()

        # Subsample L2/3 activations for participation ratio (every 10th step)
        if t % 10 == 0:
            l23_samples.append(region.l23.active.astype(np.float64))

        diag.step(t, region)

        if t > 0 and t % log_interval == 0:
            bc = diag._burst_count
            pc = diag._precise_count
            l4_recall = pc / (bc + pc) if (bc + pc) > 0 else 0
            print(
                f"t={t:,} rcl={l4_recall:.0%} bpc={bpc_probe.recent_bpc:.2f} "
                f"ctx={ctx_tracker.discrimination():.3f} | ",
                end="",
                flush=True,
            )

    elapsed = time.monotonic() - start
    summ = diag.summary()
    snap = diag.snapshots[-1] if diag.snapshots else None

    # Permanence percentiles
    lat_perm = region.l4_lat_seg_perm
    _p10, p50, _p90 = np.percentile(lat_perm, [10, 50, 90])

    # Compute L2/3 KPIs
    l23_ctx = ctx_tracker.discrimination()
    l23_dim = participation_ratio(l23_samples)
    l4_precision = l4_pred_correct / max(l4_pred_total, 1)
    l4_sparseness = float(np.mean(l4_sparseness_values)) if l4_sparseness_values else 0

    result = Result(
        name=name,
        params={
            "n_columns": n_columns,
            "n_l4": n_l4,
            "n_l23": n_l23,
            "n_l5": n_l5,
            "k_columns": k_columns,
            "n_l4_lat_segments": n_l4_lat_segments,
            "n_l23_segments": n_l23_segments,
            "n_synapses_per_segment": n_synapses_per_segment,
            "seg_activation_threshold": seg_activation_threshold,
            "perm_init": perm_init,
            "perm_increment": perm_increment,
            "perm_decrement": perm_decrement,
            "perm_threshold": perm_threshold,
            "pre_trace_decay": pre_trace_decay,
            "prediction_gain": prediction_gain,
            "learning_rate": learning_rate,
            "ltd_rate": ltd_rate,
        },
        elapsed=elapsed,
        # L4 KPIs
        l4_pred_recall=1.0 - summ["burst_rate"],
        l4_pred_precision=l4_precision,
        l4_pop_sparseness=l4_sparseness,
        # L2/3 KPIs
        l23_bpc=bpc_probe.recent_bpc,
        l23_ctx_disc=l23_ctx,
        l23_eff_dim=l23_dim,
        # Supporting diagnostics
        den_pct=(snap.n_active_lat_segments / max(region.n_l4_total, 1)) if snap else 0,
        lat_connected_frac=snap.lat_seg_connected_frac if snap else 0,
        lat_perm_p50=float(p50),
        l23_connected_frac=snap.l23_seg_connected_frac if snap else 0,
        column_entropy_ratio=summ["column_entropy_ratio"],
    )

    print(
        f"DONE rcl={result.l4_pred_recall:.1%} prc={l4_precision:.1%} "
        f"bpc={result.l23_bpc:.2f} ctx={l23_ctx:.3f} dim={l23_dim:.1f} "
        f"({elapsed:.1f}s)"
    )
    return result


# --- Sweep configs ---


def build_configs() -> list[tuple[str, dict]]:
    """Parameter grid for S1 sweep."""
    configs = []

    # === Group 1: Baseline ===
    configs.append(("baseline", {}))

    # === Group 2: Segment activation threshold ===
    # Current default: threshold=2, synapses=24
    # Lower threshold = easier to fire but noisier
    configs.append(("thresh=3", {"seg_activation_threshold": 3}))
    configs.append(("thresh=4", {"seg_activation_threshold": 4}))
    configs.append(("thresh=6", {"seg_activation_threshold": 6}))

    # === Group 3: Synapses per segment ===
    # Fewer synapses = faster to fill, but less specific
    configs.append(("syn=12", {"n_synapses_per_segment": 12}))
    configs.append(("syn=8", {"n_synapses_per_segment": 8}))
    configs.append(("syn=32", {"n_synapses_per_segment": 32}))

    # === Group 4: More segments ===
    configs.append(("8seg", {"n_l4_lat_segments": 8, "n_l23_segments": 8}))
    configs.append(("12seg", {"n_l4_lat_segments": 12, "n_l23_segments": 12}))

    # === Group 5: Permanence dynamics ===
    # Higher init = start connected, learn by pruning bad ones
    configs.append(("pinit=0.8", {"perm_init": 0.8}))
    # Faster growth
    configs.append(("pinc=0.3", {"perm_increment": 0.3}))
    # More aggressive pruning
    configs.append(("pdec=0.1", {"perm_decrement": 0.1}))
    # Combined: fast growth + aggressive pruning
    configs.append(
        (
            "fast_perm",
            {"perm_increment": 0.3, "perm_decrement": 0.1, "perm_init": 0.7},
        )
    )

    # === Group 6: Neuron counts (asymmetric) ===
    # Biology: L4 thickest in sensory, L5 thinnest
    configs.append(("n_l5=0", {"n_l5": 0}))  # L5 inert in S1-only, verify
    configs.append(("n_l4=6", {"n_l4": 6}))
    configs.append(("n_l4=8", {"n_l4": 8}))
    configs.append(("l4=6_l23=4_l5=2", {"n_l4": 6, "n_l23": 4, "n_l5": 2}))
    configs.append(("l4=8_l23=4_l5=2", {"n_l4": 8, "n_l23": 4, "n_l5": 2}))
    configs.append(("l4=4_l23=6_l5=2", {"n_l4": 4, "n_l23": 6, "n_l5": 2}))

    # === Group 7: Trace decay ===
    configs.append(("trace=0.6", {"pre_trace_decay": 0.6}))
    configs.append(("trace=0.9", {"pre_trace_decay": 0.9}))
    configs.append(("trace=0.0", {"pre_trace_decay": 0.0}))

    # === Group 8: Prediction gain ===
    configs.append(("gain=1.5", {"prediction_gain": 1.5}))
    configs.append(("gain=4.0", {"prediction_gain": 4.0}))

    # === Group 9: Combined "best guess" configs ===
    configs.append(
        (
            "combo_seg",
            {
                "seg_activation_threshold": 4,
                "n_synapses_per_segment": 12,
                "n_l4_lat_segments": 8,
                "perm_increment": 0.3,
                "perm_decrement": 0.1,
            },
        )
    )
    configs.append(
        (
            "combo_neuron",
            {
                "n_l4": 6,
                "n_l23": 4,
                "n_l5": 2,
                "n_l4_lat_segments": 8,
                "seg_activation_threshold": 4,
            },
        )
    )
    configs.append(
        (
            "combo_all",
            {
                "n_l4": 6,
                "n_l23": 4,
                "n_l5": 2,
                "n_l4_lat_segments": 8,
                "n_l23_segments": 8,
                "n_synapses_per_segment": 12,
                "seg_activation_threshold": 4,
                "perm_increment": 0.3,
                "perm_decrement": 0.1,
                "pre_trace_decay": 0.9,
            },
        )
    )

    return configs


def save_results(results: list[Result], out_dir: Path):
    """Save results to CSV and JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = out_dir / "sweep_results.csv"
    fieldnames = [
        "name",
        "elapsed",
        "l4_pred_recall",
        "l4_pred_precision",
        "l4_pop_sparseness",
        "l23_bpc",
        "l23_ctx_disc",
        "l23_eff_dim",
        "den_pct",
        "lat_connected_frac",
        "lat_perm_p50",
        "l23_connected_frac",
        "column_entropy_ratio",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: getattr(r, k) for k in fieldnames}
            writer.writerow(row)

    # JSON with full params
    json_path = out_dir / "sweep_results.json"
    data = []
    for r in results:
        d = {k: getattr(r, k) for k in fieldnames}
        d["params"] = r.params
        data.append(d)
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to {csv_path} and {json_path}")


def print_summary(results: list[Result]):
    """Print comparison table sorted by BPC."""
    results_sorted = sorted(results, key=lambda r: r.l23_bpc)

    print("\n" + "=" * 140)
    print(
        f"{'Config':<22} {'Time':>5} "
        f"{'L4Rcl':>6} {'L4Prc':>6} {'L4Spr':>6} "
        f"{'L23BPC':>7} {'CtxDsc':>7} {'EffDim':>7} "
        f"{'Den%':>6} {'LatCon':>7} {'Entrpy':>7}"
    )
    print("-" * 140)
    print(
        f"{'TARGETS':<22} {'':>5} "
        f"{'>70%':>6} {'>50%':>6} {'~.016':>6} "
        f"{'<6.0':>7} {'>0.80':>7} {'>50':>7} "
        f"{'':>6} {'':>7} {'':>7}"
    )
    print("=" * 140)

    for r in results_sorted:
        print(
            f"{r.name:<22} {r.elapsed:>4.0f}s "
            f"{r.l4_pred_recall:>5.1%} {r.l4_pred_precision:>5.1%} "
            f"{r.l4_pop_sparseness:>5.3f} "
            f"{r.l23_bpc:>6.2f} {r.l23_ctx_disc:>6.3f} {r.l23_eff_dim:>6.1f} "
            f"{r.den_pct:>5.1%} {r.lat_connected_frac:>6.1%} "
            f"{r.column_entropy_ratio:>6.3f}"
        )

    print("=" * 140)
    best = results_sorted[0]
    print(f"\nBest L2/3 BPC: {best.name} ({best.l23_bpc:.2f})")
    print(
        f"  l4_recall={best.l4_pred_recall:.1%} "
        f"precision={best.l4_pred_precision:.1%} "
        f"ctx_disc={best.l23_ctx_disc:.3f} eff_dim={best.l23_eff_dim:.1f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Sweep S1 parameters")
    parser.add_argument(
        "--tokens", type=int, default=30000, help="Tokens per config for initial sweep"
    )
    parser.add_argument(
        "--confirm-tokens",
        type=int,
        default=0,
        help="If >0, re-run top 3 configs at this token count",
    )
    parser.add_argument("--log-interval", type=int, default=5000)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Run only this config name (for debugging)",
    )
    parser.add_argument("--out-dir", type=str, default="experiments/runs/sweep_s1")
    args = parser.parse_args()

    # Load data
    print("Loading TinyDialogues (char-level)...")
    max_needed = max(args.tokens, args.confirm_tokens)
    tokens = prepare_tokens_tinydialogues(max_needed)

    encoder = CharbitEncoder()

    # Build configs
    all_configs = build_configs()
    if args.config:
        all_configs = [(n, p) for n, p in all_configs if n == args.config]
        if not all_configs:
            print(f"Config '{args.config}' not found")
            sys.exit(1)

    sweep_tokens = tokens[: args.tokens]
    print(f"\nSweeping {len(all_configs)} configs x {args.tokens:,} tokens\n")

    # Run sweep
    results = []
    for name, overrides in all_configs:
        result = run_config(name, sweep_tokens, encoder, args.log_interval, **overrides)
        results.append(result)

    # Summary
    print_summary(results)

    # Save
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir) / ts
    save_results(results, out_dir)

    # Confirmation runs
    if args.confirm_tokens > 0 and len(results) > 1:
        top3 = sorted(results, key=lambda r: r.l23_bpc)[:3]
        print(f"\n\n{'=' * 60}")
        print(f"Confirming top 3 at {args.confirm_tokens:,} tokens...")
        print(f"{'=' * 60}\n")

        confirm_tokens = tokens[: args.confirm_tokens]
        confirm_results = []
        for r in top3:
            result = run_config(
                f"{r.name}_confirm",
                confirm_tokens,
                encoder,
                args.log_interval,
                **r.params,
            )
            confirm_results.append(result)

        print_summary(confirm_results)
        save_results(confirm_results, out_dir / "confirm")

    return results


if __name__ == "__main__":
    main()
