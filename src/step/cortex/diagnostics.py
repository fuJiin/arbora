"""Cortex diagnostic instrumentation.

All metrics are computed from existing arrays via numpy reductions —
no extra allocations per step. Snapshots are taken at configurable
intervals to avoid overhead on the hot path.
"""

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from step.cortex.sensory import SensoryRegion


@dataclass
class Snapshot:
    """Single-timestep diagnostic snapshot."""

    t: int

    # Weight health (per matrix)
    ff_mean: float = 0.0
    ff_std: float = 0.0
    ff_max: float = 0.0
    ff_sparsity: float = 0.0
    fb_mean: float = 0.0
    fb_std: float = 0.0
    fb_max: float = 0.0
    fb_sparsity: float = 0.0
    lat_mean: float = 0.0
    lat_std: float = 0.0
    lat_max: float = 0.0
    lat_sparsity: float = 0.0
    l23_lat_mean: float = 0.0
    l23_lat_std: float = 0.0
    l23_lat_max: float = 0.0
    l23_lat_sparsity: float = 0.0

    # Excitability vs voltage
    excitability_l4_max: float = 0.0
    excitability_l4_mean: float = 0.0
    voltage_l4_max: float = 0.0

    # Trace health
    trace_l4_mean: float = 0.0
    trace_l4_nonzero: int = 0
    trace_l23_mean: float = 0.0
    trace_l23_nonzero: int = 0

    # Prediction signal breakdown
    prediction_max: float = 0.0
    feedback_contribution: float = 0.0
    lateral_contribution: float = 0.0


@dataclass
class CortexDiagnostics:
    """Collects diagnostics from a running cortex region."""

    snapshot_interval: int = 100
    snapshots: list[Snapshot] = field(default_factory=list)

    # Activation diversity (accumulated per step, cheap)
    _column_counts: Counter = field(default_factory=Counter)
    _l4_neuron_window: list[int] = field(default_factory=list)
    _l23_neuron_window: list[int] = field(default_factory=list)
    _l4_l23_matches: int = 0
    _l4_l23_total: int = 0
    _unique_col_sets: list[frozenset] = field(default_factory=list)
    _burst_count: int = 0
    _precise_count: int = 0

    def step(self, t: int, region: SensoryRegion) -> None:
        """Call after each region.process(). Cheap per-step bookkeeping."""
        # Track which columns activated
        active_cols = np.nonzero(region.active_columns)[0]
        for c in active_cols:
            self._column_counts[int(c)] += 1

        # Track active neuron indices for diversity window
        l4_active = np.nonzero(region.active_l4)[0]
        l23_active = np.nonzero(region.active_l23)[0]
        self._l4_neuron_window.extend(int(i) for i in l4_active)
        self._l23_neuron_window.extend(int(i) for i in l23_active)

        # L4-L2/3 match rate: does L2/3 winner match L4 winner position?
        for col in active_cols:
            l4_winner = l4_active[l4_active // region.n_l4 == col]
            l23_winner = l23_active[l23_active // region.n_l23 == col]
            if len(l4_winner) > 0 and len(l23_winner) > 0:
                l4_pos = int(l4_winner[0]) % region.n_l4
                l23_pos = int(l23_winner[0]) % region.n_l23
                self._l4_l23_matches += int(l4_pos == l23_pos)
                self._l4_l23_total += 1

        # Track distinct column sets
        self._unique_col_sets.append(frozenset(int(c) for c in active_cols))

        # Track burst vs precise activations
        n_bursting = int(region.bursting_columns.sum())
        n_active = len(active_cols)
        self._burst_count += n_bursting
        self._precise_count += n_active - n_bursting

        # Periodic snapshot
        if t % self.snapshot_interval == 0:
            self.snapshots.append(self._take_snapshot(t, region))

    def _take_snapshot(self, t: int, region: SensoryRegion) -> Snapshot:
        snap = Snapshot(t=t)

        # Weight health
        for prefix, w in [
            ("ff", region.ff_weights),
            ("fb", region.fb_weights),
            ("lat", region.lateral_weights),
            ("l23_lat", region.l23_lateral_weights),
        ]:
            setattr(snap, f"{prefix}_mean", float(np.mean(w)))
            setattr(snap, f"{prefix}_std", float(np.std(w)))
            setattr(snap, f"{prefix}_max", float(np.max(w)))
            setattr(snap, f"{prefix}_sparsity", float(np.mean(w < 1e-6)))

        # Excitability vs voltage
        snap.excitability_l4_max = float(np.max(region.excitability_l4))
        snap.excitability_l4_mean = float(np.mean(region.excitability_l4))
        snap.voltage_l4_max = float(np.max(np.abs(region.voltage_l4)))

        # Trace health
        snap.trace_l4_mean = float(np.mean(region.trace_l4))
        snap.trace_l4_nonzero = int(np.count_nonzero(region.trace_l4 > 0.01))
        snap.trace_l23_mean = float(np.mean(region.trace_l23))
        snap.trace_l23_nonzero = int(np.count_nonzero(region.trace_l23 > 0.01))

        # Prediction signal breakdown (dendritic spike model)
        v = region.voltage_l4 * region.voltage_decay
        fb_signal = np.zeros_like(v)
        lat_signal = np.zeros_like(v)

        if region.active_l23.any():
            fb_raw = region.active_l23.astype(np.float64) @ region.fb_weights
            fb_signal = region.fb_boost * (fb_raw > region.fb_boost_threshold)

        if region.active_l4.any():
            lat_raw = region.active_l4.astype(np.float64) @ region.lateral_weights
            lat_signal = region.fb_boost * (lat_raw > region.fb_boost_threshold)

        snap.prediction_max = float(np.max(v + fb_signal + lat_signal))
        snap.feedback_contribution = float(np.max(fb_signal))
        snap.lateral_contribution = float(np.max(lat_signal))

        return snap

    def summary(self) -> dict:
        """Return summary statistics for the full run."""
        n_cols = max(self._column_counts.keys(), default=0) + 1
        col_hist = [self._column_counts.get(i, 0) for i in range(n_cols)]
        total_activations = sum(col_hist) or 1

        # Column entropy (uniformity of column usage)
        probs = np.array(col_hist, dtype=np.float64) / total_activations
        probs = probs[probs > 0]
        entropy = -float(np.sum(probs * np.log2(probs)))
        max_entropy = np.log2(n_cols) if n_cols > 0 else 1.0

        # Unique neurons over full run
        unique_l4 = len(set(self._l4_neuron_window))
        unique_l23 = len(set(self._l23_neuron_window))

        # Unique column-set diversity
        unique_col_sets = len(set(self._unique_col_sets))

        # L4-L2/3 match rate
        match_rate = (
            self._l4_l23_matches / self._l4_l23_total if self._l4_l23_total > 0 else 0.0
        )

        # Burst rate
        total_cols = self._burst_count + self._precise_count
        burst_rate = self._burst_count / total_cols if total_cols > 0 else 0.0

        return {
            "column_entropy": entropy,
            "column_entropy_ratio": entropy / max_entropy if max_entropy > 0 else 0,
            "unique_l4_neurons": unique_l4,
            "unique_l23_neurons": unique_l23,
            "unique_column_sets": unique_col_sets,
            "l4_l23_match_rate": match_rate,
            "burst_rate": burst_rate,
        }

    def print_report(self) -> None:
        """Print a human-readable diagnostic report."""
        if not self.snapshots:
            print("No snapshots collected.")
            return

        s = self.snapshots[-1]
        summ = self.summary()

        print("\n--- Cortex Diagnostics ---")

        print("\nWeight health (latest snapshot):")
        for name in ["ff", "fb", "lat", "l23_lat"]:
            m = getattr(s, f"{name}_mean")
            sd = getattr(s, f"{name}_std")
            mx = getattr(s, f"{name}_max")
            sp = getattr(s, f"{name}_sparsity")
            print(
                f"  {name:>8s}: mean={m:.4f} std={sd:.4f} max={mx:.4f} sparse={sp:.1%}"
            )

        print("\nExcitability vs voltage:")
        print(
            f"  excitability L4: max={s.excitability_l4_max:.1f}"
            f" mean={s.excitability_l4_mean:.1f}"
        )
        print(f"  voltage L4 max:  {s.voltage_l4_max:.4f}")
        ratio = s.excitability_l4_max / max(s.voltage_l4_max, 1e-10)
        if ratio > 10:
            print(f"  WARNING: excitability dominates voltage by {ratio:.0f}x")

        print("\nTrace health:")
        print(f"  L4:  mean={s.trace_l4_mean:.4f} nonzero={s.trace_l4_nonzero}")
        print(f"  L2/3: mean={s.trace_l23_mean:.4f} nonzero={s.trace_l23_nonzero}")

        print("\nPrediction signal:")
        print(f"  max prediction voltage: {s.prediction_max:.4f}")
        print(f"  feedback contribution:  {s.feedback_contribution:.4f}")
        print(f"  lateral contribution:   {s.lateral_contribution:.4f}")

        print("\nActivation diversity:")
        max_ent = np.log2(max(self._column_counts.keys(), default=1) + 1)
        ent = summ["column_entropy"]
        ent_r = summ["column_entropy_ratio"]
        print(f"  column entropy: {ent:.2f} / {max_ent:.2f} ({ent_r:.1%} of max)")
        print(f"  unique L4 neurons:  {summ['unique_l4_neurons']}")
        print(f"  unique L2/3 neurons: {summ['unique_l23_neurons']}")
        print(f"  unique column sets: {summ['unique_column_sets']}")
        print(f"  L4-L2/3 match rate: {summ['l4_l23_match_rate']:.1%}")
        print(f"  burst rate: {summ['burst_rate']:.1%}")
