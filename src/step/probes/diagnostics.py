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

    # Prediction quality diagnostics
    n_predicted_neurons: int = 0

    # Dendritic segment health
    fb_seg_perm_mean: float = 0.0
    fb_seg_connected_frac: float = 0.0
    lat_seg_perm_mean: float = 0.0
    lat_seg_connected_frac: float = 0.0
    n_active_fb_segments: int = 0
    n_active_lat_segments: int = 0

    # L2/3 segment health
    l23_seg_perm_mean: float = 0.0
    l23_seg_connected_frac: float = 0.0
    n_active_l23_segments: int = 0
    n_predicted_l23: int = 0

    # Apical gain health (S2 → S1 feedback)
    apical_gain_mean: float = 0.0
    apical_gain_max: float = 0.0


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

    # Prediction diversity (accumulated per step)
    _unique_prediction_sets: list[frozenset] = field(default_factory=list)
    _prediction_correct_neuron: int = 0
    _prediction_correct_column: int = 0
    _prediction_total: int = 0

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

        # Track prediction diversity: what did predicted_l4 look like?
        predicted_neurons = np.nonzero(region.predicted_l4)[0]
        self._unique_prediction_sets.append(
            frozenset(int(i) for i in predicted_neurons)
        )

        # Prediction-activation alignment
        if len(predicted_neurons) > 0:
            self._prediction_total += 1
            predicted_set = set(int(i) for i in predicted_neurons)
            active_set = set(int(i) for i in l4_active)
            # Neuron-level: did any predicted neuron actually fire?
            if predicted_set & active_set:
                self._prediction_correct_neuron += 1
            # Column-level: did predicted columns match active columns?
            predicted_cols = set(i // region.n_l4 for i in predicted_set)
            active_col_set = set(int(c) for c in active_cols)
            if predicted_cols & active_col_set:
                self._prediction_correct_column += 1

        # Periodic snapshot
        if t % self.snapshot_interval == 0:
            self.snapshots.append(self._take_snapshot(t, region))

    def _take_snapshot(self, t: int, region: SensoryRegion) -> Snapshot:
        snap = Snapshot(t=t)

        # Weight health
        for prefix, w in [
            ("ff", region.ff_weights),
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

        # Prediction state from dendritic segments
        snap.n_predicted_neurons = int(region.predicted_l4.sum())

        # Dendritic segment health
        if hasattr(region, "fb_seg_perm"):
            fb_perm = region.fb_seg_perm
            lat_perm = region.lat_seg_perm
            snap.fb_seg_perm_mean = float(np.mean(fb_perm))
            snap.fb_seg_connected_frac = float(np.mean(fb_perm > region.perm_threshold))
            snap.lat_seg_perm_mean = float(np.mean(lat_perm))
            snap.lat_seg_connected_frac = float(
                np.mean(lat_perm > region.perm_threshold)
            )

            # Count active segments (would fire given current activity)
            if region.active_l23.any():
                fb_active = region.active_l23[region.fb_seg_indices]
                fb_conn = fb_perm > region.perm_threshold
                fb_counts = (fb_active & fb_conn).sum(axis=2)
                snap.n_active_fb_segments = int(
                    (fb_counts >= region.seg_activation_threshold).sum()
                )
            if region.active_l4.any():
                lat_active = region.active_l4[region.lat_seg_indices]
                lat_conn = lat_perm > region.perm_threshold
                lat_counts = (lat_active & lat_conn).sum(axis=2)
                snap.n_active_lat_segments = int(
                    (lat_counts >= region.seg_activation_threshold).sum()
                )

        # L2/3 lateral segment health
        if hasattr(region, "l23_seg_perm"):
            l23_perm = region.l23_seg_perm
            snap.l23_seg_perm_mean = float(np.mean(l23_perm))
            snap.l23_seg_connected_frac = float(
                np.mean(l23_perm > region.perm_threshold)
            )
            snap.n_predicted_l23 = int(region.predicted_l23.sum())

            if region.active_l23.any():
                l23_active = region.active_l23[region.l23_seg_indices]
                l23_conn = l23_perm > region.perm_threshold
                l23_counts = (l23_active & l23_conn).sum(axis=2)
                snap.n_active_l23_segments = int(
                    (l23_counts >= region.seg_activation_threshold).sum()
                )

        # Apical gain health
        if region.has_apical and region._apical_gain_weights is not None:
            snap.apical_gain_mean = float(np.mean(region._apical_gain_weights))
            snap.apical_gain_max = float(np.max(region._apical_gain_weights))

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

        # Prediction diversity
        unique_pred_sets = len(set(self._unique_prediction_sets))
        pred_neuron_rate = (
            self._prediction_correct_neuron / self._prediction_total
            if self._prediction_total > 0
            else 0.0
        )
        pred_column_rate = (
            self._prediction_correct_column / self._prediction_total
            if self._prediction_total > 0
            else 0.0
        )

        return {
            "column_entropy": entropy,
            "column_entropy_ratio": entropy / max_entropy if max_entropy > 0 else 0,
            "unique_l4_neurons": unique_l4,
            "unique_l23_neurons": unique_l23,
            "unique_column_sets": unique_col_sets,
            "l4_l23_match_rate": match_rate,
            "burst_rate": burst_rate,
            "unique_prediction_sets": unique_pred_sets,
            "prediction_hit_neuron": pred_neuron_rate,
            "prediction_hit_column": pred_column_rate,
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
        for name in ["ff", "l23_lat"]:
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

        print("\nPrediction quality:")
        print(f"  predicted neurons:       {s.n_predicted_neurons}")
        print(f"  unique prediction sets:  {summ['unique_prediction_sets']}")
        print(f"  hit rate (neuron):       {summ['prediction_hit_neuron']:.1%}")
        print(f"  hit rate (column):       {summ['prediction_hit_column']:.1%}")

        print("\nDendritic segments:")
        print(
            f"  fb: perm_mean={s.fb_seg_perm_mean:.4f}"
            f" connected={s.fb_seg_connected_frac:.1%}"
            f" active_segs={s.n_active_fb_segments}"
        )
        print(
            f"  lat: perm_mean={s.lat_seg_perm_mean:.4f}"
            f" connected={s.lat_seg_connected_frac:.1%}"
            f" active_segs={s.n_active_lat_segments}"
        )

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
