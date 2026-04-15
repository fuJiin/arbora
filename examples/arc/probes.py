"""ARC-specific visual decodability probes.

Diagnostic probes that answer whether Arbor's cortical hierarchy is
learning useful visual representations for ARC-AGI-3 grids.

Three probes, all lightweight (numpy only, no external deps):

1. ChangeLocalizationProbe — do V1 bursts track grid changes?
2. TimerSelectivityProbe — do specific V1 columns track the timer?
3. RepresentationStabilityProbe — does V2 form stable object codes?

These live in examples/arc/ (not core framework) because they depend
on the ARC encoder layout (32x32 downsampled, encoding_width=18).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from examples.arc.encoder import _block_mode_pool

# ARC encoder constants (must match encoder.py)
_DOWNSAMPLED = 32
_N_CELLS = _DOWNSAMPLED * _DOWNSAMPLED  # 1024
_ENCODING_WIDTH = 18  # 16 color + 2 change
_N_COLORS = 16
# Timer occupies the bottom rows of the 64x64 grid.
# In downsampled (32x32) space: rows 30-31.
_TIMER_ROW_START = 30
_TIMER_ROW_END = 32  # exclusive


def _changed_cells(prev_down: np.ndarray | None, curr_down: np.ndarray) -> np.ndarray:
    """Return boolean mask (32x32) of which downsampled cells changed."""
    if prev_down is None:
        return np.ones((_DOWNSAMPLED, _DOWNSAMPLED), dtype=np.bool_)
    return prev_down != curr_down


# ---------------------------------------------------------------------------
# 1. Change Localization Probe
# ---------------------------------------------------------------------------


class ChangeLocalizationProbe:
    """Can we decode WHERE the grid changed from V1's burst pattern?

    Measures correlation between grid change magnitude and V1 burst
    magnitude. Since V1 columns tile the feature dimension (not spatial
    positions), this probe tracks:

    - Burst-change correlation: does burst fraction increase when more
      cells change? High correlation = V1 bursts are driven by novelty.
    - Per-step localization score: fraction of changed encoding bits
      that fall within bursting columns' receptive fields (ff_mask).

    Call observe() after each circuit.process() step.
    """

    def __init__(self, window: int = 50):
        self._window = window
        self._change_fractions: list[float] = []
        self._burst_fractions: list[float] = []
        self._localization_scores: list[float] = []
        self._prev_down: np.ndarray | None = None

    def observe(self, circuit, encoder, grid: np.ndarray) -> None:
        """Record change and burst statistics for one step.

        Args:
            circuit: The Circuit after process().
            encoder: ArcGridEncoder (for encoding_width layout).
            grid: The raw 64x64 grid for this step.
        """
        v1 = circuit.region("V1")

        # Compute which cells changed in downsampled space
        curr_down = _block_mode_pool(grid)
        changed = _changed_cells(self._prev_down, curr_down)
        self._prev_down = curr_down.copy()

        n_changed = int(changed.sum())
        change_frac = n_changed / _N_CELLS
        self._change_fractions.append(change_frac)

        # V1 burst fraction
        n_active = max(int(v1.active_columns.sum()), 1)
        n_bursting = int(v1.bursting_columns.sum())
        burst_frac = n_bursting / n_active
        self._burst_fractions.append(burst_frac)

        # Localization score: do bursting columns' receptive fields
        # overlap with the encoding positions that changed?
        # Build the set of encoding indices that correspond to changed cells.
        changed_flat = changed.ravel()
        changed_indices = np.nonzero(changed_flat)[0]

        if len(changed_indices) > 0 and n_bursting > 0:
            # Encoding bits for changed cells: for each changed cell,
            # the temporal-change bit is at cell_idx * encoding_width + N_COLORS
            changed_bits = set()
            for ci in changed_indices:
                # The "changed" bit in the encoding
                changed_bits.add(ci * _ENCODING_WIDTH + _N_COLORS)
                # Also the spatial color bits for the changed cell
                for b in range(_N_COLORS):
                    changed_bits.add(ci * _ENCODING_WIDTH + b)

            # Which columns are bursting?
            burst_cols = np.nonzero(v1.bursting_columns)[0]

            # Check ff_mask overlap: how many changed bits fall in burst
            # columns' receptive fields?
            if hasattr(v1, "ff_mask"):
                ff_mask = v1.ff_mask  # (input_dim, n_columns)
                # Bits covered by bursting columns
                burst_covered = set()
                for col in burst_cols:
                    col_bits = np.nonzero(ff_mask[:, col])[0]
                    burst_covered.update(int(b) for b in col_bits)

                overlap = len(changed_bits & burst_covered)
                score = overlap / len(changed_bits)
            else:
                # No ff_mask: fall back to simple burst fraction
                score = burst_frac
            self._localization_scores.append(score)
        else:
            self._localization_scores.append(0.0)

        # Keep only recent window
        if len(self._change_fractions) > self._window * 2:
            self._change_fractions = self._change_fractions[-self._window :]
            self._burst_fractions = self._burst_fractions[-self._window :]
            self._localization_scores = self._localization_scores[-self._window :]

    def snapshot(self) -> dict:
        """Return summary metrics."""
        n = len(self._change_fractions)
        if n < 2:
            return {
                "n_steps": n,
                "burst_change_corr": 0.0,
                "mean_localization": 0.0,
                "mean_burst_frac": 0.0,
                "mean_change_frac": 0.0,
            }

        recent_change = np.array(self._change_fractions[-self._window :])
        recent_burst = np.array(self._burst_fractions[-self._window :])
        recent_loc = np.array(self._localization_scores[-self._window :])

        # Pearson correlation between change fraction and burst fraction
        if np.std(recent_change) > 1e-8 and np.std(recent_burst) > 1e-8:
            corr = float(np.corrcoef(recent_change, recent_burst)[0, 1])
        else:
            corr = 0.0

        return {
            "n_steps": n,
            "burst_change_corr": corr,
            "mean_localization": float(np.mean(recent_loc)),
            "mean_burst_frac": float(np.mean(recent_burst)),
            "mean_change_frac": float(np.mean(recent_change)),
        }

    def reset(self) -> None:
        """Reset per-episode state."""
        self._change_fractions.clear()
        self._burst_fractions.clear()
        self._localization_scores.clear()
        self._prev_down = None


# ---------------------------------------------------------------------------
# 2. Timer Selectivity Probe
# ---------------------------------------------------------------------------


class TimerSelectivityProbe:
    """Do specific V1 columns consistently track the timer?

    The ARC-AGI-3 timer is encoded in the bottom rows of the grid
    (rows 60-63 raw, rows 30-31 in 32x32 downsampled space). It
    changes nearly every frame, producing a reliable temporal signal.

    This probe tracks which V1 columns burst preferentially on frames
    where the timer region changed, and identifies "timer-selective"
    columns: those with burst rate on timer frames >> non-timer frames.

    Call observe() after each circuit.process() step.
    """

    def __init__(self, selectivity_threshold: float = 2.0):
        """
        Args:
            selectivity_threshold: a column is "timer-selective" if its
                burst rate on timer-change frames is >= this multiple of
                its burst rate on non-timer-change frames.
        """
        self._selectivity_threshold = selectivity_threshold
        # Per-column: count of bursts on timer-change frames
        self._timer_bursts: dict[int, int] = defaultdict(int)
        # Per-column: count of bursts on non-timer-change frames
        self._nontimer_bursts: dict[int, int] = defaultdict(int)
        self._timer_frames = 0
        self._nontimer_frames = 0
        self._prev_down: np.ndarray | None = None

    def observe(self, circuit, encoder, grid: np.ndarray) -> None:
        """Record burst patterns relative to timer changes."""
        v1 = circuit.region("V1")

        curr_down = _block_mode_pool(grid)
        changed = _changed_cells(self._prev_down, curr_down)
        self._prev_down = curr_down.copy()

        # Did the timer region change?
        timer_changed = bool(changed[_TIMER_ROW_START:_TIMER_ROW_END, :].any())
        # Did anything OUTSIDE the timer change?
        non_timer_changed = bool(changed[:_TIMER_ROW_START, :].any())

        burst_cols = np.nonzero(v1.bursting_columns)[0]

        if timer_changed and not non_timer_changed:
            # Pure timer frame: only timer region changed
            self._timer_frames += 1
            for col in burst_cols:
                self._timer_bursts[int(col)] += 1
        elif not timer_changed:
            # Non-timer frame
            self._nontimer_frames += 1
            for col in burst_cols:
                self._nontimer_bursts[int(col)] += 1
        # Mixed frames (both timer and content change) are skipped —
        # they confound the selectivity measurement.

    def snapshot(self) -> dict:
        """Return timer selectivity metrics."""
        if self._timer_frames < 2 or self._nontimer_frames < 2:
            return {
                "timer_frames": self._timer_frames,
                "nontimer_frames": self._nontimer_frames,
                "n_selective": 0,
                "selective_columns": [],
                "mean_timer_burst_rate": 0.0,
                "mean_nontimer_burst_rate": 0.0,
            }

        all_cols = set(self._timer_bursts.keys()) | set(self._nontimer_bursts.keys())

        selective = []
        timer_rates = []
        nontimer_rates = []

        for col in sorted(all_cols):
            timer_rate = self._timer_bursts.get(col, 0) / self._timer_frames
            nontimer_rate = self._nontimer_bursts.get(col, 0) / self._nontimer_frames
            timer_rates.append(timer_rate)
            nontimer_rates.append(nontimer_rate)

            # Selectivity: timer burst rate >> non-timer burst rate
            if nontimer_rate > 0:
                ratio = timer_rate / nontimer_rate
            elif timer_rate > 0:
                ratio = float("inf")
            else:
                ratio = 0.0

            if ratio >= self._selectivity_threshold and timer_rate > 0.05:
                selective.append(
                    {
                        "column": col,
                        "timer_burst_rate": round(timer_rate, 3),
                        "nontimer_burst_rate": round(nontimer_rate, 3),
                        "selectivity_ratio": (
                            round(ratio, 1) if ratio != float("inf") else "inf"
                        ),
                    }
                )

        return {
            "timer_frames": self._timer_frames,
            "nontimer_frames": self._nontimer_frames,
            "n_selective": len(selective),
            "selective_columns": selective[:10],  # top 10
            "mean_timer_burst_rate": (
                float(np.mean(timer_rates)) if timer_rates else 0.0
            ),
            "mean_nontimer_burst_rate": (
                float(np.mean(nontimer_rates)) if nontimer_rates else 0.0
            ),
        }

    def reset(self) -> None:
        """Reset per-episode state."""
        self._timer_bursts.clear()
        self._nontimer_bursts.clear()
        self._timer_frames = 0
        self._nontimer_frames = 0
        self._prev_down = None


# ---------------------------------------------------------------------------
# 3. Representation Stability Probe
# ---------------------------------------------------------------------------


class RepresentationStabilityProbe:
    """Does V2 form stable object representations?

    Tracks V2's active column pattern frame-to-frame and computes
    Jaccard similarity (intersection / union of active column sets).
    High stability means V2 has learned consistent representations
    that don't flicker with every new frame.

    Call observe() after each circuit.process() step.
    """

    def __init__(self, *, window: int = 50, region_name: str = "V2"):
        self._window = window
        self._region_name = region_name
        self._prev_active: set[int] | None = None
        self._jaccard_history: list[float] = []

    def observe(self, circuit, encoder, grid: np.ndarray) -> None:
        """Record V2 active column overlap with previous step."""
        try:
            region = circuit.region(self._region_name)
        except KeyError:
            return  # V2 not in circuit (e.g. simpler architecture)

        active_cols = set(int(c) for c in np.nonzero(region.active_columns)[0])

        if self._prev_active is not None and len(active_cols) > 0:
            intersection = len(active_cols & self._prev_active)
            union = len(active_cols | self._prev_active)
            jaccard = intersection / union if union > 0 else 0.0
            self._jaccard_history.append(jaccard)

        self._prev_active = active_cols

        # Keep only recent window
        if len(self._jaccard_history) > self._window * 2:
            self._jaccard_history = self._jaccard_history[-self._window :]

    def snapshot(self) -> dict:
        """Return representation stability metrics."""
        n = len(self._jaccard_history)
        if n == 0:
            return {
                "region": self._region_name,
                "n_steps": 0,
                "mean_jaccard": 0.0,
                "std_jaccard": 0.0,
                "min_jaccard": 0.0,
                "max_jaccard": 0.0,
            }

        recent = np.array(self._jaccard_history[-self._window :])
        return {
            "region": self._region_name,
            "n_steps": n,
            "mean_jaccard": float(np.mean(recent)),
            "std_jaccard": float(np.std(recent)),
            "min_jaccard": float(np.min(recent)),
            "max_jaccard": float(np.max(recent)),
        }

    def reset(self) -> None:
        """Reset per-episode state."""
        self._prev_active = None
        self._jaccard_history.clear()


# ---------------------------------------------------------------------------
# Convenience: all probes in one bundle
# ---------------------------------------------------------------------------


class ArcProbeBundle:
    """Convenience wrapper that manages all three ARC probes together.

    Usage:
        probes = ArcProbeBundle()
        # in training loop, after circuit.process():
        probes.observe(circuit, encoder, grid)
        # at episode end:
        probes.print_report()
        probes.reset()
    """

    def __init__(self, *, window: int = 50):
        self.change = ChangeLocalizationProbe(window=window)
        self.timer = TimerSelectivityProbe()
        self.stability = RepresentationStabilityProbe(window=window)

    def observe(self, circuit, encoder, grid: np.ndarray) -> None:
        """Observe all probes. Call after circuit.process() each step."""
        self.change.observe(circuit, encoder, grid)
        self.timer.observe(circuit, encoder, grid)
        self.stability.observe(circuit, encoder, grid)

    def snapshot(self) -> dict:
        """Return combined metrics from all probes."""
        return {
            "change_localization": self.change.snapshot(),
            "timer_selectivity": self.timer.snapshot(),
            "representation_stability": self.stability.snapshot(),
        }

    def reset(self) -> None:
        """Reset all probes for a new episode."""
        self.change.reset()
        self.timer.reset()
        self.stability.reset()

    def print_report(self) -> None:
        """Print a human-readable probe report."""
        snap = self.snapshot()

        print("\n--- ARC Visual Probes ---")

        cl = snap["change_localization"]
        print(f"\n  Change Localization ({cl['n_steps']} steps):")
        print(f"    burst-change correlation: {cl['burst_change_corr']:+.3f}")
        print(f"    mean localization score:  {cl['mean_localization']:.3f}")
        print(
            f"    mean burst frac: {cl['mean_burst_frac']:.3f}, "
            f"mean change frac: {cl['mean_change_frac']:.3f}"
        )

        ts = snap["timer_selectivity"]
        print(
            f"\n  Timer Selectivity "
            f"({ts['timer_frames']} timer / {ts['nontimer_frames']} non-timer frames):"
        )
        print(f"    timer-selective columns: {ts['n_selective']}")
        if ts["selective_columns"]:
            for sc in ts["selective_columns"][:5]:
                print(
                    f"      col {sc['column']:3d}: "
                    f"timer={sc['timer_burst_rate']:.3f} "
                    f"non-timer={sc['nontimer_burst_rate']:.3f} "
                    f"ratio={sc['selectivity_ratio']}"
                )
        print(f"    mean timer burst rate:     {ts['mean_timer_burst_rate']:.3f}")
        print(f"    mean non-timer burst rate: {ts['mean_nontimer_burst_rate']:.3f}")

        rs = snap["representation_stability"]
        print(f"\n  V2 Representation Stability ({rs['n_steps']} steps):")
        print(
            f"    Jaccard: mean={rs['mean_jaccard']:.3f} "
            f"std={rs['std_jaccard']:.3f} "
            f"range=[{rs['min_jaccard']:.3f}, {rs['max_jaccard']:.3f}]"
        )
        if rs["mean_jaccard"] > 0.5:
            print("    -> STABLE representations (good)")
        elif rs["mean_jaccard"] > 0.2:
            print("    -> MODERATE stability")
        else:
            print("    -> UNSTABLE (V2 not yet forming consistent codes)")
