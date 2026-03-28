"""Core probe protocol and input-agnostic lamina probe.

Probe — minimal protocol for observing circuit state.
LaminaProbe — per-lamina KPI accumulator (L4 prediction/sparseness,
  L2/3 dimensionality).
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from step.snapshots.core import L4Snapshot, L23Snapshot, LaminaRegionSnapshot

if TYPE_CHECKING:
    from step.cortex.circuit import Circuit


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Probe(Protocol):
    """Minimal probe interface. Input-agnostic."""

    name: str

    def observe(self, circuit: Circuit, **kwargs) -> None:
        """Observe circuit state after process(). Read-only."""
        ...

    def snapshot(self) -> dict:
        """Point-in-time KPI values. Computed lazily where possible."""
        ...


# ---------------------------------------------------------------------------
# LaminaProbe — input-agnostic
# ---------------------------------------------------------------------------


class LaminaProbe:
    """Per-lamina KPI accumulator. Works for any environment.

    Walks circuit._regions, inspects each region's laminae, accumulates
    prediction recall/precision/sparseness (L4) and effective
    dimensionality (L2/3). Tracks per (region_name, lamina_id).
    """

    name: str = "lamina"

    def __init__(self, *, l23_sample_interval: int = 10):
        # Per-region L4 accumulators
        self._l4_predicted_total: dict[str, int] = defaultdict(int)
        self._l4_predicted_correct: dict[str, int] = defaultdict(int)
        self._l4_active_total: dict[str, int] = defaultdict(int)
        self._l4_active_predicted: dict[str, int] = defaultdict(int)
        self._l4_sparseness: dict[str, list[float]] = defaultdict(list)

        # Per-region L2/3 accumulators
        self._l23_samples: dict[str, list[np.ndarray]] = defaultdict(list)
        self._l23_sample_interval = l23_sample_interval
        self._step_count = 0

    def observe(self, circuit: Circuit, **kwargs) -> None:
        """Read circuit state, accumulate KPIs. Never writes to circuit."""
        self._step_count += 1

        for region_name, state in circuit._regions.items():
            region = state.region

            if region.n_l4 > 0:
                self._observe_l4(region, region_name)

            if region.n_l23 > 0:
                self._observe_l23(region, region_name)

    def snapshot(self) -> dict[str, LaminaRegionSnapshot]:
        """Compute current KPI values."""
        result: dict[str, LaminaRegionSnapshot] = {}

        all_regions = set(
            list(self._l4_active_total.keys()) + list(self._l23_samples.keys())
        )

        for name in sorted(all_regions):
            result[name] = LaminaRegionSnapshot(
                l4=self._snapshot_l4(name),
                l23=self._snapshot_l23(name),
            )

        return result

    # -----------------------------------------------------------------------
    # Per-lamina observe
    # -----------------------------------------------------------------------

    def _observe_l4(self, region, region_name: str) -> None:
        """Accumulate L4 prediction recall, precision, and sparseness."""
        l4 = region.l4
        predicted = l4.predicted
        active = l4.active

        # Recall: of active neurons, how many were predicted?
        n_active = int(active.sum())
        if n_active > 0:
            n_predicted_and_active = int((predicted & active).sum())
            self._l4_active_total[region_name] += n_active
            self._l4_active_predicted[region_name] += n_predicted_and_active

        # Precision: of predicted neurons, how many fired?
        n_predicted = int(predicted.sum())
        if n_predicted > 0:
            n_correct = int((predicted & active).sum())
            self._l4_predicted_total[region_name] += n_predicted
            self._l4_predicted_correct[region_name] += n_correct

        # Population sparseness (Treves-Rolls)
        r = active.astype(np.float64)
        mean_r = r.mean()
        mean_r2 = (r**2).mean()
        if mean_r2 > 0:
            self._l4_sparseness[region_name].append(float(mean_r**2 / mean_r2))

    def _observe_l23(self, region, region_name: str) -> None:
        """Sample L2/3 activations at configured interval."""
        if self._step_count % self._l23_sample_interval == 0:
            self._l23_samples[region_name].append(region.l23.active.astype(np.float64))

    # -----------------------------------------------------------------------
    # Per-lamina snapshot
    # -----------------------------------------------------------------------

    def _snapshot_l4(self, name: str) -> L4Snapshot:
        """Compute L4 KPIs from accumulated state."""
        total = self._l4_active_total.get(name, 0)
        recall = self._l4_active_predicted.get(name, 0) / total if total > 0 else 0.0

        pred_total = self._l4_predicted_total.get(name, 0)
        precision = (
            self._l4_predicted_correct.get(name, 0) / pred_total
            if pred_total > 0
            else 0.0
        )

        vals = self._l4_sparseness.get(name, [])
        sparseness = float(np.mean(vals)) if vals else 0.0

        return L4Snapshot(recall=recall, precision=precision, sparseness=sparseness)

    def _snapshot_l23(self, name: str) -> L23Snapshot:
        """Compute L2/3 KPIs from accumulated state."""
        samples = self._l23_samples.get(name, [])
        return L23Snapshot(eff_dim=_participation_ratio(samples))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _participation_ratio(activations: list[np.ndarray]) -> float:
    """Effective dimensionality via participation ratio.

    PR = (sum(eigenvalues))^2 / sum(eigenvalues^2)
    High = rich representation. Low = collapsed.
    """
    if len(activations) < 10:
        return 0.0
    X = np.array(activations, dtype=np.float64)
    X -= X.mean(axis=0)
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    lambdas = s**2 / (len(activations) - 1)
    sum_l = lambdas.sum()
    sum_l2 = (lambdas**2).sum()
    if sum_l2 < 1e-12:
        return 0.0
    return float(sum_l**2 / sum_l2)
