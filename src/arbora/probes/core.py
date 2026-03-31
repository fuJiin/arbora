"""Core probe protocol and input-agnostic lamina probe.

Probe — minimal protocol for observing circuit state.
LaminaProbe — functional KPI accumulator. Measures input reception
  (recall/precision/sparsity) on the input lamina (L4 or L2/3) and
  association (eff_dim) on L2/3.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from arbora.snapshots.core import (
    AssociationSnapshot,
    InputSnapshot,
    LaminaRegionSnapshot,
)

if TYPE_CHECKING:
    from arbora.cortex.circuit import Circuit


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
    """Functional KPI accumulator. Works for any environment.

    Measures by function, not by lamina name:
    - Input reception (recall, precision, sparsity): on input_port (L4 or L2/3)
    - Association (eff_dim): on L2/3

    Walks circuit._regions, routes to the correct lamina per region.
    """

    name: str = "lamina"

    def __init__(self, *, l23_sample_interval: int = 10):
        # Per-region input reception accumulators
        self._input_predicted_total: dict[str, int] = defaultdict(int)
        self._input_predicted_correct: dict[str, int] = defaultdict(int)
        self._input_active_total: dict[str, int] = defaultdict(int)
        self._input_active_predicted: dict[str, int] = defaultdict(int)
        self._input_sparseness: dict[str, list[float]] = defaultdict(list)

        # Per-region association accumulators
        self._l23_samples: dict[str, list[np.ndarray]] = defaultdict(list)
        self._l23_sample_interval = l23_sample_interval
        self._step_count = 0

    def observe(self, circuit: Circuit, **kwargs) -> None:
        """Read circuit state, accumulate KPIs. Never writes to circuit."""
        self._step_count += 1

        for region_name, state in circuit._regions.items():
            region = state.region
            self._observe_input(region, region_name)
            self._observe_association(region, region_name)

    def snapshot(self) -> dict:
        """Compute current KPI values."""
        result: dict[str, LaminaRegionSnapshot] = {}

        all_regions = set(
            list(self._input_active_total.keys()) + list(self._l23_samples.keys())
        )

        for name in sorted(all_regions):
            result[name] = LaminaRegionSnapshot(
                input=self._snapshot_input(name),
                association=self._snapshot_association(name),
            )

        return result

    # -----------------------------------------------------------------------
    # Functional observe
    # -----------------------------------------------------------------------

    def _observe_input(self, region, region_name: str) -> None:
        """Accumulate input reception KPIs on the input lamina."""
        lamina = region.input_port
        predicted = lamina.predicted
        active = lamina.active

        # Recall: of active neurons, how many were predicted?
        n_active = int(active.sum())
        if n_active > 0:
            n_predicted_and_active = int((predicted & active).sum())
            self._input_active_total[region_name] += n_active
            self._input_active_predicted[region_name] += n_predicted_and_active

        # Precision: of predicted neurons, how many fired?
        n_predicted = int(predicted.sum())
        if n_predicted > 0:
            n_correct = int((predicted & active).sum())
            self._input_predicted_total[region_name] += n_predicted
            self._input_predicted_correct[region_name] += n_correct

        # Population sparseness (Treves-Rolls)
        r = active.astype(np.float64)
        mean_r = r.mean()
        mean_r2 = (r**2).mean()
        if mean_r2 > 0:
            self._input_sparseness[region_name].append(float(mean_r**2 / mean_r2))

    def _observe_association(self, region, region_name: str) -> None:
        """Sample L2/3 activations at configured interval."""
        if self._step_count % self._l23_sample_interval == 0:
            self._l23_samples[region_name].append(region.l23.active.astype(np.float64))

    # -----------------------------------------------------------------------
    # Functional snapshot
    # -----------------------------------------------------------------------

    def _snapshot_input(self, name: str) -> InputSnapshot:
        """Compute input reception KPIs from accumulated state."""
        total = self._input_active_total.get(name, 0)
        recall = self._input_active_predicted.get(name, 0) / total if total > 0 else 0.0

        pred_total = self._input_predicted_total.get(name, 0)
        precision = (
            self._input_predicted_correct.get(name, 0) / pred_total
            if pred_total > 0
            else 0.0
        )

        vals = self._input_sparseness.get(name, [])
        sparseness = float(np.mean(vals)) if vals else 0.0

        return InputSnapshot(recall=recall, precision=precision, sparseness=sparseness)

    def _snapshot_association(self, name: str) -> AssociationSnapshot:
        """Compute association KPIs from accumulated state."""
        samples = self._l23_samples.get(name, [])
        return AssociationSnapshot(eff_dim=_participation_ratio(samples))


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
