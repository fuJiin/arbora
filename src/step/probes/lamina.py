"""Per-lamina KPI probes for cortical circuit measurement.

Two probe classes:

LaminaProbe — input-agnostic, reads only lamina/region state:
  L4: prediction recall, precision, population sparseness
  L2/3: effective dimensionality (participation ratio)
  L5: stub (deferred until multi-region testing)

ChatLaminaProbe — extends with stimulus-labeled metrics:
  L2/3: linear probe accuracy, context discrimination

Probes observe circuit state after each process() call. They never
write to the circuit. The runner owns the probe and calls observe().
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from step.cortex.circuit import Circuit


# ---------------------------------------------------------------------------
# Protocols
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


@runtime_checkable
class ChatProbe(Probe, Protocol):
    """Probe with dialogue/episode boundary support."""

    def boundary(self) -> None:
        """Reset per-episode state (e.g., bigram tracking)."""
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

            # --- L4 KPIs ---
            if region.n_l4 > 0:
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

            # --- L2/3 KPIs (effective dimensionality) ---
            if region.n_l23 > 0 and self._step_count % self._l23_sample_interval == 0:
                self._l23_samples[region_name].append(
                    region.l23.active.astype(np.float64)
                )

    def snapshot(self) -> dict:
        """Compute current KPI values."""
        result = {}

        all_regions = set(
            list(self._l4_active_total.keys()) + list(self._l23_samples.keys())
        )

        for name in sorted(all_regions):
            l4_kpis = {}
            l23_kpis = {}

            # L4 recall
            total = self._l4_active_total.get(name, 0)
            if total > 0:
                l4_kpis["recall"] = self._l4_active_predicted.get(name, 0) / total
            else:
                l4_kpis["recall"] = 0.0

            # L4 precision
            pred_total = self._l4_predicted_total.get(name, 0)
            if pred_total > 0:
                l4_kpis["precision"] = (
                    self._l4_predicted_correct.get(name, 0) / pred_total
                )
            else:
                l4_kpis["precision"] = 0.0

            # L4 sparseness
            vals = self._l4_sparseness.get(name, [])
            l4_kpis["sparseness"] = float(np.mean(vals)) if vals else 0.0

            # L2/3 effective dimensionality (participation ratio)
            samples = self._l23_samples.get(name, [])
            l23_kpis["eff_dim"] = _participation_ratio(samples)

            result[name] = {"l4": l4_kpis, "l23": l23_kpis}

        return result


# ---------------------------------------------------------------------------
# ChatLaminaProbe — adds stimulus-labeled metrics
# ---------------------------------------------------------------------------


class ChatLaminaProbe(LaminaProbe):
    """Extends LaminaProbe with chat-specific L2/3 KPIs.

    Requires stimulus_id kwarg in observe() for:
    - Linear probe accuracy (SGDClassifier on frozen L2/3 -> token)
    - Context discrimination (Jaccard distance across contexts)
    """

    name: str = "chat_lamina"

    def __init__(
        self,
        *,
        l23_sample_interval: int = 10,
        linear_probe_fit_interval: int = 5000,
        linear_probe_window: int = 3000,
        ctx_disc_min_contexts: int = 3,
    ):
        super().__init__(l23_sample_interval=l23_sample_interval)
        self._fit_interval = linear_probe_fit_interval
        self._probe_window = linear_probe_window
        self._ctx_min = ctx_disc_min_contexts

        # Per-region linear probe state
        self._probe_X: dict[str, list[np.ndarray]] = defaultdict(list)
        self._probe_y: dict[str, list[int]] = defaultdict(list)
        self._probe_accuracy: dict[str, float] = {}

        # Per-region context discrimination state
        self._prev_token: int | None = None
        self._bigram_patterns: dict[
            str, dict[tuple[int, int], list[frozenset[int]]]
        ] = defaultdict(lambda: defaultdict(list))
        self._token_contexts: dict[str, dict[int, set[int]]] = defaultdict(
            lambda: defaultdict(set)
        )

    def observe(self, circuit: Circuit, **kwargs) -> None:
        """Observe with optional stimulus_id for chat-specific KPIs."""
        super().observe(circuit, **kwargs)

        stimulus_id = kwargs.get("stimulus_id")
        if stimulus_id is None:
            return

        for region_name, state in circuit._regions.items():
            region = state.region
            if region.n_l23 == 0:
                continue

            l23_active = region.l23.active

            # Linear probe: accumulate (activation, label) pairs
            self._probe_X[region_name].append(l23_active.astype(np.float32))
            self._probe_y[region_name].append(stimulus_id)
            # Cap buffer
            window = self._probe_window * 2
            if len(self._probe_X[region_name]) > window:
                self._probe_X[region_name] = self._probe_X[region_name][
                    -self._probe_window :
                ]
                self._probe_y[region_name] = self._probe_y[region_name][
                    -self._probe_window :
                ]

            # Context discrimination: bigram tracking
            if self._prev_token is not None:
                neurons = frozenset(int(n) for n in np.nonzero(l23_active)[0])
                key = (self._prev_token, stimulus_id)
                self._bigram_patterns[region_name][key].append(neurons)
                self._token_contexts[region_name][stimulus_id].add(self._prev_token)

        self._prev_token = stimulus_id

        # Periodic linear probe fit
        if self._step_count % self._fit_interval == 0:
            self._fit_linear_probes()

    def boundary(self) -> None:
        """Reset per-dialogue state."""
        self._prev_token = None

    def snapshot(self) -> dict:
        """Extend parent snapshot with chat-specific L2/3 KPIs."""
        result = super().snapshot()

        # Ensure we have a recent fit
        self._fit_linear_probes()

        for region_name in result:
            # Linear probe accuracy
            result[region_name]["l23"]["linear_probe"] = self._probe_accuracy.get(
                region_name, 0.0
            )

            # Context discrimination
            result[region_name]["l23"]["ctx_disc"] = self._compute_ctx_disc(region_name)

        return result

    def _fit_linear_probes(self) -> None:
        """Fit SGD linear classifiers on accumulated L2/3 data."""
        try:
            from sklearn.linear_model import SGDClassifier
        except ImportError:
            return

        for region_name in list(self._probe_X.keys()):
            X_list = self._probe_X[region_name]
            y_list = self._probe_y[region_name]
            if len(X_list) < 100:
                continue

            X = np.array(X_list[-self._probe_window :])
            y = np.array(y_list[-self._probe_window :])

            # 80/20 split
            split = int(len(X) * 0.8)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            if len(set(y_test)) < 2:
                continue

            clf = SGDClassifier(
                loss="modified_huber", max_iter=100, random_state=42, n_jobs=1
            )
            clf.fit(X_train, y_train)
            self._probe_accuracy[region_name] = float(clf.score(X_test, y_test))

    def _compute_ctx_disc(self, region_name: str) -> float:
        """Mean Jaccard distance for same token across different contexts."""
        bigrams = self._bigram_patterns.get(region_name, {})
        contexts = self._token_contexts.get(region_name, {})
        if not bigrams:
            return 0.0

        rng = np.random.default_rng(42)
        all_dists: list[float] = []

        for tid, ctx_set in contexts.items():
            if len(ctx_set) < self._ctx_min:
                continue
            # Collect all patterns for this token across contexts
            patterns: list[frozenset[int]] = []
            for (_prev, t), pats in bigrams.items():
                if t == tid:
                    patterns.extend(pats)
            if len(patterns) < self._ctx_min:
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
