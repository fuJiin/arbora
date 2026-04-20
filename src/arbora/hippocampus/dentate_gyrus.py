"""Dentate gyrus: k-WTA pattern separation between EC and CA3.

DG granule cells implement sparse pattern separation — orthogonalizing
inputs so that similar EC codes produce non-overlapping DG codes. This
creates the episodic distinctiveness that CA3's attractor dynamics can
then bind and complete.

Biology:
  - DG is overprovisioned relative to CA3 (~4-10x in rodents). This
    creates headroom for graceful capacity degradation.
  - Pattern separation is achieved via k-WTA: strong lateral inhibition
    restricts activity to the most-driven ~1-2% of granule cells.
  - DG neurogenesis (granule cells born in adulthood) is deferred
    to v2; v1 uses a fixed projection.

DG maturation (sparsity_schedule):
  In altricial species, DG is dramatically immature at birth. Early
  episodes are coarse and high-overlap; as DG matures, pattern
  separation sharpens. Computationally: if HC activates before cortical
  representations are fully stable, low sparsity (high k → more overlap)
  produces blurry associative support robust to drift. As cortex
  stabilizes, decreasing k sharpens separation.

  Exposed via `k_schedule: Callable[[int], int] | None`. If provided,
  `forward(pattern, step=...)` calls it to determine k dynamically.
  Default: fixed k at construction time.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from arbora.hippocampus._kwta import kwta


class DentateGyrus:
    """Sparse pattern separation via k-WTA on a fixed random projection.

    Parameters
    ----------
    input_dim : int
        EC output dimension (upstream).
    output_dim : int
        DG granule cell count. Large for overprovisioning (~4-10x CA3).
    k : int | None
        Fixed number of active units in output. Required if `k_schedule`
        is None.
    k_schedule : Callable[[int], int] | None
        If provided, called with step index to determine k dynamically.
        Overrides fixed k. Use for DG maturation ramps.
    seed : int
        Random seed for projection matrix.

    Attributes
    ----------
    weights : np.ndarray, shape (input_dim, output_dim)
        Fixed Gaussian random projection.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        k: int | None = None,
        k_schedule: Callable[[int], int] | None = None,
        seed: int = 0,
    ):
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError(
                f"dims must be positive; "
                f"got input_dim={input_dim}, output_dim={output_dim}"
            )
        if k is None and k_schedule is None:
            raise ValueError("either k or k_schedule must be provided")
        if k is not None and (k < 1 or k > output_dim):
            raise ValueError(
                f"k must be in [1, output_dim]; got k={k}, output_dim={output_dim}"
            )

        self.input_dim = input_dim
        self.output_dim = output_dim
        self._k = k
        self._k_schedule = k_schedule

        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(input_dim)
        self.weights = rng.normal(0.0, scale, size=(input_dim, output_dim))

    def k_at(self, step: int | None) -> int:
        """Active-unit count at the given step.

        If a schedule is configured, calls it with `step` (or 0 if None).
        Otherwise returns the fixed k. Result is clamped to [1, output_dim].
        """
        if self._k_schedule is not None:
            k = self._k_schedule(step if step is not None else 0)
            return max(1, min(k, self.output_dim))
        assert self._k is not None  # guaranteed by __init__
        return self._k

    def forward(self, ec_pattern: np.ndarray, *, step: int | None = None) -> np.ndarray:
        """Project EC input to a sparse binary DG code.

        Parameters
        ----------
        ec_pattern : np.ndarray
            EC output. Any shape that flattens to `input_dim`.
        step : int | None
            Training step for schedule lookup; ignored if no k_schedule.

        Returns
        -------
        np.ndarray, dtype=bool, shape=(output_dim,)
            Sparse binary vector with exactly `k_at(step)` active units.
        """
        flat = ec_pattern.reshape(-1).astype(np.float64, copy=False)
        if flat.shape[0] != self.input_dim:
            raise ValueError(
                f"input has {flat.shape[0]} elements, expected {self.input_dim}"
            )
        k = self.k_at(step)
        projected = flat @ self.weights
        return kwta(projected, k)
