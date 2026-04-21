"""Canary-retrieval tracker for a HippocampalRegion.

Maintains a fixed set of canary observations, snapshots the CA3 state
they bind to at setup time, and later re-encodes them
non-destructively to measure how well HC retains those memories as
training continues.

Non-destructive measurement is achieved by saving and restoring the
HC's learned state (`ca3.lateral_weights`, `ca3.state`, `last_match`,
`last_ec_pattern`, `last_dg_pattern`, `output_port.firing_rate`)
around each canary re-encoding. This keeps ongoing training unaffected
by the measurement.

Usage::

    tracker = CanaryTracker(hc_region, canaries=[obs1, obs2, obs3])
    # ... run some training ...
    overlaps = tracker.measure()
    # overlaps[i] in [0, 1] = Jaccard of CA3 state vs. the snapshot
    # taken at canary-set creation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from arbora.hippocampus import HippocampalRegion


class CanaryTracker:
    """Measure CA3 retention for a held-out observation set.

    Parameters
    ----------
    region : HippocampalRegion
        The HC region to probe. Must already be wired; the canary set
        is encoded immediately on construction.
    canaries : sequence of np.ndarray
        Cortical-dim observations (typically S1 L2/3 firing patterns).
        Each is stored at setup time; subsequent `measure()` calls
        compare retrieval against the stored state.
    """

    def __init__(
        self,
        region: HippocampalRegion,
        canaries: list[np.ndarray],
    ):
        if not canaries:
            raise ValueError("canaries must be a non-empty list")
        self.region = region
        # Store copies so external mutation doesn't affect tracking.
        self.canaries: list[np.ndarray] = [np.asarray(c).copy() for c in canaries]
        self.initial_states: list[np.ndarray] = []

        # Prime each canary — writes lateral weights and state. This is
        # intentional: we want canaries to be treated as legitimate
        # memories the system should retain.
        for c in self.canaries:
            region.process(c)
            self.initial_states.append(region.ca3.state.copy())

    def measure(self) -> list[float]:
        """Re-encode each canary non-destructively and return Jaccard.

        Returns a list of per-canary overlap values in [0, 1], same
        order as `self.canaries`. 1.0 = CA3 state identical to the
        snapshot; 0.0 = disjoint.

        Non-destructive in two senses:
        - `CA3.learning_enabled` is toggled off during measurement so
          re-encoding does not apply fresh LTP to the lateral weights
          (which would contaminate the retention measurement itself —
          the canary's retrieval would succeed purely because we just
          re-bound it).
        - Transient state (`ca3.state`, `last_match`, intermediate
          patterns, output_port firing rate) is snapshotted and
          restored so ongoing training is unperturbed.
        """
        overlaps: list[float] = []
        ca3 = self.region.ca3
        was_enabled = ca3.learning_enabled
        ca3.learning_enabled = False
        try:
            for canary, initial in zip(self.canaries, self.initial_states, strict=True):
                with _FrozenHC(self.region):
                    self.region.process(canary)
                    current = self.region.ca3.state
                    union = int((current | initial).sum())
                    if union == 0:
                        overlap = 1.0
                    else:
                        overlap = float((current & initial).sum()) / union
                    overlaps.append(overlap)
        finally:
            ca3.learning_enabled = was_enabled
        return overlaps


class _FrozenHC:
    """Context manager that snapshots and restores learned HC state.

    Used by `CanaryTracker.measure` to make re-encoding a canary
    non-destructive. We restore every attribute that `HippocampalRegion
    .process()` mutates so the post-block state is byte-identical to
    the pre-block state.
    """

    def __init__(self, region: HippocampalRegion):
        self.region = region
        self._lat: np.ndarray | None = None
        self._state: np.ndarray | None = None
        self._last_match: float = 0.0
        self._ec_pat: np.ndarray | None = None
        self._dg_pat: np.ndarray | None = None
        self._out_rate: np.ndarray | None = None

    def __enter__(self) -> _FrozenHC:
        r = self.region
        self._lat = r.ca3.lateral_weights.copy()
        self._state = r.ca3.state.copy()
        self._last_match = r.last_match
        self._ec_pat = r.last_ec_pattern.copy()
        self._dg_pat = r.last_dg_pattern.copy()
        self._out_rate = r.output_port.firing_rate.copy()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        r = self.region
        assert self._lat is not None
        assert self._state is not None
        assert self._ec_pat is not None
        assert self._dg_pat is not None
        assert self._out_rate is not None
        r.ca3.lateral_weights[:] = self._lat
        r.ca3.state[:] = self._state
        r.last_match = self._last_match
        r.last_ec_pattern[:] = self._ec_pat
        r.last_dg_pattern[:] = self._dg_pat
        r.output_port.firing_rate[:] = self._out_rate
