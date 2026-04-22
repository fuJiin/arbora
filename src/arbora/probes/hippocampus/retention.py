"""Memory-retention tracker for a HippocampalRegion.

Maintains a fixed set of reference patterns, snapshots the CA3 state
they bind to at setup time, and later re-encodes them non-destructively
to measure how well HC retains those memories as training continues.
The output is a forgetting curve: Jaccard overlap vs. the initial
snapshot for each pattern, over time.

Non-destructive measurement is achieved in two layers:
1. `CA3.learning_enabled` is toggled off during re-encoding so the
   measurement itself does not apply fresh LTP to the lateral weights
   (which would make the retention test trivially pass — the pattern
   would "retrieve" simply because we just re-bound it).
2. All transient HC state (`ca3.state`, `last_match`, intermediate
   patterns, `output_port.firing_rate`) is snapshotted and restored
   around each re-encoding so ongoing training is unperturbed.

Usage::

    tracker = RetentionTracker(hc_region, patterns=[obs1, obs2, obs3])
    # ... run some training ...
    overlaps = tracker.measure()
    # overlaps[i] in [0, 1] = Jaccard of CA3 state vs. the snapshot
    # taken at tracker construction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from arbora.hippocampus import HippocampalRegion


class RetentionTracker:
    """Measure CA3 memory retention for a fixed reference pattern set.

    Parameters
    ----------
    region : HippocampalRegion
        The HC region to probe. Must already be wired; the reference
        patterns are encoded immediately on construction so they're
        treated as legitimate memories the system should retain.
    patterns : sequence of np.ndarray
        Cortical-dim observations (typically T1 L2/3 firing patterns).
        Each is stored at setup time; subsequent `measure()` calls
        compare retrieval against the stored state.
    """

    def __init__(
        self,
        region: HippocampalRegion,
        patterns: list[np.ndarray],
    ):
        if not patterns:
            raise ValueError("patterns must be a non-empty list")
        self.region = region
        # Store copies so external mutation doesn't affect tracking.
        self.patterns: list[np.ndarray] = [np.asarray(p).copy() for p in patterns]
        self.initial_states: list[np.ndarray] = []

        # Prime each pattern — writes lateral weights and state. This is
        # intentional: we want these patterns to be treated as
        # legitimate memories the system should retain.
        for p in self.patterns:
            region.process(p)
            self.initial_states.append(region.ca3.state.copy())

    def measure(self) -> list[float]:
        """Re-encode each pattern non-destructively and return Jaccard.

        Returns a list of per-pattern overlap values in [0, 1], same
        order as `self.patterns`. 1.0 = CA3 state identical to the
        snapshot taken at construction; 0.0 = disjoint.

        Non-destructive in two senses — see the module docstring.
        """
        overlaps: list[float] = []
        ca3 = self.region.ca3
        was_enabled = ca3.learning_enabled
        ca3.learning_enabled = False
        try:
            for pattern, initial in zip(
                self.patterns, self.initial_states, strict=True
            ):
                with _FrozenHC(self.region):
                    self.region.process(pattern)
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

    Used by `RetentionTracker.measure` to make re-encoding a reference
    pattern non-destructive. Restores every attribute that
    `HippocampalRegion.process()` mutates so the post-block state is
    byte-identical to the pre-block state.
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
