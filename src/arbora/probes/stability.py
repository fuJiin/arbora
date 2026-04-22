"""Representation-stability tracker for cortical regions.

Cortical analog of `arbora.probes.hippocampus.RetentionTracker`:
primes with a fixed set of reference encodings, snapshots the region's
L2/3 active-column pattern for each, and later re-encodes non-
destructively to report Jaccard drift. The output is a "drift curve":
how much the region's response to a fixed stimulus has changed over
training.

Motivation: HC binds memories keyed on cortical L2/3 patterns, so if
those patterns drift as the cortex learns, HC retrieval breaks — the
keys no longer match what CA3 stored them against. This tracker
measures drift directly so it can be disambiguated from HC-internal
capacity failures (CA3 saturation, which is covered by
`HippocampalProbe` / `RetentionTracker`).

Non-destructive measurement
---------------------------
Implemented via `copy.deepcopy(region)` per measurement: the copy is
put into learning-disabled mode, the encoding is pushed through its
`process()`, the resulting L2/3 active pattern is read off, and the
copy is discarded. This avoids the fragile "save every field that
process() might mutate" pattern — `CorticalRegion` has many internal
state arrays (segment traces, prediction contexts, RNG state, etc.)
and missing any of them breaks measurement determinism.

Cost is O(N_refs) deepcopies per `measure()` call; for a moderate S1
(~1024 neurons, a few MB of weights) that's tens of MB of temporary
allocation per call. Acceptable given the tracker is used a handful
of times per ablation run, not per-step.

Usage
-----
::

    tracker = CortexStabilityTracker(s1_region, encodings=[enc1, enc2, ...])
    # ... run some training ...
    overlaps = tracker.measure()
    # overlaps[i] ∈ [0, 1] = Jaccard between current L2/3 active pattern
    # and the pattern at tracker construction.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from arbora.cortex.region import CorticalRegion


class CortexStabilityTracker:
    """Measure L2/3 representational drift for a cortical region.

    Parameters
    ----------
    region : CorticalRegion
        The region whose L2/3 stability we're tracking. Typically S1.
    encodings : list of np.ndarray
        Raw input encodings (dimensioned to `region.input_dim`). These
        are the fixed reference stimuli re-presented at each
        `measure()` call.
    """

    def __init__(
        self,
        region: CorticalRegion,
        encodings: list[np.ndarray],
    ):
        if not encodings:
            raise ValueError("encodings must be a non-empty list")
        self.region = region
        self.encodings: list[np.ndarray] = [np.asarray(e).copy() for e in encodings]
        self.initial_states: list[np.ndarray] = [
            self._measure_one(e) for e in self.encodings
        ]

    def measure(self) -> list[float]:
        """Re-encode each reference non-destructively; return Jaccard overlaps.

        Returns a list of per-encoding overlap values in [0, 1], same
        order as `self.encodings`. 1.0 = L2/3 active pattern identical
        to the initial snapshot; 0.0 = disjoint.
        """
        overlaps: list[float] = []
        for encoding, initial in zip(self.encodings, self.initial_states, strict=True):
            current = self._measure_one(encoding)
            union = int((current | initial).sum())
            overlap = float((current & initial).sum()) / union if union > 0 else 1.0
            overlaps.append(overlap)
        return overlaps

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _measure_one(self, encoding: np.ndarray) -> np.ndarray:
        """Run `encoding` through a deepcopy of the region; return L2/3 active.

        Deepcopy keeps the real region bit-for-bit identical — no state
        of any kind leaks out of this method, including RNG state,
        segment internals, prediction contexts, and eligibility traces.
        """
        region_copy = copy.deepcopy(self.region)
        region_copy.learning_enabled = False
        region_copy.process(encoding)
        return region_copy.l23.active.copy()
