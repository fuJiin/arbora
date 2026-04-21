"""Mechanistic probe for HippocampalRegion.

Records per-step observables that verify HC's design properties hold
on real inputs (not just the synthetic inputs covered by unit tests).

Measurements captured per step
------------------------------
- `ec_active` / `dg_active` / `ca3_active`: active-unit counts on
  each layer. Confirms sparsity matches spec and doesn't collapse.
- `ca1_match`: cosine similarity between CA3 and direct-EC drives at
  CA1. Should be high on revisits of bound observations, low on novel.
- `ca3_lateral_sum` / `_nnz` / `_sat_frac`: bulk statistics on the
  learned lateral weight matrix. Tracks capacity accumulation and
  clip-saturation over training.
- `ec_hash` / `ca3_hash`: content hashes used by `snapshot()` to
  detect revisits and measure CA3 attractor stability.

Summary stats (from `snapshot()`)
---------------------------------
- Mean active counts across all steps.
- `ca3_revisit_stability`: among EC patterns seen more than once,
  fraction whose CA3 state matched the first-visit state. Aggregated
  across all revisiting patterns. Values near 1.0 mean CA3 is reliably
  pattern-completing to the stored attractor; lower values mean the
  attractor is drifting or getting overwritten.
- `ca1_match_on_revisit_vs_first`: mean CA1 match on EC revisits minus
  mean on first-visits. Positive values mean CA1 successfully tags
  familiar inputs.

Usage::

    probe = HippocampalProbe()
    harness = MiniGridHarness(env, agent, probes=[probe])
    result = harness.run()
    metrics = probe.snapshot()
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from arbora.hippocampus import HippocampalRegion

if TYPE_CHECKING:
    from arbora.cortex.circuit import Circuit


class HippocampalProbe:
    """Per-step mechanistic diagnostics for a HippocampalRegion in a Circuit.

    Duck-typed against the probe protocol used by MiniGridHarness:
    implements `name`, `observe(circuit, **kwargs)`, `snapshot()`.
    Silently no-ops on circuits that do not contain a HippocampalRegion.

    Parameters
    ----------
    max_steps : int | None
        Optional cap on the per-step log length. When the cap is hit,
        further steps are summarized into rolling counters but not
        appended to the step log. Default None = unbounded.
    """

    name: str = "hippocampus"

    def __init__(self, *, max_steps: int | None = None):
        self._max_steps = max_steps
        self._steps: list[dict] = []
        self._region: HippocampalRegion | None = None
        # Rolling accumulators so summary stats work even when step log
        # is capped.
        self._n_observed: int = 0
        self._ec_active_sum: int = 0
        self._dg_active_sum: int = 0
        self._ca3_active_sum: int = 0
        self._ca1_match_sum: float = 0.0

    def observe(self, circuit: Circuit, **kwargs) -> None:
        """Record one step of HC state. No-op if no HC region present."""
        if self._region is None:
            self._region = self._find_hc(circuit)
            if self._region is None:
                return
        hc = self._region

        ec_active = int(hc.last_ec_pattern.sum())
        dg_active = int(hc.last_dg_pattern.sum())
        ca3_active = int(hc.ca3.state.sum())
        ca1_match = float(hc.last_match)

        lat = hc.ca3.lateral_weights
        lat_nnz = int((lat > 0).sum())
        lat_sum = float(lat.sum())
        total = lat.size
        lat_sat_frac = float((lat >= 1.0).sum() / total) if total > 0 else 0.0

        self._n_observed += 1
        self._ec_active_sum += ec_active
        self._dg_active_sum += dg_active
        self._ca3_active_sum += ca3_active
        self._ca1_match_sum += ca1_match

        if self._max_steps is None or len(self._steps) < self._max_steps:
            self._steps.append(
                {
                    "step": int(kwargs.get("step", -1)),
                    "ec_active": ec_active,
                    "dg_active": dg_active,
                    "ca3_active": ca3_active,
                    "ca1_match": ca1_match,
                    "ca3_lateral_sum": lat_sum,
                    "ca3_lateral_nnz": lat_nnz,
                    "ca3_lateral_sat_frac": lat_sat_frac,
                    "ec_hash": hash(bytes(hc.last_ec_pattern)),
                    "ca3_hash": hash(bytes(hc.ca3.state)),
                }
            )

    def snapshot(self) -> dict:
        """Return the per-step log plus aggregate summary stats."""
        summary = self._summarize()
        return {"steps": list(self._steps), "summary": summary}

    def _summarize(self) -> dict:
        if self._n_observed == 0:
            return {"n_steps": 0}

        n = self._n_observed
        summary: dict[str, float | int] = {
            "n_steps": n,
            "mean_ec_active": self._ec_active_sum / n,
            "mean_dg_active": self._dg_active_sum / n,
            "mean_ca3_active": self._ca3_active_sum / n,
            "mean_ca1_match": self._ca1_match_sum / n,
        }
        if self._steps:
            last = self._steps[-1]
            summary["final_ca3_lateral_sum"] = last["ca3_lateral_sum"]
            summary["final_ca3_lateral_nnz"] = last["ca3_lateral_nnz"]
            summary["final_ca3_lateral_sat_frac"] = last["ca3_lateral_sat_frac"]

        # Revisit-based metrics — only meaningful if the step log has
        # enough entries to contain repeats.
        visits_by_ec: dict[int, list[dict]] = defaultdict(list)
        for s in self._steps:
            visits_by_ec[s["ec_hash"]].append(s)

        revisit_matches = 0
        revisit_total = 0
        revisit_match_sum = 0.0
        first_visit_match_sum = 0.0
        n_first_visits = 0
        for _ec_hash, visits in visits_by_ec.items():
            if not visits:
                continue
            first = visits[0]
            first_visit_match_sum += first["ca1_match"]
            n_first_visits += 1
            if len(visits) < 2:
                continue
            for v in visits[1:]:
                if v["ca3_hash"] == first["ca3_hash"]:
                    revisit_matches += 1
                revisit_total += 1
                revisit_match_sum += v["ca1_match"]

        if revisit_total > 0:
            summary["ca3_revisit_stability"] = revisit_matches / revisit_total
            summary["mean_ca1_match_revisit"] = revisit_match_sum / revisit_total
        if n_first_visits > 0:
            summary["mean_ca1_match_first_visit"] = (
                first_visit_match_sum / n_first_visits
            )
        if revisit_total > 0 and n_first_visits > 0:
            summary["ca1_match_revisit_minus_first"] = (
                summary["mean_ca1_match_revisit"]  # type: ignore[operator]
                - summary["mean_ca1_match_first_visit"]
            )
        return summary

    @staticmethod
    def _find_hc(circuit: Circuit) -> HippocampalRegion | None:
        """Locate a HippocampalRegion in the circuit, or None."""
        regions = getattr(circuit, "_regions", None)
        if regions is None:
            return None
        for state in regions.values():
            region = getattr(state, "region", None)
            if isinstance(region, HippocampalRegion):
                return region
        return None
