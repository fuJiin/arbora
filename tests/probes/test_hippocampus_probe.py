"""Tests for HippocampalProbe — mechanistic HC diagnostics.

The probe only reads `circuit._regions`, so tests use a minimal fake
Circuit rather than constructing a full cortical pipeline.
"""

from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np

from arbora.hippocampus import HippocampalRegion
from arbora.probes import HippocampalProbe


@dataclass
class _FakeState:
    region: Any


class _FakeCircuit:
    """Minimal stand-in exposing `_regions`, which is all the probe reads."""

    def __init__(self, regions: dict[str, Any]):
        self._regions = {name: _FakeState(r) for name, r in regions.items()}


def _hc(input_dim: int = 64, seed: int = 0) -> HippocampalRegion:
    return HippocampalRegion(
        input_dim=input_dim,
        ec_dim=200,
        dg_dim=800,
        ca3_dim=200,
        seed=seed,
    )


def _encoding(seed: int, dim: int = 64) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=dim)


class TestSilentOnNonHCCircuit:
    def test_observe_noop_when_no_hc(self):
        """Probe silently no-ops on circuits without a HippocampalRegion."""

        # Any non-HC object satisfies the search predicate.
        class Other:
            pass

        circuit = _FakeCircuit({"other": Other()})
        probe = HippocampalProbe()
        probe.observe(circuit, step=0)
        snap = probe.snapshot()
        assert snap["summary"] == {"n_steps": 0}
        assert snap["steps"] == []

    def test_observe_noop_on_empty_circuit(self):
        circuit = _FakeCircuit({})
        probe = HippocampalProbe()
        probe.observe(circuit, step=0)
        assert probe.snapshot()["summary"] == {"n_steps": 0}


class TestPerStepLogging:
    def test_records_one_step_per_observe(self):
        hc = _hc()
        circuit = _FakeCircuit({"HC": hc})
        probe = HippocampalProbe()
        for i in range(3):
            hc.process(_encoding(i))
            probe.observe(circuit, step=i)
        snap = probe.snapshot()
        assert len(snap["steps"]) == 3
        assert snap["summary"]["n_steps"] == 3

    def test_recorded_counts_match_hc_state(self):
        hc = _hc()
        circuit = _FakeCircuit({"HC": hc})
        probe = HippocampalProbe()
        hc.process(_encoding(0))
        probe.observe(circuit, step=0)
        step = probe.snapshot()["steps"][0]
        assert step["ec_active"] == int(hc.last_ec_pattern.sum())
        assert step["dg_active"] == int(hc.last_dg_pattern.sum())
        assert step["ca3_active"] == int(hc.ca3.state.sum())
        assert step["ca1_match"] == hc.last_match

    def test_lateral_weight_stats_grow_with_encoding(self):
        """CA3 lateral_sum and nnz must increase as new patterns bind."""
        hc = _hc()
        circuit = _FakeCircuit({"HC": hc})
        probe = HippocampalProbe()
        for i in range(5):
            hc.process(_encoding(i))
            probe.observe(circuit, step=i)
        steps = probe.snapshot()["steps"]
        # Monotonic non-decreasing (LTP is additive, clip at 1.0).
        for prev, cur in pairwise(steps):
            assert cur["ca3_lateral_sum"] >= prev["ca3_lateral_sum"]
            assert cur["ca3_lateral_nnz"] >= prev["ca3_lateral_nnz"]


class TestMaxStepsCap:
    def test_cap_limits_step_log_but_counts_continue(self):
        hc = _hc()
        circuit = _FakeCircuit({"HC": hc})
        probe = HippocampalProbe(max_steps=3)
        for i in range(10):
            hc.process(_encoding(i))
            probe.observe(circuit, step=i)
        snap = probe.snapshot()
        assert len(snap["steps"]) == 3
        assert snap["summary"]["n_steps"] == 10  # rolling counter keeps going


class TestRevisitDetection:
    def test_ca3_stability_on_exact_repeats(self):
        """Exact repeats of an encoding should produce identical CA3 state."""
        hc = _hc()
        circuit = _FakeCircuit({"HC": hc})
        probe = HippocampalProbe()

        x = _encoding(42)
        for i in range(3):
            hc.process(x)
            probe.observe(circuit, step=i)
        # Also visit a different encoding so first-visit stats aren't empty.
        hc.process(_encoding(99))
        probe.observe(circuit, step=3)

        summary = probe.snapshot()["summary"]
        # Revisits of `x` should match the first visit's CA3 state
        # (deterministic LTP: same input → same k-WTA pattern).
        assert summary["ca3_revisit_stability"] == 1.0

    def test_ca1_match_delta_is_reported_when_revisits_occur(self):
        """Confirms the summary includes revisit-vs-first delta when
        at least one EC pattern shows up more than once."""
        hc = _hc()
        circuit = _FakeCircuit({"HC": hc})
        probe = HippocampalProbe()
        x = _encoding(7)
        hc.process(x)
        probe.observe(circuit, step=0)
        hc.process(x)
        probe.observe(circuit, step=1)
        summary = probe.snapshot()["summary"]
        assert "ca1_match_revisit_minus_first" in summary


class TestHCStateAttrs:
    def test_last_patterns_populated_after_process(self):
        """Regression: the observable attrs that the probe depends on
        must be set by every process() call."""
        hc = _hc()
        assert hc.last_ec_pattern.shape == (200,)
        assert hc.last_dg_pattern.shape == (800,)
        assert not hc.last_ec_pattern.any()
        assert not hc.last_dg_pattern.any()
        hc.process(_encoding(0))
        assert hc.last_ec_pattern.any()
        assert hc.last_dg_pattern.any()

    def test_last_patterns_cleared_by_reset_working_memory(self):
        hc = _hc()
        hc.process(_encoding(0))
        hc.reset_working_memory()
        assert not hc.last_ec_pattern.any()
        assert not hc.last_dg_pattern.any()
