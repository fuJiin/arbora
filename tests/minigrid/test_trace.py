"""Tests for the compact TraceProbe."""

import io

import pytest

from examples.minigrid.encoder import MiniGridEncoder
from examples.minigrid.presets import (
    build_baseline_circuit,
    build_hippocampal_circuit,
)
from examples.minigrid.trace import TraceProbe


def _encoding():
    import numpy as np

    rng = np.random.default_rng(0)
    out = np.zeros(984, dtype=bool)
    out[rng.choice(984, size=148, replace=False)] = True
    return out


class TestInit:
    def test_rejects_every_below_one(self):
        with pytest.raises(ValueError):
            TraceProbe(every=0)


class TestObserveFormatting:
    def test_baseline_circuit_line_contains_s1_m1_bg(self):
        buf = io.StringIO()
        probe = TraceProbe(stream=buf)
        circuit = build_baseline_circuit(MiniGridEncoder())
        circuit.process(_encoding())
        probe.observe(circuit, step=0)
        line = buf.getvalue().strip()
        assert "t=00000" in line
        assert "S1=" in line
        assert "M1=" in line
        assert "BG_top3=" in line
        # Baseline has no HC — should omit HC fields.
        assert "EC=" not in line
        assert "DG=" not in line
        assert "CA3=" not in line

    def test_hippocampal_circuit_line_includes_hc_fields(self):
        buf = io.StringIO()
        probe = TraceProbe(stream=buf)
        circuit = build_hippocampal_circuit(MiniGridEncoder())
        circuit.process(_encoding())
        probe.observe(circuit, step=42)
        line = buf.getvalue().strip()
        assert "t=00042" in line
        assert "EC=" in line
        assert "DG=" in line
        assert "CA3=" in line
        assert "match=" in line


class TestSubsampling:
    def test_every_n_prints_only_every_n(self):
        buf = io.StringIO()
        probe = TraceProbe(stream=buf, every=3)
        circuit = build_baseline_circuit(MiniGridEncoder())
        circuit.process(_encoding())
        for t in range(7):
            probe.observe(circuit, step=t)
        n_lines = len(buf.getvalue().rstrip().splitlines())
        # steps 0, 3, 6 → 3 lines
        assert n_lines == 3


class TestEpisodeBoundary:
    def test_episode_end_emits_marker(self):
        buf = io.StringIO()
        probe = TraceProbe(stream=buf)
        probe.episode_end(success=True, steps=10, reward=0.5)
        out = buf.getvalue()
        assert "episode 1 end" in out
        assert "steps=10" in out
        assert "reward=0.500" in out


class TestSnapshot:
    def test_snapshot_counts_emitted_lines(self):
        buf = io.StringIO()
        probe = TraceProbe(stream=buf, every=2)
        circuit = build_baseline_circuit(MiniGridEncoder())
        circuit.process(_encoding())
        for t in range(4):
            probe.observe(circuit, step=t)
        snap = probe.snapshot()
        assert snap["lines_emitted"] == 2
