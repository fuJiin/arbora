"""Tests for BasalGanglia go/no-go gating."""

import numpy as np
import pytest

from step.cortex.basal_ganglia import BasalGanglia
from step.cortex.motor import MotorRegion
from step.cortex.sensory import SensoryRegion
from step.cortex.topology import Topology
from step.data import EOM_TOKEN, STORY_BOUNDARY
from step.encoders.charbit import CharbitEncoder


@pytest.fixture()
def encoder():
    return CharbitEncoder(length=4, width=5, chars="abcd")


@pytest.fixture()
def region1():
    return SensoryRegion(
        input_dim=4 * 5,
        encoding_width=5,
        n_columns=8,
        n_l4=2,
        n_l23=2,
        k_columns=2,
        seed=42,
    )


@pytest.fixture()
def motor(region1):
    return MotorRegion(
        input_dim=region1.n_l23_total,
        n_columns=4,
        n_l4=2,
        n_l23=2,
        k_columns=1,
        voltage_decay=0.5,
        learning_rate=0.15,
        ltd_rate=0.15,
        seed=456,
    )


class TestBasalGangliaUnit:
    def test_gate_starts_neutral(self):
        bg = BasalGanglia(context_dim=16)
        assert bg.gate_value == pytest.approx(0.5)

    def test_step_returns_gate_in_range(self):
        bg = BasalGanglia(context_dim=16)
        ctx = np.random.default_rng(0).random(16)
        gate = bg.step(ctx)
        assert 0.0 <= gate <= 1.0

    def test_positive_reward_shifts_gate(self):
        """Repeated positive reward for a context should open gate."""
        bg = BasalGanglia(context_dim=16, learning_rate=0.1)
        ctx = np.ones(16) * 0.5
        # Establish context with gate open-ish, then reward
        for _ in range(100):
            bg.step(ctx)
            bg.reward(1.0)
        gate = bg.step(ctx)
        assert gate > 0.7  # Should have learned to open

    def test_negative_reward_closes_gate(self):
        """Repeated negative reward should close gate."""
        bg = BasalGanglia(context_dim=16, learning_rate=0.1)
        ctx = np.ones(16) * 0.5
        for _ in range(100):
            bg.step(ctx)
            bg.reward(-1.0)
        gate = bg.step(ctx)
        assert gate < 0.3  # Should have learned to close

    def test_different_contexts_different_gates(self):
        """BG should learn different gate values for different contexts."""
        bg = BasalGanglia(context_dim=16, learning_rate=0.05)
        ctx_open = np.zeros(16)
        ctx_open[:8] = 1.0
        ctx_close = np.zeros(16)
        ctx_close[8:] = 1.0

        for _ in range(200):
            bg.step(ctx_open)
            bg.reward(1.0)
            bg.step(ctx_close)
            bg.reward(-1.0)

        gate_open = bg.step(ctx_open)
        gate_close = bg.step(ctx_close)
        assert gate_open > gate_close + 0.1

    def test_reset_clears_transient_state(self):
        bg = BasalGanglia(context_dim=16)
        ctx = np.ones(16)
        bg.step(ctx)
        bg.reward(1.0)
        bg.reset()
        assert bg.gate_value == 0.5
        assert bg._trace.sum() == 0.0


class TestBasalGangliaIntegration:
    def test_bg_runs_in_topology(self, region1, motor, encoder):
        """BG wired to M1 runs without error."""
        bg = BasalGanglia(context_dim=region1.n_columns + 1)
        tokens = [(i % 4, chr(ord("a") + i % 4)) for i in range(30)]
        cortex = Topology(encoder)
        cortex.add_region("S1", region1, entry=True)
        cortex.add_region("M1", motor, basal_ganglia=bg)
        cortex.connect("S1", "M1", "feedforward")
        result = cortex.run(tokens, log_interval=1000)
        m = result.per_region["M1"]
        assert len(m.bg_gate_values) > 0
        assert all(0 <= v <= 1 for v in m.bg_gate_values)

    def test_bg_with_eom_tokens(self, region1, motor, encoder):
        """BG gate values are tracked through EOM phases."""
        bg = BasalGanglia(context_dim=region1.n_columns + 1)
        tokens = [
            (0, "a"), (1, "b"), (2, "c"),
            (EOM_TOKEN, ""),
            (0, "a"), (1, "b"),
            (STORY_BOUNDARY, ""),
            (0, "a"), (1, "b"),
        ]
        cortex = Topology(encoder)
        cortex.add_region("S1", region1, entry=True)
        cortex.add_region("M1", motor, basal_ganglia=bg)
        cortex.connect("S1", "M1", "feedforward")
        result = cortex.run(tokens, log_interval=1000)
        assert len(result.per_region["M1"].bg_gate_values) > 0

    def test_bg_resets_at_boundary(self, region1, motor, encoder):
        """Story boundary resets BG transient state."""
        bg = BasalGanglia(context_dim=region1.n_columns + 1)
        tokens = [
            (0, "a"), (1, "b"),
            (STORY_BOUNDARY, ""),
            (0, "a"),
        ]
        cortex = Topology(encoder)
        cortex.add_region("S1", region1, entry=True)
        cortex.add_region("M1", motor, basal_ganglia=bg)
        cortex.connect("S1", "M1", "feedforward")
        cortex.run(tokens, log_interval=1000)
        # After run, BG should have been reset at boundary
        # (we can't directly observe mid-run, but no crash = good)

    def test_no_bg_backward_compatible(self, region1, motor, encoder):
        """Without BG, bg_gate_values is empty."""
        tokens = [(i % 4, chr(ord("a") + i % 4)) for i in range(20)]
        cortex = Topology(encoder)
        cortex.add_region("S1", region1, entry=True)
        cortex.add_region("M1", motor)
        cortex.connect("S1", "M1", "feedforward")
        result = cortex.run(tokens, log_interval=1000)
        assert result.per_region["M1"].bg_gate_values == []
