"""Tests for reward wiring: env → harness → agent → circuit → motor/BG."""

import numpy as np
import pytest

from arbora.basal_ganglia import BasalGangliaRegion
from arbora.cortex import SensoryRegion
from arbora.cortex.circuit import Circuit, ConnectionRole
from arbora.cortex.motor import MotorRegion
from examples.minigrid.agent import MiniGridAgent
from examples.minigrid.encoder import MiniGridEncoder
from examples.minigrid.env import MiniGridEnv
from examples.minigrid.harness import MiniGridHarness


@pytest.fixture()
def encoder():
    return MiniGridEncoder()


@pytest.fixture()
def setup(encoder):
    """Build circuit with BG for reward testing."""
    s1 = SensoryRegion(
        input_dim=encoder.input_dim,
        encoding_width=encoder.encoding_width,
        n_columns=16,
        n_l4=2,
        n_l23=2,
        n_l5=0,
        k_columns=2,
        seed=42,
    )
    m1 = MotorRegion(
        input_dim=s1.n_l23_total,
        n_columns=8,
        n_l4=0,
        n_l23=2,
        k_columns=1,
        n_output_tokens=7,
        seed=123,
    )
    bg = BasalGangliaRegion(input_dim=s1.n_l23_total, n_actions=7, seed=789)
    circuit = Circuit(encoder)
    circuit.add_region("S1", s1, entry=True)
    circuit.add_region("BG", bg)
    circuit.add_region("M1", m1)
    circuit.connect(s1.output_port, bg.input_port, ConnectionRole.FEEDFORWARD)
    circuit.connect(s1.output_port, m1.input_port, ConnectionRole.FEEDFORWARD)
    circuit.connect(bg.output_port, m1.input_port, ConnectionRole.MODULATORY)
    circuit.finalize()
    return circuit, m1, bg


class TestCircuitApplyReward:
    def test_routes_to_motor(self, setup):
        circuit, m1, _bg = setup
        # Process one step to create eligibility traces
        enc = np.random.default_rng(0).random(circuit._encoder.input_dim)
        circuit.process(enc, motor_active=True)
        m1.observe_token(3)

        w_before = m1.output_weights.copy()
        circuit.apply_reward(1.0)
        assert not np.array_equal(m1.output_weights, w_before)

    def test_routes_to_bg(self, setup):
        circuit, _m1, bg = setup
        # Process multiple steps so BG builds eligibility trace
        rng = np.random.default_rng(0)
        for _ in range(5):
            enc = rng.random(circuit._encoder.input_dim)
            circuit.process(enc, motor_active=True)

        w_before = bg.go_weights.copy()
        circuit.apply_reward(1.0)
        assert not np.array_equal(bg.go_weights, w_before)

    def test_zero_reward_no_change(self, setup):
        circuit, m1, _bg = setup
        enc = np.random.default_rng(0).random(circuit._encoder.input_dim)
        circuit.process(enc, motor_active=True)

        w_before = m1.ff_weights.copy()
        circuit.apply_reward(0.0)
        # Zero reward should produce no weight change (or negligible)
        # ff_weights use three-factor: reward * trace, so 0 * trace = 0
        np.testing.assert_array_equal(m1.ff_weights, w_before)


class TestRewardInHarness:
    def test_reward_flows_through_harness(self, encoder, setup):
        circuit, _m1, _bg = setup
        env = MiniGridEnv("MiniGrid-Empty-5x5-v0", max_episodes=3)
        agent = MiniGridAgent(encoder=encoder, circuit=circuit)
        harness = MiniGridHarness(env, agent, log_interval=10000)
        result = harness.run()
        assert result.elapsed_seconds > 0
        # If agent ever reached the goal, reward was applied
        # Can't guarantee goal reached, but harness ran without error
