"""Tests for MiniGridMotorProbe — episode success, purposeful ratio, consolidation."""

import pytest

from arbor.basal_ganglia import BasalGangliaRegion
from arbor.cortex import SensoryRegion
from arbor.cortex.circuit import Circuit, ConnectionRole
from arbor.cortex.motor import MotorRegion
from examples.minigrid.agent import MiniGridAgent
from examples.minigrid.encoder import MiniGridEncoder
from examples.minigrid.env import MiniGridEnv
from examples.minigrid.harness import MiniGridHarness
from examples.minigrid.probes import MiniGridMotorProbe


@pytest.fixture()
def encoder():
    return MiniGridEncoder()


@pytest.fixture()
def circuit(encoder):
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
    bg = BasalGangliaRegion(input_dim=s1.n_l23_total, n_actions=7, seed=789)
    m1 = MotorRegion(
        input_dim=s1.n_l23_total,
        n_columns=8,
        n_l4=0,
        n_l23=2,
        k_columns=1,
        n_output_tokens=7,
        seed=123,
    )
    c = Circuit(encoder)
    c.add_region("S1", s1, entry=True)
    c.add_region("BG", bg)
    c.add_region("M1", m1)
    c.connect(s1.output_port, bg.input_port, ConnectionRole.FEEDFORWARD)
    c.connect(s1.output_port, m1.input_port, ConnectionRole.FEEDFORWARD)
    c.connect(bg.output_port, m1.input_port, ConnectionRole.MODULATORY)
    c.finalize()
    return c


class TestMiniGridMotorProbe:
    def test_probe_accumulates_during_training(self, encoder, circuit):
        env = MiniGridEnv("MiniGrid-Empty-5x5-v0", max_episodes=5)
        agent = MiniGridAgent(encoder=encoder, circuit=circuit)
        probe = MiniGridMotorProbe()
        harness = MiniGridHarness(env, agent, probes=[probe], log_interval=10000)
        harness.run()

        snap = probe.snapshot()["minigrid"]
        assert snap.total_steps > 0
        assert len(snap.episode_successes) == 5
        assert snap.purposeful_ratio >= 0

    def test_episode_metrics_tracked(self, encoder, circuit):
        env = MiniGridEnv("MiniGrid-Empty-5x5-v0", max_episodes=3)
        agent = MiniGridAgent(encoder=encoder, circuit=circuit)
        probe = MiniGridMotorProbe()
        harness = MiniGridHarness(env, agent, probes=[probe], log_interval=10000)
        harness.run()

        snap = probe.snapshot()["minigrid"]
        assert len(snap.episode_rewards) == 3
        assert len(snap.episode_steps) == 3
        assert all(isinstance(s, bool) for s in snap.episode_successes)

    def test_action_counts(self, encoder, circuit):
        env = MiniGridEnv("MiniGrid-Empty-5x5-v0", max_episodes=3)
        agent = MiniGridAgent(encoder=encoder, circuit=circuit)
        probe = MiniGridMotorProbe()
        harness = MiniGridHarness(env, agent, probes=[probe], log_interval=10000)
        harness.run()

        snap = probe.snapshot()["minigrid"]
        total_actions = sum(snap.action_counts.values())
        assert total_actions > 0

    def test_tonic_da_tracked(self, encoder, circuit):
        env = MiniGridEnv("MiniGrid-Empty-5x5-v0", max_episodes=3)
        agent = MiniGridAgent(encoder=encoder, circuit=circuit)
        probe = MiniGridMotorProbe()
        harness = MiniGridHarness(env, agent, probes=[probe], log_interval=10000)
        harness.run()

        snap = probe.snapshot()["minigrid"]
        assert len(snap.tonic_da_history) > 0

    def test_snapshot_in_train_result(self, encoder, circuit):
        env = MiniGridEnv("MiniGrid-Empty-5x5-v0", max_episodes=2)
        agent = MiniGridAgent(encoder=encoder, circuit=circuit)
        probe = MiniGridMotorProbe()
        harness = MiniGridHarness(env, agent, probes=[probe], log_interval=10000)
        result = harness.run()

        assert "minigrid_motor" in result.probe_snapshots
