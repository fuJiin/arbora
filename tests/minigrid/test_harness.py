import pytest

from arbora.cortex import SensoryRegion
from arbora.cortex.circuit import Circuit, ConnectionRole
from arbora.cortex.motor import MotorRegion
from arbora.probes.core import LaminaProbe
from examples.minigrid.agent import MiniGridAgent
from examples.minigrid.encoder import MiniGridEncoder
from examples.minigrid.env import MiniGridEnv
from examples.minigrid.harness import MiniGridHarness


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
    c.add_region("M1", m1)
    c.connect(s1.output_port, m1.input_port, ConnectionRole.FEEDFORWARD)
    c.finalize()
    return c


class TestMiniGridHarness:
    def test_runs_to_completion(self, encoder, circuit):
        """Harness should complete 5 episodes without error."""
        env = MiniGridEnv("MiniGrid-Empty-5x5-v0", max_episodes=5)
        agent = MiniGridAgent(encoder=encoder, circuit=circuit)
        harness = MiniGridHarness(env, agent, log_interval=10000)
        result = harness.run()
        assert result.elapsed_seconds > 0
        assert env.episode_count == 5

    def test_lamina_probe_observed(self, encoder, circuit):
        """LaminaProbe should accumulate data during training."""
        env = MiniGridEnv("MiniGrid-Empty-5x5-v0", max_episodes=3)
        agent = MiniGridAgent(encoder=encoder, circuit=circuit)
        probe = LaminaProbe()
        harness = MiniGridHarness(env, agent, probes=[probe], log_interval=10000)
        result = harness.run()
        assert "lamina" in result.probe_snapshots
        snap = result.probe_snapshots["lamina"]
        assert "S1" in snap
        assert snap["S1"].input.recall >= 0

    def test_train_result_has_snapshots(self, encoder, circuit):
        """TrainResult should include probe snapshots."""
        env = MiniGridEnv("MiniGrid-Empty-5x5-v0", max_episodes=2)
        agent = MiniGridAgent(encoder=encoder, circuit=circuit)
        probe = LaminaProbe()
        harness = MiniGridHarness(env, agent, probes=[probe], log_interval=10000)
        result = harness.run()
        assert isinstance(result.probe_snapshots, dict)
        assert len(result.probe_snapshots) == 1
