import numpy as np
import pytest

from arbora.cortex import SensoryRegion
from arbora.cortex.circuit import Circuit, ConnectionRole
from arbora.cortex.motor import MotorRegion
from examples.minigrid.agent import MiniGridAgent
from examples.minigrid.encoder import MiniGridEncoder
from examples.minigrid.env import MiniGridEnv, MiniGridObs


@pytest.fixture()
def encoder():
    return MiniGridEncoder()


@pytest.fixture()
def circuit(encoder):
    """Minimal S1 -> M1 circuit for MiniGrid."""
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


@pytest.fixture()
def agent(encoder, circuit):
    return MiniGridAgent(encoder=encoder, circuit=circuit)


@pytest.fixture()
def sample_obs():
    image = np.zeros((7, 7, 3), dtype=np.uint8)
    image[:, :, 0] = 1  # empty cells
    return MiniGridObs(image=image, direction=0)


class TestMiniGridAgent:
    def test_step_stores_encoding(self, agent, sample_obs):
        agent.step(sample_obs)
        assert agent.last_encoding is not None
        assert agent.last_encoding.shape == (984,)

    def test_step_stores_output(self, agent, sample_obs):
        agent.step(sample_obs)
        assert agent.last_output is not None

    def test_decode_action_returns_valid(self, agent, sample_obs):
        agent.step(sample_obs)
        action = agent.decode_action()
        assert isinstance(action, int)
        assert 0 <= action < 7

    def test_act_convenience(self, agent, sample_obs):
        action = agent.act(sample_obs, reward=0.0)
        assert isinstance(action, int)
        assert 0 <= action < 7

    def test_reset_clears_state(self, agent, sample_obs):
        agent.step(sample_obs)
        agent.decode_action()
        assert agent.last_action is not None
        agent.reset()
        assert agent.last_action is None

    def test_circuit_property(self, agent, circuit):
        assert agent.circuit is circuit

    def test_encoder_property(self, agent, encoder):
        assert agent.encoder is encoder

    def test_multiple_steps(self, agent):
        """Agent should handle multiple steps without error."""
        env = MiniGridEnv("MiniGrid-Empty-5x5-v0", max_episodes=1)
        obs = env.reset()
        for _ in range(10):
            agent.step(obs)
            action = agent.decode_action()
            obs, _reward = env.step(action)
            if env.done:
                break
