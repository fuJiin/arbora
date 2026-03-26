"""Tests for Environment and Agent abstractions."""

import numpy as np

from step.agent import ChatAgent
from step.data import EOM_TOKEN, STORY_BOUNDARY
from step.environment import BOUNDARY_OBS, EOM_OBS, ChatEnv, ChatObs


class TestChatObs:
    def test_content_obs(self):
        obs = ChatObs(token_id=65, token_str="A")
        assert obs.token_id == 65
        assert obs.token_str == "A"
        assert not obs.is_boundary
        assert not obs.is_eom

    def test_boundary_obs(self):
        assert BOUNDARY_OBS.is_boundary
        assert not BOUNDARY_OBS.is_eom

    def test_eom_obs(self):
        assert EOM_OBS.is_eom
        assert not EOM_OBS.is_boundary

    def test_frozen(self):
        import pytest

        obs = ChatObs(token_id=65, token_str="A")
        with pytest.raises(AttributeError):
            obs.token_id = 66  # type: ignore[misc]


class TestChatEnvCorpus:
    """Pure listening mode (babble_ratio=0)."""

    def _tokens(self):
        return [(ord("a"), "a"), (ord("b"), "b"), (ord("c"), "c")]

    def test_iterates_tokens(self):
        env = ChatEnv(self._tokens())
        obs = env.reset()
        assert obs.token_id == ord("a")

        obs, _ = env.step(None)
        assert obs.token_id == ord("b")

        obs, _ = env.step(None)
        assert obs.token_id == ord("c")

        obs, _ = env.step(None)
        assert env.done

    def test_listen_count(self):
        env = ChatEnv(self._tokens())
        env.reset()
        while not env.done:
            env.step(None)
        assert env.total_listen_steps == 3

    def test_boundary_signaling(self):
        tokens = [(ord("a"), "a"), (STORY_BOUNDARY, ""), (ord("b"), "b")]
        env = ChatEnv(tokens)
        env.reset()  # 'a'
        obs, _ = env.step(None)  # boundary
        assert obs.is_boundary
        obs, _ = env.step(None)  # 'b'
        assert obs.token_id == ord("b")

    def test_eom_signaling(self):
        tokens = [(ord("a"), "a"), (EOM_TOKEN, ""), (ord("b"), "b")]
        env = ChatEnv(tokens)
        env.reset()  # 'a'
        obs, _ = env.step(None)  # eom
        assert obs.is_eom
        assert env.in_eom
        obs, _ = env.step(None)  # 'b'
        assert not obs.is_eom


class TestChatEnvReward:
    def test_silent_during_input_positive(self):
        tokens = [(ord("a"), "a"), (ord("b"), "b")]
        env = ChatEnv(tokens)
        env.reset()
        _, reward = env.step(None)  # silent during input
        assert reward > 0  # +0.2

    def test_speak_during_input_negative(self):
        tokens = [(ord("a"), "a"), (ord("b"), "b")]
        env = ChatEnv(tokens)
        env.reset()
        _, reward = env.step(ord("x"))  # speak during input
        assert reward < 0  # -0.5

    def test_silent_during_eom_negative(self):
        tokens = [(EOM_TOKEN, ""), (ord("a"), "a")]
        env = ChatEnv(tokens)
        env.reset()  # eom obs
        env.step(None)  # process eom, sets in_eom
        _, reward = env.step(None)  # silent during eom
        assert reward < 0  # -0.3

    def test_speak_during_eom_positive(self):
        tokens = [(EOM_TOKEN, ""), (ord("a"), "a")]
        env = ChatEnv(tokens)
        env.reset()  # eom obs
        env.step(None)  # process eom
        _, reward = env.step(ord("h"))  # speak during eom
        assert reward > 0  # +0.5


class TestChatEnvInterleaved:
    def test_babble_mode(self):
        tokens = [(ord(c), c) for c in "abcdefghij"]
        env = ChatEnv(tokens, babble_ratio=0.5, listen_chunk=3, babble_chunk=3)
        env.reset()

        # Run through some steps
        for _ in range(20):
            if env.done:
                break
            env.step(ord("x"))

        assert env.total_babble_steps > 0
        assert env.total_listen_steps > 0


class TestChatAgent:
    def _make_agent(self):
        from step.cortex.canonical import build_canonical_circuit
        from step.encoders.positional import PositionalCharEncoder

        alphabet = "abcdefghijklmnopqrstuvwxyz .,"
        encoder = PositionalCharEncoder(alphabet, max_positions=4)
        circuit = build_canonical_circuit(encoder)
        return ChatAgent(encoder=encoder, circuit=circuit), circuit

    def test_act_returns_action(self):
        agent, _ = self._make_agent()
        obs = ChatObs(token_id=ord("h"), token_str="h")
        action = agent.act(obs, 0.0)
        # Action is int or None
        assert action is None or isinstance(action, (int, np.integer))

    def test_boundary_resets(self):
        agent, _circuit = self._make_agent()
        # Process a token first
        agent.act(ChatObs(token_id=ord("h"), token_str="h"), 0.0)
        # Boundary should reset
        result = agent.act(BOUNDARY_OBS, 0.0)
        assert result is None

    def test_eom_activates_motor(self):
        agent, _ = self._make_agent()
        assert not agent._motor_active
        agent.act(EOM_OBS, 0.0)
        assert agent._motor_active

    def test_encoding_stored(self):
        agent, _ = self._make_agent()
        agent.act(ChatObs(token_id=ord("a"), token_str="a"), 0.0)
        assert agent.last_encoding is not None
        assert agent.last_output is not None

    def test_full_loop(self):
        """End-to-end: ChatEnv + ChatAgent."""
        agent, _ = self._make_agent()
        tokens = [(ord(c), c) for c in "hello"]
        env = ChatEnv(tokens)
        obs = env.reset()
        reward = 0.0
        steps = 0
        while not env.done:
            action = agent.act(obs, reward)
            obs, reward = env.step(action)
            steps += 1
        assert steps == 5
