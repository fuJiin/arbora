"""Environment abstraction for the STEP training loop.

An Environment provides observations and evaluates actions following
the standard RL interface::

    obs = env.reset()
    while not env.done:
        action = agent.act(obs, reward)
        obs, reward = env.step(action)

The generic Environment protocol is modality-agnostic. ChatEnv is the
concrete implementation for char-level text.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Generic protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class Observation(Protocol):
    """Minimal observation interface.

    Concrete types (ChatObs, etc.) add domain-specific fields.
    """

    ...


@runtime_checkable
class Environment(Protocol):
    """World that provides observations and evaluates actions.

    The step(action) -> (obs, reward) contract follows standard RL:
    the environment advances one timestep, returns what the agent
    perceives next and the reward for the previous action.
    """

    def reset(self) -> Observation:
        """Reset to initial state. Returns first observation."""
        ...

    def step(self, action: int | None) -> tuple[Observation, float]:
        """Advance one timestep.

        Args:
            action: Agent's output (e.g. token_id), or None (silence).

        Returns:
            (observation, reward) — next percept and reward for this action.
        """
        ...

    @property
    def done(self) -> bool:
        """True when the episode is over."""
        ...


# ---------------------------------------------------------------------------
# Chat-specific types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChatObs:
    """Single-token observation for char-level chat.

    Replaces the (token_id, token_str) tuples and STORY_BOUNDARY/EOM_TOKEN
    sentinel checks scattered across the old codebase.
    """

    token_id: int
    token_str: str
    is_boundary: bool = False
    is_eom: bool = False


# Sentinel observations
BOUNDARY_OBS = ChatObs(token_id=-1, token_str="", is_boundary=True)
EOM_OBS = ChatObs(token_id=-2, token_str="", is_eom=True)


# ---------------------------------------------------------------------------
# ChatEnv
# ---------------------------------------------------------------------------


class ChatEnv:
    """Token-by-token chat environment.

    Wraps a token stream and handles:
    - Sequential token presentation during listen episodes
    - Babble episodes where the agent's action feeds back as observation
    - STORY_BOUNDARY / EOM signaling via ChatObs flags
    - Turn-taking reward computation

    The environment streams tokens lazily from an iterable. During babble
    episodes, the agent's last action becomes the next observation.

    Args:
        tokens: Iterable of (token_id, token_str) pairs. Consumed lazily.
        babble_ratio: Fraction of total steps that are babbling (0-1).
            0 = pure listening, 0.5 = equal listen/babble. Default 0.
        listen_chunk: Tokens per listening episode before switching to babble.
        babble_chunk: Steps per babbling episode.
        reward_source: Pluggable reward function. If None, uses default
            turn-taking reward. Should have step(char, context) -> float.
        max_speak_steps: Anti-rambling cutoff for turn-taking reward.
    """

    # Default turn-taking reward values
    REWARD_SPEAK_IN_TURN = 0.5
    REWARD_SILENT_IN_TURN = -0.3
    REWARD_SPEAK_OUT_OF_TURN = -0.5
    REWARD_SILENT_OUT_OF_TURN = 0.2
    REWARD_RAMBLING = -1.0

    def __init__(
        self,
        tokens: Iterable[tuple[int, str]],
        *,
        babble_ratio: float = 0.0,
        listen_chunk: int = 200,
        babble_chunk: int = 50,
        reward_source=None,
        max_speak_steps: int = 20,
    ):
        self._token_iter = iter(tokens)
        self._babble_ratio = babble_ratio
        self._listen_chunk = listen_chunk
        self._babble_chunk = babble_chunk
        self._reward_source = reward_source
        self._max_speak_steps = max_speak_steps

        # Episode state
        self._done = False
        self._in_eom = False
        self._eom_steps = 0

        # Interleaved state
        self._mode: str = "listen"  # "listen" or "babble"
        self._listen_remaining = listen_chunk
        self._babble_remaining = 0
        self._last_action_char: str = " "

        # Track totals for metrics
        self.total_listen_steps = 0
        self.total_babble_steps = 0

    @property
    def done(self) -> bool:
        return self._done

    @property
    def in_eom(self) -> bool:
        return self._in_eom

    @property
    def eom_steps(self) -> int:
        return self._eom_steps

    def reset(self) -> ChatObs:
        """Reset transient state and return first observation."""
        self._in_eom = False
        self._eom_steps = 0
        self._mode = "listen"
        self._listen_remaining = self._listen_chunk
        self._babble_remaining = 0
        self._last_action_char = " "
        if self._reward_source is not None:
            _reset = getattr(self._reward_source, "reset", None)
            if _reset is not None:
                _reset()
        return self._next_listen_obs()

    def step(self, action: int | None) -> tuple[ChatObs, float]:
        """Advance one timestep.

        Args:
            action: Agent's output token_id, or None for silence.

        Returns:
            (next_obs, reward) — what the agent sees next and its reward.
        """
        if self._done:
            raise StopIteration("Environment is done")

        # Track agent's action for babble feedback
        spoke = action is not None and action >= 0
        if spoke and 32 <= action < 127:
            self._last_action_char = chr(action)

        # Compute reward for this action
        reward = self._compute_reward(action)

        # Update turn-taking state
        if self._in_eom:
            self._eom_steps += 1
            if self._eom_steps > self._max_speak_steps:
                self._in_eom = False

        # Decide next observation based on mode
        if self._should_babble():
            obs = self._next_babble_obs()
        else:
            obs = self._next_listen_obs()

        return obs, reward

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self, action: int | None) -> float:
        """Compute reward for the agent's action."""
        spoke = action is not None and action >= 0

        if self._reward_source is not None:
            char = chr(action) if spoke and 32 <= action < 127 else None
            return self._reward_source.step(char, 0.0)

        return self._turn_taking_reward(spoke)

    def _turn_taking_reward(self, spoke: bool) -> float:
        """Default turn-taking reward.

        +0.5  speak during EOM (correct)
        -0.5  speak during input (interruption)
        +0.2  silent during input (correct)
        -0.3  silent during EOM (unresponsive)
        -1.0  rambling past max_speak_steps

        TODO: turn-taking reward should eventually be learned by
        BasalGanglia rather than hardcoded here. The BG would learn
        when to gate motor output based on context, making this
        external reward signal unnecessary.
        """
        if self._in_eom:
            if self._eom_steps > self._max_speak_steps and spoke:
                return self.REWARD_RAMBLING
            return self.REWARD_SPEAK_IN_TURN if spoke else self.REWARD_SILENT_IN_TURN
        else:
            if spoke:
                return self.REWARD_SPEAK_OUT_OF_TURN
            return self.REWARD_SILENT_OUT_OF_TURN

    # ------------------------------------------------------------------
    # Observation sourcing
    # ------------------------------------------------------------------

    def _next_listen_obs(self) -> ChatObs:
        """Pull next token from the corpus stream."""
        try:
            token_id, token_str = next(self._token_iter)
        except StopIteration:
            self._done = True
            return BOUNDARY_OBS

        self.total_listen_steps += 1

        # Translate sentinel values to ChatObs flags
        from step.data import EOM_TOKEN, STORY_BOUNDARY

        if token_id == STORY_BOUNDARY:
            self._in_eom = False
            self._eom_steps = 0
            return BOUNDARY_OBS
        if token_id == EOM_TOKEN:
            self._in_eom = True
            self._eom_steps = 0
            return EOM_OBS

        # Decrement listen counter for interleaved mode
        self._listen_remaining -= 1

        return ChatObs(token_id=token_id, token_str=token_str)

    def _next_babble_obs(self) -> ChatObs:
        """During babble: agent's last action becomes next observation."""
        self._babble_remaining -= 1
        self.total_babble_steps += 1
        char = self._last_action_char
        return ChatObs(token_id=ord(char), token_str=char)

    def _should_babble(self) -> bool:
        """Decide whether the next step should be babble or listen.

        TODO: babble/listen alternation is a training curriculum concern.
        Eventually BasalGanglia should learn when to self-initiate
        (babble) vs. attend to input (listen), making the external
        chunking schedule unnecessary.
        """
        if self._babble_ratio <= 0:
            return False

        # Currently babbling — continue until chunk exhausted
        if self._mode == "babble":
            if self._babble_remaining > 0:
                return True
            # Switch to listen
            self._mode = "listen"
            self._listen_remaining = self._listen_chunk
            return False

        # Currently listening — switch to babble when chunk exhausted
        if self._listen_remaining <= 0:
            self._mode = "babble"
            self._babble_remaining = self._babble_chunk
            return True

        return False
