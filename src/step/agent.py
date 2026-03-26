"""Agent abstraction for the STEP training loop.

An Agent perceives observations and produces actions::

    action = agent.act(obs, reward)

The generic Agent protocol is modality-agnostic. ChatAgent is the
concrete implementation for char-level text, wrapping an encoder,
a Circuit, and a decoder.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from step.cortex.motor import MotorRegion
from step.environment import ChatObs

if TYPE_CHECKING:
    from step.cortex.circuit import Circuit
    from step.cortex.circuit_types import Encoder


# ---------------------------------------------------------------------------
# Generic protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Agent(Protocol):
    """Agent that perceives observations and produces actions.

    The act(obs, reward) -> action contract follows standard RL:
    the agent receives the current observation and the reward from
    its previous action, then returns its next action.
    """

    def act(self, obs, reward: float) -> int | None:
        """Process one observation and return an action.

        Args:
            obs: Current observation from the environment.
            reward: Reward from previous action.

        Returns:
            Action (e.g. token_id), or None for silence.
        """
        ...


# ---------------------------------------------------------------------------
# ChatAgent
# ---------------------------------------------------------------------------


class ChatAgent:
    """Agent wrapping a Circuit with encoder/decoder for char-level chat.

    Owns the encoder (text -> vector), the circuit (vector -> vector),
    and the decoder (vector -> action). Token-level learning
    (observe_token, decoder training) happens here, not in the circuit.

    The circuit's process() is pure neural computation. The agent
    handles everything else: encoding, decoding, boundary/EOM dispatch,
    motor learning, efference copy, and force-gate policy.

    Args:
        encoder: Encodes token strings to input vectors.
        circuit: The neural circuit (process(encoding) -> output).
        entry_name: Name of the entry (sensory) region. Default "S1".
    """

    def __init__(
        self,
        encoder: Encoder,
        circuit: Circuit,
        *,
        entry_name: str | None = None,
    ):
        self._encoder = encoder
        self._circuit = circuit
        self._entry_name = entry_name or circuit._entry_name

        # Motor state — agent policy, not circuit state
        self._motor_active = False
        self.force_gate_open = False

        # Last step state (readable by runners for metrics)
        self.last_encoding: np.ndarray | None = None
        self.last_output: np.ndarray | None = None
        self.last_action: int | None = None
        self.last_token_str: str = ""

    @property
    def encoder(self) -> Encoder:
        return self._encoder

    @property
    def circuit(self) -> Circuit:
        return self._circuit

    def act(self, obs: ChatObs, reward: float) -> int | None:
        """Process one observation and return an action.

        Handles the full pipeline: boundary/EOM dispatch, encoding,
        neural processing, motor learning, decoding, efference copy.

        Args:
            obs: Current observation from ChatEnv.
            reward: Reward from previous action.

        Returns:
            Token ID of the agent's action, or None for silence.
        """
        # Boundary: reset circuit working memory
        if obs.is_boundary:
            self._circuit.reset()
            self._motor_active = False
            self.last_action = None
            return None

        # EOM: signal turn boundary
        if obs.is_eom:
            self._circuit.mark_eom()
            self._motor_active = True
            return None

        # Encode observation
        encoding = self._encoder.encode(obs.token_str)
        self.last_encoding = encoding
        self.last_token_str = obs.token_str

        # Efference copy: feed last motor output to entry region
        # (must happen before process() so it's available this step)
        if self.last_action is not None and self.force_gate_open:
            entry = self._circuit.region(self._entry_name)
            ef = self._encoder.encode(
                chr(self.last_action) if self.last_action < 128 else "",
            )
            entry.set_efference_copy(ef)

        # Update circuit's turn-taking state for _step_motor_inline
        # (pragmatic coupling — will be removed when motor gating
        # moves fully to the agent)
        self._circuit.force_gate_open = self.force_gate_open

        # Neural processing
        output = self._circuit.process(encoding)
        self.last_output = output

        # Token-level motor learning (agent's responsibility)
        motor_active = self._motor_active or self.force_gate_open
        if motor_active:
            for s in self._circuit._regions.values():
                if s.motor and isinstance(s.region, MotorRegion):
                    s.region.observe_token(obs.token_id)

        # Decode motor output to action
        action = self._decode_action()
        self.last_action = action
        return action

    def _decode_action(self) -> int | None:
        """Decode motor region output to a token ID.

        Reads M1's population output. Returns token_id if motor
        produced something, None otherwise (silence).
        """
        for s in self._circuit._regions.values():
            if s.motor and isinstance(s.region, MotorRegion):
                m_id, _conf = s.region.last_output
                if m_id >= 0:
                    return m_id
                return None
        return None
