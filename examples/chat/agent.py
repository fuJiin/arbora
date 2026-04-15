"""Agent abstraction for the STEP training loop.

An Agent perceives observations and produces actions::

    action = agent.act(obs, reward)

The generic Agent protocol is modality-agnostic. ChatAgent is the
concrete implementation for char-level text, wrapping an encoder,
a Circuit, and a decoder.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from arbora.agent import BaseAgent
from arbora.cortex.motor import MotorRegion
from examples.chat.env import ChatObs

if TYPE_CHECKING:
    from arbora.cortex.circuit import Circuit
    from arbora.cortex.circuit_types import Encoder


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


class ChatAgent(BaseAgent):
    """Agent wrapping a Circuit with encoder/decoder for char-level chat.

    Owns the encoder (text -> vector), the circuit (vector -> vector),
    and the decoder (vector -> action). Token-level learning
    (observe_token, decoder training) happens here, not in the circuit.

    The circuit's process() is pure neural computation. The agent
    handles everything else: encoding, decoding, boundary dispatch,
    motor learning, efference copy, and force-gate policy.

    The harness interleaves probes between step() and decode_action()::

        agent.step(obs)              # encode + process + motor learning
        probe.observe(circuit)       # harness observes
        action = agent.decode_action()  # read motor output

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
        super().__init__(encoder, circuit, entry_name=entry_name)

        # Motor state — agent policy, not circuit state
        self._motor_active = False
        self.force_gate_open = False

        # Chat-specific state
        self.last_token_str: str = ""

    def reset(self) -> None:
        """Reset at dialogue boundary: clear circuit and agent state."""
        super().reset()
        for conn in self._circuit._connections:
            if conn.reward_modulator is not None:
                conn.reward_modulator.reset()
        if self._circuit._reward_source is not None:
            _rs_reset = getattr(self._circuit._reward_source, "reset", None)
            if _rs_reset is not None:
                _rs_reset()
        self._motor_active = False

    def step(self, obs: ChatObs) -> None:
        """Encode observation, run neural processing, do motor learning.

        Does NOT produce an action — call decode_action() after probe
        observation. Encoding is available via self.last_encoding.
        """
        # Encode
        encoding = self._encoder.encode(obs.token_str)
        self.last_encoding = encoding
        self.last_token_str = obs.token_str

        # Efference copy (before process)
        if self.last_action is not None and self.force_gate_open:
            entry = self._circuit.region(self._entry_name)
            action_char = chr(self.last_action) if self.last_action < 128 else ""
            ef = self._encoder.encode(action_char)
            entry.set_efference_copy(ef)

        # Neural processing — motor always processes, BG gates output.
        output = self._circuit.process(encoding)
        self.last_output = output

        # Token-level motor learning (only when agent is speaking)
        if self._motor_active or self.force_gate_open:
            for motor in self._circuit.output_regions:
                if isinstance(motor, MotorRegion):
                    motor.observe_token(obs.token_id)

    def decode_action(self) -> int | None:
        """Decode motor region output to an action.

        Returns token_id if motor produced output, None for silence.
        Agent-level gating: only act when it's the agent's turn.
        """
        if not (self._motor_active or self.force_gate_open):
            self.last_action = None
            return None
        output_regions = self._circuit.output_regions
        if not output_regions:
            self.last_action = None
            return None
        motor = output_regions[0]
        m_id, _conf = motor.last_output
        if m_id >= 0:
            self.last_action = m_id
            return m_id
        self.last_action = None
        return None

    def act(self, obs: ChatObs, reward: float) -> int | None:
        """Process one observation and return an action.

        Convenience method combining step() + decode_action(). Use them
        separately when you need to interleave probe observation.
        """
        if obs.is_boundary:
            self.reset()
            return None

        if obs.is_eom:
            self._motor_active = True
            return None

        self.step(obs)
        return self.decode_action()
