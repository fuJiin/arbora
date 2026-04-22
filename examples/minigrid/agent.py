"""MiniGrid agent wrapping a Circuit for gymnasium environments.

Unlike ChatAgent:
- No turn-taking / listen-speak phases
- No efference copy
- Random action fallback when M1 is silent (always acts)
- No EOM/boundary concept (episode boundary = reset)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from arbora.agent import BaseAgent
from examples.minigrid.env import MiniGridObs

if TYPE_CHECKING:
    from arbora.cortex.circuit import Circuit
    from examples.minigrid.encoder import MiniGridEncoder


class MiniGridAgent(BaseAgent):
    """Agent wrapping a Circuit for MiniGrid environments.

    The harness interleaves probes between step() and decode_action()::

        agent.step(obs)              # encode + process
        probe.observe(circuit)       # harness observes
        action = agent.decode_action()  # read motor output

    Args:
        encoder: Encodes MiniGridObs to sparse binary vectors.
        circuit: The neural circuit (process(encoding) -> output).
        n_actions: Number of discrete actions (default 7 for MiniGrid).
        entry_name: Name of the entry (sensory) region. Default "T1".
    """

    def __init__(
        self,
        encoder: MiniGridEncoder,
        circuit: Circuit,
        *,
        n_actions: int = 7,
        entry_name: str | None = None,
    ):
        super().__init__(encoder, circuit, entry_name=entry_name)
        self._n_actions = n_actions
        self._rng = np.random.default_rng(42)

    def step(self, obs: MiniGridObs) -> None:
        """Encode observation and run circuit processing.

        Does NOT produce an action -- call decode_action() after probe
        observation.
        """
        encoding = self._encoder.encode(obs)
        self.last_encoding = encoding
        output = self._circuit.process(encoding)
        self.last_output = output

    def decode_action(self) -> int:
        """Read motor region output. Always returns a valid action.

        If M1 produces no output (BG suppressed or step 0), falls
        back to a random action. MiniGrid requires an action every step.
        """
        motor = self._circuit.output_regions[0]
        m_id, _conf = motor.last_output
        if m_id >= 0:
            self.last_action = m_id
            return m_id
        # Random fallback
        action = int(self._rng.integers(self._n_actions))
        self.last_action = action
        return action

    def act(self, obs: MiniGridObs, reward: float) -> int:
        """Convenience: step + decode_action."""
        self.step(obs)
        return self.decode_action()
