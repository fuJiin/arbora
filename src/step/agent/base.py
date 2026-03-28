"""Base agent with shared state for all STEP agents.

Provides encoder/circuit wiring, last-step state tracking, and
base reset behavior. Subclasses implement step(), decode_action(),
and act() for their specific modality.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from step.cortex.circuit import Circuit
    from step.cortex.circuit_types import Encoder


class BaseAgent:
    """Shared state and wiring for all STEP agents.

    Subclasses must implement:
    - step(obs) — encode + process
    - decode_action() — read motor output
    - act(obs, reward) — convenience step + decode

    The harness interleaves probes between step() and decode_action()::

        agent.step(obs)
        probe.observe(circuit)
        action = agent.decode_action()
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

        # Last step state (readable by harness for probes/metrics)
        self.last_encoding: np.ndarray | None = None
        self.last_output: np.ndarray | None = None
        self.last_action: int | None = None

    @property
    def encoder(self) -> Encoder:
        return self._encoder

    @property
    def circuit(self) -> Circuit:
        return self._circuit

    def reset(self) -> None:
        """Reset circuit and clear agent state. Subclasses should call super()."""
        self._circuit.reset()
        self.last_action = None

    def apply_reward(self, reward: float) -> None:
        """Route reward to circuit (BG + motor regions internally)."""
        self._circuit.apply_reward(reward)
