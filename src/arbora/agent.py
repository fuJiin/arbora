"""Base agent and training result for Arbor circuits.

BaseAgent — shared state and wiring for all agents.
TrainResult — result of a training run (probe snapshots + elapsed time).

Concrete agents (ChatAgent, MiniGridAgent) live in examples/.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from arbora.cortex.circuit import Circuit
    from arbora.cortex.circuit_types import Encoder


@dataclass
class TrainResult:
    """Result of a training run. All metrics live in probe_snapshots."""

    probe_snapshots: dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0


class BaseAgent:
    """Shared state and wiring for all Arbor agents.

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
