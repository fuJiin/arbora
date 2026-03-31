"""ARC-AGI-3 agent: V1 → BG → M1 with efference copy and intrinsic reward.

Biological sensorimotor loop:
  V1 encodes grid → BG selects action → M1 executes → V1 gets efference
  copy of expected next state → mismatch (burst rate) drives intrinsic
  reward → BG learns from surprise signal.

No epsilon-greedy or other hacks. Exploration comes from BG's tonic DA.
Internal reward comes from V1's burst rate (surprise/novelty).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from arbor.agent import BaseAgent
from arbor.basal_ganglia import BasalGangliaRegion
from arbor.cortex import SensoryRegion
from arbor.cortex.circuit import Circuit, ConnectionRole
from arbor.cortex.motor import MotorRegion
from examples.arc.encoder import ArcGridEncoder

if TYPE_CHECKING:
    from arcengine.enums import GameAction


def build_circuit(
    encoder: ArcGridEncoder,
    *,
    n_actions: int = 7,
    # V1 config
    v1_columns: int = 64,
    v1_k: int = 8,
    v1_cells: int = 4,
    # BG config — higher learning rate for sparse reward
    bg_learning_rate: float = 0.1,
    seed: int = 42,
) -> Circuit:
    """Build a V1 → BG → M1 circuit for ARC-AGI-3."""
    v1 = SensoryRegion(
        input_dim=encoder.input_dim,
        encoding_width=encoder.encoding_width,
        n_columns=v1_columns,
        n_l4=v1_cells,
        n_l23=v1_cells,
        n_l5=0,
        k_columns=v1_k,
        n_l4_lat_segments=8,
        n_synapses_per_segment=32,
        seg_activation_threshold=4,
        seed=seed,
    )
    bg = BasalGangliaRegion(
        input_dim=v1.n_l23_total,
        n_actions=n_actions,
        learning_rate=bg_learning_rate,
        tonic_da_init=2.0,
        tonic_da_min=0.5,
        # No seed: biological tonic DA noise is genuinely stochastic.
        # This is the primary exploration mechanism — different noise
        # each episode produces different action sequences.
        seed=np.random.default_rng().integers(2**31),
    )
    m1 = MotorRegion(
        input_dim=v1.n_l23_total,
        n_columns=16,
        n_l4=0,
        n_l23=4,
        k_columns=2,
        n_output_tokens=n_actions,
        seed=seed + 200,
    )

    circuit = Circuit(encoder)
    circuit.add_region("V1", v1, entry=True)
    circuit.add_region("BG", bg)
    circuit.add_region("M1", m1)
    circuit.connect(v1.output_port, bg.input_port, ConnectionRole.FEEDFORWARD)
    circuit.connect(v1.output_port, m1.input_port, ConnectionRole.FEEDFORWARD)
    circuit.connect(bg.output_port, m1.input_port, ConnectionRole.MODULATORY)
    circuit.finalize()
    return circuit


class ArcAgent(BaseAgent):
    """Agent with efference copy loop and intrinsic reward from V1 surprise.

    Sensorimotor loop each step:
      1. Set efference copy (predicted next grid = last grid encoding)
      2. Encode new grid through V1 (efference copy suppresses expected input)
      3. Measure surprise: V1 burst rate after efference copy suppression
      4. Route surprise as intrinsic reward to BG
      5. BG + M1 select action

    Exploration comes from BG tonic DA noise, not epsilon-greedy.
    """

    def __init__(
        self,
        encoder: ArcGridEncoder,
        circuit: Circuit,
        *,
        available_actions: list[int],
    ):
        super().__init__(encoder, circuit, entry_name="V1")
        self.available_actions = available_actions
        self._rng = np.random.default_rng()
        self._action_map = self._build_action_map(available_actions)
        self._last_encoding: np.ndarray | None = None
        self._step_count = 0

    def _build_action_map(self, available_actions: list[int]) -> dict[int, int]:
        """Map M1 output indices (0..n-1) to actual GameAction values."""
        return {i: a for i, a in enumerate(available_actions)}

    def step(self, grid: np.ndarray) -> None:
        """Encode grid with efference copy, process circuit, compute intrinsic reward."""
        v1 = self._circuit.region(self._entry_name)

        # Efference copy: V1 expects the grid to look like last frame.
        # Mismatch between expected and actual = what changed = surprise.
        if self._last_encoding is not None:
            v1.set_efference_copy(self._last_encoding)

        # Encode and process
        encoding = self._encoder.encode(grid)
        self.last_encoding = encoding
        output = self._circuit.process(encoding, motor_active=True)
        self.last_output = output

        # Intrinsic reward from V1 burst rate.
        # High burst = unexpected state = positive surprise → curiosity reward.
        # Low burst = predicted state = expected → no reward.
        # This is the dopaminergic response to novel stimuli.
        n_active = max(int(v1.active_columns.sum()), 1)
        n_bursting = int(v1.bursting_columns.sum())
        burst_rate = n_bursting / n_active

        # Convert burst rate to intrinsic reward:
        # baseline burst rate ~0.5 for random input, so center around that.
        # Above baseline = positive surprise, below = negative (boring/expected).
        intrinsic_reward = (burst_rate - 0.5) * 0.1
        if abs(intrinsic_reward) > 0.001:
            self.apply_reward(intrinsic_reward)

        # Save encoding for next step's efference copy
        self._last_encoding = encoding.copy()
        self._step_count += 1

    def decode_action(self) -> tuple[int, dict | None]:
        """Read M1 output. Falls back to BG-biased random if M1 silent."""
        for s in self._circuit._regions.values():
            if s.motor and isinstance(s.region, MotorRegion):
                m_id, _conf = s.region.last_output
                if m_id >= 0 and m_id in self._action_map:
                    action = self._action_map[m_id]
                    data = self._click_data(action) if action >= 6 else None
                    self.last_action = m_id
                    return action, data

        # M1 silent (step 0): use BG output directly to pick action
        bg = self._circuit.region("BG")
        bg_scores = bg._output_group.firing_rate
        best_idx = int(np.argmax(bg_scores))
        action = self.available_actions[best_idx]
        data = self._click_data(action) if action >= 6 else None
        self.last_action = best_idx
        return action, data

    def _click_data(self, action: int) -> dict:
        """Generate click coordinates for click actions (6, 7)."""
        x = int(self._rng.integers(64))
        y = int(self._rng.integers(64))
        return {"x": x, "y": y}

    def act(self, grid: np.ndarray, reward: float) -> tuple[int, dict | None]:
        """Process one frame and return an action."""
        if reward != 0.0:
            self.apply_reward(reward)
        self.step(grid)
        return self.decode_action()

    def reset_episode(self) -> None:
        """Reset per-episode state. Learned weights persist."""
        self._circuit.reset()
        self._encoder.reset()
        self._last_encoding = None
        self._step_count = 0
