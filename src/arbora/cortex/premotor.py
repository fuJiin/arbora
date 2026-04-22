"""Premotor cortex (M2): sequential motor planning.

Translates abstract goals from PFC into temporal character sequences
for M1 to execute. Uses the same minicolumn architecture as all other
regions — the key difference is HOW it's wired:

- PFC feedforward: static goal (WHAT to produce)
- T2 feedforward: word-level context (linguistic context)
- Lateral segments: temporal sequencing (WHERE in the sequence)
- M1 feedforward output: current step (drive M1's column competition)

The lateral segments learn character sequence patterns through normal
Hebbian learning — the same mechanism T2 uses to learn word patterns.
T2 predicts "what comes next" bottom-up; M2 generates "what comes next"
top-down. Same segments, different direction.

Biologically maps to Broca's area / ventral premotor cortex:
- Patients with Broca's damage can understand words but can't sequence
  them for production
- Mirror neurons fire both when hearing and producing sequences
- Preparatory activity encodes upcoming sequence steps
"""

import numpy as np

from arbora.config import PlasticityRule
from arbora.cortex.region import CorticalRegion


class PremotorRegion(CorticalRegion):
    """Premotor cortex: goal-driven sequence generation.

    Receives PFC goal (feedforward) and generates a temporal sequence
    of M1 drive patterns. Lateral L2/3 segments learn common character
    sequences so M2 can "play back" learned motor programs.

    The output to M1 is feedforward (additive drive), replacing the
    direct PFC->M1 goal drive for sequences longer than 2-3 chars.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        n_columns: int = 32,
        k_columns: int = 4,
        voltage_decay: float = 0.7,
        learning_rate: float = 0.05,
        plasticity_rule: PlasticityRule = PlasticityRule.THREE_FACTOR,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(
            input_dim=input_dim,
            n_columns=n_columns,
            k_columns=k_columns,
            voltage_decay=voltage_decay,
            learning_rate=learning_rate,
            plasticity_rule=plasticity_rule,
            seed=seed,
            **kwargs,
        )

        # Goal drive from PFC (set externally each step)
        self._goal_drive: np.ndarray | None = None
        self._goal_weights: np.ndarray | None = None

    def init_goal_input(self, pfc_dim: int) -> None:
        """Initialize PFC→M2 goal weights."""
        self._goal_weights = self._rng.uniform(
            0,
            0.01,
            size=(pfc_dim, self.n_l4_total),
        )
        mask = self._rng.random((pfc_dim, self.n_l4_total)) < 0.5
        self._goal_weights *= mask

    def set_goal_drive(self, pfc_firing_rate: np.ndarray) -> None:
        """Set PFC goal for next process() call."""
        self._goal_drive = pfc_firing_rate

    def process(
        self,
        encoding: np.ndarray,
        *,
        forced_columns: np.ndarray | None = None,
    ) -> np.ndarray:
        """Feedforward with additive PFC goal drive.

        encoding: T2 word-level context (feedforward from sensory)
        goal_drive: PFC goal (additive, biases which sequence unfolds)

        The combination of T2 context + PFC goal + lateral segment
        predictions drives column selection. Lateral segments carry
        temporal state ("just produced 'h', next is 'i'").
        """
        flat = encoding.flatten().astype(np.float64)
        neuron_drive = flat @ self.ff_weights

        # Add PFC goal drive
        if self._goal_drive is not None and self._goal_weights is not None:
            neuron_drive += self._goal_drive @ self._goal_weights
            self._goal_drive = None

        self.last_column_drive = neuron_drive.reshape(self.n_columns, self.n_l4).max(
            axis=1
        )
        active = self.step(neuron_drive)

        if self.learning_enabled:
            self._learn_ff(flat)

        return active
