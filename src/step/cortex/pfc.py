"""Prefrontal cortex: goal maintenance, context integration, and response planning.

Subclasses CorticalRegion with PFC-specific properties:
- Slow voltage decay (working memory via sustained activity)
- Multi-source input (S2 word-level + S3 topic-level concatenated)
- Three-factor learning with longer eligibility traces
- BG per-stripe input gating (controls when goals update vs maintain)

PFC is agranular cortex — thin/absent L4, receives processed cortical
input rather than raw sensory. Same minicolumn architecture, different
parameters.

Development mirrors infant PFC:
  1. Echo mode: maintain "reproduce input" as a goal
  2. Babble mode: maintain "produce novel output" as a goal
  3. Context-dependent: maintain topic/intent across response generation
"""

import numpy as np

from step.cortex.region import CorticalRegion


class PFCRegion(CorticalRegion):
    """Prefrontal cortex with working memory and goal maintenance.

    Key differences from base CorticalRegion:
    - Slow voltage decay → activity persists across many timesteps
    - Per-stripe organization → groups of columns hold independent goals
    - Confidence signal → derived from activation strength, available
      for downstream monitoring ("I don't know" detection)
    """

    def __init__(
        self,
        input_dim: int,
        *,
        n_columns: int = 16,
        k_columns: int = 4,
        voltage_decay: float = 0.97,
        learning_rate: float = 0.02,
        eligibility_decay: float = 0.98,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(
            input_dim=input_dim,
            n_columns=n_columns,
            k_columns=k_columns,
            voltage_decay=voltage_decay,
            learning_rate=learning_rate,
            eligibility_decay=eligibility_decay,
            seed=seed,
            **kwargs,
        )

        # Global gate: True = accept new input, False = maintain goal.
        # BG controls this via Go/NoGo.
        #
        # Future: per-stripe gating (groups of columns independently
        # updatable) for holding multiple goals. Biologically maps to
        # PFC macrocolumns with separate cortico-BG-thalamic loops.
        # See O'Reilly & Frank 2006 (PBWM) for the stripe model.
        self.gate_open: bool = True

        # Context snapshot: what was active when goal was set.
        # Used for echo mode (M1 tries to reproduce this pattern).
        self._goal_context = np.zeros(self.n_l23_total, dtype=np.float64)

    @property
    def confidence(self) -> float:
        """Current confidence: how strongly are columns activated?

        High activation = strong goal representation = high confidence.
        Low activation = weak/degraded goal = low confidence.
        Can be used downstream: low confidence → "I don't know" response.
        """
        if not self.active_columns.any():
            return 0.0
        # Mean L2/3 firing rate of active columns
        rates = self.firing_rate_l23.reshape(self.n_columns, self.n_l23)
        active_rates = rates[self.active_columns].mean()
        return float(active_rates)

    def process(self, encoding: np.ndarray) -> np.ndarray:
        """Feedforward with global gating.

        When gate is open: process new input normally (update goal).
        When gate is closed: zero feedforward drive, rely on slow
        voltage decay to maintain previous activation (hold goal).
        """
        flat = encoding.flatten().astype(np.float64)

        if self.gate_open:
            neuron_drive = flat @ self.ff_weights
        else:
            # Gate closed: no new input, maintain via slow decay
            neuron_drive = np.zeros(self.n_l4_total)

        self.last_column_drive = neuron_drive.reshape(self.n_columns, self.n_l4).max(
            axis=1
        )
        active = self.step(neuron_drive)

        if self.learning_enabled and self.gate_open:
            self._learn_ff(flat)

        return active

    def snapshot_goal(self) -> None:
        """Capture current L2/3 state as the active goal.

        Called after PFC settles on a goal representation.
        Used for echo mode (compare M1 output against this goal)
        and confidence monitoring.
        """
        self._goal_context[:] = self.firing_rate_l23

    @property
    def goal_context(self) -> np.ndarray:
        """Current goal representation (L2/3 firing rates)."""
        return self._goal_context

    def reset_working_memory(self):
        """Reset transient state, preserving learned weights."""
        super().reset_working_memory()
        self.gate_open = True
        self._goal_context[:] = 0.0
