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
        n_stripes: int = 4,
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
        self.n_stripes = n_stripes
        self._cols_per_stripe = n_columns // n_stripes

        # Per-stripe gating state: True = accept new input, False = maintain
        self._stripe_gates = np.ones(n_stripes, dtype=np.bool_)

        # Context snapshot: what was active when goal was set
        # Used for confidence monitoring and echo mode
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

    def set_stripe_gates(self, gates: np.ndarray) -> None:
        """Set per-stripe gating (from BG).

        gates[i] = True → stripe i accepts new input (update goal)
        gates[i] = False → stripe i maintains current state (hold goal)
        """
        self._stripe_gates[:] = gates

    def process(self, encoding: np.ndarray) -> np.ndarray:
        """Feedforward with per-stripe gating.

        Only stripes with open gates process new input. Closed stripes
        maintain their current activation via slow voltage decay.
        """
        flat = encoding.flatten().astype(np.float64)

        # Compute drive for all columns
        neuron_drive = flat @ self.ff_weights

        # Apply stripe gating: zero drive for gated (maintaining) stripes
        for s in range(self.n_stripes):
            if not self._stripe_gates[s]:
                start = s * self._cols_per_stripe * self.n_l4
                end = (s + 1) * self._cols_per_stripe * self.n_l4
                neuron_drive[start:end] = 0.0

        self.last_column_drive = neuron_drive.reshape(
            self.n_columns, self.n_l4
        ).max(axis=1)
        active = self.step(neuron_drive)

        if self.learning_enabled:
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
        self._stripe_gates[:] = True  # Open all gates on reset
        self._goal_context[:] = 0.0
