"""Prefrontal cortex: goal maintenance, context integration, and response planning.

Subclasses CorticalRegion with PFC-specific properties:
- Slow voltage decay (working memory via sustained activity)
- Multi-source input (S2 word-level + S3 topic-level concatenated)
- Three-factor learning: eligibility traces + reward consolidation
- BG per-stripe input gating (controls when goals update vs maintain)

PFC is agranular cortex — thin/absent L4, receives processed cortical
input rather than raw sensory. Same minicolumn architecture, different
parameters.

Three-factor learning on ff_weights:
  During listen phase (gate open), PFC processes S2+S3 input and records
  Hebbian coincidences in eligibility traces (not weights). After the
  speak phase, reward consolidates traces into weights. This teaches PFC
  to produce activation patterns that lead to downstream reward — not
  just patterns that represent the input.

  Biologically: PFC neurons receive dopaminergic projections from VTA.
  DA modulates synaptic plasticity via D1/D5 receptors, implementing
  the three-factor rule: pre x post x DA -> dw. This is how PFC learns
  which representations are useful for goal-directed behavior.

Development mirrors infant PFC:
  1. Echo mode: maintain "reproduce input" as a goal
  2. Babble mode: maintain "produce novel output" as a goal
  3. Context-dependent: maintain topic/intent across response generation
"""

import numpy as np

from step.config import PlasticityRule
from step.cortex.region import CorticalRegion


class PFCRegion(CorticalRegion):
    """Prefrontal cortex with working memory and goal maintenance.

    Key differences from base CorticalRegion:
    - Slow voltage decay -> activity persists across many timesteps
    - Three-factor learning -> ff_weights learn from reward, not just
      input statistics. PFC learns to produce useful goal patterns.
    - Per-stripe organization -> groups of columns hold independent goals
    - Confidence signal -> derived from activation strength, available
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
        eligibility_clip: float = 0.05,
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
            eligibility_decay=eligibility_decay,
            plasticity_rule=plasticity_rule,
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

        # Eligibility clip: prevents unbounded trace accumulation.
        self._eligibility_clip = eligibility_clip

    @property
    def confidence(self) -> float:
        """Current confidence: how strongly are columns activated?

        High activation = strong goal representation = high confidence.
        Low activation = weak/degraded goal = low confidence.
        Can be used downstream: low confidence -> "I don't know" response.
        """
        if not self.active_columns.any():
            return 0.0
        # Mean L2/3 firing rate of active columns
        rates = self.l23.firing_rate.reshape(self.n_columns, self.n_l23)
        active_rates = rates[self.active_columns].mean()
        return float(active_rates)

    def process(
        self,
        encoding: np.ndarray,
        *,
        forced_columns: np.ndarray | None = None,
    ) -> np.ndarray:
        """Feedforward with global gating.

        When gate is open: process new input normally (update goal).
        When gate is closed: zero feedforward drive, rely on slow
        voltage decay to maintain previous activation (hold goal).

        When gate is closed, eligibility traces still decay -- otherwise
        stale traces from the listen phase persist through the entire
        hold phase and get consolidated by reward earned many steps later.
        (When gate is open, the base _learn_ff handles the decay.)
        """
        flat = encoding.flatten().astype(np.float64)

        if self.gate_open:
            neuron_drive = flat @ self.ff_weights
        else:
            # Gate closed: no new input, maintain via slow decay.
            # Decay eligibility traces even though we skip _learn_ff.
            if self._ff_eligibility is not None:
                self._ff_eligibility *= self.eligibility_decay
            neuron_drive = np.zeros(self.n_l4_total)

        self.last_column_drive = neuron_drive.reshape(self.n_columns, self.n_l4).max(
            axis=1
        )
        active = self.step(neuron_drive)

        if self.learning_enabled and self.gate_open:
            self._learn_ff(flat)

        return active

    # _learn_ff() and apply_reward() inherited from CorticalRegion.
    # Base _learn_ff dispatches to _learn_ff_three_factor which handles
    # eligibility trace decay + accumulation. apply_reward() handles
    # clip + consolidation into ff_weights.

    def snapshot_goal(self) -> None:
        """Capture current L2/3 state as the active goal.

        Called after PFC settles on a goal representation.
        Used for echo mode (compare M1 output against this goal)
        and confidence monitoring.
        """
        self._goal_context[:] = self.l23.firing_rate

    @property
    def goal_context(self) -> np.ndarray:
        """Current goal representation (L2/3 firing rates)."""
        return self._goal_context

    def reset_working_memory(self):
        """Reset transient state, preserving learned weights."""
        super().reset_working_memory()
        self.gate_open = True
        self._goal_context[:] = 0.0
