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

from step.cortex.region import CorticalRegion


class PFCRegion(CorticalRegion):
    """Prefrontal cortex with working memory and goal maintenance.

    Key differences from base CorticalRegion:
    - Slow voltage decay → activity persists across many timesteps
    - Three-factor learning → ff_weights learn from reward, not just
      input statistics. PFC learns to produce useful goal patterns.
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
        eligibility_clip: float = 0.05,
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

        # Three-factor learning: eligibility traces on ff_weights.
        # Slower decay (0.98 ≈ 50-step window) than Motor (0.95 ≈ 20-step)
        # to match PFC's slow voltage dynamics.
        self._ff_eligibility = np.zeros_like(self.ff_weights)
        self._eligibility_clip = eligibility_clip

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

        Eligibility traces decay unconditionally — even when gate is
        closed. Otherwise stale traces from the listen phase persist
        through the entire hold phase and get consolidated by reward
        that was earned many steps later.
        """
        # Always decay eligibility traces (even gate-closed)
        self._ff_eligibility *= self.eligibility_decay

        flat = encoding.flatten().astype(np.float64)

        if self.gate_open:
            neuron_drive = flat @ self.ff_weights
        else:
            # Gate closed: no new input, maintain via slow decay
            neuron_drive = np.zeros(self.n_l4_total)

        self.last_column_drive = neuron_drive.reshape(
            self.n_columns, self.n_l4
        ).max(axis=1)
        active = self.step(neuron_drive)

        if self.learning_enabled and self.gate_open:
            self._learn_ff(flat)

        return active

    def _learn_ff(self, flat_input: np.ndarray):
        """Three-factor learning with STDP-like presynaptic traces.

        Uses pre_trace (if enabled) for temporal credit in the
        eligibility trace. Inputs that fired before PFC activated
        get credit, not just inputs active at the same time.
        Weights updated only when apply_reward() is called.
        """
        # Update presynaptic trace
        if self._pre_trace is not None:
            self._pre_trace *= self._pre_trace_decay
            self._pre_trace += flat_input
            ltp_signal = self._pre_trace
        else:
            ltp_signal = flat_input

        # Note: eligibility trace decay happens in process()

        active_cols = np.nonzero(self.active_columns)[0]
        if len(active_cols) == 0:
            return

        voltage_by_col = self.voltage_l4.reshape(
            self.n_columns, self.n_l4
        )
        active_by_col = self.active_l4.reshape(
            self.n_columns, self.n_l4
        )
        is_burst = self.bursting_columns[active_cols]

        winner_indices = np.empty(len(active_cols), dtype=np.intp)
        if is_burst.any():
            winner_indices[is_burst] = (
                active_cols[is_burst] * self.n_l4
                + voltage_by_col[active_cols[is_burst]].argmax(axis=1)
            )
        precise = ~is_burst
        if precise.any():
            winner_indices[precise] = (
                active_cols[precise] * self.n_l4
                + active_by_col[active_cols[precise]].argmax(axis=1)
            )

        # Record in eligibility trace using temporal signal
        self._ff_eligibility[:, winner_indices] += (
            self.learning_rate * ltp_signal[:, np.newaxis]
        )

    def apply_reward(self, reward: float) -> None:
        """Consolidate eligibility traces into weights using reward.

        Three-factor rule: dw = reward * eligibility_trace
        Positive reward → strengthen mappings that led to good output
        Negative reward → weaken mappings that led to bad output

        Biologically: dopamine from VTA gates synaptic consolidation
        in PFC via D1/D5 receptors. High DA = strengthen recent
        synaptic changes. Low DA = let them decay.
        """
        if not self.learning_enabled:
            return
        if abs(reward) < 1e-6:
            return

        # Clamp eligibility traces before consolidation
        if self._eligibility_clip > 0:
            np.clip(
                self._ff_eligibility,
                -self._eligibility_clip,
                self._eligibility_clip,
                out=self._ff_eligibility,
            )

        self.ff_weights += reward * self._ff_eligibility
        self.ff_weights *= self.ff_mask
        np.clip(self.ff_weights, 0, 1, out=self.ff_weights)

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
        self._ff_eligibility[:] = 0.0
