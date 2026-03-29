"""Basal ganglia region: per-action Go/NoGo with tonic DA exploration.

Models the cortico-basal ganglia-thalamic loop:
  Cortex → Striatum (Go/NoGo per action) → GPi → Thalamus → M1 modulation

Per-action channels (not a scalar gate):
  Go[a] = cortical_input @ go_weights[:, a]  (D1 direct pathway)
  NoGo[a] = cortical_input @ nogo_weights[:, a]  (D2 indirect pathway)
  action_bias[a] = Go[a] - NoGo[a] + tonic_da_noise

Tonic dopamine models exploration:
  High tonic DA (reward uncertainty) → noisy action selection → exploration
  Low tonic DA (stable rewards) → sharp selection → exploitation

Learning is asymmetric via reward prediction error (RPE):
  +RPE → Go pathway LTP (strengthen what worked)
  -RPE → NoGo pathway LTP (suppress what didn't)

Output arrives at M1 as a MODULATORY connection — additive bias on
M1's input_port voltage before k-WTA column selection.
"""

from __future__ import annotations

import numpy as np

from step.cortex.lamina import Lamina, LaminaID


class BasalGangliaRegion:
    """Per-action Go/NoGo channels with tonic dopamine exploration.

    Not a CorticalRegion — no dendritic segments, no k-WTA, no apical.
    Satisfies the Region protocol: has input_port/output_port, process(),
    apply_reward(), reset_working_memory().

    The output_port's firing_rate IS the per-action disinhibition signal,
    delivered to M1 via a MODULATORY connection.
    """

    def __init__(
        self,
        input_dim: int,
        n_actions: int,
        *,
        learning_rate: float = 0.01,
        eligibility_decay: float = 0.95,
        tonic_da_init: float = 0.5,
        tonic_da_decay: float = 0.99,
        seed: int = 0,
    ):
        self._rng = np.random.default_rng(seed)
        self._input_dim = input_dim
        self._n_actions = n_actions
        self.learning_rate = learning_rate
        self.eligibility_decay = eligibility_decay

        # D1 (Go) and D2 (NoGo) corticostriatal weights
        self.go_weights = self._rng.normal(0, 0.01, size=(input_dim, n_actions))
        self.nogo_weights = self._rng.normal(0, 0.01, size=(input_dim, n_actions))

        # Eligibility traces (three-factor: context x action selection)
        self._go_trace = np.zeros((input_dim, n_actions))
        self._nogo_trace = np.zeros((input_dim, n_actions))

        # Tonic DA: tracks reward uncertainty for exploration
        self._tonic_da = tonic_da_init
        self._tonic_da_decay = tonic_da_decay
        self._rpe_var_ema = tonic_da_init**2
        self._reward_baseline = 0.0

        # Lamina-compatible ports for circuit.connect()
        # Input: single "column" with input_dim neurons
        self._input_lam = Lamina(
            n_columns=1, n_per_col=input_dim, lamina_id=LaminaID.L4
        )
        self._input_lam.region = self
        # Output: single "column" with n_actions neurons
        self._output_lam = Lamina(
            n_columns=1, n_per_col=n_actions, lamina_id=LaminaID.L23
        )
        self._output_lam.region = self

        # Stub properties for circuit compatibility
        self.n_columns = 1
        self.n_l4 = input_dim
        self.n_l23 = n_actions
        self.active_columns = np.ones(1, dtype=np.bool_)
        self.bursting_columns = np.zeros(1, dtype=np.bool_)
        self.learning_enabled = True

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def n_actions(self) -> int:
        return self._n_actions

    @property
    def input_port(self) -> Lamina:
        return self._input_lam

    @property
    def output_port(self) -> Lamina:
        return self._output_lam

    def get_lamina(self, lid: LaminaID) -> Lamina:
        """Look up a lamina by ID (for circuit wiring compatibility)."""
        if lid == LaminaID.L4:
            return self._input_lam
        if lid == LaminaID.L23:
            return self._output_lam
        raise KeyError(f"BasalGangliaRegion has no lamina {lid}")

    def process(self, cortical_input: np.ndarray, **kwargs) -> np.ndarray:
        """Compute per-action disinhibition from cortical firing rate.

        Go - NoGo + tonic DA noise → output_port firing_rate.
        """
        flat = cortical_input.flatten().astype(np.float64)

        # D1 (Go) and D2 (NoGo) striatal activation
        go_act = flat @ self.go_weights  # (n_actions,)
        nogo_act = flat @ self.nogo_weights  # (n_actions,)

        # Action value: Go - NoGo
        action_value = go_act - nogo_act

        # Tonic DA exploration noise
        noise = self._rng.normal(0, max(self._tonic_da, 0.01), size=self._n_actions)
        action_bias = action_value + noise

        # Sigmoid to [0, 1] range for output
        action_bias = 1.0 / (1.0 + np.exp(-np.clip(action_bias, -10, 10)))

        # Update eligibility traces: record cortical input for all actions.
        # The asymmetry comes at reward time (+RPE → Go, -RPE → NoGo),
        # not at trace accumulation time.
        self._go_trace *= self.eligibility_decay
        self._nogo_trace *= self.eligibility_decay
        self._go_trace += flat[:, np.newaxis]
        self._nogo_trace += flat[:, np.newaxis]

        # Set output port firing rate (consumed by MODULATORY connection)
        self._output_lam.firing_rate[:] = action_bias

        return action_bias

    def apply_reward(self, reward: float) -> None:
        """Asymmetric three-factor learning from reward prediction error.

        +RPE → Go weights LTP (strengthen actions that led to reward)
        -RPE → NoGo weights LTP (suppress actions that led to punishment)

        Also updates tonic DA level (exploration temperature).
        """
        rpe = reward - self._reward_baseline
        self._reward_baseline += 0.01 * (reward - self._reward_baseline)

        # Asymmetric learning
        if rpe > 0:
            self.go_weights += self.learning_rate * rpe * self._go_trace
        elif rpe < 0:
            self.nogo_weights += self.learning_rate * abs(rpe) * self._nogo_trace

        # Clip weights to prevent unbounded growth
        np.clip(self.go_weights, -1.0, 1.0, out=self.go_weights)
        np.clip(self.nogo_weights, -1.0, 1.0, out=self.nogo_weights)

        # Update tonic DA from RPE variance (exploration temperature)
        self._rpe_var_ema = (
            self._tonic_da_decay * self._rpe_var_ema
            + (1 - self._tonic_da_decay) * rpe**2
        )
        self._tonic_da = float(np.sqrt(self._rpe_var_ema))

    def reset_working_memory(self) -> None:
        """Reset transient state. Preserves learned weights + tonic DA."""
        self._go_trace[:] = 0.0
        self._nogo_trace[:] = 0.0
        self._output_lam.firing_rate[:] = 0.5
        self._reward_baseline = 0.0
