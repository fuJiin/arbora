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

from step.neuron_group import NeuronGroup


class BasalGangliaRegion:
    """Per-action Go/NoGo channels with tonic dopamine exploration.

    Not a CorticalRegion — no laminae, no columns, no dendritic segments.
    Uses NeuronGroup (not Lamina) for its input/output ports.

    Satisfies the Region protocol: has input_port/output_port, process(),
    apply_reward(), reset_working_memory().
    """

    def __init__(
        self,
        input_dim: int,
        n_actions: int,
        *,
        learning_rate: float = 0.01,
        eligibility_decay: float = 0.95,
        tonic_da_init: float = 2.0,
        tonic_da_decay: float = 0.99,
        seed: int = 0,
    ):
        self._rng = np.random.default_rng(seed)
        self._input_dim = input_dim
        self._n_actions = n_actions
        self.learning_rate = learning_rate
        self.eligibility_decay = eligibility_decay
        self.learning_enabled = True

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

        # NeuronGroup ports for circuit.connect()
        # Striatum: where cortical projections arrive (input)
        # GPi: where disinhibition signal leaves (output to thalamus→M1)
        self._input_group = NeuronGroup(
            n_neurons=input_dim, group_id="striatum", region=self
        )
        self._output_group = NeuronGroup(
            n_neurons=n_actions, group_id="gpi", region=self
        )

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def n_actions(self) -> int:
        return self._n_actions

    @property
    def input_port(self) -> NeuronGroup:
        return self._input_group

    @property
    def output_port(self) -> NeuronGroup:
        return self._output_group

    def get_lamina(self, lid: str | object) -> NeuronGroup:
        """Look up a neuron group by ID (for circuit wiring compatibility).

        Accepts string IDs ("striatum", "gpi") or LaminaID enum values.
        """
        key = lid.value if hasattr(lid, "value") else str(lid)
        if key in ("striatum", "L4"):
            return self._input_group
        if key in ("gpi", "L2/3"):
            return self._output_group
        raise KeyError(f"BasalGangliaRegion has no group {lid!r}")

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

        # Tonic DA exploration noise (large early -> exploration, small late -> exploit)
        noise = self._rng.normal(0, max(self._tonic_da, 0.01), size=self._n_actions)
        action_bias = action_value + noise

        # Clamp to prevent extreme values
        np.clip(action_bias, -3.0, 3.0, out=action_bias)

        # Update eligibility traces: decay old, accumulate current input.
        self._go_trace *= self.eligibility_decay
        self._nogo_trace *= self.eligibility_decay
        self._go_trace += (1 - self.eligibility_decay) * flat[:, np.newaxis]
        self._nogo_trace += (1 - self.eligibility_decay) * flat[:, np.newaxis]

        # Set output group firing rate (consumed by MODULATORY connection)
        self._output_group.firing_rate[:] = action_bias

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
        self._output_group.firing_rate[:] = 0.0
        self._reward_baseline = 0.0
