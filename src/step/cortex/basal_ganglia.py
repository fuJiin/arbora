"""Basal ganglia: reward-modulated go/no-go gating on motor output.

Models the cortico-basal ganglia-thalamic loop:
  Cortex (S1 precision) → Striatum → GPi → Thalamus → gate on M1 output

The striatum learns a context→gate mapping via three-factor plasticity:
  dw = learning_rate * reward * eligibility_trace

Direct pathway (D1): context predicts GO → open gate → allow speech
Indirect pathway (D2): context predicts NO-GO → close gate → silence

Dopamine (reward signal) biases the balance:
  positive reward → strengthen D1 (go) → open gate
  negative reward → strengthen D2 (no-go) → close gate

Exploration noise (models tonic dopamine variability in striatum):
  Gaussian noise on activation prevents gate collapse during early learning.
  Decays as weights grow stronger (signal-to-noise improves with experience).
"""

import numpy as np


class BasalGanglia:
    """Go/no-go gate learned from reward and cortical precision context.

    Receives per-column precision state (1 = predicted correctly, 0 = bursting)
    plus overall precision fraction. During EOM repetition, precision is high
    (dense 1s). During novel input, precision is low (sparse 1s).

    The gate is a scalar in [0, 1]:
      0.0 = fully closed (no-go dominates, M1 silenced)
      1.0 = fully open (go dominates, M1 speaks freely)
    """

    def __init__(
        self,
        context_dim: int,
        *,
        learning_rate: float = 0.01,
        eligibility_decay: float = 0.95,
        exploration_noise: float = 0.5,
        seed: int = 0,
    ):
        self._rng = np.random.default_rng(seed)
        self.context_dim = context_dim
        self.learning_rate = learning_rate
        self.eligibility_decay = eligibility_decay
        self.exploration_noise = exploration_noise

        # Corticostriatal weights: context → go signal
        # Small random init centered near 0 (gate starts neutral)
        self.go_weights = self._rng.normal(0, 0.01, size=context_dim)

        # Eligibility trace for three-factor learning
        self._trace = np.zeros(context_dim)

        # Current gate state (updated each step)
        self.gate_value: float = 0.5
        self._last_context = np.zeros(context_dim)

    def step(self, context: np.ndarray) -> float:
        """Compute gate value from cortical precision context.

        Args:
            context: Per-column precision state + precision fraction.

        Returns:
            Gate value in [0, 1]. Multiply with M1 output_scores.
        """
        self._last_context = context.copy()

        # Striatal activation: weighted sum of context
        activation = float(np.dot(self.go_weights, context))

        # Exploration noise: prevents gate collapse during early learning.
        # Models tonic dopamine variability in striatum.
        noise = self._rng.normal(0, self.exploration_noise)
        activation += noise

        # Sigmoid squash to [0, 1]
        self.gate_value = 1.0 / (1.0 + np.exp(-np.clip(activation, -10, 10)))

        # Update eligibility trace: decay + current context contribution
        self._trace *= self.eligibility_decay
        # Trace tracks which context features were active (D1/D2 model:
        # reward sign determines go vs no-go, not gate state)
        self._trace += context

        return self.gate_value

    def reward(self, reward_signal: float) -> None:
        """Apply reward to update gate weights.

        Three-factor update: dw = lr * reward * eligibility_trace

        D1/D2 pathway model:
          Positive reward (dopamine) → strengthen D1 (go) → open gate
          Negative reward (no dopamine) → strengthen D2 (no-go) → close gate
        """
        self.go_weights += (
            self.learning_rate * reward_signal * self._trace
        )

    def reset(self) -> None:
        """Reset transient state at story boundaries."""
        self._trace[:] = 0.0
        self.gate_value = 0.5
        self._last_context[:] = 0.0
