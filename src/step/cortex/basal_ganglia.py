"""Basal ganglia: reward-modulated go/no-go gating on motor output.

Models the cortico-basal ganglia-thalamic loop:
  Cortex (S1 L2/3) → Striatum → GPi → Thalamus → gate on M1 output

The striatum learns a context→gate mapping via three-factor plasticity:
  dw = learning_rate * reward * context * gate_error

Direct pathway (D1): context predicts GO → open gate → allow speech
Indirect pathway (D2): context predicts NO-GO → close gate → silence

Dopamine (reward signal) biases the balance:
  positive reward → strengthen active pathway (reinforce current decision)
  negative reward → weaken active pathway (change behavior)
"""

import numpy as np


class BasalGanglia:
    """Go/no-go gate learned from reward and sensory context.

    Receives S1's L2/3 firing rates as context (same signal M1 sees).
    Learns which contexts should open (speak) vs close (silent) the gate.
    Gate signal multiplies M1's output scores before the speak/silent
    decision.

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
        seed: int = 0,
    ):
        self._rng = np.random.default_rng(seed)
        self.context_dim = context_dim
        self.learning_rate = learning_rate
        self.eligibility_decay = eligibility_decay

        # Corticostriatal weights: context → go signal
        # Small random init centered near 0 (gate starts neutral)
        self.go_weights = self._rng.normal(0, 0.01, size=context_dim)

        # Eligibility trace for three-factor learning
        self._trace = np.zeros(context_dim)

        # Current gate state (updated each step)
        self.gate_value: float = 0.5
        self._last_context = np.zeros(context_dim)

    def step(self, context: np.ndarray) -> float:
        """Compute gate value from sensory context.

        Args:
            context: S1's L2/3 firing rates (same as M1's input).

        Returns:
            Gate value in [0, 1]. Multiply with M1 output_scores.
        """
        self._last_context = context.copy()

        # Striatal activation: weighted sum of context
        activation = float(np.dot(self.go_weights, context))

        # Sigmoid squash to [0, 1]
        self.gate_value = 1.0 / (1.0 + np.exp(-activation))

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
