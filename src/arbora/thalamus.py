"""Thalamic nucleus: gated relay between cortical regions.

Models thalamic relay nuclei (LGN, pulvinar, VA/VL) as lightweight
NeuronGroup-based regions. All cortico-cortical communication in the
brain passes through the thalamus — there are no direct V1→V2
connections. The thalamus gates what gets relayed.

Each nucleus has:
  - Driver input (cortical L5 projections via FEEDFORWARD)
  - Modulatory gate (cortical L6 / TRN / BG via MODULATORY)
  - Relay output (projects to target cortex L4)

Two firing modes:
  - Tonic: faithful relay, gate open. Driver signal passes through
    with learned weights. This is the "attending" state.
  - Burst: amplified transient when gate reopens after inhibition.
    Novel or salient input gets a boost. This is the "orienting" state.

The gate signal arrives via ConnectionRole.MODULATORY on input_port,
same mechanism as BG→M1. Positive modulation opens the gate (tonic
mode), negative closes it (suppressed). Transition from closed→open
triggers burst mode (transient amplification).

Instances of the same class serve different roles:
  - LGN: encoder → V1 (first-order sensory relay)
  - Pulvinar: V1 L5 → V2 (higher-order, attention-gated)
  - VA/VL: cortex → M1 (motor relay, BG-gated)

References:
  Sherman & Guillery (2006) — driver/modulator framework
  Sherman (2016) — first-order vs higher-order thalamic nuclei
"""

from __future__ import annotations

import numpy as np

from arbora.neuron_group import NeuronGroup


class ThalamicNucleus:
    """Gated relay nucleus. NeuronGroup-based, satisfies Region protocol.

    Parameters
    ----------
    input_dim : int
        Dimension of driver input (cortical L5 projections).
    relay_dim : int
        Dimension of relay output (projects to target cortex L4).
    relay_gain : float
        Scaling of driver→relay in tonic mode. Default 1.0.
    burst_gain : float
        Transient amplification when gate transitions closed→open.
        Models T-type calcium channel rebound burst. Default 3.0.
    gate_decay : float
        How quickly the gate state decays toward resting. Default 0.9.
    gate_threshold : float
        Modulation level above which the gate is considered "open".
        Default 0.0 (any positive modulation opens the gate).
    seed : int
        Random seed for weight initialization.
    """

    # NeuronGroup IDs
    RELAY_IN = "relay_in"
    RELAY_OUT = "relay_out"

    def __init__(
        self,
        input_dim: int,
        relay_dim: int,
        *,
        relay_gain: float = 1.0,
        burst_gain: float = 3.0,
        gate_decay: float = 0.9,
        gate_threshold: float = 0.0,
        learning_rate: float = 0.01,
        seed: int = 0,
    ):
        self._rng = np.random.default_rng(seed)
        self._input_dim = input_dim
        self._relay_dim = relay_dim
        self.relay_gain = relay_gain
        self.burst_gain = burst_gain
        self.gate_decay = gate_decay
        self.gate_threshold = gate_threshold
        self.learning_rate = learning_rate
        self.learning_enabled = True

        # Relay weights: driver → output (learned)
        self.relay_weights = self._rng.uniform(0.1, 0.5, size=(input_dim, relay_dim))
        # Clip to [0, 1] consistent with cortical ff_weights
        np.clip(self.relay_weights, 0.0, 1.0, out=self.relay_weights)

        # Gate state: scalar tracking how "open" the relay is.
        # Driven by modulatory input. Decays toward 0 (resting/closed).
        self._gate = 0.0
        self._prev_gate = 0.0  # for detecting closed→open transitions
        self._in_burst = False  # currently in burst mode

        # NeuronGroup ports
        self._input_group = NeuronGroup(
            n_neurons=input_dim, group_id=self.RELAY_IN, region=self
        )
        self._output_group = NeuronGroup(
            n_neurons=relay_dim, group_id=self.RELAY_OUT, region=self
        )

    # ------------------------------------------------------------------
    # Region protocol
    # ------------------------------------------------------------------

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def relay_dim(self) -> int:
        return self._relay_dim

    @property
    def input_port(self) -> NeuronGroup:
        return self._input_group

    @property
    def output_port(self) -> NeuronGroup:
        return self._output_group

    def get_lamina(self, lid: str | object) -> NeuronGroup:
        key = lid.value if hasattr(lid, "value") else str(lid)
        if key == self.RELAY_IN:
            return self._input_group
        if key == self.RELAY_OUT:
            return self._output_group
        raise KeyError(f"ThalamicNucleus has no group {lid!r}")

    def process(self, driver_input: np.ndarray, **kwargs) -> np.ndarray:
        """Relay driver input through learned weights, gated by modulation.

        1. Read modulatory gate signal (set by circuit before process())
        2. Detect burst condition (gate reopening after inhibition)
        3. Compute relay: driver @ relay_weights * gate_gain
        4. Hebbian learning on relay weights (when gate is open)
        5. Set output_port firing_rate

        Returns relay output vector.
        """
        flat = driver_input.flatten().astype(np.float64)

        # --- Gate update ---
        # Modulatory input arrives on input_port via add_modulation()
        self._prev_gate = self._gate
        if self._input_group.modulation is not None:
            # Gate driven by mean modulation (positive = open, negative = close)
            gate_drive = float(self._input_group.modulation.mean())
            self._input_group.clear_modulation()
        else:
            gate_drive = 0.0

        # Gate decays toward 0, modulation pushes it
        self._gate = self._gate * self.gate_decay + gate_drive

        # --- Burst detection ---
        # Burst fires when gate transitions from closed to open
        # (T-type calcium channel rebound after hyperpolarization)
        gate_open = self._gate > self.gate_threshold
        was_closed = self._prev_gate <= self.gate_threshold
        self._in_burst = gate_open and was_closed

        # --- Compute relay ---
        # Tonic: faithful relay with learned weights
        relay = flat @ self.relay_weights  # (relay_dim,)

        # Apply gate: scale by how open the gate is
        if gate_open:
            gain = self.relay_gain
            if self._in_burst:
                gain *= self.burst_gain  # transient amplification
            relay *= gain
        else:
            # Gate closed: strongly attenuate (not fully zero — leak)
            relay *= 0.1

        # --- Learning ---
        # Hebbian: when gate is open, strengthen driver→relay connections
        # for currently active driver neurons → active relay neurons.
        if self.learning_enabled and gate_open:
            # Simple Hebbian: co-active driver and relay neurons
            active_driver = flat > 0.01
            active_relay = relay > 0.01
            if active_driver.any() and active_relay.any():
                # Outer product LTP on active pairs, masked by existing weights
                driver_idx = np.flatnonzero(active_driver)
                relay_idx = np.flatnonzero(active_relay)
                self.relay_weights[np.ix_(driver_idx, relay_idx)] += (
                    self.learning_rate * flat[driver_idx, np.newaxis]
                )
                np.clip(self.relay_weights, 0.0, 1.0, out=self.relay_weights)

        # --- Set output ---
        self._output_group.firing_rate[:] = np.clip(relay, 0.0, None)

        return self._output_group.firing_rate

    def apply_reward(self, reward: float) -> None:
        """No-op. Thalamic relay doesn't do reward-modulated learning.

        Thalamic plasticity is Hebbian (activity-dependent), not
        reward-gated. The gate signal from cortex/BG carries the
        "what to attend to" information; reward shapes that signal
        at its source (cortex, BG), not in the thalamus itself.
        """

    def reset_working_memory(self) -> None:
        """Reset transient state. Preserves learned relay weights."""
        self._gate = 0.0
        self._prev_gate = 0.0
        self._in_burst = False
        self._output_group.firing_rate[:] = 0.0
        self._input_group.clear_modulation()
