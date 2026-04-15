"""ARC-AGI-3 agent: V1 → pulvinar → V2 → BG → M1 with efference copy.

Biological sensorimotor loop with hierarchical visual processing:
  V1 encodes grid → pulvinar relays V1 L5 to V2 (transthalamic pathway)
  → V2 learns higher-level abstractions → V2 gates pulvinar (attention)
  → V2 sends apical feedback to V1 → BG selects action → M1 executes
  → V1 gets efference copy → mismatch (burst rate) drives intrinsic
  reward → BG learns from surprise signal.

No epsilon-greedy or other hacks. Exploration comes from BG's tonic DA.
Internal reward comes from V1's burst rate (surprise/novelty).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from arbora.agent import BaseAgent
from arbora.basal_ganglia import BasalGangliaRegion
from arbora.cortex import SensoryRegion
from arbora.cortex.circuit import Circuit, ConnectionRole
from arbora.cortex.motor import MotorRegion
from arbora.thalamus import ThalamicNucleus
from examples.arc.encoder import ArcGridEncoder

if TYPE_CHECKING:
    pass


def build_circuit(
    encoder: ArcGridEncoder,
    *,
    n_actions: int = 7,
    # V1 config — 128 cols, k=16 (12.5% sparsity, enough capacity for
    # diverse visual features without forgetting)
    v1_columns: int = 128,
    v1_k: int = 16,
    v1_cells: int = 4,
    # V2 config — 64 cols, k=8 (higher-level, more abstract)
    v2_columns: int = 64,
    v2_k: int = 8,
    v2_cells: int = 4,
    # BG config — higher learning rate for sparse reward
    bg_learning_rate: float = 0.1,
    seed: int = 42,
) -> Circuit:
    """Build a V1 → pulvinar → V2 → BG → M1 circuit for ARC-AGI-3.

    Transthalamic pathway: V1 L5 → pulvinar → V2 L4.
    All cortico-cortical communication passes through the thalamus —
    there are no direct V1→V2 connections. The pulvinar gates what
    gets relayed based on V2's top-down attention signal.

    V2 sends apical feedback to V1, modulating L5 output via BAC
    firing (coincident feedforward + apical → amplified response).
    """
    v1 = SensoryRegion(
        input_dim=encoder.input_dim,
        encoding_width=encoder.encoding_width,
        n_columns=v1_columns,
        n_l4=v1_cells,
        n_l23=v1_cells,
        # L5 provides saliency-weighted output for BG and pulvinar.
        # Bursting columns → all L5 cells fire (strong signal).
        # Predicted columns → 1 L5 cell fires (weak signal).
        # This routes spatial novelty to action selection and V2.
        n_l5=v1_cells,
        k_columns=v1_k,
        n_l4_lat_segments=8,
        n_synapses_per_segment=32,
        seg_activation_threshold=4,
        seed=seed,
    )

    # Pulvinar: higher-order thalamic relay between V1 and V2.
    # Driver input: V1 L5 (saliency-weighted cortical output).
    # Modulatory gate: V2 L2/3 (top-down attention — what to relay).
    # Gate starts closed — V2 must learn what to attend to before
    # the relay opens. This prevents noise from flooding V2 early.
    pulvinar = ThalamicNucleus(
        input_dim=v1.n_l5_total,  # V1 L5: 128 cols * 4 cells = 512
        relay_dim=v1.n_l5_total,  # Same dimensionality for relay
        relay_gain=1.0,
        burst_gain=3.0,
        gate_threshold=0.0,
        learning_rate=0.01,
        seed=seed + 100,
    )

    # V2: higher-level visual region. Fewer columns (more abstract),
    # same cells per column (maintains representational capacity).
    # Input comes from pulvinar relay, not raw encoding — no encoding_width.
    v2 = SensoryRegion(
        input_dim=pulvinar.relay_dim,  # 512 (pulvinar relay output)
        n_columns=v2_columns,
        n_l4=v2_cells,
        n_l23=v2_cells,
        # L5 for future corticostriatal projections (V2→BG).
        # Not wired to BG yet — V1 L5→BG handles saliency for now.
        n_l5=v2_cells,
        k_columns=v2_k,
        n_l4_lat_segments=8,
        n_synapses_per_segment=32,
        seg_activation_threshold=4,
        seed=seed + 50,
    )

    bg = BasalGangliaRegion(
        # BG receives V1 L5 (saliency-weighted), not L2/3 (full representation).
        # Corticostriatal projections come from L5 — biologically grounded.
        input_dim=v1.n_l5_total,
        n_actions=n_actions,
        learning_rate=bg_learning_rate,
        tonic_da_init=2.0,
        tonic_da_min=1.0,
        # No seed: biological tonic DA noise is genuinely stochastic.
        # This is the primary exploration mechanism — different noise
        # each episode produces different action sequences.
        seed=np.random.default_rng().integers(2**31),
    )
    m1 = MotorRegion(
        input_dim=v1.n_l23_total,
        n_columns=16,
        n_l4=0,
        n_l23=4,
        k_columns=2,
        n_output_tokens=n_actions,
        seed=seed + 200,
    )

    circuit = Circuit(encoder)
    circuit.add_region("V1", v1, entry=True, input_region=True)
    circuit.add_region("pulvinar", pulvinar)
    circuit.add_region("V2", v2)
    circuit.add_region("BG", bg)
    circuit.add_region("M1", m1, output_region=True)

    # --- Transthalamic pathway: V1 → pulvinar → V2 ---
    # V1 L5 → pulvinar: driver input (corticothalamic from L5).
    circuit.connect(v1.l5, pulvinar.input_port, ConnectionRole.FEEDFORWARD)
    # Pulvinar → V2 L4: relay output drives V2's input layer.
    circuit.connect(pulvinar.output_port, v2.input_port, ConnectionRole.FEEDFORWARD)
    # V2 L2/3 → pulvinar: top-down attention gate (MODULATORY).
    # V2's learned representations modulate what V1 activity gets relayed.
    # Gate starts closed — V2 must learn useful representations before
    # it can selectively gate the relay.
    circuit.connect(v2.l23, pulvinar.input_port, ConnectionRole.MODULATORY)

    # --- V2 → V1 apical feedback ---
    # V2 L2/3 sends context to V1 L5 via apical dendrites.
    # This enables BAC firing: V1 L5 neurons that receive both
    # feedforward drive (from L2/3) AND apical context (from V2)
    # fire more strongly than those with feedforward alone.
    circuit.connect(v2.l23, v1.l5, ConnectionRole.APICAL)

    # --- V1 L5 → BG: saliency pathway (unchanged) ---
    # Bursting columns send 4x signal vs predicted columns.
    circuit.connect(v1.l5, bg.input_port, ConnectionRole.FEEDFORWARD)

    # --- V1 L2/3 → M1: motor pathway (unchanged) ---
    circuit.connect(v1.l23, m1.input_port, ConnectionRole.FEEDFORWARD)
    circuit.connect(bg.output_port, m1.input_port, ConnectionRole.MODULATORY)

    circuit.finalize()
    return circuit


class ArcAgent(BaseAgent):
    """Agent with efference copy loop and intrinsic reward from V1 surprise.

    Sensorimotor loop each step:
      1. Set efference copy (predicted next grid = last grid encoding)
      2. Encode new grid through V1 (efference copy suppresses expected input)
      3. Measure surprise: V1 burst rate after efference copy suppression
      4. Route surprise as intrinsic reward to BG
      5. BG + M1 select action

    Exploration comes from BG tonic DA noise, not epsilon-greedy.
    """

    def __init__(
        self,
        encoder: ArcGridEncoder,
        circuit: Circuit,
        *,
        available_actions: list[int],
    ):
        super().__init__(encoder, circuit, entry_name="V1")
        self.available_actions = available_actions
        self._rng = np.random.default_rng()
        self._action_map = self._build_action_map(available_actions)
        self._last_encoding: np.ndarray | None = None
        self._step_count = 0

    def _build_action_map(self, available_actions: list[int]) -> dict[int, int]:
        """Map M1 output indices (0..n-1) to actual GameAction values."""
        return {i: a for i, a in enumerate(available_actions)}

    def step(self, grid: np.ndarray) -> None:
        """Encode grid with efference copy, process circuit, reward."""
        v1 = self._circuit.region(self._entry_name)

        # Efference copy: predict "no change" (simplest prediction).
        # Real efference copy would predict sensory consequences of the
        # motor command; for the baseline, last frame is the prediction.
        # Any mismatch (from agent action or environment) = surprise.
        if self._last_encoding is not None:
            v1.set_efference_copy(self._last_encoding)

        # Encode and process — BG gating controls M1 output.
        encoding = self._encoder.encode(grid)
        self.last_encoding = encoding
        output = self._circuit.process(encoding)
        self.last_output = output

        # Intrinsic reward from V1 burst rate.
        # Pass raw burst rate as reward — let BG's own RPE baseline
        # adapt to the typical level. Above-baseline bursts (genuine
        # surprise) produce positive RPE → Go pathway. Below-baseline
        # (habituated) produces negative RPE → NoGo pathway.
        # No centering or scaling — the BG's reward_baseline handles it.
        n_active = max(int(v1.active_columns.sum()), 1)
        n_bursting = int(v1.bursting_columns.sum())
        burst_rate = n_bursting / n_active

        self.apply_reward(burst_rate)

        # Save encoding for next step's efference copy
        self._last_encoding = encoding.copy()
        self._step_count += 1

    def decode_action(self) -> tuple[int, dict | None]:
        """Read M1 output. Falls back to BG-biased random if M1 silent."""
        motor = self._circuit.output_regions[0]
        m_id, _conf = motor.last_output
        if m_id >= 0 and m_id in self._action_map:
            action = self._action_map[m_id]
            data = self._click_data(action) if action >= 6 else None
            self.last_action = m_id
            return action, data

        # M1 silent (step 0 or BG fully suppressed): use BG output
        # directly to pick action.
        bg = self._circuit.region("BG")
        bg_scores = bg._output_group.firing_rate
        best_idx = int(np.argmax(bg_scores))
        action = self.available_actions[best_idx]
        data = self._click_data(action) if action >= 6 else None
        self.last_action = best_idx
        return action, data

    def _click_data(self, action: int) -> dict:
        """Generate click coordinates for click actions (6, 7)."""
        x = int(self._rng.integers(64))
        y = int(self._rng.integers(64))
        return {"x": x, "y": y}

    def act(self, grid: np.ndarray, reward: float) -> tuple[int, dict | None]:
        """Process one frame and return an action."""
        if reward != 0.0:
            self.apply_reward(reward)
        self.step(grid)
        return self.decode_action()

    def reset_episode(self) -> None:
        """Reset per-episode state. Learned weights persist."""
        self._circuit.reset()
        self._encoder.reset()
        self._last_encoding = None
        self._step_count = 0
