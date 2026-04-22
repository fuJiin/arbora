"""Canonical circuit factory -- single source of truth for the 6-region topology.

T1 -> T2 -> T3 -> PFC -> M2 -> M1 with apical feedback, surprise modulation,
thalamic gating, and basal ganglia. All experiment scripts, the REPL, and
sweeps should use build_canonical_circuit() instead of wiring their own.

Override specific region configs or connection params via kwargs. The
factory handles dimension chaining (T1->T2 buffer depth affects T2 input_dim,
PFC receives T2+T3 concatenated, M2 receives T2+PFC, M1 receives M2).
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from arbora.config import (
    CortexConfig,
    _default_motor_config,
    _default_pfc_config,
    _default_premotor_config,
    _default_region2_config,
    _default_region3_config,
    _default_t1_config,
    make_motor_region,
    make_pfc_region,
    make_premotor_region,
    make_sensory_region,
)
from arbora.cortex.circuit import Circuit, ConnectionRole
from arbora.cortex.modulators import SurpriseTracker, ThalamicGate

if TYPE_CHECKING:
    from arbora.cortex.circuit_types import Encoder


def build_canonical_circuit(
    encoder: Encoder,
    *,
    # Circuit-level config
    log_interval: int = 100,
    timeline_interval: int = 100,
    # Per-region config overrides (replace fields on defaults)
    t1_overrides: dict | None = None,
    t2_overrides: dict | None = None,
    t3_overrides: dict | None = None,
    pfc_overrides: dict | None = None,
    m2_overrides: dict | None = None,
    m1_overrides: dict | None = None,
    # Connection params
    t1_t2_buffer_depth: int = 4,
    t2_t3_buffer_depth: int = 8,
    # BG config
    bg_learning_rate: float = 0.05,
    # Whether to finalize
    finalize: bool = True,
) -> Circuit:
    """Build the canonical 6-region cortical circuit.

    Returns a Circuit ready to run. All dimension chaining is handled
    automatically -- you only need to provide the encoder and optional
    config overrides.

    Args:
        encoder: Token encoder (e.g., PositionalCharEncoder).
        t1_overrides: Dict of CortexConfig field overrides for T1.
        t2_overrides: Dict of CortexConfig field overrides for T2.
        t3_overrides: Dict of CortexConfig field overrides for T3.
        pfc_overrides: Dict of CortexConfig field overrides for PFC.
        m2_overrides: Dict of CortexConfig field overrides for M2.
        m1_overrides: Dict of CortexConfig field overrides for M1.
        t1_t2_buffer_depth: Temporal buffer on T1->T2 (default 4).
        t2_t3_buffer_depth: Temporal buffer on T2->T3 (default 8).
        bg_learning_rate: Basal ganglia learning rate.
        finalize: Whether to call finalize() (default True).

    Returns:
        Configured Circuit instance.
    """

    def _apply_overrides(cfg: CortexConfig, overrides: dict | None) -> CortexConfig:
        if overrides:
            return replace(cfg, **overrides)
        return cfg

    # --- Regions ---

    t1_cfg = _apply_overrides(_default_t1_config(), t1_overrides)
    t1 = make_sensory_region(
        t1_cfg,
        encoder.input_dim,
        encoder.encoding_width,
    )

    t2_cfg = _apply_overrides(_default_region2_config(), t2_overrides)
    t2 = make_sensory_region(t2_cfg, t1.n_l23_total * t1_t2_buffer_depth, seed=123)

    t3_cfg = _apply_overrides(_default_region3_config(), t3_overrides)
    t3 = make_sensory_region(t3_cfg, t2.n_l23_total * t2_t3_buffer_depth, seed=789)

    m2_cfg = _apply_overrides(_default_premotor_config(), m2_overrides)
    m2_n_l23 = m2_cfg.n_columns * m2_cfg.n_l23

    m1_cfg = _apply_overrides(_default_motor_config(), m1_overrides)
    output_vocab = [
        ord(ch)
        for ch in encoder._char_to_idx  # type: ignore[attr-defined]
    ]
    m1 = make_motor_region(m1_cfg, m2_n_l23, seed=456)
    # Set vocabulary for L5 output mapping
    m1._output_vocab = np.array(output_vocab, dtype=np.int64)
    m1.n_output_tokens = len(output_vocab)
    n_l5 = m1.n_l5_total
    m1.output_weights = m1._rng.uniform(0, 0.01, size=(n_l5, len(output_vocab)))
    m1.output_mask = (m1._rng.random((n_l5, len(output_vocab))) < 0.5).astype(
        np.float64
    )
    m1.output_weights *= m1.output_mask
    m1._output_eligibility = np.zeros((n_l5, len(output_vocab)))

    pfc_cfg = _apply_overrides(_default_pfc_config(), pfc_overrides)
    pfc = make_pfc_region(
        pfc_cfg,
        t2.n_l23_total + t3.n_l23_total,
        seed=999,
        source_dims=[t2.n_l23_total, t3.n_l23_total],
    )

    m2 = make_premotor_region(
        m2_cfg,
        t2.n_l23_total + pfc.n_l23_total,
        seed=321,
        source_dims=[t2.n_l23_total, pfc.n_l23_total],
    )

    # --- Circuit assembly ---

    circuit = Circuit(
        encoder,
        enable_timeline=timeline_interval > 0,
        timeline_interval=max(timeline_interval, 1),
        diagnostics_interval=log_interval,
    )

    circuit.add_region("T1", t1, entry=True, input_region=True)
    circuit.add_region("T2", t2)
    circuit.add_region("T3", t3)
    circuit.add_region("M1", m1, output_region=True)
    circuit.add_region("PFC", pfc)
    circuit.add_region("M2", m2)

    # --- Feedforward connections (L2/3 -> L4, corticocortical) ---
    # L2/3 is the canonical feedforward source (Felleman & Van Essen 1991).
    # L5 is the feedback/subcortical output layer, NOT the FF source.
    circuit.connect(
        t1.l23,
        t2.l4,
        ConnectionRole.FEEDFORWARD,
        buffer_depth=t1_t2_buffer_depth,
        burst_gate=True,
        surprise_tracker=SurpriseTracker(),
    )
    circuit.connect(
        t2.l23,
        t3.l4,
        ConnectionRole.FEEDFORWARD,
        buffer_depth=t2_t3_buffer_depth,
        burst_gate=True,
        surprise_tracker=SurpriseTracker(),
    )
    circuit.connect(t2.l23, pfc.l4, ConnectionRole.FEEDFORWARD)
    circuit.connect(t3.l23, pfc.l4, ConnectionRole.FEEDFORWARD)
    circuit.connect(t2.l23, m2.l4, ConnectionRole.FEEDFORWARD)
    circuit.connect(pfc.l23, m2.l4, ConnectionRole.FEEDFORWARD)
    circuit.connect(m2.l23, m1.l4, ConnectionRole.FEEDFORWARD)

    # --- Apical feedback (L5 -> {L2/3, L5}, top-down context) ---
    # L5 projects back to lower regions' L2/3 and L5 apical dendrites
    # (via L1 in biology). Each pathway targets both layers.
    gate = ThalamicGate
    # Sensory hierarchy (top-down)
    circuit.connect(t2.l5, t1.l23, ConnectionRole.APICAL, thalamic_gate=gate())
    circuit.connect(t2.l5, t1.l5, ConnectionRole.APICAL, thalamic_gate=gate())
    circuit.connect(t3.l5, t2.l23, ConnectionRole.APICAL, thalamic_gate=gate())
    circuit.connect(t3.l5, t2.l5, ConnectionRole.APICAL, thalamic_gate=gate())
    # Motor hierarchy (bottom-up monitoring)
    circuit.connect(m1.l5, m2.l23, ConnectionRole.APICAL, thalamic_gate=gate())
    circuit.connect(m1.l5, m2.l5, ConnectionRole.APICAL, thalamic_gate=gate())
    circuit.connect(m2.l5, pfc.l23, ConnectionRole.APICAL, thalamic_gate=gate())
    circuit.connect(m2.l5, pfc.l5, ConnectionRole.APICAL, thalamic_gate=gate())
    # Cross-hierarchy
    circuit.connect(
        t1.l5,
        m1.l23,
        ConnectionRole.APICAL,
        thalamic_gate=gate(),
        surprise_tracker=SurpriseTracker(),
    )
    circuit.connect(t1.l5, m1.l5, ConnectionRole.APICAL, thalamic_gate=gate())
    circuit.connect(m1.l5, t1.l23, ConnectionRole.APICAL, thalamic_gate=gate())
    circuit.connect(m1.l5, t1.l5, ConnectionRole.APICAL, thalamic_gate=gate())

    if finalize:
        circuit.finalize()

    return circuit
