"""Canonical circuit factory -- single source of truth for the 6-region topology.

S1 -> S2 -> S3 -> PFC -> M2 -> M1 with apical feedback, surprise modulation,
thalamic gating, and basal ganglia. All experiment scripts, the REPL, and
sweeps should use build_canonical_circuit() instead of wiring their own.

Override specific region configs or connection params via kwargs. The
factory handles dimension chaining (S1->S2 buffer depth affects S2 input_dim,
PFC receives S2+S3 concatenated, M2 receives S2+PFC, M1 receives M2).
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from step.config import (
    CortexConfig,
    _default_motor_config,
    _default_pfc_config,
    _default_premotor_config,
    _default_region2_config,
    _default_region3_config,
    _default_s1_config,
    make_motor_region,
    make_pfc_region,
    make_premotor_region,
    make_sensory_region,
)
from step.cortex.basal_ganglia import BasalGanglia
from step.cortex.circuit import Circuit, ConnectionRole
from step.cortex.modulators import SurpriseTracker, ThalamicGate

if TYPE_CHECKING:
    from step.cortex.circuit_types import Encoder


def build_canonical_circuit(
    encoder: Encoder,
    *,
    # Circuit-level config
    log_interval: int = 100,
    timeline_interval: int = 100,
    # Per-region config overrides (replace fields on defaults)
    s1_overrides: dict | None = None,
    s2_overrides: dict | None = None,
    s3_overrides: dict | None = None,
    pfc_overrides: dict | None = None,
    m2_overrides: dict | None = None,
    m1_overrides: dict | None = None,
    # Connection params
    s1_s2_buffer_depth: int = 4,
    s2_s3_buffer_depth: int = 8,
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
        s1_overrides: Dict of CortexConfig field overrides for S1.
        s2_overrides: Dict of CortexConfig field overrides for S2.
        s3_overrides: Dict of CortexConfig field overrides for S3.
        pfc_overrides: Dict of CortexConfig field overrides for PFC.
        m2_overrides: Dict of CortexConfig field overrides for M2.
        m1_overrides: Dict of CortexConfig field overrides for M1.
        s1_s2_buffer_depth: Temporal buffer on S1->S2 (default 4).
        s2_s3_buffer_depth: Temporal buffer on S2->S3 (default 8).
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

    s1_cfg = _apply_overrides(_default_s1_config(), s1_overrides)
    s1 = make_sensory_region(
        s1_cfg,
        encoder.input_dim,
        encoder.encoding_width,  # type: ignore[attr-defined]
    )

    s2_cfg = _apply_overrides(_default_region2_config(), s2_overrides)
    s2 = make_sensory_region(s2_cfg, s1.n_l23_total * s1_s2_buffer_depth, seed=123)

    s3_cfg = _apply_overrides(_default_region3_config(), s3_overrides)
    s3 = make_sensory_region(s3_cfg, s2.n_l23_total * s2_s3_buffer_depth, seed=789)

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
        s2.n_l23_total + s3.n_l23_total,
        seed=999,
        source_dims=[s2.n_l23_total, s3.n_l23_total],
    )

    m2 = make_premotor_region(
        m2_cfg,
        s2.n_l23_total + pfc.n_l23_total,
        seed=321,
        source_dims=[s2.n_l23_total, pfc.n_l23_total],
    )

    # --- Circuit assembly ---

    circuit = Circuit(
        encoder,
        enable_timeline=timeline_interval > 0,
        timeline_interval=max(timeline_interval, 1),
        diagnostics_interval=log_interval,
    )

    circuit.add_region("S1", s1, entry=True)
    circuit.add_region("S2", s2)
    circuit.add_region("S3", s3)
    bg = BasalGanglia(
        context_dim=s1.n_columns + 1,
        learning_rate=bg_learning_rate,
        seed=789,
    )
    circuit.add_region("M1", m1, basal_ganglia=bg)
    circuit.add_region("PFC", pfc)
    circuit.add_region("M2", m2)

    # --- Feedforward connections (L2/3 -> L4, corticocortical) ---
    circuit.connect(
        s1.l23,
        s2.l4,
        ConnectionRole.FEEDFORWARD,
        buffer_depth=s1_s2_buffer_depth,
        burst_gate=True,
        surprise_tracker=SurpriseTracker(),
    )
    circuit.connect(
        s2.l23,
        s3.l4,
        ConnectionRole.FEEDFORWARD,
        buffer_depth=s2_s3_buffer_depth,
        burst_gate=True,
        surprise_tracker=SurpriseTracker(),
    )
    circuit.connect(s2.l23, pfc.l4, ConnectionRole.FEEDFORWARD)
    circuit.connect(s3.l23, pfc.l4, ConnectionRole.FEEDFORWARD)
    circuit.connect(s2.l23, m2.l4, ConnectionRole.FEEDFORWARD)
    circuit.connect(pfc.l23, m2.l4, ConnectionRole.FEEDFORWARD)
    circuit.connect(m2.l23, m1.l4, ConnectionRole.FEEDFORWARD)

    # --- Apical feedback (L2/3 -> {L2/3, L5}, top-down context) ---
    # Each feedback pathway targets both L2/3 and L5 apical dendrites
    # on the target region, matching cortical L1 input to apical tufts.
    gate = ThalamicGate
    # Sensory hierarchy (top-down)
    circuit.connect(s2.l23, s1.l23, ConnectionRole.APICAL, thalamic_gate=gate())
    circuit.connect(s2.l23, s1.l5, ConnectionRole.APICAL, thalamic_gate=gate())
    circuit.connect(s3.l23, s2.l23, ConnectionRole.APICAL, thalamic_gate=gate())
    circuit.connect(s3.l23, s2.l5, ConnectionRole.APICAL, thalamic_gate=gate())
    # Motor hierarchy (bottom-up monitoring)
    circuit.connect(m1.l23, m2.l23, ConnectionRole.APICAL, thalamic_gate=gate())
    circuit.connect(m1.l23, m2.l5, ConnectionRole.APICAL, thalamic_gate=gate())
    circuit.connect(m2.l23, pfc.l23, ConnectionRole.APICAL, thalamic_gate=gate())
    circuit.connect(m2.l23, pfc.l5, ConnectionRole.APICAL, thalamic_gate=gate())
    # Cross-hierarchy
    circuit.connect(
        s1.l23,
        m1.l23,
        ConnectionRole.APICAL,
        thalamic_gate=gate(),
        surprise_tracker=SurpriseTracker(),
    )
    circuit.connect(s1.l23, m1.l5, ConnectionRole.APICAL, thalamic_gate=gate())
    circuit.connect(m1.l23, s1.l23, ConnectionRole.APICAL, thalamic_gate=gate())
    circuit.connect(m1.l23, s1.l5, ConnectionRole.APICAL, thalamic_gate=gate())

    if finalize:
        circuit.finalize()

    return circuit
