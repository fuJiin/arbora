"""Data types for the circuit system.

Extracted from circuit.py to separate type definitions from the
builder and runner logic. All types are re-exported from circuit.py
for backward compatibility.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np

from step.cortex.basal_ganglia import BasalGanglia
from step.cortex.lamina import LaminaID
from step.cortex.modulators import RewardModulator, SurpriseTracker, ThalamicGate
from step.cortex.region import CorticalRegion
from step.decoders import DendriticDecoder, InvertedIndexDecoder, SynapticDecoder
from step.probes.diagnostics import CortexDiagnostics
from step.probes.representation import RepresentationTracker
from step.probes.timeline import Timeline

if TYPE_CHECKING:
    from step.decoders.word import WordDecoder


class Encoder(Protocol):
    """Minimal encoder interface for the runner."""

    def encode(self, token: str) -> np.ndarray: ...


@dataclass
class RunMetrics:
    overlaps: list[float] = field(default_factory=list)
    accuracies: list[float] = field(default_factory=list)
    synaptic_accuracies: list[float] = field(default_factory=list)
    column_accuracies: list[float] = field(default_factory=list)
    dendritic_accuracies: list[float] = field(default_factory=list)
    motor_accuracies: list[float] = field(default_factory=list)
    motor_decoder_accuracies: list[float] = field(default_factory=list)
    motor_population_accuracies: list[float] = field(default_factory=list)
    motor_confidences: list[float] = field(default_factory=list)
    motor_rewards: list[float] = field(default_factory=list)
    # Basal ganglia gate values (when BG is wired)
    bg_gate_values: list[float] = field(default_factory=list)
    # Turn-taking behavioral metrics (Stage 1 RL)
    turn_interruptions: int = 0  # Spoke during input phase
    turn_unresponsive: int = 0  # Silent during EOM phase
    turn_correct_speak: int = 0  # Spoke during EOM phase
    turn_correct_silent: int = 0  # Silent during input phase
    turn_rambles: int = 0  # Spoke past max_speak_steps
    turn_eom_steps: int = 0  # Total steps in EOM phase
    turn_input_steps: int = 0  # Total steps in input phase
    # Bits-per-character (entry region only, uses dendritic decoder)
    bpc: float = 0.0
    bpc_recent: float = 0.0
    # Per-dialogue BPC breakdown (forgetting diagnosis)
    bpc_per_dialogue: list[float] = field(default_factory=list)
    bpc_boundary: list[float] = field(default_factory=list)
    bpc_steady: list[float] = field(default_factory=list)
    # Centroid-based BPC (non-learned, for comparison)
    centroid_bpc: float = 0.0
    centroid_bpc_recent: float = 0.0
    elapsed_seconds: float = 0.0
    representation: dict = field(default_factory=dict)


@dataclass
class _RegionState:
    """Per-region bookkeeping created by add_region()."""

    region: CorticalRegion
    rep_tracker: RepresentationTracker
    diagnostics: CortexDiagnostics | None
    timeline: Timeline | None
    entry: bool = False
    motor: bool = False
    basal_ganglia: BasalGanglia | None = None
    # Entry region only:
    decode_index: InvertedIndexDecoder | None = None
    syn_decoder: SynapticDecoder | None = None
    dendritic_decoder: DendriticDecoder | None = None
    # Motor region decoder (maps M1 L2/3 → token predictions):
    motor_decoder: DendriticDecoder | None = None
    # Word-level decoder (maps L2/3 → word predictions):
    word_decoder: WordDecoder | None = None


class ConnectionRole(enum.Enum):
    """Structural role of a connection between cortical regions."""

    FEEDFORWARD = "feedforward"
    APICAL = "apical"


@dataclass
class Connection:
    source: str
    target: str
    role: ConnectionRole
    source_lamina: LaminaID = LaminaID.L23
    target_lamina: LaminaID = LaminaID.L4
    surprise_tracker: SurpriseTracker | None = None
    reward_modulator: RewardModulator | None = None
    buffer_depth: int = 1
    burst_gate: bool = False
    thalamic_gate: ThalamicGate | None = None
    enabled: bool = True
    trace_decay: float = 0.8
    _trace: np.ndarray | None = field(default=None, repr=False)
    _buffer: np.ndarray | None = field(default=None, repr=False)
    _buffer_pos: int = 0


@dataclass
class CortexResult:
    per_region: dict[str, RunMetrics] = field(default_factory=dict)
    surprise_modulators: dict[str, list[float]] = field(default_factory=dict)
    thalamic_readiness: dict[str, list[float]] = field(default_factory=dict)
    reward_modulators: dict[str, list[float]] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    # Babble metrics (populated when babble_ratio > 0)
    babble_rewards: list[float] = field(default_factory=list)
    babble_tokens_produced: list[str] = field(default_factory=list)
    babble_unique_tokens: list[str] = field(default_factory=list)
    total_listen_steps: int = 0
    total_babble_steps: int = 0
