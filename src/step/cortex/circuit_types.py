"""Data types for the circuit system.

Extracted from circuit.py to separate type definitions from the
builder and runner logic. All types are re-exported from circuit.py
for backward compatibility.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, TypeVar, runtime_checkable

import numpy as np

from step.cortex.lamina import Lamina, LaminaID
from step.cortex.modulators import RewardModulator, SurpriseTracker, ThalamicGate
from step.decoders import DendriticDecoder, InvertedIndexDecoder, SynapticDecoder
from step.probes.diagnostics import CortexDiagnostics
from step.probes.representation import RepresentationTracker
from step.probes.timeline import Timeline

if TYPE_CHECKING:
    from step.decoders.word import WordDecoder


# Contravariant: Encoder accepts observations (parameter position).
# Encoder[str] for chat, Encoder[MiniGridObs] for grid envs.
T_obs = TypeVar("T_obs", contravariant=True)


class Encoder(Protocol[T_obs]):
    """Encoder interface: observation → sparse binary vector.

    Generic over the observation type. Chat encoders implement
    Encoder[str], grid encoders implement Encoder[MiniGridObs], etc.
    """

    @property
    def input_dim(self) -> int:
        """Total flattened encoding dimension for SensoryRegion."""
        ...

    @property
    def encoding_width(self) -> int:
        """Width of a single position for receptive field tiling.

        Return 0 if the encoding has no positional structure.
        """
        ...

    def encode(self, obs: T_obs) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# Region protocol — shared interface for circuit participation
# ---------------------------------------------------------------------------


@runtime_checkable
class Region(Protocol):
    """Minimal interface for anything that participates in a Circuit.

    CorticalRegion (sensory, motor, PFC) and subcortical regions (BG)
    both satisfy this protocol. The circuit uses input_port/output_port
    for wiring, process() for computation, and apply_reward()/
    reset_working_memory() for lifecycle.
    """

    @property
    def input_port(self) -> Lamina:
        """Connectable input surface (target of feedforward/modulatory)."""
        ...

    @property
    def output_port(self) -> Lamina:
        """Connectable output surface (source of feedforward/modulatory)."""
        ...

    @property
    def input_dim(self) -> int:
        """Expected input dimension for validation."""
        ...

    def process(self, encoding: np.ndarray, **kwargs) -> np.ndarray:
        """Transform input → output. Called once per circuit step."""
        ...

    def apply_reward(self, reward: float) -> None:
        """Consolidate eligibility traces using reward signal."""
        ...

    def reset_working_memory(self) -> None:
        """Reset transient state at episode/story boundaries."""
        ...


# ---------------------------------------------------------------------------
# Per-region bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class _RegionState:
    """Per-region bookkeeping created by add_region().

    Supports both cortical and subcortical regions. Cortical-specific
    fields (rep_tracker, diagnostics, timeline, decoders) are None
    for non-cortical regions.
    """

    region: Region  # type: ignore[assignment]
    rep_tracker: RepresentationTracker | None
    diagnostics: CortexDiagnostics | None
    timeline: Timeline | None
    entry: bool = False
    motor: bool = False
    # Entry region only:
    decode_index: InvertedIndexDecoder | None = None
    syn_decoder: SynapticDecoder | None = None
    dendritic_decoder: DendriticDecoder | None = None
    # Motor region decoder (maps M1 L2/3 → token predictions):
    motor_decoder: DendriticDecoder | None = None
    # Word-level decoder (maps L2/3 → word predictions):
    word_decoder: WordDecoder | None = None


class ConnectionRole(enum.Enum):
    """Structural role of a connection between regions."""

    FEEDFORWARD = "feedforward"
    APICAL = "apical"
    MODULATORY = "modulatory"


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
