"""Data types for the circuit system.

Extracted from circuit.py to separate type definitions from the
builder and runner logic. All types are re-exported from circuit.py
for backward compatibility.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, TypeVar

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
