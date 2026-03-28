"""Typed snapshot dataclasses for probe output.

Core snapshots (input-agnostic): L4Snapshot, L23Snapshot, LaminaRegionSnapshot.
Chat-specific snapshots: ChatL23Snapshot, ChatLaminaRegionSnapshot, MotorRegionSnapshot.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Core snapshots (LaminaProbe — input-agnostic)
# ---------------------------------------------------------------------------


@dataclass
class L4Snapshot:
    """L4 lamina KPIs: prediction quality and sparseness."""

    recall: float = 0.0
    precision: float = 0.0
    sparseness: float = 0.0


@dataclass
class L23Snapshot:
    """L2/3 lamina KPIs: representation quality (input-agnostic)."""

    eff_dim: float = 0.0


@dataclass
class LaminaRegionSnapshot:
    """Per-region lamina snapshot (input-agnostic)."""

    l4: L4Snapshot = field(default_factory=L4Snapshot)
    l23: L23Snapshot = field(default_factory=L23Snapshot)


# ---------------------------------------------------------------------------
# Chat-specific snapshots (ChatLaminaProbe, ChatMotorProbe)
# ---------------------------------------------------------------------------


@dataclass
class ChatL23Snapshot(L23Snapshot):
    """L2/3 KPIs with chat-specific metrics."""

    linear_probe: float = 0.0
    ctx_disc: float = 0.0


@dataclass
class ChatLaminaRegionSnapshot:
    """Per-region lamina snapshot with chat-specific L2/3."""

    l4: L4Snapshot = field(default_factory=L4Snapshot)
    l23: ChatL23Snapshot = field(default_factory=ChatL23Snapshot)


@dataclass
class MotorRegionSnapshot:
    """Per-region motor metrics (chat-specific)."""

    motor_accuracies: list[float] = field(default_factory=list)
    motor_confidences: list[float] = field(default_factory=list)
    motor_rewards: list[float] = field(default_factory=list)
    bg_gate_values: list[float] = field(default_factory=list)
    turn_eom_steps: int = 0
    turn_input_steps: int = 0
    turn_correct_speak: int = 0
    turn_correct_silent: int = 0
    turn_interruptions: int = 0
    turn_unresponsive: int = 0
    turn_rambles: int = 0
