"""Chat-specific snapshot types for probe output.

ChatAssociationSnapshot, ChatLaminaRegionSnapshot — used by ChatLaminaProbe.
MotorRegionSnapshot — used by ChatMotorProbe.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from arbora.snapshots.core import AssociationSnapshot, InputSnapshot


@dataclass
class ChatAssociationSnapshot(AssociationSnapshot):
    """Association KPIs with chat-specific metrics."""

    linear_probe: float = 0.0
    ctx_disc: float = 0.0


@dataclass
class ChatLaminaRegionSnapshot:
    """Per-region functional snapshot with chat-specific association."""

    input: InputSnapshot = field(default_factory=InputSnapshot)
    association: ChatAssociationSnapshot = field(
        default_factory=ChatAssociationSnapshot
    )


@dataclass
class MotorRegionSnapshot:
    """Per-region motor metrics."""

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
