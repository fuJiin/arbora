"""Chat-specific snapshot types for probe output.

ChatL23Snapshot, ChatLaminaRegionSnapshot — used by ChatLaminaProbe.
MotorRegionSnapshot — used by ChatMotorProbe.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from step.snapshots.core import L4Snapshot, L23Snapshot


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
