"""Typed snapshot dataclasses for probe output.

Replaces untyped dicts with structured types for autocompletion,
type checking, and self-documenting probe output.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Lamina snapshots (LaminaProbe / ChatLaminaProbe)
# ---------------------------------------------------------------------------


@dataclass
class L4Snapshot:
    """L4 lamina KPIs: prediction quality and sparseness."""

    recall: float = 0.0
    precision: float = 0.0
    sparseness: float = 0.0


@dataclass
class L23Snapshot:
    """L2/3 lamina KPIs: representation quality."""

    eff_dim: float = 0.0
    # Chat-specific (set by ChatLaminaProbe)
    linear_probe: float = 0.0
    ctx_disc: float = 0.0


@dataclass
class LaminaRegionSnapshot:
    """Per-region lamina snapshot."""

    l4: L4Snapshot = field(default_factory=L4Snapshot)
    l23: L23Snapshot = field(default_factory=L23Snapshot)


# ---------------------------------------------------------------------------
# Motor snapshots (ChatMotorProbe)
# ---------------------------------------------------------------------------


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
