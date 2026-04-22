"""Core snapshot types for probe output (input-agnostic).

Functional snapshots: InputSnapshot, AssociationSnapshot, LaminaRegionSnapshot.
KPIs are function-specific, not lamina-specific:
- Input reception (recall, precision, sparsity): measured on L4 or L2/3
- Association (eff_dim): measured on L2/3
- Output (confidence, decodability): measured on L5 or L2/3 (future)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class InputSnapshot:
    """Input reception KPIs (measured on the input lamina: L4 or L2/3)."""

    recall: float = 0.0
    precision: float = 0.0
    sparseness: float = 0.0


@dataclass
class AssociationSnapshot:
    """Association KPIs (measured on L2/3).

    `recall`, `precision`, `sparseness` mirror the `InputSnapshot`
    fields but on L2/3 — useful for regions where L2/3 carries its
    own prediction (e.g. T1 next-char next-L2/3 learning). Default 0
    so callers that only care about `eff_dim` keep working.
    """

    eff_dim: float = 0.0
    recall: float = 0.0
    precision: float = 0.0
    sparseness: float = 0.0


@dataclass
class LaminaRegionSnapshot:
    """Per-region functional snapshot."""

    input: InputSnapshot = field(default_factory=InputSnapshot)
    association: AssociationSnapshot = field(default_factory=AssociationSnapshot)
