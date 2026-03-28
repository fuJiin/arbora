"""Core snapshot types for probe output (input-agnostic).

L4Snapshot, L23Snapshot, LaminaRegionSnapshot — used by LaminaProbe.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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


@dataclass
class LaminaRegionSnapshot:
    """Per-region lamina snapshot."""

    l4: L4Snapshot = field(default_factory=L4Snapshot)
    l23: L23Snapshot = field(default_factory=L23Snapshot)
