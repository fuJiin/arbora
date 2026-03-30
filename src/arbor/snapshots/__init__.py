"""Typed snapshot dataclasses for probe output.

Core (input-agnostic): InputSnapshot, AssociationSnapshot, LaminaRegionSnapshot.
"""

from arbor.snapshots.core import (
    AssociationSnapshot,
    InputSnapshot,
    LaminaRegionSnapshot,
)

__all__ = [
    "AssociationSnapshot",
    "InputSnapshot",
    "LaminaRegionSnapshot",
]
