"""ARC-AGI-3 environment utilities."""

from __future__ import annotations

import arc_agi


def list_games() -> list[arc_agi.EnvironmentInfo]:
    """List all available ARC-AGI-3 public demo environments."""
    arcade = arc_agi.Arcade()
    return arcade.get_environments()


