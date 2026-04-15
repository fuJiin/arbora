"""ARC-AGI-3 environment utilities."""

from __future__ import annotations

import arc_agi


def list_games() -> list[arc_agi.EnvironmentInfo]:
    """List all available ARC-AGI-3 public demo environments."""
    arcade = arc_agi.Arcade()
    return arcade.get_environments()


def keyboard_only_games() -> list[arc_agi.EnvironmentInfo]:
    """Return games that only use keyboard actions (1-5), no clicks."""
    games = list_games()
    result = []
    arcade = arc_agi.Arcade()
    for g in games:
        env = arcade.make(g.game_id.split("-")[0])
        f = env.reset()
        if all(a <= 5 for a in f.available_actions):
            result.append(g)
    return result
