"""ARC-AGI-3 environment utilities."""

from __future__ import annotations

import arc_agi


def list_games() -> list[arc_agi.EnvironmentInfo]:
    """List all available ARC-AGI-3 public demo environments."""
    arcade = arc_agi.Arcade()
    return arcade.get_environments()


def keyboard_only_games(arcade: arc_agi.Arcade | None = None) -> list[str]:
    """Return game IDs that only use keyboard actions (1-5), no clicks."""
    if arcade is None:
        arcade = arc_agi.Arcade()
    game_ids = []
    for e in arcade.get_environments():
        gid = e.game_id.split("-")[0]
        env = arcade.make(gid)
        f = env.reset()
        if all(a <= 5 for a in f.available_actions):
            game_ids.append(gid)
    return game_ids
