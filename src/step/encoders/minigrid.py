"""MiniGrid observation encoder.

Converts a MiniGridObs (7x7x3 symbolic grid + direction) into a sparse
binary vector suitable for SensoryRegion input.

Encoding layout (984 bits):
- 49 cells x 20 bits each = 980 bits
  - 11 bits: object type one-hot (unseen, empty, wall, floor, door, key,
    ball, box, goal, lava, agent)
  - 6 bits: color one-hot (red, green, blue, purple, yellow, grey)
  - 3 bits: state one-hot (open, closed, locked)
- 4 bits: agent direction one-hot (right, down, left, up)

Active bits per step: 49*3 + 1 = 148 (each cell has 3 active channels,
plus 1 direction bit). Density ~15%.
"""

from __future__ import annotations

import numpy as np

from step.environment.minigrid import MiniGridObs

# MiniGrid symbolic encoding ranges
_N_OBJECT_TYPES = 11
_N_COLORS = 6
_N_STATES = 3
_BITS_PER_CELL = _N_OBJECT_TYPES + _N_COLORS + _N_STATES  # 20
_GRID_SIZE = 7
_N_CELLS = _GRID_SIZE * _GRID_SIZE  # 49
_N_DIRECTIONS = 4
_TOTAL_DIM = _N_CELLS * _BITS_PER_CELL + _N_DIRECTIONS  # 984


class MiniGridEncoder:
    """Encode MiniGrid observations as sparse binary vectors.

    Produces a 984-bit boolean vector from a 7x7x3 symbolic grid plus
    agent direction. Each cell's 3 channels are one-hot encoded and
    concatenated, followed by a one-hot direction suffix.

    Properties match the Encoder[MiniGridObs] protocol for SensoryRegion
    integration: input_dim=984, encoding_width=20 (per-cell feature width
    for receptive field tiling).
    """

    @property
    def input_dim(self) -> int:
        """Total flattened encoding dimension."""
        return _TOTAL_DIM

    @property
    def encoding_width(self) -> int:
        """Width of a single cell encoding (for receptive field tiling)."""
        return _BITS_PER_CELL

    def encode(self, obs: MiniGridObs) -> np.ndarray:
        """Encode observation to a sparse boolean vector.

        Returns a (984,) boolean array with ~148 active bits.
        """
        out = np.zeros(_TOTAL_DIM, dtype=np.bool_)
        image = obs.image  # (7, 7, 3) uint8

        for r in range(_GRID_SIZE):
            for c in range(_GRID_SIZE):
                cell_idx = r * _GRID_SIZE + c
                base = cell_idx * _BITS_PER_CELL

                obj_type = int(image[r, c, 0])
                color = int(image[r, c, 1])
                state = int(image[r, c, 2])

                # Object type one-hot (bits 0-10)
                if 0 <= obj_type < _N_OBJECT_TYPES:
                    out[base + obj_type] = True

                # Color one-hot (bits 11-16)
                if 0 <= color < _N_COLORS:
                    out[base + _N_OBJECT_TYPES + color] = True

                # State one-hot (bits 17-19)
                if 0 <= state < _N_STATES:
                    out[base + _N_OBJECT_TYPES + _N_COLORS + state] = True

        # Direction one-hot (last 4 bits)
        direction = obs.direction
        if 0 <= direction < _N_DIRECTIONS:
            out[_N_CELLS * _BITS_PER_CELL + direction] = True

        return out

    def reset(self) -> None:
        """No-op (stateless encoder)."""
