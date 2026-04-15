"""Grid encoder for ARC-AGI-3 environments.

Encodes a 64x64 grid frame (16 colors) from the arc-agi SDK into a
sparse binary vector suitable for SensoryRegion input.

Two encoding channels per cell, interleaved at stride 18:
1. **Spatial** (bits 0-15): one-hot color from 2x2 block mode-pool → 32x32.
2. **Temporal** (bits 16-17): change/unchanged vs previous frame.
   Like retinal ganglion ON/OFF cells.

Layout: 1024 cells * 18 bits/cell = 18432 dimensions.
encoding_width = 18 (the per-cell stride for receptive field tiling).
"""

from __future__ import annotations

import numpy as np

_GRID_SIZE = 64
_BLOCK_SIZE = 2
_DOWNSAMPLED = _GRID_SIZE // _BLOCK_SIZE  # 32
_N_COLORS = 16
_N_CHANGE = 2  # changed / unchanged
_N_CELLS = _DOWNSAMPLED * _DOWNSAMPLED  # 1024
_SPATIAL_DIM = _N_CELLS * _N_COLORS  # 16384
_TEMPORAL_DIM = _N_CELLS * _N_CHANGE  # 2048
_TOTAL_DIM = _SPATIAL_DIM + _TEMPORAL_DIM  # 18432
_ENCODING_WIDTH = _N_COLORS + _N_CHANGE  # 18


class ArcGridEncoder:
    """Encode a 64x64 ARC-AGI-3 grid frame with spatial + temporal channels.

    Spatial: 2x2 block mode-pool → 32x32, one-hot color (16384 bits).
    Temporal: per-cell change vs previous frame (2048 bits).

    The temporal channel acts like retinal ON/OFF cells — it highlights
    what changed between frames, giving V1 a change signal even when
    the static spatial content is fully predicted.
    """

    def __init__(self) -> None:
        self._prev_down: np.ndarray | None = None

    @property
    def input_dim(self) -> int:
        return _TOTAL_DIM

    @property
    def encoding_width(self) -> int:
        return _ENCODING_WIDTH

    def encode(self, grid: np.ndarray) -> np.ndarray:
        """Encode a 64x64 grid frame.

        Returns a boolean vector of shape (18432,).
        Active bits: 1024 spatial + ~50-200 temporal change bits.
        """
        downsampled = _block_mode_pool(grid, _BLOCK_SIZE)
        out = np.zeros(_TOTAL_DIM, dtype=np.bool_)

        flat = downsampled.ravel()

        # Spatial channel: one-hot color per cell
        for i, color in enumerate(flat):
            c = int(color)
            if 0 <= c < _N_COLORS:
                out[i * _ENCODING_WIDTH + c] = True

        # Temporal channel: change detection vs previous frame
        if self._prev_down is not None:
            prev_flat = self._prev_down.ravel()
            for i in range(len(flat)):
                changed = flat[i] != prev_flat[i]
                # Bit 0 of change channel = changed, bit 1 = unchanged
                out[i * _ENCODING_WIDTH + _N_COLORS + (0 if changed else 1)] = True
        else:
            # First frame: mark all as "changed" (novel)
            for i in range(len(flat)):
                out[i * _ENCODING_WIDTH + _N_COLORS + 0] = True

        self._prev_down = downsampled.copy()
        return out

    def reset(self) -> None:
        """Reset temporal state (new episode)."""
        self._prev_down = None


def _block_mode_pool(grid: np.ndarray, block_size: int = _BLOCK_SIZE) -> np.ndarray:
    """Downsample grid by taking the mode (most common value) of each block."""
    h, w = grid.shape
    bh = h // block_size
    bw = w // block_size
    out = np.zeros((bh, bw), dtype=np.int8)
    for r in range(bh):
        for c in range(bw):
            block = grid[
                r * block_size : (r + 1) * block_size,
                c * block_size : (c + 1) * block_size,
            ]
            counts = np.bincount(block.ravel(), minlength=_N_COLORS)
            out[r, c] = np.argmax(counts)
    return out
