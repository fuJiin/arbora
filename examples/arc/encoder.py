"""Grid encoder for ARC-AGI-3 environments.

Encodes a 64x64 grid frame (16 colors) from the arc-agi SDK into a
sparse binary vector suitable for SensoryRegion input.

The 64x64 grid is downsampled to 16x16 via 4x4 block mode-pooling
(most common color per block), then each cell is one-hot encoded
over 16 colors.

Layout: 16*16 cells × 16 colors = 4096 dimensions.
encoding_width = 16 (one cell's color bits), so SensoryRegion tiles
its columns across the 256 spatial positions — a natural retinotopic map.
"""

from __future__ import annotations

import numpy as np

_GRID_SIZE = 64
_BLOCK_SIZE = 4
_DOWNSAMPLED = _GRID_SIZE // _BLOCK_SIZE  # 16
_N_COLORS = 16
_N_CELLS = _DOWNSAMPLED * _DOWNSAMPLED  # 256
_TOTAL_DIM = _N_CELLS * _N_COLORS  # 4096


class ArcGridEncoder:
    """Encode a 64x64 ARC-AGI-3 grid frame as a sparse binary vector.

    Downsamples via 4x4 block mode-pooling, then one-hot encodes
    each cell's color. Produces a 4096-bit vector with 256 active bits
    (one per downsampled cell).

    Properties match the Encoder protocol for SensoryRegion integration.
    """

    @property
    def input_dim(self) -> int:
        return _TOTAL_DIM

    @property
    def encoding_width(self) -> int:
        return _N_COLORS

    def encode(self, grid: np.ndarray) -> np.ndarray:
        """Encode a 64x64 grid frame.

        Parameters
        ----------
        grid : np.ndarray
            Shape (64, 64), dtype int8, values 0-15.

        Returns
        -------
        np.ndarray
            Boolean vector of shape (4096,) with 256 active bits.
        """
        # Downsample via 4x4 block mode pooling
        downsampled = _block_mode_pool(grid, _BLOCK_SIZE)

        # One-hot encode
        out = np.zeros(_TOTAL_DIM, dtype=np.bool_)
        flat = downsampled.ravel()
        for i, color in enumerate(flat):
            c = int(color)
            if 0 <= c < _N_COLORS:
                out[i * _N_COLORS + c] = True
        return out

    def reset(self) -> None:
        """No-op (stateless encoder)."""


def _block_mode_pool(grid: np.ndarray, block_size: int) -> np.ndarray:
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
            # Mode: most common value in the block
            counts = np.bincount(block.ravel(), minlength=_N_COLORS)
            out[r, c] = np.argmax(counts)
    return out
