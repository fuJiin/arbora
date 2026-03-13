"""Positional character encoder for character-level tokenization.

Encodes a character as a (max_positions, vocab_size) boolean matrix where the
character identity is a one-hot column and the row encodes its position within
the current word.  Word boundaries (space, punctuation) reset the position
counter.

This gives the cortex two axes of information per timestep:
- WHAT character (column)
- WHERE in the word (row)

The flattened input is max_positions * vocab_size dimensions with exactly
one active bit, providing structured sparsity without dead padding.
"""

import numpy as np

# Characters that reset the word-position counter
_BOUNDARY_CHARS = frozenset(" .!?'-")


class PositionalCharEncoder:
    """Encodes a character with its position within the current word.

    Parameters
    ----------
    chars : str
        Character alphabet (unique). Unknown chars map to last column.
    max_positions : int
        Number of position rows. Characters beyond this position are
        clamped to the last row. Default 8 covers 98.8% of BabyLM words.
    """

    def __init__(self, chars: str, *, max_positions: int = 8) -> None:
        if len(chars) != len(set(chars)):
            raise ValueError("chars must not contain duplicates")
        self._char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self._idx_to_char = {i: ch for ch, i in self._char_to_idx.items()}
        self.vocab_size = len(chars) + 1  # +1 for unknown
        self._unknown_idx = len(chars)
        self.max_positions = max_positions
        self._position = 0

    @property
    def input_dim(self) -> int:
        """Total flattened encoding dimension."""
        return self.max_positions * self.vocab_size

    @property
    def encoding_width(self) -> int:
        """Width of each position row (for ff_mask tiling)."""
        return self.vocab_size

    def encode(self, char: str) -> np.ndarray:
        """Encode a character with its current word position.

        Returns a (max_positions, vocab_size) boolean matrix with one
        active bit at (position, char_index).
        """
        out = np.zeros((self.max_positions, self.vocab_size), dtype=np.bool_)
        idx = self._char_to_idx.get(char, self._unknown_idx)
        row = min(self._position, self.max_positions - 1)
        out[row, idx] = True

        # Advance or reset position
        if char in _BOUNDARY_CHARS:
            self._position = 0
        else:
            self._position += 1

        return out

    def reset(self) -> None:
        """Reset position counter (call at story boundaries)."""
        self._position = 0

    def decode_idx(self, idx: int) -> str:
        """Decode a character index back to a character."""
        return self._idx_to_char.get(idx, "?")
