"""One-hot character encoder for character-level tokenization.

Produces a compact one-hot vector (vocab_size,) for a single character.
No positional padding — each character is one input vector, and temporal
structure comes from the sequence of steps.

This is the natural pairing with character-level tokenization: the input
to S1 L4 is exactly the vocabulary size, with one bit active per step.
"""

import numpy as np


class OneHotCharEncoder:
    """Encodes a single character as a one-hot vector.

    Parameters
    ----------
    chars : str
        The character alphabet. Each must be unique.
        Unknown characters map to the last position.
    """

    def __init__(self, chars: str) -> None:
        if len(chars) != len(set(chars)):
            raise ValueError("chars must not contain duplicates")
        self._char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self._idx_to_char = {i: ch for ch, i in self._char_to_idx.items()}
        self.vocab_size = len(chars) + 1  # +1 for unknown
        self._unknown_idx = len(chars)

    @property
    def input_dim(self) -> int:
        """Dimension of the encoding vector."""
        return self.vocab_size

    @property
    def encoding_width(self) -> int:
        """No positional structure in one-hot encoding."""
        return 0

    def encode(self, char: str) -> np.ndarray:
        """Encode a single character as a one-hot vector."""
        out = np.zeros(self.vocab_size, dtype=np.bool_)
        idx = self._char_to_idx.get(char, self._unknown_idx)
        out[idx] = True
        return out

    def decode_idx(self, idx: int) -> str:
        """Decode an index back to a character."""
        return self._idx_to_char.get(idx, "?")
