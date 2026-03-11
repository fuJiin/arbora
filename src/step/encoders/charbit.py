import numpy as np


class CharbitEncoder:
    """Character-level binary encoder.

    Encodes a token string into a (length, width) boolean matrix where each
    row is a one-hot vector over the character alphabet. Characters not in
    the alphabet map to the last column (unknown). Tokens shorter than
    `length` are zero-padded; tokens longer are truncated.

    Parameters
    ----------
    length : int
        Maximum number of character positions to encode.
    width : int
        Total columns in the output matrix. Must be >= len(chars) + 1
        (the extra column is reserved for unknown characters).
    chars : str
        The character alphabet. Each character must be unique.
    """

    def __init__(self, length: int, width: int, chars: str) -> None:
        if len(chars) != len(set(chars)):
            raise ValueError("chars must not contain duplicate characters")
        if width < len(chars) + 1:
            raise ValueError(
                f"width ({width}) must be >= len(chars) + 1 ({len(chars) + 1})"
            )

        self.length = length
        self.width = width
        self._unknown_col = width - 1
        # Map each character to its column index
        self._char_to_col = {ch: i for i, ch in enumerate(chars)}

    def encode(self, token: str) -> np.ndarray:
        """Encode a token string into a (length, width) boolean matrix."""
        out = np.zeros((self.length, self.width), dtype=np.bool_)
        for i, ch in enumerate(token[: self.length]):
            col = self._char_to_col.get(ch, self._unknown_col)
            out[i, col] = True
        return out
