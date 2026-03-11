import numpy as np

from step.config import EncoderConfig


class RandomEncoder:
    """Hash-based random encoder.

    Deterministically maps a token_id to a sparse binary encoding
    using a seeded RNG. Same token_id always produces the same encoding.
    """

    def __init__(self, config: EncoderConfig) -> None:
        self.config = config

    def encode(self, token_id: int) -> frozenset[int]:
        """Encode a token_id into a sparse binary encoding."""
        rng = np.random.default_rng(token_id)
        indices = rng.choice(self.config.n, self.config.k, replace=False)
        return frozenset(int(i) for i in indices)
