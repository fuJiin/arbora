import numpy as np

from step.config import EncoderConfig


def encode_token(token_id: int, config: EncoderConfig) -> frozenset[int]:
    rng = np.random.default_rng(token_id)
    indices = rng.choice(config.n, config.k, replace=False)
    return frozenset(int(i) for i in indices)
