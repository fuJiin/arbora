import numpy as np

from step.config import EncoderConfig


def encode_token(token_id: int, config: EncoderConfig) -> frozenset[int]:
    rng = np.random.default_rng(token_id)
    indices = rng.choice(config.n, config.k, replace=False)
    return frozenset(int(i) for i in indices)


class AdaptiveEncoder:
    """Context-seeded SDR encoder.

    First encounter of a token: seed context_fraction of bits from
    recently active bits (weighted by frequency), rest random.
    Subsequent encounters: return cached SDR.
    No context available: falls back to hash-based encoding.
    """

    def __init__(self, config: EncoderConfig, seed: int = 42):
        self.config = config
        self.context_fraction = config.context_fraction
        self._token_sdrs: dict[int, frozenset[int]] = {}
        self._rng = np.random.default_rng(seed)

    def encode(
        self, token_id: int, active_bits: list[int] | None = None
    ) -> frozenset[int]:
        if token_id in self._token_sdrs:
            return self._token_sdrs[token_id]

        k = self.config.k
        n = self.config.n
        chosen: set[int] = set()

        # Context-seeded portion: sample from active bits weighted by frequency
        if active_bits and self.context_fraction > 0:
            k_context = int(k * self.context_fraction)
            bit_counts: dict[int, int] = {}
            for b in active_bits:
                bit_counts[b] = bit_counts.get(b, 0) + 1

            unique_bits = list(bit_counts.keys())
            weights = np.array([bit_counts[b] for b in unique_bits], dtype=np.float64)
            weights /= weights.sum()

            n_sample = min(k_context, len(unique_bits))
            sampled = self._rng.choice(
                np.array(unique_bits), n_sample, replace=False, p=weights
            )
            chosen.update(int(b) for b in sampled)

        # Random portion: deterministic per token_id, fill remaining bits
        remaining = k - len(chosen)
        if remaining > 0:
            available = np.array([i for i in range(n) if i not in chosen])
            token_rng = np.random.default_rng(token_id)
            random_bits = token_rng.choice(available, remaining, replace=False)
            chosen.update(int(b) for b in random_bits)

        sdr = frozenset(chosen)
        self._token_sdrs[token_id] = sdr
        return sdr

    @property
    def known_tokens(self) -> int:
        return len(self._token_sdrs)
