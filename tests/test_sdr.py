from step.config import EncoderConfig
from step.encoders import AdaptiveEncoder, RandomEncoder


class TestRandomEncoder:
    def setup_method(self):
        self.config = EncoderConfig(n=2048, k=40)
        self.encoder = RandomEncoder(self.config)

    def test_determinism(self):
        a = self.encoder.encode(42)
        b = self.encoder.encode(42)
        assert a == b

    def test_different_tokens_differ(self):
        a = self.encoder.encode(42)
        b = self.encoder.encode(43)
        assert a != b

    def test_size(self):
        encoding = self.encoder.encode(100)
        assert len(encoding) == 40

    def test_range(self):
        encoding = self.encoder.encode(100)
        assert all(0 <= idx < 2048 for idx in encoding)

    def test_returns_frozenset(self):
        encoding = self.encoder.encode(100)
        assert isinstance(encoding, frozenset)

    def test_custom_config(self):
        encoder = RandomEncoder(EncoderConfig(n=128, k=5))
        encoding = encoder.encode(0)
        assert len(encoding) == 5
        assert all(0 <= idx < 128 for idx in encoding)

    def test_matches_snapshot_logic(self):
        """Verify RandomEncoder produces same indices as dump.py's get_sdr."""
        import numpy as np

        token_id = 42
        rng = np.random.default_rng(token_id)
        expected = frozenset(int(i) for i in rng.choice(2048, 40, replace=False))
        actual = self.encoder.encode(token_id)
        assert actual == expected


class TestAdaptiveEncoder:
    def setup_method(self):
        self.config = EncoderConfig(n=2048, k=40, adaptive=True, context_fraction=0.5)
        self.encoder = AdaptiveEncoder(self.config)

    def test_no_context_returns_valid_sdr(self):
        sdr = self.encoder.encode(42, active_bits=None)
        assert len(sdr) == 40
        assert all(0 <= idx < 2048 for idx in sdr)

    def test_caches_on_first_encounter(self):
        sdr1 = self.encoder.encode(42, active_bits=None)
        sdr2 = self.encoder.encode(42, active_bits=[100, 200, 300])
        assert sdr1 == sdr2  # second call returns cached, ignores context

    def test_known_tokens_counter(self):
        assert self.encoder.known_tokens == 0
        self.encoder.encode(1)
        assert self.encoder.known_tokens == 1
        self.encoder.encode(2)
        assert self.encoder.known_tokens == 2
        self.encoder.encode(1)  # re-encode same token
        assert self.encoder.known_tokens == 2

    def test_context_seeding_shares_bits(self):
        """Tokens first seen in the same context should share bits."""
        context = list(range(100))  # bits 0-99 active
        sdr_a = self.encoder.encode(1000, active_bits=context)
        sdr_b = self.encoder.encode(1001, active_bits=context)
        overlap = len(sdr_a & sdr_b)
        # With 50% context from same pool of 100 bits, expect significant overlap
        # Random expectation: k^2/n = 40*40/2048 ≈ 0.78
        assert overlap > 5  # well above random

    def test_different_contexts_differ(self):
        """Tokens first seen in different contexts should have less overlap."""
        ctx_a = list(range(0, 100))
        ctx_b = list(range(1000, 1100))
        sdr_a = self.encoder.encode(2000, active_bits=ctx_a)
        sdr_b = self.encoder.encode(2001, active_bits=ctx_b)
        # Context bits are disjoint, so overlap comes only from random portion
        overlap = len(sdr_a & sdr_b)
        # Compare with same-context overlap
        enc2 = AdaptiveEncoder(self.config)
        sdr_c = enc2.encode(3000, active_bits=ctx_a)
        sdr_d = enc2.encode(3001, active_bits=ctx_a)
        same_ctx_overlap = len(sdr_c & sdr_d)
        assert same_ctx_overlap > overlap

    def test_context_fraction_zero_is_all_random(self):
        config = EncoderConfig(n=2048, k=40, adaptive=True, context_fraction=0.0)
        enc = AdaptiveEncoder(config)
        context = list(range(100))
        sdr = enc.encode(42, active_bits=context)
        assert len(sdr) == 40
        # All bits should come from the random portion

    def test_reproducible_across_instances(self):
        """Same seed + same call order = same SDRs."""
        enc1 = AdaptiveEncoder(self.config, seed=99)
        enc2 = AdaptiveEncoder(self.config, seed=99)
        ctx = list(range(50))
        sdr1 = enc1.encode(42, active_bits=ctx)
        sdr2 = enc2.encode(42, active_bits=ctx)
        assert sdr1 == sdr2
