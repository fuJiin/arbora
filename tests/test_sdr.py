from step.config import EncoderConfig
from step.sdr import encode_token


class TestEncodeToken:
    def setup_method(self):
        self.config = EncoderConfig(n=2048, k=40)

    def test_determinism(self):
        a = encode_token(42, self.config)
        b = encode_token(42, self.config)
        assert a == b

    def test_different_tokens_differ(self):
        a = encode_token(42, self.config)
        b = encode_token(43, self.config)
        assert a != b

    def test_size(self):
        sdr = encode_token(100, self.config)
        assert len(sdr) == 40

    def test_range(self):
        sdr = encode_token(100, self.config)
        assert all(0 <= idx < 2048 for idx in sdr)

    def test_returns_frozenset(self):
        sdr = encode_token(100, self.config)
        assert isinstance(sdr, frozenset)

    def test_custom_config(self):
        config = EncoderConfig(n=128, k=5)
        sdr = encode_token(0, config)
        assert len(sdr) == 5
        assert all(0 <= idx < 128 for idx in sdr)

    def test_matches_snapshot_logic(self):
        """Verify new encode_token produces same indices as dump.py's get_sdr."""
        import numpy as np

        token_id = 42
        rng = np.random.default_rng(token_id)
        expected = frozenset(int(i) for i in rng.choice(2048, 40, replace=False))
        actual = encode_token(token_id, self.config)
        assert actual == expected
