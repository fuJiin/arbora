"""Tests for the BPCProbe bits-per-character metric."""

import math
from unittest.mock import MagicMock

import numpy as np
import pytest

import step.cortex.topology  # noqa: F401  # resolve circular import chain
from step.probes.bpc import BPCProbe


def _make_decoder(n_tokens: int = 10, scores: dict | None = None):
    """Create a mock DendriticDecoder with controllable scores."""
    decoder = MagicMock()
    decoder.n_tokens = n_tokens
    decoder.decode_scores.return_value = scores if scores is not None else {}
    return decoder


class TestBPCProbe:
    def test_starts_at_inf(self):
        probe = BPCProbe()
        assert probe.bpc == float("inf")
        assert probe.recent_bpc == float("inf")

    def test_finite_after_steps(self):
        probe = BPCProbe()
        state = np.ones(64, dtype=np.float32)
        decoder = _make_decoder(n_tokens=10, scores={5: 8, 3: 2, 7: 1})

        for _ in range(5):
            probe.step(token_id=5, l23_state=state, decoder=decoder)

        assert np.isfinite(probe.bpc)
        assert probe.bpc > 0

    def test_reset_clears_state(self):
        probe = BPCProbe()
        state = np.ones(64, dtype=np.float32)
        decoder = _make_decoder(n_tokens=10, scores={5: 8})

        probe.step(token_id=5, l23_state=state, decoder=decoder)
        assert probe.bpc != float("inf")

        probe.reset()
        assert probe.bpc == float("inf")
        assert probe.recent_bpc == float("inf")

    def test_recent_bpc_tracks_rolling_window(self):
        probe = BPCProbe()
        probe._window_size = 3
        state = np.ones(64, dtype=np.float32)

        high_decoder = _make_decoder(n_tokens=10, scores={1: 10})
        for _ in range(3):
            probe.step(token_id=1, l23_state=state, decoder=high_decoder)
        low_recent = probe.recent_bpc

        low_decoder = _make_decoder(n_tokens=10, scores={2: 1, 1: 1})
        for _ in range(3):
            probe.step(token_id=2, l23_state=state, decoder=low_decoder)
        high_recent = probe.recent_bpc

        assert high_recent > low_recent

    def test_step_returns_bits(self):
        probe = BPCProbe()
        state = np.ones(64, dtype=np.float32)
        decoder = _make_decoder(n_tokens=10, scores={5: 8, 3: 2})

        bits = probe.step(token_id=5, l23_state=state, decoder=decoder)
        assert np.isfinite(bits)
        assert bits >= 0

    def test_unknown_token_still_finite(self):
        probe = BPCProbe()
        state = np.ones(64, dtype=np.float32)
        decoder = _make_decoder(n_tokens=10, scores={5: 8})

        bits = probe.step(token_id=99, l23_state=state, decoder=decoder)
        assert np.isfinite(bits)
        assert bits > 0

    def test_empty_scores_uniform(self):
        probe = BPCProbe()
        state = np.ones(64, dtype=np.float32)
        decoder = _make_decoder(n_tokens=10, scores={})

        bits = probe.step(token_id=5, l23_state=state, decoder=decoder)
        assert bits == pytest.approx(math.log2(10))

    def test_zero_n_tokens_skips(self):
        probe = BPCProbe()
        state = np.ones(64, dtype=np.float32)
        decoder = _make_decoder(n_tokens=0, scores={})

        bits = probe.step(token_id=5, l23_state=state, decoder=decoder)
        assert bits == 0.0
        assert probe._n_chars == 0
        assert probe.bpc == float("inf")
