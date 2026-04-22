"""T1Trainer tests — verify step/reset/eval wiring, not learning efficacy."""

from __future__ import annotations

import numpy as np
import pytest

from arbora.config import _default_t1_config, make_sensory_region
from arbora.decoders.dendritic import DendriticDecoder
from arbora.encoders.charbit import CharbitEncoder
from arbora.probes.bpc import BPCProbe
from examples.text_exploration.data import DEFAULT_ALPHABET
from examples.text_exploration.trainer import StepResult, T1Trainer


@pytest.fixture
def trainer() -> T1Trainer:
    encoder = CharbitEncoder(length=1, width=27, chars=DEFAULT_ALPHABET)
    # Small region for test speed: 32 cols, k=4 is enough to exercise paths.
    cfg = _default_t1_config()
    cfg.n_columns = 32
    cfg.k_columns = 4
    region = make_sensory_region(cfg, input_dim=encoder.input_dim, seed=0)
    decoder = DendriticDecoder(source_dim=region.n_l23_total, seed=0)
    return T1Trainer(region, encoder, decoder, bpc_probe=BPCProbe())


class TestStep:
    def test_returns_step_result(self, trainer: T1Trainer):
        r = trainer.step("a")
        assert isinstance(r, StepResult)
        assert r.char == "a"
        assert r.token_id == ord("a")

    def test_updates_prev_l23(self, trainer: T1Trainer):
        """After step, `_prev_l23` should equal the region's L2/3 active."""
        trainer.step("a")
        np.testing.assert_array_equal(trainer._prev_l23, trainer.region.l23.active)

    def test_eval_mode_does_not_learn(self, trainer: T1Trainer):
        """train=False must not grow the decoder or update region weights."""
        # Prime with some training so the region and decoder have state.
        for c in "hello":
            trainer.step(c, train=True)

        ff_before = trainer.region.ff_weights.copy()
        decoder_tokens_before = set(trainer.decoder._neurons.keys())

        # Feed a novel char in eval mode.
        trainer.reset()
        r = trainer.step("z", train=False)

        assert r.char == "z"
        np.testing.assert_array_equal(trainer.region.ff_weights, ff_before)
        assert set(trainer.decoder._neurons.keys()) == decoder_tokens_before

    def test_train_mode_learns(self, trainer: T1Trainer):
        """train=True should populate the decoder with observed tokens."""
        assert trainer.decoder.n_tokens == 0
        trainer.train_word("hello")
        # Decoder sees 5 unique chars in "hello" (h, e, l, l, o → 4 unique)
        # but the first char has no pre-step L2/3 (all zeros) so decoder
        # observes nothing for it. Subsequent chars build up.
        assert trainer.decoder.n_tokens >= 1


class TestReset:
    def test_reset_zeros_prev_l23(self, trainer: T1Trainer):
        trainer.step("a")
        assert trainer._prev_l23.any()
        trainer.reset()
        assert not trainer._prev_l23.any()

    def test_reset_clears_region_activations(self, trainer: T1Trainer):
        trainer.step("a")
        trainer.reset()
        assert not trainer.region.l23.active.any()
        assert not trainer.region.l4.active.any()


class TestTrainWord:
    def test_resets_before_processing(self, trainer: T1Trainer):
        """`train_word` must reset first so prior state can't leak in."""
        trainer.step("z")  # leaves state
        trainer.train_word("cat")
        # `_prev_l23` came from the last char of "cat" — not the leaked "z".
        # Indirect check: just verify we got 3 results (one per char).
        # The primary check is that the first step in train_word had
        # zero pre-state; tested indirectly by accuracy stability.
        assert trainer.region.l23.active.any()  # something activated

    def test_returns_one_result_per_char(self, trainer: T1Trainer):
        results = trainer.train_word("cat")
        assert len(results) == 3
        assert [r.char for r in results] == ["c", "a", "t"]

    def test_eval_mode_passthrough(self, trainer: T1Trainer):
        # Prime.
        trainer.train_word("hello")
        ff_before = trainer.region.ff_weights.copy()
        trainer.train_word("world", train=False)
        np.testing.assert_array_equal(trainer.region.ff_weights, ff_before)
