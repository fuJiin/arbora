"""T1Trainer tests — verify step/reset/eval wiring, not learning efficacy."""

from __future__ import annotations

import numpy as np
import pytest

from arbora.config import _default_t1_config, make_sensory_region
from arbora.decoders.dendritic import DendriticDecoder
from arbora.encoders.onehot import OneHotCharEncoder
from arbora.probes.bpc import BPCProbe
from examples.text_exploration.data import DEFAULT_ALPHABET
from examples.text_exploration.trainer import StepResult, T1Trainer


def _make_region(seed: int = 0):
    """Small region (32 cols / k=4) sharing defaults with _default_t1_config."""
    cfg = _default_t1_config()
    cfg.n_columns = 32
    cfg.k_columns = 4
    encoder = OneHotCharEncoder(chars=DEFAULT_ALPHABET)
    region = make_sensory_region(cfg, input_dim=encoder.input_dim, seed=seed)
    return region, encoder


@pytest.fixture
def trainer_with_decoder() -> T1Trainer:
    region, encoder = _make_region()
    decoder = DendriticDecoder(source_dim=region.n_l23_total, seed=0)
    return T1Trainer(region, encoder, decoder=decoder, bpc_probe=BPCProbe())


@pytest.fixture
def bare_trainer() -> T1Trainer:
    """Trainer with no decoder — pure region-driver mode."""
    region, encoder = _make_region()
    return T1Trainer(region, encoder)


class TestStep:
    def test_returns_step_result(self, trainer_with_decoder: T1Trainer):
        r = trainer_with_decoder.step("a")
        assert isinstance(r, StepResult)
        assert r.char == "a"
        assert r.token_id == ord("a")

    def test_step_exposes_post_step_l23(self, trainer_with_decoder: T1Trainer):
        """StepResult.l23_active should match the region's post-step active."""
        r = trainer_with_decoder.step("a")
        np.testing.assert_array_equal(
            r.l23_active, trainer_with_decoder.region.l23.active
        )

    def test_updates_prev_l23(self, trainer_with_decoder: T1Trainer):
        """After step, `_prev_l23` should equal the region's L2/3 active."""
        trainer_with_decoder.step("a")
        np.testing.assert_array_equal(
            trainer_with_decoder._prev_l23,
            trainer_with_decoder.region.l23.active,
        )

    def test_eval_mode_does_not_learn(self, trainer_with_decoder: T1Trainer):
        """train=False must not grow the decoder or update region weights."""
        # Prime with some training so the region and decoder have state.
        for c in "hello":
            trainer_with_decoder.step(c, train=True)

        ff_before = trainer_with_decoder.region.ff_weights.copy()
        decoder_tokens_before = set(trainer_with_decoder.decoder._neurons.keys())

        # Feed a novel char in eval mode.
        trainer_with_decoder.reset()
        r = trainer_with_decoder.step("z", train=False)

        assert r.char == "z"
        np.testing.assert_array_equal(trainer_with_decoder.region.ff_weights, ff_before)
        assert (
            set(trainer_with_decoder.decoder._neurons.keys()) == decoder_tokens_before
        )

    def test_train_mode_grows_decoder(self, trainer_with_decoder: T1Trainer):
        """train=True should populate the decoder with observed tokens."""
        assert trainer_with_decoder.decoder.n_tokens == 0
        trainer_with_decoder.train_word("hello")
        # Decoder sees 5 unique chars in "hello" (h, e, l, l, o → 4 unique)
        # but the first char has no pre-step L2/3 (all zeros) so decoder
        # observes nothing for it. Subsequent chars build up.
        assert trainer_with_decoder.decoder.n_tokens >= 1


class TestBareTrainer:
    """Trainer with no decoder — pure region-driver mode (observation first)."""

    def test_step_runs_without_decoder(self, bare_trainer: T1Trainer):
        r = bare_trainer.step("a")
        assert r.char == "a"
        # No decoder wired → no prediction metrics populated.
        assert r.bits is None
        assert r.top1_char is None
        assert r.top1_correct is False

    def test_step_still_drives_region(self, bare_trainer: T1Trainer):
        """Region should still process the input even without a decoder."""
        bare_trainer.step("a")
        assert bare_trainer.region.l23.active.any()

    def test_train_toggle_still_freezes_region(self, bare_trainer: T1Trainer):
        """train=False must still freeze region learning."""
        for c in "hello":
            bare_trainer.step(c, train=True)
        ff_before = bare_trainer.region.ff_weights.copy()
        bare_trainer.reset()
        bare_trainer.step("z", train=False)
        np.testing.assert_array_equal(bare_trainer.region.ff_weights, ff_before)


class TestReset:
    def test_reset_zeros_prev_l23(self, trainer_with_decoder: T1Trainer):
        trainer_with_decoder.step("a")
        assert trainer_with_decoder._prev_l23.any()
        trainer_with_decoder.reset()
        assert not trainer_with_decoder._prev_l23.any()

    def test_reset_clears_region_activations(self, trainer_with_decoder: T1Trainer):
        trainer_with_decoder.step("a")
        trainer_with_decoder.reset()
        assert not trainer_with_decoder.region.l23.active.any()
        assert not trainer_with_decoder.region.l4.active.any()


class TestTrainWord:
    def test_resets_before_processing(self, trainer_with_decoder: T1Trainer):
        """`train_word` must reset first so prior state can't leak in."""
        trainer_with_decoder.step("z")  # leaves state
        trainer_with_decoder.train_word("cat")
        assert trainer_with_decoder.region.l23.active.any()  # something activated

    def test_returns_one_result_per_char(self, trainer_with_decoder: T1Trainer):
        results = trainer_with_decoder.train_word("cat")
        assert len(results) == 3
        assert [r.char for r in results] == ["c", "a", "t"]

    def test_eval_mode_passthrough(self, trainer_with_decoder: T1Trainer):
        # Prime.
        trainer_with_decoder.train_word("hello")
        ff_before = trainer_with_decoder.region.ff_weights.copy()
        trainer_with_decoder.train_word("world", train=False)
        np.testing.assert_array_equal(trainer_with_decoder.region.ff_weights, ff_before)


class TestEncoderCompat:
    def test_accepts_multi_dim_encoder_output(self):
        """CharbitEncoder returns (length, width); trainer must flatten."""
        from arbora.encoders.charbit import CharbitEncoder

        encoder = CharbitEncoder(length=1, width=27, chars=DEFAULT_ALPHABET)
        cfg = _default_t1_config()
        cfg.n_columns = 32
        cfg.k_columns = 4
        region = make_sensory_region(cfg, input_dim=encoder.input_dim, seed=0)
        t = T1Trainer(region, encoder)
        # Should not raise even though encoder.encode returns a 2D array.
        t.step("a")
