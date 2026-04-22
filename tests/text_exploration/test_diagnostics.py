"""Tests for the three T1 diagnostic checkpoints (ARB-131 PR B).

Focuses on:
- Non-destructiveness (trainer state unchanged after measurement).
- Correct partitioning of within-vowel, within-consonant, across pairs.
- Pathology flags fire on synthetic extreme weights.
"""

from __future__ import annotations

import numpy as np
import pytest

from arbora.config import _default_t1_config, make_sensory_region
from arbora.decoders.dendritic import DendriticDecoder
from arbora.encoders.charbit import CharbitEncoder
from examples.text_exploration.data import DEFAULT_ALPHABET
from examples.text_exploration.diagnostics import (
    CONSONANTS,
    VOWELS,
    WeightStats,
    _jaccard,
    character_sdr_overlap,
    context_sensitivity,
    format_diagnostics,
    weight_distribution,
)
from examples.text_exploration.trainer import T1Trainer


@pytest.fixture
def primed_trainer() -> T1Trainer:
    """Trainer with a few words of training, small region for speed."""
    encoder = CharbitEncoder(length=1, width=27, chars=DEFAULT_ALPHABET)
    cfg = _default_t1_config()
    cfg.n_columns = 32
    cfg.k_columns = 4
    region = make_sensory_region(cfg, input_dim=encoder.input_dim, seed=0)
    decoder = DendriticDecoder(source_dim=region.n_l23_total, seed=0)
    trainer = T1Trainer(region, encoder, decoder)
    for w in ["cat", "dog", "sun", "run", "hat"]:
        trainer.train_word(w)
    return trainer


class TestJaccard:
    def test_identical_is_one(self):
        a = np.array([True, False, True, True])
        assert _jaccard(a, a) == 1.0

    def test_disjoint_is_zero(self):
        a = np.array([True, False, True, False])
        b = np.array([False, True, False, True])
        assert _jaccard(a, b) == 0.0

    def test_half_overlap(self):
        a = np.array([True, True, False, False])
        b = np.array([True, False, True, False])
        assert _jaccard(a, b) == pytest.approx(1 / 3)

    def test_both_empty_is_zero(self):
        a = np.zeros(4, dtype=bool)
        assert _jaccard(a, a) == 0.0


class TestCharacterSDROverlap:
    def test_returns_sdr_per_char(self, primed_trainer: T1Trainer):
        r = character_sdr_overlap(primed_trainer)
        for c in DEFAULT_ALPHABET:
            assert c in r.per_char_sdr
            assert r.per_char_sdr[c].dtype == np.bool_

    def test_within_and_across_pair_counts(self, primed_trainer: T1Trainer):
        """Partition math: 5 vowels -> C(5,2)=10 within-vowel pairs,
        21 consonants -> C(21,2)=210 within-consonant, 5*21=105 across."""
        r = character_sdr_overlap(primed_trainer)
        assert len(r.within_vowel) == 10
        assert len(r.within_consonant) == 210
        assert len(r.across) == 105

    def test_non_destructive_of_region_and_prev_l23(self, primed_trainer: T1Trainer):
        saved_prev = primed_trainer._prev_l23.copy()
        saved_ff = primed_trainer.region.ff_weights.copy()
        saved_decoder = set(primed_trainer.decoder._neurons.keys())
        character_sdr_overlap(primed_trainer)
        np.testing.assert_array_equal(primed_trainer._prev_l23, saved_prev)
        np.testing.assert_array_equal(primed_trainer.region.ff_weights, saved_ff)
        assert set(primed_trainer.decoder._neurons.keys()) == saved_decoder

    def test_vowels_constant_covers_exactly_five(self):
        assert set(VOWELS) == set("aeiou")

    def test_consonants_is_alphabet_minus_vowels(self):
        assert set(CONSONANTS) == set(DEFAULT_ALPHABET) - set(VOWELS)


class TestContextSensitivity:
    def test_one_prefix_per_entry(self, primed_trainer: T1Trainer):
        prefixes = ["ca", "de"]
        r = context_sensitivity(primed_trainer, prefixes)
        assert set(r.top_k_per_prefix.keys()) == {"ca", "de"}

    def test_top_k_length_bounded(self, primed_trainer: T1Trainer):
        r = context_sensitivity(primed_trainer, ["ca"], k=3)
        assert len(r.top_k_per_prefix["ca"]) <= 3

    def test_pairwise_count(self, primed_trainer: T1Trainer):
        """Four prefixes → C(4,2) = 6 pairs."""
        r = context_sensitivity(primed_trainer, ["a", "b", "c", "d"])
        assert len(r.pairwise_overlap) == 6

    def test_non_destructive(self, primed_trainer: T1Trainer):
        saved_ff = primed_trainer.region.ff_weights.copy()
        saved_decoder = set(primed_trainer.decoder._neurons.keys())
        context_sensitivity(primed_trainer, ["ca", "de"])
        np.testing.assert_array_equal(primed_trainer.region.ff_weights, saved_ff)
        assert set(primed_trainer.decoder._neurons.keys()) == saved_decoder


class TestWeightDistribution:
    def test_has_all_sections(self, primed_trainer: T1Trainer):
        r = weight_distribution(primed_trainer)
        assert r.ff.name == "ff_weights"
        assert r.l4_lat_perm.name == "l4_lat_seg_perm"
        assert r.l23_seg_perm.name == "l23_seg_perm"

    def test_stats_are_finite(self, primed_trainer: T1Trainer):
        r = weight_distribution(primed_trainer)
        for stats in (r.ff, r.l4_lat_perm, r.l23_seg_perm):
            assert np.isfinite(stats.mean)
            assert np.isfinite(stats.std)
            assert 0.0 <= stats.frac_near_zero <= 1.0
            assert 0.0 <= stats.frac_near_one <= 1.0


class TestPathologyFlags:
    def test_saturated_flag_fires_above_30pct(self):
        s = WeightStats(
            name="x",
            n=100,
            mean=0.9,
            std=0.1,
            frac_near_zero=0.0,
            frac_near_one=0.31,
            frac_below_threshold=0.0,
        )
        assert s.saturated is True

    def test_saturated_flag_does_not_fire_at_30pct(self):
        s = WeightStats(
            name="x",
            n=100,
            mean=0.9,
            std=0.1,
            frac_near_zero=0.0,
            frac_near_one=0.30,
            frac_below_threshold=0.0,
        )
        assert s.saturated is False

    def test_collapsed_flag_fires_above_80pct(self):
        s = WeightStats(
            name="x",
            n=100,
            mean=0.01,
            std=0.01,
            frac_near_zero=0.81,
            frac_near_one=0.0,
            frac_below_threshold=0.95,
        )
        assert s.collapsed is True


class TestFormatDiagnostics:
    def test_includes_all_three_sections(self, primed_trainer: T1Trainer):
        sdr = character_sdr_overlap(primed_trainer)
        ctx = context_sensitivity(primed_trainer, ["ca", "de"])
        weights = weight_distribution(primed_trainer)
        text = format_diagnostics(sdr, ctx, weights)
        assert "Checkpoint 1" in text
        assert "Checkpoint 2" in text
        assert "Checkpoint 3" in text
        assert "ff_weights" in text
