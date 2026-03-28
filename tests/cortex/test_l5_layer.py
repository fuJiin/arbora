"""Tests for L5 output layer on CorticalRegion base class."""

import numpy as np

from step.cortex.motor import MotorRegion
from step.cortex.region import CorticalRegion


class TestL5BaseLayer:
    """L5 layer state and activation on CorticalRegion."""

    def _make_region(self, n_l5=None, **kwargs):
        defaults = dict(
            input_dim=16,
            n_columns=8,
            n_l4=4,
            n_l23=4,
            k_columns=2,
            seed=42,
        )
        defaults.update(kwargs)
        if n_l5 is not None:
            defaults["n_l5"] = n_l5
        return CorticalRegion(**defaults)

    def test_l5_defaults_to_n_l23(self):
        r = self._make_region()
        assert r.n_l5 == r.n_l23
        assert r.n_l5_total == r.n_columns * r.n_l5

    def test_l5_custom_size(self):
        r = self._make_region(n_l5=2)
        assert r.n_l5 == 2
        assert r.n_l5_total == 8 * 2

    def test_l5_state_shapes(self):
        r = self._make_region()
        assert r.l5.active.shape == (r.n_l5_total,)
        assert r.l5.firing_rate.shape == (r.n_l5_total,)
        assert r.output_scores.shape == (r.n_columns,)

    def test_l5_activates_on_step(self):
        r = self._make_region()
        drive = np.random.default_rng(0).random(r.n_l4_total)
        r.step(drive)
        # k=2 columns active, so L5 should have active neurons
        assert r.l5.active.any()

    def test_l5_only_active_in_active_columns(self):
        r = self._make_region()
        drive = np.random.default_rng(0).random(r.n_l4_total)
        r.step(drive)
        l5_by_col = r.l5.active.reshape(r.n_columns, r.n_l5)
        for col in range(r.n_columns):
            if not r.active_columns[col]:
                assert not l5_by_col[col].any()

    def test_l5_firing_rate_tracks_activation(self):
        r = self._make_region()
        rng = np.random.default_rng(1)
        for _ in range(20):
            r.step(rng.random(r.n_l4_total))
        # After multiple steps, firing rate should be nonzero for active L5 neurons
        assert r.l5.firing_rate.sum() > 0

    def test_output_scores_per_column(self):
        r = self._make_region()
        rng = np.random.default_rng(1)
        for _ in range(10):
            r.step(rng.random(r.n_l4_total))
        # Active columns should have nonzero output scores
        assert r.output_scores.sum() > 0
        # Output scores should match per-column mean L5 firing rate
        expected = r.l5.firing_rate.reshape(r.n_columns, r.n_l5).mean(axis=1)
        np.testing.assert_allclose(r.output_scores, expected)

    def test_reset_clears_l5(self):
        r = self._make_region()
        rng = np.random.default_rng(0)
        for _ in range(5):
            r.step(rng.random(r.n_l4_total))
        r.reset_working_memory()
        assert not r.l5.active.any()
        assert r.l5.firing_rate.sum() == 0
        assert r.output_scores.sum() == 0

    def test_burst_columns_all_l5_fire(self):
        """In burst columns, all L5 neurons should fire (mirroring L2/3)."""
        r = self._make_region()
        # First step with strong drive to get some columns active
        drive = np.zeros(r.n_l4_total)
        drive[: r.n_l4] = 1.0  # Strong drive to column 0
        drive[r.n_l4 : 2 * r.n_l4] = 0.8  # Strong drive to column 1
        r.step(drive)
        # Check burst columns have all L5 neurons active
        l5_by_col = r.l5.active.reshape(r.n_columns, r.n_l5)
        for col in range(r.n_columns):
            if r.bursting_columns[col] and r.active_columns[col]:
                assert l5_by_col[col].all(), (
                    f"Burst column {col} should have all L5 active"
                )


class TestMotorL5:
    """MotorRegion using base L5 for output weights."""

    def _make_motor(self, **kwargs):
        defaults = dict(
            input_dim=16,
            n_columns=8,
            n_l4=4,
            n_l23=4,
            k_columns=2,
            n_output_tokens=32,
            seed=42,
        )
        defaults.update(kwargs)
        return MotorRegion(**defaults)

    def test_output_weights_use_l5_dim(self):
        m = self._make_motor()
        assert m.output_weights.shape == (m.n_l5_total, m.n_output_tokens)

    def test_output_eligibility_uses_l5_dim(self):
        m = self._make_motor()
        assert m._output_eligibility.shape == (m.n_l5_total, m.n_output_tokens)

    def test_observe_token_uses_l5(self):
        m = self._make_motor()
        rng = np.random.default_rng(0)
        # Run a step to get L5 active
        encoding = rng.random(m.input_dim)
        m.process(encoding)
        # Observe a token — should record L5 coincidence
        m.observe_token(5)
        # Eligibility should be nonzero where L5 was active
        l5_active_idx = np.nonzero(m.l5.active)[0]
        if len(l5_active_idx) > 0:
            assert m._output_eligibility[l5_active_idx, 5].sum() > 0

    def test_get_population_output_uses_l5(self):
        m = self._make_motor()
        rng = np.random.default_rng(0)
        # Run steps to build up L5 firing rate and output weights
        for _ in range(10):
            encoding = rng.random(m.input_dim)
            m.process(encoding)
            m.observe_token(rng.integers(0, m.n_output_tokens))
        token_id, _confidence = m.get_population_output()
        # Should produce some output
        assert token_id >= 0 or not m.l5.active.any()

    def test_output_scores_inherited_from_base(self):
        """Motor output_scores from base CorticalRegion L5."""
        m = self._make_motor()
        rng = np.random.default_rng(0)
        for _ in range(5):
            m.process(rng.random(m.input_dim))
        # Should match per-column mean L5 firing rate
        expected = m.l5.firing_rate.reshape(m.n_columns, m.n_l5).mean(axis=1)
        np.testing.assert_allclose(m.output_scores, expected)

    def test_babble_activates_l5(self):
        m = self._make_motor()
        m.exploration_noise = 1.0
        encoding = np.random.default_rng(0).random(m.input_dim)
        m.process(encoding)
        assert m.l5.active.any()
