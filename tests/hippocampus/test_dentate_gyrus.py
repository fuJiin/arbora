"""Tests for DentateGyrus — k-WTA pattern separation."""

import numpy as np
import pytest

from arbora.hippocampus import DentateGyrus


class TestInit:
    def test_shapes(self):
        dg = DentateGyrus(input_dim=200, output_dim=2000, k=40, seed=0)
        assert dg.weights.shape == (200, 2000)
        assert dg.input_dim == 200
        assert dg.output_dim == 2000

    def test_invalid_dims_raise(self):
        with pytest.raises(ValueError):
            DentateGyrus(input_dim=0, output_dim=100, k=2)
        with pytest.raises(ValueError):
            DentateGyrus(input_dim=100, output_dim=0, k=2)

    def test_requires_k_or_schedule(self):
        with pytest.raises(ValueError):
            DentateGyrus(input_dim=100, output_dim=500)

    def test_k_bounds(self):
        with pytest.raises(ValueError):
            DentateGyrus(input_dim=100, output_dim=500, k=0)
        with pytest.raises(ValueError):
            DentateGyrus(input_dim=100, output_dim=500, k=501)


class TestFixedK:
    def test_exact_k_active(self):
        dg = DentateGyrus(input_dim=200, output_dim=2000, k=40, seed=0)
        x = np.random.default_rng(1).normal(size=200)
        out = dg.forward(x)
        assert out.shape == (2000,)
        assert out.dtype == np.bool_
        assert out.sum() == 40

    def test_deterministic(self):
        dg = DentateGyrus(input_dim=200, output_dim=2000, k=40, seed=0)
        x = np.random.default_rng(1).normal(size=200)
        np.testing.assert_array_equal(dg.forward(x), dg.forward(x))

    def test_accepts_boolean_input(self):
        """Sparse binary EC input should project cleanly."""
        dg = DentateGyrus(input_dim=200, output_dim=2000, k=40, seed=0)
        x = np.zeros(200, dtype=np.bool_)
        x[::10] = True
        out = dg.forward(x)
        assert out.sum() == 40

    def test_wrong_dim_raises(self):
        dg = DentateGyrus(input_dim=200, output_dim=2000, k=40, seed=0)
        with pytest.raises(ValueError):
            dg.forward(np.zeros(199))


class TestSchedule:
    def test_schedule_overrides_k(self):
        """k comes from the schedule when provided."""
        dg = DentateGyrus(
            input_dim=200,
            output_dim=2000,
            k_schedule=lambda step: 100 if step < 10 else 20,
            seed=0,
        )
        x = np.random.default_rng(1).normal(size=200)
        early = dg.forward(x, step=0)
        late = dg.forward(x, step=100)
        assert early.sum() == 100
        assert late.sum() == 20

    def test_schedule_result_clamped_to_bounds(self):
        """Schedule returning out-of-range k is clamped into [1, output_dim]."""
        dg = DentateGyrus(
            input_dim=200,
            output_dim=500,
            k_schedule=lambda step: -5,  # negative
            seed=0,
        )
        x = np.random.default_rng(1).normal(size=200)
        out = dg.forward(x, step=0)
        assert out.sum() == 1

        dg_over = DentateGyrus(
            input_dim=200,
            output_dim=500,
            k_schedule=lambda step: 10_000,  # too large
            seed=0,
        )
        out_over = dg_over.forward(x, step=0)
        assert out_over.sum() == 500  # all active

    def test_step_none_defaults_to_zero(self):
        dg = DentateGyrus(
            input_dim=200,
            output_dim=2000,
            k_schedule=lambda step: 30 if step == 0 else 999,
            seed=0,
        )
        x = np.random.default_rng(1).normal(size=200)
        out = dg.forward(x)  # no step kwarg
        assert out.sum() == 30


class TestPatternSeparation:
    """DG's signature property: similar inputs → less-overlapping outputs."""

    def test_expansion_reduces_overlap_vs_ec_like_direct_copy(self):
        """Two slightly different inputs produce low-Jaccard DG outputs."""
        dg = DentateGyrus(input_dim=200, output_dim=4000, k=80, seed=0)
        rng = np.random.default_rng(0)
        x1 = rng.normal(size=200)
        x1_perturbed = x1 + 0.6 * rng.normal(size=200)  # meaningful noise

        out1 = dg.forward(x1)
        out1_perturbed = dg.forward(x1_perturbed)

        def jaccard(a: np.ndarray, b: np.ndarray) -> float:
            union = (a | b).sum()
            return float((a & b).sum() / union) if union else 1.0

        # With a noisy input, DG should produce a substantially different
        # pattern. Exact thresholds depend on noise scale; we just want
        # to confirm separation is occurring rather than being a pass-
        # through.
        overlap = jaccard(out1, out1_perturbed)
        assert overlap < 0.5, (
            f"expected separation under 30% noise; got jaccard={overlap:.3f}"
        )

    def test_dissimilar_inputs_near_orthogonal(self):
        dg = DentateGyrus(input_dim=200, output_dim=4000, k=80, seed=0)
        rng = np.random.default_rng(0)
        x1 = rng.normal(size=200)
        x2 = rng.normal(size=200)

        out1 = dg.forward(x1)
        out2 = dg.forward(x2)

        def jaccard(a: np.ndarray, b: np.ndarray) -> float:
            union = (a | b).sum()
            return float((a & b).sum() / union) if union else 1.0

        # Random-chance Jaccard: k / (2 * output_dim - k) ≈ 80 / 7920 ≈ 0.01.
        assert jaccard(out1, out2) < 0.05
