"""Tests for EntorhinalLayer — fixed random projection, forward + reverse."""

import numpy as np
import pytest

from arbora.hippocampus import EntorhinalLayer


class TestInit:
    def test_shapes(self):
        ec = EntorhinalLayer(input_dim=100, output_dim=500, sparsity=0.02, seed=0)
        assert ec.forward_weights.shape == (100, 500)
        assert ec.reverse_weights.shape == (500, 100)
        assert ec.k == 10

    def test_k_rounds_to_at_least_one(self):
        # Very small sparsity * output_dim should still yield k=1.
        ec = EntorhinalLayer(input_dim=10, output_dim=10, sparsity=0.01, seed=0)
        assert ec.k == 1

    def test_invalid_dims_raise(self):
        with pytest.raises(ValueError):
            EntorhinalLayer(input_dim=0, output_dim=10)
        with pytest.raises(ValueError):
            EntorhinalLayer(input_dim=10, output_dim=0)

    def test_invalid_sparsity_raises(self):
        with pytest.raises(ValueError):
            EntorhinalLayer(input_dim=10, output_dim=10, sparsity=0.0)
        with pytest.raises(ValueError):
            EntorhinalLayer(input_dim=10, output_dim=10, sparsity=1.5)

    def test_forward_and_reverse_are_independent_matrices(self):
        # Reverse must not be forward.T — the two pathways are independent.
        ec = EntorhinalLayer(input_dim=100, output_dim=500, seed=0)
        assert not np.allclose(ec.forward_weights.T, ec.reverse_weights)

    def test_different_seeds_produce_different_projections(self):
        ec0 = EntorhinalLayer(input_dim=100, output_dim=500, seed=0)
        ec1 = EntorhinalLayer(input_dim=100, output_dim=500, seed=1)
        assert not np.allclose(ec0.forward_weights, ec1.forward_weights)


class TestForward:
    def test_exact_k_active(self):
        ec = EntorhinalLayer(input_dim=100, output_dim=500, sparsity=0.02, seed=0)
        x = np.random.default_rng(42).normal(size=100)
        out = ec.forward(x)
        assert out.shape == (500,)
        assert out.dtype == np.bool_
        assert out.sum() == 10

    def test_deterministic(self):
        """Same input → same output (frozen projection)."""
        ec = EntorhinalLayer(input_dim=100, output_dim=500, seed=0)
        x = np.random.default_rng(42).normal(size=100)
        out1 = ec.forward(x)
        out2 = ec.forward(x)
        np.testing.assert_array_equal(out1, out2)

    def test_accepts_boolean_input(self):
        """Sparse binary cortical firing rates should project cleanly."""
        ec = EntorhinalLayer(input_dim=100, output_dim=500, seed=0)
        x = np.zeros(100, dtype=np.bool_)
        x[::10] = True
        out = ec.forward(x)
        assert out.sum() == ec.k

    def test_accepts_multidim_input(self):
        """Flattens automatically, e.g. (n_columns, n_per_col) laminae."""
        ec = EntorhinalLayer(input_dim=100, output_dim=500, seed=0)
        x = np.random.default_rng(0).normal(size=(10, 10))
        out = ec.forward(x)
        assert out.shape == (500,)

    def test_wrong_dim_raises(self):
        ec = EntorhinalLayer(input_dim=100, output_dim=500, seed=0)
        with pytest.raises(ValueError):
            ec.forward(np.zeros(99))


class TestReverse:
    def test_output_shape_and_dtype(self):
        ec = EntorhinalLayer(input_dim=100, output_dim=500, seed=0)
        y = np.zeros(500, dtype=np.bool_)
        y[:10] = True
        out = ec.reverse(y)
        assert out.shape == (100,)
        assert out.dtype == np.float64

    def test_accepts_continuous_input(self):
        ec = EntorhinalLayer(input_dim=100, output_dim=500, seed=0)
        y = np.random.default_rng(0).normal(size=500)
        out = ec.reverse(y)
        assert out.shape == (100,)

    def test_wrong_dim_raises(self):
        ec = EntorhinalLayer(input_dim=100, output_dim=500, seed=0)
        with pytest.raises(ValueError):
            ec.reverse(np.zeros(499))

    def test_reverse_is_not_sparsified(self):
        """Reverse is continuous — no k-WTA. Downstream regions sparsify."""
        ec = EntorhinalLayer(input_dim=100, output_dim=500, seed=0)
        y = np.random.default_rng(0).normal(size=500)
        out = ec.reverse(y)
        # Should have many non-zero entries (not a sparse binary vector).
        assert (out != 0).sum() > 50


class TestDistancePreservation:
    """Forward projection should approximately preserve input similarity."""

    def test_jaccard_separates_similar_from_dissimilar(self):
        """Small perturbations produce much higher overlap than random inputs."""
        ec = EntorhinalLayer(input_dim=200, output_dim=1000, sparsity=0.02, seed=0)
        rng = np.random.default_rng(42)
        x1 = rng.normal(size=200)
        x1_perturbed = x1 + 0.02 * rng.normal(size=200)  # small noise
        x2 = rng.normal(size=200)  # unrelated input

        out1 = ec.forward(x1)
        out1_perturbed = ec.forward(x1_perturbed)
        out2 = ec.forward(x2)

        def jaccard(a: np.ndarray, b: np.ndarray) -> float:
            union = (a | b).sum()
            if union == 0:
                return 1.0
            return float((a & b).sum() / union)

        sim_similar = jaccard(out1, out1_perturbed)
        sim_dissimilar = jaccard(out1, out2)

        # Expected random-chance Jaccard: ~k/(2*output_dim - k) ≈ 0.01.
        assert sim_similar > sim_dissimilar
        assert sim_similar > 0.5, (
            f"small perturbation should preserve most active units; "
            f"got jaccard={sim_similar:.3f}"
        )
        assert sim_dissimilar < 0.1, (
            f"unrelated inputs should be near-orthogonal; "
            f"got jaccard={sim_dissimilar:.3f}"
        )

    def test_graded_similarity_matches_graded_perturbation(self):
        """Larger perturbations produce smaller Jaccard overlaps."""
        ec = EntorhinalLayer(input_dim=200, output_dim=1000, sparsity=0.02, seed=0)
        rng = np.random.default_rng(0)
        x = rng.normal(size=200)
        base = ec.forward(x)

        def jaccard(a: np.ndarray, b: np.ndarray) -> float:
            union = (a | b).sum()
            return float((a & b).sum() / union) if union else 1.0

        sims = []
        for noise_scale in [0.01, 0.1, 1.0]:
            perturbed = x + noise_scale * rng.normal(size=200)
            sims.append(jaccard(base, ec.forward(perturbed)))

        # Monotonic: as noise grows, overlap shrinks.
        assert sims[0] > sims[1] > sims[2]
