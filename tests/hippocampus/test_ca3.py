"""Tests for CA3AttractorNetwork — Hebbian recurrent, one-shot binding."""

import numpy as np
import pytest

from arbora.hippocampus import CA3AttractorNetwork


def _sparse_pattern(dim: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Make a sparse binary CA3 pattern with exactly k active units."""
    idx = rng.choice(dim, size=k, replace=False)
    out = np.zeros(dim, dtype=np.bool_)
    out[idx] = True
    return out


class TestInit:
    def test_shapes(self):
        ca3 = CA3AttractorNetwork(dim=500, dg_dim=4000, seed=0)
        assert ca3.mossy_weights.shape == (4000, 500)
        assert ca3.recurrent.shape == (500, 500)
        assert ca3.state.shape == (500,)
        assert ca3.state.dtype == np.bool_

    def test_k_derivation(self):
        ca3 = CA3AttractorNetwork(dim=1000, dg_dim=5000, k_active=0.02)
        assert ca3.k == 20

    def test_mossy_k_derivation(self):
        ca3 = CA3AttractorNetwork(dim=500, dg_dim=4000, mossy_sparsity=0.01)
        # Each CA3 cell receives from 40 DG cells.
        assert ca3.mossy_k == 40
        # Each column of mossy_weights has exactly mossy_k non-zero entries.
        nonzero_per_col = (ca3.mossy_weights != 0).sum(axis=0)
        np.testing.assert_array_equal(nonzero_per_col, 40)

    def test_mossy_k_explicit(self):
        ca3 = CA3AttractorNetwork(dim=100, dg_dim=1000, mossy_k=25)
        assert ca3.mossy_k == 25

    def test_invalid_dims_raise(self):
        with pytest.raises(ValueError):
            CA3AttractorNetwork(dim=0, dg_dim=100)
        with pytest.raises(ValueError):
            CA3AttractorNetwork(dim=100, dg_dim=0)

    def test_invalid_k_active_raises(self):
        with pytest.raises(ValueError):
            CA3AttractorNetwork(dim=100, dg_dim=100, k_active=0.0)
        with pytest.raises(ValueError):
            CA3AttractorNetwork(dim=100, dg_dim=100, k_active=1.5)

    def test_invalid_mossy_k_raises(self):
        with pytest.raises(ValueError):
            CA3AttractorNetwork(dim=100, dg_dim=1000, mossy_k=0)
        with pytest.raises(ValueError):
            CA3AttractorNetwork(dim=100, dg_dim=1000, mossy_k=1001)


class TestEncode:
    def test_encode_returns_sparse_binary(self):
        ca3 = CA3AttractorNetwork(dim=500, dg_dim=4000, k_active=0.02, seed=0)
        rng = np.random.default_rng(0)
        dg_pat = _sparse_pattern(4000, 80, rng)
        out = ca3.encode(dg_pat)
        assert out.shape == (500,)
        assert out.dtype == np.bool_
        assert out.sum() == ca3.k

    def test_encode_strengthens_coactive_pairs(self):
        """After encoding, off-diagonal weights among co-active units > 0."""
        ca3 = CA3AttractorNetwork(
            dim=200, dg_dim=2000, k_active=0.05, learning_rate=0.5, seed=0
        )
        rng = np.random.default_rng(0)
        dg_pat = _sparse_pattern(2000, 40, rng)
        active = ca3.encode(dg_pat)
        idx = np.flatnonzero(active)
        # All off-diagonal pairs among active units should be strengthened.
        submatrix = ca3.recurrent[np.ix_(idx, idx)]
        off_diag_mask = ~np.eye(submatrix.shape[0], dtype=bool)
        assert (submatrix[off_diag_mask] > 0).all()
        # Diagonal must remain zero (no self-loops).
        assert (np.diag(ca3.recurrent) == 0).all()

    def test_encode_updates_state(self):
        ca3 = CA3AttractorNetwork(dim=200, dg_dim=2000, seed=0)
        rng = np.random.default_rng(0)
        dg_pat = _sparse_pattern(2000, 40, rng)
        out = ca3.encode(dg_pat)
        np.testing.assert_array_equal(ca3.state, out)

    def test_encode_empty_pattern_does_not_crash(self):
        """All-zero DG input produces an active set (from arbitrary top-k)."""
        ca3 = CA3AttractorNetwork(dim=100, dg_dim=1000, seed=0)
        out = ca3.encode(np.zeros(1000, dtype=np.bool_))
        # Zero drive; top-k is arbitrary but well-defined.
        assert out.shape == (100,)

    def test_wrong_dg_dim_raises(self):
        ca3 = CA3AttractorNetwork(dim=100, dg_dim=1000, seed=0)
        with pytest.raises(ValueError):
            ca3.encode(np.zeros(999))


class TestRetrieve:
    def test_full_cue_returns_same_pattern(self):
        """encode X, retrieve with X → get X back (one-shot binding)."""
        ca3 = CA3AttractorNetwork(
            dim=500, dg_dim=4000, k_active=0.02, learning_rate=0.5, seed=0
        )
        rng = np.random.default_rng(0)
        dg_pat = _sparse_pattern(4000, 80, rng)
        encoded = ca3.encode(dg_pat)
        retrieved = ca3.retrieve(encoded, n_iter=3)
        np.testing.assert_array_equal(retrieved, encoded)

    def test_partial_cue_completes_pattern(self):
        """Masking out half of a stored pattern still retrieves it."""
        ca3 = CA3AttractorNetwork(
            dim=500, dg_dim=4000, k_active=0.02, learning_rate=0.5, seed=0
        )
        rng = np.random.default_rng(0)
        dg_pat = _sparse_pattern(4000, 80, rng)
        encoded = ca3.encode(dg_pat)

        # Drop half the active CA3 units from the cue.
        active_idx = np.flatnonzero(encoded)
        keep_idx = active_idx[: len(active_idx) // 2]
        partial = np.zeros_like(encoded)
        partial[keep_idx] = True

        retrieved = ca3.retrieve(partial, n_iter=5)
        # Should recover most of the original active units.
        overlap = (retrieved & encoded).sum()
        assert overlap >= 0.9 * encoded.sum(), (
            f"pattern completion failed; recovered {overlap}/{encoded.sum()}"
        )

    def test_capacity_degrades_gracefully(self):
        """Multiple distinct patterns remain retrievable beyond a handful."""
        ca3 = CA3AttractorNetwork(
            dim=500, dg_dim=4000, k_active=0.02, learning_rate=0.5, seed=0
        )
        rng = np.random.default_rng(0)
        patterns = [_sparse_pattern(4000, 80, rng) for _ in range(5)]
        encodings = [ca3.encode(p) for p in patterns]

        # Each stored pattern should be retrievable from its own full cue.
        for enc in encodings:
            retrieved = ca3.retrieve(enc, n_iter=3)
            overlap = (retrieved & enc).sum()
            assert overlap >= 0.8 * enc.sum()

    def test_retrieve_wrong_dim_raises(self):
        ca3 = CA3AttractorNetwork(dim=100, dg_dim=1000, seed=0)
        with pytest.raises(ValueError):
            ca3.retrieve(np.zeros(99))


class TestReset:
    def test_reset_preserves_recurrent(self):
        ca3 = CA3AttractorNetwork(dim=200, dg_dim=2000, seed=0)
        rng = np.random.default_rng(0)
        ca3.encode(_sparse_pattern(2000, 40, rng))
        before = ca3.recurrent.copy()
        ca3.reset()
        assert not ca3.state.any()
        np.testing.assert_array_equal(ca3.recurrent, before)

    def test_reset_memory_clears_recurrent(self):
        ca3 = CA3AttractorNetwork(dim=200, dg_dim=2000, seed=0)
        rng = np.random.default_rng(0)
        ca3.encode(_sparse_pattern(2000, 40, rng))
        assert ca3.recurrent.any()
        ca3.reset_memory()
        assert not ca3.state.any()
        assert not ca3.recurrent.any()

    def test_memory_survives_reset_for_retrieval(self):
        """After reset(), a stored pattern can still be retrieved via cue."""
        ca3 = CA3AttractorNetwork(
            dim=500, dg_dim=4000, k_active=0.02, learning_rate=0.5, seed=0
        )
        rng = np.random.default_rng(0)
        dg_pat = _sparse_pattern(4000, 80, rng)
        encoded = ca3.encode(dg_pat)
        ca3.reset()
        # Pass the originally encoded CA3 pattern back in as a cue.
        retrieved = ca3.retrieve(encoded, n_iter=3)
        np.testing.assert_array_equal(retrieved, encoded)
