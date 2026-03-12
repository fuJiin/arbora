"""Tests for RepresentationTracker."""

import numpy as np
import pytest

from step.probes.representation import RepresentationTracker


@pytest.fixture
def tracker():
    return RepresentationTracker(n_columns=8, n_l4=4)


def _make_columns(cols, n_columns=8):
    """Make active_columns array from list of column indices."""
    arr = np.zeros(n_columns, dtype=np.bool_)
    arr[list(cols)] = True
    return arr


def _make_neurons(neurons, n_total=32):
    """Make active_l4 array from list of neuron indices."""
    arr = np.zeros(n_total, dtype=np.bool_)
    arr[list(neurons)] = True
    return arr


class TestObserve:
    def test_tracks_token_columns(self, tracker):
        cols = _make_columns([0, 1])
        neurons = _make_neurons([0, 4])
        tracker.observe(42, cols, neurons)

        assert 42 in tracker._token_columns
        assert len(tracker._token_columns[42]) == 1
        assert tracker._token_columns[42][0] == frozenset({0, 1})

    def test_tracks_column_tokens(self, tracker):
        cols = _make_columns([0, 1])
        neurons = _make_neurons([0, 4])
        tracker.observe(42, cols, neurons)

        assert tracker._column_tokens[0][42] == 1
        assert tracker._column_tokens[1][42] == 1

    def test_tracks_bigrams(self, tracker):
        cols = _make_columns([0])
        neurons = _make_neurons([0])
        tracker.observe(10, cols, neurons)
        tracker.observe(20, cols, neurons)

        assert (10, 20) in tracker._bigram_neurons

    def test_reset_context_breaks_bigram(self, tracker):
        cols = _make_columns([0])
        neurons = _make_neurons([0])
        tracker.observe(10, cols, neurons)
        tracker.reset_context()
        tracker.observe(20, cols, neurons)

        assert (10, 20) not in tracker._bigram_neurons

    def test_step_count(self, tracker):
        cols = _make_columns([0])
        neurons = _make_neurons([0])
        tracker.observe(1, cols, neurons)
        tracker.observe(2, cols, neurons)
        assert tracker._n_steps == 2


class TestColumnSelectivity:
    def test_single_token_per_column_is_selective(self, tracker):
        """Column that only responds to one token = max selectivity."""
        # Column 0 only sees token 1
        tracker.observe(1, _make_columns([0]), _make_neurons([0]))
        tracker.observe(1, _make_columns([0]), _make_neurons([0]))
        # Column 1 only sees token 2
        tracker.observe(2, _make_columns([1]), _make_neurons([4]))
        tracker.observe(2, _make_columns([1]), _make_neurons([4]))

        sel = tracker.column_selectivity()
        # Columns 0 and 1 each see exactly one token -> entropy = 0
        assert sel["per_column"][0] == 0.0
        assert sel["per_column"][1] == 0.0

    def test_uniform_column_not_selective(self, tracker):
        """Column responding to many tokens equally = low selectivity."""
        for tid in range(20):
            tracker.observe(tid, _make_columns([0]), _make_neurons([0]))

        sel = tracker.column_selectivity()
        # Column 0 sees 20 tokens uniformly -> high entropy
        assert sel["per_column"][0] > 0.8

    def test_mean_in_range(self, tracker):
        for tid in range(5):
            tracker.observe(tid, _make_columns([tid % 8]), _make_neurons([0]))
        sel = tracker.column_selectivity()
        assert 0.0 <= sel["mean"] <= 1.0


class TestRepresentationSimilarity:
    def test_identical_tokens_high_similarity(self, tracker):
        """Tokens that always activate same columns = high similarity."""
        for _ in range(10):
            tracker.observe(1, _make_columns([0, 1]), _make_neurons([0]))
            tracker.observe(2, _make_columns([0, 1]), _make_neurons([0]))

        sim = tracker.representation_similarity()
        assert sim["mean"] > 0.8

    def test_disjoint_tokens_low_similarity(self, tracker):
        """Tokens that activate different columns = low similarity."""
        for _ in range(10):
            tracker.observe(1, _make_columns([0, 1]), _make_neurons([0, 4]))
            tracker.observe(2, _make_columns([6, 7]), _make_neurons([24, 28]))

        sim = tracker.representation_similarity()
        assert sim["mean"] < 0.2

    def test_nontrivial_structure(self, tracker):
        """Mix of similar and dissimilar tokens = nontrivial."""
        for _ in range(10):
            # Tokens 1,2 share columns
            tracker.observe(1, _make_columns([0, 1]), _make_neurons([0]))
            tracker.observe(2, _make_columns([0, 2]), _make_neurons([0]))
            # Token 3 is different
            tracker.observe(3, _make_columns([6, 7]), _make_neurons([24]))

        sim = tracker.representation_similarity()
        assert sim["std"] > 0.01


class TestContextDiscrimination:
    def test_same_neurons_different_context_no_discrimination(self, tracker):
        """Same neuron pattern regardless of context = 0 discrimination."""
        neurons = _make_neurons([0, 4])
        cols = _make_columns([0, 1])
        # Token 99 preceded by different tokens, always same neurons
        for prev in range(5):
            tracker.observe(prev, cols, neurons)
            tracker.observe(99, cols, neurons)
            tracker.reset_context()

        ctx = tracker.context_discrimination(min_contexts=3)
        if ctx["n_eligible_tokens"] > 0:
            assert ctx["mean_discrimination"] < 0.1

    def test_different_neurons_different_context_high_discrimination(self, tracker):
        """Different neuron pattern per context = high discrimination."""
        cols = _make_columns([0, 1])
        for prev in range(5):
            # Each context produces different neurons for token 99
            neurons = _make_neurons([prev * 4, prev * 4 + 1])
            tracker.observe(prev, cols, neurons)
            tracker.observe(99, cols, neurons)
            tracker.reset_context()

        ctx = tracker.context_discrimination(min_contexts=3)
        if ctx["n_eligible_tokens"] > 0:
            assert ctx["mean_discrimination"] > 0.3


class TestFFConvergence:
    def test_sparse_weights(self, tracker):
        """Mostly-zero weights = high sparsity."""
        ff = np.zeros((100, 8))
        ff[:5, 0] = 0.5  # Only 5 nonzero entries in column 0

        conv = tracker.ff_convergence(ff)
        assert conv["weight_sparsity"] > 0.9

    def test_differentiated_columns(self, tracker):
        """Columns with different weight patterns = low cosine."""
        ff = np.zeros((100, 8))
        for col in range(8):
            start = col * 12
            ff[start : start + 5, col] = 1.0

        conv = tracker.ff_convergence(ff)
        assert conv["cross_col_cosine_mean"] < 0.1

    def test_identical_columns_high_cosine(self, tracker):
        """Columns with same weights = high cosine."""
        ff = np.zeros((100, 8))
        for col in range(8):
            ff[:10, col] = 1.0  # All columns same pattern

        conv = tracker.ff_convergence(ff)
        assert conv["cross_col_cosine_mean"] > 0.9


class TestSummary:
    def test_summary_has_all_keys(self, tracker):
        for tid in range(5):
            tracker.observe(
                tid,
                _make_columns([tid % 8]),
                _make_neurons([tid * 4]),
            )

        ff = np.random.default_rng(0).random((100, 8))
        s = tracker.summary(ff)

        assert "column_selectivity_mean" in s
        assert "similarity_mean" in s
        assert "context_discrimination" in s
        assert "ff_sparsity" in s
        assert "n_unique_tokens" in s
        assert s["n_unique_tokens"] == 5

    def test_summary_without_ff(self, tracker):
        tracker.observe(1, _make_columns([0]), _make_neurons([0]))
        s = tracker.summary()
        assert "column_selectivity_mean" in s
        assert "ff_sparsity" not in s
