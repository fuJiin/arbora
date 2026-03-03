import numpy as np
import pytest

from step.config import ModelConfig
from step.model import (
    ModelState,
    _local_normalize,
    initial_state,
    learn,
    observe,
    predict,
)


class TestInitialState:
    def test_empty(self):
        config = ModelConfig(n=64, k=4)
        state = initial_state(config)
        assert state.weights.shape == (64, 64)
        assert np.all(state.weights == 0)
        assert state.history == {}

    def test_no_config(self):
        state = initial_state()
        assert state.weights.shape == (0, 0)
        assert state.history == {}


class TestPredict:
    def setup_method(self):
        self.config = ModelConfig(n=64, k=4, eligibility_window=20)

    def test_returns_k_indices(self):
        weights = np.zeros((64, 64), dtype=np.float32)
        weights[0] = np.random.default_rng(0).random(64).astype(np.float32)
        state = ModelState(
            weights=weights,
            history={0: frozenset({0, 1, 2})},
        )
        pred = predict(state, t=1, config=self.config)
        assert len(pred) == 4

    def test_all_in_range(self):
        weights = np.zeros((64, 64), dtype=np.float32)
        weights[10] = np.ones(64, dtype=np.float32)
        state = ModelState(
            weights=weights,
            history={0: frozenset({10})},
        )
        pred = predict(state, t=1, config=self.config)
        assert all(0 <= idx < 64 for idx in pred)

    def test_returns_frozenset(self):
        state = initial_state(self.config)
        pred = predict(state, t=0, config=self.config)
        assert isinstance(pred, frozenset)

    def test_empty_state(self):
        state = initial_state(self.config)
        pred = predict(state, t=0, config=self.config)
        assert len(pred) == 4

    def test_handles_missing_history_steps(self):
        """History may have gaps; predict should skip missing steps."""
        weights = np.zeros((64, 64), dtype=np.float32)
        state = ModelState(weights=weights, history={5: frozenset({0})})
        pred = predict(state, t=10, config=self.config)
        assert len(pred) == 4


class TestLearn:
    def setup_method(self):
        self.config = ModelConfig(
            n=64,
            k=4,
            max_lr=0.5,
            weight_decay=0.999,
            penalty_factor=0.5,
            eligibility_window=20,
        )

    def test_returns_iou(self):
        weights = np.zeros((64, 64), dtype=np.float32)
        state = ModelState(weights=weights, history={0: frozenset({10})})
        iou = learn(
            state,
            t=1,
            current_sdr=frozenset({0, 1, 2, 3}),
            predicted_sdr=frozenset({0, 1, 4, 5}),
            config=self.config,
        )
        assert iou == 0.5

    def test_reinforces_correct_bits(self):
        current = frozenset({0, 1, 2, 3})
        predicted = frozenset({0, 1, 4, 5})
        weights = np.zeros((64, 64), dtype=np.float32)
        state = ModelState(weights=weights, history={0: frozenset({10})})
        learn(
            state, t=1, current_sdr=current, predicted_sdr=predicted, config=self.config
        )
        for idx in current:
            assert state.weights[10][idx] > 0

    def test_penalizes_false_positives(self):
        current = frozenset({0, 1, 2, 3})
        predicted = frozenset({0, 1, 4, 5})
        weights = np.zeros((64, 64), dtype=np.float32)
        state = ModelState(weights=weights, history={0: frozenset({10})})
        learn(
            state, t=1, current_sdr=current, predicted_sdr=predicted, config=self.config
        )
        for idx in predicted - current:
            assert state.weights[10][idx] < 0

    def test_weight_decay(self):
        weights = np.ones((64, 64), dtype=np.float32) * 100.0
        state = ModelState(
            weights=weights,
            history={0: frozenset({10})},
        )
        original = state.weights[10].copy()
        # Perfect prediction -> eta=0, only decay applies
        current = frozenset({0, 1, 2, 3})
        learn(
            state, t=1, current_sdr=current, predicted_sdr=current, config=self.config
        )
        # With perfect iou (eta=0), decay still applied to src row 10
        for idx in range(4, 64):
            assert abs(state.weights[10][idx] - original[idx] * 0.999) < 1e-3


class TestObserve:
    def setup_method(self):
        self.config = ModelConfig(n=64, k=4, eligibility_window=5)

    def test_adds_to_history(self):
        state = initial_state(self.config)
        sdr = frozenset({1, 2, 3})
        state = observe(state, t=0, sdr=sdr, config=self.config)
        assert state.history[0] == sdr

    def test_prunes_beyond_window(self):
        state = initial_state(self.config)
        for t in range(10):
            state = observe(state, t=t, sdr=frozenset({t}), config=self.config)
        assert 0 not in state.history
        assert 4 not in state.history
        assert 5 in state.history
        assert 9 in state.history

    def test_preserves_weights_reference(self):
        """observe returns new history but same weights array."""
        state = initial_state(self.config)
        new_state = observe(state, t=0, sdr=frozenset({1}), config=self.config)
        assert new_state.weights is state.weights


class TestRoundTrip:
    def test_predict_learn_observe_cycle(self):
        config = ModelConfig(n=64, k=4, eligibility_window=10)
        state = initial_state(config)

        sdrs = [frozenset({i, i + 1, i + 2, i + 3}) for i in range(0, 40, 4)]
        ious = []

        for t, sdr in enumerate(sdrs):
            if t > 0:
                pred = predict(state, t, config)
                iou = learn(state, t, sdr, pred, config)
                ious.append(iou)
            state = observe(state, t, sdr, config)

        assert len(ious) == len(sdrs) - 1
        assert all(0.0 <= x <= 1.0 for x in ious)


class TestLocalNormalize:
    def test_positive_values(self):
        vec = np.array([2.0, 4.0, 1.0])
        result = _local_normalize(vec)
        np.testing.assert_array_almost_equal(result, [0.5, 1.0, 0.25])

    def test_max_becomes_one(self):
        vec = np.array([3.0, 7.0, 5.0])
        result = _local_normalize(vec)
        assert result.max() == pytest.approx(1.0)

    def test_zero_vector_unchanged(self):
        vec = np.zeros(5)
        result = _local_normalize(vec)
        np.testing.assert_array_equal(result, np.zeros(5))

    def test_all_negative_unchanged(self):
        vec = np.array([-1.0, -2.0, -3.0])
        result = _local_normalize(vec)
        np.testing.assert_array_equal(result, vec)

    def test_mixed_with_positive_max(self):
        vec = np.array([-1.0, 2.0, 0.0])
        result = _local_normalize(vec)
        np.testing.assert_array_almost_equal(result, [-0.5, 1.0, 0.0])
