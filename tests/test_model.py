import numpy as np

from step.config import ModelConfig
from step.model import ModelState, initial_state, observe, predict, update


class TestInitialState:
    def test_empty(self):
        state = initial_state()
        assert state.weights == {}
        assert state.history == {}


class TestPredict:
    def setup_method(self):
        self.config = ModelConfig(n=64, k=4, eligibility_window=20)

    def test_returns_k_indices(self):
        state = ModelState(
            weights={0: np.random.default_rng(0).random(64)},
            history={0: frozenset({0, 1, 2})},
        )
        pred = predict(state, t=1, config=self.config)
        assert len(pred) == 4

    def test_all_in_range(self):
        state = ModelState(
            weights={10: np.ones(64)},
            history={0: frozenset({10})},
        )
        pred = predict(state, t=1, config=self.config)
        assert all(0 <= idx < 64 for idx in pred)

    def test_returns_frozenset(self):
        state = initial_state()
        pred = predict(state, t=0, config=self.config)
        assert isinstance(pred, frozenset)

    def test_empty_state(self):
        state = initial_state()
        pred = predict(state, t=0, config=self.config)
        assert len(pred) == 4

    def test_handles_missing_history_steps(self):
        """History may have gaps; predict should skip missing steps."""
        state = ModelState(weights={}, history={5: frozenset({0})})
        pred = predict(state, t=10, config=self.config)
        assert len(pred) == 4


class TestUpdate:
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
        state = ModelState(weights={}, history={0: frozenset({10})})
        iou = update(
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
        state = ModelState(weights={}, history={0: frozenset({10})})
        update(
            state, t=1, current_sdr=current, predicted_sdr=predicted, config=self.config
        )
        for idx in current:
            assert state.weights[10][idx] > 0

    def test_penalizes_false_positives(self):
        current = frozenset({0, 1, 2, 3})
        predicted = frozenset({0, 1, 4, 5})
        state = ModelState(weights={}, history={0: frozenset({10})})
        update(
            state, t=1, current_sdr=current, predicted_sdr=predicted, config=self.config
        )
        for idx in predicted - current:
            assert state.weights[10][idx] < 0

    def test_weight_decay(self):
        state = ModelState(
            weights={10: np.ones(64) * 100.0},
            history={0: frozenset({10})},
        )
        original = state.weights[10].copy()
        # Perfect prediction → eta=0, only decay applies
        current = frozenset({0, 1, 2, 3})
        update(
            state, t=1, current_sdr=current, predicted_sdr=current, config=self.config
        )
        for idx in range(4, 64):
            assert abs(state.weights[10][idx] - original[idx] * 0.999) < 1e-10


class TestObserve:
    def setup_method(self):
        self.config = ModelConfig(n=64, k=4, eligibility_window=5)

    def test_adds_to_history(self):
        state = initial_state()
        sdr = frozenset({1, 2, 3})
        state = observe(state, t=0, sdr=sdr, config=self.config)
        assert state.history[0] == sdr

    def test_prunes_beyond_window(self):
        state = initial_state()
        for t in range(10):
            state = observe(state, t=t, sdr=frozenset({t}), config=self.config)
        # Window is 5, so t=5 prunes t=0, t=6 prunes t=1, etc.
        # At t=9, history should contain t=5..9
        assert 0 not in state.history
        assert 4 not in state.history
        assert 5 in state.history
        assert 9 in state.history

    def test_preserves_weights_reference(self):
        """observe returns new history but same weights dict."""
        state = initial_state()
        state.weights[0] = np.zeros(10)
        new_state = observe(state, t=0, sdr=frozenset({1}), config=self.config)
        assert new_state.weights is state.weights


class TestRoundTrip:
    def test_predict_update_observe_cycle(self):
        config = ModelConfig(n=64, k=4, eligibility_window=10)
        state = initial_state()

        sdrs = [frozenset({i, i + 1, i + 2, i + 3}) for i in range(0, 40, 4)]
        ious = []

        for t, sdr in enumerate(sdrs):
            if t > 0:
                pred = predict(state, t, config)
                iou = update(state, t, sdr, pred, config)
                ious.append(iou)
            state = observe(state, t, sdr, config)

        assert len(ious) == len(sdrs) - 1
        assert all(0.0 <= x <= 1.0 for x in ious)
