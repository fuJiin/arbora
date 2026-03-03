from step.config import EncoderConfig, ModelConfig, TrainingConfig
from step.model import initial_state
from step.sdr import encode_token
from step.training import train, train_step


class TestTrainStep:
    def test_first_step_no_iou(self):
        config = ModelConfig(n=64, k=4, eligibility_window=10)
        state = initial_state()
        sdr = frozenset({0, 1, 2, 3})
        state, iou = train_step(state, t=0, current_sdr=sdr, config=config)
        assert iou is None
        assert 0 in state.history

    def test_second_step_has_iou(self):
        config = ModelConfig(n=64, k=4, eligibility_window=10)
        state = initial_state()
        sdr0 = frozenset({0, 1, 2, 3})
        state, _ = train_step(state, t=0, current_sdr=sdr0, config=config)
        sdr1 = frozenset({4, 5, 6, 7})
        state, iou = train_step(state, t=1, current_sdr=sdr1, config=config)
        assert iou is not None
        assert 0.0 <= iou <= 1.0


class TestTrain:
    def test_respects_max_tokens(self):
        encoder_config = EncoderConfig(n=128, k=5)
        model_config = ModelConfig(n=128, k=5, eligibility_window=10)
        training_config = TrainingConfig(
            max_tokens=20, log_interval=5, rolling_window=5
        )

        stream = [(t, t % 10, encode_token(t % 10, encoder_config)) for t in range(20)]

        log_calls: list[tuple[int, float]] = []

        def mock_log(t: int, rolling: float) -> None:
            log_calls.append((t, rolling))

        state = train(
            iter(stream),
            model_config,
            encoder_config,
            training_config,
            log_fn=mock_log,
        )

        assert len(state.history) > 0

    def test_calls_log_fn_at_intervals(self):
        encoder_config = EncoderConfig(n=128, k=5)
        model_config = ModelConfig(n=128, k=5, eligibility_window=10)
        training_config = TrainingConfig(
            max_tokens=30, log_interval=10, rolling_window=5
        )

        stream = [(t, t % 5, encode_token(t % 5, encoder_config)) for t in range(30)]

        log_calls: list[tuple[int, float]] = []

        def mock_log(t: int, rolling: float) -> None:
            log_calls.append((t, rolling))

        train(
            iter(stream), model_config, encoder_config, training_config, log_fn=mock_log
        )

        # Should log at t=10 and t=20 (multiples of log_interval that are > 0)
        logged_ts = [t for t, _ in log_calls]
        assert 10 in logged_ts
        assert 20 in logged_ts

    def test_empty_stream(self):
        encoder_config = EncoderConfig(n=128, k=5)
        model_config = ModelConfig(n=128, k=5, eligibility_window=10)
        training_config = TrainingConfig(max_tokens=0, log_interval=10)

        state = train(iter([]), model_config, encoder_config, training_config)
        assert state.weights == {}
        assert state.history == {}
