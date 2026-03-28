"""Tests for MotorRegion with n_l4=0 (agranular motor cortex)."""

import numpy as np
import pytest

from step.cortex.motor import MotorRegion


@pytest.fixture()
def motor():
    return MotorRegion(
        input_dim=32,
        n_columns=8,
        n_l4=0,
        n_l23=4,
        k_columns=2,
        n_output_tokens=7,
        seed=42,
    )


@pytest.fixture()
def motor_granular():
    """Standard granular motor region for comparison."""
    return MotorRegion(
        input_dim=32,
        n_columns=8,
        n_l4=4,
        n_l23=4,
        k_columns=2,
        n_output_tokens=7,
        seed=42,
    )


class TestMotorAgranular:
    def test_ff_weights_target_l23(self, motor):
        assert motor.ff_weights.shape == (32, motor.n_l23_total)

    def test_process_runs(self, motor):
        encoding = np.random.default_rng(0).random(32)
        result = motor.process(encoding)
        assert isinstance(result, np.ndarray)

    def test_produces_output(self, motor):
        encoding = np.random.default_rng(0).random(32)
        motor.process(encoding)
        token_id, _conf = motor.get_population_output()
        # May or may not produce output depending on threshold
        assert isinstance(token_id, (int, np.integer))

    def test_babble_runs(self, motor):
        motor.babbling_noise = 1.0
        encoding = np.random.default_rng(0).random(32)
        result = motor.process(encoding)
        assert isinstance(result, np.ndarray)
        assert motor.l23.active.any()

    def test_babble_activates_l5(self, motor):
        motor.babbling_noise = 1.0
        encoding = np.random.default_rng(0).random(32)
        motor.process(encoding)
        assert motor.l5.active.any()

    def test_observe_token(self, motor):
        encoding = np.random.default_rng(0).random(32)
        motor.process(encoding)
        motor.observe_token(3)  # should not error

    def test_apply_reward(self, motor):
        encoding = np.random.default_rng(0).random(32)
        motor.process(encoding)
        motor.observe_token(3)
        w_before = motor.output_weights.copy()
        motor.apply_reward(1.0)
        # Weights should change after positive reward
        assert not np.array_equal(motor.output_weights, w_before)

    def test_goal_drive(self, motor):
        motor.init_goal_drive(source_dim=16)
        assert motor._goal_weights.shape == (16, motor.n_l23_total)
        motor.set_goal_drive(np.random.default_rng(0).random(16))
        encoding = np.random.default_rng(0).random(32)
        motor.process(encoding)

    def test_multiple_episodes(self, motor):
        """Simulate multiple episodes without error."""
        rng = np.random.default_rng(0)
        for _ep in range(5):
            for _step in range(20):
                motor.process(rng.random(32))
            motor.apply_reward(rng.random())


class TestMotorGranularUnchanged:
    def test_ff_weights_target_l4(self, motor_granular):
        assert motor_granular.ff_weights.shape == (32, motor_granular.n_l4_total)

    def test_process_runs(self, motor_granular):
        encoding = np.random.default_rng(0).random(32)
        result = motor_granular.process(encoding)
        assert isinstance(result, np.ndarray)

    def test_babble_runs(self, motor_granular):
        motor_granular.babbling_noise = 1.0
        encoding = np.random.default_rng(0).random(32)
        result = motor_granular.process(encoding)
        assert isinstance(result, np.ndarray)
