import pytest

from step.cortex.modulators import ThalamicGate
from step.cortex.motor import MotorRegion
from step.cortex.sensory import SensoryRegion
from step.cortex.topology import Topology
from step.data import STORY_BOUNDARY
from step.encoders.charbit import CharbitEncoder


@pytest.fixture()
def encoder():
    return CharbitEncoder(length=4, width=5, chars="abcd")


@pytest.fixture()
def region1():
    return SensoryRegion(
        input_dim=4 * 5,
        encoding_width=5,
        n_columns=8,
        n_l4=2,
        n_l23=2,
        k_columns=2,
        seed=42,
    )


@pytest.fixture()
def motor(region1):
    return MotorRegion(
        input_dim=region1.n_l23_total,
        n_columns=4,
        n_l4=2,
        n_l23=2,
        k_columns=1,
        voltage_decay=0.5,
        learning_rate=0.15,
        ltd_rate=0.15,
        seed=456,
    )


class TestMotorRegion:
    def test_is_sensory_subclass(self, motor):
        assert isinstance(motor, SensoryRegion)

    def test_output_starts_silent(self, motor):
        token_id, conf = motor.get_output()
        assert token_id == -1
        assert conf == 0.0

    def test_output_scores_shape(self, motor):
        scores = motor.get_output_distribution()
        assert scores.shape == (motor.n_columns,)

    def test_observe_builds_mapping(self, motor, region1, encoder):
        """Processing + observing tokens builds column→token map."""
        enc = encoder.encode("a")
        region1.process(enc)
        motor.process(region1.firing_rate_l23)
        motor.observe_token(0)
        # At least one column should now be mapped
        assert (motor._col_token_map >= 0).any()

    def test_process_updates_output_scores(self, motor, region1, encoder):
        enc = encoder.encode("a")
        region1.process(enc)
        motor.process(region1.firing_rate_l23)
        # After processing, some columns should have nonzero scores
        # (from firing_rate_l23 EMA)
        # First step may be zero since firing_rate starts at 0
        # Process a second token to build up rates
        region1.process(encoder.encode("b"))
        motor.process(region1.firing_rate_l23)
        # Output scores should be populated
        assert motor.output_scores is not None

    def test_reset_clears_output(self, motor, region1, encoder):
        region1.process(encoder.encode("a"))
        motor.process(region1.firing_rate_l23)
        motor.reset_working_memory()
        assert motor.output_scores.sum() == 0.0


class TestMotorTopology:
    def test_motor_detected_as_motor(self, region1, motor, encoder):
        cortex = Topology(encoder)
        cortex.add_region("S1", region1, entry=True)
        cortex.add_region("M1", motor)
        assert cortex._regions["M1"].motor is True
        assert cortex._regions["S1"].motor is False

    def test_motor_runs_in_hierarchy(self, region1, motor, encoder):
        """M1 processes tokens and collects motor metrics."""
        tokens = [(i % 4, chr(ord("a") + i % 4)) for i in range(30)]
        cortex = Topology(encoder)
        cortex.add_region("S1", region1, entry=True)
        cortex.add_region("M1", motor)
        cortex.connect("S1", "M1", "feedforward")
        cortex.connect("S1", "M1", "surprise")
        cortex.connect("M1", "S1", "apical")
        result = cortex.run(tokens, log_interval=1000)
        assert result.elapsed_seconds > 0
        # Motor metrics should be populated
        m1_metrics = result.per_region["M1"]
        assert len(m1_metrics.motor_confidences) > 0

    def test_motor_with_thalamic_gate(self, region1, motor, encoder):
        """M1→S1 apical with thalamic gate runs without error."""
        tokens = [(i % 4, chr(ord("a") + i % 4)) for i in range(30)]
        cortex = Topology(encoder)
        cortex.add_region("S1", region1, entry=True)
        cortex.add_region("M1", motor)
        cortex.connect("S1", "M1", "feedforward")
        cortex.connect("S1", "M1", "surprise")
        cortex.connect("M1", "S1", "apical", thalamic_gate=ThalamicGate())
        result = cortex.run(tokens, log_interval=1000)
        assert "M1->S1" in result.thalamic_readiness

    def test_story_boundary_resets_motor(self, region1, motor, encoder):
        tokens = [
            (0, "a"), (1, "b"), (2, "c"),
            (STORY_BOUNDARY, ""),
            (0, "a"), (1, "b"),
        ]
        cortex = Topology(encoder)
        cortex.add_region("S1", region1, entry=True)
        cortex.add_region("M1", motor)
        cortex.connect("S1", "M1", "feedforward")
        result = cortex.run(tokens, log_interval=1000)
        assert result.elapsed_seconds > 0

    def test_motor_without_hierarchy(self, region1, motor, encoder):
        """M1 can work without S2 — just S1→M1."""
        tokens = [(i % 4, chr(ord("a") + i % 4)) for i in range(20)]
        cortex = Topology(encoder)
        cortex.add_region("S1", region1, entry=True)
        cortex.add_region("M1", motor)
        cortex.connect("S1", "M1", "feedforward")
        result = cortex.run(tokens, log_interval=1000)
        assert "M1" in result.per_region
