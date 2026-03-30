import numpy as np
import pytest

from arbor.cortex.circuit import Circuit, ConnectionRole
from arbor.cortex.modulators import SurpriseTracker, ThalamicGate
from arbor.cortex.motor import MotorRegion
from arbor.cortex.sensory import SensoryRegion
from arbor.encoders.charbit import CharbitEncoder
from arbor.probes.modulators import ModulatorProbe
from examples.chat.data import STORY_BOUNDARY
from examples.chat.probes import ChatMotorProbe
from tests.conftest import run_circuit


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
    def test_is_cortical_subclass(self, motor):
        from arbor.cortex.region import CorticalRegion

        assert isinstance(motor, CorticalRegion)

    def test_output_starts_silent(self, motor):
        token_id, conf = motor.get_output()
        assert token_id == -1
        assert conf == 0.0

    def test_output_scores_shape(self, motor):
        scores = motor.get_output_distribution()
        assert scores.shape == (motor.n_columns,)

    def test_observe_strengthens_output_weights(self, motor, region1, encoder):
        """Processing + observing tokens strengthens L5 output weights."""
        enc = encoder.encode("a")
        region1.process(enc)
        motor.process(region1.l23.firing_rate)
        weights_before = motor.output_weights[:, ord("a")].copy()
        motor.observe_token(ord("a"))
        weights_after = motor.output_weights[:, ord("a")]
        # Active L2/3 neurons should have increased weights to this token
        assert (weights_after >= weights_before).all()

    def test_process_updates_output_scores(self, motor, region1, encoder):
        enc = encoder.encode("a")
        region1.process(enc)
        motor.process(region1.l23.firing_rate)
        # After processing, some columns should have nonzero scores
        # (from firing_rate_l23 EMA)
        # First step may be zero since firing_rate starts at 0
        # Process a second token to build up rates
        region1.process(encoder.encode("b"))
        motor.process(region1.l23.firing_rate)
        # Output scores should be populated
        assert motor.output_scores is not None

    def test_reset_clears_output(self, motor, region1, encoder):
        region1.process(encoder.encode("a"))
        motor.process(region1.l23.firing_rate)
        motor.reset_working_memory()
        assert motor.output_scores.sum() == 0.0


class TestMotorCircuit:
    def test_motor_detected_as_motor(self, region1, motor, encoder):
        cortex = Circuit(encoder)
        cortex.add_region("S1", region1, entry=True)
        cortex.add_region("M1", motor)
        assert cortex._regions["M1"].motor is True
        assert cortex._regions["S1"].motor is False

    def test_motor_runs_in_hierarchy(self, region1, motor, encoder):
        """M1 processes tokens and collects motor metrics."""
        tokens = [(i % 4, chr(ord("a") + i % 4)) for i in range(30)]
        cortex = Circuit(encoder)
        cortex.add_region("S1", region1, entry=True)
        cortex.add_region("M1", motor)
        cortex.connect(
            region1.l23,
            motor.l4,
            ConnectionRole.FEEDFORWARD,
            surprise_tracker=SurpriseTracker(),
        )
        cortex.connect(motor.l23, region1.l4, ConnectionRole.APICAL)
        probe = ChatMotorProbe()
        result = run_circuit(cortex, tokens, probes=[probe])
        assert result.elapsed_seconds > 0
        # Motor metrics should be populated
        snap = result.probe_snapshots["motor"]
        assert len(snap["M1"].motor_confidences) > 0

    def test_motor_with_thalamic_gate(self, region1, motor, encoder):
        """M1→S1 apical with thalamic gate runs without error."""
        tokens = [(i % 4, chr(ord("a") + i % 4)) for i in range(30)]
        cortex = Circuit(encoder)
        cortex.add_region("S1", region1, entry=True)
        cortex.add_region("M1", motor)
        cortex.connect(
            region1.l23,
            motor.l4,
            ConnectionRole.FEEDFORWARD,
            surprise_tracker=SurpriseTracker(),
        )
        cortex.connect(
            motor.l23, region1.l4, ConnectionRole.APICAL, thalamic_gate=ThalamicGate()
        )
        mod_probe = ModulatorProbe()
        result = run_circuit(cortex, tokens, probes=[mod_probe])
        mod_snap = result.probe_snapshots["modulators"]
        assert "M1->S1" in mod_snap.thalamic

    def test_story_boundary_resets_motor(self, region1, motor, encoder):
        tokens = [
            (0, "a"),
            (1, "b"),
            (2, "c"),
            (STORY_BOUNDARY, ""),
            (0, "a"),
            (1, "b"),
        ]
        cortex = Circuit(encoder)
        cortex.add_region("S1", region1, entry=True)
        cortex.add_region("M1", motor)
        cortex.connect(region1.l23, motor.l4, ConnectionRole.FEEDFORWARD)
        result = run_circuit(cortex, tokens)
        assert result.elapsed_seconds > 0

    def test_motor_without_hierarchy(self, region1, motor, encoder):
        """M1 can work without S2 — just S1→M1."""
        tokens = [(i % 4, chr(ord("a") + i % 4)) for i in range(20)]
        cortex = Circuit(encoder)
        cortex.add_region("S1", region1, entry=True)
        cortex.add_region("M1", motor)
        cortex.connect(region1.l23, motor.l4, ConnectionRole.FEEDFORWARD)
        probe = ChatMotorProbe()
        result = run_circuit(cortex, tokens, probes=[probe])
        snap = result.probe_snapshots["motor"]
        assert "M1" in snap


class TestExploreDirect:
    """Tests for _explore_direct() — random column forcing for motor exploration."""

    def test_explore_activates_k_columns(self, motor, region1, encoder):
        """_explore_direct should activate exactly k_columns columns."""
        enc = encoder.encode("a")
        region1.process(enc)
        motor._explore_direct(region1.l23.firing_rate)
        active_cols = np.nonzero(motor.active_columns)[0]
        assert len(active_cols) == motor.k_columns

    def test_explore_returns_active_l4_indices(self, motor, region1, encoder):
        """_explore_direct should return nonzero array of active L4 indices."""
        enc = encoder.encode("a")
        region1.process(enc)
        active = motor._explore_direct(region1.l23.firing_rate)
        assert len(active) > 0
        # Active indices should be valid L4 neuron indices
        assert active.max() < motor.n_l4_total

    def test_explore_trains_ff_weights(self, motor, region1, encoder):
        """_explore_direct should populate ff eligibility trace for learning."""
        enc = encoder.encode("a")
        region1.process(enc)
        assert motor._ff_eligibility.sum() == 0.0
        motor._explore_direct(region1.l23.firing_rate)
        # Eligibility trace should be nonzero after babbling with input
        assert motor._ff_eligibility.sum() != 0.0

    def test_explore_diverse_across_steps(self, motor, region1, encoder):
        """Repeated babbling should explore different columns over time."""
        enc = encoder.encode("a")
        region1.process(enc)
        activated_sets = set()
        for _ in range(20):
            motor._explore_direct(region1.l23.firing_rate)
            cols = tuple(sorted(np.nonzero(motor.active_columns)[0]))
            activated_sets.add(cols)
        # With 4 columns and k=1, should see multiple different selections
        assert len(activated_sets) > 1

    def test_process_uses_explore_when_noise_full(self, motor, region1, encoder):
        """process() with exploration_noise=1.0 always uses _explore_direct path."""
        motor.exploration_noise = 1.0
        enc = encoder.encode("a")
        region1.process(enc)
        # Run multiple steps — all should activate exactly k columns
        for _ in range(10):
            motor.process(region1.l23.firing_rate)
            active_cols = np.nonzero(motor.active_columns)[0]
            assert len(active_cols) == motor.k_columns

    def test_process_uses_explore_probabilistically(self, motor, region1, encoder):
        """process() with 0 < exploration_noise < 1 mixes babble and normal."""
        motor.exploration_noise = 0.5
        enc = encoder.encode("b")
        region1.process(enc)
        # Run enough steps that both paths should be taken
        for _ in range(20):
            motor.process(region1.l23.firing_rate)
            active_cols = np.nonzero(motor.active_columns)[0]
            # Both paths produce valid activations
            assert len(active_cols) == motor.k_columns

    def test_explore_without_encoding(self, motor):
        """_explore_direct with None encoding should still activate columns."""
        active = motor._explore_direct(None)
        assert len(active) > 0
        active_cols = np.nonzero(motor.active_columns)[0]
        assert len(active_cols) == motor.k_columns

    def test_explore_without_encoding_skips_ff_learning(self, motor):
        """_explore_direct with None encoding should not update ff eligibility."""
        assert motor._ff_eligibility.sum() == 0.0
        motor._explore_direct(None)
        # ff_weights learning requires encoding input
        assert motor._ff_eligibility.sum() == 0.0
