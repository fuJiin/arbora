import numpy as np
import pytest

from step.cortex import CorticalRegion, SensoryRegion

# ---------------------------------------------------------------------------
# CorticalRegion: initialization
# ---------------------------------------------------------------------------


class TestCorticalRegionInit:
    def test_dimensions(self):
        r = CorticalRegion(n_columns=64, n_l4=8, n_l23=8, k_columns=6)
        assert r.voltage_l4.shape == (512,)
        assert r.voltage_l23.shape == (512,)
        assert r.fb_weights.shape == (512, 512)
        assert r.lateral_weights.shape == (512, 512)
        assert r.l23_lateral_weights.shape == (512, 512)
        assert r.trace_l4.shape == (512,)
        assert r.trace_l23.shape == (512,)

    def test_initial_state_is_zero(self):
        r = CorticalRegion(n_columns=32, n_l4=4, n_l23=4, k_columns=4)
        assert r.voltage_l4.sum() == 0
        assert r.excitability_l4.sum() == 0
        assert not r.active_l4.any()
        assert not r.active_l23.any()
        assert r.trace_l4.sum() == 0


# ---------------------------------------------------------------------------
# CorticalRegion: activation mechanics
# ---------------------------------------------------------------------------


class TestActivation:
    @pytest.fixture()
    def region(self):
        return CorticalRegion(n_columns=16, n_l4=4, n_l23=4, k_columns=3)

    def test_activates_k_columns(self, region):
        drive = np.zeros(16)
        drive[2] = 1.0
        drive[7] = 0.8
        drive[11] = 0.6
        region.step(drive)
        assert region.active_columns.sum() == 3
        assert region.active_columns[2]
        assert region.active_columns[7]
        assert region.active_columns[11]

    def test_burst_on_first_step(self, region):
        """First step has no predictions — all active columns should burst."""
        drive = np.zeros(16)
        drive[0] = 1.0
        drive[1] = 0.8
        drive[2] = 0.6
        region.step(drive)
        for col in np.nonzero(region.active_columns)[0]:
            assert region.bursting_columns[col]
            # All neurons in burst column should be active
            col_neurons = region.active_l4[col * 4: (col + 1) * 4]
            assert col_neurons.all()

    def test_precise_when_predicted(self, region):
        """If a neuron was predicted, its column should activate precisely."""
        # Set up feedback so L2/3 neuron 0 predicts L4 neuron 3
        region.fb_weights[0, 3] = 2.0
        region.active_l23[0] = True  # pretend active last step
        region.fb_boost_threshold = 0.0  # ensure prediction triggers

        drive = np.zeros(16)
        drive[0] = 1.0  # drive col 0
        drive[4] = 0.5  # drive col 1 (to fill k)
        drive[8] = 0.3  # drive col 2
        region.step(drive)

        # Col 0 should be precise (not bursting)
        assert region.active_columns[0]
        assert not region.bursting_columns[0]
        # Only the predicted neuron should be active
        col0_neurons = region.active_l4[0:4]
        assert col0_neurons.sum() == 1
        assert region.active_l4[3]  # the predicted neuron

    def test_excitability_breaks_ties(self, region):
        # All neurons in col 0 get same feedforward voltage,
        # but neuron 2 (global idx 2) has high excitability → wins.
        # Also need prediction on neuron 2 to avoid burst
        region.excitability_l4[2] = 10.0
        # Set up prediction for neuron 2 so col 0 is precise
        region.fb_weights[0, 2] = 2.0
        region.active_l23[0] = True
        region.fb_boost_threshold = 0.0

        drive = np.zeros(16)
        drive[0] = 1.0
        drive[1] = 0.5
        drive[2] = 0.3
        region.step(drive)
        assert region.active_l4[2]

    def test_inactive_columns_have_no_active_neurons(self, region):
        drive = np.zeros(16)
        drive[0] = 1.0
        region.step(drive)
        for col in range(16):
            if not region.active_columns[col]:
                assert not region.active_l4[col * 4: (col + 1) * 4].any()
                assert not region.active_l23[col * 4: (col + 1) * 4].any()


# ---------------------------------------------------------------------------
# CorticalRegion: burst mechanics
# ---------------------------------------------------------------------------


class TestBurst:
    def test_all_neurons_fire_on_burst(self):
        """Burst column has all L4 and L2/3 neurons active."""
        r = CorticalRegion(n_columns=4, n_l4=4, n_l23=4, k_columns=1)
        r.step(np.array([1.0, 0.0, 0.0, 0.0]))
        assert r.bursting_columns[0]
        assert r.active_l4[0:4].all()
        assert r.active_l23[0:4].all()

    def test_burst_rate_decreases_with_learning(self):
        """As feedback weights develop, burst rate should decrease."""
        r = CorticalRegion(
            n_columns=4,
            n_l4=2,
            n_l23=2,
            k_columns=1,
            learning_rate=0.5,
            synapse_decay=1.0,
            fb_boost_threshold=0.0,
        )
        # Alternate two columns so feedback can learn
        early_bursts = 0
        for i in range(20):
            col = i % 2
            drive = np.zeros(4)
            drive[col] = 1.0
            r.step(drive)
            early_bursts += int(r.bursting_columns[col])

        late_bursts = 0
        for i in range(20):
            col = i % 2
            drive = np.zeros(4)
            drive[col] = 1.0
            r.step(drive)
            late_bursts += int(r.bursting_columns[col])

        # Late burst count should be less than or equal to early
        assert late_bursts <= early_bursts

    def test_burst_trace_goes_to_best_match(self):
        """During burst, only the highest-voltage neuron gets the trace."""
        r = CorticalRegion(n_columns=4, n_l4=4, n_l23=4, k_columns=1)
        # Give neuron 2 in col 0 extra voltage so it's the "best match"
        r.voltage_l4[2] = 0.5
        r.step(np.array([1.0, 0.0, 0.0, 0.0]))
        assert r.bursting_columns[0]
        # All neurons active, but only best-match should have trace
        assert r.active_l4[0:4].all()
        # Neuron 2 had highest pre-existing voltage + drive
        assert r.trace_l4[2] == 1.0

    def test_burst_learning_scale(self):
        """Burst columns should produce larger weight updates."""
        r = CorticalRegion(
            n_columns=4,
            n_l4=2,
            n_l23=2,
            k_columns=1,
            learning_rate=0.1,
            synapse_decay=1.0,
            burst_learning_scale=5.0,
        )
        # Step 1: establish trace
        r.step(np.array([1.0, 0.0, 0.0, 0.0]))
        trace_neuron = np.where(r.trace_l4 > 0)[0][0]

        # Step 2: drive different column — will burst (no predictions)
        r.step(np.array([0.0, 1.0, 0.0, 0.0]))
        active_neurons = np.where(r.active_l4)[0]

        # Lateral weight from trace_neuron to active neurons should be
        # scaled by burst_learning_scale
        for n in active_neurons:
            weight = r.lateral_weights[trace_neuron, n]
            # With lr=0.1, scale=5, trace≈0.9(decayed): ~0.45
            assert weight > 0.3


# ---------------------------------------------------------------------------
# CorticalRegion: voltage dynamics
# ---------------------------------------------------------------------------


class TestVoltage:
    def test_active_neuron_voltage_resets(self):
        r = CorticalRegion(n_columns=4, n_l4=2, n_l23=2, k_columns=1)
        drive = np.array([1.0, 0.0, 0.0, 0.0])
        r.step(drive)
        for idx in np.where(r.active_l4)[0]:
            assert r.voltage_l4[idx] == 0.0

    def test_voltage_accumulates_over_steps(self):
        r = CorticalRegion(
            n_columns=4,
            n_l4=2,
            n_l23=2,
            k_columns=1,
            voltage_decay=0.9,
        )
        # Step 1: drive col 0 strongly, col 3 weakly — col 0 wins,
        # col 3 gets voltage but isn't selected.
        drive = np.array([5.0, 0.0, 0.0, 0.3])
        r.step(drive)
        # Step 2: again col 0 wins, col 3 accumulates
        drive2 = np.array([5.0, 0.0, 0.0, 0.3])
        r.step(drive2)
        # Col 3 neurons: decayed + new drive should accumulate
        col3_start = 3 * 2
        assert r.voltage_l4[col3_start: col3_start + 2].max() > 0.5

    def test_voltage_decays_each_step(self):
        r = CorticalRegion(
            n_columns=4,
            n_l4=2,
            n_l23=2,
            k_columns=1,
            voltage_decay=0.5,
        )
        r.voltage_l4[6] = 1.0  # col 3, neuron 0
        drive = np.array([10.0, 0.0, 0.0, 0.0])
        r.step(drive)
        # Col 3 neuron 0: 1.0 * 0.5 (decay) + 0.0 (no drive) = 0.5
        assert r.voltage_l4[6] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# CorticalRegion: excitability
# ---------------------------------------------------------------------------


class TestExcitability:
    def test_grows_for_inactive_neurons(self):
        r = CorticalRegion(n_columns=8, n_l4=2, n_l23=2, k_columns=1)
        drive = np.zeros(8)
        drive[0] = 1.0
        r.step(drive)
        inactive = ~r.active_l4
        assert (r.excitability_l4[inactive] > 0).all()

    def test_resets_on_activation(self):
        r = CorticalRegion(n_columns=4, n_l4=2, n_l23=2, k_columns=1)
        r.excitability_l4[:] = 5.0
        drive = np.array([1.0, 0.0, 0.0, 0.0])
        r.step(drive)
        for idx in np.where(r.active_l4)[0]:
            assert r.excitability_l4[idx] == 0.0

    def test_excitability_ensures_rotation(self):
        """Over many steps, all neurons should eventually activate."""
        r = CorticalRegion(n_columns=4, n_l4=4, n_l23=4, k_columns=1)
        drive = np.array([1.0, 0.0, 0.0, 0.0])
        activated = set()
        for _ in range(20):
            r.step(drive)
            activated.update(np.where(r.active_l4)[0].tolist())
        # All 4 neurons in col 0 should have activated at some point
        col0_neurons = {0, 1, 2, 3}
        assert col0_neurons.issubset(activated)


# ---------------------------------------------------------------------------
# CorticalRegion: eligibility traces
# ---------------------------------------------------------------------------


class TestEligibility:
    def test_set_on_activation(self):
        r = CorticalRegion(n_columns=4, n_l4=2, n_l23=2, k_columns=1)
        drive = np.array([1.0, 0.0, 0.0, 0.0])
        r.step(drive)
        # During burst, only best-match neuron gets trace
        assert r.trace_l4.sum() > 0
        assert r.trace_l23.sum() > 0

    def test_decays_over_time(self):
        r = CorticalRegion(
            n_columns=4,
            n_l4=2,
            n_l23=2,
            k_columns=1,
            eligibility_decay=0.5,
        )
        drive = np.array([1.0, 0.0, 0.0, 0.0])
        r.step(drive)
        traced_neurons = np.where(r.trace_l4 > 0)[0].copy()
        # Step with different column so first neurons are not re-activated
        drive2 = np.array([0.0, 1.0, 0.0, 0.0])
        r.step(drive2)
        for idx in traced_neurons:
            if not r.active_l4[idx]:
                assert r.trace_l4[idx] == pytest.approx(0.5)

    def test_inactive_neurons_have_zero_trace(self):
        r = CorticalRegion(n_columns=4, n_l4=2, n_l23=2, k_columns=1)
        r.step(np.array([1.0, 0.0, 0.0, 0.0]))
        # Neurons that aren't trace winners should have 0 trace
        # (burst: all active but only best-match gets trace)
        trace_count = (r.trace_l4 > 0).sum()
        assert trace_count == 1  # only one trace winner per burst column


# ---------------------------------------------------------------------------
# CorticalRegion: learning
# ---------------------------------------------------------------------------


class TestLearning:
    def test_lateral_weights_strengthen(self):
        """If L4 neuron A fires at t, then neuron B fires at t+1,
        lateral_weights[A, B] should increase."""
        r = CorticalRegion(
            n_columns=4,
            n_l4=2,
            n_l23=2,
            k_columns=1,
            learning_rate=0.1,
            synapse_decay=1.0,
            eligibility_decay=0.9,
        )
        r.step(np.array([1.0, 0.0, 0.0, 0.0]))
        traced = np.where(r.trace_l4 > 0)[0][0]

        r.step(np.array([0.0, 1.0, 0.0, 0.0]))
        active = np.where(r.active_l4)[0]

        # lateral_weights[traced, active] should be > 0
        for n in active:
            assert r.lateral_weights[traced, n] > 0

    def test_fb_weights_strengthen(self):
        """If L2/3 neuron fires at t, then L4 neuron fires at t+1,
        fb_weights[L2/3, L4] should increase."""
        r = CorticalRegion(
            n_columns=4,
            n_l4=2,
            n_l23=2,
            k_columns=1,
            learning_rate=0.1,
            synapse_decay=1.0,
            eligibility_decay=0.9,
        )
        r.step(np.array([1.0, 0.0, 0.0, 0.0]))
        l23_traced = np.where(r.trace_l23 > 0)[0][0]
        r.step(np.array([0.0, 1.0, 0.0, 0.0]))
        l4_active = np.where(r.active_l4)[0]
        for n in l4_active:
            assert r.fb_weights[l23_traced, n] > 0

    def test_no_learning_without_trace(self):
        """Synapses don't strengthen if source has no eligibility."""
        r = CorticalRegion(
            n_columns=4,
            n_l4=2,
            n_l23=2,
            k_columns=1,
            learning_rate=0.1,
            synapse_decay=1.0,
        )
        r.step(np.array([1.0, 0.0, 0.0, 0.0]))
        assert r.lateral_weights.sum() == 0.0

    def test_weights_clipped_to_unit(self):
        """Weights stay in [0, 1]."""
        r = CorticalRegion(
            n_columns=4,
            n_l4=2,
            n_l23=2,
            k_columns=1,
            learning_rate=1.0,
            synapse_decay=1.0,
        )
        for _ in range(100):
            drive = np.zeros(4)
            drive[0] = 1.0
            r.step(drive)
        assert r.lateral_weights.max() <= 1.0
        assert r.lateral_weights.min() >= 0.0


# ---------------------------------------------------------------------------
# CorticalRegion: feedback influences activation
# ---------------------------------------------------------------------------


class TestFeedback:
    def test_feedback_biases_neuron_selection(self):
        """Strong feedback to a specific L4 neuron should make it win
        and produce a precise (non-burst) activation."""
        r = CorticalRegion(
            n_columns=4,
            n_l4=4,
            n_l23=4,
            k_columns=1,
            fb_boost_threshold=0.0,
        )
        target = 3  # col 0, neuron 3
        r.fb_weights[0, target] = 2.0
        r.active_l23[0] = True

        drive = np.array([1.0, 0.0, 0.0, 0.0])
        r.step(drive)
        assert r.active_l4[target]
        assert not r.bursting_columns[0]  # should be precise


# ---------------------------------------------------------------------------
# CorticalRegion: L2/3 lateral connections
# ---------------------------------------------------------------------------


class TestL23Lateral:
    def test_l23_burst_on_first_step(self):
        """First step: L2/3 also bursts since L4 bursts."""
        r = CorticalRegion(n_columns=4, n_l4=4, n_l23=4, k_columns=1)
        r.step(np.array([1.0, 0.0, 0.0, 0.0]))
        assert r.bursting_columns[0]
        # All L2/3 neurons in burst column should be active
        assert r.active_l23[0:4].all()

    def test_lateral_overrides_l4_match(self):
        """Strong L2/3 lateral input can make a different L2/3
        neuron win than the L4-matching one."""
        r = CorticalRegion(
            n_columns=4, n_l4=4, n_l23=4, k_columns=1,
            fb_boost_threshold=0.0,
        )
        # Set up prediction so col 0 is precise
        r.fb_weights[0, 0] = 2.0
        r.active_l23[0] = True

        # Set up L2/3 lateral so neuron 3 in col 0 gets
        # strong input from a previously active L2/3 neuron
        source_l23 = 7  # col 1, neuron 3
        target_l23 = 3  # col 0, neuron 3
        r.l23_lateral_weights[source_l23, target_l23] = 5.0
        r.active_l23[source_l23] = True

        r.step(np.array([1.0, 0.0, 0.0, 0.0]))
        assert r.active_l4[0]  # L4 picks predicted neuron 0
        assert r.active_l23[target_l23]  # L2/3 picks neuron 3

    def test_l23_lateral_weights_learn(self):
        """Co-active L2/3 neurons strengthen lateral connections."""
        r = CorticalRegion(
            n_columns=4,
            n_l4=2,
            n_l23=2,
            k_columns=1,
            learning_rate=0.1,
            synapse_decay=1.0,
            eligibility_decay=0.9,
        )
        r.step(np.array([1.0, 0.0, 0.0, 0.0]))
        l23_traced = np.where(r.trace_l23 > 0)[0][0]
        r.step(np.array([0.0, 1.0, 0.0, 0.0]))
        l23_active = np.where(r.active_l23)[0]
        for n in l23_active:
            assert r.l23_lateral_weights[l23_traced, n] > 0

    def test_l23_voltage_resets_on_activation(self):
        """Active L2/3 neurons have voltage reset after step."""
        r = CorticalRegion(n_columns=4, n_l4=2, n_l23=2, k_columns=1)
        r.step(np.array([1.0, 0.0, 0.0, 0.0]))
        for idx in np.where(r.active_l23)[0]:
            assert r.voltage_l23[idx] == 0.0


# ---------------------------------------------------------------------------
# SensoryRegion
# ---------------------------------------------------------------------------


class TestSensoryRegion:
    def test_ff_weights_shape(self):
        s = SensoryRegion(
            input_dim=100,
            n_columns=32,
            n_l4=4,
            n_l23=4,
            k_columns=4,
        )
        assert s.ff_weights.shape == (100, 32)

    def test_process_activates_correct_column_count(self):
        s = SensoryRegion(
            input_dim=10,
            n_columns=8,
            n_l4=2,
            n_l23=2,
            k_columns=2,
            seed=42,
        )
        encoding = np.zeros(10)
        encoding[0] = 1.0
        encoding[5] = 1.0
        s.process(encoding)
        assert s.active_columns.sum() == 2

    def test_different_inputs_different_columns(self):
        s1 = SensoryRegion(
            input_dim=10,
            n_columns=8,
            n_l4=2,
            n_l23=2,
            k_columns=2,
            seed=42,
        )
        enc1 = np.zeros(10)
        enc1[0] = 1.0
        s1.process(enc1)
        cols1 = s1.active_columns.copy()

        s2 = SensoryRegion(
            input_dim=10,
            n_columns=8,
            n_l4=2,
            n_l23=2,
            k_columns=2,
            seed=42,
        )
        enc2 = np.zeros(10)
        enc2[9] = 1.0
        s2.process(enc2)
        cols2 = s2.active_columns.copy()

        assert not np.array_equal(cols1, cols2)

    def test_process_accepts_2d_encoding(self):
        """process() flattens multi-dimensional input."""
        s = SensoryRegion(
            input_dim=20,
            n_columns=8,
            n_l4=2,
            n_l23=2,
            k_columns=2,
            seed=42,
        )
        encoding_2d = np.zeros((4, 5))
        encoding_2d[0, 0] = 1.0
        s.process(encoding_2d)
        assert s.active_columns.sum() == 2

    def test_charbit_integration(self):
        """End-to-end: CharbitEncoder → SensoryRegion."""
        from step.encoders import CharbitEncoder

        enc = CharbitEncoder(length=4, width=5, chars="abcd")
        encoding = enc.encode("abc")

        s = SensoryRegion(
            input_dim=4 * 5,
            n_columns=16,
            n_l4=4,
            n_l23=4,
            k_columns=3,
            seed=42,
        )
        s.process(encoding)
        assert s.active_columns.sum() == 3
