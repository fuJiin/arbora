import numpy as np
import pytest

from step.cortex import CorticalRegion, SensoryRegion

# ---------------------------------------------------------------------------
# CorticalRegion: initialization
# ---------------------------------------------------------------------------


class TestCorticalRegionInit:
    def test_dimensions(self):
        r = CorticalRegion(
            n_columns=64, n_l4=8, n_l23=8, k_columns=6
        )
        assert r.voltage_l4.shape == (512,)
        assert r.voltage_l23.shape == (512,)
        assert r.fb_weights.shape == (512, 512)
        assert r.lateral_weights.shape == (512, 512)
        assert r.l23_lateral_weights.shape == (512, 512)
        assert r.trace_l4.shape == (512,)
        assert r.trace_l23.shape == (512,)

    def test_initial_state_is_zero(self):
        r = CorticalRegion(
            n_columns=32, n_l4=4, n_l23=4, k_columns=4
        )
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
        return CorticalRegion(
            n_columns=16, n_l4=4, n_l23=4, k_columns=3
        )

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

    def test_one_neuron_per_active_column(self, region):
        drive = np.zeros(16)
        drive[0] = 1.0
        drive[1] = 0.8
        drive[2] = 0.6
        region.step(drive)
        for col in range(16):
            col_neurons = region.active_l4[col * 4 : (col + 1) * 4]
            if region.active_columns[col]:
                assert col_neurons.sum() == 1
            else:
                assert col_neurons.sum() == 0

    def test_excitability_breaks_ties(self, region):
        # All neurons in col 0 get same feedforward voltage,
        # but neuron 2 (global idx 2) has high excitability → wins.
        region.excitability_l4[2] = 10.0
        drive = np.zeros(16)
        drive[0] = 1.0
        drive[1] = 0.5
        drive[2] = 0.3
        region.step(drive)
        assert region.active_l4[2]

    def test_l23_mirrors_l4_with_equal_layer_sizes(self, region):
        drive = np.zeros(16)
        drive[0] = 1.0
        drive[5] = 0.8
        drive[10] = 0.6
        region.step(drive)
        for col in range(16):
            for i in range(4):
                l4 = region.active_l4[col * 4 + i]
                l23 = region.active_l23[col * 4 + i]
                assert l4 == l23

    def test_inactive_columns_have_no_active_neurons(self, region):
        drive = np.zeros(16)
        drive[0] = 1.0
        region.step(drive)
        for col in range(16):
            if not region.active_columns[col]:
                assert not region.active_l4[
                    col * 4 : (col + 1) * 4
                ].any()
                assert not region.active_l23[
                    col * 4 : (col + 1) * 4
                ].any()


# ---------------------------------------------------------------------------
# CorticalRegion: voltage dynamics
# ---------------------------------------------------------------------------


class TestVoltage:
    def test_active_neuron_voltage_resets(self):
        r = CorticalRegion(
            n_columns=4, n_l4=2, n_l23=2, k_columns=1
        )
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
        # Drive column 3 weakly twice — voltage should accumulate
        drive = np.array([0.0, 0.0, 0.0, 0.3])
        r.step(drive)
        # Column 3 neurons got 0.3 voltage, but col 3 may or may not
        # have been selected. Let's check a non-selected neuron's voltage.
        # Drive column 0 strongly so column 3 is NOT selected.
        drive2 = np.array([5.0, 0.0, 0.0, 0.3])
        r.step(drive2)
        # Col 3 neurons: decayed (0.3*0.9) + new drive (0.3) = 0.57
        col3_start = 3 * 2
        # At least one neuron in col 3 should have accumulated voltage
        assert r.voltage_l4[col3_start:col3_start + 2].max() > 0.5

    def test_voltage_decays_each_step(self):
        r = CorticalRegion(
            n_columns=4,
            n_l4=2,
            n_l23=2,
            k_columns=1,
            voltage_decay=0.5,
        )
        # Manually set voltage, then step with zero drive
        r.voltage_l4[6] = 1.0  # col 3, neuron 0
        # Drive col 0 so col 3 is not selected (no reset)
        drive = np.array([10.0, 0.0, 0.0, 0.0])
        r.step(drive)
        # Col 3 neuron 0: 1.0 * 0.5 (decay) + 0.0 (no drive) = 0.5
        assert r.voltage_l4[6] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# CorticalRegion: excitability
# ---------------------------------------------------------------------------


class TestExcitability:
    def test_grows_for_inactive_neurons(self):
        r = CorticalRegion(
            n_columns=8, n_l4=2, n_l23=2, k_columns=1
        )
        drive = np.zeros(8)
        drive[0] = 1.0
        r.step(drive)
        inactive = ~r.active_l4
        assert (r.excitability_l4[inactive] > 0).all()

    def test_resets_on_activation(self):
        r = CorticalRegion(
            n_columns=4, n_l4=2, n_l23=2, k_columns=1
        )
        r.excitability_l4[:] = 5.0
        drive = np.array([1.0, 0.0, 0.0, 0.0])
        r.step(drive)
        for idx in np.where(r.active_l4)[0]:
            assert r.excitability_l4[idx] == 0.0

    def test_excitability_ensures_rotation(self):
        """Over many steps, all neurons should eventually activate."""
        r = CorticalRegion(
            n_columns=4, n_l4=4, n_l23=4, k_columns=1
        )
        # Same drive every step — excitability should cause
        # different neurons to win within the active column.
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
        r = CorticalRegion(
            n_columns=4, n_l4=2, n_l23=2, k_columns=1
        )
        drive = np.array([1.0, 0.0, 0.0, 0.0])
        r.step(drive)
        for idx in np.where(r.active_l4)[0]:
            assert r.trace_l4[idx] == 1.0
        for idx in np.where(r.active_l23)[0]:
            assert r.trace_l23[idx] == 1.0

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
        first_active = np.where(r.active_l4)[0].copy()
        # Step with different column so first neurons are not re-activated
        drive2 = np.array([0.0, 1.0, 0.0, 0.0])
        r.step(drive2)
        for idx in first_active:
            if not r.active_l4[idx]:
                assert r.trace_l4[idx] == pytest.approx(0.5)

    def test_inactive_neurons_have_zero_trace(self):
        r = CorticalRegion(
            n_columns=4, n_l4=2, n_l23=2, k_columns=1
        )
        r.step(np.array([1.0, 0.0, 0.0, 0.0]))
        inactive = ~r.active_l4
        assert (r.trace_l4[inactive] == 0.0).all()


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
            synapse_decay=1.0,  # no decay for clarity
            eligibility_decay=0.9,
        )
        # Step 1: drive col 0
        r.step(np.array([1.0, 0.0, 0.0, 0.0]))
        neuron_a = np.where(r.active_l4)[0][0]

        # Step 2: drive col 1
        r.step(np.array([0.0, 1.0, 0.0, 0.0]))
        neuron_b = np.where(r.active_l4)[0][0]

        # lateral_weights[neuron_a, neuron_b] should be > 0
        assert r.lateral_weights[neuron_a, neuron_b] > 0

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
        l23_neuron = np.where(r.active_l23)[0][0]
        r.step(np.array([0.0, 1.0, 0.0, 0.0]))
        l4_neuron = np.where(r.active_l4)[0][0]
        assert r.fb_weights[l23_neuron, l4_neuron] > 0

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
        # First step ever — no prior trace exists
        r.step(np.array([1.0, 0.0, 0.0, 0.0]))
        assert r.lateral_weights.sum() == 0.0

    def test_weights_clipped_to_unit(self):
        """Weights stay in [0, 1]."""
        r = CorticalRegion(
            n_columns=4,
            n_l4=2,
            n_l23=2,
            k_columns=1,
            learning_rate=1.0,  # aggressive
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
        """Strong feedback to a specific L4 neuron should make it win."""
        r = CorticalRegion(
            n_columns=4,
            n_l4=4,
            n_l23=4,
            k_columns=1,
            fb_boost_threshold=0.0,  # low threshold so feedback always applies
        )
        # Manually set fb_weights so L2/3 neuron 0 → L4 neuron 3
        # (col 0, idx 3) gets a strong boost
        target = 3  # col 0, neuron 3
        r.fb_weights[0, target] = 2.0
        # Pretend L2/3 neuron 0 was active last step
        r.active_l23[0] = True

        drive = np.array([1.0, 0.0, 0.0, 0.0])
        r.step(drive)
        assert r.active_l4[target]


# ---------------------------------------------------------------------------
# CorticalRegion: L2/3 lateral connections
# ---------------------------------------------------------------------------


class TestL23Lateral:
    def test_l23_defaults_to_l4_match_without_context(self):
        """Without lateral context, L2/3 winner matches L4 winner."""
        r = CorticalRegion(
            n_columns=4, n_l4=4, n_l23=4, k_columns=1
        )
        r.excitability_l4[2] = 10.0  # force L4 neuron 2 to win
        r.step(np.array([1.0, 0.0, 0.0, 0.0]))
        assert r.active_l4[2]
        assert r.active_l23[2]  # L2/3 mirrors L4

    def test_lateral_overrides_l4_match(self):
        """Strong L2/3 lateral input can make a different L2/3
        neuron win than the L4-matching one."""
        r = CorticalRegion(
            n_columns=4, n_l4=4, n_l23=4, k_columns=1
        )
        # L4 neuron 0 (col 0, idx 0) will win via feedforward
        # But set up L2/3 lateral so neuron 3 in col 0 gets
        # strong input from a previously active L2/3 neuron
        source_l23 = 7  # col 1, neuron 3
        target_l23 = 3  # col 0, neuron 3
        r.l23_lateral_weights[source_l23, target_l23] = 5.0
        r.active_l23[source_l23] = True  # pretend active last step

        r.step(np.array([1.0, 0.0, 0.0, 0.0]))
        assert r.active_l4[0]  # L4 still picks neuron 0
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
        l23_a = np.where(r.active_l23)[0][0]
        r.step(np.array([0.0, 1.0, 0.0, 0.0]))
        l23_b = np.where(r.active_l23)[0][0]
        assert r.l23_lateral_weights[l23_a, l23_b] > 0

    def test_l23_one_winner_per_column(self):
        """Exactly one L2/3 neuron per active column."""
        r = CorticalRegion(
            n_columns=8, n_l4=4, n_l23=4, k_columns=3
        )
        r.step(np.ones(8) * 0.5)
        for col in range(8):
            col_l23 = r.active_l23[col * 4 : (col + 1) * 4]
            if r.active_columns[col]:
                assert col_l23.sum() == 1
            else:
                assert col_l23.sum() == 0

    def test_l23_voltage_resets_on_activation(self):
        """Active L2/3 neurons have voltage reset after step."""
        r = CorticalRegion(
            n_columns=4, n_l4=2, n_l23=2, k_columns=1
        )
        r.step(np.array([1.0, 0.0, 0.0, 0.0]))
        for idx in np.where(r.active_l23)[0]:
            assert r.voltage_l23[idx] == 0.0

    def test_l23_excitability_rotation(self):
        """L2/3 neurons rotate via excitability when L4 matching
        bias is absent (n_l4 > n_l23, so no match bonus)."""
        r = CorticalRegion(
            n_columns=4, n_l4=8, n_l23=4, k_columns=1
        )
        # L4 winner will be idx >= 4 (beyond n_l23), so no L2/3
        # neuron gets the matching bonus. All L2/3 neurons in the
        # active column get equal base drive (0.5).
        # Excitability should rotate them.
        # Bias L4 neuron 7 (idx > n_l23) to always win in col 0
        r.excitability_l4[7] = 100.0
        drive = np.array([1.0, 0.0, 0.0, 0.0])
        activated_l23 = set()
        for _ in range(20):
            r.step(drive)
            activated_l23.update(np.where(r.active_l23)[0].tolist())
        # All 4 L2/3 neurons in col 0 should have activated
        assert {0, 1, 2, 3}.issubset(activated_l23)


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

    def test_process_activates_correct_count(self):
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
        assert s.active_l4.sum() == 2
        assert s.active_l23.sum() == 2

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
        assert s.active_l4.sum() == 2

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
        assert s.active_l4.sum() == 3
        assert s.active_l23.sum() == 3
