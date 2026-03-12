import numpy as np
import pytest

from step.cortex import CorticalRegion, SensoryRegion


def col_drive(col_values: list[float], n_l4: int) -> np.ndarray:
    """Convert column-level drive to per-neuron drive by repeating."""
    return np.repeat(np.array(col_values), n_l4)


# ---------------------------------------------------------------------------
# CorticalRegion: initialization
# ---------------------------------------------------------------------------


class TestCorticalRegionInit:
    def test_dimensions(self):
        r = CorticalRegion(n_columns=64, n_l4=8, n_l23=8, k_columns=6)
        assert r.voltage_l4.shape == (512,)
        assert r.voltage_l23.shape == (512,)
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
        drive = np.zeros(64)  # 16 cols x 4 neurons
        # Drive specific neurons in cols 2, 7, 11
        drive[2 * 4] = 1.0
        drive[7 * 4] = 0.8
        drive[11 * 4] = 0.6
        region.step(drive)
        assert region.active_columns.sum() == 3
        assert region.active_columns[2]
        assert region.active_columns[7]
        assert region.active_columns[11]

    def test_burst_on_first_step(self, region):
        """First step has no predictions — all active columns should burst."""
        drive = col_drive([1.0, 0.8, 0.6] + [0.0] * 13, 4)
        region.step(drive)
        for col in np.nonzero(region.active_columns)[0]:
            assert region.bursting_columns[col]
            col_neurons = region.active_l4[col * 4 : (col + 1) * 4]
            assert col_neurons.all()

    def test_precise_when_predicted(self, region):
        """If a neuron was predicted, its column should activate precisely."""
        region.predict_neuron(3, 0, segment_type="fb")
        region.active_l23[0] = True

        drive = col_drive([1.0, 0.5, 0.3] + [0.0] * 13, 4)
        region.step(drive)

        assert region.active_columns[0]
        assert not region.bursting_columns[0]
        col0_neurons = region.active_l4[0:4]
        assert col0_neurons.sum() == 1
        assert region.active_l4[3]

    def test_excitability_breaks_ties(self, region):
        region.excitability_l4[2] = 10.0
        region.predict_neuron(2, 0, segment_type="fb")
        region.active_l23[0] = True

        drive = col_drive([1.0, 0.5, 0.3] + [0.0] * 13, 4)
        region.step(drive)
        assert region.active_l4[2]

    def test_inactive_columns_have_no_active_neurons(self, region):
        drive = col_drive([1.0] + [0.0] * 15, 4)
        region.step(drive)
        for col in range(16):
            if not region.active_columns[col]:
                assert not region.active_l4[col * 4 : (col + 1) * 4].any()
                assert not region.active_l23[col * 4 : (col + 1) * 4].any()


# ---------------------------------------------------------------------------
# CorticalRegion: burst mechanics
# ---------------------------------------------------------------------------


class TestBurst:
    def test_all_neurons_fire_on_burst(self):
        """Burst column has all L4 and L2/3 neurons active."""
        r = CorticalRegion(n_columns=4, n_l4=4, n_l23=4, k_columns=1)
        r.step(col_drive([1.0, 0.0, 0.0, 0.0], 4))
        assert r.bursting_columns[0]
        assert r.active_l4[0:4].all()
        assert r.active_l23[0:4].all()

    def test_burst_rate_decreases_with_learning(self):
        """As segments develop, burst rate should decrease."""
        r = CorticalRegion(
            n_columns=4,
            n_l4=2,
            n_l23=2,
            k_columns=1,
            learning_rate=0.5,
            synapse_decay=1.0,
            fb_boost_threshold=0.0,
        )
        early_bursts = 0
        for i in range(20):
            col = i % 2
            drive = [0.0] * 4
            drive[col] = 1.0
            r.step(col_drive(drive, 2))
            early_bursts += int(r.bursting_columns[col])

        late_bursts = 0
        for i in range(20):
            col = i % 2
            drive = [0.0] * 4
            drive[col] = 1.0
            r.step(col_drive(drive, 2))
            late_bursts += int(r.bursting_columns[col])

        assert late_bursts <= early_bursts

    def test_burst_trace_goes_to_best_match(self):
        """During burst, only the highest-voltage neuron gets the trace."""
        r = CorticalRegion(n_columns=4, n_l4=4, n_l23=4, k_columns=1)
        r.voltage_l4[2] = 0.5
        r.step(col_drive([1.0, 0.0, 0.0, 0.0], 4))
        assert r.bursting_columns[0]
        assert r.active_l4[0:4].all()
        assert r.trace_l4[2] == 1.0

    def test_burst_learning_scale_l23(self):
        """Burst columns should produce larger L2/3 lateral weight updates."""
        r = CorticalRegion(
            n_columns=4,
            n_l4=2,
            n_l23=2,
            k_columns=1,
            learning_rate=0.1,
            synapse_decay=1.0,
            burst_learning_scale=5.0,
        )
        r.step(col_drive([1.0, 0.0, 0.0, 0.0], 2))
        l23_traced = np.where(r.trace_l23 > 0)[0][0]

        r.step(col_drive([0.0, 1.0, 0.0, 0.0], 2))
        l23_active = np.where(r.active_l23)[0]

        for n in l23_active:
            weight = r.l23_lateral_weights[l23_traced, n]
            assert weight > 0.3


# ---------------------------------------------------------------------------
# CorticalRegion: voltage dynamics
# ---------------------------------------------------------------------------


class TestVoltage:
    def test_active_neuron_voltage_resets(self):
        r = CorticalRegion(n_columns=4, n_l4=2, n_l23=2, k_columns=1)
        r.step(col_drive([1.0, 0.0, 0.0, 0.0], 2))
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
        r.step(col_drive([5.0, 0.0, 0.0, 0.3], 2))
        r.step(col_drive([5.0, 0.0, 0.0, 0.3], 2))
        col3_start = 3 * 2
        assert r.voltage_l4[col3_start : col3_start + 2].max() > 0.5

    def test_voltage_decays_each_step(self):
        r = CorticalRegion(
            n_columns=4,
            n_l4=2,
            n_l23=2,
            k_columns=1,
            voltage_decay=0.5,
        )
        r.voltage_l4[6] = 1.0  # col 3, neuron 0
        r.step(col_drive([10.0, 0.0, 0.0, 0.0], 2))
        # Col 3 neuron 0: 1.0 * 0.5 (decay) + 0.0 (no drive) = 0.5
        assert r.voltage_l4[6] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# CorticalRegion: excitability
# ---------------------------------------------------------------------------


class TestExcitability:
    def test_grows_for_inactive_neurons(self):
        r = CorticalRegion(n_columns=8, n_l4=2, n_l23=2, k_columns=1)
        r.step(col_drive([1.0] + [0.0] * 7, 2))
        inactive = ~r.active_l4
        assert (r.excitability_l4[inactive] > 0).all()

    def test_resets_on_activation(self):
        r = CorticalRegion(n_columns=4, n_l4=2, n_l23=2, k_columns=1)
        r.excitability_l4[:] = 5.0
        r.step(col_drive([1.0, 0.0, 0.0, 0.0], 2))
        for idx in np.where(r.active_l4)[0]:
            assert r.excitability_l4[idx] == 0.0

    def test_excitability_ensures_rotation(self):
        """Over many steps, all neurons should eventually activate."""
        r = CorticalRegion(n_columns=4, n_l4=4, n_l23=4, k_columns=1)
        drive = col_drive([1.0, 0.0, 0.0, 0.0], 4)
        activated = set()
        for _ in range(20):
            r.step(drive)
            activated.update(np.where(r.active_l4)[0].tolist())
        col0_neurons = {0, 1, 2, 3}
        assert col0_neurons.issubset(activated)


# ---------------------------------------------------------------------------
# CorticalRegion: eligibility traces
# ---------------------------------------------------------------------------


class TestEligibility:
    def test_set_on_activation(self):
        r = CorticalRegion(n_columns=4, n_l4=2, n_l23=2, k_columns=1)
        r.step(col_drive([1.0, 0.0, 0.0, 0.0], 2))
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
        r.step(col_drive([1.0, 0.0, 0.0, 0.0], 2))
        traced_neurons = np.where(r.trace_l4 > 0)[0].copy()
        r.step(col_drive([0.0, 1.0, 0.0, 0.0], 2))
        for idx in traced_neurons:
            if not r.active_l4[idx]:
                assert r.trace_l4[idx] == pytest.approx(0.5)

    def test_inactive_neurons_have_zero_trace(self):
        r = CorticalRegion(n_columns=4, n_l4=2, n_l23=2, k_columns=1)
        r.step(col_drive([1.0, 0.0, 0.0, 0.0], 2))
        trace_count = (r.trace_l4 > 0).sum()
        assert trace_count == 1


# ---------------------------------------------------------------------------
# CorticalRegion: learning
# ---------------------------------------------------------------------------


class TestLearning:
    def test_l23_lateral_weights_strengthen_on_sequence(self):
        """If L2/3 neuron A fires at t, then neuron B fires at t+1,
        l23_lateral_weights[A, B] should increase."""
        r = CorticalRegion(
            n_columns=4,
            n_l4=2,
            n_l23=2,
            k_columns=1,
            learning_rate=0.1,
            synapse_decay=1.0,
            eligibility_decay=0.9,
        )
        r.step(col_drive([1.0, 0.0, 0.0, 0.0], 2))
        l23_traced = np.where(r.trace_l23 > 0)[0][0]

        r.step(col_drive([0.0, 1.0, 0.0, 0.0], 2))
        l23_active = np.where(r.active_l23)[0]

        for n in l23_active:
            assert r.l23_lateral_weights[l23_traced, n] > 0

    def test_l23_weights_clipped_to_unit(self):
        """L2/3 lateral weights stay in [0, 1]."""
        r = CorticalRegion(
            n_columns=4,
            n_l4=2,
            n_l23=2,
            k_columns=1,
            learning_rate=1.0,
            synapse_decay=1.0,
        )
        for _ in range(100):
            r.step(col_drive([1.0, 0.0, 0.0, 0.0], 2))
        assert r.l23_lateral_weights.max() <= 1.0
        assert r.l23_lateral_weights.min() >= 0.0


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
        )
        target = 3  # col 0, neuron 3
        r.predict_neuron(target, 0, segment_type="fb")
        r.active_l23[0] = True

        r.step(col_drive([1.0, 0.0, 0.0, 0.0], 4))
        assert r.active_l4[target]
        assert not r.bursting_columns[0]


# ---------------------------------------------------------------------------
# CorticalRegion: L2/3 lateral connections
# ---------------------------------------------------------------------------


class TestL23Lateral:
    def test_l23_burst_on_first_step(self):
        """First step: L2/3 also bursts since L4 bursts."""
        r = CorticalRegion(n_columns=4, n_l4=4, n_l23=4, k_columns=1)
        r.step(col_drive([1.0, 0.0, 0.0, 0.0], 4))
        assert r.bursting_columns[0]
        assert r.active_l23[0:4].all()

    def test_lateral_overrides_l4_match(self):
        """Strong L2/3 lateral input can make a different L2/3
        neuron win than the L4-matching one."""
        r = CorticalRegion(
            n_columns=4,
            n_l4=4,
            n_l23=4,
            k_columns=1,
        )
        r.predict_neuron(0, 0, segment_type="fb")
        r.active_l23[0] = True

        source_l23 = 7  # col 1, neuron 3
        target_l23 = 3  # col 0, neuron 3
        r.l23_lateral_weights[source_l23, target_l23] = 5.0
        r.active_l23[source_l23] = True

        r.step(col_drive([1.0, 0.0, 0.0, 0.0], 4))
        assert r.active_l4[0]
        assert r.active_l23[target_l23]

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
        r.step(col_drive([1.0, 0.0, 0.0, 0.0], 2))
        l23_traced = np.where(r.trace_l23 > 0)[0][0]
        r.step(col_drive([0.0, 1.0, 0.0, 0.0], 2))
        l23_active = np.where(r.active_l23)[0]
        for n in l23_active:
            assert r.l23_lateral_weights[l23_traced, n] > 0

    def test_l23_voltage_resets_on_activation(self):
        """Active L2/3 neurons have voltage reset after step."""
        r = CorticalRegion(n_columns=4, n_l4=2, n_l23=2, k_columns=1)
        r.step(col_drive([1.0, 0.0, 0.0, 0.0], 2))
        for idx in np.where(r.active_l23)[0]:
            assert r.voltage_l23[idx] == 0.0


# ---------------------------------------------------------------------------
# CorticalRegion: dendritic segments
# ---------------------------------------------------------------------------


class TestDendriticSegments:
    def test_segment_arrays_initialized(self):
        r = CorticalRegion(n_columns=4, n_l4=2, n_l23=2, k_columns=1)
        assert r.fb_seg_indices.shape == (8, 4, 16)
        assert r.fb_seg_perm.shape == (8, 4, 16)
        assert r.lat_seg_indices.shape == (8, 4, 16)
        assert r.lat_seg_perm.shape == (8, 4, 16)

    def test_segments_start_disconnected(self):
        """All permanences start at 0 — no predictions initially."""
        r = CorticalRegion(n_columns=4, n_l4=2, n_l23=2, k_columns=1)
        assert r.fb_seg_perm.max() == 0.0
        assert r.lat_seg_perm.max() == 0.0

    def test_no_predictions_initially(self):
        """With all permanences at 0, no neurons should be predicted."""
        r = CorticalRegion(n_columns=4, n_l4=2, n_l23=2, k_columns=1)
        r.active_l23[0] = True
        pred = r.get_prediction(k=4)
        assert len(pred) == 0

    def test_predict_neuron_helper(self):
        """predict_neuron sets up a segment that fires."""
        r = CorticalRegion(n_columns=4, n_l4=2, n_l23=2, k_columns=1)
        r.predict_neuron(3, 0, segment_type="fb")
        r.active_l23[0] = True
        pred = r.get_prediction(k=4)
        assert 3 in pred

    def test_segment_growth_on_burst(self):
        """Bursting should grow segment permanences above zero."""
        r = CorticalRegion(
            n_columns=4,
            n_l4=2,
            n_l23=2,
            k_columns=1,
            synapse_decay=1.0,
        )
        r.step(col_drive([1.0, 0.0, 0.0, 0.0], 2))
        r.step(col_drive([0.0, 1.0, 0.0, 0.0], 2))

        col1_neurons = [2, 3]
        any_grown = False
        for n in col1_neurons:
            if r.fb_seg_perm[n].max() > 0 or r.lat_seg_perm[n].max() > 0:
                any_grown = True
        assert any_grown

    def test_segments_reduce_burst_rate(self):
        """After learning, segments should predict and reduce bursting."""
        r = CorticalRegion(
            n_columns=4,
            n_l4=2,
            n_l23=2,
            k_columns=1,
            synapse_decay=1.0,
            perm_init=0.7,
            perm_increment=0.15,
            seg_activation_threshold=2,
        )
        for _ in range(50):
            r.step(col_drive([1.0, 0.0, 0.0, 0.0], 2))
            r.step(col_drive([0.0, 1.0, 0.0, 0.0], 2))

        bursts = 0
        total = 0
        for _ in range(10):
            r.step(col_drive([1.0, 0.0, 0.0, 0.0], 2))
            total += 1
            bursts += int(r.bursting_columns[0])
            r.step(col_drive([0.0, 1.0, 0.0, 0.0], 2))
            total += 1
            bursts += int(r.bursting_columns[1])

        burst_rate = bursts / total
        assert burst_rate < 1.0, f"burst_rate={burst_rate}, segments not learning"

    def test_false_prediction_punished(self):
        """Segments that predict incorrectly should have permanences reduced."""
        r = CorticalRegion(
            n_columns=4,
            n_l4=2,
            n_l23=2,
            k_columns=1,
            perm_decrement=0.2,
        )
        r.predict_neuron(0, 0, segment_type="fb")
        r.active_l23[0] = True

        initial_perm = r.fb_seg_perm[0, 0, 0]
        r.step(col_drive([0.0, 1.0, 0.0, 0.0], 2))
        assert r.fb_seg_perm[0, 0, 0] < initial_perm

    def test_sensory_local_connectivity(self):
        """SensoryRegion segments should only connect to local neurons."""
        s = SensoryRegion(
            input_dim=10,
            n_columns=8,
            n_l4=2,
            n_l23=2,
            k_columns=2,
        )
        radius = max(1, 8 // 4)  # 2
        for s_idx in range(s.n_fb_segments):
            for syn_idx in range(s.n_synapses_per_segment):
                source = s.fb_seg_indices[0, s_idx, syn_idx]
                source_col = source // s.n_l23
                assert source_col <= radius, (
                    f"neuron 0 has fb synapse to col {source_col}, "
                    f"expected <= {radius}"
                )


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
        # Per-neuron: (input_dim, n_l4_total)
        assert s.ff_weights.shape == (100, 128)

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
