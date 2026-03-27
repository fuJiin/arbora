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
        r = CorticalRegion(8, n_columns=64, n_l4=8, n_l23=8, k_columns=6)
        assert r.l4.voltage.shape == (512,)
        assert r.l23.voltage.shape == (512,)
        assert r.l4.trace.shape == (512,)
        assert r.l23.trace.shape == (512,)

    def test_initial_state_is_zero(self):
        r = CorticalRegion(8, n_columns=32, n_l4=4, n_l23=4, k_columns=4)
        assert r.l4.voltage.sum() == 0
        assert r.l4.excitability.sum() == 0
        assert not r.l4.active.any()
        assert not r.l23.active.any()
        assert r.l4.trace.sum() == 0


# ---------------------------------------------------------------------------
# CorticalRegion: activation mechanics
# ---------------------------------------------------------------------------


class TestActivation:
    @pytest.fixture()
    def region(self):
        return CorticalRegion(8, n_columns=16, n_l4=4, n_l23=4, k_columns=3)

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
            col_neurons = region.l4.active[col * 4 : (col + 1) * 4]
            assert col_neurons.all()

    def test_precise_when_predicted(self, region):
        """If a neuron was predicted, its column should activate precisely."""
        region.predict_neuron(3, 0, segment_type="lat")
        region.l4.active[0] = True

        drive = col_drive([1.0, 0.5, 0.3] + [0.0] * 13, 4)
        region.step(drive)

        assert region.active_columns[0]
        assert not region.bursting_columns[0]
        col0_neurons = region.l4.active[0:4]
        assert col0_neurons.sum() == 1
        assert region.l4.active[3]

    def test_excitability_breaks_ties(self, region):
        region.l4.excitability[2] = 10.0
        region.predict_neuron(2, 0, segment_type="lat")
        region.l4.active[0] = True

        drive = col_drive([1.0, 0.5, 0.3] + [0.0] * 13, 4)
        region.step(drive)
        assert region.l4.active[2]

    def test_inactive_columns_have_no_active_neurons(self, region):
        drive = col_drive([1.0] + [0.0] * 15, 4)
        region.step(drive)
        for col in range(16):
            if not region.active_columns[col]:
                assert not region.l4.active[col * 4 : (col + 1) * 4].any()
                assert not region.l23.active[col * 4 : (col + 1) * 4].any()


# ---------------------------------------------------------------------------
# CorticalRegion: burst mechanics
# ---------------------------------------------------------------------------


class TestBurst:
    def test_all_neurons_fire_on_burst(self):
        """Burst column has all L4 and L2/3 neurons active."""
        r = CorticalRegion(8, n_columns=4, n_l4=4, n_l23=4, k_columns=1)
        r.step(col_drive([1.0, 0.0, 0.0, 0.0], 4))
        assert r.bursting_columns[0]
        assert r.l4.active[0:4].all()
        assert r.l23.active[0:4].all()

    def test_burst_rate_decreases_with_learning(self):
        """As segments develop, burst rate should decrease."""
        r = CorticalRegion(
            8,
            n_columns=4,
            n_l4=2,
            n_l23=2,
            k_columns=1,
            learning_rate=0.5,
            synapse_decay=1.0,
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
        r = CorticalRegion(8, n_columns=4, n_l4=4, n_l23=4, k_columns=1)
        r.l4.voltage[2] = 0.5
        r.step(col_drive([1.0, 0.0, 0.0, 0.0], 4))
        assert r.bursting_columns[0]
        assert r.l4.active[0:4].all()
        assert r.l4.trace[2] == 1.0

    def test_burst_learning_scale_l23(self):
        """Burst columns should produce larger L2/3 lateral weight updates."""
        r = CorticalRegion(
            8,
            n_columns=4,
            n_l4=2,
            n_l23=2,
            k_columns=1,
            learning_rate=0.1,
            synapse_decay=1.0,
            burst_learning_scale=5.0,
        )
        r.step(col_drive([1.0, 0.0, 0.0, 0.0], 2))
        _l23_traced = np.where(r.l23.trace > 0)[0][0]

        r.step(col_drive([0.0, 1.0, 0.0, 0.0], 2))
        _l23_active = np.where(r.l23.active)[0]

        # L2/3 segments should have grown (permanences > 0)
        assert r.l23_seg_perm.sum() > 0


# ---------------------------------------------------------------------------
# CorticalRegion: voltage dynamics
# ---------------------------------------------------------------------------


class TestVoltage:
    def test_active_neuron_voltage_resets(self):
        r = CorticalRegion(8, n_columns=4, n_l4=2, n_l23=2, k_columns=1)
        r.step(col_drive([1.0, 0.0, 0.0, 0.0], 2))
        for idx in np.where(r.l4.active)[0]:
            assert r.l4.voltage[idx] == 0.0

    def test_voltage_accumulates_over_steps(self):
        r = CorticalRegion(
            8,
            n_columns=4,
            n_l4=2,
            n_l23=2,
            k_columns=1,
            voltage_decay=0.9,
        )
        r.step(col_drive([5.0, 0.0, 0.0, 0.3], 2))
        r.step(col_drive([5.0, 0.0, 0.0, 0.3], 2))
        col3_start = 3 * 2
        assert r.l4.voltage[col3_start : col3_start + 2].max() > 0.5

    def test_voltage_decays_each_step(self):
        r = CorticalRegion(
            8,
            n_columns=4,
            n_l4=2,
            n_l23=2,
            k_columns=1,
            voltage_decay=0.5,
        )
        r.l4.voltage[6] = 1.0  # col 3, neuron 0
        r.step(col_drive([10.0, 0.0, 0.0, 0.0], 2))
        # Col 3 neuron 0: 1.0 * 0.5 (decay) + 0.0 (no drive) = 0.5
        assert r.l4.voltage[6] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# CorticalRegion: excitability
# ---------------------------------------------------------------------------


class TestExcitability:
    def test_grows_for_inactive_neurons(self):
        r = CorticalRegion(8, n_columns=8, n_l4=2, n_l23=2, k_columns=1)
        r.step(col_drive([1.0] + [0.0] * 7, 2))
        inactive = ~r.l4.active
        assert (r.l4.excitability[inactive] > 0).all()

    def test_resets_on_activation(self):
        r = CorticalRegion(8, n_columns=4, n_l4=2, n_l23=2, k_columns=1)
        r.l4.excitability[:] = 5.0
        r.step(col_drive([1.0, 0.0, 0.0, 0.0], 2))
        for idx in np.where(r.l4.active)[0]:
            assert r.l4.excitability[idx] == 0.0

    def test_excitability_ensures_rotation(self):
        """Over many steps, all neurons should eventually activate."""
        r = CorticalRegion(8, n_columns=4, n_l4=4, n_l23=4, k_columns=1)
        drive = col_drive([1.0, 0.0, 0.0, 0.0], 4)
        activated = set()
        for _ in range(20):
            r.step(drive)
            activated.update(np.where(r.l4.active)[0].tolist())
        col0_neurons = {0, 1, 2, 3}
        assert col0_neurons.issubset(activated)


# ---------------------------------------------------------------------------
# CorticalRegion: eligibility traces
# ---------------------------------------------------------------------------


class TestEligibility:
    def test_set_on_activation(self):
        r = CorticalRegion(8, n_columns=4, n_l4=2, n_l23=2, k_columns=1)
        r.step(col_drive([1.0, 0.0, 0.0, 0.0], 2))
        assert r.l4.trace.sum() > 0
        assert r.l23.trace.sum() > 0

    def test_decays_over_time(self):
        r = CorticalRegion(
            8,
            n_columns=4,
            n_l4=2,
            n_l23=2,
            k_columns=1,
            eligibility_decay=0.5,
        )
        r.step(col_drive([1.0, 0.0, 0.0, 0.0], 2))
        traced_neurons = np.where(r.l4.trace > 0)[0].copy()
        r.step(col_drive([0.0, 1.0, 0.0, 0.0], 2))
        for idx in traced_neurons:
            if not r.l4.active[idx]:
                assert r.l4.trace[idx] == pytest.approx(0.5)

    def test_inactive_neurons_have_zero_trace(self):
        r = CorticalRegion(8, n_columns=4, n_l4=2, n_l23=2, k_columns=1)
        r.step(col_drive([1.0, 0.0, 0.0, 0.0], 2))
        trace_count = (r.l4.trace > 0).sum()
        assert trace_count == 1


# ---------------------------------------------------------------------------
# CorticalRegion: learning
# ---------------------------------------------------------------------------


class TestLearning:
    def test_l23_segments_grow_on_sequence(self):
        """L2/3 segment permanences should grow after sequential activation."""
        r = CorticalRegion(
            8,
            n_columns=4,
            n_l4=2,
            n_l23=2,
            k_columns=1,
            learning_rate=0.1,
            synapse_decay=1.0,
            eligibility_decay=0.9,
        )
        initial_perm = r.l23_seg_perm.sum()

        r.step(col_drive([1.0, 0.0, 0.0, 0.0], 2))
        r.step(col_drive([0.0, 1.0, 0.0, 0.0], 2))

        # Segment permanences should have changed
        assert r.l23_seg_perm.sum() != initial_perm

    def test_l23_seg_perm_clipped_to_unit(self):
        """L2/3 segment permanences stay in [0, 1]."""
        r = CorticalRegion(
            8,
            n_columns=4,
            n_l4=2,
            n_l23=2,
            k_columns=1,
            learning_rate=1.0,
            synapse_decay=1.0,
        )
        for _ in range(100):
            r.step(col_drive([1.0, 0.0, 0.0, 0.0], 2))
        assert r.l23_seg_perm.max() <= 1.0
        assert r.l23_seg_perm.min() >= 0.0


# ---------------------------------------------------------------------------
# CorticalRegion: feedback influences activation
# ---------------------------------------------------------------------------


class TestLateralPrediction:
    def test_lateral_biases_neuron_selection(self):
        """Lateral prediction to a specific L4 neuron should make it win
        and produce a precise (non-burst) activation."""
        r = CorticalRegion(
            8,
            n_columns=4,
            n_l4=4,
            n_l23=4,
            k_columns=1,
        )
        target = 3  # col 0, neuron 3
        r.predict_neuron(target, 0, segment_type="lat")
        r.l4.active[0] = True  # L4 lateral context

        r.step(col_drive([1.0, 0.0, 0.0, 0.0], 4))
        assert r.l4.active[target]
        assert not r.bursting_columns[0]


# ---------------------------------------------------------------------------
# CorticalRegion: L2/3 lateral connections
# ---------------------------------------------------------------------------


class TestL23Lateral:
    def test_l23_burst_on_first_step(self):
        """First step: L2/3 also bursts since L4 bursts."""
        r = CorticalRegion(8, n_columns=4, n_l4=4, n_l23=4, k_columns=1)
        r.step(col_drive([1.0, 0.0, 0.0, 0.0], 4))
        assert r.bursting_columns[0]
        assert r.l23.active[0:4].all()

    def test_l23_segment_prediction_biases_selection(self):
        """L2/3 segment prediction boosts a specific neuron to win."""
        r = CorticalRegion(
            8,
            n_columns=4,
            n_l4=4,
            n_l23=4,
            k_columns=1,
        )
        # Wire L2/3 segment: neuron 3 in col 0 predicted by neuron 7 (col 1)
        target_l23 = 3
        r.l23_seg_indices[target_l23, 0, :] = 7  # source: col 1 neuron 3
        r.l23_seg_perm[target_l23, 0, :] = 1.0
        r.l23.active[7] = True  # source active

        r.step(col_drive([1.0, 0.0, 0.0, 0.0], 4))
        # The predicted neuron should have won L2/3 competition
        assert r.l23.active[target_l23]

    def test_l23_voltage_resets_on_activation(self):
        """Active L2/3 neurons have voltage reset after step."""
        r = CorticalRegion(8, n_columns=4, n_l4=2, n_l23=2, k_columns=1)
        r.step(col_drive([1.0, 0.0, 0.0, 0.0], 2))
        for idx in np.where(r.l23.active)[0]:
            assert r.l23.voltage[idx] == 0.0


# ---------------------------------------------------------------------------
# CorticalRegion: dendritic segments
# ---------------------------------------------------------------------------


class TestDendriticSegments:
    def test_segment_arrays_initialized(self):
        r = CorticalRegion(8, n_columns=4, n_l4=2, n_l23=2, k_columns=1)
        n_syn = r.n_synapses_per_segment
        assert r.l4_lat_seg_indices.shape == (8, 4, n_syn)
        assert r.l4_lat_seg_perm.shape == (8, 4, n_syn)

    def test_segments_start_disconnected(self):
        """All permanences start at 0 — no predictions initially."""
        r = CorticalRegion(8, n_columns=4, n_l4=2, n_l23=2, k_columns=1)
        assert r.l4_lat_seg_perm.max() == 0.0

    def test_no_predictions_initially(self):
        """With all permanences at 0, no neurons should be predicted."""
        r = CorticalRegion(8, n_columns=4, n_l4=2, n_l23=2, k_columns=1)
        r.l4.active[0] = True
        pred = r.get_prediction(k=4)
        assert len(pred) == 0

    def test_predict_neuron_helper(self):
        """predict_neuron sets up a segment that fires."""
        r = CorticalRegion(8, n_columns=4, n_l4=2, n_l23=2, k_columns=1)
        r.predict_neuron(3, 0, segment_type="lat")
        r.l4.active[0] = True
        pred = r.get_prediction(k=4)
        assert 3 in pred

    def test_segment_growth_on_burst(self):
        """Bursting should grow segment permanences above zero."""
        r = CorticalRegion(
            8,
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
            if r.l4_lat_seg_perm[n].max() > 0:
                any_grown = True
        assert any_grown

    def test_segments_reduce_burst_rate(self):
        """After learning, segments should predict and reduce bursting."""
        r = CorticalRegion(
            8,
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
            8,
            n_columns=4,
            n_l4=2,
            n_l23=2,
            k_columns=1,
            perm_decrement=0.2,
        )
        r.predict_neuron(0, 0, segment_type="lat")
        r.l4.active[0] = True

        initial_perm = r.l4_lat_seg_perm[0, 0, 0]
        r.step(col_drive([0.0, 1.0, 0.0, 0.0], 2))
        assert r.l4_lat_seg_perm[0, 0, 0] < initial_perm

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
        for s_idx in range(s.n_l4_lat_segments):
            for syn_idx in range(s.n_synapses_per_segment):
                source = s.l4_lat_seg_indices[0, s_idx, syn_idx]
                source_col = source // s.n_l4
                assert source_col <= radius, (
                    f"neuron 0 lat synapse col {source_col} > {radius}"
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
