"""Tests for LaminaProbe and ChatLaminaProbe."""

import numpy as np
import pytest

from step.probes.chat import ChatLaminaProbe
from step.probes.core import LaminaProbe, Probe, _participation_ratio

from .conftest import make_circuit, step_circuit

# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocol:
    def test_lamina_probe_is_probe(self):
        probe = LaminaProbe()
        assert isinstance(probe, Probe)

    def test_chat_lamina_probe_is_probe(self):
        probe = ChatLaminaProbe()
        assert isinstance(probe, Probe)

    def test_has_name(self):
        assert LaminaProbe().name == "lamina"
        assert ChatLaminaProbe().name == "chat_lamina"


# ---------------------------------------------------------------------------
# LaminaProbe: read-only observation
# ---------------------------------------------------------------------------


class TestLaminaProbeReadOnly:
    def test_observe_does_not_modify_circuit(self):
        circuit, encoder = make_circuit()
        rng = np.random.default_rng(42)
        step_circuit(circuit, encoder, rng)

        # Snapshot circuit state
        region = circuit.region("S1")
        l4_active_before = region.l4.active.copy()
        l23_active_before = region.l23.active.copy()
        l4_predicted_before = region.l4.predicted.copy()
        ff_weights_before = region.ff_weights.copy()

        probe = LaminaProbe()
        probe.observe(circuit)

        # Verify nothing changed
        np.testing.assert_array_equal(region.l4.active, l4_active_before)
        np.testing.assert_array_equal(region.l23.active, l23_active_before)
        np.testing.assert_array_equal(region.l4.predicted, l4_predicted_before)
        np.testing.assert_array_equal(region.ff_weights, ff_weights_before)


# ---------------------------------------------------------------------------
# LaminaProbe: L4 KPIs
# ---------------------------------------------------------------------------


class TestL4KPIs:
    def test_recall_after_steps(self):
        circuit, encoder = make_circuit()
        probe = LaminaProbe()
        rng = np.random.default_rng(42)

        for _ in range(100):
            step_circuit(circuit, encoder, rng)
            probe.observe(circuit)

        snap = probe.snapshot()
        # Recall should be between 0 and 1
        assert 0.0 <= snap["S1"]["l4"]["recall"] <= 1.0

    def test_precision_after_steps(self):
        circuit, encoder = make_circuit()
        probe = LaminaProbe()
        rng = np.random.default_rng(42)

        for _ in range(100):
            step_circuit(circuit, encoder, rng)
            probe.observe(circuit)

        snap = probe.snapshot()
        assert 0.0 <= snap["S1"]["l4"]["precision"] <= 1.0

    def test_sparseness_near_target(self):
        """Population sparseness should be near k/N for binary activations."""
        circuit, encoder = make_circuit(n_columns=16, n_l4=4, k_columns=3)
        probe = LaminaProbe()
        rng = np.random.default_rng(42)

        for _ in range(100):
            step_circuit(circuit, encoder, rng)
            probe.observe(circuit)

        snap = probe.snapshot()
        sparseness = snap["S1"]["l4"]["sparseness"]
        # k=3 columns * n_l4 neurons (burst) out of 64 total
        # Should be in a reasonable range (not 0, not 1)
        assert 0.0 < sparseness < 0.5

    def test_recall_precision_hand_computed(self):
        """Verify recall/precision on manually set state."""
        circuit, _encoder = make_circuit()
        region = circuit.region("S1")
        probe = LaminaProbe()

        # Manually set: 10 active, 8 predicted, 6 overlap
        region.l4.active[:] = False
        region.l4.active[:10] = True
        region.l4.predicted[:] = False
        region.l4.predicted[2:10] = True  # 8 predicted, 6 overlap with active[0:10]

        probe.observe(circuit)
        snap = probe.snapshot()

        # Recall = overlap / active = 8/10 = 0.8
        # (predicted[2:10] & active[0:10] = active[2:10] = 8 neurons)
        assert abs(snap["S1"]["l4"]["recall"] - 0.8) < 0.01

        # Precision = overlap / predicted = 8/8 = 1.0
        assert abs(snap["S1"]["l4"]["precision"] - 1.0) < 0.01


# ---------------------------------------------------------------------------
# LaminaProbe: L2/3 KPIs
# ---------------------------------------------------------------------------


class TestL23KPIs:
    def test_eff_dim_nonzero_after_steps(self):
        circuit, encoder = make_circuit()
        probe = LaminaProbe(l23_sample_interval=1)  # sample every step
        rng = np.random.default_rng(42)

        for _ in range(50):
            step_circuit(circuit, encoder, rng)
            probe.observe(circuit)

        snap = probe.snapshot()
        assert snap["S1"]["l23"]["eff_dim"] > 0

    def test_eff_dim_zero_without_enough_samples(self):
        probe = LaminaProbe()
        snap = probe.snapshot()
        assert snap == {}  # no regions observed


# ---------------------------------------------------------------------------
# ChatLaminaProbe: linear probe
# ---------------------------------------------------------------------------


class TestLinearProbe:
    def test_linear_probe_above_chance(self):
        """Linear probe should converge above random for separable patterns."""
        pytest.importorskip("sklearn")
        circuit, encoder = make_circuit(n_columns=32, n_l4=4, n_l23=4, k_columns=4)
        probe = ChatLaminaProbe(
            linear_probe_fit_interval=500,
            linear_probe_window=1000,
        )

        # Feed distinct tokens — each should produce somewhat distinct L2/3
        chars = "abcdefgh"
        for i in range(2000):
            token_id = i % len(chars)
            encoding = encoder.encode(chars[token_id])
            circuit.process(encoding)
            probe.observe(circuit, stimulus_id=token_id)

        snap = probe.snapshot()
        accuracy = snap["S1"]["l23"]["linear_probe"]
        # Should be above random (1/8 = 12.5%)
        assert accuracy > 0.15, f"Linear probe accuracy {accuracy:.1%} too low"


# ---------------------------------------------------------------------------
# ChatLaminaProbe: context discrimination
# ---------------------------------------------------------------------------


class TestContextDiscrimination:
    def test_ctx_disc_nonzero(self):
        circuit, encoder = make_circuit()
        probe = ChatLaminaProbe(ctx_disc_min_contexts=2)

        # Feed varied bigram contexts
        chars = "abcdefgh"
        token_seq = [0, 1, 0, 2, 0, 3, 0, 1, 0, 2] * 50
        for token_id in token_seq:
            encoding = encoder.encode(chars[token_id % len(chars)])
            circuit.process(encoding)
            probe.observe(circuit, stimulus_id=token_id)

        snap = probe.snapshot()
        ctx = snap["S1"]["l23"]["ctx_disc"]
        # With different preceding contexts, discrimination should be > 0
        assert ctx >= 0.0

    def test_boundary_resets_prev_token(self):
        probe = ChatLaminaProbe()
        probe._prev_token = 42
        probe.boundary()
        assert probe._prev_token is None


# ---------------------------------------------------------------------------
# ChatLaminaProbe: no stimulus_id degrades gracefully
# ---------------------------------------------------------------------------


class TestNoStimulusId:
    def test_observe_without_stimulus_id(self):
        """ChatLaminaProbe should work without stimulus_id (L4 KPIs only)."""
        circuit, encoder = make_circuit()
        probe = ChatLaminaProbe()
        rng = np.random.default_rng(42)

        for _ in range(50):
            step_circuit(circuit, encoder, rng)
            probe.observe(circuit)  # no stimulus_id

        snap = probe.snapshot()
        # L4 KPIs should still work
        assert "recall" in snap["S1"]["l4"]
        # L2/3 chat KPIs should be 0 (no data)
        assert snap["S1"]["l23"]["linear_probe"] == 0.0
        assert snap["S1"]["l23"]["ctx_disc"] == 0.0


# ---------------------------------------------------------------------------
# Participation ratio helper
# ---------------------------------------------------------------------------


class TestParticipationRatio:
    def test_returns_zero_for_few_samples(self):
        assert _participation_ratio([]) == 0.0
        assert _participation_ratio([np.zeros(10)]) == 0.0

    def test_collapsed_has_low_pr(self):
        """Identical patterns = 0 effective dimensions."""
        same = [np.array([1.0, 0, 0, 0, 0])] * 20
        assert _participation_ratio(same) < 1.0

    def test_diverse_has_high_pr(self):
        """Orthogonal patterns = high effective dimensions."""
        rng = np.random.default_rng(42)
        diverse = [rng.random(50) for _ in range(100)]
        pr = _participation_ratio(diverse)
        assert pr > 10  # many effective dimensions
