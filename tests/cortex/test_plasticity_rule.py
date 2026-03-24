"""Tests for PlasticityRule enum and unified _learn_ff dispatch."""

import numpy as np

from step.config import PlasticityRule
from step.cortex.motor import MotorRegion
from step.cortex.pfc import PFCRegion
from step.cortex.premotor import PremotorRegion
from step.cortex.region import CorticalRegion

# ---------------------------------------------------------------------------
# PlasticityRule enum basics
# ---------------------------------------------------------------------------


class TestPlasticityRuleEnum:
    def test_values(self):
        assert PlasticityRule.HEBBIAN.value == "hebbian"
        assert PlasticityRule.THREE_FACTOR.value == "three_factor"

    def test_round_trip(self):
        assert PlasticityRule("hebbian") is PlasticityRule.HEBBIAN
        assert PlasticityRule("three_factor") is PlasticityRule.THREE_FACTOR


# ---------------------------------------------------------------------------
# Base CorticalRegion: plasticity_rule param
# ---------------------------------------------------------------------------


class TestCorticalRegionPlasticityRule:
    def test_default_is_hebbian(self):
        r = CorticalRegion(8, n_columns=4, n_l4=2, n_l23=2, k_columns=1)
        assert r.plasticity_rule == PlasticityRule.HEBBIAN
        assert r._ff_eligibility is None

    def test_three_factor_allocates_eligibility(self):
        r = CorticalRegion(
            8,
            n_columns=4,
            n_l4=2,
            n_l23=2,
            k_columns=1,
            plasticity_rule=PlasticityRule.THREE_FACTOR,
        )
        assert r.plasticity_rule == PlasticityRule.THREE_FACTOR
        assert r._ff_eligibility is not None
        assert r._ff_eligibility.shape == r.ff_weights.shape

    def test_hebbian_does_not_allocate_eligibility(self):
        r = CorticalRegion(
            8,
            n_columns=4,
            n_l4=2,
            n_l23=2,
            k_columns=1,
            plasticity_rule=PlasticityRule.HEBBIAN,
        )
        assert r._ff_eligibility is None


# ---------------------------------------------------------------------------
# HEBBIAN path: immediate weight updates
# ---------------------------------------------------------------------------


class TestHebbianPath:
    def test_immediate_weight_update(self):
        """HEBBIAN rule updates ff_weights directly in _learn_ff."""
        r = CorticalRegion(
            8,
            n_columns=4,
            n_l4=2,
            n_l23=2,
            k_columns=1,
            learning_rate=0.5,
            synapse_decay=1.0,
            plasticity_rule=PlasticityRule.HEBBIAN,
        )
        weights_before = r.ff_weights.copy()
        inp = np.zeros(8)
        inp[0] = 1.0
        r.process(inp)
        # Weights should change immediately (LTP + subthreshold)
        assert not np.array_equal(r.ff_weights, weights_before)

    def test_apply_reward_is_noop_for_hebbian(self):
        """apply_reward should be a no-op when no eligibility trace exists."""
        r = CorticalRegion(
            8,
            n_columns=4,
            n_l4=2,
            n_l23=2,
            k_columns=1,
            plasticity_rule=PlasticityRule.HEBBIAN,
        )
        inp = np.zeros(8)
        inp[0] = 1.0
        r.process(inp)
        weights_before = r.ff_weights.copy()
        r.apply_reward(1.0)
        # No change -- no eligibility trace to consolidate
        np.testing.assert_array_equal(r.ff_weights, weights_before)


# ---------------------------------------------------------------------------
# THREE_FACTOR path: eligibility trace accumulation + reward consolidation
# ---------------------------------------------------------------------------


class TestThreeFactorPath:
    def test_no_immediate_weight_update(self):
        """THREE_FACTOR should NOT update ff_weights in _learn_ff."""
        r = CorticalRegion(
            8,
            n_columns=4,
            n_l4=2,
            n_l23=2,
            k_columns=1,
            learning_rate=0.5,
            synapse_decay=1.0,
            plasticity_rule=PlasticityRule.THREE_FACTOR,
        )
        # Process input -- this fills eligibility but should NOT change weights
        # (beyond structural initialization). Capture weights right before _learn_ff.
        inp = np.zeros(8)
        inp[0] = 1.0

        # Manually run the pipeline to isolate _learn_ff behavior:
        flat = inp.flatten().astype(np.float64)
        neuron_drive = flat @ r.ff_weights
        r.step(neuron_drive)
        weights_before = r.ff_weights.copy()
        r._learn_ff(flat)
        # Weights unchanged -- only eligibility trace updated
        np.testing.assert_array_equal(r.ff_weights, weights_before)
        # But eligibility trace should be nonzero
        assert r._ff_eligibility.sum() != 0.0

    def test_apply_reward_consolidates(self):
        """apply_reward should consolidate eligibility into ff_weights."""
        r = CorticalRegion(
            8,
            n_columns=4,
            n_l4=2,
            n_l23=2,
            k_columns=1,
            learning_rate=0.5,
            synapse_decay=1.0,
            plasticity_rule=PlasticityRule.THREE_FACTOR,
        )
        inp = np.zeros(8)
        inp[0] = 1.0
        r.process(inp)
        weights_before = r.ff_weights.copy()
        r.apply_reward(1.0)
        # Weights should now change
        assert not np.array_equal(r.ff_weights, weights_before)

    def test_eligibility_decays(self):
        """Eligibility trace should decay each time _learn_ff is called."""
        r = CorticalRegion(
            8,
            n_columns=4,
            n_l4=2,
            n_l23=2,
            k_columns=1,
            learning_rate=0.5,
            eligibility_decay=0.5,
            plasticity_rule=PlasticityRule.THREE_FACTOR,
        )
        inp = np.zeros(8)
        inp[0] = 1.0
        r.process(inp)
        elig_after_first = r._ff_eligibility.copy()

        # Process with zero input -- triggers decay but no new accumulation
        zero_inp = np.zeros(8)
        r.process(zero_inp)
        # Eligibility should have decayed by eligibility_decay
        max_after_first = np.abs(elig_after_first).max()
        max_after_second = np.abs(r._ff_eligibility).max()
        if max_after_first > 0:
            assert max_after_second < max_after_first

    def test_reset_clears_eligibility(self):
        """reset_working_memory should zero out eligibility traces."""
        r = CorticalRegion(
            8,
            n_columns=4,
            n_l4=2,
            n_l23=2,
            k_columns=1,
            plasticity_rule=PlasticityRule.THREE_FACTOR,
        )
        inp = np.zeros(8)
        inp[0] = 1.0
        r.process(inp)
        assert r._ff_eligibility.sum() != 0.0
        r.reset_working_memory()
        assert r._ff_eligibility.sum() == 0.0


# ---------------------------------------------------------------------------
# Subclass defaults: PFC, Motor, Premotor all default to THREE_FACTOR
# ---------------------------------------------------------------------------


class TestSubclassDefaults:
    def test_pfc_defaults_to_three_factor(self):
        pfc = PFCRegion(input_dim=16, n_columns=4, n_l4=2, n_l23=2, k_columns=2)
        assert pfc.plasticity_rule == PlasticityRule.THREE_FACTOR
        assert pfc._ff_eligibility is not None

    def test_motor_defaults_to_three_factor(self):
        m1 = MotorRegion(input_dim=16, n_columns=4, n_l4=2, n_l23=2, k_columns=1)
        assert m1.plasticity_rule == PlasticityRule.THREE_FACTOR
        assert m1._ff_eligibility is not None

    def test_premotor_defaults_to_three_factor(self):
        m2 = PremotorRegion(input_dim=16, n_columns=4, n_l4=2, n_l23=2, k_columns=2)
        assert m2.plasticity_rule == PlasticityRule.THREE_FACTOR
        assert m2._ff_eligibility is not None

    def test_base_region_defaults_to_hebbian(self):
        r = CorticalRegion(8, n_columns=4, n_l4=2, n_l23=2, k_columns=1)
        assert r.plasticity_rule == PlasticityRule.HEBBIAN

    def test_pfc_can_be_overridden_to_hebbian(self):
        pfc = PFCRegion(
            input_dim=16,
            n_columns=4,
            n_l4=2,
            n_l23=2,
            k_columns=2,
            plasticity_rule=PlasticityRule.HEBBIAN,
        )
        assert pfc.plasticity_rule == PlasticityRule.HEBBIAN
        assert pfc._ff_eligibility is None

    def test_motor_can_be_overridden_to_hebbian(self):
        m1 = MotorRegion(
            input_dim=16,
            n_columns=4,
            n_l4=2,
            n_l23=2,
            k_columns=1,
            plasticity_rule=PlasticityRule.HEBBIAN,
        )
        assert m1.plasticity_rule == PlasticityRule.HEBBIAN
        assert m1._ff_eligibility is None


# ---------------------------------------------------------------------------
# PFC gate-closed eligibility decay
# ---------------------------------------------------------------------------


class TestPFCGateClosedDecay:
    def test_gate_closed_decays_eligibility(self):
        """When gate is closed, eligibility should still decay."""
        pfc = PFCRegion(
            input_dim=8,
            n_columns=4,
            n_l4=2,
            n_l23=2,
            k_columns=2,
            eligibility_decay=0.5,
        )
        # Process with gate open to build up eligibility
        inp = np.ones(8) * 0.5
        pfc.process(inp)
        elig_before = pfc._ff_eligibility.copy()
        assert elig_before.sum() != 0.0

        # Close gate and process -- eligibility should decay
        pfc.gate_open = False
        pfc.process(inp)
        # After gate-closed step, eligibility should be smaller
        assert np.abs(pfc._ff_eligibility).max() < np.abs(elig_before).max()

    def test_gate_closed_no_accumulation(self):
        """When gate is closed, no new eligibility should accumulate."""
        pfc = PFCRegion(
            input_dim=8,
            n_columns=4,
            n_l4=2,
            n_l23=2,
            k_columns=2,
            eligibility_decay=1.0,  # no decay
        )
        # Close gate from the start
        pfc.gate_open = False
        inp = np.ones(8) * 0.5
        pfc.process(inp)
        # No accumulation when gate is closed
        assert pfc._ff_eligibility.sum() == 0.0


# ---------------------------------------------------------------------------
# Motor three-factor learning end-to-end
# ---------------------------------------------------------------------------


class TestMotorThreeFactorEndToEnd:
    def test_motor_eligibility_accumulates_on_process(self):
        """MotorRegion.process() should accumulate eligibility, not update weights."""
        m1 = MotorRegion(
            input_dim=8,
            n_columns=4,
            n_l4=2,
            n_l23=2,
            k_columns=1,
            learning_rate=0.5,
            synapse_decay=1.0,
        )
        inp = np.zeros(8)
        inp[0] = 1.0
        m1._process_with_goal(inp)
        assert m1._ff_eligibility.sum() != 0.0

    def test_motor_reward_changes_weights(self):
        """apply_reward on MotorRegion consolidates eligibility into weights."""
        m1 = MotorRegion(
            input_dim=8,
            n_columns=4,
            n_l4=2,
            n_l23=2,
            k_columns=1,
            learning_rate=0.5,
            synapse_decay=1.0,
        )
        inp = np.zeros(8)
        inp[0] = 1.0
        m1._process_with_goal(inp)
        weights_before = m1.ff_weights.copy()
        m1.apply_reward(1.0)
        assert not np.array_equal(m1.ff_weights, weights_before)


# ---------------------------------------------------------------------------
# Config factory integration
# ---------------------------------------------------------------------------


class TestConfigFactories:
    def test_default_pfc_config_is_three_factor(self):
        from step.config import _default_pfc_config

        cfg = _default_pfc_config()
        assert cfg.plasticity_rule == PlasticityRule.THREE_FACTOR

    def test_default_motor_config_is_three_factor(self):
        from step.config import _default_motor_config

        cfg = _default_motor_config()
        assert cfg.plasticity_rule == PlasticityRule.THREE_FACTOR

    def test_default_premotor_config_is_three_factor(self):
        from step.config import _default_premotor_config

        cfg = _default_premotor_config()
        assert cfg.plasticity_rule == PlasticityRule.THREE_FACTOR

    def test_default_s1_config_is_hebbian(self):
        from step.config import _default_s1_config

        cfg = _default_s1_config()
        assert cfg.plasticity_rule == PlasticityRule.HEBBIAN

    def test_make_pfc_region_passes_rule(self):
        from step.config import _default_pfc_config, make_pfc_region

        cfg = _default_pfc_config()
        pfc = make_pfc_region(cfg, input_dim=16)
        assert pfc.plasticity_rule == PlasticityRule.THREE_FACTOR

    def test_make_motor_region_passes_rule(self):
        from step.config import _default_motor_config, make_motor_region

        cfg = _default_motor_config()
        m1 = make_motor_region(cfg, input_dim=16)
        assert m1.plasticity_rule == PlasticityRule.THREE_FACTOR

    def test_make_premotor_region_passes_rule(self):
        from step.config import _default_premotor_config, make_premotor_region

        cfg = _default_premotor_config()
        m2 = make_premotor_region(cfg, input_dim=16)
        assert m2.plasticity_rule == PlasticityRule.THREE_FACTOR

    def test_make_sensory_region_is_hebbian(self):
        from step.config import CortexConfig, make_sensory_region

        cfg = CortexConfig()
        s1 = make_sensory_region(cfg, input_dim=16)
        assert s1.plasticity_rule == PlasticityRule.HEBBIAN
