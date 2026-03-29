"""Tests for BasalGangliaRegion — per-action Go/NoGo with tonic DA."""

import numpy as np

from step.cortex.basal_ganglia import BasalGangliaRegion


class TestBasalGangliaRegionUnit:
    def test_init_shapes(self):
        bg = BasalGangliaRegion(input_dim=32, n_actions=7)
        assert bg.go_weights.shape == (32, 7)
        assert bg.nogo_weights.shape == (32, 7)
        assert bg.input_dim == 32
        assert bg.n_actions == 7

    def test_process_returns_correct_shape(self):
        bg = BasalGangliaRegion(input_dim=32, n_actions=7)
        inp = np.random.default_rng(0).random(32)
        bias = bg.process(inp)
        assert bias.shape == (7,)
        assert all(0 <= b <= 1 for b in bias)

    def test_output_port_firing_rate_matches(self):
        bg = BasalGangliaRegion(input_dim=16, n_actions=4)
        inp = np.random.default_rng(0).random(16)
        bias = bg.process(inp)
        np.testing.assert_array_equal(bias, bg.output_port.firing_rate)

    def test_positive_reward_strengthens_go(self):
        bg = BasalGangliaRegion(input_dim=16, n_actions=4, learning_rate=0.1)
        inp = np.ones(16) * 0.5
        initial_bias = bg.process(inp).copy()
        for _ in range(100):
            bg.process(inp)
            bg.apply_reward(1.0)
        final_bias = bg.process(inp)
        assert final_bias.mean() > initial_bias.mean()

    def test_negative_reward_strengthens_nogo(self):
        bg = BasalGangliaRegion(input_dim=16, n_actions=4, learning_rate=0.1)
        inp = np.ones(16) * 0.5
        for _ in range(100):
            bg.process(inp)
            bg.apply_reward(-1.0)
        bias = bg.process(inp)
        # NoGo should dominate -> bias below 0.5
        assert bias.mean() < 0.5

    def test_different_contexts_different_bias(self):
        bg = BasalGangliaRegion(input_dim=16, n_actions=4, learning_rate=0.05)
        inp_a = np.zeros(16)
        inp_a[:8] = 1.0
        inp_b = np.zeros(16)
        inp_b[8:] = 1.0
        for _ in range(200):
            bg.process(inp_a)
            bg.apply_reward(1.0)
            bg.process(inp_b)
            bg.apply_reward(-1.0)
        bias_a = bg.process(inp_a)
        bias_b = bg.process(inp_b)
        assert bias_a.mean() > bias_b.mean()

    def test_tonic_da_tracks_uncertainty(self):
        bg = BasalGangliaRegion(input_dim=16, n_actions=4, tonic_da_init=0.1)
        inp = np.ones(16)
        # High variance rewards -> high tonic DA
        for i in range(200):
            bg.process(inp)
            bg.apply_reward(1.0 if i % 2 == 0 else -1.0)
        high_var_da = bg._tonic_da
        # Low variance rewards -> low tonic DA
        bg2 = BasalGangliaRegion(input_dim=16, n_actions=4, tonic_da_init=0.1)
        for _ in range(200):
            bg2.process(inp)
            bg2.apply_reward(0.5)
        low_var_da = bg2._tonic_da
        assert high_var_da > low_var_da

    def test_reset_clears_traces(self):
        bg = BasalGangliaRegion(input_dim=16, n_actions=4)
        bg.process(np.ones(16))
        bg.apply_reward(1.0)
        bg.reset_working_memory()
        assert bg._go_trace.sum() == 0.0
        assert bg._nogo_trace.sum() == 0.0

    def test_get_lamina(self):
        from step.cortex.lamina import LaminaID

        bg = BasalGangliaRegion(input_dim=16, n_actions=4)
        assert bg.get_lamina(LaminaID.L4) is bg.input_port
        assert bg.get_lamina(LaminaID.L23) is bg.output_port

    def test_input_port_dims(self):
        bg = BasalGangliaRegion(input_dim=32, n_actions=7)
        assert bg.input_port.n_total == 32

    def test_output_port_dims(self):
        bg = BasalGangliaRegion(input_dim=32, n_actions=7)
        assert bg.output_port.n_total == 7


class TestBasalGangliaCircuitIntegration:
    def test_bg_as_region_in_circuit(self):
        """BG wired as proper region with MODULATORY output to M1."""
        from step.cortex import SensoryRegion
        from step.cortex.circuit import Circuit, ConnectionRole
        from step.cortex.motor import MotorRegion
        from step.encoders.charbit import CharbitEncoder

        encoder = CharbitEncoder(length=4, width=5, chars="abcd")
        s1 = SensoryRegion(
            input_dim=4 * 5,
            encoding_width=5,
            n_columns=8,
            n_l4=2,
            n_l23=2,
            k_columns=2,
            seed=42,
        )
        bg = BasalGangliaRegion(input_dim=s1.n_l23_total, n_actions=7, seed=789)
        m1 = MotorRegion(
            input_dim=s1.n_l23_total,
            n_columns=4,
            n_l4=2,
            n_l23=2,
            k_columns=1,
            seed=456,
        )
        circuit = Circuit(encoder)
        circuit.add_region("S1", s1, entry=True)
        circuit.add_region("BG", bg)
        circuit.add_region("M1", m1)
        circuit.connect(s1.l23, bg.input_port, ConnectionRole.FEEDFORWARD)
        circuit.connect(s1.l23, m1.l4, ConnectionRole.FEEDFORWARD)
        circuit.connect(bg.output_port, m1.l4, ConnectionRole.MODULATORY)
        circuit.finalize()

        # Process should not crash
        enc = encoder.encode("ab")
        circuit.process(enc, motor_active=True)

    def test_reward_routes_to_bg(self):
        from step.cortex import SensoryRegion
        from step.cortex.circuit import Circuit, ConnectionRole
        from step.cortex.motor import MotorRegion
        from step.encoders.charbit import CharbitEncoder

        encoder = CharbitEncoder(length=4, width=5, chars="abcd")
        s1 = SensoryRegion(
            input_dim=4 * 5,
            encoding_width=5,
            n_columns=8,
            n_l4=2,
            n_l23=2,
            k_columns=2,
            seed=42,
        )
        bg = BasalGangliaRegion(
            input_dim=s1.n_l23_total,
            n_actions=7,
            learning_rate=0.1,
            seed=789,
        )
        m1 = MotorRegion(
            input_dim=s1.n_l23_total,
            n_columns=4,
            n_l4=2,
            n_l23=2,
            k_columns=1,
            seed=456,
        )
        circuit = Circuit(encoder)
        circuit.add_region("S1", s1, entry=True)
        circuit.add_region("BG", bg)
        circuit.add_region("M1", m1)
        circuit.connect(s1.l23, bg.input_port, ConnectionRole.FEEDFORWARD)
        circuit.connect(s1.l23, m1.l4, ConnectionRole.FEEDFORWARD)
        circuit.connect(bg.output_port, m1.l4, ConnectionRole.MODULATORY)
        circuit.finalize()

        # Process a few steps so traces accumulate
        for _ in range(5):
            circuit.process(encoder.encode("ab"), motor_active=True)

        w_before = bg.go_weights.copy()
        circuit.apply_reward(1.0)
        assert not np.array_equal(bg.go_weights, w_before)
