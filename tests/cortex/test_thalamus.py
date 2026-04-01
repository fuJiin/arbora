"""Tests for ThalamicNucleus — gated relay between cortical regions."""

import numpy as np

from arbora.thalamus import ThalamicNucleus


class TestThalamicNucleusUnit:
    def test_init_shapes(self):
        th = ThalamicNucleus(input_dim=32, relay_dim=16)
        assert th.input_dim == 32
        assert th.relay_dim == 16
        assert th.input_port.n_total == 32
        assert th.output_port.n_total == 16
        assert th.relay_weights.shape == (32, 16)

    def test_process_returns_correct_shape(self):
        th = ThalamicNucleus(input_dim=32, relay_dim=16)
        inp = np.random.default_rng(0).random(32)
        out = th.process(inp)
        assert out.shape == (16,)

    def test_output_port_matches_process(self):
        th = ThalamicNucleus(input_dim=32, relay_dim=16)
        inp = np.random.default_rng(0).random(32)
        out = th.process(inp)
        np.testing.assert_array_equal(out, th.output_port.firing_rate)

    def test_gate_closed_attenuates(self):
        """With no modulatory input, gate is closed → output attenuated."""
        th = ThalamicNucleus(input_dim=16, relay_dim=8)
        inp = np.ones(16)
        # No modulation → gate stays at 0 (closed)
        out_closed = th.process(inp).copy()
        # Manually open the gate via modulation
        th.input_port.add_modulation(np.ones(16) * 1.0)
        out_open = th.process(inp).copy()
        # Open gate should produce stronger output
        assert out_open.sum() > out_closed.sum()

    def test_gate_modulation_opens_relay(self):
        """Positive modulation opens the gate, increasing relay output."""
        th = ThalamicNucleus(input_dim=16, relay_dim=8, gate_threshold=0.0)
        inp = np.ones(16) * 0.5
        # Step 1: closed
        out1 = th.process(inp).copy()
        # Step 2: open via positive modulation
        th.input_port.add_modulation(np.ones(16) * 2.0)
        out2 = th.process(inp).copy()
        assert out2.sum() > out1.sum() * 2  # should be much stronger

    def test_burst_on_gate_reopen(self):
        """Gate reopening after closure triggers burst (amplified output)."""
        th = ThalamicNucleus(
            input_dim=16, relay_dim=8,
            burst_gain=3.0, gate_threshold=0.0,
        )
        inp = np.ones(16) * 0.5
        # Open the gate for several steps to establish tonic mode
        for _ in range(5):
            th.input_port.add_modulation(np.ones(16) * 1.0)
            th.process(inp)
        tonic_out = th.process(inp).copy()  # no modulation → will decay
        # Let gate close
        for _ in range(10):
            th.process(inp)
        # Reopen → should burst
        th.input_port.add_modulation(np.ones(16) * 2.0)
        burst_out = th.process(inp).copy()
        # Burst output should exceed tonic output
        assert burst_out.sum() > tonic_out.sum()

    def test_tonic_mode_faithful_relay(self):
        """In tonic mode (gate open, no burst), relay is proportional to input."""
        th = ThalamicNucleus(input_dim=16, relay_dim=8, gate_threshold=-1.0)
        # Gate threshold -1 means gate is always open
        inp_weak = np.ones(16) * 0.2
        inp_strong = np.ones(16) * 0.8
        out_weak = th.process(inp_weak).copy()
        out_strong = th.process(inp_strong).copy()
        # Stronger input → stronger output
        assert out_strong.sum() > out_weak.sum()

    def test_reset_clears_transient_state(self):
        th = ThalamicNucleus(input_dim=16, relay_dim=8)
        th.input_port.add_modulation(np.ones(16))
        th.process(np.ones(16))
        assert th._gate != 0.0
        th.reset_working_memory()
        assert th._gate == 0.0
        assert th._prev_gate == 0.0
        assert not th._in_burst
        assert th.output_port.firing_rate.sum() == 0.0

    def test_relay_weights_learn(self):
        """Hebbian learning updates relay weights when gate is open."""
        th = ThalamicNucleus(
            input_dim=16, relay_dim=8,
            learning_rate=0.1, gate_threshold=-1.0,
        )
        initial_weights = th.relay_weights.copy()
        inp = np.ones(16) * 0.5
        for _ in range(50):
            th.process(inp)
        # Weights should have changed
        assert not np.allclose(th.relay_weights, initial_weights)
        # Weights stay in [0, 1]
        assert th.relay_weights.min() >= 0.0
        assert th.relay_weights.max() <= 1.0

    def test_get_lamina(self):
        th = ThalamicNucleus(input_dim=16, relay_dim=8)
        assert th.get_lamina(ThalamicNucleus.RELAY_IN) is th.input_port
        assert th.get_lamina(ThalamicNucleus.RELAY_OUT) is th.output_port

    def test_apply_reward_is_noop(self):
        th = ThalamicNucleus(input_dim=16, relay_dim=8)
        weights_before = th.relay_weights.copy()
        th.apply_reward(1.0)
        th.apply_reward(-1.0)
        np.testing.assert_array_equal(th.relay_weights, weights_before)


class TestThalamicNucleusCircuitIntegration:
    def test_thalamus_in_circuit(self):
        """ThalamicNucleus wired between V1 and a downstream region."""
        from arbora.cortex import SensoryRegion
        from arbora.cortex.circuit import Circuit, ConnectionRole
        from arbora.encoders.charbit import CharbitEncoder

        encoder = CharbitEncoder(length=4, width=5, chars="abcd")
        v1 = SensoryRegion(
            input_dim=4 * 5, encoding_width=5,
            n_columns=8, n_l4=2, n_l23=2, n_l5=2, k_columns=2, seed=42,
        )
        th = ThalamicNucleus(
            input_dim=v1.n_l5_total, relay_dim=16, seed=0,
        )
        v2 = SensoryRegion(
            input_dim=th.relay_dim, n_columns=4, n_l4=2, n_l23=2,
            k_columns=1, seed=99,
        )

        circuit = Circuit(encoder)
        circuit.add_region("V1", v1, entry=True)
        circuit.add_region("thal", th)
        circuit.add_region("V2", v2)

        # V1 L5 → thalamus → V2
        circuit.connect(v1.l5, th.input_port, ConnectionRole.FEEDFORWARD)
        circuit.connect(th.output_port, v2.l4, ConnectionRole.FEEDFORWARD)
        # V2 → thalamus (modulatory gate)
        circuit.connect(v2.l23, th.input_port, ConnectionRole.MODULATORY)

        circuit.finalize()

        # Process a few steps
        for ch in "abcd":
            enc = encoder.encode(ch)
            out = circuit.process(enc)
            assert out is not None

        # V2 should have activations
        assert v2.active_columns.sum() > 0

    def test_thalamus_with_bg_gate(self):
        """Motor thalamus: BG gates the relay to M1."""
        from arbora.basal_ganglia import BasalGangliaRegion
        from arbora.cortex import SensoryRegion
        from arbora.cortex.circuit import Circuit, ConnectionRole
        from arbora.cortex.motor import MotorRegion
        from arbora.encoders.charbit import CharbitEncoder

        encoder = CharbitEncoder(length=4, width=5, chars="abcd")
        v1 = SensoryRegion(
            input_dim=4 * 5, encoding_width=5,
            n_columns=8, n_l4=2, n_l23=2, n_l5=2, k_columns=2, seed=42,
        )
        bg = BasalGangliaRegion(input_dim=v1.n_l5_total, n_actions=4, seed=100)
        motor_thal = ThalamicNucleus(
            input_dim=v1.n_l23_total, relay_dim=8, seed=200,
        )
        m1 = MotorRegion(
            input_dim=motor_thal.relay_dim,
            n_columns=4, n_l4=0, n_l23=2, k_columns=1,
            n_output_tokens=4, seed=300,
        )

        circuit = Circuit(encoder)
        circuit.add_region("V1", v1, entry=True)
        circuit.add_region("BG", bg)
        circuit.add_region("motor_thal", motor_thal)
        circuit.add_region("M1", m1)

        # V1 L5 → BG (saliency)
        circuit.connect(v1.l5, bg.input_port, ConnectionRole.FEEDFORWARD)
        # V1 L2/3 → motor_thal (driver)
        circuit.connect(v1.l23, motor_thal.input_port, ConnectionRole.FEEDFORWARD)
        # BG → motor_thal (gate)
        circuit.connect(bg.output_port, motor_thal.input_port, ConnectionRole.MODULATORY)
        # motor_thal → M1
        circuit.connect(motor_thal.output_port, m1.input_port, ConnectionRole.FEEDFORWARD)

        circuit.finalize()

        for ch in "abcd":
            enc = encoder.encode(ch)
            circuit.process(enc, motor_active=True)

        assert m1.active_columns.sum() > 0
