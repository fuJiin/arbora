"""Tests for Lamina state container and LaminaID enum."""

from step.cortex.lamina import Lamina, LaminaID


class TestLaminaID:
    def test_values(self):
        assert LaminaID.L4.value == "L4"
        assert LaminaID.L23.value == "L2/3"
        assert LaminaID.L5.value == "L5"

    def test_all_members(self):
        assert set(LaminaID) == {LaminaID.L4, LaminaID.L23, LaminaID.L5}


class TestLamina:
    def test_dimensions(self):
        lam = Lamina(8, 4)
        assert lam.n_per_col == 4
        assert lam.n_columns == 8
        assert lam.n_total == 32

    def test_all_features_enabled(self):
        lam = Lamina(8, 4)
        assert lam.active.shape == (32,)
        assert lam.predicted.shape == (32,)
        assert lam.voltage.shape == (32,)
        assert lam.excitability.shape == (32,)
        assert lam.trace.shape == (32,)
        assert lam.firing_rate.shape == (32,)

    def test_optional_features_disabled(self):
        lam = Lamina(
            8,
            4,
            has_voltage=False,
            has_excitability=False,
            has_trace=False,
            has_firing_rate=False,
        )
        assert lam.active is not None  # always present
        assert lam.predicted is not None  # always present
        assert lam.voltage is None
        assert lam.excitability is None
        assert lam.trace is None
        assert lam.firing_rate is None

    def test_l4_config(self):
        """L4: voltage + excitability + trace, no firing rate."""
        lam = Lamina(8, 4, has_firing_rate=False)
        assert lam.voltage is not None
        assert lam.excitability is not None
        assert lam.trace is not None
        assert lam.firing_rate is None

    def test_l5_config(self):
        """L5: firing rate only, no voltage/excitability/trace."""
        lam = Lamina(
            8,
            4,
            has_voltage=False,
            has_excitability=False,
            has_trace=False,
            has_firing_rate=True,
        )
        assert lam.voltage is None
        assert lam.firing_rate is not None

    def test_reset(self):
        lam = Lamina(8, 4)
        lam.active[0] = True
        lam.predicted[1] = True
        lam.voltage[2] = 0.5
        lam.excitability[3] = 0.1
        lam.trace[4] = 0.3
        lam.firing_rate[5] = 0.7
        lam.reset()
        assert not lam.active.any()
        assert not lam.predicted.any()
        assert lam.voltage.sum() == 0.0
        assert lam.excitability.sum() == 0.0
        assert lam.trace.sum() == 0.0
        assert lam.firing_rate.sum() == 0.0

    def test_reset_with_disabled_features(self):
        lam = Lamina(8, 4, has_voltage=False, has_trace=False)
        lam.active[0] = True
        lam.firing_rate[1] = 0.5
        lam.reset()  # should not crash
        assert not lam.active.any()
        assert lam.firing_rate.sum() == 0.0


class TestConnectionLaminaFields:
    """Verify Connection has source_lamina/target_lamina fields."""

    def test_connection_defaults(self):
        from step.cortex.topology import Connection, ConnectionRole

        conn = Connection(source="S1", target="S2", role=ConnectionRole.FEEDFORWARD)
        assert conn.source_lamina == LaminaID.L23
        assert conn.target_lamina == LaminaID.L4

    def test_connection_custom_lamina(self):
        from step.cortex.topology import Connection, ConnectionRole

        conn = Connection(
            source="S1",
            target="S2",
            role=ConnectionRole.FEEDFORWARD,
            source_lamina=LaminaID.L5,
            target_lamina=LaminaID.L4,
        )
        assert conn.source_lamina == LaminaID.L5

    def test_connect_api_accepts_lamina(self):
        from step.cortex.region import CorticalRegion
        from step.cortex.topology import ConnectionRole, Topology
        from step.encoders.positional import PositionalCharEncoder

        encoder = PositionalCharEncoder("abc", max_positions=4)
        r1 = CorticalRegion(
            input_dim=encoder.input_dim,
            n_columns=8,
            n_l4=4,
            n_l23=4,
            k_columns=2,
        )
        r2 = CorticalRegion(
            input_dim=r1.n_l23_total,
            n_columns=8,
            n_l4=4,
            n_l23=4,
            k_columns=2,
        )
        cortex = Topology(encoder)
        cortex.add_region("S1", r1, entry=True)
        cortex.add_region("S2", r2)
        cortex.connect(
            "S1",
            "S2",
            ConnectionRole.FEEDFORWARD,
            source_lamina=LaminaID.L23,
            target_lamina=LaminaID.L4,
        )
        conn = cortex._connections[0]
        assert conn.source_lamina == LaminaID.L23
        assert conn.target_lamina == LaminaID.L4
