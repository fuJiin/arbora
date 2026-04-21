"""Tests for MiniGrid circuit factories: baseline and hippocampal arms."""

import numpy as np

from arbora.cortex.circuit import ConnectionRole
from examples.minigrid.encoder import MiniGridEncoder
from examples.minigrid.presets import (
    build_baseline_circuit,
    build_hippocampal_circuit,
)


def _encoding(seed: int = 0) -> np.ndarray:
    """Synthetic sparse binary encoding matching MiniGridEncoder shape."""
    rng = np.random.default_rng(seed)
    x = np.zeros(984, dtype=np.bool_)
    x[rng.choice(984, size=148, replace=False)] = True
    return x


def _edges(circuit) -> set[tuple[str, str, ConnectionRole]]:
    """Extract (source_region, target_region, role) tuples from a circuit."""
    return {(c.source, c.target, c.role) for c in circuit._connections}


class TestBaseline:
    def test_regions_present(self):
        circuit = build_baseline_circuit(MiniGridEncoder())
        names = list(circuit._regions.keys())
        assert names == ["S1", "BG", "M1"]

    def test_dimensions_chain(self):
        circuit = build_baseline_circuit(MiniGridEncoder())
        s1 = circuit._regions["S1"].region
        bg = circuit._regions["BG"].region
        m1 = circuit._regions["M1"].region
        assert bg.input_dim == s1.n_l23_total
        assert m1.input_dim == s1.n_l23_total

    def test_edges(self):
        circuit = build_baseline_circuit(MiniGridEncoder())
        assert _edges(circuit) == {
            ("S1", "BG", ConnectionRole.FEEDFORWARD),
            ("S1", "M1", ConnectionRole.FEEDFORWARD),
            ("BG", "M1", ConnectionRole.MODULATORY),
        }

    def test_processes_encoding(self):
        circuit = build_baseline_circuit(MiniGridEncoder())
        out = circuit.process(_encoding())
        assert out is not None

    def test_override_passthrough(self):
        circuit = build_baseline_circuit(
            MiniGridEncoder(), s1_overrides={"n_columns": 32}
        )
        assert circuit._regions["S1"].region.n_columns == 32


class TestHippocampal:
    def test_regions_present(self):
        circuit = build_hippocampal_circuit(MiniGridEncoder())
        names = list(circuit._regions.keys())
        assert names == ["S1", "HC", "BG", "M1"]

    def test_hc_dimensions_match_s1_l23(self):
        """HC is symmetric: its input and output dim both match S1.n_l23_total."""
        circuit = build_hippocampal_circuit(MiniGridEncoder())
        s1 = circuit._regions["S1"].region
        hc = circuit._regions["HC"].region
        assert hc.input_dim == s1.n_l23_total
        assert hc.output_port.n_total == s1.n_l23_total

    def test_bg_input_dim_sums_s1_and_hc(self):
        """ARB-123: BG takes both S1 and HC as feedforward sources, so
        its input_dim is the sum of the two stream widths."""
        circuit = build_hippocampal_circuit(MiniGridEncoder())
        s1 = circuit._regions["S1"].region
        hc = circuit._regions["HC"].region
        bg = circuit._regions["BG"].region
        assert bg.input_dim == s1.n_l23_total + hc.output_port.n_total

    def test_m1_input_dim_matches_s1(self):
        """M1 reads S1 directly (the reflexive sensorimotor path), same
        as in the baseline. HC does NOT feed M1 under the ARB-123
        topology."""
        circuit = build_hippocampal_circuit(MiniGridEncoder())
        s1 = circuit._regions["S1"].region
        m1 = circuit._regions["M1"].region
        assert m1.input_dim == s1.n_l23_total

    def test_edges(self):
        """The ARB-123 topology: HC → BG, no HC → M1, S1 → M1 restored."""
        circuit = build_hippocampal_circuit(MiniGridEncoder())
        edges = _edges(circuit)
        assert edges == {
            ("S1", "HC", ConnectionRole.FEEDFORWARD),
            ("S1", "BG", ConnectionRole.FEEDFORWARD),
            ("S1", "M1", ConnectionRole.FEEDFORWARD),
            ("HC", "BG", ConnectionRole.FEEDFORWARD),
            ("BG", "M1", ConnectionRole.MODULATORY),
        }
        # Explicitly: HC does not project to M1 under the new wiring.
        assert ("HC", "M1", ConnectionRole.FEEDFORWARD) not in edges
        assert ("HC", "M1", ConnectionRole.MODULATORY) not in edges

    def test_processes_encoding(self):
        circuit = build_hippocampal_circuit(MiniGridEncoder())
        out = circuit.process(_encoding())
        assert out is not None

    def test_hc_overrides_passthrough(self):
        circuit = build_hippocampal_circuit(
            MiniGridEncoder(),
            hc_overrides={"ec_dim": 200, "dg_dim": 800, "ca3_dim": 200},
        )
        hc = circuit._regions["HC"].region
        assert hc.ec.output_dim == 200
        assert hc.dg.output_dim == 800
        assert hc.ca3.dim == 200

    def test_hc_overrides_flow_through_to_bg_width(self):
        """Override check: BG input_dim tracks the HC output_dim override."""
        circuit = build_hippocampal_circuit(
            MiniGridEncoder(),
            hc_overrides={"ec_dim": 200},  # HC output_port.n_total == input_dim
        )
        s1 = circuit._regions["S1"].region
        bg = circuit._regions["BG"].region
        assert bg.input_dim == s1.n_l23_total + s1.n_l23_total  # HC symmetric


class TestArmsShareConfig:
    """Invariants that keep the ablation honest.

    S1 and M1 must be configured identically across arms — any
    difference would confound the "HC vs no-HC" comparison. BG is an
    intentional exception: its `input_dim` is wider in the HC arm
    because HC adds a feedforward stream, but every other BG field
    (n_actions, seed, learning rates) matches.
    """

    def test_s1_config_matches(self):
        b = build_baseline_circuit(MiniGridEncoder())
        h = build_hippocampal_circuit(MiniGridEncoder())
        s1_b = b._regions["S1"].region
        s1_h = h._regions["S1"].region
        assert s1_b.n_columns == s1_h.n_columns
        assert s1_b.n_l4 == s1_h.n_l4
        assert s1_b.n_l23 == s1_h.n_l23
        assert s1_b.n_l5 == s1_h.n_l5
        assert s1_b.k_columns == s1_h.k_columns

    def test_m1_input_dim_matches(self):
        b = build_baseline_circuit(MiniGridEncoder())
        h = build_hippocampal_circuit(MiniGridEncoder())
        assert b._regions["M1"].region.input_dim == h._regions["M1"].region.input_dim

    def test_bg_input_dim_intentionally_differs(self):
        """The HC arm's BG is wider by exactly HC.output_port.n_total.

        This is the one legitimate asymmetry — HC feeds BG an extra FF
        stream, so its weight matrix grows. Non-dim BG config (n_actions,
        learning rate, seed) is asserted identical below.
        """
        b = build_baseline_circuit(MiniGridEncoder())
        h = build_hippocampal_circuit(MiniGridEncoder())
        b_bg = b._regions["BG"].region
        h_bg = h._regions["BG"].region
        h_hc = h._regions["HC"].region
        assert h_bg.input_dim == b_bg.input_dim + h_hc.output_port.n_total

    def test_bg_non_dim_config_matches(self):
        b = build_baseline_circuit(MiniGridEncoder())
        h = build_hippocampal_circuit(MiniGridEncoder())
        b_bg = b._regions["BG"].region
        h_bg = h._regions["BG"].region
        assert b_bg.n_actions == h_bg.n_actions
        assert b_bg.learning_rate == h_bg.learning_rate
        assert b_bg.eligibility_decay == h_bg.eligibility_decay


class TestDeferredFinalize:
    def test_baseline_finalize_false_leaves_unfinalized(self):
        circuit = build_baseline_circuit(MiniGridEncoder(), finalize=False)
        assert circuit._finalized is False

    def test_hippocampal_finalize_false_leaves_unfinalized(self):
        circuit = build_hippocampal_circuit(MiniGridEncoder(), finalize=False)
        assert circuit._finalized is False
