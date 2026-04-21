"""Tests for MiniGrid circuit factories: baseline and hippocampal arms."""

import numpy as np

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
        # BG and M1 both take S1's L2/3 total as input dim.
        assert bg.input_dim == s1.n_l23_total
        assert m1.input_dim == s1.n_l23_total

    def test_processes_encoding(self):
        circuit = build_baseline_circuit(MiniGridEncoder())
        out = circuit.process(_encoding())
        assert out is not None

    def test_override_passthrough(self):
        """s1_overrides reach the SensoryRegion constructor."""
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

    def test_m1_input_dim_matches_hc_output(self):
        """M1 receives HC output in the with-HC arm; dimensions must line up."""
        circuit = build_hippocampal_circuit(MiniGridEncoder())
        hc = circuit._regions["HC"].region
        m1 = circuit._regions["M1"].region
        assert m1.input_dim == hc.output_port.n_total

    def test_bg_still_reads_s1(self):
        """Per ARB-118: BG sees raw sensory state in both arms so action
        selection is identical; HC only changes M1's feedforward drive."""
        circuit = build_hippocampal_circuit(MiniGridEncoder())
        s1 = circuit._regions["S1"].region
        bg = circuit._regions["BG"].region
        assert bg.input_dim == s1.n_l23_total

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


class TestArmsShareConfig:
    """Shared regions (S1, BG, M1) must be configured identically across arms.

    This is the ablation-integrity invariant — any difference in shared
    config would confound the 'HC vs no-HC' comparison.
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

    def test_bg_input_dim_matches(self):
        b = build_baseline_circuit(MiniGridEncoder())
        h = build_hippocampal_circuit(MiniGridEncoder())
        assert b._regions["BG"].region.input_dim == h._regions["BG"].region.input_dim
        assert b._regions["BG"].region.n_actions == h._regions["BG"].region.n_actions

    def test_m1_input_dim_matches(self):
        b = build_baseline_circuit(MiniGridEncoder())
        h = build_hippocampal_circuit(MiniGridEncoder())
        assert b._regions["M1"].region.input_dim == h._regions["M1"].region.input_dim


class TestDeferredFinalize:
    def test_baseline_finalize_false_leaves_unfinalized(self):
        circuit = build_baseline_circuit(MiniGridEncoder(), finalize=False)
        assert circuit._finalized is False

    def test_hippocampal_finalize_false_leaves_unfinalized(self):
        circuit = build_hippocampal_circuit(MiniGridEncoder(), finalize=False)
        assert circuit._finalized is False
