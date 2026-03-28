"""Tests for runner-probe integration (STEP-63 Phase 2).

Verifies that train() and run_cortex() correctly wire probes:
- observe() called after each circuit.process()
- boundary() called at story boundaries
- snapshot() stored in result.probe_snapshots
- LaminaProbe and ChatLaminaProbe produce real KPIs through the runner
"""

import pytest

from step.agent import ChatAgent
from step.cortex import CorticalRegion
from step.cortex.circuit import Circuit
from step.data import STORY_BOUNDARY
from step.encoders.positional import PositionalCharEncoder
from step.environment import ChatEnv
from step.probes.core import LaminaProbe
from step.runner import run_cortex
from step.train import train

# ---------------------------------------------------------------------------
# Spy probe for verifying call protocol
# ---------------------------------------------------------------------------


class SpyProbe:
    """Probe that records all calls for test assertions."""

    name: str = "spy"

    def __init__(self):
        self.observe_count = 0
        self.boundary_count = 0
        self.snapshot_count = 0
        self.last_kwargs: dict = {}

    def observe(self, circuit, **kwargs):
        self.observe_count += 1
        self.last_kwargs = kwargs

    def boundary(self):
        self.boundary_count += 1

    def snapshot(self) -> dict:
        self.snapshot_count += 1
        return {
            "observe_count": self.observe_count,
            "boundary_count": self.boundary_count,
        }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_tokens(text: str) -> list[tuple[int, str]]:
    """Convert text to (token_id, char) pairs with story boundary prefix."""
    tokens = [(STORY_BOUNDARY, "")]
    for ch in text:
        tokens.append((ord(ch), ch))
    return tokens


def _make_circuit_and_env(text: str):
    """Create a minimal circuit, encoder, agent, and env for testing."""
    encoder = PositionalCharEncoder("abcdefgh", max_positions=1)
    region = CorticalRegion(
        encoder.input_dim,
        n_columns=16,
        n_l4=4,
        n_l23=4,
        k_columns=3,
        seed=42,
    )
    circuit = Circuit(encoder)
    circuit.add_region("S1", region, entry=True)
    circuit.finalize()

    tokens = _make_tokens(text)
    env = ChatEnv(tokens)
    agent = ChatAgent(encoder=encoder, circuit=circuit)
    return env, agent, tokens


# ---------------------------------------------------------------------------
# train() integration
# ---------------------------------------------------------------------------


class TestTrainProbeWiring:
    def test_observe_called_per_step(self):
        """probe.observe() is called once per non-boundary, non-EOM step."""
        env, agent, _tokens = _make_circuit_and_env("abcdef")
        spy = SpyProbe()

        train(env, agent, log_interval=9999, probes=[spy])

        # 6 characters = 6 observe calls (boundary is skipped, not processed)
        assert spy.observe_count == 6

    def test_stimulus_id_passed(self):
        """observe() receives stimulus_id=token_id."""
        env, agent, _ = _make_circuit_and_env("ab")
        spy = SpyProbe()

        train(env, agent, log_interval=9999, probes=[spy])

        # Last token was 'b' = ord('b') = 98
        assert spy.last_kwargs["stimulus_id"] == ord("b")

    def test_boundary_called(self):
        """probe.boundary() is called at story boundaries."""
        # Two stories separated by a boundary
        tokens = [
            (STORY_BOUNDARY, ""),
            *((ord(c), c) for c in "abc"),
            (STORY_BOUNDARY, ""),
            *((ord(c), c) for c in "de"),
        ]
        encoder = PositionalCharEncoder("abcdefgh", max_positions=1)
        region = CorticalRegion(
            encoder.input_dim, n_columns=16, n_l4=4, n_l23=4, k_columns=3, seed=42
        )
        circuit = Circuit(encoder)
        circuit.add_region("S1", region, entry=True)
        circuit.finalize()
        env = ChatEnv(tokens)
        agent = ChatAgent(encoder=encoder, circuit=circuit)

        spy = SpyProbe()
        train(env, agent, log_interval=9999, probes=[spy])

        assert spy.boundary_count == 2  # two STORY_BOUNDARY tokens
        assert spy.observe_count == 5  # abc + de

    def test_snapshot_in_result(self):
        """probe.snapshot() is stored in result.probe_snapshots."""
        env, agent, _ = _make_circuit_and_env("abcdef")
        spy = SpyProbe()

        result = train(env, agent, log_interval=9999, probes=[spy])

        assert "spy" in result.probe_snapshots
        assert result.probe_snapshots["spy"]["observe_count"] == 6

    def test_multiple_probes(self):
        """Multiple probes all get called."""
        env, agent, _ = _make_circuit_and_env("abc")
        spy1 = SpyProbe()
        spy1.name = "spy1"
        spy2 = SpyProbe()
        spy2.name = "spy2"

        result = train(env, agent, log_interval=9999, probes=[spy1, spy2])

        assert spy1.observe_count == 3
        assert spy2.observe_count == 3
        assert "spy1" in result.probe_snapshots
        assert "spy2" in result.probe_snapshots

    def test_no_probes_backward_compatible(self):
        """train() works without probes (default)."""
        env, agent, _ = _make_circuit_and_env("abc")
        result = train(env, agent, log_interval=9999)
        assert result.probe_snapshots == {}

    def test_probe_without_boundary_ok(self):
        """Probes without boundary() method don't crash at boundaries."""

        class MinimalProbe:
            name = "minimal"

            def observe(self, circuit, **kwargs):
                pass

            def snapshot(self):
                return {}

        tokens = [
            (STORY_BOUNDARY, ""),
            (ord("a"), "a"),
            (STORY_BOUNDARY, ""),
            (ord("b"), "b"),
        ]
        encoder = PositionalCharEncoder("abcdefgh", max_positions=1)
        region = CorticalRegion(
            encoder.input_dim, n_columns=16, n_l4=4, n_l23=4, k_columns=3, seed=42
        )
        circuit = Circuit(encoder)
        circuit.add_region("S1", region, entry=True)
        circuit.finalize()
        env = ChatEnv(tokens)
        agent = ChatAgent(encoder=encoder, circuit=circuit)

        result = train(env, agent, log_interval=9999, probes=[MinimalProbe()])
        assert "minimal" in result.probe_snapshots


# ---------------------------------------------------------------------------
# run_cortex() integration
# ---------------------------------------------------------------------------


class TestRunCortexProbes:
    def test_probes_threaded_through_run_cortex(self):
        """run_cortex() passes probes to train(); spy records observations."""
        encoder = PositionalCharEncoder("abcdefgh", max_positions=1)
        region = CorticalRegion(
            encoder.input_dim, n_columns=16, n_l4=4, n_l23=4, k_columns=3, seed=42
        )
        tokens = _make_tokens("abcdef")
        spy = SpyProbe()

        run_cortex(region, encoder, tokens, log_interval=9999, probes=[spy])

        # Probe was called for each character
        assert spy.observe_count == 6


# ---------------------------------------------------------------------------
# End-to-end: real probes produce real KPIs through runner
# ---------------------------------------------------------------------------


class TestEndToEndProbeKPIs:
    def test_lamina_probe_via_train(self):
        """LaminaProbe produces nonzero L4 KPIs when run through train()."""
        env, agent, _ = _make_circuit_and_env("abcdefgh" * 20)
        probe = LaminaProbe(l23_sample_interval=1)

        result = train(env, agent, log_interval=9999, probes=[probe])

        snap = result.probe_snapshots["lamina"]
        assert "S1" in snap
        assert snap["S1"].l4.recall > 0
        assert snap["S1"].l4.sparseness > 0
        assert snap["S1"].l23.eff_dim > 0

    def test_chat_lamina_probe_via_train(self):
        """ChatLaminaProbe produces linear probe accuracy through train()."""
        pytest.importorskip("sklearn")
        from step.probes.chat import ChatLaminaProbe

        env, agent, _ = _make_circuit_and_env("abcdefgh" * 250)
        probe = ChatLaminaProbe(
            l23_sample_interval=1,
            linear_probe_fit_interval=500,
            linear_probe_window=1000,
        )

        result = train(env, agent, log_interval=9999, probes=[probe])

        snap = result.probe_snapshots["chat_lamina"]
        assert "S1" in snap
        # Linear probe should have attempted a fit
        assert snap["S1"].l23.linear_probe is not None
