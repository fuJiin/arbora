# Shared test fixtures and helpers for STEP cortex tests.

import numpy as np

from arbor.cortex import CorticalRegion
from arbor.cortex.circuit import Circuit
from arbor.encoders.positional import PositionalCharEncoder
from examples.chat.agent import ChatAgent
from examples.chat.env import ChatEnv
from examples.chat.harness import ChatTrainHarness, TrainResult


def make_circuit(n_columns=16, n_l4=4, n_l23=4, k_columns=3, seed=42):
    """Create a minimal single-region circuit for testing."""
    encoder = PositionalCharEncoder("abcdefgh", max_positions=1)
    region = CorticalRegion(
        encoder.input_dim,
        n_columns=n_columns,
        n_l4=n_l4,
        n_l23=n_l23,
        k_columns=k_columns,
        seed=seed,
    )
    circuit = Circuit(encoder)
    circuit.add_region("S1", region, entry=True)
    circuit.finalize()
    return circuit, encoder


def step_circuit(circuit, encoder, rng=None):
    """Process one random encoding through the circuit."""
    if rng is None:
        rng = np.random.default_rng(0)
    chars = "abcdefgh"
    ch = chars[rng.integers(0, len(chars))]
    encoding = encoder.encode(ch)
    circuit.process(encoding)


def run_circuit(
    cortex,
    tokens,
    *,
    log_interval=1000,
    rolling_window=100,
    probes=(),
) -> TrainResult:
    """Test helper: run a circuit via ChatTrainHarness."""
    agent = ChatAgent(encoder=cortex._encoder, circuit=cortex)
    env = ChatEnv(tokens)
    harness = ChatTrainHarness(
        env,
        agent,
        probes=probes,
        log_interval=log_interval,
        rolling_window=rolling_window,
    )
    return harness.run()
