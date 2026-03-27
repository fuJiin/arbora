# Shared test fixtures and helpers for STEP cortex tests.

import numpy as np

from step.agent import ChatAgent
from step.cortex import CorticalRegion
from step.cortex.circuit import Circuit
from step.cortex.circuit_types import CortexResult
from step.encoders.positional import PositionalCharEncoder
from step.environment import ChatEnv
from step.train import train


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
    show_predictions=0,
    metric_interval=0,
    babble_ratio=0.0,
) -> CortexResult:
    """Test helper: run a circuit through ChatEnv + ChatAgent + train().

    Drop-in replacement for cortex.run(tokens, ...) in tests.
    """
    agent = ChatAgent(encoder=cortex._encoder, circuit=cortex)
    env = ChatEnv(tokens, babble_ratio=babble_ratio)
    return train(
        env,
        agent,
        log_interval=log_interval,
        rolling_window=rolling_window,
        show_predictions=show_predictions,
        metric_interval=metric_interval,
    )
