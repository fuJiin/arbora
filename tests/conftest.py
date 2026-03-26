# Shared test fixtures and helpers for STEP cortex tests.

from step.agent import ChatAgent
from step.cortex.circuit_types import CortexResult
from step.environment import ChatEnv
from step.train import train


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
