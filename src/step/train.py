"""Backward-compatible train() function.

Delegates to ChatTrainHarness. Prefer using ChatTrainHarness directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from step.harness.chat.train import TrainResult
from step.probes.core import Probe

if TYPE_CHECKING:
    from step.agent import ChatAgent
    from step.environment import ChatEnv

# Re-export for backward compatibility
__all__ = ["TrainResult", "train"]


def train(
    env: ChatEnv,
    agent: ChatAgent,
    *,
    log_interval: int = 100,
    rolling_window: int = 100,
    probes: Sequence[Probe] = (),
    decoder_training: bool = False,
) -> TrainResult:
    """Run training loop. Thin wrapper around ChatTrainHarness.run()."""
    from step.harness.chat.train import ChatTrainHarness

    harness = ChatTrainHarness(
        env,
        agent,
        probes=probes,
        log_interval=log_interval,
        rolling_window=rolling_window,
        decoder_training=decoder_training,
    )
    return harness.run()
