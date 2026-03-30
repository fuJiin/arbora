"""Training harnesses for STEP circuits.

Harnesses wrap pure neural Circuit computation with environment
orchestration, probe telemetry, and reporting.
"""

from arbor.harness.chat import ChatTrainHarness

__all__ = ["ChatTrainHarness", "MiniGridHarness"]


def __getattr__(name: str):
    if name == "MiniGridHarness":
        from arbor.harness.minigrid import MiniGridHarness

        return MiniGridHarness
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
