"""Environment abstractions for the STEP training loop.

Environment — modality-agnostic protocol (reset/step/done).
ChatEnv — char-level text environment.
MiniGridEnv — gymnasium MiniGrid wrapper.
"""

from step.environment.chat import (
    BOUNDARY_OBS,
    EOM_OBS,
    ChatEnv,
    ChatObs,
    Environment,
    Observation,
)

__all__ = [
    "BOUNDARY_OBS",
    "EOM_OBS",
    "ChatEnv",
    "ChatObs",
    "Environment",
    "MiniGridEnv",
    "MiniGridObs",
    "Observation",
]


def __getattr__(name: str):
    if name in ("MiniGridEnv", "MiniGridObs"):
        from step.environment import minigrid

        return getattr(minigrid, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
