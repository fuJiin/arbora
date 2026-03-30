"""Agent abstractions for the STEP training loop.

Agent — modality-agnostic protocol (act(obs, reward) -> action).
ChatAgent — concrete implementation for char-level text chat.
MiniGridAgent — concrete implementation for MiniGrid gymnasium envs.
"""

from arbor.agent.base import BaseAgent
from arbor.agent.chat import Agent, ChatAgent

__all__ = ["Agent", "BaseAgent", "ChatAgent", "MiniGridAgent"]


def __getattr__(name: str):
    if name == "MiniGridAgent":
        from arbor.agent.minigrid import MiniGridAgent

        return MiniGridAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
