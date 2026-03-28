"""Agent abstractions for the STEP training loop.

Agent — modality-agnostic protocol (act(obs, reward) -> action).
ChatAgent — concrete implementation for char-level text chat.
MiniGridAgent — concrete implementation for MiniGrid gymnasium envs.
"""

from step.agent.chat import Agent, ChatAgent
from step.agent.minigrid import MiniGridAgent

__all__ = ["Agent", "ChatAgent", "MiniGridAgent"]
