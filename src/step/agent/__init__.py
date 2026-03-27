"""Agent abstractions for the STEP training loop.

Agent — modality-agnostic protocol (act(obs, reward) -> action).
ChatAgent — concrete implementation for char-level text chat.
"""

from step.agent.chat import Agent, ChatAgent

__all__ = ["Agent", "ChatAgent"]
