"""Minimal causal transformer for baseline comparison.

A from-scratch ~100 line implementation (not nanoGPT).
Full implementation deferred to a future session.
"""

from dataclasses import dataclass


@dataclass
class MiniGPTConfig:
    vocab_size: int = 50257  # GPT-2 vocab
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 2
    block_size: int = 101  # Match STEP's eligibility window
    dropout: float = 0.1


# TODO: Implement MiniGPT
# - CausalSelfAttention (masked multi-head attention)
# - MLP (feed-forward block)
# - TransformerBlock (attention + MLP + layer norms)
# - MiniGPT (embedding + blocks + LM head)
