"""Wrap MiniGPT (nn.Module) into the Model protocol."""

import torch
import torch.nn.functional as F

from baselines.mini_gpt import MiniGPT, MiniGPTConfig


class MiniGPTModel:
    """MiniGPT conforming to the Model protocol.

    Maintains a rolling context buffer of token IDs (truncated to block_size).
    Inference-only during eval: learn() computes cross-entropy loss but does
    not update weights.
    """

    def __init__(self, model: MiniGPT, config: MiniGPTConfig):
        self.model = model
        self.config = config
        self._context: list[int] = []

    def predict_token(self, t: int) -> int:
        if not self._context:
            return -1
        self.model.eval()
        with torch.no_grad():
            idx = torch.tensor([self._context], dtype=torch.long)
            logits = self.model(idx)
            return int(logits[0, -1].argmax().item())

    def predict_sdr(self, t: int) -> frozenset[int]:
        return frozenset()

    def learn(
        self, t: int, actual_sdr: frozenset[int], predicted_sdr: frozenset[int]
    ) -> float:
        """Compute cross-entropy loss as a metric. Does NOT update weights."""
        if len(self._context) < 2:
            return 0.0
        self.model.eval()
        with torch.no_grad():
            idx = torch.tensor([self._context], dtype=torch.long)
            logits = self.model(idx)
            # Loss over all positions: predict next token from each position
            loss = F.cross_entropy(
                logits[0, :-1], torch.tensor(self._context[1:], dtype=torch.long)
            )
            return float(loss.item())

    def observe(self, t: int, token_id: int, sdr: frozenset[int]) -> None:
        self._context.append(token_id)
        if len(self._context) > self.config.block_size:
            self._context = self._context[-self.config.block_size :]
