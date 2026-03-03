"""Model protocol for running STEP and baselines through the same harness."""

from typing import Protocol


class Model(Protocol):
    def predict_token(self, t: int) -> int:
        """Predict the next token ID."""
        ...

    def predict_sdr(self, t: int) -> frozenset[int]:
        """Predict the next SDR."""
        ...

    def learn(
        self, t: int, actual_sdr: frozenset[int], predicted_sdr: frozenset[int]
    ) -> float:
        """Update model from prediction error. Returns IoU."""
        ...

    def observe(self, t: int, token_id: int, sdr: frozenset[int]) -> None:
        """Record a token observation."""
        ...
