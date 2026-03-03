"""Wrap in-memory STEP model into the Model protocol."""

from step.config import EncoderConfig, ModelConfig
from step.model import ModelState, initial_state, learn, observe, predict


class StepMemoryModel:
    """In-memory STEP model conforming to the Model protocol.

    Delegates to the pure functions in model.py and maintains
    an inverted index of token SDRs for fast decode.
    """

    def __init__(self, model_config: ModelConfig, encoder_config: EncoderConfig):
        self.model_config = model_config
        self.encoder_config = encoder_config
        self._state: ModelState = initial_state(model_config)
        # Inverted index decode: bit_index -> [token_indices]
        self._token_ids: list[int] = []
        self._token_id_to_idx: dict[int, int] = {}
        self._inverted_index: dict[int, list[int]] = {}

    def predict_token(self, t: int) -> int:
        sdr = self.predict_sdr(t)
        return self._decode(sdr)

    def predict_sdr(self, t: int) -> frozenset[int]:
        return predict(self._state, t, self.model_config)

    def learn(
        self, t: int, actual_sdr: frozenset[int], predicted_sdr: frozenset[int]
    ) -> float:
        return learn(self._state, t, actual_sdr, predicted_sdr, self.model_config)

    def observe(self, t: int, token_id: int, sdr: frozenset[int]) -> None:
        if token_id not in self._token_id_to_idx:
            idx = len(self._token_ids)
            self._token_ids.append(token_id)
            self._token_id_to_idx[token_id] = idx
            for bit in sdr:
                self._inverted_index.setdefault(bit, []).append(idx)
        self._state = observe(self._state, t, sdr, self.model_config)

    def _decode(self, sdr: frozenset[int]) -> int:
        """Find best-matching token via inverted index overlap count."""
        if not sdr or not self._token_ids:
            return -1
        counts: dict[int, int] = {}
        for bit in sdr:
            for idx in self._inverted_index.get(bit, ()):
                counts[idx] = counts.get(idx, 0) + 1
        if not counts:
            return -1
        best_idx = max(counts, key=counts.__getitem__)
        return self._token_ids[best_idx]
