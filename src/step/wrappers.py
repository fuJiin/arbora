"""Wrap in-memory STEP model into the Model protocol."""

from step.config import EncoderConfig, ModelConfig
from step.data import STORY_BOUNDARY
from step.model import (
    ModelState,
    initial_state,
    learn,
    observe,
    predict,
    predict_with_vector,
)
from step.sdr import AdaptiveEncoder


class StepMemoryModel:
    """In-memory STEP model conforming to the Model protocol.

    Delegates to the pure functions in model.py and maintains
    an inverted index of token SDRs for fast decode.
    """

    def __init__(self, model_config: ModelConfig, encoder_config: EncoderConfig):
        self.model_config = model_config
        self.encoder_config = encoder_config
        self._state: ModelState = initial_state(model_config)
        self._encoder: AdaptiveEncoder | None = None
        if encoder_config.adaptive:
            self._encoder = AdaptiveEncoder(encoder_config)
        # Inverted index decode: bit_index -> [token_indices]
        self._token_ids: list[int] = []
        self._token_id_to_idx: dict[int, int] = {}
        self._inverted_index: dict[int, list[int]] = {}

    def encode_token_sdr(self, token_id: int, t: int = -1) -> frozenset[int]:
        """Encode a token using adaptive encoder (if enabled) or hash-based."""
        if self._encoder is not None:
            # Only compute context for NEW tokens
            if token_id not in self._encoder._token_sdrs:
                context = self._get_seeding_context(t)
            else:
                context = None
            return self._encoder.encode(token_id, context)
        from step.sdr import encode_token

        return encode_token(token_id, self.encoder_config)

    def _get_seeding_context(self, t: int) -> list[int] | None:
        """Get context bits for seeding a new token's SDR."""
        if self.encoder_config.seeding == "predicted" and t >= 0:
            # Use model's prediction as context — seed from what was EXPECTED
            predicted = predict(self._state, t, self.model_config)
            return list(predicted) if predicted else None
        # Default "active": use all bits from the eligibility window
        bits: list[int] = []
        for sdr in self._state.history.values():
            bits.extend(sdr)
        return bits or None

    def predict_token(self, t: int) -> int:
        sdr, vector = predict_with_vector(self._state, t, self.model_config)
        return self._decode(sdr, vector)

    def predict_sdr(self, t: int) -> frozenset[int]:
        return predict(self._state, t, self.model_config)

    def learn(
        self, t: int, actual_sdr: frozenset[int], predicted_sdr: frozenset[int]
    ) -> float:
        return learn(self._state, t, actual_sdr, predicted_sdr, self.model_config)

    def observe(self, t: int, token_id: int, sdr: frozenset[int]) -> None:
        if token_id == STORY_BOUNDARY:
            self._state.history.clear()
            return
        if token_id not in self._token_id_to_idx:
            idx = len(self._token_ids)
            self._token_ids.append(token_id)
            self._token_id_to_idx[token_id] = idx
            for bit in sdr:
                self._inverted_index.setdefault(bit, []).append(idx)
        self._state = observe(self._state, t, sdr, self.model_config)

    def _decode(self, sdr: frozenset[int], vector=None) -> int:
        """Find best-matching token via inverted index.

        If vector is provided, weights each bit's contribution by
        prediction strength (weight-aware decode). Otherwise falls
        back to simple overlap counting.
        """
        if not sdr or not self._token_ids:
            return -1
        scores: dict[int, float] = {}
        for bit in sdr:
            w = float(vector[bit]) if vector is not None else 1.0
            for idx in self._inverted_index.get(bit, ()):
                scores[idx] = scores.get(idx, 0.0) + w
        if not scores:
            return -1
        best_idx = max(scores, key=scores.__getitem__)
        return self._token_ids[best_idx]
