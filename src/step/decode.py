"""Shared inverted-index decode: maps activation patterns to token IDs."""


class DecodeIndex:
    """Maps sparse activation patterns to token IDs via inverted index.

    Each token is recorded as a set of active indices. To decode a query
    pattern, finds the stored token with highest overlap.
    """

    def __init__(self):
        self._token_ids: list[int] = []
        self._token_id_to_idx: dict[int, int] = {}
        self._inverted_index: dict[int, list[int]] = {}

    def observe(self, token_id: int, active_indices: frozenset[int]) -> None:
        if token_id in self._token_id_to_idx:
            return
        idx = len(self._token_ids)
        self._token_ids.append(token_id)
        self._token_id_to_idx[token_id] = idx
        for bit in active_indices:
            self._inverted_index.setdefault(bit, []).append(idx)

    def decode(self, query_indices: frozenset[int]) -> int:
        if not query_indices or not self._token_ids:
            return -1
        scores: dict[int, float] = {}
        for bit in query_indices:
            for idx in self._inverted_index.get(bit, ()):
                scores[idx] = scores.get(idx, 0.0) + 1.0
        if not scores:
            return -1
        best_idx = max(scores, key=scores.__getitem__)
        return self._token_ids[best_idx]
