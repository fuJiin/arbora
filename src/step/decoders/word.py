"""Word-level decoder for S2: maps L2/3 patterns to words.

Wraps DendriticDecoder to operate at word boundaries. Accumulates
characters, and when a boundary is hit (space, punctuation), records
the completed word against S2's L2/3 state.

Lets us see what words S2 is "thinking about" at any moment —
qualitative insight into the context S2 sends to S1 via apical.
"""

import numpy as np

from step.decoders.dendritic import DendriticDecoder

_BOUNDARY_CHARS = frozenset(" .!?'-,\n")


class WordDecoder:
    """Word-level decoder for a region's L2/3 state.

    Usage:
        decoder = WordDecoder(source_dim=region.n_l23_total)
        # Each step:
        decoder.step(char, region.l23.firing_rate)
        # To read:
        decoder.predict(region.l23.firing_rate, k=3)
    """

    def __init__(
        self,
        source_dim: int,
        *,
        min_word_len: int = 2,
        max_vocab: int = 500,
        seed: int = 0,
    ):
        self._decoder = DendriticDecoder(
            source_dim,
            n_segments=4,
            n_synapses=24,
            seed=seed,
        )
        self.min_word_len = min_word_len
        self.max_vocab = max_vocab

        # Word ↔ ID mapping
        self._word_to_id: dict[str, int] = {}
        self._id_to_word: dict[int, str] = {}
        self._next_id: int = 0

        # Accumulator for current word
        self._current_chars: list[str] = []

        # Last L2/3 state (for observation at boundary)
        self._last_l23: np.ndarray | None = None

    def _get_word_id(self, word: str) -> int:
        """Get or create ID for a word."""
        if word not in self._word_to_id:
            if self._next_id >= self.max_vocab:
                return -1  # Vocab full
            wid = self._next_id
            self._word_to_id[word] = wid
            self._id_to_word[wid] = word
            self._next_id += 1
            return wid
        return self._word_to_id[word]

    def step(self, char: str, l23_state: np.ndarray) -> str | None:
        """Process one character. Returns completed word if at boundary.

        Call every step with the character and region's L2/3 state.
        At word boundaries, trains the decoder on the completed word.
        """
        if char in _BOUNDARY_CHARS or not char:
            # Word boundary — observe completed word
            word = None
            if len(self._current_chars) >= self.min_word_len:
                word = "".join(self._current_chars)
                wid = self._get_word_id(word)
                if wid >= 0 and self._last_l23 is not None:
                    # Train decoder: S2's state at end of word → word ID
                    self._decoder.observe(wid, self._last_l23)
            self._current_chars.clear()
            self._last_l23 = l23_state.copy()
            return word
        else:
            self._current_chars.append(char)
            self._last_l23 = l23_state.copy()
            return None

    def predict(self, l23_state: np.ndarray, k: int = 3) -> list[tuple[str, int]]:
        """Predict top-k words from current L2/3 state.

        Returns list of (word, score) tuples.
        """
        scores = self._decoder.decode_scores(l23_state)
        if not scores:
            return []

        sorted_ids = sorted(scores, key=lambda t: scores[t], reverse=True)
        results = []
        for wid in sorted_ids[:k]:
            word = self._id_to_word.get(wid, f"<{wid}>")
            results.append((word, scores[wid]))
        return results

    @property
    def n_words(self) -> int:
        """Number of words in vocabulary."""
        return self._next_id

    def summary(self) -> dict:
        """Stats for display."""
        return {
            "n_words": self.n_words,
            "top_words": sorted(
                self._word_to_id.keys(),
                key=lambda w: self._word_to_id[w],
            )[:20],
        }
