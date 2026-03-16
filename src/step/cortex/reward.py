"""Reward sources for BG-gated motor learning.

Each reward source computes a scalar signal that trains the basal ganglia
to gate motor output. Different training stages use different reward sources.

Stage 1 (turn-taking): Did M1 speak at the right time?
Stage 3 (guided babbling): Did M1 produce a word S2 recognizes?
"""

import numpy as np


class TurnTakingReward:
    """Stage 1: reward for speaking during EOM, silence during input."""

    def compute(
        self, spoke: bool, in_eom: bool, eom_steps: int, max_speak_steps: int,
    ) -> float:
        if in_eom:
            if eom_steps > max_speak_steps and spoke:
                return -1.0  # Rambling
            return 0.5 if spoke else -0.3
        else:
            return -0.5 if spoke else 0.2


class WordReward:
    """Stage 3: reward based on S2 word-pattern consistency.

    Tracks S2's L2/3 activation patterns during M1's output sequence.
    When a word boundary is hit (space/punctuation), compares the
    accumulated S2 pattern against known word patterns.

    Reward signal:
      +1.0  S2 pattern matches a previously seen word (high Jaccard)
      +0.3  S2 pattern is novel but internally consistent
      -0.3  S2 pattern is noisy/incoherent
       0.0  No word produced (too short)
    """

    def __init__(
        self,
        n_columns: int,
        *,
        consistency_threshold: float = 0.3,
        recognition_threshold: float = 0.5,
        min_word_length: int = 2,
    ):
        self.n_columns = n_columns
        self.consistency_threshold = consistency_threshold
        self.recognition_threshold = recognition_threshold
        self.min_word_length = min_word_length

        # Known word patterns: word_str -> list of column sets
        self._known_words: dict[str, list[frozenset[int]]] = {}

        # Current word accumulator
        self._current_chars: list[str] = []
        self._current_col_sets: list[frozenset[int]] = []

        # Stats
        self.words_produced: int = 0
        self.words_recognized: int = 0
        self.words_novel: int = 0

    def step(
        self,
        char: str | None,
        s2_active_columns: np.ndarray,
    ) -> float | None:
        """Process one M1-produced character through S2 observation.

        Args:
            char: The character M1 produced (None if silent).
            s2_active_columns: S2's active columns after processing.

        Returns:
            Reward if word boundary hit, None otherwise.
        """
        if char is None:
            return None

        is_boundary = char in (" ", ".", ",", "!", "?", "\n", "")

        if is_boundary:
            reward = self._evaluate_word()
            self._current_chars.clear()
            self._current_col_sets.clear()
            return reward
        else:
            self._current_chars.append(char)
            cols = frozenset(int(c) for c in np.nonzero(s2_active_columns)[0])
            self._current_col_sets.append(cols)
            return None

    def _evaluate_word(self) -> float:
        """Evaluate the completed word and return reward."""
        if len(self._current_chars) < self.min_word_length:
            return 0.0

        word = "".join(self._current_chars)
        self.words_produced += 1

        # Compute internal consistency: mean Jaccard across consecutive
        # S2 activations within this word. High = S2 maintained a stable
        # pattern throughout (coherent input).
        consistency = self._internal_consistency()

        # Check recognition: does this word match a known pattern?
        if word in self._known_words:
            known_sets = self._known_words[word]
            # Compare current pattern (union of all char activations)
            current_union = frozenset().union(*self._current_col_sets) if self._current_col_sets else frozenset()
            similarities = []
            for known in known_sets:
                if current_union or known:
                    j = len(current_union & known) / len(current_union | known) if (current_union | known) else 0.0
                    similarities.append(j)
            best_match = max(similarities) if similarities else 0.0

            if best_match >= self.recognition_threshold:
                self.words_recognized += 1
                self._record_word(word)
                return 1.0  # Known word, well-produced

        # Novel word — reward consistency
        if consistency >= self.consistency_threshold:
            self.words_novel += 1
            self._record_word(word)
            return 0.3  # Novel but coherent

        return -0.3  # Incoherent

    def _internal_consistency(self) -> float:
        """Mean pairwise Jaccard of consecutive S2 activations."""
        if len(self._current_col_sets) < 2:
            return 0.0

        jaccards = []
        for i in range(len(self._current_col_sets) - 1):
            a = self._current_col_sets[i]
            b = self._current_col_sets[i + 1]
            if a or b:
                j = len(a & b) / len(a | b) if (a | b) else 0.0
                jaccards.append(j)

        return float(np.mean(jaccards)) if jaccards else 0.0

    def _record_word(self, word: str) -> None:
        """Store this word's S2 pattern for future recognition."""
        union = frozenset().union(*self._current_col_sets) if self._current_col_sets else frozenset()
        if word not in self._known_words:
            self._known_words[word] = []
        self._known_words[word].append(union)
        # Keep at most 10 recent patterns per word
        if len(self._known_words[word]) > 10:
            self._known_words[word].pop(0)

    def seed_from_sensory(self, word_probe) -> None:
        """Bootstrap known words from a Stage 1 WordSelectivityProbe.

        Loads word→column patterns learned during sensory training so
        Stage 3 can immediately recognize words M1 produces.
        """
        for word, col_sets in word_probe._word_columns.items():
            if len(col_sets) >= word_probe.min_observations:
                self._known_words[word] = list(col_sets[-10:])

    def summary(self) -> dict:
        return {
            "words_produced": self.words_produced,
            "words_recognized": self.words_recognized,
            "words_novel": self.words_novel,
            "known_vocabulary": len(self._known_words),
            "recognition_rate": (
                self.words_recognized / max(self.words_produced, 1)
            ),
        }
