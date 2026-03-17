"""Reward sources for BG-gated motor learning.

Each reward source computes a scalar signal that trains the basal ganglia
to gate motor output. Different training stages use different reward sources.

Stage 1 (turn-taking): Did M1 speak at the right time?
Stage 3 (guided babbling): Is M1 producing word-like sequences?
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


class S1PredictionReward:
    """Stage 3: reward based on S1's prediction accuracy (burst rate).

    S1 was trained on real text and knows which character transitions
    are natural. When M1 produces a char that S1 predicted (low burst),
    it's producing natural language-like sequences. High burst means
    M1 produced something S1 didn't expect — an unnatural transition.

    Reward = scale * (1 - 2 * burst_fraction)
      burst_fraction=0.0 (all predicted) → +scale
      burst_fraction=0.5 (half bursting)  → 0.0
      burst_fraction=1.0 (all bursting)   → -scale
    """

    def __init__(self, scale: float = 0.5):
        self.scale = scale

    def step(self, char, s1_burst_fraction: float) -> float:
        if char is None:
            return 0.0
        return self.scale * (1.0 - 2.0 * s1_burst_fraction)


class WordReward:
    """Stage 3: continuous reward based on S2 pattern coherence.

    Every step M1 speaks, computes reward from S2's activation pattern
    stability — no need to wait for word boundaries. This gives BG
    gradient on every step rather than sparse boundary-only signal.

    Per-step reward:
      Positive: S2 pattern is stable (high Jaccard with previous step).
                Means S2 is recognizing a coherent sequence.
      Negative: S2 pattern is unstable (low Jaccard). Means M1 is
                producing gibberish that S2 can't parse.
      Bonus:    At word boundaries, extra reward if the accumulated
                word matches a known S2 pattern.

    Reward range: [-0.5, +1.0]
    """

    def __init__(
        self,
        n_columns: int,
        *,
        recognition_threshold: float = 0.5,
        min_word_length: int = 2,
        stability_scale: float = 0.3,
    ):
        self.n_columns = n_columns
        self.recognition_threshold = recognition_threshold
        self.min_word_length = min_word_length
        self.stability_scale = stability_scale

        # Known word patterns: word_str -> list of column sets
        self._known_words: dict[str, list[frozenset[int]]] = {}

        # Current word accumulator
        self._current_chars: list[str] = []
        self._current_col_sets: list[frozenset[int]] = []

        # Previous step's S2 columns (for stability computation)
        self._prev_cols: frozenset[int] = frozenset()

        # Stats
        self.words_produced: int = 0
        self.words_recognized: int = 0

    def step(
        self,
        char: str | None,
        s2_active_columns: np.ndarray,
    ) -> float:
        """Compute reward for one M1-produced character.

        Always returns a reward (never None). This replaces the
        turn-taking reward entirely during Stage 3.

        Args:
            char: The character M1 produced (None if silent).
            s2_active_columns: S2's active columns after processing.

        Returns:
            Reward in [-0.5, +1.0].
        """
        if char is None:
            return 0.0  # Silent — neutral (no turn-taking penalty)

        cols = frozenset(int(c) for c in np.nonzero(s2_active_columns)[0])

        # Per-step stability reward: how consistent is S2's pattern?
        stability = 0.0
        if self._prev_cols and cols:
            union = self._prev_cols | cols
            stability = len(self._prev_cols & cols) / len(union) if union else 0.0
        self._prev_cols = cols

        # Base reward: stability scaled to [-stability_scale, +stability_scale]
        # stability=0 → -scale, stability=1 → +scale
        reward = self.stability_scale * (2.0 * stability - 1.0)

        # Track current word
        is_boundary = char in (" ", ".", ",", "!", "?", "\n", "")

        if is_boundary:
            # Word boundary: check recognition bonus
            bonus = self._evaluate_word()
            reward += bonus
            self._current_chars.clear()
            self._current_col_sets.clear()
        else:
            self._current_chars.append(char)
            self._current_col_sets.append(cols)

        return reward

    def _evaluate_word(self) -> float:
        """Evaluate completed word, return bonus reward."""
        if len(self._current_chars) < self.min_word_length:
            return 0.0

        word = "".join(self._current_chars)
        self.words_produced += 1

        # Check recognition against known vocabulary
        if word in self._known_words:
            known_sets = self._known_words[word]
            current_union = (
                frozenset().union(*self._current_col_sets)
                if self._current_col_sets else frozenset()
            )
            similarities = []
            for known in known_sets:
                if current_union or known:
                    j = (
                        len(current_union & known) / len(current_union | known)
                        if (current_union | known) else 0.0
                    )
                    similarities.append(j)
            best_match = max(similarities) if similarities else 0.0

            if best_match >= self.recognition_threshold:
                self.words_recognized += 1
                self._record_word(word)
                return 0.7  # Big bonus for recognized word

        # Novel but coherent (internal consistency)
        consistency = self._internal_consistency()
        if consistency > 0.3:
            self._record_word(word)
            return 0.2  # Small bonus for novel coherent word

        return -0.2  # Penalty for incoherent word

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
        """Store word's S2 pattern for future recognition."""
        union = (
            frozenset().union(*self._current_col_sets)
            if self._current_col_sets else frozenset()
        )
        if word not in self._known_words:
            self._known_words[word] = []
        self._known_words[word].append(union)
        if len(self._known_words[word]) > 10:
            self._known_words[word].pop(0)

    def seed_from_sensory(self, word_probe) -> None:
        """Bootstrap known words from a Stage 1 WordSelectivityProbe."""
        for word, col_sets in word_probe._word_columns.items():
            if len(col_sets) >= word_probe.min_observations:
                self._known_words[word] = list(col_sets[-10:])

    def reset_word(self) -> None:
        """Reset at story/dialogue boundary."""
        self._current_chars.clear()
        self._current_col_sets.clear()
        self._prev_cols = frozenset()

    def summary(self) -> dict:
        return {
            "words_produced": self.words_produced,
            "words_recognized": self.words_recognized,
            "known_vocabulary": len(self._known_words),
            "recognition_rate": (
                self.words_recognized / max(self.words_produced, 1)
            ),
        }
