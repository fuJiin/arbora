"""Reward sources for BG-gated motor learning.

Each reward source computes a scalar signal that trains the basal ganglia
to gate motor output.

Two biological reward pathways modeled:
  Intrinsic (curiosity/RPE): dopamine from VTA for prediction improvement
  Extrinsic (caregiver): social reward when M1 produces a recognized word
"""

import numpy as np


class CuriosityReward:
    """Reward for prediction improvement (dopamine RPE model).

    Tracks expected S1 burst rate per bigram. Reward = improvement
    over expectation: expected_burst - actual_burst. This gives:

    - High reward for new bigrams where S1 is learning to predict
    - Zero reward for fully-predicted bigrams (nothing left to learn)
    - Negative reward for bigrams that got WORSE (shouldn't happen often)

    Naturally solves exploration: known patterns stop being rewarding,
    pushing M1 to discover new ones. No need for explicit diversity
    bonus or repetition penalty — they emerge from the RPE structure.

    Models dopaminergic RPE: DA neurons fire for unexpected reward
    (prediction improvement), not expected reward (prediction accuracy).
    """

    def __init__(
        self,
        scale: float = 1.0,
        baseline_decay: float = 0.99,
    ):
        self.scale = scale
        self.baseline_decay = baseline_decay

        # Per-bigram expected burst rate (EMA)
        # Key: (prev_char, char) → expected burst fraction
        self._expected_burst: dict[tuple[str | None, str], float] = {}
        self._prev_char: str | None = None

        # Global baseline for unseen bigrams
        self._global_burst_ema: float = 0.5

    def step(self, char, s1_burst_fraction: float) -> float:
        if char is None:
            return 0.0

        bigram = (self._prev_char, char)

        # Get expected burst for this bigram (or global baseline)
        if bigram in self._expected_burst:
            expected = self._expected_burst[bigram]
        else:
            expected = self._global_burst_ema

        # RPE: positive when actual burst < expected (better than expected)
        rpe = expected - s1_burst_fraction

        # Update expected burst (EMA toward actual)
        alpha = 1.0 - self.baseline_decay
        if bigram in self._expected_burst:
            self._expected_burst[bigram] += alpha * (
                s1_burst_fraction - self._expected_burst[bigram]
            )
        else:
            self._expected_burst[bigram] = s1_burst_fraction

        # Update global baseline
        self._global_burst_ema += alpha * (
            s1_burst_fraction - self._global_burst_ema
        )

        self._prev_char = char
        return self.scale * rpe

    def reset(self):
        self._prev_char = None
        # Keep learned expectations — they transfer across dialogues


class CaregiverReward:
    """Combined intrinsic (curiosity) + extrinsic (word recognition) reward.

    Models infant speech development:
    - Curiosity (VTA/SNc dopamine RPE): drives exploration of new bigrams
    - Caregiver (social reward): bonus when M1 output forms a known word

    The caregiver signal guides M1 toward actual language patterns.
    Without it, curiosity alone discovers all chars but doesn't converge
    on English — any predictable pattern is equally rewarding.

    Word detection: accumulates M1's chars, checks at word boundaries
    (space, punctuation) against a known vocabulary seeded from S2's
    word patterns learned during Stage 1 sensory training.
    """

    def __init__(
        self,
        known_words: set[str] | None = None,
        *,
        curiosity_scale: float = 1.0,
        word_bonus: float = 0.5,
        min_word_length: int = 2,
        baseline_decay: float = 0.99,
    ):
        self._curiosity = CuriosityReward(
            scale=curiosity_scale, baseline_decay=baseline_decay,
        )
        self.word_bonus = word_bonus
        self.min_word_length = min_word_length

        # Known vocabulary (seeded from corpus or S2 probe)
        self._known_words = known_words or set()

        # Current word accumulator
        self._current_word: list[str] = []

        # Stats
        self.words_attempted: int = 0
        self.words_recognized: int = 0

    def seed_vocabulary(self, words: set[str]) -> None:
        """Add words to known vocabulary and build prefix index."""
        self._known_words |= words
        # Build prefix set for fast partial matching
        self._prefixes: set[str] = set()
        for w in self._known_words:
            for i in range(2, len(w) + 1):
                self._prefixes.add(w[:i])
        # Pre-compute extension counts: how many next-chars continue each prefix
        self._prefix_extensions: dict[str, int] = {}
        for p in self._prefixes:
            self._prefix_extensions[p] = sum(
                1 for p2 in self._prefixes
                if p2.startswith(p) and len(p2) == len(p) + 1
            )

    def step(self, char, s1_burst_fraction: float) -> float:
        """Compute combined reward for one M1-produced character.

        Live word tracking: every char gets feedback based on how many
        word completions remain. Like a caregiver's face as baby babbles:
        "t" → mild interest, "th" → excited, "the" → thrilled,
        "thq" → disappointed (no words start with "thq").
        """
        # Base: curiosity RPE (always active)
        reward = self._curiosity.step(char, s1_burst_fraction)

        if char is None:
            return reward

        is_boundary = char in (" ", ".", ",", "!", "?", "'", "-")

        if is_boundary:
            # Word boundary: check for full word match
            if len(self._current_word) >= self.min_word_length:
                word = "".join(self._current_word)
                self.words_attempted += 1
                if word in self._known_words:
                    self.words_recognized += 1
                    # Caregiver goes WILD. This is the biggest reward
                    # in the system — completing a real word. Scales
                    # with word length (longer words = harder = more reward).
                    reward += self.word_bonus * max(len(word), 2)
            self._current_word.clear()
        else:
            self._current_word.append(char)
            # Live prefix tracking: reward based on remaining completions
            if len(self._current_word) >= self.min_word_length:
                reward += self._prefix_signal()
            # Caregiver attention span: reset after max word length.
            # Long babbles without boundaries lose caregiver engagement.
            if len(self._current_word) > 8:
                self._current_word.clear()

        return reward

    def _prefix_signal(self) -> float:
        """Live reward signal based on word completion optionality.

        Scales reward by how many words this prefix could become:
        - Many completions → high reward (promising direction)
        - Few completions → low reward (narrowing)
        - Complete word → strong "finish now" signal
        - Dead end → zero (no penalty, curiosity handles exploration)

        Models caregiver excitement proportional to how word-like
        the babbling sounds.
        """
        if not hasattr(self, '_prefixes'):
            return 0.0

        prefix = "".join(self._current_word)

        if prefix in self._known_words:
            # This IS a complete word — strong signal to produce a space.
            # Caregiver is thrilled: "did you just say 'the'?!"
            return self.word_bonus * 1.0

        if prefix in self._prefixes:
            # Valid prefix: scale by how many next-chars continue a word.
            # More possible completions → more reward (promising direction).
            n_ext = self._prefix_extensions.get(prefix, 0)
            optionality = min(n_ext / 10.0, 1.0)
            return self.word_bonus * 0.1 * optionality

        # Dead end: no penalty.
        return 0.0

    def reset(self):
        self._curiosity.reset()
        self._current_word.clear()

    def summary(self) -> dict:
        return {
            "words_attempted": self.words_attempted,
            "words_recognized": self.words_recognized,
            "known_vocabulary": len(self._known_words),
            "recognition_rate": (
                self.words_recognized / max(self.words_attempted, 1)
            ),
            "bigrams_tracked": len(self._curiosity._expected_burst),
        }

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
        """Reset word accumulator."""
        self._current_chars.clear()
        self._current_col_sets.clear()
        self._prev_cols = frozenset()

    def reset(self) -> None:
        """Reset at story/dialogue boundary."""
        self.reset_word()

    def summary(self) -> dict:
        return {
            "words_produced": self.words_produced,
            "words_recognized": self.words_recognized,
            "known_vocabulary": len(self._known_words),
            "recognition_rate": (
                self.words_recognized / max(self.words_produced, 1)
            ),
        }
