"""Word-level selectivity probe for higher cortical regions.

Measures whether a region's columns learn word-level representations
by tracking column activations across character-by-character input and
correlating with word boundaries.

Two analyses:
1. Per-word column selectivity: Do specific columns fire consistently
   for specific words? Measured by entropy of word distribution per column.
2. Word-level decoding: Can we predict the current word from the region's
   L2/3 state? Uses a dendritic decoder mapping L2/3 → word_id.
"""

from collections import Counter, defaultdict

import numpy as np

from step.decoders.dendritic import DendriticDecoder


class WordSelectivityProbe:
    """Track word-level selectivity of a cortical region's columns.

    Feed characters one at a time via step(). The probe accumulates
    the current word and, at each space/punctuation boundary, records
    which columns were active during that word.
    """

    def __init__(
        self,
        n_columns: int,
        *,
        min_word_length: int = 2,
        min_observations: int = 3,
    ):
        self.n_columns = n_columns
        self.min_word_length = min_word_length
        self.min_observations = min_observations

        # Current word accumulator
        self._current_word: list[str] = []
        # Columns active during current word (union across chars)
        self._current_cols: set[int] = set()
        # Per-column activation counts during current word
        self._current_col_counts: Counter = Counter()
        self._current_word_chars: int = 0

        # Completed word statistics
        # word → list of column sets (one per occurrence)
        self._word_columns: dict[str, list[frozenset[int]]] = defaultdict(list)
        # column → Counter of words it fired for
        self._col_word_counts: list[Counter] = [
            Counter() for _ in range(n_columns)
        ]
        self._total_words: int = 0

    def step(
        self,
        char: str,
        active_columns: np.ndarray,
    ) -> str | None:
        """Process one character, return completed word if boundary hit."""
        is_boundary = char in (" ", ".", ",", "!", "?", "\n", "") or char == ""

        if is_boundary:
            word = self._finish_word()
            # Don't start a new word with the boundary char
            self._current_word = []
            self._current_cols = set()
            self._current_col_counts = Counter()
            self._current_word_chars = 0
            return word
        else:
            self._current_word.append(char)
            self._current_word_chars += 1
            # Record which columns are active for this character
            active = set(int(c) for c in np.nonzero(active_columns)[0])
            self._current_cols |= active
            for c in active:
                self._current_col_counts[c] += 1
            return None

    def _finish_word(self) -> str | None:
        """Record statistics for the completed word."""
        if len(self._current_word) < self.min_word_length:
            return None

        word = "".join(self._current_word)
        cols = frozenset(self._current_cols)

        self._word_columns[word].append(cols)
        for col in cols:
            self._col_word_counts[col][word] += 1
        self._total_words += 1

        return word

    def column_selectivity(self) -> list[tuple[int, float, str]]:
        """Per-column word selectivity: (col, entropy, best_word).

        Lower entropy = more selective (fires for fewer words).
        Returns list sorted by entropy (most selective first).
        """
        results = []
        for col in range(self.n_columns):
            counts = self._col_word_counts[col]
            if not counts:
                continue

            total = sum(counts.values())
            if total < self.min_observations:
                continue

            # Normalized entropy
            probs = np.array(list(counts.values()), dtype=np.float64)
            probs = probs / probs.sum()
            entropy = -float((probs * np.log2(probs + 1e-10)).sum())
            max_entropy = np.log2(len(probs)) if len(probs) > 1 else 1.0
            norm_entropy = entropy / max(max_entropy, 1e-10)

            best_word = counts.most_common(1)[0][0]
            results.append((col, norm_entropy, best_word))

        results.sort(key=lambda x: x[1])
        return results

    def word_consistency(self) -> list[tuple[str, float, int]]:
        """Per-word column consistency: (word, mean_jaccard, n_observations).

        Higher Jaccard = same columns fire consistently for this word.
        Only includes words with >= min_observations occurrences.
        """
        results = []
        for word, col_sets in self._word_columns.items():
            if len(col_sets) < self.min_observations:
                continue

            # Mean pairwise Jaccard similarity
            jaccards = []
            for i in range(len(col_sets)):
                for j in range(i + 1, len(col_sets)):
                    a, b = col_sets[i], col_sets[j]
                    if not a and not b:
                        continue
                    jaccard = len(a & b) / len(a | b) if (a | b) else 0.0
                    jaccards.append(jaccard)

            mean_j = float(np.mean(jaccards)) if jaccards else 0.0
            results.append((word, mean_j, len(col_sets)))

        results.sort(key=lambda x: -x[1])
        return results

    def summary(self) -> dict:
        """Summary statistics for printing."""
        selectivity = self.column_selectivity()
        consistency = self.word_consistency()

        n_selective = sum(1 for _, e, _ in selectivity if e < 0.7)
        n_consistent = sum(1 for _, j, _ in consistency if j > 0.3)

        return {
            "total_words": self._total_words,
            "unique_words": len(self._word_columns),
            "columns_with_words": len(selectivity),
            "selective_columns": n_selective,
            "mean_selectivity": (
                float(np.mean([e for _, e, _ in selectivity]))
                if selectivity else 1.0
            ),
            "consistent_words": n_consistent,
            "mean_consistency": (
                float(np.mean([j for _, j, _ in consistency]))
                if consistency else 0.0
            ),
            "top_selective": selectivity[:5],
            "top_consistent": consistency[:10],
        }
