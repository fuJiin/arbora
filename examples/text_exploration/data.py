"""Dictionary stream: load words, split train/test, iterate chars.

The char stream emits raw characters; word boundaries are a separate
signal delivered by iterating `words` externally (the trainer resets
between words). Deliberately no in-band sentinel/delimiter — the
training regime in ARB-131 specifies "no spaces, externally provided
word boundaries".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT_WORDS_PATH = Path(__file__).resolve().parent / "data" / "common_words.txt"

DEFAULT_ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def load_words(path: Path | str | None = None) -> list[str]:
    """Load a newline-separated wordlist. Lowercases, strips, dedupes
    (preserving first-seen order), skips blanks."""
    p = Path(path) if path is not None else DEFAULT_WORDS_PATH
    seen: set[str] = set()
    words: list[str] = []
    for raw in p.read_text().splitlines():
        w = raw.strip().lower()
        if not w or w in seen:
            continue
        seen.add(w)
        words.append(w)
    return words


def train_test_split(
    words: list[str],
    *,
    test_frac: float = 0.2,
    seed: int = 0,
) -> tuple[list[str], list[str]]:
    """Deterministic train/test split. Shuffles with `seed`, slices off
    `test_frac` of the words for test."""
    if not 0.0 < test_frac < 1.0:
        raise ValueError(f"test_frac must be in (0, 1), got {test_frac}")
    rng = np.random.default_rng(seed)
    shuffled = list(words)
    rng.shuffle(shuffled)
    n_test = max(1, int(len(shuffled) * test_frac))
    return shuffled[n_test:], shuffled[:n_test]


def alphabet_filter(words: list[str], alphabet: str = DEFAULT_ALPHABET) -> list[str]:
    """Keep only words composed entirely of `alphabet` chars.

    Dictionary entries can sneak in hyphens, apostrophes, or digits;
    this gives the caller a clean way to restrict to a single alphabet
    without having to handle unknown-column encodings downstream.
    """
    chars = set(alphabet)
    return [w for w in words if w and all(c in chars for c in w)]
