"""Char-level data sources for the T1 text-exploration experiments.

Two loaders:

- `load_words` / `alphabet_filter` — the original bundled wordlist
  (ARB-131's synthetic test data). Now space-separated when streamed
  continuously so the region doesn't learn artifactual cross-word
  bigrams.
- `load_natural_chunks` — delegates to `examples.chat.data` to load
  real text datasets (TinyStories, BabyLM, TinyDialogues, PersonaChat)
  from HuggingFace at char level, split into per-document "chunks".

Both paths return a list of "chunks" — strings the caller iterates
char-by-char. Chunks are the unit at which callers may optionally
reset region working memory (`reset_per_chunk` at word boundary for
the wordlist, at document boundary for natural text).

`shuffle_chunks` gives deterministic per-epoch shuffling, which the
original pipeline was missing (words iterated in fixed order every
epoch, effectively one repeated super-sentence).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT_WORDS_PATH = Path(__file__).resolve().parent / "data" / "common_words.txt"

# a-z + space. Space matters: in continuous mode it's the natural
# word delimiter; without it the stream teaches artifactual cross-word
# bigrams.
DEFAULT_ALPHABET = "abcdefghijklmnopqrstuvwxyz "


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


def wordlist_chunks(words: list[str], *, append_space: bool = True) -> list[str]:
    """Convert a wordlist into stream chunks with optional trailing space.

    Each word becomes one chunk. When `append_space=True` (default),
    the chunk has a trailing space so concatenating chunks produces a
    natural space-separated stream. Callers that reset between chunks
    still get word boundaries; callers that don't get realistic char
    transitions across words.
    """
    if not append_space:
        return list(words)
    return [w + " " for w in words]


def load_natural_chunks(
    dataset: str = "tinystories",
    *,
    max_chars: int = 50_000,
    alphabet: str = DEFAULT_ALPHABET,
) -> list[str]:
    """Load a natural-text dataset as chunks (documents), char-level.

    Delegates to `examples.chat.data.prepare_tokens_charlevel` and
    groups tokens by `STORY_BOUNDARY` into per-document chunks.
    Characters outside `alphabet` are dropped (default is a-z+space,
    so punctuation/uppercase get stripped — keep the alphabet simple
    for the M1 substrate work).

    Requires `datasets` and `transformers` via HuggingFace; imports
    lazily so the common wordlist path stays dependency-light.
    """
    from examples.chat.data import STORY_BOUNDARY, prepare_tokens_charlevel

    tokens = prepare_tokens_charlevel(max_tokens=max_chars, dataset=dataset)
    allowed = set(alphabet)
    chunks: list[str] = []
    buf: list[str] = []
    for tid, s in tokens:
        if tid == STORY_BOUNDARY:
            if buf:
                chunks.append("".join(buf))
                buf = []
            continue
        c = s.lower() if s else ""
        for ch in c:
            if ch in allowed:
                buf.append(ch)
    if buf:
        chunks.append("".join(buf))
    return [c for c in chunks if c]


def split_chunks(
    chunks: list[str],
    *,
    test_frac: float = 0.2,
    seed: int = 0,
) -> tuple[list[str], list[str]]:
    """Deterministic train/test split on a chunk list."""
    if not 0.0 < test_frac < 1.0:
        raise ValueError(f"test_frac must be in (0, 1), got {test_frac}")
    rng = np.random.default_rng(seed)
    shuffled = list(chunks)
    rng.shuffle(shuffled)
    n_test = max(1, int(len(shuffled) * test_frac))
    return shuffled[n_test:], shuffled[:n_test]


def shuffle_chunks(chunks: list[str], *, rng: np.random.Generator) -> list[str]:
    """Return a new shuffled list. Pass a live RNG for per-epoch reshuffling."""
    out = list(chunks)
    rng.shuffle(out)
    return out
