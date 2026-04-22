"""Dictionary loader + train/test split tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from examples.text_exploration.data import (
    DEFAULT_ALPHABET,
    DEFAULT_WORDS_PATH,
    alphabet_filter,
    load_words,
    train_test_split,
)


class TestLoadWords:
    def test_default_path_exists(self):
        """The bundled wordlist must ship with the repo."""
        assert DEFAULT_WORDS_PATH.exists()

    def test_default_returns_nonempty(self):
        words = load_words()
        assert len(words) >= 100

    def test_lowercases_and_strips(self, tmp_path: Path):
        p = tmp_path / "w.txt"
        p.write_text("  Hello  \n\nWORLD\n hello \n")
        words = load_words(p)
        assert words == ["hello", "world"]  # dedupes "hello"

    def test_skips_blanks(self, tmp_path: Path):
        p = tmp_path / "w.txt"
        p.write_text("\n\na\n\nb\n\n\n")
        assert load_words(p) == ["a", "b"]

    def test_preserves_first_seen_order_on_dedupe(self, tmp_path: Path):
        p = tmp_path / "w.txt"
        p.write_text("b\na\nb\nc\na\n")
        assert load_words(p) == ["b", "a", "c"]


class TestTrainTestSplit:
    def test_split_sizes(self):
        words = [f"w{i:03d}" for i in range(100)]
        train, test = train_test_split(words, test_frac=0.2)
        assert len(train) == 80
        assert len(test) == 20

    def test_disjoint(self):
        words = [f"w{i:03d}" for i in range(100)]
        train, test = train_test_split(words, test_frac=0.3)
        assert set(train).isdisjoint(set(test))
        assert set(train) | set(test) == set(words)

    def test_deterministic_for_seed(self):
        words = [f"w{i:03d}" for i in range(50)]
        a = train_test_split(words, seed=42)
        b = train_test_split(words, seed=42)
        assert a == b

    def test_different_seeds_differ(self):
        """Seeds changing means the split changes — no silently stuck RNG."""
        words = [f"w{i:03d}" for i in range(50)]
        a = train_test_split(words, seed=0)
        b = train_test_split(words, seed=1)
        assert a != b

    def test_rejects_bad_frac(self):
        with pytest.raises(ValueError):
            train_test_split(["a", "b"], test_frac=0.0)
        with pytest.raises(ValueError):
            train_test_split(["a", "b"], test_frac=1.0)

    def test_min_one_test_word(self):
        """Very small corpora should still produce at least one test word,
        so downstream evaluation isn't silently empty."""
        _, test = train_test_split(["a", "b", "c"], test_frac=0.01)
        assert len(test) >= 1


class TestAlphabetFilter:
    def test_keeps_clean_words(self):
        words = ["hello", "world", "cat"]
        assert alphabet_filter(words) == words

    def test_drops_words_with_hyphens(self):
        assert alphabet_filter(["hello", "hi-there", "cat"]) == ["hello", "cat"]

    def test_drops_words_with_digits_or_punct(self):
        assert alphabet_filter(["abc", "a1b", "a'b", "xyz"]) == ["abc", "xyz"]

    def test_custom_alphabet(self):
        """Custom alphabet that includes '-' should keep hyphenated words."""
        assert alphabet_filter(
            ["hi-there", "plain"], alphabet=DEFAULT_ALPHABET + "-"
        ) == ["hi-there", "plain"]
