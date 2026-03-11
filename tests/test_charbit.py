import numpy as np
import pytest

from step.encoders.charbit import CharbitEncoder

# Default test alphabet: printable ASCII (space through ~)
PRINTABLE_ASCII = "".join(chr(c) for c in range(32, 127))


class TestCharbitEncoderInit:
    def test_stores_params(self):
        w = len(PRINTABLE_ASCII) + 1
        enc = CharbitEncoder(length=8, width=w, chars=PRINTABLE_ASCII)
        assert enc.length == 8
        assert enc.width == w

    def test_width_must_match_chars_plus_unknown(self):
        """width = len(chars) + 1 for the unknown column."""
        enc = CharbitEncoder(length=8, width=96, chars=PRINTABLE_ASCII)
        assert enc.width == 96  # 95 printable + 1 unknown

    def test_width_too_small_raises(self):
        """width must be >= len(chars) + 1."""
        with pytest.raises(ValueError, match="width"):
            CharbitEncoder(length=8, width=10, chars=PRINTABLE_ASCII)

    def test_duplicate_chars_raises(self):
        with pytest.raises(ValueError, match="duplicate"):
            CharbitEncoder(length=8, width=5, chars="aab")


class TestCharbitEncoderOutput:
    @pytest.fixture()
    def encoder(self):
        return CharbitEncoder(
            length=8, width=len(PRINTABLE_ASCII) + 1, chars=PRINTABLE_ASCII
        )

    def test_output_shape(self, encoder):
        result = encoder.encode("hello")
        assert isinstance(result, np.ndarray)
        assert result.shape == (8, encoder.width)

    def test_output_dtype(self, encoder):
        result = encoder.encode("hello")
        assert result.dtype == np.bool_

    def test_one_hot_per_position(self, encoder):
        """Each row for a present character should have exactly one 1."""
        result = encoder.encode("abc")
        for i in range(3):
            assert result[i].sum() == 1
        # Padding rows should be all zeros
        for i in range(3, 8):
            assert result[i].sum() == 0

    def test_padding_is_zeros(self, encoder):
        """Tokens shorter than length get zero-padded."""
        result = encoder.encode("hi")
        assert result[2:].sum() == 0

    def test_empty_string(self, encoder):
        result = encoder.encode("")
        assert result.shape == (8, encoder.width)
        assert result.sum() == 0

    def test_truncation_at_length(self, encoder):
        """Tokens longer than length are truncated."""
        result = encoder.encode("a" * 20)
        assert result.shape == (8, encoder.width)
        # All 8 positions should be active (the character 'a')
        for i in range(8):
            assert result[i].sum() == 1


class TestCharbitEncoderCharMapping:
    @pytest.fixture()
    def encoder(self):
        # Small alphabet for easy reasoning
        return CharbitEncoder(length=4, width=5, chars="abcd")

    def test_different_chars_different_columns(self, encoder):
        result = encoder.encode("abcd")
        # Each character should activate a different column
        active_cols = [np.argmax(result[i]) for i in range(4)]
        assert len(set(active_cols)) == 4

    def test_same_char_same_column(self, encoder):
        result = encoder.encode("aaaa")
        col = np.argmax(result[0])
        for i in range(1, 4):
            assert np.argmax(result[i]) == col

    def test_deterministic(self, encoder):
        a = encoder.encode("test")
        b = encoder.encode("test")
        np.testing.assert_array_equal(a, b)

    def test_unknown_char_uses_unknown_column(self, encoder):
        """Characters not in the alphabet map to the unknown column."""
        result = encoder.encode("z")  # 'z' not in "abcd"
        assert result[0].sum() == 1
        # The unknown column should be the last one
        assert result[0, encoder.width - 1] == True  # noqa: E712

    def test_known_and_unknown_mixed(self, encoder):
        result = encoder.encode("azb")
        # 'a' -> known column, 'z' -> unknown column, 'b' -> known column
        assert result[0, encoder.width - 1] == False  # noqa: E712  # 'a' is known
        assert result[1, encoder.width - 1] == True  # noqa: E712  # 'z' is unknown
        assert result[2, encoder.width - 1] == False  # noqa: E712  # 'b' is known


class TestCharbitEncoderEdgeCases:
    def test_space_in_alphabet(self):
        enc = CharbitEncoder(length=4, width=4, chars=" ab")
        result = enc.encode(" a")
        assert result[0].sum() == 1  # space is encoded
        assert result[1].sum() == 1  # 'a' is encoded

    def test_full_ascii_alphabet(self):
        chars = "".join(chr(c) for c in range(128))
        enc = CharbitEncoder(length=16, width=129, chars=chars)
        result = enc.encode("Hello, World!")
        assert result.shape == (16, 129)
        for i in range(13):
            assert result[i].sum() == 1

    def test_single_char_length(self):
        enc = CharbitEncoder(length=1, width=4, chars="abc")
        result = enc.encode("abc")
        assert result.shape == (1, 4)
        assert result[0].sum() == 1  # only first char encoded
