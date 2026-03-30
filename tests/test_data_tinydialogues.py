"""Tests for prepare_tokens_tinydialogues() speaker-turn loading."""

import pytest

from arbor.data import EOM_TOKEN, STORY_BOUNDARY, prepare_tokens_tinydialogues


@pytest.mark.slow
class TestTinyDialogues:
    def test_returns_tokens_with_sentinels(self):
        """Output contains EOM_TOKEN and STORY_BOUNDARY sentinels."""
        tokens = prepare_tokens_tinydialogues(max_tokens=2000, speak_window=2)
        ids = [tid for tid, _ in tokens]
        assert EOM_TOKEN in ids, "Expected EOM_TOKEN (-2) in output"
        assert STORY_BOUNDARY in ids, "Expected STORY_BOUNDARY (-1) in output"

    def test_child_turns_preceded_by_eom(self):
        """EOM_TOKEN appears before child utterance characters."""
        tokens = prepare_tokens_tinydialogues(max_tokens=5000, speak_window=2)
        ids = [tid for tid, _ in tokens]
        eom_indices = [i for i, tid in enumerate(ids) if tid == EOM_TOKEN]
        assert len(eom_indices) > 0, "No EOM tokens found"

        # After each EOM, the speak_window tokens should be repeated chars,
        # then child utterance characters (positive ords) should follow.
        for eom_idx in eom_indices[:5]:  # check first few
            # Skip speak_window padding, then look for a real character
            scan_start = eom_idx + 1
            found_char = False
            for j in range(scan_start, min(scan_start + 20, len(ids))):
                if ids[j] >= 0 and ids[j] != STORY_BOUNDARY:
                    found_char = True
                    break
            assert found_char, f"No character found after EOM at index {eom_idx}"

    def test_respects_max_tokens(self):
        """Output length is bounded by max_tokens."""
        limit = 500
        tokens = prepare_tokens_tinydialogues(max_tokens=limit, speak_window=2)
        # Count only chars (non-sentinel), which is what t counts
        char_count = sum(1 for tid, _ in tokens if tid >= 0)
        # Total includes sentinels; char_count should be at most limit
        assert char_count <= limit

    def test_char_level_encoding(self):
        """Regular tokens are (ord(char), char) pairs."""
        tokens = prepare_tokens_tinydialogues(max_tokens=500, speak_window=0)
        for tid, tstr in tokens:
            if tid >= 0:
                assert tid == ord(tstr), (
                    f"Expected ord({tstr!r})={ord(tstr)}, got {tid}"
                )
