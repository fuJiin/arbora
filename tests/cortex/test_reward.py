"""Tests for RewardModulator, EchoReward, and Stage 1 turn-taking reward."""

import pytest

from step.cortex.modulators import RewardModulator
from step.cortex.motor import MotorRegion
from step.cortex.reward import EchoReward
from step.cortex.sensory import SensoryRegion
from step.cortex.topology import Topology
from step.data import EOM_TOKEN, STORY_BOUNDARY, inject_eom_tokens
from step.encoders.charbit import CharbitEncoder


@pytest.fixture()
def encoder():
    return CharbitEncoder(length=4, width=5, chars="abcd")


@pytest.fixture()
def region1():
    return SensoryRegion(
        input_dim=4 * 5,
        encoding_width=5,
        n_columns=8,
        n_l4=2,
        n_l23=2,
        k_columns=2,
        seed=42,
    )


@pytest.fixture()
def motor(region1):
    return MotorRegion(
        input_dim=region1.n_l23_total,
        n_columns=4,
        n_l4=2,
        n_l23=2,
        k_columns=1,
        voltage_decay=0.5,
        learning_rate=0.15,
        ltd_rate=0.15,
        seed=456,
    )


class TestRewardModulator:
    def test_starts_neutral(self):
        rm = RewardModulator()
        assert rm.value == pytest.approx(1.0)

    def test_positive_reward_increases_modulator(self):
        rm = RewardModulator()
        for _ in range(50):
            mod = rm.update(0.5)
        assert mod > 1.0

    def test_negative_reward_decreases_modulator(self):
        rm = RewardModulator()
        for _ in range(50):
            mod = rm.update(-0.5)
        assert mod < 1.0

    def test_zero_reward_stays_neutral(self):
        rm = RewardModulator()
        for _ in range(100):
            mod = rm.update(0.0)
        assert mod == pytest.approx(1.0, abs=0.01)

    def test_clipped_to_range(self):
        rm = RewardModulator()
        # Extreme positive
        for _ in range(200):
            mod = rm.update(10.0)
        assert mod <= 2.0
        assert mod >= 0.0

        # Extreme negative
        rm2 = RewardModulator()
        for _ in range(200):
            mod = rm2.update(-10.0)
        assert mod >= 0.0
        assert mod <= 2.0

    def test_reset(self):
        rm = RewardModulator()
        for _ in range(50):
            rm.update(1.0)
        rm.reset()
        assert rm.value == pytest.approx(1.0)


class TestTurnTakingReward:
    _fn = staticmethod(Topology._compute_turn_reward)

    def test_reward_function_speak_during_eom(self):
        """Speaking during EOM phase should be rewarded."""
        r = self._fn(
            spoke=True,
            in_eom=True,
            eom_steps=1,
            max_speak_steps=20,
        )
        assert r > 0

    def test_reward_function_silent_during_input(self):
        """Silence during input phase should be mildly rewarded."""
        r = self._fn(
            spoke=False,
            in_eom=False,
            eom_steps=0,
            max_speak_steps=20,
        )
        assert r > 0

    def test_reward_function_speak_during_input(self):
        """Speaking during input phase should be penalized."""
        r = self._fn(
            spoke=True,
            in_eom=False,
            eom_steps=0,
            max_speak_steps=20,
        )
        assert r < 0

    def test_reward_function_silent_during_eom(self):
        """Silence during EOM phase should be mildly penalized."""
        r = self._fn(
            spoke=False,
            in_eom=True,
            eom_steps=1,
            max_speak_steps=20,
        )
        assert r < 0

    def test_reward_function_rambling(self):
        """Speaking past max steps should be penalized most."""
        r = self._fn(
            spoke=True,
            in_eom=True,
            eom_steps=25,
            max_speak_steps=20,
        )
        assert r < 0
        # Should be harsher than speaking during input
        r_input = self._fn(
            spoke=True,
            in_eom=False,
            eom_steps=0,
            max_speak_steps=20,
        )
        assert r < r_input


class TestRewardIntegration:
    def test_reward_connection_runs(self, region1, motor, encoder):
        """Reward connection runs without error and collects metrics."""
        tokens = [(i % 4, chr(ord("a") + i % 4)) for i in range(30)]
        cortex = Topology(encoder)
        cortex.add_region("S1", region1, entry=True)
        cortex.add_region("M1", motor)
        cortex.connect("S1", "M1", "feedforward")
        cortex.connect("M1", "M1", "feedforward", reward_modulator=RewardModulator())
        result = cortex.run(tokens, log_interval=1000)
        m1_metrics = result.per_region["M1"]
        assert len(m1_metrics.motor_rewards) > 0

    def test_reward_with_eom_tokens(self, region1, motor, encoder):
        """EOM tokens trigger turn-taking state changes."""
        tokens = [
            (0, "a"),
            (1, "b"),
            (2, "c"),
            (EOM_TOKEN, ""),
            (0, "a"),
            (1, "b"),
            (STORY_BOUNDARY, ""),
            (0, "a"),
            (1, "b"),
        ]
        cortex = Topology(encoder)
        cortex.add_region("S1", region1, entry=True)
        cortex.add_region("M1", motor)
        cortex.connect("S1", "M1", "feedforward")
        cortex.connect("M1", "M1", "feedforward", reward_modulator=RewardModulator())
        result = cortex.run(tokens, log_interval=1000)
        assert len(result.per_region["M1"].motor_rewards) > 0

    def test_reward_modulators_in_result(self, region1, motor, encoder):
        """reward_modulators dict is populated when reward connections exist."""
        tokens = [(i % 4, chr(ord("a") + i % 4)) for i in range(30)]
        cortex = Topology(encoder)
        cortex.add_region("S1", region1, entry=True)
        cortex.add_region("M1", motor)
        cortex.connect("S1", "M1", "feedforward")
        cortex.connect("M1", "M1", "feedforward", reward_modulator=RewardModulator())
        result = cortex.run(tokens, log_interval=1000)
        assert "M1" in result.reward_modulators
        assert len(result.reward_modulators["M1"]) > 0

    def test_no_reward_backward_compatible(self, region1, motor, encoder):
        """Without reward connections, reward_modulators is empty."""
        tokens = [(i % 4, chr(ord("a") + i % 4)) for i in range(20)]
        cortex = Topology(encoder)
        cortex.add_region("S1", region1, entry=True)
        cortex.add_region("M1", motor)
        cortex.connect("S1", "M1", "feedforward")
        result = cortex.run(tokens, log_interval=1000)
        assert result.reward_modulators == {}


class TestEomTokenInjection:
    def test_inject_adds_eom_before_boundaries(self):
        tokens = [
            (0, "a"),
            (1, "b"),
            (STORY_BOUNDARY, ""),
            (2, "c"),
            (STORY_BOUNDARY, ""),
        ]
        result = inject_eom_tokens(tokens, speak_window=2)
        # EOM + 2 repeated tokens + BOUNDARY for each
        assert result == [
            (0, "a"),
            (1, "b"),
            (EOM_TOKEN, ""),
            (1, "b"),
            (1, "b"),  # speak window repeats last token
            (STORY_BOUNDARY, ""),
            (2, "c"),
            (EOM_TOKEN, ""),
            (2, "c"),
            (2, "c"),
            (STORY_BOUNDARY, ""),
        ]

    def test_inject_no_boundaries(self):
        tokens = [(0, "a"), (1, "b"), (2, "c")]
        result = inject_eom_tokens(tokens, speak_window=0)
        assert result == tokens

    def test_inject_preserves_token_content(self):
        tokens = [(65, "A"), (66, "B"), (STORY_BOUNDARY, "")]
        result = inject_eom_tokens(tokens, speak_window=0)
        assert (65, "A") in result
        assert (66, "B") in result

    def test_inject_with_segment_length(self):
        """segment_length creates synthetic turn boundaries."""
        tokens = [(i, chr(ord("a") + i)) for i in range(10)]
        result = inject_eom_tokens(
            tokens,
            segment_length=3,
            speak_window=0,
        )
        eom_count = sum(1 for tid, _ in result if tid == EOM_TOKEN)
        assert eom_count == 3  # at positions 3, 6, 9

    def test_segment_length_zero_no_synthetic(self):
        """segment_length=0 only injects at natural boundaries."""
        tokens = [(i, chr(ord("a") + i)) for i in range(10)]
        result = inject_eom_tokens(tokens, segment_length=0)
        assert result == tokens

    def test_speak_window_adds_tokens(self):
        """speak_window pads EOM phase with repeated tokens."""
        tokens = [
            (0, "a"),
            (1, "b"),
            (STORY_BOUNDARY, ""),
        ]
        result = inject_eom_tokens(tokens, speak_window=5)
        # EOM + 5 repeated last tokens + BOUNDARY
        eom_idx = next(i for i, (tid, _) in enumerate(result) if tid == EOM_TOKEN)
        # 5 tokens between EOM and BOUNDARY
        boundary_idx = next(
            i for i in range(eom_idx + 1, len(result)) if result[i][0] == STORY_BOUNDARY
        )
        assert boundary_idx - eom_idx - 1 == 5


class TestTurnTakingCounters:
    def test_counters_with_eom(self, region1, motor, encoder):
        """Turn-taking counters accumulate correctly with EOM tokens."""
        tokens = [
            # Input phase: 3 tokens
            (0, "a"),
            (1, "b"),
            (2, "c"),
            # EOM: M1 should speak
            (EOM_TOKEN, ""),
            (0, "a"),
            (1, "b"),
            # Reset
            (STORY_BOUNDARY, ""),
            # Another input phase
            (0, "a"),
            (1, "b"),
            (2, "c"),
        ]
        cortex = Topology(encoder)
        cortex.add_region("S1", region1, entry=True)
        cortex.add_region("M1", motor)
        cortex.connect("S1", "M1", "feedforward")
        cortex.connect("M1", "M1", "feedforward", reward_modulator=RewardModulator())
        result = cortex.run(tokens, log_interval=1000)
        m = result.per_region["M1"]
        # Should have counted both phases
        assert m.turn_input_steps > 0
        assert m.turn_eom_steps > 0
        # Totals should add up: every motor step is one of the 4 categories
        total = (
            m.turn_interruptions
            + m.turn_correct_silent
            + m.turn_correct_speak
            + m.turn_unresponsive
            + m.turn_rambles
        )
        assert total == m.turn_input_steps + m.turn_eom_steps

    def test_counters_without_eom(self, region1, motor, encoder):
        """Without EOM, all steps are input phase."""
        tokens = [(i % 4, chr(ord("a") + i % 4)) for i in range(20)]
        cortex = Topology(encoder)
        cortex.add_region("S1", region1, entry=True)
        cortex.add_region("M1", motor)
        cortex.connect("S1", "M1", "feedforward")
        cortex.connect("M1", "M1", "feedforward", reward_modulator=RewardModulator())
        result = cortex.run(tokens, log_interval=1000)
        m = result.per_region["M1"]
        assert m.turn_eom_steps == 0
        assert m.turn_input_steps > 0
        assert m.turn_unresponsive == 0
        assert m.turn_correct_speak == 0
        assert m.turn_rambles == 0


class TestEchoReward:
    def _make_echo(self, heard: str, **kwargs) -> EchoReward:
        er = EchoReward(**kwargs)
        for ch in heard:
            er.hear(ch)
        er.start_speak()
        return er

    def test_exact_match_positive(self):
        """Exact position match gives positive RPE."""
        er = self._make_echo("abc")
        r = er.step("a", 0.5)
        # match_score=1.0 minus small baseline (~0.03) → positive RPE
        assert r > 0

    def test_wrong_char_negative(self):
        """Wrong char at a position with no nearby match gives negative RPE."""
        er = self._make_echo("abc")
        r = er.step("z", 0.5)
        # match_score=0.0 minus baseline → negative RPE
        assert r < 0

    def test_no_anywhere_in_word_credit(self):
        """Char present in word but far from position gets zero match score.

        This is the 'h' attractor fix: 'h' in "the" should NOT get credit
        at position 2 (target 'e') because 'h' at position 1 is within
        tolerance — but if we're checking position 0 (target 't'), 'h'
        at position 1 is within tolerance so it DOES get credit there.

        The key test: for a long word, a char far from echo_pos should
        get zero credit.
        """
        er = self._make_echo("abcdef")
        # Advance past first 4 positions
        for ch in "abcd":
            er.step(ch, 0.5)
        # Now at echo_pos=4 (target 'e'). 'a' is at position 0, dist=4.
        # With tolerance=1, near-position checks [3,6), 'a' not there.
        # Without anywhere-in-word, match_score should be 0.
        r = er.step("a", 0.5)
        # match_score=0 → RPE is negative (0 - baseline), so total echo
        # contribution is negative. The curiosity base may offset slightly,
        # but the match RPE component should drag it down.
        # Verify by checking match_score would have been non-zero before fix:
        # 'a' IS in the heard word (pos 0), but dist=4 > tolerance=1.
        assert r < 0

    def test_h_no_free_riding_at_distant_positions(self):
        """'h' should not get credit at positions far from where it appears.

        Regression test for the 'h' attractor bug: in words like "the",
        'h' was getting partial credit at every position via the
        anywhere-in-word fallback.
        """
        # "beautiful" — 'h' is not in this word at all
        er = self._make_echo("beautiful")
        r = er.step("h", 0.5)
        # 'h' not anywhere near position 0 → zero match → negative RPE
        assert r < 0

    def test_near_position_gets_partial_credit(self):
        """Char at position ±1 from target gets partial credit (not full)."""
        er = self._make_echo("ab")
        # echo_pos=0, target='a', produce 'b' (which is at position 1, dist=1)
        r = er.step("b", 0.5)
        # match_score = 1.0/(1+1) = 0.5, baseline ~0.03 → positive RPE
        assert r > 0

    def test_stats_tracking(self):
        er = self._make_echo("abc")
        er.step("a", 0.5)  # exact match
        er.step("x", 0.5)  # miss
        er.step("c", 0.5)  # exact match
        assert er.exact_matches == 2
        assert er.chars_echoed == 3

    def test_reset_clears_state(self):
        er = self._make_echo("abc")
        er.step("a", 0.5)
        er.reset()
        assert not er._in_speak_phase
        assert len(er._heard) == 0
        assert er._echo_pos == 0
