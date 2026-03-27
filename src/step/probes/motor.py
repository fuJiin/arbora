"""Chat-specific motor probe for turn-taking and motor output metrics.

ChatMotorProbe accumulates motor accuracy, BG gate values, turn-taking
counters, and reward signals. Chat-prefixed because turn-taking
(EOM/input phases) is dialogue-specific.

Reads circuit motor region state directly (friend access). Does NOT
train motor decoders — that stays in RunHooks (future ChatReplRunner).
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from step.cortex.motor import MotorRegion

if TYPE_CHECKING:
    from step.cortex.circuit import Circuit


class ChatMotorProbe:
    """Per-region motor metrics accumulator for chat environments.

    Tracks motor output accuracy, BG gating, turn-taking behavior,
    and reward signals. Reads motor region state after each process().
    """

    name: str = "motor"
    MAX_SPEAK_STEPS: int = 20

    def __init__(self):
        # Per-region accumulators
        self._motor_accuracies: dict[str, list[float]] = defaultdict(list)
        self._motor_confidences: dict[str, list[float]] = defaultdict(list)
        self._motor_rewards: dict[str, list[float]] = defaultdict(list)
        self._bg_gate_values: dict[str, list[float]] = defaultdict(list)
        # Turn-taking counters
        self._turn_eom_steps: dict[str, int] = defaultdict(int)
        self._turn_input_steps: dict[str, int] = defaultdict(int)
        self._turn_correct_speak: dict[str, int] = defaultdict(int)
        self._turn_correct_silent: dict[str, int] = defaultdict(int)
        self._turn_interruptions: dict[str, int] = defaultdict(int)
        self._turn_unresponsive: dict[str, int] = defaultdict(int)
        self._turn_rambles: dict[str, int] = defaultdict(int)

    def observe(self, circuit: Circuit, **kwargs) -> None:
        """Read motor state from circuit after process()."""
        stimulus_id = kwargs.get("stimulus_id")

        for name, s in circuit._regions.items():
            if not s.motor:
                continue
            if not isinstance(s.region, MotorRegion):
                continue
            motor = s.region

            if circuit._total_steps == 0:
                continue

            # BG gate value
            if s.basal_ganglia is not None:
                self._bg_gate_values[name].append(motor.last_gate)

            # Motor output
            m_id, m_conf = motor.last_output
            self._motor_confidences[name].append(m_conf)

            if m_id >= 0 and stimulus_id is not None:
                self._motor_accuracies[name].append(1.0 if m_id == stimulus_id else 0.0)

            # Reward
            self._motor_rewards[name].append(motor.last_reward)

            # Turn-taking
            spoke = m_id >= 0
            if circuit._in_eom:
                self._turn_eom_steps[name] += 1
                if spoke:
                    if circuit._eom_steps > self.MAX_SPEAK_STEPS:
                        self._turn_rambles[name] += 1
                    else:
                        self._turn_correct_speak[name] += 1
                else:
                    self._turn_unresponsive[name] += 1
            else:
                self._turn_input_steps[name] += 1
                if spoke:
                    self._turn_interruptions[name] += 1
                else:
                    self._turn_correct_silent[name] += 1

    def boundary(self) -> None:
        """Reset per-dialogue turn counters."""
        # Turn counters accumulate across the full run, not per-dialogue.
        # boundary() exists for protocol conformance but is a no-op here.

    def snapshot(self) -> dict:
        """Return per-region motor metrics."""
        all_regions = set(
            list(self._motor_confidences.keys()) + list(self._bg_gate_values.keys())
        )
        result = {}
        for name in sorted(all_regions):
            result[name] = {
                "motor_accuracies": self._motor_accuracies.get(name, []),
                "motor_confidences": self._motor_confidences.get(name, []),
                "motor_rewards": self._motor_rewards.get(name, []),
                "bg_gate_values": self._bg_gate_values.get(name, []),
                "turn_eom_steps": self._turn_eom_steps.get(name, 0),
                "turn_input_steps": self._turn_input_steps.get(name, 0),
                "turn_correct_speak": self._turn_correct_speak.get(name, 0),
                "turn_correct_silent": self._turn_correct_silent.get(name, 0),
                "turn_interruptions": self._turn_interruptions.get(name, 0),
                "turn_unresponsive": self._turn_unresponsive.get(name, 0),
                "turn_rambles": self._turn_rambles.get(name, 0),
            }
        return result
