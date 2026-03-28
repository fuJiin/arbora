"""Chat-specific reporter for periodic training log lines.

Reads typed probe snapshots (LaminaProbe, ChatMotorProbe) and prints
formatted progress lines. No learning, no circuit access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from step.probes.chat import ChatMotorProbe
    from step.probes.core import LaminaProbe


def _rolling_mean(vals: list[float], window: int) -> float:
    if not vals:
        return 0.0
    tail = vals[-window:]
    return sum(tail) / len(tail)


class ChatReporter:
    """Periodic log lines from typed probe snapshots for chat training."""

    def __init__(self, *, log_interval: int = 100, rolling_window: int = 100):
        self._log_interval = log_interval
        self._rolling_window = rolling_window

    def log_at_interval(
        self,
        t: int,
        elapsed: float,
        *,
        lamina: LaminaProbe | None = None,
        motor: ChatMotorProbe | None = None,
        surprise_modulators: dict[str, list[float]] | None = None,
        thalamic_readiness: dict[str, list[float]] | None = None,
        reward_modulators: dict[str, list[float]] | None = None,
    ) -> None:
        """Print a log line if t is at a log interval."""
        if t == 0 or t % self._log_interval != 0:
            return
        self._log(
            t,
            elapsed,
            lamina=lamina,
            motor=motor,
            surprise_modulators=surprise_modulators or {},
            thalamic_readiness=thalamic_readiness or {},
            reward_modulators=reward_modulators or {},
        )

    def _log(
        self,
        t: int,
        elapsed: float,
        *,
        lamina: LaminaProbe | None,
        motor: ChatMotorProbe | None,
        surprise_modulators: dict[str, list[float]],
        thalamic_readiness: dict[str, list[float]],
        reward_modulators: dict[str, list[float]],
    ) -> None:
        """Format and print a log line from typed probe snapshots."""
        rw = self._rolling_window

        # Lamina metrics (first region)
        lamina_str = ""
        if lamina is not None:
            for _rn, snap in lamina.snapshot().items():
                burst = 1.0 - snap.l4.recall
                lamina_str = (
                    f"recall={snap.l4.recall:.2f} "
                    f"prec={snap.l4.precision:.2f} "
                    f"sparse={snap.l4.sparseness:.2f} "
                    f"burst={burst:.0%} "
                    f"dim={snap.l23.eff_dim:.1f}"
                )
                lp = getattr(snap.l23, "linear_probe", 0.0)
                if lp > 0:
                    lamina_str += f" lprobe={lp:.2f}"
                break

        # Motor metrics (first region)
        motor_str = ""
        if motor is not None:
            for _rn, m in motor.snapshot().items():
                if m.motor_accuracies:
                    motor_str += f" M1={_rolling_mean(m.motor_accuracies, rw):.4f}"
                if m.bg_gate_values:
                    motor_str += f" bg={_rolling_mean(m.bg_gate_values, rw):.2f}"
                if m.turn_eom_steps > 0 or m.turn_input_steps > 0:
                    intr = (
                        m.turn_interruptions / m.turn_input_steps
                        if m.turn_input_steps > 0
                        else 0
                    )
                    unr = (
                        m.turn_unresponsive / m.turn_eom_steps
                        if m.turn_eom_steps > 0
                        else 0
                    )
                    motor_str += f" int={intr:.0%} unr={unr:.0%}"
                break

        # Modulator info
        mod_str = ""
        for _tgt, mods in surprise_modulators.items():
            if mods:
                mod_str += f" mod={_rolling_mean(mods, rw):.2f}"
                break
        for _key, vals in thalamic_readiness.items():
            if vals:
                mod_str += f" gate={_rolling_mean(vals, rw):.2f}"
                break
        for _tgt, rews in reward_modulators.items():
            if rews:
                mod_str += f" rew={_rolling_mean(rews, rw):.2f}"
                break

        print(f"  t={t:,} {lamina_str}{motor_str}{mod_str} ({elapsed:.1f}s)")
