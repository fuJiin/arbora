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
            lamina_snap = lamina.snapshot()
            for _rn, region_snap in lamina_snap.items():
                l4 = region_snap.get("l4", {})
                l23 = region_snap.get("l23", {})
                recall = l4.get("recall", 0.0)
                precision = l4.get("precision", 0.0)
                sparseness = l4.get("sparseness", 0.0)
                eff_dim = l23.get("eff_dim", 0.0)
                burst = 1.0 - recall
                lamina_str = (
                    f"recall={recall:.2f} "
                    f"prec={precision:.2f} "
                    f"sparse={sparseness:.2f} "
                    f"burst={burst:.0%} "
                    f"dim={eff_dim:.1f}"
                )
                lp = l23.get("linear_probe")
                if lp is not None and lp > 0:
                    lamina_str += f" lprobe={lp:.2f}"
                break

        # Motor metrics (first region)
        motor_str = ""
        if motor is not None:
            motor_snap = motor.snapshot()
            for _rn, m in motor_snap.items():
                accs = m.get("motor_accuracies", [])
                if accs:
                    motor_str += f" M1={_rolling_mean(accs, rw):.4f}"
                gates = m.get("bg_gate_values", [])
                if gates:
                    motor_str += f" bg={_rolling_mean(gates, rw):.2f}"
                eom = m.get("turn_eom_steps", 0)
                inp = m.get("turn_input_steps", 0)
                if eom > 0 or inp > 0:
                    intr = m.get("turn_interruptions", 0) / inp if inp > 0 else 0
                    unr = m.get("turn_unresponsive", 0) / eom if eom > 0 else 0
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
