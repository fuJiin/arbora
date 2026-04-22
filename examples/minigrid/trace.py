"""Compact per-step trace for MiniGrid HC circuits.

Prints one line per harness observe() call so you can scan an
episode's behavior at a glance. Designed for devbox debugging — no
visualization dependencies, works over SSH, scales to thousands of
steps via `every=N` subsampling.

Line format
-----------
::

    t=0042 ep=1 T1=16 EC=10 DG=40 CA3=10 match=+0.12 M1=2 BG_top3=[+0.40 -0.21 -0.33]

- `t`       step index within the run (not episode)
- `ep`      1-indexed episode number
- `T1`      count of active T1 L2/3 neurons
- `EC/DG/CA3`  active-unit counts inside HC (omitted if no HC in circuit)
- `match`   CA1 cosine-similarity of CA3 vs direct-EC drive
- `M1`      chosen action id (from motor region's last_output)
- `BG_top3` highest-biased action values (from BG output port)

Usage
-----
Enable with `--trace` on the ablation CLI. Off by default; zero
overhead when unset.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, TextIO

from arbora.hippocampus import HippocampalRegion

if TYPE_CHECKING:
    from arbora.cortex.circuit import Circuit


class TraceProbe:
    """Per-step compact trace writer. Duck-typed against MiniGridHarness.

    Parameters
    ----------
    stream : TextIO
        Where to write. Default sys.stdout.
    every : int
        Print every N-th observe() call. Useful for very long runs.
    sensory_name : str
        Name of the sensory region in the circuit. Default "T1".
    motor_name : str
        Name of the motor region. Default "M1".
    bg_name : str
        Name of the basal ganglia region. Default "BG".
    """

    name = "trace"

    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        every: int = 1,
        sensory_name: str = "T1",
        motor_name: str = "M1",
        bg_name: str = "BG",
    ):
        if every < 1:
            raise ValueError(f"every must be >= 1; got {every}")
        self.stream = stream if stream is not None else sys.stdout
        self.every = every
        self.sensory_name = sensory_name
        self.motor_name = motor_name
        self.bg_name = bg_name
        self._count = 0
        self._episode = 1

    def observe(self, circuit: Circuit, step: int = 0) -> None:
        if self._count % self.every == 0:
            line = self._format_line(circuit, step)
            self.stream.write(line + "\n")
        self._count += 1

    def boundary(self) -> None:
        pass

    def episode_end(self, success: bool, steps: int, reward: float) -> None:
        self.stream.write(
            f"--- episode {self._episode} end: "
            f"steps={steps} reward={reward:.3f} term={success} ---\n"
        )
        self._episode += 1

    def snapshot(self) -> dict:
        return {"lines_emitted": self._count // self.every}

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def _format_line(self, circuit: Circuit, step: int) -> str:
        parts = [f"t={step:05d}", f"ep={self._episode}"]

        t1 = self._get_region(circuit, self.sensory_name)
        if t1 is not None and hasattr(t1, "l23"):
            parts.append(f"T1={int(t1.l23.active.sum())}")

        hc = _find_hc(circuit)
        if hc is not None:
            parts.append(f"EC={int(hc.last_ec_pattern.sum())}")
            parts.append(f"DG={int(hc.last_dg_pattern.sum())}")
            parts.append(f"CA3={int(hc.ca3.state.sum())}")
            parts.append(f"match={hc.last_match:+.2f}")

        m1 = self._get_region(circuit, self.motor_name)
        if m1 is not None and hasattr(m1, "last_output"):
            m_id, _conf = m1.last_output
            parts.append(f"M1={m_id}")

        bg = self._get_region(circuit, self.bg_name)
        if bg is not None and hasattr(bg, "output_port"):
            bias = bg.output_port.firing_rate
            # Top-3 by value, preserving sign.
            order = bias.argsort()[::-1][:3]
            top = "[" + " ".join(f"{bias[i]:+.2f}" for i in order) + "]"
            parts.append(f"BG_top3={top}")

        return " ".join(parts)

    @staticmethod
    def _get_region(circuit: Circuit, name: str):
        state = circuit._regions.get(name)
        return state.region if state is not None else None


def _find_hc(circuit: Circuit) -> HippocampalRegion | None:
    """Locate a HippocampalRegion in the circuit, or None."""
    for state in circuit._regions.values():
        if isinstance(state.region, HippocampalRegion):
            return state.region
    return None
