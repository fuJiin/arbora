"""Modulator probe — tracks inter-region signal values.

Input-agnostic: reads surprise, thalamic, and reward modulator state
from circuit connections. Works for any environment.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from arbora.cortex.circuit_types import ConnectionRole

if TYPE_CHECKING:
    from arbora.cortex.circuit import Circuit


@dataclass
class ModulatorSnapshot:
    """Snapshot of inter-region modulator time series."""

    surprise: dict[str, list[float]] = field(default_factory=dict)
    thalamic: dict[str, list[float]] = field(default_factory=dict)
    reward: dict[str, list[float]] = field(default_factory=dict)


class ModulatorProbe:
    """Track surprise, thalamic, and reward modulator values per step.

    Reads connection state after each process() call. Each modulator
    type is keyed by target region name (surprise, reward) or
    "source->target" (thalamic).
    """

    name: str = "modulators"

    def __init__(self):
        self._surprise: dict[str, list[float]] = defaultdict(list)
        self._thalamic: dict[str, list[float]] = defaultdict(list)
        self._reward: dict[str, list[float]] = defaultdict(list)

    def observe(self, circuit: Circuit, **kwargs) -> None:
        """Record modulator values from circuit connections."""
        for conn in circuit._connections:
            if not conn.enabled:
                continue
            if conn.surprise_tracker is not None:
                self._surprise[conn.target].append(conn.surprise_tracker.modulator)
            if conn.role == ConnectionRole.APICAL and conn.thalamic_gate is not None:
                tgt = circuit._regions[conn.target].region
                if tgt.has_apical:
                    key = f"{conn.source}->{conn.target}"
                    self._thalamic[key].append(conn.thalamic_gate.readiness)

        for _name, s in circuit._regions.items():
            if s.motor:
                for conn in circuit._connections:
                    if conn.source == _name and conn.reward_modulator is not None:
                        tgt = circuit._regions[conn.target].region
                        self._reward[conn.target].append(tgt.reward_modulator)

    def snapshot(self) -> ModulatorSnapshot:
        """Return accumulated modulator time series."""
        return ModulatorSnapshot(
            surprise=dict(self._surprise),
            thalamic=dict(self._thalamic),
            reward=dict(self._reward),
        )
