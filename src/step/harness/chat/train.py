"""Chat training harness with probe-based telemetry.

Usage::

    harness = ChatTrainHarness(
        env=ChatEnv(tokens),
        agent=ChatAgent(encoder=encoder, circuit=circuit),
        probes=[LaminaProbe(), ChatMotorProbe()],
    )
    result = harness.run()
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from step.cortex.circuit_types import ConnectionRole
from step.probes.chat import ChatMotorProbe
from step.probes.core import LaminaProbe, Probe
from step.reporting.chat import ChatReporter

if TYPE_CHECKING:
    from step.agent import ChatAgent
    from step.cortex.circuit import Circuit
    from step.environment import ChatEnv


@dataclass
class TrainResult:
    """Result of a training run. All metrics live in probe_snapshots."""

    probe_snapshots: dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    # Inter-region signal time series (TODO: move to ModulatorProbe)
    surprise_modulators: dict[str, list[float]] = field(default_factory=dict)
    thalamic_readiness: dict[str, list[float]] = field(default_factory=dict)
    reward_modulators: dict[str, list[float]] = field(default_factory=dict)


class ChatTrainHarness:
    """Train a Circuit on chat token sequences with probe telemetry.

    Wraps the ChatEnv + ChatAgent loop with:
    - Probe observation after each agent.act() step
    - ChatReporter for periodic log lines
    - Optional decoder training (for REPL warmup)
    - Modulator tracking

    The harness operates on the agent abstraction, not the circuit
    directly. The agent owns encoding, processing, decoding, and reset.
    """

    def __init__(
        self,
        env: ChatEnv,
        agent: ChatAgent,
        *,
        probes: Sequence[Probe] = (),
        log_interval: int = 100,
        rolling_window: int = 100,
        decoder_training: bool = False,
    ):
        self._env = env
        self._agent = agent
        self._probes = probes
        self._decoder_training = decoder_training
        self._reporter = ChatReporter(
            log_interval=log_interval, rolling_window=rolling_window
        )

        # Resolve typed probes for reporter
        self._lamina_probe: LaminaProbe | None = None
        self._motor_probe: ChatMotorProbe | None = None
        for p in probes:
            if isinstance(p, LaminaProbe) and self._lamina_probe is None:
                self._lamina_probe = p
            if isinstance(p, ChatMotorProbe) and self._motor_probe is None:
                self._motor_probe = p

        # Initialize modulator tracking from circuit connections
        # TODO(STEP-86): move to ModulatorProbe
        circuit = agent.circuit
        self._surprise_mods: dict[str, list[float]] = {}
        self._thalamic_ready: dict[str, list[float]] = {}
        self._reward_mods: dict[str, list[float]] = {}
        for conn in circuit._connections:
            if conn.surprise_tracker is not None:
                self._surprise_mods[conn.target] = []
            if conn.thalamic_gate is not None:
                self._thalamic_ready[f"{conn.source}->{conn.target}"] = []
            if conn.reward_modulator is not None:
                self._reward_mods[conn.target] = []

    def run(self) -> TrainResult:
        """Execute the training loop. Returns TrainResult with probe snapshots."""
        env = self._env
        agent = self._agent
        circuit = agent.circuit
        probes = self._probes
        start = time.monotonic()

        obs = env.reset()
        t = 0

        while not env.done:
            if obs.is_boundary:
                agent.reset()
                for probe in probes:
                    if hasattr(probe, "boundary"):
                        probe.boundary()
                obs, _reward = env.step(None)
                continue

            if obs.is_eom:
                agent._motor_active = True
                obs, _reward = env.step(None)
                continue

            # Encode
            encoding = agent.encoder.encode(obs.token_str)
            agent.last_encoding = encoding
            agent.last_token_str = obs.token_str

            # Snapshot L2/3 before processing (for decoder training)
            prev_l23 = None
            prev_motor_l23: dict[str, object] = {}
            if self._decoder_training:
                prev_l23, prev_motor_l23 = _snapshot_l23(circuit)

            # TODO(STEP-87): move to agent.step() — harness shouldn't
            # touch circuit directly. Split act() into step() + decode().
            motor_active = agent._motor_active or agent.force_gate_open
            if agent.last_action is not None and agent.force_gate_open:
                entry = circuit.region(agent._entry_name)
                action_char = chr(agent.last_action) if agent.last_action < 128 else ""
                ef = agent.encoder.encode(action_char)
                entry.set_efference_copy(ef)

            output = circuit.process(encoding, motor_active=motor_active)
            agent.last_output = output

            # Probes observe circuit state
            for probe in probes:
                probe.observe(
                    circuit,
                    stimulus_id=obs.token_id,
                    in_eom=env.in_eom,
                    eom_steps=env.eom_steps,
                )

            # Modulator tracking (TODO: move to ModulatorProbe)
            _record_modulators(
                circuit,
                self._surprise_mods,
                self._thalamic_ready,
                self._reward_mods,
            )

            # Diagnostics / timeline capture
            # TODO(STEP-87): move to agent or viz probe
            _capture_diagnostics(circuit, t)

            # Optional decoder training
            if self._decoder_training:
                _train_decoders(
                    circuit,
                    obs.token_id,
                    obs.token_str,
                    encoding,
                    prev_l23,
                    prev_motor_l23,
                )

            # Periodic logging
            elapsed = time.monotonic() - start
            self._reporter.log_at_interval(
                t,
                elapsed,
                lamina=self._lamina_probe,
                motor=self._motor_probe,
                surprise_modulators=self._surprise_mods,
                thalamic_readiness=self._thalamic_ready,
                reward_modulators=self._reward_mods,
            )

            # Decode action + step env
            action = agent._decode_action()
            agent.last_action = action
            obs, _reward = env.step(action)
            t += 1

        # Build result
        elapsed = time.monotonic() - start
        result = TrainResult(
            elapsed_seconds=elapsed,
            surprise_modulators=self._surprise_mods,
            thalamic_readiness=self._thalamic_ready,
            reward_modulators=self._reward_mods,
        )
        for probe in probes:
            result.probe_snapshots[probe.name] = probe.snapshot()
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _snapshot_l23(circuit: Circuit) -> tuple[object, dict[str, object]]:
    """Snapshot L2/3 state before processing (for decoder training)."""
    entry_name = circuit._entry_name
    assert entry_name is not None
    entry_region = circuit._regions[entry_name].region
    prev_l23 = entry_region.l23.active.copy()
    prev_motor_l23: dict[str, object] = {}
    for _mn, _ms in circuit._regions.items():
        if _ms.motor and _ms.motor_decoder is not None:
            prev_motor_l23[_mn] = _ms.region.l23.active.copy()
    return prev_l23, prev_motor_l23


def _record_modulators(
    circuit: Circuit,
    surprise_mods: dict[str, list[float]],
    thalamic_ready: dict[str, list[float]],
    reward_mods: dict[str, list[float]],
) -> None:
    """Record inter-region signal values from connection objects."""
    for conn in circuit._connections:
        if not conn.enabled:
            continue
        if conn.surprise_tracker is not None:
            surprise_mods[conn.target].append(conn.surprise_tracker.modulator)
        if conn.role == ConnectionRole.APICAL and conn.thalamic_gate is not None:
            tgt = circuit._regions[conn.target].region
            if tgt.has_apical:
                key = f"{conn.source}->{conn.target}"
                thalamic_ready[key].append(conn.thalamic_gate.readiness)
    for _name, s in circuit._regions.items():
        if s.motor:
            for conn in circuit._connections:
                if conn.source == _name and conn.reward_modulator is not None:
                    tgt = circuit._regions[conn.target].region
                    reward_mods[conn.target].append(tgt.reward_modulator)


def _capture_diagnostics(circuit: Circuit, t: int) -> None:
    """Capture diagnostics and timeline snapshots (viz observability)."""
    for _name, s in circuit._regions.items():
        if s.diagnostics is not None:
            s.diagnostics.step(t, s.region)
        if s.timeline is not None and t % circuit._timeline_interval == 0:
            s.timeline.capture(
                len(s.timeline.frames),
                s.region,
                s.region.last_column_drive,
            )


def _train_decoders(
    circuit: Circuit,
    token_id: int,
    token_str: str,
    encoding: object,
    prev_l23: object,
    prev_motor_l23: dict[str, object],
) -> None:
    """Train entry and motor decoders (opt-in, for REPL warmup)."""
    entry_name = circuit._entry_name
    assert entry_name is not None
    entry_state = circuit._regions[entry_name]
    entry_region = entry_state.region

    if (
        entry_state.decode_index is not None
        and token_id not in entry_state.decode_index._token_id_to_idx
    ):
        active_set = frozenset(int(i) for i in np.nonzero(entry_region.l4.active)[0])
        entry_state.decode_index.observe(token_id, active_set)
    if entry_state.syn_decoder is not None:
        entry_state.syn_decoder.observe(
            token_id, token_str, encoding, entry_region.active_columns
        )
    if entry_state.dendritic_decoder is not None and prev_l23 is not None:
        entry_state.dendritic_decoder.observe(token_id, prev_l23)

    for _name, s in circuit._regions.items():
        if s.motor_decoder is not None and _name in prev_motor_l23:
            s.motor_decoder.observe(token_id, prev_motor_l23[_name])

    for _name, s in circuit._regions.items():
        if s.word_decoder is not None:
            s.word_decoder.step(token_str, s.region.l23.firing_rate)
