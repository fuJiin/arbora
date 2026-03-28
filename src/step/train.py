"""Chat training loop with probe-based telemetry.

Usage::

    env = ChatEnv(tokens, babble_ratio=0.3)
    agent = ChatAgent(encoder=encoder, circuit=circuit)
    result = train(env, agent, probes=[LaminaProbe()], log_interval=100)
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from step.cortex.circuit_types import ConnectionRole
from step.cortex.motor import MotorRegion
from step.environment import ChatEnv
from step.probes.chat import ChatMotorProbe
from step.probes.core import LaminaProbe, Probe
from step.reporting.chat import ChatReporter

if TYPE_CHECKING:
    from step.agent import ChatAgent
    from step.cortex.circuit import Circuit


@dataclass
class TrainResult:
    """Result of a train() run. All metrics live in probe_snapshots."""

    probe_snapshots: dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    # Inter-region signal time series
    surprise_modulators: dict[str, list[float]] = field(default_factory=dict)
    thalamic_readiness: dict[str, list[float]] = field(default_factory=dict)
    reward_modulators: dict[str, list[float]] = field(default_factory=dict)


def train(
    env: ChatEnv,
    agent: ChatAgent,
    *,
    log_interval: int = 100,
    rolling_window: int = 100,
    probes: Sequence[Probe] = (),
    decoder_training: bool = False,
) -> TrainResult:
    """Run training loop: ChatEnv drives observations, ChatAgent acts.

    All metrics flow through probes. ChatReporter prints periodic log
    lines from probe snapshots.

    Args:
        env: ChatEnv providing observations and computing reward.
        agent: ChatAgent wrapping encoder + circuit + decoder.
        log_interval: Steps between log lines.
        rolling_window: Window for rolling averages in log lines.
        probes: Sequence of Probe objects to observe circuit state.
        decoder_training: If True, train entry/motor decoders every step.
            Used for REPL warmup. Default False for experiments.
    """
    circuit = agent.circuit
    reporter = ChatReporter(log_interval=log_interval, rolling_window=rolling_window)
    start = time.monotonic()

    # Initialize modulator tracking from circuit connections
    surprise_mods: dict[str, list[float]] = {}
    thalamic_ready: dict[str, list[float]] = {}
    reward_mods: dict[str, list[float]] = {}
    for conn in circuit._connections:
        if conn.surprise_tracker is not None:
            surprise_mods[conn.target] = []
        if conn.thalamic_gate is not None:
            thalamic_ready[f"{conn.source}->{conn.target}"] = []
        if conn.reward_modulator is not None:
            reward_mods[conn.target] = []

    # Resolve typed probes for reporter (once, not per step)
    _lamina_probe: LaminaProbe | None = None
    _motor_probe: ChatMotorProbe | None = None
    for p in probes:
        if isinstance(p, LaminaProbe) and _lamina_probe is None:
            _lamina_probe = p
        if isinstance(p, ChatMotorProbe) and _motor_probe is None:
            _motor_probe = p

    obs = env.reset()
    t = 0

    while not env.done:
        if obs.is_boundary:
            # Reset reward modulators at dialogue boundary
            for conn in circuit._connections:
                if conn.reward_modulator is not None:
                    conn.reward_modulator.reset()
            if circuit._reward_source is not None:
                _rs_reset = getattr(circuit._reward_source, "reset", None)
                if _rs_reset is not None:
                    _rs_reset()
            circuit.reset()
            agent._motor_active = False
            agent.last_action = None
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

        # Snapshot L2/3 before processing (needed for decoder training)
        prev_l23 = None
        prev_motor_l23: dict[str, object] = {}
        if decoder_training:
            entry_name = circuit._entry_name
            assert entry_name is not None
            entry_region = circuit._regions[entry_name].region
            prev_l23 = entry_region.l23.active.copy()
            # Motor L2/3 snapshots
            for _mn, _ms in circuit._regions.items():
                if _ms.motor and _ms.motor_decoder is not None:
                    prev_motor_l23[_mn] = _ms.region.l23.active.copy()

        # Efference copy (before process so it's available this step)
        motor_active = agent._motor_active or agent.force_gate_open
        if agent.last_action is not None and agent.force_gate_open:
            entry = circuit.region(agent._entry_name)
            action_char = chr(agent.last_action) if agent.last_action < 128 else ""
            ef = agent.encoder.encode(action_char)
            entry.set_efference_copy(ef)

        # Neural processing
        output = circuit.process(encoding, motor_active=motor_active)
        agent.last_output = output

        # Token-level motor learning
        if motor_active:
            for s in circuit._regions.values():
                if s.motor and isinstance(s.region, MotorRegion):
                    s.region.observe_token(obs.token_id)

        # Probes: observe circuit state after processing
        for probe in probes:
            probe.observe(
                circuit,
                stimulus_id=obs.token_id,
                in_eom=env.in_eom,
                eom_steps=env.eom_steps,
            )

        # Record inter-region signal values
        _record_modulators(circuit, surprise_mods, thalamic_ready, reward_mods)

        # Diagnostics / timeline capture
        _capture_diagnostics(circuit, t)

        # Optional decoder training (REPL warmup)
        if decoder_training:
            _train_decoders(
                circuit,
                obs.token_id,
                obs.token_str,
                encoding,
                prev_l23,
                prev_motor_l23,
            )

        # Periodic logging from probes
        elapsed = time.monotonic() - start
        reporter.log_at_interval(
            t,
            elapsed,
            lamina=_lamina_probe,
            motor=_motor_probe,
            surprise_modulators=surprise_mods,
            thalamic_readiness=thalamic_ready,
            reward_modulators=reward_mods,
        )

        # Decode action
        action = agent._decode_action()
        agent.last_action = action

        # Step environment
        obs, _reward = env.step(action)
        t += 1

    # Build result
    elapsed = time.monotonic() - start
    result = TrainResult(
        elapsed_seconds=elapsed,
        surprise_modulators=surprise_mods,
        thalamic_readiness=thalamic_ready,
        reward_modulators=reward_mods,
    )
    for probe in probes:
        result.probe_snapshots[probe.name] = probe.snapshot()
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    # Reward modulators (from motor connections)
    for _name, s in circuit._regions.items():
        if s.motor:
            for conn in circuit._connections:
                if conn.source == _name and conn.reward_modulator is not None:
                    tgt = circuit._regions[conn.target].region
                    reward_mods[conn.target].append(tgt.reward_modulator)


def _capture_diagnostics(circuit: Circuit, t: int) -> None:
    """Capture diagnostics and timeline snapshots."""
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
    import numpy as np

    entry_name = circuit._entry_name
    assert entry_name is not None
    entry_state = circuit._regions[entry_name]
    entry_region = entry_state.region

    # Entry decoders
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

    # Motor decoders
    for _name, s in circuit._regions.items():
        if s.motor_decoder is not None and _name in prev_motor_l23:
            s.motor_decoder.observe(token_id, prev_motor_l23[_name])

    # Word decoders (non-entry regions)
    for _name, s in circuit._regions.items():
        if s.word_decoder is not None:
            s.word_decoder.step(token_str, s.region.l23.firing_rate)
