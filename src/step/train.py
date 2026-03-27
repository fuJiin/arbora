"""Training loop bridging ChatEnv + ChatAgent with RunHooks metrics.

This is the transitional training function that connects the new
Environment/Agent abstractions with the existing RunHooks metrics
system. It will evolve into TrainRunner in a future PR.

Usage::

    env = ChatEnv(tokens, babble_ratio=0.3)
    agent = ChatAgent(encoder=encoder, circuit=circuit)
    result = train(env, agent, log_interval=100)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from step.cortex.circuit_hooks import RunHooks
from step.cortex.circuit_types import CortexResult
from step.cortex.motor import MotorRegion
from step.environment import ChatEnv
from step.probes.core import Probe

if TYPE_CHECKING:
    from step.agent import ChatAgent


def train(
    env: ChatEnv,
    agent: ChatAgent,
    *,
    log_interval: int = 100,
    rolling_window: int = 100,
    show_predictions: int = 0,
    metric_interval: int = 0,
    probes: Sequence[Probe] = (),
) -> CortexResult:
    """Run training loop: ChatEnv provides observations, ChatAgent acts.

    Bridges the new abstractions with RunHooks for metrics/logging.
    Returns CortexResult for backward compatibility with existing
    analysis and save_run() code.

    Args:
        env: ChatEnv providing observations and computing reward.
        agent: ChatAgent wrapping encoder + circuit + decoder.
        log_interval: Steps between log lines.
        rolling_window: Window for rolling averages.
        show_predictions: Number of recent predictions to show.
        metric_interval: Interval for expensive metrics (0 = log_interval).
    """
    circuit = agent.circuit

    hooks = RunHooks(
        circuit,
        log_interval=log_interval,
        rolling_window=rolling_window,
        show_predictions=show_predictions,
        metric_interval=metric_interval,
    )

    obs = env.reset()
    t = 0

    while not env.done:
        if obs.is_boundary:
            hooks.on_boundary(circuit)
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
        hooks._current_token_str = obs.token_str

        # Hooks: before (motor_active resolved here, passed to hooks)
        motor_active = agent._motor_active or agent.force_gate_open
        hooks.on_before_step(
            circuit, t, obs.token_id, encoding, motor_active=motor_active
        )

        # Efference copy (before process so it's available this step)
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

        # Hooks: after
        hooks.on_after_step(circuit, t, obs.token_id, encoding)

        # Decode action
        action = agent._decode_action()
        agent.last_action = action

        # Step environment (reward tracked internally by env/hooks)
        obs, _reward = env.step(action)
        t += 1

    result = hooks.finalize()
    for probe in probes:
        result.probe_snapshots[probe.name] = probe.snapshot()
    return result
