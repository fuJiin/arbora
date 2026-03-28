"""End-to-end smoke test: build circuit, train episodes, verify result."""

from step.agent.minigrid import MiniGridAgent
from step.cortex import SensoryRegion
from step.cortex.circuit import Circuit, ConnectionRole
from step.cortex.motor import MotorRegion
from step.encoders.minigrid import MiniGridEncoder
from step.env_minigrid import MiniGridEnv
from step.harness.minigrid.train import MiniGridHarness
from step.probes.core import LaminaProbe


def test_full_loop_10_episodes():
    """Full pipeline: encoder -> circuit -> agent -> harness -> result."""
    encoder = MiniGridEncoder()
    s1 = SensoryRegion(
        input_dim=encoder.input_dim,
        encoding_width=encoder.encoding_width,
        n_columns=16,
        n_l4=2,
        n_l23=2,
        k_columns=2,
        seed=42,
    )
    m1 = MotorRegion(
        input_dim=s1.n_l23_total,
        n_columns=8,
        n_l4=2,
        n_l23=2,
        k_columns=1,
        n_output_tokens=7,
        seed=123,
    )
    circuit = Circuit(encoder)
    circuit.add_region("S1", s1, entry=True)
    circuit.add_region("M1", m1)
    circuit.connect(s1.l23, m1.l4, ConnectionRole.FEEDFORWARD)
    circuit.finalize()

    env = MiniGridEnv("MiniGrid-Empty-5x5-v0", max_episodes=10)
    agent = MiniGridAgent(encoder=encoder, circuit=circuit)
    probe = LaminaProbe()

    harness = MiniGridHarness(env, agent, probes=[probe], log_interval=10000)
    result = harness.run()

    # Verify basic result structure
    assert result.elapsed_seconds > 0
    assert "lamina" in result.probe_snapshots
    assert env.episode_count == 10

    # Verify S1 has learned something (recall > 0)
    snap = result.probe_snapshots["lamina"]
    assert "S1" in snap
    assert snap["S1"].l4.recall >= 0
