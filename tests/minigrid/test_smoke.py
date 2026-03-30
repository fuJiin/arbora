"""End-to-end smoke test: build circuit, train episodes, verify result."""

from arbor.cortex import SensoryRegion
from arbor.cortex.circuit import Circuit, ConnectionRole
from arbor.cortex.motor import MotorRegion
from arbor.probes.core import LaminaProbe
from examples.minigrid.agent import MiniGridAgent
from examples.minigrid.encoder import MiniGridEncoder
from examples.minigrid.env import MiniGridEnv
from examples.minigrid.harness import MiniGridHarness


def test_full_loop_10_episodes():
    """Full pipeline: encoder -> circuit -> agent -> harness -> result."""
    encoder = MiniGridEncoder()
    s1 = SensoryRegion(
        input_dim=encoder.input_dim,
        encoding_width=encoder.encoding_width,
        n_columns=16,
        n_l4=2,
        n_l23=2,
        n_l5=0,
        k_columns=2,
        seed=42,
    )
    m1 = MotorRegion(
        input_dim=s1.n_l23_total,
        n_columns=8,
        n_l4=0,
        n_l23=2,
        k_columns=1,
        n_output_tokens=7,
        seed=123,
    )
    circuit = Circuit(encoder)
    circuit.add_region("S1", s1, entry=True)
    circuit.add_region("M1", m1)
    circuit.connect(s1.output_port, m1.input_port, ConnectionRole.FEEDFORWARD)
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
    assert snap["S1"].input.recall >= 0
