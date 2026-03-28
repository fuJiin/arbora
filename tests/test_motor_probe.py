"""Tests for ChatMotorProbe — chat-specific motor and turn-taking metrics."""

from step.agent import ChatAgent
from step.cortex.basal_ganglia import BasalGanglia
from step.cortex.circuit import Circuit, ConnectionRole
from step.cortex.modulators import SurpriseTracker
from step.cortex.motor import MotorRegion
from step.cortex.sensory import SensoryRegion
from step.data import STORY_BOUNDARY
from step.encoders.charbit import CharbitEncoder
from step.environment import ChatEnv
from step.harness.chat import ChatTrainHarness
from step.probes.chat import ChatMotorProbe


def _make_motor_circuit():
    """Build S1→M1 circuit with BG."""
    encoder = CharbitEncoder(length=4, width=5, chars="abcd")
    s1 = SensoryRegion(
        input_dim=4 * 5,
        encoding_width=5,
        n_columns=8,
        n_l4=2,
        n_l23=2,
        k_columns=2,
        seed=42,
    )
    m1 = MotorRegion(
        input_dim=s1.n_l23_total,
        n_columns=4,
        n_l4=2,
        n_l23=2,
        k_columns=1,
        seed=456,
    )
    circuit = Circuit(encoder)
    circuit.add_region("S1", s1, entry=True)
    circuit.add_region("M1", m1, basal_ganglia=BasalGanglia(context_dim=9))
    circuit.connect(
        s1.l23,
        m1.l4,
        ConnectionRole.FEEDFORWARD,
        surprise_tracker=SurpriseTracker(),
    )
    return circuit, encoder


def _train_with_probe(circuit, encoder, text="abcdabcd" * 5):
    """Run train() with ChatMotorProbe wired in."""
    tokens = [
        (STORY_BOUNDARY, ""),
        *((ord(c) - ord("a"), c) for c in text),
    ]
    env = ChatEnv(tokens)
    agent = ChatAgent(encoder=encoder, circuit=circuit)
    probe = ChatMotorProbe()
    result = ChatTrainHarness(env, agent, log_interval=9999, probes=[probe]).run()
    return result, probe


class TestChatMotorProbeObserve:
    def test_bg_gate_values_accumulated(self):
        """BG gate values should be recorded for motor regions with BG."""
        circuit, encoder = _make_motor_circuit()
        result, _probe = _train_with_probe(circuit, encoder)

        snap = result.probe_snapshots["motor"]
        assert "M1" in snap
        assert len(snap["M1"].bg_gate_values) > 0

    def test_motor_confidences_accumulated(self):
        circuit, encoder = _make_motor_circuit()
        _, probe = _train_with_probe(circuit, encoder)

        snap = probe.snapshot()
        assert len(snap["M1"].motor_confidences) > 0

    def test_turn_taking_counters(self):
        """Turn-taking counters should accumulate across steps."""
        circuit, encoder = _make_motor_circuit()
        _, probe = _train_with_probe(circuit, encoder)

        snap = probe.snapshot()
        m1 = snap["M1"]
        # Input steps should be nonzero (tokens processed outside EOM)
        assert m1.turn_input_steps > 0
        # Total = input + eom
        total = m1.turn_input_steps + m1.turn_eom_steps
        assert total > 0

    def test_motor_rewards_accumulated(self):
        circuit, encoder = _make_motor_circuit()
        _, probe = _train_with_probe(circuit, encoder)

        snap = probe.snapshot()
        assert len(snap["M1"].motor_rewards) > 0


class TestChatMotorProbeNoMotor:
    def test_no_motor_regions_empty_snapshot(self):
        """Probe should produce empty snapshot for non-motor circuits."""
        from tests.conftest import make_circuit

        circuit, encoder = make_circuit()
        tokens = [
            (STORY_BOUNDARY, ""),
            *((ord(c) - ord("a"), c) for c in "abcdef"),
        ]
        env = ChatEnv(tokens)
        agent = ChatAgent(encoder=encoder, circuit=circuit)
        probe = ChatMotorProbe()
        ChatTrainHarness(env, agent, log_interval=9999, probes=[probe]).run()

        assert probe.snapshot() == {}
