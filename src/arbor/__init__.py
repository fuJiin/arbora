"""Arbor: biologically-plausible cortical learning framework.

Build circuits from cortical regions, connect them, and train with
Hebbian + three-factor reward-modulated learning. No backprop.

Quick start::

    from arbor import Circuit, SensoryRegion, MotorRegion, ConnectionRole

    s1 = SensoryRegion(input_dim=100, n_columns=32, n_l4=4, n_l23=4, k_columns=4)
    m1 = MotorRegion(input_dim=s1.n_l23_total, n_columns=16, n_l4=0, n_l23=4,
                     k_columns=2, n_output_tokens=7)

    circuit = Circuit(encoder)
    circuit.add_region("S1", s1, entry=True)
    circuit.add_region("M1", m1)
    circuit.connect(s1.output_port, m1.input_port, ConnectionRole.FEEDFORWARD)
    circuit.finalize()

    output = circuit.process(encoding)
"""

__version__ = "0.1.0"

# Building blocks
# Base classes
from arbor.agent import BaseAgent, TrainResult
from arbor.basal_ganglia import BasalGangliaRegion
from arbor.config import PlasticityRule
from arbor.cortex import (
    Circuit,
    Connection,
    ConnectionRole,
    CorticalRegion,
    Lamina,
    LaminaID,
    MotorRegion,
    PFCRegion,
    PremotorRegion,
    RewardModulator,
    SensoryRegion,
    SurpriseTracker,
    ThalamicGate,
)

# Protocols
from arbor.cortex.circuit_types import Encoder, Region
from arbor.neuron_group import NeuronGroup

# Probes
from arbor.probes.core import LaminaProbe, Probe

__all__ = [
    "BasalGangliaRegion",
    "BaseAgent",
    "Circuit",
    "Connection",
    "ConnectionRole",
    "CorticalRegion",
    "Encoder",
    "Lamina",
    "LaminaID",
    "LaminaProbe",
    "MotorRegion",
    "NeuronGroup",
    "PFCRegion",
    "PlasticityRule",
    "PremotorRegion",
    "Probe",
    "Region",
    "RewardModulator",
    "SensoryRegion",
    "SurpriseTracker",
    "ThalamicGate",
    "TrainResult",
    "__version__",
]
