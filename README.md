# Arbora

Biologically-plausible cortical learning framework. Build neural circuits from cortical regions, wire them together, and train with Hebbian + three-factor reward-modulated learning. No backprop.

Named for dendritic arbors -- the branching structures through which neurons receive and integrate signals.

## Quick start

```python
from arbora import (
    Circuit, SensoryRegion, MotorRegion, BasalGangliaRegion,
    ConnectionRole, PlasticityRule,
)

# Build regions
s1 = SensoryRegion(input_dim=100, n_columns=32, n_l4=4, n_l23=4, k_columns=4)
m1 = MotorRegion(input_dim=s1.n_l23_total, n_columns=16, n_l4=0, n_l23=4,
                 k_columns=2, n_output_tokens=7)
bg = BasalGangliaRegion(input_dim=s1.n_l23_total, n_actions=7)

# Wire circuit
circuit = Circuit(encoder)
circuit.add_region("S1", s1, entry=True)
circuit.add_region("BG", bg)
circuit.add_region("M1", m1)
circuit.connect(s1.output_port, bg.input_port, ConnectionRole.FEEDFORWARD)
circuit.connect(s1.output_port, m1.input_port, ConnectionRole.FEEDFORWARD)
circuit.connect(bg.output_port, m1.input_port, ConnectionRole.MODULATORY)
circuit.finalize()

# Process input
output = circuit.process(encoding, motor_active=True)
circuit.apply_reward(reward)
```

## What this is

A framework for building biologically-grounded neural circuits that learn from reward, not gradients. Every component maps to real neuroscience:

| Arbora concept | Biology | What it does |
|---------------|---------|-------------|
| `SensoryRegion` | Granular cortex (V1, S1) | L4 input reception, L2/3 association, dendritic prediction |
| `MotorRegion` | Agranular cortex (M1) | L2/3 input, L5 action output, three-factor RL |
| `BasalGangliaRegion` | Striatum + GPi | Per-action Go/NoGo gating, tonic DA exploration |
| `ConnectionRole.FEEDFORWARD` | Corticocortical L2/3 projections | Drive, content, commands |
| `ConnectionRole.APICAL` | L1 feedback projections | Gain modulation, context, attention |
| `ConnectionRole.MODULATORY` | BG-thalamocortical loop | Action selection bias before k-WTA |
| `NeuronGroup` | Any neural population | Universal connectable surface |
| `Lamina(NeuronGroup)` | Cortical layer | Column structure, voltage, predictions, eligibility |
| Three-factor learning | DA-modulated Hebbian plasticity | dW = reward x eligibility_trace |

## Architecture

```
NeuronGroup          -- universal base (firing_rate + modulation)
  Lamina             -- cortex-specific (+ voltage, active, predicted, trace)

Region types:
  Sensory (S1):  L4 --> L2/3           (granular, n_l5=0)
  Motor (M1):    L2/3 --> L5           (agranular, n_l4=0)
  BG:            striatum --> gpi      (subcortical, NeuronGroup)
  Full (S2+):    L4 --> L2/3 --> L5

Connection roles:
  FEEDFORWARD   additive drive through ff_weights
  APICAL        gain modulation via dendritic segments
  MODULATORY    additive voltage bias before k-WTA
```

## Installation

```bash
pip install -e .
# or with uv:
uv sync
```

Requires Python 3.12+.

## Examples

See `examples/` for complete applications built on Arbora:

- **`examples/chat/`** -- Character-level text learning with sensory-motor hierarchy (S1->S2->S3->PFC->M2->M1)
- **`examples/minigrid/`** -- Grid navigation with MiniGrid (S1->BG->M1)

Run an example:

```bash
uv run examples/minigrid/train.py --episodes 100
uv run examples/minigrid/benchmark.py --episodes 1000
```

## Core package

```
src/arbora/
  neuron_group.py       NeuronGroup base
  basal_ganglia.py      BasalGangliaRegion (Go/NoGo + tonic DA)
  agent.py              BaseAgent, TrainResult
  config.py             PlasticityRule
  cortex/
    region.py           CorticalRegion (agranular support)
    sensory.py          SensoryRegion
    motor.py            MotorRegion
    pfc.py              PFCRegion
    circuit.py          Circuit builder
    circuit_types.py    Region protocol, Connection, ConnectionRole
    lamina.py           Lamina(NeuronGroup), LaminaID
    modulators.py       SurpriseTracker, ThalamicGate, RewardModulator
  encoders/             CharbitEncoder, OneHotCharEncoder, PositionalCharEncoder
  decoders/             DendriticDecoder, InvertedIndexDecoder, SynapticDecoder
  probes/               Probe protocol, LaminaProbe (functional KPIs)
  snapshots/            InputSnapshot, AssociationSnapshot
```

## Development

```bash
uv run pytest tests/              # 445 tests
uv run ruff check src/ tests/     # lint
uv run ty check src/arbora/        # typecheck
```

## Version

0.1.0 -- API is evolving. Expect breaking changes in 0.x releases.
