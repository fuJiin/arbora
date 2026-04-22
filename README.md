# Arbora

Biologically-plausible cortical learning framework. Build neural circuits from cortical regions, wire them together, and train with Hebbian + three-factor reward-modulated learning. No backprop.

Named for dendritic arbors - the branching structures through which neurons receive and integrate signals - and for aurora, evoking both a new dawn and the light of brain activity.

> **Status:** alpha research code. APIs evolving, breaking changes likely, not all mechanisms fully learn yet. Use at your own risk; issues and discussion welcome. Apache-2.0 licensed (see [LICENSE](LICENSE)).

## Quick start

```python
from arbora import (
    Circuit, SensoryRegion, MotorRegion, BasalGangliaRegion,
    ConnectionRole, PlasticityRule,
)

# Build regions
t1 = SensoryRegion(input_dim=100, n_columns=32, n_l4=4, n_l23=4, k_columns=4)
m1 = MotorRegion(input_dim=t1.n_l23_total, n_columns=16, n_l4=0, n_l23=4,
                 k_columns=2, n_output_tokens=7)
bg = BasalGangliaRegion(input_dim=t1.n_l23_total, n_actions=7)

# Wire circuit
circuit = Circuit(encoder)
circuit.add_region("T1", t1, entry=True, input_region=True)
circuit.add_region("BG", bg)
circuit.add_region("M1", m1, output_region=True)
circuit.connect(t1.output_port, bg.input_port, ConnectionRole.FEEDFORWARD)
circuit.connect(t1.output_port, m1.input_port, ConnectionRole.FEEDFORWARD)
circuit.connect(bg.output_port, m1.input_port, ConnectionRole.MODULATORY)
circuit.finalize()

# Process input
output = circuit.process(encoding)
circuit.apply_reward(reward)
```

## What this is

A framework for building biologically-grounded neural circuits that learn from reward, not gradients. Every component maps to real neuroscience:

| Arbora concept | Biology | What it does |
|---------------|---------|-------------|
| `SensoryRegion` | Granular cortex (V1, T1) | L4 input reception, L2/3 association, dendritic prediction |
| `MotorRegion` | Agranular cortex (M1) | L2/3 input, L5 action output, three-factor RL |
| `BasalGangliaRegion` | Striatum + GPi | Per-action Go/NoGo gating, tonic DA exploration |
| `ThalamicNucleus` | Pulvinar / higher-order thalamus | Gated relay between cortical regions |
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
  Sensory (T1):  L4 --> L2/3           (granular, n_l5=0)
  Motor (M1):    L2/3 --> L5           (agranular, n_l4=0)
  BG:            striatum --> gpi      (subcortical, NeuronGroup)
  Thalamus:      driver --> relay      (gated, Hebbian learned)
  Full (T2+):    L4 --> L2/3 --> L5

Connection roles:
  FEEDFORWARD   additive drive through ff_weights
  APICAL        gain modulation via dendritic segments
  MODULATORY    additive voltage bias before k-WTA
```

## Getting started

Requires Python 3.12+. No PyPI package yet — clone and run from source:

```bash
git clone https://github.com/fuJiin/arbora.git
cd arbora
./scripts/bootstrap.sh
```

## Examples

See `examples/` for complete applications built on Arbora:

- **`examples/arc/`** -- ARC-AGI-3 spatial reasoning with transthalamic hierarchy (V1->pulvinar->V2->BG->M1)
- **`examples/chat/`** -- Character-level text learning with sensory-motor hierarchy (T1->T2->T3->PFC->M2->M1)
- **`examples/minigrid/`** -- Grid navigation with MiniGrid (T1->BG->M1)

Run an example (examples import each other as a package, so invoke them with `python -m`):

```bash
# MiniGrid requires the optional gymnasium/minigrid deps:
uv sync --extra minigrid
uv run python -m examples.minigrid.train --episodes 100
uv run python -m examples.minigrid.benchmark --episodes 1000

# ARC requires the optional arc-agi dependency:
uv sync --extra arc
uv run python -m examples.arc.train --keyboard-only --episodes 5
```

## Development

```bash
uv run pytest tests/              # ~480 tests (install --extra minigrid to exercise minigrid suite)
uv run ruff check src/ tests/     # lint
uv run ty check src/arbora/        # typecheck
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup details.

## Further reading

- [docs/LAMINA_KPIS.md](docs/LAMINA_KPIS.md) -- what each probe measures and why
- [docs/BIBLIOGRAPHY.md](docs/BIBLIOGRAPHY.md) -- papers the framework is grounded in
