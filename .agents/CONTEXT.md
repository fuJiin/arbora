# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Biologically-plausible cortical learning. Minicolumn architecture, Hebbian + three-factor RL, no backprop. Full sensory-motor hierarchy with PFC goal maintenance.

## Integrations
- **Linear Project:** STEP (Sparse Temporal Eligibility Propagation)

## Architecture
```
[Chat]     ChatEnv → ChatAgent → Circuit.process(encoding)
[MiniGrid] MiniGridEnv → MiniGridAgent → Circuit.process(encoding)

BaseAgent → Circuit.process() + apply_reward() → pure neural

Topo: S1 → BG → M1 (MiniGrid) | S1 → S2 → S3 → PFC → M2 → M1 (Chat)
Region protocol: input_port / output_port (NeuronGroup) for circuit wiring
ConnectionRole: FEEDFORWARD, APICAL, MODULATORY

NeuronGroup: base (firing_rate + modulation). String group_id.
Lamina(NeuronGroup): cortex-specific (+ voltage, active, predicted, excitability, trace)

Region types:
  Sensory (S1):  L4 → L2/3       (n_l5=0)
  Motor (M1):    L2/3 → L5       (n_l4=0, agranular)
  BG:            striatum → gpi  (subcortical, NeuronGroup not Lamina)
  Full (S2+):    L4 → L2/3 → L5

Functional KPIs: InputSnapshot (recall/prec/sparse), AssociationSnapshot (eff_dim).
```

## Package structure
```
src/step/
├── neuron_group.py      # NeuronGroup base (universal connectable surface)
├── basal_ganglia.py     # BasalGangliaRegion (Go/NoGo + tonic DA)
├── cortex/
│   ├── lamina.py        # Lamina(NeuronGroup), LaminaID enum
│   ├── region.py        # CorticalRegion (agranular: n_l4=0, n_l5=0)
│   ├── motor.py         # MotorRegion (L5 output, exploration)
│   ├── circuit.py       # Circuit builder + apply_reward() + MODULATORY
│   └── circuit_types.py # Region protocol, Connection, ConnectionRole
├── environment/
│   ├── chat.py          # ChatEnv, ChatObs
│   └── minigrid.py      # MiniGridEnv, MiniGridObs
├── agent/
│   ├── base.py          # BaseAgent (shared state, apply_reward)
│   ├── chat.py          # ChatAgent
│   └── minigrid.py      # MiniGridAgent
├── harness/
│   ├── chat/train.py    # ChatTrainHarness, TrainResult
│   └── minigrid/train.py # MiniGridHarness
├── encoders/            # Encoder[T] protocol, CharbitEncoder, MiniGridEncoder
├── probes/core.py       # Probe protocol, LaminaProbe (functional KPIs)
├── snapshots/core.py    # InputSnapshot, AssociationSnapshot
└── reporting/chat.py    # ChatReporter

experiments/scripts/
├── chat/                # train.py, repl.py, sweep_s1.py
└── minigrid/            # train_minigrid.py, benchmark.py
```

## Session: 2026-03-28 → 2026-03-30

### Merged PRs
1. PR #51 — MiniGrid Phase 0: pipeline end-to-end
2. PR #52 — Agranular regions (STEP-77) + reward wiring (STEP-88)
3. PR #54 — BG redesign: proper region with Go/NoGo + tonic DA (STEP-90)

### Key decisions
- **Region protocol** — shared interface (input_port/output_port, process, apply_reward, reset)
- **NeuronGroup** — universal base (firing_rate + modulation). String group_id. Brian2 naming.
- **Lamina(NeuronGroup)** — cortex-specific (voltage, active, predicted, excitability, trace)
- **ConnectionRole.MODULATORY** — additive bias on target voltage before k-WTA
- **BG as proper DAG node** — BasalGangliaRegion with per-action Go/NoGo, tonic DA exploration
- **BG uses NeuronGroup** — striatum/gpi ports, not Lamina. Class constants for IDs.
- **add_modulation() / clear_modulation()** — public API on NeuronGroup for modulatory signals
- **input_port / output_port** — renamed from input_lamina/output_lamina (universal, not cortex-specific)
- **Corticostriatal from L5** — biologically, cortex→BG comes from L5. Using output_port for v1 (S1 has no L5).

### Key findings (benchmarks)
- Without exploration: M1 collapses to one action, 0.2% success
- Epsilon-greedy (hack): 49.9% success, peak 62%
- BG tonic DA exploration: peak 84% at ep 300, then collapses (tonic DA decays too fast)
- Tonic DA decay rate needs tuning: floor or slower decay

## Remaining tickets

**Done:**
- [x] STEP-63, STEP-77, STEP-78–84, STEP-86–88 (probe framework + agranular + reward)
- [x] STEP-90 BG redesign (PR #54)

**Next:**
- [ ] Tune BG tonic DA (floor or slower decay to prevent exploration collapse)
- [ ] STEP-89 MiniGrid Phase 2 — PFC working memory
- [ ] Motor KPI probes (episode success rate, purposeful ratio, consolidation rate)
- [ ] STEP-30 Region Protocol typing — partially done (Region protocol exists)

**Features:**
- [ ] STEP-61 Adaptive gating — learned interleaving (XL)
- [ ] STEP-20 Cerebellar forward model (XL)
