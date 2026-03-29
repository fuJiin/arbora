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
Region protocol: input_port / output_port for circuit wiring
ConnectionRole: FEEDFORWARD, APICAL, MODULATORY

Laminae by region type:
  Sensory (S1):  L4 → L2/3       (n_l5=0)
  Motor (M1):    L2/3 → L5       (n_l4=0, agranular)
  BG:            input → Go/NoGo → output (subcortical, not cortex)
  Full (S2+):    L4 → L2/3 → L5

Functional KPIs: InputSnapshot (recall/prec/sparse), AssociationSnapshot (eff_dim).
```

## Package structure
```
src/step/
├── cortex/              # Pure neural: Circuit, Region, connections, modulators
│   ├── region.py        # CorticalRegion (agranular support: n_l4=0, n_l5=0)
│   ├── motor.py         # MotorRegion (L5 output, exploration, goal drive)
│   ├── basal_ganglia.py # BG go/no-go gating (NEEDS REDESIGN)
│   └── circuit.py       # Circuit builder + apply_reward()
├── environment/
│   ├── chat.py          # ChatEnv, ChatObs
│   └── minigrid.py      # MiniGridEnv, MiniGridObs
├── agent/
│   ├── base.py          # BaseAgent (shared state, apply_reward)
│   ├── chat.py          # ChatAgent
│   └── minigrid.py      # MiniGridAgent (epsilon-greedy TEMP)
├── harness/
│   ├── chat/train.py    # ChatTrainHarness, TrainResult
│   └── minigrid/train.py # MiniGridHarness
├── encoders/
│   ├── positional.py    # PositionalCharEncoder (Encoder[str])
│   ├── charbit.py       # CharbitEncoder (Encoder[str])
│   └── minigrid.py      # MiniGridEncoder (Encoder[MiniGridObs], 984-bit)
├── probes/core.py       # Probe protocol, LaminaProbe (functional KPIs)
├── snapshots/core.py    # InputSnapshot, AssociationSnapshot
└── reporting/chat.py    # ChatReporter

experiments/scripts/
├── chat/                # train.py, repl.py, sweep_s1.py
└── minigrid/            # train_minigrid.py, benchmark.py, diagnose.py, visualize.py
```

## Session: 2026-03-28 → 2026-03-29 (MiniGrid RL + agranular)

### Merged PRs
1. PR #51 — MiniGrid Phase 0: pipeline end-to-end
2. PR #52 — Agranular regions (STEP-77) + reward wiring (STEP-88) + consolidation
3. PR #53 (pending) — Epsilon-greedy exploration, benchmark scripts

### Key decisions
- **Encoder[T] generic protocol** — Encoder[str] for chat, Encoder[MiniGridObs] for grid
- **BaseAgent** — shared state, apply_reward() delegates to circuit
- **environment/ package** — chat.py + minigrid.py with backward-compat re-exports
- **Agranular regions** — n_l4=0 for motor, n_l5=0 for sensory. input_lamina/output_lamina route processing.
- **Functional KPIs** — InputSnapshot/AssociationSnapshot measured on the function's lamina, not hardcoded L4/L23
- **circuit.apply_reward()** — routes to BG + motor internally, harness never knows about BG
- **forced_columns on step()/process()** — enables exploration without duplicating pipeline
- **exploration_noise** (was babbling_noise) — modality-agnostic naming
- **Goal drive in base CorticalRegion** — any region can receive goal signals, not just motor

### Key findings
- S1 plateau: recall=0.73 by ep 20, holds flat without reward
- Without exploration: M1 collapses to one action (toggle), 0.2% success vs 37% random
- With epsilon-greedy: 49.9% success, peak 62% rolling, 13% faster episodes
- **BG gate stuck at 1.0** — `motor_active=True` overrides gate to 1.0 (chat-specific hack)
- Epsilon-greedy is not biologically grounded — need BG redesign

## Current work: BG Redesign

**Problem:** BG is a side-car (arg to add_region), not a real circuit participant. Gate is overridden to 1.0 when motor_active. Exploration is hacked into agent (epsilon-greedy) instead of circuit (BG-modulated action selection).

**Goal:** BG as a proper region that receives cortical input, learns from reward, and modulates M1 action selection. Exploration emerges from BG-cortex loop, not agent policy.

## Remaining tickets

**Done:**
- [x] STEP-63, STEP-78–84, STEP-86, STEP-87 (probe framework)
- [x] STEP-77 Agranular regions (PR #52)
- [x] STEP-88 Reward wiring (PR #52)
- [x] MiniGrid Phase 0 (PR #51)

**In progress:**
- [ ] BG redesign — proper region, exploration via circuit not agent
- [ ] STEP-89 MiniGrid Phase 2 — PFC working memory (blocked on BG)

**Architecture:**
- [ ] STEP-30 Region Protocol typing (M)

**Features:**
- [ ] STEP-61 Adaptive gating — learned interleaving (XL)
- [ ] STEP-20 Cerebellar forward model (XL)
