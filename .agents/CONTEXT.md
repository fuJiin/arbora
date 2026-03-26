# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Biologically-plausible cortical learning. Minicolumn architecture, Hebbian + three-factor RL, no backprop. Full sensory-motor hierarchy with PFC goal maintenance.

## Integrations
- **Linear Project:** STEP (Sparse Temporal Eligibility Propagation)

## Architecture
```
TrainRunner / ReplRunner (hooks, metrics, param sweeps) — future
  └── Environment.step(action) -> (obs, reward)       — ChatEnv
        └── Agent.act(obs, reward) -> action           — ChatAgent
              └── Circuit.process(encoding) -> ndarray — pure neural (908 LOC)

Topo: S1 → S2 → S3 → PFC → M2 → M1
Layers: L4 (input) → L2/3 (associative) → L5 (output)
Connections: connect(source_lamina, target_lamina, role) — Lamina objects only
Learning: PlasticityRule (HEBBIAN, THREE_FACTOR), STDP pre-traces, segments
```

## File organization
- `environment.py` — Observation/Environment protocols, ChatEnv, ChatObs
- `agent.py` — Agent protocol, ChatAgent (encoder + circuit + decoder)
- `train.py` — train() bridging ChatEnv + ChatAgent with RunHooks
- `cortex/circuit.py` — Circuit (process + builder + checkpoint, 908 LOC)
- `cortex/canonical.py` — build_canonical_circuit() factory
- `cortex/region.py` — CorticalRegion with Lamina composition
- `.agents/LEARNING.md` — Learning mechanism audit (pre-STEP-62)

## Session: 2026-03-26 (35 PRs total)

### Completed this session
- **STEP-69** (PR #26): Circuit.process(encoding) -> ndarray
- **STEP-70** (PRs #27-30): ChatEnv + ChatAgent + train(), full migration
- **STEP-72** (PR #31): Remove deprecated methods (circuit.py 1780→908)
- **PR #32**: Prune 14 broken experiment scripts (4,467 lines)
- **STEP-64** (PR #33): connect() takes Lamina objects only (no strings)
- **STEP-54** (PR #34): L5 continuous traces for lateral segments
- **Learning audit** committed to .agents/LEARNING.md

### Key decisions this session
- Environment/Agent/Circuit layered architecture (clean separation)
- connect(lamina, lamina) — no string-based routing
- motor_active param on process() decouples EOM from Circuit
- Turn-taking reward + babble chunking → TODO: BasalGanglia (STEP-61)
- Encoder/decoder are Agent attributes, not Circuit (STEP-71 cancelled)

### In progress: STEP-62 planning
Learning audit complete. Architecture questions raised:
- FF should use L5→L4 (not L2/3→L4) for biological accuracy
- Apical should be L5(higher)→L2/3(lower), not L2/3→L4
- All connections should carry decaying trace of source signal
- Linear gain apical should be replaced with segments or three-factor
- L2/3 should get apical context from external sources (like L5 does)
See .agents/LEARNING.md for full audit and open questions.

## Remaining tickets
- [ ] STEP-62 Uniform learning mechanics (L) — next up, needs design decisions
- [ ] STEP-61 Adaptive gating — learned interleaving (XL)
- [ ] STEP-30 Region Protocol typing (M)
- [ ] STEP-48 Checkpoint validation (S)
- [ ] STEP-50 Generate clean baseline (XS)
- [ ] STEP-58 RunHooks verbosity cleanup (S)
- [ ] STEP-20 Cerebellar forward model (XL)
- [ ] Extract TrainRunner / ReplRunner
- [ ] Remove _in_eom/force_gate_open/mark_eom from Circuit
