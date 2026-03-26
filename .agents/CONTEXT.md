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

Layers: L4 (input) → L2/3 (associative) → L5 (output)
  Lamina class, direct access (region.l4.voltage), all fields non-optional

Connections: ConnectionRole (FEEDFORWARD, APICAL), source/target lamina
Learning: PlasticityRule (HEBBIAN, THREE_FACTOR), orthogonal to STDP traces
```

## File organization
- `environment.py` — Observation protocol, ChatObs, Environment protocol, ChatEnv
- `agent.py` — Agent protocol, ChatAgent (encoder + circuit + decoder)
- `train.py` — train() loop bridging ChatEnv + ChatAgent with RunHooks
- `cortex/circuit.py` — Circuit class (process + builder + checkpoint, 908 LOC)
- `cortex/circuit_types.py` — Connection, ConnectionRole, CortexResult, RunMetrics
- `cortex/circuit_hooks.py` — StepHooks protocol, RunHooks
- `cortex/canonical.py` — build_canonical_circuit() factory
- `cortex/stages.py` — configure functions (configure_sensory, configure_babbling, etc.)
- `cortex/lamina.py` — Lamina, LaminaID
- `cortex/region.py` — CorticalRegion with Lamina composition

## Session: 2026-03-26 (32 PRs total)

### Completed
- Previous: 25 PRs (Cycle 1, architecture, lamina, circuit split, canonical)
- **STEP-69** (PR #26): Circuit.process(encoding) -> ndarray
- **STEP-70** (PRs #27-30): ChatEnv + ChatAgent + train(), migrate all primary callers, deprecation + motor_active decoupling
- **STEP-72** (PR #31): Migrate tests, remove all deprecated methods (870 lines deleted, circuit.py 1780→908)
- **STEP-71** cancelled (encoder/decoder are Agent attributes)
- **STEP-44** marked Done (all subtasks complete)
- **STEP-38** cancelled (duplicate of STEP-61)

### Key decisions
- `Circuit.process()` (neural), `Agent.act()` (obs→action), `Environment.step()` (action→obs,reward)
- ChatObs: token_id, token_str, is_boundary, is_eom. Streaming via iterable.
- motor_active param on process() decouples from _in_eom/force_gate_open
- Turn-taking reward + babble chunking: TODO to move to BasalGanglia (learned gating)
- 14 experiment sweep scripts still reference cortex.run() — need migration or archiving

## Remaining tickets
- [ ] STEP-64 Redesign connect() for Lamina objects (M)
- [ ] STEP-54 L5 segment continuous traces (S)
- [ ] STEP-62 Uniform learning mechanics (L)
- [ ] STEP-61 Adaptive gating — learned interleaving (XL)
- [ ] STEP-30 Region Protocol typing (M)
- [ ] STEP-48 Checkpoint validation (S)
- [ ] STEP-50 Generate clean baseline (XS)
- [ ] STEP-58 RunHooks verbosity cleanup (S)
- [ ] STEP-20 Cerebellar forward model (XL)
- [ ] Migrate/archive 14 experiment sweep scripts
- [ ] Extract TrainRunner / ReplRunner
- [ ] Remove _in_eom/force_gate_open/mark_eom from Circuit
