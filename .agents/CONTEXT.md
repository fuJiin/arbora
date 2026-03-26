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
              └── Circuit.process(encoding) -> ndarray — pure neural

Layers: L4 (input) → L2/3 (associative) → L5 (output)
  Lamina class, direct access (region.l4.voltage), all fields non-optional

Connections: ConnectionRole (FEEDFORWARD, APICAL), source/target lamina
Learning: PlasticityRule (HEBBIAN, THREE_FACTOR), orthogonal to STDP traces
```

## File organization
- `environment.py` — Observation protocol, ChatObs, Environment protocol, ChatEnv
- `agent.py` — Agent protocol, ChatAgent (encoder + circuit + decoder)
- `train.py` — train() loop bridging ChatEnv + ChatAgent with RunHooks
- `cortex/circuit.py` — Circuit class (process + builder + checkpoint)
- `cortex/circuit_types.py` — Connection, ConnectionRole, CortexResult, RunMetrics
- `cortex/circuit_hooks.py` — StepHooks protocol, RunHooks
- `cortex/canonical.py` — build_canonical_circuit() factory
- `cortex/stages.py` — configure functions (configure_sensory, configure_babbling, etc.)
- `cortex/lamina.py` — Lamina, LaminaID
- `cortex/region.py` — CorticalRegion with Lamina composition

## Session: 2026-03-26

### Completed (30 PRs total)
- Previous: 25 PRs (Cycle 1, architecture, lamina, circuit split, canonical)
- **STEP-69** (PR #26): Circuit.process(encoding) -> ndarray. Pure neural computation. step() kept as compat wrapper.
- **STEP-70** (PRs #27-30): Environment + Agent abstractions
  - PR #27: ChatEnv + ChatAgent + 18 tests
  - PR #28: train() function, migrate cortex_staged.py
  - PR #29: Migrate runner.py + cortex_repl.py
  - PR #30 (open): Deprecation warnings + motor_active param decouples EOM from process()
- **STEP-71** cancelled: encoder/decoder are Agent attributes, not Circuit

### Key decisions
- **Naming**: `Circuit.process()` (neural), `Agent.act()` (obs→action), `Environment.step()` (action→obs,reward)
- **Observation**: Generic `Observation` protocol, concrete `ChatObs` with token_id, token_str, is_boundary, is_eom
- **Streaming**: ChatEnv takes iterable, consumes lazily. Interleaved is default mode (babble_ratio=0 = pure listen)
- **motor_active param**: process() accepts explicit motor_active flag, decoupling from _in_eom/force_gate_open
- **EOM/boundary**: Still on Circuit for deprecated methods; Agent sets motor_active explicitly for new path
- **Turn-taking reward + babble chunking**: TODO to move to BasalGanglia (learned gating)

## Remaining tickets
- [ ] STEP-30 Region Protocol typing (M)
- [ ] STEP-48 Checkpoint validation (S)
- [ ] STEP-64 Redesign connect() for Lamina objects (M)
- [ ] STEP-58 RunHooks verbosity cleanup (S)
- [ ] STEP-54 L5 segment continuous traces (S)
- [ ] STEP-62 Uniform learning mechanics (L)
- [ ] STEP-61 Adaptive gating (XL)
- [ ] STEP-20 Cerebellar forward model (XL)
- [ ] Migrate remaining 15 experiment scripts to ChatEnv + ChatAgent
- [ ] Extract TrainRunner / ReplRunner from train() + cortex_repl.py
- [ ] Remove _in_eom/force_gate_open/mark_eom from Circuit entirely
