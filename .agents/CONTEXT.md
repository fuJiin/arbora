# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Biologically-plausible cortical learning. Minicolumn architecture, Hebbian + three-factor RL, no backprop. Full sensory-motor hierarchy with PFC goal maintenance.

## Integrations
- **Linear Project:** STEP (Sparse Temporal Eligibility Propagation)

## Architecture
```
Topo order: S1 → S2 → S3 → PFC → M2 → M1

Layers: L4 (input) → L2/3 (associative) → L5 (output)
  Lamina class, non-optional fields, direct access (region.l4.voltage)

Connections: ConnectionRole (FEEDFORWARD, APICAL), source/target lamina
Learning: PlasticityRule (HEBBIAN, THREE_FACTOR), orthogonal to STDP traces
Execution: step() is primitive, run() = step-in-a-loop with RunHooks
```

## File organization
- `canonical.py` — build_canonical_circuit() factory (single source of truth)
- `circuit.py` — Circuit class (builder + step + checkpoint)
- `circuit_types.py` — Connection, ConnectionRole, CortexResult, RunMetrics
- `circuit_hooks.py` — StepHooks protocol, RunHooks
- `stages.py` — configure functions (configure_sensory, configure_babbling, etc.)
- `lamina.py` — Lamina, LaminaID
- `region.py` — CorticalRegion with Lamina composition
- `motor.py`, `pfc.py`, `sensory.py`, `premotor.py` — subclasses

## Session: 2026-03-24/25

### Completed (23 PRs)
- Cycle 1 cleanup: STEP-19, 24, 25, 26, 29, 49
- Architecture: STEP-33, 31, 32, 43, 35
- Lamina: STEP-45, 46, 47
- Circuit split: STEP-51, 56, 57 (+ STEP-53 rename)
- Canonical: STEP-59 factory, STEP-65 stages simplify
- Infra: lint/typecheck (#7), REPL fixes

### In Progress
- **STEP-66** Unify run_interleaved as canonical loop — next up

## Design vision

### Canonical circuit (STEP-59) ✅
Single factory for 6-region topology. All scripts call it.

### Canonical curriculum (STEP-60) — in progress
- STEP-65 ✅ Simplify stages (config functions)
- STEP-66 Unify run_interleaved as canonical loop
- STEP-67 Collapse run_echo into PFC goal injection
- STEP-68 Remove run_dialogue

### Adaptive gating (STEP-61)
Circuit self-regulates training phases based on readiness metrics.

### Uniform learning (STEP-62)
All connection types: segments, continuous traces, prediction penalties.

### Probe protocol (STEP-63)
Clean measurement at column/lamina/region/circuit levels.

## Remaining tickets
- [ ] STEP-66 Unify run_interleaved (M)
- [ ] STEP-67 Collapse run_echo (M)
- [ ] STEP-68 Remove run_dialogue (S)
- [ ] STEP-30 Region Protocol typing (M)
- [ ] STEP-48 Checkpoint validation (S)
- [ ] STEP-64 Redesign connect() for Lamina objects (M)
- [ ] STEP-58 RunHooks verbosity cleanup (S)
- [ ] STEP-54 L5 segment continuous traces (S)
- [ ] STEP-62 Uniform learning mechanics (L)
- [ ] STEP-61 Adaptive gating (XL)
- [ ] STEP-20 Cerebellar forward model (XL)
