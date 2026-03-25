# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Biologically-plausible cortical learning. Minicolumn architecture, Hebbian + three-factor RL, no backprop. Full sensory-motor hierarchy with PFC goal maintenance.

## Integrations
- **Linear Project:** STEP (Sparse Temporal Eligibility Propagation)

## Architecture
```
Topo order: S1 → S2 → S3 → PFC → M2 → M1

Layers: L4 (input) → L2/3 (associative) → L5 (output)
  Lamina class, non-optional fields, register_lamina pattern

Connections: ConnectionRole (FEEDFORWARD, APICAL), source/target lamina
Learning: PlasticityRule (HEBBIAN, THREE_FACTOR), orthogonal to STDP traces
Execution: step() is primitive, run() = step-in-a-loop with RunHooks
```

## File organization
- `circuit.py` — Circuit class (builder + step + checkpoint)
- `circuit_types.py` — Connection, ConnectionRole, CortexResult, RunMetrics
- `circuit_hooks.py` — StepHooks protocol, RunHooks
- `lamina.py` — Lamina, LaminaID
- `region.py` — CorticalRegion with Lamina composition
- `motor.py`, `pfc.py`, `sensory.py`, `premotor.py` — subclasses

## Design vision (2026-03-25)

### Canonical circuit (STEP-59)
One factory builds the standard 6-region topology. All scripts call it.
Sweeps override params via kwargs. Eliminates 5+ duplicated wiring copies.

### Canonical curriculum (STEP-60)
One configurable loop replaces run/run_babbling/run_echo/run_interleaved/run_dialogue.
Sensory-only = motor gate closed. Interleaved = gate opens adaptively.
Echo/babbling = different PFC goals, not different loops.
REPL = another modality for interacting with the same circuit.

### Adaptive gating (STEP-61)
Circuit self-regulates training phases. BG/motor decides when babbling is useful
based on readiness metrics (surprise rate, representation stability). No hardcoded
stages. Ramble threshold as penalty (PFC learns conciseness), not hard cutoff.
Motor confidence threshold: training (babble freely) vs production (think first).

### Uniform learning (STEP-62)
All connection types: segments (structural sparsity), continuous traces (temporal
credit), prediction-based credit assignment, penalty on incorrect predictions.
L2/3 lateral is the gold standard. Generalize to apical and L5.

### Probe protocol (STEP-63)
Clean measurement at column/lamina/region/circuit levels. Probe interface
registered on circuit. RunHooks becomes probe orchestrator.

## Remaining Cycle 1
- [ ] STEP-30 Region Protocol typing (M)
- [ ] STEP-49 Alias cleanup, segment naming, DRY learning (M)
- [ ] STEP-48 Checkpoint validation (S)

## Next priorities (post Cycle 1)
- [ ] STEP-59 Canonical circuit factory (S)
- [ ] STEP-60 Canonical curriculum (L)
- [ ] STEP-62 Uniform learning (L)
- [ ] STEP-61 Adaptive gating (XL)
- [ ] STEP-63 Probe protocol (M)
