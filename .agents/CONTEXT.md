# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Biologically-plausible cortical learning. Minicolumn architecture, Hebbian + three-factor RL, no backprop. Full sensory-motor hierarchy with PFC goal maintenance.

## Integrations
- **Linear Project:** STEP (Sparse Temporal Eligibility Propagation)

## Architecture
```
Experiment (params, logging, metrics)
  └── Environment (token source, feedback loop, timing)
        └── Circuit (step: encoding_in → encoding_out)
              └── Regions + Connections (internal wiring)

Layers: L4 (input) → L2/3 (associative) → L5 (output)
  Lamina class, direct access (region.l4.voltage), all fields non-optional

Connections: ConnectionRole (FEEDFORWARD, APICAL), source/target lamina
Learning: PlasticityRule (HEBBIAN, THREE_FACTOR), orthogonal to STDP traces
Execution: step() is primitive, run(babble_ratio=...) is unified training loop
```

## File organization
- `canonical.py` — build_canonical_circuit() factory (single source of truth)
- `circuit.py` — Circuit class (builder + step + run + checkpoint)
- `circuit_types.py` — Connection, ConnectionRole, CortexResult, RunMetrics
- `circuit_hooks.py` — StepHooks protocol, RunHooks
- `stages.py` — configure functions (configure_sensory, configure_babbling, etc.)
- `lamina.py` — Lamina, LaminaID
- `region.py` — CorticalRegion with Lamina composition

## Completed (25 PRs, 2026-03-24/25/26)
- Cycle 1 cleanup: STEP-19, 24, 25, 26, 29, 49
- Architecture: STEP-33, 31, 32, 43, 35
- Lamina: STEP-44 (phases 1-3), STEP-49 (alias removal)
- Circuit split: STEP-28 (types, hooks, rename)
- Canonical: STEP-59 factory, STEP-60 curriculum (stages, unified run, echo/dialogue)

## Design direction

### Target architecture (next major work)
- STEP-69: step(encoding) → encoding | None. Modality-agnostic circuit.
- STEP-70: Environment abstraction (token source, feedback, encoder/decoder)
- STEP-71: Encoder/decoder as circuit attributes (learnable)

### Other vision items
- STEP-61: Adaptive gating — circuit self-regulates training phases (XL)
- STEP-62: Uniform learning — all connections use segments/traces/penalties (L)
- STEP-63: Probe protocol — clean measurement at all levels (M)

## Remaining tickets
- [ ] STEP-69 Redesign step() interface (L)
- [ ] STEP-70 Extract Environment abstraction (L)
- [ ] STEP-71 Move encoder/decoder to Circuit (M)
- [ ] STEP-30 Region Protocol typing (M)
- [ ] STEP-48 Checkpoint validation (S)
- [ ] STEP-64 Redesign connect() for Lamina objects (M)
- [ ] STEP-58 RunHooks verbosity cleanup (S)
- [ ] STEP-54 L5 segment continuous traces (S)
- [ ] STEP-62 Uniform learning mechanics (L)
- [ ] STEP-61 Adaptive gating (XL)
- [ ] STEP-20 Cerebellar forward model (XL)
