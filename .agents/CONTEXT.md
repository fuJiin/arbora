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
  Lamina class, direct access (region.l4.voltage)

Connections: ConnectionRole (FEEDFORWARD, APICAL), source/target lamina
Learning: PlasticityRule (HEBBIAN, THREE_FACTOR), orthogonal to STDP traces
Execution: step() is primitive, run(babble_ratio=...) is unified training loop
```

## File organization
- `canonical.py` — build_canonical_circuit() factory
- `circuit.py` — Circuit class (builder + step + run + checkpoint)
- `circuit_types.py` — Connection, ConnectionRole, CortexResult, RunMetrics
- `circuit_hooks.py` — StepHooks protocol, RunHooks
- `stages.py` — configure functions (configure_sensory, configure_babbling, etc.)
- `lamina.py` — Lamina, LaminaID
- `region.py` — CorticalRegion with Lamina composition

## Design direction

### Target architecture
- Circuit: step(encoding) → encoding | None. Modality-agnostic. (STEP-69)
- Environment: encoder/decoder, token source, feedback policy. (STEP-70)
- Experiment: circuit + environment + metrics config.
- Encoder/decoder as circuit attributes (learnable). (STEP-71)

### Canonical curriculum (STEP-60)
- STEP-65 ✅ Simplify stages
- STEP-66 ✅ Unified run with babble_ratio
- STEP-67 Collapse run_echo into PFC goal injection
- STEP-68 Remove run_dialogue

### Other vision items
- STEP-61 Adaptive gating (XL)
- STEP-62 Uniform learning mechanics (L)
- STEP-63 Probe protocol (M)

## Session: 2026-03-24/25/26 — 24 PRs merged

## Immediate next
- [ ] STEP-67 Collapse run_echo into PFC goal injection (M)
- [ ] STEP-68 Remove run_dialogue (S)
- [ ] STEP-69 Redesign step() interface (L)
- [ ] STEP-70 Extract Environment abstraction (L)
- [ ] STEP-30 Region Protocol typing (M)
- [ ] STEP-48 Checkpoint validation (S)
