# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Biologically-plausible cortical learning. Minicolumn architecture, Hebbian + three-factor RL, no backprop. Full sensory-motor hierarchy with PFC goal maintenance.

## Integrations
- **Linear Project:** STEP (Sparse Temporal Eligibility Propagation)

## Architecture
```
Topo order: S1 → S2 → S3 → PFC → M2 → M1

Layers per region: L4 (input) → L2/3 (associative) → L5 (output)
  - Lamina class encapsulates per-layer state (voltage, active, firing_rate, etc.)
  - CorticalRegion composes Laminae via register_lamina() / get_lamina()
  - Back-references: lamina.region points to parent CorticalRegion

Connections: ConnectionRole enum (FEEDFORWARD, APICAL)
  - source_lamina / target_lamina fields on Connection (default L23→L4)
  - Modulators (surprise, reward, thalamic_gate) as optional properties
```

## Inter-region pathways: current vs biological

| Pathway | Status | Notes |
|---------|--------|-------|
| FF: src.L2/3 → tgt.L4 | Done | Default, correct |
| FF: src.L5 → tgt.L4 | Missing | Parallel ascending pathway |
| Apical: src.L2/3 → tgt.L5 (segments) | Done (STEP-32) | BAC firing, +18% PFC |
| Apical: src.L2/3 → tgt.L4 (linear gain) | Stopgap | Wrong target lamina |
| Feedback: src.L5 → tgt.L4 via thal | Deferred | Cross-region, needs thalamic relay |
| L5→L5 lateral | Missing | STEP-43 |

## Key abstractions
- **Lamina**: per-layer state container (lamina.py). LaminaID enum: L4, L23, L5.
- **Connection**: role + source/target lamina + modulators (topology.py)
- **Segment prediction**: shared `_check_segments()` helper for L4/L2/3/L5
- **apply_reward()**: base class with motor override

## Known bugs
- Checkpoint missing encoder alphabet: old checkpoints can't be loaded by REPL if alphabet differs. Fixed in save_checkpoint() but existing checkpoints need re-generation.
- Checkpoint missing S2/PFC regions: full_pipeline_l5apical.py checkpoint appears corrupt (only 4 of 6 regions). Needs investigation or re-run.

## Validated Results
- L5 apical segments (50k): PFC +18.4%, burst 58.8%→54.8%, BPC 12.60→11.92
- Full pipeline (300k sensory + 100k babbling): 29 vocab chars, no 'h' attractor, diverse output
- STDP traces: 7.3% echo, 300k sensory decoder BPC 3.63

## Session: 2026-03-24

### Completed
- STEP-19, 24, 25, 26, 33, 31, 29, 32 (PRs #3-11)
- STEP-45 (Lamina Phase 1, PR #12), STEP-46 (Lamina Phase 2, PR #13)
- Lint/typecheck CI (#7), REPL checkpoint fix

### Key Decisions
- `Lamina` as state container, not self-running unit. register_lamina() pattern.
- Connection gets source_lamina/target_lamina (defaults preserve existing behavior)
- L5→L4 feedback is cross-region, deferred until thalamic relay exists
- L5 should have all features (voltage, excitability, trace, firing_rate)
- REPL delegates to cortex_staged.build_topology() for dimension consistency

## Next Steps
- [ ] **STEP-44 Phase 3** Connection routing via lamina fields (STEP-44)
- [ ] **STEP-43** L5→L5 lateral segments (S)
- [ ] **STEP-28** Split topology.py builder/runner (L)
- [ ] **STEP-42** Reward-gated apical learning in frontal regions (M)
- [ ] **STEP-20** Cerebellar forward model (XL)
- [ ] **STEP-40** Better echo partial credit (M)
- [ ] Re-run full pipeline to get clean checkpoint with encoder alphabet
