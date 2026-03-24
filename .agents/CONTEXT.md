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
- STEP-47 (Lamina Phase 3 — connection routing, PR #14)
- Lint/typecheck CI (#7), REPL checkpoint fix

### In Review
- STEP-43 L5 lateral segments (PR #15) — disabled by default after regression

### Key Decisions
- `Lamina` as state container, not self-running unit. register_lamina() pattern.
- Connection gets source_lamina/target_lamina (defaults preserve existing behavior)
- L5→L4 feedback is cross-region, deferred until thalamic relay exists
- L5 should have all features (voltage, excitability, trace, firing_rate)
- REPL delegates to cortex_staged.build_topology() for dimension consistency
- L5 lateral segments default off (`n_l5_segments=0`): immature segments regress ctx_disc at 50k. Enable post-refactor during tuning phase.
- Param tuning (neurons per column per lamina, segment counts) deferred until after refactors

### Technical debt (captured in STEP-49)
- Inconsistent lateral segment naming (lat_seg, l23_seg, l5_seg)
- Duplicated segment learning code (_learn_segments, _learn_l23_segments, _learn_l5_lateral_segments)
- SensoryRegion overrides _init_segments entirely instead of extending base
- Backward-compat aliases on CorticalRegion (voltage_l4 etc.)

## Next Steps
- [ ] **STEP-43** Merge L5 lateral PR (in review)
- [ ] **STEP-28** Split topology.py builder/runner (L)
- [ ] **STEP-49** Phase 4: alias cleanup, segment naming, DRY learning (M)
- [ ] **STEP-35** PlasticityRule enum (M)
- [ ] **STEP-30** Region Protocol typing (M)
- [ ] **STEP-42** Reward-gated apical learning (M)
- [ ] **STEP-20** Cerebellar forward model (XL)
- [ ] **STEP-50** Generate clean baseline checkpoint
- [ ] Param tuning sweep (n_l5, n_l5_segments, neurons per column per lamina)
