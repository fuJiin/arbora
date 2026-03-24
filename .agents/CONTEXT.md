# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Biologically-plausible cortical learning. Minicolumn architecture, Hebbian + three-factor RL, no backprop. Full sensory-motor hierarchy with PFC goal maintenance.

## Integrations
- **Linear Project:** STEP (Sparse Temporal Eligibility Propagation)

## Architecture (DAG-validated, finalize() enforced)
```
Topo order: S1 → S2 → S3 → PFC → M2 → M1

Feedforward (source-aware sparsity on PFC/M2):
  S1→S2 (buf=4), S2→S3 (buf=8)
  S2+S3→PFC (40% sparse), S2+PFC→M2 (40% sparse)
  M2→M1

Apical (multi-source, per-source gain weights):
  S3→S2, S2→S1, M1→M2, M2→PFC, S1→M1, M1→S1

Surprise: modulator property on feedforward/apical connections (ConnectionRole enum)
```

## Layers: L4 → L2/3 → L5 (all regions)
- **L4** (input): feedforward drive, dendritic segment predictions
- **L2/3** (associative): lateral context, firing rate EMA
- **L5** (output): intra-columnar drive from L2/3, projects subcortically (BG, cerebellum, thalamus)
- L5 added to CorticalRegion base class (PR #9). Motor output_weights now map L5→tokens.
- `n_l5` config parameter (defaults to n_l23)

## Connection System (refactored 2026-03-24)
- ConnectionRole enum: FEEDFORWARD, APICAL (was string "kind")
- SurpriseTracker/RewardModulator are optional properties on any connection
- No more separate "surprise"/"reward" connection kinds

## Learning: STDP Presynaptic Traces (DEFAULT ON)
`pre_trace_decay=0.8` in CortexConfig, all regions from construction.
- Three-factor (PFC, M1): pre_trace feeds eligibility → reward

## CI
- GitHub Actions: lint (ruff check + format), typecheck (ty), test (pytest)
- ty overrides: viz/ ignores unresolved-import (plotly optional), experiments/ relaxed
- Pre-commit hooks: ruff, ty, pytest

## Validated Results
- STDP traces from construction: 7.3% echo (best config)
- Structural sparsity: 38% echo improvement
- PFC three-factor: 3.1% → 8.2% echo
- 300k trace sensory: decoder BPC 3.63
- 50k staged sensory (with L5, 2026-03-24): S2 ctx_disc=0.913, S3=0.935, M1=0.842

## Session: 2026-03-24

### Completed (merged)
- **STEP-19** Fix echo reward 'h' attractor (PR #3)
- **STEP-24** Extract _find_winners() helper (PR #4)
- **STEP-25** Remove unused WordReward (PR #5)
- **STEP-26** Add tests for _babble_direct (PR #6)
- **STEP-33** Connection kind→role refactor + ConnectionRole enum (PR #8)
- Lint/typecheck fix — 0 warnings (PR #7)

### In Review
- **STEP-31** L5 output layer in all regions (PR #9) — rebased on PZO-33
- **STEP-29** Unify apply_reward() base implementation — in progress

### Key decisions
- Echo reward: removed broken partial credit, created STEP-40 for redesign
- _babble_direct is NOT dead code (used at noise=0.5 by stages) — kept + tested
- ConnectionRole enum replaces string kinds; "role" replaces "kind"
- L5 activation: one winner per column from L2/3 firing rate, all-fire on burst
- Reward sparsity acknowledged as unsolved (STEP-40 backlogged)

## Next Steps
See Linear project (STEP) for full backlog. Key priorities:
- [ ] **STEP-31** Merge L5 layer PR (pending review)
- [ ] **STEP-29** Merge apply_reward unification (in progress)
- [ ] **STEP-40** Design better echo partial credit (backlog)
- [ ] **STEP-20** Cerebellar forward model (blocked by STEP-31)
- [ ] **STEP-28** Split topology.py into builder + runner (cleanup, L)
- [ ] **STEP-35** PlasticityRule enum (cleanup, M)
- [ ] Run L5 training comparison (in progress, results pending)
- [ ] Sweep L5 n_l5 configs per region
