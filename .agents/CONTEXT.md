# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Biologically-plausible cortical learning. Minicolumn architecture, Hebbian + three-factor RL, no backprop. Full sensory-motor hierarchy with PFC goal maintenance.

## Integrations
- **Linear Project:** STEP (Sparse Temporal Eligibility Propagation)

## Architecture
```
Topo order: S1 → S2 → S3 → PFC → M2 → M1

Layers per region: L4 (input) → L2/3 (associative) → L5 (output)
  - L5 projects subcortically (BG, cerebellum, thalamus)
  - n_l5 defaults to n_l23, configurable per region

Feedforward: S1→S2 (buf=4), S2→S3 (buf=8), S2+S3→PFC, S2+PFC→M2, M2→M1
Apical: S3→S2, S2→S1, M1→M2, M2→PFC, S1→M1, M1→S1
Connections: ConnectionRole enum (FEEDFORWARD, APICAL), modulators as properties
```

## Inter-region pathways: current vs biological

Connections implicitly hardcode source_lamina=L2/3, target_lamina=L4. STEP-44 will make these explicit.

| Pathway | Current impl | Biological target | Status |
|---------|-------------|-------------------|--------|
| FF: src.L2/3 → tgt.L4 | `firing_rate_l23` → `ff_weights` | Correct (corticocortical) | Done |
| FF: src.L5 → tgt.L4 | — | Parallel ascending pathway | Missing |
| Apical: src.L2/3 → tgt.L4 | Linear gain on L4 voltage | Wrong target — should be tgt.L5 apical | Stopgap (default) |
| Apical: src.L2/3 → tgt.L5 | L5 apical segments | Correct (BAC firing) | STEP-32 |
| Feedback: src.L5 → tgt.L4 via thal | fb_segments sourced from L2/3 | Wrong source — should be L5 | Deferred |
| L5→L5 lateral | — | Output-layer sequence prediction | STEP-43 |

## Apical feedback: two modes
- **Linear gain** (default): src.L2/3 → tgt.L4 voltage. Fast, simple, wrong lamina target.
- **L5 apical segments** (`use_l5_apical_segments=True`): src.L2/3 → tgt.L5 segments. Context-specific BAC firing. +18% PFC ctx_disc, -4% surprise at 50k.

## Intra-region segments
- **L4 fb_segments**: L2/3→L4 (should be L5→thal→L4 eventually)
- **L4 lat_segments**: L4→L4 temporal patterns
- **L2/3 lateral segments**: L2/3→L2/3 pattern prediction
- **L5 apical segments**: cross-region top-down (STEP-32)
- **L5 lateral segments**: missing (STEP-43)

## Learning
- STDP pre-traces default on (`pre_trace_decay=0.8`)
- Three-factor (PFC, M1): eligibility traces consolidated by reward (`apply_reward` on base class)
- Segment learning: grow/reinforce/punish via shared `_check_segments` helper

## Validated Results
- STDP traces: 7.3% echo (best config)
- Structural sparsity: 38% echo improvement
- 300k sensory: decoder BPC 3.63
- L5 apical segments (50k): PFC +18.4%, burst 58.8%→54.8%, BPC 12.60→11.92

## Session: 2026-03-24

### Completed (merged)
- STEP-19 Echo reward fix (#3), STEP-24 _find_winners (#4), STEP-25 WordReward (#5), STEP-26 _babble_direct (#6)
- STEP-33 Connection kind→role + ConnectionRole enum (#8)
- STEP-31 L5 output layer in all regions (#9)
- STEP-29 Unify apply_reward base implementation (#10)
- Lint/typecheck fix (#7)

### In Review
- **STEP-32** L5 apical dendritic segments (PR #11) — training results positive, PR feedback addressed

### In Progress
- Full pipeline run (300k sensory + 100k babbling) with L5 apical segments

### Key Decisions
- L5→L4 feedback (replacing L2/3→L4 fb_segments) is cross-region, deferred
- Connections need explicit source_lamina/target_lamina (STEP-44)
- `Lamina` class to DRY per-layer code, paired with connection lamina routing (STEP-44)
- Apical reward-gated learning deferred to STEP-42

## Next Steps
- [ ] **STEP-32** Merge L5 apical segments PR (in review)
- [ ] **STEP-44** Lamina abstraction + connection lamina routing (XL, High)
- [ ] **STEP-43** L5→L5 lateral segments (S, backlog)
- [ ] **STEP-28** Split topology.py builder/runner (L, after STEP-32 merges)
- [ ] **STEP-42** Reward-gated apical learning in frontal regions (M, backlog)
- [ ] **STEP-20** Cerebellar forward model (XL, blocked by L5 work)
- [ ] **STEP-40** Better echo partial credit (M, backlog)
