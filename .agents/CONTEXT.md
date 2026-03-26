# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Biologically-plausible cortical learning. Minicolumn architecture, Hebbian + three-factor RL, no backprop. Full sensory-motor hierarchy with PFC goal maintenance.

## Integrations
- **Linear Project:** STEP (Sparse Temporal Eligibility Propagation)

## Architecture
```
Environment.step(action) -> (obs, reward)       — ChatEnv
  └── Agent.act(obs, reward) -> action           — ChatAgent
        └── Circuit.process(encoding) -> ndarray — pure neural (908 LOC)

Topo: S1 → S2 → S3 → PFC → M2 → M1
Layers: L4 (input) → L2/3 (associative) → L5 (output)

Feedforward: L5 → L4 (corticocortical, all inter-region)
Apical: L5 → {L2/3, L5} (top-down context, dual target)
Intra-region: L4 → L2/3 → L5 (within column)

Learning: segments everywhere (lateral, feedback, apical)
  Per-connection traces (temporal credit, decay=0.8)
  Hebbian (sensory) vs three-factor (motor/PFC)
  Surprise modulates FF only, not segments
```

## Session: 2026-03-26 (38 PRs total)

### Completed
- STEP-69: Circuit.process(encoding) -> ndarray
- STEP-70: ChatEnv + ChatAgent + train(), full migration
- STEP-72: Remove deprecated methods (circuit.py 1780→908)
- STEP-64: connect() takes Lamina objects only
- STEP-54: L5 continuous traces for lateral segments
- STEP-62: Uniform learning (per-connection traces, remove linear gain, apical→{L2/3,L5}, limit surprise)
- STEP-73: L5 as universal corticocortical output
- Pruned 14 broken experiment scripts

### Key decisions
- L5 is the universal corticocortical output (FF source + apical source)
- Apical segments only mode (linear gain removed)
- Per-connection traces on all pathways
- L2/3→L5 intra-region uses firing rate proxy (STEP-74: proper ff weights)
- Segments everywhere, optimize later (sparse weights as perf lever)

### In progress
- Baseline training run (300k tokens sensory stage)

## Remaining tickets
- [ ] STEP-74 L2/3→L5 intra-region ff weights (M, high priority)
- [ ] STEP-50 Generate clean baseline (XS, in progress)
- [ ] STEP-61 Adaptive gating — learned interleaving (XL)
- [ ] STEP-30 Region Protocol typing (M)
- [ ] STEP-48 Checkpoint validation (S)
- [ ] STEP-58 RunHooks verbosity cleanup (S)
- [ ] STEP-20 Cerebellar forward model (XL)
- [ ] Remove _in_eom/force_gate_open/mark_eom from Circuit
