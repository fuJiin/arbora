# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Biologically-plausible cortical learning. Minicolumn architecture, Hebbian + three-factor RL, no backprop. Full sensory-motor hierarchy with PFC goal maintenance.

## Integrations
- **Linear Project:** STEP (Sparse Temporal Eligibility Propagation)

## Architecture
```
Environment.step(action) -> (obs, reward)       — ChatEnv
  └── Agent.act(obs, reward) -> action           — ChatAgent
        └── Circuit.process(encoding) -> ndarray — pure neural

Topo: S1 → S2 → S3 → PFC → M2 → M1
Layers: L4 (input) → L2/3 (associative) → L5 (output/feedback)

Feedforward: L2/3 → L4 (canonical, Felleman & Van Essen 1991)
Apical: L5 → {L2/3, L5} (top-down context via L1)
Intra-region: L4 → L2/3 → L5 (per-column learned weights)

Learning: segments for prediction, per-connection traces, Hebbian/3-factor
  Surprise modulates FF only, not segments
```

## Session: 2026-03-26/27 (40 PRs total)

### Completed this session
- STEP-69/70/72: Environment/Agent/Circuit architecture + migration
- STEP-64: connect() takes Lamina objects only
- STEP-54: L5 continuous traces
- STEP-62: Uniform learning (traces, remove linear gain, apical→{L2/3,L5})
- STEP-73: L5 as corticocortical output (then reverted FF to L2/3)
- STEP-74: Per-column L4→L2/3 and L2/3→L5 ff weights
- STEP-75: Remove fb_seg + revert FF to L2/3 + rename lat→l4_lat
- Biology audit: confirmed L2/3=FF source, L5=feedback/subcortical
- Baseline run (STEP-73 config): BPC 10.98 at 300k tokens

### Baselines
| Config | BPC (300k) | Notes |
|--------|-----------|-------|
| STEP-73 (L5 FF, proxy weights) | 10.98 | L5 as FF source, firing rate proxy |
| Current (L2/3 FF, per-col weights, no fb_seg) | TBD | Running next |

## Remaining tickets
- [ ] STEP-50 Baseline with current architecture (in progress)
- [ ] STEP-61 Adaptive gating — learned interleaving (XL)
- [ ] STEP-30 Region Protocol typing (M)
- [ ] STEP-48 Checkpoint validation (S)
- [ ] STEP-58 RunHooks verbosity cleanup (S)
- [ ] STEP-20 Cerebellar forward model (XL)
- [ ] Remove _in_eom/force_gate_open from Circuit
- [ ] Agranular motor/PFC regions (no true L4)
- [ ] L6 layer (thalamic gain control)

See .agents/BIOLOGY_AUDIT.md for full connection accuracy audit.
