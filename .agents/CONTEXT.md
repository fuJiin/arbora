# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Biologically-plausible cortical learning. Minicolumn architecture, Hebbian + three-factor RL, no backprop. Full sensory-motor hierarchy with PFC goal maintenance.

## Architecture (DAG-validated, finalize() enforced)
```
Topo order: S1 → S2 → S3 → PFC → M2 → M1

Feedforward (concatenated, source-aware sparsity on PFC/M2):
  S1→S2 (buf=4), S2→S3 (buf=8)
  S2+S3→PFC (word+topic → goal, 40% sparse per source)
  S2+PFC→M2 (word+goal → sequence, 40% sparse per source)
  M2→M1 (sequence → execution)

Apical (multi-source, per-source gain weights):
  Sensory top-down: S3→S2, S2→S1
  Motor monitoring: M1→M2, M2→PFC (corollary discharge)
  Cross: S1→M1 (sensory context), M1→S1 (efference copy)
  S1 receives from both S2 AND M1 (gains sum additively)

Surprise: S1→S2, S2→S3, S1→M1
```

Learning types by region:
- SensoryRegion (S1/S2/S3): two-factor Hebbian (traces available but default off)
- PFCRegion (PFC): three-factor (eligibility traces + reward), slow decay 0.97
- PremotorRegion (M2): two-factor Hebbian, temporal sequencing via lateral segments
- MotorRegion (M1): three-factor (eligibility traces + reward), L5 output, babbling

## Key Results

### Structural sparsity validated
40% per-source sparsity on PFC/M2 improves echo 38% (6.9% vs 5.0%).

### Echo (PFC→M2→M1)
- PFC three-factor biggest win: 3.1% → 8.2%
- Eligibility clip (0.05) only consistent tuning fix
- Babbling warmup before echo hurts (proactive interference)

### Sensory eligibility traces — wrong approach, not wrong idea
Swept 9 configs — no improvement. BUT: we implemented joint coincidence traces (pre AND post must co-occur, then smooth over time). This is temporal smoothing, not temporal credit assignment. The RIGHT approach is STDP-like presynaptic traces:
- Pre fires → trace on outbound synapses (regardless of post)
- Post fires later → strengthen synapses WITH traces, weaken WITHOUT
- This gives credit to inputs that PRECEDED activation, not just coincided

### Biology comparison (language focus)
**Good alignment**: columnar org, hierarchical abstraction, temporal integration, apical gain (BAC), dendritic segments, surprise modulation, BG gating, developmental stages, efference copy, mirror-like M2 activation during listen+speak
**Key gaps**: no cerebellum (forward model error correction), no thalamic relay, no STDP, passive PFC decay (should be recurrent), no dual stream distinction, no pattern completion (segments do temporal prediction only)

### Architecture Fixes (this session)
- Apical multi-source (was silently dropping signals)
- Multi-ff structural sparsity (40% per source on PFC/M2)
- Topology.step() multi-ff (proper concatenation)
- PFC three-factor (replaced reward_modulator replay hack)
- EchoReward RPE + partial credit
- S2 WordDecoder (trained during run(), saved in checkpoints)

## Uncommitted
- `.github/workflows/ci.yml` — typecheck scoped to core modules

## Next Steps (Priority Order)
- [ ] **STDP-like presynaptic traces in CorticalRegion** — foundation for all regions. Pre_trace decays + accumulates on input activity. Used when post fires (k-WTA). Third factor (reward) optional per region. Replaces current joint-coincidence approach.
- [ ] **Minimal cerebellar forward model** — M1 output → predicted S1 → error → M2. Addresses echo oscillation.
- [ ] **Recurrent PFC maintenance** — self-excitation within active columns, replacing passive voltage decay
- [ ] **M2 three-factor** — credit assignment gap
- [ ] **Per-stripe PFC gating** — concurrent goals
- [ ] **Longer echo runs** (5k+ episodes)
