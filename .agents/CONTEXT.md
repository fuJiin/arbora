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
- SensoryRegion (S1/S2/S3): **two-factor Hebbian** (traces available but default off)
- PFCRegion (PFC): **three-factor** (eligibility traces + reward), slow decay 0.97
- PremotorRegion (M2): two-factor Hebbian, temporal sequencing via lateral segments
- MotorRegion (M1): **three-factor** (eligibility traces + reward), L5 output, babbling

## Key Results

### Echo (PFC→M2→M1) — 4 sweep iterations
- **PFC three-factor** was biggest win: baseline 3.1% → 8.2%
- **Eligibility clip (0.05)** only consistently helpful tuning fix
- **RPE match reward** cleaner but slightly underperforms (5.7% vs 8.2%)
- **Babbling warmup before echo hurts** — proactive interference
- **Motor surprise** (M1→M2, M2→M1): neutral to harmful

### Sensory Eligibility Traces — negative result
Swept 9 configs (trace_fraction x decay). No improvement on burst rate or centroid BPC. Best config (tf=0.1, decay=0.90) was within noise. Two-factor Hebbian with surprise modulation is already well-tuned for sensory learning — character representations are per-step, temporal echoes add noise. Infrastructure preserved (SensoryRegion.trace_fraction) but default off.

### Architecture Fixes (this session)
- **Apical multi-source**: fixed overwrite bug, per-source gain weights
- **Multi-ff structural sparsity**: 40% sparse per source on PFC/M2
- **Topology.step() multi-ff**: proper concatenation for PFC/M2
- **PFC three-factor**: eligibility traces + reward consolidation
- **EchoReward RPE**: self-dampening match signal, partial credit
- **S2 WordDecoder**: word-level predictions in REPL

### Demo Tools
- **REPL**: full architecture, /echo, /babble, /probe, burst surprise %, S2 word context, checkpoint loading
- **Dashboard**: hierarchy tabs, all fixes, served on port 8080

## Uncommitted
- `.github/workflows/ci.yml` — typecheck scoped to core modules (needs workflow OAuth scope)

## Next Steps (Priority Order)
- [ ] **M2 three-factor** — credit assignment gap: PFC→[2f]→M2→[2f]→M1
- [ ] **Longer echo runs** (5k+ episodes) with best config
- [ ] **Per-stripe PFC gating** — needed for multiple concurrent goals
- [ ] **Dialogue training** with stable echo as foundation
- [ ] **Apical-triggered sensory consolidation** — if we revisit sensory traces, use apical calcium signal instead of surprise EMA
