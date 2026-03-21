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
- SensoryRegion (S1/S2/S3): **two-factor Hebbian** (no eligibility traces)
- PFCRegion (PFC): **three-factor** (eligibility traces + reward), slow decay 0.97
- PremotorRegion (M2): two-factor Hebbian, temporal sequencing via lateral segments
- MotorRegion (M1): **three-factor** (eligibility traces + reward), L5 output, babbling

## Key Results

### Echo (PFC→M2→M1) — 4 sweep iterations
- **PFC three-factor** was biggest win: baseline 3.1% → 8.2%
- **Eligibility clip (0.05)** only consistently helpful tuning fix across all sweeps
- **RPE match reward** cleaner but slightly underperforms (5.7% vs 8.2%)
- **Babbling warmup before echo hurts** — proactive interference from babbling-trained weights conflicts with echo objective
- **Motor surprise** (M1→M2, M2→M1): neutral to harmful in current testing

### Architecture Fixes (this session)
- **Apical multi-source**: S1 was silently dropping S2 apical when M1 also connected (overwrite bug). Now per-source gain weights, additive combination.
- **Multi-ff structural sparsity**: PFC/M2 had full connectivity to concatenated inputs. Now 40% sparse per source for column specialization.
- **Topology.step() multi-ff**: was using naive first-connection-break loop, crashed on PFC. Now uses proper concatenation like _propagate_feedforward.
- **PFC three-factor**: replaced crude reward_modulator replay hack with eligibility traces + reward consolidation.
- **EchoReward RPE**: match signal is now RPE-based (actual - expected), self-dampening. Partial credit for right-char-wrong-position.

### Demo Tools (updated this session)
- **REPL**: full S1→S2→S3→PFC→M2→M1 architecture, /echo command, burst surprise % display, checkpoint loading
- **Dashboard**: fixed crashes, hierarchy tabs, title links to index, sorted by mtime

## Uncommitted
- `.github/workflows/ci.yml` — typecheck scoped to core modules (needs workflow OAuth scope)

## Next Steps (Priority Order)
- [ ] **Eligibility traces in sensory regions** — three-factor with surprise as consolidation signal. Biologically grounded (synaptic tagging is universal). Would enable longer-range causal learning in S1/S2/S3.
- [ ] **S2 word-level dendritic decoder** — qualitative tool for REPL, shows what words S2 is recognizing
- [ ] **Fresh training run** with apical fix + structural sparsity (old checkpoints incompatible)
- [ ] **M2 three-factor** — credit assignment gap: PFC→[2-factor]→M2→[2-factor]→M1. M2 is a blind relay.
- [ ] **Per-stripe PFC gating** — needed for multiple concurrent goals
- [ ] **Longer echo runs** (5k+ episodes) to test convergence
