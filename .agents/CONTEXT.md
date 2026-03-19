# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Biologically-plausible cortical learning. Minicolumn architecture, Hebbian + three-factor RL, no backprop. Full sensory-motor hierarchy with PFC goal maintenance.

## Architecture (DAG-validated, finalize() enforced)
```
Topo order: S1 → S2 → S3 → PFC → M2 → M1

Feedforward (concatenated for multi-source):
  S1→S2 (buf=4), S2→S3 (buf=8)
  S2+S3→PFC (word+topic → goal)
  S2+PFC→M2 (word+goal → sequence)
  M2→M1 (sequence → execution)

Apical:
  Sensory top-down: S3→S2, S2→S1
  Motor monitoring: M1→M2, M2→PFC (corollary discharge)
  Cross: S1→M1 (sensory context), M1→S1 (efference copy)

Surprise: S1→S2, S2→S3, S1→M1
```

Learning types by region:
- SensoryRegion (S1/S2/S3): **two-factor Hebbian** on ff_weights (no eligibility traces)
- PFCRegion (PFC): **three-factor** (eligibility traces + reward consolidation), slow decay 0.97
- PremotorRegion (M2): two-factor Hebbian, temporal sequencing via lateral segments
- MotorRegion (M1): **three-factor** (eligibility traces + reward), L5 output, babbling

## Key Results

### Echo (PFC→M2→M1) — 4 sweep iterations this session
- **PFC three-factor learning** was the biggest single win: baseline 3.1% → 8.2%
- **Eligibility clip (0.05)** is the only consistently helpful tuning fix across all sweeps
- **RPE-based match reward** is architecturally cleaner but slightly underperforms fixed reward (5.7% vs 8.2%) — self-dampening kicks in before sufficient learning
- **Batch reward** helped pre-PFC-3-factor (7.5%) but harms post (0.8%) — PFC traces decay during speak phase, batch delays consolidation past decay window
- **Motor surprise** (M1→M2, M2→M1): neutral to harmful in current testing
- **Proper staging** (sensory → babbling → echo) being tested — hypothesis that M1 cold-start is a major oscillation source

### Architecture Insights
- **Apical = bias/mode, feedforward = content/command**
- **PFC three-factor**: teaches PFC to produce activation patterns useful for downstream reward, not just input-representative patterns. Replaced crude reward_modulator replay hack.
- **Eligibility clip** needed in PFC/Motor because traces accumulate over multiple steps. PFC is worst due to slow voltage decay keeping columns active ~30 steps. Sensory regions don't have traces so don't need clip.

## Uncommitted
- `.github/workflows/ci.yml` — typecheck scoped to core modules (needs workflow OAuth scope)

## Next Steps (Priority Order)
- [ ] **Babbling warmup results** — test running. Does sensory+babbling before echo reduce oscillation?
- [ ] **Eligibility traces in sensory regions** — currently two-factor only. Three-factor with surprise as consolidation signal could improve representations. Biologically grounded (synaptic tagging is universal cortical machinery, not PFC-specific).
- [ ] **M2 three-factor** — M2 currently two-factor Hebbian. Credit assignment chain: S2+S3→[3-factor]→PFC→[2-factor]→M2→[2-factor]→M1→[3-factor]→output. M2 is a blind relay.
- [ ] **Per-stripe PFC gating** — needed for multiple concurrent goals (echo vs dialogue). Not blocking for single-task echo.
- [ ] **Longer echo runs** (5k+ episodes) with best config to test convergence
- [ ] **Dialogue training** with stable echo as foundation
