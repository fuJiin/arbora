# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Biologically-plausible cortical learning for next-token prediction. Minicolumn architecture, Hebbian + three-factor RL, no backprop.

## Architecture
```
CorticalRegion (L4/L2/3, segments, apical, Hebbian)
  ├── SensoryRegion (local connectivity) → S1 (128c/k8), S2 (32c/k4), S3 (32c/k4)
  ├── MotorRegion (L5 output, three-factor) → M1 (32c/k4)
  └── PFCRegion (slow decay, global gate) → PFC (16c/k4)

S1→S2→S3→PFC (ff), S3→S2→S1 (apical), S1→M1 (ff), PFC→M1 (apical)
```

## What's Implemented
- **Sensory hierarchy**: S1→S2→S3 with apical feedback. Centroid BPC ~4.59 at 300k.
- **Motor babbling**: M1 with L5, three-factor RL, curiosity + caregiver reward. Interleaved listen+babble produces English words ("the", "mom", "ask", "him").
- **PFC region**: Slow voltage decay (0.97) for working memory. Global gate (open=update, closed=maintain). S3→PFC feedforward, PFC→M1 apical.
- **Echo mode**: `run_echo()` — hear word → PFC holds goal → M1 rewarded for matching. EchoReward with position-tolerant character matching.

## Runs In Progress
- **500k interleaved** (old reward) — 283k/500k, ~4 hours remaining
- **300k sensory with PFC** — just started, ~15 min. Will produce checkpoint with PFC weights for echo testing.

## Session Summary (40+ commits)
Major milestones: centroid BPC, L5 output layer, three-factor learning, curiosity RPE, caregiver reward (optionality + habituation), adaptive noise, interleaved training, M1 listening, PFC region, echo mode.

Engineering: DRY refactor (shared run loop methods), 5 bug fixes, perf pre-allocation, MotorRegion→CorticalRegion inheritance, code/perf audits.

## Next Steps
- [ ] **Test echo mode with pre-trained checkpoint** — when 300k sensory finishes
- [ ] **Analyze 500k interleaved** — when it finishes
- [ ] **Echo training runs** — does M1 learn to reproduce heard words?
- [ ] **Dialogue training** — structured listen→respond turns
- [ ] **Tune caregiver habituation** — may be too aggressive
