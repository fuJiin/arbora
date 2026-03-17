# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Biologically-plausible cortical learning for next-token prediction. Minicolumn architecture, Hebbian + three-factor RL, no backprop. NumPy + Numba, Python 3.12+, uv.

## Architecture
```
CorticalRegion (region.py) — L4/L2/3, segments, apical, Hebbian
  ├── SensoryRegion — local connectivity → S1 (128c/k8), S2 (32c/k4), S3 (32c/k4)
  ├── MotorRegion — L5 output, babbling, three-factor → M1 (32c/k4)
  └── PFCRegion — slow decay, global gate → PFC (16c/k4) [NEW]

Connections: S1→S2→S3→PFC (feedforward), S3→S2→S1 (apical), S1→M1 (ff), PFC→M1 (apical)
```

## PFCRegion (implemented, not yet trained)
- `voltage_decay=0.97` → activity persists ~30 steps (working memory)
- Global gate: open=accept new input, closed=maintain goal via slow decay
- Receives S3 output (topic/phrase level context)
- PFC→M1 apical: biases motor output toward goal
- Confidence signal from activation strength
- Goal snapshot for echo mode comparison
- Future: per-stripe gating (PBWM macrocolumns) when multiple goals needed

## Reward Stack
- **Curiosity (RPE)**: per-bigram habituating. Drives exploration.
- **Caregiver**: optionality-scaled prefix signal + massive word completion bonus (scales with length) + habituation (0.7^count). Attention span 8 chars.
- **Anti-attractor**: curiosity + caregiver habituation + adaptive noise + three-factor negative reward.

## Results Summary
- **100k interleaved (old reward)**: 6 real 3-letter words (the, ask, has, not, him, mom). Best output.
- **100k interleaved (new reward)**: Higher bigram counts (it:115 vs 88) but fewer 3-char words. Habituation may be too aggressive.
- **500k interleaved**: In progress (~180k/500k). Will be longest run.

## Engineering This Session
- Extracted shared run loop methods (DRY)
- Fixed 5 bugs (learning gate, eligibility resets, checkpoints)
- Pre-allocated hot-path buffers (perf)
- MotorRegion inherits CorticalRegion (not SensoryRegion)
- PFCRegion implemented with simplified global gate

## Training Pipeline
1. **Sensory + M1/PFC listening** (300k) — S1→S2→S3, M1+PFC observe ✅
2. **Interleaved babble** (100k+) — curiosity + caregiver, English words ✅
3. **Echo mode** — PFC maintains "reproduce input" goal, M1 rewarded for matching ← NEXT
4. **Dialogue training** — listen → PFC → respond
5. **M2 sequencing** — when word length is bottleneck

## Runs In Progress
- 500k interleaved (old reward) — ~180k/500k, ~5 more hours

## Key Files
- `src/step/cortex/pfc.py` — PFCRegion (slow decay, global gate, confidence)
- `src/step/cortex/motor.py` — MotorRegion (L5, babbling, three-factor)
- `src/step/cortex/reward.py` — CuriosityReward, CaregiverReward
- `src/step/cortex/topology.py` — run(), run_babbling(), run_interleaved()
- `experiments/scripts/cortex_staged.py` — staged runner with PFC

## Next Steps
- [ ] **Echo mode training** — PFC gate open during listen, closed during speak. M1 rewarded for reproducing input. First PFC-gated behavior.
- [ ] **Analyze 500k run** when it finishes
- [ ] **Tune caregiver habituation** — may be too aggressive (fewer 3-char words)
- [ ] **Dialogue training** — structured listen→respond turns
- [ ] **S2/S3 global connectivity experiment**
