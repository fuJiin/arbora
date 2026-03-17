# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Biologically-plausible cortical learning for next-token prediction. Minicolumn model with L4/L2/3, dendritic segments, per-neuron ff_weights, apical gain, L5 output layer. NumPy + Numba, Python 3.12+, uv.

## Architecture
- **S1**: 128 cols, k=8. **S2**: 32 cols, k=4, buf=4. **S3**: 32 cols, k=4, buf=8.
- **M1**: 32 cols, k=4. L5 output (learned, sparse, three-factor). Babbling via direct column forcing.

## Reward Architecture
- **Curiosity (RPE)**: per-bigram expected vs actual S1 burst. Known patterns stop being rewarding.
- **Caregiver**: live prefix tracking against BabyLM vocabulary (~732 words). Positive-only. Attention span auto-reset at 8 chars.
- **Adaptive noise**: reward EMA delta + stagnation detection. Self-regulating.

## Best Results: Interleaved Listen+Babble (100k babble + 400k listen)

Space is #1 char (17.5%). Top chars: space, i, h, t, s, o, e, a — all high-frequency English.

| English bigrams | Count | Words found |
|----------------|-------|-------------|
| hi | 110 | "hi":18, "the":1, "mom":1 |
| it | 88 | "it":7, "ask":1, "has":1 |
| at | 74 | "ah":8, "not":1, "him":1 |
| is | 71 | "as":5, "he":4, "go":3 |
| th | 59 | "oh":4, "to":3, "is":2 |

6 different 3-letter words emerged: **the, ask, has, not, him, mom**.
English bigram density stable at ~6% throughout (no degradation).
Sample output t=10k: `'iteeidnt .e h heht h h h ihen '`

## Key Insight: Interleaved > Sequential
Pure babbling degrades L5 over time (500k run: English bigrams dropped from 55→10 "th"). Interleaved training keeps L5 calibrated by continually reinforcing token mappings from corpus. Biologically grounded: infants listen and babble simultaneously.

## Training Pipeline (current)
1. **Sensory + M1 listening** (300k corpus): S1→S2→S3 + M1 observes. ✅
2. **Interleaved babble** (100k babble + 400k listen): Curiosity + caregiver. English words emerge. ✅

## Session Stats: 32 commits
Major milestones: centroid BPC probe, L5 output layer, three-factor learning, curiosity RPE, caregiver reward, adaptive noise, interleaved training, M1 listening.

## Key Files
- `src/step/cortex/motor.py` — L5, three-factor, `_babble_direct()`
- `src/step/cortex/reward.py` — `CuriosityReward`, `CaregiverReward`
- `src/step/cortex/topology.py` — `run()`, `run_babbling()`, `run_interleaved()`
- `src/step/cortex/stages.py` — SENSORY, BABBLING, GUIDED stage defs
- `experiments/scripts/cortex_staged.py` — staged runner

## Checkpoints
- `stage1_sensory.ckpt` — 300k sensory with M1 listening
- `stage2_babbling.ckpt` — 100k interleaved babble

## Next Steps
- [ ] **REPL babbling mode** — watch M1 babble interactively, qualitative evaluation
- [ ] **Longer interleaved (500k babble)** — do more 3+ letter words emerge?
- [ ] **Dialogue training** — listen to utterance, babble response, listen to next
- [ ] **M2 sequencing** — once M1 reliably produces common bigrams
- [ ] **PFC design** — working memory, per-stripe BG, three-factor learning
- [ ] **Apical gain tuning** — deferred, use centroid BPC to A/B
