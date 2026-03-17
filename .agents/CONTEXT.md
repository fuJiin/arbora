# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Biologically-plausible cortical learning for next-token prediction. Minicolumn model with L4/L2/3, dendritic segments, per-neuron ff_weights, apical gain, L5 output layer. NumPy + Numba, Python 3.12+, uv.

## Architecture
- **S1**: 128 cols, k=8. **S2**: 32 cols, k=4, buf=4. **S3**: 32 cols, k=4, buf=8.
- **M1**: 32 cols, k=4. L5 output (learned, sparse, three-factor). Babbling via direct column forcing.

## Reward Architecture
- **Curiosity (RPE)**: per-bigram expected vs actual S1 burst. Known patterns stop being rewarding.
- **Caregiver**: live prefix tracking against vocabulary. Positive-only (no dead-end penalty). Attention span auto-reset at 8 chars.
- **Adaptive noise**: reward EMA delta + stagnation detection. Self-regulating.

## Breakthrough: M1 Listening + Babbling Produces English

### M1 listening during sensory stage
M1 connected during Stage 1 with all learning enabled. Processes every corpus token: ff_weights learn S1→M1 mapping, segments learn temporal prediction, L5 learns token associations. Like infants hearing language before babbling.

### Results (300k listening → 100k babbling)

| Metric | Without listening | With listening |
|--------|------------------|----------------|
| Vocab at t=100 | 6-8 | **25** |
| English bigrams | 40 | **538** |
| "th" | 28 | **55** |
| "is" | 0 | **123** |
| "it" | 0 | **65** |
| "ha" | 0 | **68** |
| Real words found | "go":9 | **"is":8, "it":8, "hi":22, "at":3, "as":5, "be":2, "if":2** |

First babbling output (t=100): `'o osmlsnphfsabssmm wowf?s sa o'`
t=1000: `"lol'lgnccsbpbcsrsln?n?gtlltden"`

## Reward Design Lessons
- Curiosity (intrinsic) = exploration engine. Never remove.
- Caregiver (extrinsic) = gentle bias toward English. Positive-only.
- Dead-end penalties cause attractor collapse. Only reward progress.
- L5 random init → non-English chars dominate. Listening phase fixes this.

## Training Pipeline
1. **Sensory + M1 listening** (300k corpus): S1→S2→S3 + M1 observes. ✅
2. **Babbling** (100k autoregressive): Curiosity + caregiver. English words emerge. ✅
3. Stages 2/3 collapsed (both use caregiver + curiosity).

## Key Files
- `src/step/cortex/motor.py` — L5, three-factor, `_babble_direct()`
- `src/step/cortex/reward.py` — `CuriosityReward`, `CaregiverReward`
- `src/step/cortex/topology.py` — `run_babbling()`, adaptive noise
- `src/step/cortex/stages.py` — SENSORY (with M1), BABBLING, GUIDED
- `experiments/scripts/cortex_staged.py` — staged runner

## Next Steps
- [ ] **Longer babbling (500k)** — do more/longer words emerge?
- [ ] **REPL babbling mode** — watch M1 babble interactively
- [ ] **Analyze word frequency** — does word distribution match BabyLM?
- [ ] **M2 sequencing** — plan multi-char motor sequences
- [ ] **PFC design** — working memory, per-stripe BG, three-factor learning
