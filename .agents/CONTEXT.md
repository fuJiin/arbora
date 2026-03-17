# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Biologically-plausible cortical learning. Minicolumn architecture, Hebbian + three-factor RL, no backprop.

## Architecture
```
CorticalRegion (L4/L2/3, segments, apical, Hebbian)
  ├── SensoryRegion → S1 (128c/k8), S2 (32c/k4), S3 (32c/k4)
  ├── MotorRegion (L5, three-factor) → M1 (32c/k4)
  └── PFCRegion (slow decay 0.97, global gate) → PFC (16c/k4)

S1→S2→S3 (ff), S2→PFC (ff), PFC→M1 (apical), S1→M1 (ff)
```

PFC receives S2 (word-level) for goal specificity. PFC→M1 is modulatory (apical gain), not feedforward. PFC→M2 (ff) will be added when M2 is built. Biologically: PFC→premotor is ff, PFC→M1 is modulatory.

## Echo Mode (implemented, training)
- Listen: word flows through S1→S2→PFC (gate open). PFC learns word representation.
- PFC snapshots goal, closes gate.
- Speak: M1 produces chars. EchoReward compares against heard word.
- Reward→PFC: modulates PFC learning rate, replays heard word. Good echo → PFC representation strengthened.
- First result: "you"→"yoy" (first char match!), "huh"→"uuu" (vowel captured)
- 5k episodes: 4.1% match (1.3x chance). 50k episode run in progress.

## Motor Babbling (completed)
- Interleaved listen+babble with curiosity + caregiver reward
- Best: 6 real 3-letter words (the, mom, ask, him, not, has) from 100k babble
- 500k run in progress (~310k/500k)

## Runs In Progress
- **500k interleaved babble** — ~310k/500k
- **50k echo episodes** — just started

## Next Steps
- [ ] **Analyze 50k echo** — does match rate climb?
- [ ] **Analyze 500k babble** — best word production at scale
- [ ] **Dialogue training** — structured listen→respond
- [ ] **M2 design** — PFC→M2 (ff) → M1 (ff). Sequential motor planning.
