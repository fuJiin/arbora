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

## Key Architectural Learning: Apical ≠ Feedforward

**Echo mode failed at 4.2% match (50k episodes)** because PFC→M1 is apical (gain modulation). Apical can bias which neurons are more excitable, but can't select specific outputs. M1 collapses to producing 'e' regardless of apical signal. Direct S2→M1 apical also failed (9.1%, still 'e' collapse).

**The biological lesson:** PFC→M1 is modulatory (correct for mode/bias). Content-specific commands go PFC→M2→M1 via feedforward. This is why premotor cortex (M2) exists — it translates abstract goals into concrete motor sequences.

**Implication:** Echo/imitation needs M2 as a feedforward intermediary. PFC says "echo mode", S2 provides the word pattern, M2 translates to a char sequence plan, M1 executes. PFC→M1 apical is for mode selection only.

## What Works
- **Sensory**: S1→S2→S3, cbpc 4.93 at 300k. Centroid BPC (non-learned) as metric.
- **Babbling**: Interleaved listen+babble produces English words ("the", "mom", "ask"). Curiosity RPE + caregiver reward with habituation.
- **PFC region**: Implemented with slow decay, global gate, confidence signal. Learns from S2 input during sensory stage.

## What Doesn't Work Yet
- **Echo mode via apical**: Apical modulation can't drive specific outputs.
- **Caregiver habituation**: May be too aggressive (fewer 3-char words in new reward vs old).

## Runs
- **500k interleaved**: ~345k/500k, still running

## Next Steps (Priority Order)
- [ ] **M2 design + implementation** — feedforward intermediary: PFC→M2 (ff, goal→plan), S2→M2 (ff, word context), M2→M1 (ff, sequence→execution). This is the critical missing piece for echo/imitation.
- [ ] **Analyze 500k interleaved** when it finishes
- [ ] **Dialogue training** — structured listen→respond with PFC mode gating
- [ ] **Tune caregiver reward** — habituation rate, word bonus magnitude
