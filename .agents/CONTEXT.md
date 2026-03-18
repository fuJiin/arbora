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

Region types (all inherit CorticalRegion):
- SensoryRegion (S1/S2/S3): local connectivity
- PFCRegion (PFC): slow decay 0.97, global gate
- PremotorRegion (M2): temporal sequencing via lateral segments
- MotorRegion (M1): L5 output, three-factor RL, babbling

## Key Results

### Babbling
- English words ("the", "mom", "ask") from interleaved listen+babble
- Curiosity RPE + caregiver reward (optionality-scaled, habituating)

### Echo (PFC→M2→M1)
- **Peak: 13% match** (M2 single-ff), **10% match** (clean multi-ff + monitoring)
- Avg: 4.3-7.5% across configurations (2-4x above chance)
- **Core problem: oscillation.** Three-factor learning overshoots — match climbs to 10-13%, then drops to 2%, then recovers. Doesn't converge.
- Motor monitoring apical (M1→M2→PFC) gave marginal improvement (+0.4%)

### Architecture Insights
- **Apical = bias/mode, feedforward = content/command.** Echo via apical: 4.2%. Echo via ff: 13% peak.
- **Multiple ff to same target**: concatenated (biologically correct convergent input)
- **Motor monitoring apical** mirrors sensory feedback (M1→M2→PFC ↔ S3→S2→S1)

## Engineering This Session (65+ commits)
- Multi-ff concatenation with pre-allocated buffers
- DAG validation (finalize, cycle detection, dimension checking)
- Motor hierarchy apical (M1→M2→PFC monitoring)
- CI fixed (lint, format, types — 0 errors)
- Mermaid architecture diagram in README
- "Why this matters" section in README
- PFCRegion, PremotorRegion implementations
- Echo mode + dialogue training loops
- Dead code removal (old apical segments)
- Perf: cached ff connection lists, pre-allocated buffers

## Uncommitted
- `.github/workflows/ci.yml` — typecheck scoped to core modules (needs workflow OAuth scope to push)

## Next Steps (Priority Order)
- [ ] **Fix echo oscillation** — batch reward accumulation, learning rate scheduling, or curriculum (2-char → 3-char). The architecture works (peaks at 13%), the learning dynamics don't converge.
- [ ] **Numba for L2/3 segments** — 10-20% speedup, defer until training objective stabilizes
- [ ] **Dialogue training** with stable echo as foundation
- [ ] **Push CI workflow change** (requires workflow scope)
