# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Biologically-plausible cortical learning. Minicolumn architecture, Hebbian + three-factor RL, no backprop.

## Architecture (DAG-validated)
```
Topo order: S1 → S2 → S3 → PFC → M2 → M1

Feedforward (concatenated for multi-source targets):
  S1→S2 (buf=4, burst-gated), S2→S3 (buf=8, burst-gated)
  S2+S3→PFC, S2+PFC→M2, M2→M1

Apical: S3→S2, S2→S1, S1→M1, M1→S1 (disabled)
Surprise: S1→S2, S2→S3, S1→M1
```

Topology.finalize() validates: no ff cycles, dimension matching (with buffer_depth), entry exists. Auto-finalizes on first use.

## Key Architectural Insights
- **Apical = bias/mode. Feedforward = content/command.** Echo via apical: 4.2%. Echo via ff: 13% peak.
- **Multiple ff to same target**: concatenated (convergent input). PFC gets S2+S3. M2 gets S2+PFC. Biologically correct.
- **PFC→M2→M1** is the motor hierarchy. PFC holds goal, M2 sequences, M1 executes.

## Current Status
- **Echo with M2**: 8.2% match (clean architecture, 100 eps). Run in progress: 300k sensory + 10k echo.
- **Learning rate tuned**: goal weights at 0.1x lr, 0.3x reward consolidation for stability.

## Engineering This Session
- Multi-ff support (concatenation instead of break-on-first)
- DAG validation (finalize, cycle detection, dim checking)
- CI fixed (61 lint errors, formatting)
- README with accurate connection table + "why this matters" section
- run() uses _propagate_feedforward() (was inlined single-ff)

## Runs In Progress
- 10k echo with clean multi-ff architecture

## Next Steps
- [ ] **Analyze 10k echo results** — does M2 pathway keep improving?
- [ ] **Type checker fixes** (ty check has optional attribute issues)
- [ ] **Longer echo/dialogue** with tuned learning rates
- [ ] **M2 temporal sequence learning** — do lateral segments learn char sequences?
