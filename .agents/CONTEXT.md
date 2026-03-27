# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Biologically-plausible cortical learning. Minicolumn architecture, Hebbian + three-factor RL, no backprop. Full sensory-motor hierarchy with PFC goal maintenance.

## Integrations
- **Linear Project:** STEP (Sparse Temporal Eligibility Propagation)

## Architecture
```
Environment.step(action) -> (obs, reward)       — ChatEnv
  └── Agent.act(obs, reward) -> action           — ChatAgent
        └── Circuit.process(encoding) -> ndarray — pure neural

Topo: S1 → S2 → S3 → PFC → M2 → M1
Layers: L4 (input) → L2/3 (associative) → L5 (output/feedback)

Feedforward: L2/3 → L4 (canonical, Felleman & Van Essen 1991)
Apical: L5 → {L2/3, L5} (top-down context via L1)
Intra-region: L4 → L2/3 → L5 (per-column learned weights)

Learning: segments for prediction, per-connection traces, Hebbian/3-factor
  Surprise modulates FF only, not segments
```

## Lamina KPI Framework (decided 2026-03-27)

**L4** — Input reception + temporal prediction
1. Prediction recall (>70%) — of active neurons, fraction predicted. 1 - burst_rate.
2. Prediction precision (>50%) — of predicted neurons, fraction that fired
3. Population sparseness (~k/N) — proper sparse code per timestep (Treves-Rolls)

**L2/3** — Context-enriched, decodable output
1. BPC (<6.0) — downstream decodability. log2(vocab) is random baseline.
2. Context discrimination (>0.80) — same token, different patterns in different contexts
3. Effective dimensionality (>50) — participation ratio of activation covariance

**L5** — Feedback signal + subcortical output
1. Apical modulation index — top-down context effect on L5 patterns
2. Cross-layer divergence (CKA) — L5 carrying different info than L2/3
3. Decodability (only when L5 is primary output, i.e., motor regions)

Agranular regions (M1/M2/PFC): no L4, L2/3 absorbs L4 KPIs.

See .agents/METRICS_AUDIT.md for full framework, docs/BIBLIOGRAPHY.md for sources.

## Session: 2026-03-27 (S1 sweep + KPI framework)

### Key findings
- **Bug: L4 firing_rate never updated** — per-column L4→L2/3 weights had zero effect.
  Fixed by adding L4 firing_rate EMA update before _activate_l23. All 317 tests pass.
- **L2/3 burst ~70% even when L4 burst ~29%** — L4 prediction success was not flowing
  to L2/3 because the weight pathway was dead. L2/3 telemetry was a blind spot.
- **n_l5=0 matches baseline in S1-only** — L5 is inert without multi-region apical.
  Added guards for n_l5=0 in region.py.
- **trace=0.6 best for L4-level prediction** (BPC 1.94 vs 2.33 baseline at 30k)
- **syn=32 second best** — more synapses per segment = better specificity
- **prediction_gain has zero effect** — gain=1.5, 2.5, 4.0 all identical

### Baselines
| Config | BPC (300k) | Notes |
|--------|-----------|-------|
| STEP-73 (L5 FF, proxy weights) | 10.98 | L5 as FF source, firing rate proxy |
| Current (L2/3 FF, per-col weights, no fb_seg) | 10.80 | +1.6%, segments slowly engaging (4% den) |

### In progress
- S1 sweep with full KPI dashboard (L4 burst/precision/sparseness, L2/3 BPC/ctx_disc/eff_dim)
- Sweep script: experiments/scripts/sweep_s1.py

## Remaining tickets

**Ship next:**
- [ ] STEP-78 Fix L4 firing_rate bug (Urgent — per-column weights dead)

**Architecture:**
- [ ] STEP-63 Probe protocol — LaminaProbe/ChatLaminaProbe, runner ownership (M)
- [ ] STEP-79 Streamline Circuit — remove _in_eom/force_gate_open, pure process() (M, blocked by STEP-63)
- [ ] STEP-77 Agranular motor/PFC regions — skip L4 (M)
- [ ] STEP-30 Region Protocol typing (M, reassess after STEP-79)

**Features:**
- [ ] STEP-61 Adaptive gating — learned interleaving (XL)
- [ ] STEP-76 Consider pruning L5→L2/3 apical (S)
- [ ] STEP-20 Cerebellar forward model (XL)
- [ ] STEP-48 Checkpoint validation (S)
- [ ] STEP-37 Mermaid diagram generation (S)

**Deferred:**
- L6 layer (thalamic gain control)
- L5 KPIs (blocked on multi-region)

**In progress:**
- STEP-41 Bibliography — partial (docs/BIBLIOGRAPHY.md started, needs full pass)

**Closed this session:** STEP-38 (dup of 61), STEP-42 (stale), STEP-58 (merged into 63)

See .agents/BIOLOGY_AUDIT.md for connection accuracy audit.
