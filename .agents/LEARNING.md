# Learning Mechanism Audit

> **DELETE THIS FILE** once findings are absorbed and actioned into STEP-62.
> Created 2026-03-26 as pre-work for unifying learning mechanics.

## Connections

| Connection | Source | Target | Role | Learning | Notes |
|---|---|---|---|---|---|
| S1→S2 | L2/3 | L4 | FF | Hebbian + burst gate + surprise | Buffer depth, structural sparsity |
| S2→S3 | L2/3 | L4 | FF | Hebbian + burst gate + surprise | Buffer depth |
| S2→PFC | L2/3 | L4 | FF | Three-factor | |
| S3→PFC | L2/3 | L4 | FF | Three-factor | |
| S2→M2 | L2/3 | L4 | FF | Three-factor | |
| PFC→M2 | L2/3 | L4 | FF | Three-factor | |
| M2→M1 | L2/3 | L4 | FF | Three-factor | |
| S2→S1 | L2/3 | L4 | Apical | Linear gain (slow Hebbian) | |
| S3→S2 | L2/3 | L4 | Apical | Linear gain | |
| M1→M2 | L2/3 | L4 | Apical | Linear gain | |
| M2→PFC | L2/3 | L4 | Apical | Linear gain | |
| S1→M1 | L2/3 | L4 | Apical | Linear gain + surprise | |
| M1→S1 | L2/3 | L4 | Apical | Linear gain | Efference copy |

## Segment types per region

| Segment | Layer | Context Source | Traces | Bio basis | Status |
|---|---|---|---|---|---|
| fb_seg (feedback) | L4 | L2/3 activity | Yes | L2/3→L4 feedback dendrites | Working |
| lat_seg (lateral) | L4 | L4 activity | Yes | L4→L4 lateral prediction | Working |
| l23_seg (lateral) | L2/3 | L2/3 activity | Yes | L2/3→L2/3 sequence learning | Working |
| l5_seg (lateral) | L5 | L5 activity | Yes (STEP-54) | L5→L5 output sequence | Off by default, regressed at 50k |
| apical_seg | L5 | External source | **No traces** | Apical BAC firing | +18% PFC ctx_disc |

## Feedforward rules

| Rule | Regions | Mechanism | Evidence |
|---|---|---|---|
| HEBBIAN | S1, S2, S3 | Immediate LTP on pre-trace, LTD on inactive. Surprise modulates. | BPC improves, representations form |
| THREE_FACTOR | PFC, M2, M1 | Eligibility trace accumulates, consolidated on reward | Motor output works, echo partial credit |

## Apical modes

| Mode | Mechanism | Evidence |
|---|---|---|
| Linear gain (default) | Slow Hebbian weight matrix, multiplicative on L4 voltage. LTP-only. | Modest improvement |
| Segments (opt-in) | Dendritic segments on L5, grow/reinforce/punish. Boolean external context. | +18% PFC but no traces |

## Known issues

1. **Apical segments lack traces** — boolean external context while all others use continuous traces
2. **L5 lateral off by default** — regressed at 50k (immature predictions bias selection)
3. **Apical gain is LTP-only** — no LTD, passive decay only. Slow saturation risk
4. **No FF prediction penalty** — wrong predictions only punish segments, not FF weights
5. **Surprise scales ALL learning** — segments learn structure, may not want surprise gating
6. **Pre-trace vs eligibility naming confusion** — orthogonal but confusingly similar

## Perf impact

| Mechanism | Cost | Notes |
|---|---|---|
| FF Hebbian | Low | Matrix multiply on winners only, ~5% of step |
| FF Three-factor | Very low | Just accumulate eligibility; reward consolidation separate |
| Segment prediction | **Highest** | Per-neuron scoring dominates when n_segments high |
| Segment learning | Moderate | Batch grow/adapt per burst/precise neuron |
| Trace updates | Negligible | O(n_neurons) per layer |
| Apical gain | Low | Sparse outer product on active neurons |

## Open questions (pre-STEP-62)

### L5→L4 feedforward?
Do we need L5 (lower) → L4 (higher) in addition to L2/3 (lower) → L4 (higher)?
Could increase neurons/columns to accommodate.

### S2→PFC reward gating
S2→PFC and S3→PFC are both three-factor, but the question is whether
sensory→frontal should be reward-gated at all vs immediate Hebbian.

### Motor FF learning rule
Should all motor FF be three-factor? Should reward just modulate by
increasing LR rather than gating entirely?

### Apical pathway biology
Are L2/3 (higher) → L4 (lower) apical pathways real? Biology suggests:
- L5 (higher) → L2/3 (lower) for feedback
- L5 (lower) → L2/3 (higher) for motor efference

### Motor hierarchy accuracy
Is M2 really premotor cortex (PMC)? How does human M1/PMC/SMA wiring look?

### L4 prediction mechanism
L2/3→L4 apical (same region, fb_seg) drives predictions. But biology has
L4→L4 lateral for prediction. How is the current architecture doing
context discrimination? Is fb_seg serving as a proxy?

### L2/3 apical from external sources
Shouldn't L2/3 also get apical segments from external sources like L5 does?
This would replace the current fb_seg (L2/3→L4 within region) with
proper top-down context on L2/3.

### Eligibility traces on all FF
Hebbian on input→L4 makes sense for sensory. Should other FF connections
(S2→S3, S2→PFC) use eligibility traces? Or experiment first?

### Linear gain vs three-factor apical
Linear gain is a simplistic Hebbian gain modulation. Should we replace
with three-factor on all apical connections?

### Segments vs neuron-to-neuron
Are segments worth the complexity? Alternative: direct neuron-to-neuron
connections with structural sparsity (like FF weights). Segments add
combinatorial context sensitivity but at high computational cost.
