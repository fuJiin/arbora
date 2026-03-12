# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Research project exploring biologically-plausible learning for next-token prediction. Two architectures: (1) original flat STEP model, (2) cortical region model (neocortical minicolumn with L4/L2/3 layers, dendritic segments). Built with NumPy, Python 3.12+, managed with uv.

## Architecture (`src/step/cortex/`)

- **L4/L2/3 minicolumn model** with burst/precise activation
- **Dendritic segments** for prediction: fb (L2/3→L4) + lat (L4→L4), HTM-style permanence learning
- **SensoryRegion**: local connectivity (radius = n_columns//4), structural ff/fb/lat masks
- **RepresentationTracker** (`representation.py`): primary metrics for evaluating sensory cortex quality
- **Synaptic decoder**: kept for monitoring, NOT the primary optimization target
- **Dense weights**: kept for diagnostics/decoder backward compat

## Current Work

### Branch: `main`
- Dendritic segments committed (`15ff0e6`)
- Uncommitted: RepresentationTracker + eval script + BabyLM support

### Representation metrics (new, primary)
Replaced decoder accuracy as primary evaluation with representation quality:
1. **Column selectivity** — normalized entropy per column (0=feature detector, 1=uniform)
2. **Representational similarity** — pairwise Jaccard of column profiles (non-trivial structure check)
3. **Context discrimination** — Jaccard distance of neuron patterns for same token in different contexts
4. **FF convergence** — weight sparsity, RF entropy, cross-column cosine

### Results at 10K tokens (random encoder, t2+i0.2 segments)
| Metric | TinyStories | BabyLM |
|---|---|---|
| Selectivity | 0.605 | 0.625 |
| Similarity (nontrivial) | 0.054 ✓ | 0.052 ✓ |
| Context discrimination | 0.602 | 0.591 |
| Cross-col cosine | 0.018 | 0.016 |
| Burst rate | 36.5% | 35.2% |

Both datasets produce similar results with random encoder (expected). Switching to BabyLM as default for biological plausibility.

## Key Decisions
- **Representation quality over decoder accuracy** — sensory cortex builds representations for downstream regions (motor cortex, PFC), not for next-token prediction
- **Motor cortex will generate responses** (not predict next token) — important architectural distinction
- **BabyLM over TinyStories** — more naturalistic, developmentally plausible data
- **Dendritic segments over dense Hebbian** for prediction
- **Segment params**: thresh=2, perm_inc=0.2 (best from sweep)

## Long-term Architecture Vision
- Sensory cortex (current) → low-level features, semantic representations
- Secondary sensory region → higher-level features (planned)
- Motor cortex → response generation (planned)
- PFC → longer-range reasoning (planned)

## Next Steps
- [ ] Switch default dataset to BabyLM
- [ ] Improve column selectivity (currently 0.6, want lower)
- [ ] Add secondary sensory region for higher-level features
- [ ] Consider real (learned) encoder to replace random binary encoder
- [ ] Explore motor cortex design for response generation
