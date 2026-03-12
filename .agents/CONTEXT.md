# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Research project exploring biologically-plausible learning for next-token prediction. Two architectures: (1) original flat STEP model (dense weight matrix + Hebbian learning), (2) new cortical region model (neocortical minicolumn architecture with L4/L2/3 layers). Built with NumPy, Python 3.12+, managed with uv.

## Current Work

### Branch: `main` (uncommitted changes)

### Dendritic segments implemented (not yet committed)

Replaced the dense fb_weights/lateral_weights prediction mechanism with HTM-style dendritic segments:
- Each L4 neuron has multiple dendritic segments (fb: L2/3→L4, lateral: L4→L4)
- Each segment has sparse synapses with permanence values
- A segment fires when enough connected synapses have active sources (threshold-based pattern matching)
- Provides the specificity needed for temporal association learning

**Modified files**: `region.py`, `sensory.py`, `config.py`, `diagnostics.py`, `test_region.py`

### Early results (500 tokens)
- Segments growing: 2.7% fb connected, 1.8% lat connected
- Burst rate 63.4% (down from 100% initial — segments are predicting)
- Dense weights kept alongside for backward compat/diagnostics
- All 137 tests pass + 8 new segment-specific tests

### Critical finding (still relevant): prediction mechanism was broken
- Random encoder control proved encoding is NOT the bottleneck
- Dense Hebbian (trace × active → strengthen) too diffuse for temporal patterns
- Dendritic segments are the fix: specific synapse matching vs whole-neuron Hebbian

## Architecture Summary (`src/step/cortex/`)

- **L4/L2/3 minicolumn model** with burst/precise mechanism
- **Dendritic segments** for prediction: `fb_seg_indices/perm` (L2/3→L4), `lat_seg_indices/perm` (L4→L4)
- **Segment learning**: grow on burst, reinforce on precise, punish false predictions
- **SensoryRegion** overrides segments with local connectivity (radius = n_columns//4)
- **Dense weights** kept for diagnostics/decoder backward compat
- **Structural sparsity**: width-based ff_mask, local lateral/fb masks
- **Synaptic decoder**: nearest-neighbor via ff_weight reconstruction

## Key Decisions
- **Dendritic segments over dense Hebbian** for prediction — encoding sweep proved dense approach cannot learn temporal patterns
- **Hybrid approach**: dense weights kept for diagnostics alongside segment-based prediction
- **Segment params**: 4 fb + 4 lat segments, 16 synapses each, threshold 4, perm_init 0.6
- **predict_neuron() helper** for test setup (fills segment with single source index)

## Next Steps
- [ ] **Run longer experiment** (10K+ tokens) to evaluate segment learning convergence
- [ ] **Tune segment parameters** — activation threshold, perm_increment/decrement, n_segments
- [ ] **Commit dendritic segments** once validated on longer run
- [ ] Consider removing dense fb/lateral weights if segments prove sufficient
- [ ] Encoding improvements (strip spaces, custom tokenizer, prebuilt embeddings)
