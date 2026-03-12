# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Research project exploring biologically-plausible learning for next-token prediction. Two architectures: (1) original flat STEP model (dense weight matrix + Hebbian learning), (2) new cortical region model (neocortical minicolumn architecture with L4/L2/3 layers). Built with NumPy, Python 3.12+, managed with uv.

## Current Work

### Branch: `main`

### Critical finding: prediction mechanism is broken

Encoding sweep (charbit, charbit-nosp, random) on 10K tokens proves the prediction pathway itself fails, not the encoding:
- **Random encoder**: 82.4% uniquely identifiable tokens, 98.9% entropy, max ambiguity 14 — **but 0.5% accuracy** (worse than charbit's 1.8%)
- Even with perfectly discriminative representations, the cortex cannot learn temporal patterns
- The Hebbian learning rule (trace × active → strengthen) is too diffuse to learn specific temporal associations

### Previous findings (still relevant)
- Space character monopoly in charbit encoding (cols 29-31 fire for everything)
- Prediction LTD works (sparser predictions, more diverse) but doesn't enable actual learning
- Per-neuron ff_weights and more neurons per column don't help (capacity sweep confirmed)

## Architecture Summary (`src/step/cortex/`)

- **L4/L2/3 minicolumn model** with burst/precise mechanism
- **Structural sparsity**: width-based ff_mask, local lateral/fb masks
- **LTD on ff_weights** + **prediction LTD on fb/lateral weights**
- **Per-neuron ff option**: `per_neuron_ff=True` (no accuracy improvement)
- **Synaptic decoder**: nearest-neighbor via ff_weight reconstruction

## Key Decisions
- **Prediction LTD** reduces predicted neurons 56→3-9 (good) but accuracy still ~1%
- **Shared ff_weights** puts all disambiguation on context pathway — which can't learn
- **Random encoder control** definitively proves encoding is not the bottleneck

## Next Steps
- [ ] **Fix the prediction learning rule** — current Hebbian is too diffuse for temporal associations
- [ ] Consider multiple dendritic segments (specific synapse matching vs whole-neuron Hebbian)
- [ ] Consider segment-level learning (HTM's dendritic segment permanence model)
- [ ] Performance audit before dendritic segment implementation
