# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Research project exploring biologically-plausible learning for next-token prediction. Two architectures: (1) original flat STEP model (dense weight matrix + Hebbian learning), (2) new cortical region model (neocortical minicolumn architecture with L4/L2/3 layers). Built with NumPy, Python 3.12+, managed with uv.

## Current Work

### Branch: `main`

Cortex PoC with structural sparsity, LTD, burst mechanism, prediction LTD, synaptic decoder, per-neuron ff option.

### Critical finding: space character monopoly

Columns 29-31 fire for almost every token because `string.printable` puts space at index 94, which falls in col 31's receptive field. With k=4 and 3 columns fixed, the model has effectively k=1 discriminative column (~30 patterns for 1,105 tokens). Only 14.4% of tokens are uniquely identifiable by column set.

**This is the root cause of low accuracy**, not neuron count or prediction quality.

### Capacity sweep results (2026-03-12)
Swept 6 configs (shared/per-neuron ff × 4/8/16 neurons) on 10K tokens. All land at ~1-2% accuracy. Neither per-neuron ff_weights nor more neurons per column helps — because the column representation itself is undiscriminative.

### Prediction LTD (implemented, working)
- Predicted neurons dropped from 56 → 3-9 (much sparser)
- Unique prediction sets: 7,000+ (diverse)
- Weight differentiation improved (cosine 0.42 → 0.22)
- But accuracy unchanged because column sets are ambiguous

### Per-neuron ff_weights (implemented, no accuracy improvement)
- `per_neuron_ff=True` flag on SensoryRegion
- Each neuron gets own ff_weights within column's mask
- Better entropy (79% vs 74%) and more prediction diversity
- But still ~1% accuracy (same column ambiguity problem)

## Architecture Summary (`src/step/cortex/`)

- **L4/L2/3 minicolumn model** with burst/precise mechanism
- **Structural sparsity**: width-based ff_mask, local lateral/fb masks
- **LTD on ff_weights** (BCM-inspired) + **prediction LTD on fb/lateral weights**
- **Synaptic decoder** (`decoder.py`): nearest-neighbor via ff_weight reconstruction + column-level inverted index
- **Per-neuron ff option**: `per_neuron_ff=True` gives each neuron its own ff_weights

## Key Decisions
- **Prediction LTD**: weaken fb/lateral synapses when predicted neuron doesn't fire (heterosynaptic depression)
- **Per-neuron ff_weights**: biologically more accurate (neurons sample different synapses), but doesn't help until column representation is fixed
- **Shared ff is HTM-faithful** but puts all disambiguation on context pathway

## Next Steps
- [ ] **Fix space character monopoly** — reorder alphabet, input normalization, or inhibitory mechanism so universal chars don't dominate
- [ ] Consider k-WTA per receptive field region instead of global top-k
- [ ] Performance audit for multiple dendritic segments
- [ ] Fix pre-commit hook config (ty scope, pytest entry)
