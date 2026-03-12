# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Research project exploring biologically-plausible learning for next-token prediction. Two architectures: (1) original flat STEP model (dense weight matrix + Hebbian learning), (2) new cortical region model (neocortical minicolumn architecture with L4/L2/3 layers). Built with NumPy, Python 3.12+, managed with uv.

## Current Work

### Branch: `main`

Cortex PoC with three major additions addressing column monopoly:
1. **LTD (Long-Term Depression)** — weakens ff_weights to inactive inputs when column fires, per-column local sparsity scaling
2. **Structural sparsity** — width-based ff_mask (columns tile encoding_width), local connectivity masks for fb/lateral weights
3. **Burst mechanism** — unpredicted columns fire all neurons (surprise signal, 3x learning rate), predicted columns fire one neuron (precise)

### Results after all three fixes
- Column entropy: **72.1%** (was 42%)
- Unique column sets: **411** (was 7)
- Burst rate: **6.5%** (decreases as learning progresses)
- 34 tests passing

### Interactive dashboard
`experiments/scripts/cortex_dashboard.py` — Plotly dashboard on port 80 with 5 charts (entropy, heatmap, ff_weight divergence, voltage/excitability, drive distribution).

## Cortex Architecture (`src/step/cortex/`)

Two-layer neocortical minicolumn model:
- **L4 (input):** feedforward drive + dendritic spike prediction from feedback/lateral
- **L2/3 (associative):** L4 feedforward + lateral context
- **Burst/precise:** predicted columns → one neuron (precise), unpredicted → all neurons (burst, stronger learning)
- **LTD:** per-column weakening of inactive input connections (BCM-inspired)
- **Structural sparsity:** width-based receptive fields, local lateral/feedback masks

## Key Decisions
- **LTD replaces synapse_decay on ff_weights** — double decay killed weights
- **Width-based tiling** (not position) — columns detect character ranges across all positions
- **Per-column local LTD scaling** — global scaling was wrong for masked connectivity
- **Burst fires all neurons** — HTM-inspired surprise signal for unpredicted activations
- **Pre-commit hooks**: installed but ty/pytest hooks have pre-existing failures. CI only checks ruff format.

## Next Steps
- [ ] Scale to more tokens (10K+) and measure long-term learning curves
- [ ] Consider sparse weight updates for scaling beyond 128 neurons
- [ ] Experiment with activation thresholds instead of top-k columns
- [ ] Fix pre-commit hook config (ty scope, pytest entry)
