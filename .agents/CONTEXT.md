# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Research project exploring biologically-plausible learning for next-token prediction. Two architectures: (1) original flat STEP model (dense weight matrix + Hebbian learning), (2) new cortical region model (neocortical minicolumn architecture with L4/L2/3 layers). Built with NumPy, Python 3.12+, managed with uv.

## Current Work

### Branch: `cortext` — PR #2 (open)
https://github.com/fuJiin/STEP/pull/2

Core cortex implementation + PoC tooling + tuning. Encoder refactoring (SDR -> encoders package) also included.

### Cortex PoC status
Diagnostics reveal issues not yet fixed:
- Column monopoly (31/128 neurons, 49% entropy) — ff_weights not differentiating
- Voltage accumulation on non-winners drives monopoly

**PoC design:**
- Own training loop (not shimming into old predict/learn/observe protocol)
- Prediction = pre-activation voltage from feedback/lateral before next input
- Measurement = overlap between pre-activated and actually-activated neurons
- CharbitEncoder for cortex, RandomEncoder for STEP baseline
- Params matched: cortex ~72K vs STEP ~65K (n=256, k=10)
- Story boundaries: reset working memory, keep synaptic weights

## Cortex Architecture (`src/step/cortex/`)

Two-layer neocortical minicolumn model:
- **L4 (input):** feedforward drive + feedback context
- **L2/3 (associative):** L4 feedforward + lateral context, enables associative binding

Activation: top-k columns -> winner-take-all L4 neuron -> winner-take-all L2/3 neuron (L4 bias + lateral + excitability).

### Performance concerns (deferred, fine at PoC scale)
Dense weight matrices O(n^2), full outer product learning, full matvec feedback — needs sparse updates for >128 neurons.

## Baseline Results

| Model | Accuracy |
|-------|----------|
| TinyStories-1M ceiling | **42.2%** |
| Bigram baseline | **28.9%** |
| STEP w=3 (embedding SDRs, 200K) | **28.9%** |

**Key finding:** Hebbian rule can't do credit assignment at scale — cortex architecture aims to fix this via structured columns and competitive inhibition.

## Key Decisions
- **Story boundaries**: reset working memory, keep synaptic weights
- **Tackle issues one-by-one** for publishable attribution
- **Goal**: prove online continual learning without catastrophic forgetting, then match transformer accuracy
- **Argmax winner selection** for PoC, activation thresholds + bursting for later
- **Pre-commit hooks**: installed but ty/pytest hooks have pre-existing failures. CI only checks ruff format.

## Next Steps
- [ ] Fix column diversity / monopoly — ff_weights differentiation is the bottleneck
- [ ] Address voltage accumulation on non-winners
- [ ] Consider sparse weight updates for scaling beyond 128 neurons
- [ ] Fix pre-commit hook config (ty scope, pytest entry)
