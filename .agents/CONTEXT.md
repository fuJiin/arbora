# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Research project comparing biologically-plausible learning (eligibility propagation + sparse distributed representations) against standard backprop in transformers. Built with NumPy (STEP model) and PyTorch (baseline GPT). Python 3.12+, managed with uv.

## Architecture
- **Dual model implementations**:
  - `model.py`: Functional core with explicit state threading via `ModelState` NamedTuple (dense numpy (n,n) weight matrix)
  - `db.py`: SQLite-backed `StepModel` class — dropped from exp0 due to O(80K) SQL upserts/token. Future: hybrid numpy compute + SQLite checkpoint.
- **Model wrappers** (`step/wrappers.py`, `baselines/wrappers.py`): Wrap implementations into `Model` protocol
  - `StepMemoryModel`: In-memory STEP with inverted index decode (O(k) per query)
  - `MiniGPTModel`: Inference-only GPT wrapper with rolling context buffer
  - `TinyStories1MModel`: Pre-trained HuggingFace model wrapper (inference-only, 3.7M params)
- **Model protocol** (`protocol.py`): `Model` protocol for running STEP and baselines through the same harness
- **Key modules**: `sdr.py` (encoding), `metrics.py` (IoU, rolling mean), `data.py` (tokenizer + dataset caching + vocab filtering + .env HF token)
- **Experiment infra**: `experiment.py` (generic `run_experiment()` + `pretrain_step_model()` with callback hooks, JSON logging)
- **Diagnostics** (`step/diagnostics.py`): `DiagnosticCollector` for weight stats, prediction logging, bigram SDR overlap, per-position accuracy
- **Baselines**: `mini_gpt.py` (2-layer transformer), `pretrain.py` (training loop with cosine LR schedule), `wrappers.py`

## exp2 Results (completed)
Branch: `worktree-exp0` — TinyStories-1M ceiling + STEP diagnostics (200K pretrain, 10K eval)

| Model | Params | Mean Accuracy | Final Rolling Accuracy | Native Metric |
|-------|--------|--------------|----------------------|---------------|
| TinyStories-1M (ceiling) | 3.7M | 24.14% | 30.00% | CE Loss: 4.70 / pplx 110 |
| step_memory | 4.2M (n=2048) | 9.47% | 3.00% | IoU: 10.76% |

**Note on ceiling quality**: CE loss 4.70 (pplx 110) is much worse than published 1.71 (pplx 5.5) because we feed clamped 10K-vocab tokens to a model trained on 50K vocab — it sees UNK constantly where it expects real tokens. Still valid as a *relative* ceiling (24% vs 3%) but absolute numbers are degraded. Revisit closer to publishing: either use a 10K-vocab variant or fine-tune on clamped data.

### Diagnostic Findings (priority order)
1. **Weight explosion (Critical)**: Weights grow linearly without bound (mean 0→1687, max→100K+ by 200K steps). 100% of weights > 1.0 by step ~15K. The Hebbian learning rule is purely additive with no effective normalization. This is the #1 fix needed.
2. **Prediction collapse**: All top-20 confusions are `token_0 -> <actual>`. STEP predicts the same dominant SDR for everything, consequence of weight explosion.
3. **SDR encoding has zero structure**: Bigram SDR overlap averages 0.9 bits (excluding self-bigrams), essentially random (k²/n = 0.78). 17/48 non-self bigrams have zero overlap. Hash-based encoding provides no similarity between related tokens.
4. **Position accuracy**: Flat ~8-10% across story positions 0-230, drops to 0% at 240+.
5. **IoU distribution**: Bimodal — mostly near 0, with a bump at ~0.45 from repeated tokens.

## exp1 Results (completed)
Ceiling transformer attempt with 10K vocab, 2M tokens — both models stuck at 3% accuracy. Key finding: 2M tokens severely insufficient for MiniGPT (needs 50-100M+).

## exp0 Results (completed)
Two-model comparison on TinyStories (50K vocab, 100K train tokens) — both at 6% accuracy.

## Key Decisions
- `vocab_size` added to `EncoderConfig` (default 50257 for backward compat)
- Tokens >= vocab_size clamped to UNK (token 0) in data.py and pretrain.py
- Inverted index decode replaces dense matrix multiply (identical results, ~2600x fewer ops)
- Cosine LR schedule with linear warmup added to MiniGPT pretraining
- TinyStories-1M used as ceiling instead of training MiniGPT (pre-trained, no training needed)
- **Tackle issues one-by-one** for publishable attribution: weight stability first, then SDR adaptation

## Planned: exp3 — Weight Stability

### Problem
Hebbian learning rule is purely additive. weight_decay=1.0 is in config but not effective. Need to investigate how it's applied in model.py and fix it.

### Biological grounding
- **Synaptic scaling** (Turrigiano 2008): Neurons multiplicatively scale all incoming synapses to maintain target firing rate → maps to global weight decay
- **Heterosynaptic depression**: When some synapses strengthen (LTP), neighboring inactive ones weaken (LTD). Net synaptic weight roughly conserved → maps to misfire penalty
- **BCM theory**: Threshold for potentiation vs depression slides based on recent activity → dynamic equilibrium

Core principle: **total synaptic weight per neuron should be roughly conserved over time.**

### Two levers
1. **Global weight decay**: Multiplicative decay each step (already a config param, needs to actually work)
2. **Misfire penalty** (`penalty_factor`): Penalize false positives (predicted bits that weren't in actual SDR). Already a config param at 0.0 — needs nonzero value.

### TODO
- Investigate why weight_decay=1.0 isn't preventing explosion
- Determine sensible values for both params
- Run exp3 with fix, compare to exp2 baseline

## Planned: exp4 — SDR Adaptation (Adaptive Encoding + Structural Plasticity)

### Problem
Hash-based SDR encoding produces random bit patterns. Semantically related tokens share no structure (0.9 bits overlap vs 0.78 random expectation). Synapses can't learn associations when the representations have nothing in common.

### Two mechanisms (do in order)

#### 1. Adaptive Encoding (simpler, do first)
When encoding a NEW token for the first time, use eligibility traces of recently active bits to determine ~50% of its bits. Remaining 50% random. This ensures partial overlap with context from the start.
- **Biological analog**: Hebbian priming — new representations are shaped by the context in which they first appear
- **Hyperparameter**: overlap fraction (start at 50%, tune later)
- Stateless operation — just look at eligibility traces when encoding

#### 2. Structural Plasticity (bit swapping)
Track per-token, per-bit statistics:
- **Misfires** (false positives): bit fired in prediction but wasn't in actual SDR → accumulates negative score
- **False negatives**: bit should have fired (was in actual) but wasn't in prediction → accumulates positive score

When an SDR has enough deadwood (high-misfire bits) and high-potential candidates (high false-negative bits), swap them. Run every X steps (e.g., 1000).
- **Biological analog**: Receptor trafficking — AMPA receptor insertion/removal at synapses based on activity
- **Concern**: Bit swaps partially invalidate existing synaptic weights for that token. Mitigate with small swaps (1-2 bits at a time) and gradual change.

#### 3. Sleep Consolidation (deferred, future experiment)
After bit swaps, replay low-IoU sequences involving modified SDRs to let weights adjust. This is a separate, more complex mechanism — good standalone paper contribution. Lightweight version (replay last N occurrences) could accompany structural plasticity if transient accuracy dips are too large.

### Sequencing rationale
- Adaptive encoding first: simpler, no replay, immediate benefit for new tokens
- Structural plasticity second: local operation, tractable without replay
- Sleep consolidation third: complex machinery, best as its own experiment for clean attribution
