# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Research project comparing biologically-plausible learning (eligibility propagation + sparse distributed representations) against standard backprop in transformers. Built with NumPy (STEP model) and PyTorch (baseline GPT). Python 3.12+, managed with uv.

## Architecture
- **Model core** (`model.py`): Functional with explicit state threading via `ModelState` NamedTuple (dense numpy (n,n) weight matrix). Learning rule: Hebbian reinforcement of source→target connections + anti-Hebbian penalty for false positives. Experimental three-factor gated rule (relevance_gate, weight_init params).
- **Model wrappers** (`step/wrappers.py`, `baselines/wrappers.py`): Wrap implementations into `Model` protocol
  - `StepMemoryModel`: In-memory STEP with inverted index decode (O(k) per query), optional `AdaptiveEncoder`, weight-aware decode
  - `TinyStories1MModel`: Pre-trained HuggingFace model wrapper (inference-only, 3.7M params, 10K vocab restriction, per-story context reset)
- **Key modules**: `sdr.py` (hash-based encoding + `AdaptiveEncoder`), `data.py` (tokenizer + dataset caching + story boundary detection via STORY_BOUNDARY sentinel + vocab filtering)
- **Experiment infra**: `experiment.py` (generic `run_experiment()` + `pretrain_step_model()` with story boundary handling, callback hooks, JSON logging)

## Corrected Accuracy Measurements (with story boundaries)

| Model | Accuracy | Notes |
|-------|----------|-------|
| TinyStories-1M ceiling | **42.2%** | Clamped 10K vocab, per-story context reset |
| Bigram baseline | **28.9%** | |
| STEP w=3 (embedding SDRs, 200K) | **28.9%** | Matches bigrams with optimal SDRs |
| STEP w=3 (hash, 200K) | **21.7%** | Below bigrams with hash encoding |

Previous 30% measurements were inflated by cross-story context bleeding. Fixed in commit 6dd4d5d.

## Key Findings (exp3)

### 1. Encoding is NOT the bottleneck
Embedding-derived SDRs (from TinyStories-1M word embeddings) match bigrams at w=3 (28.9%) but w=5 still degrades (28.4%). Even with optimal SDRs, the learning rule can't use longer context.

### 2. Learning rule is the bottleneck
The rule treats all source bits in the eligibility window equally (modulo time decay). With w>3, noise from irrelevant context drowns the signal. No mechanism to focus on relevant context vs noise.

### 3. Per-bit gating doesn't work
Three-factor gated learning (gate source bits by W[i,:] relevance to target) hurts at all thresholds. Root cause: each W[i,:] is a superposition of ~195 tokens' prediction targets, so per-bit relevance to one specific target is low even for genuinely relevant source bits. Positive weight initialization doesn't fix this.

### 4. Token-level credit assignment needed
The model needs to learn which CONTEXT TOKENS (not bits) are relevant for each prediction. This is fundamentally what attention solves in transformers. Biologically-plausible approaches: neuromodulatory gating (dopamine), competitive inhibition, or separate fast/slow pathways.

## Adaptive Encoding Results (superseded by learning rule work)
- `AdaptiveEncoder` in `sdr.py`: seeds context_fraction of bits from context
- Active f=0.3 makes w=5 viable (doesn't degrade), but doesn't beat bigrams
- Predict f=0.5: high IoU but accuracy collapses (discrimination failure — 1767 confusable tokens)
- Weight-aware decode, IoU lift metric implemented
- Decode diagnostics: hash errors = prediction failure (IoU 0.08), predict f=0.5 errors = discrimination failure (IoU 0.46)

## Open Questions / Next Steps
- **Token-level gating**: Gate entire context tokens instead of individual bits. Compute per-token contribution to the prediction, gate based on that.
- **Biologically-plausible attention**: Can competitive inhibition among context tokens implement attention? Different cortical layers for different timescales?
- **Three-factor learning at token level**: The user's insight that activation (weight × trace) should determine IF plasticity happens, while plasticity magnitude should be uniform, hasn't been fully explored at the token level.
- **Structural plasticity**: Moving SDR bits based on prediction errors. Discussed but deprioritized since encoding isn't the bottleneck.

## Key Decisions
- **Story boundaries**: STORY_BOUNDARY sentinel (-1) between stories. All models reset context.
- **10K vocab ceiling**: TinyStories1MModel restricts argmax to vocab_size
- Inverted index decode with weight-aware scoring
- **IoU is primary STEP-internal metric** (accuracy for external comparison)
- **Tackle issues one-by-one** for publishable attribution
