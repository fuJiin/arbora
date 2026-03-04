# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Research project comparing biologically-plausible learning (eligibility propagation + sparse distributed representations) against standard backprop in transformers. Built with NumPy (STEP model) and PyTorch (baseline GPT). Python 3.12+, managed with uv.

## Architecture
- **Model core** (`model.py`): Functional with explicit state threading via `ModelState` NamedTuple (dense numpy (n,n) weight matrix). Learning rule: Hebbian reinforcement of source→target connections + anti-Hebbian penalty for false positives.
- **Model wrappers** (`step/wrappers.py`, `baselines/wrappers.py`): Wrap implementations into `Model` protocol
  - `StepMemoryModel`: In-memory STEP with inverted index decode (O(k) per query), optional `AdaptiveEncoder`
  - `TinyStories1MModel`: Pre-trained HuggingFace model wrapper (inference-only, 3.7M params, 10K vocab restriction)
- **Key modules**: `sdr.py` (hash-based encoding + `AdaptiveEncoder`), `data.py` (tokenizer + dataset caching + story boundary detection + vocab filtering)
- **Experiment infra**: `experiment.py` (generic `run_experiment()` + `pretrain_step_model()` with story boundary handling, callback hooks, JSON logging)

## Corrected Accuracy Measurements (with story boundaries)

| Model | Accuracy | Notes |
|-------|----------|-------|
| TinyStories-1M ceiling | **42.2%** | Clamped 10K vocab, per-story context reset |
| Bigram baseline | **28.9%** | |
| STEP w=3 (hash, 200K) | **21.7%** | Below bigrams — old 30% was inflated by cross-story contamination |
| STEP w=3 (embedding SDRs, 200K) | **28.9%** | Matches bigrams with optimal SDRs |

Previous 30% measurements were inflated by cross-story context bleeding. Fixed in commit 6dd4d5d.

## Key Finding: Learning Rule is the Bottleneck

**Encoding is NOT the bottleneck.** Embedding-derived SDRs (from TinyStories-1M word embeddings with pre-trained semantic structure) match bigrams at w=3 but w=5 still degrades. Even with optimal SDRs, the learning rule can't use longer context.

**Root cause**: The learning rule treats all source bits in the eligibility window equally (modulo time decay). With w>3, noise from irrelevant context tokens drowns the signal. The model has no mechanism to focus on relevant context vs noise — which is exactly what attention solves in transformers.

## Current Work: Three-Factor Gated Learning Rule
Branch: `worktree-exp0`

### Motivation
Current rule: `W[src, dst] += eta * trace_strength` for ALL source bits in window.
Proposed: gate eligibility by relevance — only learn from source bits that actually contributed to the prediction.

### Design (biologically-plausible three-factor rule)
1. **Activation** = `W[i, :] * trace_strength[i]` (synapse weight × pre-synaptic trace)
2. **Relevance gate** = how well this source bit's activation matches the actual target (neuromodulatory signal)
3. **Learning** = flat `eta` for everyone who passes the gate

Maps to neuroscience: eligibility traces exist on all synapses, but only convert to actual plasticity when a neuromodulatory signal (dopamine/prediction error) confirms relevance. This is a "three-factor learning rule" in computational neuroscience: pre × post × neuromodulator.

Key insight from user: the current rule incorrectly uses trace_strength for both activation AND learning magnitude. In biology, activation (weight × trace) determines IF plasticity happens, while the plasticity magnitude is relatively uniform (LTP/LTD has characteristic magnitudes).

## exp3 Results — Adaptive Encoding (completed, superseded)

Adaptive encoding (seeding SDR bits from context) adds marginal IoU lift but doesn't break the bigram ceiling. Decode diagnostics separated prediction failure (hash) from discrimination failure (predict f=0.5). Weight-aware decode implemented but didn't fix the core issue. All this work confirmed encoding isn't the bottleneck.

Key files: `sdr.py` (AdaptiveEncoder), `sweep_final.py` (IoU lift), `diagnose_decode.py`, `sweep_embedding_sdr.py`

## Previous Experiments
- **exp2c**: STEP matches TinyStories-1M at 30% with w=3 (now known to be inflated)
- **exp2b**: Weight decay/penalty necessary but not sufficient
- **exp1**: 10K vocab, 2M tokens — both models stuck at 3%
- **exp0**: Initial comparison, both at 6%

## Key Decisions
- **Story boundaries**: STORY_BOUNDARY sentinel (-1) between stories in token_stream. All models reset context at boundaries.
- **10K vocab ceiling**: TinyStories1MModel restricts argmax to vocab_size (10K) since targets are clamped
- Inverted index decode with weight-aware scoring
- **Learning rule is the focus** — encoding experiments completed, bottleneck identified
