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

## exp2c Results — STEP matches ceiling (completed)
Branch: `worktree-exp0` — Short eligibility window (w=3) + weight decay/penalty

| Model | Params | Final Rolling Accuracy | Config |
|-------|--------|----------------------|-------|
| TinyStories-1M (ceiling) | 3.7M | 30.00% | pre-trained, inference only |
| **step_memory (exp2c)** | 4.2M (n=2048) | **30.00%** | w=3, decay=0.999, penalty=0.5 |
| step_memory (exp2) | 4.2M (n=2048) | 3.00% | w=101, decay=1.0, penalty=0.0 |

**Key finding**: Reducing eligibility_window from 101→3 was the breakthrough. Long window floods the (n,n) weight matrix with noise (80K updates/step across all source→target pairs). Short window concentrates learning on bigram associations where the actual predictive signal lives. 200K pretrain in 110s vs 2+ hours.

### exp2b intermediate result
Enabling weight_decay=0.999 + penalty_factor=0.5 with w=101: weights stabilize (mean ~5.9 vs 1687) but accuracy unchanged at ~2%. Matrix becomes uniformly dense — bounded but no discriminative signal. Weight stability is necessary but not sufficient without the short window.

### exp2c weight dynamics
Weights stabilize by ~100K steps: mean=1.35, max~10, 60% > 1.0. Healthy equilibrium — not exploding, not collapsing.

### exp2 diagnostic findings (still relevant for SDR work)
- **SDR encoding has zero structure**: Bigram SDR overlap averages 0.9 bits (random expectation k²/n = 0.78). Hash-based encoding provides no similarity between related tokens.
- **Note on ceiling quality**: CE loss 4.70 (pplx 110) vs published 1.71 (pplx 5.5) due to 10K vocab clamping on 50K-trained model. Valid as relative ceiling.

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
- **Tackle issues one-by-one** for publishable attribution
- **eligibility_window=3** is the correct operating point for now (bigram model). Longer windows can be revisited after SDR encoding has structure.

## Critical finding: STEP with w=3 is a bigram model

Window sweep (w=3,5,10,20) on 50K pretrain + 5K eval:
| Window | Mean Accuracy |
|--------|--------------|
| 3 | 27.1% |
| 5 | 26.6% |
| 10 | 21.8% |
| 20 | 13.7% |
| **Pure bigram baseline** | **28.8%** |

STEP with w=3 ≈ bigram frequency lookup. Longer windows strictly hurt. Hash-based SDRs (zero overlap between tokens) mean every additional context token adds noise, not signal. The eligibility window mechanism works — it just needs representations that make longer contexts useful.

## Next: exp3 — Adaptive Encoding (beat bigrams)

**Goal**: Show STEP can generalize beyond bigrams. Success metric: window=5-10 helps instead of hurting, accuracy > 28.8% (bigram ceiling).

**The bottleneck is SDR encoding.** If related tokens shared bits, longer windows would let the model learn patterns like "after a determiner, predict a noun" — actual generalization. Right now each token's bits are random.

**Dataset strategy**: TinyStories is fine for proving the point — it has pronoun resolution, verb agreement, story templates that all require context > 1 token. For a paper: also PTB (classic, well-studied bigram baselines), WikiText-2 (standard), BabyLM (data-efficient, fits bio-plausible narrative).

### SDR Adaptation (Adaptive Encoding + Structural Plasticity)
The SDR encoding has zero structure. Fixing this enables longer windows (the original vision). Two mechanisms planned:

#### 1. Adaptive Encoding (simpler, do first)
When encoding a NEW token for the first time, use eligibility traces of recently active bits to determine ~50% of its bits. Remaining 50% random.
- **Biological analog**: Hebbian priming — new representations are shaped by context
- Stateless operation — just look at eligibility traces when encoding

#### 2. Structural Plasticity (bit swapping)
Track per-token, per-bit misfires/false-negatives. When an SDR has enough deadwood, swap bits.
- **Biological analog**: Receptor trafficking — AMPA receptor insertion/removal
- **Concern**: Bit swaps partially invalidate existing synaptic weights

#### 3. Sleep Consolidation (deferred)
Replay low-IoU sequences after bit swaps. Separate, more complex mechanism — good standalone paper contribution.
