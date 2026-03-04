# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Research project comparing biologically-plausible learning (eligibility propagation + sparse distributed representations) against standard backprop in transformers. Built with NumPy (STEP model) and PyTorch (baseline GPT). Python 3.12+, managed with uv.

## Architecture
- **Dual model implementations**:
  - `model.py`: Functional core with explicit state threading via `ModelState` NamedTuple (dense numpy (n,n) weight matrix)
  - `db.py`: SQLite-backed `StepModel` class — dropped from exp0 due to O(80K) SQL upserts/token. Future: hybrid numpy compute + SQLite checkpoint.
- **Model wrappers** (`step/wrappers.py`, `baselines/wrappers.py`): Wrap implementations into `Model` protocol
  - `StepMemoryModel`: In-memory STEP with inverted index decode (O(k) per query), optional `AdaptiveEncoder`
  - `MiniGPTModel`: Inference-only GPT wrapper with rolling context buffer
  - `TinyStories1MModel`: Pre-trained HuggingFace model wrapper (inference-only, 3.7M params)
- **Model protocol** (`protocol.py`): `Model` protocol for running STEP and baselines through the same harness
- **Key modules**: `sdr.py` (hash-based encoding + `AdaptiveEncoder`), `metrics.py` (IoU, rolling mean), `data.py` (tokenizer + dataset caching + vocab filtering + .env HF token)
- **Experiment infra**: `experiment.py` (generic `run_experiment()` + `pretrain_step_model()` with callback hooks, JSON logging, adaptive SDR support via `encode_token_sdr`)
- **Diagnostics** (`step/diagnostics.py`): `DiagnosticCollector` for weight stats, prediction logging, bigram SDR overlap, per-position accuracy
- **Baselines**: `mini_gpt.py` (2-layer transformer), `pretrain.py` (training loop with cosine LR schedule), `wrappers.py`

## exp3 Results — Adaptive Encoding (in progress)
Branch: `worktree-exp0`

### Implementation
- `AdaptiveEncoder` class in `sdr.py`: seeds context_fraction of new token bits from context, rest random per token_id
- Two seeding modes: `"active"` (eligibility window bits) and `"predicted"` (model's prediction at encode time)
- `EncoderConfig` extended: `adaptive: bool`, `context_fraction: float`, `seeding: str`
- Experiment loop calls `model.encode_token_sdr(token_id, t)` when available

### Seeding sweep results (50K pretrain, 5K eval)

| Seeding | w=3 Acc | w=3 IoU | w=5 Acc | w=5 IoU | w=10 Acc | w=10 IoU |
|---------|---------|---------|---------|---------|----------|----------|
| hash | **30.0%** | 0.2650 | 28.0% | 0.2452 | 27.0% | 0.2082 |
| active f=0.3 | 29.0% | 0.2692 | **29.0%** | 0.2603 | 22.0% | 0.2260 |
| predict f=0.3 | 28.0% | 0.2908 | 22.0% | 0.2722 | 19.0% | 0.2652 |
| predict f=0.5 | 8.0% | 0.4840 | 7.0% | **0.4873** | 7.0% | 0.4740 |

### Key findings

1. **Active f=0.3: w=5 doesn't degrade.** Accuracy holds at 29% for both w=3 and w=5 (hash drops 30→28%). First evidence that longer context can be used without harm. IoU gap also halves (0.0089 vs 0.0198).

2. **Predicted f=0.5: w=5 IoU > w=3 IoU.** (0.4873 vs 0.4840) — first time longer window actually HELPS on IoU. But accuracy collapses to 7-8% because shared bits destroy decode discrimination.

3. **Two separate problems identified:**
   - **Representation structure** (IoU) — adaptive encoding helps, prediction-based helps even more
   - **Decode discrimination** (accuracy) — inverted index overlap count can't distinguish tokens that share context bits

### What this means
The adaptive encoding IS creating useful SDR structure — the model learns better predictions (higher IoU) and longer windows become less harmful. But the inverted index decode (simple overlap count) can't turn structured predictions into correct tokens when tokens share bits.

### Potential next steps (not yet started)
1. **Weight-aware decode**: Score tokens by sum of weight magnitudes from predicted bits, not just overlap count. Higher architectural complexity but directly addresses discrimination.
2. **Further tune active f=0.3**: May already be enough with more training data (200K pretrain) or window tuning.
3. **Structural plasticity**: Post-hoc bit adjustment based on prediction errors — different angle on same problem.

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

## Critical finding: STEP with w=3 is a bigram model

Window sweep (w=3,5,10,20) on 50K pretrain + 5K eval:
| Window | Mean Accuracy |
|--------|--------------|
| 3 | 27.1% |
| 5 | 26.6% |
| 10 | 21.8% |
| 20 | 13.7% |
| **Pure bigram baseline** | **28.8%** |

STEP with w=3 ≈ bigram frequency lookup. Longer windows strictly hurt. Hash-based SDRs (zero overlap between tokens) mean every additional context token adds noise, not signal.

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
- **IoU is the primary STEP-internal metric** — accuracy matters for external comparison but IoU better reflects representational learning quality
- **eligibility_window=3** is the correct operating point for hash-based encoding. Active-seeded adaptive encoding at f=0.3 makes w=5 viable.
