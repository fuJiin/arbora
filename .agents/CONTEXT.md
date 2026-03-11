# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Research project exploring biologically-plausible learning for next-token prediction. Two architectures: (1) original flat STEP model (dense weight matrix + Hebbian learning), (2) new cortical region model (neocortical minicolumn architecture with L4/L2/3 layers). Built with NumPy, Python 3.12+, managed with uv.

## Current Work

### Branch: `cortext` (not yet pushed to remote)
5 commits on top of `main`:
- Add CharbitEncoder, promote RandomEncoder to class, rename SDR -> encoding
- Migrate all imports from step.sdr to step.encoders, remove encode_token()
- Add cortical region abstraction with L4/L2/3 layers
- Add L2/3 lateral connections for associative binding
- Reorganize tests into subdirectories mirroring source layout

### Next: Cortex PoC (in progress)
Building proof-of-concept to compare cortex model against original STEP on TinyStories. Design decisions made:

**Architecture (first-principles, not shimming into old protocol):**
- Cortex gets its own training loop — no fake predict/learn/observe cycle
- Prediction = pre-activation voltage from feedback/lateral signals BEFORE next input
- Measurement = overlap between pre-activated neurons and actually-activated neurons
- Shared result types so both models produce comparable metrics
- `DecodeIndex` extracted as shared utility for token decode

**Encoding:** CharbitEncoder for cortex (character-level binary, structured 2D), RandomEncoder for original STEP baseline comparison.

**PoC parameters (matched ~72K params to best existing STEP at ~65K):**
- CharbitEncoder: length=8, width=96 (95 printable ASCII + unknown), input_dim=768
- Cortex: n_columns=32, n_l4=4, n_l23=4, k_columns=4
- Baseline: existing STEP at n=256, k=10 (best performer from prior experiments)
- Start with 1K tokens, then scale

**Story boundaries:** Reset voltages/traces/excitability (working memory), keep weights (long-term memory). Biologically: "clear your mind" between stories while retaining learned knowledge. Supports continual learning narrative.

**Neuron bursting (future):** When no neuron in a column activates confidently, burst all neurons as a surprise signal. For PoC: argmax forces 1 winner per column (sufficient for MVP). Bursting deferred — will eventually use activation thresholds instead of argmax, with bursting to signal surprise to downstream regions and modulate learning.

## Cortex Architecture (`src/step/cortex/`)

Two-layer neocortical minicolumn model:
- **L4 (input):** receives feedforward drive, modulated by feedback context
- **L2/3 (associative):** receives L4 feedforward + lateral context, enables associative binding

Activation: top-k columns by strongest L4 score -> 1 winning L4 neuron per column -> 1 winning L2/3 neuron per column (L4 bias + lateral context + excitability).

Key files:
- `cortex/region.py`: CorticalRegion base class (all dynamics)
- `cortex/sensory.py`: SensoryRegion (adds feedforward weights from encoding)
- `cortex/__init__.py`: exports both

### Performance concerns (identified, not yet fixed)
- Dense weight matrices scale O(n^2) — needs sparse updates for >128 neurons per layer
- `_learn()` does full outer product every step — should index into active rows/cols only
- `_apply_feedback()` does full matvec — should index into active source neurons only
- Synapse decay touches all weights — should skip zeros
- All fine at PoC scale (128 neurons), blocking at production scale

## Encoder Refactoring (completed)

- `step/encoders/` package replaces old `step/sdr.py`
- `CharbitEncoder`: char-level binary (length, width) boolean matrix, 1-hot per position
- `RandomEncoder`: promoted from bare function to class, hash-based deterministic encoding
- `AdaptiveEncoder`: context-seeded encoding (moved from sdr.py)
- Terminology: "SDR" -> "encoding" throughout

## Original STEP Model (baseline)

- `model.py`: functional style, ModelState NamedTuple, dense (n,n) weight matrix
- `wrappers.py`: StepMemoryModel with inverted index decode
- `protocol.py`: Model protocol (predict_token, predict_sdr, learn, observe)

### Baseline Results

| Model | Accuracy | Notes |
|-------|----------|-------|
| TinyStories-1M ceiling | **42.2%** | 3.7M params, 10K vocab |
| Bigram baseline | **28.9%** | |
| STEP w=3 (embedding SDRs, 200K) | **28.9%** | Matches bigrams |
| STEP w=3 (hash, 200K) | **21.7%** | Below bigrams |

**Inverse scaling:** n=256,k=10 gets 3.3x better IoU than n=2048,k=40. Smaller models win because the learning rule can't do credit assignment at scale.

### Key finding: learning rule is the bottleneck
Encoding is not the problem. The Hebbian rule treats all source bits equally (modulo time decay) — can't focus on relevant vs irrelevant context. Token-level credit assignment is needed, which is what the cortex architecture aims to provide via structured columns and competitive inhibition.

## Key Decisions
- **Story boundaries**: reset working memory, keep synaptic weights
- **Tackle issues one-by-one** for publishable attribution
- **Goal**: prove online continual learning without catastrophic forgetting, then match transformer accuracy
- **Argmax winner selection** for PoC, activation thresholds + bursting for later
