# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Research project exploring biologically-plausible learning for next-token prediction. Cortical region model: neocortical minicolumn with L4/L2/3 layers, dendritic segments for prediction, per-neuron feedforward weights. Built with NumPy, Python 3.12+, managed with uv.

## Project Structure

- **`src/step/config.py`** — `CortexConfig`, `HierarchyConfig`
- **`src/step/cortex/`** — models: `region.py`, `sensory.py`, `surprise.py`, `topology.py`
- **`src/step/probes/`** — observation: `diagnostics.py`, `representation.py`, `timeline.py`
- **`src/step/runner.py`** — thin wrappers `run_cortex()`, `run_hierarchy()` delegating to `Topology`
- **`src/step/data.py`** — token loading: `prepare_tokens()`, `prepare_tokens_charlevel()`, `STORY_BOUNDARY`
- **`src/step/viz/`** — dashboard chart builders (`cards.py`, `charts.py`, `layout.py`)
- **`src/step/encoders/`** — `CharbitEncoder`, `OneHotCharEncoder`, `PositionalCharEncoder`
- **`src/step/decoders/`** — `InvertedIndexDecoder`, `SynapticDecoder`, `DendriticDecoder`

## Two-Region Hierarchy
- **S1** (sensory): encoder → 32 cols, k=4, ltd=0.05 (char-level)
- **S2** (secondary): S1's L2/3 firing rate → 16 cols, sliding window receptive fields
- **Feedforward**: `firing_rate_l23` EMA signal S1→S2, with optional temporal buffer + burst gating
  - `buffer_depth=N`: S2 sees sliding window of N recent S1 snapshots (oldest-first), preserving temporal order
  - `burst_gate=True`: zeros precisely-predicted columns, only forwarding novel/surprising events
  - Buffer lives on `Connection`, validated at `connect()` time (input_dim must match)
- **Feedback**: S2 `firing_rate_l23` → S1 apical segments (disabled by default, `enable_apical_feedback=False`)
- **Precision-weighted gating**: when enabled, feedback scaled by S2 confidence `(1 - burst_rate)`
- **Surprise modulation**: S1 burst rate → SurpriseTracker → scales all S2 learning

## Current Encoding: PositionalCharEncoder (256-dim)
- **Char-level tokenization** on BabyLM (32 unique chars: 26 lowercase + space + `!'-?.`)
- **PositionalCharEncoder**: encodes (char_identity, position_in_word) as 8×32 = 256-dim boolean matrix
- Position resets at word boundaries (space, punctuation)
- **Best config**: 32 cols, k=4, ltd=0.05 → 19.9% top-1 (beats 19.7% majority baseline)
- Dashboard supports `--char-level` flag for char tokenization + positional encoding

## Key Decisions
- **Char-level over BPE**: 32 vocab tractable for motor output (BPE gives 1538)
- **Positional encoding wins**: 16.3% top-1 vs 14.8% Charbit vs 9.9% OneHot
- **LTD=0.05 for char-level**: default 0.2 too aggressive
- **S2 needs high LR (0.20)**: S1's EMA has high inter-token cosine similarity (0.48)
- **Apical feedback disabled**: S2 "precise but wrong" — feedback hurts S1
- **Temporal buffer on Connection, not Region**: different connections can have different depths
- **Burst gating before buffering**: each slot captures what was novel at that moment
- **Representation quality > decoder accuracy** — build representations for downstream regions
- **Firing rate > boolean for inter-region** — rate-coded EMA is biologically grounded

## Performance (20k chars, char-level, positional)
- **S1**: burst 46%→14%, overlap 0.38→0.85, ctx_disc 0.932
- **S2 baseline**: ctx_disc 0.737
- **S2 + buffer_depth=4**: ctx_disc 0.899 (+22%)
- **S2 + buffer_depth=4 + burst_gate**: ctx_disc 0.912 (+24%), higher sparsity (0.949), lower cross-col cosine (0.041)
- Burst gating benefit emerges at scale (neutral at 500 tokens with ~50% burst, +1.3% at 20k with ~14% burst)

## Next Steps
- [ ] Revisit apical feedback: S2 ctx_disc now 0.912 with buffer+burst — test if feedback helps S1
- [ ] Motor cortex design: babbling loop (char-by-char output, 32 classes)
- [ ] Consider L5 (motor output) and L6 (thalamic control) layers
