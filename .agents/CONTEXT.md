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
- **Feedforward**: `firing_rate_l23` EMA signal S1→S2
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
- **Char-level over BPE**: BPE gives 1538 vocab (too many for 128-dim L2/3). Char-level gives 32 vocab, tractable for motor output.
- **Positional encoding wins**: 16.3% top-1 vs 14.8% (Charbit 808-dim) vs 9.9% (OneHot 32-dim). Position-in-word info helps.
- **LTD=0.05 for char-level**: Default 0.2 too aggressive. Sweep showed 0.05 > 0.10 > 0.20.
- **Apical feedback disabled by default**: S2 is "precise but wrong" (low burst but bad predictions), so feedback hurts S1. Precision weighting `(1 - burst_rate)` doesn't gate enough. Needs S2 to mature first.
- **S2 needs high LR (0.20)**: S1's firing_rate_l23 EMA has high cosine similarity between tokens (mean 0.48). S2 needs 20x default LR to amplify subtle differences.
- **Representation quality over decoder accuracy** — sensory cortex builds representations for downstream regions
- **Motor cortex will generate char-by-char** — 32 possible outputs makes motor learning tractable
- **Firing rate > boolean for inter-region** — rate-coded EMA is biologically grounded
- **Decodability before motor cortex** — validate representations are actionable before investing in output architecture

## Dashboard
- Single-region tabbed (Activity/Representations/Segments), `--hierarchy` for dual-region (Overview/S1/S2/Feedback)
- Config banner shows encoder, region params, LTD, token count
- Stat cards with health-based color coding
- `--char-level` flag switches to PositionalCharEncoder + char tokenization + ltd=0.05

## Hierarchy Performance (20k chars, positional, ltd=0.05)
- S1 burst: 46% → 14% (excellent learning)
- S1 overlap: 0.38 → 0.75-0.85
- S1 context discrimination: 0.932
- S2 context discrimination: 0.950 (improves on S1)
- Zero dead columns

## Next Steps
- [ ] Motor cortex design: babbling loop (char-by-char output, 32 classes)
- [ ] Investigate throughput: char-level is ~4-5x more steps per text unit than BPE
- [ ] Consider L5 (motor output) and L6 (thalamic control) layers
- [ ] Revisit apical feedback once S2 representations improve
