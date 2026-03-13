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
- **Temporal buffer on Connection, not Region**: different connections can have different depths
- **Burst gating before buffering**: each slot captures what was novel at that moment
- **Apical feedback works with buffer+burst**: previously S2 was "precise but wrong", now S2 ctx_disc 0.947 and apical boosts S1 ctx_disc 0.657→0.890
- **Apical tradeoff**: S1 gains ctx_disc but loses selectivity (0.580→0.684) — columns become more context-dependent, less token-specific. Acceptable for feeding motor cortex.
- **Dendritic decoder must use active_l23 (boolean)**: firing_rate_l23 EMA is 128/128 nonzero due to decay, making `> 0` threshold useless for segment discrimination
- **Dendritic decoder tuned to 16seg/48syn**: sweep over {4,8,16,32}×{24,48,96} — 16 segments is sweet spot, 32 dilutes learning. 48 syn covers ~37% of 128-dim L2/3 per segment.
- **Firing rate > boolean for inter-region** — rate-coded EMA is biologically grounded

## Performance (20k chars, char-level, positional, buffer+burst+apical)
- **S1**: burst 33.7%, ctx_disc 0.890, overlap ~0.46
- **S2**: ctx_disc 0.947
- **Dendritic decoder** (16seg/48syn): 13.8% last500, 16% last100 (chance=3.2%, still climbing)
- Other decoders: index 5%, column 6%, synaptic 4%

## Dashboard CLI
- `--hierarchy --char-level --buffer-depth 4 --burst-gate --apical` — full pipeline

## Next Steps
- [ ] Motor cortex design: babbling loop (char-by-char output, 32 classes)
- [ ] Consider L5 (motor output) and L6 (thalamic control) layers
