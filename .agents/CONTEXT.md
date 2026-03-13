# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Research project exploring biologically-plausible learning for next-token prediction. Cortical region model: neocortical minicolumn with L4/L2/3 layers, dendritic segments for prediction, per-neuron feedforward weights. Built with NumPy, Python 3.12+, managed with uv.

## Project Structure

- **`src/step/config.py`** — `CortexConfig`, `HierarchyConfig`
- **`src/step/cortex/`** — models: `region.py`, `sensory.py`, `surprise.py`, `topology.py`
- **`src/step/probes/`** — observation: `diagnostics.py`, `representation.py`, `timeline.py`
- **`src/step/runner.py`** — thin wrappers `run_cortex()`, `run_hierarchy()` delegating to `Topology`
- **`src/step/data.py`** — token loading: `prepare_tokens()`, `prepare_tokens_charlevel()`, `STORY_BOUNDARY`
- **`src/step/runs.py`** — run serialization: `save_run`/`load_run`/`list_runs`/`auto_name`
- **`src/step/viz/`** — dashboard chart builders (`cards.py`, `charts.py`, `layout.py`, `build_index_html`)
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

## Dashboard Architecture
- **`cortex_run.py`**: runs cortex, saves `data.pkl` + `meta.json` to `experiments/runs/{name}--{timestamp}/`
- **`cortex_dashboard.py`**: generates HTML from saved runs (`--latest`, `--all`, `--run DIR`, `--index-only`), serves with `--serve --port N`
- **Index page**: `build_index_html()` lists all runs with metrics, tags, links
- Legacy inline mode still works: `--tokens N --char-level --hierarchy`

## Architecture: Thalamic Gating + Motor Cortex

### Thalamic Gate (build first)
- **Problem**: feedback connections (S2→S1, future M1→S1) disrupt receiver early when sender is still learning garbage
- **Two complementary mechanisms**:
  1. **Sender precision** (already have): feedback scaled by sender's `(1 - burst_rate)`
  2. **Receiver gating** (new): receiver's smoothed surprise tells thalamus to suppress incoming feedback
- **Formula**: `effective_feedback = signal * sender_confidence * receiver_readiness`
- **Biology**: pulvinar (higher-order thalamic nucleus) modulated by L6 projections from receiving cortex
- Per-connection, lives on Connection infrastructure — apply to S2→S1 immediately to validate

### Motor Cortex (M1)
- **L2/3 (always on)**: receives S1/S2 L2/3, predicts next char via dendritic segments. Learns sensory→motor mapping continuously.
- **L5 (gated output)**: actual "speak this char" signal. Default: inhibited. Fires when prediction confidence > threshold.
- **Babbling→speech arc**: early on confidence low → mostly silent; as learning proceeds → increasingly commits to outputs
- **32 output classes** (chars), receives S1 L2/3 as input
- Basal ganglia deferred — confidence threshold captures disinhibition for now

### Sequencing
1. ThalamicGate on feedback connections, validate on S2→S1
2. Motor region M1: L2/3 predicts, L5 gated output
3. M1→S1 feedback through thalamic gate (auto-suppressed early)

## Next Steps
- [ ] Implement ThalamicGate (receiver surprise-based gating on feedback connections)
- [ ] Validate on S2→S1: compare with current precision-only gating
- [ ] Motor cortex M1: L2/3 prediction + L5 gated output
