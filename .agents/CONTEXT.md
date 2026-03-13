# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Research project exploring biologically-plausible learning for next-token prediction. Cortical region model: neocortical minicolumn with L4/L2/3 layers, dendritic segments for prediction, per-neuron feedforward weights. Built with NumPy, Python 3.12+, managed with uv.

## Project Structure

- **`src/step/config.py`** — `CortexConfig`, `HierarchyConfig`, `_default_motor_config()`
- **`src/step/cortex/`** — models: `region.py`, `sensory.py`, `motor.py`, `surprise.py`, `topology.py`
- **`src/step/probes/`** — observation: `diagnostics.py`, `representation.py`, `timeline.py`
- **`src/step/runner.py`** — thin wrappers `run_cortex()`, `run_hierarchy()` delegating to `Topology`
- **`src/step/data.py`** — token loading: `prepare_tokens()`, `prepare_tokens_charlevel()`, `STORY_BOUNDARY`
- **`src/step/runs.py`** — run serialization: `save_run`/`load_run`/`list_runs`/`auto_name`
- **`src/step/viz/`** — dashboard chart builders (`cards.py`, `charts.py`, `layout.py`, `build_index_html`)
- **`src/step/encoders/`** — `CharbitEncoder`, `OneHotCharEncoder`, `PositionalCharEncoder`
- **`src/step/decoders/`** — `InvertedIndexDecoder`, `SynapticDecoder`, `DendriticDecoder`

## Three-Region Architecture

### S1 (Sensory)
- Encoder → 32 cols, k=4, ltd=0.05 (char-level with PositionalCharEncoder, 256-dim)

### S2 (Secondary Sensory)
- S1's L2/3 firing rate → 32 cols, sliding window receptive fields
- Feedforward: `firing_rate_l23` EMA signal S1→S2, with temporal buffer + burst gating
- Feedback: S2→S1 apical segments (precision-weighted + thalamic-gated)
- Surprise modulation: S1 burst rate → SurpriseTracker → scales S2 learning

### M1 (Motor Cortex) — NEW
- **`MotorRegion`** subclasses `SensoryRegion` (encoding_width=0), adds L5 output gating
- Receives S1's L2/3, learns context→token mapping via same Hebbian/dendritic algorithm
- **L5 readout**: per-column mean L2/3 firing rate, thresholded (default 0.3) for output
- **Self-organizing column→token map**: columns track which token activates them most
- **k=1 (winner-take-all)**: parameter sweep showed k=1 → 60% accuracy, k=4 → 26%. Architecture dominates tuning.
- Wiring: S1→M1 feedforward, S1→M1 surprise, M1→S1 apical (thalamic-gated)

### ThalamicGate — NEW
- **Receiver-side gating** on feedback connections (S2→S1, M1→S1)
- `readiness = 1.0 - smoothed_burst_rate` (EMA, decay=0.95, starts closed)
- `effective_feedback = signal * sender_confidence * receiver_readiness`
- Resets at story boundaries; opens within ~30-50 tokens as receiver stabilizes
- At current scale, gate doesn't measurably help/hurt S2→S1 feedback (signal too weak), but infrastructure ready for M1 where feedback is stronger

## Key Decisions
- **k=1 for M1**: sweep of k={1,2,4} × {lr, ltd, voltage_decay, threshold} (13+14 configs). k=1 gives 60% accuracy, robust to other params. k=2 gives 46%, k=4 gives 26%.
- **Motor params insensitive**: with k=1, lr={0.10-0.30}, ltd={0.10-0.30}, voltage_decay={0.3-0.7} all give 58-61%. Architecture > tuning.
- **output_threshold=0.3 correct**: score distribution shows per-step max almost always >0.3. Threshold filters bad guesses, not good ones.
- **Thalamic gate starts closed, opens fast**: half-life ~14 steps. At 5k tokens, gate at 0.63-0.90. Dips at story boundaries.
- **M1 feedback affects S1**: S1 burst rate higher early (48% vs 23% without M1) but columns become more context-dependent

## Performance (5k chars, char-level, positional, buffer+burst+apical+gate+motor, k=1)
- **S1**: burst ~42%, ctx_disc 0.888, overlap ~0.40, dendritic ~18%
- **S2**: ctx_disc 0.944
- **M1**: 53-64% accuracy (rolling 100), 17-33% silence, selectivity 0.536

## Dashboard
- **`cortex_run.py`**: `--motor` and `--gate-feedback` flags added
- **`cortex_dashboard.py`**: Motor tab (accuracy/silence chart, column activation, selectivity, usage, segment health). M1 Accuracy + M1 Selectivity score cards at top. Thalamic Gate chart in Feedback tab.
- Motor accuracy chart: dual panel (rolling accuracy when speaking + silence rate)
- Thalamic gate chart: readiness time series with raw + rolling average

## Next Steps
- [ ] Longer motor runs (20k+) to see if M1 accuracy keeps climbing
- [ ] PFC / reasoning region (receives S2, plans multi-step)
- [ ] M1 feedback impact analysis: compare S1 representation quality with/without M1
- [ ] Consider developmental curriculum: start M1 silent, gradually lower threshold
