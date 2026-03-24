# Context: STEP (Sparse Temporal Eligibility Propagation)

## Overview
Biologically-plausible cortical learning. Minicolumn architecture, Hebbian + three-factor RL, no backprop. Full sensory-motor hierarchy with PFC goal maintenance.

## Integrations
- **Linear Project:** PZO / STEP (Sparse Temporal Eligibility Propagation)

## Architecture (DAG-validated, finalize() enforced)
```
Topo order: S1 → S2 → S3 → PFC → M2 → M1

Feedforward (source-aware sparsity on PFC/M2):
  S1→S2 (buf=4), S2→S3 (buf=8)
  S2+S3→PFC (40% sparse), S2+PFC→M2 (40% sparse)
  M2→M1

Apical (multi-source, per-source gain weights):
  S3→S2, S2→S1, M1→M2, M2→PFC, S1→M1, M1→S1

Surprise: S1→S2, S2→S3, S1→M1
```

## Learning: STDP Presynaptic Traces (DEFAULT ON)

`pre_trace_decay=0.8` in CortexConfig, all regions from construction.
- FF traces + segment traces for plasticity, prediction stays boolean
- Three-factor (PFC, M1): pre_trace feeds eligibility → reward

## Echo Status

### 2k episode run (traces default, 50k sensory warmup)
- 0-500: 7.3%, 500-1000: 6.7%, 1000-1500: 6.8%, 1500-2000: **8.0%**
- Stable, slight upward trend, no oscillation
- **Problem**: M1 converges on 'h' during echo (not during babble)
- 'h' gets partial credit from RPE because it appears in many common words
- This is a reward signal problem, not a motor/representation problem
- Babble output is diverse (17 unique chars, dominated by 'g'/'?'/space)

### Echo reward issue
RPE partial credit gives 0.2 for char-anywhere-in-word. 'h' appears in "the", "that", "what", "here", "have", etc. — consistent positive signal regardless of target. M1 learns "'h' is always somewhat right" and stops exploring.

Fix options:
1. Remove partial credit entirely (back to position-only matching)
2. Make partial credit position-dependent only (no anywhere-in-word)
3. Add exploration bonus / curiosity to echo reward
4. Cerebellar error signal: "you produced 'h' but S1 expected 't'" — per-step corrective, not reward-based

## Validated Results
- STDP traces from construction: 7.3% echo (best config)
- Structural sparsity: 38% echo improvement
- PFC three-factor: 3.1% → 8.2% echo
- 300k trace sensory: decoder BPC 3.63

## Uncommitted
- `.github/workflows/ci.yml` — needs workflow OAuth scope

## Architecture Decisions (from braindump review 2026-03-23)

### Layers
- **L5 must be a proper layer in all regions** — not just motor. L5 is where cerebellum connects (climbing fiber error), BG gets cortical input (corticostriatal projections), and top-down apical feedback lands (BAC firing). Do before cerebellum.
- **Column structure in all layers** — L4 strict (receptive field bound), L2/3 looser (wider lateral), L5 output-organized (subcortical projection). Column is the fundamental unit across all three layers.
- **Keep laminar names (l4, l23, l5)** — abstract names (input/associative/output) collide with existing concepts (input_dim, output_weights). Docstrings map the functional roles.

### Connections
- **Apical: linear gain now, dendritic segments on L5 later** — linear gain is adequate short-term but misses context-dependent gating. When L5 is added, apical segments on L5 apical dendrites.
- **Connection modulators are properties, not types** — surprise/reward fold into optional properties on feedforward/apical/lateral connections. Supersedes separate "surprise" and "reward" connection kinds.
- **fb_segments model L6→L4, not L2/3→L4** — direct L2/3→L4 isn't a major biological pathway. Current impl is a stopgap; will become L5→L4 recurrent when L5 is added.

### Learning
- **STDP pre-traces are universal, consolidation varies** — two orthogonal axes: pre_trace_decay (temporal credit, all regions) and consolidation rule (immediate for sensory, reward-gated for PFC/Motor).
- **Structural plasticity is optimization, not critical path** — defer to tuning phase.

### Subcortical
- **Cerebellum = separate forward model module, BG = gating** — cerebellum predicts sensory consequence of motor command; BG evaluates and gates. Cerebellum provides per-step corrective error, BG provides go/no-go.
- **Hippocampus: not yet** — park until PFC capacity limits are hit during multi-turn dialogue. PFC + cerebellum + BG is the right next set.

### Regions
- **Region types by laminar profile**: granular sensory (S1), association (S2/S3), agranular frontal (PFC), motor (M1/M2)

### KPIs
- **Primary: surprise rate** — most biologically grounded, directly measures prediction quality
- **Secondary: representation stability** — critical for downstream learning; always report when evaluating new configs
- **Tertiary: decoder BPC** — useful for literature comparison (current: 3.63, good LM target: ~1.0-1.5)
- **Decoder approach: char-only prediction** — simplest, most direct. Charbit vector prediction adds encoding noise.

## Next Steps
See Linear project (PZO) for full backlog. Key priorities:
- [ ] **PZO-19** Fix echo reward — partial credit creating 'h' attractor (Urgent, PR #3 open)
- [ ] **PZO-31** L5 as a proper layer in all regions (High, blocks cerebellum)
- [ ] **PZO-33** Connection modulator refactor (High, cleanup)
- [ ] **PZO-20** Cerebellar forward model (High, blocked by PZO-31)
- [ ] **PZO-34** PFC per-stripe gating (Medium)
