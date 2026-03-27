# Lamina KPI Framework

> **DELETE THIS FILE** once findings are absorbed into CONTEXT.md/Linear and
> implementation is underway.
> Created 2026-03-27. Defines the top 3 KPIs per lamina and how to handle
> missing lamina in agranular regions.

## Design Principles

- Each lamina has a biological role; KPIs measure whether it fulfills that role
- Metrics should be per-layer, not per-region (same KPI set across S1/S2/S3 etc.)
- Avoid sparse-binary-specific jargon; use standard neuroscience/information-theory terms
- Missing lamina (agranular regions): surviving lamina absorb the missing layer's KPIs

## L4 — Input Reception + Temporal Prediction

**Role**: Receive feedforward drive, represent input faithfully, predict next
input via lateral dendritic segments.

| # | KPI | Formula | Target | Why this target |
|---|-----|---------|--------|-----------------|
| 1 | **Prediction recall** | \|predicted ∩ active\| / \|active\| | >70% | Below 50% means segments predict less than half of activations — mostly guessing. 70%+ means temporal patterns are being learned. (This is 1 - burst_rate.) |
| 2 | **Prediction precision** | \|predicted ∩ active\| / \|predicted\| | >50% | Random precision for k=8 active out of 128 columns is ~6%. 50%+ means segments are selective, not just predicting everything. Low precision with high recall = promiscuous segments. |
| 3 | **Population sparseness** | (Σr/n)² / (Σr²/n) (Treves-Rolls) | ~k/N (0.016 for S1) | For binary codes, Treves-Rolls equals the fraction of active neurons. Should match the architectural target: k_columns × n_l4 / (n_columns × n_l4) = 8/128 ≈ 0.063 at column level. Drift means excitability or WTA is broken. |

**Current state**: We have column-level burst rate (= 1 - recall) and
prediction hit rate (conflates precision/recall). Missing: neuron-level
precision, population sparseness.

## L2/3 — Context-Enriched, Decodable Output

**Role**: Integrate L4 input + lateral context, produce representation that
downstream regions can use. L2/3 is the feedforward output — its quality
determines the information available to the rest of the hierarchy.

| # | KPI | Formula | Target | Why this target |
|---|-----|---------|--------|-----------------|
| 1 | **BPC** | Dendritic decoder softmax on L2/3 state | <6.0 | log2(65 chars) ≈ 6.02 is random baseline. Below 6.0 means the decoder extracts more info than uniform guessing. Good char-level models reach 1.0-1.5. |
| 2 | **Context discrimination** | Mean Jaccard distance (same token, different contexts) | >0.80 | 0.0 = identical patterns regardless of context (pure input relay). 1.0 = completely different patterns (no token identity preserved). 0.80+ means strong context enrichment while retaining decodability. |
| 3 | **Effective dimensionality** | Participation ratio: (Σλ)² / Σλ² from activation covariance | >50 (of 512 dims) | Low PR (~1-10) means collapsed to few stereotyped patterns. PR should exceed L4's (evidence of context enrichment adding dimensions). 50+ uses ~10% of representational capacity — enough for downstream regions to discriminate. |

**Current state**: BPC and context discrimination exist. Missing: effective
dimensionality.

## L5 — Feedback Signal + Subcortical Output

**Role**: Receive L2/3 drive + top-down apical context. Project to subcortical
targets (BG, thalamus, brainstem) and provide feedback to lower regions.
In predictive coding, L5 carries the "ground truth" that generates prediction
errors in target regions.

| # | KPI | Formula | What it catches |
|---|-----|---------|-----------------|
| 1 | **Apical modulation index** | Jaccard(L5 with apical, L5 without apical) | Top-down influence — is apical context reaching L5? |
| 2 | **Cross-layer divergence** | 1 - CKA(L2/3 activations, L5 activations) | Information routing — is L5 adding value beyond copying L2/3? |
| 3 | **Output decodability** | Linear probe accuracy (L5 → task labels) | Signal quality — can targets use L5 output? |

**Current state**: Zero L5-specific metrics. All three are missing.

## Agranular Regions (M1, M2, PFC)

Motor and prefrontal regions lack a true L4 (granular layer). Input arrives
directly at L2/3. This changes the KPI mapping:

| Lamina | Granular (S1/S2/S3) | Agranular (M1/M2/PFC) |
|--------|---------------------|----------------------|
| L4 | Input + prediction (3 KPIs) | *absent* |
| L2/3 | Context enrichment (3 KPIs) | Input + context (absorbs L4 KPIs) |
| L5 | Feedback + output (3 KPIs) | Feedback + output (unchanged) |

**L2/3 in agranular regions** gets an expanded KPI set:
1. BPC (decodability — same as granular)
2. Prediction recall (absorbed from L4 — L2/3 now receives direct input)
3. Context discrimination (same — but also tests input fidelity)

Effective dimensionality drops to secondary. The key question shifts from
"is L2/3 enriching beyond L4?" to "is L2/3 both receiving input faithfully
AND adding context?"

## Implementation Priority

1. **L2/3 BPC** — done (just added to sweep via BPCProbe)
2. **L2/3 burst rate** — done (just added to sweep via L23BurstTracker)
3. **Prediction precision** on L4 — cheap, add to sweep next
4. **Population sparseness** (Treves-Rolls) on L4 and L2/3 — cheap
5. **Effective dimensionality** on L2/3 — moderate (needs window of activations)
6. **L5 metrics** — blocked on L5 having meaningful input (needs multi-region)

## References

See [docs/BIBLIOGRAPHY.md](../../docs/BIBLIOGRAPHY.md) for full citations. Key sources for this framework:
- Lazar et al. 2025 (layer-specific roles: L4=relay, L2/3=predictor, L5=teaching signal)
- Willmore & Tolhurst 2001 (population vs lifetime sparseness)
- Kornblith et al. 2019 (CKA for cross-layer comparison)
- Bastos et al. 2012 (predictive coding mapped to cortical layers)
