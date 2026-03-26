# Learning Mechanism Audit + STEP-62 Decisions

> **DELETE THIS FILE** once findings are absorbed and actioned.
> Created 2026-03-26. Decisions agreed 2026-03-26.

## Current state

### Connections
| Connection | Source | Target | Role | Learning |
|---|---|---|---|---|
| S1→S2 | L2/3 | L4 | FF | Hebbian + burst gate + surprise |
| S2→S3 | L2/3 | L4 | FF | Hebbian + burst gate + surprise |
| S2→PFC | L2/3 | L4 | FF | Three-factor |
| S3→PFC | L2/3 | L4 | FF | Three-factor |
| S2→M2 | L2/3 | L4 | FF | Three-factor |
| PFC→M2 | L2/3 | L4 | FF | Three-factor |
| M2→M1 | L2/3 | L4 | FF | Three-factor |
| S2→S1 | L2/3 | L4 | Apical | Linear gain |
| S3→S2 | L2/3 | L4 | Apical | Linear gain |
| M1→M2 | L2/3 | L4 | Apical | Linear gain |
| M2→PFC | L2/3 | L4 | Apical | Linear gain |
| S1→M1 | L2/3 | L4 | Apical | Linear gain + surprise |
| M1→S1 | L2/3 | L4 | Apical | Linear gain (efference copy) |

### Segment types
| Segment | Layer | Context | Traces | Status |
|---|---|---|---|---|
| fb_seg | L4 | L2/3 (intra-region) | Yes | Working |
| lat_seg | L4 | L4 (lateral) | Yes | Working |
| l23_seg | L2/3 | L2/3 (lateral) | Yes | Working |
| l5_seg | L5 | L5 (lateral) | Yes (STEP-54) | Off by default |
| apical_seg | L5 | External source | **No traces** | +18% PFC ctx_disc |

### Known issues
1. Apical segments lack traces (boolean external context)
2. L5 lateral off by default (regressed at 50k)
3. Linear gain is LTP-only, no LTD
4. No FF prediction penalty
5. Surprise scales ALL learning (including structure)

---

## STEP-62 Decisions (agreed 2026-03-26)

### 1. Per-connection traces ✅
Every Connection carries a decaying trace of its source lamina's firing
rate. All pathways (FF and apical) get temporal credit. One mechanism,
universally applied. Replaces the need for separate per-mechanism traces.

### 2. Apical target: L4 → {L2/3, L5} ✅
Move apical destination from L4 to both L2/3 and L5. Both layers have
apical dendrites reaching L1 in biology. Implemented as two connections
per feedback pathway (one to L2/3, one to L5), or a single connection
targeting both.

### 3. External apical segments on L2/3 ✅
L2/3 gets apical segment infrastructure for top-down context from higher
regions. Keep existing fb_seg (L2/3→L4 intra-region) — it serves a
different role (within-region prediction, proxy for L6→L4).

### 4. Segments as default for all apical ✅
Remove linear gain mode. Apical segments become the only apical mode.
Simplifies code (one path, not two). Rally around segments everywhere,
optimize later (sparse weights as a perf lever if needed, or Rust/async).

### 5. Limit surprise to FF only ✅
Surprise modulation applies to feedforward weight learning only.
Segment learning (structure/prediction) is not surprise-gated.

---

## Deferred work (post-STEP-62)

### L5 as universal corticocortical output (HIGH PRIORITY)
**This is the next major architectural bet after STEP-62.**

L5 becomes:
- **FF source** to higher L4 (replaces current L2/3→L4 FF)
- **Apical source** to lower L2/3 and L5 (replaces current L2/3 apical source)
- **Subcortical output** (BG, cerebellum)

**Blocker:** L5 lateral segments need to work reliably first.
Re-evaluate L5 lateral with the new continuous traces (STEP-54).
If L5 carries a good signal, switching FF/apical source to L5 is a
one-line change per connection in canonical.py.

### Other deferred items
- FF prediction penalty (punish FF weights for wrong L4 predictions)
- Segments vs sparse weights (profile first, optimize in fewer places)
- fb_seg removal (evaluate after L2/3 apical is working)
- M2 → PMC rename (cosmetic, low priority)
- Pre-trace vs eligibility naming cleanup

---

## Perf notes
| Mechanism | Cost | Notes |
|---|---|---|
| Segment prediction | **Highest** | Per-neuron scoring, dominates at high n_segments |
| Segment learning | Moderate | Batch grow/adapt |
| FF learning | Low | Matrix multiply on winners |
| Trace updates | Negligible | O(n_neurons) per layer |
| Optimization levers | | Sparse weights, Rust, async, reduce n_segments |
