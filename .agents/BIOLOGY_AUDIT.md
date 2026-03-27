# Biology Audit Results

> **DELETE THIS FILE** once findings are absorbed and actioned.
> Created 2026-03-27. Based on primary source research.

## Connection accuracy summary

| Connection | Accuracy | Action |
|---|---|---|
| FF: L2/3→L4 (inter-region) | **HIGH** | Reverted from L5 (PR #39) |
| Apical: L5→{L2/3,L5} | **HIGH** | Correct |
| Input→L4 | **HIGH** | Correct |
| L4→L2/3 (intra-column) | **HIGH** | Per-column weights (PR #38) |
| L2/3→L5 (intra-column) | **HIGH** | Per-column weights (PR #38) |
| L4→L4 lateral (lat_seg) | **HIGH** | Correct |
| L2/3→L4 feedback (fb_seg) | **LOW** | Removed (PR #39) |
| L2/3→L2/3 lateral | **HIGH** | Correct |
| L5→L5 lateral | **HIGH** | Correct |
| L5 apical segments | **HIGH** | Correct (BAC firing) |
| L2/3 apical segments | **MEDIUM-HIGH** | Keep — weaker than L5 but functional |
| L4 in motor/PFC | **LOW** | Defer — agranular regions need different pipeline |

## Layer role clarification

- **L2/3**: Feedforward source (canonical, Felleman & Van Essen 1991)
- **L5**: Feedback/apical source + subcortical output (NOT FF source)
- **L6**: Missing — thalamic gain control, major feedback source

## Major biological gaps (deferred)

- L6 layer (thalamic gain control + corticocortical feedback)
- Inhibitory interneurons (PV/SST/VIP circuits)
- Agranular motor/PFC regions (no true L4)
- Motor hierarchy direction (PFC→M2→M1 top-down, not M1→M2)

## Sources

- Felleman & Van Essen 1991 (cortical hierarchy)
- Bastos et al. 2012 (canonical microcircuits for predictive coding)
- Larkum 2013 (BAC firing, cellular mechanism for cortical associations)
- Sherman & Guillery (transthalamic pathways)
- eLife 2024 (L6 corticocortical feedback)
