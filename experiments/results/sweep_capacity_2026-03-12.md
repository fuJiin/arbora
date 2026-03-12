# Capacity Sweep Results — 2026-03-12

10,000 tokens (1,105 unique, 58 stories) from TinyStories.
32 columns, k=4, with prediction LTD (rate=0.1).

## Summary Table

| Config | Params | Time | Overlap | Idx Acc | Col Acc | Syn Acc | Entropy | ColSets | Burst | Pred Sets | Pred# | Fb Cos |
|--------|--------|------|---------|---------|---------|---------|---------|---------|-------|-----------|-------|--------|
| shared-4n | 75K | 13s | 0.232 | 1.8% | 1.9% | 0.5% | 74% | 673 | 39% | 7,197 | 9 | 0.231 |
| shared-8n | 222K | 17s | 0.148 | 1.9% | 1.9% | 0.5% | 74% | 704 | 39% | 7,590 | 26 | 0.215 |
| shared-16n | 812K | 34s | 0.099 | 1.2% | 1.1% | 0.3% | 75% | 709 | 40% | 7,638 | 68 | 0.200 |
| neuron-4n | 153K | 14s | 0.155 | 1.3% | 1.3% | 0.1% | 79% | 719 | 49% | 8,430 | 11 | 0.233 |
| neuron-8n | 403K | 23s | 0.084 | 1.0% | 1.0% | 0.1% | 79% | 675 | 49% | 8,461 | 25 | 0.256 |
| neuron-16n | 1.2M | 47s | 0.052 | 1.3% | 1.3% | 0.2% | 79% | 616 | 48% | 8,505 | 62 | 0.201 |

## Key Findings

### Neither change improves accuracy
All configs land at ~1-2% token prediction accuracy. More neurons per column and per-neuron ff_weights don't help.

### Overlap drops with more neurons (expected)
With burst, active set size = k_columns × n_l4 (when bursting). More neurons per column = larger denominator for same number of correct predictions.

### Per-neuron ff improves diversity slightly
- Higher column entropy (79% vs 74%)
- More unique prediction sets (8,400+ vs 7,200+)
- But higher burst rate (49% vs 39%) — predictions are less effective

### Representation is the bottleneck
~700 unique column sets for 1,105 unique tokens. Many tokens share column sets, making them inherently indistinguishable at the prediction level. No amount of neuron-level refinement fixes this.

## Root Cause Analysis

The prediction pathway (lateral/feedback) can only discriminate between tokens that have DIFFERENT column representations. With 32 columns and k=4:
- Theoretical capacity: C(32,4) = 35,960 unique patterns
- Observed: ~700 unique patterns — only 2% utilization
- 1,105 tokens / 700 patterns ≈ 1.6 tokens per pattern on average

The ff_weights don't differentiate enough to give unique column patterns per token. Even with structural sparsity and LTD, the column competition (top-k by max drive) isn't producing discriminative representations.

## Column Set Ambiguity Analysis

Ran `analyze_colset_ambiguity.py` on the same 10K tokens:

- 1,105 unique tokens → only 673 unique column sets
- Only **14.4%** of tokens are uniquely identifiable by any column set
- Most ambiguous set `[2, 29, 30, 31]`: **253 tokens** share it
- Columns 29, 30, 31 appear in almost every column set

### Root cause: space character monopoly

`string.printable` puts space at index 94. Column 31 (w_start=93, window=9) covers indices 93-101 (with wraparound to 0). Since ~79% of tokens start with space, col 31 fires for almost everything. Cols 29-30 cover `{|}~` etc which overlap with the same region.

With 3 of 4 active columns fixed (29,30,31), the model has effectively k=1 discriminative column, giving ~30 possible patterns for 1,105 tokens.

### This means:
- The capacity issue is not about neuron count or ff_weight granularity
- It's about the character alphabet ordering + receptive field layout
- Columns tiling `string.printable` puts all the discriminative letters (a-z) in cols 3-12, while the universal space eats cols 29-31

## Next Directions to Consider

1. **Fix the encoding**: reorder alphabet so space is distributed, or use a better encoding that doesn't have universal characters dominating specific columns
2. **k-WTA per receptive field region**: instead of global top-k, ensure diversity across receptive field regions
3. **Input normalization**: subtract mean column drive so universally-active columns don't always win
4. **Inhibitory columns**: columns that see space should inhibit each other, not all fire together
5. **More columns + higher k**: brute-force more discriminative capacity (though this doesn't fix the space monopoly)
