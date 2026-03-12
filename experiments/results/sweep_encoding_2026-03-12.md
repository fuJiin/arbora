# Encoding Sweep Results — 2026-03-12

10,000 tokens (1,105 unique, 58 stories) from TinyStories.
32 columns, k=4, with prediction LTD (rate=0.1).

## Summary Table

| Config | ColSets | UniqID% | MaxAmb | Entropy | Accuracy | Burst |
|--------|---------|---------|--------|---------|----------|-------|
| charbit | 673 | 14.4% | 253 | 74.2% | 1.8% | 38.5% |
| charbit-nosp | 860 | 14.5% | 131 | 79.1% | 0.9% | 42.7% |
| random-808 | 4,893 | 82.4% | 14 | 98.9% | 0.5% | 50.2% |

## Key Finding: The prediction mechanism is broken

Random encoder gives near-perfect representation (82.4% uniquely identifiable, 98.9% entropy, max 14 tokens per column set) but **worse accuracy (0.5%)** than charbit (1.8%).

This proves:
1. The encoding was NOT the bottleneck — fixing it doesn't help
2. The lateral/feedback learning rule cannot learn temporal patterns
3. More diverse representations actually make prediction harder (more things to distinguish)

## Why the prediction pathway fails

The Hebbian rule `trace × active → strengthen` learns "neurons that were active before this neuron tend to co-occur." But:
- With 50% burst rate (random), most activations are bursting → all neurons fire → traces spread across too many neurons
- The prediction LTD helps (sparser predictions) but the LTP signal is still too diffuse
- The eligibility window (decay=0.95, ~20 steps) means we're trying to learn temporal associations over many steps, but the signal-to-noise ratio is too low

## Space-stripping analysis

Removing leading spaces improved column diversity:
- Column sets: 673 → 860
- Max ambiguity: 253 → 131
- Entropy: 74.2% → 79.1%

But accuracy dropped (1.8% → 0.9%) — more diverse representations actually made it harder for the (broken) prediction mechanism to find patterns.

## Conclusion

The next step is NOT encoding improvement. It's fixing the prediction learning rule:
- Current: flat Hebbian (trace × active → strengthen)
- Needed: something that can learn "token A predicts token B" with high specificity
- The biological analog might be dendritic segment matching with specific synapse subsets, rather than whole-neuron Hebbian learning
