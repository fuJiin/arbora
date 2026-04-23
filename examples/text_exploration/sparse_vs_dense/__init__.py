"""Sparse-vs-dense word embedding comparison (ARB-139).

Compare T1 (sparse binary, local Hebbian) against word2vec (dense
continuous, gradient-trained) on the same data + same vocab + same
benchmarks. Headline metrics:

- SimLex-999 Spearman correlation (semantic similarity)
- Google analogy top-1 accuracy (vector arithmetic)
- Sample efficiency curves (metric vs training tokens)

All architectures take one-hot per word from text8, so the substrate
is the only difference.
"""
