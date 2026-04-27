"""Sparse-vs-dense word embedding comparison (ARB-139).

Four-way comparison on shared text8 data + vocab + eval battery,
isolating "sparse-binary representation" from "local learning rule"
from "architecture":

- word2vec (dense continuous, gradient-trained shallow NN)
- random_indexing (sparse binary, no learning, flat random projection)
- brown_cluster (sparse binary, hierarchical, greedy clustering)
- t1_sparse (sparse binary, local Hebbian + k-WTA, cortical circuit)

Headline metrics:

- SimLex-999 Spearman correlation (semantic similarity)
- Google analogy top-1 accuracy (vector arithmetic)
- Sample efficiency curves (metric vs training tokens)
- Bundling capacity (VSA-style superposition recovery)
- Capacity / collision / effective dimensionality
- Corruption robustness (Hamming flips for sparse, Gaussian noise for dense)
- Storage cost + NN-query speed
- Continual learning forgetting (word2vec + t1 today, others later)

All architectures take one-hot per word from text8, so the substrate
is the only difference.
"""
