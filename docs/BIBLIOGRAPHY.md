# Bibliography

Organized by contribution area. Papers may appear in multiple sections.

## Cortical Architecture & Layer Roles

| Reference | Relevance |
|-----------|-----------|
| [Felleman & Van Essen (1991)](https://doi.org/10.1093/cercor/1.1.1). "Distributed hierarchical processing in the primate cerebral cortex." *Cerebral Cortex* 1(1):1-47. | Canonical FF/FB hierarchy. L2/3 = FF source, L5/L6 = FB source. |
| [Bastos et al. (2012)](https://doi.org/10.1016/j.neuron.2012.10.038). "Canonical microcircuits for predictive coding." *Neuron* 76(4):695-711. | Predictive coding mapped onto cortical layers. L2/3 predictions, deep layers carry errors. |
| [Larkum (2013)](https://doi.org/10.1016/j.tins.2012.11.006). "A cellular mechanism for cortical associations." *Trends in Neurosciences* 36(3):141-151. | BAC firing in L5. Apical dendrites in L1 for top-down context. Basis for apical segments. |
| [Harris & Shepherd (2015)](https://doi.org/10.1038/nn.3917). "The neocortical circuit: themes and variations." *Nature Neuroscience* 18(2):170-181. | Canonical microcircuit review. L4 input, L2/3 associative, L5 subcortical. |
| [Lazar et al. (2025)](https://doi.org/10.7554/eLife.93839). "Self-supervised predictive learning accounts for cortical layer-specificity." *eLife* 12. | L4=relay, L2/3=predictor, L5=teaching signal. Validates our three-layer KPI framework. |

## Sparse Binary Representations

| Reference | Relevance |
|-----------|-----------|
| [Ahmad & Hawkins (2016)](https://arxiv.org/abs/1601.00720). "How do neurons operate on sparse distributed representations?" *arXiv:1601.00720*. | Mathematical properties of sparse binary codes: overlap, false positives, unions. |
| [Willmore & Tolhurst (2001)](https://doi.org/10.1088/0954-898X/12/3/302). "Characterizing the sparseness of neural codes." *Network* 12(3):255-270. | Population vs lifetime sparseness. Treves-Rolls measure. Foundation for sparseness KPIs. |

## Information Theory in Neural Coding

| Reference | Relevance |
|-----------|-----------|
| [Borst & Theunissen (1999)](https://doi.org/10.1038/14731). "Information theory and neural coding." *Nature Neuroscience* 2(11):947-957. | Foundational. MI, transfer entropy for directed information flow between layers. |
| [Timme & Lapish (2018)](https://doi.org/10.1523/ENEURO.0052-18.2018). "A tutorial for information theory in neuroscience." *eNeuro* 5(3). | Practical MI, conditional MI, partial information decomposition. |
| [Williams & Beer (2010)](https://arxiv.org/abs/1004.2515). "Nonnegative decomposition of multivariate information." *arXiv:1004.2515*. | Partial information decomposition: unique, redundant, synergistic components. |

## Representation Quality Metrics

| Reference | Relevance |
|-----------|-----------|
| [Kornblith et al. (2019)](https://arxiv.org/abs/1905.00414). "Similarity of neural network representations revisited." *ICML*. | CKA for comparing representations across layers. Basis for cross-layer divergence KPI. |
| [Alain & Bengio (2016)](https://arxiv.org/abs/1610.01644). "Understanding intermediate layers using linear classifier probes." *arXiv:1610.01644*. | Linear probing methodology for measuring decodability at each layer. |
| [Hewitt & Ethayarajh (2021)](https://aclanthology.org/2021.emnlp-main.122). "Conditional probing: measuring usable information beyond a baseline." *EMNLP*. | V-usable information. Additional info a representation provides beyond baseline. |
| [Gao & Ganguli (2015)](https://doi.org/10.1016/j.conb.2015.04.005). "On simplicity and complexity in large-scale neuroscience." *Current Opinion in Neurobiology* 32:148-155. | Participation ratio (effective dimensionality) for neural populations. |
| [Eastwood & Williams (2018)](https://openreview.net/forum?id=By-7dz-AZ). "A framework for quantitative evaluation of disentangled representations." *ICLR*. | Disentanglement, completeness, informativeness metrics. |

## Predictive Coding & Temporal Processing

| Reference | Relevance |
|-----------|-----------|
| [Rao & Ballard (1999)](https://doi.org/10.1038/4580). "Predictive coding in the visual cortex." *Nature Neuroscience* 2(1):79-87. | Original predictive coding model. Errors propagate up, predictions propagate down. |
| [Keller & Mrsic-Flogel (2018)](https://doi.org/10.1016/j.neuron.2018.10.003). "Predictive processing: a canonical cortical computation." *Neuron* 100(2):424-435. | Modern review. Prediction errors, mismatch responses, layer-specific roles. |

## Cortical Column Models

| Reference | Relevance |
|-----------|-----------|
| [Mountcastle (1997)](https://doi.org/10.1093/brain/120.4.701). "The columnar organization of the neocortex." *Brain* 120(4):701-722. | Original minicolumn hypothesis. Foundation for column-based architecture. |
| [Hawkins & Ahmad (2016)](https://doi.org/10.3389/fncir.2016.00023). "Why neurons have thousands of synapses." *Frontiers in Neural Circuits* 10:23. | Dendritic segments for sequence prediction. Theoretical basis for segment learning. |
| [Hawkins, Ahmad & Cui (2017)](https://doi.org/10.3389/fncir.2017.00081). "A theory of how columns enable learning world structure." *Frontiers in Neural Circuits* 11:81. | Multi-column object recognition. Lateral connections for consensus. |

## Thalamic & Subcortical Circuits

| Reference | Relevance |
|-----------|-----------|
| [Sherman & Guillery (2011)](https://doi.org/10.1152/jn.00429.2011). "Distinct functions for direct and transthalamic corticocortical connections." *J Neurophysiology* 106(3):1068-1077. | Transthalamic pathways. Relevant when we add L6/thalamic relay. |

## Noise Robustness & Stability

| Reference | Relevance |
|-----------|-----------|
| [Schoonover et al. (2021)](https://doi.org/10.1038/s41586-021-03628-7). "Representational drift in primary olfactory cortex." *Nature* 594:541-546. | Stable task performance despite changing neural codes. |
| [Gallego et al. (2018)](https://doi.org/10.1038/s41593-018-0141-6). "Cortical population activity within a preserved neural manifold." *Nature Neuroscience* 21:1061-1074. | Stable subspace despite neuron drift. Supports effective dimensionality over individual neuron stats. |
