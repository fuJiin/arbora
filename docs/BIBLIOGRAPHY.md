# Bibliography

Organized by contribution area. Papers may appear in multiple sections.

## Cortical Architecture & Layer Roles

- Felleman DJ, Van Essen DC (1991). "Distributed hierarchical processing in the primate cerebral cortex." *Cerebral Cortex* 1(1):1-47.
  - Canonical feedforward/feedback hierarchy. L2/3 = FF source, L5/L6 = FB source.

- Bastos AM, Usrey WM, Adams RA, Mangun GR, Fries P, Friston KJ (2012). "Canonical microcircuits for predictive coding." *Neuron* 76(4):695-711.
  - Predictive coding framework mapped onto cortical layers. L2/3 generates predictions, deep layers carry prediction errors.

- Larkum M (2013). "A cellular mechanism for cortical associations: an organizing principle for the cerebral cortex." *Trends in Neurosciences* 36(3):141-151.
  - BAC firing in L5 thick-tufted neurons. Apical dendrites in L1 for top-down context. Basis for our apical segment model.

- Harris KD, Shepherd GMG (2015). "The neocortical circuit: themes and variations." *Nature Neuroscience* 18(2):170-181.
  - Review of canonical microcircuit. L4 input, L2/3 associative/output, L5 subcortical projection.

- Lazar A, Bhatt D, Bhalla US (2025). "Self-supervised predictive learning accounts for cortical layer-specificity." *eLife* 12.
  - L4 as relay, L2/3 as predictor, L5 as teaching signal. Uses linear decoding accuracy, Treves-Rolls sparseness, PCA dimensionality. Directly validates our three-layer KPI framework.

## Sparse Binary Representations

- Ahmad S, Hawkins J (2016). "How do neurons operate on sparse distributed representations? A mathematical theory of sparsity, neurons and active dendrites." *arXiv:1601.00720*.
  - Mathematical properties of sparse binary codes: overlap probability, false positive analysis, union properties.

- Willmore B, Tolhurst DJ (2001). "Characterizing the sparseness of neural codes." *Network: Computation in Neural Systems* 12(3):255-270.
  - Population sparseness vs lifetime sparseness. Treves-Rolls measure. Foundation for our sparseness KPIs.

## Information Theory in Neural Coding

- Borst A, Theunissen FE (1999). "Information theory and neural coding." *Nature Neuroscience* 2(11):947-957.
  - Foundational. Mutual information, transfer entropy for directed information flow between layers.

- Timme NM, Lapish C (2018). "A tutorial for information theory in neuroscience." *eNeuro* 5(3).
  - Practical MI, conditional MI, partial information decomposition for neural data.

- Williams PL, Beer RD (2010). "Nonnegative decomposition of multivariate information." *arXiv:1004.2515*.
  - Partial information decomposition: unique, redundant, synergistic components. Framework for understanding population codes.

## Representation Quality Metrics

- Kornblith S, Norouzi M, Lee H, Hinton GE (2019). "Similarity of neural network representations revisited." *ICML*.
  - Centered Kernel Alignment (CKA) for comparing representations across layers. Basis for our cross-layer divergence KPI.

- Alain G, Bengio Y (2016). "Understanding intermediate layers using linear classifier probes." *arXiv:1610.01644*.
  - Linear probing methodology for measuring decodability at each layer.

- Hewitt J, Ethayarajh K (2021). "Conditional probing: measuring usable information beyond a baseline." *EMNLP*.
  - V-usable information. Measures how much *additional* information a representation provides beyond a simple baseline.

- Gao P, Ganguli S (2015). "On simplicity and complexity in the brave new world of large-scale neuroscience." *Current Opinion in Neurobiology* 32:148-155.
  - Participation ratio (effective dimensionality) for neural population analysis.

- Eastwood C, Williams CKI (2018). "A framework for the quantitative evaluation of disentangled representations." *ICLR*.
  - Disentanglement, completeness, informativeness metrics. Potentially applicable to column specialization.

## Predictive Coding & Temporal Processing

- Rao RP, Ballard DH (1999). "Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects." *Nature Neuroscience* 2(1):79-87.
  - Original predictive coding model. Prediction errors propagate up, predictions propagate down.

- Keller GB, Mrsic-Flogel TD (2018). "Predictive processing: a canonical cortical computation." *Neuron* 100(2):424-435.
  - Modern review. Prediction errors, mismatch responses, layer-specific roles in predictive processing.

## Cortical Column Models

- Mountcastle VB (1997). "The columnar organization of the neocortex." *Brain* 120(4):701-722.
  - Original minicolumn hypothesis. Foundation for column-based architecture.

- Hawkins J, Ahmad S (2016). "Why neurons have thousands of synapses, a theory of sequence memory in neocortex." *Frontiers in Neural Circuits* 10:23.
  - Dendritic segments for sequence prediction. Theoretical basis for our segment learning mechanism.

- Hawkins J, Ahmad S, Cui Y (2017). "A theory of how columns in the neocortex enable learning the structure of the world." *Frontiers in Neural Circuits* 11:81.
  - Multi-column object recognition. Lateral connections between columns for consensus.

## Thalamic & Subcortical Circuits (Deferred)

- Sherman SM, Guillery RW (2011). "Distinct functions for direct and transthalamic corticocortical connections." *Journal of Neurophysiology* 106(3):1068-1077.
  - Transthalamic pathways. L5 -> higher-order thalamus -> cortex. Relevant when we add L6/thalamic relay.

## Noise Robustness & Stability

- Schoonover CE, Ohashi SN, Axel R, Bhalla US (2021). "Representational drift in primary olfactory cortex." *Nature* 594:541-546.
  - Representational drift: stable task performance despite changing neural codes. Relevant to our stability metrics.

- Gallego JA, Perich MG, Naufel SN, Ethier C, Solla SA, Miller LE (2018). "Cortical population activity within a preserved neural manifold." *Nature Neuroscience* 21:1061-1074.
  - Stable subspace despite individual neuron drift. Supports measuring effective dimensionality over individual neuron statistics.
