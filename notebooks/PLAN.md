# ARB-139 Notebook Series — Plan

A guided walk through the theory and experiments behind sparse-binary
representations + local learning, building toward a blog post on the
research direction.

## Status

| Part | Topic | Status | File |
|---|---|---|---|
| 1 | What word2vec is doing, and how SSH translates it | ✅ Built | `arb139_part1_word2vec_to_ssh.ipynb` |
| 2 | Math floor — Oja's rule, PCA convergence, the unifying equation | 🚧 Outlined | `arb139_part2_oja_and_unification.ipynb` |
| 3 | Stability primitives — synaptic scaling, BCM, Sanger | 🚧 Outlined | `arb139_part3_stability_primitives.ipynb` |
| 4 | Metrics — semantically simple, comparable across CDR/SBR | 🚧 Outlined | `arb139_part4_metrics.ipynb` |
| 5 | Implementation tradeoffs + blog post outline | 🚧 Outlined | `arb139_part5_implementation_and_blog.ipynb` |

`build_<part>.py` files generate the corresponding `.ipynb`. Run with
`uv run python notebooks/build_<part>.py`. The script approach keeps the
source readable and git-diffable; the `.ipynb` is what you open and run.

## Key terms used throughout

- **CDR** — continuous dense representation (e.g., word2vec vectors)
- **SBR** — sparse binary representation (top-k of an accumulator)
- **SSH** — Sparse Skip-gram Hebbian (the SBR + local Hebbian analog of word2vec)
- **Hebbian** — pre × post coincidence-based update, no error signal
- **Anti-Hebbian** — symmetric weakening on uncoincident or noise pairs
- **Modulator** — scalar gate on the update, derived from local activity
- **Oja's rule** — Hebbian + weight decay term that yields PCA convergence
- **k-WTA** — k-winner-take-all sparsification (top-k binarization)

## Part 1 — word2vec → SSH (built)

Walks through:
1. Skip-gram + negative sampling — same in word2vec and SSH
2. Word2vec representation (CDR) and the dot-product score
3. The SGD update rule and the **sigmoid as triple-duty stabilizer**
   (magnitude bound, surprise modulation, sliding threshold)
4. SSH representation (SBR via top-k of accumulator)
5. The Hebbian update rule — symmetric LTP and anti-Hebbian LTD
6. Locality side-by-side — word2vec touches all D dims, SSH touches k
7. The translation table — and the **empty stabilizer row** for SSH
8. Sample-efficiency curves on text8 from the actual sweep

**Key insight from this part:** the difference between word2vec and SSH
isn't "gradient vs Hebbian." It's "stabilized vs not." Word2vec's sigmoid
factor is doing three jobs simultaneously; SSH needs separate primitives
(modulation, decay, BCM threshold) to recover those jobs.

## Part 2 — Oja's rule and the unifying equation

Plan:
1. Why plain Hebbian explodes — the boundedness problem
2. **Oja's rule**: `dw = η · y · (x − y · w)` and why the `−y²w` term saves us
3. **Oja's theorem**: this rule converges to the principal eigenvector of
   the input covariance, `||w|| → 1`. So Hebbian + the right decay term IS
   streaming PCA. Local learning has provable convergence properties.
4. **Sanger's rule (GHA)**: extension to multiple components. Streaming
   multi-component PCA via Hebbian + Gram-Schmidt-like projection.
5. **The unifying parametric form**:
   ```
   Δw = η · m(·) · [f_pre(x) · f_post(y) − s(w, θ)]
   ```
   - `m`: modulator (1 for vanilla, σ(-score) for word2vec, surprise for SSH)
   - `f_pre`, `f_post`: pre/post activity nonlinearities
   - `s(w, θ)`: stability term (decay, subtractive norm, BCM)
6. Each classical rule mapped to its (m, f_pre, f_post, s) tuple
7. Sanity-check: implement Oja's rule on synthetic 2D data, watch it
   converge to the principal eigenvector
8. Reference reading: Oja (1982), Sanger (1989), Földiák (1990),
   Bienenstock-Cooper-Munro (1982)

## Part 3 — Stability primitives in detail

Plan:
1. **Weight decay (Oja-like)**: uniform shrinkage. Lazy implementation.
2. **Subtractive normalization** (Miller & MacKay): `w -= mean(w)` per row.
   Forces zero-sum competition.
3. **Synaptic scaling / row-norm bound**: divisive normalization. The
   "soft k-WTA" continuous relaxation.
4. **BCM sliding threshold**: postsynaptic activity above threshold → LTP,
   below → LTD, threshold slides with running activity. Prevents dead and
   runaway neurons.
5. How each primitive interacts with the modulator — the "do you need
   decay if you have modulation?" question and our empirical answer
   (mostly no at small scale, yes at moderate scale, scale-dependent).
6. Map T1 (arbora cortex) onto these primitives. Show that T1 already
   has Oja-like decay (`synapse_decay=0.999`), surprise modulation (burst
   signal), and column-level k-WTA. The unifying form makes T1's design
   choices legible.

## Part 4 — Metrics

Plan:
1. The naïve metrics we used (SimLex Spearman, analogy, capacity, bundling,
   corruption, partial-cue, storage, train cost) — strengths and weaknesses
2. Why they don't all transfer cleanly across CDR/SBR
   - SimLex via Jaccard has discrete-jump problem on binary codes
   - Bundling capacity definition has to bridge bit-OR (sparse) and
     vector-mean (dense)
   - Effective dimensionality means different things in dense vs sparse
3. **Semantically simple, representation-agnostic versions**:
   - Pair-similarity rank correlation (Spearman/Kendall) — works for both
     once you pick the similarity function appropriate to representation
   - Bundling capacity as "fraction recoverable from superposition" —
     unify dense and sparse via fraction-recovered metric, not raw margin
   - Continual-learning retention as fraction of old-knowledge preserved
   - Corruption resilience as area-under-degradation-curve
4. Build a clean reusable evaluation harness for the multi-seed runs

## Part 5 — Implementation tradeoffs + blog post outline

Plan:
1. **Performance options**:
   - Lazy decay (per-row timestamp + amortized shrinkage)
   - Batched-by-word Hebbian updates (snapshot codes, aggregate deltas)
   - Numba/Cython inner loop (already in)
   - Heap-based top-k for asymptotic improvement
2. **Math-vs-ablation**: when do we need to run the full ablation, and
   when can a closed-form analysis tell us the answer?
3. **Bringing primitives back to T1**: refactor the cortex update rule
   into the unified form. Win is interpretability + ablatability, not
   speed.
4. **Blog post outline**:
   - Hook: word2vec works because of its stabilizer, not its update rule
   - Sparse binary + local Hebbian retains word2vec's structure with
     stronger continual-learning properties
   - Empirical: SSH(modulated+decay) crushes word2vec on small-data
     SimLex; loses on absolute peak at large scale
   - Continual learning experiment (forthcoming) is where SBR + local
     fully wins
   - Connection to existing work (Bricken/Anthropic SDM, HTM, VSA, Levy &
     Goldberg PMI factorization)
   - The deeper claim: gradient descent and modulated local Hebbian are
     the same algorithm in shallow models; arbora is the deep cortical
     embodiment of this

## Empirical milestones (running tally)

| experiment | date | finding |
|---|---|---|
| 5-baseline sweep on text8 | apr 24 | word2vec wins absolute; sparse methods win on corruption + storage |
| SSH baseline on text8 | apr 24 | SimLex 0.124±0.030 at 500k vs word2vec −0.001; non-monotonic curve |
| Modulated SSH ablation | apr 25 | Modulation alone prevents 1M dip (vanilla -0.020 → modulated +0.073) |
| Decay rate sweep at 1M | apr 25 | Bimodal — decay=0 (0.073) and decay=3e-4 (0.098) both good, in-between hurts |
| Cross-scale decay test | apr 25 | decay=3e-4 is scale-dependent — hurts at 100k, helps from 500k onward |

## Open experimental questions

1. **Decay schedule**: should `decay` scale with corpus size, e.g.,
   `decay = const / sqrt(N)` to keep total shrinkage constant?
2. **Multi-seed at headline points**: confirm the +0.159 SimLex at 500k
   mod+decay is robust across seeds
3. **Continual learning** with new vocabulary: does SBR + local Hebbian
   preserve old word embeddings while integrating novel ones, where
   word2vec catastrophically forgets?
4. **Performance**: lazy decay (quick win), batched-by-word update (bigger
   refactor)
5. **Theoretical analysis**: can we derive a closed-form fixed point for
   modulated SSH analogous to Oja's theorem?

## Reading list (anchored to discussions in the conversation)

- Oja, E. (1982). "A simplified neuron model as a principal component analyzer." J. Math. Bio. 15.
- Sanger, T. (1989). "Optimal unsupervised learning in a single-layer linear feedforward neural network." Neural Networks 2.
- Bienenstock, Cooper, Munro (1982). "Theory for the development of neuron selectivity..." J. Neuroscience 2.
- Földiák, P. (1990). "Forming sparse representations by local anti-Hebbian learning." Biol. Cybern. 64.
- Miller & MacKay (1994). "The role of constraints in Hebbian learning." Neural Computation 6.
- Levy & Goldberg (2014). "Neural word embedding as implicit matrix factorization." NeurIPS.
- Bricken & Pehlevan (2021). "Attention approximates sparse distributed memory." NeurIPS.
- Bricken et al. (2023). "Sparse Distributed Memory is a Continual Learner." ICLR.
- Sahlgren (2005, 2006). Random Indexing literature.
- Kanerva (2009). "Hyperdimensional Computing."
- Joshi et al. (2017). "Language Geometry using Random Indexing."
- Bricken et al. (2023). "Towards Monosemanticity" / Templeton et al. (2024). "Scaling Monosemanticity."
