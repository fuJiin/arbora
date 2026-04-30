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
| k_eval sweep at 1M | apr 27 | top-k aperture matters: best SimLex at k_eval≈160 (k_train=40); decoupled from training-time k |
| Continual v2: cross-domain (text8→Gutenberg) | apr 28 | At 5M each phase, SSH forgets MORE than w2v on shared/cross — capacity-ceiling effect |
| Continual sweep: small-update size | apr 28 | Crossover at ~500k–1M Gutenberg: SSH preserves better below, w2v scales better above |
| Continual sweep: D=2048 vs D=1024 at 1M | apr 28 | Doubling D does NOT reduce SSH forgetting; bottleneck is k aperture, not raw dimensionality |
| BCM-style consolidation bonus at 5M cross-domain | apr 28 | Non-monotonic in λ; sweet spot at λ=0.5 cuts cross drift from -0.119 to -0.014 (~90% reduction) with mild shared cost; λ=1.0 over-freezes (cross drift -0.278) because sigmoid bound makes phase-2 updates vanish |
| W2v 50-epoch sweep at 5M text8 | apr 28 | Peak SimLex 0.265 at ep8; plateau ~0.245 through ep25; slow overfit drift to 0.210 by ep50 (-21% from peak). Matches Schnabel 2015 / Salle 2016 reports of intrinsic-eval peak-then-decline on small corpora. Implication: our 1-epoch continual-learning experiments leave both methods at ~0.188, well below convergence; the "w2v rebound at b=5M" finding is partially "w2v finishes the training that didn't fit in phase 1." |
| Best historical SSH peak (1 epoch) | apr 24-28 | SimLex 0.204 at 5M with sigmoid+single_table+modulated, k_eval=40. Extrapolated to k_eval=160: ~0.30 (estimate based on the apr 27 k_eval sweep showing ~50% lift). Multi-epoch SSH at 5M with k_eval=160 not yet measured — the experiment that closes the convergence-comparison loop. |
| Hogwild!-parallel SSH infrastructure | apr 28 | Added `n_workers` to modulated SSH baseline; numba.prange splits token stream into contiguous slices, lock-free shared updates to A_center/A_context (Niu et al. 2011 Hogwild). 4 workers gives 2× speedup with ΔSimLex ≤ 0.005 (within seed-noise). 8 workers gives 3.2× but with measurable race noise (ΔSimLex 0.04). 2× ceiling at 4 workers is cache-line contention on the 40MB A-table working set (doesn't fit in L3). Same memory-pressure ceiling gensim hits at production scale. |
| Matched 8-epoch continual at 5M cross-domain | apr 29 | First like-for-like comparison with both methods at ~converged phase 1 (k_eval=160 + bonus=0.5 for SSH). W2v shared 0.251→0.230 (drift -0.021, 92% retention). SSH shared 0.154→0.089 (drift -0.065, 58% retention). Earlier "extrapolated SSH ~0.30" projection was wrong — actual SSH peak with all best practices is 0.154 on shared. **W2v wins absolute by ~63% AND retains better proportionally.** Cross drift roughly tied (+0.21 each). |
| Peak-detected SSH phase 1 + bonus=0.5 | apr 29 | Per-epoch SimLex eval + early-stop (patience=3); SSH peaked at ep4 (0.207 on shared), reverted from later epochs. **But phase-2 retention got WORSE**: ep4 model lost more in phase 2 (drift -0.143, 31% retention) than the noisier 8-epoch hardcoded model (drift -0.065, 58% retention). Plasticity-stability tradeoff: sharper phase-1 representations are more specialized to A's statistics → more disrupted by B's. Hogwild noise in the 8-epoch run acted as implicit regularization, producing a flatter, more transferable representation. |

## Open experimental questions

1. **Decay schedule**: should `decay` scale with corpus size, e.g.,
   `decay = const / sqrt(N)` to keep total shrinkage constant?
2. **Multi-seed at headline points**: confirm the +0.159 SimLex at 500k
   mod+decay is robust across seeds
3. ~~**Continual learning** with new vocabulary~~ — answered apr 28:
   SSH preserves better than w2v at small B (≤500k tokens), worse at
   large B (≥1M). Crossover at update size 500k–1M. The story isn't
   "SBR wins continual learning" — it's "SBR is more update-stable
   per token, but its k-bit aperture caps how much new information
   can be absorbed without overwriting old." More dimensions don't help.
   **Update**: a one-shot BCM-like consolidation bonus (λ=0.5 added
   to top-40 phase-1 bits at the phase-2 boundary) eliminates ~90%
   of the cross-domain forgetting at 5M B-tokens. Confirms the
   k-aperture bottleneck hypothesis: the issue isn't capacity, it's
   *plasticity asymmetry* — phase-1 bits need explicit protection
   to survive top-k competition. λ=1.0 is too strong; sigmoid bound
   makes phase-2 updates vanish on protected bits and the system
   can't form phase-2-aware representations.
4. **Performance**: lazy decay (quick win), batched-by-word update (bigger
   refactor)
5. **Theoretical analysis**: can we derive a closed-form fixed point for
   modulated SSH analogous to Oja's theorem?
6. **Capacity translation** (open): bits framing
   `D_dense · b_eff ≈ k · log₂(D_sparse · e / k)` predicts
   D_dense=100·float32 ≈ SDR(1024, 80–160), matching the empirical
   k_eval optimum. Needs a controlled experiment matching information
   budgets explicitly to test the equivalence.
7. **Consolidation follow-ups**: (a) does λ=0.5 generalize to b=1M
   and b=500k or does it over-protect? (b) skip uninitialized words
   (max(A_phase1[w]) < threshold) so B-only words are fully plastic
   in phase 2; (c) protect top-160 (matches eval aperture) instead
   of top-40; (d) sliding threshold per-bit activity counter (true
   BCM rather than one-shot snapshot).
8. **Multi-epoch SSH convergence**: does SSH show the same
   peak-then-decline curve as w2v over many epochs on the same corpus?
   Sequential at 5M is ~32 hours for 50 epochs (intractable);
   Hogwild parallel at 4 workers brings it to ~16 hours (overnight).
   Two outcomes are possible: (1) SSH peaks higher than w2v's 0.265
   when given convergence-time training (with k_eval=160, extrapolation
   suggests ~0.30+) — headline finding; (2) SSH peaks at the same
   place w2v does and declines similarly — equivalence finding. Either
   informative.
9. **Top-k vs dropout**: dropout-augmented SSH as a different fix
   for continual learning. BCM consolidation = "remember the specific
   subset that won at phase 1." Dropout-during-training = "make many
   subsets equally valid so phase-2 perturbation isn't catastrophic."
   One-line change: multiply A_w by a Bernoulli mask before computing
   top-k during training. Tests whether implicit-ensemble training
   makes SSH naturally robust to phase-2 reorganization.
10. **Proportional consolidation bonus**: the apr 29 plasticity-stability
    finding suggests λ=0.5 (absolute) is calibrated wrong for a more-mature
    accumulator. A `λ · A[w, i]` (proportional) bonus would scale with
    each bit's actual learned magnitude — well-trained bits get
    proportionally larger protection, untrained bits stay near zero.
    Could plausibly close the SSH-vs-w2v retention gap at peak-detected
    phase 1.
11. **Multi-epoch phase 2 with per-epoch eval**: currently both methods
    use 1-epoch phase 2. Methodologically incomplete but unlikely to
    change the qualitative finding (w2v's shared SimLex peaks at ep0
    of phase 2 since phase 1 was already optimal; SSH's might have
    a brief uptick before declining). Worth running for completeness
    (~1.5 hrs SSH + ~3 min w2v with cached phase-1 accumulator) but
    not load-bearing for the main story.

## Session insights (apr 28)

### The local-learning ↔ gradient-descent equivalence is exact for shallow models

For skip-gram with negative sampling, the loss `-log σ(v_w · v_c)`
factorizes such that `∂L/∂v_w[i] = -σ(-score) · v_c[i]`. The
gradient is **already zero** wherever `v_c[i] = 0`. So when v_c is a
sparse top-k binary code, "gradient descent with bottom (D-k) zeroed
out" is the same operation as "local Hebbian update on the support of
E_c." The two framings differ only in the modulator shape:

  - Word2vec / STE: m = σ(-score)  (sigmoid)
  - Modulated SSH:  m = (1 - overlap/k)  (linear)

Both saturate to 0 when fully aligned and 1 when fully misaligned. The
linear modulator avoids the σ(0)=0.5 bootstrap problem at init, which
explains the empirical gap we measured between `gradient_ste_baseline`
and `sparse_skipgram_hebbian_modulated_baseline` — same algorithm,
different schedule.

This dichotomy ("local learning" vs "gradient descent") only becomes
real in *deep* networks, where backprop introduces non-local
gradient signals through hidden layers. Shallow embedding methods like
word2vec and SSH are the same algorithm in different vocabularies.
arbora's Part-2 unifying equation captures this:

  Δw = η · m(·) · [f_pre · f_post − s(w, θ)]

with classical rules as different (m, f_pre, f_post, s) tuples.

### Top-k vs dropout: related, not dual

Both produce sparse activation patterns by zeroing some dimensions.
Difference:

  - Top-k = "this dimension *deserves* to be active" (value-based, deterministic)
  - Dropout = "this dimension *happens* to be active" (random, stochastic)

Top-k is for *the representation*; dropout is for *regularization*.
Dropout-trained models reconstruct a dense representation at inference
(via averaging across draws) — they do not give SDRs by themselves.

But there's a useful synthesis: **dropout-during-training plus top-k-at-readout**.
The implicit-ensemble interpretation of dropout (Srivastava 2014) says
the model learns redundant encodings — many subsets of bits all carry
usable information. For SSH continual learning, this would make the
phase-1 representation robust to *which* bits end up winning the top-k
post-phase-2 reorganization. Different mechanism from BCM consolidation
(which protects a specific subset); see open question #9.

### Memory bandwidth ceiling on Hogwild parallel SSH

4-worker speedup capped at ~2× because the 40 MB A-table working set
exceeds L3 cache. False sharing on overlapping rows ping-pongs cache
lines between cores. Gensim doesn't have a magic fix — its typical
D=100 keeps the working set at ~4 MB (fits in L3), so it scales to
~3× at 4 workers. At our D=1024 (or gensim's production V=1M, D=300
where syn0 is 1.2 GB), both hit the same wall. Fixes that work:
word-partitioned ownership (no shared writes), worker-local A +
periodic sync (Federated-style), or dimensionality reduction to
D=256–512.

### Plasticity-stability tradeoff in SSH continual learning (apr 29)

Surprising result from the peak-detection run: the SSH phase-1 model
with HIGHER absolute SimLex (peak-detected ep4 at 0.207) had WORSE
phase-2 retention (drift -0.143, 31% retained) than the same setup
with a noisier 8-epoch hardcoded phase-1 (SimLex 0.154 but drift
-0.065, 58% retained). More phase-1 training → sharper, more
specialized representation → more disrupted by phase-2 evidence.

Three readings:

1. **Specialization-driven forgetting**: well-tuned phase-1
   representations encode more A-specific structure, which is exactly
   what phase 2 disrupts. Less-tuned representations are more
   "generic" and less attached to A's statistics.

2. **Hogwild noise as implicit regularization**: the 8-epoch run
   accumulated race-condition noise across passes, producing a
   flatter accumulator that's robust to phase-2 perturbation. The
   peak-detected ep4 captured a sharper local maximum that's less
   stable.

3. **Consolidation bonus calibration depends on phase-1 maturity**:
   λ=0.5 was sweet-spotted on the under-trained 1-epoch phase 1 where
   accumulator values are still small. For a more-mature accumulator,
   the proportional protection is weaker — the *relative* gap
   between protected and unprotected bits shrinks. A scaled
   consolidation bonus (λ proportional to per-bit A magnitude) is
   the natural fix.

This is a known phenomenon in continual learning literature
("plasticity-stability dilemma" — Mermillod et al. 2013, Parisi et al.
2019) but we observed it here in a setup where the same SSH
algorithm and same consolidation mechanism produced opposite
continual-learning behavior depending on phase-1 training depth. Tells
us that "more phase-1 training is better" is not always true under
this measurement.

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
