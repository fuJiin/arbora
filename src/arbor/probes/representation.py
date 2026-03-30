"""Representation quality metrics for sensory cortex.

Measures whether the cortex is learning useful internal representations,
independent of any external decoder. These metrics answer:

1. Column selectivity: are columns becoming feature detectors?
2. Representational similarity: do similar tokens get similar columns?
3. Context discrimination: does L2/3 context differentiate same-token
   activations in different contexts?
4. Receptive field convergence: are ff_weights stabilizing?

These are the metrics that matter for building good representations
that downstream regions (motor cortex, PFC) can work with.
"""

from collections import Counter, defaultdict

import numpy as np


class RepresentationTracker:
    """Accumulates per-step activation data for representation analysis.

    Call observe() after each step. Call analysis methods at report time.
    Per-step cost is minimal (just appending to lists/counters).
    """

    def __init__(self, n_columns: int, n_l4: int):
        self.n_columns = n_columns
        self.n_l4 = n_l4

        # token_id -> list of column activation sets (one per occurrence)
        self._token_columns: dict[int, list[frozenset[int]]] = defaultdict(list)
        # token_id -> list of L4 neuron activation sets
        self._token_neurons: dict[int, list[frozenset[int]]] = defaultdict(list)
        # column -> Counter of token_ids that activated it
        self._column_tokens: dict[int, Counter] = defaultdict(Counter)

        # Context discrimination: (prev_token, token) -> neuron sets
        self._bigram_neurons: dict[tuple[int, int], list[frozenset[int]]] = defaultdict(
            list
        )

        self._prev_token_id: int | None = None
        self._n_steps = 0

    def observe(
        self,
        token_id: int,
        active_columns: np.ndarray,
        active_l4: np.ndarray,
    ) -> None:
        """Record one activation. Call after region.process()."""
        cols = frozenset(int(c) for c in np.nonzero(active_columns)[0])
        neurons = frozenset(int(n) for n in np.nonzero(active_l4)[0])

        self._token_columns[token_id].append(cols)
        self._token_neurons[token_id].append(neurons)

        for c in cols:
            self._column_tokens[c][token_id] += 1

        if self._prev_token_id is not None:
            key = (self._prev_token_id, token_id)
            self._bigram_neurons[key].append(neurons)

        self._prev_token_id = token_id
        self._n_steps += 1

    def reset_context(self) -> None:
        """Call on story boundary to prevent cross-story bigrams."""
        self._prev_token_id = None

    def column_selectivity(self) -> dict:
        """Per-column selectivity: how peaked is each column's response.

        Uses normalized entropy of token distribution per column:
        - 0.0 = responds to exactly one token (perfect feature detector)
        - 1.0 = responds uniformly to all tokens (no selectivity)

        Returns dict with per-column values and summary stats.
        """
        selectivities = []
        n_tokens = len(self._token_columns)
        if n_tokens <= 1:
            return {
                "per_column": [],
                "mean": 0.0,
                "std": 0.0,
                "best_5": [],
            }

        max_entropy = np.log2(n_tokens)
        if max_entropy == 0:
            max_entropy = 1.0

        for col in range(self.n_columns):
            counts = self._column_tokens.get(col)
            if not counts:
                selectivities.append(1.0)  # never fired = no selectivity
                continue
            total = sum(counts.values())
            probs = np.array(list(counts.values()), dtype=np.float64)
            probs /= total
            entropy = -float(np.sum(probs * np.log2(probs)))
            selectivities.append(entropy / max_entropy)

        sel = np.array(selectivities)
        # Best columns = lowest normalized entropy
        best_indices = np.argsort(sel)[:5]
        best_5 = [
            (int(i), float(sel[i]), len(self._column_tokens.get(i, {})))
            for i in best_indices
        ]

        return {
            "per_column": selectivities,
            "mean": float(np.mean(sel)),
            "std": float(np.std(sel)),
            "best_5": best_5,
        }

    def representation_similarity(self, top_n: int = 50) -> dict:
        """Pairwise column overlap between most frequent tokens.

        Computes Jaccard similarity of column sets for each token pair.
        Good representations have non-trivial structure: neither all
        identical (collapsed) nor all disjoint (no generalization).

        Returns similarity stats and whether structure is non-trivial.
        """
        # Find most frequent tokens
        token_freq = {tid: len(cols) for tid, cols in self._token_columns.items()}
        if len(token_freq) < 2:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "n_tokens": 0,
                "nontrivial": False,
            }

        top_tokens = sorted(token_freq, key=lambda t: token_freq[t], reverse=True)[
            :top_n
        ]

        # For each token, compute its "typical" column set (most frequent)
        token_col_profile: dict[int, set[int]] = {}
        for tid in top_tokens:
            # Union of all column activations, weighted by frequency
            col_counts: Counter = Counter()
            for cols in self._token_columns[tid]:
                for c in cols:
                    col_counts[c] += 1
            # Take columns that activated > 50% of the time
            threshold = len(self._token_columns[tid]) * 0.3
            token_col_profile[tid] = {
                c for c, n in col_counts.items() if n >= threshold
            }

        # Pairwise Jaccard similarity
        sims = []
        for i, t1 in enumerate(top_tokens):
            for t2 in top_tokens[i + 1 :]:
                s1 = token_col_profile[t1]
                s2 = token_col_profile[t2]
                union = len(s1 | s2)
                if union == 0:
                    sims.append(0.0)
                else:
                    sims.append(len(s1 & s2) / union)

        sims_arr = np.array(sims) if sims else np.array([0.0])

        # Non-trivial = not all same and not all different
        mean = float(np.mean(sims_arr))
        std = float(np.std(sims_arr))
        nontrivial = std > 0.01 and 0.01 < mean < 0.99

        return {
            "mean": mean,
            "std": std,
            "min": float(np.min(sims_arr)),
            "max": float(np.max(sims_arr)),
            "n_tokens": len(top_tokens),
            "nontrivial": nontrivial,
        }

    def context_discrimination(self, min_contexts: int = 3) -> dict:
        """How much does neuron pattern vary for same token in different
        preceding contexts.

        For each token seen in >= min_contexts different bigram contexts,
        measures pairwise Jaccard distance between neuron activation sets.
        High distance = L2/3 context is successfully differentiating.

        Returns mean discrimination and per-token breakdown for top tokens.
        """
        # Group by target token: which different contexts produced which
        # neuron patterns?
        token_context_patterns: dict[int, list[frozenset[int]]] = defaultdict(list)
        # Track which preceding tokens we've seen for each target
        token_contexts: dict[int, set[int]] = defaultdict(set)

        for (prev_tid, tid), patterns in self._bigram_neurons.items():
            token_contexts[tid].add(prev_tid)
            token_context_patterns[tid].extend(patterns)

        # Filter to tokens with enough different contexts
        eligible = {
            tid: patterns
            for tid, patterns in token_context_patterns.items()
            if len(token_contexts[tid]) >= min_contexts
            and len(patterns) >= min_contexts
        }

        if not eligible:
            return {
                "mean_discrimination": 0.0,
                "n_eligible_tokens": 0,
                "per_token": [],
            }

        per_token = []
        all_discriminations = []

        for tid in sorted(eligible, key=lambda t: -len(eligible[t]))[:20]:
            patterns = eligible[tid]
            # Sample pairwise Jaccard distances
            dists = []
            n = len(patterns)
            pairs = min(50, n * (n - 1) // 2)
            rng = np.random.default_rng(tid)
            for _ in range(pairs):
                i, j = rng.choice(n, 2, replace=False)
                s1, s2 = patterns[i], patterns[j]
                union = len(s1 | s2)
                if union > 0:
                    dists.append(1.0 - len(s1 & s2) / union)
                else:
                    dists.append(0.0)

            mean_dist = float(np.mean(dists)) if dists else 0.0
            per_token.append(
                {
                    "token_id": tid,
                    "n_contexts": len(token_contexts[tid]),
                    "n_observations": len(patterns),
                    "discrimination": mean_dist,
                }
            )
            all_discriminations.append(mean_dist)

        return {
            "mean_discrimination": float(np.mean(all_discriminations)),
            "n_eligible_tokens": len(eligible),
            "per_token": per_token,
        }

    def ff_convergence(self, ff_weights: np.ndarray) -> dict:
        """Measure how well-formed receptive fields are.

        Checks:
        - Weight sparsity: fraction of near-zero weights (sharper = better)
        - Per-column weight entropy: peaked = specialized feature detector
        - Cross-column similarity: low = columns detect different features
        """
        n_cols = ff_weights.shape[1]

        # Per-column weight distributions
        col_entropies = []
        col_sparsities = []
        for col in range(n_cols):
            w = ff_weights[:, col]
            nonzero = w[w > 0.01]
            col_sparsities.append(1.0 - len(nonzero) / len(w))
            if len(nonzero) > 0:
                p = nonzero / nonzero.sum()
                ent = -float(np.sum(p * np.log2(p + 1e-10)))
                max_ent = np.log2(len(nonzero)) if len(nonzero) > 1 else 1.0
                col_entropies.append(ent / max_ent if max_ent > 0 else 0.0)
            else:
                col_entropies.append(1.0)

        # Cross-column cosine similarity (are columns learning different
        # features?)
        norms = np.linalg.norm(ff_weights, axis=0)
        valid = norms > 1e-8
        cosines = []
        if valid.sum() >= 2:
            normed = ff_weights[:, valid] / norms[valid]
            gram = normed.T @ normed
            # Upper triangle only (excluding diagonal)
            triu = np.triu_indices(gram.shape[0], k=1)
            cosines = gram[triu].tolist()

        cos_arr = np.array(cosines) if cosines else np.array([0.0])

        return {
            "weight_sparsity": float(np.mean(col_sparsities)),
            "rf_entropy_mean": float(np.mean(col_entropies)),
            "cross_col_cosine_mean": float(np.mean(cos_arr)),
            "cross_col_cosine_std": float(np.std(cos_arr)),
        }

    def summary(self, ff_weights: np.ndarray | None = None) -> dict:
        """Compute all representation metrics."""
        sel = self.column_selectivity()
        sim = self.representation_similarity()
        ctx = self.context_discrimination()

        result = {
            "column_selectivity_mean": sel["mean"],
            "column_selectivity_std": sel["std"],
            "similarity_mean": sim["mean"],
            "similarity_std": sim["std"],
            "similarity_nontrivial": sim["nontrivial"],
            "context_discrimination": ctx["mean_discrimination"],
            "context_n_eligible": ctx["n_eligible_tokens"],
            "n_unique_tokens": len(self._token_columns),
            "n_steps": self._n_steps,
        }

        if ff_weights is not None:
            conv = self.ff_convergence(ff_weights)
            result.update(
                {
                    "ff_sparsity": conv["weight_sparsity"],
                    "rf_entropy": conv["rf_entropy_mean"],
                    "ff_cross_col_cosine": conv["cross_col_cosine_mean"],
                }
            )

        return result

    def print_report(self, ff_weights: np.ndarray | None = None) -> None:
        """Print human-readable representation quality report."""
        print("\n--- Representation Quality ---")

        sel = self.column_selectivity()
        print("\nColumn selectivity (0=perfect detector, 1=uniform):")
        print(f"  mean={sel['mean']:.3f} std={sel['std']:.3f}")
        if sel["best_5"]:
            print("  most selective columns:")
            for col, entropy, n_tokens in sel["best_5"]:
                print(f"    col {col}: entropy={entropy:.3f} ({n_tokens} tokens)")

        sim = self.representation_similarity()
        print(f"\nRepresentational similarity ({sim['n_tokens']} tokens):")
        print(
            f"  Jaccard: mean={sim['mean']:.3f} std={sim['std']:.3f}"
            f" range=[{sim['min']:.3f}, {sim['max']:.3f}]"
        )
        if sim["nontrivial"]:
            print("  Structure: NON-TRIVIAL (good)")
        else:
            if sim["std"] <= 0.01:
                print("  Structure: COLLAPSED (all similar)")
            else:
                print("  Structure: DEGENERATE")

        ctx = self.context_discrimination()
        print(f"\nContext discrimination ({ctx['n_eligible_tokens']} tokens):")
        print(
            f"  mean Jaccard distance={ctx['mean_discrimination']:.3f}"
            f" (1=max discrimination)"
        )
        if ctx["per_token"][:5]:
            print("  top tokens:")
            for t in ctx["per_token"][:5]:
                print(
                    f"    token {t['token_id']}: "
                    f"disc={t['discrimination']:.3f} "
                    f"({t['n_contexts']} contexts, "
                    f"{t['n_observations']} obs)"
                )

        if ff_weights is not None:
            conv = self.ff_convergence(ff_weights)
            print("\nReceptive field quality:")
            print(f"  weight sparsity: {conv['weight_sparsity']:.3f}")
            print(f"  RF entropy (normalized): {conv['rf_entropy_mean']:.3f}")
            print(
                f"  cross-column cosine: "
                f"mean={conv['cross_col_cosine_mean']:.3f} "
                f"std={conv['cross_col_cosine_std']:.3f}"
            )
            if conv["cross_col_cosine_mean"] > 0.9:
                print("  WARNING: columns not differentiated")
