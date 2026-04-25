"""Generate skeleton notebooks for ARB-139 parts 2-5.

Each skeleton is markdown-only with the planned section structure, so
parts can be opened in Jupyter even before they're filled in. As we
build out each part in conversation, replace the skeleton with a
proper build_partN.py.

Run: uv run python notebooks/build_skeletons.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def build_skeleton(filename: str, title: str, sections: list[tuple[str, str]]) -> None:
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_markdown_cell(
        f"# {title}\n\n"
        "**Status: skeleton.** Section structure committed; content to be filled in "
        "as we build through this part in conversation. See `notebooks/PLAN.md` for "
        "the overall plan."
    ))

    for section_title, section_body in sections:
        cells.append(nbf.v4.new_markdown_cell(
            f"## {section_title}\n\n{section_body}"
        ))

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    }
    out_path = Path(__file__).parent / filename
    nbf.write(nb, out_path)
    print(f"Wrote {out_path}  ({len(cells)} cells)")


# Part 2 — Oja's rule and the unifying equation
build_skeleton(
    "arb139_part2_oja_and_unification.ipynb",
    "ARB-139 Part 2 — Oja's Rule and the Unifying Equation",
    [
        ("Why plain Hebbian explodes",
         "Set up the boundedness problem. Demonstrate on synthetic 2D data: "
         "plain `dw = η · pre · post` causes weights to grow without bound."),
        ("Oja's rule",
         "Introduce `dw = η · y · (x − y · w)`. Why the `−y²w` term saves us. "
         "Geometric intuition: shrinkage along the weight direction."),
        ("Oja's theorem",
         "**Theorem**: Oja's rule converges to the principal eigenvector of the input "
         "covariance matrix, with `||w|| → 1`. Local learning has provable convergence. "
         "Hebbian + the right decay term *is* streaming PCA."),
        ("Sanger's rule (Generalized Hebbian Algorithm)",
         "Extension of Oja to multiple components. Streaming multi-component PCA "
         "with purely local updates per neuron."),
        ("The unifying parametric form",
         "All Hebbian variants written in one equation:\n"
         "$$\\Delta w = \\eta \\cdot m(\\cdot) \\cdot \\left[ f_{\\text{pre}}(x) \\cdot f_{\\text{post}}(y) - s(w, \\theta) \\right]$$\n\n"
         "- $m$: modulator (1 for vanilla, $\\sigma(-\\text{score})$ for word2vec, "
         "$1 - \\text{overlap}/k$ for SSH)\n"
         "- $f_{\\text{pre}}, f_{\\text{post}}$: pre/post activity nonlinearities\n"
         "- $s$: stability term (decay, subtractive normalization, BCM)"),
        ("Mapping classical rules to (m, f_pre, f_post, s)",
         "Table showing where Oja, Sanger, BCM, covariance, three-factor, and our "
         "modulated SSH each sit in this taxonomy."),
        ("Sanity check on synthetic data",
         "Implement Oja's rule, run on data drawn from a known covariance, watch "
         "the weight vector converge to the first eigenvector."),
        ("Reading",
         "Oja (1982), Sanger (1989), BCM (1982), Földiák (1990), Miller & MacKay (1994)."),
    ],
)


# Part 3 — Stability primitives
build_skeleton(
    "arb139_part3_stability_primitives.ipynb",
    "ARB-139 Part 3 — Stability Primitives in Detail",
    [
        ("Weight decay (Oja-like)",
         "Per-update uniform shrinkage. Lazy implementation via per-row last-decayed "
         "timestamp + amortized scaling. Demonstrate on SSH that decay magnitude must "
         "scale inversely with corpus size to be effective."),
        ("Subtractive normalization (Miller & MacKay)",
         "`w -= mean(w)` per row. Forces zero-sum competition: positive updates on "
         "some bits implicitly weaken others. Plot trajectory comparison vs Oja decay."),
        ("Synaptic scaling / row-norm bound",
         "Divisive normalization. The 'soft k-WTA' continuous relaxation. "
         "Connection to attention's softmax."),
        ("BCM sliding threshold",
         "Postsynaptic activity above threshold → LTP, below → LTD. Threshold slides "
         "with running activity (`θ_t = (1-α)θ_{t-1} + α·⟨y²⟩`). Prevents dead and "
         "runaway neurons. Demonstrate on a toy 'silent neuron' problem."),
        ("Interaction with modulators",
         "When does decay help in addition to surprise modulation? Empirical answer "
         "from our cross-scale decay test: scale-dependent — modulator handles small "
         "data, decay becomes useful at moderate-to-large data."),
        ("Map T1 onto these primitives",
         "T1 already has `synapse_decay=0.999` (Oja-like) and burst signal (modulator). "
         "What it lacks: per-cell BCM threshold, subtractive normalization. "
         "The unifying form makes T1's design choices legible."),
    ],
)


# Part 4 — Metrics
build_skeleton(
    "arb139_part4_metrics.ipynb",
    "ARB-139 Part 4 — Metrics: Semantically Simple, Representation-Agnostic",
    [
        ("The metrics we used in the sweep",
         "SimLex Spearman, analogy top-1, capacity (mean_sim, coll_frac, eff_dim), "
         "bundling capacity, corruption resilience, partial-cue retention, storage "
         "bytes, train cost. What each is, what it's good for, where it breaks."),
        ("Where the metrics don't transfer cleanly across CDR/SBR",
         "- SimLex via Jaccard has discrete-jump artifacts on binary codes\n"
         "- Bundling capacity bridges bit-OR (sparse) and vector-mean (dense) — different ops\n"
         "- Effective dim has different meaning in continuous vs binary codes\n"
         "- Corruption resilience uses different noise models for sparse vs dense"),
        ("Semantically simple, representation-agnostic versions",
         "- **Pair-similarity rank correlation**: pick a similarity function appropriate "
         "to the representation, then compute Spearman/Kendall against human ratings. "
         "Same protocol, different inner similarity.\n"
         "- **Bundling capacity as fraction recoverable**: unify dense and sparse via "
         "fraction-of-members-in-top-k metric, not raw margin.\n"
         "- **Continual-learning retention**: fraction of pre-task-B SimLex correlation "
         "preserved after continuing on task B.\n"
         "- **Corruption resilience**: area under degradation curve from 0% to 50% noise."),
        ("Reusable evaluation harness",
         "One function: `evaluate_all(emb, simlex, analogy, ...)`. Single seed in, "
         "stat dict out. Wrap with multi-seed averaging. Used for the multi-seed runs."),
        ("Confidence intervals and visual conventions",
         "How to report jumpy metrics: error bars at headline points, shaded bands for "
         "curves. Standard ML conventions adapted to the SBR/CDR comparison."),
    ],
)


# Part 5 — Implementation + blog post outline
build_skeleton(
    "arb139_part5_implementation_and_blog.ipynb",
    "ARB-139 Part 5 — Implementation Tradeoffs and Blog Post Outline",
    [
        ("Performance options ranked by effort/payoff",
         "1. **Lazy decay** (per-row timestamp + amortized shrinkage). Cheap. ~10-20% speedup.\n"
         "2. **Batched-by-word updates**. Snapshot codes, aggregate deltas, apply once. "
         "Bigger refactor. ~5-10x speedup possible.\n"
         "3. **Numba/Cython inner loop** (already in). Modest gain over numpy.\n"
         "4. **Heap-based top-k**. Asymptotic improvement on the dominant op. ~3-5x.\n"
         "5. **Multi-pass / mini-epoch** with frozen codes per epoch. Algorithmic change."),
        ("Mathematical insight vs ablation runs",
         "When does math tell you the answer without an experiment? Oja's theorem says "
         "Hebbian + decay → PCA — closed form. But interaction effects (modulator + decay + "
         "k-WTA) need ablation. Rule of thumb: closed forms exist for shallow Hebbian "
         "without competition; competition + nonlinearity forces empirical work."),
        ("Bringing primitives back to T1",
         "Refactor the cortex update rule into the unified form `Δw = η · m · (f · g − s)`. "
         "Win: interpretability + ablatability, not speed. Side benefit: cleaner story "
         "for papers/blogs."),
        ("Blog post outline",
         "**Hook**: word2vec works because of its stabilizer, not its update rule.\n\n"
         "**Section 1**: skip-gram is data shaping, sigmoid is triple-duty stabilizer\n\n"
         "**Section 2**: SSH retains the data shaping, swaps representation + update\n\n"
         "**Section 3**: empirical results — modulated+decay SSH crushes word2vec on "
         "small data, ties at moderate, loses at peak\n\n"
         "**Section 4**: the structural advantage — corruption resilience, storage cost, "
         "and especially **continual learning**\n\n"
         "**Section 5**: connection to existing work — Bricken/Anthropic SDM, HTM, VSA, "
         "Levy-Goldberg PMI\n\n"
         "**Section 6**: the deeper claim — gradient descent and modulated local Hebbian "
         "are the same algorithm in shallow models; arbora is the deep cortical "
         "embodiment of this"),
        ("Continual-learning experiment (the big payoff)",
         "Two probes: (1) shared vocab, new relationships; (2) novel vocabulary "
         "introduction. SSH should structurally win on both, especially #2. Per-word "
         "drift histogram is the visually compelling figure."),
    ],
)

print("All skeletons written.")
