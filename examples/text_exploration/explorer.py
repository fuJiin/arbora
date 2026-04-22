"""Explorer views + state wrapper for T1 interactive exploration.

PR C of ARB-131 M1. Goal is intuition, not polish — keep views minimal:
one function per view, each returning a `plotly.graph_objects.Figure`
that Jupyter renders inline.

Views (from the ticket):
1. `spike_raster` — lamina activations across recent timesteps.
2. `sdr_overlap_matrix` — pairwise Jaccard heatmap of L2/3 SDRs per char.
3. `weight_histograms` — ff + segment-permanence distributions.
4. `dendritic_segment_view` — basal segments for a chosen L2/3 neuron.
5. `pca_population` — 2D PCA projection of L2/3 history.

`ExplorerState` wraps a `T1Trainer` to add per-step history of L4/L2/3/L5
active patterns (needed for the raster + PCA), plus named
snapshot/restore via `copy.deepcopy`.

Dependencies: `plotly`, `scikit-learn` for PCA. Install with the
`explorer` extras: `uv sync --extra explorer`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from examples.text_exploration.diagnostics import (
    VOWELS,
    _jaccard,
    character_sdr_overlap,
)
from examples.text_exploration.trainer import StepResult, T1Trainer


@dataclass
class ExplorerState:
    """Drive a T1Trainer with per-step history + named snapshots.

    History lists grow unbounded — call `clear_history()` periodically
    in long runs, or use a bounded-window view from the view functions.
    """

    trainer: T1Trainer
    history: list[StepResult] = field(default_factory=list)
    l4_history: list[np.ndarray] = field(default_factory=list)
    l23_history: list[np.ndarray] = field(default_factory=list)
    l5_history: list[np.ndarray] = field(default_factory=list)
    snapshots: dict[str, T1Trainer] = field(default_factory=dict)

    def step(self, char: str, *, train: bool = True) -> StepResult:
        r = self.trainer.step(char, train=train)
        self.history.append(r)
        self.l4_history.append(self.trainer.region.l4.active.copy())
        self.l23_history.append(self.trainer.region.l23.active.copy())
        self.l5_history.append(self.trainer.region.l5.active.copy())
        return r

    def reset(self) -> None:
        self.trainer.reset()

    def step_word(self, word: str, *, train: bool = True) -> list[StepResult]:
        self.reset()
        return [self.step(c, train=train) for c in word]

    def clear_history(self) -> None:
        self.history.clear()
        self.l4_history.clear()
        self.l23_history.clear()
        self.l5_history.clear()

    def snapshot(self, name: str) -> None:
        """Save a deepcopy of the trainer under `name`."""
        self.snapshots[name] = copy.deepcopy(self.trainer)

    def restore(self, name: str) -> None:
        """Replace the current trainer with a deepcopy of the named snapshot.

        Also clears history — the restored state starts a new timeline.
        """
        if name not in self.snapshots:
            raise KeyError(f"no snapshot named {name!r}")
        self.trainer = copy.deepcopy(self.snapshots[name])
        self.clear_history()


# ---------------------------------------------------------------------------
# Lazy plotly import so importing `explorer` doesn't require the extras
# ---------------------------------------------------------------------------


def _plotly():
    """Lazy import. Raises a clear error if `plotly` isn't installed."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as e:
        raise ImportError(
            "explorer views require plotly. Install with `uv sync --extra explorer`."
        ) from e
    return go, make_subplots


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------


def spike_raster(state: ExplorerState, window: int = 50):
    """Lamina activations across recent timesteps.

    Rows = neurons stacked L4 / L2/3 / L5 (separated by red lines),
    cols = timesteps (most recent on the right). Binary heatmap.
    """
    go, _ = _plotly()
    if not state.history:
        return go.Figure().update_layout(
            title="spike_raster: history is empty — step first"
        )

    w = state.history[-window:]
    l4 = np.stack(state.l4_history[-window:])
    l23 = np.stack(state.l23_history[-window:])
    l5 = np.stack(state.l5_history[-window:])
    combined = np.concatenate([l4, l23, l5], axis=1).T.astype(int)
    chars = [r.char for r in w]

    fig = go.Figure(
        data=go.Heatmap(
            z=combined,
            x=chars,
            colorscale=[[0, "white"], [1, "black"]],
            showscale=False,
            hovertemplate="t=%{x}<br>neuron=%{y}<br>active=%{z}<extra></extra>",
        )
    )
    n_l4 = l4.shape[1]
    n_l23 = l23.shape[1]
    fig.add_hline(y=n_l4 - 0.5, line_color="red", line_width=1)
    fig.add_hline(y=n_l4 + n_l23 - 0.5, line_color="red", line_width=1)
    fig.update_layout(
        title=f"Spike raster — last {len(w)} steps (L4 / L2·3 / L5 separated)",
        xaxis_title="char at step",
        yaxis_title="neuron",
        height=600,
    )
    return fig


def sdr_overlap_matrix(
    state: ExplorerState,
    chars: str | None = None,
):
    """Side-by-side L4 and L2/3 pairwise-Jaccard heatmaps.

    Showing both laminae disambiguates where a representation collapse
    is happening: if L4 already looks uniform, ff_weights are the
    problem; if L4 is sharp but L2/3 collapses, the L4→L2/3 projection
    is smushing things together.

    Chars are reordered so vowels come first; a red cross marks the
    vowel/consonant divider on both axes so phonetic clustering (or
    lack of it) is visually obvious.
    """
    go, make_subplots = _plotly()
    from examples.text_exploration.data import DEFAULT_ALPHABET

    if chars is None:
        chars = VOWELS + "".join(c for c in DEFAULT_ALPHABET if c not in VOWELS)

    result = character_sdr_overlap(state.trainer, chars)
    n = len(chars)

    def _heatmap(lamina_stats) -> np.ndarray:
        m = np.zeros((n, n))
        per_char = lamina_stats.per_char_sdr
        for i, c1 in enumerate(chars):
            for j, c2 in enumerate(chars):
                m[i, j] = _jaccard(per_char[c1], per_char[c2])
        return m

    l4_matrix = _heatmap(result.l4)
    l23_matrix = _heatmap(result.l23)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"L4 (within-vowel={result.l4.within_vowel_mean:.2f}, "
            f"across={result.l4.across_mean:.2f})",
            f"L2/3 (within-vowel={result.l23.within_vowel_mean:.2f}, "
            f"across={result.l23.across_mean:.2f})",
        ),
        horizontal_spacing=0.12,
    )
    for col_idx, matrix in enumerate([l4_matrix, l23_matrix], start=1):
        fig.add_trace(
            go.Heatmap(
                z=matrix,
                x=list(chars),
                y=list(chars),
                colorscale="Viridis",
                zmin=0.0,
                zmax=1.0,
                showscale=(col_idx == 2),
                hovertemplate="%{y} vs %{x}<br>Jaccard=%{z:.3f}<extra></extra>",
            ),
            row=1,
            col=col_idx,
        )
        fig.update_xaxes(title_text="char", row=1, col=col_idx)
        fig.update_yaxes(title_text="char", row=1, col=col_idx, autorange="reversed")
        # Vowel/consonant dividers.
        fig.add_vline(
            x=len(VOWELS) - 0.5, line_color="red", line_width=1, row=1, col=col_idx
        )
        fig.add_hline(
            y=len(VOWELS) - 0.5, line_color="red", line_width=1, row=1, col=col_idx
        )
    fig.update_layout(
        title="SDR pairwise Jaccard — L4 vs L2/3 (reset + 1 char)",
        height=550,
    )
    return fig


def weight_histograms(state: ExplorerState):
    """Histograms of ff_weights + segment permanences."""
    go, make_subplots = _plotly()
    region = state.trainer.region

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "ff_weights (learnable only)",
            "l4_lat_seg_perm",
            "l23_seg_perm",
            "l5_seg_perm",
        ),
    )
    ff = region.ff_weights[region.ff_mask]
    fig.add_trace(go.Histogram(x=ff, nbinsx=40), row=1, col=1)
    fig.add_trace(
        go.Histogram(x=region.l4_lat_seg_perm.reshape(-1), nbinsx=40), row=1, col=2
    )
    fig.add_trace(
        go.Histogram(x=region.l23_seg_perm.reshape(-1), nbinsx=40), row=2, col=1
    )
    if region.l5_seg_perm.size > 0:
        fig.add_trace(
            go.Histogram(x=region.l5_seg_perm.reshape(-1), nbinsx=40), row=2, col=2
        )
    fig.update_layout(
        title="Weight / permanence distributions",
        height=600,
        showlegend=False,
        bargap=0.05,
    )
    return fig


def dendritic_segment_view(state: ExplorerState, neuron_idx: int):
    """For a chosen L2/3 neuron, plot segment-wise permanences.

    Bar colors:
      - red   = source is active AND perm > threshold (actively firing)
      - blue  = perm > threshold but source not active
      - gray  = perm <= threshold (unused synapse)

    A horizontal dashed line marks `perm_threshold`. Segments that have
    enough red bars above threshold will be the ones predicting the
    current step.
    """
    go, make_subplots = _plotly()
    region = state.trainer.region
    if not 0 <= neuron_idx < region.n_l23_total:
        raise ValueError(
            f"neuron_idx {neuron_idx} out of range [0, {region.n_l23_total})"
        )

    indices = region.l23_seg_indices[neuron_idx]  # (n_segs, n_syn)
    perms = region.l23_seg_perm[neuron_idx]
    ctx = region._pred_context_l23

    n_segs = indices.shape[0]
    fig = make_subplots(
        rows=n_segs, cols=1, subplot_titles=[f"seg {i}" for i in range(n_segs)]
    )
    for s in range(n_segs):
        srcs = indices[s]
        perm_s = perms[s]
        active_src = ctx[srcs]
        above = perm_s > region.perm_threshold
        colors = [
            "red" if a and ab else ("blue" if ab else "lightgray")
            for a, ab in zip(active_src, above, strict=True)
        ]
        fig.add_trace(
            go.Bar(
                x=srcs,
                y=perm_s,
                marker_color=colors,
                showlegend=False,
            ),
            row=s + 1,
            col=1,
        )
        fig.add_hline(
            y=region.perm_threshold,
            row=s + 1,
            col=1,
            line_dash="dash",
            line_color="black",
            line_width=1,
        )
    fig.update_layout(
        title=f"Basal dendritic segments for L2/3 neuron {neuron_idx}",
        height=180 * n_segs,
    )
    return fig


def pca_population(state: ExplorerState, window: int | None = None):
    """2D PCA of the L2/3 population vectors from `l23_history`.

    If `window` is set, only the last `window` steps are used.
    """
    go, _ = _plotly()
    try:
        from sklearn.decomposition import PCA
    except ImportError as e:
        raise ImportError(
            "pca_population requires scikit-learn. Install with "
            "`uv sync --extra explorer` or ensure dev extras are installed."
        ) from e

    l23 = state.l23_history if window is None else state.l23_history[-window:]
    if len(l23) < 2:
        return go.Figure().update_layout(
            title="pca_population: need ≥ 2 history points — step first"
        )

    X = np.stack(l23).astype(np.float32)
    # Guard: if all rows identical, PCA would error.
    if X.std(axis=0).sum() == 0:
        return go.Figure().update_layout(
            title="pca_population: no variance in L2/3 history"
        )

    n_components = min(2, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=n_components)
    X2 = pca.fit_transform(X)
    chars = [
        r.char for r in (state.history if window is None else state.history[-window:])
    ]
    x = X2[:, 0]
    y = X2[:, 1] if X2.shape[1] > 1 else np.zeros(X2.shape[0])

    fig = go.Figure(
        data=go.Scatter(
            x=x,
            y=y,
            mode="markers+text",
            text=chars,
            textposition="top center",
            marker=dict(
                size=9,
                color=list(range(len(chars))),
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="step"),
            ),
            hovertemplate="t=%{marker.color}<br>char=%{text}<extra></extra>",
        )
    )
    explained = pca.explained_variance_ratio_
    ev_text = " + ".join(f"{r:.2f}" for r in explained)
    fig.update_layout(
        title=f"PCA of L2/3 (n={len(l23)}, explained={ev_text})",
        xaxis_title="PC1",
        yaxis_title="PC2",
        height=500,
    )
    return fig
