"""Tests for `examples.text_exploration.explorer` (ARB-131 PR C).

The views produce plotly Figures — we don't assert visual correctness,
just that each function produces a Figure without raising on a trainer
with realistic history. Visual verification lives in `explorer.ipynb`.
"""

from __future__ import annotations

import pytest

plotly = pytest.importorskip("plotly")

from arbora.config import _default_t1_config, make_sensory_region  # noqa: E402
from arbora.decoders.dendritic import DendriticDecoder  # noqa: E402
from arbora.encoders.charbit import CharbitEncoder  # noqa: E402
from examples.text_exploration.data import DEFAULT_ALPHABET  # noqa: E402
from examples.text_exploration.explorer import (  # noqa: E402
    ExplorerState,
    dendritic_segment_view,
    pca_population,
    sdr_overlap_matrix,
    spike_raster,
    weight_histograms,
)
from examples.text_exploration.trainer import T1Trainer  # noqa: E402


@pytest.fixture
def state() -> ExplorerState:
    """Explorer state with a few steps of history."""
    encoder = CharbitEncoder(length=1, width=27, chars=DEFAULT_ALPHABET)
    cfg = _default_t1_config()
    cfg.n_columns = 32
    cfg.k_columns = 4
    region = make_sensory_region(cfg, input_dim=encoder.input_dim, seed=0)
    decoder = DendriticDecoder(source_dim=region.n_l23_total, seed=0)
    trainer = T1Trainer(region, encoder, decoder)
    s = ExplorerState(trainer=trainer)
    for w in ["cat", "dog", "sun", "run"]:
        s.step_word(w)
    return s


class TestExplorerState:
    def test_step_appends_history(self, state: ExplorerState):
        n_before = len(state.history)
        state.step("z")
        assert len(state.history) == n_before + 1
        assert len(state.l4_history) == n_before + 1
        assert len(state.l23_history) == n_before + 1
        assert len(state.l5_history) == n_before + 1

    def test_step_word_resets_between_words(self, state: ExplorerState):
        state.step_word("hi")
        # After reset + 2 steps, region.l23.active reflects second char
        assert state.trainer.region.l23.active.any()

    def test_clear_history(self, state: ExplorerState):
        assert state.history
        state.clear_history()
        assert state.history == []
        assert state.l23_history == []

    def test_snapshot_and_restore_roundtrip(self, state: ExplorerState):
        """Snapshot then mutate, then restore, and mutations should be gone."""
        state.snapshot("before")
        ff_at_snapshot = state.trainer.region.ff_weights.copy()

        # Mutate: run more training.
        for w in ["big", "run"]:
            state.step_word(w)
        # Training changed weights.
        assert not (state.trainer.region.ff_weights == ff_at_snapshot).all()

        state.restore("before")
        # Weights back to snapshot value.
        assert (state.trainer.region.ff_weights == ff_at_snapshot).all()
        # History cleared.
        assert state.history == []

    def test_restore_nonexistent_raises(self, state: ExplorerState):
        with pytest.raises(KeyError):
            state.restore("nope")

    def test_snapshot_is_independent_deepcopy(self, state: ExplorerState):
        """Mutating current trainer must not mutate the snapshot."""
        state.snapshot("s")
        snap_ff = state.snapshots["s"].region.ff_weights.copy()
        state.step_word("big")
        # Snapshot weights unchanged despite trainer weights changing.
        assert (state.snapshots["s"].region.ff_weights == snap_ff).all()


class TestViewsSmoke:
    def test_spike_raster_returns_figure(self, state: ExplorerState):
        fig = spike_raster(state, window=10)
        assert hasattr(fig, "data")

    def test_spike_raster_empty_history(self):
        encoder = CharbitEncoder(length=1, width=27, chars=DEFAULT_ALPHABET)
        region = make_sensory_region(
            _default_t1_config(), input_dim=encoder.input_dim, seed=0
        )
        decoder = DendriticDecoder(source_dim=region.n_l23_total, seed=0)
        s = ExplorerState(trainer=T1Trainer(region, encoder, decoder))
        fig = spike_raster(s)
        # Renders a placeholder rather than raising.
        assert "empty" in (fig.layout.title.text or "").lower()

    def test_sdr_overlap_matrix(self, state: ExplorerState):
        fig = sdr_overlap_matrix(state, chars="aeioubcdf")
        assert hasattr(fig, "data")
        assert len(fig.data) == 1  # heatmap

    def test_weight_histograms(self, state: ExplorerState):
        fig = weight_histograms(state)
        # Should have at least 3 histograms: ff, l4_lat, l23.
        assert len(fig.data) >= 3

    def test_dendritic_segment_view(self, state: ExplorerState):
        fig = dendritic_segment_view(state, neuron_idx=0)
        assert hasattr(fig, "data")

    def test_dendritic_segment_view_out_of_range(self, state: ExplorerState):
        with pytest.raises(ValueError):
            dendritic_segment_view(state, neuron_idx=10**9)

    def test_pca_population(self, state: ExplorerState):
        fig = pca_population(state)
        assert hasattr(fig, "data")

    def test_pca_population_insufficient_history(self):
        encoder = CharbitEncoder(length=1, width=27, chars=DEFAULT_ALPHABET)
        region = make_sensory_region(
            _default_t1_config(), input_dim=encoder.input_dim, seed=0
        )
        decoder = DendriticDecoder(source_dim=region.n_l23_total, seed=0)
        s = ExplorerState(trainer=T1Trainer(region, encoder, decoder))
        fig = pca_population(s)
        # Renders placeholder, no raise.
        assert "need" in (fig.layout.title.text or "").lower()
