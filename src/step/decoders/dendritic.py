"""Dendritic segment decoder: nonlinear pattern matching for token prediction.

Uses the same dendritic segment mechanism as the cortex — each "decoder
neuron" (one per observed token) has segments that learn to recognize
specific conjunctions of active L2/3 neurons. This tests whether our
representations support nonlinear decoding, matching how downstream
cortical regions would actually read them.

Learning follows the cortex pattern:
- Grow: find best-matching segment, strengthen active synapses, replace
  weakest inactive ones with connections to active sources.
- Reinforce: strengthen segments that correctly predicted.
- Punish: weaken segments that falsely predicted (optional).
"""

import numpy as np


class DendriticDecoder:
    """Decoder using dendritic segments to map L2/3 patterns to tokens."""

    def __init__(
        self,
        source_dim: int,
        *,
        n_segments: int = 4,
        n_synapses: int = 24,
        seg_threshold: int = 2,
        perm_threshold: float = 0.5,
        perm_init: float = 0.6,
        perm_increment: float = 0.2,
        perm_decrement: float = 0.05,
        seed: int = 0,
    ):
        self.source_dim = source_dim
        self.n_segments = n_segments
        self.n_synapses = n_synapses
        self.seg_threshold = seg_threshold
        self.perm_threshold = perm_threshold
        self.perm_init = perm_init
        self.perm_increment = perm_increment
        self.perm_decrement = perm_decrement
        self._rng = np.random.default_rng(seed)
        self._source_pool = np.arange(source_dim)

        # Lazy per-token segment storage: token_id -> (indices, permanences)
        self._neurons: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    @property
    def n_tokens(self) -> int:
        """Number of tokens the decoder has observed."""
        return len(self._neurons)

    def _alloc_neuron(self, token_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Allocate segment arrays for a new token."""
        indices = np.zeros(
            (self.n_segments, self.n_synapses), dtype=np.int32
        )
        for s in range(self.n_segments):
            indices[s] = self._rng.choice(
                self._source_pool, self.n_synapses,
                replace=self.source_dim < self.n_synapses,
            )
        perm = np.zeros((self.n_segments, self.n_synapses))
        self._neurons[token_id] = (indices, perm)
        return indices, perm

    def observe(self, token_id: int, l23_state: np.ndarray) -> None:
        """Learn: grow/reinforce segments for token_id given preceding L2/3 state.

        Call after seeing token_id, passing the L2/3 state from the
        PREVIOUS timestep (the pattern that should predict this token).
        """
        ctx = l23_state > 0 if l23_state.dtype != np.bool_ else l23_state
        if not ctx.any():
            return

        if token_id not in self._neurons:
            self._alloc_neuron(token_id)

        indices, perm = self._neurons[token_id]
        self._grow_best_segment(indices, perm, ctx)

    def decode(self, l23_state: np.ndarray, k: int = 5) -> list[int]:
        """Return top-k token predictions given current L2/3 state."""
        ctx = l23_state > 0 if l23_state.dtype != np.bool_ else l23_state
        if not ctx.any():
            return []

        if not self._neurons:
            return []

        # Batch all neurons: stack indices/perm, compute overlaps in one shot
        token_ids = list(self._neurons.keys())
        all_indices = np.stack([self._neurons[t][0] for t in token_ids])  # (N, n_seg, n_syn)
        all_perm = np.stack([self._neurons[t][1] for t in token_ids])
        active_at_syn = ctx[all_indices]  # (N, n_seg, n_syn)
        connected = all_perm > self.perm_threshold
        counts = (active_at_syn & connected).sum(axis=2)  # (N, n_seg)
        max_overlaps = counts.max(axis=1)  # (N,)

        # Filter by threshold and get top-k
        above = max_overlaps >= self.seg_threshold
        if not above.any():
            return []
        valid_idx = np.flatnonzero(above)
        valid_overlaps = max_overlaps[valid_idx]
        top = valid_idx[np.argsort(valid_overlaps)[::-1][:k]]
        return [token_ids[i] for i in top]

    def decode_scores(self, l23_state: np.ndarray) -> dict[int, int]:
        """Return all token scores (best segment overlap) for analysis."""
        ctx = l23_state > 0 if l23_state.dtype != np.bool_ else l23_state
        if not ctx.any():
            return {}

        if not self._neurons:
            return {}

        # Batch all neurons
        token_ids = list(self._neurons.keys())
        all_indices = np.stack([self._neurons[t][0] for t in token_ids])
        all_perm = np.stack([self._neurons[t][1] for t in token_ids])
        active_at_syn = ctx[all_indices]
        connected = all_perm > self.perm_threshold
        counts = (active_at_syn & connected).sum(axis=2)
        max_overlaps = counts.max(axis=1)

        return {
            token_ids[i]: int(max_overlaps[i])
            for i in range(len(token_ids))
            if max_overlaps[i] > 0
        }

    def _best_segment_overlap(
        self,
        indices: np.ndarray,
        perm: np.ndarray,
        ctx: np.ndarray,
    ) -> int:
        """Max overlap across segments: connected synapses with active sources."""
        active_at_syn = ctx[indices]  # (n_seg, n_syn)
        connected = perm > self.perm_threshold
        counts = (active_at_syn & connected).sum(axis=1)  # (n_seg,)
        return int(counts.max())

    def _grow_best_segment(
        self,
        indices: np.ndarray,
        perm: np.ndarray,
        ctx: np.ndarray,
    ) -> None:
        """Find best-matching segment, reinforce active synapses, grow new ones."""
        # Vectorized: find segment with most context overlap
        overlaps = ctx[indices].sum(axis=1)
        best_seg = int(overlaps.argmax())
        if overlaps[best_seg] <= 0:
            return

        idx = indices[best_seg].copy()
        p = perm[best_seg].copy()
        syn_active = ctx[idx]

        # Strengthen active, weaken inactive
        p[syn_active] = np.minimum(p[syn_active] + self.perm_increment, 1.0)
        p[~syn_active] = np.maximum(p[~syn_active] - self.perm_decrement, 0.0)

        # Grow: replace weakest inactive synapses with active sources
        active_sources = np.nonzero(ctx)[0]
        new_sources = active_sources[~np.isin(active_sources, idx)]

        if len(new_sources) > 0:
            inactive_slots = np.where(~syn_active)[0]
            if len(inactive_slots) > 0:
                order = np.argsort(p[inactive_slots])
                n_grow = min(len(new_sources), len(inactive_slots))
                slots = inactive_slots[order[:n_grow]]
                idx[slots] = new_sources[:n_grow]
                p[slots] = self.perm_init

        indices[best_seg] = idx
        perm[best_seg] = p
