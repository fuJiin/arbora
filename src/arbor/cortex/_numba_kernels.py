"""Numba-accelerated kernels for dendritic segment operations.

These are standalone @njit functions that operate on raw numpy arrays.
Called from CorticalRegion methods to replace hot Python/NumPy paths
that create large temporaries or require per-element branching.

Import pattern:
    from arbor.cortex._numba_kernels import predict_segments, ...

Falls back gracefully if numba is not installed (pure numpy path remains).
"""

import numpy as np
from numba import njit, prange, types  # noqa: F401


@njit(cache=True)
def predict_segments(
    active_source: np.ndarray,  # (source_dim,) bool
    seg_indices: np.ndarray,  # (n_neurons, n_segments, n_synapses) int32
    seg_perm: np.ndarray,  # (n_neurons, n_segments, n_synapses) float64
    perm_threshold: float,
    seg_threshold: int,
) -> np.ndarray:
    """Check which neurons have active dendritic segments.

    Replaces the NumPy path:
        active_at_syn = active_source[seg_indices]
        connected = seg_perm > perm_threshold
        counts = (active_at_syn & connected).sum(axis=2)
        predicted = (counts >= seg_threshold).any(axis=1)

    This avoids allocating the (n_neurons, n_seg, n_syn) temporary arrays.
    """
    n_neurons = seg_indices.shape[0]
    n_segments = seg_indices.shape[1]
    n_synapses = seg_indices.shape[2]
    predicted = np.zeros(n_neurons, dtype=np.bool_)

    for i in range(n_neurons):
        for s in range(n_segments):
            count = 0
            for syn in range(n_synapses):
                if (
                    active_source[seg_indices[i, s, syn]]
                    and seg_perm[i, s, syn] > perm_threshold
                ):
                    count += 1
                    if count >= seg_threshold:
                        predicted[i] = True
                        break
            if predicted[i]:
                break

    return predicted


@njit(cache=True)
def grow_segment(
    neuron: int,
    seg_indices: np.ndarray,  # (n_neurons, n_segments, n_synapses)
    seg_perm: np.ndarray,  # (n_neurons, n_segments, n_synapses)
    ctx: np.ndarray,  # (source_dim,) bool
    pool: np.ndarray,  # (pool_size,) int — valid source indices
    perm_increment: float,
    perm_decrement: float,
    perm_init: float,
):
    """Grow the best-matching segment for a bursting neuron.

    Finds segment with most context overlap, strengthens matching
    synapses, weakens non-matching, replaces weakest inactive with
    new connections to active sources.
    """
    n_segments = seg_indices.shape[1]
    n_synapses = seg_indices.shape[2]

    # Find best-matching segment
    best_seg = 0
    best_overlap = 0
    for s in range(n_segments):
        overlap = 0
        for syn in range(n_synapses):
            if ctx[seg_indices[neuron, s, syn]]:
                overlap += 1
        if overlap > best_overlap:
            best_overlap = overlap
            best_seg = s

    if best_overlap <= 0:
        return

    # Update permanences
    for syn in range(n_synapses):
        if ctx[seg_indices[neuron, best_seg, syn]]:
            seg_perm[neuron, best_seg, syn] = min(
                seg_perm[neuron, best_seg, syn] + perm_increment, 1.0
            )
        else:
            seg_perm[neuron, best_seg, syn] = max(
                seg_perm[neuron, best_seg, syn] - perm_decrement, 0.0
            )

    # Build set of existing indices for this segment
    existing = set()
    for syn in range(n_synapses):
        existing.add(seg_indices[neuron, best_seg, syn])

    # Collect new sources: active in pool but not already connected
    new_sources = []
    for p in range(len(pool)):
        src = pool[p]
        if ctx[src] and src not in existing:
            new_sources.append(src)

    if len(new_sources) == 0:
        return

    # Find inactive synapse slots, sorted by weakest permanence
    inactive_slots = []
    inactive_perms = []
    for syn in range(n_synapses):
        if not ctx[seg_indices[neuron, best_seg, syn]]:
            inactive_slots.append(syn)
            inactive_perms.append(seg_perm[neuron, best_seg, syn])

    if len(inactive_slots) == 0:
        return

    # Sort inactive slots by permanence (weakest first)
    for i in range(len(inactive_slots)):
        for j in range(i + 1, len(inactive_slots)):
            if inactive_perms[j] < inactive_perms[i]:
                inactive_slots[i], inactive_slots[j] = (
                    inactive_slots[j],
                    inactive_slots[i],
                )
                inactive_perms[i], inactive_perms[j] = (
                    inactive_perms[j],
                    inactive_perms[i],
                )

    # Replace weakest inactive with new sources
    n_grow = min(len(new_sources), len(inactive_slots))
    for i in range(n_grow):
        slot = inactive_slots[i]
        seg_indices[neuron, best_seg, slot] = new_sources[i]
        seg_perm[neuron, best_seg, slot] = perm_init


@njit(cache=True)
def adapt_segments_batch(
    neurons: np.ndarray,  # (n_neurons,) int — neuron indices
    seg_indices: np.ndarray,  # (total_neurons, n_segments, n_synapses)
    seg_perm: np.ndarray,  # (total_neurons, n_segments, n_synapses)
    ctx: np.ndarray,  # (source_dim,) bool
    perm_threshold: float,
    seg_threshold: int,
    perm_increment: float,
    perm_decrement: float,
    reinforce: bool,
):
    """Reinforce or punish active segments for a batch of neurons.

    For each neuron, checks which segments are active (enough connected
    synapses with active sources), then updates permanences.
    """
    n_segments = seg_indices.shape[1]
    n_synapses = seg_indices.shape[2]

    for ni in range(len(neurons)):
        neuron = neurons[ni]
        for s in range(n_segments):
            # Count active connected synapses
            count = 0
            for syn in range(n_synapses):
                if (
                    ctx[seg_indices[neuron, s, syn]]
                    and seg_perm[neuron, s, syn] > perm_threshold
                ):
                    count += 1

            if count < seg_threshold:
                continue

            # This segment is active — update permanences
            if reinforce:
                for syn in range(n_synapses):
                    if ctx[seg_indices[neuron, s, syn]]:
                        seg_perm[neuron, s, syn] = min(
                            seg_perm[neuron, s, syn] + perm_increment, 1.0
                        )
                    else:
                        seg_perm[neuron, s, syn] = max(
                            seg_perm[neuron, s, syn] - perm_decrement, 0.0
                        )
            else:
                # Punish: only connected + active synapses
                for syn in range(n_synapses):
                    if (
                        ctx[seg_indices[neuron, s, syn]]
                        and seg_perm[neuron, s, syn] > perm_threshold
                    ):
                        seg_perm[neuron, s, syn] = max(
                            seg_perm[neuron, s, syn] - perm_decrement, 0.0
                        )
