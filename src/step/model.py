from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from step.config import ModelConfig


class ModelState(NamedTuple):
    weights: NDArray[np.floating]  # (n, n) dense matrix
    history: dict[int, frozenset[int]]


def initial_state(config: ModelConfig | None = None) -> ModelState:
    n = config.n if config is not None else 0
    init_val = config.weight_init if config is not None else 0.0
    weights = np.full((n, n), init_val, dtype=np.float32)
    return ModelState(weights=weights, history={})


def _local_normalize(vector: NDArray[np.floating]) -> NDArray[np.floating]:
    max_val = np.max(vector)
    return vector / max_val if max_val > 0 else vector


def _compute_prediction_vector(
    state: ModelState, t: int, config: ModelConfig
) -> NDArray[np.floating] | None:
    """Compute the raw prediction vector (weighted sum of weight rows).

    Returns None if no history is available.
    """
    window = config.eligibility_window
    src_bits: list[int] = []
    strengths: list[float] = []
    for i in range(max(0, t - window), t):
        if i not in state.history:
            continue
        strength = 1 - ((t - i) / window)
        for bit_idx in state.history[i]:
            src_bits.append(bit_idx)
            strengths.append(strength)

    if not src_bits:
        return None

    src_arr = np.array(src_bits, dtype=np.intp)
    str_arr = np.array(strengths, dtype=np.float32)
    prediction_vector = (state.weights[src_arr] * str_arr[:, np.newaxis]).sum(axis=0)
    return _local_normalize(prediction_vector)


def predict(state: ModelState, t: int, config: ModelConfig) -> frozenset[int]:
    prediction_vector = _compute_prediction_vector(state, t, config)

    if prediction_vector is None:
        top_k_indices = np.arange(config.k)
        return frozenset(int(idx) for idx in top_k_indices)

    top_k_indices = np.argpartition(prediction_vector, -config.k)[-config.k :]
    return frozenset(int(idx) for idx in top_k_indices)


def predict_with_vector(
    state: ModelState, t: int, config: ModelConfig
) -> tuple[frozenset[int], NDArray[np.floating] | None]:
    """Like predict(), but also returns the raw prediction vector."""
    prediction_vector = _compute_prediction_vector(state, t, config)

    if prediction_vector is None:
        top_k_indices = np.arange(config.k)
        return frozenset(int(idx) for idx in top_k_indices), None

    top_k_indices = np.argpartition(prediction_vector, -config.k)[-config.k :]
    return frozenset(int(idx) for idx in top_k_indices), prediction_vector


def learn(
    state: ModelState,
    t: int,
    current_sdr: frozenset[int],
    predicted_sdr: frozenset[int],
    config: ModelConfig,
) -> float:
    """Update weights in-place via vectorized numpy ops. Returns IoU."""
    overlap = len(current_sdr & predicted_sdr)
    iou = overlap / config.k
    actual_eta = config.max_lr * (1.0 - iou)
    window = config.eligibility_window

    # Aggregate source bits with summed strengths (same bit from different
    # timesteps gets its strengths added together)
    src_strength: dict[int, float] = {}
    for i in range(max(0, t - window), t):
        if i not in state.history:
            continue
        trace_strength = 1 - ((t - i) / window)
        for p_idx in state.history[i]:
            src_strength[p_idx] = src_strength.get(p_idx, 0.0) + trace_strength

    if not src_strength:
        return iou

    src_arr = np.array(list(src_strength.keys()), dtype=np.intp)
    str_arr = np.array(list(src_strength.values()), dtype=np.float32)

    # Weight decay (applied to all affected rows at once)
    if config.weight_decay != 1.0:
        state.weights[src_arr] *= config.weight_decay

    if actual_eta == 0.0:
        return iou

    dst_arr = np.array(list(current_sdr), dtype=np.intp)

    if config.relevance_gate > 0:
        # Three-factor gated learning (per source bit):
        # activation = W[i,:] * trace_strength (synapse x trace)
        # relevance = fraction of top-k(activation) in actual target
        # Only source bits passing the gate get flat eta update.
        # Requires weight_init > 0 to bootstrap.
        activations = state.weights[src_arr] * str_arr[:, np.newaxis]
        top_k_per_src = np.argpartition(activations, -config.k, axis=1)[:, -config.k :]
        relevances = np.zeros(len(src_arr), dtype=np.float32)
        dst_set = current_sdr
        for idx in range(len(src_arr)):
            hits = sum(1 for b in top_k_per_src[idx] if b in dst_set)
            relevances[idx] = hits / config.k

        gate_mask = relevances >= config.relevance_gate
        if not gate_mask.any():
            return iou

        gated_src = src_arr[gate_mask]
        # Flat eta for gated sources (no trace_strength scaling)
        state.weights[np.ix_(gated_src, dst_arr)] += actual_eta

        false_positives = predicted_sdr - current_sdr
        if false_positives and config.penalty_factor > 0:
            fp_arr = np.array(list(false_positives), dtype=np.intp)
            penalty = actual_eta * config.penalty_factor
            state.weights[np.ix_(gated_src, fp_arr)] -= penalty
    else:
        # Original learning rule
        delta = actual_eta * str_arr  # (num_src,)
        state.weights[np.ix_(src_arr, dst_arr)] += delta[:, np.newaxis]

        false_positives = predicted_sdr - current_sdr
        if false_positives and config.penalty_factor > 0:
            fp_arr = np.array(list(false_positives), dtype=np.intp)
            penalty = actual_eta * config.penalty_factor * str_arr
            state.weights[np.ix_(src_arr, fp_arr)] -= penalty[:, np.newaxis]

    return iou


def observe(
    state: ModelState, t: int, sdr: frozenset[int], config: ModelConfig
) -> ModelState:
    new_history = dict(state.history)
    new_history[t] = sdr
    new_history.pop(t - config.eligibility_window, None)
    return ModelState(weights=state.weights, history=new_history)
