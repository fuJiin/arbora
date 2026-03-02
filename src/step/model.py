from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from step.config import ModelConfig
from step.normalize import local_normalize


class ModelState(NamedTuple):
    weights: dict[int, NDArray[np.floating]]
    history: dict[int, frozenset[int]]


def initial_state() -> ModelState:
    return ModelState(weights={}, history={})


def predict(state: ModelState, t: int, config: ModelConfig) -> frozenset[int]:
    prediction_vector = np.zeros(config.n)
    window = config.eligibility_window

    for i in range(max(0, t - window), t):
        if i not in state.history:
            continue
        past_sdr = state.history[i]
        strength = 1 - ((t - i) / window)
        for bit_idx in past_sdr:
            if bit_idx in state.weights:
                prediction_vector += state.weights[bit_idx] * strength

    prediction_vector = local_normalize(prediction_vector)

    top_k_indices = np.argpartition(prediction_vector, -config.k)[-config.k :]
    return frozenset(int(idx) for idx in top_k_indices)


def update(
    state: ModelState,
    t: int,
    current_sdr: frozenset[int],
    predicted_sdr: frozenset[int],
    config: ModelConfig,
) -> float:
    """Update weights in-place (intentional mutation for performance). Returns IoU."""
    overlap = len(current_sdr & predicted_sdr)
    iou = overlap / config.k
    actual_eta = config.max_lr * (1.0 - iou)
    window = config.eligibility_window

    for i in range(max(0, t - window), t):
        if i not in state.history:
            continue
        past_indices = state.history[i]
        trace_strength = 1 - ((t - i) / window)

        for p_idx in past_indices:
            if p_idx not in state.weights:
                state.weights[p_idx] = np.zeros(config.n)

            # Weight decay
            state.weights[p_idx] *= config.weight_decay

            # Reinforce bits that should have been active
            for c_idx in current_sdr:
                state.weights[p_idx][c_idx] += actual_eta * trace_strength

            # Penalize false positives
            for f_idx in predicted_sdr - current_sdr:
                state.weights[p_idx][f_idx] -= (
                    actual_eta * trace_strength * config.penalty_factor
                )

    return iou


def observe(
    state: ModelState, t: int, sdr: frozenset[int], config: ModelConfig
) -> ModelState:
    new_history = dict(state.history)
    new_history[t] = sdr
    new_history.pop(t - config.eligibility_window, None)
    return ModelState(weights=state.weights, history=new_history)
