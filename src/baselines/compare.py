"""Comparison harness: STEP vs MiniGPT on shared metrics.

Shared metric: next-token accuracy (top-1 match).
  - STEP: for each predicted SDR, find which token in the recent vocabulary
    has the highest IoU with the prediction.
  - GPT: argmax of logits.

Model-native metrics:
  - STEP: IoU (intersection over union of predicted vs actual SDR)
  - GPT: perplexity (exp of cross-entropy loss)

Note: torch is required for GPT experiments.
Install with: uv sync --extra comparison
"""

from dataclasses import dataclass

import numpy as np

from step.config import EncoderConfig, ModelConfig, TrainingConfig
from step.metrics import compute_iou
from step.model import initial_state, learn, observe, predict
from step.sdr import encode_token


@dataclass
class ComparisonConfig:
    max_tokens: int = 10000
    eval_interval: int = 100
    dataset_name: str = "roneneldan/TinyStories"
    dataset_split: str = "train"


def step_next_token_accuracy(
    encoder_config: EncoderConfig,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    seed: int = 0,
    vocab_sample_size: int = 1000,
) -> dict:
    """Run STEP and compute both IoU and next-token accuracy.

    For next-token accuracy, we build a vocabulary of recently-seen tokens
    and check if the predicted SDR's best-matching token is the actual token.
    """
    rng = np.random.default_rng(seed)
    vocab_size = 50257
    token_ids = rng.integers(0, vocab_size, size=training_config.max_tokens)

    state = initial_state()
    ious: list[float] = []
    accuracies: list[float] = []
    seen_tokens: dict[int, frozenset[int]] = {}

    for t, tid in enumerate(token_ids):
        tid = int(tid)
        current_sdr = encode_token(tid, encoder_config)

        if t > 0 and len(seen_tokens) > 1:
            predicted = predict(state, t, model_config)
            iou = learn(state, t, current_sdr, predicted, model_config)
            ious.append(iou)

            # Next-token accuracy: find best matching token from seen vocab
            best_tid = -1
            best_iou = -1.0
            for candidate_tid, candidate_sdr in seen_tokens.items():
                candidate_iou = compute_iou(predicted, candidate_sdr)
                if candidate_iou > best_iou:
                    best_iou = candidate_iou
                    best_tid = candidate_tid

            accuracies.append(1.0 if best_tid == tid else 0.0)
        elif t > 0:
            predicted = predict(state, t, model_config)
            iou = learn(state, t, current_sdr, predicted, model_config)
            ious.append(iou)

        state = observe(state, t, current_sdr, model_config)
        seen_tokens[tid] = current_sdr

    return {
        "ious": ious,
        "accuracies": accuracies,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "mean_accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
    }
