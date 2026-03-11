"""STEP diagnostic instrumentation for exp2.

Collects weight statistics, prediction logs, and computes bigram SDR overlap
and per-position accuracy for diagnosing why STEP is at low accuracy.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from step.config import EncoderConfig, TrainingConfig
from step.sdr import encode_token

if TYPE_CHECKING:
    from step.protocol import Model


class DiagnosticCollector:
    """Collects diagnostics during STEP pretraining and evaluation."""

    def __init__(
        self,
        weight_snapshot_interval: int = 1000,
        log_predictions: bool = True,
        story_boundaries: list[int] | None = None,
    ):
        self.weight_snapshot_interval = weight_snapshot_interval
        self.log_predictions = log_predictions
        self.story_boundaries = story_boundaries or []

        # Weight stats over training
        self.weight_stats: list[dict] = []

        # Prediction logs (t, predicted, actual, native_metric)
        self.predictions: list[tuple[int, int, int, float]] = []

        # Per-position accuracy tracking
        self._position_correct: Counter[int] = Counter()
        self._position_total: Counter[int] = Counter()

        # Precompute story position lookup
        self._story_starts: list[int] = sorted(self.story_boundaries)

    def on_pretrain_step(self, t: int, model: Model) -> None:
        """Called during STEP pretraining to snapshot weight statistics."""
        if t % self.weight_snapshot_interval != 0:
            return
        state = getattr(model, "_state", None)
        if state is None:
            return
        weights = state.weights
        self.weight_stats.append(
            {
                "step": t,
                "mean": float(np.mean(weights)),
                "std": float(np.std(weights)),
                "max": float(np.max(weights)),
                "min": float(np.min(weights)),
                "near_zero_frac": float(np.mean(np.abs(weights) < 0.01)),
                "gt_one_frac": float(np.mean(np.abs(weights) > 1.0)),
                "nonzero_frac": float(np.mean(weights != 0.0)),
            }
        )

    def on_eval_step(
        self, t: int, predicted_token: int, actual_token: int, native_metric: float
    ) -> None:
        """Called during evaluation to log predictions and track accuracy."""
        if self.log_predictions:
            self.predictions.append((t, predicted_token, actual_token, native_metric))

        # Track per-position accuracy
        pos = self._get_story_position(t)
        correct = 1 if predicted_token == actual_token else 0
        self._position_correct[pos] += correct
        self._position_total[pos] += 1

    def _get_story_position(self, t: int) -> int:
        """Get position within current story using binary search."""
        if not self._story_starts:
            return t
        # Find the last story start <= t
        lo, hi = 0, len(self._story_starts) - 1
        result = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if self._story_starts[mid] <= t:
                result = self._story_starts[mid]
                lo = mid + 1
            else:
                hi = mid - 1
        return t - result

    def get_position_accuracy(self) -> dict[int, float]:
        """Return mean accuracy by story position."""
        result = {}
        for pos in sorted(self._position_total.keys()):
            total = self._position_total[pos]
            if total > 0:
                result[pos] = self._position_correct[pos] / total
        return result

    def save(self, output_dir: Path) -> None:
        """Write all diagnostics as JSON files."""
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.weight_stats:
            (output_dir / "weight_stats.json").write_text(
                json.dumps(self.weight_stats, indent=2)
            )

        if self.predictions:
            (output_dir / "predictions.json").write_text(json.dumps(self.predictions))

        position_acc = self.get_position_accuracy()
        if position_acc:
            (output_dir / "position_accuracy.json").write_text(
                json.dumps({str(k): v for k, v in position_acc.items()}, indent=2)
            )


def compute_bigram_sdr_overlap(
    token_cache: list[tuple[int, frozenset[int]]],
    encoder_config: EncoderConfig,
    top_n: int = 50,
) -> list[dict]:
    """Count bigram frequencies and compute SDR bit overlap for top-N bigrams.

    Returns list of dicts with keys: token_a, token_b, count, overlap, overlap_frac.
    """
    # Count bigrams
    bigram_counts: Counter[tuple[int, int]] = Counter()
    for i in range(len(token_cache) - 1):
        a_id = token_cache[i][0]
        b_id = token_cache[i + 1][0]
        bigram_counts[(a_id, b_id)] += 1

    # Get top-N bigrams
    top_bigrams = bigram_counts.most_common(top_n)

    results = []
    for (a_id, b_id), count in top_bigrams:
        sdr_a = encode_token(a_id, encoder_config)
        sdr_b = encode_token(b_id, encoder_config)
        overlap = len(sdr_a & sdr_b)
        results.append(
            {
                "token_a": a_id,
                "token_b": b_id,
                "count": count,
                "overlap": overlap,
                "overlap_frac": overlap / encoder_config.k
                if encoder_config.k > 0
                else 0.0,
            }
        )

    return results


def compute_story_boundaries(
    training_config: TrainingConfig,
    encoder_config: EncoderConfig,
) -> list[int]:
    """Find cumulative token positions where new stories begin.

    Re-iterates dataset (tokenize only, no SDR encoding) to find story starts.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    import step.env  # noqa: F401

    tokenizer = AutoTokenizer.from_pretrained(encoder_config.model_name)
    dataset = load_dataset(
        training_config.dataset_name,
        split=training_config.dataset_split,
    )

    boundaries = [0]
    t = 0
    for example in dataset:
        boundaries.append(t)
        token_ids = tokenizer.encode(example["text"])  # type: ignore[union-attr]
        t += len(token_ids)
        if t >= training_config.max_tokens:
            break

    return boundaries


def save_bigram_overlap(results: list[dict], output_dir: Path) -> None:
    """Save bigram overlap analysis to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "bigram_overlap.json").write_text(json.dumps(results, indent=2))
