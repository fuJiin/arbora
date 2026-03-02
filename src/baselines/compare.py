"""Comparison harness: STEP vs MiniGPT on shared metrics.

Shared metric: next-token accuracy (top-1 match).
Model-native metrics: IoU for STEP, perplexity for GPT.

Full implementation deferred to a future session.
"""

from dataclasses import dataclass


@dataclass
class ComparisonConfig:
    max_tokens: int = 10000
    eval_interval: int = 100
    dataset_name: str = "roneneldan/TinyStories"
    dataset_split: str = "train"


# TODO: Implement comparison harness
# - run_step(config) → metrics dict
# - run_gpt(config) → metrics dict
# - compare(step_metrics, gpt_metrics) → summary
