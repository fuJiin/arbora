#!/usr/bin/env python3
"""Run STEP experiments from a config file. Saves raw JSON results only."""

import json
import sys
from pathlib import Path

from step.config import EncoderConfig, ModelConfig, TrainingConfig
from step.experiment import ExperimentConfig, run_multi_seed

ROOT = Path(__file__).resolve().parent.parent


def main(config_path: str | None = None) -> None:
    if config_path is None:
        config_path = str(ROOT / "configs" / "exp0_tinystories.json")

    with open(config_path) as f:
        raw = json.load(f)

    base_config = ExperimentConfig(
        encoder=EncoderConfig(**raw["encoder"]),
        model=ModelConfig(**raw["model"]),
        training=TrainingConfig(**raw["training"]),
        name=raw["name"],
    )
    seeds = raw.get("seeds", [0, 1, 2, 3, 4])
    output_dir = ROOT / "runs" / raw["name"]

    print(f"Running {raw['name']} with seeds {seeds}...")
    print(f"  Tokens per run: {raw['training']['max_tokens']}")
    print(f"  Output: {output_dir}")
    print()

    results = run_multi_seed(base_config, seeds, output_dir)

    for r in results:
        final_iou = r.rolling_ious[-1][1] if r.rolling_ious else 0.0
        print(
            f"  Seed {r.config.seed}: "
            f"final rolling IoU = {final_iou:.4f}, "
            f"elapsed = {r.elapsed_seconds:.1f}s"
        )

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_path)
