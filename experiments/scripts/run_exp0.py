#!/usr/bin/env python3
"""Run exp0 three-model comparison on TinyStories.

Pre-trains all models on train split, then evaluates on validation split.
Usage: uv run --extra comparison experiments/scripts/run_exp0.py [config_path]
"""

import json
import sys
from pathlib import Path

from step.config import EncoderConfig, ModelConfig, TrainingConfig
from step.data import prepare_token_cache
from step.experiment import (
    ComparisonRunResult,
    ExperimentConfig,
    pretrain_step_model,
    run_experiment,
    save_comparison_result,
)

ROOT = Path(__file__).resolve().parent.parent


def make_step_memory_factory():
    from step.wrappers import StepMemoryModel

    def factory(config: ExperimentConfig):
        return StepMemoryModel(config.model, config.encoder)

    return factory


def make_step_sqlite_factory(db_path: Path):
    from step.db import StepModel

    def factory(config: ExperimentConfig):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return StepModel(db_path, config.model, config.encoder)

    return factory


def make_tinystories_1m_factory():
    import torch
    from transformers import AutoModelForCausalLM

    from baselines.wrappers import TinyStories1MModel

    def factory(config: ExperimentConfig):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-1M")
        model.to(device)
        model.eval()
        return TinyStories1MModel(model, context_length=512)

    return factory


def make_mini_gpt_factory(checkpoint_path: Path, gpt_config_dict: dict):
    import torch

    from baselines.mini_gpt import MiniGPT, MiniGPTConfig
    from baselines.wrappers import MiniGPTModel

    gpt_config = MiniGPTConfig(**gpt_config_dict)

    def factory(config: ExperimentConfig):
        model = MiniGPT(gpt_config)
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        model.eval()
        return MiniGPTModel(model, gpt_config)

    return factory


def main(config_path: str | None = None) -> None:
    if config_path is None:
        config_path = str(ROOT / "configs" / "exp0_comparison.json")

    with open(config_path) as f:
        raw = json.load(f)

    exp_config = ExperimentConfig(
        encoder=EncoderConfig(**raw["encoder"]),
        model=ModelConfig(**raw["model"]),
        training=TrainingConfig(**raw["training"]),
        name=raw["name"],
    )
    models_to_run: list[str] = raw["models"]
    output_base = ROOT / "runs" / raw["name"]

    print(f"Experiment: {raw['name']}")
    print(f"  Eval split: {raw['training']['dataset_split']}")
    print(f"  Eval tokens: {raw['training']['max_tokens']:,}")
    print(f"  Models: {models_to_run}")
    print()

    # --- Cache datasets ---
    step_pretrain_cfg = raw.get("step_pretrain")
    train_cache = None
    if step_pretrain_cfg and any(m.startswith("step_") for m in models_to_run):
        print("--- Caching train split ---")
        train_tc = TrainingConfig(
            dataset_name=raw["training"]["dataset_name"],
            dataset_split=step_pretrain_cfg["dataset_split"],
            max_tokens=step_pretrain_cfg["max_tokens"],
        )
        train_cache = prepare_token_cache(train_tc, exp_config.encoder)
        print(f"  Cached {len(train_cache):,} train tokens")
        print()

    print("--- Caching validation split ---")
    eval_cache = prepare_token_cache(exp_config.training, exp_config.encoder)
    print(f"  Cached {len(eval_cache):,} eval tokens")
    print()

    # --- Pre-train MiniGPT ---
    checkpoint_path = ROOT / "checkpoints" / f"mini_gpt_{raw['name']}.pt"
    if "mini_gpt" in models_to_run:
        from baselines.mini_gpt import MiniGPTConfig
        from baselines.pretrain import PretrainConfig, pretrain_mini_gpt

        gpt_config = MiniGPTConfig(**raw["mini_gpt"])
        pretrain_cfg = PretrainConfig(
            dataset_name=raw["training"]["dataset_name"],
            dataset_split=raw["pretrain"]["dataset_split"],
            max_tokens=raw["pretrain"]["max_tokens"],
            batch_size=raw["pretrain"]["batch_size"],
            epochs=raw["pretrain"]["epochs"],
            lr=raw["pretrain"]["lr"],
            warmup_steps=raw["pretrain"].get("warmup_steps", 0),
            min_lr=raw["pretrain"].get("min_lr", 0.0),
        )
        print("--- Pre-training MiniGPT ---")
        pretrain_mini_gpt(gpt_config, pretrain_cfg, checkpoint_path)
        print()

    # --- Pre-train STEP models, then eval ---
    factories: dict[str, tuple[callable, str]] = {}
    if "step_memory" in models_to_run:
        factories["step_memory"] = (make_step_memory_factory(), "iou")
    if "step_sqlite" in models_to_run:
        db_path = output_base / "dbs" / "step_sqlite.db"
        factories["step_sqlite"] = (make_step_sqlite_factory(db_path), "iou")
    if "mini_gpt" in models_to_run:
        factories["mini_gpt"] = (
            make_mini_gpt_factory(checkpoint_path, raw["mini_gpt"]),
            "cross_entropy_loss",
        )
    if "tinystories_1m" in models_to_run:
        factories["tinystories_1m"] = (
            make_tinystories_1m_factory(),
            "cross_entropy_loss",
        )

    # --- Setup diagnostics (if configured) ---
    diag_cfg = raw.get("diagnostics", {})
    diagnostics = None
    if diag_cfg.get("enabled") and any(m.startswith("step_") for m in models_to_run):
        from step.diagnostics import (
            DiagnosticCollector,
            compute_bigram_sdr_overlap,
            compute_story_boundaries,
            save_bigram_overlap,
        )

        print("--- Computing diagnostic baselines ---")
        # Compute story boundaries from the pretrain dataset
        story_boundaries = []
        if step_pretrain_cfg and train_cache is not None:
            pretrain_tc = TrainingConfig(
                dataset_name=raw["training"]["dataset_name"],
                dataset_split=step_pretrain_cfg["dataset_split"],
                max_tokens=step_pretrain_cfg["max_tokens"],
            )
            story_boundaries = compute_story_boundaries(
                pretrain_tc, exp_config.encoder
            )
            print(f"  Found {len(story_boundaries):,} story boundaries")

        # Compute bigram SDR overlap from eval cache
        bigram_results = compute_bigram_sdr_overlap(
            eval_cache,
            exp_config.encoder,
            top_n=diag_cfg.get("bigram_top_n", 50),
        )
        diag_dir = output_base / "diagnostics"
        save_bigram_overlap(bigram_results, diag_dir)
        print(f"  Computed bigram overlap for {len(bigram_results)} bigrams")

        diagnostics = DiagnosticCollector(
            weight_snapshot_interval=diag_cfg.get("weight_snapshot_interval", 1000),
            log_predictions=diag_cfg.get("log_predictions", True),
            story_boundaries=story_boundaries,
        )
        print()

    results: list[ComparisonRunResult] = []
    for model_name in models_to_run:
        factory, native_metric_name = factories[model_name]

        # Prepare diagnostic callbacks for STEP models
        on_step_cb = None
        on_eval_cb = None
        if diagnostics is not None and model_name.startswith("step_"):
            on_step_cb = diagnostics.on_pretrain_step
            on_eval_cb = diagnostics.on_eval_step

        # Pre-train STEP models on train split
        if model_name.startswith("step_") and train_cache is not None:
            print(f"--- Pre-training {model_name} on {len(train_cache):,} tokens ---")
            pretrain_config = ExperimentConfig(
                encoder=exp_config.encoder,
                model=exp_config.model,
                training=TrainingConfig(
                    dataset_name=raw["training"]["dataset_name"],
                    dataset_split=step_pretrain_cfg["dataset_split"],
                    max_tokens=step_pretrain_cfg["max_tokens"],
                    log_interval=exp_config.training.log_interval,
                    rolling_window=exp_config.training.rolling_window,
                ),
                name=exp_config.name,
            )
            # Create model, pre-train it, then reuse for eval
            model_instance = factory(exp_config)
            pretrain_step_model(
                model_instance, pretrain_config, train_cache, on_step=on_step_cb
            )
            print()

            # Wrap in a factory that returns the pre-trained instance
            _model = model_instance

            def pretrained_factory(config, _m=_model):
                return _m

            print(f"--- Evaluating {model_name} ---")
            result = run_experiment(
                exp_config, pretrained_factory, model_name,
                native_metric_name, eval_cache, on_eval_step=on_eval_cb,
            )
        else:
            print(f"--- Evaluating {model_name} ---")
            result = run_experiment(
                exp_config, factory, model_name,
                native_metric_name, eval_cache,
            )

        results.append(result)

        # Save result
        model_dir = output_base / model_name
        save_comparison_result(result, model_dir)

        final_acc = (
            result.rolling_accuracies[-1][1] if result.rolling_accuracies else 0.0
        )
        final_native = (
            result.rolling_native[-1][1] if result.rolling_native else 0.0
        )
        print(
            f"  {model_name}: "
            f"final rolling accuracy = {final_acc:.4f}, "
            f"final rolling {native_metric_name} = {final_native:.4f}, "
            f"elapsed = {result.elapsed_seconds:.1f}s"
        )
        print()

    # Save diagnostics
    if diagnostics is not None:
        diag_dir = output_base / "diagnostics"
        diagnostics.save(diag_dir)
        print(f"Diagnostics saved to: {diag_dir}")

    print(f"Results saved to: {output_base}")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_path)
