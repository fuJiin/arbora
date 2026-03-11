#!/usr/bin/env python3
"""Diagnose why high IoU lift doesn't translate to accuracy.

For each wrong prediction, computes:
- IoU between predicted_sdr and correct token's SDR
- IoU between predicted_sdr and decoded (wrong) token's SDR
- Number of "confusable" tokens (IoU with predicted_sdr above threshold)
- Whether the correct token was in the top-N candidates

Usage: uv run --extra comparison experiments/scripts/diagnose_decode.py
"""

import numpy as np

from step.config import EncoderConfig, ModelConfig, TrainingConfig
from step.data import prepare_token_cache
from step.experiment import ExperimentConfig, pretrain_step_model
from step.wrappers import StepMemoryModel

PRETRAIN_TOKENS = 50_000
EVAL_TOKENS = 2_000


def iou(a: frozenset[int], b: frozenset[int], k: int) -> float:
    return len(a & b) / k


def run_diagnosis(label, enc_cfg, window, train_cache, eval_cache):
    model_cfg = ModelConfig(
        n=2048,
        k=40,
        max_lr=0.5,
        weight_decay=0.999,
        penalty_factor=0.5,
        eligibility_window=window,
    )
    train_tc = TrainingConfig(
        dataset_name="roneneldan/TinyStories",
        dataset_split="train",
        max_tokens=PRETRAIN_TOKENS,
        log_interval=50_000,
    )
    eval_tc = TrainingConfig(
        dataset_name="roneneldan/TinyStories",
        dataset_split="validation",
        max_tokens=EVAL_TOKENS,
        log_interval=50_000,
    )
    pretrain_cfg = ExperimentConfig(
        encoder=enc_cfg,
        model=model_cfg,
        training=train_tc,
        name="diag",
    )

    model = StepMemoryModel(model_cfg, enc_cfg)
    pretrain_step_model(model, pretrain_cfg, train_cache)

    # Collect all token SDRs from the model's encoder
    # (either adaptive cached or hash-based)
    k = enc_cfg.k

    # Run eval and collect decode diagnostics
    correct_count = 0
    wrong_count = 0
    iou_correct_when_wrong = []  # IoU(pred_sdr, correct_sdr) when wrong
    iou_decoded_when_wrong = []  # IoU(pred_sdr, decoded_sdr) when wrong
    iou_correct_when_right = []  # IoU(pred_sdr, correct_sdr) when right
    rank_of_correct = []  # rank of correct token among all candidates
    n_confusable = []  # tokens with IoU > threshold
    from step.experiment import _make_stream

    eval_cfg = ExperimentConfig(
        encoder=enc_cfg,
        model=model_cfg,
        training=eval_tc,
        name="diag",
    )

    for t, token_id, _sdr in _make_stream(eval_cfg, eval_cache):
        if hasattr(model, "encode_token_sdr"):
            sdr = model.encode_token_sdr(token_id, t)
        else:
            sdr = _sdr

        if t > 0:
            predicted_sdr = model.predict_sdr(t)
            predicted_token = model.predict_token(t)

            # Get correct token's SDR
            correct_iou = iou(predicted_sdr, sdr, k)

            if predicted_token == token_id:
                correct_count += 1
                iou_correct_when_right.append(correct_iou)
            else:
                wrong_count += 1
                iou_correct_when_wrong.append(correct_iou)

                # Get decoded token's SDR
                if predicted_token >= 0 and predicted_token in model._token_id_to_idx:
                    idx = model._token_id_to_idx[predicted_token]
                    # Reconstruct decoded token's SDR from inverted index
                    decoded_sdr = set()
                    for bit, idxs in model._inverted_index.items():
                        if idx in idxs:
                            decoded_sdr.add(bit)
                    decoded_sdr = frozenset(decoded_sdr)
                    decoded_iou_val = iou(predicted_sdr, decoded_sdr, k)
                    iou_decoded_when_wrong.append(decoded_iou_val)
                else:
                    iou_decoded_when_wrong.append(0.0)

                # Count confusable tokens (IoU with predicted_sdr > correct_iou * 0.8)
                threshold = correct_iou * 0.8
                n_conf = 0
                correct_rank = 1
                for _tid, tidx in model._token_id_to_idx.items():
                    tok_sdr = set()
                    for bit, idxs in model._inverted_index.items():
                        if tidx in idxs:
                            tok_sdr.add(bit)
                    tok_iou = iou(predicted_sdr, frozenset(tok_sdr), k)
                    if tok_iou >= threshold:
                        n_conf += 1
                    if tok_iou > correct_iou:
                        correct_rank += 1
                n_confusable.append(n_conf)
                rank_of_correct.append(correct_rank)

        model.observe(t, token_id, sdr)

        # Limit for speed (reconstructing SDRs from inverted index is slow)
        if t >= 500:
            break

    total = correct_count + wrong_count
    print(f"\n=== {label} (w={window}) ===")
    print(f"Accuracy: {correct_count}/{total} = {correct_count / total:.1%}")
    print(f"When CORRECT ({correct_count} cases):")
    if iou_correct_when_right:
        print(f"  Mean IoU(pred, correct): {np.mean(iou_correct_when_right):.4f}")
    print(f"When WRONG ({wrong_count} cases):")
    if iou_correct_when_wrong:
        print(f"  Mean IoU(pred, correct):  {np.mean(iou_correct_when_wrong):.4f}")
        print(f"  Mean IoU(pred, decoded):  {np.mean(iou_decoded_when_wrong):.4f}")
        print(f"  Mean rank of correct:     {np.mean(rank_of_correct):.1f}")
        print(f"  Median rank of correct:   {np.median(rank_of_correct):.0f}")
        print(f"  Mean confusable tokens:   {np.mean(n_confusable):.1f}")
        # How often decoded has HIGHER iou than correct?
        n_decoded_wins = sum(
            1
            for d, c in zip(iou_decoded_when_wrong, iou_correct_when_wrong, strict=True)
            if d > c
        )
        n_tie = sum(
            1
            for d, c in zip(iou_decoded_when_wrong, iou_correct_when_wrong, strict=True)
            if d == c
        )
        print(
            f"  Decoded IoU > Correct IoU: {n_decoded_wins}/{wrong_count} "
            f"({n_decoded_wins / wrong_count:.1%})"
        )
        print(
            f"  Decoded IoU = Correct IoU: {n_tie}/{wrong_count} "
            f"({n_tie / wrong_count:.1%})"
        )


def main():
    base_enc = EncoderConfig(model_name="gpt2", n=2048, k=40, vocab_size=10000)
    train_tc = TrainingConfig(
        dataset_name="roneneldan/TinyStories",
        dataset_split="train",
        max_tokens=PRETRAIN_TOKENS,
    )
    eval_tc = TrainingConfig(
        dataset_name="roneneldan/TinyStories",
        dataset_split="validation",
        max_tokens=EVAL_TOKENS,
    )
    print("Caching data...")
    train_cache = prepare_token_cache(train_tc, base_enc)
    eval_cache = prepare_token_cache(eval_tc, base_enc)

    schemes = [
        ("hash", EncoderConfig(model_name="gpt2", n=2048, k=40, vocab_size=10000)),
        (
            "predict f=0.5",
            EncoderConfig(
                model_name="gpt2",
                n=2048,
                k=40,
                vocab_size=10000,
                adaptive=True,
                context_fraction=0.5,
                seeding="predicted",
            ),
        ),
        (
            "active f=0.3",
            EncoderConfig(
                model_name="gpt2",
                n=2048,
                k=40,
                vocab_size=10000,
                adaptive=True,
                context_fraction=0.3,
                seeding="active",
            ),
        ),
    ]

    for label, enc_cfg in schemes:
        run_diagnosis(label, enc_cfg, 3, train_cache, eval_cache)


if __name__ == "__main__":
    main()
