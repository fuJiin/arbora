#!/usr/bin/env python3
"""Linear probe: can L2/3 representations predict the next token?

Runs the cortex model, collects (L2/3 activation, next_token_id) pairs,
then trains a simple softmax regression to test whether next-token
information is present in the representations.

This is a sanity check, not a generation test. If the probe works,
the representations contain recoverable temporal structure. If it fails,
the representations may not support downstream use.

Evaluation modes:
  --top-k K         Filter to top-K most frequent tokens (fair test at our
                    dimensionality). 0 = all tokens (default).
  --burst-analysis  Print per-token burst rate analysis showing which tokens
                    the dendritic segments have learned to anticipate.

Usage: uv run experiments/scripts/probe_l23.py [--tokens 5000] [--lr 0.01]
"""

import argparse
import string
import time
from collections import defaultdict

import numpy as np

import step.env  # noqa: F401
from step.config import CortexConfig
from step.cortex.sensory import SensoryRegion
from step.data import STORY_BOUNDARY, prepare_tokens
from step.encoders.charbit import CharbitEncoder

CHARS = string.printable
CHAR_LENGTH = 8
CHAR_WIDTH = len(CHARS) + 1


def collect_activations(
    tokens: list[tuple[int, str]],
    cfg: CortexConfig,
    *,
    use_firing_rate: bool = False,
    track_bursts: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[int, list[float]] | None]:
    """Run cortex and collect (L2/3 activation, next_token_id) pairs.

    Returns X of shape (n_samples, n_l23_total), y of shape (n_samples,),
    and optionally a dict mapping token_id -> list of per-step burst rates.

    Each sample pairs the L2/3 state AFTER processing token t with the
    token_id of token t+1 (the "next token" to predict).

    If use_firing_rate=True, uses the continuous firing_rate_l23 EMA
    instead of boolean active_l23. This captures temporal context.

    If track_bursts=True, records the burst rate after processing each token
    (bursting_columns.sum() / active_columns.sum()).
    """
    charbit = CharbitEncoder(length=CHAR_LENGTH, width=CHAR_WIDTH, chars=CHARS)
    input_dim = CHAR_LENGTH * CHAR_WIDTH
    region = SensoryRegion(
        input_dim=input_dim,
        encoding_width=CHAR_WIDTH,
        n_columns=cfg.n_columns,
        n_l4=cfg.n_l4,
        n_l23=cfg.n_l23,
        k_columns=cfg.k_columns,
        voltage_decay=cfg.voltage_decay,
        eligibility_decay=cfg.eligibility_decay,
        synapse_decay=cfg.synapse_decay,
        learning_rate=cfg.learning_rate,
        max_excitability=cfg.max_excitability,
        fb_boost=cfg.fb_boost,
        ltd_rate=cfg.ltd_rate,
        burst_learning_scale=cfg.burst_learning_scale,
        seed=cfg.seed,
    )

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    burst_rates: dict[int, list[float]] | None = (
        defaultdict(list) if track_bursts else None
    )

    prev_l23: np.ndarray | None = None
    prev_was_boundary = False

    start = time.monotonic()
    for t, (token_id, token_str) in enumerate(tokens):
        if token_id == STORY_BOUNDARY:
            region.reset_working_memory()
            prev_l23 = None
            prev_was_boundary = True
            continue

        encoding = charbit.encode(token_str)
        region.process(encoding)

        # Track per-token burst rate
        if burst_rates is not None:
            n_active = region.active_columns.sum()
            rate = (
                region.bursting_columns.sum() / n_active
                if n_active > 0
                else 1.0
            )
            burst_rates[token_id].append(float(rate))

        # Pair previous L2/3 state with current token_id
        if prev_l23 is not None and not prev_was_boundary:
            X_list.append(prev_l23)
            y_list.append(token_id)

        # Save current L2/3 state
        if use_firing_rate:
            prev_l23 = region.firing_rate_l23.astype(np.float32).copy()
        else:
            prev_l23 = region.active_l23.astype(np.float32).copy()
        prev_was_boundary = False

        if t > 0 and t % 1000 == 0:
            elapsed = time.monotonic() - start
            print(f"  collected {len(X_list):,} samples ({elapsed:.1f}s)")

    elapsed = time.monotonic() - start
    print(f"  {len(X_list):,} samples collected ({elapsed:.1f}s)")

    X = np.array(X_list)
    y = np.array(y_list)
    return X, y, dict(burst_rates) if burst_rates is not None else None


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over last axis."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def train_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    *,
    lr: float = 0.01,
    epochs: int = 10,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Train softmax regression via mini-batch SGD.

    Returns (W, b) where logits = X @ W + b.
    """
    n_features = X_train.shape[1]
    rng = np.random.default_rng(42)
    W = rng.normal(0, 0.01, (n_features, n_classes)).astype(np.float32)
    b = np.zeros(n_classes, dtype=np.float32)

    n = len(X_train)
    for epoch in range(epochs):
        perm = rng.permutation(n)
        total_loss = 0.0

        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            Xb = X_train[idx]
            yb = y_train[idx]
            bs = len(idx)

            logits = Xb @ W + b
            probs = softmax(logits)

            # Cross-entropy loss
            log_probs = np.log(probs[np.arange(bs), yb] + 1e-10)
            total_loss += -log_probs.sum()

            # Gradient
            probs[np.arange(bs), yb] -= 1.0  # probs - one_hot
            dW = Xb.T @ probs / bs
            db = probs.mean(axis=0)

            W -= lr * dW
            b -= lr * db

        avg_loss = total_loss / n
        print(f"  epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}")

    return W, b


def evaluate(
    X: np.ndarray,
    y: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
) -> dict:
    """Evaluate top-1 and top-5 accuracy."""
    logits = X @ W + b
    top1_pred = logits.argmax(axis=1)
    top1_acc = (top1_pred == y).mean()

    # Top-5
    top5_indices = np.argsort(logits, axis=1)[:, -5:]
    top5_acc = np.array([y[i] in top5_indices[i] for i in range(len(y))]).mean()

    return {"top1": float(top1_acc), "top5": float(top5_acc)}


def _filter_top_k(
    X: np.ndarray, y: np.ndarray, top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter samples to only include the top-K most frequent tokens."""
    token_ids, counts = np.unique(y, return_counts=True)
    top_k_indices = np.argsort(counts)[-top_k:]
    keep_tokens = set(token_ids[top_k_indices])

    mask = np.array([t in keep_tokens for t in y])
    X_filtered = X[mask]
    y_filtered = y[mask]

    print(f"\nTop-K filtering: kept {len(keep_tokens)} tokens, "
          f"{mask.sum():,}/{len(y):,} samples ({mask.mean() * 100:.1f}%)")

    return X_filtered, y_filtered


def _print_burst_analysis(burst_rates: dict[int, list[float]]) -> None:
    """Print top-20 tokens by burst rate (lowest = best predicted)."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Compute mean burst rate per token, require minimum occurrences
    min_occurrences = 5
    token_stats: list[tuple[int, float, int]] = []
    for token_id, rates in burst_rates.items():
        if len(rates) >= min_occurrences:
            token_stats.append((token_id, float(np.mean(rates)), len(rates)))

    token_stats.sort(key=lambda x: x[1])

    print(f"\n{'=' * 60}")
    print("Burst rate analysis (lowest = best anticipated by dendrites)")
    print(f"  {len(token_stats)} tokens with >= {min_occurrences} occurrences")
    print(f"{'=' * 60}")

    # Overall burst rate
    all_rates = [r for rates in burst_rates.values() for r in rates]
    overall = float(np.mean(all_rates))
    print(f"  Overall mean burst rate: {overall:.4f}")

    print("\n  Top 20 best-predicted tokens (lowest burst rate):")
    print(f"  {'Token':>14s}  {'BurstRate':>9s}  {'Count':>6s}")
    print(f"  {'-' * 14}  {'-' * 9}  {'-' * 6}")
    for token_id, mean_rate, count in token_stats[:20]:
        tok_str = repr(tokenizer.decode([token_id]))
        print(f"  {tok_str:>14s}  {mean_rate:9.4f}  {count:6d}")

    print("\n  Top 20 worst-predicted tokens (highest burst rate):")
    print(f"  {'Token':>14s}  {'BurstRate':>9s}  {'Count':>6s}")
    print(f"  {'-' * 14}  {'-' * 9}  {'-' * 6}")
    for token_id, mean_rate, count in token_stats[-20:][::-1]:
        tok_str = repr(tokenizer.decode([token_id]))
        print(f"  {tok_str:>14s}  {mean_rate:9.4f}  {count:6d}")

    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument(
        "--firing-rate", action="store_true",
        help="Use continuous firing_rate_l23 instead of boolean active_l23",
    )
    parser.add_argument(
        "--top-k", type=int, default=0,
        help="Filter to top-K most frequent tokens (0 = all tokens)",
    )
    parser.add_argument(
        "--burst-analysis", action="store_true",
        help="Print per-token burst rate analysis (which tokens are best predicted)",
    )
    args = parser.parse_args()

    tokens = prepare_tokens(args.tokens)
    cfg = CortexConfig()

    mode = (
        "firing_rate_l23 (continuous)" if args.firing_rate
        else "active_l23 (boolean)"
    )
    print(f"\nRunning cortex to collect L2/3 activations ({mode})...")
    X, y, burst_rates = collect_activations(
        tokens, cfg, use_firing_rate=args.firing_rate,
        track_bursts=args.burst_analysis,
    )

    # --- Burst rate analysis ---
    if args.burst_analysis and burst_rates is not None:
        _print_burst_analysis(burst_rates)

    # --- Top-K token filtering ---
    if args.top_k > 0:
        X, y = _filter_top_k(X, y, args.top_k)

    # Map token IDs to contiguous class indices
    unique_tokens = np.unique(y)
    n_classes = len(unique_tokens)
    token_to_class = {tid: i for i, tid in enumerate(unique_tokens)}
    class_to_token = {i: tid for tid, i in token_to_class.items()}
    y_mapped = np.array([token_to_class[t] for t in y])

    print(f"\n{X.shape[0]} samples, {X.shape[1]} features, {n_classes} classes")

    # Train/test split (temporal — no shuffling, to avoid leaking future)
    split = int(len(X) * args.train_frac)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y_mapped[:split], y_mapped[split:]
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Random baseline: predict most frequent class
    class_counts = np.bincount(y_train, minlength=n_classes)
    majority_class = class_counts.argmax()
    random_top1 = (y_test == majority_class).mean()
    # Uniform random baseline
    uniform_top1 = 1.0 / n_classes
    uniform_top5 = min(5.0 / n_classes, 1.0)

    print("\nBaselines:")
    print(f"  Majority class: top1={random_top1:.4f}")
    print(f"  Uniform random: top1={uniform_top1:.4f} top5={uniform_top5:.4f}")

    # Train probe
    print(f"\nTraining linear probe (lr={args.lr}, epochs={args.epochs})...")
    W, b = train_probe(X_train, y_train, n_classes, lr=args.lr, epochs=args.epochs)

    # Evaluate
    train_metrics = evaluate(X_train, y_train, W, b)
    test_metrics = evaluate(X_test, y_test, W, b)

    print("\nResults:")
    print(f"  Train: top1={train_metrics['top1']:.4f} top5={train_metrics['top5']:.4f}")
    print(f"  Test:  top1={test_metrics['top1']:.4f} top5={test_metrics['top5']:.4f}")
    print(f"  Majority baseline: {random_top1:.4f}")
    print(f"  Uniform baseline:  top1={uniform_top1:.4f} top5={uniform_top5:.4f}")

    lift = test_metrics["top1"] / max(random_top1, 1e-10)
    print(f"\n  Lift over majority: {lift:.2f}x")

    if test_metrics["top1"] > random_top1 * 2:
        print("\n  -> Representations contain recoverable next-token information.")
    elif test_metrics["top1"] > random_top1 * 1.2:
        print("\n  -> Weak signal — some information present but marginal.")
    else:
        print("\n  -> No meaningful signal above baseline.")

    # Show top predicted classes
    print("\nTop 10 most common test tokens:")
    test_counts = np.bincount(y_test, minlength=n_classes)
    top_classes = np.argsort(test_counts)[-10:][::-1]
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    for cls in top_classes:
        tid = class_to_token[cls]
        tok_str = repr(tokenizer.decode([tid]))
        count = test_counts[cls]
        pct = count / len(y_test) * 100
        # Per-class accuracy
        mask = y_test == cls
        if mask.any():
            logits = X_test[mask] @ W + b
            cls_acc = (logits.argmax(axis=1) == cls).mean()
        else:
            cls_acc = 0.0
        print(f"  {tok_str:>12s}: {count:4d} ({pct:4.1f}%) probe_acc={cls_acc:.3f}")


if __name__ == "__main__":
    main()
