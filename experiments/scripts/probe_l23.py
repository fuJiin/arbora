#!/usr/bin/env python3
"""Linear probe: can L2/3 representations predict the next token?

Runs the cortex model, collects (L2/3 activation, next_token_id) pairs,
then trains a simple softmax regression to test whether next-token
information is present in the representations.

This is a sanity check, not a generation test. If the probe works,
the representations contain recoverable temporal structure. If it fails,
the representations may not support downstream use.

Usage: uv run experiments/scripts/probe_l23.py [--tokens 5000] [--lr 0.01]
"""

import argparse
import string
import time

import numpy as np

import step.env  # noqa: F401
from step.cortex.config import CortexConfig
from step.cortex.runner import STORY_BOUNDARY
from step.cortex.sensory import SensoryRegion
from step.encoders.charbit import CharbitEncoder

CHARS = string.printable
CHAR_LENGTH = 8
CHAR_WIDTH = len(CHARS) + 1


def prepare_tokens(max_tokens: int) -> list[tuple[int, str]]:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    print("Loading dataset...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    dataset = load_dataset("roneneldan/TinyStories", split="train")

    tokens: list[tuple[int, str]] = []
    t = 0
    first_story = True
    for example in dataset:
        if not first_story:
            tokens.append((STORY_BOUNDARY, ""))
            t += 1
            if t >= max_tokens:
                break
        first_story = False
        for tid in tokenizer.encode(example["text"]):
            tokens.append((tid, tokenizer.decode([tid])))
            t += 1
            if t >= max_tokens:
                break
        if t >= max_tokens:
            break

    unique = len({tid for tid, _ in tokens if tid != STORY_BOUNDARY})
    print(f"  {len(tokens):,} tokens, {unique} unique")
    return tokens


def collect_activations(
    tokens: list[tuple[int, str]],
    cfg: CortexConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Run cortex and collect (L2/3 activation, next_token_id) pairs.

    Returns X of shape (n_samples, n_l23_total) and y of shape (n_samples,).
    Each sample pairs the L2/3 state AFTER processing token t with the
    token_id of token t+1 (the "next token" to predict).
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

        # Pair previous L2/3 state with current token_id
        if prev_l23 is not None and not prev_was_boundary:
            X_list.append(prev_l23)
            y_list.append(token_id)

        # Save current L2/3 activation (as float for the probe)
        prev_l23 = region.active_l23.astype(np.float32).copy()
        prev_was_boundary = False

        if t > 0 and t % 1000 == 0:
            elapsed = time.monotonic() - start
            print(f"  collected {len(X_list):,} samples ({elapsed:.1f}s)")

    elapsed = time.monotonic() - start
    print(f"  {len(X_list):,} samples collected ({elapsed:.1f}s)")

    X = np.array(X_list)
    y = np.array(y_list)
    return X, y


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-frac", type=float, default=0.8)
    args = parser.parse_args()

    tokens = prepare_tokens(args.tokens)
    cfg = CortexConfig()

    print("\nRunning cortex to collect L2/3 activations...")
    X, y = collect_activations(tokens, cfg)

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
