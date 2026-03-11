#!/usr/bin/env python3
"""Test if STEP can beat bigrams with embedding-derived SDRs.

Uses TinyStories-1M embeddings to create structured SDRs where
semantically similar tokens share bits. Tests whether the learning
rule can leverage this structure at w=3,5,10.

If w>3 still can't beat w=3, the learning rule is the bottleneck.
If w>3 helps, encoding was the bottleneck.

Usage: uv run --extra comparison experiments/scripts/sweep_embedding_sdr.py
"""

import time
from collections import Counter

import numpy as np
from transformers import AutoModelForCausalLM

from step.config import EncoderConfig, ModelConfig, TrainingConfig
from step.data import (
    STORY_BOUNDARY,
    prepare_token_cache,
)
from step.model import learn, observe, predict, predict_with_vector

PRETRAIN_TOKENS = 200_000
EVAL_TOKENS = 10_000
VOCAB_SIZE = 10000
N = 2048  # SDR dimensionality
K = 40  # SDR sparsity


def build_embedding_sdrs(n: int = N, k: int = K, seed: int = 42):
    """Build SDRs from TinyStories-1M embeddings via random projection.

    1. Load 64-dim embeddings for 10K tokens
    2. Random project to n dimensions
    3. For each token, take top-k activations as the SDR

    Similar tokens will share bits because their projections
    will be correlated.
    """
    model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-1M")
    emb = model.transformer.wte.weight[:VOCAB_SIZE].detach().numpy()
    del model

    rng = np.random.default_rng(seed)
    # Random projection matrix: (64, n)
    projection = rng.standard_normal((emb.shape[1], n)).astype(np.float32)
    # Normalize columns for stability
    projection /= np.linalg.norm(projection, axis=0, keepdims=True)

    # Project: (10000, 64) @ (64, n) -> (10000, n)
    projected = emb @ projection

    # For each token, take top-k indices as SDR
    sdrs: dict[int, frozenset[int]] = {}
    for token_id in range(VOCAB_SIZE):
        top_k = np.argpartition(projected[token_id], -k)[-k:]
        sdrs[token_id] = frozenset(int(i) for i in top_k)

    return sdrs


def compute_sdr_stats(sdrs: dict[int, frozenset[int]], k: int = K):
    """Compute overlap statistics for the SDR set."""
    rng = np.random.default_rng(42)
    sdr_list = list(sdrs.values())
    overlaps = []
    for _ in range(5000):
        i, j = rng.choice(len(sdr_list), 2, replace=False)
        overlap = len(sdr_list[i] & sdr_list[j])
        overlaps.append(overlap)
    overlaps = np.array(overlaps)
    return overlaps.mean(), overlaps.std(), overlaps.max()


def bigram_baseline(train_cache, eval_cache):
    """Compute bigram accuracy with story boundaries."""
    bigrams: Counter = Counter()
    prev = None
    for token_id, _sdr in train_cache:
        if token_id == STORY_BOUNDARY:
            prev = None
            continue
        if prev is not None:
            bigrams[(prev, token_id)] += 1
        prev = token_id

    best_next: dict[int, int] = {}
    for (a, b), count in bigrams.items():
        if a not in best_next or count > bigrams.get((a, best_next[a]), 0):
            best_next[a] = b

    correct = 0
    total = 0
    prev = None
    for token_id, _sdr in eval_cache:
        if token_id == STORY_BOUNDARY:
            prev = None
            continue
        if prev is not None:
            if best_next.get(prev) == token_id:
                correct += 1
            total += 1
        prev = token_id

    return correct / total if total > 0 else 0.0


class EmbeddingSTEP:
    """STEP model that uses pre-computed embedding SDRs."""

    def __init__(self, model_config: ModelConfig, sdrs: dict[int, frozenset]):
        self.model_config = model_config
        from step.model import initial_state

        self._state = initial_state(model_config)
        self._sdrs = sdrs
        # Inverted index
        self._token_ids: list[int] = []
        self._token_id_to_idx: dict[int, int] = {}
        self._inverted_index: dict[int, list[int]] = {}

    def observe(self, t: int, token_id: int, sdr: frozenset[int]) -> None:
        if token_id == STORY_BOUNDARY:
            self._state.history.clear()
            return
        if token_id not in self._token_id_to_idx:
            idx = len(self._token_ids)
            self._token_ids.append(token_id)
            self._token_id_to_idx[token_id] = idx
            for bit in sdr:
                self._inverted_index.setdefault(bit, []).append(idx)
        self._state = observe(self._state, t, sdr, self.model_config)

    def predict_token(self, t: int) -> int:
        sdr, vector = predict_with_vector(self._state, t, self.model_config)
        return self._decode(sdr, vector)

    def predict_sdr(self, t: int) -> frozenset[int]:
        return predict(self._state, t, self.model_config)

    def learn(self, t, actual_sdr, predicted_sdr) -> float:
        return learn(self._state, t, actual_sdr, predicted_sdr, self.model_config)

    def get_sdr(self, token_id: int) -> frozenset[int]:
        return self._sdrs.get(token_id, frozenset())

    def _decode(self, sdr, vector=None) -> int:
        if not sdr or not self._token_ids:
            return -1
        scores: dict[int, float] = {}
        for bit in sdr:
            w = float(vector[bit]) if vector is not None else 1.0
            for idx in self._inverted_index.get(bit, ()):
                scores[idx] = scores.get(idx, 0.0) + w
        if not scores:
            return -1
        best_idx = max(scores, key=scores.__getitem__)
        return self._token_ids[best_idx]


def run_one(sdrs, model_cfg, train_cache, eval_cache):
    """Pretrain and eval with embedding SDRs."""
    model = EmbeddingSTEP(model_cfg, sdrs)

    # Pretrain
    start = time.monotonic()
    after_boundary = False
    for t, (token_id, _hash_sdr) in enumerate(train_cache):
        if t >= PRETRAIN_TOKENS:
            break
        if token_id == STORY_BOUNDARY:
            model.observe(t, token_id, frozenset())
            after_boundary = True
            continue
        sdr = sdrs.get(token_id, frozenset())
        if t > 0 and not after_boundary:
            predicted_sdr = model.predict_sdr(t)
            model.learn(t, sdr, predicted_sdr)
        after_boundary = False
        model.observe(t, token_id, sdr)

        if t > 0 and t % 100_000 == 0:
            elapsed = time.monotonic() - start
            print(f"    [pretrain] t={t:,}/{PRETRAIN_TOKENS:,} ({elapsed:.1f}s)")

    pretrain_time = time.monotonic() - start
    print(f"    [pretrain] done in {pretrain_time:.1f}s")

    # Eval
    correct = 0
    total = 0
    ious = []
    after_boundary = False
    for t, (token_id, _hash_sdr) in enumerate(eval_cache):
        if t >= EVAL_TOKENS:
            break
        if token_id == STORY_BOUNDARY:
            model.observe(t, token_id, frozenset())
            after_boundary = True
            continue
        sdr = sdrs.get(token_id, frozenset())
        if t > 0 and not after_boundary:
            pred_token = model.predict_token(t)
            pred_sdr = model.predict_sdr(t)
            iou_val = len(sdr & pred_sdr) / K if sdr else 0.0
            ious.append(iou_val)
            if pred_token == token_id:
                correct += 1
            total += 1
            model.learn(t, sdr, pred_sdr)
        after_boundary = False
        model.observe(t, token_id, sdr)

    acc = correct / total if total > 0 else 0.0
    mean_iou = np.mean(ious) if ious else 0.0
    return acc, mean_iou


def main():
    print("Building embedding SDRs...")
    sdrs = build_embedding_sdrs()
    mean_overlap, std_overlap, max_overlap = compute_sdr_stats(sdrs)
    print(
        f"  SDR overlap stats: mean={mean_overlap:.2f}, "
        f"std={std_overlap:.2f}, max={max_overlap}"
    )
    print(f"  Expected random: {K * K / N:.2f}")

    # Also build hash SDRs for comparison
    from step.sdr import encode_token

    hash_enc = EncoderConfig(model_name="gpt2", n=N, k=K, vocab_size=VOCAB_SIZE)
    hash_sdrs = {tid: encode_token(tid, hash_enc) for tid in range(VOCAB_SIZE)}
    hmean, hstd, hmax = compute_sdr_stats(hash_sdrs)
    print(f"  Hash overlap stats: mean={hmean:.2f}, std={hstd:.2f}, max={hmax}")

    # Cache data
    print("\nCaching data...")
    base_enc = EncoderConfig(model_name="gpt2", n=N, k=K, vocab_size=VOCAB_SIZE)
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
    train_cache = prepare_token_cache(train_tc, base_enc)
    eval_cache = prepare_token_cache(eval_tc, base_enc)
    print(f"  Train: {len(train_cache):,}, Eval: {len(eval_cache):,}")

    # Bigram baseline
    bigram_acc = bigram_baseline(train_cache, eval_cache)
    print(f"\nBigram baseline: {bigram_acc:.1%}")

    # Sweep
    windows = [3, 5, 10]
    results = []
    for sdr_type, sdr_dict in [("hash", hash_sdrs), ("embedding", sdrs)]:
        for w in windows:
            print(f"\n  {sdr_type} w={w}:")
            model_cfg = ModelConfig(
                n=N,
                k=K,
                max_lr=0.5,
                weight_decay=0.999,
                penalty_factor=0.5,
                eligibility_window=w,
            )
            start = time.monotonic()
            acc, iou = run_one(sdr_dict, model_cfg, train_cache, eval_cache)
            elapsed = time.monotonic() - start
            results.append((sdr_type, w, acc, iou, elapsed))
            print(f"    acc={acc:.1%} iou={iou:.4f} ({elapsed:.0f}s)")

    print(f"\n{'SDR Type':12s} {'w':>3s} {'Acc':>7s} {'IoU':>7s}")
    print(f"{'bigram':12s} {'-':>3s} {bigram_acc:7.1%} {'-':>7s}")
    for sdr_type, w, acc, iou, _ in results:
        print(f"{sdr_type:12s} {w:3d} {acc:7.1%} {iou:7.4f}")


if __name__ == "__main__":
    main()
