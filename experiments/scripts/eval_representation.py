#!/usr/bin/env python3
"""Evaluate representation quality with random encoder + dendritic segments.

Runs the cortex with a random binary encoder (near-perfect token
discriminability) and reports the new representation metrics alongside
traditional decoder metrics.

Usage:
  uv run experiments/scripts/eval_representation.py [--tokens N]
  uv run experiments/scripts/eval_representation.py --dataset babylm
"""

import argparse
import string
import time

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

import step.env  # noqa: F401
from step.cortex.diagnostics import CortexDiagnostics
from step.cortex.representation import RepresentationTracker
from step.cortex.runner import STORY_BOUNDARY
from step.cortex.sensory import SensoryRegion
from step.decoders import InvertedIndexDecoder, SynapticDecoder
from step.encoders.charbit import CharbitEncoder

CHARS = string.printable
CHAR_LENGTH = 8
CHAR_WIDTH = len(CHARS) + 1


class RandomBinaryEncoder:
    """Deterministic random sparse binary encoder."""

    def __init__(self, n: int, k: int):
        self.n = n
        self.k = k
        self._cache: dict[int, np.ndarray] = {}

    def encode(self, token_id: int) -> np.ndarray:
        if token_id not in self._cache:
            rng = np.random.default_rng(token_id + 12345)
            vec = np.zeros(self.n, dtype=np.bool_)
            indices = rng.choice(self.n, self.k, replace=False)
            vec[indices] = True
            self._cache[token_id] = vec
        return self._cache[token_id]


def prepare_tokens_tinystories(max_tokens: int):
    print("Loading TinyStories...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    dataset = load_dataset("roneneldan/TinyStories", split="train")

    tokens: list[tuple[int, str]] = []
    t = 0
    first = True
    for ex in dataset:
        if not first:
            tokens.append((STORY_BOUNDARY, ""))
            t += 1
            if t >= max_tokens:
                break
        first = False
        for tid in tokenizer.encode(ex["text"]):
            tokens.append((tid, tokenizer.decode([tid])))
            t += 1
            if t >= max_tokens:
                break
        if t >= max_tokens:
            break

    unique = len({tid for tid, _ in tokens if tid != STORY_BOUNDARY})
    boundaries = sum(1 for tid, _ in tokens if tid == STORY_BOUNDARY)
    print(
        f"  {len(tokens):,} tokens, {unique} unique, "
        f"{boundaries + 1} documents\n"
    )
    return tokens


def prepare_tokens_babylm(max_tokens: int):
    print("Loading BabyLM (10M)...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    dataset = load_dataset("nilq/babylm-10M", split="train")

    tokens: list[tuple[int, str]] = []
    t = 0
    # Group consecutive non-empty lines into documents;
    # empty lines are document boundaries.
    in_doc = False
    for ex in dataset:
        text = ex.get("text", "").strip()
        if not text:
            if in_doc:
                tokens.append((STORY_BOUNDARY, ""))
                t += 1
                in_doc = False
            if t >= max_tokens:
                break
            continue
        in_doc = True
        for tid in tokenizer.encode(text):
            tokens.append((tid, tokenizer.decode([tid])))
            t += 1
            if t >= max_tokens:
                break
        if t >= max_tokens:
            break

    unique = len({tid for tid, _ in tokens if tid != STORY_BOUNDARY})
    boundaries = sum(1 for tid, _ in tokens if tid == STORY_BOUNDARY)
    print(
        f"  {len(tokens):,} tokens, {unique} unique, "
        f"{boundaries + 1} documents\n"
    )
    return tokens


def prepare_tokens(dataset_name: str, max_tokens: int):
    if dataset_name == "babylm":
        return prepare_tokens_babylm(max_tokens)
    return prepare_tokens_tinystories(max_tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=10000)
    parser.add_argument("--log-interval", type=int, default=2000)
    parser.add_argument(
        "--dataset",
        choices=["tinystories", "babylm"],
        default="tinystories",
    )
    parser.add_argument(
        "--encoder",
        choices=["random", "charbit"],
        default="random",
    )
    args = parser.parse_args()

    tokens = prepare_tokens(args.dataset, args.tokens)

    if args.encoder == "charbit":
        charbit = CharbitEncoder(
            length=CHAR_LENGTH, width=CHAR_WIDTH, chars=CHARS
        )
        input_dim = CHAR_LENGTH * CHAR_WIDTH
        encoding_width = CHAR_WIDTH

        def encode_token(_tid: int, tok_str: str) -> np.ndarray:
            return charbit.encode(tok_str)
    else:
        rand_enc = RandomBinaryEncoder(n=808, k=8)
        input_dim = rand_enc.n
        encoding_width = 0

        def encode_token(tid: int, _tok_str: str) -> np.ndarray:
            return rand_enc.encode(tid)

    # Use best config from previous sweep: thresh=2, inc=0.2
    region = SensoryRegion(
        input_dim=input_dim,
        encoding_width=encoding_width,
        n_columns=32,
        n_l4=4,
        n_l23=4,
        k_columns=4,
        voltage_decay=0.5,
        eligibility_decay=0.95,
        synapse_decay=0.999,
        learning_rate=0.05,
        max_excitability=0.2,
        fb_boost=0.4,
        ltd_rate=0.2,
        burst_learning_scale=3.0,
        n_fb_segments=4,
        n_lat_segments=4,
        n_synapses_per_segment=16,
        seg_activation_threshold=2,
        perm_init=0.6,
        perm_increment=0.2,
        perm_decrement=0.05,
        perm_threshold=0.5,
        seed=42,
    )

    diag = CortexDiagnostics(snapshot_interval=args.log_interval)
    rep = RepresentationTracker(region.n_columns, region.n_l4)
    decode_index = InvertedIndexDecoder()
    syn_decoder = SynapticDecoder()
    k = region.k_columns
    start = time.monotonic()

    # Accumulators for decoder metrics (kept for monitoring)
    syn_accs: list[float] = []
    overlaps: list[float] = []

    print(
        f"--- {args.dataset} + {args.encoder} encoder "
        f"+ segments (t2+i0.2) ---\n"
    )

    for t, (token_id, token_str) in enumerate(tokens):
        if token_id == STORY_BOUNDARY:
            region.reset_working_memory()
            rep.reset_context()
            continue

        # Prediction before input
        predicted_neurons = region.get_prediction(k)
        predicted_set = frozenset(int(i) for i in predicted_neurons)
        syn_id, _ = syn_decoder.decode_synaptic(predicted_neurons, region)

        # Process
        encoding = encode_token(token_id, token_str)
        active_neurons = region.process(encoding)
        active_set = frozenset(int(i) for i in active_neurons)

        # Track
        rep.observe(token_id, region.active_columns, region.active_l4)
        diag.step(t, region)

        if t > 0:
            if active_set:
                overlap = len(predicted_set & active_set) / len(active_set)
            else:
                overlap = 0.0
            overlaps.append(overlap)
            syn_accs.append(1.0 if syn_id == token_id else 0.0)

        decode_index.observe(token_id, active_set)
        syn_decoder.observe(
            token_id, token_str, encoding, region.active_columns
        )

        if t > 0 and t % args.log_interval == 0 and overlaps:
            tail_s = syn_accs[-100:]
            tail_o = overlaps[-100:]
            bc = diag._burst_count
            pc = diag._precise_count
            total = bc + pc
            elapsed = time.monotonic() - start
            print(
                f"  t={t:,} "
                f"syn={sum(tail_s)/len(tail_s):.4f} "
                f"overlap={sum(tail_o)/len(tail_o):.4f} "
                f"burst={bc/total:.1%} "
                f"({elapsed:.1f}s)"
            )

    elapsed = time.monotonic() - start
    print(f"\nDone in {elapsed:.1f}s")

    # Decoder summary (monitoring only)
    avg_syn = sum(syn_accs) / len(syn_accs) if syn_accs else 0
    tail_syn = syn_accs[-100:]
    avg_o = sum(overlaps) / len(overlaps) if overlaps else 0
    summ = diag.summary()
    print("\n--- Decoder metrics (monitoring) ---")
    last_syn = sum(tail_syn) / len(tail_syn)
    print(f"  syn accuracy: avg={avg_syn:.1%} last-100={last_syn:.1%}")
    print(f"  overlap: avg={avg_o:.4f}")
    print(f"  burst rate: {summ['burst_rate']:.1%}")

    # Representation quality (primary metrics)
    rep.print_report(region.ff_weights)

    # Compact summary
    rs = rep.summary(region.ff_weights)
    print("\n--- Summary ---")
    print(
        f"  selectivity={rs['column_selectivity_mean']:.3f} "
        f"similarity={rs['similarity_mean']:.3f} "
        f"(nontrivial={rs['similarity_nontrivial']}) "
        f"ctx_disc={rs['context_discrimination']:.3f}"
    )
    print(
        f"  ff_sparsity={rs['ff_sparsity']:.3f} "
        f"rf_entropy={rs['rf_entropy']:.3f} "
        f"cross_col_cos={rs['ff_cross_col_cosine']:.3f}"
    )
    print(f"  burst={summ['burst_rate']:.1%} tokens={rs['n_unique_tokens']}")


if __name__ == "__main__":
    main()
