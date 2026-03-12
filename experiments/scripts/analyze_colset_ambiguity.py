#!/usr/bin/env python3
"""Analyze how many tokens share column activation patterns."""

import string
from collections import Counter, defaultdict

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

import step.env  # noqa: F401
from step.config import CortexConfig
from step.cortex.sensory import SensoryRegion
from step.encoders.charbit import CharbitEncoder
from step.runner import STORY_BOUNDARY

CHARS = string.printable
CHAR_LENGTH = 8
CHAR_WIDTH = len(CHARS) + 1


def main():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    tokens = []
    t = 0
    first = True
    for ex in dataset:
        if not first:
            tokens.append((STORY_BOUNDARY, ""))
            t += 1
            if t >= 10000:
                break
        first = False
        for tid in tokenizer.encode(ex["text"]):
            tokens.append((tid, tokenizer.decode([tid])))
            t += 1
            if t >= 10000:
                break
        if t >= 10000:
            break

    cfg = CortexConfig()
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

    colset_to_tokens = defaultdict(set)
    token_to_colsets = defaultdict(set)
    tid_to_str = {}

    for tid, tstr in tokens:
        if tid == STORY_BOUNDARY:
            region.reset_working_memory()
            continue
        enc = charbit.encode(tstr)
        region.process(enc)
        cols = frozenset(int(c) for c in np.nonzero(region.active_columns)[0])
        colset_to_tokens[cols].add(tid)
        token_to_colsets[tid].add(cols)
        tid_to_str[tid] = tstr

    n_unique_tokens = len(token_to_colsets)
    n_unique_colsets = len(colset_to_tokens)
    tokens_per_colset = [len(tids) for tids in colset_to_tokens.values()]
    colsets_per_token = [len(cs) for cs in token_to_colsets.values()]

    print(f"Unique tokens: {n_unique_tokens}")
    print(f"Unique column sets: {n_unique_colsets}")
    print(
        f"Tokens per column set: "
        f"min={min(tokens_per_colset)} max={max(tokens_per_colset)} "
        f"mean={np.mean(tokens_per_colset):.1f} "
        f"median={np.median(tokens_per_colset):.0f}"
    )
    print(
        f"Column sets per token: "
        f"min={min(colsets_per_token)} max={max(colsets_per_token)} "
        f"mean={np.mean(colsets_per_token):.1f} "
        f"median={np.median(colsets_per_token):.0f}"
    )

    # Distribution
    counts = Counter(tokens_per_colset)
    print("\nTokens-per-colset distribution:")
    for k in sorted(counts.keys())[:20]:
        print(f"  {k} tokens: {counts[k]} column sets")
    if max(counts.keys()) > 20:
        print(f"  ... up to {max(counts.keys())} tokens")

    # Uniquely identifiable tokens
    uniquely_identifiable = 0
    for _tid, colsets in token_to_colsets.items():
        for cs in colsets:
            if len(colset_to_tokens[cs]) == 1:
                uniquely_identifiable += 1
                break
    pct = uniquely_identifiable / n_unique_tokens
    print(
        f"\nTokens uniquely identifiable by at least one column set: "
        f"{uniquely_identifiable}/{n_unique_tokens} ({pct:.1%})"
    )

    # Worst offenders
    worst = sorted(colset_to_tokens.items(), key=lambda x: -len(x[1]))[:10]
    print("\nMost ambiguous column sets:")
    for cols, tids in worst:
        sample = [repr(tid_to_str.get(t, "?")) for t in list(tids)[:6]]
        sample_str = ", ".join(sample)
        print(f"  cols={sorted(cols)}: {len(tids)} tokens — e.g. {sample_str}")

    # Max theoretical accuracy from column-set decode
    # For each token, best case: the column set it most often maps to
    # is unique to it. Accuracy = fraction of uniquely identifiable tokens.
    print(f"\nTheoretical max column-set accuracy: {pct:.1%}")
    print("(assuming perfect temporal prediction of column sets)")


if __name__ == "__main__":
    main()
