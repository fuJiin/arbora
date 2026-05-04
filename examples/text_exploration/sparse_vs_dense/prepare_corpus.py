"""ARB-139: unified corpus preparation for fair SSH ↔ w2v comparison.

Both models train on the same text8 stream, but historically have
diverged in subtle data-handling choices:

    word2vec (gensim):  1000-token sentences, sentence-shuffled per epoch,
                        Mikolov frequent-word subsampling, min_count filter,
                        5 epochs by default.
    SSH (current):      whole corpus as one stream, no shuffling, no
                        subsampling, no min_count filter, 1 epoch.

This module exposes one function — `prepare_corpus` — that produces a
corpus pre-processed identically for both. Each model converts the output
to its native format (gensim wants `list[list[str]]`; SSH wants flat
`list[int]`).

Default behavior matches Mikolov / gensim conventions:
  - chunk_size=1000  (gensim's default sentence size for text8)
  - subsample_threshold=1e-3  (Mikolov's `t` parameter)
  - shuffle_chunks=True  (gensim shuffles sentences per epoch — for our
    1-epoch comparison we shuffle once at the start)
  - min_count=5  (gensim default)

Each switch can be turned off independently for ablation.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np


@dataclass
class CorpusPlan:
    """Spec for how to prepare the corpus. All knobs default to gensim-style."""

    chunk_size: int = 1000
    """Tokens per pseudo-sentence. Skip-gram windows don't cross chunk
    boundaries. 0 disables chunking (whole corpus as one stream)."""

    subsample_threshold: float = 1e-3
    """Mikolov subsampling threshold `t`. Word `w` with corpus frequency
    `f(w)` is kept with probability `min(1, sqrt(t / f(w)))`. Set to 0 to
    disable subsampling (every token kept)."""

    min_count: int = 5
    """Drop words with fewer than this many occurrences from the vocabulary
    (and from the token stream). 1 disables filtering."""

    shuffle_chunks: bool = True
    """Shuffle the order of chunks once before flattening. Preserves
    within-chunk local context but breaks long-range autocorrelation —
    matches gensim's per-epoch sentence shuffling for a 1-epoch comparison.
    """

    seed: int = 0


@dataclass
class PreparedCorpus:
    """Output of `prepare_corpus`."""

    flat_tokens: list[str]
    """Token stream after subsampling, min_count filter, and (optional)
    chunk shuffling. Used by SSH."""

    chunks: list[list[str]]
    """Token stream split into chunks of `chunk_size` (or one giant chunk
    if chunking disabled). Order matches `flat_tokens`. Used by gensim
    (it expects `list[list[str]]` as `sentences=`)."""

    vocab: list[str]
    """Vocabulary list (post-min_count filter), in frequency-descending
    order with `<unk>` at index 0."""

    token_to_id: dict[str, int]
    """Lookup from token string to integer ID. Used by SSH."""

    flat_token_ids: list[int]
    """`flat_tokens` mapped through `token_to_id`. Used directly by SSH
    training. (Tokens not in vocab are mapped to 0 = `<unk>`.)"""

    plan: CorpusPlan
    """The plan used to produce this corpus."""

    stats: dict = field(default_factory=dict)
    """Per-stage counts: tokens before/after subsampling, vocab size,
    chunk count, etc."""


def prepare_corpus(
    raw_tokens: Sequence[str],
    *,
    plan: CorpusPlan | None = None,
) -> PreparedCorpus:
    """Apply chunking, subsampling, min_count filtering, and chunk
    shuffling to a raw token stream. Both SSH and word2vec consume
    the result through their respective adapters.

    Args:
        raw_tokens: list of pre-tokenized strings (text8 is already
            space-separated lowercase, so `tokens = raw_text.split()`).
        plan: knobs (see `CorpusPlan`). If None, gensim-style defaults.

    Returns:
        A `PreparedCorpus` with both flat and chunked views.
    """
    if plan is None:
        plan = CorpusPlan()

    rng = np.random.default_rng(plan.seed)
    n_raw = len(raw_tokens)

    # Step 1: count word frequencies in the raw stream.
    counts = Counter(raw_tokens)
    total = sum(counts.values())

    # Step 2: build vocab with min_count filter.
    most_common = [tok for tok, c in counts.most_common() if c >= plan.min_count]
    vocab = ["<unk>", *most_common]
    token_to_id = {tok: i for i, tok in enumerate(vocab)}

    # Step 3: Mikolov subsampling. For each token in the raw stream,
    # drop with probability 1 - sqrt(t / f(w)). f(w) is the *unigram
    # frequency* — fraction of total tokens that are this word.
    if plan.subsample_threshold > 0:
        # Pre-compute keep probability per word.
        t = plan.subsample_threshold
        keep_prob: dict[str, float] = {}
        for tok, c in counts.items():
            f = c / total
            if f <= t:
                keep_prob[tok] = 1.0
            else:
                # Mikolov 2013, eq. (5): P(keep) = sqrt(t/f) + t/f
                # Some implementations use just sqrt(t/f). We use the
                # original two-term formula matching gensim's default.
                keep_prob[tok] = (t / f) ** 0.5 + (t / f)
        # Sample drops; tokens not in keep_prob (i.e. filtered by
        # min_count below) are dropped here too.
        kept: list[str] = []
        random_draws = rng.random(n_raw)
        for tok, r in zip(raw_tokens, random_draws, strict=False):
            if tok not in token_to_id:
                continue  # filtered by min_count
            if r < keep_prob.get(tok, 1.0):
                kept.append(tok)
    else:
        kept = [t for t in raw_tokens if t in token_to_id]

    n_kept = len(kept)

    # Step 4: chunk into pseudo-sentences.
    if plan.chunk_size > 0:
        chunks = [
            kept[i : i + plan.chunk_size] for i in range(0, n_kept, plan.chunk_size)
        ]
    else:
        chunks = [kept]

    # Step 5: shuffle chunks if requested (preserves within-chunk locality).
    if plan.shuffle_chunks and len(chunks) > 1:
        order = rng.permutation(len(chunks))
        chunks = [chunks[i] for i in order]

    flat_tokens = [t for chunk in chunks for t in chunk]
    flat_token_ids = [token_to_id.get(t, 0) for t in flat_tokens]

    return PreparedCorpus(
        flat_tokens=flat_tokens,
        chunks=chunks,
        vocab=vocab,
        token_to_id=token_to_id,
        flat_token_ids=flat_token_ids,
        plan=plan,
        stats={
            "n_raw_tokens": n_raw,
            "n_kept_tokens": n_kept,
            "fraction_kept": n_kept / max(n_raw, 1),
            "vocab_size": len(vocab),
            "n_chunks": len(chunks),
            "mean_chunk_size": float(np.mean([len(c) for c in chunks]))
            if chunks
            else 0.0,
            "shuffled": plan.shuffle_chunks,
        },
    )
