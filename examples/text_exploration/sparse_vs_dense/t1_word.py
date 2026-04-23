"""Word-level T1 baseline for ARB-139.

Each step processes one word (one-hot input over `vocab_size`). After
training, the embedding for word `w` is the L2/3 active SDR produced
when w is presented after a reset (i.e., context-free representation).

Design decisions:

- **One-hot per word, vocab_size = input_dim.** Same input format
  word2vec sees. If we want to test whether morphological structure
  (CharbitEncoder) helps, that's a separate experiment.
- **No reset between words during training.** text8 has no document
  boundaries; the natural regime is continuous streaming. Each word's
  step uses lateral context from prior words (the "n-gram-like"
  contextual signal that segments learn from).
- **Context-free embeddings at eval.** For each vocab word, reset the
  region, present the word once, capture L2/3.active. That's the
  embedding. (Alternative: average L2/3 over all training contexts the
  word appeared in. Saved for later if needed.)
"""

from __future__ import annotations

import time

import numpy as np

from arbora.config import _default_t1_config, make_sensory_region


class _OneHotIDEncoder:
    """One-hot encoder over an integer ID space. Duck-typed for `T1Trainer`.

    Caches encoded vectors to avoid reallocating one bool array per
    step. The region's fast path uses `np.flatnonzero` on the bool
    array, which is fine to share across calls (the array is read,
    not mutated).
    """

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self.input_dim = vocab_size
        self.encoding_width = 0  # no positional sub-structure
        self._cache: dict[int, np.ndarray] = {}

    def encode(self, word_id: int) -> np.ndarray:
        v = self._cache.get(word_id)
        if v is None:
            v = np.zeros(self.vocab_size, dtype=np.bool_)
            v[word_id] = True
            self._cache[word_id] = v
        return v


class T1WordEmbeddings:
    """`Embeddings`-compatible wrapper for trained T1 sparse SDRs."""

    name = "t1_sparse"

    def __init__(self, sdrs: dict[str, np.ndarray]) -> None:
        self._sdrs = sdrs

    def vocab(self) -> list[str]:
        return list(self._sdrs.keys())

    def get(self, word: str) -> np.ndarray | None:
        return self._sdrs.get(word)

    def is_sparse(self) -> bool:
        return True


def build_t1_for_words(
    *,
    vocab_size: int,
    n_columns: int = 256,
    k_columns: int = 16,
    n_l4: int = 4,
    n_l23: int = 4,
    pre_trace_decay: float = 0.5,
    ltd_rate: float = 0.20,
    synapse_decay: float = 0.999,
    learning_rate: float = 0.02,
    seed: int = 0,
):
    """Build a T1 region sized for word-level processing.

    Defaults follow the ARB-131 sweep findings (ptd=0.50, ltd=0.20,
    decay=0.999, lr=0.02). `n_columns=256` is up from the char-level
    default of 128 — vocab is 100-1000x larger so we want more capacity.
    `n_l5=0` since nothing reads it.
    """
    cfg = _default_t1_config()
    cfg.n_l5 = 0
    cfg.n_columns = n_columns
    cfg.k_columns = k_columns
    cfg.n_l4 = n_l4
    cfg.n_l23 = n_l23
    cfg.ltd_rate = ltd_rate
    cfg.synapse_decay = synapse_decay
    cfg.learning_rate = learning_rate
    cfg.pre_trace_decay = pre_trace_decay
    return make_sensory_region(cfg, input_dim=vocab_size, encoding_width=0, seed=seed)


def train_t1_word(
    token_ids: list[int],
    *,
    id_to_token: list[str],
    epochs: int = 1,
    log_every: int = 0,
    region_kwargs: dict | None = None,
    seed: int = 0,
) -> tuple[T1WordEmbeddings, dict]:
    """Train word-level T1 on `token_ids`, return embeddings + stats.

    No per-word reset (continuous stream — text8 is one corpus). Per-
    epoch shuffle is intentional: reshuffles the word ORDER across
    epochs, even though within an epoch the order matters (sequence
    learning). This is a different regime than word2vec which uses a
    sliding context window — but it's the natural T1 regime.

    Embeddings are extracted context-free: reset, present each vocab
    word once, capture L2/3 active.
    """
    region_kwargs = region_kwargs or {}
    encoder = _OneHotIDEncoder(vocab_size=len(id_to_token))
    region = build_t1_for_words(vocab_size=len(id_to_token), seed=seed, **region_kwargs)

    t0 = time.monotonic()
    for epoch in range(epochs):
        # Shuffle is debatable here — for true sequence learning we'd
        # keep order. For sample-efficiency comparison vs word2vec we
        # iterate in order (matches text8's distributional structure).
        # No shuffle.
        for i, tid in enumerate(token_ids):
            encoding = encoder.encode(tid)
            region.process(encoding)
            if log_every and (i + 1) % log_every == 0:
                print(
                    f"  epoch {epoch + 1}/{epochs} step {i + 1}/{len(token_ids)} "
                    f"({time.monotonic() - t0:.1f}s)"
                )

    # Extract context-free embeddings: reset region, present each word,
    # capture L2/3 active.
    sdrs: dict[str, np.ndarray] = {}
    region.learning_enabled = False
    for word_id, word in enumerate(id_to_token):
        region.reset_working_memory()
        region.process(encoder.encode(word_id))
        sdrs[word] = region.l23.active.copy()

    return T1WordEmbeddings(sdrs), {
        "elapsed_s": time.monotonic() - t0,
        "vocab_size": len(id_to_token),
        "n_columns": region.n_columns,
        "n_l23_total": region.n_l23_total,
        "n_train_tokens": len(token_ids) * epochs,
        "active_per_word_mean": float(np.mean([s.sum() for s in sdrs.values()])),
    }
