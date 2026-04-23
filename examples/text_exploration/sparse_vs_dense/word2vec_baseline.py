"""gensim Word2Vec baseline for ARB-139.

Wraps gensim's Word2Vec into the `Embeddings` interface so it slots into
the same evaluation harness as our T1 sparse embeddings.

Validation: text8-trained Skip-gram should hit roughly SimLex Spearman
~0.30 and Google analogy ~30-40% per the gensim tutorial. If we don't
hit those, the pipeline has a bug — we shouldn't trust the sparse
comparison until this is calibrated.
"""

from __future__ import annotations

import time

import numpy as np


class Word2VecEmbeddings:
    """`Embeddings`-compatible wrapper around gensim's KeyedVectors."""

    name = "word2vec"

    def __init__(self, kv) -> None:
        self.kv = kv  # gensim.models.keyedvectors.KeyedVectors

    def vocab(self) -> list[str]:
        return list(self.kv.key_to_index.keys())

    def get(self, word: str) -> np.ndarray | None:
        if word not in self.kv.key_to_index:
            return None
        return self.kv[word].astype(np.float32)

    def is_sparse(self) -> bool:
        return False


def train_word2vec(
    tokens: list[str],
    *,
    vocab: list[str] | None = None,
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 5,
    epochs: int = 5,
    workers: int = 4,
    sg: int = 1,  # 1 = skip-gram (matches Mikolov; cbow=0)
    seed: int = 0,
) -> tuple[Word2VecEmbeddings, dict]:
    """Train Skip-gram word2vec on `tokens`. Returns (embeddings, stats).

    If `vocab` is provided, the model's vocabulary is restricted to
    those words (others map to OOV) — this lets us share vocab with T1
    so the comparison is apples-to-apples.
    """
    from gensim.models import Word2Vec

    t0 = time.monotonic()
    # Chunk the stream into manageable "sentences." gensim doesn't
    # iterate epochs reliably on a single giant sentence — the default
    # text8-style preprocessing in gensim's own tutorials chunks into
    # ~1000-token sentences. Same here.
    SENTENCE_LEN = 1000
    sentences = [
        tokens[i : i + SENTENCE_LEN] for i in range(0, len(tokens), SENTENCE_LEN)
    ]
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        epochs=epochs,
        workers=workers,
        sg=sg,
        seed=seed,
    )

    if vocab is not None:
        # Restrict KeyedVectors to the shared vocab. Words in `vocab` not
        # in the trained model are silently dropped (will be OOV at eval).
        present = [w for w in vocab if w in model.wv.key_to_index]
        from gensim.models.keyedvectors import KeyedVectors

        kv = KeyedVectors(vector_size=vector_size)
        kv.add_vectors(present, [model.wv[w] for w in present])
        out = Word2VecEmbeddings(kv)
    else:
        out = Word2VecEmbeddings(model.wv)

    return out, {
        "elapsed_s": time.monotonic() - t0,
        "vocab_size": len(out.vocab()),
        "vector_size": vector_size,
        "epochs": epochs,
        "n_train_tokens": len(tokens),
    }
