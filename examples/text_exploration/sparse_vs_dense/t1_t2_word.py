"""T1 + T2 word-level baseline for ARB-139.

Adds T2 above T1 to test the architectural bet that hierarchy +
sparse + local-Hebbian produces semantic structure that flat T1
alone doesn't. T2 receives T1's L2/3 as feedforward (with a temporal
buffer so T2 sees a sliding window of recent T1 states), and T2.l5
projects back to T1.l23 / T1.l5 as apical context.

Key design choices for the conviction comparison:

- **Inputs are still one-hot per word.** Same encoder as T1-only path.
- **T2 input_dim = T1.n_l23_total * buffer_depth.** Default buffer
  depth = 4 (matches the chat preset default), so T2 sees a temporal
  window of the last 4 T1 L2/3 states. This is the "context-aware"
  representation T2 forms — exactly the contextual signal that
  context-free T1 extraction was discarding.
- **Apical T2→T1 enabled by default** (thalamic-gated, matches chat
  preset). The apical feedback is what lets higher regions sharpen
  lower-region predictions; disabling makes this just a feedforward
  cascade.
- **Embeddings extracted from T2's L2/3** (context-free reset+1step,
  same protocol as T1). For T2, "present a word" means: reset both
  regions, run circuit.process(one-hot(word_id)), read T2.l23.active.
  T2's representation reflects whatever T2 learned about that word's
  L2/3-pattern signature during training.
"""

from __future__ import annotations

import time

import numpy as np

from arbora.config import (
    _default_region2_config,
    _default_t1_config,
    make_sensory_region,
)
from arbora.cortex.circuit import Circuit, ConnectionRole
from arbora.cortex.modulators import ThalamicGate
from examples.text_exploration.sparse_vs_dense.t1_word import (
    T1WordEmbeddings,
    _OneHotIDEncoder,
)


def build_t1_t2_circuit(
    *,
    vocab_size: int,
    t1_cols: int = 128,
    t1_k: int = 8,
    t2_cols: int = 256,
    t2_k: int = 16,
    buffer_depth: int = 4,
    apical: bool = True,
    seed: int = 0,
):
    """Wire T1 → T2 (FF) and optionally T2.l5 → T1.{l23,l5} (APICAL)."""
    encoder = _OneHotIDEncoder(vocab_size=vocab_size)

    t1_cfg = _default_t1_config()
    t1_cfg.n_columns = t1_cols
    t1_cfg.k_columns = t1_k
    # Need L5 if we wire apical *to* T1.l5 from above.
    if not apical:
        t1_cfg.n_l5 = 0
    t1_cfg.ltd_rate = 0.20
    t1_cfg.synapse_decay = 0.999
    t1_cfg.learning_rate = 0.02
    t1_cfg.pre_trace_decay = 0.5
    t1 = make_sensory_region(t1_cfg, input_dim=vocab_size, encoding_width=0, seed=seed)

    t2_cfg = _default_region2_config()
    t2_cfg.n_columns = t2_cols
    t2_cfg.k_columns = t2_k
    t2_cfg.n_l5 = 4 if apical else 0  # T2 needs L5 to source apical
    t2 = make_sensory_region(
        t2_cfg,
        input_dim=t1.n_l23_total * buffer_depth,
        encoding_width=0,
        seed=seed + 1,
    )

    circuit = Circuit(encoder)
    circuit.add_region("T1", t1, entry=True, input_region=True)
    circuit.add_region("T2", t2)

    # T1 → T2 feedforward with temporal buffer (T2 sees a sliding
    # window of the last `buffer_depth` T1.l23 states).
    circuit.connect(
        t1.l23,
        t2.l4,
        ConnectionRole.FEEDFORWARD,
        buffer_depth=buffer_depth,
    )

    if apical:
        # T2.l5 → T1.{l23, l5} apical feedback, thalamic-gated.
        gate = ThalamicGate
        circuit.connect(t2.l5, t1.l23, ConnectionRole.APICAL, thalamic_gate=gate())
        circuit.connect(t2.l5, t1.l5, ConnectionRole.APICAL, thalamic_gate=gate())

    circuit.finalize()
    return circuit, encoder, t1, t2


def train_t1_t2_word(
    token_ids: list[int],
    *,
    id_to_token: list[str],
    epochs: int = 1,
    log_every: int = 0,
    apical: bool = True,
    t1_cols: int = 128,
    t1_k: int = 8,
    t2_cols: int = 256,
    t2_k: int = 16,
    buffer_depth: int = 4,
    seed: int = 0,
) -> tuple[T1WordEmbeddings, T1WordEmbeddings, dict]:
    """Train T1+T2, return (t1_embeddings, t2_embeddings, stats).

    Continuous stream (no per-word reset) — same regime as T1-only.
    Both regions learn end-to-end via Hebbian on their own connections.
    """
    circuit, encoder, t1, t2 = build_t1_t2_circuit(
        vocab_size=len(id_to_token),
        t1_cols=t1_cols,
        t1_k=t1_k,
        t2_cols=t2_cols,
        t2_k=t2_k,
        buffer_depth=buffer_depth,
        apical=apical,
        seed=seed,
    )

    t0 = time.monotonic()
    for epoch in range(epochs):
        for i, tid in enumerate(token_ids):
            circuit.process(encoder.encode(tid))
            if log_every and (i + 1) % log_every == 0:
                print(
                    f"  epoch {epoch + 1}/{epochs} step {i + 1}/{len(token_ids)} "
                    f"({time.monotonic() - t0:.1f}s)"
                )

    # Embedding extraction: reset both regions, present each word
    # once, capture T1.l23 (for comparison) and T2.l23 (the headline).
    t1.learning_enabled = False
    t2.learning_enabled = False
    t1_sdrs: dict[str, np.ndarray] = {}
    t2_sdrs: dict[str, np.ndarray] = {}
    for word_id, word in enumerate(id_to_token):
        # Reset working memory on both regions; the buffer between
        # them lives on the connection — Circuit doesn't expose a
        # clean reset for it, so we just step a few zeros first to
        # flush, then the actual word.
        t1.reset_working_memory()
        t2.reset_working_memory()
        # Flush buffer with a few empty steps so T2 doesn't see stale
        # context from the prior word's surroundings.
        empty = np.zeros(len(id_to_token), dtype=np.bool_)
        for _ in range(buffer_depth):
            circuit.process(empty)
        circuit.process(encoder.encode(word_id))
        t1_sdrs[word] = t1.l23.active.copy()
        t2_sdrs[word] = t2.l23.active.copy()

    return (
        T1WordEmbeddings(t1_sdrs),
        T1WordEmbeddings(t2_sdrs),
        {
            "elapsed_s": time.monotonic() - t0,
            "vocab_size": len(id_to_token),
            "t1_n_columns": t1.n_columns,
            "t1_n_l23_total": t1.n_l23_total,
            "t2_n_columns": t2.n_columns,
            "t2_n_l23_total": t2.n_l23_total,
            "buffer_depth": buffer_depth,
            "apical": apical,
            "n_train_tokens": len(token_ids) * epochs,
            "t1_active_per_word_mean": float(
                np.mean([s.sum() for s in t1_sdrs.values()])
            ),
            "t2_active_per_word_mean": float(
                np.mean([s.sum() for s in t2_sdrs.values()])
            ),
        },
    )
