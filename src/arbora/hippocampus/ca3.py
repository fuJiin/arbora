"""CA3: Hebbian recurrent attractor network with one-shot binding.

CA3 is the episodic memory substrate of the hippocampus. It receives a
strong sparse "detonator" input from DG via the mossy fibers, binds
coincident activity through Hebbian LTP on its recurrent (lateral)
collaterals, and performs pattern completion from partial cues.

Biology:
  - DG → CA3 mossy fibers: ~14 contacts per DG neuron onto CA3 pyramidal
    cells, via very large synapses ("detonator synapses"). A single DG
    spike can drive a CA3 cell to threshold.
  - CA3 recurrent collaterals: each pyramidal cell synapses on ~10^4
    others, making CA3 a dense auto-associative network. We call this
    the `lateral_weights` matrix, consistent with cortical within-region
    naming.
  - BTSP (behavioral timescale synaptic plasticity): a single supra-
    threshold plateau potential triggers persistent weight change
    across a ~2-second window (Bittner et al. 2017). Biological basis
    for one-shot binding.

v1 simplifications:
  - Mossy fibers: fixed sparse random projection. Each CA3 unit
    receives from `mossy_k` DG units with strong weight `mossy_weight`.
  - Lateral: dense Hebbian LTP only (no LTD). Co-active pairs gain
    weight on encode; BTSP reduces to "a single encode call is enough
    for retrieval."
  - No three-factor / DA modulation; no theta phases; no replay. CA3
    lateral weights are the entire HC memory in v1.

Reset semantics:
  - `reset()` clears the current activation state only. Lateral
    weights (the memory) are preserved.
  - `reset_memory()` clears state AND lateral weights. Use at task
    boundaries as a crude consolidation-via-forgetting when cortical
    drift has invalidated stored keys (see ARB-105 spec §3).
"""

from __future__ import annotations

import numpy as np

from arbora.sparse import kwta


class CA3:
    """Hebbian recurrent attractor with BTSP-like one-shot binding.

    Parameters
    ----------
    dim : int
        CA3 unit count.
    dg_dim : int
        Upstream DG dimension (for mossy input).
    mossy_k : int | None
        Number of DG units projecting to each CA3 unit. If None,
        derived from `mossy_sparsity`.
    mossy_sparsity : float
        Fraction of DG units projecting to each CA3 unit. Ignored if
        `mossy_k` is set. Default 0.02.
    mossy_weight : float
        Per-contact strength for mossy fiber input. Default 2.0.
    k_active : float
        Fraction of CA3 units active after k-WTA. Default 0.02.
    learning_rate : float
        LTP increment per encode call (one-shot). Default 0.5.
    seed : int
        Random seed for mossy mask.

    Attributes
    ----------
    mossy_weights : np.ndarray, shape (dg_dim, dim)
        Sparse mossy projection (non-zero entries = mossy_weight).
    lateral_weights : np.ndarray, shape (dim, dim)
        Learned within-region recurrent collaterals. Clipped to [0, 1].
    state : np.ndarray, dtype=bool, shape (dim,)
        Current activation. Updated by encode/retrieve.
    """

    def __init__(
        self,
        dim: int,
        dg_dim: int,
        *,
        mossy_k: int | None = None,
        mossy_sparsity: float = 0.02,
        mossy_weight: float = 2.0,
        k_active: float = 0.02,
        learning_rate: float = 0.5,
        seed: int = 0,
    ):
        if dim <= 0 or dg_dim <= 0:
            raise ValueError(f"dims must be positive; got dim={dim}, dg_dim={dg_dim}")
        if not 0.0 < k_active <= 1.0:
            raise ValueError(f"k_active must be in (0, 1]; got {k_active}")

        self.dim = dim
        self.dg_dim = dg_dim
        self.k = max(1, round(dim * k_active))
        self.learning_rate = learning_rate

        if mossy_k is None:
            mossy_k = max(1, round(dg_dim * mossy_sparsity))
        if mossy_k < 1 or mossy_k > dg_dim:
            raise ValueError(
                f"mossy_k must be in [1, dg_dim]; "
                f"got mossy_k={mossy_k}, dg_dim={dg_dim}"
            )
        self.mossy_k = mossy_k
        self.mossy_weight = mossy_weight
        # Plasticity gate. When False, `encode()` still computes the
        # mossy-driven activation and updates `state`, but skips the
        # Hebbian LTP on `lateral_weights`. Used by probes that need a
        # non-destructive read of "what would this input retrieve?"
        # without perturbing the learned memory.
        self.learning_enabled: bool = True

        rng = np.random.default_rng(seed)
        # Sparse mossy mask: each CA3 cell receives from mossy_k DG cells.
        self.mossy_weights = np.zeros((dg_dim, dim))
        for i in range(dim):
            idx = rng.choice(dg_dim, size=mossy_k, replace=False)
            self.mossy_weights[idx, i] = mossy_weight

        self.lateral_weights = np.zeros((dim, dim))
        self.state = np.zeros(dim, dtype=np.bool_)

    def encode(self, dg_pattern: np.ndarray) -> np.ndarray:
        """Bind a DG input pattern as a CA3 attractor.

        One-shot Hebbian LTP: compute CA3 activation from DG detonator
        via mossy weights, apply k-WTA, then strengthen lateral weights
        among co-active CA3 units. Stores result as current state.

        Returns a copy of the CA3 activation pattern.
        """
        flat = dg_pattern.reshape(-1).astype(np.float64, copy=False)
        if flat.shape[0] != self.dg_dim:
            raise ValueError(
                f"dg_pattern has {flat.shape[0]} elements, expected {self.dg_dim}"
            )

        drive = flat @ self.mossy_weights
        active = kwta(drive, self.k)

        if self.learning_enabled and active.any():
            idx = np.flatnonzero(active)
            self.lateral_weights[np.ix_(idx, idx)] += self.learning_rate
            np.fill_diagonal(self.lateral_weights, 0.0)
            np.clip(self.lateral_weights, 0.0, 1.0, out=self.lateral_weights)

        self.state[:] = active
        return active.copy()

    def retrieve(self, partial_cue: np.ndarray, n_iter: int = 3) -> np.ndarray:
        """Pattern-complete from a partial cue via recurrent dynamics.

        Parameters
        ----------
        partial_cue : np.ndarray, shape (dim,)
            Partial CA3 pattern. Boolean or continuous.
        n_iter : int
            Recurrent iterations. Default 3.

        Returns
        -------
        np.ndarray, dtype=bool, shape (dim,)
            Completed pattern. Also stored in `state`.
        """
        flat = partial_cue.reshape(-1)
        if flat.shape[0] != self.dim:
            raise ValueError(
                f"partial_cue has {flat.shape[0]} elements, expected {self.dim}"
            )

        current = flat.astype(np.float64, copy=True)
        for _ in range(n_iter):
            drive = current @ self.lateral_weights
            current = kwta(drive, self.k).astype(np.float64)

        result = current.astype(np.bool_)
        self.state[:] = result
        return result.copy()

    def reset(self) -> None:
        """Clear current activation state. Preserves lateral weights."""
        self.state[:] = False

    def reset_memory(self) -> None:
        """Clear both activation state and lateral weights.

        Episodic memory reset. Call at task boundaries when cortical
        drift has invalidated stored CA3 keys — crude consolidation-
        via-forgetting until full replay (ARB-120) lands.
        """
        self.state[:] = False
        self.lateral_weights[:] = 0.0
