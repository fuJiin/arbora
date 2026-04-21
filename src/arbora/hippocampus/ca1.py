"""CA1: dual-input comparator between memory (CA3) and sensory (direct EC).

CA1 pyramidal neurons receive two inputs on different dendritic
compartments:

  - CA3 via Schaffer collaterals land on proximal apical dendrites
    (stratum radiatum) — the pattern-completed "prediction from memory".
  - EC-III via the temporoammonic path lands on distal apical dendrites
    (stratum lacunosum-moleculare) — fresh sensory input.

The mismatch between these streams is the classic substrate for novelty
detection, and it is CA1 itself that performs the comparison — not EC.

v1 scope:
  - Two fixed random projections (CA3 → output, EC_direct → output),
    both Gaussian with JL-style scaling.
  - forward() returns (output_vector, match_signal).
  - match_signal is cosine similarity of the two drive vectors:
    range [-1, 1]; high = match, near-zero = novelty.
  - No learning. Learned CA1 is deferred to v2.
"""

from __future__ import annotations

import numpy as np


class CA1:
    """Dual-input comparator. Projects CA3 + EC_direct and reports match.

    Parameters
    ----------
    ca3_dim : int
        CA3 input dimension (Schaffer path).
    ec_direct_dim : int
        EC direct input dimension (temporoammonic path).
    output_dim : int
        CA1 output dimension. Typically matches EC dim so the output
        routes back through `EntorhinalCortex.reverse` to cortex.
    seed : int
        Random seed.

    Attributes
    ----------
    ca3_weights : np.ndarray, shape (ca3_dim, output_dim)
    ec_weights : np.ndarray, shape (ec_direct_dim, output_dim)
    """

    def __init__(
        self,
        ca3_dim: int,
        ec_direct_dim: int,
        output_dim: int,
        *,
        seed: int = 0,
    ):
        if ca3_dim <= 0 or ec_direct_dim <= 0 or output_dim <= 0:
            raise ValueError(
                f"dims must be positive; "
                f"got ca3_dim={ca3_dim}, ec_direct_dim={ec_direct_dim}, "
                f"output_dim={output_dim}"
            )

        self.ca3_dim = ca3_dim
        self.ec_direct_dim = ec_direct_dim
        self.output_dim = output_dim

        rng = np.random.default_rng(seed)
        self.ca3_weights = rng.normal(
            0.0, 1.0 / np.sqrt(ca3_dim), size=(ca3_dim, output_dim)
        )
        self.ec_weights = rng.normal(
            0.0, 1.0 / np.sqrt(ec_direct_dim), size=(ec_direct_dim, output_dim)
        )

    def forward(
        self,
        ca3_pattern: np.ndarray,
        ec_direct: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Compare memory (CA3) against fresh sensory (EC direct).

        Parameters
        ----------
        ca3_pattern : np.ndarray, shape (ca3_dim,)
            Output from the CA3 attractor. Boolean or continuous.
        ec_direct : np.ndarray, shape (ec_direct_dim,)
            EC direct input (temporoammonic path).

        Returns
        -------
        output : np.ndarray, shape (output_dim,)
            Summed drive vector from both paths.
        match : float
            Cosine similarity of the two drives, in [-1, 1].
            Positive = CA3 retrieval aligns with fresh input (match).
            Near zero = unrelated (novelty).
        """
        ca3_flat = ca3_pattern.reshape(-1).astype(np.float64, copy=False)
        ec_flat = ec_direct.reshape(-1).astype(np.float64, copy=False)
        if ca3_flat.shape[0] != self.ca3_dim:
            raise ValueError(
                f"ca3_pattern has {ca3_flat.shape[0]} elements, expected {self.ca3_dim}"
            )
        if ec_flat.shape[0] != self.ec_direct_dim:
            raise ValueError(
                f"ec_direct has {ec_flat.shape[0]} elements, "
                f"expected {self.ec_direct_dim}"
            )

        ca3_drive = ca3_flat @ self.ca3_weights
        ec_drive = ec_flat @ self.ec_weights
        output = ca3_drive + ec_drive
        match = _cosine(ca3_drive, ec_drive)
        return output, match


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity. Returns 0.0 when either vector has zero norm."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
