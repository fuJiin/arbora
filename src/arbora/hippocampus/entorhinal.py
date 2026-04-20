"""Entorhinal layer: fixed random projection between cortex and hippocampus.

The entorhinal cortex is the primary interface between neocortex and the
hippocampal formation. In biology:

  - EC-II stellate cells project to DG via the perforant path.
  - EC-III pyramidal cells project to CA1 directly (temporoammonic path).
  - CA1 → EC-V → deep-layer cortex is the main HC output route.

For HC v1 we collapse EC-II and EC-III into a single forward projection
(the distinction matters only once EC develops learned representations in
v2). Reverse corresponds to the CA1 → EC-V → neocortex output path.

Forward is a Gaussian random projection followed by k-WTA sparsification,
yielding a sparse binary vector compatible with DG's pattern-separation
expectations. Reverse is a separate Gaussian random projection back to
cortical dimension; downstream cortical regions handle their own
sparsification.

Forward and reverse use independent random matrices — reverse is NOT the
transpose of forward. Biologically the two pathways are distinct axonal
populations; computationally, independent matrices let downstream CA1
output carry signal into cortex without reconstructing the input.

No learning in v1. Projection matrices are frozen at initialization.
"""

from __future__ import annotations

import numpy as np

from arbora.hippocampus._kwta import kwta


class EntorhinalLayer:
    """Fixed random projection between cortex and hippocampus.

    Forward projects cortical input into a sparse binary EC code (k-WTA).
    Reverse projects EC-space vectors back to cortical dimension
    (continuous, for downstream sparsification).

    Parameters
    ----------
    input_dim : int
        Cortical input dimension (e.g. upstream L2/3 total neurons).
    output_dim : int
        EC-space dimension. ~500-2000 for v1.
    sparsity : float
        Fraction of units active in the forward output. Default 0.02.
        Matches downstream DG expectation.
    seed : int
        Random seed for projection matrix initialization.

    Attributes
    ----------
    forward_weights : np.ndarray, shape (input_dim, output_dim)
        Gaussian random projection, scaled to preserve distances (JL).
    reverse_weights : np.ndarray, shape (output_dim, input_dim)
        Independent Gaussian random projection for the return path.
    k : int
        Number of active units in forward output (output_dim * sparsity).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        sparsity: float = 0.02,
        seed: int = 0,
    ):
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError(
                f"input_dim and output_dim must be positive; "
                f"got input_dim={input_dim}, output_dim={output_dim}"
            )
        if not 0.0 < sparsity <= 1.0:
            raise ValueError(f"sparsity must be in (0, 1]; got {sparsity}")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sparsity = sparsity
        self.k = max(1, round(output_dim * sparsity))

        rng = np.random.default_rng(seed)
        # JL-style scaling: entries N(0, 1/input_dim) preserve inner
        # products in expectation. Forward and reverse use independent
        # draws — the reverse is not a mathematical inverse.
        forward_scale = 1.0 / np.sqrt(input_dim)
        self.forward_weights = rng.normal(
            0.0, forward_scale, size=(input_dim, output_dim)
        )
        reverse_scale = 1.0 / np.sqrt(output_dim)
        self.reverse_weights = rng.normal(
            0.0, reverse_scale, size=(output_dim, input_dim)
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Project cortical input to sparse binary EC code.

        Parameters
        ----------
        x : np.ndarray
            Cortical input. Any shape that flattens to input_dim.
            Boolean or continuous; both are treated as real-valued.

        Returns
        -------
        np.ndarray, dtype=bool, shape=(output_dim,)
            Sparse binary vector with exactly `k` active units
            (k-WTA on the projected drive).
        """
        flat = x.reshape(-1).astype(np.float64, copy=False)
        if flat.shape[0] != self.input_dim:
            raise ValueError(
                f"input has {flat.shape[0]} elements, expected {self.input_dim}"
            )
        projected = flat @ self.forward_weights
        return kwta(projected, self.k)

    def reverse(self, y: np.ndarray) -> np.ndarray:
        """Project EC-space vector back to cortical dimension.

        Parameters
        ----------
        y : np.ndarray
            EC-space vector, shape (output_dim,). Boolean or continuous.

        Returns
        -------
        np.ndarray, dtype=float64, shape=(input_dim,)
            Continuous vector. Downstream regions apply their own k-WTA.
        """
        flat = y.reshape(-1).astype(np.float64, copy=False)
        if flat.shape[0] != self.output_dim:
            raise ValueError(
                f"input has {flat.shape[0]} elements, expected {self.output_dim}"
            )
        return flat @ self.reverse_weights
