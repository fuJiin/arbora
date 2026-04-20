"""Shared k-WTA helper for hippocampal layers.

Binary k-Winners-Take-All: returns a bool vector with the top-k entries
of the drive active. Used by EC forward, DG, and CA3's attractor
dynamics. Centralized here so sparsification behavior stays consistent
across the module.
"""

from __future__ import annotations

import numpy as np


def kwta(drive: np.ndarray, k: int) -> np.ndarray:
    """Return a sparse binary vector with the top-k entries of `drive` active.

    Parameters
    ----------
    drive : np.ndarray, shape (n,)
        Continuous drive signal.
    k : int
        Number of active units. If k >= n, returns all-True.

    Returns
    -------
    np.ndarray, dtype=bool, shape=(n,)
    """
    n = drive.shape[0]
    out = np.zeros(n, dtype=np.bool_)
    if k >= n:
        out[:] = True
        return out
    top_k = np.argpartition(drive, -k)[-k:]
    out[top_k] = True
    return out
