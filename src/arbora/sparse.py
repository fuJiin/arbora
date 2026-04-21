"""Sparse-activation primitives shared across the codebase.

Currently hosts the binary k-Winners-Take-All helper used by hippocampal
layers (EC forward, DG, CA3 attractor dynamics). Cortical code uses the
same `argpartition` idiom inline for column selection; that can be
migrated here once the refactor is worthwhile.
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
