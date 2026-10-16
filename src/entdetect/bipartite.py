"""Bipartite entanglement criteria."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from entdetect.linalg import partial_transpose, realign


def log_negativity(rho: np.ndarray, sys: Sequence[int], dims: Sequence[int]) -> float:
    """Logarithmic negativity ``log2 ||rho^{T_sys}||_1``."""
    eig = np.linalg.eigvalsh(partial_transpose(rho, sys, dims))
    return float(np.log2(np.abs(eig).sum()))


def is_ppt(rho: np.ndarray, sys: Sequence[int], dims: Sequence[int], tol: float = 1e-10) -> bool:
    """Whether ``rho`` has a positive partial transpose on ``sys``."""
    return bool(np.linalg.eigvalsh(partial_transpose(rho, sys, dims)).min() >= -tol)


def ccnr(rho: np.ndarray, dims: Sequence[int]) -> float:
    """Trace norm of the realigned state; a value above 1 certifies entanglement.

    This is the computable cross-norm or realignment criterion of Rudolph
    (Quantum Inf. Process. 4, 219 (2005)) and Chen and Wu (Quantum Inf.
    Comput. 3, 193 (2003)). It detects some PPT entangled states.
    """
    return float(np.linalg.svd(realign(rho, dims), compute_uv=False).sum())
