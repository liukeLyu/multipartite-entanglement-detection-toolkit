"""Bipartite entanglement criteria.

[1] A. Peres, PRL 77, 1413 (1996), arXiv:quant-ph/9604005.
[2] M. Horodecki, P. Horodecki, R. Horodecki, Phys. Lett. A 223, 1 (1996),
    arXiv:quant-ph/9605038.
[3] G. Vidal, R. F. Werner, PRA 65, 032314 (2002), arXiv:quant-ph/0102117.
[4] O. Rudolph, Quantum Inf. Process. 4, 219 (2005), arXiv:quant-ph/0202121.
[5] K. Chen, L.-A. Wu, Quantum Inf. Comput. 3, 193 (2003), arXiv:quant-ph/0205017.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from medt.linalg import partial_transpose, realign


def log_negativity(rho: np.ndarray, sys: Sequence[int], dims: Sequence[int]) -> float:
    """Logarithmic negativity ``log2 ||rho^{T_sys}||_1`` [3]."""
    eig = np.linalg.eigvalsh(partial_transpose(rho, sys, dims))
    return float(np.log2(np.abs(eig).sum()))


def is_ppt(rho: np.ndarray, sys: Sequence[int], dims: Sequence[int], tol: float = 1e-10) -> bool:
    """Whether ``rho`` has a positive partial transpose on ``sys`` [1, 2]."""
    return bool(np.linalg.eigvalsh(partial_transpose(rho, sys, dims)).min() >= -tol)


def ccnr(rho: np.ndarray, dims: Sequence[int]) -> float:
    """Trace norm of the realigned state; a value above 1 certifies entanglement.

    This is the computable cross-norm or realignment criterion [4, 5]. It
    detects some PPT entangled states.
    """
    return float(np.linalg.svd(realign(rho, dims), compute_uv=False).sum())
