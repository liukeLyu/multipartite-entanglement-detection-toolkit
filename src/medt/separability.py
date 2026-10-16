"""Certifying separability.

Kampermann, Gühne, Wilmott and Bruß [1] certify that a state lies in a convex
set by repeatedly subtracting extremal points of the set (here product states)
with large overlap. Each step keeps the remainder a valid state and makes it
more mixed. Once the remainder is inside a ball of separable states around
the maximally mixed state, the original state is a mixture of the subtracted
product states and that remainder, hence separable.

For a bipartite state of total dimension ``D``, every state with
``tr rho^2 <= 1/(D - 1)`` is separable (Gurvits and Barnum [2]).

[1] H. Kampermann, O. Gühne, C. Wilmott, D. Bruß, PRA 86, 032307 (2012).
[2] L. Gurvits, H. Barnum, PRA 66, 062311 (2002).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.linalg import sqrtm


@dataclass
class SeparabilityResult:
    separable: bool
    """True only when separability is certified; False means inconclusive."""
    iterations: int
    purity: float
    """Purity of the final remainder; certification needs ``purity <= 1/(D-1)``."""
    remainder: np.ndarray


def max_product_overlap(
    op: np.ndarray, dims: Sequence[int], *, n_starts: int = 5, max_iter: int = 500,
    tol: float = 1e-12, rng: np.random.Generator | None = None,
) -> tuple[float, np.ndarray]:
    """Maximize ``<a b| op |a b>`` over product kets by alternating eigenvector updates."""
    rng = np.random.default_rng() if rng is None else rng
    dA, dB = dims
    t = op.reshape(dA, dB, dA, dB)
    best, best_ket = -np.inf, None
    for _ in range(n_starts):
        a = rng.normal(size=dA) + 1j * rng.normal(size=dA)
        a /= np.linalg.norm(a)
        prev = -np.inf
        for _ in range(max_iter):
            XB = np.einsum("i,ijkl,k->jl", a.conj(), t, a)
            b = np.linalg.eigh(XB)[1][:, -1]
            XA = np.einsum("j,ijkl,l->ik", b.conj(), t, b)
            w, v = np.linalg.eigh(XA)
            a, value = v[:, -1], w[-1]
            if value - prev < tol:
                break
            prev = value
        if value > best:
            best, best_ket = value, np.kron(a, b)
    return float(best), best_ket


def kampermann_sep(
    rho: np.ndarray,
    dims: Sequence[int],
    *,
    max_iter: int = 10_000,
    step: float = 1.0,
    seed: int | None = None,
) -> SeparabilityResult:
    """Try to certify that a bipartite state is separable [1].

    The state should have full rank, since each step can only remove as much
    weight as the smallest eigenvalue allows. Increase ``max_iter`` for states
    close to the boundary of the separable set.
    """
    rho = np.asarray(rho, dtype=complex)
    dims = tuple(int(d) for d in dims)
    if len(dims) != 2 or rho.shape != (np.prod(dims),) * 2:
        raise ValueError("kampermann_sep takes a bipartite state with dims (dA, dB)")
    rng = np.random.default_rng(seed)
    threshold = 1 / (rho.shape[0] - 1)
    r = (rho + rho.conj().T) / 2
    for k in range(max_iter + 1):
        purity = float(np.real(np.trace(r @ r)))
        if purity <= threshold:
            return SeparabilityResult(True, k, purity, r)
        if k == max_iter:
            break
        # Searching with sqrt(r) flattens the spectrum and finds better directions.
        _, psi = max_product_overlap(sqrtm(r), dims, rng=rng)
        c = float(np.real(psi.conj() @ r @ psi))
        if c <= purity:
            continue
        eps = min((c - purity) / (1 - c), step * np.linalg.eigvalsh(r)[0])
        if eps <= 0:
            break
        r = (r - eps * np.outer(psi, psi.conj())) / (1 - eps)
        r = (r + r.conj().T) / 2
    return SeparabilityResult(False, k, purity, r)
