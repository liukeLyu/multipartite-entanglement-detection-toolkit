"""Genuine multipartite concurrence (GMC).

For pure states, ``C_GME(psi) = min_gamma sqrt(2 (1 - tr rho_gamma^2))`` over
bipartitions ``gamma``, extended to mixed states by the convex roof [1]. The
convex roof is hard to evaluate, but it is known exactly for X states [2] and
bounded from below by the ``I2`` criterion of Ma et al. [1].

[1] Z.-H. Ma, Z.-H. Chen, J.-L. Chen, C. Spengler, A. Gabriel, M. Huber,
    PRA 83, 062325 (2011).
[2] S. M. Hashemi Rafsanjani, M. Huber, C. J. Broadbent, J. H. Eberly,
    PRA 86, 062303 (2012).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.optimize import minimize

from medt.linalg import bipartitions, partial_trace


def gmc_pure(psi: np.ndarray, dims: Sequence[int] | None = None) -> float:
    """Genuine multipartite concurrence of a pure state."""
    psi = np.asarray(psi, dtype=complex).ravel()
    psi = psi / np.linalg.norm(psi)
    if dims is None:
        dims = (2,) * int(round(np.log2(psi.size)))
    rho = np.outer(psi, psi.conj())
    purities = [np.real(np.trace(r @ r)) for r in
                (partial_trace(rho, M, dims) for M in bipartitions(len(dims)))]
    return float(np.sqrt(max(0.0, 2 * (1 - max(purities)))))


def x_state_parts(rho: np.ndarray, tol: float = 1e-10):
    """Split an N-qubit X state into ``(a, b, z)``.

    ``a[i] = rho[i, i]``, ``b[i] = rho[D-1-i, D-1-i]``, ``z[i] = rho[i, D-1-i]`` for
    ``i < D/2``. Raises ``ValueError`` if ``rho`` has weight outside the X shape.
    """
    rho = np.asarray(rho)
    D = rho.shape[0]
    mask = np.eye(D, dtype=bool) | np.fliplr(np.eye(D, dtype=bool))
    if np.abs(rho[~mask]).max(initial=0) > tol:
        raise ValueError("rho is not an X state")
    i = np.arange(D // 2)
    return rho[i, i].real, rho[D - 1 - i, D - 1 - i].real, rho[i, D - 1 - i]


def gmc_x_state(rho: np.ndarray) -> float:
    """Exact genuine multipartite concurrence of an N-qubit X state [2].

    ``C = 2 max(0, max_i |z_i| - sum_{j != i} sqrt(a_j b_j))``.
    """
    a, b, z = x_state_parts(rho)
    s = np.sqrt(np.clip(a * b, 0, None))
    return float(2 * max(0.0, np.max(np.abs(z) - (s.sum() - s))))


def _i2(rho, kets1, kets2, cuts):
    def kron_all(kets):
        out = np.ones(1, dtype=complex)
        for k in kets:
            out = np.kron(out, k)
        return out

    phi1, phi2 = kron_all(kets1), kron_all(kets2)
    value = np.abs(phi1.conj() @ rho @ phi2)
    for M in cuts:
        s1 = [kets2[k] if k in M else kets1[k] for k in range(len(kets1))]
        s2 = [kets1[k] if k in M else kets2[k] for k in range(len(kets1))]
        u, v = kron_all(s1), kron_all(s2)
        value -= np.sqrt(max(0.0, np.real(u.conj() @ rho @ u) * np.real(v.conj() @ rho @ v)))
    return value


def gmc_lower_bound(
    rho: np.ndarray,
    dims: Sequence[int] | None = None,
    *,
    n_starts: int = 20,
    seed: int | None = None,
) -> float:
    """``I2`` lower bound on the genuine multipartite concurrence [1].

    Maximizes ``2 (|<phi1|rho|phi2>| - sum_gamma sqrt(<phi1'|rho|phi1'> <phi2'|rho|phi2'>))``
    over fully product kets ``phi1, phi2``, where ``phi1', phi2'`` swap the
    parties in ``gamma`` between them. A positive value certifies genuine
    multipartite entanglement. The optimization is local with random restarts,
    so the result is a valid lower bound but may not be the best one.
    """
    rho = np.asarray(rho, dtype=complex)
    if dims is None:
        dims = (2,) * int(round(np.log2(rho.shape[0])))
    dims = tuple(dims)
    cuts = bipartitions(len(dims))
    offsets = np.cumsum((0,) + dims)
    n_real = 2 * offsets[-1]

    def unpack(x):
        c = x[: offsets[-1]] + 1j * x[offsets[-1] :]
        kets = [c[offsets[k] : offsets[k + 1]] for k in range(len(dims))]
        return [k / np.linalg.norm(k) for k in kets]

    def objective(x):
        return -_i2(rho, unpack(x[:n_real]), unpack(x[n_real:]), cuts)

    rng = np.random.default_rng(seed)
    starts = [rng.normal(size=2 * n_real) for _ in range(n_starts)]
    best = max(-minimize(objective, x0, method="BFGS").fun for x0 in starts)
    return float(2 * max(0.0, best))
