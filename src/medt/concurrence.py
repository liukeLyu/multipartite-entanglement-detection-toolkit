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


def _gmc_unnormalized(phi_rows, dims, cuts):
    """``f(phi) = sqrt(2 (n^2 - max_M tr rho_M^2))`` per row and its Wirtinger gradient."""
    K, D = phi_rows.shape
    n = np.einsum("kd,kd->k", phi_rows.conj(), phi_rows).real
    best_P = np.full(K, -np.inf)
    best_grad = np.zeros_like(phi_rows)
    for M in cuts:
        rest = [k for k in range(len(dims)) if k not in M]
        dA = int(np.prod([dims[k] for k in M]))
        t = phi_rows.reshape((K,) + tuple(dims)).transpose([0] + [1 + k for k in M] + [1 + k for k in rest])
        Phi = t.reshape(K, dA, D // dA)
        rhoA = Phi @ Phi.conj().transpose(0, 2, 1)
        P = np.einsum("kab,kba->k", rhoA, rhoA).real
        g = (rhoA @ Phi).reshape((K,) + tuple(dims[k] for k in list(M) + rest))
        inv = np.argsort([0] + [1 + k for k in M] + [1 + k for k in rest])
        g = g.transpose(inv).reshape(K, D)
        better = P > best_P
        best_P[better], best_grad[better] = P[better], g[better]
    f2 = np.clip(2 * (n**2 - best_P), 0, None)
    f = np.sqrt(f2)
    with np.errstate(divide="ignore", invalid="ignore"):
        grad = np.where(f[:, None] > 1e-14, (2 * n[:, None] * phi_rows - 2 * best_grad) / f[:, None], 0)
    return f, grad


def gmc_convex_roof(
    rho: np.ndarray,
    dims: Sequence[int] | None = None,
    *,
    n_terms: int | None = None,
    n_starts: int = 5,
    max_iter: int = 2000,
    tol: float = 1e-10,
    seed: int | None = None,
) -> float:
    """Upper bound on the genuine multipartite concurrence by convex-roof optimization.

    Every decomposition ``rho = sum_k |phi_k><phi_k|`` into ``n_terms``
    unnormalized pure states has the form ``phi_k = sum_j U_kj sqrt(l_j) |e_j>``
    with ``U`` an isometry and ``(l_j, e_j)`` the eigenpairs of ``rho`` [1].
    This minimizes ``sum_k ||phi_k||^2 C(phi_k / ||phi_k||)`` over isometries by
    Riemannian gradient descent on the Stiefel manifold, as used in Lyu et al. [3].
    Any decomposition gives an upper bound; the minimum is the convex roof.

    [3] L. Lyu, M. Song, T.-T. Wang, Z. Y. Meng, W. Witczak-Krempa, "Multiparty
        entanglement microscopy of quantum Ising models in 1d, 2d and 3d",
        arXiv:2412.12533.
    """
    rho = np.asarray(rho, dtype=complex)
    if dims is None:
        dims = (2,) * int(round(np.log2(rho.shape[0])))
    dims = tuple(dims)
    cuts = bipartitions(len(dims))
    w, v = np.linalg.eigh((rho + rho.conj().T) / 2)
    keep = w > 1e-12
    A = (np.sqrt(w[keep])[:, None] * v[:, keep].T)  # rows sqrt(l_j) e_j^T
    r = A.shape[0]
    K = n_terms or 2 * r
    rng = np.random.default_rng(seed)

    def cost_grad(U):
        f, g = _gmc_unnormalized(U @ A, dims, cuts)
        return f.sum(), g @ A.conj().T

    best = np.inf
    for _ in range(n_starts):
        U, _r = np.linalg.qr(rng.normal(size=(K, r)) + 1j * rng.normal(size=(K, r)))
        cost, G = cost_grad(U)
        step = 1.0
        for _ in range(max_iter):
            xi = G - U @ (U.conj().T @ G + G.conj().T @ U) / 2
            gnorm = np.real(np.vdot(xi, xi))
            if gnorm < tol:
                break
            while step > 1e-12:
                Unew, R = np.linalg.qr(U - step * xi)
                Unew = Unew * np.sign(np.diag(R)).conj()
                new_cost, new_G = cost_grad(Unew)
                if new_cost <= cost - 1e-4 * step * gnorm:
                    break
                step /= 2
            else:
                break
            if cost - new_cost < tol:
                U, cost, G = Unew, new_cost, new_G
                break
            U, cost, G = Unew, new_cost, new_G
            step *= 2
        best = min(best, cost)
    return float(best)
