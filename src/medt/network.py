"""Certifying genuine network multipartite entanglement (GNME) by ring inflation.

A tripartite state has GNME if it cannot be prepared by distributing
bipartite states between the parties and applying local channels (a triangle
network). Two copies of such a network can be rewired into a ring of six
parties. Navascués, Wolfe, Rosset and Pozas-Kerstjens [1] showed that any
network state ``rho_ABC`` then admits inflated states ``tau`` and ``gamma`` on
``A1 B1 C1 A2 B2 C2`` with matching marginals and PPT reductions. This SDP is
an outer relaxation of the network set, so if no inflation exists for a state
it has GNME.

Following Lyu et al. [2], the inflated states are restricted to the
symmetric and antisymmetric subspaces of the copy swap, which roughly halves
the variables. :func:`ring_inflation_robustness` returns the largest ``t``
for which ``t rho + (1 - t) I/D`` passes the test; ``t < 1`` certifies GNME.
The level-3 inflation of [2] (three copies of each source, inflated states on
nine parties) gives tighter bounds at a higher cost.

[1] M. Navascués, E. Wolfe, D. Rosset, A. Pozas-Kerstjens, PRL 125, 240505
    (2020), arXiv:2002.02773.
[2] L. Lyu, P. Lauand, W. Witczak-Krempa, "Network-irreducible multiparty
    entanglement in quantum matter", PRL 137, 110204 (2026), arXiv:2512.11118.
"""

from __future__ import annotations

from collections.abc import Sequence

import cvxpy as cp
import numpy as np
from scipy import sparse

from medt.linalg import partial_transpose, permute_subsystems


def _sym_bases(d: int) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal bases of the symmetric and antisymmetric subspaces of C^d (x) C^d."""
    sym, anti = [], []
    for i in range(d):
        for j in range(i, d):
            v = np.zeros(d * d)
            if i == j:
                v[i * d + i] = 1
                sym.append(v)
                continue
            v[i * d + j] = v[j * d + i] = 1 / np.sqrt(2)
            sym.append(v)
            u = np.zeros(d * d)
            u[i * d + j], u[j * d + i] = 1 / np.sqrt(2), -1 / np.sqrt(2)
            anti.append(u)
    return np.array(sym).T, np.array(anti).T


def _trace_map(keep: Sequence[int], dims: Sequence[int]) -> sparse.csr_matrix:
    """Sparse matrix of ``vec(X) -> vec(partial_trace(X, keep, dims))`` (row-major)."""
    D = int(np.prod(dims))
    n = len(dims)
    keep = list(keep)
    traced = [k for k in range(n) if k not in keep]
    dk = int(np.prod([dims[k] for k in keep]))
    dt = D // dk
    idx = np.arange(D * D).reshape(tuple(dims) * 2)
    idx = idx.transpose(keep + traced + [n + k for k in keep] + [n + k for k in traced]).reshape(dk, dt, dk, dt)
    cols = np.einsum("atbt->tab", idx).reshape(dt, dk * dk)
    rows = np.tile(np.arange(dk * dk), dt)
    return sparse.csr_matrix((np.ones(rows.size), (rows, cols.ravel())), shape=(dk * dk, D * D))


def _apply(M: sparse.csr_matrix, X: cp.Expression, d_out: int) -> cp.Expression:
    return cp.reshape(M @ cp.vec(X, order="C"), (d_out, d_out), order="C")


def _cp_partial_transpose(X: cp.Expression, sys: Sequence[int], dims: Sequence[int]) -> cp.Expression:
    D = int(np.prod(dims))
    idx = partial_transpose(np.arange(D * D).reshape(D, D), sys, dims).ravel()
    return cp.reshape(cp.vec(X, order="C")[idx], (D, D), order="C")


def _cp_permute(X: cp.Expression, order: Sequence[int], dims: Sequence[int]) -> cp.Expression:
    D = int(np.prod(dims))
    idx = permute_subsystems(np.arange(D * D).reshape(D, D), order, dims).ravel()
    return cp.reshape(cp.vec(X, order="C")[idx], (D, D), order="C")


def ring_inflation_robustness(
    rho: np.ndarray,
    dims: Sequence[int],
    *,
    level: int = 2,
    solver: str | None = None,
    **solver_kwargs,
) -> float:
    """White-noise robustness of ``rho`` against ring inflation.

    Parameters
    ----------
    rho : (D, D) array
        State of three parties ``A B C``.
    dims : (dA, dB, dC)
        Party dimensions. Level 2 lives on ``D^2`` and level 3 on ``D^3``, so
        level 3 is practical for three qubits only.
    level : {2, 3}
        Inflation level: two or three copies of each source.

    Returns
    -------
    float
        ``t_max``. If ``t_max < 1``, ``rho`` has genuine network multipartite
        entanglement; the noise threshold certified by this test is ``1 - t_max``
        for pure ``rho``.
    """
    rho = np.asarray(rho, dtype=complex)
    dims3 = [int(d) for d in dims]
    if len(dims3) != 3:
        raise ValueError("ring inflation is implemented for three parties")
    if level == 3:
        return _level3_robustness(rho, dims3, solver, solver_kwargs)
    if level != 2:
        raise ValueError("level must be 2 or 3")
    D3 = int(np.prod(dims3))
    dims6 = dims3 + dims3
    Qp, Qm = _sym_bases(D3)

    def swap_invariant():
        Yp = cp.Variable((Qp.shape[1],) * 2, hermitian=True)
        Ym = cp.Variable((Qm.shape[1],) * 2, hermitian=True)
        X = Qp @ Yp @ Qp.T + Qm @ Ym @ Qm.T
        return X, [Yp >> 0, Ym >> 0, cp.real(cp.trace(Yp) + cp.trace(Ym)) == 1]

    tau, cons_tau = swap_invariant()
    gamma, cons_gam = swap_invariant()
    t = cp.Variable()
    cons = cons_tau + cons_gam

    def marg(X, keep):
        d_out = int(np.prod([dims6[k] for k in keep]))
        return _apply(_trace_map(keep, dims6), X, d_out)

    # Each copy of the original parties reproduces the noisy state.
    cons.append(marg(tau, [0, 1, 2]) == t * rho + (1 - t) * np.eye(D3) / D3)
    # Four-party marginals shared by the ring and the doubled network.
    cons.append(marg(tau, [0, 1, 3, 4]) == marg(gamma, [0, 1, 3, 4]))
    cons.append(marg(tau, [1, 2, 4, 5]) == marg(gamma, [1, 2, 4, 5]))
    cons.append(marg(tau, [2, 0, 5, 3]) == marg(gamma, [2, 3, 5, 0]))
    # PPT relaxations of separability.
    cons.append(_cp_partial_transpose(tau, [3, 4, 5], dims6) >> 0)
    for keep in ([0, 1, 2, 4], [1, 2, 3, 5], [2, 3, 4, 0]):
        sub = [dims6[k] for k in keep]
        cons.append(_cp_partial_transpose(marg(gamma, keep), [3], sub) >> 0)

    return _solve(t, cons, solver, solver_kwargs)


def _solve(t: cp.Variable, cons: list, solver: str | None, solver_kwargs: dict) -> float:
    prob = cp.Problem(cp.Maximize(t), cons)
    if solver is None:
        solver = "SCS"
        solver_kwargs.setdefault("eps", 1e-7)
    prob.solve(solver=solver, **solver_kwargs)
    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"SDP solver returned status {prob.status!r}")
    return float(t.value)


def _level3_robustness(rho: np.ndarray, dims3: list[int], solver: str | None, solver_kwargs: dict) -> float:
    """Level-3 inflation SDP of Lyu et al. [2], Supplemental Material Eq. (A5).

    Parties are ordered ``A1 B1 C1 A2 B2 C2 A3 B3 C3`` (0-based indices 0-8).
    ``sigma`` holds three disjoint triangles, ``tau`` a 6-ring and a triangle,
    and ``gamma`` the 9-ring.
    """
    D3 = int(np.prod(dims3))
    dims9 = dims3 * 3
    copies = [D3] * 3
    real = bool(np.allclose(rho.imag, 0))
    rho = rho.real if real else rho

    def state():
        X = cp.Variable((D3**3,) * 2, symmetric=True) if real else cp.Variable((D3**3,) * 2, hermitian=True)
        tr = cp.trace(X) if real else cp.real(cp.trace(X))
        return X, [X >> 0, tr == 1]

    gamma, c1 = state()
    tau, c2 = state()
    sigma, c3 = state()
    t = cp.Variable()
    cons = c1 + c2 + c3 + [t >= 0]

    # Symmetries of the inflated networks under permuting copies.
    cons.append(_cp_permute(tau, [1, 0, 2], copies) == tau)
    cons.append(_cp_permute(gamma, [1, 2, 0], copies) == gamma)
    cons.append(_cp_permute(sigma, [1, 0, 2], copies) == sigma)
    cons.append(_cp_permute(sigma, [0, 2, 1], copies) == sigma)

    def marg(X, keep):
        d_out = int(np.prod([dims9[k] for k in keep]))
        return _apply(_trace_map(keep, dims9), X, d_out)

    def ppt(X, keep, transposed):
        sub = [dims9[k] for k in keep]
        return _cp_partial_transpose(marg(X, keep), [keep.index(k) for k in transposed], sub) >> 0

    cons.append(marg(sigma, [0, 1, 2]) == t * rho + (1 - t) * np.eye(D3) / D3)
    # Seven-party marginals shared by gamma and tau.
    cons.append(marg(gamma, [7, 8, 0, 1, 2, 4, 5]) == marg(tau, [4, 5, 0, 1, 2, 7, 8]))
    cons.append(marg(gamma, [8, 0, 1, 2, 3, 5, 6]) == marg(tau, [5, 0, 1, 2, 3, 8, 6]))
    cons.append(marg(gamma, [0, 1, 2, 3, 4, 6, 7]) == marg(tau, [0, 1, 2, 3, 4, 6, 7]))
    # Seven-party marginals shared by tau and sigma.
    cons.append(marg(tau, [0, 1, 3, 4, 6, 7, 8]) == marg(sigma, [0, 1, 3, 4, 6, 7, 8]))
    cons.append(marg(tau, [1, 2, 4, 5, 6, 7, 8]) == marg(sigma, [1, 2, 4, 5, 6, 7, 8]))
    cons.append(marg(tau, [2, 3, 5, 0, 6, 7, 8]) == marg(sigma, [2, 0, 5, 3, 6, 7, 8]))
    # PPT relaxations of separability between disconnected groups of parties.
    cons.append(_cp_partial_transpose(tau, [2], copies) >> 0)
    cons.append(_cp_partial_transpose(sigma, [0], copies) >> 0)
    for keep in ([0, 2, 3, 4, 5, 6, 7], [1, 3, 4, 5, 6, 7, 8], [2, 4, 5, 6, 7, 8, 0]):
        cons.append(ppt(gamma, keep, [keep[0]]))
    for keep in ([0, 1, 2, 4, 5, 6, 7], [1, 2, 3, 5, 6, 7, 8], [2, 3, 4, 6, 7, 8, 0]):
        cons.append(ppt(gamma, keep, keep[:3]))
    for keep in ([0, 2, 3, 4, 6, 7, 8], [1, 3, 4, 5, 6, 7, 8], [2, 4, 5, 0, 6, 7, 8]):
        cons.append(ppt(tau, keep, [keep[0]]))
        cons.append(ppt(tau, keep, keep[1:4]))
    return _solve(t, cons, solver, solver_kwargs)


def gilbert_criterion(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Noise level above which ``rho`` is certified to be a network state.

    Given any network state ``sigma`` (for instance the closest one found by a
    Gilbert-type algorithm) at Hilbert-Schmidt distance ``delta`` from ``rho``,
    the state ``(1 - eps) rho + eps I/D`` is a network state for every
    ``eps >= delta / (delta + r)``, where ``r = 1/sqrt(D (D - 1))`` is the radius
    of a ball of separable states around ``I/D`` [3]. See Lyu et al. [2],
    Supplemental Material Sec. A2. The same bound holds for any convex set that
    contains ``sigma`` and that ball, such as the biseparable states.

    [3] L. Gurvits, H. Barnum, PRA 66, 062311 (2002).
    """
    rho = np.asarray(rho, dtype=complex)
    D = rho.shape[0]
    delta = float(np.linalg.norm(rho - np.asarray(sigma), "fro"))
    r = 1 / np.sqrt(D * (D - 1))
    return delta / (delta + r)
