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

[1] M. Navascués, E. Wolfe, D. Rosset, A. Pozas-Kerstjens, PRL 125, 240505
    (2020), arXiv:2002.02773.
[2] L. Lyu et al., PRL 137, 110204 (2026), arXiv:2512.11118.
"""

from __future__ import annotations

from collections.abc import Sequence

import cvxpy as cp
import numpy as np
from scipy import sparse

from medt.linalg import partial_transpose


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


def ring_inflation_robustness(
    rho: np.ndarray,
    dims: Sequence[int],
    *,
    solver: str | None = None,
    **solver_kwargs,
) -> float:
    """White-noise robustness of ``rho`` against the second-order ring inflation.

    Parameters
    ----------
    rho : (D, D) array
        State of three parties ``A B C``.
    dims : (dA, dB, dC)
        Party dimensions. The SDP lives on ``D^2``, so this is practical for
        three qubits and small qudits.

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

    prob = cp.Problem(cp.Maximize(t), cons)
    if solver is None:
        solver = "SCS"
        solver_kwargs.setdefault("eps", 1e-7)
    prob.solve(solver=solver, **solver_kwargs)
    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"SDP solver returned status {prob.status!r}")
    return float(t.value)


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
