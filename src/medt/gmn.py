"""Genuine multiparticle negativity (GMN) via semidefinite programming.

A state is biseparable if it is a mixture of states separable across some
bipartition. Jungnitsch, Moroder and Gühne [1] relax this to mixtures of states
with positive partial transpose (PPT mixtures) and detect genuine multipartite
entanglement with fully decomposable witnesses. The optimal witness value gives
the GMN. This module implements the renormalized GMN of Hofmann et al. [2],
which drops the ``P_M <= 1`` constraint and is an entanglement monotone.

For a state ``rho`` on ``n`` parties, GMN is ``-min tr(W rho)`` over Hermitian
``W`` such that for every bipartition ``M``

    W = P_M + Q_M^{T_M},   P_M >= 0,   0 <= Q_M <= 1.

GMN > 0 certifies genuine multipartite entanglement. For ``n = 2`` it equals
the negativity ``(||rho^{T_A}||_1 - 1) / 2``.

[1] B. Jungnitsch, T. Moroder, O. Gühne, PRL 106, 190502 (2011).
[2] M. Hofmann, T. Moroder, O. Gühne, J. Phys. A 47, 155301 (2014).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from medt.linalg import bipartitions, partial_transpose


@dataclass
class GMNResult:
    value: float
    witness: np.ndarray
    status: str


def _default_dims(rho: np.ndarray) -> tuple[int, ...]:
    n = int(round(np.log2(rho.shape[0])))
    if 2**n != rho.shape[0]:
        raise ValueError("dims must be given when rho is not a multi-qubit state")
    return (2,) * n


def _validate(rho, dims):
    rho = np.asarray(rho, dtype=complex)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("rho must be a square matrix")
    if not np.allclose(rho, rho.conj().T, atol=1e-10):
        raise ValueError("rho must be Hermitian")
    dims = _default_dims(rho) if dims is None else tuple(int(d) for d in dims)
    if any(d < 2 for d in dims):
        raise ValueError("subsystem dimensions must be at least 2")
    if np.prod(dims) != rho.shape[0]:
        raise ValueError(f"dims {dims} do not match rho of size {rho.shape[0]}")
    return (rho + rho.conj().T) / 2, dims


def _pt_index(sys: Sequence[int], dims: Sequence[int]) -> np.ndarray:
    """Flat index map ``idx`` with ``partial_transpose(X).ravel() == X.ravel()[idx]``."""
    D = int(np.prod(dims))
    return partial_transpose(np.arange(D * D).reshape(D, D), sys, dims).ravel()


def _cp_partial_transpose(X: cp.Expression, sys: Sequence[int], dims: Sequence[int]):
    # One gather instead of cp.partial_transpose's sum of Kronecker products,
    # which runs out of memory for six qubits.
    D = X.shape[0]
    idx = _pt_index(sys, dims)
    return cp.reshape(cp.vec(X, order="C")[idx], (D, D), order="C")


def gmn(
    rho: np.ndarray,
    dims: Sequence[int] | None = None,
    *,
    solver: str | None = None,
    return_witness: bool = False,
    **solver_kwargs,
):
    """Renormalized genuine multiparticle negativity of ``rho``.

    Parameters
    ----------
    rho : (D, D) array
        Density matrix.
    dims : sequence of int, optional
        Local dimensions of the parties. Defaults to qubits. To treat a block
        of qubits as one party, group them (e.g. six qubits as ``(4, 4, 4)``)
        after reordering with :func:`medt.permute_subsystems` if needed.
    solver : str, optional
        Any CVXPY SDP solver (``"CLARABEL"``, ``"SCS"``, ``"MOSEK"``, ...).
        By default, the interior-point solver Clarabel is used up to total
        dimension 32 and the first-order solver SCS (``eps=1e-8``) above it:
        interior-point memory grows too fast beyond five qubits.
    return_witness : bool
        If True, return a :class:`GMNResult` with the optimal witness ``W``.
    """
    rho, dims = _validate(rho, dims)
    D = rho.shape[0]
    W = cp.Variable((D, D), hermitian=True)
    eye = np.eye(D)
    constraints = []
    for M in bipartitions(len(dims)):
        P = cp.Variable((D, D), hermitian=True)
        Q = _cp_partial_transpose(W - P, M, dims)
        constraints += [P >> 0, Q >> 0, eye - Q >> 0]
    objective = cp.real(cp.trace(rho @ W))
    prob = cp.Problem(cp.Minimize(objective), constraints)
    if solver is None:
        solver = "CLARABEL" if D <= 32 else "SCS"
        if solver == "SCS":
            solver_kwargs.setdefault("eps", 1e-8)
    prob.solve(solver=solver, **solver_kwargs)
    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"SDP solver returned status {prob.status!r}")
    value = float(-prob.value)
    if not return_witness:
        return value
    return GMNResult(value=value, witness=np.asarray(W.value), status=prob.status)


def negativity(
    rho: np.ndarray, sys: Sequence[int] = (0,), dims: Sequence[int] | None = None
) -> float:
    """Negativity ``(||rho^{T_sys}||_1 - 1) / 2`` across the cut ``sys | rest``."""
    rho, dims = _validate(rho, dims)
    eig = np.linalg.eigvalsh(partial_transpose(rho, sys, dims))
    return float(-eig[eig < 0].sum())


def fermionic_gmn(
    rho: np.ndarray,
    n_modes: int | None = None,
    *,
    solver: str | None = None,
    return_witness: bool = False,
    **solver_kwargs,
):
    """Fermionic genuine multiparticle negativity of a state of ``n_modes`` modes.

    The PPT-mixture SDP of :func:`gmn` with each partial transpose replaced by
    the fermionic partial transpose of :mod:`medt.fermions`, following
    Allen and Witczak-Krempa, arXiv:2509.12109. Each party is one fermionic
    mode. For two modes it equals the fermionic negativity.
    """
    from medt.fermions import compute_fermionic_partial_transpose_matrix

    rho = np.asarray(rho, dtype=complex)
    D = rho.shape[0]
    if n_modes is None:
        n_modes = int(round(np.log2(D)))
    if rho.shape != (2**n_modes, 2**n_modes):
        raise ValueError(f"rho must be {2**n_modes} x {2**n_modes} for {n_modes} modes")
    dims = [2] * n_modes
    eye = np.eye(D)
    W = cp.Variable((D, D), complex=True)
    constraints = []
    for M in bipartitions(n_modes):
        F = compute_fermionic_partial_transpose_matrix(list(M), dims, include_particle_hole_unitary=False)
        P = cp.Variable((D, D), complex=True)
        Q = cp.Variable((D, D), complex=True)
        # The fermionic partial transpose does not preserve Hermiticity, so Q is
        # a general matrix with singular values of 2Q - 1 at most one.
        disk = 2 * Q - eye
        Q_ft = cp.reshape(F @ cp.vec(Q, order="C"), (D, D), order="C")
        constraints += [
            W == P + Q_ft,
            P + P.H >> 0,
            cp.bmat([[eye, disk], [disk.H, eye]]) >> 0,
        ]
    prob = cp.Problem(cp.Minimize(cp.real(cp.trace(rho @ W))), constraints)
    if solver is None:
        solver = "CLARABEL" if D <= 16 else "SCS"
        if solver == "SCS":
            solver_kwargs.setdefault("eps", 1e-8)
    prob.solve(solver=solver, **solver_kwargs)
    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"SDP solver returned status {prob.status!r}")
    value = float(-prob.value)
    if not return_witness:
        return value
    return GMNResult(value=value, witness=np.asarray(W.value), status=prob.status)
