"""Adaptive polytope certification of separability.

Ohst, Yu, Gühne and Sarbicki [1] approximate the separable set from inside by
mixtures ``sum_i sigma_i (x) tau_i`` in which the states ``sigma_i`` of some
parties are fixed and the operators ``tau_i >= 0`` of the others are free.
For fixed ``sigma_i`` the white-noise robustness

    t* = max t  such that  t rho + (1 - t) I/D = sum_i sigma_i (x) tau_i

is an SDP. The optimal ``tau_i`` become the fixed states of the next round,
and the roles alternate. Every feasible point is a separable decomposition,
so ``t* >= 1`` certifies that ``rho`` is separable, and in general
``t* rho + (1 - t*) I/D`` is certified separable.

Pairs of subsystems of dimensions 2x2 or 2x3 can be freed together under a
PPT constraint, since PPT equals separability there [1].

[1] T. Ohst, X.-D. Yu, O. Gühne, G. Sarbicki, "Certifying quantum separability
    with adaptive polytopes", SciPost Phys. 16, 063 (2024), arXiv:2210.10054.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np

from medt.linalg import bipartitions, partial_transpose, permute_subsystems


@dataclass
class PolytopeResult:
    robustness: float
    """Largest ``t`` found with ``t rho + (1 - t) I/D`` certified in the set.

    The iteration stops as soon as ``t >= 1``, so for certified states this is
    a lower bound on the true robustness, not its value.
    """
    history: list[float] = field(default_factory=list)

    @property
    def certified(self) -> bool:
        return self.robustness >= 1 - 1e-7


def _random_pure(d: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=d) + 1j * rng.normal(size=d)
    v /= np.linalg.norm(v)
    return np.outer(v, v.conj())


def _normalize(tau: np.ndarray | None, d: int, rng: np.random.Generator) -> np.ndarray:
    if tau is None or np.real(np.trace(tau)) < 1e-6:
        return _random_pure(d, rng)
    tau = (tau + tau.conj().T) / 2
    w, v = np.linalg.eigh(tau)
    tau = (v * np.clip(w, 0, None)) @ v.conj().T
    return tau / np.real(np.trace(tau))


def _kron_gather(perm: Sequence[int], dims: Sequence[int], n_fixed: int) -> np.ndarray:
    """Index map from ``K = S @ T`` to ``sum_i sigma_i (x) tau_i`` in the original order.

    ``S[:, i] = vec(sigma_i)`` lives on subsystems ``perm[:n_fixed]`` and
    ``T[i, :] = vec(tau_i)`` on ``perm[n_fixed:]``.
    """
    pdims = [dims[p] for p in perm]
    dA = int(np.prod(pdims[:n_fixed]))
    dB = int(np.prod(pdims[n_fixed:]))
    idx = np.arange(dA * dA * dB * dB).reshape(dA, dA, dB, dB).transpose(0, 2, 1, 3)
    idx = idx.reshape(dA * dB, dA * dB)
    return permute_subsystems(idx, list(np.argsort(perm)), pdims).ravel()


class _Block:
    """One family of vertices ``sigma_i (x) tau_i`` for a split of the subsystems."""

    def __init__(self, fixed, free, dims, n_points, ppt_free, rng):
        self.fixed, self.free, self.dims = list(fixed), list(free), dims
        self.dF = int(np.prod([dims[k] for k in self.fixed]))
        self.dT = int(np.prod([dims[k] for k in self.free]))
        self.ppt_free = ppt_free
        self.sigmas = [_random_pure(self.dF, rng) for _ in range(n_points)]
        self.idx = _kron_gather(self.fixed + self.free, dims, len(self.fixed))

    def build(self):
        D = self.dF * self.dT
        self.taus = [cp.Variable((self.dT, self.dT), hermitian=True) for _ in self.sigmas]
        cons = [tau >> 0 for tau in self.taus]
        if self.ppt_free:
            sub = [self.dims[k] for k in self.free]
            cons += [_cp_pt(tau, sub) >> 0 for tau in self.taus]
        S = np.stack([s.ravel() for s in self.sigmas], axis=1)
        T = cp.vstack([cp.reshape(tau, (1, self.dT**2), order="C") for tau in self.taus])
        total = cp.reshape(cp.vec(S @ T, order="C")[self.idx], (D, D), order="C")
        return total, cons


def _cp_pt(X, sub):
    D = int(np.prod(sub))
    idx = partial_transpose(np.arange(D * D).reshape(D, D), [0], sub).ravel()
    return cp.reshape(cp.vec(X, order="C")[idx], (D, D), order="C")


def _robustness(rho, blocks_per_constraint, solver, solver_kwargs):
    """Solve for t with one decomposition per entry of ``blocks_per_constraint``.

    Each entry is a list of blocks whose sum must equal the noisy state; several
    entries impose several decompositions of the same state at once.
    """
    D = rho.shape[0]
    t = cp.Variable()
    target = t * rho + (1 - t) * np.eye(D) / D
    cons = []
    for blocks in blocks_per_constraint:
        total = 0
        for b in blocks:
            expr, c = b.build()
            total, cons = total + expr, cons + c
        cons.append(total == target)
    prob = cp.Problem(cp.Maximize(t), cons)
    prob.solve(solver=solver, **solver_kwargs)
    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"SDP solver returned status {prob.status!r}")
    return float(t.value)


def _swap(block: _Block, rng) -> _Block:
    """Fix the optimized free states and free the previously fixed subsystems."""
    new = object.__new__(_Block)
    new.fixed, new.free, new.dims = block.free, block.fixed, block.dims
    new.dF, new.dT, new.ppt_free = block.dT, block.dF, False
    new.sigmas = [_normalize(tau.value, block.dT, rng) for tau in block.taus]
    new.idx = _kron_gather(new.fixed + new.free, block.dims, len(new.fixed))
    return new


def _adapt(rho, make_blocks, rotate, max_iter, tol, solver, solver_kwargs):
    rho = np.asarray(rho, dtype=complex)
    blocks = make_blocks()
    history = []
    for _ in range(max_iter):
        history.append(_robustness(rho, blocks, solver, solver_kwargs))
        if history[-1] >= 1 - 1e-7:
            break
        if len(history) > 2 and abs(history[-1] - history[-3]) < tol:
            break
        blocks = rotate(blocks)
    return PolytopeResult(max(history), history)


def polytope_separable(
    rho: np.ndarray,
    dims: Sequence[int],
    *,
    n_points: int = 100,
    max_iter: int = 30,
    tol: float = 1e-4,
    seed: int | None = None,
    solver: str = "SCS",
    **solver_kwargs,
) -> PolytopeResult:
    """Robustness certificate for separability across the cut ``dims = (dA, dB)``."""
    if len(dims) != 2:
        raise ValueError("polytope_separable takes dims = (dA, dB); use polytope_fully_separable")
    return polytope_fully_separable(
        rho, dims, n_points=n_points, max_iter=max_iter, tol=tol, seed=seed, solver=solver, **solver_kwargs
    )


def polytope_fully_separable(
    rho: np.ndarray,
    dims: Sequence[int],
    *,
    n_points: int = 100,
    max_iter: int = 30,
    tol: float = 1e-4,
    seed: int | None = None,
    solver: str = "SCS",
    **solver_kwargs,
) -> PolytopeResult:
    """Robustness certificate for full separability of ``len(dims)`` parties.

    Each round frees one party (or a 2x2 / 2x3 pair under a PPT constraint)
    and fixes the states of all others.
    """
    rng = np.random.default_rng(seed)
    dims = list(dims)
    groups = _ppt_groups(dims) if len(dims) > 2 else [[0], [1]]
    n = len(groups)
    state = {"k": 0, "vertices": None}

    def block_for(k):
        free = groups[k]
        fixed = [s for j, g in enumerate(groups) if j != k for s in g]
        b = _Block(fixed, free, dims, n_points, ppt_free=len(free) > 1, rng=rng)
        if state["vertices"] is not None:
            b.sigmas = [
                _kron_product([state["vertices"][j][i] for j in range(n) if j != k]) for i in range(n_points)
            ]
        return b

    def make_blocks():
        state["vertices"] = [
            [_group_state(len(g), dims, g, rng) for _ in range(n_points)] for g in groups
        ]
        return [[block_for(0)]]

    def rotate(blocks):
        k = state["k"]
        b = blocks[0][0]
        state["vertices"][k] = [_normalize(tau.value, b.dT, rng) for tau in b.taus]
        state["k"] = (k + 1) % n
        return [[block_for(state["k"])]]

    return _adapt(rho, make_blocks, rotate, max_iter, tol, solver, solver_kwargs)


def _group_state(size, dims, group, rng):
    return _kron_product([_random_pure(dims[s], rng) for s in group])


def _kron_product(mats):
    out = np.ones((1, 1), dtype=complex)
    for m in mats:
        out = np.kron(out, m)
    return out


def _ppt_groups(dims):
    """Pair up 2x3 and 2x2 subsystems, where PPT equals separability."""
    left = list(range(len(dims)))
    groups = []
    for want in ((2, 3), (2, 2)):
        i = 0
        while i < len(left):
            a = left[i]
            match = next(
                (b for b in left[i + 1 :] if sorted((dims[a], dims[b])) == list(want)), None
            )
            if match is not None:
                groups.append([a, match])
                left.remove(a)
                left.remove(match)
            else:
                i += 1
    return groups + [[a] for a in left]


def polytope_biseparable(
    rho: np.ndarray,
    dims: Sequence[int],
    *,
    n_points: int = 50,
    max_iter: int = 30,
    tol: float = 1e-4,
    seed: int | None = None,
    solver: str = "SCS",
    **solver_kwargs,
) -> PolytopeResult:
    """Robustness certificate for biseparability (no genuine multipartite entanglement).

    The noisy state is decomposed as a sum over bipartitions ``M | M^c`` of
    mixtures of states separable across that cut.
    """
    rng = np.random.default_rng(seed)
    dims = list(dims)
    n = len(dims)
    cuts = [(list(M), [k for k in range(n) if k not in M]) for M in bipartitions(n)]

    def make_blocks():
        return [[_Block(A, B, dims, n_points, ppt_free=False, rng=rng) for A, B in cuts]]

    def rotate(blocks):
        return [[_swap(b, rng) for b in blocks[0]]]

    return _adapt(rho, make_blocks, rotate, max_iter, tol, solver, solver_kwargs)


def polytope_separable_all_cuts(
    rho: np.ndarray,
    dims: Sequence[int],
    *,
    n_points: int = 50,
    max_iter: int = 30,
    tol: float = 1e-4,
    seed: int | None = None,
    solver: str = "SCS",
    **solver_kwargs,
) -> PolytopeResult:
    """Robustness certificate for separability across every bipartition at once."""
    rng = np.random.default_rng(seed)
    dims = list(dims)
    n = len(dims)
    cuts = [(list(M), [k for k in range(n) if k not in M]) for M in bipartitions(n)]

    def make_blocks():
        return [[_Block(A, B, dims, n_points, ppt_free=False, rng=rng)] for A, B in cuts]

    def rotate(blocks):
        return [[_swap(bl[0], rng)] for bl in blocks]

    return _adapt(rho, make_blocks, rotate, max_iter, tol, solver, solver_kwargs)
