"""Tensor-product helpers for density matrices.

Subsystems are ordered as in ``np.kron(A_1, A_2, ..., A_n)``: subsystem 0 is the
most significant index.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _check_dims(rho: np.ndarray, dims: Sequence[int]) -> tuple[int, ...]:
    dims = tuple(int(d) for d in dims)
    if rho.shape != (np.prod(dims), np.prod(dims)):
        raise ValueError(f"rho of shape {rho.shape} does not match dims {dims}")
    return dims


def partial_transpose(rho: np.ndarray, sys: Sequence[int], dims: Sequence[int]) -> np.ndarray:
    """Transpose the subsystems listed in ``sys`` (0-based)."""
    rho = np.asarray(rho)
    dims = _check_dims(rho, dims)
    n = len(dims)
    t = rho.reshape(dims + dims)
    perm = list(range(2 * n))
    for k in sys:
        perm[k], perm[n + k] = n + k, k
    return t.transpose(perm).reshape(rho.shape)


def permute_subsystems(rho: np.ndarray, order: Sequence[int], dims: Sequence[int]) -> np.ndarray:
    """Reorder subsystems so that new subsystem ``i`` is old subsystem ``order[i]``."""
    rho = np.asarray(rho)
    dims = _check_dims(rho, dims)
    n = len(dims)
    order = list(order)
    if sorted(order) != list(range(n)):
        raise ValueError(f"order {order} is not a permutation of range({n})")
    t = rho.reshape(dims + dims).transpose(order + [n + k for k in order])
    return t.reshape(rho.shape)


def bipartitions(n: int) -> list[tuple[int, ...]]:
    """All 2**(n-1) - 1 inequivalent bipartitions M|M^c, as the subsystem set M.

    The last subsystem is never in M, which picks one representative per pair.
    """
    return [tuple(k for k in range(n) if m >> k & 1) for m in range(1, 2 ** (n - 1))]


def partial_trace(rho: np.ndarray, keep: Sequence[int], dims: Sequence[int]) -> np.ndarray:
    """Reduced state on the subsystems ``keep``, in the order given."""
    rho = np.asarray(rho)
    dims = _check_dims(rho, dims)
    n = len(dims)
    keep = list(keep)
    traced = [k for k in range(n) if k not in keep]
    t = rho.reshape(dims + dims)
    # Move kept row indices, then traced row indices, then the same for columns.
    t = t.transpose(keep + traced + [n + k for k in keep] + [n + k for k in traced])
    dk = int(np.prod([dims[k] for k in keep]))
    dt = int(np.prod([dims[k] for k in traced]))
    return np.einsum("ajbj->ab", t.reshape(dk, dt, dk, dt))


def realign(rho: np.ndarray, dims: Sequence[int]) -> np.ndarray:
    """Realigned matrix ``R[(i,j),(k,l)] = rho[(i,k),(j,l)]`` of a bipartite operator."""
    rho = np.asarray(rho)
    dA, dB = _check_dims(rho, dims)
    return rho.reshape(dA, dB, dA, dB).transpose(0, 2, 1, 3).reshape(dA * dA, dB * dB)
