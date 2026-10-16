"""Tensor-product helpers for states and density matrices.

Subsystems are ordered as in ``np.kron(A_1, A_2, ..., A_n)``: subsystem 0 is the
most significant index, and subsystem labels are 0-based. :func:`partial_trace`
takes the subsystems to trace out; :func:`keep_systems` takes the subsystems
to keep, in the order wanted for the output.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import sparse


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


def _split_index(idx: np.ndarray, dims: Sequence[int], keep: Sequence[int], traced: Sequence[int]):
    """Split flat basis indices into flat indices of the kept and traced subsystems."""
    digits = np.unravel_index(idx, dims)
    a = np.ravel_multi_index([digits[k] for k in keep], [dims[k] for k in keep]) if keep else np.zeros_like(idx)
    b = np.ravel_multi_index([digits[k] for k in traced], [dims[k] for k in traced]) if traced else np.zeros_like(idx)
    return a, b


def keep_systems(state, keep: Sequence[int], dims: Sequence[int]) -> np.ndarray:
    """Reduced density matrix on the subsystems ``keep``, ordered as listed in ``keep``.

    ``state`` can be a ket (1D array or column vector) or a density matrix, and
    either can be a SciPy sparse array. A ket is never expanded into a density
    matrix: it is reshaped into the ``d_keep x d_rest`` matrix ``M`` and the
    result is ``M M^dagger``. Memory therefore scales with the length of the
    ket, so a dense 24-qubit state (2^24 amplitudes) is fine, and a sparse ket
    only touches its nonzero amplitudes. The output is a dense
    ``d_keep x d_keep`` array.
    """
    dims = tuple(int(d) for d in dims)
    n = len(dims)
    keep = [int(k) for k in keep]
    if len(set(keep)) != len(keep) or any(not 0 <= k < n for k in keep):
        raise ValueError(f"keep={keep} is not a list of distinct subsystems of {n}")
    traced = [k for k in range(n) if k not in keep]
    D = int(np.prod(dims))
    dk = int(np.prod([dims[k] for k in keep]))
    dt = D // dk

    if sparse.issparse(state):
        X = sparse.coo_array(state)
        if 1 in X.shape or X.ndim == 1:  # sparse ket
            if max(X.shape) != D:
                raise ValueError(f"ket of length {max(X.shape)} does not match dims {dims}")
            idx = X.coords[0] if X.ndim == 1 else (X.row if X.shape[1] == 1 else X.col)
            a, b = _split_index(idx, dims, keep, traced)
            M = sparse.csr_array((X.data, (a, b)), shape=(dk, dt))
            return (M @ M.conj().T).toarray()
        if X.shape != (D, D):
            raise ValueError(f"rho of shape {X.shape} does not match dims {dims}")
        ar, br = _split_index(X.row, dims, keep, traced)
        ac, bc = _split_index(X.col, dims, keep, traced)
        on = br == bc
        out = np.zeros((dk, dk), dtype=np.result_type(X.dtype, float))
        np.add.at(out, (ar[on], ac[on]), X.data[on])
        return out

    state = np.asarray(state)
    if state.ndim == 1 or (state.ndim == 2 and state.shape[1] == 1 and D > 1):
        psi = state.reshape(-1)
        if psi.size != D:
            raise ValueError(f"ket of length {psi.size} does not match dims {dims}")
        M = psi.reshape(dims).transpose(keep + traced).reshape(dk, dt)
        return M @ M.conj().T
    rho = state
    _check_dims(rho, dims)
    t = rho.reshape(dims + dims)
    # Move kept row indices, then traced row indices, then the same for columns.
    t = t.transpose(keep + traced + [n + k for k in keep] + [n + k for k in traced])
    return np.einsum("ajbj->ab", t.reshape(dk, dt, dk, dt))


def partial_trace(state, trace_out: Sequence[int], dims: Sequence[int]) -> np.ndarray:
    """Trace out the subsystems ``trace_out``; the rest keep their original order.

    Accepts the same inputs as :func:`keep_systems`.
    """
    trace_out = {int(k) for k in trace_out}
    if any(not 0 <= k < len(dims) for k in trace_out):
        raise ValueError(f"trace_out={sorted(trace_out)} is out of range for {len(dims)} subsystems")
    return keep_systems(state, [k for k in range(len(dims)) if k not in trace_out], dims)


def realign(rho: np.ndarray, dims: Sequence[int]) -> np.ndarray:
    """Realigned matrix ``R[(i,j),(k,l)] = rho[(i,k),(j,l)]`` of a bipartite operator."""
    rho = np.asarray(rho)
    dA, dB = _check_dims(rho, dims)
    return rho.reshape(dA, dB, dA, dB).transpose(0, 2, 1, 3).reshape(dA * dA, dB * dB)
