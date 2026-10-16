"""Reduced density matrices of the 1D transverse-field Ising chain.

For the infinite chain ``H = -sum_i (X_i X_{i+1} + h Z_i)`` the model maps to
free fermions, so every spin correlation is a Pfaffian of Majorana two-point
functions [1, 2]. The reduced state of any set of sites, at zero or finite
temperature, follows from its Pauli expansion

    rho_S = 2^{-|S|} sum_P <P> P.

This dense version sums all ``4^|S|`` Pauli strings and suits up to about
eight sites.

[1] E. Lieb, T. Schultz, D. Mattis, Ann. Phys. 16, 407 (1961).
[2] P. Pfeuty, Ann. Phys. 57, 79 (1970).
"""

from __future__ import annotations

import functools
from collections.abc import Sequence

import numpy as np
from scipy.integrate import quad
from scipy.linalg import toeplitz

_PAULI = [
    np.eye(2, dtype=complex),
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]], dtype=complex),
    np.array([[1, 0], [0, -1]], dtype=complex),
]


def tfim_rdm(h: float, sites: Sequence[int], beta: float | None = None) -> np.ndarray:
    """Reduced state of ``sites`` (0-based, any subset) at field ``h``.

    ``beta=None`` gives the ground state, otherwise the Gibbs state at inverse
    temperature ``beta``. Sites are ordered as given after sorting.
    """
    sites = sorted(int(s) for s in sites)
    if not sites or sites[0] < 0 or len(set(sites)) != len(sites):
        raise ValueError("sites must be distinct non-negative integers")
    beta = None if beta is None else float(beta)
    # Translation invariance: shift so the first site is 1 (1-based internally).
    one_based = [s - sites[0] + 1 for s in sites]
    rho = _rdm_rec(float(h), beta, tuple(one_based), {})
    return rho / 2 ** len(sites)


def _pfaffian(A: np.ndarray) -> complex:
    """Pfaffian of a skew-symmetric matrix by Parlett-Reid style elimination."""
    A = np.array(A, dtype=complex)
    n = A.shape[0]
    if n % 2:
        return 0.0
    pf = 1.0 + 0j
    for i in range(0, n - 2, 2):
        p = i + 1 + int(np.argmax(np.abs(A[i, i + 1 :])))
        if abs(A[i, p]) < 1e-15:
            return 0.0
        if p != i + 1:
            A[[i + 1, p], :] = A[[p, i + 1], :]
            A[:, [i + 1, p]] = A[:, [p, i + 1]]
            pf = -pf
        val = A[i, i + 1]
        pf *= val
        for j in range(i + 2, n):
            f = A[i, j] / val
            A[j, :] -= f * A[i + 1, :]
            A[:, j] -= f * A[:, i + 1]
    return pf * A[n - 2, n - 1]


@functools.cache
def _g(l: int, h: float, beta: float | None) -> complex:
    def integrand(k):
        eps = np.sqrt(1 + h**2 + 2 * h * np.cos(k))
        val = (h * np.cos(l * k) + np.cos((l - 1) * k)) / eps
        return val if beta is None else val * np.tanh(beta * eps)

    val, _ = quad(integrand, 0, np.pi)
    return val * ((-1.0) ** (l + 1) * 1j / np.pi)


@functools.cache
def _gamma(h: float, l: int, beta: float | None) -> np.ndarray:
    m = np.tril(toeplitz(np.arange(l)))
    t = m - m.T
    res = np.zeros((2 * l, 2 * l), dtype=complex)
    for i in range(l):
        for j in range(l):
            res[2 * i : 2 * i + 2, 2 * j : 2 * j + 2] = [[0, _g(t[i, j], h, beta)], [-_g(-t[i, j], h, beta), 0]]
    return -res


def _majorana_corr(ops, h, beta):
    if not ops:
        return 1.0
    gam = _gamma(h, int(np.ceil(max(ops) / 2)), beta)
    idx = [i - 1 for i in ops]
    return _pfaffian(gam[np.ix_(idx, idx)])


_F_MAP = {
    0: {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1)},
    1: {0: (1, 1), 1: (0, 1), 2: (1, 0), 3: (0, 0)},
}
_PAULI_ALGEBRA = {
    (0, 0, 0): 1, (0, 0, 1): 1, (0, 1, 0): 1, (0, 1, 1): -1j,
    (1, 0, 0): 1, (1, 0, 1): 1j, (1, 1, 0): -1j, (1, 1, 1): -1j,
}


def _jordan_wigner(pauli_string):
    """Majorana occupation vector of a Pauli string (Jordan-Wigner)."""
    parity, rows = 0, []
    for p in reversed(pauli_string):
        a = _F_MAP[parity][p]
        parity = (parity + sum(a)) % 2
        rows.append(a)
    return np.array(rows[::-1], dtype=int).ravel()


def _phase(mu):
    xy = mu.reshape(-1, 2)
    after = xy[1:].sum(axis=1)
    z = np.append(np.flip(np.cumsum(np.flip(after))), 0) % 2
    coef = 1.0 + 0j
    for (a, b), c in zip(xy, z, strict=True):
        coef *= _PAULI_ALGEBRA[(a, b, int(c))]
    return coef


def _expectation(pauli_string, h, beta):
    mu = _jordan_wigner(pauli_string)
    ops = [i + 1 for i, v in enumerate(mu) if v]
    return float(np.real(_phase(mu) * _majorana_corr(ops, h, beta)))


def _rdm_rec(h, beta, sites, fixed):
    free = [s for s in sites if s not in fixed]
    string = [0] * max(sites)
    for s, v in fixed.items():
        string[s - 1] = v
    if len(free) == 1:
        rho = np.zeros((2, 2), dtype=complex)
        for k in range(4):
            string[free[0] - 1] = k
            rho += _expectation(string, h, beta) * _PAULI[k]
        return rho
    last = free[-1]
    rho = 0
    for k in range(4):
        rho = rho + np.kron(_rdm_rec(h, beta, sites, {**fixed, last: k}), _PAULI[k])
    return rho
