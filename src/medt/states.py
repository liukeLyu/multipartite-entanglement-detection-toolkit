"""Canonical states for benchmarks and tests.

Kets are 1D arrays and density matrices 2D arrays, in the computational basis
with subsystem 0 most significant.
"""

from __future__ import annotations

import itertools
import math

import numpy as np


def ket_to_dm(psi: np.ndarray) -> np.ndarray:
    psi = np.asarray(psi, dtype=complex).ravel()
    psi = psi / np.linalg.norm(psi)
    return np.outer(psi, psi.conj())


def white_noise(rho: np.ndarray, p: float) -> np.ndarray:
    """``(1 - p) rho + p I/D``."""
    D = rho.shape[0]
    return (1 - p) * np.asarray(rho) + p * np.eye(D) / D


def basis_state(digits: str, d: int = 2) -> np.ndarray:
    """Computational basis ket, e.g. ``basis_state("0110")``."""
    psi = np.zeros(d ** len(digits), dtype=complex)
    psi[int(digits, d)] = 1
    return psi


def ghz(n: int, d: int = 2) -> np.ndarray:
    """``sum_k |k...k> / sqrt(d)`` on ``n`` parties of dimension ``d``."""
    psi = np.zeros(d**n, dtype=complex)
    psi[np.arange(d) * ((d**n - 1) // (d - 1))] = 1 / np.sqrt(d)
    return psi


def dicke(n: int, k: int) -> np.ndarray:
    """Symmetric ``n``-qubit state with ``k`` excitations."""
    if not 0 <= k <= n:
        raise ValueError("k must be between 0 and n")
    psi = np.zeros(2**n, dtype=complex)
    for ones in itertools.combinations(range(n), k):
        psi[sum(1 << (n - 1 - q) for q in ones)] = 1
    return psi / np.sqrt(math.comb(n, k))


def w(n: int) -> np.ndarray:
    """``n``-qubit W state, the Dicke state with one excitation."""
    return dicke(n, 1)


def werner(d: int, alpha: float) -> np.ndarray:
    """Werner state ``(I - alpha F) / (d^2 - alpha d)`` with ``F`` the swap."""
    F = np.eye(d * d).reshape(d, d, d, d).transpose(0, 1, 3, 2).reshape(d * d, d * d)
    return (np.eye(d * d) - alpha * F) / (d * d - alpha * d)


def isotropic(d: int, alpha: float) -> np.ndarray:
    """``alpha |Phi+><Phi+| + (1 - alpha) I/d^2`` with ``|Phi+>`` maximally entangled."""
    phi = np.eye(d).ravel() / np.sqrt(d)
    return alpha * np.outer(phi, phi) + (1 - alpha) * np.eye(d * d) / (d * d)
