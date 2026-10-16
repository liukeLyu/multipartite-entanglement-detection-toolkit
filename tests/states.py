import numpy as np


def ket_to_rho(psi):
    psi = np.asarray(psi, dtype=complex)
    psi = psi / np.linalg.norm(psi)
    return np.outer(psi, psi.conj())


def ghz(n):
    psi = np.zeros(2**n)
    psi[0] = psi[-1] = 1
    return ket_to_rho(psi)


def w_state(n):
    psi = np.zeros(2**n)
    for k in range(n):
        psi[1 << k] = 1
    return ket_to_rho(psi)


def horodecki_3x3(a):
    """Horodecki's 3x3 PPT entangled state, entangled for 0 < a < 1."""
    rho = np.zeros((9, 9))
    for i in (0, 4, 8):
        for j in (0, 4, 8):
            rho[i, j] = a
    for i in (1, 2, 3, 5, 7):
        rho[i, i] = a
    rho[6, 6] = rho[8, 8] = (1 + a) / 2
    rho[6, 8] = rho[8, 6] = np.sqrt(1 - a**2) / 2
    return rho / (8 * a + 1)
