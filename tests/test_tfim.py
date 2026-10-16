import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from medt.tfim import tfim_rdm


def ring_ground_state_rdm(L, h, keep):
    """Reduced state of the first ``keep`` sites of H = -sum X X - h sum Z on a ring."""
    eye, X, Z = sp.identity(2), sp.csr_matrix([[0, 1], [1, 0]]), sp.csr_matrix([[1.0, 0], [0, -1]])

    def op(o, i):
        out = sp.identity(1)
        for j in range(L):
            out = sp.kron(out, o if j == i else eye, format="csr")
        return out

    H = sum(-op(X, i) @ op(X, (i + 1) % L) - h * op(Z, i) for i in range(L))
    psi = sla.eigsh(H, k=1, which="SA")[1][:, 0].reshape(2**keep, -1)
    return psi @ psi.conj().T


@pytest.mark.parametrize("h", [0.5, 2.0])
def test_matches_exact_diagonalization(h):
    # A ring of 16 sites is close to the infinite chain away from h = 1.
    assert np.abs(tfim_rdm(h, [0, 1, 2]) - ring_ground_state_rdm(16, h, 3)).max() < 1e-4


@pytest.mark.parametrize("beta", [None, 2.0])
def test_is_a_state(beta):
    rho = tfim_rdm(1.0, [0, 2, 3, 5], beta=beta)
    assert np.trace(rho) == pytest.approx(1)
    assert np.allclose(rho, rho.conj().T)
    assert np.linalg.eigvalsh(rho).min() > -1e-10


def test_translation_invariance_and_high_temperature():
    assert np.allclose(tfim_rdm(0.7, [3, 4]), tfim_rdm(0.7, [0, 1]))
    assert np.allclose(tfim_rdm(1.0, [0, 1], beta=1e-4), np.eye(4) / 4, atol=1e-3)


def test_invalid_sites():
    with pytest.raises(ValueError):
        tfim_rdm(1.0, [0, 0])
