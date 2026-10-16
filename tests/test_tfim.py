import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from medt.tfim import tfim_ground_state, tfim_rdm


def ring_ground_state_rdm(L, h, keep):
    """Reduced state of the first ``keep`` sites of the ring ground state."""
    psi = tfim_ground_state(L, h)[1].reshape(2**keep, -1)
    return psi @ psi.conj().T


def dense_ring_hamiltonian(L, h):
    eye, X, Z = sp.identity(2), sp.csr_matrix([[0, 1], [1, 0]]), sp.csr_matrix([[1.0, 0], [0, -1]])

    def op(o, i):
        out = sp.identity(1)
        for j in range(L):
            out = sp.kron(out, o if j == i else eye, format="csr")
        return out

    return sum(-op(X, i) @ op(X, (i + 1) % L) - h * op(Z, i) for i in range(L))


@pytest.mark.parametrize("h", [0.3, 1.0, 2.5])
def test_lanczos_matches_sparse_matrix_and_free_fermions(h):
    L = 10
    energy, psi = tfim_ground_state(L, h)
    H = dense_ring_hamiltonian(L, h)
    assert energy == pytest.approx(sla.eigsh(H, k=1, which="SA")[0][0], abs=1e-8)
    assert np.linalg.norm(H @ psi - energy * psi) < 1e-6
    # Parity-even sector: fermion momenta k = pi (2m + 1) / L, single-particle energies 2 sqrt(1 + h^2 - 2 h cos k).
    k = np.pi * (2 * np.arange(L) + 1) / L
    assert energy == pytest.approx(-np.sum(np.sqrt(1 + h**2 - 2 * h * np.cos(k))), abs=1e-8)


def test_open_chain():
    energy, _ = tfim_ground_state(8, 0.7, periodic=False)
    H = dense_ring_hamiltonian(8, 0.7)
    X = sp.csr_matrix([[0, 1], [1, 0]])
    wrap = sp.kron(sp.kron(X, sp.identity(2**6)), X)  # X_0 X_7 bond
    assert energy == pytest.approx(sla.eigsh(H + wrap, k=1, which="SA")[0][0], abs=1e-8)


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


@pytest.mark.slow
def test_six_spin_gmn_matches_published_value():
    # Three adjacent pairs at h = 0.99, Lyu et al., PRL 137, 110204 (2026), Fig. 3 data.
    from medt import gmn

    assert gmn(tfim_rdm(0.99, range(6)), (4, 4, 4)) == pytest.approx(0.1315414, abs=1e-5)
