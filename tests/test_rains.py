import numpy as np
import pytest

from medt import gmre
from medt.states import ghz, ket_to_dm, w

pytest.importorskip("qics")
pytest.importorskip("picos")


def test_two_qubits_pure_state_is_entanglement_entropy():
    # For two parties GMRE is the Rains relative entropy, which equals the
    # entanglement entropy on pure states.
    p = 0.8
    psi = np.array([np.sqrt(p), 0, 0, np.sqrt(1 - p)])
    expected = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    assert gmre(ket_to_dm(psi)) == pytest.approx(expected, abs=1e-5)


def test_ghz_and_w():
    assert gmre(ket_to_dm(ghz(3))) == pytest.approx(1, abs=1e-5)
    assert gmre(ket_to_dm(w(3))) == pytest.approx(np.log2(1.5), abs=1e-5)


def test_biseparable_state_is_zero():
    rho = np.kron(ket_to_dm([1, 0, 0, 1]), np.eye(2) / 2)
    assert gmre(rho) == pytest.approx(0, abs=1e-5)


# GMRE in bits of transverse-field Ising chain reduced states, computed by Liuke Lyu's
# MATLAB code (gilbert-GMRE/data/TFIM_MRE_results.mat and TFIM_MRE_4spin_results.mat).
@pytest.mark.parametrize(
    "h, sites, dims, ref",
    [
        (1.0, [0, 1, 2], [2, 2, 2], 0.028250680508351153),
        (1.2, [0, 1, 2], [2, 2, 2], 0.06291866336849225),
        (2.0, [0, 1, 2], [2, 2, 2], 0.036875095276514155),
        (1.3, [0, 1, 2, 3], [2, 4, 2], 0.0676558640111351),
    ],
)
def test_tfim_matches_reference(h, sites, dims, ref):
    from medt.tfim import tfim_rdm

    assert gmre(tfim_rdm(h, sites), dims) == pytest.approx(ref, abs=2e-5)
