import numpy as np
import pytest

from medt import is_ppt
from medt.states import basis_state, dicke, ghz, isotropic, ket_to_dm, w, werner, white_noise


def test_ghz_and_w():
    assert np.allclose(ghz(3)[[0, 7]], 1 / np.sqrt(2))
    assert np.allclose(ghz(2, d=3)[[0, 4, 8]], 1 / np.sqrt(3))
    assert np.flatnonzero(w(3)).tolist() == [1, 2, 4]


@pytest.mark.parametrize("n, k", [(4, 2), (6, 3), (5, 0)])
def test_dicke(n, k):
    psi = dicke(n, k)
    assert np.linalg.norm(psi) == pytest.approx(1)
    assert all(bin(i).count("1") == k for i in np.flatnonzero(psi))


def test_basis_state():
    assert np.flatnonzero(basis_state("0110")).tolist() == [6]


def test_werner_and_isotropic_ppt_thresholds():
    # Werner states (I - a F)/(d^2 - a d) are entangled iff a > 1/d; isotropic iff alpha > 1/(d+1).
    d = 3
    assert is_ppt(werner(d, 0.3), (0,), (d, d)) and not is_ppt(werner(d, 0.4), (0,), (d, d))
    assert is_ppt(isotropic(d, 0.24), (0,), (d, d)) and not is_ppt(isotropic(d, 0.26), (0,), (d, d))
    for rho in (werner(d, 0.5), isotropic(d, 0.5)):
        assert np.trace(rho) == pytest.approx(1)


def test_white_noise():
    rho = white_noise(ket_to_dm(ghz(3)), 0.25)
    assert rho[0, 0] == pytest.approx(0.75 / 2 + 0.25 / 8)
