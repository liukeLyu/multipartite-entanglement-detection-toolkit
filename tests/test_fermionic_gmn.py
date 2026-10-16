# Adapted from Liuke Lyu's original tests of the fermionic GMN.
import numpy as np
import pytest

from entdetect import fermionic_gmn
from entdetect.fermions import fermionic_negativity

pytestmark = pytest.mark.filterwarnings("ignore:Solution may be inaccurate:UserWarning")


def _ketbra(psi):
    return np.outer(psi, psi.conj())


def _white_noise_mix(psi, p):
    dim = psi.size
    return p * np.eye(dim, dtype=complex) / dim + (1 - p) * _ketbra(psi)


def _ghz_x_state_3():
    psi = np.zeros(8, dtype=complex)
    psi[[4, 2, 1, 7]] = 1 / 2
    return psi


def _single_occupancy_w_state(num_modes):
    psi = np.zeros(2**num_modes, dtype=complex)
    for mode in range(num_modes):
        psi[1 << (num_modes - 1 - mode)] = 1 / np.sqrt(num_modes)
    return psi


def test_reduces_to_bipartite_fermionic_negativity():
    psi = np.array([0, 1 / np.sqrt(2), -1 / np.sqrt(2), 0], dtype=complex)
    rho = _ketbra(psi)
    assert fermionic_gmn(rho) == pytest.approx(fermionic_negativity(rho, [0], [2, 2]), abs=1e-6)


def test_returns_witness():
    psi = np.array([0, 1 / np.sqrt(2), -1 / np.sqrt(2), 0], dtype=complex)
    res = fermionic_gmn(_ketbra(psi), return_witness=True)
    assert res.witness.shape == (4, 4)
    assert np.isfinite(res.witness).all()


def test_pure_w_state_is_minimum_bipartite_negativity():
    rho = _ketbra(_single_occupancy_w_state(3))
    expected = min(fermionic_negativity(rho, [mode], [2, 2, 2]) for mode in range(3))
    assert fermionic_gmn(rho) == pytest.approx(expected, abs=5e-4)


def test_vanishes_for_product_state():
    psi = np.zeros(8, dtype=complex)
    psi[7] = 1
    assert fermionic_gmn(_ketbra(psi)) == pytest.approx(0, abs=1e-6)


@pytest.mark.parametrize(
    ("psi", "p_below", "p_above"),
    [(_ghz_x_state_3(), 0.790, 0.810), (_single_occupancy_w_state(3), 0.720, 0.735)],
    ids=["GHZ3", "W3"],
)
def test_white_noise_thresholds(psi, p_below, p_above):
    assert fermionic_gmn(_white_noise_mix(psi, p_below)) > 1e-6
    assert fermionic_gmn(_white_noise_mix(psi, p_above)) <= 1e-6
