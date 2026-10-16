import numpy as np
import pytest

from entdetect import gmc_lower_bound, gmc_pure, gmc_x_state
from tests.states import ghz, w_state


def noisy_ghz(p, n=3):
    return p * ghz(n) + (1 - p) * np.eye(2**n) / 2**n


def test_pure_ghz_and_w():
    psi = np.zeros(8)
    psi[[0, 7]] = 1
    assert gmc_pure(psi) == pytest.approx(1)
    psi = np.zeros(8)
    psi[[1, 2, 4]] = 1
    # min over cuts of sqrt(2 (1 - tr rho_A^2)), with tr rho_A^2 = 5/9 for W.
    assert gmc_pure(psi) == pytest.approx(np.sqrt(8 / 9))


@pytest.mark.parametrize("p", [0.3, 3 / 7, 0.6, 1.0])
def test_x_state_formula_on_noisy_ghz(p):
    assert gmc_x_state(noisy_ghz(p)) == pytest.approx(max(0, p - 3 * (1 - p) / 4))


def test_x_state_rejects_non_x():
    with pytest.raises(ValueError):
        gmc_x_state(w_state(3))


@pytest.mark.parametrize("p", [0.6, 0.9])
def test_i2_bound_is_tight_for_noisy_ghz(p):
    # For X states the I2 bound reaches the exact value.
    rho = noisy_ghz(p)
    assert gmc_lower_bound(rho, seed=0) == pytest.approx(gmc_x_state(rho), abs=1e-5)


def test_i2_bound_below_pure_value_for_w():
    psi = np.zeros(8)
    psi[[1, 2, 4]] = 1
    bound = gmc_lower_bound(w_state(3), seed=0)
    assert 0 < bound <= gmc_pure(psi) + 1e-8


def test_i2_bound_zero_for_biseparable():
    bell = np.zeros((4, 4))
    bell[np.ix_([0, 3], [0, 3])] = 0.5
    rho = np.kron(bell, np.eye(2) / 2)
    assert gmc_lower_bound(rho, seed=0) == 0
