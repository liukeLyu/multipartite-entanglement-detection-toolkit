import numpy as np
import pytest

from medt import gilbert_criterion, ring_inflation_robustness
from medt.states import ghz, ket_to_dm, w, white_noise

pytestmark = pytest.mark.filterwarnings("ignore:Solution may be inaccurate:UserWarning")


def test_ghz_has_network_irreducible_entanglement():
    t = ring_inflation_robustness(ket_to_dm(ghz(3)), (2, 2, 2))
    # Second-order ring inflation with PPT certifies GHZ3 up to about 37% white noise.
    assert t == pytest.approx(0.632, abs=2e-3)


def test_noisy_w_not_certified():
    assert ring_inflation_robustness(white_noise(ket_to_dm(w(3)), 0.5), (2, 2, 2)) >= 1


def test_gilbert_criterion():
    rho = ket_to_dm(ghz(3))
    sigma = np.diag([0.5, 0, 0, 0, 0, 0, 0, 0.5])  # a separable (hence network) state
    eps = gilbert_criterion(rho, sigma)
    delta = np.linalg.norm(rho - sigma)
    assert eps == pytest.approx(delta / (delta + 1 / np.sqrt(56)))
    assert gilbert_criterion(sigma, sigma) == 0
