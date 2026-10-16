import numpy as np
import pytest

from medt import gilbert_criterion, ring_inflation_robustness
from medt.states import ghz, ket_to_dm, w, white_noise

pytestmark = pytest.mark.filterwarnings("ignore:Solution may be inaccurate:UserWarning")


# Level-2 values computed with MOSEK by Liuke's MATLAB code for Lyu et al., PRL 137, 110204 (2026).
@pytest.mark.parametrize("psi, t_ref", [(ghz(3), 0.632115682541130), (w(3), 0.699876830472147)])
def test_level2_matches_reference(psi, t_ref):
    t = ring_inflation_robustness(ket_to_dm(psi), (2, 2, 2))
    assert t == pytest.approx(t_ref, abs=2e-4)


def test_noisy_w_not_certified():
    assert ring_inflation_robustness(white_noise(ket_to_dm(w(3)), 0.5), (2, 2, 2)) >= 1


def test_gilbert_criterion():
    rho = ket_to_dm(ghz(3))
    sigma = np.diag([0.5, 0, 0, 0, 0, 0, 0, 0.5])  # a separable (hence network) state
    eps = gilbert_criterion(rho, sigma)
    delta = np.linalg.norm(rho - sigma)
    assert eps == pytest.approx(delta / (delta + 1 / np.sqrt(56)))
    assert gilbert_criterion(sigma, sigma) == 0


# Level-3 GNME noise thresholds p_c = 1 - t from Lyu et al., PRL 137, 110204 (2026), Table I.
@pytest.mark.slow
def test_level3_ghz_matches_paper():
    t = ring_inflation_robustness(ket_to_dm(ghz(3)), (2, 2, 2), level=3)
    assert 1 - t == pytest.approx(0.3878, abs=2e-4)
