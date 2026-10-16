import numpy as np
import pytest

from medt import (
    polytope_biseparable,
    polytope_fully_separable,
    polytope_separable,
    polytope_separable_all_cuts,
)

pytestmark = pytest.mark.filterwarnings("ignore:Solution may be inaccurate:UserWarning")


def noisy(p, n):
    ghz = np.zeros((2**n, 2**n))
    ghz[np.ix_([0, -1], [0, -1])] = 0.5
    return p * ghz + (1 - p) * np.eye(2**n) / 2**n


def test_bell_robustness_is_exact():
    # p * Bell + noise is separable iff p <= 1/3, so the robustness is 1/(3p).
    res = polytope_separable(noisy(0.5, 2), (2, 2), n_points=30, seed=0)
    assert res.robustness == pytest.approx(2 / 3, abs=1e-5)
    assert not res.certified


def test_bell_certified_below_threshold():
    assert polytope_separable(noisy(0.3, 2), (2, 2), n_points=30, seed=0).certified


@pytest.mark.parametrize("p, certified", [(0.15, True), (0.3, False)])
def test_ghz_full_separability(p, certified):
    # Noisy GHZ3 is fully separable iff p <= 1/5.
    res = polytope_fully_separable(noisy(p, 3), (2, 2, 2), n_points=40, seed=0)
    assert res.certified == certified


@pytest.mark.parametrize("p, certified", [(0.35, True), (0.5, False)])
def test_ghz_biseparability(p, certified):
    # Noisy GHZ3 is biseparable iff p <= 3/7.
    res = polytope_biseparable(noisy(p, 3), (2, 2, 2), n_points=30, seed=0)
    assert res.certified == certified


def test_all_cuts():
    assert polytope_separable_all_cuts(noisy(0.15, 3), (2, 2, 2), n_points=30, seed=0).certified
