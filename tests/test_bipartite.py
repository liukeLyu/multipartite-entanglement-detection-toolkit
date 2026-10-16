import numpy as np
import pytest

from medt import ccnr, is_ppt, log_negativity, partial_trace, realign
from tests.states import horodecki_3x3, ket_to_rho


def test_realign_of_product():
    rng = np.random.default_rng(0)
    A, B = rng.normal(size=(2, 2)), rng.normal(size=(3, 3))
    assert np.allclose(realign(np.kron(A, B), (2, 3)), np.outer(A.ravel(), B.ravel()))


def test_partial_trace_of_product():
    rng = np.random.default_rng(1)
    A, B, C = (rng.normal(size=(d, d)) for d in (2, 3, 2))
    X = np.kron(np.kron(A, B), C)
    assert np.allclose(partial_trace(X, (2, 0), (2, 3, 2)), np.trace(B) * np.kron(C, A))


def test_bell_state():
    bell = ket_to_rho([1, 0, 0, 1])
    assert log_negativity(bell, (0,), (2, 2)) == pytest.approx(1)
    assert ccnr(bell, (2, 2)) == pytest.approx(2)
    assert not is_ppt(bell, (0,), (2, 2))


def test_horodecki_ppt_entangled_state_detected_by_ccnr():
    rho = horodecki_3x3(0.25)
    assert is_ppt(rho, (0,), (3, 3))
    assert ccnr(rho, (3, 3)) > 1


def test_product_state_not_detected():
    rho = np.kron(ket_to_rho([1, 1]), np.eye(3) / 3)
    assert is_ppt(rho, (0,), (2, 3))
    assert ccnr(rho, (2, 3)) <= 1 + 1e-12
