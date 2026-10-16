import numpy as np
import pytest

from entdetect import is_ppt, kampermann_sep
from entdetect.separability import max_product_overlap
from tests.states import ket_to_rho


def random_product_mixture(dA, dB, k, noise, rng):
    rho = noise * np.eye(dA * dB, dtype=complex) / (dA * dB)
    for w in rng.dirichlet(np.ones(k)) * (1 - noise):
        a = rng.normal(size=dA) + 1j * rng.normal(size=dA)
        b = rng.normal(size=dB) + 1j * rng.normal(size=dB)
        rho += w * ket_to_rho(np.kron(a, b))
    return rho


def test_product_overlap_of_product_state():
    psi = np.kron([1, 1j], [1, 0, 1]) / 2
    value, _ = max_product_overlap(ket_to_rho(psi), (2, 3), rng=np.random.default_rng(0))
    assert value == pytest.approx(1)


def test_certifies_noisy_product_state():
    rho = 0.5 * ket_to_rho([1, 0, 0, 0]) + 0.5 * np.eye(4) / 4
    assert kampermann_sep(rho, (2, 2), seed=0).separable


@pytest.mark.parametrize("dims", [(2, 3), (3, 3)])
def test_certifies_random_separable_states(dims):
    rng = np.random.default_rng(1)
    rho = random_product_mixture(*dims, k=8, noise=0.2, rng=rng)
    res = kampermann_sep(rho, dims, seed=0)
    assert res.separable
    assert res.purity <= 1 / (np.prod(dims) - 1)


def test_does_not_certify_entangled_state():
    rho = 0.6 * ket_to_rho([1, 0, 0, 1]) + 0.4 * np.eye(4) / 4
    assert not is_ppt(rho, (0,), (2, 2))
    assert not kampermann_sep(rho, (2, 2), max_iter=500, seed=0).separable
