from pathlib import Path

import numpy as np
import pytest

from medt import gmn, negativity
from tests.states import ghz, ket_to_rho, w_state

DATA = Path(__file__).parent / "data"

def test_two_qubits_equals_negativity():
    rho = 0.7 * ket_to_rho([1, 0, 0, 1]) + 0.3 * np.eye(4) / 4
    assert gmn(rho) == pytest.approx(negativity(rho), abs=1e-6)


@pytest.mark.parametrize("n", [3, 4])
def test_ghz(n):
    assert gmn(ghz(n)) == pytest.approx(0.5, abs=1e-6)


def test_w_bounded_by_bipartite_negativity():
    rho = w_state(3)
    value = gmn(rho)
    assert 0.1 < value <= negativity(rho, (0,)) + 1e-6


def test_biseparable_state():
    bell = ket_to_rho([1, 0, 0, 1])
    rho = np.kron(bell, ket_to_rho([1, 1]))
    assert gmn(rho) == pytest.approx(0, abs=1e-6)


@pytest.mark.parametrize("p, entangled", [(0.40, False), (0.45, True)])
def test_noisy_ghz_threshold(p, entangled):
    # p * GHZ + (1 - p) * I/8 is genuinely multipartite entangled iff p > 3/7.
    rho = p * ghz(3) + (1 - p) * np.eye(8) / 8
    assert (gmn(rho) > 1e-6) == entangled


def test_witness_detects_state():
    res = gmn(ghz(3), return_witness=True)
    assert np.real(np.trace(res.witness @ ghz(3))) == pytest.approx(-res.value, abs=1e-6)


def test_non_hermitian_rejected():
    with pytest.raises(ValueError):
        gmn(np.triu(np.ones((8, 8))))


# Reference values from Lyu, Chandorkar, Kapoor, Takei, Sorensen, Witczak-Krempa,
# "Multiparty entanglement loops in quantum spin liquids", arXiv:2505.18124
# (Kitaev honeycomb model in a [111] field, N=32), computed with MOSEK.
kitaev = np.load(DATA / "kitaev_rdms.npz")


@pytest.mark.parametrize("name", ["rdm_adj3_0", "rdm_adj3_25"])
def test_kitaev_three_sites(name):
    assert max(gmn(kitaev[f"{name}_rho"], (2, 2, 2)), 0) == pytest.approx(
        max(float(kitaev[f"{name}_ref"]), 0), abs=1e-6
    )


@pytest.mark.slow
def test_kitaev_hexagon():
    # Six spins grouped into three pairs; about a minute with SCS.
    value = gmn(kitaev["rdm_hexagonal_0_rho"], (4, 4, 4))
    assert value == pytest.approx(float(kitaev["rdm_hexagonal_0_ref"]), abs=1e-6)
