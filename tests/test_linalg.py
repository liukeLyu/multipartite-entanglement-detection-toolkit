import numpy as np
import pytest
from scipy import sparse

from medt.linalg import bipartitions, keep_systems, partial_trace, partial_transpose, permute_subsystems


def rand_mat(d, rng):
    return rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))


def test_partial_transpose_of_product():
    rng = np.random.default_rng(0)
    A, B, C = rand_mat(2, rng), rand_mat(3, rng), rand_mat(2, rng)
    X = np.kron(np.kron(A, B), C)
    expected = np.kron(np.kron(A.T, B), C.T)
    assert np.allclose(partial_transpose(X, (0, 2), (2, 3, 2)), expected)


def test_full_transpose():
    X = rand_mat(8, np.random.default_rng(1))
    assert np.allclose(partial_transpose(X, (0, 1, 2), (2, 2, 2)), X.T)


def test_permute_subsystems_of_product():
    rng = np.random.default_rng(2)
    A, B, C = rand_mat(2, rng), rand_mat(3, rng), rand_mat(4, rng)
    X = np.kron(np.kron(A, B), C)
    assert np.allclose(permute_subsystems(X, (2, 0, 1), (2, 3, 4)), np.kron(np.kron(C, A), B))


def test_bad_dims():
    with pytest.raises(ValueError):
        partial_transpose(np.eye(8), (0,), (2, 2))


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_bipartition_count(n):
    cuts = bipartitions(n)
    assert len(cuts) == 2 ** (n - 1) - 1
    assert len(set(cuts)) == len(cuts)


def _random_ket(D, seed):
    rng = np.random.default_rng(seed)
    psi = rng.normal(size=D) + 1j * rng.normal(size=D)
    return psi / np.linalg.norm(psi)


@pytest.mark.parametrize("keep", [[0], [2, 0], [3, 1, 2], [0, 1, 2, 3]])
def test_keep_systems_of_ket_matches_density_matrix(keep):
    dims = (2, 3, 2, 2)
    psi = _random_ket(24, seed=1)
    rho = np.outer(psi, psi.conj())
    expected = keep_systems(rho, keep, dims)
    assert np.allclose(keep_systems(psi, keep, dims), expected)
    assert np.allclose(keep_systems(psi[:, None], keep, dims), expected)
    assert np.allclose(keep_systems(sparse.csr_array(psi[:, None]), keep, dims), expected)
    assert np.allclose(keep_systems(sparse.coo_array(psi), keep, dims), expected)
    assert np.allclose(keep_systems(sparse.csr_array(rho), keep, dims), expected)


def test_keep_systems_of_large_sparse_w_state():
    n = 24
    psi = sparse.coo_array((np.full(n, 1 / np.sqrt(n)), (1 << np.arange(n),)), shape=(2**n,))
    rho = keep_systems(psi, [3, 17], [2] * n)
    expected = np.zeros((4, 4))
    expected[0, 0] = (n - 2) / n
    expected[1:3, 1:3] = 1 / n
    assert np.allclose(rho, expected)


def test_keep_systems_of_large_dense_ghz_state():
    n = 20
    psi = np.zeros(2**n)
    psi[[0, -1]] = 1 / np.sqrt(2)
    rho = keep_systems(psi, [0, 9, 19], [2] * n)
    assert np.allclose(rho, np.diag([0.5, 0, 0, 0, 0, 0, 0, 0.5]))


def test_partial_trace_traces_out_and_keeps_order():
    rng = np.random.default_rng(2)
    dims = (2, 3, 2, 3)
    mats = [rand_mat(d, rng) for d in dims]
    X = mats[0]
    for M in mats[1:]:
        X = np.kron(X, M)
    expected = np.trace(mats[2]) * np.kron(np.kron(mats[0], mats[1]), mats[3])
    assert np.allclose(partial_trace(X, [2], dims), expected)
    assert np.allclose(partial_trace(X, [2], dims), keep_systems(X, [0, 1, 3], dims))
    with pytest.raises(ValueError):
        partial_trace(X, [4], dims)


@pytest.mark.parametrize("seed", range(4))
def test_keep_systems_equals_partial_trace_then_permute(seed):
    # keep_systems(keep) == permute(partial_trace(complement), order of keep), for kets and density matrices.
    rng = np.random.default_rng(seed)
    dims = (2, 3, 2, 4, 3)
    psi = _random_ket(int(np.prod(dims)), seed)
    rho = np.outer(psi, psi.conj())
    for _ in range(4):
        keep = list(rng.permutation(5)[: rng.integers(1, 6)])
        trace_out = [k for k in range(5) if k not in keep]
        ranked = sorted(keep)
        reduced = partial_trace(rho, trace_out, dims)
        expected = permute_subsystems(reduced, [ranked.index(k) for k in keep], [dims[k] for k in ranked])
        assert np.allclose(keep_systems(rho, keep, dims), expected)
        assert np.allclose(keep_systems(psi, keep, dims), expected)
        assert np.allclose(partial_trace(psi, trace_out, dims), reduced)


def test_partial_trace_in_steps_matches_keep_systems():
    dims = (3, 2, 2, 3)
    psi = _random_ket(36, seed=9)
    # Trace out subsystem 2, then (in the remaining 0, 1, 3) subsystem 0, then reorder to (3, 1).
    step = partial_trace(partial_trace(psi, [2], dims), [0], [3, 2, 3])
    assert np.allclose(permute_subsystems(step, [1, 0], [2, 3]), keep_systems(psi, [3, 1], dims))
