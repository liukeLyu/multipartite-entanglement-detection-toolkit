import numpy as np
import pytest

from medt.linalg import bipartitions, partial_transpose, permute_subsystems


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
