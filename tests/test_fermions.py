import numpy as np
import pytest

from entdetect.fermions import (
    compute_fermionic_partial_transpose_matrix,
    fermionic_gaussian_thermal_state,
    fermionic_log_negativity,
    fermionic_negativity,
    fermionic_partial_trace,
    fermionic_partial_transpose,
    fermionic_permute_systems,
    gaussian_wick_four_point_violations,
    jordan_wigner_majoranas,
    random_fermionic_density_matrix,
    random_mixed_fermionic_state,
)
from entdetect.linalg import partial_transpose

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def _log_trace_norm(matrix):
    return np.log(np.linalg.norm(matrix, ord="nuc"))


def _werner_state(p):
    psi_singlet = np.array([0, -1 / np.sqrt(2), 1 / np.sqrt(2), 0], dtype=complex)
    return ((1 - p) / 4) * np.eye(4, dtype=complex) + p * np.outer(psi_singlet, psi_singlet.conj())


def _ketbra(psi):
    return np.outer(psi, psi.conj())


def _normalize(psi):
    return psi / np.linalg.norm(psi)


def _partial_trace_last_mode(rho, modes):
    tensor = rho.reshape([2] * (2 * modes))
    reduced = np.trace(tensor, axis1=modes - 1, axis2=2 * modes - 1)
    return reduced.reshape(2 ** (modes - 1), 2 ** (modes - 1))


def _partial_trace_grouped(rho, kept_count, traced_count):
    modes = kept_count + traced_count
    tensor = rho.reshape([2] * (2 * modes))
    for _ in range(traced_count):
        tensor = np.trace(tensor, axis1=kept_count, axis2=2 * kept_count + traced_count)
        traced_count -= 1
    return tensor.reshape(2**kept_count, 2**kept_count)


def _majorana_c_odd(mode, modes):
    operator = np.zeros((2**modes, 2**modes), dtype=complex)
    for col in range(2**modes):
        bits = tuple((col >> shift) & 1 for shift in range(modes - 1, -1, -1))
        sign = -1 if sum(bits[:mode]) % 2 else 1
        row_bits = list(bits)
        row_bits[mode] = 1 - row_bits[mode]

        row = 0
        for bit in row_bits:
            row = (row << 1) | bit
        operator[row, col] = sign

    return operator


def _random_physical_density_matrix(rng, modes):
    size = 2**modes
    parity = np.array([bin(index).count("1") % 2 for index in range(size)])
    rho = np.zeros((size, size), dtype=complex)

    for sector in [0, 1]:
        indices = np.flatnonzero(parity == sector)
        ginibre = rng.normal(size=(len(indices), len(indices))) + 1j * rng.normal(
            size=(len(indices), len(indices))
        )
        block = ginibre @ ginibre.conj().T
        rho[np.ix_(indices, indices)] = block

    return rho / np.trace(rho)


def test_random_fermionic_density_matrix_properties():
    rho = random_fermionic_density_matrix(3, seed=123)
    parity = np.array([index.bit_count() % 2 for index in range(8)])

    assert rho.shape == (8, 8)
    assert np.isclose(np.trace(rho), 1)
    assert np.allclose(rho, rho.conj().T)
    assert np.min(np.linalg.eigvalsh(rho)) > -1e-12
    assert np.allclose(rho[parity[:, None] != parity[None, :]], 0)


def test_random_fermionic_density_matrix_seed_reproducibility_and_alias():
    rho_a = random_fermionic_density_matrix(3, k=2, seed=321)
    rho_b = random_fermionic_density_matrix(3, k=2, seed=321)
    rho_c = random_mixed_fermionic_state(3, k=2, seed=321)

    assert np.allclose(rho_a, rho_b)
    assert np.allclose(rho_a, rho_c)


def test_jordan_wigner_majoranas_satisfy_clifford_relations():
    modes = 3
    majoranas = jordan_wigner_majoranas(modes)
    identity = np.eye(2**modes, dtype=complex)

    assert len(majoranas) == 2 * modes
    for index, majorana in enumerate(majoranas):
        assert majorana.shape == identity.shape
        assert np.allclose(majorana, majorana.conj().T)
        if index % 2 == 0:
            assert np.allclose(majorana, _majorana_c_odd(index // 2, modes))

    for j, cj in enumerate(majoranas):
        for k, ck in enumerate(majoranas):
            expected = 2 * identity if j == k else np.zeros_like(identity)
            assert np.allclose(cj @ ck + ck @ cj, expected)


def test_fermionic_gaussian_thermal_state_constructs_density_matrix():
    A = np.zeros((4, 4), dtype=float)
    A[0, 1] = 0.7
    A[1, 0] = -0.7
    A[2, 3] = -0.2
    A[3, 2] = 0.2

    rho = fermionic_gaussian_thermal_state(A, beta=0.5)

    assert rho.shape == (4, 4)
    assert np.allclose(rho, rho.conj().T)
    assert np.isclose(np.trace(rho), 1)
    assert np.min(np.linalg.eigvalsh(rho)) >= -1e-12


def test_fermionic_gaussian_thermal_state_satisfies_four_point_wick_identity():
    A = np.zeros((6, 6), dtype=float)
    A[0, 1] = 0.7
    A[1, 0] = -0.7
    A[2, 5] = -0.4
    A[5, 2] = 0.4
    A[3, 4] = 0.2
    A[4, 3] = -0.2

    rho = fermionic_gaussian_thermal_state(A, beta=0.9)
    violations = gaussian_wick_four_point_violations(rho, num_modes=3)

    assert max(abs(residual) for residual in violations.values()) < 1e-12


def test_fermionic_partial_transpose_core_pauli_rules():
    assert np.allclose(
        fermionic_partial_transpose(
            np.kron(X, I), [0], [2, 2], include_particle_hole_unitary=False
        ),
        1j * np.kron(X, I),
    )
    assert np.allclose(
        fermionic_partial_transpose(
            np.kron(Y, I), [0], [2, 2], include_particle_hole_unitary=False
        ),
        -1j * np.kron(Y, I),
    )
    assert np.allclose(
        fermionic_partial_transpose(
            np.kron(Z, I), [0], [2, 2], include_particle_hole_unitary=False
        ),
        np.kron(Z, I),
    )


def test_fermionic_partial_transpose_twice_is_local_parity_conjugation():
    rng = np.random.default_rng(123)
    rho = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    rho = rho + rho.conj().T

    first = fermionic_partial_transpose(rho, [0], [2, 2])
    second = fermionic_partial_transpose(first, [0], [2, 2])

    parity_a = np.kron(Z, I)
    assert np.allclose(second, parity_a @ rho @ parity_a)


def test_fermionic_partial_transpose_rejects_non_leading_modes():
    with pytest.raises(NotImplementedError, match="leading normal-ordered block"):
        fermionic_partial_transpose(np.eye(4), [1], [2, 2])


def test_fermionic_partial_transpose_rejects_non_qubit_dimensions():
    with pytest.raises(ValueError, match=r"dim == \[2\] \* N"):
        fermionic_partial_transpose(np.eye(4), [0], [4])


def test_fermionic_partial_transpose_rejects_unknown_method():
    with pytest.raises(ValueError, match="method"):
        fermionic_partial_transpose(np.eye(4), [0], [2, 2], method="fast")


def test_fermionic_permute_systems_includes_exchange_sign_for_vectors():
    state = np.zeros(4, dtype=complex)
    state[3] = 1

    permuted = fermionic_permute_systems(state, [1, 0], [2, 2])

    expected = np.zeros(4, dtype=complex)
    expected[3] = -1
    assert np.allclose(permuted, expected)


def test_fermionic_permute_systems_round_trips_operator():
    rng = np.random.default_rng(4321)
    rho = _random_physical_density_matrix(rng, modes=3)
    perm = [2, 0, 1]
    inverse = [perm.index(mode) for mode in range(3)]

    permuted = fermionic_permute_systems(rho, perm, [2, 2, 2])
    restored = fermionic_permute_systems(permuted, inverse, [2, 2, 2])

    assert np.allclose(restored, rho)


def test_fermionic_partial_trace_matches_grouped_bosonic_trace():
    rng = np.random.default_rng(55)
    rho = _random_physical_density_matrix(rng, modes=3)

    got = fermionic_partial_trace(rho, [2], [2, 2, 2])
    expected = _partial_trace_grouped(rho, kept_count=2, traced_count=1)

    assert np.allclose(got, expected)


def test_fermionic_partial_trace_uses_signed_permutation_for_arbitrary_modes():
    rng = np.random.default_rng(56)
    rho = _random_physical_density_matrix(rng, modes=3)

    got = fermionic_partial_trace(rho, [0], [2, 2, 2])
    grouped = fermionic_permute_systems(rho, [1, 2, 0], [2, 2, 2])
    expected = _partial_trace_grouped(grouped, kept_count=2, traced_count=1)

    assert np.allclose(got, expected)


def test_fermionic_partial_trace_order_independence_for_random_five_fermion_rdm():
    rho = random_fermionic_density_matrix(5, seed=57)

    # Path 1: trace original mode 2, then original mode 4.  After tracing
    # original mode 2, the remaining order is [0, 1, 3, 4], so original mode 4
    # is reduced index 3.
    path_1 = fermionic_partial_trace(rho, [2], [2] * 5)
    path_1 = fermionic_partial_trace(path_1, [3], [2] * 4)

    # Path 2: trace original mode 4, then original mode 2.  After tracing
    # original mode 4, the remaining order is [0, 1, 2, 3], so original mode 2
    # remains reduced index 2.
    path_2 = fermionic_partial_trace(rho, [4], [2] * 5)
    path_2 = fermionic_partial_trace(path_2, [2], [2] * 4)

    direct = fermionic_partial_trace(rho, [2, 4], [2] * 5)

    assert path_1.shape == (8, 8)
    assert np.allclose(path_1, path_2)
    assert np.allclose(path_1, direct)


def test_fermionic_partial_trace_three_step_order_independence_for_random_five_fermion_rdm():
    rho = random_fermionic_density_matrix(5, seed=58)

    # User-facing labels 3, 1, 5 correspond to original zero-based modes
    # 2, 0, 4.  The remaining original modes are [1, 3].
    path_1 = fermionic_partial_trace(rho, [2], [2] * 5)
    path_1 = fermionic_partial_trace(path_1, [0], [2] * 4)
    path_1 = fermionic_partial_trace(path_1, [2], [2] * 3)

    # User-facing labels 5, 1, 3 correspond to original zero-based modes
    # 4, 0, 2.  Track each original mode's reduced index after prior traces.
    path_2 = fermionic_partial_trace(rho, [4], [2] * 5)
    path_2 = fermionic_partial_trace(path_2, [0], [2] * 4)
    path_2 = fermionic_partial_trace(path_2, [1], [2] * 3)

    direct = fermionic_partial_trace(rho, [0, 2, 4], [2] * 5)

    assert path_1.shape == (4, 4)
    assert np.allclose(path_1, path_2)
    assert np.allclose(path_1, direct)


def test_sparse_partial_transpose_matrix_matches_method():
    rho = _werner_state(0.4)
    sparse_map = compute_fermionic_partial_transpose_matrix([0], [2, 2])
    from_matrix = (sparse_map @ rho.reshape(-1)).reshape(rho.shape)
    from_method = fermionic_partial_transpose(rho, [0], [2, 2], method="sparse")

    assert sparse_map.shape == (16, 16)
    assert sparse_map.nnz == 16
    assert np.allclose(from_matrix, from_method)


def test_sparse_partial_transpose_supports_nonleading_modes():
    rng = np.random.default_rng(1234)
    rho = _random_physical_density_matrix(rng, modes=3)
    sparse_pt = fermionic_partial_transpose(rho, [1], [2, 2, 2], method="sparse")
    twice = fermionic_partial_transpose(sparse_pt, [1], [2, 2, 2], method="sparse")
    parity_b = np.kron(np.kron(I, Z), I)

    assert np.allclose(twice, parity_b @ rho @ parity_b)


@pytest.mark.parametrize("modes", [2, 3, 4])
@pytest.mark.parametrize("num_transposed_modes", [1, 2])
@pytest.mark.parametrize("include_particle_hole_unitary", [False, True])
def test_optimized_methods_match_bruteforce_for_random_physical_density_matrices(
    modes,
    num_transposed_modes,
    include_particle_hole_unitary,
):
    if num_transposed_modes > modes:
        pytest.skip("transposed block cannot exceed total modes")

    rng = np.random.default_rng(1000 + 10 * modes + num_transposed_modes)
    rho = _random_physical_density_matrix(rng, modes)
    sys = list(range(num_transposed_modes))
    dims = [2] * modes

    brute = fermionic_partial_transpose(
        rho,
        sys,
        dims,
        include_particle_hole_unitary=include_particle_hole_unitary,
        method="bruteforce",
    )
    tensor = fermionic_partial_transpose(
        rho,
        sys,
        dims,
        include_particle_hole_unitary=include_particle_hole_unitary,
        method="tensor",
    )
    sparse = fermionic_partial_transpose(
        rho,
        sys,
        dims,
        include_particle_hole_unitary=include_particle_hole_unitary,
        method="sparse",
    )

    assert np.allclose(tensor, brute)
    assert np.allclose(sparse, brute)


def test_two_majorana_state_has_nonzero_fermionic_negativity():
    rho = (
        np.array(
            [
                [1, 0, 0, 1],
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [1, 0, 0, 1],
            ],
            dtype=complex,
        )
        / 4
    )

    bosonic_pt = partial_transpose(rho, [0], [2, 2])
    fermionic_pt = fermionic_partial_transpose(rho, [0], [2, 2])

    bosonic_negativity = (np.linalg.norm(bosonic_pt, ord="nuc") - 1) / 2
    f_negativity = fermionic_negativity(rho, [0], [2, 2])

    assert np.isclose(bosonic_negativity, 0)
    assert f_negativity > 0
    assert np.isclose(f_negativity, (np.sqrt(2) - 1) / 2)
    assert np.isclose(fermionic_log_negativity(rho, [0], [2, 2]), 0.5)
    assert np.isclose(fermionic_log_negativity(rho, [0], [2, 2], base=None), np.log(np.sqrt(2)))
    assert np.isclose(_log_trace_norm(fermionic_pt), np.log(np.sqrt(2)))


@pytest.mark.parametrize("occupied_indices", [(0, 3), (1, 2)])
def test_two_fermion_pure_state_negativity_matches_pdf_formula(occupied_indices):
    lambda_0 = 0.6
    lambda_1 = np.sqrt(1 - lambda_0**2) * np.exp(0.3j)
    psi = np.zeros(4, dtype=complex)
    psi[occupied_indices[0]] = lambda_0
    psi[occupied_indices[1]] = lambda_1
    rho = _ketbra(psi)

    fermionic_pt = fermionic_partial_transpose(rho, [0], [2, 2])
    negativity = (np.linalg.norm(fermionic_pt, ord="nuc") - 1) / 2

    assert np.isclose(negativity, abs(lambda_0 * lambda_1))
    assert np.isclose(fermionic_negativity(rho, [0], [2, 2]), abs(lambda_0 * lambda_1))
    assert np.isclose(_log_trace_norm(fermionic_pt), np.log(1 + 2 * abs(lambda_0 * lambda_1)))


@pytest.mark.parametrize("p", [0.0, 0.1, 1 / 3, 0.5, 1.0])
def test_werner_state_log_negativity_matches_analytic_formulas(p):
    rho = _werner_state(p)

    bosonic_pt = partial_transpose(rho, [0], [2, 2])
    fermionic_pt = fermionic_partial_transpose(rho, [0], [2, 2])

    expected_bosonic = np.log((3 / 4) * (1 + p) + (1 / 4) * abs(1 - 3 * p))
    expected_fermionic = np.log((1 / 2) * (1 + p) + (1 / 2) * np.sqrt(5 * p**2 - 2 * p + 1))

    assert np.isclose(_log_trace_norm(bosonic_pt), expected_bosonic)
    assert np.isclose(_log_trace_norm(fermionic_pt), expected_fermionic)
    assert np.isclose(fermionic_log_negativity(rho, [0], [2, 2], base=None), expected_fermionic)


def test_two_complex_fermion_singlet_has_log_two_negativity():
    psi = np.array([0, 1 / np.sqrt(2), -1 / np.sqrt(2), 0], dtype=complex)
    rho = _ketbra(psi)

    bosonic_pt = partial_transpose(rho, [0], [2, 2])
    fermionic_pt = fermionic_partial_transpose(rho, [0], [2, 2])

    assert np.isclose(_log_trace_norm(bosonic_pt), np.log(2))
    assert np.isclose(_log_trace_norm(fermionic_pt), np.log(2))
    assert np.isclose(fermionic_log_negativity(rho, [0], [2, 2]), 1)


def test_w_state_log_negativity_matches_pdf_examples():
    psi = np.zeros(8, dtype=complex)
    psi[[4, 2, 1]] = 1 / np.sqrt(3)
    rho = _ketbra(psi)
    rho_ab = _partial_trace_last_mode(rho, modes=3)

    fermionic_a_bc = fermionic_partial_transpose(rho, [0], [2, 2, 2])
    fermionic_ab = fermionic_partial_transpose(rho_ab, [0], [2, 2])
    bosonic_a_bc = partial_transpose(rho, [0], [2, 2, 2])
    bosonic_ab = partial_transpose(rho_ab, [0], [2, 2])

    expected_a_bc = np.log(1 + 2 * np.sqrt(2) / 3)
    expected_ab = np.log((2 + np.sqrt(5)) / 3)

    assert np.isclose(_log_trace_norm(fermionic_a_bc), expected_a_bc)
    assert np.isclose(_log_trace_norm(fermionic_ab), expected_ab)
    assert np.isclose(_log_trace_norm(bosonic_a_bc), expected_a_bc)
    assert np.isclose(_log_trace_norm(bosonic_ab), expected_ab)


def test_generic_three_fermion_pure_state_a_negativity_matches_pdf_formula():
    lambdas = np.array([0.3 + 0.1j, -0.4j, 0.5, -0.2 + 0.65j], dtype=complex)
    lambdas = _normalize(lambdas)
    psi = np.zeros(8, dtype=complex)
    psi[0] = lambdas[0]
    psi[6] = lambdas[1]
    psi[3] = lambdas[2]
    psi[5] = lambdas[3]
    rho = _ketbra(psi)

    fermionic_pt = fermionic_partial_transpose(rho, [0], [2, 2, 2])
    expected_negativity = np.sqrt(
        (abs(lambdas[0]) ** 2 + abs(lambdas[2]) ** 2)
        * (abs(lambdas[1]) ** 2 + abs(lambdas[3]) ** 2)
    )

    assert np.isclose((np.linalg.norm(fermionic_pt, ord="nuc") - 1) / 2, expected_negativity)
    assert np.isclose(_log_trace_norm(fermionic_pt), np.log(1 + 2 * expected_negativity))


def test_ghz_fermion_state_log_negativity_matches_pdf_examples():
    psi = np.zeros(8, dtype=complex)
    psi[[4, 2, 1, 7]] = 1 / 2
    rho = _ketbra(psi)
    rho_ab = _partial_trace_last_mode(rho, modes=3)

    fermionic_a_bc = fermionic_partial_transpose(rho, [0], [2, 2, 2])
    fermionic_ab = fermionic_partial_transpose(rho_ab, [0], [2, 2])
    bosonic_ab = partial_transpose(rho_ab, [0], [2, 2])

    assert np.isclose(_log_trace_norm(fermionic_a_bc), np.log(2))
    assert np.isclose(_log_trace_norm(fermionic_ab), np.log(np.sqrt(2)))
    assert np.isclose(_log_trace_norm(bosonic_ab), 0)


def test_ghz_w_interpolation_separable_pdf_point_has_zero_negativity():
    w_state = np.zeros(8, dtype=complex)
    w_state[[4, 2, 1]] = 1 / np.sqrt(3)
    ghz_state = np.zeros(8, dtype=complex)
    ghz_state[[4, 2, 1, 7]] = 1 / 2

    p = 4 / 7
    psi = _normalize(np.sqrt(p) * ghz_state - np.sqrt(1 - p) * w_state)
    rho = _ketbra(psi)

    assert np.isclose(abs(psi[7]), 1)
    assert np.isclose(_log_trace_norm(fermionic_partial_transpose(rho, [0], [2, 2, 2])), 0)


def test_three_majorana_state_log_negativity_matches_pdf_examples():
    c1 = _majorana_c_odd(0, modes=3)
    c2 = _majorana_c_odd(1, modes=3)
    c3 = _majorana_c_odd(2, modes=3)
    rho = (np.eye(8, dtype=complex) - 1j * (c2 @ c1 + c3 @ c2 + c1 @ c3) / np.sqrt(3)) / 8
    rho_ab = _partial_trace_last_mode(rho, modes=3)

    fermionic_a_bc = fermionic_partial_transpose(rho, [0], [2, 2, 2])
    fermionic_ab = fermionic_partial_transpose(rho_ab, [0], [2, 2])
    bosonic_a_bc = partial_transpose(rho, [0], [2, 2, 2])
    bosonic_ab = partial_transpose(rho_ab, [0], [2, 2])

    assert np.allclose(rho, rho.conj().T)
    assert np.isclose(np.trace(rho), 1)
    assert np.isclose(_log_trace_norm(fermionic_a_bc), np.log(np.sqrt(5 / 3)))
    assert np.isclose(_log_trace_norm(fermionic_ab), np.log(2 / np.sqrt(3)))
    assert np.isclose(_log_trace_norm(bosonic_a_bc), 0)
    assert np.isclose(_log_trace_norm(bosonic_ab), 0)


def test_optimized_methods_match_bruteforce_for_pdf_examples():
    two_majorana = (
        np.array(
            [
                [1, 0, 0, 1],
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [1, 0, 0, 1],
            ],
            dtype=complex,
        )
        / 4
    )

    w_state = np.zeros(8, dtype=complex)
    w_state[[4, 2, 1]] = 1 / np.sqrt(3)
    w_rho = _ketbra(w_state)

    ghz_state = np.zeros(8, dtype=complex)
    ghz_state[[4, 2, 1, 7]] = 1 / 2
    ghz_rho = _ketbra(ghz_state)

    c1 = _majorana_c_odd(0, modes=3)
    c2 = _majorana_c_odd(1, modes=3)
    c3 = _majorana_c_odd(2, modes=3)
    three_majorana = (
        np.eye(8, dtype=complex) - 1j * (c2 @ c1 + c3 @ c2 + c1 @ c3) / np.sqrt(3)
    ) / 8

    examples = [
        (two_majorana, [0], [2, 2]),
        (_werner_state(0.4), [0], [2, 2]),
        (w_rho, [0], [2, 2, 2]),
        (_partial_trace_last_mode(w_rho, modes=3), [0], [2, 2]),
        (ghz_rho, [0], [2, 2, 2]),
        (_partial_trace_last_mode(ghz_rho, modes=3), [0], [2, 2]),
        (three_majorana, [0], [2, 2, 2]),
        (_partial_trace_last_mode(three_majorana, modes=3), [0], [2, 2]),
    ]

    for rho, sys, dims in examples:
        brute = fermionic_partial_transpose(rho, sys, dims, method="bruteforce")
        tensor = fermionic_partial_transpose(rho, sys, dims, method="tensor")
        sparse = fermionic_partial_transpose(rho, sys, dims, method="sparse")
        assert np.allclose(tensor, brute)
        assert np.allclose(sparse, brute)
