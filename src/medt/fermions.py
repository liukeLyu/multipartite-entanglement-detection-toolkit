"""Fermionic modes: signed permutations, partial trace, partial transpose.

States of ``N`` fermionic modes are written in the occupation basis
``|n_1 ... n_N>`` with mode 1 most significant, as for qubits. Unlike qubits,
reordering modes picks up exchange signs, and the partial transpose is the
fermionic one of Shapourian, Shiozaki and Ryu [1, 2], which respects the
parity superselection rule.

[1] H. Shapourian, K. Shiozaki, S. Ryu, PRB 95, 165101 (2017).
[2] H. Shapourian, S. Ryu, PRA 99, 022310 (2019).
"""

from itertools import combinations

import numpy as np
from scipy import sparse
from scipy.linalg import expm

_FPT_MATRIX_CACHE = {}


def random_fermionic_density_matrix(num_modes, k=None, seed=None, dtype=np.complex128):
    """Generate a random parity-preserving mixed state of ``num_modes`` fermions.

    The output is block diagonal in the global fermion-number parity sectors.
    Each parity block is sampled from an induced complex Ginibre ensemble and
    the combined state is normalized to trace one.
    """
    if num_modes < 1:
        raise ValueError("num_modes must be at least 1.")
    if k is not None and k < 1:
        raise ValueError("k must be at least 1 when provided.")

    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    size = 2**num_modes
    parity = np.array([_bit_parity(index) for index in range(size)])
    rho = np.zeros((size, size), dtype=np.complex128)

    for sector in [0, 1]:
        indices = np.flatnonzero(parity == sector)
        block_cols = len(indices) if k is None else k
        ginibre = rng.normal(size=(len(indices), block_cols))
        ginibre = ginibre + 1j * rng.normal(size=(len(indices), block_cols))
        block = ginibre @ ginibre.conj().T
        rho[np.ix_(indices, indices)] = block

    rho = rho / np.trace(rho)
    return rho.astype(dtype, copy=False)


random_mixed_fermionic_state = random_fermionic_density_matrix


def jordan_wigner_majoranas(num_modes, dtype=np.complex128):
    """Return Jordan-Wigner Majorana operators for ``num_modes`` fermions.

    The returned list is ordered as ``[c_1, c_2, ..., c_{2N}]`` with
    ``c_{2j-1} = Z_1 ... Z_{j-1} X_j`` and
    ``c_{2j} = Z_1 ... Z_{j-1} Y_j``.
    """
    if num_modes < 1:
        raise ValueError("num_modes must be at least 1.")

    identity = np.eye(2, dtype=dtype)
    pauli_x = np.array([[0, 1], [1, 0]], dtype=dtype)
    pauli_y = np.array([[0, -1j], [1j, 0]], dtype=dtype)
    pauli_z = np.array([[1, 0], [0, -1]], dtype=dtype)
    majoranas = []

    for mode in range(num_modes):
        for local_pauli in [pauli_x, pauli_y]:
            factors = []
            for position in range(num_modes):
                if position < mode:
                    factors.append(pauli_z)
                elif position == mode:
                    factors.append(local_pauli)
                else:
                    factors.append(identity)

            operator = factors[0]
            for factor in factors[1:]:
                operator = np.kron(operator, factor)
            majoranas.append(operator)

    return majoranas


def fermionic_gaussian_thermal_state(quadratic_majorana_matrix, beta=1.0, dtype=np.complex128):
    """Construct a fermionic Gaussian thermal state from a quadratic Majorana form.

    ``quadratic_majorana_matrix`` is the real antisymmetric matrix ``A`` in
    ``H = (i / 4) sum_ab A_ab c_a c_b``.  The returned state is
    ``exp(-beta H) / Tr(exp(-beta H))`` in the occupation basis defined by
    :func:`jordan_wigner_majoranas`.
    """
    A = np.asarray(quadratic_majorana_matrix, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] % 2 != 0:
        raise ValueError("quadratic_majorana_matrix must be a square even-dimensional matrix.")
    if not np.allclose(A, -A.T):
        raise ValueError("quadratic_majorana_matrix must be antisymmetric.")

    num_modes = A.shape[0] // 2
    dim = 2**num_modes
    majoranas = jordan_wigner_majoranas(num_modes)
    hamiltonian = np.zeros((dim, dim), dtype=np.complex128)
    for a in range(2 * num_modes):
        for b in range(a + 1, 2 * num_modes):
            hamiltonian += 0.5j * A[a, b] * (majoranas[a] @ majoranas[b])

    hamiltonian = 0.5 * (hamiltonian + hamiltonian.conj().T)
    rho = expm(-float(beta) * hamiltonian)
    rho = rho / np.trace(rho)
    rho = 0.5 * (rho + rho.conj().T)
    return rho.astype(dtype, copy=False)


def majorana_monomial_coefficients(operator, num_modes=None, max_degree=None, atol=0.0):
    """Expand an operator in the ordered Majorana monomial basis.

    Returns a dictionary mapping tuples of Majorana indices to coefficients in
    ``operator = sum_S coeff[S] prod_{j in S} c_j``.  The empty tuple is the
    identity coefficient.  By default all monomials are included.
    """
    matrix = np.asarray(operator, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("operator must be a square matrix.")

    dim = matrix.shape[0]
    if num_modes is None:
        n_float = np.log2(dim)
        num_modes = int(round(n_float))
        if 2**num_modes != dim:
            raise ValueError("operator dimension must be a power of 2 when num_modes is omitted.")
    if dim != 2**num_modes:
        raise ValueError("operator shape does not match num_modes.")

    majoranas = jordan_wigner_majoranas(num_modes)
    identity = np.eye(dim, dtype=np.complex128)
    coefficients = {(): np.trace(matrix) / dim}
    max_degree = 2 * num_modes if max_degree is None else int(max_degree)

    for degree in range(1, max_degree + 1):
        for indices in combinations(range(2 * num_modes), degree):
            monomial = identity
            for index in indices:
                monomial = monomial @ majoranas[index]
            coeff = np.trace(monomial.conj().T @ matrix) / dim
            if atol <= 0.0 or abs(coeff) > atol:
                coefficients[indices] = coeff

    return coefficients


def gaussian_wick_four_point_violations(operator, num_modes=None):
    """Return Wick/Pfaffian residuals for four-Majorana coefficients.

    For a Gaussian operator with nonzero identity coefficient, the Majorana
    expansion coefficients obey
    ``w_ijkl w_0 = w_ij w_kl - w_ik w_jl + w_il w_jk``.
    """
    coefficients = majorana_monomial_coefficients(operator, num_modes=num_modes, max_degree=4)
    w0 = coefficients.get((), 0.0)
    violations = {}
    if abs(w0) == 0.0:
        raise ValueError("identity Majorana coefficient is zero; four-point Wick test is singular.")

    inferred_modes = num_modes
    if inferred_modes is None:
        dim = np.asarray(operator).shape[0]
        inferred_modes = int(round(np.log2(dim)))

    for i, j, k, l in combinations(range(2 * inferred_modes), 4):
        residual = (
            coefficients.get((i, j, k, l), 0.0) * w0
            - coefficients.get((i, j), 0.0) * coefficients.get((k, l), 0.0)
            + coefficients.get((i, k), 0.0) * coefficients.get((j, l), 0.0)
            - coefficients.get((i, l), 0.0) * coefficients.get((j, k), 0.0)
        )
        violations[(i, j, k, l)] = residual

    return violations


def fermionic_permute_systems(X, perm, dim=None):
    """Permute fermionic modes with occupation-dependent exchange signs.

    ``perm`` gives the new mode order in terms of old mode indices.  For an
    operator, the signed permutation is applied to both rows and columns.
    """
    array = np.asarray(X, dtype=np.complex128)
    modes, dims = _validate_fermionic_array(array, dim)
    perm = _validate_permutation(perm, modes)
    target_indices, signs = _fermionic_permutation_basis_action(perm, modes)

    if array.ndim == 1:
        return signs * array[target_indices]
    return signs[:, None] * array[np.ix_(target_indices, target_indices)] * signs[None, :]


def fermionic_partial_trace(X, trace_sys, dim=None):
    """Trace out fermionic modes.

    The output is ordered by the kept modes in their original relative order.
    Internally, arbitrary traced modes are first moved to the end using a signed
    fermionic mode permutation, then an ordinary tensor partial trace is applied.
    """
    matrix = np.asarray(X, dtype=np.complex128)
    modes, dims = _validate_fermionic_operator(matrix, dim)
    traced_modes = _validate_modes(trace_sys, modes)
    kept_modes = [mode for mode in range(modes) if mode not in set(traced_modes)]

    if not traced_modes:
        return matrix.copy()
    if not kept_modes:
        return np.array([[np.trace(matrix)]], dtype=np.complex128)

    perm = kept_modes + traced_modes
    grouped = fermionic_permute_systems(matrix, perm, dims)
    kept_count = len(kept_modes)
    traced_count = len(traced_modes)
    tensor = grouped.reshape([2] * (2 * modes))

    for _ in range(traced_count):
        tensor = np.trace(tensor, axis1=kept_count, axis2=2 * kept_count + traced_count)
        traced_count -= 1

    return tensor.reshape(2**kept_count, 2**kept_count)


def fermionic_partial_transpose(
    X,
    sys,
    dim=None,
    *,
    include_particle_hole_unitary=True,
    method="bruteforce",
):
    """Compute the fermionic partial transpose in occupation basis.

    The implementation follows the occupation-basis formula in Shapourian and
    Ryu, Eq. (23).  The input basis must already be normal ordered as
    ``|n_A, n_B>``; equivalently, this first reference version only supports
    transposing a leading block of modes, ``sys == [0, ..., len(sys)-1]``.

    Parameters
    ----------
    X:
        Square operator matrix of shape ``(2**N, 2**N)``.
    sys:
        Fermionic modes to transpose.  Must be the leading normal-ordered block.
    dim:
        Local dimensions.  Restricted to ``[2] * N``.  If omitted, ``N`` is
        inferred from ``X``.
    include_particle_hole_unitary:
        If true, returns the paper's Eq. (23), ``U_A^dagger core(X) U_A``.
        If false, returns only the core phase-and-index map, which has the same
        trace norm and is useful as a simple negativity reference.
    method:
        ``"bruteforce"`` uses explicit matrix-entry iteration. ``"tensor"``
        uses a vectorized tensor transpose and phase tensor. ``"sparse"``
        applies a cached sparse matrix to ``vec(X)``.

    Returns
    -------
    numpy.ndarray
        The fermionic partially transposed operator.
    """
    matrix = np.asarray(X, dtype=np.complex128)
    modes, dims = _validate_fermionic_operator(matrix, dim)
    transpose_modes = _validate_modes(sys, modes)
    method = _validate_method(method)

    if method == "bruteforce":
        _validate_leading_block(transpose_modes)
        result = _fermionic_partial_transpose_core(matrix, transpose_modes, modes)
    elif method == "tensor":
        _validate_leading_block(transpose_modes)
        result = _fermionic_partial_transpose_core_tensor(matrix, transpose_modes, modes)
    else:
        return _fermionic_partial_transpose_sparse(
            matrix,
            transpose_modes,
            dims,
            include_particle_hole_unitary=include_particle_hole_unitary,
        )

    if include_particle_hole_unitary and transpose_modes:
        return _apply_particle_hole_unitary(result, transpose_modes, modes)
    return result


def fermionic_negativity(
    X,
    sys,
    dim=None,
    *,
    method="sparse",
    include_particle_hole_unitary=True,
):
    """Compute fermionic negativity from the trace norm of the FPT."""
    partial = fermionic_partial_transpose(
        X,
        sys,
        dim,
        include_particle_hole_unitary=include_particle_hole_unitary,
        method=method,
    )
    value = (np.linalg.norm(partial, ord="nuc") - 1) / 2
    return max(0.0, float(np.real_if_close(value)))


def fermionic_log_negativity(
    X,
    sys,
    dim=None,
    *,
    base=2,
    method="sparse",
    include_particle_hole_unitary=True,
):
    """Compute fermionic log-negativity.

    The default logarithm base is 2.  Pass ``base=None`` or ``base=np.e`` for
    the natural-log convention used in some analytic formulas.
    """
    partial = fermionic_partial_transpose(
        X,
        sys,
        dim,
        include_particle_hole_unitary=include_particle_hole_unitary,
        method=method,
    )
    trace_norm = max(1.0, float(np.real_if_close(np.linalg.norm(partial, ord="nuc"))))

    if base is None:
        return float(np.log(trace_norm))
    if base <= 0 or np.isclose(base, 1):
        raise ValueError("base must be positive and different from 1.")
    return float(np.log(trace_norm) / np.log(base))


def compute_fermionic_partial_transpose_matrix(
    sys,
    dim,
    *,
    include_particle_hole_unitary=True,
):
    """Compute sparse matrix for ``vec(X) -> vec(X^TA_f)``.

    Vectorization uses NumPy row-major order, matching ``X.reshape(-1)``.
    """
    dims = list(dim)
    if any(d != 2 for d in dims):
        raise ValueError(
            "compute_fermionic_partial_transpose_matrix currently requires dim == [2] * N."
        )

    modes = len(dims)
    transpose_modes = _validate_modes(sys, modes)
    return _compute_fermionic_partial_transpose_matrix(
        transpose_modes,
        modes,
        include_particle_hole_unitary=include_particle_hole_unitary,
    )


def _fermionic_partial_transpose_core(X, transpose_modes, modes):
    size = 2**modes
    transpose_set = set(transpose_modes)
    rest_modes = [mode for mode in range(modes) if mode not in transpose_set]
    result = np.zeros_like(X, dtype=np.complex128)

    for row in range(size):
        row_bits = _index_to_bits(row, modes)
        tau_a = sum(row_bits[mode] for mode in transpose_modes)
        tau_b = sum(row_bits[mode] for mode in rest_modes)

        for col in range(size):
            value = X[row, col]
            if value == 0:
                continue

            col_bits = _index_to_bits(col, modes)
            tau_a_bar = sum(col_bits[mode] for mode in transpose_modes)
            tau_b_bar = sum(col_bits[mode] for mode in rest_modes)

            new_row_bits = list(row_bits)
            new_col_bits = list(col_bits)
            for mode in transpose_modes:
                new_row_bits[mode] = col_bits[mode]
                new_col_bits[mode] = row_bits[mode]

            phase = _fermionic_transpose_phase(tau_a, tau_a_bar, tau_b, tau_b_bar)
            new_row = _bits_to_index(new_row_bits)
            new_col = _bits_to_index(new_col_bits)
            result[new_row, new_col] += phase * value

    return result


def _fermionic_partial_transpose_core_tensor(X, transpose_modes, modes):
    tensor = X.reshape([2] * (2 * modes))
    phase = _fermionic_transpose_phase_tensor(transpose_modes, modes)
    axes = list(range(2 * modes))

    for mode in transpose_modes:
        axes[mode], axes[modes + mode] = axes[modes + mode], axes[mode]

    return np.transpose(phase * tensor, axes=axes).reshape(X.shape)


def _fermionic_partial_transpose_sparse(X, transpose_modes, dims, *, include_particle_hole_unitary):
    matrix = _get_fpt_matrix_cached(
        transpose_modes,
        dims,
        include_particle_hole_unitary=include_particle_hole_unitary,
    )
    return (matrix @ X.reshape(-1)).reshape(X.shape)


def _get_fpt_matrix_cached(transpose_modes, dims, *, include_particle_hole_unitary):
    key = (tuple(transpose_modes), tuple(dims), include_particle_hole_unitary)
    if key not in _FPT_MATRIX_CACHE:
        _FPT_MATRIX_CACHE[key] = _compute_fermionic_partial_transpose_matrix(
            transpose_modes,
            len(dims),
            include_particle_hole_unitary=include_particle_hole_unitary,
        )
    return _FPT_MATRIX_CACHE[key]


def _compute_fermionic_partial_transpose_matrix(
    transpose_modes, modes, *, include_particle_hole_unitary
):
    size = 2**modes
    vec_size = size * size
    perm = list(transpose_modes) + [
        mode for mode in range(modes) if mode not in set(transpose_modes)
    ]
    inverse_perm = _inverse_permutation(perm)
    leading_modes = list(range(len(transpose_modes)))
    rest_modes = [mode for mode in range(modes) if mode not in set(leading_modes)]

    if include_particle_hole_unitary and transpose_modes:
        particle_hole_targets, particle_hole_signs = _particle_hole_basis_action(
            leading_modes, modes
        )
    else:
        particle_hole_targets = None
        particle_hole_signs = None

    input_indices = np.arange(vec_size, dtype=int)
    output_indices = np.empty(vec_size, dtype=int)
    values = np.empty(vec_size, dtype=np.complex128)

    cursor = 0
    for row in range(size):
        row_bits_old = _index_to_bits(row, modes)
        row_bits, row_perm_sign = _permute_bits_with_sign(row_bits_old, perm)
        tau_a = sum(row_bits[mode] for mode in leading_modes)
        tau_b = sum(row_bits[mode] for mode in rest_modes)

        for col in range(size):
            col_bits_old = _index_to_bits(col, modes)
            col_bits, col_perm_sign = _permute_bits_with_sign(col_bits_old, perm)
            tau_a_bar = sum(col_bits[mode] for mode in leading_modes)
            tau_b_bar = sum(col_bits[mode] for mode in rest_modes)

            new_row_bits = list(row_bits)
            new_col_bits = list(col_bits)
            for mode in leading_modes:
                new_row_bits[mode] = col_bits[mode]
                new_col_bits[mode] = row_bits[mode]

            phase = row_perm_sign * col_perm_sign
            phase *= _fermionic_transpose_phase(tau_a, tau_a_bar, tau_b, tau_b_bar)

            if particle_hole_targets is not None:
                output_row_new = _bits_to_index(new_row_bits)
                output_col_new = _bits_to_index(new_col_bits)
                phase *= particle_hole_signs[output_row_new] * particle_hole_signs[output_col_new]
                new_row_bits = list(_index_to_bits(particle_hole_targets[output_row_new], modes))
                new_col_bits = list(_index_to_bits(particle_hole_targets[output_col_new], modes))

            output_row_bits_old, output_row_sign = _permute_bits_with_sign(
                new_row_bits, inverse_perm
            )
            output_col_bits_old, output_col_sign = _permute_bits_with_sign(
                new_col_bits, inverse_perm
            )
            phase *= output_row_sign * output_col_sign

            output_row = _bits_to_index(output_row_bits_old)
            output_col = _bits_to_index(output_col_bits_old)

            output_indices[cursor] = output_row * size + output_col
            values[cursor] = phase
            cursor += 1

    return sparse.csr_matrix((values, (output_indices, input_indices)), shape=(vec_size, vec_size))


def _fermionic_transpose_phase_tensor(transpose_modes, modes):
    transpose_set = set(transpose_modes)
    rest_modes = [mode for mode in range(modes) if mode not in transpose_set]
    grids = np.indices([2] * (2 * modes), sparse=True)

    tau_a = _sum_grids(grids, transpose_modes)
    tau_b = _sum_grids(grids, rest_modes)
    tau_a_bar = _sum_grids(grids, [modes + mode for mode in transpose_modes])
    tau_b_bar = _sum_grids(grids, [modes + mode for mode in rest_modes])

    a_total = tau_a + tau_a_bar
    b_total = tau_b + tau_b_bar
    phase = np.where(a_total % 2, 1j, 1.0).astype(np.complex128)
    return phase * np.where((a_total * b_total) % 2, -1.0, 1.0)


def _sum_grids(grids, axes):
    if not axes:
        return 0
    total = grids[axes[0]]
    for axis in axes[1:]:
        total = total + grids[axis]
    return total


def _fermionic_transpose_phase(tau_a, tau_a_bar, tau_b, tau_b_bar):
    a_total = tau_a + tau_a_bar
    b_total = tau_b + tau_b_bar
    phase = 1j if a_total % 2 else 1.0
    if (a_total * b_total) % 2:
        phase *= -1
    return phase


def _particle_hole_unitary(transpose_modes, modes):
    size = 2**modes
    unitary = np.eye(size, dtype=np.complex128)
    for mode in transpose_modes:
        unitary = _majorana_c_odd(mode, modes) @ unitary
    return unitary


def _apply_particle_hole_unitary(X, transpose_modes, modes):
    target_indices, signs = _particle_hole_basis_action(transpose_modes, modes)
    source_indices = np.empty_like(target_indices)
    source_indices[target_indices] = np.arange(len(target_indices))

    left = signs[source_indices][:, None]
    right = signs[source_indices][None, :]
    return left * X[np.ix_(source_indices, source_indices)] * right


def _particle_hole_basis_action(transpose_modes, modes):
    size = 2**modes
    target_indices = np.empty(size, dtype=int)
    signs = np.empty(size, dtype=np.complex128)

    for col in range(size):
        bits = list(_index_to_bits(col, modes))
        sign = 1
        for mode in transpose_modes:
            if sum(bits[:mode]) % 2:
                sign *= -1
            bits[mode] = 1 - bits[mode]

        target_indices[col] = _bits_to_index(bits)
        signs[col] = sign

    return target_indices, signs


def _majorana_c_odd(mode, modes):
    size = 2**modes
    operator = np.zeros((size, size), dtype=np.complex128)

    for col in range(size):
        bits = _index_to_bits(col, modes)
        sign = -1 if sum(bits[:mode]) % 2 else 1
        row_bits = list(bits)
        row_bits[mode] = 1 - row_bits[mode]
        row = _bits_to_index(row_bits)
        operator[row, col] = sign

    return operator


def _validate_fermionic_operator(matrix, dim):
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("X must be a square matrix.")

    size = matrix.shape[0]
    modes_float = np.log2(size)
    modes = int(round(modes_float))
    if 2**modes != size:
        raise ValueError("X dimension must be a power of 2.")

    if dim is None:
        dims = [2] * modes
    else:
        dims = list(dim)
        if dims != [2] * modes:
            raise ValueError("fermionic_partial_transpose currently requires dim == [2] * N.")

    return modes, dims


def _validate_fermionic_array(array, dim):
    if array.ndim not in {1, 2}:
        raise ValueError("X must be a vector or square matrix.")
    if array.ndim == 2 and array.shape[0] != array.shape[1]:
        raise ValueError("X must be a vector or square matrix.")

    size = array.shape[0]
    modes_float = np.log2(size)
    modes = int(round(modes_float))
    if 2**modes != size:
        raise ValueError("X dimension must be a power of 2.")

    if dim is None:
        dims = [2] * modes
    else:
        dims = list(dim)
        if dims != [2] * modes:
            raise ValueError("fermionic operations currently require dim == [2] * N.")

    return modes, dims


def _validate_modes(sys, modes):
    if isinstance(sys, (int, np.integer)):
        transpose_modes = [int(sys)]
    else:
        transpose_modes = [int(mode) for mode in sys]

    if len(set(transpose_modes)) != len(transpose_modes):
        raise ValueError("sys contains duplicate modes.")
    if any(mode < 0 or mode >= modes for mode in transpose_modes):
        raise ValueError("sys contains a mode outside the operator dimensions.")

    transpose_modes = sorted(transpose_modes)
    return transpose_modes


def _validate_leading_modes(sys, modes):
    transpose_modes = _validate_modes(sys, modes)
    _validate_leading_block(transpose_modes)
    return transpose_modes


def _validate_leading_block(transpose_modes):
    expected = list(range(len(transpose_modes)))
    if transpose_modes != expected:
        raise NotImplementedError(
            "This brute-force reference version requires sys to be the leading "
            "normal-ordered block [0, ..., len(sys)-1]. Fermionic mode "
            "permutation signs are not implemented yet."
        )


def _validate_method(method):
    if method not in {"bruteforce", "tensor", "sparse"}:
        raise ValueError("method must be 'bruteforce', 'tensor', or 'sparse'.")
    return method


def _validate_permutation(perm, modes):
    perm = [int(mode) for mode in perm]
    if sorted(perm) != list(range(modes)):
        raise ValueError("perm must be a permutation of all mode indices.")
    return perm


def _index_to_bits(index, modes):
    return tuple((index >> shift) & 1 for shift in range(modes - 1, -1, -1))


def _bit_parity(index):
    return int(index).bit_count() % 2


def _permute_bits_with_sign(bits, perm):
    permuted_bits = tuple(bits[mode] for mode in perm)
    sign = 1
    for left in range(len(perm)):
        if not permuted_bits[left]:
            continue
        for right in range(left + 1, len(perm)):
            if permuted_bits[right] and perm[left] > perm[right]:
                sign *= -1
    return permuted_bits, sign


def _inverse_permutation(perm):
    inverse = [0] * len(perm)
    for new_pos, old_pos in enumerate(perm):
        inverse[old_pos] = new_pos
    return inverse


def _fermionic_permutation_basis_action(perm, modes):
    size = 2**modes
    target_indices = np.empty(size, dtype=int)
    signs = np.empty(size, dtype=np.complex128)

    for old_index in range(size):
        old_bits = _index_to_bits(old_index, modes)
        new_bits, sign = _permute_bits_with_sign(old_bits, perm)
        target_indices[_bits_to_index(new_bits)] = old_index
        signs[_bits_to_index(new_bits)] = sign

    return target_indices, signs


def _bits_to_index(bits):
    index = 0
    for bit in bits:
        index = (index << 1) | int(bit)
    return index


__all__ = [
    "random_fermionic_density_matrix",
    "random_mixed_fermionic_state",
    "jordan_wigner_majoranas",
    "fermionic_gaussian_thermal_state",
    "majorana_monomial_coefficients",
    "gaussian_wick_four_point_violations",
    "fermionic_permute_systems",
    "fermionic_partial_trace",
    "fermionic_partial_transpose",
    "fermionic_negativity",
    "fermionic_log_negativity",
    "compute_fermionic_partial_transpose_matrix",
]
