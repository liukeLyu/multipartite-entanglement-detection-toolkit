"""Fermionic modes: signed permutations, partial trace, partial transpose.

States of ``N`` fermionic modes are written in the occupation basis
``|n_1 ... n_N>`` with mode 1 most significant, as for qubits. Unlike qubits,
reordering modes picks up exchange signs, and the partial transpose is the
fermionic one of Shapourian, Shiozaki and Ryu [1, 2], which respects the
parity superselection rule.

[1] H. Shapourian, K. Shiozaki, S. Ryu, PRB 95, 165101 (2017), arXiv:1611.07536.
[2] H. Shapourian, S. Ryu, PRA 99, 022310 (2019), arXiv:1804.08637.
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
    """Trace out the fermionic modes ``trace_sys``; the rest keep their original order.

    ``X`` is a ket or density matrix, dense or SciPy sparse. See
    :func:`fermionic_keep_systems` for the sign convention.
    """
    size = X.shape[0] if X.shape[0] > 1 else X.shape[-1]
    modes = int(round(np.log2(size)))
    if dim is not None and (len(dim) != modes or any(int(d) != 2 for d in dim)):
        raise ValueError("fermionic modes have dimension 2")
    traced = _validate_modes(trace_sys, modes)
    return fermionic_keep_systems(X, [m for m in range(modes) if m not in set(traced)], modes)


def _move_to_front_signs(index, keep, traced, modes):
    """Sign of moving the occupied ``keep`` modes ahead of the ``traced`` ones, keeping relative order."""
    keep = set(keep)
    seen = np.zeros_like(index)  # parity of occupied traced modes passed so far
    parity = np.zeros_like(index)
    for m in range(modes):
        bit = (index >> (modes - 1 - m)) & 1
        if m in keep:
            parity ^= bit & seen
        else:
            seen ^= bit
    return 1 - 2 * parity


def fermionic_keep_systems(state, keep, n_modes=None):
    """Reduced density matrix of the fermionic modes ``keep``, in the order given.

    ``state`` is a ket or density matrix in the occupation basis of ``n_modes``
    modes, with the Jordan-Wigner ordering of :mod:`medt.fermions`; a ket can
    be a SciPy sparse array. Each basis amplitude is multiplied by the sign of
    moving the kept modes in front of the traced ones, after which the reduced
    state is an ordinary partial trace (:func:`medt.linalg.keep_systems`). The
    signs are computed in a vectorized pass over the nonzero amplitudes, so a
    ket of 24 modes is handled without forming a density matrix. If ``keep`` is
    not increasing, the kept modes are then reordered with
    :func:`fermionic_permute_systems`.

    The result is defined by ``Tr(rho_A O) = Tr(rho O')`` for every operator
    ``O`` built from the Majorana operators of the kept modes, where ``O'`` is
    the same product of the corresponding Majoranas of the full system.
    """
    order = [int(k) for k in keep]
    if len(set(order)) != len(order):
        raise ValueError("keep has repeated modes")
    rdm = _fermionic_reduced_sorted(state, order, n_modes)
    ranked = sorted(order)
    if order == ranked:
        return rdm
    return fermionic_permute_systems(rdm, [ranked.index(k) for k in order])


def _fermionic_reduced_sorted(state, keep, n_modes):
    from scipy import sparse

    from medt.linalg import keep_systems

    size = state.shape[0] if state.shape[0] > 1 else state.shape[-1]
    modes = int(round(np.log2(size))) if n_modes is None else int(n_modes)
    if 2**modes != size:
        raise ValueError("state dimension must be 2**n_modes")
    keep = sorted(int(k) for k in keep)
    traced = [m for m in range(modes) if m not in keep]
    dims = [2] * modes
    if sparse.issparse(state):
        X = sparse.coo_array(state)
        is_ket = 1 in X.shape or X.ndim == 1
        if not is_ket:
            sr = _move_to_front_signs(X.row.astype(np.int64), keep, traced, modes)
            sc = _move_to_front_signs(X.col.astype(np.int64), keep, traced, modes)
            X = sparse.coo_array((X.data * sr * sc, (X.row, X.col)), shape=X.shape)
            return keep_systems(X, keep, dims)
        idx = (X.coords[0] if X.ndim == 1 else (X.row if X.shape[1] == 1 else X.col)).astype(np.int64)
        signed = sparse.coo_array((X.data * _move_to_front_signs(idx, keep, traced, modes), (idx,)), shape=(size,))
        return keep_systems(signed, keep, dims)
    state = np.asarray(state)
    sign = _move_to_front_signs(np.arange(size, dtype=np.int64), keep, traced, modes)
    if state.ndim == 1 or state.shape[1] == 1:
        return keep_systems(sign * state.reshape(-1), keep, dims)
    return keep_systems(sign[:, None] * state * sign[None, :], keep, dims)


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
    """Index map and exchange signs of a fermionic mode permutation, vectorized over the basis."""
    new_index = np.arange(2**modes, dtype=np.int64)
    # Bit of new position p (mode 0 is the most significant bit).
    new_bits = [(new_index >> (modes - 1 - p)) & 1 for p in range(modes)]
    old_index = np.zeros_like(new_index)
    for p, old_mode in enumerate(perm):
        old_index |= new_bits[p] << (modes - 1 - old_mode)
    parity = np.zeros_like(new_index)
    for left in range(modes):
        for right in range(left + 1, modes):
            if perm[left] > perm[right]:
                parity ^= new_bits[left] & new_bits[right]
    return old_index, (1 - 2 * parity).astype(np.complex128)


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
    "fermionic_keep_systems",
    "fermionic_partial_transpose",
    "fermionic_negativity",
    "fermionic_log_negativity",
    "compute_fermionic_partial_transpose_matrix",
]
