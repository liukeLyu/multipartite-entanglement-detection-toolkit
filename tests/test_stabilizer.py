import numpy as np
import pytest

from medt import (
    gmn,
    keep_systems,
    log_negativity,
    negativity,
    stabilizer_dm,
    stabilizer_entangled,
    stabilizer_is_gme,
    stabilizer_log_negativity,
    stabilizer_negativity,
    stabilizer_restrict,
)
from medt.linalg import bipartitions
from medt.stabilizer import _MATRICES, _mul, _parse, _strings

pytestmark = pytest.mark.filterwarnings("ignore:Solution may be inaccurate:UserWarning")


# --- Building stabilizer states ---------------------------------------------------------------


def ghz(n):
    return ["X" * n] + ["I" * i + "ZZ" + "I" * (n - i - 2) for i in range(n - 1)]


def graph_state(n, edges):
    gens = []
    for v in range(n):
        g = ["I"] * n
        g[v] = "X"
        for a, b in edges:
            if v in (a, b):
                g[b if a == v else a] = "Z"
        gens.append("".join(g))
    return gens


def chain(n):
    return graph_state(n, [(i, i + 1) for i in range(n - 1)])


def ring(n):
    return graph_state(n, [(i, (i + 1) % n) for i in range(n)])


def grid(rows, cols):
    idx = lambda r, c: r * cols + c  # noqa: E731
    edges = [(idx(r, c), idx(r, c + 1)) for r in range(rows) for c in range(cols - 1)]
    edges += [(idx(r, c), idx(r + 1, c)) for r in range(rows - 1) for c in range(cols)]
    return graph_state(rows * cols, edges)


FIVE_QUBIT_CODE = ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]
STEANE_CODE = [
    *(row.replace("1", "X").replace("0", "I") for row in ("0001111", "0110011", "1010101")),
    *(row.replace("1", "Z").replace("0", "I") for row in ("0001111", "0110011", "1010101")),
]

NAMED_STATES = {
    "bell": ["XX", "ZZ"],
    "bell_minus": ["-XX", "ZZ"],
    "two_bells": ["XXII", "ZZII", "IIXX", "IIZZ"],
    **{f"ghz{n}": ghz(n) for n in range(3, 7)},
    "chain5": chain(5),
    "ring6": ring(6),
    "grid2x3": grid(2, 3),
    "five_qubit_codeword": FIVE_QUBIT_CODE + ["ZZZZZ"],
    "five_qubit_code_space": FIVE_QUBIT_CODE,
    "steane_codeword": STEANE_CODE + ["ZZZZZZZ"],
    "steane_code_space": STEANE_CODE,
    "y_state": ["YY", "ZZ"],
}


def pauli_matrix(g):
    x, z, e = _parse([g])
    M = np.ones((1, 1), dtype=complex)
    for a, b in zip(x[0], z[0], strict=True):
        M = np.kron(M, _MATRICES[{(0, 0): "I", (1, 0): "X", (1, 1): "Y", (0, 1): "Z"}[(a, b)]])
    return 1j ** e[0] * M


def product(gens, combo):
    x, z, e = _parse(gens)
    n = x.shape[1]
    px, pz, pe = np.zeros(n, dtype=np.uint8), np.zeros(n, dtype=np.uint8), 0
    for i in np.flatnonzero(combo):
        px, pz, pe = _mul(px, pz, pe, x[i], z[i], e[i])
    return _strings(px[None], pz[None], np.array([pe]))[0]


def single(n, q, letter):
    return "I" * q + letter + "I" * (n - q - 1)


def gate_images(n, gate, qubits):
    """Images ``U X_q U^dag`` and ``U Z_q U^dag`` of all single-qubit Paulis under a Clifford gate."""
    imgs = {(q, p): single(n, q, p) for q in range(n) for p in "XZ"}
    if gate == "H":
        (q,) = qubits
        imgs[q, "X"], imgs[q, "Z"] = single(n, q, "Z"), single(n, q, "X")
    elif gate == "S":
        (q,) = qubits
        imgs[q, "X"] = single(n, q, "Y")
    elif gate == "CNOT":
        c, t = qubits
        imgs[c, "X"] = product([single(n, c, "X"), single(n, t, "X")], [1, 1])
        imgs[t, "Z"] = product([single(n, c, "Z"), single(n, t, "Z")], [1, 1])
    return imgs


def conjugate(gens, gate, qubits):
    x, z, e = _parse(gens)
    n = x.shape[1]
    imgs = {k: _parse([v]) for k, v in gate_images(n, gate, qubits).items()}
    out = []
    for xi, zi, ei in zip(x, z, e, strict=True):
        # P = i^e prod_q i^{x_q z_q} X_q^{x_q} Z_q^{z_q}
        px, pz, pe = np.zeros(n, dtype=np.uint8), np.zeros(n, dtype=np.uint8), int(ei + (xi & zi).sum())
        for q in range(n):
            for p, bit in (("X", xi[q]), ("Z", zi[q])):
                if bit:
                    ix, iz, ie = imgs[q, p]
                    px, pz, pe = _mul(px, pz, pe, ix[0], iz[0], ie[0])
        out.append(_strings(px[None], pz[None], np.array([pe]))[0])
    return out


def random_clifford_state(n, rng, depth=None):
    gens = [("-" if rng.random() < 0.5 else "") + single(n, q, "Z") for q in range(n)]
    for _ in range(depth or 4 * n):
        gate = rng.choice(["H", "S", "CNOT"] if n > 1 else ["H", "S"])
        qubits = rng.choice(n, size=2 if gate == "CNOT" else 1, replace=False)
        gens = conjugate(gens, gate, [int(q) for q in qubits])
    return gens


def random_invertible_f2(k, rng):
    from medt.stabilizer import _rank_f2

    while True:
        C = rng.integers(0, 2, size=(k, k))
        if _rank_f2(C) == k:
            return C


def regenerate(gens, rng):
    """Same stabilizer group, new generating set of random products."""
    C = random_invertible_f2(len(gens), rng)
    return [product(gens, row) for row in C]


def all_cuts(n):
    return [list(M) for M in bipartitions(n)]


def assert_matches_dense(gens):
    n = len(gens[0].lstrip("+-"))
    rho = stabilizer_dm(gens)
    for sys in all_cuts(n):
        assert stabilizer_log_negativity(gens, sys) == pytest.approx(log_negativity(rho, sys, [2] * n), abs=1e-9)


# --- Dense matrix -----------------------------------------------------------------------------


@pytest.mark.parametrize("name", NAMED_STATES)
def test_dense_matrix(name):
    gens = NAMED_STATES[name]
    n, k = len(gens[0].lstrip("+-")), len(gens)
    rho = stabilizer_dm(gens)
    assert np.trace(rho) == pytest.approx(1)
    assert np.allclose(rho, rho.conj().T)
    assert np.allclose(rho @ rho, rho) == (k == n)
    assert np.trace(rho @ rho).real == pytest.approx(2.0 ** (k - n))
    for g in gens:
        assert np.trace(pauli_matrix(g) @ rho) == pytest.approx(1)


def test_qubit_order_matches_kron():
    rho = stabilizer_dm(["ZI", "IX"])  # |0> (x) |+>
    plus = np.array([1, 1]) / np.sqrt(2)
    psi = np.kron([1, 0], plus)
    assert np.allclose(rho, np.outer(psi, psi))


def cnot_matrix(n, c, t):
    idx = np.arange(2**n)
    flipped = np.where(idx >> (n - 1 - c) & 1, idx ^ (1 << (n - 1 - t)), idx)
    return np.eye(2**n)[flipped]


def test_clifford_helper_matches_dense():
    rng = np.random.default_rng(0)
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    I = np.eye(2)
    gates = {
        ("H", (1,)): np.kron(np.kron(I, H), I),
        ("S", (2,)): np.kron(np.kron(I, I), np.diag([1, 1j])),
        ("CNOT", (0, 2)): cnot_matrix(3, 0, 2),
        ("CNOT", (2, 1)): cnot_matrix(3, 2, 1),
    }
    for _ in range(20):
        g = random_clifford_state(3, rng, depth=5)[0]
        for (name, qubits), U in gates.items():
            assert np.allclose(pauli_matrix(conjugate([g], name, qubits)[0]), U @ pauli_matrix(g) @ U.conj().T)


# --- Negativity against dense numerics ------------------------------------------------------


@pytest.mark.parametrize("name", NAMED_STATES)
def test_log_negativity_named_states(name):
    assert_matches_dense(NAMED_STATES[name])


def test_negativity_matches_medt_negativity():
    for gens in (ghz(4), FIVE_QUBIT_CODE, chain(4)):
        rho = stabilizer_dm(gens)
        for sys in all_cuts(len(gens[0])):
            assert stabilizer_negativity(gens, sys) == pytest.approx(negativity(rho, sys), abs=1e-9)


def test_known_values():
    assert stabilizer_log_negativity(ghz(5), [0, 1]) == 1
    assert stabilizer_log_negativity(NAMED_STATES["two_bells"], [0, 2]) == 2
    assert stabilizer_log_negativity(NAMED_STATES["five_qubit_codeword"], [0, 1]) == 2
    assert stabilizer_log_negativity(FIVE_QUBIT_CODE, [0, 1]) == 2


@pytest.mark.parametrize("n", range(1, 8))
def test_random_pure_states(n):
    rng = np.random.default_rng(n)
    for _ in range(4 if n < 7 else 2):
        gens = random_clifford_state(n, rng)
        if n > 1:
            assert_matches_dense(gens)
        assert np.allclose(stabilizer_dm(gens) @ stabilizer_dm(gens), stabilizer_dm(gens))


@pytest.mark.parametrize("n", range(2, 8))
def test_random_mixed_states(n):
    rng = np.random.default_rng(100 + n)
    for _ in range(4 if n < 7 else 2):
        gens = random_clifford_state(n, rng)
        k = int(rng.integers(1, n))
        gens = regenerate([gens[i] for i in rng.choice(n, size=k, replace=False)], rng)
        assert_matches_dense(gens)


@pytest.mark.parametrize("n", range(3, 8))
def test_restrict_matches_partial_trace(n):
    rng = np.random.default_rng(200 + n)
    for trial in range(6):
        gens = random_clifford_state(n, rng)
        if trial % 2:  # mixed parent state
            gens = [gens[i] for i in rng.choice(n, size=max(1, n - 1 - trial % 3), replace=False)]
        size = int(rng.integers(1, n))
        keep = [int(q) for q in rng.choice(n, size=size, replace=False)]
        reduced = stabilizer_restrict(gens, keep)
        dense = keep_systems(stabilizer_dm(gens), keep, [2] * n)
        assert np.allclose(stabilizer_dm(reduced, n=size), dense, atol=1e-12)
        if reduced and size > 1:
            assert_matches_dense(reduced)


def test_restrict_examples():
    assert stabilizer_restrict(ghz(3), [0, 1]) == ["ZZ"]
    assert stabilizer_restrict(["XX", "ZZ"], [1]) == []
    assert stabilizer_restrict(["XXX", "ZZI", "-IZZ"], [2, 0]) == ["-ZZ"]


# --- Invariance ---------------------------------------------------------------------------------


def test_invariance():
    rng = np.random.default_rng(7)
    for n in (4, 5, 6):
        gens = random_clifford_state(n, rng)
        gens = gens[: n - (n % 2)]  # pure for even n, mixed for n = 5
        reference = [stabilizer_log_negativity(gens, sys) for sys in all_cuts(n)]
        new = regenerate(gens, rng)
        assert [stabilizer_log_negativity(new, sys) for sys in all_cuts(n)] == reference
        flipped = [g.lstrip("-") if g.startswith("-") else "-" + g for g in gens]
        assert [stabilizer_log_negativity(flipped, sys) for sys in all_cuts(n)] == reference
        perm = rng.permutation(n)  # new qubit perm[q] is old qubit q
        relabelled = []
        for g in gens:
            sign, body = ("-", g[1:]) if g[0] == "-" else ("", g)
            new_body = [""] * n
            for q in range(n):
                new_body[perm[q]] = body[q]
            relabelled.append(sign + "".join(new_body))
        for sys, value in zip(all_cuts(n), reference, strict=True):
            assert stabilizer_log_negativity(relabelled, [int(perm[q]) for q in sys]) == value


# --- Toric code -----------------------------------------------------------------------------------


class ToricCode:
    """Toric code on an L x L periodic square lattice, one qubit per edge.

    Edge ``h(i, j)`` joins vertices (i, j) and (i, j+1); ``v(i, j)`` joins (i, j) and (i+1, j).
    """

    def __init__(self, L):
        self.L, self.n = L, 2 * L * L

    def h(self, i, j):
        return 2 * ((i % self.L) * self.L + j % self.L)

    def v(self, i, j):
        return 2 * ((i % self.L) * self.L + j % self.L) + 1

    def star(self, i, j):
        return [self.h(i, j), self.h(i, j - 1), self.v(i, j), self.v(i - 1, j)]

    def plaquette(self, i, j):
        return [self.h(i, j), self.h(i + 1, j), self.v(i, j), self.v(i, j + 1)]

    def op(self, letter, edges):
        g = ["I"] * self.n
        for e in edges:
            g[e] = letter
        return "".join(g)

    def generators(self):
        L = self.L
        cells = [(i, j) for i in range(L) for j in range(L)]
        gens = [self.op("Z", self.star(*c)) for c in cells[1:]]
        gens += [self.op("X", self.plaquette(*c)) for c in cells[1:]]
        gens += [self.op("Z", [self.v(0, j) for j in range(L)]), self.op("Z", [self.h(i, 0) for i in range(L)])]
        return gens

    def ket(self):
        """``prod_p (1 + B_p) |0...0>``, normalized; qubit 0 is the most significant bit."""
        psi = np.zeros(2**self.n)
        psi[0] = 1
        idx = np.arange(2**self.n)
        for i in range(self.L):
            for j in range(self.L):
                mask = sum(1 << (self.n - 1 - e) for e in self.plaquette(i, j))
                psi = psi + psi[idx ^ mask]
        return psi / np.linalg.norm(psi)


@pytest.fixture(scope="module")
def toric():
    tc = ToricCode(3)
    return tc, tc.generators(), tc.ket()


def test_toric_generators_match_ket(toric):
    tc, gens, psi = toric
    assert len(gens) == tc.n
    for region in ([0, 1, 2, 3], tc.star(1, 1) + [tc.h(1, 1) + 2]):
        reduced = stabilizer_restrict(gens, region)
        assert np.allclose(stabilizer_dm(reduced, n=len(region)), keep_systems(psi, region, [2] * tc.n), atol=1e-12)


def regions(tc):
    star = tc.star(1, 1)  # edges right, left, up, down of vertex (1, 1)
    ne, sw = tc.plaquette(1, 1), tc.plaquette(0, 0)
    return {
        "tree": star + [tc.h(1, 2), tc.v(2, 1)],
        "loop": ne,
        "star_plaquette": sorted(set(star) | set(ne)),
        "star_two_plaquettes": sorted(set(star) | set(ne) | set(sw)),
    }


def region_state(toric, name):
    tc, gens, psi = toric
    region = regions(tc)[name]
    reduced = stabilizer_restrict(gens, region)
    dense = keep_systems(psi, region, [2] * tc.n)
    assert np.allclose(stabilizer_dm(reduced, n=len(region)), dense, atol=1e-12)
    return region, reduced, dense


@pytest.mark.parametrize("name", ["tree", "loop"])
def test_toric_separable_regions(toric, name):
    region, reduced, dense = region_state(toric, name)
    for sys in all_cuts(len(region)):
        assert not stabilizer_entangled(reduced, sys)
        assert log_negativity(dense, sys, [2] * len(region)) == pytest.approx(0, abs=1e-9)


def test_toric_star_and_plaquette(toric):
    tc, _, _ = toric
    region, reduced, dense = region_state(toric, "star_plaquette")
    shared = [region.index(e) for e in sorted(set(tc.star(1, 1)) & set(tc.plaquette(1, 1)))]
    for sys in all_cuts(len(region)):
        separates = (shared[0] in sys) != (shared[1] in sys)
        assert stabilizer_entangled(reduced, sys) == separates
        assert stabilizer_log_negativity(reduced, sys) == (1 if separates else 0)
        assert log_negativity(dense, sys, [2] * len(region)) == pytest.approx(1 if separates else 0, abs=1e-9)


def set_partitions(items):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for part in set_partitions(rest):
        for i in range(len(part)):
            yield part[:i] + [[first] + part[i]] + part[i + 1 :]
        yield [[first]] + part


def test_toric_star_and_two_plaquettes(toric):
    region, reduced, dense = region_state(toric, "star_two_plaquettes")
    m = len(region)
    assert m == 8
    for sys in all_cuts(m):
        assert stabilizer_log_negativity(reduced, sys) == pytest.approx(log_negativity(dense, sys, [2] * m), abs=1e-9)
    gme = {}
    for parts in set_partitions(list(range(m))):
        if len(parts) >= 2:
            gme.setdefault(len(parts), []).append(stabilizer_is_gme(reduced, parts))
    assert len(gme[3]) == 966
    assert sum(gme[3]) == 324
    assert any(gme[2]) and not all(gme[2])
    assert not any(any(gme[p]) for p in gme if p >= 4)


# --- GME against the GMN SDP ---------------------------------------------------------------------

GME_CASES = {
    "ghz3": (ghz(3), True),
    "chain4": (chain(4), True),
    "ghz3_mixed": (["ZZI", "IZZ"], False),
    "ghz4_mixed": (["XXXX", "ZZII", "IIZZ"], False),
    "chain4_mixed": (chain(4)[:3], False),
    "mixed4_gme": (["-IZXX", "XXYX", "-ZXIZ"], True),
    "two_bells": (NAMED_STATES["two_bells"], False),
}


@pytest.mark.parametrize("name", GME_CASES)
def test_gme_matches_gmn(name):
    gens, expected = GME_CASES[name]
    assert stabilizer_is_gme(gens) == expected
    assert (gmn(stabilizer_dm(gens)) > 1e-6) == expected


@pytest.mark.parametrize("gens", [ghz(3), chain(4), random_clifford_state(4, np.random.default_rng(3))])
def test_pure_gmn_equals_min_cut_negativity(gens):
    # Renormalized GMN of a pure state is the minimum bipartite negativity (Hofmann et al. 2014, Lemma 3 (vii)).
    n = len(gens)
    expected = min(stabilizer_negativity(gens, sys) for sys in all_cuts(n))
    assert gmn(stabilizer_dm(gens)) == pytest.approx(expected, abs=1e-6)


@pytest.mark.slow
@pytest.mark.parametrize("gens", [ghz(5), chain(5), ring(5)[:4], grid(2, 3)[:5]])
def test_gme_matches_gmn_slow(gens):
    assert (gmn(stabilizer_dm(gens)) > 1e-6) == stabilizer_is_gme(gens)


def test_gme_with_grouped_parties():
    gens = NAMED_STATES["two_bells"]
    assert stabilizer_is_gme(gens, [[0, 2], [1, 3]])
    assert not stabilizer_is_gme(gens, [[0, 1], [2], [3]])


# --- Errors ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "gens",
    [["XI", "ZI"], ["XX", "XX"], ["XX", "ZZ", "YY"], ["XQ"], ["XX", "Z"], ["II"]],
)
def test_invalid_generators(gens):
    with pytest.raises(ValueError):
        stabilizer_dm(gens)
    with pytest.raises(ValueError):
        stabilizer_log_negativity(gens, [0])


def test_invalid_parties():
    with pytest.raises(ValueError):
        stabilizer_is_gme(ghz(3), [[0, 1], [1, 2]])
    with pytest.raises(ValueError):
        stabilizer_is_gme(ghz(3), [[0, 1, 2]])
