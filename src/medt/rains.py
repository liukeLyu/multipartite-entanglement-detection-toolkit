"""Genuine multipartite Rains entanglement (GMRE).

Murray, Bhattacharya, Cerezo, Lyu and Wilde [1] define

    GMRE(rho) = min  D(rho || sum_M sigma_M)
                s.t. sigma_M >= 0,  sum_M ||sigma_M^{T_M}||_1 <= 1,

over one positive operator per bipartition ``M``, with ``D`` the quantum
relative entropy in bits. It is a GME monotone under selective PPT
operations and reduces to the Rains relative entropy for two parties. The
relative entropy cone is handled natively by the QICS solver [2] through
PICOS, which are optional dependencies (``pip install medt[rains]``).

[1] H. S. Murray, S. Bhattacharya, M. Cerezo, L. Lyu, M. M. Wilde,
    "Genuine multipartite Rains entanglement", arXiv:2601.09590.
[2] K. He, J. Saunderson, H. Fawzi, "QICS: Quantum Information Conic Solver",
    arXiv:2410.17803.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from medt.gmn import _validate
from medt.linalg import bipartitions


def gmre(rho: np.ndarray, dims: Sequence[int] | None = None, *, solver: str = "qics", **solver_options) -> float:
    """Genuine multipartite Rains entanglement of ``rho`` in bits."""
    try:
        import picos
    except ImportError as err:
        raise ImportError("gmre needs PICOS and QICS: pip install 'medt[rains]'") from err

    rho, dims = _validate(rho, dims)
    D = rho.shape[0]
    prob = picos.Problem()
    tau, norm_bound = 0, 0
    for k, M in enumerate(bipartitions(len(dims))):
        sigma = picos.HermitianVariable(f"sigma{k}", D)
        # ||sigma^{T_M}||_1 <= tr(P + N) with sigma^{T_M} = P - N, P, N >= 0.
        P = picos.HermitianVariable(f"P{k}", D)
        N = picos.HermitianVariable(f"N{k}", D)
        prob.add_list_of_constraints([sigma >> 0, P >> 0, N >> 0])
        prob.add_constraint(picos.partial_transpose(sigma, list(M), list(dims)) == P - N)
        norm_bound += picos.trace(P + N).real
        tau += sigma
    prob.add_constraint(norm_bound <= 1)
    prob.set_objective("min", picos.quantrelentr(picos.Constant(rho), tau))
    prob.solve(solver=solver, **solver_options)
    return float(prob.value / np.log(2))
