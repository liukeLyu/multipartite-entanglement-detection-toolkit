# Multipartite Entanglement Detection Toolkit (MEDT)

Python tools to certify and quantify entanglement in multipartite quantum states, from small benchmark states to reduced density matrices of many-body systems.

```python
import numpy as np
import medt

ghz = np.zeros((8, 8)); ghz[np.ix_([0, 7], [0, 7])] = 0.5
rho = 0.6 * ghz + 0.4 * np.eye(8) / 8

medt.gmn(rho)                     # genuine multiparticle negativity, > 0 certifies GME
medt.gmc_x_state(rho)             # exact genuine multipartite concurrence (X states)
medt.polytope_biseparable(rho, [2, 2, 2]).certified   # False: the state is GME
```

## Methods

| Task | Function | Reference |
|---|---|---|
| Genuine multiparticle negativity (renormalized PPT mixer) | `gmn` | Jungnitsch, Moroder, Gühne, PRL 106, 190502 (2011); Hofmann, Moroder, Gühne, J. Phys. A 47, 155301 (2014) |
| Fermionic GMN, fermionic partial transpose | `fermionic_gmn`, `medt.fermions` | Allen, Lyu, Witczak-Krempa, arXiv:2607.20707; Shapourian, Shiozaki, Ryu, PRB 95, 165101 (2017) |
| Genuine multipartite Rains entanglement | `gmre` | Murray, Bhattacharya, Cerezo, Lyu, Wilde, IEEE ISIT 2026, arXiv:2601.09590 |
| Genuine multipartite concurrence: pure states, exact for X states, I2 lower bound, convex-roof upper bound | `gmc_pure`, `gmc_x_state`, `gmc_lower_bound`, `gmc_convex_roof` | Ma et al., PRA 83, 062325 (2011); Hashemi Rafsanjani et al., PRA 86, 062303 (2012); Lyu, Song, Wang, Meng, Witczak-Krempa, arXiv:2412.12533 |
| Negativity, log-negativity, PPT, CCNR/realignment | `negativity`, `log_negativity`, `is_ppt`, `ccnr` | Peres; Horodecki; Rudolph; Chen and Wu |
| Separability certificate by iterative purity reduction | `kampermann_sep` | Kampermann, Gühne, Wilmott, Bruß, PRA 86, 032307 (2012) |
| Adaptive polytopes: separable, fully separable, biseparable | `polytope_*` | Ohst, Yu, Gühne, Sarbicki, arXiv:2210.10054 |
| Genuine network multipartite entanglement: ring inflation at levels 2 (symmetry reduced) and 3, Gilbert criterion | `ring_inflation_robustness`, `gilbert_criterion` | Navascués et al., PRL 125, 240505 (2020); Lyu et al., PRL 137, 110204 (2026), arXiv:2512.11118 |

Benchmark states live in `medt.states` (GHZ, W, Dicke, Werner, isotropic), and `medt.tfim.tfim_rdm(h, sites, beta)` gives exact reduced states of the infinite transverse-field Ising chain for any set of sites at zero or finite temperature.

## Solvers

SDPs go through [CVXPY](https://www.cvxpy.org), so any installed solver works (`solver="MOSEK"`, `"CLARABEL"`, `"SCS"`). GMN restricts the SDP to real variables for real states and to parity sectors for states commuting with `Z...Z`, which keeps 6-qubit problems within reach of the interior-point solver Clarabel. Larger problems default to SCS (`eps=1e-8`), which uses far less memory but can converge slowly for nearly pure states. GMRE uses the QICS solver for the quantum relative entropy cone (`pip install "medt[rains]"`).

## Install and test

```bash
pip install -e ".[dev]"
pytest              # fast suite
pytest -m slow      # includes a 6-qubit GMN check against published values
```

## Validation

Tests check each method against analytic results (GHZ and W states, noisy-GHZ thresholds 1/5, 3/7, Bell-state robustness, Horodecki's PPT entangled state) and against published numerics: GMN values for reduced states of the Kitaev honeycomb model from L. Lyu, D. Chandorkar, S. Kapoor, S. Takei, E. S. Sørensen, W. Witczak-Krempa, "Multiparty entanglement loops in quantum spin liquids", arXiv:2505.18124, and, from Lyu et al., PRL 137, 110204 (2026), 6-spin GMN of the transverse-field Ising chain and ring-inflation robustness of GHZ and W states (level 2 against the paper's MOSEK code, level 3 against Table I). The TFIM reduced density matrices are checked against exact diagonalization.

## Citing

If you use MEDT, please cite the papers for the methods you use (table above).
