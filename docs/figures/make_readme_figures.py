"""Make the two figures shown in README.md (about a minute on a laptop).

    pip install ".[examples]"
    python docs/figures/make_readme_figures.py
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

import medt
from medt.states import ghz, ket_to_dm, w, white_noise

OUT = Path(__file__).resolve().parent
INK, MUTED, GRID = "#52514e", "#8a8984", "#e6e5e1"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "font.size": 10,
})
warnings.filterwarnings("ignore", message="Solution may be inaccurate")  # SCS notes near GMN = 0


def noisy_ghz():
    ps = np.linspace(0, 0.8, 41)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for (name, psi), color in zip([("GHZ", ghz(3)), ("W", w(3))], COLORS, strict=False):
        rho = ket_to_dm(psi)
        vals = np.array([max(medt.gmn(white_noise(rho, p)), 0.0) for p in ps])
        ax.plot(ps, vals, color=color, lw=2, marker="o", ms=3.5, label=f"{name} state")
        p_dead = ps[np.argmax(vals < 1e-6)]
        print(f"{name}: GMN(p=0) = {vals[0]:.4f}, zero from p = {p_dead:.2f}")
    ax.axvline(4 / 7, color=MUTED, ls="--", lw=1)
    ax.text(4 / 7 + 0.01, 0.42, "GHZ: p = 4/7", color=MUTED)
    ax.set(xlabel="white-noise fraction p", ylabel="GMN", ylim=(-0.02, 0.52),
           title=r"GMN of $\rho = (1-p)\,|\psi\rangle\langle\psi| + p\,\mathbb{1}/8$")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "gmn_vs_noise.png", dpi=150)


def draw_chain(ax):
    """Infinite Ising chain with three adjacent spins A, B, C highlighted."""
    ax.set(xlim=(-0.6, 10.6), ylim=(-1.0, 1.05), aspect="equal")
    ax.axis("off")
    xs = np.arange(1, 10)
    highlight = {4: "A", 5: "B", 6: "C"}
    ax.add_patch(FancyBboxPatch((3.45, -0.62), 3.1, 1.24, boxstyle="round,pad=0.02,rounding_size=0.25",
                                fc="#eaf2fc", ec=COLORS[0], lw=1.5, zorder=0))
    ax.plot([xs[0], xs[-1]], [0, 0], color=MUTED, lw=1.5, zorder=1)
    for side in (0.05, 9.95):
        ax.text(side, 0, "···", ha="center", va="center", color=MUTED, fontsize=14)
    rng = np.random.default_rng(3)
    for x in xs:
        hot = x in highlight
        ax.add_patch(Circle((x, 0), 0.3, fc=COLORS[0] if hot else "white", ec=COLORS[0] if hot else MUTED,
                            lw=1.5, zorder=2))
        tilt = rng.uniform(-0.35, 0.35)  # spins point along the field, smeared by quantum and thermal fluctuations
        ax.add_patch(FancyArrowPatch((x - 0.22 * np.sin(tilt), -0.22 * np.cos(tilt)),
                                     (x + 0.22 * np.sin(tilt), 0.22 * np.cos(tilt)),
                                     arrowstyle="-|>", mutation_scale=9, lw=1.3,
                                     color="white" if hot else INK, zorder=3))
        if hot:
            ax.text(x, -0.85, highlight[x], ha="center", va="center", color=COLORS[0], fontsize=11, weight="bold")
    ax.annotate("", xy=(2, -0.5), xytext=(3, -0.5),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1, shrinkA=0, shrinkB=0))
    ax.text(2.5, -0.85, r"$-X_iX_{i+1}$", ha="center", va="center", color=INK, fontsize=10)
    ax.text(8, -0.85, r"$-h\,Z_i$ on each spin", ha="center", va="center", color=INK, fontsize=10)
    ax.text(5, 0.9, r"$\rho_{ABC} = \mathrm{Tr}_{\mathrm{rest}}\, e^{-H/T}/Z$   $\rightarrow$   GMN$(\rho_{ABC})$",
            ha="center", color=INK, fontsize=11)


def thermal_tfim():
    temps = np.linspace(0.02, 2.2, 45)
    fields = (0.5, 1.0, 1.5, 3.0)
    fig, (ax0, ax) = plt.subplots(2, 1, figsize=(6.4, 5.6), gridspec_kw={"height_ratios": [1, 1.7]})
    draw_chain(ax0)
    for h, color in zip(fields, COLORS, strict=True):
        vals = np.array([max(medt.gmn(medt.tfim.tfim_rdm(h, [0, 1, 2], beta=1 / T)), 0.0) for T in temps])
        ax.plot(temps, vals, color=color, lw=2, marker="o", ms=3.5, label=f"h = {h}")
        print(f"h = {h}: GMN(T->0) = {vals[0]:.4f}, zero from T = {temps[np.argmax(vals < 1e-6)]:.2f}")
    ax.set(xlabel="temperature T", ylabel="GMN of A, B, C")
    ax.legend(frameon=False, title="transverse field", title_fontsize=9, alignment="left")
    fig.tight_layout()
    fig.savefig(OUT / "gmn_vs_temperature.png", dpi=150)


if __name__ == "__main__":
    noisy_ghz()
    thermal_tfim()
