"""Phase-1 validation: sprinkling + causal order + Brightwell--Gregory geodesic law.

Reproduce with:
    python experiments/exp00_phase1_validation.py

Produces:
    * a printed table (with standard errors over realisations), and
    * figures/exp00_brightwell_gregory.png

Physics being validated
------------------------
For Poisson sprinklings into a 1+1 D causal diamond the causal order is the 2-D
random (dominance) order. Its longest chain ``L`` -- the discrete timelike
geodesic length -- satisfies the Brightwell--Gregory law (PRL 66, 260 (1991)):

    Panel A:  E[L] / sqrt(N)  ->  2   as N grows   (Vershik--Kerov / Logan--Shepp
              constant for Ulam's problem; approach is from below ~ 2 - c N^{-1/3}).
    Panel B:  at fixed density, L is proportional to proper time tau, slope
              sqrt(2 rho), because N = rho tau^2 / 2.

Everything is regenerable from the fixed seeds recorded below (Integrity Rule 1).
No data is smoothed or interpolated (Rule 1); error bars are standard errors over
independent realisations (Rule 3).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from causet import order, sprinkle  # noqa: E402

# ---- Documented parameters (Integrity Rule 4) -----------------------------
VKLS_CONSTANT = 2.0
SEED_PANEL_A = 20260714
SEED_PANEL_B = 70418202
# Realisation counts and density ladder chosen so the whole script runs in a
# couple of minutes on a laptop while still spanning ~16x in N (the tests cover
# the physics rigorously; this script produces the illustrative figure). The
# top density gives N ~= rho*tau^2/2 = 4000 elements per realisation.
N_REAL_A = 40
N_REAL_B = 40
TAU_A = 2.0
RHO_B = 1000.0
DENSITIES_A = [125.0, 250.0, 500.0, 1000.0, 2000.0]
TAUS_B = [1.0, 1.5, 2.0, 2.5, 3.0]


def measure(rho: float, tau: float, n_real: int, seed0: int):
    """Return arrays of (L, N) over ``n_real`` independent sprinklings."""
    lengths = np.empty(n_real)
    counts = np.empty(n_real)
    for k in range(n_real):
        s = sprinkle.sprinkle_diamond_1d(rho, tau, seed=seed0 + k, include_endpoints=True)
        c = order.causal_matrix_1d(s.u, s.v)
        lengths[k] = order.longest_chain_length(c)
        counts[k] = s.n
    return lengths, counts


def panel_a():
    print("\n=== Panel A: E[L]/sqrt(N) -> 2 (Brightwell--Gregory constant) ===")
    print(f"{'rho':>8} {'<N>':>8} {'<L>':>9} {'SE(L)':>7} {'<L>/sqrt(<N>)':>14}")
    xs, ys, yerr = [], [], []
    for rho in DENSITIES_A:
        L, N = measure(rho, TAU_A, N_REAL_A, SEED_PANEL_A)
        mean_n = N.mean()
        mean_l = L.mean()
        se_l = L.std(ddof=1) / np.sqrt(len(L))
        ratio = mean_l / np.sqrt(mean_n)
        print(f"{rho:>8.0f} {mean_n:>8.1f} {mean_l:>9.2f} {se_l:>7.2f} {ratio:>14.4f}")
        xs.append(mean_n)
        ys.append(ratio)
        yerr.append(se_l / np.sqrt(mean_n))
    return np.array(xs), np.array(ys), np.array(yerr)


def panel_b():
    print("\n=== Panel B: L proportional to proper time tau (fixed rho) ===")
    print(f"{'tau':>6} {'<N>':>8} {'<L>':>9} {'SE(L)':>7}")
    taus, means, ses = [], [], []
    for tau in TAUS_B:
        L, N = measure(RHO_B, tau, N_REAL_B, SEED_PANEL_B)
        means.append(L.mean())
        ses.append(L.std(ddof=1) / np.sqrt(len(L)))
        taus.append(tau)
        print(f"{tau:>6.2f} {N.mean():>8.1f} {L.mean():>9.2f} {ses[-1]:>7.2f}")
    taus = np.array(taus)
    means = np.array(means)
    ses = np.array(ses)
    slope = float(np.sum(taus * means) / np.sum(taus * taus))
    predicted = np.sqrt(2.0 * RHO_B)
    print(f"\nfitted slope L/tau = {slope:.3f}   |   sqrt(2 rho) = {predicted:.3f}")
    return taus, means, ses, slope, predicted


def main() -> None:
    xa, ya, yerra = panel_a()
    tb, mb, seb, slope, predicted = panel_b()

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10, 4.2))

    # Panel A: ratio vs N (log x), with the theoretical asymptote 2.
    axA.errorbar(xa, ya, yerr=yerra, fmt="o-", color="black", capsize=3, ms=4)
    axA.axhline(VKLS_CONSTANT, ls="--", color="0.4", lw=1)
    axA.text(xa[0], VKLS_CONSTANT + 0.005, r"asymptote $=2$", color="0.4", fontsize=9)
    # Finite-size guide 2 - c N^{-1/3}, c fitted for display only (labelled).
    cfit = float(np.mean((VKLS_CONSTANT - ya) * xa ** (1 / 3)))
    ngrid = np.linspace(xa.min(), xa.max(), 200)
    axA.plot(ngrid, VKLS_CONSTANT - cfit * ngrid ** (-1 / 3), ls=":", color="0.5", lw=1,
             label=rf"$2 - {cfit:.2f}\,N^{{-1/3}}$ (guide)")
    axA.set_xscale("log")
    axA.set_xlabel(r"mean number of elements $\langle N\rangle$")
    axA.set_ylabel(r"$\langle L\rangle/\sqrt{\langle N\rangle}$")
    axA.set_title("A. Longest-chain constant")
    axA.legend(frameon=False, fontsize=8, loc="lower right")

    # Panel B: L vs tau with through-origin fit.
    axB.errorbar(tb, mb, yerr=seb, fmt="s", color="black", capsize=3, ms=4,
                 label="measured")
    tgrid = np.linspace(0, tb.max() * 1.02, 100)
    axB.plot(tgrid, slope * tgrid, "-", color="0.3", lw=1,
             label=rf"fit $L={slope:.1f}\,\tau$")
    axB.plot(tgrid, predicted * tgrid, "--", color="0.6", lw=1,
             label=rf"$\sqrt{{2\rho}}\,\tau={predicted:.1f}\,\tau$")
    axB.set_xlabel(r"proper time $\tau$")
    axB.set_ylabel(r"longest chain $\langle L\rangle$")
    axB.set_title(rf"B. $L\propto\tau$ at $\rho={RHO_B:.0f}$")
    axB.legend(frameon=False, fontsize=8, loc="upper left")

    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / "figures" / "exp00_brightwell_gregory.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"\nSaved figure -> {out}")
    print(f"Seeds: panelA={SEED_PANEL_A}, panelB={SEED_PANEL_B} (regenerable).")


if __name__ == "__main__":
    main()
