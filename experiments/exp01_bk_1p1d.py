"""Phase-2a experiment: Boguñá--Krioukov causal-overlap distance in 1+1 D.

Reproduce with:
    python experiments/exp01_bk_1p1d.py

Produces:
    * a printed table (mean estimate, standard error over realisations,
      admissible-c counts), and
    * figures/exp01_bk_convergence.png

Physics being reproduced (qualitatively -- different seeds, not the paper's exact
numbers). Boguñá--Krioukov (arXiv:2401.17376) show that their causal-overlap
spacelike-distance estimator converges to the true continuum separation as the
sprinkling density grows (their Fig. 3/4 pattern). We reproduce that shape in
their d = 1 case:

  Fix a spacelike target pair a = (t_ab, +s/2), b = (t_ab, -s/2) -- equal time,
  proper spatial separation s. Sprinkle a bounding causal diamond (past tip
  c0 = (0,0)) at increasing density rho. For each realisation run the FULL
  intrinsic pipeline:
     1. enumerate common past events c (c prec a, c prec b),
     2. keep those passing Filter 2 (eq. 34) -- symmetric vantage points,
     3. measure overlap O_C (eq. 28), estimate depth tau_c (eq. 38, alpha_1=1/sqrt2),
     4. map to distance via the exact d=1 closed form (eq. 24),
     5. average over admissible c.
  Plot the mean estimate vs rho with standard-error bars and the true value s;
  and the relative error vs rho on log-log axes to read off the convergence rate.

ACCEPTANCE (Phase-2a gate): relative error shrinks (up to realisation noise) as
rho increases; the actual numeric convergence rate is reported, not forced.

Everything regenerable from the fixed seeds recorded below (Integrity Rule 1).
No data is smoothed (Rule 1); error bars are standard errors over independent
realisations and admissible-c counts are reported (Rule 3).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from causet import causal_overlap as co  # noqa: E402
from causet import order, sprinkle  # noqa: E402

# ---- Documented parameters (Integrity Rule 4) -----------------------------
SEED0 = 20260721  # base seed; realisation k uses SEED0 + k.
N_REAL = 40  # independent sprinklings per density.
T_AB = 3.0  # target-pair time coordinate (mid-slice of the bounding diamond).
SEPARATION = 1.0  # true proper spacelike separation s between a and b.
# Density ladder: E[N] in the bounding diamond ~ rho * (2*T_AB)^2 / 2 = 18*rho,
# so this spans ~ 900 .. 28800 elements per realisation.
DENSITIES = [25.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0]

# Continuum depth of the on-axis common event c0=(0,0): proper time c0 -> a.
TAU_C_CONTINUUM = np.sqrt(T_AB**2 - (0.5 * SEPARATION) ** 2)
# Overlap the exact eq. 24 demands for (s, tau_c) -- the value <O_C> should
# converge to (used only for the diagnostic overlap panel, not the estimate).
O_PRED = co.overlap_predicted_from_distance(SEPARATION, TAU_C_CONTINUUM)


def _sprinkle_fixed_pair(rho: float, seed: int):
    """Sprinkle the bounding diamond and inject c0, a, b.

    Returns (C, ic, ia, ib, n_interior, u, v). The null coordinates u, v are
    returned so the experiment can use the O(N log N) LIS chain counter (identical
    result to the matrix DP; Phase 1 verified this) instead of the O(N^2) default
    -- a pure speed optimisation, no change to the estimator's physics.
    """
    diamond_tau = 2.0 * T_AB
    sp = sprinkle.sprinkle_diamond_1d(rho, diamond_tau, seed=seed, include_endpoints=False)
    t = np.concatenate([sp.t, [0.0, T_AB, T_AB]])
    x = np.concatenate([sp.x, [0.0, +0.5 * SEPARATION, -0.5 * SEPARATION]])
    u, v = t + x, t - x
    c_mat = order.causal_matrix_1d(u, v)
    n = t.shape[0]
    return c_mat, n - 3, n - 2, n - 1, sp.n, u, v


def _make_lis_chain_counter(c_mat, u, v):
    """Return a fast n_C(c, x) using 1+1 D LIS over the interval's null coords.

    Numerically identical to causet.causal_overlap.chain_count (Phase 1 proved the
    LIS and matrix-DP longest chains agree); used only to make the density ladder
    tractable.
    """

    def counter(c_idx: int, x_idx: int) -> int:
        interior = co.alexandrov_interval(x_idx, c_idx, c_mat)
        idx = np.concatenate(([c_idx], interior, [x_idx])).astype(int)
        return order.chain_via_lis_1d(u[idx], v[idx])

    return counter


def measure(rho: float):
    """Run N_REAL realisations at density rho.

    Returns dict with per-realisation distance estimates, the diagnostic mean
    overlap from the fixed on-axis c0, and admissible-c counts.
    """
    dist_est = []
    overlaps_c0 = []
    n_c_used = []
    n_elems = []
    for k in range(N_REAL):
        c_mat, ic, ia, ib, n_interior, u, v = _sprinkle_fixed_pair(rho, SEED0 + k)
        n_elems.append(n_interior)
        # Diagnostic: overlap from the fixed on-axis common event c0 (index ic).
        o0 = co.causal_overlap(ia, ib, ic, c_mat)
        if np.isfinite(o0):
            overlaps_c0.append(o0)
        # Full intrinsic pipeline (Filter-2-selected c, eq. 38 tau_c, eq. 24).
        chain_fn = _make_lis_chain_counter(c_mat, u, v)
        res = co.distance_causal_overlap(ia, ib, c_mat, rho, chain_count_fn=chain_fn)
        if np.isfinite(res.distance):
            dist_est.append(res.distance)
            n_c_used.append(res.n_c)
    return {
        "dist": np.asarray(dist_est),
        "overlap_c0": np.asarray(overlaps_c0),
        "n_c": np.asarray(n_c_used),
        "n_elems": np.asarray(n_elems),
    }


def main():
    print("=" * 74)
    print("Phase 2a -- Boguñá--Krioukov causal-overlap distance (1+1 D)")
    print(f"  true separation s = {SEPARATION},  tau_c (continuum) = {TAU_C_CONTINUUM:.4f}")
    print(f"  O predicted by eq. 24 for this (s, tau_c): {O_PRED:.4f}")
    print(f"  seeds: {SEED0}+k, k=0..{N_REAL - 1};  realisations/density = {N_REAL}")
    print("=" * 74)
    header = (
        f"{'rho':>8} {'<N_int>':>9} {'<est d>':>9} {'SE(d)':>8} "
        f"{'rel.err':>9} {'<O@c0>':>8} {'med.#c':>7}"
    )
    print(header)

    rhos = []
    est_mean, est_sem, rel_err = [], [], []
    o_mean = []
    for rho in DENSITIES:
        m = measure(rho)
        d = m["dist"]
        mu = float(d.mean())
        se = float(d.std(ddof=1) / np.sqrt(d.size))
        re = abs(mu - SEPARATION) / SEPARATION
        omu = float(m["overlap_c0"].mean())
        med_c = int(np.median(m["n_c"]))
        rhos.append(rho)
        est_mean.append(mu)
        est_sem.append(se)
        rel_err.append(re)
        o_mean.append(omu)
        print(
            f"{rho:>8.0f} {m['n_elems'].mean():>9.0f} {mu:>9.4f} {se:>8.4f} "
            f"{re:>9.4f} {omu:>8.4f} {med_c:>7d}"
        )

    rhos = np.asarray(rhos)
    est_mean = np.asarray(est_mean)
    est_sem = np.asarray(est_sem)
    rel_err = np.asarray(rel_err)
    o_mean = np.asarray(o_mean)

    # Convergence-rate fit: rel.err ~ C * rho^(-p) via log-log least squares.
    # Two fits are reported honestly:
    #   * ALL densities: dominated by the low-rho points, which are noise-limited
    #     (few admissible c, large SE) and sit artificially close to s, so the
    #     global "rate" is meaningless (often even positive). Reported anyway.
    #   * WELL-SAMPLED regime rho >= RHO_REGIME: where the per-realisation noise is
    #     sub-dominant to the systematic. This is the physically meaningful rate.
    RHO_REGIME = 200.0
    mask = rel_err > 0
    slope_all, icpt_all = np.polyfit(np.log(rhos[mask]), np.log(rel_err[mask]), 1)
    reg = mask & (rhos >= RHO_REGIME)
    slope_reg, icpt_reg = np.polyfit(np.log(rhos[reg]), np.log(rel_err[reg]), 1)
    print("-" * 74)
    print(
        f"Log-log fit, ALL rho:            rel.err ~ {np.exp(icpt_all):.3g}"
        f" * rho^({slope_all:+.3f})   [noise-contaminated at low rho]"
    )
    print(
        f"Log-log fit, rho >= {RHO_REGIME:.0f} (regime): rel.err ~ {np.exp(icpt_reg):.3g}"
        f" * rho^({slope_reg:+.3f})   [compare N^(-1/3) => rho^(-0.333)]"
    )
    monotone_all = bool(np.all(np.diff(rel_err) <= est_sem[1:] / SEPARATION + 1e-9))
    monotone_reg = bool(
        np.all(np.diff(rel_err[rhos >= RHO_REGIME]) <= 1e-9)
    )
    print(f"Relative error monotone-decreasing (all rho, within noise): {monotone_all}")
    print(f"Relative error monotone-decreasing (rho >= {RHO_REGIME:.0f}):        {monotone_reg}")
    slope = slope_all  # kept for the figure's illustrative guide line

    # ---- figure -----------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    ax1.axhline(SEPARATION, color="k", ls="--", lw=1, label=f"true s = {SEPARATION}")
    ax1.errorbar(
        rhos, est_mean, yerr=est_sem, fmt="o-", capsize=3, color="C0",
        label="causal-overlap estimate",
    )
    ax1.set_xscale("log")
    ax1.set_xlabel(r"sprinkling density $\rho$")
    ax1.set_ylabel(r"estimated spacelike distance $d_{M^2}(a,b)$")
    ax1.set_title("Distance estimate converges to true separation")
    ax1.legend(frameon=False)

    ax2.loglog(rhos, rel_err, "s-", color="C3", label="relative error")
    ax2.loglog(
        rhos, np.exp(intercept) * rhos**slope, ":", color="gray",
        label=rf"fit $\propto \rho^{{{slope:.2f}}}$",
    )
    ax2.set_xlabel(r"sprinkling density $\rho$")
    ax2.set_ylabel(r"relative error $|\hat d - s|/s$")
    ax2.set_title("Convergence rate")
    ax2.legend(frameon=False)

    fig.suptitle(
        "Boguñá--Krioukov causal-overlap distance, 1+1 D "
        f"(s={SEPARATION}, $\\tau_c$={TAU_C_CONTINUUM:.2f}); seeds {SEED0}+k",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(__file__).resolve().parents[1] / "figures" / "exp01_bk_convergence.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"\nFigure written: {out}")


if __name__ == "__main__":
    main()
