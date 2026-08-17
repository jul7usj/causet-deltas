"""Phase 2b, Part 2 GATE A: is the Rideout--Wallden 2-link distance stable?

Reproduce with:
    python experiments/exp03_2link_stability.py             # uses the cache
    python experiments/exp03_2link_stability.py --remeasure # ignores the cache

Produces:
    * printed tables (standard errors over realisations, explicit N everywhere),
    * figures/exp03_2link_stability.png,
    * data/exp03_measurements.npz (raw per-realisation values + seeds),
    * an explicit PASS/FAIL statement for each part of the Part-2 Gate A.

The gate (Rideout--Wallden Fig. 14)
-----------------------------------
Their Fig. 14 asks one question: hold a pair of target events **fixed**, hold the
sprinkling **density** fixed, and grow the sprinkling **region** around them --
does the estimated spacelike distance stay put? A distance estimator that only
works when the region is snugly cropped around the pair is useless, because a
real causal set does not come pre-cropped. This is precisely where the naive
construction of their Section II.B fails for spacetime dimension d >= 3.

Protocol implemented here
-------------------------
* Target pair fixed at ``x = (T/2, -D/2, 0)``, ``y = (T/2, +D/2, 0)`` with
  ``D = 1``: spacelike, separation exactly ``D``, at the centre of the region.
  They are appended to the Poisson sprinkling (a measure-zero modification, the
  same device ``sprinkle3d`` uses for diamond endpoints).
* Density fixed at ``rho = 60``. Fixed density is what makes this a test of
  *region growth* rather than of the continuum limit.
* Region: the box of ``sprinkle3d.sprinkle_box_3d`` with extents
  ``(T, Lx, Ly) = lambda * (4, 2.5, 5) * D``, scaled by ``lambda`` over the
  ladder below -- a factor 18 in spacetime volume, 500 -> 9200 elements.

  Why that shape rather than a cube. The 2-links and their matching past
  partners live near the intersection of the two light cones, which for this
  pair is the curve ``X = 0, t = +/- sqrt(D^2/4 + Y^2)``: elongated in ``t`` and
  ``y``, narrow in ``x``. A box shaped that way buys far more of the boost
  freedom the construction depends on per element sprinkled. The shape is held
  FIXED across the ladder -- only ``lambda`` changes -- so it cannot manufacture
  or hide a trend.
* 30-40 realisations at every region size (40 here). Stability is a statement
  about a *mean* holding steady, so the error bars come from realisation count,
  not from N: 5 realisations at huge N could not resolve stable from drifting no
  matter how large N was.

Why the naive estimator is measured on the *same* sprinklings
--------------------------------------------------------------
"The 2-link distance did not move" is not on its own evidence of anything -- an
estimator that ignored its input would also not move. So every realisation is
also measured with the naive double-minimum of Section II.B
(``rideout_wallden.naive_distance``), the estimator the 2-link construction
replaces. It shares the sprinkling, the target pair and the calibration, so any
difference in behaviour is attributable to the estimator alone.

Statistics: what is averaged, and why
--------------------------------------
The 2-links found within one causet are *not* independent samples -- they share
the sprinkling and the target pair. Pooling them and dividing by sqrt(total)
would understate the error. The primary statistic is therefore the
**per-realisation** 2-link mean (Step 5 of Section V.A, run once per causet),
averaged over realisations with the standard error taken *across realisations*.
Causets yielding no 2-link at all contribute nothing and are counted separately;
that count is reported, not hidden, because it is the estimator's failure rate.

Calibration and its known bias (reported, not corrected)
---------------------------------------------------------
Link counts are converted to lengths with eqs. (1)-(2) using the published
``m_3 = 2.296``. eq. (1) is an asymptotic ``rho V -> infinity`` statement, yet
Step 2 deliberately picks the *smallest* interval available -- typically a few
tens of elements here. Part 1 measured ``m_3^eff`` well below 2.296 at such
sizes, so the calibrated distance is expected to come out systematically LOW by
roughly ``m_3^eff / m_3``. The mean minimising-interval size is printed at every
region size so the regime is visible, and the expected ratio is quoted from
Rideout--Wallden's own Fig.-4 curve evaluated there. Nothing is corrected by it:
it is a fixed-density constant that cannot create or mask a trend, which is why
the gate is about the *slope*, not the offset.

Acceptance criteria, fixed before the verdict
----------------------------------------------
Two conditions, because "flat" is meaningless without "and we could have seen it
if it were not":

  A1 STABILITY  -- the weighted slope of distance against ``log2(V)`` is
                   consistent with zero at <= 2 sigma.
  A2 RESOLUTION -- that slope's uncertainty is small enough that a degradation
                   of 20% of ``D`` across the full volume range would have been
                   a >= 2 sigma detection. Without A2, wide error bars would pass
                   A1 by default (Part-2 Constraint 3).

Both are also computed for the naive control, whose behaviour is reported
whatever it turns out to be.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from causet import rideout_wallden as rw, sprinkle3d  # noqa: E402
from causet.order3d import causal_matrix_3d  # noqa: E402

# ---- Experiment parameters (Integrity Rule 4: named, documented) -------------
SEED_BASE = 20260817
TARGET_SEPARATION = 1.0  # D: the true spacelike distance the estimators must find
RHO = 60.0  # fixed sprinkling density
BASE_SHAPE = (4.0, 2.5, 5.0)  # (T, Lx, Ly) in units of D at lambda = 1
#: Region scale factors. Volume goes as lambda^3, so this is a factor 18.4 in V.
LAMBDAS = [0.55, 0.70, 0.85, 1.00, 1.15, 1.30, 1.45]
N_REAL = 40  # realisations per region size (Part-2 Constraint 3: 30-40)

#: A degradation this large across the whole ladder is what the gate must be able
#: to see; expressed as a fraction of the true separation D.
MEANINGFUL_DRIFT_FRACTION = 0.20

CACHE = Path(__file__).resolve().parents[1] / "data" / "exp03_measurements.npz"

# ---- Rideout--Wallden Fig.-4 curve, used only to quote the expected offset ----
RW_FIT_A = -1.087
RW_FIT_B = -0.1201


def rw_m3_effective(rho_volume: float) -> float:
    """``m_3^eff`` at a finite interval size, from RW's own Fig.-4 fit.

    ``f(N) = m_3 + a e^{b log_2 N}`` with their published ``a``, ``b``. Used
    only to state how much of this experiment's low offset the finite-size
    calibration already accounts for. Nothing is corrected with it.
    """
    return rw.RW_M3 + RW_FIT_A * math.exp(RW_FIT_B * math.log2(max(rho_volume, 2.0)))


# ------------------------------------------------------------------ measurement --
def region_extents(lam: float) -> tuple[float, float, float]:
    return tuple(s * lam * TARGET_SEPARATION for s in BASE_SHAPE)  # type: ignore[return-value]


def measure_region(lam: float, n_real: int, index: int) -> dict:
    """Measure both estimators over ``n_real`` sprinklings at region scale ``lam``.

    Nothing is averaged or discarded here (Integrity Rule 1): per-realisation
    arrays come back raw, with ``nan`` marking a realisation in which an
    estimator produced nothing.
    """
    t_ext, x_ext, y_ext = region_extents(lam)
    volume = sprinkle3d.box_volume_3d(t_ext, x_ext, y_ext)
    half_d = 0.5 * TARGET_SEPARATION

    two_link = np.full(n_real, np.nan)
    naive = np.full(n_real, np.nan)
    n_2links = np.zeros(n_real, dtype=int)
    interval_size = np.full(n_real, np.nan)
    chain_links = np.full(n_real, np.nan)
    naive_pairs = np.zeros(n_real, dtype=int)
    realised_n = np.zeros(n_real, dtype=int)

    t0 = time.time()
    for k in range(n_real):
        seed = SEED_BASE + 10_000 * index + k
        s = sprinkle3d.sprinkle_box_3d(RHO, t_ext, x_ext, y_ext, seed=seed)
        # Append the two fixed targets: spacelike, separation exactly D, centred.
        t = np.concatenate((s.t, [0.5 * t_ext, 0.5 * t_ext]))
        x = np.concatenate((s.x, [-half_d, half_d]))
        y = np.concatenate((s.y, [0.0, 0.0]))
        i_x, i_y = t.size - 2, t.size - 1
        cm = causal_matrix_3d(t, x, y)
        realised_n[k] = t.size

        res = rw.two_link_distance(i_x, i_y, cm, rw.RW_M3, rho=RHO)
        n_2links[k] = res.n_two_links
        if res.n_two_links:
            two_link[k] = res.mean
            interval_size[k] = res.mean_interval_size
            chain_links[k] = float(res.per_link_chain_links.mean())

        nres = rw.naive_distance(i_x, i_y, cm, rw.RW_M3, rho=RHO)
        naive[k] = nres.distance
        naive_pairs[k] = nres.n_pairs
        del cm

    elapsed = time.time() - t0
    print(
        f"  lambda={lam:4.2f}  V={volume:7.2f}  <N>={realised_n.mean():7.0f}  "
        f"#real={n_real:3d}  2-links/causet={n_2links.mean():4.2f}  "
        f"empty={int((n_2links == 0).sum()):2d}  ({elapsed:6.1f} s)",
        flush=True,
    )
    return {
        "lam": lam,
        "volume": volume,
        "n_real": n_real,
        "two_link": two_link,
        "naive": naive,
        "n_2links": n_2links,
        "interval_size": interval_size,
        "chain_links": chain_links,
        "naive_pairs": naive_pairs,
        "realised_n": realised_n,
        "elapsed": elapsed,
    }


def load_or_measure(remeasure: bool) -> list[dict]:
    key = (
        f"{SEED_BASE}|{RHO}|{TARGET_SEPARATION}|{BASE_SHAPE}|{N_REAL}|"
        + ",".join(f"{l:.2f}" for l in LAMBDAS)
    )
    fields = (
        "two_link", "naive", "n_2links", "interval_size",
        "chain_links", "naive_pairs", "realised_n",
    )
    if not remeasure and CACHE.exists():
        z = np.load(CACHE, allow_pickle=False)
        if str(z["key"]) == key:
            print(f"Loaded cached measurements from {CACHE}")
            rows = []
            for i, lam in enumerate(LAMBDAS):
                row = {
                    "lam": lam,
                    "volume": sprinkle3d.box_volume_3d(*region_extents(lam)),
                    "n_real": N_REAL,
                    "elapsed": float(z[f"elapsed_{i}"]),
                }
                row.update({f: z[f"{f}_{i}"] for f in fields})
                rows.append(row)
            return rows
        print("Cache present but parameters differ -- remeasuring.")

    print(
        f"Measuring {len(LAMBDAS)} region sizes x {N_REAL} realisations "
        "(the naive control dominates the cost; ~12 min)..."
    )
    rows = [measure_region(lam, N_REAL, i) for i, lam in enumerate(LAMBDAS)]
    CACHE.parent.mkdir(exist_ok=True)
    payload = {"key": np.array(key)}
    for i, r in enumerate(rows):
        for f in fields:
            payload[f"{f}_{i}"] = r[f]
        payload[f"elapsed_{i}"] = np.array(r["elapsed"])
    np.savez(CACHE, **payload)
    print(f"Saved raw measurements -> {CACHE}")
    return rows


# -------------------------------------------------------------------- analysis --
def mean_se(values: np.ndarray) -> tuple[float, float, int]:
    """``(mean, standard error, count)`` over the non-nan entries."""
    v = values[np.isfinite(values)]
    n = int(v.size)
    if n == 0:
        return float("nan"), float("nan"), 0
    if n == 1:
        return float(v[0]), float("nan"), 1
    return float(v.mean()), float(v.std(ddof=1) / math.sqrt(n)), n


def weighted_line_fit(xs: np.ndarray, ys: np.ndarray, ses: np.ndarray) -> dict:
    """Weighted least-squares straight-line fit ``y = intercept + slope * x``.

    Written out rather than delegated so the covariance is explicit: with
    ``w = 1/se^2``, ``S = sum w``, ``Sx = sum w x`` etc., the slope variance is
    ``S / (S Sxx - Sx^2)``. Points with a non-finite standard error (a region
    size where a single realisation produced a value) are dropped and counted.
    """
    ok = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(ses) & (ses > 0)
    x, y, s = xs[ok], ys[ok], ses[ok]
    if x.size < 3:
        raise ValueError(f"need >= 3 usable points for a slope, got {x.size}")
    w = 1.0 / s**2
    sw, swx, swy = w.sum(), (w * x).sum(), (w * y).sum()
    swxx, swxy = (w * x * x).sum(), (w * x * y).sum()
    denom = sw * swxx - swx * swx
    slope = (sw * swxy - swx * swy) / denom
    intercept = (swxx * swy - swx * swxy) / denom
    slope_err = math.sqrt(sw / denom)
    resid = y - (intercept + slope * x)
    chi2 = float((w * resid**2).sum())
    return {
        "slope": float(slope),
        "slope_err": float(slope_err),
        "intercept": float(intercept),
        "chi2": chi2,
        "dof": int(x.size - 2),
        "n_used": int(x.size),
        "n_dropped": int(ok.size - x.size),
    }


def summarise(rows: list[dict], field: str) -> tuple[np.ndarray, ...]:
    log2v = np.array([math.log2(r["volume"]) for r in rows])
    mean = np.empty(len(rows))
    se = np.empty(len(rows))
    count = np.empty(len(rows), dtype=int)
    for i, r in enumerate(rows):
        mean[i], se[i], count[i] = mean_se(np.asarray(r[field], dtype=float))
    return log2v, mean, se, count


def report_gate(name: str, fit: dict, span: float) -> tuple[bool, bool]:
    """Print and evaluate criteria A1 (stability) and A2 (resolution)."""
    drift = MEANINGFUL_DRIFT_FRACTION * TARGET_SEPARATION
    slope_that_matters = drift / span
    pull = abs(fit["slope"]) / fit["slope_err"]
    resolvable = fit["slope_err"] <= slope_that_matters / 2.0
    stable = pull <= 2.0
    print(f"\n  --- {name} ---")
    print(
        f"    slope = {fit['slope']:+.5f} +/- {fit['slope_err']:.5f} per log2(V)"
        f"   ({pull:.2f} sigma from zero)"
    )
    print(
        f"    over the full span of {span:.2f} in log2(V) that is a total change of "
        f"{fit['slope'] * span:+.4f} +/- {fit['slope_err'] * span:.4f}"
    )
    print(f"    chi2/dof = {fit['chi2']:.2f}/{fit['dof']}   points used: {fit['n_used']}")
    if fit["n_dropped"]:
        # Never drop a point silently (Integrity Rule 1): a zero standard error
        # means every realisation at that size landed on the same quantised link
        # count, which carries no weight information.
        print(
            f"    NOTE: {fit['n_dropped']} region size(s) dropped from the fit "
            "for having a zero or undefined standard error."
        )
    print(
        f"    A1 stability  (|slope| <= 2 sigma)                : "
        f"{'pass' if stable else 'FAIL'}"
    )
    print(
        f"    A2 resolution (sigma_slope <= {slope_that_matters / 2:.5f}, i.e. a "
        f"{MEANINGFUL_DRIFT_FRACTION:.0%}-of-D drift would be >= 2 sigma): "
        f"{'pass' if resolvable else 'FAIL'}"
    )
    return stable, resolvable


# ------------------------------------------------------------------------ main --
def main(remeasure: bool = False) -> None:
    print("Phase 2b Part 2 GATE A -- Rideout--Wallden 2-link distance stability")
    print(f"Seeds: SEED_BASE + 10000*i + k   (SEED_BASE={SEED_BASE})")
    print(
        f"Fixed target separation D = {TARGET_SEPARATION}, fixed density rho = {RHO}, "
        f"region shape (T,Lx,Ly) = {BASE_SHAPE} x lambda"
    )
    print(
        f"Calibration: eqs. (1)-(2) with m_3 = {rw.RW_M3} (link convention "
        "throughout -- Part-1 Constraint 1)"
    )

    rows = load_or_measure(remeasure)

    log2v, tl_mean, tl_se, tl_n = summarise(rows, "two_link")
    _, nv_mean, nv_se, nv_n = summarise(rows, "naive")

    print("\n=== Measurements (errors are standard errors ACROSS realisations) ===")
    print(
        f"{'lam':>5} {'V':>8} {'<N>':>7} {'#real':>6} {'2L/causet':>10} {'tot2L':>6} "
        f"{'empty':>6} {'d_2link':>9} {'SE':>7} {'nz':>4} {'d_naive':>9} {'SE':>7} "
        f"{'<L>':>5} {'<|[p,f]|>':>10} {'nPairs':>7}"
    )
    for i, r in enumerate(rows):
        n2 = np.asarray(r["n_2links"])
        print(
            f"{r['lam']:>5.2f} {r['volume']:>8.2f} {np.mean(r['realised_n']):>7.0f} "
            f"{r['n_real']:>6d} {n2.mean():>10.2f} {int(n2.sum()):>6d} "
            f"{int((n2 == 0).sum()):>6d} "
            f"{tl_mean[i]:>9.4f} {tl_se[i]:>7.4f} {tl_n[i]:>4d} "
            f"{nv_mean[i]:>9.4f} {nv_se[i]:>7.4f} "
            f"{np.nanmean(r['chain_links']):>5.2f} "
            f"{np.nanmean(r['interval_size']):>10.1f} "
            f"{np.mean(r['naive_pairs']):>7.0f}"
        )

    span = float(log2v.max() - log2v.min())
    print(f"\n  Volume range: {2 ** log2v.min():.1f} -> {2 ** log2v.max():.1f} "
          f"(factor {2 ** span:.1f}), i.e. {span:.2f} in log2(V)")

    # ------------------------------------------------------------- the gate --
    print("\n=== GATE A: does the estimate hold still as the region grows? ===")
    tl_fit = weighted_line_fit(log2v, tl_mean, tl_se)
    nv_fit = weighted_line_fit(log2v, nv_mean, nv_se)
    a1, a2 = report_gate("Rideout--Wallden 2-link distance", tl_fit, span)
    nv_a1, nv_a2 = report_gate("naive double minimum (control, Sec. II.B)", nv_fit, span)

    # ------------------------------------- the offset, quoted but not applied --
    grand = float(np.nanmean(tl_mean))
    mean_interval = float(np.nanmean(np.concatenate([r["interval_size"] for r in rows])))
    m3_eff = rw_m3_effective(mean_interval)
    print("\n=== Scale offset (reported, NOT corrected) ===")
    print(f"  grand mean of the 2-link distance : {grand:.4f}   vs true D = "
          f"{TARGET_SEPARATION:.4f}   ratio {grand / TARGET_SEPARATION:.3f}")
    print(f"  mean minimising-interval size     : {mean_interval:.1f} elements")
    print(f"  RW's own Fig.-4 curve at that size: m_3^eff = {m3_eff:.3f} "
          f"(vs asymptotic {rw.RW_M3}), so eq. (1) applied there under-reports")
    print(f"  lengths by a factor {m3_eff / rw.RW_M3:.3f} on its own.")
    print("  The offset is a fixed-density constant: it shifts every point equally")
    print("  and therefore cannot create or mask the slope the gate tests.")

    # -------------------------------------------------- sample-size finding --
    total_2links = int(sum(int(np.sum(r["n_2links"])) for r in rows))
    total_real = int(sum(r["n_real"] for r in rows))
    total_empty = int(sum(int(np.sum(np.asarray(r["n_2links"]) == 0)) for r in rows))
    per_causet = np.array([np.mean(r["n_2links"]) for r in rows])
    yield_se = np.array(
        [np.std(r["n_2links"], ddof=1) / math.sqrt(r["n_real"]) for r in rows]
    )
    print("\n=== 2-link yield (reported always -- it bounds the estimator's usability) ===")
    print(f"  {total_2links} 2-links over {total_real} causets "
          f"= {total_2links / total_real:.2f} per causet")
    print(f"  {total_empty}/{total_real} causets ({100 * total_empty / total_real:.0f}%) "
          "yielded NO 2-link at all: on those the estimator returns nothing.")
    print(f"  per-causet yield across a factor {2 ** span:.0f} in volume: "
          f"{per_causet.min():.2f} -> {per_causet.max():.2f} "
          f"(first to last: {per_causet[0]:.2f} -> {per_causet[-1]:.2f})")

    # How does the sample size actually grow? Rideout--Wallden state that for
    # n < d (here 2 < 3) infinite Minkowski holds infinitely many n-links, but say
    # nothing about the rate, and the rate is what decides usability. Two
    # falsifiable models are fitted to the same points and compared by chi^2.
    volumes = np.array([r["volume"] for r in rows])
    w = 1.0 / yield_se**2
    ln_v = np.log(volumes)
    b_log = float((w * ln_v * per_causet).sum() / (w * ln_v * ln_v).sum())
    b_log_err = float(math.sqrt(1.0 / (w * ln_v * ln_v).sum()))
    chi2_log = float((w * (per_causet - b_log * ln_v) ** 2).sum())
    b_lin = float((w * volumes * per_causet).sum() / (w * volumes * volumes).sum())
    chi2_lin = float((w * (per_causet - b_lin * volumes) ** 2).sum())
    dof = len(rows) - 1
    print(f"\n  yield ~ {b_log:.3f} +/- {b_log_err:.3f} * ln V   "
          f"chi2/dof = {chi2_log:.2f}/{dof} = {chi2_log / dof:.2f}")
    print(f"  yield ~ {b_lin:.5f} * V (proportional-to-volume alternative)   "
          f"chi2/dof = {chi2_lin:.2f}/{dof} = {chi2_lin / dof:.2f}")
    print("  => the number of 2-links grows only LOGARITHMICALLY with the region.")
    print("     Consistent with 'infinitely many in infinite Minkowski' (n=2 < d=3),")
    print("     but it means no computationally reachable region yields a large sample:")
    # y = b ln V, so doubling y needs ln V' = 2 ln V, i.e. V' = V^2: each further
    # doubling of the sample SQUARES the region rather than multiplying it.
    print(f"     {2 ** span:.0f}x the volume bought {per_causet[-1] / per_causet[0]:.1f}x "
          "the 2-links, and doubling the yield again requires SQUARING the volume")
    print(f"     ({volumes[-1]:.0f} -> {volumes[-1] ** 2:.0f}, i.e. "
          f"N ~ {RHO * volumes[-1] ** 2:.2e} elements).")

    passed = a1 and a2
    print("\n" + "=" * 78)
    print(f"PART 2 GATE A: {'PASS' if passed else 'FAIL'}")
    print(f"  A1 2-link distance does not drift     : {'pass' if a1 else 'FAIL'}"
          f"   (slope {tl_fit['slope']:+.5f} +/- {tl_fit['slope_err']:.5f})")
    print(f"  A2 error bars can resolve a drift     : {'pass' if a2 else 'FAIL'}")
    print(f"  control: naive estimator A1           : "
          f"{'flat too' if nv_a1 else 'DRIFTS'}"
          f"   (slope {nv_fit['slope']:+.5f} +/- {nv_fit['slope_err']:.5f})")
    print(f"  control: naive estimator A2           : "
          f"{'resolved' if nv_a2 else 'under-resolved'}")
    print("=" * 78)

    make_figure(rows, log2v, tl_mean, tl_se, tl_n, nv_mean, nv_se, tl_fit, nv_fit)


def make_figure(rows, log2v, tl_mean, tl_se, tl_n, nv_mean, nv_se, tl_fit, nv_fit) -> None:
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15.5, 4.4))
    grid = np.linspace(log2v.min() - 0.2, log2v.max() + 0.2, 100)

    # Panel A: the gate itself.
    axA.axhline(TARGET_SEPARATION, color="0.35", ls="--", lw=1.2,
                label=rf"true separation $D={TARGET_SEPARATION:g}$")
    axA.plot(grid, tl_fit["intercept"] + tl_fit["slope"] * grid, "-",
             color="tab:blue", lw=1.2,
             label=rf"RW fit: slope $={tl_fit['slope']:+.4f}\pm{tl_fit['slope_err']:.4f}$")
    axA.errorbar(log2v, tl_mean, yerr=tl_se, fmt="o", color="tab:blue", capsize=3,
                 ms=5, zorder=5, label="RW 2-link distance")
    axA.plot(grid, nv_fit["intercept"] + nv_fit["slope"] * grid, "-",
             color="tab:red", lw=1.0, alpha=0.7,
             label=rf"naive fit: slope $={nv_fit['slope']:+.4f}\pm{nv_fit['slope_err']:.4f}$")
    axA.errorbar(log2v, nv_mean, yerr=nv_se, fmt="s", mfc="none", color="tab:red",
                 capsize=3, ms=5, zorder=4, label="naive double min. (control)")
    axA.set_xlabel(r"$\log_2 V$   (fixed $\rho$, fixed target pair)")
    axA.set_ylabel("estimated spacelike distance")
    axA.set_title("A. GATE A: stability under region growth")
    # Headroom above D = 1 so the legend does not sit on top of the points.
    lo = float(np.nanmin(nv_mean - nv_se))
    axA.set_ylim(lo - 0.04, TARGET_SEPARATION + 0.10)
    axA.legend(frameon=False, fontsize=7, loc="upper left")

    # Panel B: the yield -- the estimator's own sample size.
    n2_mean = np.array([np.mean(r["n_2links"]) for r in rows])
    n2_se = np.array([np.std(r["n_2links"], ddof=1) / math.sqrt(r["n_real"]) for r in rows])
    empty_frac = np.array([np.mean(np.asarray(r["n_2links"]) == 0) for r in rows])
    axB.errorbar(log2v, n2_mean, yerr=n2_se, fmt="^", color="tab:purple", capsize=3,
                 ms=5, label="2-links per causet")
    axB.set_xlabel(r"$\log_2 V$")
    axB.set_ylabel("2-links found per causet")
    axB.set_ylim(bottom=0.0)
    axB2 = axB.twinx()
    axB2.plot(log2v, 100 * empty_frac, "v--", color="tab:orange", ms=5, lw=1.0,
              label="causets with none")
    axB2.set_ylabel("% of causets yielding no 2-link", color="tab:orange")
    axB2.tick_params(axis="y", colors="tab:orange")
    axB2.set_ylim(0, 100)
    handles = axB.get_legend_handles_labels()[0] + axB2.get_legend_handles_labels()[0]
    labels = axB.get_legend_handles_labels()[1] + axB2.get_legend_handles_labels()[1]
    axB.legend(handles, labels, frameon=False, fontsize=7, loc="upper left")
    axB.set_title("B. Sample size the average rests on")

    # Panel C: the discreteness of the estimator.
    all_links = np.concatenate([np.asarray(r["chain_links"], dtype=float) for r in rows])
    all_links = all_links[np.isfinite(all_links)]
    axC.hist(all_links, bins=np.arange(1.75, all_links.max() + 0.75, 0.5),
             color="tab:blue", alpha=0.75, edgecolor="white")
    axC.axvline(2.0, color="tab:red", ls=":", lw=1.4,
                label="hard floor: 2 links\n($p \\prec x,y \\prec f$ always)")
    # One value per causet: the mean over that causet's 2-links of the Step-3
    # minimum. With ~1.2 2-links per causet it is usually a single integer, so
    # the integer-centred bins below show the estimator's raw quantisation.
    axC.set_xlabel("per-causet mean of the minimised $d(p,f_i)$  [links]")
    axC.set_ylabel("causets")
    axC.set_title("C. The estimator is coarsely quantised\n"
                  f"1 link $= {rw.proper_time_from_chain_links(1.0, RHO):.3f}$ in length units "
                  f"$= {100 * rw.proper_time_from_chain_links(1.0, RHO) / TARGET_SEPARATION:.0f}\\%$ of $D$")
    axC.legend(frameon=False, fontsize=7, loc="upper right")

    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / "figures" / "exp03_2link_stability.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"\nSaved figure -> {out}")


if __name__ == "__main__":
    main(remeasure="--remeasure" in sys.argv)
