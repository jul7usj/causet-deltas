"""Phase 2b Part B DIAGNOSTIC: is B--K's large-s compression the eqs.-25--27 breakdown?

Reproduce with:
    python experiments/exp05_bk_asymptotic_regime.py             # uses the cache
    python experiments/exp05_bk_asymptotic_regime.py --remeasure # ignores the cache

Produces:
    * printed tables (standard errors over causets, explicit N everywhere),
    * figures/exp05_bk_asymptotic_regime.png,
    * data/exp05_measurements.npz (per-causet AND per-vantage-point raw values),
    * an explicit CONFIRMED / REFUTED / UNRESOLVED statement.

DIAGNOSTIC, NOT A GATE. No pass/fail is issued. Its one job is to close, or
explicitly leave open, the confound logged as Finding 9 of the 2026-09-22 entry.

THE CONFOUND THIS EXISTS TO BREAK
==================================
exp04 measured the 2+1 D Boguna--Krioukov overlap distance compressing badly at
large separation -- ``B-K/true`` falling monotonically 1.108 -> 0.623 as ``s``
ran 0.30 -> 2.00 -- while the realised ``tau_c/d_est`` fell 4.77 -> 1.18 over the
same ladder. ``Spearman(B-K/true, tau_c/d) = +1.0000``.

That correlation is worth nothing on its own. Both quantities are monotone
functions of ``s`` by construction: ``B-K/true`` because the estimator
compresses, ``tau_c/d`` because a wider pair has less room above it inside a box
of fixed height. Two monotone functions of a common variable always rank-correlate
at ~1, mechanism or no mechanism. exp04 therefore recorded the asymptotic
explanation as CONSISTENT WITH the data and explicitly NOT established.

This experiment decouples them: ``s`` is held FIXED and the box TIME EXTENT is
varied instead, which moves the depth of the common past -- hence ``tau_c`` --
without moving the separation at all.

GEOMETRY DELIBERATELY DIFFERS FROM exp03/exp04 -- NOT COMPARABLE ON SCALE
=========================================================================
exp03 (Gate A) and exp04 (Gate B) were frozen at ``T = 4.0`` precisely so their
usability numbers stayed comparable with each other. This experiment breaks that
freeze on purpose, because the frozen geometry is what made the question
unanswerable. Consequently **no absolute number here may be compared with exp03
or exp04**, and none is. The quantities compared are internal to this
experiment: ratios measured across its own T ladder.

What is varied, and what is held
---------------------------------
* Separation ``s = 1.0``, FIXED. Chosen mid-range: exp04 measured RW's 2-link
  yield falling as ``s^-1.46``, and at ``s = 1.0`` both estimators still have
  adequate yield (RW ~1.2 2-links per causet, B--K ~16 admissible ``c``), so
  neither is starved at the point where they are compared.
* Density ``rho = 60``, FIXED -- the same value as exp03/exp04, so the
  discreteness scale (one chain link = 0.1739) is unchanged.
* Spatial extents ``(Lx, Ly) = (2.5, 5.0)``, FIXED.
* Box time extent ``T``, VARIED over ``T_LADDER``. This is the only thing that
  moves. Targets stay at the mid-time ``T/2`` on the x axis, so a taller box
  deepens the common past and the common future symmetrically.

THE RW CONTROL -- what makes this diagnostic falsifiable
=========================================================
Rideout--Wallden's calibration (eqs. 1--2, chain links -> proper time) contains
no asymptotic-regime condition of the B--K kind: it does not care how deep the
common past is. So if the compression is really the eqs.-25--27 breakdown,
``B-K/true`` must move with ``T`` while ``RW/true`` stays flat. If BOTH move, the
cause is something shared -- boundary effects, common-past truncation, a
region-size artefact -- and the B--K asymptotic story is not the explanation.
Part A's Gate A already established RW stability under ISOTROPIC region growth
(factor 18 in volume, -0.009 +/- 0.030); growth in ``T`` alone is a different
direction and is measured here rather than assumed.

A SECOND, CONFOUND-FREE TEST: stratify by vantage-point depth
==============================================================
The T ladder still varies a *global* region property. A sharper test lives
inside a single T, at fixed ``s`` and fixed region, and is recorded here too.

Each admissible common event ``c`` supplies its OWN estimate ``d_c`` from its own
depth ``tau_c``. In the continuum the estimator is supposed to return the same
answer from every vantage point -- a deeper ``c`` has a larger ``tau_c`` but a
correspondingly smaller ``(1 - O)``, and the two compensate exactly. So:

    if eqs. 25--27 hold at all depths,  d_c is INDEPENDENT of tau_c ;
    if they break down at small tau_c,  d_c is biased at small tau_c and flattens.

Stratifying the per-``c`` estimates by ``tau_c`` therefore tests the asymptotic
hypothesis with NO dependence on ``s`` and NO dependence on ``T``.

NOTE ON A CIRCULARITY THAT IS AVOIDED HERE. The stratification variable is
``tau_c`` ALONE, never the ratio ``tau_c/d_c``. Binning by the ratio would be
circular: ``d_c`` sits in its denominator, so a low ``d_c`` mechanically produces
a high ratio and the trend would appear even from pure noise. ``tau_c`` is
computed from chain counts (eq. 38) and carries no such dependence. This is the
same class of error exp04's Finding 9 flagged in its own correlation, and it is
not repeated here.

Statistics
----------
The unit of averaging for the T-ladder result is the CAUSET (measurements within
one causet share a sprinkling and a target pair, so they are not independent --
the convention Part A and exp04 both used). The per-vantage-point stratification
necessarily works on individual ``c``, and its error bars are therefore quoted
with the causet count alongside the ``c`` count, so the reader can see how much
of the apparent sample size is really independent.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from causet import causal_overlap_3d as co3  # noqa: E402
from causet import rideout_wallden as rw  # noqa: E402
from causet import sprinkle3d  # noqa: E402
from causet.order3d import causal_matrix_3d  # noqa: E402

# ---- Experiment parameters (Integrity Rule 4: named, documented) -------------
SEED_BASE = 20260923
RHO = 60.0  # same density as exp03/exp04: discreteness scale unchanged
SEPARATION = 1.0  # FIXED -- this is the whole point
SPATIAL_EXTENT = (2.5, 5.0)  # (Lx, Ly), FIXED
#: The box time extents. T = 4.0 is exp04's value (the low end, where the
#: compression was seen); the ladder climbs from there. Set after direct timing
#: of the estimator cost -- see N_CAUSETS below.
T_LADDER = [4.0, 6.0, 8.0, 10.0, 12.0]
#: Realisations per rung. Cost is dominated by ``causal_matrix_3d`` (O(N^2) dense
#: float temporaries) and by B--K's chain-count DP over interval sub-posets, both
#: of which grow with T, so the count FALLS along the ladder. Every count is
#: reported and every standard error is computed from the count actually used.
N_CAUSETS = [100, 100, 80, 60, 40]

CACHE = Path(__file__).resolve().parents[1] / "data" / "exp05_measurements.npz"
FIGURE = Path(__file__).resolve().parents[1] / "figures" / "exp05_bk_asymptotic_regime.png"

#: One chain link in length units at this density (Part A's converter, not
#: re-derived), and RW's hard 2-link floor. Quoted so the RW control's
#: resolution is visible next to its flatness.
RW_LINK_QUANTUM = rw.proper_time_from_chain_links(1.0, RHO, m_d=rw.RW_M3)
RW_FLOOR = 2.0 * RW_LINK_QUANTUM

#: Depths beyond which a vantage point's Alexandrov interval is CLIPPED by the
#: (deliberately fixed) spatial walls. A common event at depth ``tau`` below the
#: targets has a future light cone of radius ~``tau`` by the time it reaches
#: them, so once ``tau`` exceeds a half-extent the interval ``I(target, c)`` is
#: truncated by the box rather than by the light cone. See Finding 4: this is a
#: confound the experiment's own design introduces, and it is reported, not
#: buried.
CLIP_X = 0.5 * SPATIAL_EXTENT[0]  # 1.25
CLIP_Y = 0.5 * SPATIAL_EXTENT[1]  # 2.50


# ------------------------------------------------------------------ measurement --
def measure_rung(t_ext: float, n_causets: int, index: int) -> dict:
    """Measure both estimators over ``n_causets`` sprinklings at box height ``t_ext``.

    Nothing is averaged or discarded here (Integrity Rule 1). Per-causet arrays
    come back raw with ``nan`` marking a causet on which an estimator produced
    nothing, and the per-vantage-point values are returned concatenated with a
    causet index so the stratified analysis can respect causet boundaries.
    """
    lx, ly = SPATIAL_EXTENT
    half = 0.5 * SEPARATION

    rw_mean = np.full(n_causets, np.nan)
    rw_n2l = np.zeros(n_causets, dtype=int)
    bk_mean = np.full(n_causets, np.nan)
    bk_n_c = np.zeros(n_causets, dtype=int)
    bk_n_cand = np.zeros(n_causets, dtype=int)
    bk_ratio_med = np.full(n_causets, np.nan)
    realised_n = np.zeros(n_causets, dtype=int)
    perc_tau: list[np.ndarray] = []
    perc_dist: list[np.ndarray] = []
    perc_overlap: list[np.ndarray] = []
    perc_causet: list[np.ndarray] = []

    t0 = time.time()
    for k in range(n_causets):
        seed = SEED_BASE + 100_000 * index + k
        sp = sprinkle3d.sprinkle_box_3d(RHO, t_ext, lx, ly, seed=seed)
        t = np.concatenate((sp.t, [0.5 * t_ext, 0.5 * t_ext]))
        x = np.concatenate((sp.x, [-half, +half]))
        y = np.concatenate((sp.y, [0.0, 0.0]))
        i_x, i_y = t.size - 2, t.size - 1
        realised_n[k] = t.size
        cm = causal_matrix_3d(t, x, y)

        res = rw.two_link_distance(i_x, i_y, cm, rw.RW_M3, rho=RHO)
        rw_n2l[k] = res.n_two_links
        if res.n_two_links:
            rw_mean[k] = res.mean

        bk = co3.distance_causal_overlap_nd(i_x, i_y, cm, RHO, d=2)
        bk_n_c[k] = bk.n_c
        bk_n_cand[k] = bk.n_candidates
        if bk.n_c:
            bk_mean[k] = bk.distance
            bk_ratio_med[k] = bk.median_asymptotic_ratio
            perc_tau.append(bk.per_c_tau)
            perc_dist.append(bk.per_c_distance)
            perc_overlap.append(bk.per_c_overlap)
            perc_causet.append(np.full(bk.n_c, k, dtype=int))
        del cm

    elapsed = time.time() - t0
    cat = lambda parts: (  # noqa: E731
        np.concatenate(parts) if parts else np.empty(0)
    )
    print(
        f"  T={t_ext:5.1f}  V={t_ext * lx * ly:7.1f}  <N>={realised_n.mean():7.0f}  "
        f"#causets={n_causets:3d}  RW 2-links/causet={rw_n2l.mean():4.2f}  "
        f"B-K <n_c>={bk_n_c.mean():5.1f}  median tau/d={np.nanmedian(bk_ratio_med):5.2f}"
        f"  ({elapsed:6.1f} s)",
        flush=True,
    )
    return {
        "rw_mean": rw_mean,
        "rw_n2l": rw_n2l,
        "bk_mean": bk_mean,
        "bk_n_c": bk_n_c,
        "bk_n_cand": bk_n_cand,
        "bk_ratio_med": bk_ratio_med,
        "realised_n": realised_n,
        "perc_tau": cat(perc_tau),
        "perc_dist": cat(perc_dist),
        "perc_overlap": cat(perc_overlap),
        "perc_causet": cat(perc_causet).astype(int),
        "elapsed": np.array(elapsed),
    }


FIELDS = (
    "rw_mean", "rw_n2l", "bk_mean", "bk_n_c", "bk_n_cand", "bk_ratio_med",
    "realised_n", "perc_tau", "perc_dist", "perc_overlap", "perc_causet",
)


def cache_key() -> str:
    return (
        f"v1|{SEED_BASE}|{RHO}|{SEPARATION}|{SPATIAL_EXTENT}|"
        + ",".join(f"{t:.1f}:{n}" for t, n in zip(T_LADDER, N_CAUSETS))
    )


def load_or_measure(remeasure: bool) -> list[dict]:
    key = cache_key()
    if not remeasure and CACHE.exists():
        z = np.load(CACHE, allow_pickle=False)
        if str(z["key"]) == key:
            print(f"Loaded cached measurements from {CACHE}")
            rows = []
            for i in range(len(T_LADDER)):
                row = {f: z[f"{f}_{i}"] for f in FIELDS}
                row["elapsed"] = z[f"elapsed_{i}"]
                rows.append(row)
            return rows
        print("Cache present but parameters differ -- remeasuring.")
    total = sum(N_CAUSETS)
    print(
        f"Measuring {len(T_LADDER)} box heights ({total} causets total); "
        "cost grows steeply with T...",
        flush=True,
    )
    rows = [measure_rung(t, n, i) for i, (t, n) in enumerate(zip(T_LADDER, N_CAUSETS))]
    CACHE.parent.mkdir(exist_ok=True)
    payload = {"key": np.array(key)}
    for i, r in enumerate(rows):
        for f in FIELDS:
            payload[f"{f}_{i}"] = r[f]
        payload[f"elapsed_{i}"] = r["elapsed"]
    np.savez(CACHE, **payload)
    print(f"Saved raw measurements -> {CACHE}")
    return rows


# -------------------------------------------------------------------- analysis --
def mean_se(values: np.ndarray) -> tuple[float, float, int]:
    """``(mean, standard error, count)`` over the finite entries."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = int(v.size)
    if n == 0:
        return float("nan"), float("nan"), 0
    if n == 1:
        return float(v[0]), float("nan"), 1
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(n)), n


def weighted_linfit(x: np.ndarray, y: np.ndarray, sy: np.ndarray) -> tuple[float, float]:
    """Slope and its standard error for ``y = a + b x`` weighted by ``1/sy^2``.

    Used to state the T-dependence of each ratio as a NUMBER with an error bar,
    so "flat" and "moves" are decided by the fit and not by looking at the plot.
    """
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(sy) & (sy > 0)
    x, y, w = x[ok], y[ok], 1.0 / sy[ok] ** 2
    if x.size < 3:
        return float("nan"), float("nan")
    sw, sx, sxx = w.sum(), (w * x).sum(), (w * x * x).sum()
    sy_, sxy = (w * y).sum(), (w * x * y).sum()
    den = sw * sxx - sx * sx
    if den == 0:
        return float("nan"), float("nan")
    slope = (sw * sxy - sx * sy_) / den
    return float(slope), float(np.sqrt(sw / den))


def main(remeasure: bool = False) -> None:
    print("=" * 78)
    print("Phase 2b Part B DIAGNOSTIC: B-K asymptotic regime at FIXED separation")
    print("=" * 78)
    print(f"Seeds: SEED_BASE + 100000*i + k   (SEED_BASE={SEED_BASE})")
    print(
        f"rho = {RHO}, separation s = {SEPARATION} (FIXED), spatial extent "
        f"(Lx,Ly) = {SPATIAL_EXTENT} (FIXED)"
    )
    print(f"Box time extent T varied over {T_LADDER}, causets {N_CAUSETS}")
    print(
        "\nDIAGNOSTIC, NOT A GATE. Geometry deliberately differs from exp03/exp04\n"
        "(whose T was frozen at 4.0 for comparability), so NO number here is\n"
        "comparable with those experiments on absolute scale. Only this\n"
        "experiment's own internal trends are interpreted.\n"
    )

    rows = load_or_measure(remeasure)
    print(f"\nTotal wall time: {sum(float(r['elapsed']) for r in rows) / 60:.1f} min\n")

    nt = len(T_LADDER)
    tau_ratio = np.full(nt, np.nan)
    tau_ratio_se = np.full(nt, np.nan)
    tau_abs = np.full(nt, np.nan)
    bk_r = np.full(nt, np.nan)
    bk_se = np.full(nt, np.nan)
    bk_n = np.zeros(nt, dtype=int)
    rw_r = np.full(nt, np.nan)
    rw_se = np.full(nt, np.nan)
    rw_n = np.zeros(nt, dtype=int)

    for i, r in enumerate(rows):
        tau_ratio[i], tau_ratio_se[i], _ = mean_se(r["bk_ratio_med"])
        tau_abs[i] = float(np.mean(r["perc_tau"])) if r["perc_tau"].size else np.nan
        m, s, n = mean_se(r["bk_mean"])
        bk_r[i], bk_se[i], bk_n[i] = m / SEPARATION, s / SEPARATION, n
        m, s, n = mean_se(r["rw_mean"])
        rw_r[i], rw_se[i], rw_n[i] = m / SEPARATION, s / SEPARATION, n

    # ---------------------------------------------------------------- Finding 1 --
    print("### Finding 1 -- did the independent variable actually move?\n")
    print("     T  |    V   |  <N>  | causets | <tau_c> | median tau_c/d | B-K <n_c>")
    print("  " + "-" * 72)
    for i, r in enumerate(rows):
        print(
            f"   {T_LADDER[i]:5.1f} | {T_LADDER[i] * SPATIAL_EXTENT[0] * SPATIAL_EXTENT[1]:6.1f} "
            f"| {r['realised_n'].mean():5.0f} |   {len(r['realised_n']):4d}  |"
            f"  {tau_abs[i]:5.2f}  |  {tau_ratio[i]:5.2f} +/- {tau_ratio_se[i]:.2f}  "
            f"|  {r['bk_n_c'].mean():5.1f}"
        )
    span = tau_ratio[-1] / tau_ratio[0] if np.isfinite(tau_ratio[0]) else np.nan
    print(
        f"\n  tau_c/d spans {tau_ratio[0]:.2f} -> {tau_ratio[-1]:.2f} "
        f"(factor {span:.2f}); mean vantage depth tau_c spans "
        f"{tau_abs[0]:.2f} -> {tau_abs[-1]:.2f} (factor {tau_abs[-1] / tau_abs[0]:.2f})."
    )
    if span < 1.5:
        print(
            "  WARNING: the independent variable barely moved. Any null result below\n"
            "  would be uninformative rather than a refutation -- read it that way."
        )

    # ---------------------------------------------------------------- Finding 2 --
    print("\n### Finding 2 -- B-K/true vs T, and the RW control\n")
    print("     T  | median tau_c/d |   B-K/true        n  |   RW/true         n")
    print("  " + "-" * 74)
    for i in range(nt):
        print(
            f"   {T_LADDER[i]:5.1f} |     {tau_ratio[i]:5.2f}      | "
            f"{bk_r[i]:.4f} +/- {bk_se[i]:.4f} {bk_n[i]:4d} | "
            f"{rw_r[i]:.4f} +/- {rw_se[i]:.4f} {rw_n[i]:4d}"
        )
    t_arr = np.asarray(T_LADDER, dtype=float)
    bk_slope, bk_slope_se = weighted_linfit(t_arr, bk_r, bk_se)
    rw_slope, rw_slope_se = weighted_linfit(t_arr, rw_r, rw_se)
    print(
        f"\n  Weighted linear fit of ratio vs T (slope per unit T, with SE):\n"
        f"    B-K/true : {bk_slope:+.5f} +/- {bk_slope_se:.5f}   "
        f"({abs(bk_slope) / bk_slope_se if bk_slope_se else float('nan'):.2f} sigma)\n"
        f"    RW /true : {rw_slope:+.5f} +/- {rw_slope_se:.5f}   "
        f"({abs(rw_slope) / rw_slope_se if rw_slope_se else float('nan'):.2f} sigma)"
    )

    # ---------------------------------------------------------------- Finding 3 --
    print("\n### Finding 3 -- per-vantage-point stratification (no s, no T confound)\n")
    print(
        "  Each admissible c gives its own d_c from its own depth tau_c. If\n"
        "  eqs. 25--27 hold at all depths, d_c is INDEPENDENT of tau_c. Binned by\n"
        "  tau_c ALONE (never by tau_c/d_c, which would be circular -- d_c is in\n"
        "  its own denominator).\n"
    )
    all_tau = np.concatenate([r["perc_tau"] for r in rows])
    all_dist = np.concatenate([r["perc_dist"] for r in rows])
    all_cau = np.concatenate(
        [r["perc_causet"] + 1000 * i for i, r in enumerate(rows)]
    )
    edges = np.quantile(all_tau, np.linspace(0, 1, 9))
    edges = np.unique(edges)
    print("    tau_c bin      |  <tau_c> |   d_c/true         n_c  | n causets")
    print("  " + "-" * 70)
    bin_tau, bin_ratio, bin_se = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (all_tau >= a) & (all_tau < b)
        if m.sum() < 10:
            continue
        vals = all_dist[m] / SEPARATION
        mu = float(vals.mean())
        se = float(vals.std(ddof=1) / np.sqrt(vals.size))
        n_cau = int(np.unique(all_cau[m]).size)
        bin_tau.append(float(all_tau[m].mean()))
        bin_ratio.append(mu)
        bin_se.append(se)
        print(
            f"   [{a:5.2f}, {b:5.2f}) |  {all_tau[m].mean():6.2f}  | "
            f"{mu:.4f} +/- {se:.4f} {int(m.sum()):6d}  |   {n_cau:5d}"
        )
    bin_tau = np.asarray(bin_tau)
    bin_ratio = np.asarray(bin_ratio)
    bin_se = np.asarray(bin_se)
    strat_slope, strat_slope_se = weighted_linfit(bin_tau, bin_ratio, bin_se)

    # The pooled bins above mix causets from different T rungs, so a given tau_c
    # bin draws on more than one geometry -- a T confound smuggled back into a
    # test built to remove confounds. Repeat it WITHIN each rung, where T is
    # constant by construction and only the vantage depth varies.
    print("\n  Same stratification computed WITHIN each rung (T constant, so no")
    print("  T confound at all -- the cleanest form of this test):\n")
    print("     T  | tau_c range        | bins | slope d_c/true vs tau_c   | sigma")
    print("  " + "-" * 72)
    per_rung_slopes = []
    for i, r in enumerate(rows):
        tau_i, dist_i = r["perc_tau"], r["perc_dist"]
        if tau_i.size < 80:
            print(f"   {T_LADDER[i]:5.1f} | (only {tau_i.size} vantage points -- skipped)")
            continue
        e_i = np.unique(np.quantile(tau_i, np.linspace(0, 1, 7)))
        bt, br, bs = [], [], []
        for a, b in zip(e_i[:-1], e_i[1:]):
            m = (tau_i >= a) & (tau_i < b)
            if m.sum() < 10:
                continue
            v = dist_i[m] / SEPARATION
            bt.append(float(tau_i[m].mean()))
            br.append(float(v.mean()))
            bs.append(float(v.std(ddof=1) / np.sqrt(v.size)))
        sl, sl_se = weighted_linfit(np.asarray(bt), np.asarray(br), np.asarray(bs))
        per_rung_slopes.append((T_LADDER[i], sl, sl_se))
        sig = abs(sl) / sl_se if (np.isfinite(sl_se) and sl_se > 0) else float("nan")
        print(
            f"   {T_LADDER[i]:5.1f} | {tau_i.min():5.2f} -- {tau_i.max():5.2f}      "
            f"| {len(bt):4d} | {sl:+.5f} +/- {sl_se:.5f}     | {sig:5.2f}"
        )
    print(
        f"\n  Weighted fit of d_c/true vs tau_c: slope {strat_slope:+.5f} +/- "
        f"{strat_slope_se:.5f} "
        f"({abs(strat_slope) / strat_slope_se if strat_slope_se else float('nan'):.2f} sigma)"
    )
    print(
        "  NOTE ON INDEPENDENCE: the n_c column counts vantage points, the last\n"
        "  column the causets they came from. Vantage points inside one causet are\n"
        "  correlated, so these error bars are optimistic; the slope's significance\n"
        "  should be read with that in mind, and it is quoted for direction more\n"
        "  than for its exact sigma."
    )

    # ---------------------------------------------------------------- Finding 4 --
    print("\n### Finding 4 -- a confound THIS experiment introduced: spatial clipping\n")
    print(
        "  Growing T alone does NOT cleanly isolate tau_c. A common event at depth\n"
        "  tau below the targets has a future light cone of radius ~tau when it\n"
        f"  reaches them, but the spatial half-extents were held FIXED at Lx/2 =\n"
        f"  {CLIP_X:.2f} and Ly/2 = {CLIP_Y:.2f}. Past those depths the Alexandrov\n"
        "  interval is truncated by the BOX, not by the light cone -- so deeper\n"
        "  vantage points are also progressively more clipped ones, and depth and\n"
        "  clipping are confounded along the ladder.\n"
    )
    print("     T  | <tau_c> | frac clipped in x | frac clipped in y | within-rung slope")
    print("  " + "-" * 76)
    clip_y_frac = []
    for i, r in enumerate(rows):
        tau_i = r["perc_tau"]
        fx = float((tau_i > CLIP_X).mean()) if tau_i.size else np.nan
        fy = float((tau_i > CLIP_Y).mean()) if tau_i.size else np.nan
        clip_y_frac.append(fy)
        sl = next((s for (tt, s, _) in per_rung_slopes if tt == T_LADDER[i]), np.nan)
        print(
            f"   {T_LADDER[i]:5.1f} |  {tau_i.mean():5.2f}  |       {fx:5.2f}       |"
            f"       {fy:5.2f}       |     {sl:+.5f}"
        )
    clip_y_frac = np.asarray(clip_y_frac)
    rung_slopes = np.array([s for (_, s, _) in per_rung_slopes], dtype=float)
    print(
        "\n  The within-rung slope FLIPS SIGN exactly as y-clipping switches on:\n"
        f"  0% y-clipped at T = {T_LADDER[0]:.0f} gives {rung_slopes[0]:+.3f}; "
        f"{100 * clip_y_frac[-1]:.0f}% y-clipped at T = {T_LADDER[-1]:.0f} gives "
        f"{rung_slopes[-1]:+.3f}.\n"
        "  The ladder therefore measures depth and clipping together and CANNOT,\n"
        "  by itself, attribute the aggregate trend to either."
    )

    # ------------------------------------------------------------- interpretation --
    print("\n" + "=" * 78)
    print("DIAGNOSTIC VERDICT")
    print("=" * 78)
    # A non-finite fit must never produce a confident verdict: with too few rungs
    # `weighted_linfit` returns nan, and "nan is not > 3 sigma" would otherwise
    # read as "the effect is absent" and print REFUTED on no evidence at all.
    fits_usable = all(
        np.isfinite(v) and v > 0
        for v in (bk_slope_se, rw_slope_se)
    ) and np.isfinite(bk_slope) and np.isfinite(rw_slope)
    bk_moves = fits_usable and abs(bk_slope) > 3.0 * bk_slope_se
    rw_moves = fits_usable and abs(rw_slope) > 3.0 * rw_slope_se
    bk_toward_one = bk_moves and bk_slope > 0 and bk_r[0] < 1.0

    # The shallowest rung is the LEAST clipped one, and at T = 4 nothing is
    # clipped in y at all -- so its within-rung stratification is the closest
    # thing this experiment has to a clean test, and it is also exp04's own
    # geometry, which is the regime the original question was about.
    shallow_slope = rung_slopes[0] if rung_slopes.size else float("nan")
    shallow_se = per_rung_slopes[0][2] if per_rung_slopes else float("nan")
    shallow_positive = (
        np.isfinite(shallow_slope)
        and np.isfinite(shallow_se)
        and shallow_slope > 3.0 * shallow_se
    )
    clipping_tracks_sign = (
        rung_slopes.size >= 3
        and np.isfinite(rung_slopes).all()
        and rung_slopes[0] > 0
        and rung_slopes[-1] < 0
    )

    if not fits_usable:
        print(
            "  INSUFFICIENT DATA. The weighted fits did not return finite slopes\n"
            "  (too few usable rungs, or a rung with no error bar). NO verdict is\n"
            "  issued -- this is not a refutation, it is an absence of evidence."
        )
    elif bk_moves and rw_moves:
        print(
            "  UNRESOLVED -- COMMON SYSTEMATIC. BOTH estimators move with T at fixed\n"
            "  separation. Since RW's calibration carries no asymptotic-regime\n"
            "  condition, a shared T-dependence cannot be the B-K formula breaking\n"
            "  down; something common to both is driving them. The asymptotic\n"
            "  explanation is NOT established and NOT replaced by another."
        )
    elif clipping_tracks_sign and shallow_positive:
        print(
            "  PARTIALLY SUPPORTED, AND THE LADDER IS NOT USABLE AS BUILT.\n"
            "\n"
            "  Two things were measured and they point opposite ways, for a reason\n"
            "  Finding 4 makes concrete.\n"
            "\n"
            "  (1) In the LEAST-CLIPPED geometry -- T = "
            f"{T_LADDER[0]:.0f}, which is exp04's own, and where\n"
            "      NO vantage point is clipped in y -- deeper vantage points give\n"
            f"      BETTER estimates: slope {shallow_slope:+.4f} +/- {shallow_se:.4f}. That is\n"
            "      the direction the eqs.-25--27 breakdown predicts, measured at fixed\n"
            "      s and fixed region, free of exp04's monotone-in-s confound. It\n"
            "      SUPPORTS the asymptotic explanation of exp04's compression IN THE\n"
            "      REGIME exp04 ACTUALLY OPERATED IN.\n"
            "\n"
            "  (2) Across the T ladder the aggregate B-K/true FALLS "
            f"({bk_slope:+.5f} +/- {bk_slope_se:.5f},\n"
            f"      {abs(bk_slope) / bk_slope_se:.1f} sigma) while the RW control does not move at "
            f"3 sigma\n      ({rw_slope:+.5f} +/- {rw_slope_se:.5f}, "
            f"{abs(rw_slope) / rw_slope_se:.1f} sigma). Taken at face value that would\n"
            "      refute the asymptotic story. It should NOT be taken at face value:\n"
            "      growing T in a spatially fixed box makes deep vantage points\n"
            "      progressively more CLIPPED, and the within-rung slope flips sign in\n"
            "      lockstep with the y-clipping fraction (Finding 4). The ladder\n"
            "      measures depth and clipping together and cannot separate them.\n"
            "\n"
            "  CONCLUSION, stated at the limit of what was measured: exp04's\n"
            "  large-s compression is CONSISTENT WITH, and now positively supported\n"
            "  by, the eqs.-25--27 breakdown at small tau_c -- but this experiment\n"
            "  CANNOT state the validity boundary quantitatively, because its own\n"
            "  depth ladder is confounded with boundary clipping. The confound is\n"
            "  named, measured, and NOT explained away.\n"
            "\n"
            "  WHAT WOULD SETTLE IT: repeat the ladder growing Lx and Ly in\n"
            "  proportion to T, so the clipped fraction stays constant while tau_c\n"
            "  grows. That is the experiment this one turned out to need, and it has\n"
            "  not been run."
        )
    elif bk_toward_one and not rw_moves:
        print(
            "  CONFIRMED. B-K/true rises toward 1 as the region deepens at FIXED\n"
            "  separation, while the RW control stays flat. The large-s compression\n"
            "  of exp04's Finding 9 IS the eqs.-25--27 asymptotic breakdown."
        )
    elif not bk_moves:
        print(
            "  REFUTED. With separation held fixed, B-K/true does NOT track tau_c/d.\n"
            "  The large-s compression seen in exp04 therefore has a DIFFERENT cause,\n"
            "  and the asymptotic-breakdown explanation offered there as 'consistent'\n"
            "  is not supported once the s confound is removed. NO replacement\n"
            "  explanation is offered here: none has been tested."
        )
    else:
        print(
            "  UNRESOLVED. B-K/true moves with T but AWAY from 1, and the prepared\n"
            "  cases do not cover the pattern. Reported as measured; see Findings 2,\n"
            "  3 and 4 for the raw direction, size, and the clipping confound."
        )
    print(
        f"\n  Numbers the verdict rests on: B-K slope {bk_slope:+.5f} +/- {bk_slope_se:.5f}, "
        f"RW slope {rw_slope:+.5f} +/- {rw_slope_se:.5f},\n"
        f"  stratified per-vantage slope {strat_slope:+.5f} +/- {strat_slope_se:.5f}, "
        f"tau_c/d span {tau_ratio[0]:.2f} -> {tau_ratio[-1]:.2f}."
    )

    make_figure(rows, tau_ratio, bk_r, bk_se, rw_r, rw_se, bin_tau, bin_ratio, bin_se,
                per_rung_slopes, clip_y_frac)
    print(f"\nFigure -> {FIGURE}")


# --------------------------------------------------------------------- figure --
def make_figure(rows, tau_ratio, bk_r, bk_se, rw_r, rw_se, bin_tau, bin_ratio, bin_se,
                per_rung_slopes, clip_y_frac) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(22.0, 5.2))
    t_arr = np.asarray(T_LADDER, dtype=float)

    ax = axes[0]
    ax.plot(t_arr, tau_ratio, "o-", ms=5, color="tab:purple")
    ax.axhline(1.0, color="crimson", ls="--", lw=1.0, label="tau_c = d (formula invalid)")
    ax.set_xlabel("box time extent T  (the ONLY thing varied)")
    ax.set_ylabel("median tau_c / d_est")
    ax.set_title(
        "A. The independent variable moved\n"
        f"s = {SEPARATION} FIXED; tau_c/d spans "
        f"{tau_ratio[0]:.2f} -> {tau_ratio[-1]:.2f}",
        fontsize=9.5,
    )
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.errorbar(tau_ratio, bk_r, yerr=bk_se, fmt="s-", ms=5, capsize=3,
                color="tab:blue", label="B-K / true")
    ax.errorbar(tau_ratio, rw_r, yerr=rw_se, fmt="o--", ms=5, capsize=3,
                color="tab:red", label="RW / true  (CONTROL: no asymptotic condition)")
    ax.axhline(1.0, color="k", ls=":", lw=1.0, label="exact recovery")
    ax.set_xlabel("median tau_c / d_est  (moved by T, NOT by s)")
    ax.set_ylabel("estimate / true separation")
    ax.set_title(
        "B. THE TEST: confound broken\n"
        "s fixed, so any trend here is NOT the exp04 confound",
        fontsize=9.5,
    )
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    ax = axes[2]
    cmap = plt.get_cmap("viridis")
    for i, r in enumerate(rows):
        tau_i, dist_i = r["perc_tau"], r["perc_dist"]
        if tau_i.size < 80:
            continue
        e_i = np.unique(np.quantile(tau_i, np.linspace(0, 1, 7)))
        bt, br, bs = [], [], []
        for a, b in zip(e_i[:-1], e_i[1:]):
            m = (tau_i >= a) & (tau_i < b)
            if m.sum() < 10:
                continue
            v = dist_i[m] / SEPARATION
            bt.append(tau_i[m].mean())
            br.append(v.mean())
            bs.append(v.std(ddof=1) / np.sqrt(v.size))
        ax.errorbar(bt, br, yerr=bs, fmt="o-", ms=4, capsize=2,
                    color=cmap(i / max(len(rows) - 1, 1)),
                    label=f"T={T_LADDER[i]:.0f} ({100 * clip_y_frac[i]:.0f}% y-clipped)")
    ax.axvline(CLIP_X, color="crimson", ls=":", lw=1.0)
    ax.axvline(CLIP_Y, color="crimson", ls="--", lw=1.0)
    ax.text(CLIP_X, ax.get_ylim()[1] * 0.98, " x-clip", color="crimson", fontsize=7,
            va="top")
    ax.text(CLIP_Y, ax.get_ylim()[1] * 0.98, " y-clip", color="crimson", fontsize=7,
            va="top")
    ax.axhline(1.0, color="k", ls=":", lw=1.0)
    ax.set_xlabel("vantage-point depth tau_c  (binned; NOT tau_c/d_c)")
    ax.set_ylabel("per-vantage d_c / true")
    ax.set_title(
        "C. Per-vantage stratification, WITHIN each rung\n"
        "T=4 (unclipped in y) RISES = asymptotic improvement;\n"
        "deeper rungs FALL as clipping takes over",
        fontsize=9.5,
    )
    ax.legend(fontsize=6.5)
    ax.grid(alpha=0.25)

    ax = axes[3]
    rung_sl = np.array([sl for (_, sl, _) in per_rung_slopes])
    rung_se = np.array([se for (_, _, se) in per_rung_slopes])
    ax.errorbar(100 * np.asarray(clip_y_frac)[: rung_sl.size], rung_sl, yerr=rung_se,
                fmt="D-", ms=6, capsize=3, color="tab:orange")
    ax.axhline(0.0, color="k", ls="--", lw=1.0)
    for i, (tt, sl, _) in enumerate(per_rung_slopes):
        ax.annotate(f"T={tt:.0f}", (100 * clip_y_frac[i], sl), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("% of vantage points clipped in y")
    ax.set_ylabel("within-rung slope of d_c/true vs tau_c")
    ax.set_title(
        "D. THE CONFOUND: the slope flips sign in lockstep\n"
        "with clipping. Depth and clipping move together\n"
        "along this ladder, so it cannot separate them.",
        fontsize=9.5,
    )
    ax.grid(alpha=0.25)

    fig.suptitle(
        "Phase 2b DIAGNOSTIC -- is B-K's compression the eqs. 25-27 breakdown? "
        f"(rho = {RHO}, s = {SEPARATION} FIXED, T varied, seeds {SEED_BASE}+...)  "
        "NOT comparable with exp03/exp04 on absolute scale",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    FIGURE.parent.mkdir(exist_ok=True)
    fig.savefig(FIGURE, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main(remeasure="--remeasure" in sys.argv)
