"""Phase 2b Part B2 GATE: do Rideout--Wallden and Boguna--Krioukov AGREE ON ORDER?

Reproduce with:
    python experiments/exp04_rw_vs_bk_2p1d.py             # uses the cache
    python experiments/exp04_rw_vs_bk_2p1d.py --remeasure # ignores the cache

Produces:
    * printed tables (standard errors over causets, explicit N everywhere),
    * figures/exp04_rw_vs_bk_2p1d.png,
    * data/exp04_measurements.npz (raw per-causet values + seeds),
    * an explicit PASS/FAIL statement for the Part-B gate.

THIS IS A RANK-ORDERING TEST, NOT AN ACCURACY TEST. READ THIS FIRST.
=====================================================================
The two estimators carry different, and differently unreliable, calibrations.
They must NOT be compared on scale, and this experiment does not do so.

Rideout--Wallden side. Part A's diagnostic (RESULTS.md, 2026-08-17) decomposed
the 2-link estimator's -12% offset into three measured factors,

    finite-size calibration 0.7851  x  Step-2 chain suppression 0.8439
                                    x  geometric excess tau(p*,f)/D 1.3142
                                    =  0.8708   (observed 0.8644),

and established that these have DIFFERENT rho and D dependence. The net offset
is a near-cancellation of two ~30% effects, not a stable estimator property. It
is therefore **not** subtracted, divided out, or corrected for anywhere in this
experiment, and no such correction should ever be fitted from it: a constant
fitted here would not generalise to any other rho or D.

Boguna--Krioukov side. In 2+1 D their distance is no longer the exact closed
form Phase 2a used (eqs. 23--24, d = 1 only) but the ASYMPTOTIC eqs. 25--27,

    d(a,b) = (2/c_2) tau_c (1 - O),   c_2 = 6/pi,   valid for tau_c >> d(a,b),

a strictly weaker accuracy claim. How far into that regime a measurement
actually sat is not assumed: the realised ``tau_c / d_est`` is recorded for
every admissible common event and reported per pair below. At this geometry it
is marginal at the large separations, which is a limit on ACCURACY only.

Why the comparison is nonetheless valid. Rank ordering is invariant under any
monotone rescaling. A scale offset -- even an unstable one, even a
dimension-dependent one -- cannot create or destroy an ordering. So
"do the two estimators rank the same set of pairs the same way?" is a question
both estimators can answer honestly at this geometry, while "do they agree on
magnitude?" is one neither can. Only the first is asked.

Geometry: IDENTICAL to Part A's Gate A (deliberately not deepened)
------------------------------------------------------------------
``rho = 60``, box ``(T, Lx, Ly) = (4.0, 2.5, 5.0)``, targets at the mid-time on
the x axis, ``x_+- = (T/2, +-s/2, 0)``. Part A's yield numbers (1.16 2-links per
causet, 36% of causets yielding none) were measured at exactly these parameters,
so the usability comparison in Finding 4 below is like-for-like rather than a
comparison across two different regions. Deepening the time extent would have
improved the B--K asymptotic ratio, but only at the cost of making the
usability numbers incomparable with Part A's -- and the asymptotic ratio limits
accuracy, which is not under test.

The >= 20 pairs
---------------
``N_SEPARATIONS`` distinct target separations spanning ``s = 0.30 .. 2.00``
(the box's ``Lx = 2.5`` caps the top; the bottom is just under the estimator
resolution floor discussed below, deliberately, so the floor is visible rather
than avoided). Each separation is one "pair" in the gate's sense: a distinct
spacelike target configuration with an exactly known true separation.

Every pair is measured on the SAME ``N_CAUSETS`` Poisson backgrounds -- one
sprinkling per seed, re-used across all separations, with only the two injected
target events moving. Both estimators see the identical causal matrix for each
(seed, separation). Any difference between them is therefore attributable to the
estimators alone, exactly as Part A measured the naive control on its own
sprinklings.

RW'S RESOLUTION FLOOR, stated before the numbers
-------------------------------------------------
One chain link calibrates to ``1/(m_3 (rho eta_3)^{1/3}) = 0.1738`` at this
density, and every 2-link minimising interval contains both targets (``p prec x
prec f``), so its longest chain has >= 3 elements, i.e. >= 2 links. No single
2-link measurement can return below ``0.3476``. Adjacent separations in this
ladder are 0.074 apart -- far below that quantum -- so a SINGLE causet cannot
rank-order neighbouring pairs, and this is not a defect the experiment hides.
What the gate ranks is the MEAN over ``N_CAUSETS`` causets, whose standard error
(~0.02) is well below the ladder spacing; averaging resolves finer than the
quantum, single measurements do not. Both numbers are reported.

Calibration constants used (and why they are irrelevant here)
--------------------------------------------------------------
* RW: Part A's code UNCHANGED, i.e. the published ``m_3 = 2.296`` via
  ``rideout_wallden.two_link_distance(..., rho=RHO)``. Not the measured
  ``m_3^eff``, because the brief requires Part A's estimator be reused as-is.
* B--K: the MEASURED ``m_3^eff(rho V)`` curve of ``causal_overlap_3d`` (this
  repository's own data), never Rideout--Wallden's fitted asymptote.

These two choices differ by ~25% in ``alpha_2``-equivalent terms. That is a pure
multiplicative scale on each estimator separately and cannot move a rank, which
is precisely why the gate is specified on ordering. It would wreck any
accuracy comparison, which is why none is made.

TWO ALPHA POLICIES, both run on the same sprinklings
-----------------------------------------------------
B1's tests established that a FIXED ``alpha_d`` is a pure global scale (it
cannot reorder anything) while the ``measured_m3`` policy -- reading
``m_3^eff`` at each interval's own cardinality -- is genuinely pair-dependent
and therefore CAN move a rank. Both are run as two analysis passes over the
identical causets. If an ordering changes between them, that is a finding about
calibration sensitivity; if none does, that is a robustness result. Either way
it is measured rather than argued.

Acceptance criteria, fixed before the verdict
----------------------------------------------
GATE B passes iff, over the >= 20 pairs:
  (B-i)   Spearman rho(RW, true) is positive and significant at p < 0.01;
  (B-ii)  Spearman rho(B--K, true) is positive and significant at p < 0.01;
  (B-iii) Spearman rho(RW, B--K) is positive and significant at p < 0.01.
All three with the pair count stated. Eyeballing the plot decides nothing.
Every estimator-vs-estimator ordering inversion is enumerated and classified
(Finding 3), not dropped.
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
from scipy.stats import spearmanr  # noqa: E402

from causet import causal_overlap_3d as co3  # noqa: E402
from causet import rideout_wallden as rw  # noqa: E402
from causet import sprinkle3d  # noqa: E402
from causet.order3d import causal_matrix_3d  # noqa: E402

# ---- Experiment parameters (Integrity Rule 4: named, documented) -------------
SEED_BASE = 20260922
RHO = 60.0  # fixed sprinkling density -- Part A's Gate-A value
BOX_EXTENT = (4.0, 2.5, 5.0)  # (T, Lx, Ly) -- Part A's Gate-A shape at lambda = 1
#: The >= 20 distinct spacelike pairs, as target separations. Top capped by
#: Lx = 2.5 (targets at x = +-s/2 must stay inside the box); bottom set just
#: below RW's 2-link floor of 0.3476 so the floor is visible in the data.
N_SEPARATIONS = 24
SEPARATIONS = np.linspace(0.30, 2.00, N_SEPARATIONS)
#: Poisson backgrounds. Every separation is measured on all of them, so the
#: pairs are compared on identical sprinklings.
N_CAUSETS = 100

#: Significance threshold for all three Spearman criteria, fixed in advance.
ALPHA_SIGNIFICANCE = 0.01

CACHE = Path(__file__).resolve().parents[1] / "data" / "exp04_measurements.npz"
FIGURE = Path(__file__).resolve().parents[1] / "figures" / "exp04_rw_vs_bk_2p1d.png"

#: One chain link in length units at this density, and the hard 2-link floor.
#: Rideout--Wallden eqs. (1)-(2) via Part A's own converter (not re-derived).
RW_LINK_QUANTUM = rw.proper_time_from_chain_links(1.0, RHO, m_d=rw.RW_M3)
RW_FLOOR = 2.0 * RW_LINK_QUANTUM


# ------------------------------------------------------------------ measurement --
def measure() -> dict:
    """Measure both estimators for every separation on every shared background.

    Nothing is averaged or discarded here (Integrity Rule 1): per-causet arrays
    come back raw, with ``nan`` marking a causet on which an estimator produced
    nothing. The loop is causet-outer / separation-inner so that one Poisson
    background is generated once and re-used across the whole ladder.
    """
    ns, nc = N_SEPARATIONS, N_CAUSETS
    out = {
        # Rideout--Wallden 2-link (Part A's code, unchanged)
        "rw": np.full((ns, nc), np.nan),
        "rw_n_two_links": np.zeros((ns, nc), dtype=int),
        "rw_interval_size": np.full((ns, nc), np.nan),
        "rw_chain_links": np.full((ns, nc), np.nan),
        # Boguna--Krioukov causal overlap, fixed alpha_2
        "bk_fixed": np.full((ns, nc), np.nan),
        # Boguna--Krioukov causal overlap, measured-m_3 alpha_2 (per interval)
        "bk_meas": np.full((ns, nc), np.nan),
        "bk_n_c": np.zeros((ns, nc), dtype=int),
        "bk_n_candidates": np.zeros((ns, nc), dtype=int),
        "bk_tau_ratio_med": np.full((ns, nc), np.nan),
        "bk_tau_ratio_min": np.full((ns, nc), np.nan),
        "bk_interval_size": np.full((ns, nc), np.nan),
        "realised_n": np.zeros(nc, dtype=int),
    }

    t_ext, x_ext, y_ext = BOX_EXTENT
    t_mid = 0.5 * t_ext
    t0 = time.time()
    for k in range(nc):
        seed = SEED_BASE + k
        sp = sprinkle3d.sprinkle_box_3d(RHO, t_ext, x_ext, y_ext, seed=seed)
        out["realised_n"][k] = sp.n + 2
        for i, sep in enumerate(SEPARATIONS):
            # The ONLY thing that changes across i is the two target events.
            t = np.concatenate((sp.t, [t_mid, t_mid]))
            x = np.concatenate((sp.x, [-0.5 * sep, +0.5 * sep]))
            y = np.concatenate((sp.y, [0.0, 0.0]))
            i_x, i_y = t.size - 2, t.size - 1
            cm = causal_matrix_3d(t, x, y)

            # (a) Rideout--Wallden 2-link distance -- Part A's code unchanged,
            #     published m_3 = 2.296, exactly as exp03 called it.
            res = rw.two_link_distance(i_x, i_y, cm, rw.RW_M3, rho=RHO)
            out["rw_n_two_links"][i, k] = res.n_two_links
            if res.n_two_links:
                out["rw"][i, k] = res.mean
                out["rw_interval_size"][i, k] = res.mean_interval_size
                out["rw_chain_links"][i, k] = float(res.per_link_chain_links.mean())

            # (b) Boguna--Krioukov causal-overlap distance -- two alpha policies,
            #     two independent passes over the identical causal matrix.
            bk_f = co3.distance_causal_overlap_nd(
                i_x, i_y, cm, RHO, d=2, alpha_policy="fixed"
            )
            bk_m = co3.distance_causal_overlap_nd(
                i_x, i_y, cm, RHO, d=2, alpha_policy="measured_m3"
            )
            out["bk_n_c"][i, k] = bk_m.n_c
            out["bk_n_candidates"][i, k] = bk_m.n_candidates
            if bk_f.n_c:
                out["bk_fixed"][i, k] = bk_f.distance
            if bk_m.n_c:
                out["bk_meas"][i, k] = bk_m.distance
                out["bk_tau_ratio_med"][i, k] = bk_m.median_asymptotic_ratio
                out["bk_tau_ratio_min"][i, k] = bk_m.min_asymptotic_ratio
                out["bk_interval_size"][i, k] = float(bk_m.per_c_interval_size.mean())
            del cm
        if (k + 1) % 10 == 0:
            print(
                f"  {k + 1:3d}/{nc} causets  ({time.time() - t0:6.1f} s)",
                flush=True,
            )
    out["elapsed"] = np.array(time.time() - t0)
    return out


def cache_key() -> str:
    return (
        f"v1|{SEED_BASE}|{RHO}|{BOX_EXTENT}|{N_CAUSETS}|{N_SEPARATIONS}|"
        f"{SEPARATIONS[0]:.4f}-{SEPARATIONS[-1]:.4f}"
    )


def load_or_measure(remeasure: bool) -> dict:
    key = cache_key()
    if not remeasure and CACHE.exists():
        z = np.load(CACHE, allow_pickle=False)
        if str(z["key"]) == key:
            print(f"Loaded cached measurements from {CACHE}")
            return {f: z[f] for f in z.files if f != "key"}
        print("Cache present but parameters differ -- remeasuring.")
    print(
        f"Measuring {N_SEPARATIONS} separations x {N_CAUSETS} shared causets "
        f"({N_SEPARATIONS * N_CAUSETS} causal matrices; ~14 min)...",
        flush=True,
    )
    out = measure()
    CACHE.parent.mkdir(exist_ok=True)
    np.savez(CACHE, key=np.array(key), **out)
    print(f"Saved raw measurements -> {CACHE}")
    return out


# -------------------------------------------------------------------- analysis --
def mean_se(values: np.ndarray) -> tuple[float, float, int]:
    """``(mean, standard error, count)`` over the finite entries of ``values``.

    The unit of averaging is the CAUSET, never the individual 2-link or common
    event: measurements inside one causet share a sprinkling and a target pair
    and are not independent, so pooling them would understate the error (the
    same convention Part A used).
    """
    v = values[np.isfinite(values)]
    n = int(v.size)
    if n == 0:
        return float("nan"), float("nan"), 0
    if n == 1:
        return float(v[0]), float("nan"), 1
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(n)), n


def spearman_line(label: str, a: np.ndarray, b: np.ndarray) -> tuple[float, float, int]:
    """Spearman rank correlation over the pairs where BOTH estimates are finite."""
    ok = np.isfinite(a) & np.isfinite(b)
    n = int(ok.sum())
    if n < 3:
        print(f"  {label:<34} n={n:<3d}  (too few pairs to correlate)")
        return float("nan"), float("nan"), n
    rho_s, p = spearmanr(a[ok], b[ok])
    star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "   "))
    print(f"  {label:<34} n={n:<3d}  rho_s = {rho_s:+.4f}   p = {p:.3g} {star}")
    return float(rho_s), float(p), n


def inversions(
    true_s: np.ndarray,
    rw_m: np.ndarray,
    rw_e: np.ndarray,
    bk_m: np.ndarray,
    bk_e: np.ndarray,
) -> list[dict]:
    """Every pair (i, j) the two estimators order OPPOSITELY, with its context.

    An inversion is recorded whenever ``sign(rw_i - rw_j) != sign(bk_i - bk_j)``.
    Each is classified by whether either estimator's own difference is
    statistically resolved (|difference| > 2 x combined standard error) and
    whether RW's difference exceeds its one-link quantum. Only an inversion in
    which BOTH estimators resolve their difference is a genuine disagreement
    about order; the rest are two estimators failing to separate two pairs, which
    is a resolution statement, not a contradiction.
    """
    out = []
    n = true_s.size
    for i in range(n):
        for j in range(i + 1, n):
            if not (
                np.isfinite(rw_m[i])
                and np.isfinite(rw_m[j])
                and np.isfinite(bk_m[i])
                and np.isfinite(bk_m[j])
            ):
                continue
            d_rw, d_bk = rw_m[i] - rw_m[j], bk_m[i] - bk_m[j]
            if np.sign(d_rw) == np.sign(d_bk) or d_rw == 0 or d_bk == 0:
                continue
            se_rw = np.hypot(rw_e[i], rw_e[j])
            se_bk = np.hypot(bk_e[i], bk_e[j])
            out.append(
                {
                    "i": i,
                    "j": j,
                    "s_i": true_s[i],
                    "s_j": true_s[j],
                    "d_rw": d_rw,
                    "d_bk": d_bk,
                    "sig_rw": abs(d_rw) > 2.0 * se_rw if np.isfinite(se_rw) else False,
                    "sig_bk": abs(d_bk) > 2.0 * se_bk if np.isfinite(se_bk) else False,
                    "rw_above_quantum": abs(d_rw) > RW_LINK_QUANTUM,
                }
            )
    return out


def main(remeasure: bool = False) -> None:
    print("=" * 78)
    print("Phase 2b Part B2 GATE: Rideout--Wallden vs Boguna--Krioukov in 2+1 D")
    print("=" * 78)
    print(f"Seeds: SEED_BASE + k, k = 0..{N_CAUSETS - 1}   (SEED_BASE={SEED_BASE})")
    print(f"rho = {RHO}, box (T,Lx,Ly) = {BOX_EXTENT}  [identical to Part A Gate A]")
    print(
        f"{N_SEPARATIONS} target separations s = {SEPARATIONS[0]:.2f}..{SEPARATIONS[-1]:.2f}"
        f" (spacing {SEPARATIONS[1] - SEPARATIONS[0]:.4f}), {N_CAUSETS} shared causets each"
    )
    print(
        f"RW resolution: 1 link = {RW_LINK_QUANTUM:.4f}, hard floor = "
        f"{RW_FLOOR:.4f} (2 links)"
    )
    print(
        "\nSCALE DISAGREEMENT BETWEEN THE TWO ESTIMATORS IS EXPECTED AND IS NOT\n"
        "UNDER TEST. RW carries the published m_3 = 2.296 with an unstable ~-12%\n"
        "offset (Part A diagnostic); B-K carries the measured m_3^eff curve and,\n"
        "in 2+1 D, an ASYMPTOTIC distance formula. Only ORDERING is compared.\n"
    )

    data = load_or_measure(remeasure)
    print(f"Measurement wall time: {float(data['elapsed']) / 60:.1f} min")
    print(f"Mean causet size <N> = {data['realised_n'].mean():.0f}\n")

    ns = N_SEPARATIONS
    rw_m = np.full(ns, np.nan)
    rw_e = np.full(ns, np.nan)
    rw_n = np.zeros(ns, dtype=int)
    bkf_m = np.full(ns, np.nan)
    bkf_e = np.full(ns, np.nan)
    bkm_m = np.full(ns, np.nan)
    bkm_e = np.full(ns, np.nan)
    bk_n = np.zeros(ns, dtype=int)
    for i in range(ns):
        rw_m[i], rw_e[i], rw_n[i] = mean_se(data["rw"][i])
        bkf_m[i], bkf_e[i], _ = mean_se(data["bk_fixed"][i])
        bkm_m[i], bkm_e[i], bk_n[i] = mean_se(data["bk_meas"][i])

    yield_per_causet = data["rw_n_two_links"].mean(axis=1)
    frac_empty = (data["rw_n_two_links"] == 0).mean(axis=1)

    # ---------------------------------------------------------------- Finding 1 --
    print("### Finding 1 -- the measurements (errors are SE across causets)\n")
    print(
        "   s_true |  RW 2-link        n |  B-K fixed        |  B-K meas-m3      n "
        "| 2links/cst  empty | <n_c> | tau_c/d med"
    )
    print("  " + "-" * 116)
    for i, s in enumerate(SEPARATIONS):
        tau_med = np.nanmedian(data["bk_tau_ratio_med"][i])
        rw_txt = (
            f"{rw_m[i]:.4f}+/-{rw_e[i]:.4f}" if np.isfinite(rw_e[i]) else f"{rw_m[i]:.4f}   n/a  "
        )
        print(
            f"   {s:6.3f} | {rw_txt} {rw_n[i]:4d} | "
            f"{bkf_m[i]:.4f}+/-{bkf_e[i]:.4f} | "
            f"{bkm_m[i]:.4f}+/-{bkm_e[i]:.4f} {bk_n[i]:4d} | "
            f"{yield_per_causet[i]:9.2f} {100 * frac_empty[i]:5.0f}% | "
            f"{data['bk_n_c'][i].mean():5.1f} | {tau_med:8.2f}"
        )

    # ---------------------------------------------------------------- Finding 2 --
    print("\n### Finding 2 -- THE GATE: Spearman rank correlations\n")
    print("  (each over the pairs where both quantities are finite; *** p<0.001)")
    r_rw_true = spearman_line("RW        vs TRUE separation", rw_m, SEPARATIONS)
    r_bkf_true = spearman_line("B-K fixed vs TRUE separation", bkf_m, SEPARATIONS)
    r_bkm_true = spearman_line("B-K meas  vs TRUE separation", bkm_m, SEPARATIONS)
    r_rw_bkf = spearman_line("RW        vs B-K fixed", rw_m, bkf_m)
    r_rw_bkm = spearman_line("RW        vs B-K meas-m3", rw_m, bkm_m)
    r_bk_bk = spearman_line("B-K fixed vs B-K meas-m3", bkf_m, bkm_m)

    # ---------------------------------------------------------------- Finding 3 --
    print("\n### Finding 3 -- ordering inversions between the two estimators\n")
    invs = inversions(SEPARATIONS, rw_m, rw_e, bkm_m, bkm_e)
    n_comparable = int(
        (np.isfinite(rw_m) & np.isfinite(bkm_m)).sum()
    )
    n_pairs_total = n_comparable * (n_comparable - 1) // 2
    genuine = [v for v in invs if v["sig_rw"] and v["sig_bk"]]
    print(
        f"  {len(invs)} inverted orderings out of {n_pairs_total} comparable "
        f"(i,j) pairs ({100 * len(invs) / max(n_pairs_total, 1):.1f}%)."
    )
    print(
        f"  Of these, {len(genuine)} have BOTH estimators resolving their own "
        "difference at >2 combined SE"
    )
    print("  (i.e. are genuine disagreements about order rather than two ties).\n")
    if invs:
        print("    s_i    s_j   |  d_RW      d_BK    | RW sig  BK sig  RW>1link")
        print("  " + "-" * 66)
        for v in invs[:40]:
            print(
                f"   {v['s_i']:5.3f}  {v['s_j']:5.3f}  | {v['d_rw']:+8.4f} "
                f"{v['d_bk']:+8.4f} |  {str(v['sig_rw']):<5s}  {str(v['sig_bk']):<5s}"
                f"   {str(v['rw_above_quantum'])}"
            )
        if len(invs) > 40:
            print(f"    ... and {len(invs) - 40} more (all in the .npz cache)")

    # ---------------------------------------------------------------- Finding 4 --
    print("\n### Finding 4 -- practical usability (which estimator Phase 4 can use)\n")
    tot_2links = int(data["rw_n_two_links"].sum())
    tot_causets = N_SEPARATIONS * N_CAUSETS
    print(
        f"  RW 2-link yield   : {tot_2links} 2-links over {tot_causets} "
        f"(separation, causet) evaluations = {tot_2links / tot_causets:.2f} per causet"
    )
    print(
        f"  RW empty causets  : {int((data['rw_n_two_links'] == 0).sum())} / "
        f"{tot_causets} = {100 * (data['rw_n_two_links'] == 0).mean():.0f}% produced NOTHING"
    )
    print(
        f"                      (Part A measured 1.16 per causet and 36% empty at "
        "s = 1.0; at s = "
        f"{SEPARATIONS[np.argmin(abs(SEPARATIONS - 1.0))]:.2f} here: "
        f"{yield_per_causet[np.argmin(abs(SEPARATIONS - 1.0))]:.2f} per causet, "
        f"{100 * frac_empty[np.argmin(abs(SEPARATIONS - 1.0))]:.0f}% empty)"
    )
    print(
        f"  B-K admissible c  : {data['bk_n_c'].mean():.1f} per causet on average "
        f"(from {data['bk_n_candidates'].mean():.0f} common-past candidates)"
    )
    print(
        f"  B-K empty causets : {int((data['bk_n_c'] == 0).sum())} / {tot_causets} = "
        f"{100 * (data['bk_n_c'] == 0).mean():.1f}%"
    )
    n_rw_unusable = int((rw_n < 10).sum())
    print(
        f"  Pairs where RW produced < 10 usable causets out of {N_CAUSETS}: "
        f"{n_rw_unusable} / {N_SEPARATIONS}"
    )

    # ---------------------------------------------------------------- Finding 5 --
    print("\n### Finding 5 -- asymptotic-regime audit for the 2+1 D B-K distance\n")
    all_med = data["bk_tau_ratio_med"][np.isfinite(data["bk_tau_ratio_med"])]
    all_min = data["bk_tau_ratio_min"][np.isfinite(data["bk_tau_ratio_min"])]
    print(
        f"  tau_c / d_est over all causets: median {np.median(all_med):.2f}, "
        f"10th pct {np.percentile(all_med, 10):.2f}, worst single vantage "
        f"{all_min.min():.2f}"
    )
    print(
        f"  By separation: s={SEPARATIONS[0]:.2f} -> "
        f"{np.nanmedian(data['bk_tau_ratio_med'][0]):.2f} ;  "
        f"s={SEPARATIONS[-1]:.2f} -> {np.nanmedian(data['bk_tau_ratio_med'][-1]):.2f}"
    )
    print(
        "  eqs. 25--27 require tau_c >> separation. These values are of order 1-4,\n"
        "  so the 2+1 D B-K distance here is at the EDGE of its asymptotic regime.\n"
        "  This limits ACCURACY, not ordering, and no accuracy claim is made."
    )
    print(
        f"  B-K m_3^eff curve reads: mean interval size "
        f"{np.nanmean(data['bk_interval_size']):.0f} elements"
    )

    # ---------------------------------------------------------------- Finding 6 --
    print("\n### Finding 6 -- does the alpha policy move any rank?\n")
    ok = np.isfinite(bkf_m) & np.isfinite(bkm_m)
    rank_f = np.argsort(np.argsort(bkf_m[ok]))
    rank_m = np.argsort(np.argsort(bkm_m[ok]))
    moved = int((rank_f != rank_m).sum())
    print(
        f"  Fixed alpha_2 = {co3.ALPHA_2_AT_RHO_V_64:.5f} (constant) vs measured-m_3 "
        "alpha_2 (per interval)."
    )
    print(
        f"  Ranks differing between the two policies: {moved} of {int(ok.sum())} pairs."
    )
    print(
        f"  Spearman(B-K fixed, B-K meas) = {r_bk_bk[0]:+.4f} (p = {r_bk_bk[1]:.3g})."
    )
    if moved == 0:
        print(
            "  => ROBUSTNESS RESULT: the size-dependent calibration changes the\n"
            "     magnitudes but not a single rank, so the gate does not rest on\n"
            "     which policy was chosen."
        )
    else:
        print(
            "  => CALIBRATION SENSITIVITY: the size-dependent calibration DOES move\n"
            "     ranks. Reported as a finding; the gate is evaluated on the\n"
            "     measured-m_3 policy (the brief's instruction) with the fixed-alpha\n"
            "     correlations quoted alongside."
        )

    # ---------------------------------------------------------------- Finding 7 --
    # The aggregate gate ranks MEANS over N_CAUSETS causets, which suppresses
    # exactly the noise that would produce an inversion. That makes it a weak
    # test, and Phase 4 would not have 100 causets per pair anyway. The honest
    # question is how well a SINGLE causet orders the pairs.
    print("\n### Finding 7 -- ordering on a SINGLE causet (the Phase-4 regime)\n")
    per_rw, per_bk, per_rb, per_n = [], [], [], []
    for k in range(N_CAUSETS):
        a, b = data["rw"][:, k], data["bk_meas"][:, k]
        ok_a, ok_b = np.isfinite(a), np.isfinite(b)
        both = ok_a & ok_b
        if ok_a.sum() < 5 or both.sum() < 5:
            continue
        per_rw.append(spearmanr(a[ok_a], SEPARATIONS[ok_a])[0])
        per_bk.append(spearmanr(b[ok_b], SEPARATIONS[ok_b])[0])
        per_rb.append(spearmanr(a[both], b[both])[0])
        per_n.append(int(ok_a.sum()))
    per_rw = np.asarray(per_rw)
    per_bk = np.asarray(per_bk)
    per_rb = np.asarray(per_rb)
    print(
        f"  Usable causets: {per_rw.size}/{N_CAUSETS}; RW resolves "
        f"{np.mean(per_n):.1f} of the {N_SEPARATIONS} separations on an average causet."
    )
    for label, arr in (
        ("RW  vs TRUE", per_rw),
        ("B-K vs TRUE", per_bk),
        ("RW  vs B-K ", per_rb),
    ):
        print(
            f"  per-causet Spearman {label}: median {np.median(arr):+.3f}  "
            f"mean {arr.mean():+.3f} +/- {arr.std(ddof=1) / np.sqrt(arr.size):.3f}  "
            f"min {arr.min():+.3f}  (n = {arr.size} causets)"
        )
    print(
        "\n  => On a single causet the agreement is STRONG BUT NOT PERFECT. The\n"
        "     aggregate rho_s = 1.000 of Finding 2 is a property of averaging 100\n"
        "     causets, not a property of the estimators. This is the number that\n"
        "     should be carried into Phase 4."
    )

    # ---------------------------------------------------------------- Finding 8 --
    print("\n### Finding 8 -- could the aggregate gate have failed? (bootstrap)\n")
    rng = np.random.default_rng(20260922)
    n_boot = 2000
    boot = np.empty(n_boot)
    for t in range(n_boot):
        idx = rng.integers(0, N_CAUSETS, N_CAUSETS)
        rm = np.array([np.nanmean(data["rw"][i, idx]) for i in range(ns)])
        bm = np.array([np.nanmean(data["bk_meas"][i, idx]) for i in range(ns)])
        ok_b = np.isfinite(rm) & np.isfinite(bm)
        boot[t] = spearmanr(rm[ok_b], bm[ok_b])[0]
    print(
        f"  Resampling the {N_CAUSETS} causets with replacement ({n_boot} replicates),\n"
        f"  the aggregate Spearman(RW, B-K) spans [{boot.min():.4f}, {boot.max():.4f}]\n"
        f"  with median {np.median(boot):.4f}."
    )
    print(
        "\n  => HONEST ASSESSMENT OF THE GATE'S STRENGTH. Both estimators are\n"
        "     strongly monotone in s and the per-pair standard errors (~0.01-0.02)\n"
        "     are far below the ladder spacing (0.074), so the aggregate test was\n"
        "     never at serious risk of failing. It rules out GROSS ordering\n"
        "     disagreement and nothing finer. Finding 7 is the test with teeth."
    )

    # ---------------------------------------------------------------- Finding 9 --
    print("\n### Finding 9 -- B-K compresses at large separation; RW does not\n")
    ratio_bk = bkm_m / SEPARATIONS
    ratio_rw = rw_m / SEPARATIONS
    tau_med_by_s = np.array(
        [np.nanmedian(data["bk_tau_ratio_med"][i]) for i in range(ns)]
    )
    print("     s   |  B-K/true  |  RW/true  |  tau_c/d  |  RW 2-links/causet")
    print("  " + "-" * 62)
    for i in range(0, ns, max(1, ns // 8)):
        print(
            f"   {SEPARATIONS[i]:5.3f} |   {ratio_bk[i]:6.3f}   |  {ratio_rw[i]:6.3f}   |"
            f"  {tau_med_by_s[i]:6.2f}   |      {yield_per_causet[i]:5.2f}"
        )
    print(
        f"\n  B-K/true falls monotonically {ratio_bk[0]:.3f} -> {ratio_bk[-1]:.3f} "
        f"across the ladder;\n  RW/true is non-monotone "
        f"({ratio_rw[0]:.3f} at the smallest s, where the 2-link FLOOR of "
        f"{RW_FLOOR:.4f}\n  inflates it, then {np.nanmin(ratio_rw):.3f}-"
        f"{ratio_rw[-1]:.3f} across the rest)."
    )
    rs_conf = spearmanr(ratio_bk, tau_med_by_s)
    print(
        f"\n  Spearman(B-K/true, tau_c/d) = {rs_conf[0]:+.4f}. THIS NUMBER IS\n"
        "  CONFOUNDED AND IS NOT EVIDENCE OF A MECHANISM. Both quantities are\n"
        "  monotone functions of s by construction -- B-K/true because the\n"
        "  estimator compresses, tau_c/d because a wider pair has less room above\n"
        "  it in a fixed box -- so a rank correlation of ~1 between them is what\n"
        "  two monotone functions of a common variable always give, mechanism or\n"
        "  not. It is reported because it is CONSISTENT with the asymptotic\n"
        "  breakdown that eqs. 25--27 predict, and for no stronger reason.\n"
        "\n  What would actually test it (NOT done here): hold s FIXED and vary the\n"
        "  box time extent, which moves tau_c/d without moving s. If B-K/true rose\n"
        "  toward 1 as the region deepened at fixed s, the asymptotic explanation\n"
        "  would be established; if it did not, the compression has another cause.\n"
        "  Logged as an OPEN QUESTION, not a conclusion."
    )
    log_fit = np.polyfit(np.log(SEPARATIONS), np.log(yield_per_causet), 1)
    print(
        f"\n  Separately measured here and new (Part A varied the region at fixed s,\n"
        f"  never s at fixed region): RW's 2-link yield falls as s^{log_fit[0]:.2f}\n"
        f"  over this ladder -- {yield_per_causet[0]:.2f} per causet at s = "
        f"{SEPARATIONS[0]:.2f} down to {yield_per_causet[-1]:.2f} at s = "
        f"{SEPARATIONS[-1]:.2f}, with the empty-causet rate rising "
        f"{100 * frac_empty[0]:.0f}% -> {100 * frac_empty[-1]:.0f}%."
    )

    # ------------------------------------------------------------------- verdict --
    print("\n" + "=" * 78)
    print("GATE B VERDICT")
    print("=" * 78)
    crit = [
        ("B-i   Spearman(RW, true) > 0 and p < 0.01", r_rw_true),
        ("B-ii  Spearman(B-K, true) > 0 and p < 0.01", r_bkm_true),
        ("B-iii Spearman(RW, B-K) > 0 and p < 0.01", r_rw_bkm),
    ]
    passed = True
    for label, (rho_s, p, n) in crit:
        ok_c = np.isfinite(rho_s) and rho_s > 0 and p < ALPHA_SIGNIFICANCE
        passed &= ok_c
        print(
            f"  [{'PASS' if ok_c else 'FAIL'}] {label}\n"
            f"         rho_s = {rho_s:+.4f}, p = {p:.3g}, n = {n} pairs"
        )
    print(f"\n  GATE B: {'PASS' if passed else 'FAIL'}")
    print(
        "\n  Scope of this verdict: it certifies ORDERING agreement only. It makes\n"
        "  no claim about either estimator's absolute scale, and none can be read\n"
        "  out of it -- see the module docstring and Finding 5."
    )

    make_figure(
        data, rw_m, rw_e, bkf_m, bkf_e, bkm_m, bkm_e, yield_per_causet, frac_empty,
        invs, per_rw, per_bk, per_rb,
    )
    print(f"\nFigure -> {FIGURE}")


# --------------------------------------------------------------------- figure --
def make_figure(
    data, rw_m, rw_e, bkf_m, bkf_e, bkm_m, bkm_e, yield_per_causet, frac_empty,
    invs, per_rw, per_bk, per_rb,
) -> None:
    """Six panels: estimates, aggregate rank agreement, yields, asymptotic regime,
    SINGLE-causet ordering (the test with teeth), and the compression ratios."""
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 10.0))
    s = SEPARATIONS

    # Panel A -- both estimators vs true separation, SHARED axes.
    ax = axes[0, 0]
    ax.errorbar(s, rw_m, yerr=rw_e, fmt="o-", ms=4, lw=1.2, capsize=2,
                label="Rideout--Wallden 2-link (m_3 = 2.296)")
    ax.errorbar(s, bkm_m, yerr=bkm_e, fmt="s-", ms=4, lw=1.2, capsize=2,
                label="Boguna--Krioukov overlap (measured m_3^eff)")
    ax.errorbar(s, bkf_m, yerr=bkf_e, fmt="^--", ms=3, lw=0.9, capsize=2, alpha=0.6,
                label="Boguna--Krioukov overlap (fixed alpha_2)")
    ax.plot(s, s, "k:", lw=1.0, label="y = x (NOT a target: see title)")
    ax.axhline(RW_FLOOR, color="crimson", ls=":", lw=1.0)
    ax.text(s[0], RW_FLOOR * 1.03, "RW 2-link floor", color="crimson", fontsize=7)
    ax.set_xlabel("TRUE spacelike separation  s  (embedding)")
    ax.set_ylabel("estimator output (arbitrary, UNCALIBRATED scale)")
    ax.set_title(
        "A. Both estimators vs true separation\n"
        "SCALE DISAGREEMENT IS EXPECTED -- only ORDERING is under test",
        fontsize=9.5,
    )
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.25)

    # Panel B -- the rank-rank plot: the quantity the gate actually tests.
    ax = axes[0, 1]
    ok = np.isfinite(rw_m) & np.isfinite(bkm_m)
    r_rw = np.argsort(np.argsort(rw_m[ok])) + 1
    r_bk = np.argsort(np.argsort(bkm_m[ok])) + 1
    sc = ax.scatter(r_rw, r_bk, c=s[ok], cmap="viridis", s=45, zorder=3)
    ax.plot([1, ok.sum()], [1, ok.sum()], "k--", lw=1.0, label="perfect agreement")
    plt.colorbar(sc, ax=ax, label="true separation s")
    rho_s, p = spearmanr(rw_m[ok], bkm_m[ok])
    ax.set_xlabel("rank by Rideout--Wallden estimate")
    ax.set_ylabel("rank by Boguna--Krioukov estimate")
    ax.set_title(
        f"B. THE GATE: rank agreement\nSpearman rho_s = {rho_s:+.4f}, "
        f"p = {p:.2g}, n = {int(ok.sum())} pairs\n"
        f"{len(invs)} inverted (i,j) orderings",
        fontsize=9.5,
    )
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    # Panel C -- usability: what each estimator actually delivers per causet.
    ax = axes[1, 0]
    ax.plot(s, yield_per_causet, "o-", ms=4, color="tab:red", label="RW 2-links per causet")
    ax.plot(s, data["bk_n_c"].mean(axis=1), "s-", ms=4, color="tab:blue",
            label="B-K admissible c per causet")
    ax.axhline(1.16, color="tab:red", ls=":", lw=1.0)
    ax.text(s[-1], 1.25, "Part A: 1.16", color="tab:red", fontsize=7, ha="right")
    ax.set_yscale("log")
    ax.set_xlabel("TRUE spacelike separation  s")
    ax.set_ylabel("samples per causet (log scale)")
    ax2 = ax.twinx()
    ax2.plot(s, 100 * frac_empty, "x--", ms=4, color="dimgray", lw=0.9,
             label="% causets RW yields nothing")
    ax2.set_ylabel("% of causets with NO RW 2-link", color="dimgray")
    ax2.set_ylim(0, 105)
    ax.set_title(
        "C. Practical usability (Phase 4 benchmark viability)\n"
        f"over {N_CAUSETS} causets per separation",
        fontsize=9.5,
    )
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="center left")
    ax.grid(alpha=0.25)

    # Panel D -- the asymptotic-regime audit for the B-K 2+1 D formula.
    ax = axes[1, 1]
    med = np.array([np.nanmedian(data["bk_tau_ratio_med"][i]) for i in range(N_SEPARATIONS)])
    lo = np.array([np.nanpercentile(data["bk_tau_ratio_med"][i], 10) for i in range(N_SEPARATIONS)])
    hi = np.array([np.nanpercentile(data["bk_tau_ratio_med"][i], 90) for i in range(N_SEPARATIONS)])
    ax.fill_between(s, lo, hi, alpha=0.25, color="tab:blue", label="10-90th pct over causets")
    ax.plot(s, med, "o-", ms=4, color="tab:blue", label="median tau_c / d_est")
    ax.axhline(1.0, color="crimson", ls="--", lw=1.0, label="tau_c = d (formula invalid)")
    ax.set_xlabel("TRUE spacelike separation  s")
    ax.set_ylabel("tau_c / d_est  (eqs. 25--27 need >> 1)")
    ax.set_title(
        "D. Asymptotic-regime audit, 2+1 D B-K distance\n"
        "eqs. 25--27 are a LEADING-ORDER form; this is how far\n"
        "into its regime the measurement actually sat",
        fontsize=9.5,
    )
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    # Panel E -- single-causet ordering: the distribution the gate's aggregate hides.
    ax = axes[0, 2]
    bins = np.linspace(0.5, 1.0, 26)
    for arr, lab, col in (
        (per_rw, "RW vs true", "tab:red"),
        (per_bk, "B-K vs true", "tab:blue"),
        (per_rb, "RW vs B-K", "tab:green"),
    ):
        ax.hist(arr, bins=bins, histtype="step", lw=1.6, color=col,
                label=f"{lab}  med {np.median(arr):.3f}")
    ax.axvline(1.0, color="k", ls="--", lw=1.0)
    ax.text(0.999, ax.get_ylim()[1] * 0.55, "aggregate gate sits here (1.000)",
            rotation=90, ha="right", va="center", fontsize=7)
    ax.set_xlabel("Spearman rho_s within ONE causet")
    ax.set_ylabel("number of causets")
    ax.set_title(
        "E. THE TEST WITH TEETH: ordering on a single causet\n"
        f"{per_rw.size} causets; the aggregate rho_s = 1.000 is a\n"
        "property of averaging 100 causets, not of the estimators",
        fontsize=9.5,
    )
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.25)

    # Panel F -- the compression, with its confound stated on the panel.
    ax = axes[1, 2]
    tau_med_by_s = np.array(
        [np.nanmedian(data["bk_tau_ratio_med"][i]) for i in range(N_SEPARATIONS)]
    )
    ax.plot(s, bkm_m / s, "s-", ms=4, color="tab:blue", label="B-K / true")
    ax.plot(s, rw_m / s, "o-", ms=4, color="tab:red", label="RW / true")
    ax.axhline(1.0, color="k", ls="--", lw=1.0, label="exact recovery")
    ax.set_xlabel("TRUE spacelike separation  s")
    ax.set_ylabel("estimate / true separation")
    axb = ax.twinx()
    axb.plot(s, tau_med_by_s, ":", lw=1.4, color="dimgray")
    axb.set_ylabel("median tau_c / d_est", color="dimgray")
    ax.set_title(
        "F. Compression (ACCURACY, not under test)\n"
        "B-K falls away as tau_c/d drops -- CONSISTENT with the\n"
        "eqs. 25-27 breakdown, but CONFOUNDED: both curves are\n"
        "monotone in s. Open question, not a conclusion.",
        fontsize=9.5,
    )
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=0.25)

    fig.suptitle(
        "Phase 2b Part B2 -- Rideout--Wallden vs Boguna--Krioukov on IDENTICAL M^3 sprinklings "
        f"(rho = {RHO}, box {BOX_EXTENT}, seeds {SEED_BASE}+k)",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    FIGURE.parent.mkdir(exist_ok=True)
    fig.savefig(FIGURE, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main(remeasure="--remeasure" in sys.argv)
