"""Phase 2b, Part 1 GATE: reproduce Rideout--Wallden's M^3 constant m_3.

Reproduce with:
    python experiments/exp02_m3_validation.py          # uses the measurement cache
    python experiments/exp02_m3_validation.py --remeasure   # ignores the cache

Produces:
    * printed tables (standard errors over realisations, explicit N everywhere),
    * figures/exp02_m3_validation.png,
    * data/exp02_measurements.npz (raw per-realisation chain lengths + seeds, so
      the analysis can be redone without repeating the 18-minute measurement),
    * an explicit PASS/FAIL statement for each part of the Part-1 gate.

What is being validated
-----------------------
Rideout & Wallden, "Spacelike distance from discrete causal order"
(arXiv:0810.1768). Every number below is taken from the paper's own text:

* eq. (2), Sec. II.1: ``V_xy = eta(d) l_xy^d``, ``eta(d) = 2 V^s_{d-1}/(2^d d)``,
  giving ``eta(3) = pi/12`` (computed, not hardcoded, in ``sprinkle3d``).
* eq. (1), Sec. II.1: ``L (rho V)^{-1/d} -> m_d`` as ``rho V -> infinity``.
* Sec. II.1, definition of L: "we define proper time d(x,y), between two related
  elements x prec y, to be the number of **links** L in the longest chain
  between (and including) x and y".
* Fig. 4 / Sec. III.1: fitting ``f(N) = m_3 + a e^{b log_2 N}`` they obtain
  ``m_3 = 2.296 +/- 0.012, a = -1.087 +/- 0.014, b = -0.1201 +/- 0.0053``,
  over N up to 2^18 (1600 causets per point below 2^17, 400 at 2^17 and 2^17.5,
  100 at 2^18). Footnote 11 records an earlier value m_3 ~ 2.278.

Why "is the measured number ~2.29" is the wrong question
--------------------------------------------------------
Their own fitted curve, evaluated at the sizes reachable here, predicts
``m_3^eff`` rising from ~1.97 (N=2^10) to ~2.15 (N=2^17): the finite-size
correction is 0.15-0.33, an order of magnitude larger than the uncertainty on
the asymptote. Measuring 2.29 at N=2^14 would be evidence of a *bug*. So the
asymptote has to be reached by extrapolation, and the extrapolation itself has
to be calibrated. Hence four separate checks, each reported with its own number.

Three finite-size effects that must be handled explicitly
---------------------------------------------------------
1. **The link convention contributes an exactly known extra power of N.** With
   the interval endpoints included, every chain can be extended by both of them,
   so writing ``I`` for the longest chain among the ``N`` interior (Poisson)
   elements: ``E = I + 2`` elements, and ``L = E - 1 = I + 1`` links. Therefore

       m_link = L (rho V)^{-1/d} = I (rho V)^{-1/d} + (rho V)^{-1/d},

   i.e. ``m_link`` carries a ``+N^{-1/d}`` term whose coefficient is exactly 1,
   not a fitted parameter. In d=2 at N=2^10 that term is +0.031, about 15x the
   standard error, so a two-term fit that has to absorb it is biased. We
   therefore extrapolate the clean quantity ``m_inter = I (rho V)^{-1/d}`` and
   report ``m_link`` only where comparison with the paper demands it (their
   published ``a``, ``b`` describe link-convention data).
2. **The correction exponent must be fitted, not assumed.** The d=2 control
   below shows that *fixing* the exponent biases the asymptote low (and returns
   an amplitude ``a ~ -1.6`` against the exact Tracy--Widom value -1.7711),
   whereas leaving it free recovers the known answer. Fixed-exponent fits are
   still reported, as secondary numbers, with what was assumed printed next to
   them.
3. **The residual bias of the whole procedure is measured, not assumed zero.**
   See the control.

The d = 2 control: calibrating the procedure where the answer is known exactly
-----------------------------------------------------------------------------
GATE 1 and GATE 2 both compare us with the paper we are validating against, so
neither can catch an error in a *shared* procedure. The control closes that gap
using ``m_2 = 2`` exactly (Brightwell & Gregory, PRL 66, 260 (1991); quoted by
Rideout--Wallden as "The asymptote is, as stated in ref. [bg], equal to 2"; and
already validated in Phase 1). Three parts:

* **Control A -- chain-counter equivalence at scale.** A 1+1 D sprinkling is
  embedded in M^3 by setting ``y == 0``; then ``dt^2 - dx^2 - dy^2 = dt^2 - dx^2``
  is exactly the 1+1 D relation, so the new streamed 2+1 D dynamic program must
  return exactly the integer Phase 1's independent O(N log N) patience-sorting
  counter returns. This validates the new counter against trusted Phase-1 code
  at sizes where a dense causal matrix could not be formed.
* **Control B -- matched-window calibration.** The identical estimator and the
  identical fitting routine are run in d=2 over the *same* ladder of N and the
  *same* realisation counts as the d=3 measurement. Any offset from 2.000 is the
  procedure's bias at this window and noise level; it is quoted as a systematic
  uncertainty on m_3 rather than being assumed to vanish.
* **Control C -- high-statistics, longer lever arm.** The same in d=2 out to
  N = 2^18 with many more realisations, to show the bias shrinks as the fit
  window improves, i.e. that it is a finite-window artefact and not an error.

Cost note (honest scope limit)
------------------------------
The longest chain in 2+1 D needs the dense O(N^2) dynamic program: Phase 1's
O(N log N) patience-sorting shortcut is valid only in 1+1 D, where the light
cone's two flat faces make the causal order a coordinate order (see
``order3d.longest_chain_length_3d``). We use the O(N)-*memory* streamed variant
of that same quadratic DP, tested to return exactly the dense answer. Runtime is
~O(N^2) per realisation (~22 s at N=2^16, ~75 s at 2^17 on one core), capping us
at N = 2^17 with 4 realisations rather than the paper's 2^18 x 100 causets.
Realisation counts are reported per point and the error bars reflect them.

No point is dropped, nothing is smoothed, and nothing is fitted to force
agreement (Integrity Rules 1-4).
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
from scipy.optimize import curve_fit  # noqa: E402

from causet import order, order3d, sprinkle, sprinkle3d  # noqa: E402

# ---- Values quoted by Rideout & Wallden (Integrity Rule 4: named + sourced) --
RW_M3 = 2.296  # Fig. 4 / Sec. III.1 fitted asymptote
RW_M3_ERR = 0.012
RW_FIT_A = -1.087  # Fig. 4 / Sec. III.1
RW_FIT_A_ERR = 0.014
RW_FIT_B = -0.1201  # Fig. 4 / Sec. III.1, exponent of e^{b log_2 N}
RW_FIT_B_ERR = 0.0053
RW_M3_FOOTNOTE = 2.278  # earlier value, their footnote 11
RW_MAX_LOG2N = 18.0  # largest simulation size in their Fig. 4
RW_TARGET_LO = min(RW_M3_FOOTNOTE, RW_M3 - RW_M3_ERR)  # 2.278
RW_TARGET_HI = RW_M3 + RW_M3_ERR  # 2.308

# ---- Independently known d = 2 values (the control's oracle) -----------------
BG_M2 = 2.0  # Brightwell & Gregory, PRL 66, 260 (1991); Phase-1 validated
TRACY_WIDOM_MEAN = -1.7711  # exact coefficient of N^{-1/3} for m_2^inter in d=2

# ---- This experiment's parameters -------------------------------------------
SEED_BASE = 20260810
CONTROL_SEED_BASE = 8102026
TAU = 1.0  # proper time of the M^3 diamond; rho is varied to set rho V
BLOCK_SIZE = 256  # memory/speed knob of the streamed DP only; no physics
CACHE = Path(__file__).resolve().parents[1] / "data" / "exp02_measurements.npz"

# (log2 of rho V, independent realisations). Counts fall with N because runtime
# grows as N^2; every count is printed in the output tables.
LADDER: list[tuple[int, int]] = [
    (10, 800),
    (11, 600),
    (12, 400),
    (13, 200),
    (14, 100),
    (15, 40),
    (16, 14),
    (17, 4),
]
# Control C: d = 2 is cheap (Phase 1's O(N log N) counter), so it reaches the
# paper's own maximum N with far better statistics.
CONTROL_C_LADDER: list[tuple[int, int]] = [
    (10, 600), (11, 600), (12, 600), (13, 400), (14, 300),
    (15, 200), (16, 150), (17, 100), (18, 60),
]
# Control A: sizes at which the streamed 2+1 D counter is compared element-for-
# element with Phase 1's counter (kept small -- the streamed DP is O(N^2)).
CONTROL_A_SIZES: list[tuple[int, int]] = [(10, 10), (12, 6), (14, 3)]


# --------------------------------------------------------------------- models --
def rw_fit_curve(n, m3: float = RW_M3, a: float = RW_FIT_A, b: float = RW_FIT_B):
    """Rideout--Wallden Fig. 4 fitting function ``f(N) = m_3 + a e^{b log_2 N}``.

    Note ``e^{b log_2 N} = N^{b/ln 2}``: a power law with exponent -0.1733 for
    their fitted ``b``. Describes their *link-convention* data.
    """
    return m3 + a * np.exp(b * np.log2(n))


def rw_curve_uncertainty(n: np.ndarray) -> np.ndarray:
    """Uncertainty of ``rw_fit_curve`` propagated from RW's quoted parameter errors.

    ``f = m_3 + a N^{b/ln2}``, so with ``x = N^{b/ln2} = e^{b log_2 N}``::

        df/dm_3 = 1,   df/da = x,   df/db = a x log_2 N.

    CAVEAT, stated because it changes the interpretation: the paper reports
    parameter *errors* but not their *covariance*, and fitted parameters of this
    kind are strongly (anti)correlated. Adding the three terms in quadrature as
    if independent therefore **over**estimates the curve's true uncertainty, so a
    chi^2 computed with it is conservative -- lenient towards agreement. Both the
    strict (our errors only) and lenient (combined) chi^2 are reported below so
    the reader can see the whole range rather than a single chosen number.
    """
    log2n = np.log2(n)
    x = np.exp(RW_FIT_B * log2n)
    return np.sqrt(
        RW_M3_ERR**2 + (x * RW_FIT_A_ERR) ** 2 + (RW_FIT_A * x * log2n * RW_FIT_B_ERR) ** 2
    )


def power_law_model(n, m_inf: float, a: float, c: float):
    """``m_inf + a N^c`` -- the same functional form, reparametrised for fitting."""
    return m_inf + a * n**c


# ---------------------------------------------------------------- measurement --
def measure_d3(log2n: int, n_real: int) -> dict:
    """Measure the longest chain over ``n_real`` M^3 sprinklings at ``rho V = 2^log2n``.

    Returns per-realisation arrays; nothing is averaged or discarded here
    (Integrity Rule 1). ``elements`` counts chain elements *including* the two
    interval endpoints, from which both conventions follow exactly:
    ``links = elements - 1`` and ``interior = elements - 2``.
    """
    rho_volume = float(2**log2n)
    rho = rho_volume / sprinkle3d.diamond_volume_3d(TAU)

    elements = np.empty(n_real, dtype=float)
    realised_n = np.empty(n_real, dtype=float)
    t0 = time.time()
    for k in range(n_real):
        seed = SEED_BASE + 1000 * log2n + k
        s = sprinkle3d.sprinkle_diamond_3d(rho, TAU, seed=seed, include_endpoints=True)
        elements[k] = order3d.longest_chain_elements_streamed_3d(
            s.t, s.x, s.y, block_size=BLOCK_SIZE
        )
        realised_n[k] = s.n_interior
    elapsed = time.time() - t0
    print(f"  measured log2N={log2n:2d}  #real={n_real:4d}  "
          f"<elements>={elements.mean():7.2f}  ({elapsed:6.1f} s)", flush=True)
    return {
        "log2n": log2n,
        "rho_volume": rho_volume,
        "n_real": n_real,
        "elements": elements,
        "realised_n": realised_n,
        "elapsed": elapsed,
    }


def measure_d2_interior(log2n: int, n_real: int, seed_offset: int) -> np.ndarray:
    """Longest interior chain over ``n_real`` 1+1 D sprinklings at ``rho V = 2^log2n``.

    Uses Phase 1's O(N log N) counter, which returns chain *elements* including the
    two endpoints; the interior count is that minus 2.
    """
    area = float(2**log2n)
    rho = area / sprinkle.diamond_volume_1d(1.0)
    interior = np.empty(n_real, dtype=float)
    for k in range(n_real):
        s = sprinkle.sprinkle_diamond_1d(
            rho, tau=1.0, seed=CONTROL_SEED_BASE + seed_offset + 1000 * log2n + k,
            include_endpoints=True,
        )
        interior[k] = order.chain_via_lis_1d(s.u, s.v) - 2.0
    return interior


def load_or_measure(remeasure: bool) -> list[dict]:
    """Load the d=3 ladder from cache when it matches, otherwise measure and save."""
    key = f"{SEED_BASE}|{TAU}|" + ";".join(f"{a}:{b}" for a, b in LADDER)
    if not remeasure and CACHE.exists():
        z = np.load(CACHE, allow_pickle=False)
        if str(z["key"]) == key:
            print(f"Loaded cached measurements from {CACHE}")
            rows = []
            for log2n, n_real in LADDER:
                rows.append({
                    "log2n": log2n,
                    "rho_volume": float(2**log2n),
                    "n_real": n_real,
                    "elements": z[f"elements_{log2n}"],
                    "realised_n": z[f"realised_n_{log2n}"],
                    "elapsed": float(z[f"elapsed_{log2n}"]),
                })
            return rows
        print("Cache present but parameters differ -- remeasuring.")

    print("Measuring the d=3 ladder (O(N^2) per realisation; ~18 min)...")
    rows = [measure_d3(log2n, n_real) for log2n, n_real in LADDER]
    CACHE.parent.mkdir(exist_ok=True)
    payload = {"key": np.array(key)}
    for r in rows:
        payload[f"elements_{r['log2n']}"] = r["elements"]
        payload[f"realised_n_{r['log2n']}"] = r["realised_n"]
        payload[f"elapsed_{r['log2n']}"] = np.array(r["elapsed"])
    np.savez(CACHE, **payload)
    print(f"Saved raw measurements -> {CACHE}")
    return rows


# ------------------------------------------------------------------- analysis --
def convention(rows: list[dict], which: str, d: int) -> tuple[np.ndarray, ...]:
    """``(N, mean, se)`` for ``which`` in {"link", "interior"}.

    ``link``     : ``L (rho V)^{-1/d}``, ``L = elements - 1``  (Rideout--Wallden)
    ``interior`` : ``I (rho V)^{-1/d}``, ``I = elements - 2``  (clean power series)
    """
    offset = {"link": 1.0, "interior": 2.0}[which]
    n = np.array([r["rho_volume"] for r in rows], dtype=float)
    mean = np.empty(len(rows))
    se = np.empty(len(rows))
    for i, r in enumerate(rows):
        q = (r["elements"] - offset) / r["rho_volume"] ** (1.0 / d)
        mean[i] = q.mean()
        se[i] = q.std(ddof=1) / math.sqrt(r["n_real"]) if r["n_real"] > 1 else np.nan
    return n, mean, se


def fit_ladder(n, mean, se, *, label: str, fix_a=None, fix_c=None, p0_a=None,
               verbose: bool = True) -> dict:
    """Fit ``m_inf + a N^c`` weighted by ``se``, optionally holding ``a``/``c`` fixed.

    Which parameters were free is always printed, so no number here can be read
    without knowing what was assumed.
    """
    free = ["m_inf"] + ([] if fix_a is not None else ["a"]) + \
           ([] if fix_c is not None else ["c"])

    def model(nn, *p):
        it = iter(p)
        m_inf = next(it)
        a = fix_a if fix_a is not None else next(it)
        c = fix_c if fix_c is not None else next(it)
        return power_law_model(nn, m_inf, a, c)

    p0 = [RW_M3]
    if fix_a is None:
        p0.append(RW_FIT_A if p0_a is None else p0_a)
    if fix_c is None:
        p0.append(RW_FIT_B / math.log(2.0))

    popt, pcov = curve_fit(model, n, mean, p0=p0, sigma=se, absolute_sigma=True,
                           maxfev=100000)
    perr = np.sqrt(np.diag(pcov))
    chi2 = float(np.sum(((mean - model(n, *popt)) / se) ** 2))
    dof = len(n) - len(popt)

    a_val = fix_a if fix_a is not None else float(popt[free.index("a")])
    c_val = fix_c if fix_c is not None else float(popt[free.index("c")])
    a_err = None if fix_a is not None else float(perr[free.index("a")])
    c_err = None if fix_c is not None else float(perr[free.index("c")])

    if verbose:
        print(f"\n--- {label} ---")
        print(f"  free: {', '.join(free)}     points: {len(n)} "
              f"(log2N {math.log2(n.min()):.0f}-{math.log2(n.max()):.0f})")
        print(f"  m_inf = {popt[0]:.4f} +/- {perr[0]:.4f}")
        print(f"  a     = {a_val:+.4f}" +
              (f" +/- {a_err:.4f}" if a_err is not None else "   (FIXED)"))
        print(f"  c     = {c_val:+.4f}" +
              (f" +/- {c_err:.4f}" if c_err is not None else "   (FIXED)"))
        print(f"  chi2/dof = {chi2:.2f}/{dof} = {chi2 / dof:.2f}" if dof > 0
              else f"  chi2 = {chi2:.2f} (dof = 0)")
    return {
        "m_inf": float(popt[0]), "m_err": float(perr[0]),
        "a": a_val, "a_err": a_err, "c": c_val, "c_err": c_err,
        "popt_full": (float(popt[0]), a_val, c_val),
        "chi2": chi2, "dof": dof, "free": free,
    }


def sigma_outside(value: float, err: float, lo: float, hi: float) -> float:
    """Distance of ``value`` from ``[lo, hi]`` in units of ``err`` (0 if inside)."""
    if lo <= value <= hi:
        return 0.0
    return (lo - value) / err if value < lo else (value - hi) / err


# -------------------------------------------------------------------- controls --
def control_a() -> bool:
    print("\n=== Control A: streamed 2+1 D chain counter vs Phase 1's LIS counter ===")
    print("  (1+1 D sprinkling embedded in M^3 with y == 0; must match exactly)")
    print(f"  {'log2N':>6} {'#real':>6} {'N':>8} {'streamed':>9} {'Phase-1':>8} {'match':>6}")
    all_match = True
    for log2n, n_real in CONTROL_A_SIZES:
        area = float(2**log2n)
        rho = area / sprinkle.diamond_volume_1d(1.0)
        for k in range(n_real):
            s = sprinkle.sprinkle_diamond_1d(
                rho, tau=1.0, seed=CONTROL_SEED_BASE + 1000 * log2n + k,
                include_endpoints=True)
            streamed = order3d.longest_chain_elements_streamed_3d(
                s.t, s.x, np.zeros_like(s.t), block_size=BLOCK_SIZE)
            lis = order.chain_via_lis_1d(s.u, s.v)
            ok = streamed == lis
            all_match &= ok
            if k == 0 or not ok:
                print(f"  {log2n:>6d} {n_real:>6d} {s.n:>8d} {streamed:>9d} "
                      f"{lis:>8d} {'yes' if ok else 'NO':>6}")
    print(f"  all {sum(c for _, c in CONTROL_A_SIZES)} realisations match: {all_match}")
    return all_match


def run_d2_control(ladder: list[tuple[int, int]], seed_offset: int, title: str) -> dict:
    """Run the m_d procedure in d = 2, where the asymptote is exactly 2."""
    print(f"\n=== {title} ===")
    print(f"  {'log2N':>6} {'#real':>6} {'<I>':>9} {'<m2_inter>':>11} {'SE':>8}")
    n, mean, se = [], [], []
    for log2n, n_real in ladder:
        interior = measure_d2_interior(log2n, n_real, seed_offset)
        q = interior / (2.0**log2n) ** 0.5
        n.append(float(2**log2n))
        mean.append(q.mean())
        se.append(q.std(ddof=1) / math.sqrt(n_real) if n_real > 1 else np.nan)
        print(f"  {log2n:>6d} {n_real:>6d} {interior.mean():>9.2f} "
              f"{mean[-1]:>11.5f} {se[-1]:>8.5f}")
    n, mean, se = np.array(n), np.array(mean), np.array(se)

    res = fit_ladder(n, mean, se, label=f"{title}: free 2-term fit (c free)",
                     p0_a=TRACY_WIDOM_MEAN)
    res["n"], res["mean"], res["se"] = n, mean, se
    res["bias"] = res["m_inf"] - BG_M2
    res["pull"] = res["bias"] / res["m_err"]
    print(f"  recovered m_2 = {res['m_inf']:.5f} +/- {res['m_err']:.5f}  vs exact 2 "
          f"->  bias = {res['bias']:+.5f} ({res['pull']:+.2f} sigma)")
    print(f"  fitted a = {res['a']:+.4f} (exact Tracy--Widom value "
          f"{TRACY_WIDOM_MEAN:+.4f}), c = {res['c']:+.4f} (exact -0.3333)")
    return res


# ------------------------------------------------------------------------ main --
def main(remeasure: bool = False) -> None:
    print("Phase 2b Part 1 GATE -- Rideout--Wallden m_3 in M^3")
    print(f"Seeds: d=3  SEED_BASE + 1000*log2N + k  (SEED_BASE={SEED_BASE})")
    print(f"       d=2  CONTROL_SEED_BASE + offset + 1000*log2N + k "
          f"(={CONTROL_SEED_BASE})")
    print(f"tau = {TAU};  V = pi tau^3/12 = {sprinkle3d.diamond_volume_3d(TAU):.6f} "
          f"(eta(3) = {sprinkle3d.DIAMOND_VOLUME_CONSTANT_3D:.8f} = pi/12)")

    rows = load_or_measure(remeasure)
    d = order3d.SPACETIME_DIM_2P1
    n, m_link, se_link = convention(rows, "link", d)
    _, m_inter, se_inter = convention(rows, "interior", d)

    # ------------------------------- GATE 1: comparison with their own curve --
    pred = rw_fit_curve(n)
    curve_err = rw_curve_uncertainty(n)
    combined = np.hypot(se_link, curve_err)
    resid_strict = (m_link - pred) / se_link
    resid_comb = (m_link - pred) / combined

    print("\n=== Measured m_3^eff (link convention) vs RW's own Fig.-4 fit curve ===")
    print(f"{'log2N':>6} {'rhoV':>7} {'<N>':>8} {'#real':>6} {'<L>':>7} {'SE(L)':>6} "
          f"{'<m3link>':>9} {'SE':>7} {'RWfit':>7} {'+/-':>6} {'r/SE':>6} {'r/comb':>7}")
    for i, r in enumerate(rows):
        links = r["elements"] - 1.0
        print(f"{r['log2n']:>6d} {r['rho_volume']:>7.0f} {r['realised_n'].mean():>8.0f} "
              f"{r['n_real']:>6d} {links.mean():>7.2f} "
              f"{links.std(ddof=1) / math.sqrt(r['n_real']):>6.2f} "
              f"{m_link[i]:>9.4f} {se_link[i]:>7.4f} {pred[i]:>7.4f} "
              f"{curve_err[i]:>6.4f} {resid_strict[i]:>+6.2f} {resid_comb[i]:>+7.2f}")

    chi2_strict = float(np.sum(resid_strict**2))
    chi2_comb = float(np.sum(resid_comb**2))
    print("\n=== GATE 1: agreement with Rideout--Wallden's published curve ===")
    print(f"  max |measured - their curve| = {np.abs(m_link - pred).max():.4f} absolute")
    print(f"  strict (our errors only)     chi2/N = {chi2_strict / len(n):.2f}, "
          f"max resid {np.abs(resid_strict).max():.2f} SE")
    print(f"  combined (+ their param errs) chi2/N = {chi2_comb / len(n):.2f}, "
          f"max resid {np.abs(resid_comb).max():.2f} sigma")
    print("  Their curve's own uncertainty is "
          f"{curve_err.min():.4f}-{curve_err.max():.4f}, i.e. LARGER than our "
          "statistical errors")
    print("  everywhere, so the combined test is the fair one but is dominated by "
          "their fit, not ours;")
    print("  the strict test is oversensitive because it treats their fitted curve "
          "as exact. Both are")
    print("  reported. GATE 1 uses the combined test (see rw_curve_uncertainty for "
          "the covariance caveat).")
    gate1 = chi2_comb / len(n) < 4.0 and np.abs(resid_comb).max() < 4.0

    # ---------------------------------- Controls: calibrate the extrapolation --
    ok_a = control_a()
    ctrl_b = run_d2_control(LADDER, 7, "Control B: d=2, MATCHED window and realisation counts")
    ctrl_c = run_d2_control(CONTROL_C_LADDER, 500_003,
                            "Control C: d=2, high statistics to N=2^18")

    # -------------------------------------------- GATE 2: our extrapolation --
    print("\n=== GATE 2: extrapolating our ladder to the asymptote ===")
    print("  Primary quantity: m_inter = I (rho V)^(-1/3), i.e. the link-convention")
    print("  offset +(rho V)^(-1/3) (coefficient exactly 1) removed -- see docstring.")
    c_rw = RW_FIT_B / math.log(2.0)

    primary = fit_ladder(n, m_inter, se_inter,
                         label="PRIMARY: interior convention, c free")
    alt_link_free = fit_ladder(n, m_link, se_link,
                               label="link convention, c free (contaminated by "
                                     "the +N^(-1/3) offset)")
    alt_inter_cfix = fit_ladder(n, m_inter, se_inter, fix_c=c_rw,
                                label="interior convention, c FIXED to RW's b/ln2")
    alt_link_pinned = fit_ladder(n, m_link, se_link, fix_a=RW_FIT_A, fix_c=c_rw,
                                 label="link convention, a and c both pinned to RW")

    # Systematic from the matched-window control: the procedure's measured bias.
    syst = abs(ctrl_b["bias"])
    m3 = primary["m_inf"]
    stat = primary["m_err"]
    total = math.hypot(stat, syst)
    outside = sigma_outside(m3, total, RW_TARGET_LO, RW_TARGET_HI)

    print("\n  Summary of asymptote estimates (d = 3):")
    print(f"    PRIMARY  interior, c free        m_3 = {m3:.4f} +/- {stat:.4f} (stat)")
    print(f"    link, c free                     m_3 = {alt_link_free['m_inf']:.4f} "
          f"+/- {alt_link_free['m_err']:.4f}  (c = {alt_link_free['c']:+.4f})")
    print(f"    interior, c fixed to RW          m_3 = {alt_inter_cfix['m_inf']:.4f} "
          f"+/- {alt_inter_cfix['m_err']:.4f}")
    print(f"    link, a and c pinned to RW       m_3 = {alt_link_pinned['m_inf']:.4f} "
          f"+/- {alt_link_pinned['m_err']:.4f}")
    print(f"    Rideout--Wallden published       m_3 = {RW_M3:.4f} +/- {RW_M3_ERR:.4f}"
          f"   (footnote 11: {RW_M3_FOOTNOTE})")
    print(f"\n  systematic from Control B (matched-window bias in d=2): "
          f"{ctrl_b['bias']:+.4f} -> |syst| = {syst:.4f}")
    print(f"  Control C (better window) bias: {ctrl_c['bias']:+.4f} "
          "-- shrinks as the window improves" if abs(ctrl_c["bias"]) < syst
          else f"  Control C (better window) bias: {ctrl_c['bias']:+.4f}")
    print(f"  OUR RESULT:  m_3 = {m3:.4f} +/- {stat:.4f} (stat) +/- {syst:.4f} (syst) "
          f"= {m3:.4f} +/- {total:.4f}")
    print(f"  target range [{RW_TARGET_LO:.3f}, {RW_TARGET_HI:.3f}]: "
          f"{outside:.2f} sigma outside (0 = inside)")
    gate2 = outside <= 2.0

    # Arithmetic identity check (bookkeeping, not an independent cross-check).
    ident = np.abs((m_link - m_inter) - n ** (-1.0 / d)).max()
    print(f"  identity check  m_link - m_inter == (rho V)^(-1/3): "
          f"max deviation {ident:.2e}")

    # --------------------------------------------------------------- verdict --
    ok_b = abs(ctrl_b["pull"]) <= 5.0
    print("\n" + "=" * 78)
    print(f"PART 1 GATE: {'PASS' if (gate1 and gate2 and ok_a and ok_b) else 'FAIL'}")
    print(f"  GATE 1  matches RW's published curve  : {'pass' if gate1 else 'FAIL'}"
          f"   (combined chi2/N = {chi2_comb / len(n):.2f}, max "
          f"{np.abs(resid_comb).max():.2f} sigma)")
    print(f"  GATE 2  asymptote in [2.278, 2.308]   : {'pass' if gate2 else 'FAIL'}"
          f"   (m_3 = {m3:.4f} +/- {total:.4f}, {outside:.2f} sigma outside)")
    print(f"  Control A  counter == Phase 1 counter : {'pass' if ok_a else 'FAIL'}")
    print(f"  Control B  d=2 matched-window bias    : {'pass' if ok_b else 'FAIL'}"
          f"   (m_2 = {ctrl_b['m_inf']:.5f}, bias {ctrl_b['bias']:+.5f} = "
          f"{ctrl_b['pull']:+.2f} sigma)")
    print(f"  Control C  d=2 high-statistics bias   : "
          f"(m_2 = {ctrl_c['m_inf']:.5f}, bias {ctrl_c['bias']:+.5f} = "
          f"{ctrl_c['pull']:+.2f} sigma)")
    print("=" * 78)

    make_figure(n, m_link, se_link, m_inter, se_inter, pred, curve_err, resid_comb,
                primary, ctrl_c, m3, total)


def make_figure(n, m_link, se_link, m_inter, se_inter, pred, curve_err, resid_comb,
                primary, ctrl_c, m3, total) -> None:
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15.5, 4.4))
    log2n = np.log2(n)
    grid = 2.0 ** np.linspace(math.log2(n.min()), RW_MAX_LOG2N, 300)

    # Panel A: link convention vs RW's curve, with their curve's uncertainty band.
    axA.axhspan(RW_TARGET_LO, RW_TARGET_HI, color="0.85", zorder=0,
                label=rf"RW asymptote {RW_TARGET_LO:.3f}$-${RW_TARGET_HI:.3f}")
    axA.fill_between(np.log2(grid), rw_fit_curve(grid) - rw_curve_uncertainty(grid),
                     rw_fit_curve(grid) + rw_curve_uncertainty(grid),
                     color="tab:red", alpha=0.15, lw=0,
                     label="RW curve $\\pm$ its own param. errors")
    axA.plot(np.log2(grid), rw_fit_curve(grid), "--", color="tab:red", lw=1.3,
             label=r"RW Fig. 4 fit: $2.296-1.087N^{-0.173}$")
    axA.errorbar(log2n, m_link, yerr=se_link, fmt="o", color="black", capsize=3, ms=4,
                 zorder=5, label="this work, link convention")
    axA.axvline(RW_MAX_LOG2N, color="0.6", lw=0.8, ls=":")
    axA.set_xlabel(r"$\log_2(\rho V)$")
    axA.set_ylabel(r"$\langle m_3^{\rm eff}\rangle=\langle L\rangle(\rho V)^{-1/3}$")
    axA.set_title("A. GATE 1: against RW's own curve")
    axA.legend(frameon=False, fontsize=7, loc="lower right")

    # Panel B: the clean interior quantity and our extrapolation.
    axB.axhspan(RW_TARGET_LO, RW_TARGET_HI, color="0.85", zorder=0)
    axB.plot(np.log2(grid), power_law_model(grid, *primary["popt_full"]), "-",
             color="tab:blue", lw=1.4,
             label=rf"fit: $m_\infty={m3:.3f}\pm{total:.3f}$")
    axB.errorbar(log2n, m_inter, yerr=se_inter, fmt="s", color="tab:blue", capsize=3,
                 ms=4, zorder=5, label=r"$I(\rho V)^{-1/3}$ (offset removed)")
    axB.errorbar(log2n, m_link, yerr=se_link, fmt="o", mfc="none", color="0.5",
                 capsize=2, ms=4, label=r"$L(\rho V)^{-1/3}$ (link, for reference)")
    axB.axhline(RW_M3, color="tab:red", ls="--", lw=1.0, label=r"RW $m_3=2.296$")
    axB.set_xlabel(r"$\log_2(\rho V)$")
    axB.set_ylabel(r"$m_3^{\rm eff}$")
    axB.set_title("B. GATE 2: extrapolation")
    axB.legend(frameon=False, fontsize=7, loc="lower right")

    # Panel C: the d=2 control -- same procedure, exactly known answer.
    nc = ctrl_c["n"]
    axC.axhline(BG_M2, color="tab:green", ls="--", lw=1.2, label=r"exact $m_2=2$")
    cgrid = 2.0 ** np.linspace(math.log2(nc.min()), math.log2(nc.max()), 200)
    axC.plot(np.log2(cgrid), power_law_model(cgrid, *ctrl_c["popt_full"]), "-",
             color="tab:green", lw=1.2,
             label=rf"fit: $m_\infty={ctrl_c['m_inf']:.4f}\pm{ctrl_c['m_err']:.4f}$")
    axC.errorbar(np.log2(nc), ctrl_c["mean"], yerr=ctrl_c["se"], fmt="^",
                 color="black", capsize=3, ms=4, zorder=5, label=r"$d=2$ measured")
    axC.set_xlabel(r"$\log_2(\rho V)$")
    axC.set_ylabel(r"$m_2^{\rm eff}=I(\rho V)^{-1/2}$")
    axC.set_title(rf"C. Control: same procedure in $d=2$"
                  f"\nbias $={ctrl_c['bias']:+.4f}$")
    axC.legend(frameon=False, fontsize=7, loc="lower right")

    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / "figures" / "exp02_m3_validation.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"\nSaved figure -> {out}")


if __name__ == "__main__":
    main(remeasure="--remeasure" in sys.argv)
