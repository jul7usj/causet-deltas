"""Phase 3a gates: the Sorkin--Johnston construction on 1+1 D causets vs published results.

Reproduce with:
    python experiments/exp06_sj_1p1d.py gate1              # uses the cache
    python experiments/exp06_sj_1p1d.py gate1 --remeasure  # ignores the cache

GATE 1 -- the spectrum of iDelta
================================
Produces:
    * printed tables (bootstrap errors over realisations, explicit N everywhere),
    * figures/exp06_gate1_spectrum.png,
    * data/exp06_gate1_spectra.npz (every eigenvalue of every realisation),
    * an explicit PASS / FAIL per criterion.

Published claim being tested (Sorkin--Yazdi, CQG 35 (2018) 074004,
arXiv:1611.10281, Sec. 3 and their Fig. "spec", N = 200, rho = 50): the causet
spectrum of iDelta, rescaled by 1/rho, agrees with the continuum diamond
spectrum above ``lambda^cs = sqrt(N)/(4 pi)`` and is "in very poor agreement
below it", with a "break" in the causet spectrum around there.

Continuum target: ABDRSY 1207.7101 eqs. (SJfunctions1)-(SJfunctions2),
positive eigenvalues ``L/k``; for a sprinkling ``lambda_m^pred = N/(4 x_m)``
(``causet.sj_continuum``), using ``rho = N/V`` per realisation as Sorkin--Yazdi
do (``rho = N_l / 4 l^2``). Comparison is RANK-ORDERED: the m-th largest causet
eigenvalue against the m-th largest continuum eigenvalue.

Design, FIXED BEFORE THE RUN
----------------------------
One exploratory realisation (N = 1035, seed 999999, outside the seed range
used here) was inspected before this file was written. It showed agreement to
~3% down to the Sorkin--Yazdi landmark rank and then a GRADUAL roll-off below
the continuum (ratio 0.93 at rank 150, 0.45 at rank 400), not a sharp break.
A "knee" is therefore necessarily threshold-defined, and the definition below
is a family of thresholds whose agreement with each other is itself reported.

* ratio  ``r_m = lambda_m^cs / lambda_m^pred``, per realisation; ensemble mean
  ``rbar_m`` over the realisations of a rung.
* knee   ``m*(delta) = 1 + max{ m : rbar_m >= 1 - delta }`` -- the LAST rank from
  which the ensemble ratio stays below ``1 - delta``. One-sided because the
  roll-off is downward; insensitive to isolated noise dips near the top.
  ``delta in DELTAS``; PRIMARY delta = 0.10.
* knee eigenvalue ``lambda*(delta)``: ensemble-mean causet eigenvalue at rank
  ``m*``, reported in units of the Sorkin--Yazdi value ``sqrt(N)/(4 pi)``.
* Sorkin--Yazdi landmark rank ``m_SY(N)``: number of predicted eigenvalues
  ``>= sqrt(N)/(4 pi)``.
* Errors: bootstrap over realisations (``N_BOOT`` resamples, seed ``BOOT_SEED``);
  scaling exponents from the bootstrap distribution of the OLS slope of
  ``log m*`` (resp. ``log lambda*``) on ``log <N>``.

Criteria, FIXED BEFORE THE RUN
------------------------------
G1-A  agreement above the landmark: at every N,
      ``max_{m <= m_SY} |rbar_m - 1| <= AGREE_TOL`` (0.05).
G1-B  a knee exists and moves with N: the primary-delta exponent
      ``alpha = d log m* / d log N`` is > 0 by at least 3 bootstrap sigma.
GATE 1 PASSES iff G1-A and G1-B.
G1-C  comparison with the published landmark (reported, not part of PASS):
      Sorkin--Yazdi's ``lambda* ~ sqrt(N)/(4 pi)`` against ``lambda^pred ~ N/m``
      implies ``m* ~ sqrt(N)``: alpha = 1/2. CONSISTENT iff ``|alpha - 1/2|
      <= 2 sigma`` at the primary delta; otherwise DISCREPANT and logged as such.

Cost (measured before the run): one ``eigvalsh`` of the complex Hermitian
iDelta takes 0.2 s at N = 512, 0.9 s at 1024, 3.8 s at 2048, 19 s at 4096
(OpenBLAS, 8 threads). Realisation count is preferred over N (as in Phase 2b):
it halves as N doubles. Largest N = 4096.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from causet import sj  # noqa: E402
from causet import sj_continuum as sc  # noqa: E402
from causet.order import causal_matrix_1d  # noqa: E402
from causet.sprinkle import sprinkle_diamond_1d  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# ---- Gate 1 parameters (Integrity Rule 4: named, documented) -----------------
SEED_BASE = 20261005
#: Diamond proper time. Irrelevant to the spectrum (scale-free; see
#: sj_continuum) but fixed so every realisation is reproducible from its seed.
TAU = 1.0
#: Nominal (expected) element counts. rho = N_nom / V with V = tau^2/2.
N_LADDER = [256, 512, 1024, 2048, 4096]
#: Realisations per rung: halves as N doubles (cost ~ N^3).
N_REAL = [320, 160, 80, 40, 20]
#: Knee thresholds; the primary one is PRIMARY_DELTA.
DELTAS = [0.02, 0.05, 0.10, 0.20]
PRIMARY_DELTA = 0.10
#: G1-A tolerance on |rbar_m - 1| above the Sorkin--Yazdi landmark.
AGREE_TOL = 0.05
#: G1-B significance and G1-C consistency, in bootstrap sigma.
KNEE_SIGNIFICANCE = 3.0
SY_CONSISTENCY = 2.0
#: Sorkin--Yazdi 1611.10281 Sec. 3: break at lambda^cs ~ sqrt(N)/(4 pi).
SY_PREFACTOR = 1.0 / (4.0 * np.pi)
N_BOOT = 2000
BOOT_SEED = 77

CACHE1 = ROOT / "data" / "exp06_gate1_spectra.npz"
FIG1 = ROOT / "figures" / "exp06_gate1_spectrum.png"


# ------------------------------------------------------------------ measurement --
def measure_gate1_rung(i: int) -> dict:
    n_nom, n_real = N_LADDER[i], N_REAL[i]
    rho = n_nom / (0.5 * TAU * TAU)
    evs, ns, rels, secs = [], [], [], []
    t_rung = time.time()
    for k in range(n_real):
        seed = SEED_BASE + 100000 * i + k
        s = sprinkle_diamond_1d(rho, TAU, seed)
        c = causal_matrix_1d(s.u, s.v)
        t0 = time.time()
        ev = np.linalg.eigvalsh(sj.pauli_jordan(sj.retarded_green_2d(c)))
        secs.append(time.time() - t0)
        evs.append(ev)
        ns.append(s.n)
        rels.append(int(c.sum()))
    print(f"  N_nom={n_nom:5d}: {n_real} realisations in {time.time() - t_rung:7.1f} s "
          f"(eigvalsh {np.mean(secs):.2f} s each)", flush=True)
    width = max(ns)
    pad = np.full((n_real, width), np.nan)
    for k, ev in enumerate(evs):
        pad[k, : ev.size] = ev
    return {"ev": pad, "n": np.array(ns), "rel": np.array(rels), "sec": np.array(secs)}


def gate1_key() -> str:
    return f"g1v1|{SEED_BASE}|{TAU}|" + ",".join(f"{a}:{b}" for a, b in zip(N_LADDER, N_REAL))


def load_or_measure_gate1(remeasure: bool) -> list[dict]:
    key = gate1_key()
    if not remeasure and CACHE1.exists():
        z = np.load(CACHE1, allow_pickle=False)
        if str(z["key"]) == key:
            print(f"Loaded cached spectra from {CACHE1}")
            return [{f: z[f"{f}_{i}"] for f in ("ev", "n", "rel", "sec")}
                    for i in range(len(N_LADDER))]
        print("Cache present but parameters differ -- remeasuring.")
    print(f"Measuring {sum(N_REAL)} realisations over N = {N_LADDER} ...", flush=True)
    rows = [measure_gate1_rung(i) for i in range(len(N_LADDER))]
    CACHE1.parent.mkdir(exist_ok=True)
    payload = {"key": np.array(key)}
    for i, r in enumerate(rows):
        for f, v in r.items():
            payload[f"{f}_{i}"] = v
    np.savez(CACHE1, **payload)
    print(f"Saved raw spectra -> {CACHE1}")
    return rows


# -------------------------------------------------------------------- analysis --
def positive_descending(ev_row: np.ndarray, n: int) -> np.ndarray:
    ev = ev_row[:n]
    tol = sj.zero_tolerance(ev, n)
    return ev[ev > tol][::-1]


def knee_rank(rbar: np.ndarray, delta: float) -> int:
    """``1 + max{m : rbar_m >= 1 - delta}`` (1-based ranks); len+1 if never left."""
    ok = np.nonzero(rbar >= 1.0 - delta)[0]
    return int(ok.max()) + 2 if ok.size else 1


def ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.polyfit(x, y, 1)[0])


def analyse_rung(row: dict) -> dict:
    ns = row["n"].astype(int)
    pos = [positive_descending(row["ev"][k], ns[k]) for k in range(ns.size)]
    m_common = min(p.size for p in pos)
    lam = np.stack([p[:m_common] for p in pos])               # (R, M)
    x = sc.diamond_spectrum_x(m_common)
    pred = ns[:, None] / (4.0 * x[None, :])                    # (R, M)
    ratio = lam / pred
    # structural checks per realisation
    pair_err, hs_err = [], []
    for k in range(ns.size):
        ev = row["ev"][k, : ns[k]]
        pair_err.append(np.max(np.abs(ev + ev[::-1])) / np.max(np.abs(ev)))
        hs_err.append(abs(np.sum(ev**2) - row["rel"][k] / 2) / (row["rel"][k] / 2))
    n_mean = float(ns.mean())
    pred_mean = sc.predicted_causet_spectrum(n_mean, m_common)
    m_sy = int(np.sum(pred_mean >= SY_PREFACTOR * np.sqrt(n_mean)))
    return {"lam": lam, "ratio": ratio, "n_mean": n_mean, "n_sd": float(ns.std(ddof=1)),
            "m_common": m_common, "m_sy": m_sy, "pair_err": max(pair_err),
            "hs_err": max(hs_err), "R": ns.size}


def knee_stats(a: dict, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For a resample ``idx`` of realisations: rbar, m*(delta), lambda*(delta)."""
    rbar = a["ratio"][idx].mean(axis=0)
    lbar = a["lam"][idx].mean(axis=0)
    ms = np.array([knee_rank(rbar, d) for d in DELTAS])
    ls = np.array([lbar[m - 1] if m <= lbar.size else np.nan for m in ms])
    return rbar, ms, ls


def gate1(remeasure: bool) -> None:
    rows = load_or_measure_gate1(remeasure)
    an = [analyse_rung(r) for r in rows]
    n_mean = np.array([a["n_mean"] for a in an])
    logn = np.log(n_mean)
    nd = len(DELTAS)
    ip = DELTAS.index(PRIMARY_DELTA)

    # point estimates
    pt = [knee_stats(a, np.arange(a["R"])) for a in an]
    rbar = [p[0] for p in pt]
    m_pt = np.array([p[1] for p in pt], dtype=float)          # (rungs, nd)
    l_pt = np.array([p[2] for p in pt])
    rse = [a["ratio"].std(axis=0, ddof=1) / np.sqrt(a["R"]) for a in an]

    # bootstrap
    rng = np.random.default_rng(BOOT_SEED)
    m_b = np.empty((N_BOOT, len(an), nd))
    l_b = np.empty((N_BOOT, len(an), nd))
    agree_b = np.empty((N_BOOT, len(an)))
    for b in range(N_BOOT):
        for j, a in enumerate(an):
            idx = rng.integers(0, a["R"], a["R"])
            rb, m_b[b, j], l_b[b, j] = knee_stats(a, idx)
            agree_b[b, j] = np.max(np.abs(rb[: a["m_sy"]] - 1.0))
    alpha_b = np.array([[ols_slope(logn, np.log(m_b[b, :, d])) for d in range(nd)]
                        for b in range(N_BOOT)])
    beta_b = np.array([[ols_slope(logn, np.log(l_b[b, :, d])) for d in range(nd)]
                       for b in range(N_BOOT)])
    alpha_pt = np.array([ols_slope(logn, np.log(m_pt[:, d])) for d in range(nd)])
    beta_pt = np.array([ols_slope(logn, np.log(l_pt[:, d])) for d in range(nd)])
    alpha_se = alpha_b.std(axis=0, ddof=1)
    beta_se = beta_b.std(axis=0, ddof=1)

    # ---------------------------------------------------------------- report --
    print("\nGATE 1 -- spectrum of iDelta vs continuum diamond (ABDRSY 1207.7101)")
    print("=" * 78)
    print("Structural checks (max over realisations):")
    for j, a in enumerate(an):
        print(f"  N_nom={N_LADDER[j]:5d}  <N>={a['n_mean']:7.1f} +- {a['n_sd']:5.1f}  "
              f"R={a['R']:3d}  pairing |l+l'|/lmax={a['pair_err']:.1e}  "
              f"HS |sum l^2 - R/2|/(R/2)={a['hs_err']:.1e}  "
              f"eigvalsh {rows[j]['sec'].mean():.2f} s")

    print("\nG1-A: agreement down to the Sorkin--Yazdi landmark sqrt(N)/4pi")
    print(f"  {'N_nom':>6} {'m_SY':>5} {'rbar(1)':>14} {'rbar(m_SY)':>14} "
          f"{'max|rbar-1|, m<=m_SY':>22} {'at m':>5}")
    g1a = True
    for j, a in enumerate(an):
        dev = np.abs(rbar[j][: a["m_sy"]] - 1.0)
        mdev = float(dev.max())
        g1a &= mdev <= AGREE_TOL
        msy = a["m_sy"]
        print(f"  {N_LADDER[j]:6d} {msy:5d} "
              f"{rbar[j][0]:7.4f}+-{rse[j][0]:.4f} "
              f"{rbar[j][msy - 1]:7.4f}+-{rse[j][msy - 1]:.4f} "
              f"{mdev:13.4f} +- {agree_b[:, j].std(ddof=1):.4f} "
              f"{int(dev.argmax()) + 1:5d}")
    print(f"  -> G1-A {'PASS' if g1a else 'FAIL'} (tolerance {AGREE_TOL})")

    print("\nKnee m*(delta) = 1 + last rank with rbar >= 1 - delta  (bootstrap SE)")
    hdr = "".join(f"   delta={d:<4}       " for d in DELTAS)
    print(f"  {'N_nom':>6} {'m_SY':>5}" + hdr)
    for j, a in enumerate(an):
        cells = "".join(f"  {m_pt[j, d]:6.0f} +- {m_b[:, j, d].std(ddof=1):5.1f}     "
                        for d in range(nd))
        print(f"  {N_LADDER[j]:6d} {a['m_sy']:5d}" + cells)
    print("\n  knee eigenvalue lambda*(delta) in units of sqrt(N)/4pi")
    for j, a in enumerate(an):
        unit = SY_PREFACTOR * np.sqrt(a["n_mean"])
        cells = "".join(f"  {l_pt[j, d] / unit:6.3f} +- {l_b[:, j, d].std(ddof=1) / unit:5.3f}     "
                        for d in range(nd))
        print(f"  {N_LADDER[j]:6d}      " + cells)
    print("\n  Scaling exponents (OLS on log <N>, bootstrap SE):")
    for d in range(nd):
        tag = "  <- PRIMARY" if d == ip else ""
        print(f"    delta={DELTAS[d]:<4}  m* ~ N^({alpha_pt[d]:.3f} +- {alpha_se[d]:.3f})   "
              f"lambda* ~ N^({beta_pt[d]:.3f} +- {beta_se[d]:.3f}){tag}")

    a_p, s_p = alpha_pt[ip], alpha_se[ip]
    g1b = a_p > KNEE_SIGNIFICANCE * s_p
    z_sy = abs(a_p - 0.5) / s_p
    g1c = z_sy <= SY_CONSISTENCY
    print(f"\n  -> G1-B {'PASS' if g1b else 'FAIL'}: alpha = {a_p:.3f} +- {s_p:.3f} "
          f"({a_p / s_p:.1f} sigma from 0; need >= {KNEE_SIGNIFICANCE})")
    print(f"  -> G1-C {'CONSISTENT' if g1c else 'DISCREPANT'} with Sorkin--Yazdi "
          f"sqrt(N) (alpha = 1/2): {z_sy:.1f} sigma (threshold {SY_CONSISTENCY})")
    print(f"\nGATE 1: {'PASS' if (g1a and g1b) else 'FAIL'}  (G1-A and G1-B; G1-C reported)")
    print(f"Largest N used: {N_LADDER[-1]} (realised <N> = {an[-1]['n_mean']:.0f}); "
          f"eigvalsh {rows[-1]['sec'].mean():.1f} s per realisation there.")

    make_gate1_figure(an, rbar, rse, m_pt, m_b, alpha_pt, alpha_se, n_mean)
    print(f"Figure -> {FIG1}")


# --------------------------------------------------------------------- figure --
def make_gate1_figure(an, rbar, rse, m_pt, m_b, alpha_pt, alpha_se, n_mean) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(22.0, 5.4))
    cmap = plt.get_cmap("Blues")
    cols = [cmap(0.35 + 0.6 * j / (len(an) - 1)) for j in range(len(an))]
    ip = DELTAS.index(PRIMARY_DELTA)

    ax = axes[0]
    for j, a in enumerate(an):
        m = np.arange(1, a["m_common"] + 1)
        ax.loglog(m, a["lam"].mean(axis=0) / a["n_mean"], color=cols[j], lw=1.6,
                  label=f"causet <N>={a['n_mean']:.0f} (R={a['R']})")
        ax.plot(a["m_sy"], SY_PREFACTOR / np.sqrt(a["n_mean"]), "v", color=cols[j], ms=8,
                mec="k", mew=0.6)
    mm = np.arange(1, an[-1]["m_common"] + 1)
    ax.loglog(mm, 1.0 / (4.0 * sc.diamond_spectrum_x(mm.size)), "k--", lw=1.2,
              label="continuum 1/(4 x_m)  (ABDRSY)")
    ax.set_xlabel("rank m")
    ax.set_ylabel("lambda_m / N   (ensemble mean)")
    ax.set_title("A. Positive spectrum of iDelta, scaled by N\n"
                 "triangles: Sorkin-Yazdi landmark sqrt(N)/4pi", fontsize=9.5)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25, which="both")

    for ax, scale, lab in ((axes[1], "sqrt", "m / sqrt(<N>)"), (axes[2], "lin", "m / <N>")):
        for j, a in enumerate(an):
            m = np.arange(1, a["m_common"] + 1)
            xs = m / (np.sqrt(a["n_mean"]) if scale == "sqrt" else a["n_mean"])
            ax.plot(xs, rbar[j], color=cols[j], lw=1.4, label=f"<N>={a['n_mean']:.0f}")
            ax.fill_between(xs, rbar[j] - rse[j], rbar[j] + rse[j], color=cols[j],
                            alpha=0.25, lw=0)
        ax.axhline(1.0, color="k", lw=1.0, ls=":")
        ax.axhline(1.0 - PRIMARY_DELTA, color="0.4", lw=1.0, ls="--",
                   label=f"1 - delta (primary, {PRIMARY_DELTA})")
        ax.set_xscale("log")
        ax.set_ylim(0.0, 1.15)
        ax.set_xlabel(lab + "   (log)")
        ax.set_ylabel("rbar_m = lambda^cs / lambda^pred  (+- SE)")
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=7, loc="lower left")
    axes[1].set_title("B. Ratio vs rank / sqrt(N)\n"
                      "(collapse here => knee ~ sqrt(N), Sorkin-Yazdi)", fontsize=9.5)
    axes[2].set_title("C. Ratio vs rank / N\n"
                      "(collapse here => knee ~ N)", fontsize=9.5)

    ax = axes[3]
    greys = plt.get_cmap("Greys")
    for d, delta in enumerate(DELTAS):
        c = "tab:blue" if d == DELTAS.index(PRIMARY_DELTA) else greys(0.4 + 0.15 * d)
        ax.errorbar(n_mean, m_pt[:, d], yerr=m_b[:, :, d].std(axis=0, ddof=1), fmt="o-",
                    ms=5, capsize=3, color=c, lw=2.0 if d == ip else 1.0,
                    label=f"delta={delta}: N^({alpha_pt[d]:.3f} +- {alpha_se[d]:.3f})")
    msy = np.array([a["m_sy"] for a in an])
    ax.plot(n_mean, msy, "kv--", ms=6, lw=1.0, label="m_SY (sqrt(N)/4pi landmark)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("<N>")
    ax.set_ylabel("knee rank m*(delta)")
    ax.set_title("D. Knee position vs N (bootstrap SE)\n"
                 "Sorkin-Yazdi predicts slope 1/2", fontsize=9.5)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25, which="both")

    fig.suptitle(
        "Phase 3a GATE 1 -- iDelta spectrum on 1+1 D diamond causets vs continuum "
        f"(seeds {SEED_BASE}+100000 i + k; R = {N_REAL}; Hermitian eigvalsh)",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    FIG1.parent.mkdir(exist_ok=True)
    fig.savefig(FIG1, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] not in ("gate1",):
        sys.exit("usage: exp06_sj_1p1d.py gate1 [--remeasure]")
    gate1(remeasure="--remeasure" in args)
