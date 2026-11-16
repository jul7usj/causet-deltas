"""Phase 3a GATE 3: Sorkin's spacetime entanglement entropy of nested 1+1 D diamonds.

Reproduce with:
    python experiments/exp07_ssee_1p1d.py            # uses the cache
    python experiments/exp07_ssee_1p1d.py --remeasure

Published claim (Sorkin--Yazdi, CQG 35 (2018) 074004, arXiv:1611.10281, Sec. 3,
Fig. "trunc"; eqs. (gee), (s4)): with the double truncation
lambda_min ~ sqrt(N_L)/4pi (outer diamond) then ~ sqrt(N_l)/4pi (inner diamond),
l/L = 1/2, a fit S = a ln(x) + b with x = sqrt(N_l)/4pi gives
a = 0.346 +- 0.028, b = 1.883 +- 0.035 (continuum 1/3). Untruncated: S = a N_l + b
with a = 0.32, b = -6.64 at l/L = 1/2 (Fig. "sn04"; a spacetime-volume law).

Design and criteria: FIXED in RESULTS.md (entry 2026-11-16, pre-registration),
committed and pushed before any data were generated. Summary:

* Outer diamond tau = 1; inner diamond concentric, |u/L|, |v/L| <= 1/2 (l/L = 1/2).
* Ladder N_L(nominal) = 512, 724, 1024, 1448, 2048, 2896, 4096; R below.
  PRIMARY fit uses N_L >= 1024 (x ~ 1.27-2.55, inside SY's x range 1.26-4.9).
* Outer: positive eigenpairs of iDelta for the top ``TOP_FRAC`` of ranks
  (scipy eigh, driver evr, subset by index). Negative partners are conjugates.
* Variants from the same outer solve: SY cut with prefactor c in ``SWEEP``
  (applied to BOTH cuts); SXY linear truncation (sj.linear_truncation_rank,
  delta = 0.1, both cuts); CONTROL untruncated (full eigh) on the first
  ``R_CTRL`` realisations of rungs N_L <= ``CTRL_MAX_N``.
* Fit: OLS of per-realisation S on ln x; SE = max(OLS SE, bootstrap SE over
  realisations within rungs, ``N_BOOT`` resamples, seed ``BOOT_SEED``).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg as sl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from causet import sj, ssee  # noqa: E402
from causet import sj_continuum as sc  # noqa: E402
from causet.order import causal_matrix_1d  # noqa: E402
from causet.sprinkle import sprinkle_diamond_1d  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEED_BASE3 = 20261116
TAU = 1.0
N_LADDER3 = [512, 724, 1024, 1448, 2048, 2896, 4096]
N_REAL3 = [96, 64, 48, 40, 32, 24, 16]
#: Inner diamond half-width in scaled coordinates: l/L = 1/2 (SY Fig. "sn04", "trunc").
INNER_HALF = 0.5
#: Outer eigenvectors computed for the top TOP_FRAC of ranks. Every truncation here
#: keeps far fewer (n_lin/N <= 0.132 in g1lin); hitting the cap is flagged.
TOP_FRAC = 0.30
SWEEP = [0.5, 0.71, 1.0, 1.41, 2.0]
PRIMARY_MIN_N = 1024
R_CTRL = 16
CTRL_MAX_N = 2048
#: Published targets (SY 1611.10281 Sec. 3).
SY_A, SY_A_ERR, SY_B = 0.346, 0.028, 1.883
SY_VOL_A, SY_VOL_B = 0.32, -6.64
CFT_A = 1.0 / 3.0
PASS_Z = 2.0
N_BOOT = 2000
BOOT_SEED = 77

CACHE3 = ROOT / "data" / "exp07_gate3_ssee.npz"
FIG3 = ROOT / "figures" / "exp07_gate3_ssee.png"
FIELDS = ("n_L", "n_l", "sec", "S_sy", "nout_sy", "nin_sy", "S_lin", "nout_lin", "nin_lin",
          "S_ctrl", "nin_ctrl", "max_imag", "capped")


def measure_realisation(i: int, k: int) -> dict:
    n_nom = N_LADDER3[i]
    s = sprinkle_diamond_1d(n_nom / (0.5 * TAU * TAU), TAU, SEED_BASE3 + 100000 * i + k)
    i_delta = sj.pauli_jordan(sj.retarded_green_2d(causal_matrix_1d(s.u, s.v)))
    t0 = time.time()
    kk = int(np.ceil(TOP_FRAC * s.n))
    lam, vec = sl.eigh(i_delta, subset_by_index=[s.n - kk, s.n - 1], driver="evr")
    lam, vec = lam[::-1], vec[:, ::-1]
    ub, vb = sc.scaled_from_sprinkle(s.u, TAU), sc.scaled_from_sprinkle(s.v, TAU)
    inner = np.nonzero((np.abs(ub) <= INNER_HALF) & (np.abs(vb) <= INNER_HALF))[0]
    n_l = inner.size
    out = {"n_L": s.n, "n_l": n_l, "S_sy": [], "nout_sy": [], "nin_sy": []}
    imag, capped = 0.0, False
    for c in SWEEP:
        cut1 = c * sj.SY_TRUNCATION_PREFACTOR * np.sqrt(s.n)
        n_out = int(np.sum(lam >= cut1))
        capped |= n_out >= kk
        r = ssee.double_truncated_ssee(lam, vec, inner, n_out,
                                       c * sj.SY_TRUNCATION_PREFACTOR * np.sqrt(n_l))
        out["S_sy"].append(r["S"])
        out["nout_sy"].append(n_out)
        out["nin_sy"].append(r["n_inner"])
        imag = max(imag, r["max_imag"])
    n_lin = sj.linear_truncation_rank(lam)
    capped |= n_lin >= kk
    r = ssee.double_truncated_ssee(lam, vec, inner, n_lin, sj.linear_truncation_rank)
    out.update(S_lin=r["S"], nout_lin=n_lin, nin_lin=r["n_inner"])
    imag = max(imag, r["max_imag"])
    out.update(S_ctrl=np.nan, nin_ctrl=-1)
    if n_nom <= CTRL_MAX_N and k < R_CTRL:
        ev, v = np.linalg.eigh(i_delta)
        pos = ev > sj.zero_tolerance(ev, s.n)
        vi = v[np.ix_(inner, np.nonzero(pos)[0])]
        w_o = (vi * ev[pos]) @ vi.conj().T
        r = ssee.untruncated_ssee(w_o, i_delta[np.ix_(inner, inner)])
        out.update(S_ctrl=r["S"], nin_ctrl=r["n_inner"])
        imag = max(imag, r["max_imag"])
    out.update(sec=time.time() - t0, max_imag=imag, capped=capped)
    return out


def cache_key() -> str:
    return (f"g3v1|{SEED_BASE3}|{TAU}|{INNER_HALF}|{TOP_FRAC}|{SWEEP}|{R_CTRL}|{CTRL_MAX_N}|"
            + ",".join(f"{a}:{b}" for a, b in zip(N_LADDER3, N_REAL3))
            + f"|lin:{sj.LIN_DELTA},{sj.LIN_BINS_PER_DECADE},{sj.LIN_FIRST_RANK}")


def load_or_measure(remeasure: bool) -> dict:
    key = cache_key()
    part = CACHE3.with_suffix(".partial.npz")
    if not remeasure and CACHE3.exists():
        z = np.load(CACHE3, allow_pickle=False)
        if str(z["key"]) == key:
            print(f"Loaded {CACHE3.name}")
            return {f: z[f] for f in z.files if f != "key"}
    recs: list[dict] = []
    if not remeasure and part.exists():
        z = np.load(part, allow_pickle=False)
        if str(z["key"]) == key:
            recs = [{f: z[f][j] for f in FIELDS} | {"rung": int(z["rung"][j])}
                    for j in range(z["rung"].size)]
            print(f"Resuming with {len(recs)} realisations done")
    counts = [sum(1 for r in recs if r["rung"] == i) for i in range(len(N_LADDER3))]
    for i in range(len(N_LADDER3)):
        t_r = time.time()
        for k in range(counts[i], N_REAL3[i]):
            rec = measure_realisation(i, k)
            rec["rung"] = i
            recs.append(rec)
            flat = _flatten(recs)
            np.savez(part, key=np.array(key), **flat)
            if N_LADDER3[i] >= 2048:
                print(f"    N_L={N_LADDER3[i]} k={k:3d} N={rec['n_L']} N_l={rec['n_l']} "
                      f"{rec['sec']:.1f} s S(c=1)={rec['S_sy'][2]:.3f} S_lin={rec['S_lin']:.3f}",
                      flush=True)
        print(f"  N_L={N_LADDER3[i]:5d}: R={N_REAL3[i]} in {time.time() - t_r:7.1f} s", flush=True)
    flat = _flatten(recs)
    np.savez_compressed(CACHE3, key=np.array(key), **flat)
    part.unlink(missing_ok=True)
    return flat


def _flatten(recs: list[dict]) -> dict:
    return {f: np.array([r[f] for r in recs]) for f in FIELDS + ("rung",)}


# ------------------------------------------------------------------- analysis --
def ols(x, y):
    a_mat = np.c_[np.ones_like(x), x]
    coef, *_ = np.linalg.lstsq(a_mat, y, rcond=None)
    r = y - a_mat @ coef
    cov = (r @ r / (y.size - 2)) * np.linalg.inv(a_mat.T @ a_mat)
    return coef, np.sqrt(np.diag(cov)), float(r @ r)


def fit_log(d: dict, s_col: np.ndarray, sel: np.ndarray, rng) -> dict:
    """S = a ln(sqrt(N_l)/4pi) + b over the selected realisations."""
    x = np.log(np.sqrt(d["n_l"][sel]) / (4 * np.pi))
    y = s_col[sel]
    coef, se, _ = ols(x, y)
    rung = d["rung"][sel]
    idx_by = [np.nonzero(rung == r)[0] for r in np.unique(rung)]
    boot = []
    for _ in range(N_BOOT):
        j = np.concatenate([g[rng.integers(0, g.size, g.size)] for g in idx_by])
        boot.append(ols(x[j], y[j])[0])
    boot = np.array(boot)
    se_a = max(se[1], boot[:, 1].std(ddof=1))
    se_b = max(se[0], boot[:, 0].std(ddof=1))
    return {"a": coef[1], "b": coef[0], "se_a": se_a, "se_b": se_b, "boot_a": boot[:, 1]}


def main(remeasure: bool) -> None:
    d = load_or_measure(remeasure)
    rng = np.random.default_rng(BOOT_SEED)
    n_nom = np.array(N_LADDER3)[d["rung"]]
    prim = n_nom >= PRIMARY_MIN_N
    allr = np.ones_like(prim)
    ic = SWEEP.index(1.0)

    print("\nGATE 3 -- SSEE of nested diamonds (l/L = 1/2) vs Sorkin--Yazdi 1611.10281")
    print("=" * 84)
    for i, n in enumerate(N_LADDER3):
        m = d["rung"] == i
        print(f"  N_L={n:5d} R={m.sum():3d} <N_L>={d['n_L'][m].mean():7.1f} "
              f"<N_l>={d['n_l'][m].mean():6.1f} x={np.sqrt(d['n_l'][m].mean()) / 4 / np.pi:5.2f}"
              f"  solve {d['sec'][m].mean():6.1f} s  n_out(c=1) {d['nout_sy'][m, ic].mean():6.1f}"
              f" n_in(c=1) {d['nin_sy'][m, ic].mean():5.1f}  n_lin out/in "
              f"{d['nout_lin'][m].mean():6.1f}/{d['nin_lin'][m].mean():6.1f}  "
              f"<S(c=1)> {d['S_sy'][m, ic].mean():.3f}+-{d['S_sy'][m, ic].std(ddof=1) / np.sqrt(m.sum()):.3f}"
              f"  <S_lin> {d['S_lin'][m].mean():.3f}")
    print(f"  structural: max |Im lambda| = {d['max_imag'].max():.1e}; capped realisations: "
          f"{int(d['capped'].sum())}")

    # PRIMARY
    f1 = fit_log(d, d["S_sy"][:, ic], prim, rng)
    comb = np.hypot(f1["se_a"], SY_A_ERR)
    z_sy = (f1["a"] - SY_A) / comb
    z_cft = (f1["a"] - CFT_A) / f1["se_a"]
    ok = abs(z_sy) <= PASS_Z
    print(f"\nPRIMARY (c = 1, N_L >= {PRIMARY_MIN_N}): a = {f1['a']:.4f} +- {f1['se_a']:.4f}, "
          f"b = {f1['b']:.4f} +- {f1['se_b']:.4f}")
    print(f"  vs SY a = {SY_A} +- {SY_A_ERR}: z = {z_sy:+.2f} (combined {comb:.4f}) -> "
          f"{'PASS' if ok else 'FAIL'} (|z| <= {PASS_Z});  vs b = {SY_B}: "
          f"{(f1['b'] - SY_B) / np.hypot(f1['se_b'], 0.035):+.2f} sigma (reported)")
    print(f"  vs continuum 1/3: z = {z_cft:+.2f}")
    fa = fit_log(d, d["S_sy"][:, ic], allr, rng)
    print(f"  all rungs (N_L >= 512, reported): a = {fa['a']:.4f} +- {fa['se_a']:.4f}")

    # SWEEP
    print("\nPREFACTOR SWEEP (both cuts x c), N_L >= 1024:")
    fs = [fit_log(d, d["S_sy"][:, j], prim, rng) for j in range(len(SWEEP))]
    for j, c in enumerate(SWEEP):
        print(f"  c = {c:4.2f}: a = {fs[j]['a']:.4f} +- {fs[j]['se_a']:.4f}  b = {fs[j]['b']:.3f}"
              f"   a - a(1) = {fs[j]['a'] - f1['a']:+.4f}  "
              f"(|.|/SE(a(1)) = {abs(fs[j]['a'] - f1['a']) / f1['se_a']:.2f})")
    within = [abs(fs[j]["a"] - f1["a"]) < f1["se_a"] for j in range(len(SWEEP))]
    lo = ic
    while lo - 1 >= 0 and within[lo - 1]:
        lo -= 1
    hi = ic
    while hi + 1 < len(SWEEP) and within[hi + 1]:
        hi += 1
    plateau = within[ic - 1] and within[ic + 1]
    print(f"  contiguous run with |a(c) - a(1)| < SE(a(1)): c in [{SWEEP[lo]}, {SWEEP[hi]}]")
    print(f"  -> {'PLATEAU' if plateau else 'NO PLATEAU'} (needs both c = 0.71 and 1.41)")

    # SECONDARY: linear
    flin = fit_log(d, d["S_lin"], prim, rng)
    diff_b = flin["boot_a"] - f1["boot_a"]
    se_d = max(diff_b.std(ddof=1), 1e-12)
    z_d = (flin["a"] - f1["a"]) / se_d
    same = abs(z_d) <= PASS_Z
    print(f"\nSECONDARY (linear truncation): a = {flin['a']:.4f} +- {flin['se_a']:.4f}, "
          f"b = {flin['b']:.3f}")
    print(f"  a_lin - a_SY = {flin['a'] - f1['a']:+.4f} +- {se_d:.4f} (paired bootstrap), "
          f"z = {z_d:+.2f} -> {'SAME' if same else 'DIFFERENT'}")
    print(f"  a_lin vs 1/3: z = {(flin['a'] - CFT_A) / flin['se_a']:+.2f};  vs 0.346: z = "
          f"{(flin['a'] - SY_A) / np.hypot(flin['se_a'], SY_A_ERR):+.2f}")

    # CONTROL
    m = np.isfinite(d["S_ctrl"])
    nl, sv = d["n_l"][m].astype(float), d["S_ctrl"][m]
    cl, sel_, rss_l = ols(nl, sv)
    cg, seg, rss_g = ols(np.log(nl), sv)
    z_vol = cl[1] / sel_[1]
    vol = (z_vol >= 5) and (rss_l < rss_g)
    print(f"\nCONTROL (untruncated, {m.sum()} realisations, N_l = {nl.min():.0f}-{nl.max():.0f}):")
    print(f"  S = alpha N_l + beta: alpha = {cl[1]:.4f} +- {sel_[1]:.4f} ({z_vol:.1f} sigma), "
          f"beta = {cl[0]:.2f};  RSS {rss_l:.1f}")
    print(f"  S = a ln N_l + b:     a = {cg[1]:.3f} +- {seg[1]:.3f};  RSS {rss_g:.1f}")
    print(f"  SY (l/L = 1/2, N_l up to ~2000): alpha = {SY_VOL_A}, beta = {SY_VOL_B} (reported)")
    print(f"  -> VOLUME LAW {'REPRODUCED' if vol else 'NOT REPRODUCED'} "
          "(alpha >= 5 sigma and linear RSS < log RSS)")

    print(f"\nPRIMARY VERDICT: {'PASS' if ok else 'FAIL'}   a = {f1['a']:.4f} +- {f1['se_a']:.4f}"
          f" vs 0.346 +- 0.028 (z = {z_sy:+.2f}); sweep: {'PLATEAU' if plateau else 'NO PLATEAU'}")
    make_figure(d, fs, flin, prim, ic)
    print(f"Figure -> {FIG3}")


def make_figure(d, fs, flin, prim, ic) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(22.0, 5.4))
    x = np.sqrt(d["n_l"]) / (4 * np.pi)
    xx = np.linspace(x.min() * 0.95, x.max() * 1.05, 50)
    ax = axes[0]
    ax.plot(x[prim], d["S_sy"][prim, ic], "o", ms=3, color="tab:blue", alpha=0.5,
            label="c = 1 (fit range)")
    ax.plot(x[~prim], d["S_sy"][~prim, ic], "o", ms=3, color="0.6", alpha=0.5,
            label="c = 1 (below SY's x range)")
    f1 = fs[ic]
    ax.plot(xx, f1["a"] * np.log(xx) + f1["b"], color="tab:blue",
            label=f"fit a = {f1['a']:.3f} +- {f1['se_a']:.3f}")
    ax.plot(xx, SY_A * np.log(xx) + SY_B, "k--", lw=1.0, label="SY: 0.346 ln x + 1.883")
    ax.set_xscale("log")
    ax.set_xlabel("x = sqrt(N_l)/4pi")
    ax.set_ylabel("S")
    ax.set_title("A. PRIMARY: double truncation sqrt(N)/4pi, l/L = 1/2", fontsize=9.5)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25, which="both")

    ax = axes[1]
    a = np.array([f["a"] for f in fs])
    se = np.array([f["se_a"] for f in fs])
    ax.errorbar(SWEEP, a, yerr=se, fmt="o-", capsize=3, color="tab:blue", label="a(c)")
    ax.axhspan(SY_A - SY_A_ERR, SY_A + SY_A_ERR, color="0.85", label="SY 0.346 +- 0.028")
    ax.axhline(CFT_A, color="k", ls=":", label="1/3")
    ax.axhline(flin["a"], color="tab:green", ls="--", label=f"linear: {flin['a']:.3f}")
    ax.set_xscale("log")
    ax.set_xlabel("prefactor c (both cuts c sqrt(N)/4pi)")
    ax.set_ylabel("a")
    ax.set_title("B. PREFACTOR SWEEP", fontsize=9.5)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25, which="both")

    ax = axes[2]
    ax.plot(x, d["S_lin"], "^", ms=3, color="tab:green", alpha=0.5, label="linear truncation")
    ax.plot(xx, flin["a"] * np.log(xx) + flin["b"], color="tab:green")
    ax.set_xscale("log")
    ax.set_xlabel("x = sqrt(N_l)/4pi")
    ax.set_ylabel("S")
    ax.set_title("C. SECONDARY: SXY linear truncation (delta = 0.1)", fontsize=9.5)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25, which="both")

    ax = axes[3]
    m = np.isfinite(d["S_ctrl"])
    ax.plot(d["n_l"][m], d["S_ctrl"][m], "o", ms=4, color="tab:red", label="untruncated")
    nn = np.linspace(0, d["n_l"][m].max() * 1.05, 20)
    ax.plot(nn, SY_VOL_A * nn + SY_VOL_B, "k--", lw=1.0, label="SY: 0.32 N_l - 6.64")
    ax.set_xlabel("N_l")
    ax.set_ylabel("S")
    ax.set_title("D. CONTROL: no truncation (volume law expected)", fontsize=9.5)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    fig.suptitle(f"Phase 3a GATE 3 -- SSEE, nested 1+1 D diamonds (seeds {SEED_BASE3}+100000 i + k;"
                 f" R = {N_REAL3})", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    FIG3.parent.mkdir(exist_ok=True)
    fig.savefig(FIG3, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main(remeasure="--remeasure" in sys.argv[1:])
