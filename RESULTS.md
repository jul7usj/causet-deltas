# RESULTS log

Append-only. Never overwrite past entries (Scientific Integrity Rule 5).
Each entry records: date, commit, seed(s), parameters, finding, and one honest
line of interpretation.

---

## 2026-07-14 — Phase 1 acceptance: Brightwell–Gregory geodesic law

- **Commit:** `a19024f` (Phase 1: sprinkling, causal order, antichains, BG validation)
- **Experiment:** `experiments/exp00_phase1_validation.py`
- **Figure:** `figures/exp00_brightwell_gregory.png` (regenerable; not committed — seed-derived)
- **Seeds:** panel A = 20260714, panel B = 70418202
- **Realisations:** 40 independent sprinklings per data point
- **Tests:** `python -m pytest` → **22 passed** (sprinkle stats, causal transitivity,
  link = transitive reduction, longest chain = LIS, antichain maximality, BG law).

### Parameters
- 1+1 D causal diamond, endpoints p=(0,0), q=(τ,0); diamond volume V=τ²/2.
- Panel A: τ=2, densities ρ ∈ {125, 250, 500, 1000, 2000} → ⟨N⟩ ≈ 250…3994.
- Panel B: ρ=1000, τ ∈ {1.0, 1.5, 2.0, 2.5, 3.0} → ⟨N⟩ ≈ 499…4493.
- Endpoints included so the longest chain runs corner-to-corner of the diamond.

### Finding (with error bars, standard error over realisations)

Panel A — universal constant `⟨L⟩/√⟨N⟩ → 2` (Vershik–Kerov / Logan–Shepp; 1+1 D
Brightwell–Gregory, PRL 66, 260 (1991)):

| ρ    | ⟨N⟩    | ⟨L⟩    | SE(L) | ⟨L⟩/√⟨N⟩ |
|------|--------|--------|-------|----------|
| 125  | 250.6  | 29.38  | 0.32  | 1.8555   |
| 250  | 500.1  | 42.45  | 0.41  | 1.8982   |
| 500  | 999.2  | 60.00  | 0.47  | 1.8981   |
| 1000 | 1996.6 | 85.83  | 0.59  | 1.9207   |
| 2000 | 3994.4 | 121.85 | 0.59  | 1.9280   |

The ratio rises monotonically from below toward 2 as N grows, tracking the
finite-size correction `2 − c N^{−1/3}` (Tracy–Widom; fitted guide c ≈ 0.98).

Panel B — proportionality to proper time at fixed ρ:

| τ    | ⟨N⟩    | ⟨L⟩    | SE(L) |
|------|--------|--------|-------|
| 1.00 | 499.0  | 42.12  | 0.35  |
| 1.50 | 1122.2 | 64.35  | 0.50  |
| 2.00 | 1996.1 | 85.30  | 0.46  |
| 2.50 | 3119.7 | 107.12 | 0.64  |
| 3.00 | 4493.3 | 128.22 | 0.55  |

Through-origin fit: slope ⟨L⟩/τ = **42.74** vs prediction √(2ρ) = **44.72**
(ratio 0.956).

### Interpretation (honest, one line)

The geometric pipeline is validated: the longest chain reproduces the known 1+1 D
timelike-geodesic law — `⟨L⟩/√N` approaching 2 from below and `L` linear in proper
time — with the residual few-percent shortfall fully accounted for by the expected
`N^{−1/3}` discreteness correction, not a bug. Phase 1 accepted; the sprinkling,
causal order, and antichain machinery are trustworthy foundations for later phases.

---

## 2026-07-21 — Phase 2a acceptance: Boguñá–Krioukov causal-overlap distance (1+1 D)

- **Commit:** `c2e627a` (Phase 2a: Boguñá–Krioukov causal-overlap distance (1+1D))
- **Branch:** `phase2-baselines` (built on tag `phase1-accepted`)
- **Module:** `src/causet/causal_overlap.py`; **experiment:** `experiments/exp01_bk_1p1d.py`
- **Figure:** `figures/exp01_bk_convergence.png` (regenerable; not committed — seed-derived)
- **Source paper:** Boguñá–Krioukov, arXiv:2401.17376. Equations implemented:
  overlap eq. 16/28, exact d=1 distance eqs. 23–24, intrinsic Filter-2 eq. 34,
  depth estimator eq. 38 (verified against the paper's HTML, not memory).
- **Seeds:** convergence experiment `20260721 + k`, k=0..39; diagnostics `3300 + k`.
- **Realisations:** 40 independent sprinklings per density (diagnostics: 30).
- **Tests:** `python -m pytest` → **34 passed** (22 Phase 1 + 12 new causal-overlap:
  A/B/C partition, O∈[0,1], O=1 for timelike pairs, eq.24 roundtrip, and the
  sprinkling convergence `⟨O_C⟩ → O_pred`).

### Parameters (all named/documented, Rule 4)
- 1+1 D. Fixed spacelike target pair `a=(3, +0.5)`, `b=(3, −0.5)`; true proper
  separation `s = 1.0`. On-axis common event `c0=(0,0)`; continuum depth
  `τ_c = √(3² − 0.5²) = 2.9580`. Bounding diamond proper time `2·T_AB = 6`.
- `α_1 = 1/√2` (eq. 38, exact d=1) = `√2/m_2` with Phase-1 `m_2 = 2` — documented
  in the module so the two normalisations of the same Brightwell–Gregory fact are
  never conflated. Filter-2 prefactor `κ = 0.5` (paper's value, adjustable).
- Filter 1 (eq. 32) implemented but **not used** for selection: its exponent
  `β_d` was not pinned by the accessible source text; the paper flags Filter 1 as
  a speed shortcut and Filter 2 as the intrinsic claim, so Filter 2 selects c.
- Speed note: exp01 injects the O(N log N) 1+1 D LIS chain counter in place of the
  O(N²) matrix DP for the eq.-38 chain counts. Verified numerically identical
  (est 0.881511, same n_c) at ρ=100; purely a performance choice, no physics.

### Finding (error bars = standard error over 40 realisations)

**(1) Convergence to the exact eq.-24 closed form — the primary 2a gate: PASSES.**
Measured mean overlap from the symmetric vantage `c0` converges monotonically to
the value eq. 24 demands for `(s, τ_c)`, `O_pred = 0.7143`:

| ρ | ⟨N_int⟩ | ⟨O@c0⟩ | ⟨est d⟩ | SE(d) | rel.err | median #c |
|-----|--------|--------|---------|-------|---------|-----------|
| 25  | 452    | 0.7282 | 0.9704  | 0.033 | 0.030   | 11 |
| 50  | 902    | 0.7355 | 0.9475  | 0.021 | 0.053   | 17 |
| 100 | 1805   | 0.7236 | 0.9425  | 0.015 | 0.058   | 22 |
| 200 | 3604   | 0.7204 | 0.9299  | 0.010 | 0.070   | 34 |
| 400 | 7206   | 0.7190 | 0.9388  | 0.008 | 0.061   | 50 |
| 800 | 14409  | 0.7178 | 0.9493  | 0.005 | 0.051   | 72 |
| 1600| 28812  | 0.7168 | 0.9618  | 0.004 | 0.038   | 102 |

`⟨O@c0⟩` is monotone toward `O_pred` for ρ ≥ 50 (0.7355 → 0.7168, target 0.7143).
The unit test confirms the same at fixed `c0` (0.732 → 0.722 → 0.718).

**(2) Full intrinsic pipeline (Filter-2-selected c, eq.-38 estimated depth): a
stable ~4–6 % LOW bias that shrinks with density in the well-sampled regime.**
The distance estimate is NOT globally monotone: the two lowest densities (ρ=25,50)
are noise-limited — only 11–17 admissible c, SE up to 0.033 — and sit artificially
close to `s`. Once noise is sub-dominant (ρ ≥ 200) the estimate improves
monotonically, 0.9299 → 0.9618, with relative error 0.070 → 0.038.
Log-log convergence rate: **all ρ → ρ^(+0.039)** (meaningless, noise-contaminated
at low ρ); **ρ ≥ 200 → ρ^(−0.290)**, close to the expected `N^(−1/3)` ⇒ ρ^(−0.333)
Brightwell–Gregory discreteness correction.

**(3) Bias attribution (diagnostic, fixed `c0`, 30 realisations).** The residual
shortfall is entirely the eq.-38 depth estimator, not the overlap formula:

| ρ | ⟨O@c0⟩ | ⟨τ̂_c⟩ | τ̂_c/τ_true | d(true τ_c) | d(τ̂_c) |
|-----|--------|--------|------------|-------------|--------|
| 200 | 0.7233 | 2.8383 | 0.9595     | 0.9630      | 0.9239 |
| 800 | 0.7166 | 2.8483 | 0.9629     | 0.9905      | 0.9537 |

Given the *true* depth, eq. 24 recovers `s` almost exactly (0.963 → **0.9905** at
ρ=800). The eq.-38 estimate `τ_c` is ~4 % too short (ratio 0.960 → 0.963, rising
with ρ), and since `d ∝ τ_c` this maps linearly onto the ~4 % low distance. The
depth deficit is the **same** `L/√N < 2` finite-size chain shortfall validated in
Phase 1 (entering here through the chain-count `n_C`), vanishing as `N^(−1/3)`.

### Interpretation (honest, one line)

**Phase 2a gate PASSES**: the discrete causal overlap converges to the exact d=1
closed form (eq. 24) with error bars and monotone improvement in density; the
overlap + distance machinery is faithful (recovers `s` to 1 % given the true
depth). The only systematic — a ~4 % low bias in the fully-intrinsic estimate — is
not an implementation error but the eq.-38 depth estimator inheriting Phase 1's
known `N^(−1/3)` Brightwell–Gregory chain deficit, and it shrinks with density as
that correction demands. Recorded now because it will set the accuracy floor when
this estimator is used as a Δs benchmark in Phase 4.

---

## 2026-08-10 — Phase 2b Part 1 acceptance: 2+1 D (M³) infrastructure and the Rideout–Wallden m₃ gate

- **Branch:** `phase2b-2d1` (built on tags `phase1-accepted`, `phase2a-accepted`)
- **Modules:** `src/causet/sprinkle3d.py`, `src/causet/order3d.py`
- **Experiment:** `experiments/exp02_m3_validation.py`
- **Figure:** `figures/exp02_m3_validation.png`
- **Raw data:** `data/exp02_measurements.npz` (per-realisation chain lengths; committed,
  41 kB, so the analysis is re-runnable without repeating the 18-min measurement)
- **Source paper:** Rideout & Wallden, arXiv:0810.1768. Verified against the paper's
  HTML, not memory: interval volume **eq. (2), Sec. II.1**
  `V_xy = η(d) l_xy^d`, `η(d) = 2 V^s_{d−1}/(2^d d)` ⇒ `η(3) = π/12`; asymptotic
  chain law **eq. (1), Sec. II.1** `L(ρV)^(−1/d) → m_d`; the definition of `L`
  **Sec. II.1**; and the fit `f(N) = m₃ + a e^{b log₂N}` with
  `m₃ = 2.296 ± 0.012, a = −1.087 ± 0.014, b = −0.1201 ± 0.0053` over `N ≤ 2^18`
  (**Fig. 4 / Sec. III.1**; footnote 11 records an earlier `m₃ ≈ 2.278`).
- **Seeds:** d=3 `20260810 + 1000·log₂N + k`; d=2 controls `8102026 (+7, +500003) + 1000·log₂N + k`
- **Tests:** `python -m pytest` → **70 passed** (34 Phase 1/2a unchanged + 36 new).

### Parameters
- M³ causal diamond `I(p,q)`, `p=(0,0,0)`, `q=(τ,0,0)`, `τ=1`; `V = πτ³/12`. The
  diamond (not a box) is required: `m_d` of eq. (1) is defined by the longest chain
  *in an Alexandrov interval*. `sprinkle_box_3d` is also provided — its volume needs
  no derivation, so it anchors the Poisson-count tests independently, and Parts 2–3
  need a growable region holding fixed target events.
- Ladder `ρV = 2^10…2^17` with **800, 600, 400, 200, 100, 40, 14, 4** realisations
  (counts fall as runtime grows ∝ N²; all reported). Total 18.1 min, one core.
- Exact bicone sampler (no rejection, which would discard 74 % of proposals);
  uniformity verified by closed-form marginal CDFs (KS) plus a sub-box Poisson count.

### Finding 1 — GATE 1 (agreement with Rideout–Wallden's own curve): **PASSES**

`m₃^eff = L(ρV)^(−1/3)`, link convention, vs their published `f(N)`:

| log₂N | ρV | ⟨N⟩ | #real | ⟨L⟩ | SE(L) | ⟨m₃^eff⟩ | SE | RW `f(N)` | ±(their) | r/SE | r/comb |
|-----|-------|--------|-----|--------|------|--------|--------|--------|--------|-------|-------|
| 10  | 1024  | 1025   | 800 | 20.04  | 0.04 | 1.9885 | 0.0043 | 1.9689 | 0.0215 | +4.59 | +0.89 |
| 11  | 2048  | 2049   | 600 | 25.58  | 0.05 | 2.0144 | 0.0039 | 2.0059 | 0.0211 | +2.15 | +0.40 |
| 12  | 4096  | 4100   | 400 | 32.68  | 0.06 | 2.0423 | 0.0040 | 2.0388 | 0.0206 | +0.89 | +0.17 |
| 13  | 8192  | 8180   | 200 | 41.76  | 0.10 | 2.0713 | 0.0048 | 2.0679 | 0.0200 | +0.72 | +0.17 |
| 14  | 16384 | 16374  | 100 | 53.24  | 0.15 | 2.0962 | 0.0058 | 2.0937 | 0.0194 | +0.43 | +0.12 |
| 15  | 32768 | 32807  | 40  | 68.03  | 0.26 | 2.1258 | 0.0080 | 2.1166 | 0.0188 | +1.14 | +0.45 |
| 16  | 65536 | 65505  | 14  | 85.79  | 0.48 | 2.1278 | 0.0120 | 2.1369 | 0.0182 | −0.76 | −0.42 |
| 17  |131072 |131090  | 4   |109.25  | 1.03 | 2.1507 | 0.0203 | 2.1549 | 0.0176 | −0.21 | −0.16 |

Max absolute deviation **0.0195** over a factor-128 range in N. Combined-error
χ²/N = **0.18**, max residual **0.89 σ**.

**The measured values are ≈1.99–2.15, NOT ≈2.296, and that is the correct result.**
Their own fitted curve predicts exactly this: the finite-size correction
`a N^(b/ln2)` is 0.13–0.33 over this range, an order of magnitude larger than the
±0.012 on the asymptote. Measuring 2.29 at these N would have indicated a bug.

Reported both ways, because the choice changes the verdict: with **our** errors only,
χ²/N = 3.64 (max +4.59 σ at log₂N=10) — a nominal FAIL. That test is wrong: it treats
their fitted curve as exact. Propagating **their** quoted `a, b, m₃` errors gives the
curve an uncertainty of **0.0176–0.0215**, larger than our statistical errors at every
point, and the deviation vanishes. Caveat recorded in the code: the paper gives
parameter errors but no covariance, and such parameters are strongly correlated, so
quadrature over-estimates their band — the combined test is therefore conservative
(lenient). The truthful statement is that our curve and theirs agree to ≤0.02
absolute, which is the resolution their published fit can support.

### Finding 2 — GATE 2 (extrapolated asymptote): **passes, but is a low-power test**

Primary: `m₃ = 2.3427 ± 0.0975 (stat) ± 0.0180 (syst) = 2.3427 ± 0.0991`,
which is **0.35 σ** outside `[2.278, 2.308]` ⇒ consistent.

**This is consistency, not confirmation.** The error bar is 8× wider than
Rideout–Wallden's ±0.012; at ±0.099 the 2σ criterion would have accepted any central
value in ≈[2.08, 2.51], so this gate could not have detected a several-percent error.
Spread across fit variants, which exceeds every individual quoted error:

| fit variant | m₃ | note |
|---|---|---|
| interior conv., `c` free (**primary**) | 2.3427 ± 0.0975 | only variant whose error reflects extrapolation freedom |
| link conv., `c` free | 2.4722 ± 0.3113 | contaminated by the convention offset (below) |
| interior conv., `c` fixed to RW's `b/ln2` | 2.3282 ± 0.0104 | error understated — `c` assumed |
| link conv., `a` and `c` both pinned to RW | 2.3036 ± 0.0019 | error meaningless — 1 free parameter |
| **Rideout–Wallden published** | **2.2960 ± 0.0120** | |

**All four of our estimates lie at or above 2.296** (2.303, 2.328, 2.343, 2.472); none
below.

**That one-sidedness is the effective-fit overshoot, not physics.** Unmodelled
higher-order corrections bias the *fitted* exponent shallow relative to the true
asymptotic one (here −0.166 vs RW's −0.173; in the d=2 controls −0.2616 and −0.3216
against the exact −1/3), and since `m_∞ = m(N_max) + |a|·N_max^c`, a shallower `c`
mechanically inflates the extrapolated constant. The four variants are monotone in
exactly that way — `c = −0.091 → 2.472`, `c = −0.166 → 2.343`, `c = −0.173 → 2.328`,
`c` and `a` both pinned `→ 2.304` — so the ordering is a property of how much
extrapolation freedom each fit retains, not of the geometry. Its magnitude is measured
independently rather than argued: the identical procedure over the identical N window
overshoots the *exactly known* `m₂ = 2` by **+0.018** (Control B), falling to
**+0.0019** once the window improves (Control C), and the d=2 exponent bias shrinks in
step (0.072 → 0.012 shallow). Subtracting Control B's overshoot from our primary
estimate gives **2.325 ± 0.098**, statistically indistinguishable from 2.296. So the
clustering above the published value is expected from the fitting procedure and
carries no evidence for a larger m₃; logged as an observation, not a claim — ±0.099
could not support one either way.

The method-to-method spread (~0.04 among the defensible variants) is the real
uncertainty on any such extrapolation, including the published one.

### Finding 3 — the link convention contributes an exactly known power of N

**Sec. II.1 defines `L` as the number of _links_ in the longest chain "between (and
including) x and y"** — not elements. With both interval endpoints included, every
interior chain extends by both, so with `I` = longest chain among the `N` interior
Poisson elements: elements `= I+2`, `L = I+1`, hence

    m_link = L(ρV)^(−1/d) = I(ρV)^(−1/d) + (ρV)^(−1/d)

— a `+N^(−1/d)` term with coefficient **exactly 1**, not a fitted parameter. It is
+0.099 at ρV=2^10 in d=3, i.e. 23× our standard error there. Consequences measured:

- Removing it (fitting `m_inter = I(ρV)^(−1/3)`) improves the free fit's statistical
  error **3-fold, ±0.3113 → ±0.0975**, and moves `c` from −0.091 to −0.166 (much
  nearer RW's −0.173). The uncorrected quantity's flatness at low N was an artefact.
- Verified as an arithmetic identity: `max|m_link − m_inter − (ρV)^(−1/3)| = 5×10⁻¹⁶`.
- A planned "cross-check" comparing the two conventions' extrapolations was **dropped
  as vacuous**: they differ by a deterministic known amount, and with `c` fixed the
  `N^(−1/3)` offset cannot be absorbed, so the two fits are biased apart *by
  construction*. It was replaced by the d=2 controls below.

### Finding 4 — controls: the procedure calibrated where the answer is known exactly

Both gates compare us with the paper being validated, so neither can catch an error in
a *shared* procedure. Controls use `m₂ = 2` exactly (Brightwell–Gregory; Phase 1).

- **Control A — chain-counter equivalence at scale: PASSES.** A 1+1 D sprinkling
  embedded in M³ with `y ≡ 0` makes `dt²−dx²−dy²` exactly the 1+1 D relation, so the
  new streamed 2+1 D DP must return the *same integer* as Phase 1's independent
  O(N log N) patience-sorting counter. **19/19 realisations matched exactly** at
  N = 1006, 3995, 16402 — validating the new counter against trusted Phase-1 code
  well beyond where a dense causal matrix would fit.
- **Control B — matched window (same 8 N-values, same realisation counts as d=3):**
  recovers `m₂ = 2.0180 ± 0.0173`, bias **+0.018 (+1.04 σ)**. Quoted as the systematic
  on m₃ rather than assumed zero.
- **Control C — 9 points to N=2^18, higher statistics:** `m₂ = 2.0019 ± 0.0053`, bias
  **+0.0019 (+0.36 σ)**. The bias shrinks ~10× as the window improves ⇒ a finite-window
  artefact, not an implementation error.

**Caveat that limits every such extrapolation, ours and theirs.** Even at Control C's
statistics the two-term fit does not recover the *exact* d=2 coefficients: fitted
`a = −1.530` against the Tracy–Widom value **−1.7711**, and `c = −0.3216` against the
exact **−1/3**. Both drift toward truth as statistics improve (`a`: −1.079 → −1.530;
`c`: −0.262 → −0.322), so `m₃ + a N^c` is an *effective* description over a finite
window, not the true asymptotic expansion — which is why the fixed-parameter variants
above have deceptively small errors.

### Interpretation (honest)

**Part 1 gate PASSES, and the 2+1 D pipeline is trustworthy — but the strength of the
evidence is unevenly distributed.** The real validation is Finding 1 plus Control A:
our `m₃^eff(N)` lies on Rideout–Wallden's published curve to ≤0.02 absolute across a
factor-128 range in N with zero fitted parameters, and the new O(N)-memory chain
counter reproduces Phase 1's trusted counter integer-for-integer. Finding 2 is only
corroborative: our N range (≤2^17, capped by the irreducible O(N²) cost of longest
chains in 2+1 D — Phase 1's O(N log N) shortcut relies on the 1+1 D coordinate-order
equivalence and does not generalise) gives too little lever arm to pin the asymptote
better than ±0.10, so it can confirm consistency with 2.296 but could not have
detected a several-percent error in it. The single most consequential thing learned is
Finding 3: `L` counts links, and mishandling that one-element convention shifts
`m₃^eff` by `(ρV)^(−1/3)` ≈ 0.02–0.10 at reachable N — larger than the published
uncertainty on m₃ itself, and it will propagate directly into every 2-link
proper-time calibration in Part 2.

### Practical note for Part 2
Phase 1's `is_transitive`/`link_matrix` multiply **int64** matrices, which NumPy does
not route through BLAS, making them O(N³) in practice (~32 s at N≈1600). They are
correct and reused unchanged (re-exported, with a regression test asserting function
identity and identical 1+1 D results), but Part 2's `future_2links` must not be built
on them at scale.

---

## 2026-08-17 — Phase 2b Part 2 GATE A: Rideout–Wallden 2-link spacelike distance (2+1 D)

- **Branch:** `phase2b-2d1` (Part 1 accepted at `41f5607`)
- **Module:** `src/causet/rideout_wallden.py`
- **Experiment:** `experiments/exp03_2link_stability.py`
- **Figure:** `figures/exp03_2link_stability.png` (regenerable from the cache)
- **Data:** `data/exp03_measurements.npz` (raw per-realisation values)
- **Seeds:** `SEED_BASE = 20260817`; realisation seed = `SEED_BASE + 10000·i + k`
- **Tests:** `python -m pytest` → **96 passed** (26 new; the 70 Part-1 tests unchanged)

### Parameters
- 2+1 D box `[0,T]×[−Lx/2,Lx/2]×[−Ly/2,Ly/2]`, extents `λ·(4.0, 2.5, 5.0)·D`,
  `λ ∈ {0.55, 0.70, 0.85, 1.00, 1.15, 1.30, 1.45}` — shape held **fixed**, only `λ` varies.
- Target pair fixed at `x=(T/2, −D/2, 0)`, `y=(T/2, +D/2, 0)` with `D = 1` (true
  separation), appended to the sprinkling; density fixed at `ρ = 60`.
- **40 realisations per region size**, 7 sizes = 280 causets. Volume 8.32 → 152.43
  (factor **18.3**), `⟨N⟩` 495 → 9148.
- Calibration: eqs. (1)–(2), `l = L/(m₃(ρη(3))^{1/3})`, `m₃ = 2.296`, `η(3) = π/12`.
  **Link convention throughout** (Part-1 Constraint 1), isolated in
  `order3d.chain_links_from_elements`.
- Region shape is elongated in `t` and `y`, narrow in `x`, because the light cones of
  this pair meet on `X = 0, t = ±√(D²/4 + Y²)`. That shape buys the boost freedom the
  construction needs per element sprinkled; holding it fixed means it cannot
  manufacture or hide a trend.

### Finding 1 — GATE A **PASSES**: the 2-link distance does not drift

| λ | V | ⟨N⟩ | 2-links/causet | total | empty | **d_2link** | SE | n≠0 | d_naive | SE | ⟨L⟩ | ⟨\|[p,f]\|⟩ | pairs |
|------|--------|------|------|-----|----|------------|--------|----|--------|--------|------|------|------|
| 0.55 | 8.32   | 495  | 0.62 | 25  | 23 | 0.8712 | 0.0339 | 17 | 0.8174 | 0.0178 | 5.01 | 36.8 | 148  |
| 0.70 | 17.15  | 1039 | 1.20 | 48  | 11 | 0.8630 | 0.0229 | 29 | 0.8391 | 0.0196 | 4.96 | 36.7 | 554  |
| 0.85 | 30.71  | 1848 | 1.07 | 43  | 15 | 0.9287 | 0.0231 | 25 | 0.8174 | 0.0167 | 5.34 | 42.7 | 1099 |
| 1.00 | 50.00  | 3010 | 1.05 | 42  | 16 | 0.8635 | 0.0283 | 24 | 0.7913 | 0.0186 | 4.97 | 35.4 | 1671 |
| 1.15 | 76.04  | 4554 | 1.18 | 47  | 13 | 0.8754 | 0.0210 | 27 | 0.7913 | 0.0164 | 5.03 | 39.0 | 2276 |
| 1.30 | 109.85 | 6601 | 1.52 | 61  | 11 | 0.8717 | 0.0201 | 29 | 0.8043 | 0.0149 | 5.01 | 36.3 | 3601 |
| 1.45 | 152.43 | 9148 | 1.50 | 60  | 12 | 0.8731 | 0.0235 | 28 | 0.8043 | 0.0149 | 5.02 | 38.0 | 4226 |

Errors are standard errors **across realisations**, not across pooled 2-links: the
2-links inside one causet share a sprinkling and a target pair, so pooling them and
dividing by `√total` would understate the error. The primary statistic is the
per-realisation Step-5 mean; `n≠0` is how many causets contributed.

Weighted straight-line fit against `log₂V`:

    slope = −0.00218 ± 0.00707 per log₂V     (0.31 σ from zero),  χ²/dof = 5.59/5

i.e. a total change of **−0.009 ± 0.030** across a factor 18.3 in volume, on a value
of ≈0.87. Both acceptance criteria, fixed before the verdict:

- **A1 stability** — `|slope| ≤ 2σ`: **pass** (0.31 σ).
- **A2 resolution** — `σ_slope ≤ 0.0238`, so that a degradation of 20% of `D` across
  the ladder (slope 0.0476) would have been a ≥2σ detection: **pass**, and in fact
  such a drift would have shown at **6.7 σ**. Without A2 the gate would be passable
  by simply having wide error bars (Part-2 Constraint 3).

### Finding 2 — the naive control did **not** visibly fail (reported as prominently as the pass)

Rideout–Wallden motivate the whole 2-link construction by the claim that the naive
double minimum of their Section II.B degrades for spacetime dimension `d ≥ 3`, because
boosts supply an unbounded family of minimising pairs and the minimum over ever more
samples drifts downward. Measured on the **same** sprinklings, same target pair, same
calibration:

    naive slope = −0.00643 ± 0.00457 per log₂V   (1.41 σ),  total −0.027 ± 0.019

The trend is in the predicted direction and about **3× the magnitude** of the 2-link
slope — but at 1.4 σ it is **not resolved**. Sharper still: the number of `(p,f)`
candidate pairs the naive minimum ranges over grew from 148 to 4226, a factor **29**,
and 29× more chances to fluctuate low bought a decrease of at most 3%.

So this experiment does **not** reproduce a visible failure of the naive estimator at
reachable region sizes. Two readings are consistent with the data and it cannot
separate them: the drift may be real but logarithmically slow (which would match
Finding 3's mechanism), or the factor-18 range may simply be too short. Either way,
**Gate A's pass rests on the 2-link estimator's own flatness plus criterion A2, not on
a demonstrated contrast with the naive one.** The contrast remains an untested premise
of the source paper as far as this work is concerned.

(`naive_distance` uses the exact maximal-past × minimal-future reduction, which is
provably identical to minimising over the full common past and future — not an
approximation, and asserted against a brute-force scan in the tests.)

### Finding 3 — the 2-link sample size grows only **logarithmically** with the region

Rideout–Wallden state that infinite Minkowski contains infinitely many `n`-links when
`n < d` (here 2 < 3), but say nothing about the rate — and the rate is what decides
whether the estimator is usable. Fitting two falsifiable models to the same points:

    yield = 0.306 ± 0.019 · ln V        χ²/dof = 5.86/6 = 0.98    <- describes the data
    yield = 0.01457 · V                 χ²/dof = 73.3/6 = 12.2    <- decisively rejected

Consequences, all measured:

- **326 2-links over 280 causets = 1.16 per causet.** The Step-5 "average over all
  `f_i`" is, in practice, an average over **one** sample.
- **101/280 causets (36%) yielded no 2-link at all**, and the estimator returns nothing
  on those. At the smallest region it was 23/40 (58%).
- A factor **18** in volume bought a factor **2.4** in 2-links. Because the growth is
  logarithmic, doubling the yield again requires **squaring** the volume
  (152 → 23 235, i.e. `N ~ 1.4×10⁶` elements) — far beyond the O(N²) dense causal
  matrix used here, and beyond any obvious sparse rewrite.

This is consistent with the paper's existence claim and with the flatness in Finding 1,
but it is the binding practical limit on the estimator as a Δs benchmark in Phase 4.

### Finding 4 — a −12% scale offset, only partly explained by the finite-size calibration

Grand mean **0.878** against the true `D = 1` (ratio 0.878). The direction is expected:
Step 2 deliberately picks the *smallest* interval available (mean size **37.9
elements**), while eq. (1) is a `ρV → ∞` statement. Evaluating Rideout–Wallden's own
Fig.-4 curve at 37.9 gives `m₃^eff = 1.717`, which on its own predicts lengths low by a
factor **0.748**.

**The observed 0.878 is well above that 0.748, so the finite-size calibration
over-predicts the shortfall** — something compensates, and this work has not pinned it
down. The plausible candidate, flagged as unverified: the minimising interval is not a
typical Alexandrov interval of the sprinkling but one conditioned on `p` being maximal
in the common past and `f` being a 2-link, which selects intervals *emptier* than
typical for their proper time; the appropriate `ρV` at which to read `m₃^eff` would
then be larger than the realised count of 37.9, pushing the predicted ratio up. Logged
as an open discrepancy, not a claim.

The offset does not affect the gate: at fixed density it is a constant multiplier that
shifts every point equally and so cannot create or mask a slope. It **will** matter in
Part B, where the two estimators carry different calibration conventions (`m₃` vs
`c_d`) — which is why that comparison is specified on rank ordering, not scale.

**Resolution floor.** One chain link is `0.174` in length units, **17% of `D`**, and
`⟨L⟩ ≈ 5.0` throughout. There is also a hard floor of **2 links**: every minimising
interval `[p,f]` contains both `x` and `y` (since `p ≺ x ≺ f`), so its longest chain has
at least 3 elements. The estimator is coarsely quantised (figure, panel C) and cannot
report a distance below `2/(m₃(ρη)^{1/3})`.

### Finding 5 — the 2-link condition is strictly stronger than minimality, and it is guarded

`future_2links` implements Definition 2b (`n = 2`) literally: `f` in the common future
with **both** intervals `[x,f]` and `[y,f]` empty. Every 2-link is minimal in
`fut(x)∩fut(y)` (a common-future element below `f` would sit inside `[x,f]`), so the
2-links are a subset of the minimal elements — and a **strict** one, because minimality
cannot see an intervening `z` with `x ≺ z ≺ f` that misses `fut(y)`. That is the exact
failure Rideout–Wallden warn about in Section V.A, where it lets the Step-2 minimising
pair have arbitrarily large proper time.

A 13-element hand-placed M³ causet pins this down (`tests/test_rideout_wallden.py`).
It contains two deliberate negative controls, `fbad_x` and `fbad_y`, each **minimal in
the common future** yet blocked from one target by an element lying in only that
target's future. The test asserts `minimal = {f0, f2, fbad_x, fbad_y}` while
`2-links = {f0, f2}` — an implementation using minimal elements fails on both. Every
causal relation the test depends on is asserted individually, and each set was verified
against the hand derivation before being written down.

An exact reduction, proved rather than assumed: if `p ≺ p′` are both in the common past,
any chain from `p′` extends by `p`, so `d(p,f) > d(p′,f)` and only **maximal**
common-past elements can minimise Step 2. This matters because `|past(x)∩past(y)|` grows
with the region while the minimising intervals do not. `exhaustive_past=True` disables
it and the tests assert the two agree on the hand causet and on random sprinklings.

### Interpretation (honest)

**Gate A passes, and passes on a criterion strong enough to have failed.** The
Rideout–Wallden 2-link distance holds to −0.009 ± 0.030 across a factor 18.3 in
sprinkling volume around a fixed pair at fixed density, with error bars tight enough
that a 20%-of-`D` degradation would have registered at 6.7 σ. That is the property the
construction exists to provide, and it is confirmed.

Two things temper it. First, the comparison that was supposed to make the result
*meaningful* — the naive estimator visibly degrading on the same data — **did not
materialise** (Finding 2); the pass therefore establishes that the 2-link distance is
stable, not that it is stable *where the naive one is not*. Second, and more
consequential for Phase 4: the estimator's own sample size grows like `ln V`
(Finding 3), so it delivers ~1 measurement per causet and returns nothing at all 36% of
the time. Combined with a 17%-of-`D` quantum and a −12% scale offset that the
finite-size calibration only partly accounts for (Finding 4), the 2-link distance is
**stable but low-yield and low-resolution** — a sound benchmark for rank ordering and
for trend tests, and a poor one for any comparison demanding per-causet accuracy.

### Practical note for Part B

- Cost is dominated by the naive control (6.5 s/realisation at `N ≈ 9150`) and by
  `causal_matrix_3d`, whose dense `N×N` float temporaries make `N ≳ 12 000` infeasible
  in memory. The 2-link distance itself is cheap (0.11 s at that size).
- `chain_links_to_target` replaced a per-`(p,f)` interval DP and cut the control from
  21 s to 6.5 s; it is exact, and `chain_links_between` survives as the per-pair oracle
  the tests check it against.
- Head-to-head pairs must be chosen with the 17%-of-`D` quantum in mind: separations
  closer together than ~1 link cannot be rank-ordered by this estimator at `ρ = 60`,
  so the ≥20 pairs should span a wide range of true separations.
