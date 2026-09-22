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

This result is consistent with Rideout–Wallden's own reported experience rather than in tension with it: their demonstration of naive degradation required a deliberately anisotropic sprinkling region (growing in t and y at fixed x, targets at x=±4), because independent minimizing pairs concentrate near the light-cone intersection a hyperbola with a specific asymptotic direction, so isotropic volume growth adds volume mostly where such pairs do not live. They further estimated that reaching the fully degenerate result would require N ~ 512·e^512, and they too observed only a drift, never complete failure. Our isotropic factor-18 range therefore has no expectation of resolving the effect, and the unresolved contrast reflects the geometry of the test rather than a failure of the naive-degradation claim.

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

---

## 2026-08-17 — Phase 2b Part A DIAGNOSTIC: the −12% scale offset is fully accounted for

- **Branch:** `phase2b-2d1` (Part A accepted at `49abc25`)
- **Experiment:** `experiments/exp03b_offset_diagnostic.py`
- **Figure:** `figures/exp03b_offset_diagnostic.png`
- **Data:** `data/exp03b_diagnostic.npz`
- **Seeds:** selected pairs reuse exp03's stream `20260817 + 10000·i + k` (so the
  first 40 causets at each λ are literally the Gate-A causets); geometry-matched
  controls `71820260`; unconditioned baseline `30820262`
- **Scope:** diagnostic, **not a gate** — no pass/fail is issued. Its job is to close
  or explicitly leave open the discrepancy logged above under the scale-offset finding.
- **Tests:** `python -m pytest` → **98 passed** (2 new: the estimator now exposes the
  selected `(p, f_i)` pair, asserted to reproduce the reported minimum)

### Parameters
- Same geometry as Gate A: `ρ = 60`, `D = 1`, box `λ·(4.0, 2.5, 5.0)`.
- `λ ∈ {0.70, 1.00, 1.30}` with 400 / 400 / 150 causets → **1030 selected pairs**
  from 578 non-empty causets.
- **12 independent unconditioned sprinklings per causet** for the geometry-matched
  control; 4000 plain diamonds per point for the unconditioned `m₃^eff` baseline.
- Errors are standard errors **across causets** (pairs within a causet share a
  sprinkling and a target pair, so they are aggregated per causet first).

### The exact decomposition

The reported estimate factorises with no free parameters. Writing
`μ = ρ·η(3)·τ(p*,f)³` for the expected element count of the selected interval, using
the **exact embedding** proper time `τ` of the selected pair (embedding information,
used only to diagnose — never inside the estimator, the same separation Phase 2a made
between `τ_c` and `τ̂_c`):

    l_est/D  =  [ L / (m₃ μ^{1/3}) ]  ×  [ τ(p*,f)/D ]
                  calibration ratio      geometric excess (≥ 1)

| λ | pairs | causets | `l_est/D` | calibration | geometric `τ/D` | product |
|------|-----|-----|-----------------|-----------------|-----------------|--------|
| 0.70 | 361 | 222 | 0.8971 ± 0.0083 | 0.6734 ± 0.0059 | 1.3355 ± 0.0083 | 0.8994 |
| 1.00 | 480 | 259 | 0.8644 ± 0.0073 | 0.6596 ± 0.0052 | 1.3142 ± 0.0067 | 0.8668 |
| 1.30 | 189 |  97 | 0.8845 ± 0.0118 | 0.6627 ± 0.0080 | 1.3379 ± 0.0119 | 0.8866 |

The product reproduces the measured estimate at every region size, so these two
factors are a **complete** account — nothing else is left to explain.

**The second factor is the one the hypothesis did not consider.** The estimator does
not report the distance between `x` and `y`; it reports `d(p, f_i)`, the discrete
proper time of the *selected pair*. In the continuum these coincide: `min_p τ(p,f)`
over the common past equals `D` exactly, attained by the boost-matched partner. But
the sprinkling need not contain that partner, and Step 2 minimises the **chain**, not
the proper time. Hence `τ(p*,f) ≥ D` always — measured smallest value anywhere
**1.0493**, never below 1, as required — and here it runs **31% above `D`**.

### Hypothesis test — **CONFIRMED**, and it explains **0.8%** of the gap

Geometry-matched control: the identical `p, f` *coordinates* dropped into 12
independent unconditioned sprinklings of the same box at the same density. Same proper
time, same boost, same distance from the boundary; only the selection is removed. Both
samples carry the two deterministic target elements (which lie inside every selected
interval, since `p ≺ x ≺ f`), so the matched Poisson expectation is `μ + 2`.

| λ | `\|[p,f]\|` conditioned | control | ratio cond/ctrl | cond/(μ+2) | ctrl/(μ+2) |
|------|--------------|--------------|-----------------|-----------------|-----------------|
| 0.70 | 38.84 ± 0.81 | 40.56 ± 0.73 | 0.9593 ± 0.0098 | 0.9586 ± 0.0095 | 1.0012 ± 0.0029 |
| 1.00 | 35.72 ± 0.62 | 38.70 ± 0.59 | 0.9276 ± 0.0093 | 0.9267 ± 0.0088 | 1.0015 ± 0.0026 |
| 1.30 | 38.05 ± 1.22 | 40.94 ± 1.11 | 0.9263 ± 0.0154 | 0.9278 ± 0.0152 | 1.0033 ± 0.0043 |

Combined **0.9398 ± 0.0062**, i.e. **9.75 σ below 1**. The conditioned intervals really
are emptier — the hypothesis is **confirmed as a real effect**, and the mechanism is
concrete: the selection forces three sub-regions of `[p,f]` to be empty (`[x,f]` and
`[y,f]` by the 2-link condition, `fut(p) ∩ past(x) ∩ past(y)` by `p`'s maximality).

**But it does not explain the offset.** The original prediction read RW's curve at the
realised count 35.7; the hypothesis says read it at the true expected count 36.5. That
moves `m₃^eff` from 1.7110 to 1.7133 and the predicted ratio from 0.748 to **0.7490** —
closing **0.8% of the 0.130 gap**. The interval is emptier by ~6%, and `m₃^eff` varies
far too slowly with size for a 6% shift to matter. *Confirmed, and irrelevant.*

The λ = 0.70 ratio (0.959) sits above the other two (0.928, 0.926); with 2 dof that is
a mild tension, plausibly because the smallest region leaves least room for the forced-
empty sub-regions. Recorded, not resolved.

**Control validation.** `ctrl/(μ+2) = 1.0012 ± 0.0029` — the unconditioned control
reproduces the Poisson expectation exactly. That simultaneously confirms there is no
boundary clipping of these intervals and that `μ` computed from the exact embedding is
right, neither of which was assumed.

### What the conditioning actually changes: the chain, not the interval

| λ | `L` conditioned | `L` control | ratio | `m₃^eff` selected | `m₃^eff` control |
|------|-----------------|-----------------|-----------------|-----------------|-----------------|
| 0.70 | 5.1585 ± 0.0478 | 6.0337 ± 0.0432 | 0.8594 ± 0.0080 | 1.5462 ± 0.0135 | 1.8026 ± 0.0048 |
| 1.00 | 4.9702 ± 0.0423 | 5.9273 ± 0.0366 | 0.8439 ± 0.0072 | 1.5144 ± 0.0120 | 1.7992 ± 0.0046 |
| 1.30 | 5.0857 ± 0.0681 | 6.0816 ± 0.0635 | 0.8411 ± 0.0109 | 1.5215 ± 0.0183 | 1.8133 ± 0.0075 |

The chain is suppressed by **16%**, nearly three times harder than the cardinality's
6%, and for an obvious reason: **Step 2 minimises the chain over the common past**, so
the selected chain is the shortest available across those two events by construction.

This also reconciles the two framings, which superficially disagree. At matched
*geometry* the conditioned interval is **smaller** (0.94); at matched *chain length* —
the brief's intrinsic form, using no embedding at all — it is **larger**:

| L | n cond | `\|[p,f]\|` cond | n ctrl | `\|[p,f]\|` ctrl | ratio |
|---|-----|--------------|------|--------------|-----------------|
| 3 |  11 | 20.27 ± 1.29 |   36 | 18.50 ± 0.70 | 1.0958 ± 0.0810 |
| 4 | 217 | 28.27 ± 0.47 |  744 | 25.32 ± 0.20 | 1.1168 ± 0.0206 |
| 5 | 517 | 36.24 ± 0.41 | 3301 | 31.95 ± 0.13 | 1.1342 ± 0.0135 |
| 6 | 255 | 48.09 ± 0.83 | 4467 | 39.79 ± 0.14 | 1.2088 ± 0.0213 |
| 7 |  30 | 52.60 ± 3.06 | 2706 | 48.25 ± 0.22 | 1.0902 ± 0.0637 |

Both are the same fact: `L` falls 16% while cardinality falls only 6%, so at fixed `L`
the conditioned interval must be the bigger one.

### The 0.748 prediction was itself wrong

RW's Fig.-4 fit was made over `ρV = 2¹⁰…2¹⁸`. The selected intervals sit at
`ρV ≈ 36.6 = 2^5.2`, **4.8 octaves below its support** — so the original 0.748 rested
on an extrapolation with no warrant. Measured directly instead (4000 plain diamonds
per point, endpoints included, link convention, no conditioning):

| ρV | `m₃^eff` measured | RW curve extrapolated | difference |
|------|-----------------|--------|---------|
| 16.0 | 1.7592 ± 0.0052 | 1.6237 | +0.1356 |
| 24.0 | 1.7809 ± 0.0047 | 1.6693 | +0.1116 |
| 32.0 | 1.7900 ± 0.0043 | 1.6997 | +0.0903 |
| 40.0 | 1.8119 ± 0.0040 | 1.7224 | +0.0895 |
| 48.0 | 1.8238 ± 0.0039 | 1.7402 | +0.0836 |
| 64.0 | 1.8403 ± 0.0037 | 1.7672 | +0.0731 |

The extrapolation understates `m₃^eff` by 0.07–0.14, with the gap shrinking as `ρV`
rises toward the fitted window. This is **not** a disagreement with Rideout–Wallden —
it is exactly the caveat Part 1 recorded (Finding 4 of the 2026-08-10 entry): `m₃ + aN^c`
is an *effective* description over a finite window, not the true asymptotic expansion,
so it must not be read outside it. Part 1 flagged the risk; this measures the cost.

At `ρV = 36.6`: measured unconditioned `m₃^eff = 1.8027` (ratio **0.785**, not 0.748).
Independently, the geometry-matched control gives `1.7992 ± 0.0046` at the same size —
**two unrelated measurements of the unconditioned constant agreeing to 0.2%**.

### Final accounting — the discrepancy is closed

Every factor measured in this experiment, none fitted (λ = 1.00):

    finite-size calibration, unconditioned at ρV = 36.6    × 0.7851
    Step-2 minimisation suppressing the chain              × 0.8439
    geometric excess  τ(p*,f)/D                            × 1.3142
    ------------------------------------------------------------
    product                                                = 0.8708
    observed                                                 0.8644

A residual of 0.7% remains, comparable to the interpolation of the baseline curve and
to the slight non-Poisson enhancement from the two deterministic target elements inside
each interval. Not chased further; recorded.

### Interpretation (honest)

**The offset was never a calibration puzzle — it is the near-cancellation of two large,
opposite effects, and neither is the one that was hypothesised.** The estimator
under-reports by 34% (a 21% finite-size calibration shortfall compounded with a 16%
chain suppression from the Step-2 minimisation) and over-reports by 31% because the
pair it actually measures is 31% further apart in proper time than the targets are.
`0.66 × 1.31 = 0.87`. That two effects of ~30% each should cancel to 12% is a
coincidence of these parameters, not a property of the construction — and that is the
uncomfortable part, because **it means the −12% figure is not stable**. It is a
difference of large terms with different `ρ` and `D` dependence, so it should not be
carried forward as "the 2-link estimator reads ~12% low"; each application needs its
own accounting.

The logged hypothesis is **confirmed but immaterial**: conditioned intervals are
genuinely emptier (9.75 σ), and that fact explains under 1% of what it was proposed to
explain. Reported at the same prominence as the part that worked, because a confirmed
effect of negligible size is exactly the kind of result that gets quietly upgraded into
an explanation later.

Two corrections to the previous entry follow from this and should be read with it:
the original 0.748 was an unwarranted five-octave extrapolation (the right unconditioned
number is 0.785), and the speculative mechanism offered there — "the appropriate ρV
would then be larger than the realised count" — is real in direction but ~0.8% in size,
not the compensation it was floated as.

### Consequence for Part B

The geometric excess `τ(p*,f)/D = 1.31` is a property of the *estimator's target*, not
of its calibration: it does not shrink by fixing constants, only by the sprinkling
supplying a better-matched past partner. Since it enters as a multiplicative factor on
every pair, it should largely cancel in the **rank-ordering** test that Part B is
specified on — but it will not cancel in any absolute-scale comparison against
Boguñá–Krioukov, which reinforces why that comparison was specified on ordering.

---

## 2026-09-22 — Phase 2b Part B acceptance: Boguñá–Krioukov in 2+1 D and the head-to-head rank-ordering gate

- **Branch:** `phase2b-2d1` (built on Part A `49abc25` and its diagnostic `1398f53`)
- **Module:** `src/causet/causal_overlap_3d.py` (Part B1) — `causal_overlap.py` is **not modified**
- **Experiment:** `experiments/exp04_rw_vs_bk_2p1d.py` (Part B2)
- **Figure:** `figures/exp04_rw_vs_bk_2p1d.png` (6 panels)
- **Raw data:** `data/exp04_measurements.npz` (per-causet values for all 24×100 evaluations; 211 kB, committed, so every number below is re-derivable without repeating the 10.5-min measurement)
- **Source papers:** Boguñá & Krioukov, arXiv:2401.17376 (overlap eqs. 16/28, exact 1+1 D eqs. 23–24, **asymptotic eqs. 25–27**, depth estimator eq. 38, Filter 2 eq. 34); Rideout & Wallden, arXiv:0810.1768 (eqs. 1–2, Sec. V.A)
- **Seeds:** `20260922 + k`, `k = 0…99` — one Poisson background per seed, **re-used across all 24 separations**
- **Tests:** `python -m pytest` → **124 passed** (98 from Phase 1/2a/Part 1/Part A, unchanged + 26 new in `tests/test_causal_overlap_3d.py`)

### The constraint this phase was designed around

Part A's diagnostic (2026-08-17 entry) established that the 2-link estimator's −12 %
offset is a near-cancellation of a 0.7851 × 0.8439 calibration/suppression shortfall
against a 1.3142 geometric excess, with **different ρ and D dependence**. It is not a
stable estimator property and is **nowhere** subtracted, divided out, or fitted in this
work. Consequently the head-to-head is specified on **rank ordering**, which is
invariant under any monotone rescaling, and **no absolute-scale comparison between the
two estimators is made or is licensed by anything below.**

---

## Part B1 — the 2+1 D causal overlap

### The dimension split, and what was reused

The overlap **ratio** `O = N[C]/(min(N[A],N[B]) + N[C])` (eqs. 16/28) is built from
Alexandrov-interval cardinalities alone and is dimension-agnostic, so
`alexandrov_interval`, `overlap_partition`, `causal_overlap`, `common_past`,
`filter2_passes` and `chain_count` are **re-exported from Phase 2a verbatim**. Only the
distance formula and the proper-time calibration change with dimension. Phase 2a is
therefore left untouched, in the same relation `order3d.py` bears to `order.py`.

**Notation clash, documented because it silently corrupts exponents:** Boguñá–Krioukov's
`d` is the **spatial** dimension (1 for M², 2 for M³); Rideout–Wallden's `d`, used
throughout `order3d`/`sprinkle3d`/`rideout_wallden`, is the **spacetime** dimension
(2 and 3). Every crossing goes through one function, `spacetime_dim_from_spatial`.

### The `c₂` derivation, checked three ways

    c_d = (d+1)/√π · Γ(d/2)/Γ((d+1)/2)                                    (eq. 27)
    c₂  = 3/√π · Γ(1)/Γ(3/2) = 3/√π · 1/(√π/2) = 6/π = 1.9098593171027443
    ⇒ prefactor 2/c₂ = π/3 = 1.0471975511965976

Computed by `overlap_coefficient_c`, never hardcoded. Verified against
`scipy.special.gamma` for d = 1…7 (≤1e−14), against the closed form 6/π (0.0 absolute),
and — the check that actually constrains the **factor** — against Phase 2a: the same
formula gives `c₁ = 2` exactly, so the asymptotic form reads `d = τ_c(1−O)`, which is
precisely the `O → 1` limit of the exact eq. 24 `d = τ_c(1−O)/√O`. The measured ratio
is `1/√O` and converges 1.41421 → 1.00005 over `O = 0.5…0.9999`. No test of the d = 2
code alone could have caught a factor error here.

### `α_d` from the chain law, and which `m₃`

    RW eq.(1)+(2):  L = m_D · l · (ρη(D))^{1/D}  ⇒  l = [1/(m_D η(D)^{1/D})] ρ^{−1/D} L
    vs B–K eq. 38:  τ = α_d ρ^{−1/(d+1)} L,  D = d+1
    ⇒  α_d = 1 / ( m_D · η(D)^{1/D} )

Reproduces Phase 2a: `α₁ = 1/(2·(1/2)^{1/2}) = 1/√2`, agreeing with the hardcoded
`ALPHA_1` to 1 ulp (the d = 1 path keeps the Phase 2a literal, which is what lets the
regression below demand *bit* identity rather than a tolerance).

**`m₃ = 2.296` is not used.** It is the asymptote of a fit supported only over
`ρV = 2¹⁰…2¹⁸`, and eq. 38 is applied to intervals of ~37 elements — 5 octaves below
that support, the exact error Part A's diagnostic traced the discarded 0.748 prediction
to. The module instead carries `M3_EFF_MEASURED`, a 14-row **measured** curve assembled
from this repository's own committed data and nothing else: `ρV = 16…64` from
`data/exp03b_diagnostic.npz`, `ρV = 2¹⁰…2¹⁷` re-reduced from `data/exp02_measurements.npz`
with the link convention `L = elements − 1`. A test re-derives all 14 rows from those
files at 1e−12, so the literals cannot drift from their source.

| calibration | m₃ | α₂ |
|---|---|---|
| measured `m₃^eff` at ρV = 64 | 1.8403 | **0.84941** |
| published asymptote | 2.296 | 0.68083 |

A 25 % difference — not a cosmetic choice. `m3_effective()` returns a **status** with
every read (`measured` / `interpolated_gap` / `extrapolated_low` / `extrapolated_high`),
so no caller can silently consume an extrapolation. The genuinely unmeasured four-octave
gap between `ρV = 64` and `1024` is flagged, not smoothed over.

### WEAKER ACCURACY CLAIM THAN PHASE 2a — stated wherever a 2+1 D B–K number appears

Phase 2a's 1+1 D result rests on an **exact** inversion valid at every overlap. The
2+1 D number is the **leading-order** eqs. 25–27, valid only for `τ_c ≫ separation`.
The realised `τ_c/d_est` is recorded for every admissible common event
(`per_c_asymptotic_ratio`), so the regime is auditable rather than assumed — see
Finding 5.

### Tests (26 new)

Four tiers. Constants (above). **Hand-built M³ causets** with A/B/C partitions worked out
by hand and every underlying relation asserted individually: `A={p2}, B={p3},
C={p0,p1,p4}`, `O = 3/4`; a timelike pair gives `O = 1`. One element, `p5`, sits off the
`y = 0` plane specifically so that dropping the third coordinate would misclassify it
into region C — asserted as an explicit negative control, so the test provably exercises
2+1 D structure. **Documented divergence:** `order3d`'s strict convention makes
exactly-null pairs *unrelated*, so B–K's "O = 1 for timelike/null" holds here for
timelike only; measure-zero for sprinklings, asserted rather than left as a surprise.

**The 1+1 D regression passes on bit identity, not tolerance.**
`distance_causal_overlap_nd(d=1)` reproduces `causal_overlap.distance_causal_overlap`
exactly — `distance`, `sem`, and all three per-`c` arrays compared with `==` over 8
sprinklings, plus component-level identity for `estimate_tau_c_nd` and
`distance_from_overlap_nd`. This is what licenses leaving Phase 2a unmodified. (It
required implementing eq. 38 in its literal factored spelling when a single α applies:
`a·(x+y)` and `a·x+a·y` differ in the last ulp in IEEE-754.)

Two bugs were caught by these tests and fixed: `searchsorted(side="left")` labelled the
measured nodes at `ρV = 64` and `1024` as `interpolated_gap` because they border the
gap; and the table literals, first transcribed at ~9 digits, carried 3e−9 error — the
literals were regenerated at full precision rather than the provenance check loosened.

---

## Part B2 — the head-to-head, GATE B

### Parameters — identical to Part A's Gate A, deliberately not deepened

`ρ = 60`, box `(T, Lx, Ly) = (4.0, 2.5, 5.0)`, targets at the mid-time on the x axis,
`x_± = (T/2, ±s/2, 0)`. **24 distinct separations** `s = 0.30…2.00` (spacing 0.0739; top
capped by `Lx = 2.5`), each measured on the **same 100 Poisson backgrounds** — only the
two injected target events move, so both estimators see a bit-identical causal matrix for
every (seed, separation). ⟨N⟩ = 3005. 2400 causal matrices, 10.5 min, one core.

The geometry was held at Part A's values rather than deepened in `T`: deepening would
improve B–K's asymptotic ratio but make the usability numbers incomparable with Part A's
1.16 2-links/causet and 36 % empty — and the asymptotic ratio limits **accuracy**, which
is not under test.

**Calibration asymmetry, intentional:** RW uses Part A's code unchanged (published
`m₃ = 2.296`, exactly as `exp03` called it); B–K uses the measured `m₃^eff` curve. ~25 %
apart, a pure per-estimator scale, invisible to rank ordering and fatal to any accuracy
comparison. Which is the point.

**RW's resolution floor, stated before the numbers:** one chain link = **0.1739** at this
density, and every minimising interval contains both targets (`p ≺ x ≺ f`), so `L ≥ 2`
and no single measurement can return below **0.3478**. The ladder spacing (0.0739) is
*below* that quantum: a single causet cannot order neighbouring pairs. The gate ranks
means over 100 causets, whose SE (~0.01–0.02) is well under the spacing.

### Finding 1 — the measurements (abridged; full table in the experiment output)

| s_true | RW 2-link | n | B–K (meas. m₃) | n | 2-links/causet | empty | ⟨n_c⟩ | τ_c/d |
|--------|-----------|---|----------------|---|----------------|-------|-------|-------|
| 0.300 | 0.4331 ± 0.0070 | 95 | 0.3325 ± 0.0077 | 100 | 6.07 | 5 % | 39.2 | 4.77 |
| 0.522 | 0.5469 ± 0.0085 | 92 | 0.5341 ± 0.0082 | 100 | 3.22 | 8 % | 27.5 | 2.76 |
| 0.743 | 0.6976 ± 0.0098 | 83 | 0.7185 ± 0.0098 | 100 | 1.96 | 17 % | 20.0 | 2.05 |
| 0.965 | 0.8430 ± 0.0123 | 71 | 0.8723 ± 0.0104 | 100 | 1.24 | 29 % | 16.2 | 1.71 |
| 1.187 | 1.0336 ± 0.0144 | 67 | 0.9884 ± 0.0123 | 100 | 1.12 | 33 % | 13.0 | 1.49 |
| 1.409 | 1.2436 ± 0.0203 | 44 | 1.0880 ± 0.0109 | 100 | 0.68 | 56 % | 10.1 | 1.35 |
| 1.630 | 1.3682 ± 0.0235 | 39 | 1.1521 ± 0.0129 | 100 | 0.53 | 61 % | 7.7 | 1.28 |
| 2.000 | 1.6990 ± 0.0239 | 34 | 1.2468 ± 0.0142 | 100 | 0.42 | 66 % | 5.0 | 1.18 |

Errors are standard errors **across causets** (measurements within one causet share a
sprinkling and a target pair and are not independent — Part A's convention).

### Finding 2 — GATE B: **PASSES** on all three criteria

| pairing | n pairs | Spearman ρ_s | p |
|---|---|---|---|
| RW vs TRUE separation | 24 | **+1.0000** | 1.09e−173 |
| B–K (measured m₃) vs TRUE | 24 | **+1.0000** | 1.09e−173 |
| RW vs B–K (measured m₃) | 24 | **+1.0000** | 1.09e−173 |
| B–K fixed α₂ vs TRUE | 24 | +1.0000 | 1.09e−173 |
| RW vs B–K fixed α₂ | 24 | +1.0000 | 1.09e−173 |
| B–K fixed vs B–K measured | 24 | +1.0000 | 1.09e−173 |

Criteria were fixed before the verdict: ρ_s > 0 and p < 0.01 on (B-i) RW vs true,
(B-ii) B–K vs true, (B-iii) RW vs B–K. **All pass.** Finding 8 assesses how much this
is worth.

### Finding 3 — ordering inversions: **zero**

**0 inverted (i,j) orderings out of 276 comparable pairs.** The inversion check was built
to classify each discordance by whether *each* estimator resolves its own difference at
>2 combined SE — distinguishing a genuine disagreement about order from two estimators
failing to separate two pairs. It found nothing to classify.

### Finding 4 — practical usability: the two estimators are not close

| | RW 2-link | B–K causal overlap |
|---|---|---|
| samples per causet (all 2400 evaluations) | **1.61** | **15.9** admissible `c` (from 249 candidates) |
| evaluations producing **nothing** | **897/2400 = 37 %** | **1/2400 = 0.04 %** |
| at s ≈ 1.0 | 1.24/causet, 29 % empty | 16.2/causet, 0 % empty |
| pairs with <10 usable causets of 100 | 0/24 | 0/24 |

The s ≈ 1.0 row reproduces Part A's independently measured 1.16/causet and 36 % empty —
a like-for-like cross-check across two experiments and two seed streams.

**New, and not measurable from Part A's design** (which varied the region at fixed `s`,
never `s` at fixed region): RW's 2-link yield falls as **s^−1.46** over this ladder,
from 6.07 per causet at `s = 0.30` to 0.42 at `s = 2.00`, with the empty rate rising
**5 % → 66 %**. B–K's admissible-`c` count also falls (39.2 → 5.0) but never to zero.

### Finding 5 — asymptotic-regime audit: B–K sits at the EDGE of its regime

`τ_c/d_est` over all causets: median **1.54**, 10th percentile **1.19**, worst single
vantage point **0.99**. By separation it runs **4.77 at s = 0.30 down to 1.18 at
s = 2.00**. eqs. 25–27 require `τ_c ≫ separation`; values of order 1–2 are **not** that.

This is reported as a limit on **accuracy**, which no claim here depends on. The `m₃^eff`
curve was read at a mean interval size of **37 elements** — inside the measured range,
requiring no extrapolation.

### Finding 6 — the α policy moves no rank (robustness)

Fixed `α₂ = 0.84941` (a constant, hence a pure global scale) versus the `measured_m3`
policy (read per interval, hence genuinely pair-dependent — B1's tests assert both
properties). Over the same 100 causets: **0 of 24 ranks differ**, Spearman(fixed,
measured) = +1.0000. The gate does not rest on which policy was chosen.

### Finding 7 — ordering on a SINGLE causet: the test with teeth

The aggregate gate ranks *means over 100 causets*, which suppresses exactly the noise
that would produce an inversion — and Phase 4 will not have 100 causets per pair. Per
causet (RW resolves 15.0 of the 24 separations on an average causet):

| pairing | median ρ_s | mean ρ_s | min | n causets |
|---|---|---|---|---|
| RW vs TRUE | +0.976 | +0.965 ± 0.003 | +0.807 | 100 |
| B–K vs TRUE | +0.960 | +0.934 ± 0.007 | +0.660 | 100 |
| RW vs B–K | +0.949 | +0.937 ± 0.005 | +0.629 | 100 |

**Strong but not perfect.** These, not the aggregate 1.0000, are the numbers Phase 4
should carry.

### Finding 8 — could the aggregate gate have failed? Barely

Bootstrapping the 100 causets (2000 replicates, resampled with replacement), the
aggregate Spearman(RW, B–K) spans **[0.9948, 1.0000]** with median 0.9991. Both
estimators are strongly monotone in `s` and the per-pair standard errors (~0.01–0.02)
sit far below the ladder spacing (0.0739), so **the aggregate test was never at serious
risk of failing.** It rules out gross ordering disagreement and nothing finer. Recorded
at the same prominence as the PASS, because a gate that could not have failed is weak
evidence however small its p-value.

### Finding 9 — B–K compresses at large separation; RW does not. Cause NOT established

| s | B–K/true | RW/true | τ_c/d | RW 2-links/causet |
|---|---|---|---|---|
| 0.300 | 1.108 | 1.444 | 4.77 | 6.07 |
| 0.743 | 0.966 | 0.938 | 2.05 | 1.96 |
| 1.187 | 0.833 | 0.871 | 1.49 | 1.12 |
| 1.630 | 0.707 | 0.839 | 1.28 | 0.53 |
| 2.000 | 0.623 | 0.850 | 1.18 | 0.42 |

B–K/true falls monotonically **1.108 → 0.623**. RW/true is non-monotone: **1.444** at the
smallest separation, where the 2-link floor of 0.3478 inflates it, then **0.83–0.85**
across the rest.

`Spearman(B–K/true, τ_c/d) = +1.0000`. **This number is confounded and is not evidence
of a mechanism.** Both quantities are monotone functions of `s` by construction — B–K/true
because the estimator compresses, τ_c/d because a wider pair has less room above it in a
fixed box — so a rank correlation of ~1 between them is what two monotone functions of a
common variable always give, mechanism or not. It is reported because it is *consistent*
with the eqs. 25–27 breakdown, and for no stronger reason.

**What would actually test it, and was not done:** hold `s` fixed and vary the box time
extent, moving τ_c/d without moving `s`. If B–K/true rose toward 1 as the region deepened
at fixed `s`, the asymptotic explanation would be established; if it did not, the
compression has another cause. **Logged as an open question, not a conclusion.**

### Interpretation (honest)

**Gate B passes, and the pass is real but weaker than its p-value suggests.** The two
published estimators, run on bit-identical causal matrices with deliberately different
and separately unreliable calibrations, agree perfectly on the ordering of 24 spacelike
pairs and each agrees perfectly with the truth — 0 inversions in 276 comparisons. That is
the property the cross-check existed to establish, and it is established.

But Finding 8 is the honest qualifier: with both estimators strongly monotone and 100
causets of averaging, the aggregate test had almost no room to fail. The informative
result is Finding 7 — on a **single** causet the agreement is 0.95–0.98 median and dips
to 0.63 at worst. Two estimators that agree perfectly in aggregate still disagree
measurably one causet at a time, and Phase 4 lives in the single-causet regime.

The second qualifier is Finding 9. B–K's 2+1 D distance is an asymptotic form used at
`τ_c/d ≈ 1.2–1.5` over most of the ladder, and it compresses badly there — reading 62 %
of truth at `s = 2.00`. That the compression tracks the asymptotic ratio is *consistent*
with the formula's own stated validity condition, but the correlation is confounded
through `s` and does not establish it; the fixed-`s`, varying-`T` experiment that would
is named above and not done. Ordering survives the compression, which is why the gate is
specified on ordering — but the compression is monotone and would eventually collapse
ranks if extended further.

**The usability gap is the most consequential result for Phase 4, and it is not close.**
B–K returned an estimate on 2399 of 2400 evaluations with ~16 vantage points each; RW
returned nothing on 37 % of them and averaged 1.6 samples when it did, with a hard
resolution floor of 0.3478 and a yield falling as `s^−1.46`. Against that, RW tracks the
truth linearly across the whole range while B–K flattens. The two estimators fail in
opposite directions: **RW is accurate in shape but scarce and coarse; B–K is abundant
and fine-grained but compresses out of its asymptotic regime.**

### Consequence for Phase 4

- Use **B–K causal overlap** as the primary Δs benchmark: it yields ~16 independent
  vantage points per causet against RW's 1.6, and effectively never fails.
- Restrict it to the regime where its asymptotic form holds. On this geometry that means
  `s ≲ 0.75` (`τ_c/d ≳ 2`); beyond that its compression is a real distortion of magnitude,
  though not of order.
- Keep **RW 2-link** as an independent cross-check on ordering only, not as a per-causet
  measurement: 37 % of causets return nothing and the 0.3478 floor is 35 % of `D = 1`.
- Neither estimator's scale may be carried forward. RW's offset is the unstable
  cancellation of Part A's diagnostic; B–K's is an asymptotic-regime artefact measured
  here. Both are reported; neither is corrected.
- **Open:** the fixed-`s`, varying-`T` diagnostic that would confirm or refute the
  asymptotic explanation of B–K's compression (Finding 9).

---

## 2026-09-23 — Phase 2b Part B DIAGNOSTIC: breaking Finding 9's confound (fixed s, varying T)

- **Branch:** `phase2b-2d1` (follows the Part B entry of 2026-09-22)
- **Experiment:** `experiments/exp05_bk_asymptotic_regime.py`
- **Figure:** `figures/exp05_bk_asymptotic_regime.png` (4 panels)
- **Raw data:** `data/exp05_measurements.npz` (per-causet **and** per-vantage-point values — 68 575 individual `(τ_c, d_c)` pairs, committed)
- **Seeds:** `20260923 + 100000·i + k` (rung `i`, causet `k`)
- **Scope:** **diagnostic, not a gate.** No pass/fail is issued. Its one job is to close or explicitly leave open the confound logged as Finding 9 on 2026-09-22.
- **Tests:** `python -m pytest` → **124 passed** (no source module changed; this experiment adds no library code)

### The confound this existed to break

exp04 measured `B–K/true` falling 1.108 → 0.623 as `s` ran 0.30 → 2.00, while
`τ_c/d` fell 4.77 → 1.18 over the same ladder, giving
`Spearman(B–K/true, τ_c/d) = +1.0000`. That correlation was recorded as **worth
nothing on its own**: both quantities are monotone in `s` by construction. This
experiment holds `s = 1.0` FIXED and varies the box time extent instead, so depth
moves without separation moving.

**Geometry deliberately differs from exp03/exp04** (whose `T` was frozen at 4.0
for comparability). No number here is comparable with those experiments on
absolute scale, and none is so compared.

### Parameters

`ρ = 60` FIXED, `s = 1.0` FIXED, spatial extents `(Lx, Ly) = (2.5, 5.0)` FIXED.
Box time extent `T ∈ {4, 6, 8, 10, 12}`. **Causet counts fall along the ladder —
100, 100, 80, 60, 40 — because cost does forbid 100 throughout:** B–K's admissible
`c` count grows 15.7 → 683.5 per causet, each needing two chain-count dynamic
programs over larger interval sub-posets, so the rungs cost 26 s → 1052 s. Total
380 causets, ⟨N⟩ = 3004 → 9025, 30.3 min. Every count is reported and every
standard error computed from the count actually used.

### Finding 1 — the independent variable moved, and by a lot

| T | V | ⟨N⟩ | causets | ⟨τ_c⟩ | median τ_c/d | B–K ⟨n_c⟩ |
|---|---|---|---|---|---|---|
| 4 | 50 | 3004 | 100 | 1.49 | 1.67 ± 0.02 | 15.7 |
| 6 | 75 | 4500 | 100 | 2.10 | 2.29 ± 0.02 | 50.3 |
| 8 | 100 | 6010 | 80 | 3.02 | 3.75 ± 0.03 | 150.7 |
| 10 | 125 | 7495 | 60 | 3.88 | 5.53 ± 0.08 | 376.3 |
| 12 | 150 | 9025 | 40 | 4.59 | 6.94 ± 0.26 | 683.5 |

`τ_c/d` spans **1.67 → 6.94, a factor 4.16** — from the marginal regime exp04
operated in, well into `τ_c ≫ d` where eqs. 25–27 should hold. The diagnostic had
the lever arm it needed.

### Finding 2 — B–K/true FALLS as the region deepens; the RW control does not

| T | median τ_c/d | B–K/true | n | RW/true | n |
|---|---|---|---|---|---|
| 4 | 1.67 | 0.8910 ± 0.0111 | 100 | 0.8566 ± 0.0120 | 65 |
| 6 | 2.29 | 0.9128 ± 0.0063 | 100 | 0.8708 ± 0.0140 | 70 |
| 8 | 3.75 | 0.8523 ± 0.0053 | 80 | 0.9109 ± 0.0172 | 48 |
| 10 | 5.53 | 0.7584 ± 0.0075 | 60 | 0.8998 ± 0.0182 | 44 |
| 12 | 6.94 | 0.7008 ± 0.0146 | 40 | 0.9045 ± 0.0234 | 29 |

Weighted slopes of ratio vs T: **B–K `−0.02894 ± 0.00165` (17.5 σ)**, RW
`+0.00729 ± 0.00264` (**2.8 σ**).

Taken at face value this **refutes** the asymptotic explanation outright, and in
the opposite direction to the prediction: deepening the region makes B–K *worse*.
Finding 4 is why it must not be taken at face value.

**The RW control, reported honestly:** RW drifts mildly *upward*, 0.857 → 0.905
(+0.048 total). At 2.8 σ this sits below the 3 σ threshold fixed in advance, so it
is not declared "moving" — but it is not flat either, and it is recorded as a weak
positive drift rather than rounded to zero. Part A's Gate A found RW flat under
**isotropic** growth (−0.009 ± 0.030 over a factor 18 in volume); growth in `T`
alone is a different direction, and this is the first measurement of it. The
signs are **opposite** and B–K's magnitude is 4× larger, so a common systematic
driving both is excluded.

### Finding 3 — per-vantage stratification: the confound-free test

Each admissible `c` supplies its own `d_c` from its own depth `τ_c`. In the
continuum the estimator should return the same answer from every vantage point —
a deeper `c` has larger `τ_c` but correspondingly smaller `(1−O)`, and they
compensate exactly. So a slope of `d_c/true` against `τ_c` is a direct test of
eqs. 25–27, with no `s` dependence and (within a rung) no `T` dependence.

**Stratified by `τ_c` ALONE, never by `τ_c/d_c`** — binning by the ratio would
repeat exactly the circularity Finding 9 flagged, since `d_c` sits in its own
denominator and a low `d_c` mechanically produces a high ratio.

Pooled across rungs (8 equal-count bins, 8 571 vantage points each):

| ⟨τ_c⟩ | 1.54 | 2.60 | 3.31 | 3.77 | 4.15 | 4.55 | 5.00 | 5.70 |
|---|---|---|---|---|---|---|---|---|
| `d_c`/true | 0.878 | **0.928** | 0.839 | 0.770 | 0.720 | 0.671 | 0.622 | 0.562 |

**Non-monotone, peaking at τ_c ≈ 2.6.** Within each rung separately (T constant,
so no T confound at all):

| T | τ_c range | within-rung slope | σ | frac clipped in y |
|---|---|---|---|---|
| 4 | 0.68–2.57 | **+0.24292 ± 0.00952** | 25.5 | **0.00** |
| 6 | 0.68–3.73 | +0.04785 ± 0.00258 | 18.5 | 0.33 |
| 8 | 0.68–4.65 | −0.07699 ± 0.00123 | 62.8 | 0.76 |
| 10 | 0.68–5.89 | −0.09535 ± 0.00076 | 124.9 | 0.91 |
| 12 | 0.68–6.64 | −0.09715 ± 0.00050 | 192.6 | 0.95 |

In the T = 4 rung the curve rises from **0.69 at τ_c ≈ 0.9 to 1.01 at τ_c ≈ 2.1** —
essentially exact recovery at the top.

**Caveat on those error bars:** vantage points inside one causet share a
sprinkling and a target pair and are not independent, so the quoted σ are
optimistic. The slopes are quoted for **direction and relative size**, not for
their literal significance.

### Finding 4 — a confound THIS experiment introduced: spatial clipping

**Growing `T` alone does not isolate `τ_c`.** A common event at depth `τ` below the
targets has a future light cone of radius ~`τ` by the time it reaches them, but the
spatial half-extents were held fixed at `Lx/2 = 1.25` and `Ly/2 = 2.50`. Past those
depths the Alexandrov interval is truncated by the **box**, not by the light cone.

| T | ⟨τ_c⟩ | frac clipped in x | frac clipped in y | within-rung slope |
|---|---|---|---|---|
| 4 | 1.49 | 0.67 | **0.00** | **+0.243** |
| 6 | 2.10 | 0.87 | 0.33 | +0.048 |
| 8 | 3.02 | 0.96 | 0.76 | −0.077 |
| 10 | 3.88 | 0.98 | 0.91 | −0.095 |
| 12 | 4.59 | 0.99 | 0.95 | −0.097 |

**The within-rung slope flips sign in lockstep with the y-clipping fraction**, and
the pooled peak at `τ_c ≈ 2.6` sits essentially on the `Ly/2 = 2.50` threshold.
Depth and clipping move together along this ladder, so the ladder cannot attribute
the aggregate trend of Finding 2 to either one.

This is a flaw in the experiment's own design, not in the estimator, and it is
named rather than buried. Holding the spatial extent fixed was what made `τ_c` the
only *nominal* variable — but in a spatially bounded box, growing `τ_c` necessarily
grows clipping too.

### Verdict — PARTIALLY SUPPORTED, and the ladder is not usable as built

1. **In the least-clipped geometry — T = 4, exp04's own, where no vantage point is
   clipped in y — deeper vantage points give BETTER estimates** (`+0.243 ± 0.010`),
   reaching exact recovery at `τ_c ≈ 2.1`. That is the direction eqs. 25–27 predict,
   measured at fixed `s` and fixed region, free of exp04's monotone-in-`s` confound.
   **This positively supports the asymptotic explanation of exp04's large-s
   compression, in the regime exp04 actually operated in.**
2. **The aggregate T ladder points the other way** (−17.5 σ) — but it is confounded
   with clipping, which switches on across exactly the rungs where the sign flips.
   It is not evidence against (1), and it is not evidence for anything else either.

So Finding 9's confound is **partly broken, not fully**. The asymptotic-breakdown
explanation is upgraded from "consistent with" to "positively supported at small
τ_c", but **the validity boundary of the 2+1 D B–K formula still cannot be stated
quantitatively**, because this experiment's depth ladder is not clean beyond
`τ_c ≈ 2.5`.

**What would settle it, and has not been run:** repeat the ladder growing `Lx` and
`Ly` in proportion to `T`, so the clipped fraction stays constant while `τ_c` grows.
Part A's Gate A already used isotropic growth for exactly this kind of reason; this
experiment should have too.

### Interpretation (honest)

**The headline is that the prepared hypothesis survived, but only after the
experiment built to test it was found to be testing two things at once.** Had the
T-ladder result been reported alone it would have read as a clean 17.5 σ refutation,
and it would have been wrong. Had the within-rung result been reported alone it
would have read as a clean confirmation, and it would have been overstated. Both are
reported, and the reason they disagree is measured rather than argued.

**The most useful number to come out of this is one nobody asked for:** B–K's
per-vantage accuracy is strongly depth-dependent and peaks near `τ_c ≈ 2–2.5`,
where it recovers the true separation to ~1 %. The aggregate estimator averages
over *all* admissible `c`, shallow ones included, which is what drags it to
0.85–0.91. **Untested suggestion, logged as a suggestion and nothing more:**
selecting or weighting vantage points by depth — keeping `τ_c` in the range where
the per-vantage curve peaks and below the clipping threshold — may recover much of
the gap. It has not been implemented, has not been validated, and would need its
own acceptance test before being used for anything.

**A second, weaker observation worth recording:** RW drifts upward by +0.048 across
the T ladder (2.8 σ). Part A established RW flatness under isotropic growth; under
anisotropic growth in `T` alone this is the first look, and it is not perfectly
flat. Below the pre-set threshold, so not claimed as an effect — but logged, because
it is the kind of sub-threshold drift that gets quietly forgotten and then
rediscovered as a surprise.

### Consequence for Phase 4 (revising the 2026-09-22 entry)

- The recommendation to prefer **B–K** as the primary Δs benchmark is **unchanged
  and strengthened**: its accuracy problem now has a located, depth-dependent
  structure rather than being an unexplained compression.
- The previous entry's advice to restrict B–K to `s ≲ 0.75` (`τ_c/d ≳ 2`) is
  **superseded in form**: the controlling variable is the *vantage-point depth*
  `τ_c`, not the separation, and the useful window is bounded **above** by boundary
  clipping as well as below by the asymptotic breakdown. The numerical window
  cannot be stated until the isotropic-growth ladder is run.
- **Open, and now more sharply posed than before:** the isotropic version of this
  ladder (grow `Lx`, `Ly` with `T`). Until it is run, no quantitative validity
  boundary for the 2+1 D B–K distance should be quoted from this work.
