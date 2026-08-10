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
Rideout–Wallden's ±0.012; with ±0.099 anything in ≈[2.14, 2.54] would have "passed".
Spread across fit variants, which exceeds every individual quoted error:

| fit variant | m₃ | note |
|---|---|---|
| interior conv., `c` free (**primary**) | 2.3427 ± 0.0975 | only variant whose error reflects extrapolation freedom |
| link conv., `c` free | 2.4722 ± 0.3113 | contaminated by the convention offset (below) |
| interior conv., `c` fixed to RW's `b/ln2` | 2.3282 ± 0.0104 | error understated — `c` assumed |
| link conv., `a` and `c` both pinned to RW | 2.3036 ± 0.0019 | error meaningless — 1 free parameter |
| **Rideout–Wallden published** | **2.2960 ± 0.0120** | |

**All four of our estimates lie at or above 2.296** (2.303, 2.328, 2.343, 2.472); none
below. Logged as an observation, not a claim — ±0.099 cannot support it. The
method-to-method spread (~0.04 among the defensible variants) is the real uncertainty
on any such extrapolation, including the published one.

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
