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
