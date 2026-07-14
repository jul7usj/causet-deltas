# causet-deltas

Numerical verification of the **Δs spacelike-distance estimator** on causal sets,
supporting the manuscript's Standing Hypothesis 5.1:
for Poisson sprinklings into flat spacetime,
`E[Δs(a,b)] ≈ r(a,b)² / (η l_P²)` with `η = 4 ln 2`, at small spatial separation.

This repository produces the evidence (or refutation) in the format standard for
the causal set literature: sprinkling experiments, binned statistics, error bars,
convergence plots — every figure regenerable from a recorded seed.

## Status

**Phase 1 complete** — sprinkling, causal order, antichains, and the
Brightwell–Gregory geodesic-law validation. Later phases (baselines, Sorkin–
Johnston Wightman function, the Δs estimator, scaling/comparison/convergence
experiments) are not yet implemented; see the build order in the project brief.

## Install / environment

Python ≥ 3.11 with `numpy`, `scipy`, `matplotlib` (and `pytest` for tests).
No install step is required to run the code or tests — the `src/` layout is put
on `sys.path` by `tests/conftest.py` and by each experiment script. Optionally:

```bash
pip install -e .
```

## Reproduce Phase 1

Run the test suite (includes the Brightwell–Gregory acceptance checks):

```bash
python -m pytest
```

Regenerate the Phase-1 validation table and figure:

```bash
python experiments/exp00_phase1_validation.py
# -> figures/exp00_brightwell_gregory.png
```

## What Phase 1 establishes

The 1+1 D causal order of a diamond sprinkling is the 2-D dominance order, so the
longest chain (discrete timelike geodesic) obeys the exactly-known law
`E[L]/√N → 2` (Vershik–Kerov / Logan–Shepp constant; the 1+1 D Brightwell–Gregory
result). Reproducing this validates the entire geometric pipeline — sprinkling
statistics, causal matrix, transitive reduction — against the one theorem everyone
trusts, before any estimator is built on top of it.

## Layout

```
src/causet/
  sprinkle.py    Poisson sprinkling into 1+1 D Alexandrov intervals
  order.py       causal matrix, transitive reduction (links), longest chain
  antichain.py   antichains and maximal (inextendible) antichains
experiments/
  exp00_phase1_validation.py   Brightwell–Gregory validation (Phase 1)
tests/           pytest: Poisson stats, transitivity, antichain maximality, BG law
RESULTS.md       append-only log: date, seed, params, finding, interpretation
```

See `RESULTS.md` for the running experimental log.
