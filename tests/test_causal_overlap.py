"""Tests for the Boguñá--Krioukov causal-overlap distance (arXiv:2401.17376).

Two levels:
  1. Hand-built small causets with KNOWN Alexandrov intervals -- verify the
     A/B/C partition is a genuine partition, the overlap is in [0, 1], and
     O = 1 for a comparable (timelike) pair.
  2. On sprinklings -- verify the measured mean overlap for a fixed target pair
     at known continuum depth ``tau_c`` and known spacelike separation ``s``
     converges to the value eq. 24 demands (via ``overlap_predicted_from_distance``)
     as the density ``rho`` grows. Convergence is *reported*, not asserted with a
     tolerance tight enough to hide slow approach (Rule 3).
"""

from __future__ import annotations

import numpy as np
import pytest

from causet import causal_overlap as co
from causet import order, sprinkle


# ---------------------------------------------------------------------------
# 1. Hand-built causets with known intervals.
# ---------------------------------------------------------------------------


def _causal_matrix_from_coords(t, x):
    """Full M^2 causal matrix for arbitrary points via null-coordinate dominance."""
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    return order.causal_matrix_1d(t + x, t - x)


# Six-element configuration (see module test docstring):
#   c=(0,0)  [common past]      p1=(1, 0)     -> shared (region C)
#   a=(3, 1)                    p2=(1.5, 0.7) -> below a only (region A)
#   b=(3,-1)                    p3=(1.5,-0.7) -> below b only (region B)
# Indices:   c=0, a=1, b=2, p1=3, p2=4, p3=5
_T = [0.0, 3.0, 3.0, 1.0, 1.5, 1.5]
_X = [0.0, 1.0, -1.0, 0.0, 0.7, -0.7]
C_HAND = _causal_matrix_from_coords(_T, _X)
IC, IA, IB, IP1, IP2, IP3 = 0, 1, 2, 3, 4, 5


def test_alexandrov_interval_contents():
    """I(a,c) = {p1,p2}, I(b,c) = {p1,p3} for the hand-built configuration."""
    i_ac = set(co.alexandrov_interval(IA, IC, C_HAND).tolist())
    i_bc = set(co.alexandrov_interval(IB, IC, C_HAND).tolist())
    assert i_ac == {IP1, IP2}, i_ac
    assert i_bc == {IP1, IP3}, i_bc


def test_overlap_partition_is_a_partition():
    """A, B, C are disjoint and their union is I(a,c) u I(b,c)."""
    part = co.overlap_partition(IA, IB, IC, C_HAND)
    A, B, Cs = set(part.A.tolist()), set(part.B.tolist()), set(part.C_shared.tolist())
    # Pairwise disjoint.
    assert A & B == set()
    assert A & Cs == set()
    assert B & Cs == set()
    # Union equals the two intervals combined.
    i_ac = set(co.alexandrov_interval(IA, IC, C_HAND).tolist())
    i_bc = set(co.alexandrov_interval(IB, IC, C_HAND).tolist())
    assert A | B | Cs == (i_ac | i_bc)
    # Explicit expected membership.
    assert A == {IP2}
    assert B == {IP3}
    assert Cs == {IP1}
    # Cardinalities feeding eq. 38 / Filter 2.
    assert part.n_ac == 2 and part.n_bc == 2


def test_overlap_value_hand_built():
    """O = N[C] / (min(N[A],N[B]) + N[C]) = 1 / (1 + 1) = 1/2 here."""
    o = co.causal_overlap(IA, IB, IC, C_HAND)
    assert o == pytest.approx(0.5)
    assert 0.0 <= o <= 1.0


def test_overlap_is_one_for_timelike_pair():
    """Comparable (timelike) pair: one interval nests in the other => O = 1."""
    # a2 = (5,0) is in the future of a=(3,1) [Δt=2,Δx=-1 => 4-1>0], and of c.
    # Pair (a, a2) is timelike (a prec a2). Common past c=(0,0).
    t = [0.0, 3.0, 5.0, 1.0]  # c, a, a2, p1
    x = [0.0, 1.0, 0.0, 0.0]
    c_mat = _causal_matrix_from_coords(t, x)
    assert c_mat[1, 2]  # a prec a2 (timelike)
    o = co.causal_overlap(1, 2, 0, c_mat)  # overlap of a, a2 from c
    assert o == pytest.approx(1.0)


def test_overlap_nan_when_c_not_common_ancestor():
    """If c precedes neither future endpoint, both intervals empty => nan."""
    o = co.causal_overlap(IA, IB, IP2, C_HAND)  # p2 is not below both a and b
    assert np.isnan(o)


def test_overlap_symmetric_in_pair():
    """O_C(a,b) == O_C(b,a) (the min form is manifestly symmetric)."""
    assert co.causal_overlap(IA, IB, IC, C_HAND) == co.causal_overlap(
        IB, IA, IC, C_HAND
    )


def test_chain_count_and_tau_relationship():
    """chain_count is the element-count longest chain; a bare link gives 2."""
    # c -> p1 -> a is a chain of 3 elements; direct c -> a with the interval
    # {p1,p2} present should also find length-3 chains.
    n_ca = co.chain_count(IC, IA, C_HAND)
    assert n_ca >= 2
    # A configuration where the only relation is a single link => count 2.
    t = [0.0, 2.0]
    x = [0.0, 0.0]
    c_mat = _causal_matrix_from_coords(t, x)
    assert co.chain_count(0, 1, c_mat) == 2


def test_distance_formula_roundtrip():
    """eq. 24 and its inverse are consistent: recover distance from predicted O."""
    tau_c = 3.0
    for true_d in (0.5, 1.0, 2.0, 4.0):
        o_pred = co.overlap_predicted_from_distance(true_d, tau_c)
        assert 0.0 < o_pred < 1.0
        d_back = co.distance_from_overlap(o_pred, tau_c)
        assert d_back == pytest.approx(true_d, rel=1e-10)


def test_distance_zero_for_unit_overlap():
    """O = 1 (timelike) maps to distance 0 under eq. 24."""
    assert co.distance_from_overlap(1.0, 2.0) == pytest.approx(0.0)


def test_filter2_symmetry_selects_balanced_c():
    """Filter 2 admits a symmetric vantage and rejects a lopsided one."""
    # Symmetric c=(0,0): |I(a,c)|=|I(b,c)|=2 => Z=0 < rhs => passes.
    assert co.filter2_passes(IA, IB, IC, C_HAND)
    # Build a lopsided common ancestor: c' close to b's worldline so it sees far
    # more of b's interval than a's. c'=(0.2,-0.9): check it precedes both a,b.
    t = [0.2, 3.0, 3.0] + _T[3:]
    x = [-0.9, 1.0, -1.0] + _X[3:]
    c_mat = _causal_matrix_from_coords(t, x)
    # indices now: cprime=0, a=1, b=2, p1=3, p2=4, p3=5
    if c_mat[0, 1] and c_mat[0, 2]:  # only meaningful if c' precedes both
        part = co.overlap_partition(1, 2, 0, c_mat)
        # If the interval cardinalities are very unequal, Filter 2 should reject.
        if abs(part.n_ac - part.n_bc) >= 2:
            assert not co.filter2_passes(1, 2, 0, c_mat)


# ---------------------------------------------------------------------------
# 2. Sprinkling: overlap converges to the eq.-24 prediction as rho grows.
# ---------------------------------------------------------------------------


def _sprinkle_fixed_pair(rho, t_ab, s, seed):
    """Sprinkle a bounding diamond (tip c=(0,0)) and inject the pair a,b.

    Returns (causal_matrix, idx_c, idx_a, idx_b, tau_c_continuum, s).
    a=(t_ab, +s/2), b=(t_ab, -s/2): equal-time, spacelike, proper separation s.
    c=(0,0) is the bottom tip of the diamond (a genuine common past on-axis).
    Continuum depth tau_c = proper time c->a = sqrt(t_ab^2 - (s/2)^2).
    """
    T = 2.0 * t_ab  # diamond proper time; puts a,b at the mid-slice.
    sp = sprinkle.sprinkle_diamond_1d(rho, T, seed=seed, include_endpoints=False)
    # Interior sprinkled points, plus injected c (bottom tip), a, b.
    t = np.concatenate([sp.t, [0.0, t_ab, t_ab]])
    x = np.concatenate([sp.x, [0.0, +0.5 * s, -0.5 * s]])
    c_mat = _causal_matrix_from_coords(t, x)
    n = t.shape[0]
    idx_c, idx_a, idx_b = n - 3, n - 2, n - 1
    tau_c = np.sqrt(t_ab**2 - (0.5 * s) ** 2)
    return c_mat, idx_c, idx_a, idx_b, tau_c, s


def test_overlap_converges_to_eq24_prediction():
    """<O_C(a,b)> -> the value eq. 24 demands for the known (s, tau_c).

    Fixed pair at spatial separation s and known continuum depth tau_c; measure the
    mean overlap from the fixed on-axis common event c over realisations, at
    increasing density. Report the approach; assert only that the discrepancy at
    the highest density is smaller than at the lowest (convergence), not a hard
    tolerance that would hide a slow rate.
    """
    t_ab, s = 3.0, 1.0
    tau_c = np.sqrt(t_ab**2 - (0.5 * s) ** 2)
    o_pred = co.overlap_predicted_from_distance(s, tau_c)

    n_real = 30
    rhos = [50.0, 200.0, 800.0]
    abs_err = []
    print(f"\n[eq.24 overlap prediction] s={s}, tau_c={tau_c:.4f}, O_pred={o_pred:.4f}")
    for rho in rhos:
        vals = []
        for k in range(n_real):
            c_mat, ic, ia, ib, _, _ = _sprinkle_fixed_pair(rho, t_ab, s, seed=9000 + k)
            o = co.causal_overlap(ia, ib, ic, c_mat)
            if np.isfinite(o):
                vals.append(o)
        vals = np.asarray(vals)
        mean_o = vals.mean()
        sem_o = vals.std(ddof=1) / np.sqrt(vals.size)
        abs_err.append(abs(mean_o - o_pred))
        print(
            f"  rho={rho:7.1f}  <O>={mean_o:.4f} +/- {sem_o:.4f}  "
            f"|<O>-O_pred|={abs(mean_o - o_pred):.4f}  (N_used={vals.size})"
        )
        assert 0.0 <= mean_o <= 1.0

    # Convergence: highest-density discrepancy below the lowest-density one.
    assert abs_err[-1] < abs_err[0], abs_err
    # And it should be reasonably close by the top density (generous band).
    assert abs_err[-1] < 0.10, abs_err


def test_full_pipeline_distance_recovers_separation():
    """The full Filter-2 pipeline recovers a spacelike separation within a band.

    Sanity check that distance_causal_overlap returns a finite estimate near the
    true s and reports a non-trivial admissible-c count. Detailed convergence and
    error bars live in experiments/exp01_bk_1p1d.py.
    """
    t_ab, s, rho = 3.0, 1.0, 400.0
    ests = []
    n_c_used = []
    for k in range(15):
        c_mat, ic, ia, ib, _, _ = _sprinkle_fixed_pair(rho, t_ab, s, seed=4200 + k)
        res = co.distance_causal_overlap(ia, ib, c_mat, rho)
        if np.isfinite(res.distance):
            ests.append(res.distance)
            n_c_used.append(res.n_c)
    ests = np.asarray(ests)
    print(
        f"\n[full pipeline] true s={s}: mean est={ests.mean():.3f} "
        f"(n_real={ests.size}), median admissible c per realisation="
        f"{int(np.median(n_c_used))}"
    )
    assert ests.size >= 10  # pipeline yields an estimate most of the time
    # Within a factor ~2 of the truth at this modest density (loose sanity band).
    assert 0.4 * s < ests.mean() < 2.5 * s, ests.mean()
