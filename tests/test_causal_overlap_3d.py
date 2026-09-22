"""Tests for the 2+1 D Boguna--Krioukov causal overlap (Phase 2b Part B1).

Four levels, in the order the brief requires them:

  1. **The constants.** ``c_2`` is derived in the module docstring and computed
     by ``overlap_coefficient_c``; here it is checked three independent ways --
     against ``scipy.special.gamma``, against the closed form ``6/pi``, and via
     the ``d = 1`` limit where the asymptotic formula must reduce to Phase 2a's
     exact eq. 24. A factor error in ``c_d`` would silently corrupt every 2+1 D
     distance with no other failing test, which is why it gets three.
     ``alpha_d`` and the measured ``m_3^eff`` table get the same treatment, the
     table being re-derived from the committed ``.npz`` files rather than
     trusted as literals.
  2. **Hand-built 2+1 D causets** with A/B/C partitions worked out by hand, each
     underlying causal relation asserted individually so a failure localises.
     One element (``p5``) is placed off the ``y = 0`` plane specifically so that
     code which ignored the third coordinate would misclassify it.
  3. **The 1+1 D regression** -- the generalised pipeline given ``d = 1`` must
     reproduce Phase 2a's ``causal_overlap.distance_causal_overlap`` output
     BIT for bit, not within a tolerance. This is what licenses leaving the
     Phase 2a module untouched.
  4. **2+1 D pipeline behaviour** on real sprinklings: the overlap stays in
     [0, 1], the asymptotic-regime diagnostics are recorded, and a fixed
     ``alpha_d`` is confirmed to act as a pure scale (the property the Part-B
     rank-ordering gate rests on).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from scipy.special import gamma as scipy_gamma

from causet import causal_overlap as co
from causet import causal_overlap_3d as co3
from causet import order, order3d, sprinkle, sprinkle3d

DATA = Path(__file__).resolve().parents[1] / "data"


# ---------------------------------------------------------------------------
# 1. The constants: c_d, alpha_d, and the measured m_3^eff table.
# ---------------------------------------------------------------------------


def test_c_d_matches_scipy_gamma():
    """``overlap_coefficient_c`` equals eq. 27 evaluated with scipy's Gamma.

    The module uses ``math.gamma``; scipy is an independent implementation, so
    this is a genuine cross-check of the expression as written, not a tautology.
    """
    for d in range(1, 8):
        expected = (d + 1) / math.sqrt(math.pi) * scipy_gamma(d / 2.0) / scipy_gamma((d + 1) / 2.0)
        assert co3.overlap_coefficient_c(d) == pytest.approx(expected, rel=1e-14, abs=1e-15)


def test_c_2_equals_six_over_pi_by_the_documented_derivation():
    """c_2 = 3/sqrt(pi) * Gamma(1)/Gamma(3/2) = 6/pi, step by step.

    Each line of the module docstring's derivation is evaluated separately, so a
    failure says WHICH step broke rather than only that the answer moved.
    """
    step1 = 3.0 / math.sqrt(math.pi) * scipy_gamma(1.0) / scipy_gamma(1.5)
    assert scipy_gamma(1.0) == pytest.approx(1.0, abs=1e-15)
    assert scipy_gamma(1.5) == pytest.approx(math.sqrt(math.pi) / 2.0, rel=1e-15)
    step2 = 3.0 / math.sqrt(math.pi) * 1.0 / (math.sqrt(math.pi) / 2.0)
    closed = 6.0 / math.pi
    assert step1 == pytest.approx(closed, rel=1e-14)
    assert step2 == pytest.approx(closed, rel=1e-14)
    assert co3.C_D_2 == pytest.approx(closed, rel=1e-15)
    assert co3.C_D_2 == pytest.approx(1.9098593171027443, rel=1e-14)
    # The prefactor that actually multiplies the distance.
    assert 2.0 / co3.C_D_2 == pytest.approx(math.pi / 3.0, rel=1e-15)


def test_c_1_is_two():
    """c_1 = 2/sqrt(pi) * Gamma(1/2)/Gamma(1) = 2 exactly (Gamma(1/2) = sqrt(pi))."""
    assert co3.C_D_1 == pytest.approx(2.0, rel=1e-15)


def test_asymptotic_form_reduces_to_exact_eq24_as_overlap_approaches_one():
    """In d = 1 the eq.-25--27 asymptote is the O -> 1 limit of the exact eq. 24.

    Exact (eq. 24):      d = tau_c (1 - O)/sqrt(O)
    Asymptotic (c_1 = 2): d = tau_c (1 - O)
    so the ratio is 1/sqrt(O) -> 1. This validates the *normalisation* of the
    general ``c_d`` against Phase 2a's independently accepted closed form --
    a check on the factor that no d = 2 test alone could provide.
    """
    tau_c = 4.0
    print("\n[c_d normalisation check, d=1]   O      exact      asympt     ratio")
    prev = None
    for o in (0.5, 0.9, 0.99, 0.999, 0.9999):
        exact = co.distance_from_overlap(o, tau_c)
        asym = co3.distance_from_overlap_asymptotic(o, tau_c, d=1)
        ratio = exact / asym
        print(f"                             {o:<8.4f} {exact:.6f}  {asym:.6f}  {ratio:.8f}")
        assert ratio == pytest.approx(1.0 / math.sqrt(o), rel=1e-12)
        if prev is not None:
            assert abs(ratio - 1.0) < abs(prev - 1.0)  # converging to 1
        prev = ratio
    assert prev == pytest.approx(1.0, abs=1e-4)


def test_spacetime_dim_from_spatial():
    """The single conversion point between B--K's d and Rideout--Wallden's d."""
    assert co3.spacetime_dim_from_spatial(1) == 2
    assert co3.spacetime_dim_from_spatial(2) == order3d.SPACETIME_DIM_2P1 == 3
    with pytest.raises(ValueError):
        co3.spacetime_dim_from_spatial(0)


def test_alpha_d_from_m_reproduces_phase2a_alpha_1():
    """alpha_1 = 1/(m_2 eta(2)^{1/2}) with m_2 = 2 is Phase 2a's ALPHA_1.

    The two spellings differ by at most one ulp in IEEE-754 (the module docstring
    records this); that is why the d = 1 path keeps the Phase 2a literal, which
    in turn is what lets the regression test below demand bit identity.
    """
    general = co3.alpha_d_from_m(2.0, d=1)
    assert general == pytest.approx(1.0 / math.sqrt(2.0), rel=1e-15)
    assert abs(general - co.ALPHA_1) <= np.spacing(co.ALPHA_1)
    # And the identity the Phase 2a docstring states: alpha_1 = sqrt(2)/m_2.
    assert general == pytest.approx(math.sqrt(2.0) / 2.0, rel=1e-15)


def test_alpha_2_values_and_direction():
    """alpha_2 from the measured m_3^eff exceeds the one from the published 2.296.

    alpha_2 = 1/(m_3 eta(3)^{1/3}) falls as m_3 rises, and the measured
    m_3^eff (1.76-2.15 over the measured range) is everywhere below the fitted
    asymptote 2.296 -- so calibrating on the measurement gives a LARGER alpha_2
    and therefore a larger tau_c than the published asymptote would. Recorded as
    a test so the sign of that choice cannot flip unnoticed.
    """
    eta3 = sprinkle3d.interval_volume_constant(3)
    assert co3.ALPHA_2_FROM_PUBLISHED_M3 == pytest.approx(1.0 / (2.296 * eta3 ** (1 / 3)), rel=1e-14)
    assert co3.ALPHA_2_FROM_PUBLISHED_M3 == pytest.approx(0.6808298, rel=1e-6)
    assert co3.ALPHA_2_AT_RHO_V_64 > co3.ALPHA_2_FROM_PUBLISHED_M3
    print(
        f"\n[alpha_2] measured m_3^eff(rhoV=64) -> {co3.ALPHA_2_AT_RHO_V_64:.6f} ; "
        f"published m_3=2.296 -> {co3.ALPHA_2_FROM_PUBLISHED_M3:.6f} "
        f"(ratio {co3.ALPHA_2_AT_RHO_V_64 / co3.ALPHA_2_FROM_PUBLISHED_M3:.4f})"
    )


def test_m3_table_is_re_derived_from_the_committed_data():
    """``M3_EFF_MEASURED`` literals must match the .npz files they cite.

    Integrity Rule 4: a constant that has drifted from its source is worse than
    no constant. The low-rho V block comes from the Part-A diagnostic file
    verbatim; the high block is re-reduced here from Part 1's raw chain lengths
    with the link convention ``L = elements - 1`` and the *expected* count 2^K as
    the normaliser (Rideout--Wallden eq. (1) is written with rho V, not the
    realised Poisson count).
    """
    tbl = co3.M3_EFF_MEASURED
    assert tbl.shape == (14, 3)
    assert np.all(np.diff(tbl[:, 0]) > 0), "table must be sorted by rho V"

    z3b = np.load(DATA / "exp03b_diagnostic.npz", allow_pickle=True)
    lo = tbl[:6]
    assert np.array_equal(lo[:, 0], z3b["base_rho_v"])
    # Full-precision literals, so the provenance check can be essentially exact
    # rather than a rounding-tolerant approximation of one.
    assert np.allclose(lo[:, 1], z3b["base_m_eff"], rtol=0, atol=1e-12)
    assert np.allclose(lo[:, 2], z3b["base_m_eff_se"], rtol=0, atol=1e-12)

    z02 = np.load(DATA / "exp02_measurements.npz", allow_pickle=True)
    for row, k in zip(tbl[6:], range(10, 18)):
        rho_v = float(2**k)
        links = z02[f"elements_{k}"] - 1.0  # link convention (Part 1 Finding 3)
        m_eff = links / rho_v ** (1.0 / 3.0)
        assert row[0] == rho_v
        assert row[1] == pytest.approx(float(m_eff.mean()), abs=1e-12)
        assert row[2] == pytest.approx(
            float(m_eff.std(ddof=1) / math.sqrt(m_eff.size)), abs=1e-12
        )

    # It is NOT the published asymptote, and sits below it everywhere.
    assert np.all(tbl[:, 1] < 2.296)


def test_m3_effective_interpolation_and_status_flags():
    """Every read reports whether it was measured, interpolated or extrapolated."""
    grid = co3.M3_EFF_MEASURED
    # Exactly on a measured node: value reproduced, flagged measured.
    for rho_v, m, se in grid:
        read = co3.m3_effective(float(rho_v))
        assert read.value == pytest.approx(m, rel=1e-15)
        assert read.sem == pytest.approx(se, rel=1e-15)
        assert read.status == "measured", (rho_v, read.status)
        assert read.is_supported

    assert co3.m3_effective(37.0).status == "measured"
    assert co3.m3_effective(5000.0).status == "measured"
    # The genuinely unmeasured four-octave gap is flagged, not smoothed over.
    gap = co3.m3_effective(300.0)
    assert gap.status == "interpolated_gap"
    assert not gap.is_supported
    assert co3.M3_EFF_GAP_LO < 300.0 < co3.M3_EFF_GAP_HI
    # The nodes bordering the gap are measurements, not gap interior.
    assert co3.m3_effective(co3.M3_EFF_GAP_LO).status == "measured"
    assert co3.m3_effective(co3.M3_EFF_GAP_HI).status == "measured"
    # Outside the measured range: clamped AND flagged.
    lo = co3.m3_effective(5.0)
    assert lo.status == "extrapolated_low"
    assert lo.value == pytest.approx(float(grid[0, 1]), rel=1e-15)
    hi = co3.m3_effective(1e6)
    assert hi.status == "extrapolated_high"
    assert hi.value == pytest.approx(float(grid[-1, 1]), rel=1e-15)

    # Monotone increasing in rho V across the whole range (the property the
    # documented clamping direction relies on).
    xs = np.geomspace(16.0, 131072.0, 200)
    vals = np.array([co3.m3_effective(float(x)).value for x in xs])
    assert np.all(np.diff(vals) >= -1e-15)

    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            co3.m3_effective(bad)


def test_alpha_2_from_interval_size_tracks_the_curve():
    """Larger intervals -> larger m_3^eff -> smaller alpha_2, with status carried."""
    a_small, read_small = co3.alpha_2_from_interval_size(20.0)
    a_large, read_large = co3.alpha_2_from_interval_size(20000.0)
    assert read_small.status == "measured"
    assert read_large.status == "measured"
    assert a_small > a_large
    assert a_small == pytest.approx(co3.alpha_d_from_m(read_small.value, d=2), rel=1e-15)
    # An empty interval clamps and says so rather than dividing by zero.
    _, read_zero = co3.alpha_2_from_interval_size(0.0)
    assert read_zero.status == "extrapolated_low"


# ---------------------------------------------------------------------------
# 2. Hand-built 2+1 D causets with known A/B/C partitions.
# ---------------------------------------------------------------------------

# Nine events in M^3, coordinates (t, x, y), signature (-,+,+).
#
#   idx 0  c    = (0.0,  0.0, 0.0)   the common past event
#   idx 1  a    = (3.0,  1.0, 0.0)   target a
#   idx 2  b    = (3.0, -1.0, 0.0)   target b        (a, b spacelike: dt=0, dx=2)
#   idx 3  p1   = (1.0,  0.0, 0.0)   below BOTH  -> region C
#   idx 4  p2   = (1.5,  0.7, 0.0)   below a only -> region A
#   idx 5  p3   = (1.5, -0.7, 0.0)   below b only -> region B
#   idx 6  p4   = (1.5,  0.0, 0.7)   below BOTH  -> region C   (off the y=0 plane)
#   idx 7  p5   = (1.5,  0.0, 1.2)   below NEITHER            (off the y=0 plane)
#   idx 8  p0   = (0.5,  0.0, 0.0)   below p1 and everything above it
#
# p5 is the load-bearing element: with the y coordinate dropped it would read as
# (1.5, 0.0) and land strictly below both a and b, i.e. in region C. Its correct
# 2+1 D classification (in neither interval) is what a test of genuinely
# 2+1 D code must assert -- 1+1 D machinery would get it wrong.
_T3 = [0.0, 3.0, 3.0, 1.0, 1.5, 1.5, 1.5, 1.5, 0.5]
_X3 = [0.0, 1.0, -1.0, 0.0, 0.7, -0.7, 0.0, 0.0, 0.0]
_Y3 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 1.2, 0.0]
C_HAND_3D = order3d.causal_matrix_3d(np.array(_T3), np.array(_X3), np.array(_Y3))
IC, IA, IB, IP1, IP2, IP3, IP4, IP5, IP0 = range(9)


def test_hand_causet_relations_individually():
    """Assert each relation the later tests depend on, one at a time.

    Written out rather than inferred so that a failure names the broken relation
    instead of only the downstream overlap value.
    """
    cm = C_HAND_3D
    # a and b are spacelike -- the premise of the whole construction.
    assert not cm[IA, IB] and not cm[IB, IA]
    # c is a common past of a and b.
    assert cm[IC, IA] and cm[IC, IB]
    # p1 and p4 are below BOTH targets (region C).
    for p in (IP1, IP4):
        assert cm[IC, p] and cm[p, IA] and cm[p, IB], p
    # p2 is below a only; p3 below b only.
    assert cm[IC, IP2] and cm[IP2, IA] and not cm[IP2, IB]
    assert cm[IC, IP3] and cm[IP3, IB] and not cm[IP3, IA]
    # p5: in c's future but below NEITHER target (this is the y-dimension test).
    assert cm[IC, IP5]
    assert not cm[IP5, IA] and not cm[IP5, IB]
    # p0 sits below p1 (needed for the timelike O = 1 case).
    assert cm[IC, IP0] and cm[IP0, IP1] and cm[IP0, IA] and cm[IP0, IB]


def test_dropping_the_y_coordinate_would_misclassify_p5():
    """Negative control: 1+1 D machinery on the projected coords gets p5 wrong.

    Confirms the previous test's p5 assertion is testing the third dimension and
    not an accident of the t, x values -- projecting y away makes p5 precede both
    targets, i.e. moves it into region C.
    """
    t = np.array(_T3)
    x = np.array(_X3)
    projected = order.causal_matrix_1d(t + x, t - x)
    assert projected[IP5, IA] and projected[IP5, IB]  # wrong, by construction
    assert not (C_HAND_3D[IP5, IA] or C_HAND_3D[IP5, IB])  # right


def test_alexandrov_interval_contents_3d():
    """I(a,c) = {p0,p1,p2,p4}, I(b,c) = {p0,p1,p3,p4}, worked out by hand."""
    i_ac = set(co3.alexandrov_interval(IA, IC, C_HAND_3D).tolist())
    i_bc = set(co3.alexandrov_interval(IB, IC, C_HAND_3D).tolist())
    assert i_ac == {IP0, IP1, IP2, IP4}, i_ac
    assert i_bc == {IP0, IP1, IP3, IP4}, i_bc
    assert IP5 not in i_ac and IP5 not in i_bc


def test_overlap_partition_is_a_partition_3d():
    """A, B, C are pairwise disjoint and cover I(a,c) u I(b,c) exactly."""
    part = co3.overlap_partition(IA, IB, IC, C_HAND_3D)
    A, B, Cs = set(part.A.tolist()), set(part.B.tolist()), set(part.C_shared.tolist())
    assert A & B == set() and A & Cs == set() and B & Cs == set()
    i_ac = set(co3.alexandrov_interval(IA, IC, C_HAND_3D).tolist())
    i_bc = set(co3.alexandrov_interval(IB, IC, C_HAND_3D).tolist())
    assert A | B | Cs == i_ac | i_bc
    assert A == {IP2} and B == {IP3} and Cs == {IP0, IP1, IP4}
    assert (part.n_A, part.n_B, part.n_C) == (1, 1, 3)
    assert (part.n_ac, part.n_bc) == (4, 4)


def test_overlap_value_hand_built_3d():
    """O = N[C]/(min(N[A],N[B]) + N[C]) = 3/(1+3) = 0.75 for the hand causet."""
    o = co3.causal_overlap(IA, IB, IC, C_HAND_3D)
    assert o == pytest.approx(0.75, rel=1e-15)
    assert 0.0 <= o <= 1.0


def test_overlap_symmetric_in_pair_3d():
    """The ``min`` form of eq. 16/28 is manifestly symmetric in (a, b)."""
    assert co3.causal_overlap(IA, IB, IC, C_HAND_3D) == co3.causal_overlap(
        IB, IA, IC, C_HAND_3D
    )


def test_overlap_is_one_for_timelike_pair_3d():
    """O = 1 for a comparable (timelike) pair: one interval nests in the other.

    Pair (p1, a) with p1 prec a, seen from the common past c: I(p1,c) = {p0} is
    contained in I(a,c) = {p0,p1,p2,p4}, so N[A] = 0, min(N[A],N[B]) = 0 and
    O = N[C]/N[C] = 1.
    """
    assert C_HAND_3D[IP1, IA]  # they really are timelike-related
    part = co3.overlap_partition(IP1, IA, IC, C_HAND_3D)
    assert part.n_A == 0 and part.n_C == 1
    assert co3.causal_overlap(IP1, IA, IC, C_HAND_3D) == pytest.approx(1.0, rel=1e-15)


def test_null_pairs_are_unrelated_under_the_strict_3d_convention():
    """DOCUMENTED DIVERGENCE: null pairs are not comparable in ``order3d``.

    Boguna--Krioukov's ``O = 1`` statement covers timelike *and null* pairs.
    ``order3d.causal_matrix_3d`` uses the STRICT convention (``dt^2 - dx^2 - dy^2
    > 0``), under which an exactly null-separated pair is unrelated -- deliberate,
    documented in ``order3d``'s docstring, and the convention of Rideout--Wallden.
    So in this codebase a null pair yields ``nan`` (no usable common ancestor
    relation), not 1. For Poisson sprinklings the exactly-null set has measure
    zero, so no statistical result can distinguish the conventions; it shows up
    only in hand-built configurations like this one, and is asserted rather than
    left as a surprise.
    """
    t = np.array([0.0, 1.0, 2.0])
    x = np.array([0.0, 1.0, 0.0])  # event 1 is exactly null-separated from 0
    y = np.zeros(3)
    cm = order3d.causal_matrix_3d(t, x, y)
    assert not cm[0, 1] and not cm[1, 0]
    # Whereas Phase 2a's inclusive 1+1 D convention DOES relate them:
    assert order.causal_matrix_1d(t + x, t - x)[0, 1]


def test_overlap_in_unit_interval_on_3d_sprinklings():
    """O in [0, 1] (or nan) for every admissible c on real M^3 sprinklings."""
    checked = 0
    for seed in range(20260901, 20260906):
        cm, ia, ib, _ = _sprinkle_3d_fixed_pair(rho=60.0, sep=1.0, seed=seed)
        for c in co3.common_past(ia, ib, cm).tolist():
            o = co3.causal_overlap(ia, ib, c, cm)
            assert np.isnan(o) or 0.0 <= o <= 1.0, (seed, c, o)
            checked += 1
    assert checked > 50, checked
    print(f"\n[overlap range, M^3] {checked} common-past events checked, all in [0,1] or nan")


# ---------------------------------------------------------------------------
# 3. REGRESSION: d = 1 input must reproduce Phase 2a bit for bit.
# ---------------------------------------------------------------------------


def _sprinkle_1p1d_fixed_pair(rho, t_ab, s, seed):
    """Phase 2a's test geometry: diamond sprinkling with c, a, b injected.

    Deliberately the same construction as ``tests/test_causal_overlap.py`` so the
    regression runs on the kind of input Phase 2a was accepted on.
    """
    sp = sprinkle.sprinkle_diamond_1d(rho, 2.0 * t_ab, seed=seed, include_endpoints=False)
    t = np.concatenate([sp.t, [0.0, t_ab, t_ab]])
    x = np.concatenate([sp.x, [0.0, +0.5 * s, -0.5 * s]])
    cm = order.causal_matrix_1d(t + x, t - x)
    n = t.shape[0]
    return cm, n - 3, n - 2, n - 1


def test_1p1d_regression_pipeline_is_bit_identical_to_phase2a():
    """``distance_causal_overlap_nd(d=1)`` == ``distance_causal_overlap`` exactly.

    Exact equality of every returned array, not ``approx``. This is the test that
    licenses leaving ``causal_overlap.py`` untouched: the generalised pipeline is
    a strict superset of the Phase 2a one rather than a rewrite of it.
    """
    n_compared = 0
    for seed in (4200, 4201, 4202, 4203, 4204, 4205, 4206, 4207):
        cm, ic, ia, ib = _sprinkle_1p1d_fixed_pair(rho=400.0, t_ab=3.0, s=1.0, seed=seed)
        old = co.distance_causal_overlap(ia, ib, cm, 400.0)
        new = co3.distance_causal_overlap_nd(ia, ib, cm, 400.0, d=1)

        assert new.d_spatial == 1
        assert new.distance_formula == "exact_eq24"
        assert new.alpha_policy in ("fixed", "")
        assert new.n_c == old.n_c
        assert new.n_candidates == old.n_candidates
        if old.n_c == 0:
            assert np.isnan(new.distance) and np.isnan(old.distance)
            continue
        assert new.distance == old.distance, (seed, new.distance, old.distance)
        assert (new.sem == old.sem) or (np.isnan(new.sem) and np.isnan(old.sem))
        assert np.array_equal(new.per_c_distance, old.per_c_distance)
        assert np.array_equal(new.per_c_overlap, old.per_c_overlap)
        assert np.array_equal(new.per_c_tau, old.per_c_tau)
        n_compared += 1
    assert n_compared >= 6, n_compared
    print(f"\n[1+1 D regression] {n_compared} sprinklings reproduced bit-for-bit")


def test_1p1d_regression_of_the_component_functions():
    """The dispatching helpers reduce to the Phase 2a ones at d = 1, exactly."""
    for o in (0.05, 0.3, 0.5, 0.8, 0.999):
        for tau in (0.5, 2.0, 7.25):
            assert co3.distance_from_overlap_nd(o, tau, d=1) == co.distance_from_overlap(o, tau)

    cm, ic, ia, ib = _sprinkle_1p1d_fixed_pair(rho=400.0, t_ab=3.0, s=1.0, seed=4200)
    n_checked = 0
    for c in co3.common_past(ia, ib, cm).tolist()[:40]:
        old_tau = co.estimate_tau_c(ia, ib, c, cm, 400.0, d=1)
        new_tau, diag = co3.estimate_tau_c_nd(ia, ib, c, cm, 400.0, d=1)
        assert new_tau == old_tau, (c, new_tau, old_tau)
        assert diag["policy"] == "fixed"
        assert diag["alpha_a"] == diag["alpha_b"] == co.ALPHA_1
        n_checked += 1
    assert n_checked >= 10, n_checked


def test_measured_m3_policy_is_refused_in_1p1d():
    """The d = 2 calibration must not be silently applied in d = 1.

    ``m_2 = 2`` is exact (Brightwell--Gregory, validated in Phase 1); there is no
    m_3^eff curve to read and using one would replace an exact constant with an
    interpolation of unrelated data.
    """
    cm, ic, ia, ib = _sprinkle_1p1d_fixed_pair(rho=400.0, t_ab=3.0, s=1.0, seed=4200)
    c = int(co3.common_past(ia, ib, cm)[0])
    with pytest.raises(ValueError, match="d >= 2 calibration"):
        co3.estimate_tau_c_nd(ia, ib, c, cm, 400.0, d=1, alpha_policy="measured_m3")


# ---------------------------------------------------------------------------
# 4. The 2+1 D pipeline on real sprinklings.
# ---------------------------------------------------------------------------


def _sprinkle_3d_fixed_pair(rho, sep, seed, extent=(4.0, 2.5, 5.0)):
    """Box sprinkling of M^3 with a spacelike pair injected at separation ``sep``.

    Same geometry family as Part A's Gate-A experiment (box ``extent``, density
    ``rho``), with the pair placed at the mid-time on the x axis:
    ``x_+- = (T/2, +-sep/2, 0)``. Returns ``(causal_matrix, i_a, i_b, n)``.
    """
    sp = sprinkle3d.sprinkle_box_3d(rho, *extent, seed=seed)
    t_mid = 0.5 * extent[0]
    t = np.concatenate([sp.t, [t_mid, t_mid]])
    x = np.concatenate([sp.x, [+0.5 * sep, -0.5 * sep]])
    y = np.concatenate([sp.y, [0.0, 0.0]])
    cm = order3d.causal_matrix_3d(t, x, y)
    n = t.shape[0]
    return cm, n - 2, n - 1, n


def test_3d_pipeline_runs_and_records_the_asymptotic_regime():
    """The d = 2 pipeline returns a finite estimate and logs its own validity data.

    No accuracy assertion is made here -- the 2+1 D distance is an ASYMPTOTIC
    estimate (eqs. 25--27), valid only for ``tau_c >> separation``, and how far
    into that regime a given sprinkling sits is a measured property to report,
    not a tolerance to assert. What is asserted is that the diagnostics needed to
    audit the claim are present and self-consistent.
    """
    ratios = []
    ests = []
    print("\n[M^3 B-K pipeline]  seed     n_c  n_cand    d_est   min(tau/d)  med(tau/d)")
    for seed in range(20260910, 20260918):
        cm, ia, ib, _ = _sprinkle_3d_fixed_pair(rho=60.0, sep=1.0, seed=seed)
        res = co3.distance_causal_overlap_nd(ia, ib, cm, 60.0, d=2)
        assert res.d_spatial == 2
        assert res.distance_formula == "asymptotic_eq25_27"
        if res.n_c == 0:
            print(f"                  {seed}     0  {res.n_candidates:6d}   (no admissible c)")
            continue
        assert res.alpha_policy == "measured_m3"
        assert res.per_c_distance.size == res.n_c
        assert res.per_c_asymptotic_ratio.size == res.n_c
        assert res.per_c_interval_size.size == res.n_c
        assert res.per_c_alpha.size == res.n_c
        assert np.all(res.per_c_distance > 0)
        assert np.all(np.isfinite(res.per_c_asymptotic_ratio))
        # The recorded ratio must be exactly tau_c / d_est per c.
        assert np.allclose(
            res.per_c_asymptotic_ratio, res.per_c_tau / res.per_c_distance, rtol=0, atol=0
        )
        # The m_3^eff read statuses are tallied, two per admissible c.
        assert sum(res.m3_status_counts.values()) == 2 * res.n_c
        ests.append(res.distance)
        ratios.append(res.min_asymptotic_ratio)
        print(
            f"                  {seed}  {res.n_c:4d}  {res.n_candidates:6d}  "
            f"{res.distance:7.4f}   {res.min_asymptotic_ratio:8.3f}   "
            f"{res.median_asymptotic_ratio:8.3f}"
        )
    assert len(ests) >= 5, len(ests)
    assert np.all(np.isfinite(ests))
    print(
        f"                  -> {len(ests)} estimates, mean {np.mean(ests):.4f}; "
        f"worst tau_c/d over all pairs {min(ratios):.3f} "
        "(asymptotic form wants this >> 1)"
    )


def test_fixed_alpha_is_a_pure_scale_factor():
    """Doubling a FIXED alpha_d scales every per-c distance by exactly 2.

    This is the property the Part-B rank-ordering gate rests on: a fixed
    calibration constant cannot reorder anything. Asserted, not assumed, because
    the whole head-to-head is specified on ordering precisely because scale is
    not trustworthy (RESULTS.md 2026-08-17, "Consequence for Part B").
    """
    cm, ia, ib, _ = _sprinkle_3d_fixed_pair(rho=60.0, sep=1.0, seed=20260910)
    base = co3.distance_causal_overlap_nd(ia, ib, cm, 60.0, d=2, alpha_d=0.8)
    doubled = co3.distance_causal_overlap_nd(ia, ib, cm, 60.0, d=2, alpha_d=1.6)
    assert base.n_c == doubled.n_c and base.n_c > 0
    assert base.alpha_policy == "explicit"
    assert np.allclose(doubled.per_c_distance, 2.0 * base.per_c_distance, rtol=1e-14, atol=0)
    assert np.allclose(doubled.per_c_overlap, base.per_c_overlap, rtol=0, atol=0)
    assert doubled.distance == pytest.approx(2.0 * base.distance, rel=1e-14)


def test_measured_m3_policy_can_differ_from_fixed_alpha():
    """The size-dependent calibration is genuinely pair-dependent, not cosmetic.

    If the two policies always agreed there would be nothing for the experiment
    to check; this pins that they do not, so the experiment's report on whether
    the choice moves a RANK is a real question rather than a formality.
    """
    cm, ia, ib, _ = _sprinkle_3d_fixed_pair(rho=60.0, sep=1.0, seed=20260910)
    fixed = co3.distance_causal_overlap_nd(ia, ib, cm, 60.0, d=2, alpha_policy="fixed")
    measured = co3.distance_causal_overlap_nd(ia, ib, cm, 60.0, d=2, alpha_policy="measured_m3")
    assert fixed.n_c == measured.n_c > 0
    assert fixed.alpha_policy == "fixed"
    assert measured.alpha_policy == "measured_m3"
    assert np.allclose(fixed.per_c_alpha, co3.ALPHA_2_AT_RHO_V_64, rtol=0, atol=0)
    # Per-c alphas vary under the measured policy (different interval sizes).
    assert measured.per_c_alpha.std() > 0.0
    ratio = measured.per_c_distance / fixed.per_c_distance
    print(
        f"\n[alpha policy] per-c distance ratio measured/fixed: "
        f"min {ratio.min():.4f}  max {ratio.max():.4f}  (n_c={fixed.n_c})"
    )
    assert not np.allclose(ratio, ratio[0], rtol=0, atol=0)


def test_empty_result_is_reported_not_faked():
    """A pair with no admissible c returns nan with the counts intact.

    Rule 3: an unusable measurement is reported as unusable, never silently
    replaced by a number.
    """
    # Two elements at the same time, far apart, in a tiny sprinkling: almost no
    # common past, and Filter 2 rejects what little there is.
    cm, ia, ib, _ = _sprinkle_3d_fixed_pair(rho=2.0, sep=2.0, seed=777, extent=(1.0, 3.0, 1.0))
    res = co3.distance_causal_overlap_nd(ia, ib, cm, 2.0, d=2)
    if res.n_c == 0:
        assert np.isnan(res.distance) and np.isnan(res.sem)
        assert res.per_c_distance.size == 0
        assert res.d_spatial == 2 and res.distance_formula == "asymptotic_eq25_27"
        assert np.isnan(res.min_asymptotic_ratio)
    else:  # if the sprinkling did supply vantage points, the result must be sane
        assert np.isfinite(res.distance) and res.distance > 0
