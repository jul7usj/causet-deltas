"""Tests for 2+1 D Poisson sprinkling (``causet.sprinkle3d``).

Three independent things are checked, deliberately not sharing an oracle with
the implementation:

1. the volume constants, against Rideout--Wallden eq. (2) *and* against the
   already-validated Phase-1 1+1 D value ``eta(2) = 1/2``;
2. Poisson *counts*, anchored on the box (whose volume needs no derivation) and
   then on the diamond (whose volume does);
3. *uniformity* of the bicone sampler, via the closed-form marginal CDFs derived
   in the module docstring plus a sub-box count that the sampler's own
   derivation is not involved in.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from causet import sprinkle, sprinkle3d

SEED = 20260810


# ---------------------------------------------------------------- volumes ----
def test_unit_ball_volumes():
    assert sprinkle3d.unit_ball_volume(1) == pytest.approx(2.0)
    assert sprinkle3d.unit_ball_volume(2) == pytest.approx(math.pi)
    assert sprinkle3d.unit_ball_volume(3) == pytest.approx(4.0 * math.pi / 3.0)


def test_interval_volume_constant_matches_paper_and_phase1():
    """eta(d) of Rideout--Wallden eq. (2) at d = 2, 3, 4."""
    # d = 2 must reproduce the Phase-1 1+1 D diamond volume V = tau^2 / 2.
    assert sprinkle3d.interval_volume_constant(2) == pytest.approx(0.5)
    tau = 1.7
    assert sprinkle3d.interval_volume_constant(2) * tau**2 == pytest.approx(
        sprinkle.diamond_volume_1d(tau)
    )
    # d = 3: the pi/12 quoted by Rideout--Wallden for M^3.
    assert sprinkle3d.interval_volume_constant(3) == pytest.approx(math.pi / 12.0)
    assert sprinkle3d.DIAMOND_VOLUME_CONSTANT_3D == pytest.approx(math.pi / 12.0)
    # d = 4: the standard 3+1 D causal-diamond volume pi tau^4 / 24.
    assert sprinkle3d.interval_volume_constant(4) == pytest.approx(math.pi / 24.0)


def test_diamond_volume_3d_is_bicone_volume():
    """V = pi T^3/12 equals 2 x (cone of height T/2, base radius T/2)."""
    for tau in (0.5, 1.0, 3.3):
        cone = (1.0 / 3.0) * math.pi * (tau / 2.0) ** 2 * (tau / 2.0)
        assert sprinkle3d.diamond_volume_3d(tau) == pytest.approx(2.0 * cone)


def test_volume_input_validation():
    with pytest.raises(ValueError):
        sprinkle3d.diamond_volume_3d(0.0)
    with pytest.raises(ValueError):
        sprinkle3d.box_volume_3d(1.0, -1.0, 1.0)
    with pytest.raises(ValueError):
        sprinkle3d.sprinkle_diamond_3d(rho=-1.0, tau=1.0, seed=0)
    with pytest.raises(ValueError):
        sprinkle3d.interval_volume_constant(1)


# ----------------------------------------------------------- Poisson stats ---
def test_box_poisson_count_statistics():
    """Anchor test: the box volume needs no derivation, so a count mismatch here
    would indicate a bug in the Poisson machinery itself, not in geometry."""
    rho, ext = 500.0, (1.0, 2.0, 3.0)
    expected = rho * sprinkle3d.box_volume_3d(*ext)
    assert expected == pytest.approx(3000.0)
    counts = np.array(
        [sprinkle3d.sprinkle_box_3d(rho, *ext, seed=SEED + k).n for k in range(60)],
        dtype=float,
    )
    se = counts.std(ddof=1) / math.sqrt(counts.size)
    assert abs(counts.mean() - expected) < 4.0 * se
    # Poisson: variance == mean. Relative sd of the sample variance ~ sqrt(2/K).
    assert counts.var(ddof=1) == pytest.approx(expected, rel=4.0 * math.sqrt(2.0 / 60))


def test_diamond_poisson_count_statistics():
    rho, tau = 4000.0, 1.0
    expected = sprinkle3d.expected_count_diamond_3d(rho, tau)
    assert expected == pytest.approx(4000.0 * math.pi / 12.0)
    counts = np.array(
        [sprinkle3d.sprinkle_diamond_3d(rho, tau, seed=SEED + k).n for k in range(60)],
        dtype=float,
    )
    se = counts.std(ddof=1) / math.sqrt(counts.size)
    assert abs(counts.mean() - expected) < 4.0 * se
    assert counts.var(ddof=1) == pytest.approx(expected, rel=4.0 * math.sqrt(2.0 / 60))


# -------------------------------------------------------------- uniformity ---
def test_diamond_points_all_inside_bicone():
    s = sprinkle3d.sprinkle_diamond_3d(rho=2e4, tau=1.3, seed=SEED)
    assert s.n > 5000
    assert np.all(sprinkle3d.in_diamond_3d(s.t, s.x, s.y, s.tau))


def test_diamond_time_marginal_matches_closed_form_cdf():
    """F_t(t) = 4t^3/T^3 (t <= T/2), 1 - 4(T-t)^3/T^3 (t >= T/2)."""
    tau = 2.0
    s = sprinkle3d.sprinkle_diamond_3d(rho=3e4, tau=tau, seed=SEED + 101)

    def cdf_t(t):
        t = np.asarray(t, dtype=float)
        lower = 4.0 * t**3 / tau**3
        upper = 1.0 - 4.0 * (tau - t) ** 3 / tau**3
        return np.where(t <= 0.5 * tau, lower, upper)

    res = stats.kstest(s.t, cdf_t)
    assert res.pvalue > 1e-3, f"t-marginal KS p={res.pvalue:.3g}, N={s.n}"


def test_diamond_radial_marginal_matches_closed_form_cdf():
    """F_r(r) = 12 r^2/T^2 - 16 r^3/T^3 on [0, T/2]."""
    tau = 2.0
    s = sprinkle3d.sprinkle_diamond_3d(rho=3e4, tau=tau, seed=SEED + 202)
    r = np.hypot(s.x, s.y)

    def cdf_r(rr):
        rr = np.asarray(rr, dtype=float)
        return 12.0 * rr**2 / tau**2 - 16.0 * rr**3 / tau**3

    res = stats.kstest(r, cdf_r)
    assert res.pvalue > 1e-3, f"r-marginal KS p={res.pvalue:.3g}, N={s.n}"


def test_diamond_sub_box_count_is_poisson_with_rho_times_box_volume():
    """Uniformity check that never touches the sampler's own CDFs.

    A cuboid strictly inside the bicone must receive Poisson(rho * V_box) points.
    Sub-box: t in [0.4T, 0.6T], |x| <= 0.05T, |y| <= 0.05T. Its farthest corner
    has r = 0.0707T, well inside R(t) >= 0.4T over that time slab.
    """
    tau, rho = 1.0, 1.0e6
    s = sprinkle3d.sprinkle_diamond_3d(rho=rho, tau=tau, seed=SEED + 303)
    half_w = 0.05 * tau
    inside = (
        (s.t >= 0.4 * tau)
        & (s.t <= 0.6 * tau)
        & (np.abs(s.x) <= half_w)
        & (np.abs(s.y) <= half_w)
    )
    v_box = (0.2 * tau) * (2 * half_w) * (2 * half_w)
    expected = rho * v_box
    observed = int(inside.sum())
    assert expected == pytest.approx(2000.0)
    assert abs(observed - expected) < 5.0 * math.sqrt(expected), (
        f"sub-box count {observed} vs expected {expected}"
    )


def test_diamond_is_isotropic_in_the_spatial_plane():
    """x and y are statistically interchangeable and centred (rotational symmetry)."""
    s = sprinkle3d.sprinkle_diamond_3d(rho=3e5, tau=1.0, seed=SEED + 404)
    n = s.n
    for coord in (s.x, s.y):
        se = coord.std(ddof=1) / math.sqrt(n)
        assert abs(coord.mean()) < 4.0 * se
    assert s.x.var(ddof=1) == pytest.approx(s.y.var(ddof=1), rel=0.05)
    # Angles uniform on [0, 2 pi).
    theta = np.mod(np.arctan2(s.y, s.x), 2.0 * np.pi)
    res = stats.kstest(theta, lambda a: np.asarray(a) / (2.0 * np.pi))
    assert res.pvalue > 1e-3, f"angle KS p={res.pvalue:.3g}"


def test_box_marginals_are_uniform():
    ext = (1.0, 2.0, 3.0)
    s = sprinkle3d.sprinkle_box_3d(1e4, *ext, seed=SEED + 505)
    checks = (
        (s.t, 0.0, ext[0]),
        (s.x, -0.5 * ext[1], 0.5 * ext[1]),
        (s.y, -0.5 * ext[2], 0.5 * ext[2]),
    )
    for vals, lo, hi in checks:
        assert vals.min() >= lo and vals.max() <= hi
        res = stats.kstest(vals, stats.uniform(loc=lo, scale=hi - lo).cdf)
        assert res.pvalue > 1e-3


# ------------------------------------------------------ bookkeeping / repro ---
def test_seed_reproducibility():
    a = sprinkle3d.sprinkle_diamond_3d(1e3, 1.0, seed=777)
    b = sprinkle3d.sprinkle_diamond_3d(1e3, 1.0, seed=777)
    c = sprinkle3d.sprinkle_diamond_3d(1e3, 1.0, seed=778)
    assert np.array_equal(a.t, b.t) and np.array_equal(a.x, b.x) and np.array_equal(a.y, b.y)
    assert not np.array_equal(a.t, c.t)


def test_endpoints_are_placed_at_first_and_last_index():
    tau = 1.5
    s = sprinkle3d.sprinkle_diamond_3d(1e3, tau, seed=SEED + 606, include_endpoints=True)
    i_past, i_future = s.endpoint_indices
    assert (i_past, i_future) == (0, s.n - 1)
    assert (s.t[i_past], s.x[i_past], s.y[i_past]) == (0.0, 0.0, 0.0)
    assert (s.t[i_future], s.x[i_future], s.y[i_future]) == (tau, 0.0, 0.0)
    assert s.n_interior == s.n - 2

    bare = sprinkle3d.sprinkle_diamond_3d(1e3, tau, seed=SEED + 606)
    assert bare.endpoint_indices is None
    assert bare.n_interior == bare.n == s.n - 2
    # Same seed => the interior points are the identical realisation.
    assert np.array_equal(bare.t, s.t[1:-1])


def test_metadata_recorded_for_reproducibility():
    s = sprinkle3d.sprinkle_diamond_3d(123.0, 2.0, seed=42)
    assert s.seed == 42
    assert s.region == "diamond"
    assert s.tau == 2.0
    assert s.volume == pytest.approx(sprinkle3d.diamond_volume_3d(2.0))
    assert s.n_expected == pytest.approx(123.0 * s.volume)
    assert s.coords().shape == (s.n, 3)

    b = sprinkle3d.sprinkle_box_3d(10.0, 1.0, 1.0, 1.0, seed=43)
    assert b.region == "box" and b.extent == (1.0, 1.0, 1.0) and b.tau is None
