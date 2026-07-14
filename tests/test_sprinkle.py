"""Poisson statistics and geometry of the 1+1 D sprinkling."""

from __future__ import annotations

import numpy as np
import pytest

from causet import sprinkle


def test_diamond_volume_formula():
    # V(tau) = tau^2 / 2.
    assert sprinkle.diamond_volume_1d(1.0) == pytest.approx(0.5)
    assert sprinkle.diamond_volume_1d(2.0) == pytest.approx(2.0)


def test_expected_count():
    assert sprinkle.expected_count_1d(rho=100.0, tau=2.0) == pytest.approx(200.0)


def test_points_inside_diamond():
    """Every sprinkled point must lie inside the causal diamond of p and q."""
    s = sprinkle.sprinkle_diamond_1d(rho=500.0, tau=3.0, seed=1)
    # Null coords within [0, tau]; equivalently |x| <= t and |x| <= tau - t.
    assert np.all(s.u >= 0.0) and np.all(s.u <= s.tau)
    assert np.all(s.v >= 0.0) and np.all(s.v <= s.tau)
    assert np.all(np.abs(s.x) <= s.t + 1e-12)
    assert np.all(np.abs(s.x) <= s.tau - s.t + 1e-12)


def test_reproducible_seed():
    a = sprinkle.sprinkle_diamond_1d(rho=200.0, tau=2.0, seed=42)
    b = sprinkle.sprinkle_diamond_1d(rho=200.0, tau=2.0, seed=42)
    assert np.array_equal(a.t, b.t) and np.array_equal(a.x, b.x)
    c = sprinkle.sprinkle_diamond_1d(rho=200.0, tau=2.0, seed=43)
    assert not np.array_equal(a.t, c.t)


def test_poisson_count_mean_and_variance():
    """N over realisations must have mean and variance ~ rho*V (Poisson)."""
    rho, tau = 300.0, 2.0
    expected = sprinkle.expected_count_1d(rho, tau)  # = 600
    n_real = 400
    counts = np.array(
        [sprinkle.sprinkle_diamond_1d(rho, tau, seed=s).n for s in range(n_real)]
    )
    mean = counts.mean()
    var = counts.var(ddof=1)
    # Standard error of the mean of a Poisson sample.
    se_mean = np.sqrt(expected / n_real)
    assert abs(mean - expected) < 4.0 * se_mean, (mean, expected, se_mean)
    # Poisson variance == mean; allow a generous band for finite n_real.
    assert 0.8 * expected < var < 1.2 * expected, (var, expected)


def test_uniformity_in_spacetime():
    """Points should be uniform in (u, v): split the square into quadrants."""
    rho, tau = 4000.0, 2.0
    s = sprinkle.sprinkle_diamond_1d(rho, tau, seed=7)
    half = tau / 2.0
    q = [
        np.sum((s.u < half) & (s.v < half)),
        np.sum((s.u >= half) & (s.v < half)),
        np.sum((s.u < half) & (s.v >= half)),
        np.sum((s.u >= half) & (s.v >= half)),
    ]
    expected_per_q = s.n / 4.0
    # chi-square-ish sanity: each quadrant within ~5 sigma of n/4.
    sigma = np.sqrt(expected_per_q)
    for count in q:
        assert abs(count - expected_per_q) < 5.0 * sigma, (q, expected_per_q)


def test_endpoints_option():
    s = sprinkle.sprinkle_diamond_1d(rho=100.0, tau=2.0, seed=3, include_endpoints=True)
    # p=(0,0) is first, q=(tau,0) is last.
    assert s.t[0] == pytest.approx(0.0) and s.x[0] == pytest.approx(0.0)
    assert s.t[-1] == pytest.approx(s.tau) and s.x[-1] == pytest.approx(0.0)
