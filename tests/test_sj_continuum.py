"""Tests for the continuum SJ diamond reference (src/causet/sj_continuum.py)."""

from __future__ import annotations

import numpy as np

from causet import sj_continuum as sc


def test_roots_satisfy_equation_and_brackets():
    x = sc.tan_2x_roots(2000)
    n = np.arange(1, x.size + 1)
    assert np.all(x > (n - 1) * np.pi) and np.all(x < (n - 0.5) * np.pi)
    # residual of the pole-free form cos y = 2 x sin y, y = (n-1/2)pi - x
    # Recomputing y = c - x here loses ~eps*c absolutely (c ~ 6e3 at n = 2000), and
    # d(residual)/dy ~ -2x, so the attainable residual is ~ 2x * eps * c, not eps.
    c = (n - 0.5) * np.pi
    y = c - x
    bound = 10 * 2 * x * np.finfo(float).eps * c
    assert np.all(np.abs(np.cos(y) - 2 * x * np.sin(y)) < bound)
    assert np.all(np.diff(x) > 0)


def test_first_root_value():
    # tan x = 2x, smallest positive root; independent check by Newton from 1.1.
    z = 1.1
    for _ in range(50):
        z -= (np.tan(z) - 2 * z) / (1 / np.cos(z) ** 2 - 2)
    assert abs(sc.tan_2x_roots(1)[0] - z) < 1e-13
    assert abs(z - 1.16556118520721) < 1e-12


def test_g_family_completeness_abdrsy_eq46():
    """sum 1/x_g^2 = 5/6 (ABDRSY eq. (46)); tail beyond M from x_n ~ (2n-1)pi/2."""
    m = 20000
    x = sc.tan_2x_roots(m)
    n_tail = np.arange(m + 1, 4_000_001)
    tail = np.sum(1.0 / ((2 * n_tail - 1) * np.pi / 2) ** 2)
    assert abs(np.sum(1 / x**2) + tail - sc.G_FAMILY_INV_SQ_SUM) < 1e-7


def test_merged_spectrum_order_and_families():
    x = sc.diamond_spectrum_x(7)
    g = sc.tan_2x_roots(4)
    want = np.sort(np.concatenate([np.pi * np.arange(1, 4), g]))[:7]
    assert np.allclose(x, want)
    # interleaving g, f, g, f, ...: g_1 < pi < g_2 < 2pi < ...
    assert x[0] == g[0] and np.isclose(x[1], np.pi) and x[2] == g[1]


def test_predicted_spectrum_hilbert_schmidt():
    """Sum over all (+/-) predicted eigenvalues squared -> 2 (N/4)^2 = N^2/8."""
    n = 1000
    lam = sc.predicted_causet_spectrum(n, 400_000)
    total = 2 * np.sum(lam**2)
    assert abs(total / (n * n / 8) - 1) < 1e-5
