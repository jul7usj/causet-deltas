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


# --------------------------------------------------------------------------- #
# Continuum SJ Wightman function W_SJ,L (ABDRSY 1207.7101 Sec. 4)
# --------------------------------------------------------------------------- #
import pytest  # noqa: E402

_RNG = np.random.default_rng(3)
#: Generic, well-separated test points in scaled coordinates (u/L, v/L).
PTS_A = _RNG.uniform(-0.8, 0.8, (6, 2))
PTS_B = _RNG.uniform(-0.8, 0.8, (6, 2))
N_BRUTE = 100_000


@pytest.fixture(scope="module")
def brute_modes():
    n = np.arange(1, N_BRUTE + 1)
    return n, sc.tan_2x_roots(N_BRUTE)


def _f(k, u, v):
    return np.exp(-1j * k * u) - np.exp(-1j * k * v)


def _g(x, u, v):
    return np.exp(-1j * x * u) + np.exp(-1j * x * v) - 2 * np.cos(x)


def test_box_closed_form_matches_mode_sums(brute_modes):
    """w_box = eq. (sum1) + eq. (epsdef) with K -> K_0, summed by brute force.

    This is the test that exposed the sign typo in the printed eq. (SJbox).
    """
    n, _ = brute_modes
    x0 = (n - 0.5) * np.pi
    for (u, v), (u2, v2) in zip(PTS_A, PTS_B):
        s1 = np.sum(_f(n * np.pi, u, v) * np.conj(_f(n * np.pi, u2, v2)) / (8 * np.pi * n))
        g0 = (np.exp(-1j * x0 * u) + np.exp(-1j * x0 * v)) * np.conj(
            np.exp(-1j * x0 * u2) + np.exp(-1j * x0 * v2))
        s2 = np.sum(g0 / (8 * x0))
        assert abs(sc.w_box(u, v, u2, v2) - (s1 + s2)) < 2e-4


def test_w_sj_matches_eq47_mode_sum(brute_modes):
    """w_sj_matrix = ABDRSY eq. (47) summed directly over f and EXACT g modes."""
    n, x = brute_modes
    w = sc.w_sj_matrix(PTS_A[:, 0], PTS_A[:, 1], PTS_B[:, 0], PTS_B[:, 1])
    for i, (u, v) in enumerate(PTS_A):
        u2, v2 = PTS_B[i]
        s1 = np.sum(_f(n * np.pi, u, v) * np.conj(_f(n * np.pi, u2, v2)) / (8 * np.pi * n))
        s2 = np.sum(_g(x, u, v) * np.conj(_g(x, u2, v2)) / (x * (8 - 16 * np.cos(x) ** 2)))
        assert abs(w[i, i] - (s1 + s2)) < 2e-4


def test_im_w_is_half_delta():
    """Im W = Delta/2 (ABDRSY Sec. 4.1): -1/4 if X later than X', +1/4 if earlier."""
    w = sc.w_sj_matrix(PTS_A[:, 0], PTS_A[:, 1], PTS_B[:, 0], PTS_B[:, 1])
    du = PTS_A[:, 0][:, None] - PTS_B[:, 0][None, :]
    dv = PTS_A[:, 1][:, None] - PTS_B[:, 1][None, :]
    want = np.where(du * dv > 0, -0.25 * np.sign(du), 0.0)
    assert np.max(np.abs(w.imag - want)) < 1e-9


def test_w_kernel_hermitian():
    wab = sc.w_sj_matrix(PTS_A[:, 0], PTS_A[:, 1], PTS_B[:, 0], PTS_B[:, 1])
    wba = sc.w_sj_matrix(PTS_B[:, 0], PTS_B[:, 1], PTS_A[:, 0], PTS_A[:, 1])
    assert np.allclose(wab, wba.conj().T, rtol=0, atol=1e-10)


def test_epsilon_centre_matches_published():
    """eps(0,0;0,0) is ABDRSY's eps_centre (L -> infinity at fixed region): -0.063."""
    e = sc.epsilon_matrix([0.0], [0.0])[0, 0]
    assert abs(e.imag) < 1e-12
    assert abs(e.real - sc.EPS_CENTRE_ABDRSY) < 5e-4  # published to 2 sig. fig.


def test_epsilon_truncation_converged():
    a = sc.epsilon_matrix(PTS_A[:, 0], PTS_A[:, 1], n_modes=5000)
    b = sc.epsilon_matrix(PTS_A[:, 0], PTS_A[:, 1], n_modes=sc.EPS_N_MODES)
    assert np.max(np.abs(a - b)) < 1e-5


def test_centre_limit_eq_sjtpcentrecomplete():
    """Small separations at the centre: exact W -> eq. (SJtpcentrecomplete)."""
    eps0 = sc.epsilon_matrix([0.0], [0.0])[0, 0].real
    p = np.array([[0.01, 0.02], [-0.015, 0.005], [0.004, -0.012]])
    w = sc.w_sj_matrix(p[:, 0], p[:, 1])
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            c = sc.w_centre(p[i, 0], p[i, 1], p[j, 0], p[j, 1], l_half=7.3, eps_c=eps0)
            assert abs(w[i, j] - c) < 2e-3  # O(delta/L) corrections, delta ~ 0.03


def test_scaled_coordinates_map_diamond():
    tau = 1.7
    assert np.allclose(sc.scaled_from_sprinkle(np.array([0.0, tau / 2, tau]), tau),
                       [-1.0, 0.0, 1.0])
