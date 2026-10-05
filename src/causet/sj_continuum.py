"""Continuum SJ reference quantities for the massless field in the 2d causal diamond.

Comparison targets for Phase 3a, taken from Afshordi, Buck, Dowker, Rideout,
Sorkin, Yazdi, JHEP 10 (2012) 088, arXiv:1207.7101 (ABDRSY), Sec. 4.

Spectrum of iDelta (ABDRSY eqs. (SJfunctions1)-(SJfunctions2))
--------------------------------------------------------------
In lightcone coordinates ``u = (t+x)/sqrt2``, ``v = (t-x)/sqrt2`` with the diamond
``u, v in (-L, L)`` (volume ``V = 4 L^2``), the positive eigenfunctions are

    f_k = e^{-iku} - e^{-ikv},                 k = n pi / L,  n = 1, 2, ...
    g_k = e^{-iku} + e^{-ikv} - 2 cos(kL),     tan(kL) = 2 kL,  k > 0,

with eigenvalues ``lambda_k = L / k``. Writing ``x = kL`` the positive spectrum is
``L^2 / x`` over the union ``X = {n pi} u {x > 0 : tan x = 2x}``. Completeness is
checked by ABDRSY eq. (46): ``sum lambda^2`` over all (+/-) eigenvalues equals the
Hilbert--Schmidt norm ``2 L^4``, i.e. ``sum_n 1/(n pi)^2 = 1/6`` and
``sum_{g} 1/x^2 = 5/6``.

Mapping to the Phase-1 sprinkling (``sprinkle.sprinkle_diamond_1d``)
--------------------------------------------------------------------
The substrate uses ``u' = t + x``, ``v' = t - x`` in ``[0, tau]^2``, so
``u = (u' - tau/2)/sqrt2`` and ``L = tau / (2 sqrt2)``; ``V = tau^2/2 = 4 L^2``.
A causet matrix eigenvalue is ``rho`` times a continuum one (Sorkin--Yazdi
1611.10281 Sec. 3), so with ``rho = N / V`` the predicted causet spectrum is

    lambda_m^pred = rho L^2 / x_m = N / (4 x_m) ,

independent of ``tau`` (the sprinkled 2d diamond is scale-free: its causet is
the random 2-order on N elements).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

#: Exact value of sum over the g-family of 1/x^2 (ABDRSY eq. (46): 10 L^4 / 6
#: = 2 L^4 * sum 1/x_g^2). Used to validate the root finder.
G_FAMILY_INV_SQ_SUM = 5.0 / 6.0


def tan_2x_roots(n_roots: int) -> np.ndarray:
    """First ``n_roots`` positive roots of ``tan x = 2x``, ascending.

    The n-th root lies in ``((n-1) pi, (n - 1/2) pi)`` and approaches the upper
    end (ABDRSY Fig. 3). Substituting ``x = (n - 1/2) pi - y`` turns the equation
    into ``cos y - 2 x sin y = 0`` for ``y in (0, pi/2)``, which has no poles, so
    the bracketing is numerically safe for any ``n``.
    """
    if n_roots < 0:
        raise ValueError("n_roots must be >= 0")
    out = np.empty(n_roots)
    for i in range(n_roots):
        n = i + 1
        c = (n - 0.5) * np.pi

        def h(y, c=c):
            return np.cos(y) - 2.0 * (c - y) * np.sin(y)

        # n = 1: y = pi/2 is the trivial root x = 0; stop just short of it.
        hi = np.pi / 2 - (1e-6 if n == 1 else 0.0)
        out[i] = c - brentq(h, 1e-15, hi, xtol=1e-15, rtol=4 * np.finfo(float).eps)
    return out


def diamond_spectrum_x(n_values: int) -> np.ndarray:
    """The ``n_values`` smallest elements of ``X = {n pi} u {tan x = 2x}``, ascending.

    Positive continuum eigenvalues are ``L^2 / x`` for ``x`` in this list, so it
    is the spectrum in DESCENDING eigenvalue order. Roughly half of each family
    is needed; both are over-generated and merged.
    """
    if n_values <= 0:
        return np.empty(0)
    m = n_values // 2 + 2
    x = np.sort(np.concatenate([np.pi * np.arange(1, m + 1), tan_2x_roots(m)]))
    return x[:n_values]


def predicted_causet_spectrum(n_elements: int, n_values: int) -> np.ndarray:
    """Predicted positive causet spectrum ``N / (4 x_m)``, descending, m = 1..n_values."""
    return n_elements / (4.0 * diamond_spectrum_x(n_values))
