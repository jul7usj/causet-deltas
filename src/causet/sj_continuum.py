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


# --------------------------------------------------------------------------- #
# The continuum SJ two-point function W_SJ,L (ABDRSY Sec. 4)
# --------------------------------------------------------------------------- #
# All functions below take SCALED lightcone coordinates ub = u/L, vb = v/L in
# (-1, 1), with ABDRSY's u = (t+x)/sqrt2, v = (t-x)/sqrt2. The massless 2d SJ
# state is scale-free: W depends on the coordinates only through ub, vb.
# From the Phase-1 sprinkling (u' = t+x in [0, tau]):  ub = 2 u'/tau - 1.

#: ABDRSY Sec. 4.1: epsilon(u,v;u',v') -> epsilon_centre ~ -0.063 for a small
#: region at the centre as L -> infinity (2 significant figures). SSY
#: 1311.7146 eq. (Wsimp) quotes mu = (pi/4L) exp(-2 pi eps_c) = 0.0116681 at
#: L = 100; inverting gives eps_c = -0.06300, i.e. SSY used the rounded -0.063
#: and the quote carries no extra precision.
EPS_CENTRE_ABDRSY = -0.063
EPS_CENTRE_FROM_SSY_MU = -np.log(0.0116681 / (np.pi / 400.0)) / (2.0 * np.pi)

#: Number of g-family modes summed explicitly in epsilon (eq. (corr)). The
#: summand difference decays like 1/n^2 (roots approach (2n-1)pi/2 as 1/(2x)),
#: so truncation leaves an O(1/EPS_N_MODES) error, checked in the tests.
EPS_N_MODES = 20000


def scaled_from_sprinkle(u_prime: np.ndarray, tau: float) -> np.ndarray:
    """Map substrate null coordinate ``u' = t + x in [0, tau]`` to ``u/L in [-1, 1]``.

    ``u = (u' - tau/2)/sqrt2`` and ``L = tau/(2 sqrt2)`` give ``u/L = 2u'/tau - 1``.
    """
    return 2.0 * np.asarray(u_prime, dtype=float) / tau - 1.0


def _log1pm_exp(z: np.ndarray, sign: float) -> np.ndarray:
    """Principal ``Log[1 + sign * e^{-i pi z / 2}]`` (ABDRSY: Log = principal value)."""
    return np.log(1.0 + sign * np.exp(-0.5j * np.pi * z))


def w_box(ub, vb, ub2, vb2) -> np.ndarray:
    """``W_box,L`` = S1 + S2(K_0): ABDRSY eqs. (firstsum) + (secondsum), summed.

    (1/4pi){-Log[1-e^{-i pi(u-u')/2L}] - Log[1-e^{-i pi(v-v')/2L}]
            +Log[1+e^{-i pi(u-v')/2L}] + Log[1+e^{-i pi(v-u')/2L}]}.

    DISCREPANCY WITH THE PRINTED eq. (SJbox) (arXiv:1207.7101): it has
    ``Log[1 - e^{...}]`` in the two cross terms. Combining (firstsum) with
    (secondsum) term by term, with w = e^{-i pi a/2}: the cross terms give
    (1/8pi)[Log(1-w^2) + Log(1+w) - Log(1-w)] = (1/4pi) Log(1+w). Only the
    ``1 + w`` form reproduces the paper's own centre limit eq.
    (SJtpcentrecomplete) (the cross terms supply the (1/2pi) ln 2 that turns
    -(1/2pi) ln(pi/2L) into -(1/2pi) ln(pi/4L)), and it agrees with brute-force
    summation of eqs. (sum1) + (epsdef) -- see tests/test_sj_continuum.py.
    Singular on the light cone (u = u' or v = v'); callers exclude null pairs.
    """
    return (
        -_log1pm_exp(ub - ub2, -1.0) - _log1pm_exp(vb - vb2, -1.0)
        + _log1pm_exp(ub - vb2, +1.0) + _log1pm_exp(vb - ub2, +1.0)
    ) / (4.0 * np.pi)


def _g_modes(x: np.ndarray, ub: np.ndarray, vb: np.ndarray) -> np.ndarray:
    """g_k(u, v) = e^{-iku} + e^{-ikv} - 2cos(kL) at kL = x; shape (points, modes)."""
    return (np.exp(-1j * np.outer(ub, x)) + np.exp(-1j * np.outer(vb, x))
            - 2.0 * np.cos(x)[None, :])


def epsilon_matrix(ub, vb, ub2=None, vb2=None, *, n_modes: int = EPS_N_MODES,
                   chunk: int = 2000) -> np.ndarray:
    """ABDRSY eq. (corr): ``epsilon(X_i, X'_j)`` for two point sets, shape (P, Q).

    Per mode, ``(L/k)/||g_k||^2 = 1/(x (8 - 16 cos^2 x))`` with x = kL (the
    ``L^2`` cancels). Exact roots ``x_n`` of tan x = 2x minus ``x_{0,n} =
    (2n-1)pi/2``, for which cos x_0 = 0. Summed in chunks of modes.
    """
    ub = np.atleast_1d(np.asarray(ub, float))
    vb = np.atleast_1d(np.asarray(vb, float))
    ub2 = ub if ub2 is None else np.atleast_1d(np.asarray(ub2, float))
    vb2 = vb if vb2 is None else np.atleast_1d(np.asarray(vb2, float))
    x = tan_2x_roots(n_modes)
    x0 = (np.arange(1, n_modes + 1) - 0.5) * np.pi
    out = np.zeros((ub.size, ub2.size), dtype=complex)
    for s in range(0, n_modes, chunk):
        for xs, sign in ((x[s:s + chunk], 1.0), (x0[s:s + chunk], -1.0)):
            wgt = 1.0 / (xs * (8.0 - 16.0 * np.cos(xs) ** 2))
            a = _g_modes(xs, ub, vb)
            b = _g_modes(xs, ub2, vb2)
            out += sign * (a * wgt) @ b.conj().T
    return out


def w_sj_matrix(ub, vb, ub2=None, vb2=None, **kw) -> np.ndarray:
    """Exact continuum SJ Wightman function ``W_SJ,L = W_box,L + epsilon``.

    ABDRSY eq. (SJbox) with the K -> K_0 correction of eq. (corr) restored,
    i.e. eq. (47) summed exactly. Shape (P, Q); null/coincident pairs are
    singular (log divergence) and must be excluded by the caller.
    """
    ub = np.atleast_1d(np.asarray(ub, float))
    vb = np.atleast_1d(np.asarray(vb, float))
    ub2 = ub if ub2 is None else np.atleast_1d(np.asarray(ub2, float))
    vb2 = vb if vb2 is None else np.atleast_1d(np.asarray(vb2, float))
    with np.errstate(divide="ignore", invalid="ignore"):
        box = w_box(ub[:, None], vb[:, None], ub2[None, :], vb2[None, :])
    return box + epsilon_matrix(ub, vb, ub2, vb2, **kw)


def w_centre(ub, vb, ub2, vb2, l_half: float, eps_c: float = EPS_CENTRE_ABDRSY):
    """ABDRSY eq. (SJtpcentrecomplete), the large-L centre limit, in scaled coords.

    -(1/4pi) ln|du dv| - (i/4) sgn(du+dv) theta(du dv) - (1/2pi) ln(pi/4L) + eps_c,
    with du = L (ub - ub2), etc. Depends on L only through the combination
    (1/4pi) ln L^2 - (1/2pi) ln L = 0, i.e. not at all; ``l_half`` is kept as an
    argument to make that cancellation visible in the code rather than assumed.
    """
    du = l_half * (np.asarray(ub) - ub2)
    dv = l_half * (np.asarray(vb) - vb2)
    return (-np.log(np.abs(du * dv)) / (4 * np.pi)
            - 0.25j * np.sign(du + dv) * (du * dv > 0)
            - np.log(np.pi / (4 * l_half)) / (2 * np.pi) + eps_c)
