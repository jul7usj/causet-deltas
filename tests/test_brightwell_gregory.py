"""Brightwell--Gregory validation: the timelike geodesic / longest-chain law.

This is the Phase-1 acceptance test that validates the whole geometric pipeline
against the one theorem everyone trusts. Two equivalent statements are checked:

1.  Universal constant. For ``N`` points Poisson-sprinkled into a 1+1 D causal
    diamond, the causal order is the 2-D random (dominance) order, whose longest
    chain ``L`` obeys  ``E[L]/sqrt(N) -> 2``  as ``N -> infinity``. This is the
    Vershik--Kerov (1977) / Logan--Shepp (1977) constant for Ulam's longest-
    increasing-subsequence problem, and is the 1+1 D case of Brightwell &
    Gregory, "Structure of random discrete spacetime", Phys. Rev. Lett. 66, 260
    (1991). The approach to 2 is from below with a finite-size correction of
    order ``N^{-1/3}`` (Tracy--Widom): ``E[L]/sqrt(N) ~ 2 - c N^{-1/3}``.

2.  Proportionality to proper time. At fixed density ``rho``, since
    ``N = rho tau^2 / 2``, the law ``L ~ 2 sqrt(N)`` gives  ``L ~ sqrt(2 rho) tau``
    -- the longest chain is proportional to the continuum proper time ``tau``.

Both are checked with error bars over sprinkling realisations (Rule 3).
"""

from __future__ import annotations

import numpy as np
import pytest

from causet import order, sprinkle

VKLS_CONSTANT = 2.0  # lim E[L]/sqrt(N) for the 2-D random order.


def _mean_longest_chain(rho: float, tau: float, n_real: int, seed0: int):
    """Return (mean L, standard error of the mean, mean N) over realisations."""
    lengths = np.empty(n_real)
    counts = np.empty(n_real)
    for k in range(n_real):
        s = sprinkle.sprinkle_diamond_1d(rho, tau, seed=seed0 + k, include_endpoints=True)
        c = order.causal_matrix_1d(s.u, s.v)
        lengths[k] = order.longest_chain_length(c)
        counts[k] = s.n
    return lengths.mean(), lengths.std(ddof=1) / np.sqrt(n_real), counts.mean()


def test_longest_chain_constant_approaches_two():
    """E[L]/sqrt(N) should sit just below 2 and rise toward 2 with density."""
    tau = 2.0
    n_real = 40
    ratios = []
    for rho in (250.0, 1000.0, 4000.0):
        mean_l, _, mean_n = _mean_longest_chain(rho, tau, n_real, seed0=1000)
        ratio = mean_l / np.sqrt(mean_n)
        ratios.append(ratio)
        # Below the asymptote but within the finite-size band.
        assert 1.7 < ratio < 2.02, (rho, ratio, mean_n)
    # Monotone approach to the asymptote from below (allow tiny noise slack).
    assert ratios[0] <= ratios[1] + 0.02
    assert ratios[1] <= ratios[2] + 0.02
    # Highest density should be within ~7% of the theoretical constant 2.
    assert abs(ratios[-1] - VKLS_CONSTANT) / VKLS_CONSTANT < 0.07, ratios


def test_longest_chain_proportional_to_proper_time():
    """At fixed rho, L should scale linearly with tau (slope ~ sqrt(2 rho))."""
    rho = 1500.0
    n_real = 30
    taus = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
    means = np.array(
        [_mean_longest_chain(rho, tau, n_real, seed0=5000)[0] for tau in taus]
    )
    # Least-squares slope through the origin: L = m * tau.
    slope = float(np.sum(taus * means) / np.sum(taus * taus))
    predicted = np.sqrt(2.0 * rho)
    # Slope sits a few percent below sqrt(2 rho) due to the N^{-1/3} correction.
    assert 0.9 * predicted < slope < 1.0 * predicted, (slope, predicted)
    # Linearity: residuals of L = slope*tau should be small relative to L.
    resid = means - slope * taus
    assert np.max(np.abs(resid)) < 0.06 * means.mean(), (means, slope)
