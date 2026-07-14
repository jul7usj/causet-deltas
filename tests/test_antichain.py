"""Antichain correctness: antichain property and maximality/inextendibility."""

from __future__ import annotations

import numpy as np

from causet import antichain, order, sprinkle


def test_is_antichain_handbuilt():
    # a prec b prec c on the axis; {a,c} is a chain (not an antichain).
    u = np.array([0.0, 1.0, 2.0])
    v = np.array([0.0, 1.0, 2.0])
    c = order.causal_matrix_1d(u, v)
    assert antichain.is_antichain(c, [0])
    assert not antichain.is_antichain(c, [0, 2])


def test_spacelike_pair_is_antichain():
    u = np.array([1.5, 0.5])
    v = np.array([0.5, 1.5])
    c = order.causal_matrix_1d(u, v)
    assert antichain.is_antichain(c, [0, 1])


def test_extracted_antichain_is_valid_antichain():
    for seed in range(6):
        s = sprinkle.sprinkle_diamond_1d(rho=400.0, tau=2.0, seed=seed)
        c = order.causal_matrix_1d(s.u, s.v)
        a = antichain.maximal_antichain_near_midslice(c, s.t)
        assert a.size > 0
        assert antichain.is_antichain(c, a)


def test_extracted_antichain_is_maximal():
    """The greedy construction must be inextendible for every seed."""
    for seed in range(6):
        s = sprinkle.sprinkle_diamond_1d(rho=400.0, tau=2.0, seed=seed)
        c = order.causal_matrix_1d(s.u, s.v)
        a = antichain.maximal_antichain_near_midslice(c, s.t)
        assert antichain.is_maximal_antichain(c, a), seed


def test_non_maximal_antichain_detected():
    """A single spacelike element in a larger causet is an antichain but not maximal."""
    s = sprinkle.sprinkle_diamond_1d(rho=400.0, tau=2.0, seed=2)
    c = order.causal_matrix_1d(s.u, s.v)
    # A mid-slice element alone is an antichain but should not be maximal.
    t_mid = 0.5 * (s.t.min() + s.t.max())
    single = [int(np.argmin(np.abs(s.t - t_mid)))]
    assert antichain.is_antichain(c, single)
    assert not antichain.is_maximal_antichain(c, single)


def test_antichain_hugs_midslice():
    """Elements of the mid-slice antichain should sit near t_mid on average."""
    s = sprinkle.sprinkle_diamond_1d(rho=1500.0, tau=2.0, seed=5)
    c = order.causal_matrix_1d(s.u, s.v)
    t_mid = 0.5 * (s.t.min() + s.t.max())
    a = antichain.maximal_antichain_near_midslice(c, s.t, t_mid=t_mid)
    spread = np.abs(s.t[a] - t_mid).mean()
    # Antichain should be far tighter around the slice than the full diamond half-height.
    assert spread < 0.25 * s.tau, spread
