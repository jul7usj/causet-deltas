"""Causal matrix transitivity, links, and longest-chain correctness."""

from __future__ import annotations

import numpy as np

from causet import order, sprinkle


def test_causal_matrix_small_handbuilt():
    # Three points on the time axis: a prec b prec c (null coords along diagonal).
    u = np.array([0.0, 1.0, 2.0])
    v = np.array([0.0, 1.0, 2.0])
    c = order.causal_matrix_1d(u, v)
    assert c[0, 1] and c[1, 2] and c[0, 2]
    assert not c[1, 0] and not c[2, 1]
    assert not np.any(np.diag(c))  # irreflexive


def test_spacelike_pair_incomparable():
    # Two points at equal t but different x are spacelike -> incomparable.
    # (t,x)=(1, 0.5) -> (u,v)=(1.5,0.5); (t,x)=(1,-0.5) -> (u,v)=(0.5,1.5).
    u = np.array([1.5, 0.5])
    v = np.array([0.5, 1.5])
    c = order.causal_matrix_1d(u, v)
    assert not c[0, 1] and not c[1, 0]


def test_transitivity_holds_on_sprinkling():
    for seed in range(5):
        s = sprinkle.sprinkle_diamond_1d(rho=300.0, tau=2.0, seed=seed)
        c = order.causal_matrix_1d(s.u, s.v)
        assert order.is_transitive(c)


def test_link_matrix_is_transitive_reduction():
    # Chain a<b<c: only links are a-b and b-c, NOT a-c.
    u = np.array([0.0, 1.0, 2.0])
    v = np.array([0.0, 1.0, 2.0])
    c = order.causal_matrix_1d(u, v)
    lnk = order.link_matrix(c)
    assert lnk[0, 1] and lnk[1, 2]
    assert not lnk[0, 2]  # transitive relation removed


def test_link_transitive_closure_recovers_causal_matrix():
    """The transitive closure of the links must equal the original order."""
    s = sprinkle.sprinkle_diamond_1d(rho=200.0, tau=2.0, seed=11)
    c = order.causal_matrix_1d(s.u, s.v)
    lnk = order.link_matrix(c).astype(np.int64)
    # Floyd-style closure of the link matrix.
    reach = lnk.copy()
    n = reach.shape[0]
    for k in range(n):
        reach = reach | (np.outer(reach[:, k], reach[k, :]))
    reach = reach > 0
    assert np.array_equal(reach, c)


def test_longest_chain_matches_lis():
    """Dense DP and O(N log N) LIS must agree on the longest chain."""
    for seed in range(8):
        s = sprinkle.sprinkle_diamond_1d(rho=250.0, tau=2.0, seed=seed)
        c = order.causal_matrix_1d(s.u, s.v)
        dp = order.longest_chain_length(c)
        lis = order.chain_via_lis_1d(s.u, s.v)
        assert dp == lis, (seed, dp, lis)


def test_longest_chain_trivial_cases():
    empty = np.zeros((0, 0), dtype=bool)
    assert order.longest_chain_length(empty) == 0
    single = np.zeros((1, 1), dtype=bool)
    assert order.longest_chain_length(single) == 1
