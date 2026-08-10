"""Tests for the 2+1 D causal order (``causet.order3d``).

Contains, per the Phase-2b brief:

* a hand-built M^3 configuration whose entire causal matrix is written out
  literally and was computed by hand (timelike pairs, spacelike pairs and a
  mutually null-separated triple), so the relation is checked against arithmetic
  a reader can redo, not against another implementation;
* a *reuse-verification regression test* proving the re-exported Phase-1
  ``is_transitive`` / ``link_matrix`` / ``longest_chain_length`` behave
  identically on Phase 1's own 1+1 D sprinklings -- i.e. making them serve 2+1 D
  changed nothing about 1+1 D;
* equality of the ``O(N)``-memory streamed longest chain with the dense
  ``O(N^2)`` dynamic program.

Hand-built configuration (all arithmetic exact in binary floating point, which
is why 3-4-5 triples are used for the null relations)::

    e0 = (  0, 0, 0)        e3 = ( 1, 2, 0)
    e1 = (  5, 1, 0)        e4 = ( 5, 3, 4)
    e2 = ( 10, 0, 1)        e5 = (10, 6, 8)

    pair    dt   dx  dy   dt^2-dx^2-dy^2   relation
    0,1      5    1   0   25-1-0  =  24    timelike  -> 0 prec 1
    0,2     10    0   1  100-0-1  =  99    timelike  -> 0 prec 2
    0,3      1    2   0    1-4-0  =  -3    spacelike -> unrelated
    0,4      5    3   4   25-9-16 =   0    NULL      -> unrelated (strict)
    0,5     10    6   8  100-36-64 =  0    NULL      -> unrelated (strict)
    1,2      5   -1   1   25-1-1  =  23    timelike  -> 1 prec 2
    1,4      0    2   4   0-4-16  = -20    spacelike -> unrelated
    1,5      5    5   8   25-25-64= -64    spacelike -> unrelated
    3,1      4   -1   0   16-1-0  =  15    timelike  -> 3 prec 1
    3,2      9   -2   1   81-4-1  =  76    timelike  -> 3 prec 2
    3,4      4    1   4   16-1-16 =  -1    spacelike -> unrelated
    3,5      9    4   8   81-16-64=   1    timelike  -> 3 prec 5
    4,2      5   -3  -3   25-9-9  =   7    timelike  -> 4 prec 2
    4,5      5    3   4   25-9-16 =   0    NULL      -> unrelated (strict)
    2,5      0    6   7   0-36-49 = -85    spacelike -> unrelated

So {e0, e4, e5} is a mutually null-separated triple and therefore an antichain
under the strict-timelike convention documented in ``order3d``.
"""

from __future__ import annotations

import numpy as np
import pytest

from causet import order, order3d, sprinkle, sprinkle3d

SEED = 20260810

HAND_T = np.array([0.0, 5.0, 10.0, 1.0, 5.0, 10.0])
HAND_X = np.array([0.0, 1.0, 0.0, 2.0, 3.0, 6.0])
HAND_Y = np.array([0.0, 0.0, 1.0, 0.0, 4.0, 8.0])

# Rows = i, columns = j, entry True iff e_i prec e_j. Hand-computed above.
HAND_CAUSAL = np.array(
    [
        [0, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 1],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ],
    dtype=bool,
)

# Transitive reduction, also by hand: the only relations implied by a 2-step path
# are 0 prec 2 (via 1) and 3 prec 2 (via 1 or 4), so exactly those two drop out.
HAND_LINKS = np.array(
    [
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ],
    dtype=bool,
)


# ------------------------------------------------- hand-built configuration ---
def test_hand_built_causal_matrix_is_exactly_as_computed_by_hand():
    c = order3d.causal_matrix_3d(HAND_T, HAND_X, HAND_Y)
    assert c.dtype == bool
    assert np.array_equal(c, HAND_CAUSAL), f"got\n{c.astype(int)}"


def test_hand_built_timelike_spacelike_and_null_cases():
    c = order3d.causal_matrix_3d(HAND_T, HAND_X, HAND_Y)
    # timelike, future-directed
    assert c[0, 1] and not c[1, 0]
    assert c[1, 2] and c[0, 2]
    # spacelike
    assert not c[0, 3] and not c[3, 0]
    assert not c[3, 4] and not c[4, 3]
    # the mutually null-separated triple {0, 4, 5}: an antichain
    for i, j in [(0, 4), (0, 5), (4, 5)]:
        dt = HAND_T[j] - HAND_T[i]
        dx = HAND_X[j] - HAND_X[i]
        dy = HAND_Y[j] - HAND_Y[i]
        assert dt * dt - dx * dx - dy * dy == 0.0  # exactly null, no rounding
        assert not c[i, j] and not c[j, i]


def test_hand_built_links_and_longest_chain():
    c = order3d.causal_matrix_3d(HAND_T, HAND_X, HAND_Y)
    assert np.array_equal(order3d.link_matrix(c), HAND_LINKS)
    assert order3d.is_transitive(c)
    # Longest chains are 3 prec 1 prec 2 and 0 prec 1 prec 2: 3 elements, 2 links.
    assert order3d.longest_chain_length_3d(c) == 3
    assert order3d.chain_links_from_elements(3) == 2
    assert (
        order3d.longest_chain_elements_streamed_3d(HAND_T, HAND_X, HAND_Y, block_size=2) == 3
    )


def test_irreflexive_and_antisymmetric():
    c = order3d.causal_matrix_3d(HAND_T, HAND_X, HAND_Y)
    assert not np.any(np.diag(c))
    assert not np.any(c & c.T)


def test_causal_matrix_3d_input_validation():
    with pytest.raises(ValueError):
        order3d.causal_matrix_3d(np.zeros(3), np.zeros(2), np.zeros(3))
    with pytest.raises(ValueError):
        order3d.causal_matrix_3d(np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)))


# ------------------------------------- properties on genuine M^3 sprinklings ---
@pytest.mark.parametrize("seed_off", [0, 1, 2])
def test_sprinkled_order_is_a_transitive_partial_order(seed_off):
    # Size note: ``is_transitive``/``link_matrix`` multiply int64 matrices, which
    # NumPy does not route through BLAS, so they cost O(N^3) in practice (~30 s at
    # N ~ 1600). These tests check *behaviour*, not scaling, so N ~ 500 is used;
    # the experiment never calls them (it uses the streamed chain counter).
    s = sprinkle3d.sprinkle_diamond_3d(rho=1900.0, tau=1.0, seed=SEED + seed_off)
    c = order3d.causal_matrix_3d(s.t, s.x, s.y)
    assert s.n > 400
    assert order3d.is_transitive(c)
    assert not np.any(np.diag(c))
    assert not np.any(c & c.T)


def test_endpoints_precede_and_follow_every_interior_event():
    """Every event of I(p,q) lies causally between the endpoints -- so the longest
    chain of the endpoint-inclusive set really does run from p to q."""
    s = sprinkle3d.sprinkle_diamond_3d(
        rho=2000.0, tau=1.0, seed=SEED + 7, include_endpoints=True
    )
    c = order3d.causal_matrix_3d(s.t, s.x, s.y)
    i_past, i_future = s.endpoint_indices
    interior = np.ones(s.n, dtype=bool)
    interior[[i_past, i_future]] = False
    assert np.all(c[i_past, interior])
    assert np.all(c[interior, i_future])
    assert c[i_past, i_future]


def test_link_matrix_is_the_transitive_reduction_of_the_sprinkled_order():
    s = sprinkle3d.sprinkle_diamond_3d(rho=400.0, tau=1.0, seed=SEED + 11)
    c = order3d.causal_matrix_3d(s.t, s.x, s.y)
    lm = order3d.link_matrix(c)
    assert np.all(c | ~lm)  # links are a subset of relations
    # Transitive closure of the links must recover the full causal matrix.
    closure = lm.copy()
    for _ in range(order3d.longest_chain_length_3d(c) + 1):
        nxt = closure | ((closure.astype(np.int64) @ lm.astype(np.int64)) > 0)
        if np.array_equal(nxt, closure):
            break
        closure = nxt
    assert np.array_equal(closure, c)


@pytest.mark.parametrize("block_size", [1, 7, 64, 4096])
def test_streamed_longest_chain_equals_dense_dp(block_size):
    """The O(N)-memory path is the same DP, so it must agree exactly, not nearly."""
    for seed_off in range(4):
        s = sprinkle3d.sprinkle_diamond_3d(
            rho=1500.0, tau=1.0, seed=SEED + 300 + seed_off, include_endpoints=True
        )
        c = order3d.causal_matrix_3d(s.t, s.x, s.y)
        dense = order3d.longest_chain_length_3d(c)
        streamed = order3d.longest_chain_elements_streamed_3d(
            s.t, s.x, s.y, block_size=block_size
        )
        assert streamed == dense, f"seed {s.seed}: streamed {streamed} != dense {dense}"


def test_streamed_longest_chain_edge_cases():
    empty = np.array([])
    assert order3d.longest_chain_elements_streamed_3d(empty, empty, empty) == 0
    one = np.array([0.0])
    assert order3d.longest_chain_elements_streamed_3d(one, one, one) == 1
    with pytest.raises(ValueError):
        order3d.longest_chain_elements_streamed_3d(one, one, one, block_size=0)


# ----------------------------------------- reuse verification against Phase 1 --
def test_reused_helpers_are_literally_phase1_functions():
    """No copy-paste: order3d re-exports the Phase-1 implementations."""
    assert order3d.is_transitive is order.is_transitive
    assert order3d.link_matrix is order.link_matrix
    assert order3d.longest_chain_length is order.longest_chain_length


@pytest.mark.parametrize("rho", [125.0, 250.0])  # N ~ 250, 500; see size note above
def test_phase1_1p1d_regression_through_order3d(rho):
    """Regression: on Phase 1's own 1+1 D sprinklings the helpers reached through
    ``order3d`` give results identical to Phase 1's, including agreement with
    Phase 1's independent O(N log N) LIS chain counter."""
    for seed_off in range(3):
        s = sprinkle.sprinkle_diamond_1d(
            rho, tau=2.0, seed=SEED + 900 + seed_off, include_endpoints=True
        )
        c1 = order.causal_matrix_1d(s.u, s.v)

        assert order3d.is_transitive(c1) == order.is_transitive(c1) is True
        assert np.array_equal(order3d.link_matrix(c1), order.link_matrix(c1))
        assert order3d.longest_chain_length(c1) == order.longest_chain_length(c1)
        # Phase 1's independent cross-check still holds through the 2+1 D module.
        assert order3d.longest_chain_length(c1) == order.chain_via_lis_1d(s.u, s.v)


def test_generalised_helpers_on_a_non_geometric_causet():
    """The reused helpers must work on any valid boolean causal matrix, not only
    matrices produced by a sprinkling. A 4-element diamond-shaped causet:
    0 prec {1,2} prec 3, with 1, 2 incomparable."""
    c = np.array(
        [
            [0, 1, 1, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ],
        dtype=bool,
    )
    assert order3d.is_transitive(c)
    expected_links = np.array(
        [
            [0, 1, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ],
        dtype=bool,
    )
    assert np.array_equal(order3d.link_matrix(c), expected_links)
    assert order3d.longest_chain_length_3d(c) == 3

    non_transitive = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=bool)
    assert not order3d.is_transitive(non_transitive)


# ------------------------------------------------------ eq. (1) bookkeeping ----
def test_chain_links_from_elements():
    assert order3d.chain_links_from_elements(1) == 0
    assert order3d.chain_links_from_elements(5) == 4
    with pytest.raises(ValueError):
        order3d.chain_links_from_elements(0)


def test_m_d_effective_matches_equation_1():
    # m_d^eff = L (rho V)^(-1/d).
    assert order3d.m_d_effective(20, 1000.0, d=3) == pytest.approx(20.0 / 1000.0 ** (1 / 3))
    # d = 2 reproduces the Phase-1 quantity L / sqrt(rho V) whose asymptote is 2.
    assert order3d.m_d_effective(40, 400.0, d=2) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        order3d.m_d_effective(5, 0.0)
