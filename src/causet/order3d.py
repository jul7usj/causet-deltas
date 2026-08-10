"""Causal order in 2+1 D Minkowski spacetime (M^3): causal matrix, links, chains.

Companion to ``order.py`` (1+1 D, Phase 1, accepted and unmodified). This module
supplies only what genuinely changes in 2+1 D and *re-exports* the parts of
Phase 1 that are already dimension-agnostic.

Causal relation
---------------
With signature ``(-,+,+)`` and coordinates ``(t, x, y)``, event ``i`` precedes
event ``j`` iff their separation is timelike and future-directed::

    x_i prec x_j   iff   (dt)^2 - (dx)^2 - (dy)^2 > 0   AND   dt > 0,
    dt = t_j - t_i,  dx = x_j - x_i,  dy = y_j - y_i.

Convention note (documented rather than silently differing from Phase 1): this
is *strict* timelike separation, so null-separated pairs are **not** related.
``order.causal_matrix_1d`` uses the inclusive convention (``u_i <= u_j`` and
``v_i <= v_j``), under which null pairs *are* related. For Poisson sprinklings
the set of exactly null-separated pairs has measure zero, so no statistical
quantity in this project can distinguish the two conventions; the difference
only shows up in hand-built test configurations, where the strict convention
above is the one asserted. It is also the convention of Rideout & Wallden
(arXiv:0810.1768), whose relation ``prec`` is timelike separation.

No null-coordinate shortcut
---------------------------
In 1+1 D the light cone has two flat faces, so ``u = t+x``, ``v = t-x`` turn the
causal order into the 2-D dominance order -- which is why Phase 1 could compute
the longest chain in ``O(N log N)`` by patience sorting (``chain_via_lis_1d``).
In 2+1 D the light cone is a round cone with infinitely many supporting
hyperplanes, so **no** linear change of coordinates turns the causal order into
a coordinate order, and no such shortcut exists. See
``longest_chain_length_3d`` for the consequence.

What is reused from Phase 1 verbatim
------------------------------------
``is_transitive``, ``link_matrix`` and the dense longest-chain dynamic program
in ``order.longest_chain_length`` already take an arbitrary boolean causal
matrix and use no 1+1 D structure whatsoever:

* transitivity is the boolean-matrix condition ``(C @ C) => C``;
* the transitive reduction is ``L = C AND NOT (C @ C)``;
* the chain DP sorts by ancestor count, which is a valid topological order of
  *any* finite strict partial order (if ``i prec j`` then
  ``anc(i) ∪ {i} ⊆ anc(j)``, so ``|anc(i)| < |anc(j)|``).

They are therefore re-exported here unchanged rather than rewritten, and
``tests/test_order3d.py`` contains an explicit regression test proving that the
re-exported functions reproduce Phase 1's 1+1 D results bit for bit.
"""

from __future__ import annotations

import numpy as np

from causet.order import is_transitive, link_matrix, longest_chain_length

__all__ = [
    "SPACETIME_DIM_2P1",
    "causal_matrix_3d",
    "is_transitive",
    "link_matrix",
    "longest_chain_length",
    "longest_chain_length_3d",
    "longest_chain_elements_streamed_3d",
    "chain_links_from_elements",
    "m_d_effective",
]

SPACETIME_DIM_2P1 = 3


def causal_matrix_3d(t: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Strict causal matrix of a set of M^3 events.

    ``C[i, j] = True`` iff ``x_i prec x_j``, i.e. ``dt > 0`` and
    ``dt^2 - dx^2 - dy^2 > 0`` with ``dt = t_j - t_i`` etc. Irreflexive and
    antisymmetric by construction (both conditions cannot hold for ``(i,j)`` and
    ``(j,i)``), and transitive because it is the causal order of Minkowski space
    -- which ``tests/test_order3d.py`` verifies numerically rather than assumes.

    Built directly from the coordinates: the 1+1 D null-coordinate construction
    does not generalise (see module docstring).

    Complexity ``O(N^2)`` time and memory. For large ``N`` where the dense
    matrix will not fit, use ``longest_chain_elements_streamed_3d``, which needs
    only ``O(N)`` memory.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if not (t.shape == x.shape == y.shape) or t.ndim != 1:
        raise ValueError("t, x, y must be 1-D arrays of equal length")

    dt = t[None, :] - t[:, None]  # dt[i, j] = t_j - t_i
    dx = x[None, :] - x[:, None]
    dy = y[None, :] - y[:, None]
    interval_sq = dt * dt - dx * dx - dy * dy  # positive => timelike separated
    return (interval_sq > 0.0) & (dt > 0.0)


def longest_chain_length_3d(c: np.ndarray) -> int:
    """Number of *elements* in a longest chain of the causet ``C``.

    Delegates to Phase 1's ``order.longest_chain_length``: a dense ``O(N^2)``
    dynamic program over a topological order, which is dimension-agnostic (see
    module docstring).

    DO NOT "OPTIMISE" THIS WITH THE PHASE-1 FAST PATH.
    ``order.chain_via_lis_1d`` computes the longest chain in ``O(N log N)`` by
    patience sorting. That is correct **only** in 1+1 D, where the light cone's
    two flat faces make the causal order equivalent to the 2-D coordinate
    (dominance) order, so a chain is exactly an increasing subsequence. In 2+1 D
    the light cone is round: no linear coordinate change produces a dominance
    order, chains are not subsequences of any single sorted sequence, and
    longest-increasing-subsequence machinery does not apply. Anyone tempted to
    swap in a sub-quadratic method must first exhibit the total order it relies
    on -- there isn't one. The only supported alternative is
    ``longest_chain_elements_streamed_3d``, which is the *same* quadratic DP
    with ``O(N)`` instead of ``O(N^2)`` memory.
    """
    return longest_chain_length(c)


def longest_chain_elements_streamed_3d(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    block_size: int = 128,
) -> int:
    """Longest-chain element count in M^3 using ``O(N)`` memory.

    Identical mathematics to ``longest_chain_length_3d`` -- the same quadratic
    dynamic program ``best[j] = 1 + max{best[i] : i prec j}`` -- but the causal
    matrix is never materialised: events are sorted by ``t`` (a valid
    topological order, since ``dt > 0`` along every relation) and processed in
    blocks of ``block_size`` rows, so peak memory is ``O(N * block_size)``
    rather than ``O(N^2)``. This is a memory/layout choice only, **not** an
    algorithmic shortcut exploiting any dimension-specific structure; the tests
    assert it returns exactly the same integer as the dense path.

    Minkowski Gram trick used for the pair test. Writing
    ``q_i = t_i^2 - x_i^2 - y_i^2`` and ``<i,j> = t_i t_j - x_i x_j - y_i y_j``,

        (dt)^2 - (dx)^2 - (dy)^2 = q_i + q_j - 2 <i, j>,

    so a whole block of pair tests is one ``(B,3) x (3,N)`` matrix product plus
    two broadcast additions, instead of three separate coordinate-difference
    arrays. After sorting by ``t`` the ``dt > 0`` clause is automatic: for
    ``i`` earlier in the sort ``dt >= 0``, and ``dt == 0`` forces
    ``q_i + q_j - 2<i,j> = -(dx^2 + dy^2) <= 0``, so such a pair can never pass
    the positivity test.

    Returns the number of elements; use ``chain_links_from_elements`` to convert
    to the number of links, which is what Rideout--Wallden call ``L``.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if not (t.shape == x.shape == y.shape) or t.ndim != 1:
        raise ValueError("t, x, y must be 1-D arrays of equal length")
    n = t.shape[0]
    if n == 0:
        return 0
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")

    topo = np.argsort(t, kind="stable")
    ts, xs, ys = t[topo], x[topo], y[topo]
    # Rows of `p` are events; `mp` carries the (+,-,-) metric so that
    # p @ mp.T is the Minkowski Gram matrix <i, j>.
    p = np.column_stack((ts, xs, ys))
    mp = np.column_stack((ts, -xs, -ys))
    q = ts * ts - xs * xs - ys * ys

    best = np.ones(n, dtype=np.int32)

    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        blk = slice(start, stop)

        if start > 0:
            # Predecessors already finalised: all events with index < start.
            gram = p[blk] @ mp[:start].T  # (B, start)
            gram *= -2.0
            gram += q[None, :start]
            gram += q[blk, None]
            related = gram > 0.0
            cand = np.where(related, best[None, :start], np.int32(0))
            best[blk] = cand.max(axis=1) + 1
            del gram, related, cand

        # Resolve relations *within* the block sequentially, so that each row
        # sees the final value of every earlier row in the same block.
        b = stop - start
        if b > 1:
            gi = p[blk] @ mp[blk].T
            gi *= -2.0
            gi += q[blk][None, :]
            gi += q[blk][:, None]
            rel_in = gi > 0.0
            local = best[blk]
            for a in range(1, b):
                preds = np.flatnonzero(rel_in[a, :a])
                if preds.size:
                    cand_in = int(local[preds].max()) + 1
                    if cand_in > local[a]:
                        local[a] = cand_in
            best[blk] = local

    return int(best.max())


def chain_links_from_elements(n_elements: int) -> int:
    """Convert a chain's element count to its link count.

    Rideout & Wallden, Section II.1: "we define proper time d(x,y), between two
    related elements x prec y, to be the number of links L in the longest chain
    between (and including) x and y". A chain ``x = z_0 prec z_1 prec ... prec
    z_k = y`` has ``k`` links and ``k + 1`` elements, so ``L = elements - 1``.

    This one-element offset is not cosmetic at the sizes reachable here: it
    shifts ``m_d^eff = L (rho V)^{-1/d}`` by ``(rho V)^{-1/d}``, which is ~0.05
    at ``rho V ~ 10^4`` -- comparable to the +/-0.012 uncertainty Rideout and
    Wallden quote on ``m_3``. Hence it is isolated in a named, cited function.
    """
    if n_elements < 1:
        raise ValueError(f"a chain has at least one element, got {n_elements}")
    return n_elements - 1


def m_d_effective(n_links: int, rho_volume: float, d: int = SPACETIME_DIM_2P1) -> float:
    """Effective Myrheim--Meyer/Brightwell--Gregory constant of eq. (1).

    Rideout & Wallden eq. (1), Section II.1::

        L (rho V)^(-1/d)  ->  m_d      as  rho V -> infinity,

    so the finite-size estimator is ``m_d^eff = L / (rho V)^(1/d)``. The
    normaliser is the *expected* count ``rho V`` (deterministic given the
    parameters), exactly as written in eq. (1), not the realised Poisson count.

    ``d`` is the spacetime dimension: ``d = 3`` for 2+1 D. Known asymptote for
    ``d = 2`` is ``m_2 = 2`` (Brightwell & Gregory, PRL 66, 260 (1991)), which
    Phase 1 already validated; the ``d = 3`` target is ``m_3 = 2.296 +/- 0.012``
    (Rideout--Wallden Fig. 4, Section III.1).
    """
    if rho_volume <= 0:
        raise ValueError(f"rho_volume must be positive, got {rho_volume}")
    return n_links / rho_volume ** (1.0 / d)
