"""Causal order of a sprinkling: causal matrix, links, chains.

For a set of sprinkled events the fundamental object is the causal (partial)
order ``prec``:  ``x_i prec x_j``  iff  ``x_j`` lies in the causal future of
``x_i`` (they are timelike- or null-separated with ``x_j`` later). The order is
irreflexive and transitive; ``x_i prec x_j`` and ``x_j prec x_k`` implies
``x_i prec x_k``.

1+1 D null-coordinate form.
---------------------------
In null coordinates ``u = t + x``, ``v = t - x`` a point ``x_j`` is in the causal
future of ``x_i`` iff  ``u_i <= u_j``  AND  ``v_i <= v_j``  (with at least one
strict, for distinct points). This is exactly the 2-D coordinate/dominance order
underlying Ulam's longest-increasing-subsequence problem; it is why the 1+1 D
longest chain is analytically tractable (see ``longest_chain_length``).

Definitions used here
---------------------
* Causal matrix ``C`` : ``C[i, j] = 1`` iff ``x_i prec x_j`` (strict future).
* Link matrix ``L``   : the transitive reduction of ``C``. ``L[i, j] = 1`` iff
  ``x_i prec x_j`` and there is no ``k`` with ``x_i prec x_k prec x_j``. A link is
  an irreducible ("covering") causal relation; the manuscript and the causal set
  literature call these the "links" of the causet.
* Chain : a totally ordered subset. Chain length = number of elements in it.
"""

from __future__ import annotations

import numpy as np


def causal_matrix_1d(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Strict causal matrix from 1+1 D null coordinates.

    ``C[i, j] = True`` iff ``x_i prec x_j``, i.e. ``u_i <= u_j`` and ``v_i <= v_j``
    with at least one coordinate strictly smaller (so the relation is irreflexive
    and, for a.s.-distinct Poisson points, antisymmetric).

    Complexity O(N^2) time and memory. Fine for the Phase-1 target N (few 1e3);
    the module docstring in sprinkle.py notes the 3-D scaling limits.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    if u.shape != v.shape or u.ndim != 1:
        raise ValueError("u and v must be 1-D arrays of equal length")

    le_u = u[:, None] <= u[None, :]
    le_v = v[:, None] <= v[None, :]
    lt_u = u[:, None] < u[None, :]
    lt_v = v[:, None] < v[None, :]
    # future: dominance in both coords, strict in at least one (excludes diagonal).
    c = le_u & le_v & (lt_u | lt_v)
    return c


def is_transitive(c: np.ndarray) -> bool:
    """Return True iff the causal matrix ``C`` encodes a transitive relation.

    Transitivity: if ``C[i,j]`` and ``C[j,k]`` then ``C[i,k]``. Equivalent to the
    boolean matrix condition ``(C @ C) implies C`` elementwise.
    """
    c = np.asarray(c, dtype=bool)
    reachable_2 = (c.astype(np.int64) @ c.astype(np.int64)) > 0
    return bool(np.all(~reachable_2 | c))


def link_matrix(c: np.ndarray) -> np.ndarray:
    """Transitive reduction of ``C`` (the links / covering relations).

    ``L[i, j] = True`` iff ``x_i prec x_j`` and no intermediate ``k`` satisfies
    ``x_i prec x_k prec x_j``. For a (finite, acyclic) partial order the transitive
    reduction is unique and is obtained by removing every relation that is implied
    by a length-2 path:  ``L = C AND NOT (C @ C)``.
    """
    c = np.asarray(c, dtype=bool)
    two_step = (c.astype(np.int64) @ c.astype(np.int64)) > 0
    return c & ~two_step


def longest_chain_length(c: np.ndarray) -> int:
    """Number of elements in a longest chain (maximal totally ordered subset).

    Computed by dynamic programming over a topological order of the DAG ``C``.
    ``L[j] = 1 + max_{i prec j} L[i]``; the answer is ``max_j L[j]``. Complexity
    O(N^2) using the dense causal matrix.

    In 1+1 D Minkowski this is the discrete analogue of the timelike geodesic
    length. Brightwell & Gregory (Phys. Rev. Lett. 66, 260 (1991)) showed the
    longest chain is asymptotically proportional to the continuum proper time;
    equivalently, for ``N`` points the expected longest chain satisfies
    ``E[L]/sqrt(N) -> 2`` -- the Vershik--Kerov / Logan--Shepp constant for the
    2-D random order (Ulam's problem). See tests/test_brightwell_gregory.py.
    """
    c = np.asarray(c, dtype=bool)
    n = c.shape[0]
    if n == 0:
        return 0

    # Topological order: parents (causal past) must precede children. Any order
    # consistent with C works; sorting by in-degree-free peeling is overkill, so
    # we sort by number of ancestors (rows that reach j), which is monotone along
    # the order.
    n_ancestors = c.sum(axis=0)  # how many i have i prec j
    topo = np.argsort(n_ancestors, kind="stable")

    best = np.ones(n, dtype=np.int64)
    for j in topo:
        preds = np.nonzero(c[:, j])[0]
        if preds.size:
            best[j] = 1 + int(best[preds].max())
    return int(best.max())


def chain_via_lis_1d(u: np.ndarray, v: np.ndarray) -> int:
    """Longest chain in 1+1 D via O(N log N) longest-increasing-subsequence.

    Independent cross-check of ``longest_chain_length`` that never materialises
    the O(N^2) matrix: sort points by ``u`` (breaking ties by ``v`` descending so
    equal-``u`` points cannot chain), then the longest chain is the longest
    strictly-increasing subsequence in ``v`` (patience sorting). Used in tests to
    confirm the dense DP agrees with the null-coordinate structure.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    if u.shape != v.shape or u.ndim != 1:
        raise ValueError("u and v must be 1-D arrays of equal length")
    if u.size == 0:
        return 0

    order = np.lexsort((-v, u))  # primary key u ascending, tie-break v descending
    seq = v[order]

    import bisect

    tails: list[float] = []
    for val in seq:
        # strictly increasing subsequence => bisect_left on strict order
        idx = bisect.bisect_left(tails, val)
        if idx == len(tails):
            tails.append(val)
        else:
            tails[idx] = val
    return len(tails)
