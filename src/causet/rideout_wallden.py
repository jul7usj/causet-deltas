"""Rideout--Wallden 2-link spacelike distance (arXiv:0810.1768, Sections IV--V).

Phase 2b Part 2. Builds on the Part-1 2+1 D infrastructure (``sprinkle3d``,
``order3d``), which is a closed, validated dependency and is not re-derived here.

The construction in one paragraph
---------------------------------
For a *spacelike* (causally unrelated) pair ``x, y`` the continuum spatial
distance equals the shortest proper time joining the common past to the common
future::

    d(x, y) = min { tau(p, f) : p in J^-(x) n J^-(y),  f in J^+(x) n J^+(y) } .

(Check in M^3: for ``x = (0, -D/2, 0)``, ``y = (0, +D/2, 0)`` the two light cones
first meet at ``f = (D/2, 0, 0)`` and last part at ``p = (-D/2, 0, 0)``, whose
proper separation is exactly ``D``.) Discretising this "naive" definition fails
for spacetime dimension ``d >= 3``: boost freedom supplies infinitely many pairs
attaining the continuum minimum, and the *discrete* minimum over them undershoots
without bound. Rideout--Wallden's repair (their Section V.A) is to stop taking a
double minimum and instead **average** the single minimum over a distinguished,
finite family of future events -- the future *2-links* of the pair.

Definition 2b with n = 2 (their Section IV.A)
---------------------------------------------
``f`` is a **future 2-link of the pair {x, y}** iff

1. ``x prec f``  and  ``y prec f``  (``f`` is in the common future), **and**
2. there is no ``z``  with ``x prec z prec f``   (``f`` is *linked* to ``x``), **and**
3. there is no ``z'`` with ``y prec z' prec f``  (``f`` is *linked* to ``y``).

i.e. ``f`` lies in the common future and *both* Alexandrov intervals ``[x, f]``
and ``[y, f]`` are empty. Geometrically 2-links sit just inside both light cones,
hugging the light-cone intersection; in a bounded region there are finitely many
of them, so the average over them is well defined -- but the count is a property
of the region, and ``two_link_distance`` therefore always reports it.

**This is NOT "the minimal elements of ``fut(x) n fut(y)``."** Every 2-link is
minimal in the common future (if ``g`` were a common-future element below ``f``
then ``x prec g prec f``, violating (2)), so 2-links are a *subset* -- and a
strict one. An element can be minimal in the intersection while an intervening
``z`` with ``x prec z prec f`` exists, provided ``z`` is not itself in
``fut(y)``: minimality only forbids intervening elements that lie in *both*
futures. Rideout--Wallden warn in Section V.A that admitting such elements lets
the Step-2 minimising pair have arbitrarily large proper time, which is precisely
the failure the 2-link condition exists to prevent. It would surface as unbounded
drift in the Fig.-14 stability test, so it is guarded by an explicit negative
control in ``tests/test_rideout_wallden.py``.

The Section V.A algorithm (Steps 1--5), as implemented
------------------------------------------------------
1. Find every future 2-link ``f_i`` of ``{x, y}``                (``future_2links``).
2. For each ``f_i``, find the ``p`` in ``past(x) n past(y)`` **minimising** the
   timelike distance ``d(p, f_i)`` -- the longest chain from ``p`` to ``f_i``
   counted in **links** (see the convention note below), converted to a proper
   time with ``m_3`` via eqs. (1)--(2).
3. Store that minimised ``d^i(x, y)``.
4. Repeat over all ``f_i``.
5. Average. ``two_link_distance`` returns the mean, its standard error over the
   2-links, and the number of 2-links used.

LINK CONVENTION (Part-1 Constraint 1; call site of ``chain_links_from_elements``)
--------------------------------------------------------------------------------
Rideout--Wallden, Section II.1: proper time ``d(x,y)`` is "the number of **links**
L in the longest chain between (and including) x and y". A chain with ``k+1``
elements has ``k`` links, so ``L = elements - 1``. Part 1 measured that getting
this wrong injects a ``+(rho V)^{-1/d}`` term with coefficient exactly 1 into
``m_d^eff`` -- 0.02-0.10 at reachable sizes, larger than the published
uncertainty on ``m_3`` itself. The single place in this module where a chain
becomes a number is ``chain_links_between``, which converts through the named,
cited ``order3d.chain_links_from_elements``; nothing else counts chains.

Calibration to proper time (eqs. (1) and (2))
---------------------------------------------
eq. (1): ``L (rho V)^{-1/d} -> m_d``;  eq. (2): ``V = eta(d) l^d``. Eliminating
``V`` for the interval ``[p, f]`` whose endpoints are separated by proper time
``l``::

    L / ( l (rho eta(d))^{1/d} ) = m_d      =>      l = L / ( m_d (rho eta(d))^{1/d} ) ,

implemented in ``proper_time_from_chain_links``. ``eta(d)`` is taken from
``sprinkle3d.interval_volume_constant`` (computed, not hardcoded).

Two accuracy caveats that this module reports rather than hides
---------------------------------------------------------------
* **The calibration is asymptotic but is applied to small intervals.** eq. (1) is
  a ``rho V -> infinity`` statement, yet Step 2 deliberately selects the
  *smallest* interval ``[p, f_i]`` available. Part 1 measured ``m_3^eff`` rising
  from ~1.97 at ``rho V = 2^10`` to ~2.15 at ``2^17`` against the asymptote
  2.296, so using ``m_3`` on intervals holding tens of elements overestimates the
  distance by a factor of order ``m_3 / m_3^eff ~ 1.1-1.2``. The result object
  reports the mean element count of the minimising intervals so the regime is
  visible. This is a *scale* bias: at fixed density it is a constant multiplier
  and therefore does not affect the Fig.-14 stability test, but it must be
  remembered in any head-to-head against a differently calibrated estimator.
* **There is a hard resolution floor of 2 links.** Every minimising interval
  ``[p, f]`` contains both ``x`` and ``y`` (``p prec x prec f``), so its longest
  chain has at least 3 elements, i.e. ``L >= 2``. No 2-link distance can come out
  below ``2 / (m_d (rho eta)^{1/d})``, and near that floor the estimator is
  visibly quantised.

Performance (Part-1 Constraint 2)
---------------------------------
Nothing here uses ``order.is_transitive`` or ``order.link_matrix``: those
multiply int64 matrices, which NumPy does not route through BLAS, making them
effectively O(N^3) (~32 s at N ~ 1600, measured in Part 1). The 2-link test is a
pair of boolean row/column reductions, and the chain counts run the dense
Phase-1 dynamic program ``order.longest_chain_length`` (a plain DP with no matrix
product) on the *interval* sub-poset only.

The one non-obvious optimisation is exact and proved, not heuristic:

    **Step 2 need only scan the MAXIMAL elements of ``past(x) n past(y)``.**
    If ``p prec p'`` with both in the common past, then any chain from ``p'`` to
    ``f`` extends by ``p``, so ``d(p, f) >= d(p', f) + 1 > d(p', f)``. A
    minimiser is therefore never a non-maximal element.

This matters because ``|past(x) n past(y)|`` grows with the sprinkling region
while the minimising intervals do not, so the reduction is what keeps the
Fig.-14 stability test affordable. ``two_link_distance(..., exhaustive_past=True)``
disables it and scans the whole common past; the tests assert the two agree.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import order
from .order3d import SPACETIME_DIM_2P1, chain_links_from_elements
from .sprinkle3d import interval_volume_constant

__all__ = [
    "RW_M3",
    "RW_M3_ERR",
    "future_2links",
    "minimal_common_future",
    "maximal_common_past",
    "chain_links_between",
    "chain_links_to_target",
    "proper_time_from_chain_links",
    "TwoLinkDistanceResult",
    "two_link_distance",
    "NaiveDistanceResult",
    "naive_distance",
]

# ---- Values quoted by Rideout & Wallden (Integrity Rule 4: named + sourced) ----

#: Fitted asymptote of eq. (1) in d = 3, Rideout--Wallden Fig. 4 / Sec. III.1.
#: Reproduced by the Part-1 gate (``experiments/exp02_m3_validation.py``): our
#: ``m_3^eff(N)`` lies on their published curve to <= 0.02 absolute over a
#: factor-128 range in N, and our own extrapolation gives 2.343 +/- 0.099.
RW_M3 = 2.296
RW_M3_ERR = 0.012


# ---------------------------------------------------------------------------
# Future 2-links: Definition 2b with n = 2 (Section IV.A).
# ---------------------------------------------------------------------------


def future_2links(x: int, y: int, causal_matrix: np.ndarray) -> np.ndarray:
    """Future 2-links of the pair ``{x, y}`` -- Rideout--Wallden Def. 2b, ``n = 2``.

    Returns the sorted indices of every ``f`` such that ``x prec f``, ``y prec f``
    and *both* Alexandrov intervals ``[x, f]``, ``[y, f]`` are empty (equivalently:
    ``f`` is linked to ``x`` and linked to ``y`` individually).

    See the module docstring for why this is strictly stronger than "minimal
    elements of ``fut(x) n fut(y)``" and why the difference is load-bearing.

    Implementation is three boolean reductions on the causal matrix -- no matrix
    powers, no ``link_matrix`` (Part-1 Constraint 2). With ``F`` the common
    future and ``Fx = fut(x)``::

        exists z: x prec z prec f    <=>    C[Fx, f].any()

    evaluated for all ``f`` at once as ``C[ix_(Fx, F)].any(axis=0)``. Cost is
    ``O(|fut(x)| |F| + |fut(y)| |F|)`` boolean operations.

    ``x`` and ``y`` may be any two distinct elements; if they are causally
    related the result is empty (any ``f`` above the later one has the earlier
    one inside its interval), which is the correct degenerate answer. The
    spacelike requirement is enforced by ``two_link_distance``, not here, so that
    the degenerate case stays testable.
    """
    cm = np.asarray(causal_matrix, dtype=bool)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"causal_matrix must be square, got shape {cm.shape}")
    if x == y:
        raise ValueError("future_2links needs two distinct elements")

    common = np.flatnonzero(cm[x, :] & cm[y, :])
    if common.size == 0:
        return common

    fut_x = np.flatnonzero(cm[x, :])
    fut_y = np.flatnonzero(cm[y, :])
    # blocked[j] is True iff some element of fut(x) lies strictly below common[j].
    blocked_x = cm[np.ix_(fut_x, common)].any(axis=0)
    blocked_y = cm[np.ix_(fut_y, common)].any(axis=0)
    return common[~(blocked_x | blocked_y)]


def minimal_common_future(x: int, y: int, causal_matrix: np.ndarray) -> np.ndarray:
    """Minimal elements of ``fut(x) n fut(y)`` -- the WEAKER condition, for contrast.

    Provided so the negative control in the tests can be stated directly and so
    experiments can quote how many minimal elements are *not* 2-links. Rideout--
    Wallden, Section V.A, reject this set for the Step-2 minimisation: an ``f``
    minimal here may still have an intervening ``z`` with ``x prec z prec f``
    (one that simply misses ``fut(y)``), and admitting it lets the minimising
    pair's proper time grow without bound. Never use this in place of
    ``future_2links``.
    """
    cm = np.asarray(causal_matrix, dtype=bool)
    common = np.flatnonzero(cm[x, :] & cm[y, :])
    if common.size == 0:
        return common
    sub = cm[np.ix_(common, common)]
    return common[~sub.any(axis=0)]  # nothing in the set strictly below it


def maximal_common_past(x: int, y: int, causal_matrix: np.ndarray) -> np.ndarray:
    """Maximal elements of ``past(x) n past(y)``.

    The Step-2 minimisation may be restricted to these without changing its
    result: for ``p prec p'`` both in the common past, a chain from ``p'`` to
    ``f`` extends by ``p``, so ``d(p, f) > d(p', f)`` and ``p`` can never be the
    minimiser (module docstring). Returned sorted.
    """
    cm = np.asarray(causal_matrix, dtype=bool)
    common = np.flatnonzero(cm[:, x] & cm[:, y])
    if common.size == 0:
        return common
    sub = cm[np.ix_(common, common)]
    return common[~sub.any(axis=1)]  # nothing in the set strictly above it


# ---------------------------------------------------------------------------
# Timelike distance between a related pair (Step 2), in LINKS.
# ---------------------------------------------------------------------------


def chain_links_between(p: int, f: int, causal_matrix: np.ndarray) -> int:
    """Discrete proper time ``d(p, f)`` in **links**; requires ``p prec f``.

    Rideout--Wallden Section II.1: the number of links in the longest chain
    between and including ``p`` and ``f``. Computed on the interval sub-poset
    ``{p} u [p,f] u {f}`` with the Phase-1 dense dynamic program (no matrix
    product -- Part-1 Constraint 2), which returns a chain **element** count;
    the element -> link conversion is the cited ``chain_links_from_elements``
    (Part-1 Constraint 1).

    A longest chain of the sub-poset always contains both endpoints -- ``p``
    precedes and ``f`` succeeds every interior element, so any chain extends to
    include them -- hence the sub-poset maximum is the ``p``-to-``f`` maximum.

    This is the straightforward one-pair-at-a-time form, and the tests use it as
    the oracle. The estimators themselves call ``chain_links_to_target``, which
    returns the same numbers for a whole set of ``p`` from a single dynamic
    program -- the shape Step 2's minimisation actually has.
    """
    cm = np.asarray(causal_matrix, dtype=bool)
    if not cm[p, f]:
        raise ValueError(f"chain_links_between requires p ({p}) prec f ({f})")
    interior = np.flatnonzero(cm[p, :] & cm[:, f])
    idx = np.concatenate(([p], interior, [f])).astype(int)
    sub = cm[np.ix_(idx, idx)]
    return chain_links_from_elements(order.longest_chain_length(sub))


def chain_links_to_target(
    target: int, sources: np.ndarray, causal_matrix: np.ndarray
) -> np.ndarray:
    """``d(p, target)`` in links for **every** ``p`` in ``sources``, in one pass.

    Same quantity as calling ``chain_links_between(p, target, ...)`` in a loop --
    the tests assert exact agreement -- but one dynamic program serves all
    sources instead of one per source. Step 2 minimises over many ``p`` for the
    same ``f``, so this is the shape the work actually has.

    Every ``p`` in ``sources`` must satisfy ``p prec target``.

    Method. Restrict to ``S = sources u (fut(sources) n past(target)) u {target}``
    -- exactly the elements a chain from some source to ``target`` can pass
    through, so the restriction is exact, not a heuristic, and it discards the
    (large, and growing with the region) part of ``past(target)`` that lies
    outside every source's future. ``target`` is the unique maximum of ``S``, so
    every chain from ``p`` extends to it and the ordinary longest-path dynamic
    program in reverse topological order gives, for each ``p``, the longest chain
    from ``p`` to ``target`` in **elements**. Sorting by ancestor count is a
    valid topological order of any finite poset (``i prec j`` implies
    ``anc(i) u {i} subset anc(j)``), the same argument Phase 1 uses in
    ``order.longest_chain_length``.

    The element -> **link** conversion (Part-1 Constraint 1) is the cited
    ``order3d.chain_links_from_elements``, applied elementwise here.

    No matrix products anywhere (Part-1 Constraint 2): boolean masks plus a
    row-at-a-time maximum.
    """
    cm = np.asarray(causal_matrix, dtype=bool)
    src = np.asarray(sources, dtype=int).ravel()
    if src.size == 0:
        return np.empty(0, dtype=int)
    if not cm[src, target].all():
        bad = src[~cm[src, target]]
        raise ValueError(f"sources {bad.tolist()} do not precede target {target}")

    keep = cm[:, target] & cm[src, :].any(axis=0)  # strictly between a source and target
    keep[src] = True
    keep[target] = True
    idx = np.flatnonzero(keep)  # sorted, and contains target
    sub = cm[np.ix_(idx, idx)]

    topo = np.argsort(sub.sum(axis=0), kind="stable")  # ancestors first
    elements = np.ones(idx.size, dtype=np.int64)
    for i in topo[::-1]:  # descendants first: successors are already final
        succ = np.flatnonzero(sub[i])
        if succ.size:
            elements[i] = 1 + int(elements[succ].max())

    at = np.searchsorted(idx, src)
    return np.array(
        [chain_links_from_elements(int(e)) for e in elements[at]], dtype=int
    )


def proper_time_from_chain_links(
    n_chain_links: float,
    rho: float,
    m_d: float = RW_M3,
    d: int = SPACETIME_DIM_2P1,
) -> float:
    """Convert a chain-link count to a continuum proper time via eqs. (1)--(2).

    Rideout--Wallden eq. (1) ``L (rho V)^{-1/d} -> m_d`` with eq. (2)
    ``V = eta(d) l^d`` gives, for the interval whose endpoints are separated by
    proper time ``l``::

        l = L / ( m_d * (rho * eta(d))^{1/d} ) .

    ``eta(d)`` comes from ``sprinkle3d.interval_volume_constant`` (``eta(3) =
    pi/12``), computed rather than hardcoded.

    ``n_chain_links`` must already be in the **link** convention (Constraint 1).
    The result inherits the asymptotic caveat in the module docstring: eq. (1)
    holds as ``rho V -> infinity``, and Step 2 applies it to the smallest
    available interval.
    """
    if rho <= 0:
        raise ValueError(f"rho must be positive, got {rho}")
    if m_d <= 0:
        raise ValueError(f"m_d must be positive, got {m_d}")
    eta = interval_volume_constant(d)
    return float(n_chain_links) / (m_d * (rho * eta) ** (1.0 / d))


# ---------------------------------------------------------------------------
# The full Section V.A estimator (Steps 1--5).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TwoLinkDistanceResult:
    """Outcome of the 2-link distance for one spacelike pair on one causet.

    The three headline numbers required by Steps 1--5 are ``mean``,
    ``std_error`` and ``n_two_links``; ``as_tuple()`` returns exactly those.

    ``n_two_links`` is deliberately *not* called ``n_links``: this module also
    counts chain links (Constraint 1) and one ambiguous name for both quantities
    is exactly the confusion Part 1 was bitten by. It is the count of future
    2-links -- Step 1's ``f_i`` -- and is always reported: a small count means
    the average of Step 5 rests on few samples, which is a finding to log, not to
    hide (Rideout--Wallden note a bounded region admits only finitely many).

    Attributes
    ----------
    mean, std_error:
        Mean and standard error over the 2-links of the Step-3 minimised
        distances. ``nan`` when there are no usable 2-links; ``std_error`` is
        ``nan`` for a single 2-link.
    n_two_links:
        Number of 2-links that contributed to the average (Step 5's sample size).
    n_two_links_found:
        Number found by Step 1, before dropping any with an empty common past.
        Differs from ``n_two_links`` only in the degenerate case.
    units:
        ``"links"`` if no ``rho`` was supplied (raw chain-link counts, uncalibrated)
        or ``"length"`` if eqs. (1)--(2) were applied.
    per_link_distance:
        The Step-3 value for each 2-link, in ``units``.
    per_link_chain_links:
        The same values as raw link counts, always (so the discreteness of the
        estimator is inspectable regardless of calibration).
    per_link_interval_size:
        Element count of each minimising interval ``[p, f_i]`` (endpoints
        excluded). Diagnoses the asymptotic-calibration caveat: eq. (1) is being
        applied at *these* interval sizes.
    n_past_candidates, n_past_scanned:
        Size of ``past(x) n past(y)`` and the number of its elements actually
        scanned in Step 2 (equal when ``exhaustive_past=True``; otherwise the
        maximal-element count).
    """

    mean: float
    std_error: float
    n_two_links: int
    n_two_links_found: int = 0
    units: str = "links"
    per_link_distance: np.ndarray = field(default_factory=lambda: np.empty(0))
    per_link_chain_links: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    per_link_interval_size: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    n_past_candidates: int = 0
    n_past_scanned: int = 0

    def as_tuple(self) -> tuple[float, float, int]:
        """``(mean, std_error, n_two_links)`` -- the Step-5 return of Section V.A."""
        return (self.mean, self.std_error, self.n_two_links)

    @property
    def mean_interval_size(self) -> float:
        """Mean ``|[p, f_i]|`` over the 2-links; the regime eq. (1) is used in."""
        if self.per_link_interval_size.size == 0:
            return float("nan")
        return float(self.per_link_interval_size.mean())


def two_link_distance(
    x: int,
    y: int,
    causal_matrix: np.ndarray,
    m_d: float = RW_M3,
    *,
    rho: float | None = None,
    d: int = SPACETIME_DIM_2P1,
    exhaustive_past: bool = False,
    require_spacelike: bool = True,
) -> TwoLinkDistanceResult:
    """Rideout--Wallden 2-link spacelike distance between ``x`` and ``y``.

    Section V.A, Steps 1--5 (enumerated in the module docstring). Returns a
    :class:`TwoLinkDistanceResult`; ``result.as_tuple()`` is the
    ``(mean, std_error, n_two_links)`` triple.

    Parameters
    ----------
    x, y:
        Indices of a spacelike (causally unrelated) pair.
    causal_matrix:
        ``C[i, j] = True`` iff ``i prec j`` (``order3d.causal_matrix_3d``).
    m_d:
        Asymptotic constant of eq. (1); defaults to the published ``m_3 = 2.296``.
        Only used when ``rho`` is given.
    rho:
        Sprinkling density. If ``None`` the distances are returned as raw chain
        **link** counts (``units="links"``), which is sufficient for any test of
        *stability* at fixed density and keeps the asymptotic calibration out of
        it. If given, eqs. (1)--(2) convert to a proper length
        (``units="length"``).
    d:
        Spacetime dimension; 3 for 2+1 D.
    exhaustive_past:
        Scan the whole of ``past(x) n past(y)`` in Step 2 instead of only its
        maximal elements. The two are provably identical (module docstring) and
        the tests assert it; the exhaustive path exists only as that check.
    require_spacelike:
        Raise if ``x`` and ``y`` are causally related. The construction estimates
        a *spatial* separation and is meaningless for a timelike pair.
    """
    cm = np.asarray(causal_matrix, dtype=bool)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"causal_matrix must be square, got shape {cm.shape}")
    if x == y:
        raise ValueError("two_link_distance needs two distinct elements")
    if require_spacelike and (cm[x, y] or cm[y, x]):
        raise ValueError(
            f"elements {x} and {y} are causally related; the 2-link construction "
            "estimates a spacelike distance"
        )

    # Step 1: the future 2-links.
    links_f = future_2links(x, y, cm)

    common_past = np.flatnonzero(cm[:, x] & cm[:, y])
    scan = common_past if exhaustive_past else maximal_common_past(x, y, cm)

    dists_links: list[int] = []
    interval_sizes: list[int] = []
    if scan.size:
        for f in links_f.tolist():
            # Step 2: minimise d(p, f) over the common past. Every p there
            # satisfies p prec x prec f, so p prec f by transitivity and every
            # candidate is usable.
            per_p = chain_links_to_target(f, scan, cm)
            best_at = int(per_p.argmin())
            # Step 3: store the minimised value for this 2-link.
            dists_links.append(int(per_p[best_at]))
            p_star = int(scan[best_at])
            interval_sizes.append(int(np.count_nonzero(cm[p_star, :] & cm[:, f])))
        # Step 4 is the loop above; Step 5 is the average below.

    raw = np.asarray(dists_links, dtype=int)
    n_used = int(raw.size)
    if n_used == 0:
        return TwoLinkDistanceResult(
            mean=float("nan"),
            std_error=float("nan"),
            n_two_links=0,
            n_two_links_found=int(links_f.size),
            units="links" if rho is None else "length",
            n_past_candidates=int(common_past.size),
            n_past_scanned=int(scan.size),
        )

    if rho is None:
        values = raw.astype(float)
        units = "links"
    else:
        scale = proper_time_from_chain_links(1.0, rho, m_d=m_d, d=d)
        values = raw.astype(float) * scale
        units = "length"

    mean = float(values.mean())
    sem = float(values.std(ddof=1) / np.sqrt(n_used)) if n_used > 1 else float("nan")
    return TwoLinkDistanceResult(
        mean=mean,
        std_error=sem,
        n_two_links=n_used,
        n_two_links_found=int(links_f.size),
        units=units,
        per_link_distance=values,
        per_link_chain_links=raw,
        per_link_interval_size=np.asarray(interval_sizes, dtype=int),
        n_past_candidates=int(common_past.size),
        n_past_scanned=int(scan.size),
    )


# ---------------------------------------------------------------------------
# The naive predecessor estimator -- the contrast that makes the gate mean
# something (Rideout--Wallden Section II.B).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NaiveDistanceResult:
    """Outcome of the *naive* spacelike distance -- the estimator RW replace.

    Attributes
    ----------
    distance, chain_links:
        The minimum, in ``units`` and as a raw link count.
    argmin:
        The ``(p, f)`` pair attaining it, for diagnostics.
    n_pairs:
        Number of ``(p, f)`` pairs examined after the exact maximal/minimal
        reduction; a proxy for how much boost freedom the region supplied.
    units:
        ``"links"`` or ``"length"``, as for :class:`TwoLinkDistanceResult`.
    """

    distance: float
    chain_links: int
    argmin: tuple[int, int] | None
    n_pairs: int
    units: str = "links"


def naive_distance(
    x: int,
    y: int,
    causal_matrix: np.ndarray,
    m_d: float = RW_M3,
    *,
    rho: float | None = None,
    d: int = SPACETIME_DIM_2P1,
) -> NaiveDistanceResult:
    """Naive spacelike distance: a single minimum over the whole common past AND future.

        d_naive(x, y) = min { d(p, f) : p in past(x) n past(y), f in fut(x) n fut(y) }

    Rideout--Wallden, Section II.B. In the continuum this *is* the spatial
    geodesic distance. Discretely it is exact enough in 1+1 D -- the minimising
    pair is essentially unique -- but for spacetime dimension ``d >= 3`` Lorentz
    boosts supply an unbounded family of pairs all attaining the continuum
    minimum, so enlarging the sprinkling region keeps offering fresh independent
    chances to fluctuate *downwards*. The minimum of ever more samples drifts
    steadily lower: the estimator has no large-region limit. Repairing exactly
    this is why Section V.A averages a minimum over 2-links instead of taking a
    double minimum.

    Implemented here purely as the **control** for the Fig.-14 stability gate.
    Without it, "the 2-link distance did not move" is not evidence -- a broken
    estimator could also fail to move. With it, the same sprinklings show one
    estimator drifting and the other not.

    Both reductions used are exact, not approximations:
      * a minimiser ``p`` is always a *maximal* element of the common past
        (a lower ``p`` only lengthens every chain -- module docstring);
      * a minimiser ``f`` is always a *minimal* element of the common future
        (the dual argument).

    Returns ``nan`` distance and ``n_pairs = 0`` if either set is empty.
    """
    cm = np.asarray(causal_matrix, dtype=bool)
    if x == y:
        raise ValueError("naive_distance needs two distinct elements")
    if cm[x, y] or cm[y, x]:
        raise ValueError(f"elements {x} and {y} are causally related")

    past_candidates = maximal_common_past(x, y, cm)
    future_candidates = minimal_common_future(x, y, cm)
    units = "links" if rho is None else "length"
    if past_candidates.size == 0 or future_candidates.size == 0:
        return NaiveDistanceResult(
            distance=float("nan"), chain_links=0, argmin=None, n_pairs=0, units=units
        )

    best = None
    arg: tuple[int, int] | None = None
    for f in future_candidates.tolist():
        per_p = chain_links_to_target(f, past_candidates, cm)
        at = int(per_p.argmin())
        if best is None or int(per_p[at]) < best:
            best, arg = int(per_p[at]), (int(past_candidates[at]), f)

    n_pairs = int(past_candidates.size * future_candidates.size)
    if rho is None:
        value = float(best)
    else:
        value = float(best) * proper_time_from_chain_links(1.0, rho, m_d=m_d, d=d)
    return NaiveDistanceResult(
        distance=value, chain_links=int(best), argmin=arg, n_pairs=n_pairs, units=units
    )
