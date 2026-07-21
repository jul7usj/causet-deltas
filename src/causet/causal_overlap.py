"""Boguñá--Krioukov causal-overlap spacelike distance (arXiv:2401.17376).

This is one of the two published baseline estimators the Delta-s proposal must be
benchmarked against (Phase 2a). It measures the *spacelike* separation between two
causally-unrelated events ``a`` and ``b`` using only the causal-set structure: the
number of elements in the Alexandrov intervals that ``a`` and ``b`` form with a
common past event ``c``, via the number--volume correspondence ``N ~ rho * V``
(Boguñá--Krioukov eqs. 16, 28).

Geometry and notation (following the paper).
--------------------------------------------
* Causal matrix ``C``: ``C[i, j] = True`` iff ``x_i prec x_j`` (``x_i`` in the
  strict causal past of ``x_j``). Irreflexive; ``a.s.`` antisymmetric for Poisson
  sprinklings. This is exactly the matrix produced by ``order.causal_matrix_1d``.
* Alexandrov interval (causal diamond) with future endpoint ``x`` and past
  endpoint ``y`` (requires ``y prec x``):
      I(x, y) = future(y) INTERSECT past(x)
              = { z : y prec z prec x } .
  Returned as an index set over the sprinkled elements (endpoints excluded, which
  is automatic since ``C`` is irreflexive). This is the number-volume proxy for
  the continuum interval volume.
* Overlap regions of two intervals sharing the past endpoint ``c`` (their
  eqs. 16/28 with a, b the two future endpoints):
      A = I(a, c) \\ I(b, c)      (elements below a but not below b)
      B = I(b, c) \\ I(a, c)
      C = I(a, c) INTERSECT I(b, c)
  and the discrete causal overlap
      O_C(a, b) = N[C] / ( min(N[A], N[B]) + N[C] )                    (eq. 16/28)
  which lies in [0, 1]: O = 1 for a timelike/comparable pair (one interval nests
  in the other so min(N[A], N[B]) = 0), and O -> 0 as a, b move far apart
  spacelike (the shared region shrinks relative to the private regions).
  NOTE on eq. 28: the paper writes the discrete overlap as ``N[C]/(N[A]+N[C])``
  having relabelled so that A is the smaller private region; using
  ``min(N[A], N[B])`` makes the estimator manifestly symmetric in (a, b) and
  matches the continuum eq. 16, so we use the ``min`` form.

Exact 1+1 D distance (d = 1), Boguñá--Krioukov eqs. 23--24.
-----------------------------------------------------------
In M^2 the overlap has the closed form (eq. 23)
      O_{M^2}(a, b) = exp( - d_{H^1}(a, b) / tau_c )
where ``d_{H^1}`` is a hyperbolic distance and ``tau_c`` is the proper time
("depth") of the common event ``c`` below the pair. Inverting to recover the
Minkowski (spacelike) distance (eq. 24):
      d_{M^2}(a, b) = tau_c * (1 - O) / sqrt(O) .
We use eq. 24 directly (closed form, d = 1) rather than the high-d asymptotic
linearisation. ``distance_from_overlap`` implements it; the hyperbolic distance is
also exposed for completeness via ``hyperbolic_distance_from_overlap``.

Proper-time (depth) estimator, Boguñá--Krioukov eq. 38.
-------------------------------------------------------
      tau_hat_c = 0.5 * alpha_d * rho^{-1/(d+1)} * ( n_C(c, a) + n_C(c, b) )
where ``n_C(c, x)`` is the causal-set *chain count* between ``c`` and ``x`` (the
longest-chain length, i.e. the discrete proper time from ``c`` to ``x``), and
``alpha_1 = 1/sqrt(2)`` exactly for d = 1 (their Section II.A).

Relationship of ``alpha_d`` to Phase 1's ``m_2`` (Rule 4 -- do not conflate).
    Phase 1 used the Brightwell--Gregory / Vershik--Kerov constant
    ``m_2 = 2``: the longest chain of ``N`` sprinkled points obeys
    ``L ~ m_2 * sqrt(N)``. Boguñá--Krioukov's ``alpha_d`` instead relates the
    chain length directly to continuum proper time:
    ``tau = alpha_d * rho^{-1/(d+1)} * L``. In d = 1 the two are the SAME fact in
    different clothing: with ``N = rho * tau^2 / 2`` (the 1+1 D diamond volume),
    ``L = m_2 sqrt(N) = m_2 tau sqrt(rho/2)`` gives
    ``tau = L / (m_2 sqrt(rho/2)) = (sqrt(2)/m_2) * rho^{-1/2} * L``, hence
        alpha_1 = sqrt(2) / m_2 = sqrt(2)/2 = 1/sqrt(2).
    We therefore hard-code ``alpha_1 = 1/sqrt(2)`` and never reuse ``m_2 = 2``
    inside this module: they are two normalisations of one law and mixing them
    would double-count the calibration.

Event-``c`` selection (their Section IV.A double filter).
---------------------------------------------------------
For a target pair (a, b) one averages the distance estimate over admissible common
past events ``c``. Two filters restrict which ``c`` are used:
  * Filter 1 (eq. 32, a *speed shortcut* that needs knowledge of d and rho):
        | n_C(c, a) - n_C(c, b) | < rho^{ beta_d / (d+1) } .
  * Filter 2 (eq. 34, the *intrinsic* claim -- uses only causal-set counts):
        | Z_c(a, b) | < kappa * sqrt( N[A u C] + N[B u C] ) * sqrt( 1 - O_C ),
    with  Z_c(a, b) = N[A u C] - N[B u C] = |I(a, c)| - |I(b, c)|,
    and kappa a prefactor the paper sets to 1/2 (adjustable; documented parameter
    ``KAPPA_FILTER2`` below). Filter 2 keeps only events ``c`` that see ``a`` and
    ``b`` symmetrically (comparable interval cardinalities) to within the
    statistical spread expected at that overlap.
We implement Filter 2 as the selection actually used in the acceptance test (the
paper's intrinsic result); Filter 1 is provided but flagged as the shortcut.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import order

# ---------------------------------------------------------------------------
# Named, documented parameters (Scientific Integrity Rule 4).
# ---------------------------------------------------------------------------

#: Exact d = 1 proper-time-per-chain normalisation, Boguñá--Krioukov Section II.A.
#: Equals sqrt(2)/m_2 with the Phase-1 constant m_2 = 2 (see module docstring).
ALPHA_1 = 1.0 / np.sqrt(2.0)

#: Prefactor kappa in the Filter-2 symmetry test, eq. 34. The paper uses 1/2 and
#: states it is adjustable. Exposed so the acceptance test can report sensitivity.
KAPPA_FILTER2 = 0.5

#: Exponent beta_d in the Filter-1 shortcut, eq. 32. The fetched text does not pin
#: a numeric value; Filter 1 is only a speed shortcut in the paper and is NOT used
#: to select c in our acceptance test, so this default is documented-but-unvalidated.
#: (beta_d ~ d/(d+1) reproduces the "difference grows sub-linearly with density"
#: intent; treat any Filter-1 result as illustrative only.)
BETA_1_UNVALIDATED = 0.5


# ---------------------------------------------------------------------------
# Alexandrov intervals and overlap.
# ---------------------------------------------------------------------------


def past_of(x: int, c: np.ndarray) -> np.ndarray:
    """Indices strictly in the causal past of element ``x``: ``{ i : i prec x }``."""
    c = np.asarray(c, dtype=bool)
    return np.nonzero(c[:, x])[0]


def future_of(y: int, c: np.ndarray) -> np.ndarray:
    """Indices strictly in the causal future of element ``y``: ``{ j : y prec j }``."""
    c = np.asarray(c, dtype=bool)
    return np.nonzero(c[y, :])[0]


def alexandrov_interval(x: int, y: int, c: np.ndarray) -> np.ndarray:
    """Alexandrov interval ``I(x, y) = { z : y prec z prec x }`` (future x, past y).

    Requires ``y prec x``; returns a sorted index array of the interior elements
    (the endpoints ``x`` and ``y`` are excluded automatically, ``C`` being
    irreflexive). If ``y`` does not precede ``x`` the interval is empty by
    definition and an empty array is returned.

    Boguñá--Krioukov use the element count of this set as the number-volume proxy
    for the continuum interval volume (eqs. 16, 28).
    """
    c = np.asarray(c, dtype=bool)
    # z in future(y) AND z in past(x): C[y, z] and C[z, x].
    both = c[y, :] & c[:, x]
    return np.nonzero(both)[0]


@dataclass(frozen=True)
class OverlapPartition:
    """The three overlap regions of ``I(a, c)`` and ``I(b, c)`` as index sets.

    Attributes
    ----------
    A, B, C_shared:
        Index arrays for ``A = I(a,c)\\I(b,c)``, ``B = I(b,c)\\I(a,c)`` and
        ``C = I(a,c) INTERSECT I(b,c)`` respectively (paper's A, B, C).
    n_ac, n_bc:
        Cardinalities ``|I(a,c)| = N[A u C]`` and ``|I(b,c)| = N[B u C]``
        (used by the eq. 38 estimator and the Filter-2 test).
    """

    A: np.ndarray
    B: np.ndarray
    C_shared: np.ndarray
    n_ac: int
    n_bc: int

    @property
    def n_A(self) -> int:  # noqa: N802 (match paper's N[A])
        return int(self.A.size)

    @property
    def n_B(self) -> int:  # noqa: N802
        return int(self.B.size)

    @property
    def n_C(self) -> int:  # noqa: N802
        return int(self.C_shared.size)


def overlap_partition(a: int, b: int, c: int, causal_matrix: np.ndarray) -> OverlapPartition:
    """Partition ``I(a,c)`` and ``I(b,c)`` into the regions A, B, C of eq. 16.

    ``c`` must be a common past of both ``a`` and ``b`` (``c prec a`` and
    ``c prec b``); otherwise one or both intervals are empty and the partition
    degenerates (all counts zero), which callers should detect and skip.
    """
    cm = np.asarray(causal_matrix, dtype=bool)
    # Boolean membership masks (numpy, no Python sets): z in I(a,c) iff c prec z
    # prec a, i.e. C[c, z] and C[z, a]. Vectorised for speed at large N.
    mask_ac = cm[c, :] & cm[:, a]
    mask_bc = cm[c, :] & cm[:, b]
    A = np.nonzero(mask_ac & ~mask_bc)[0]
    B = np.nonzero(mask_bc & ~mask_ac)[0]
    C_shared = np.nonzero(mask_ac & mask_bc)[0]
    return OverlapPartition(
        A=A,
        B=B,
        C_shared=C_shared,
        n_ac=int(mask_ac.sum()),
        n_bc=int(mask_bc.sum()),
    )


def causal_overlap(a: int, b: int, c: int, causal_matrix: np.ndarray) -> float:
    """Discrete causal overlap ``O_C(a, b)`` seen from common past ``c`` (eq. 16/28).

        O_C(a, b) = N[C] / ( min(N[A], N[B]) + N[C] ) ,   in [0, 1].

    Returns ``1.0`` for a comparable (timelike/null) pair -- one interval nests in
    the other so ``min(N[A], N[B]) = 0`` -- and ``nan`` when both intervals are
    empty (``c`` is not a usable common ancestor: ``N[C] = 0`` and
    ``min(N[A], N[B]) = 0``), which the caller must skip rather than treat as a
    measurement.
    """
    part = overlap_partition(a, b, c, causal_matrix)
    n_c = part.n_C
    denom = min(part.n_A, part.n_B) + n_c
    if denom == 0:
        return float("nan")
    return float(n_c) / float(denom)


# ---------------------------------------------------------------------------
# Distance recovery (eqs. 23--24) and depth estimator (eq. 38).
# ---------------------------------------------------------------------------


def distance_from_overlap(overlap: float, tau_c: float) -> float:
    """Exact 1+1 D spacelike distance from overlap and depth (eq. 24).

        d_{M^2}(a, b) = tau_c * (1 - O) / sqrt(O) .

    ``overlap`` must be in ``(0, 1]``. ``O = 1`` (timelike/comparable) gives 0;
    ``O -> 0`` gives ``+inf``; ``O <= 0`` returns ``+inf``.
    """
    if tau_c <= 0:
        raise ValueError(f"tau_c must be positive, got {tau_c}")
    if not np.isfinite(overlap):
        return float("nan")
    if overlap <= 0.0:
        return float("inf")
    return float(tau_c) * (1.0 - overlap) / np.sqrt(overlap)


def hyperbolic_distance_from_overlap(overlap: float, tau_c: float) -> float:
    """Hyperbolic distance ``d_{H^1} = -tau_c * ln(O)`` (inverse of eq. 23).

    Provided for completeness; the physical spacelike separation the benchmark
    compares against is ``distance_from_overlap`` (eq. 24).
    """
    if tau_c <= 0:
        raise ValueError(f"tau_c must be positive, got {tau_c}")
    if not np.isfinite(overlap) or overlap <= 0.0:
        return float("nan")
    return -float(tau_c) * float(np.log(overlap))


def overlap_predicted_from_distance(distance: float, tau_c: float) -> float:
    """Invert eq. 24 to predict the overlap for a *known* distance and depth.

    Solving ``d = tau_c (1 - O)/sqrt(O)`` for ``w = sqrt(O)`` gives the quadratic
    ``tau_c w^2 + d w - tau_c = 0`` with the physical (positive) root
        w = ( -d + sqrt(d^2 + 4 tau_c^2) ) / (2 tau_c),   O = w^2 .
    Used by the acceptance test to compare the *measured* mean overlap against the
    value eq. 24 demands for the known continuum separation -- a direct test of
    the closed form, independent of any embedding-based hyperbolic-distance
    calculation.
    """
    if tau_c <= 0:
        raise ValueError(f"tau_c must be positive, got {tau_c}")
    d = float(distance)
    w = (-d + np.sqrt(d * d + 4.0 * tau_c * tau_c)) / (2.0 * tau_c)
    return float(w * w)


def chain_count(c_past: int, x_future: int, causal_matrix: np.ndarray) -> int:
    """Causal-set chain count ``n_C(c, x)`` = longest-chain length from ``c`` to ``x``.

    This is the discrete proper time between ``c`` and ``x`` (Brightwell--Gregory).
    Computed on the sub-poset spanned by ``{c} u I(x, c) u {x}`` (endpoints
    included, so the chain runs corner-to-corner) using the Phase-1 dense
    ``longest_chain_length``. Returns the number of *elements* in the longest such
    chain (so a bare link ``c prec x`` with empty interval gives 2), consistent
    with the ``alpha_1 = 1/sqrt(2)`` normalisation (see module docstring).

    Requires ``c prec x``; raises if not (a chain count is undefined otherwise).
    """
    cm = np.asarray(causal_matrix, dtype=bool)
    if not cm[c_past, x_future]:
        raise ValueError(
            f"chain_count requires c ({c_past}) prec x ({x_future}); they are not "
            "causally ordered that way"
        )
    interior = alexandrov_interval(x_future, c_past, cm)
    idx = np.concatenate(([c_past], interior, [x_future])).astype(int)
    sub = cm[np.ix_(idx, idx)]
    return order.longest_chain_length(sub)


def estimate_tau_c(
    a: int,
    b: int,
    c: int,
    causal_matrix: np.ndarray,
    rho: float,
    *,
    d: int = 1,
    alpha_d: float = ALPHA_1,
    chain_count_fn=None,
) -> float:
    """Estimate the depth ``tau_c`` of common event ``c`` below the pair (eq. 38).

        tau_hat_c = 0.5 * alpha_d * rho^{-1/(d+1)} * ( n_C(c, a) + n_C(c, b) ) .

    Averages the two discrete proper times ``c -> a`` and ``c -> b`` (which the
    Filter-2 selection makes comparable). For d = 1 the default ``alpha_d`` is the
    exact ``ALPHA_1 = 1/sqrt(2)``.

    ``chain_count_fn``:
        Optional ``(c_idx, x_idx) -> int`` returning the chain count ``n_C(c, x)``.
        The default (None) uses the matrix-based ``chain_count`` (the intrinsic,
        dimension-agnostic definition). In 1+1 D the null-coordinate LIS gives the
        SAME number in O(N log N) instead of O(N^2) (Phase 1 verified this
        equivalence); ``experiments/exp01`` injects it purely for speed -- it
        changes no physics.
    """
    if rho <= 0:
        raise ValueError(f"rho must be positive, got {rho}")
    if chain_count_fn is None:
        n_ca = chain_count(c, a, causal_matrix)
        n_cb = chain_count(c, b, causal_matrix)
    else:
        n_ca = chain_count_fn(c, a)
        n_cb = chain_count_fn(c, b)
    return 0.5 * alpha_d * rho ** (-1.0 / (d + 1)) * (n_ca + n_cb)


# ---------------------------------------------------------------------------
# Event-c selection filters (Section IV.A).
# ---------------------------------------------------------------------------


def common_past(a: int, b: int, causal_matrix: np.ndarray) -> np.ndarray:
    """Indices ``c`` with ``c prec a`` AND ``c prec b`` (candidate common events)."""
    cm = np.asarray(causal_matrix, dtype=bool)
    return np.nonzero(cm[:, a] & cm[:, b])[0]


def filter2_passes(
    a: int,
    b: int,
    c: int,
    causal_matrix: np.ndarray,
    *,
    kappa: float = KAPPA_FILTER2,
) -> bool:
    """Intrinsic Filter 2 (eq. 34): keep ``c`` if it sees a, b symmetrically.

        | N[A u C] - N[B u C] | < kappa * sqrt( N[A u C] + N[B u C] ) * sqrt(1 - O)

    with ``N[A u C] = |I(a,c)|`` and ``N[B u C] = |I(b,c)|``. Uses only
    causal-set counts (no embedding). Returns False if either interval is empty
    (``c`` is not a usable common ancestor) or the overlap is not finite.
    """
    part = overlap_partition(a, b, c, causal_matrix)
    if part.n_ac == 0 or part.n_bc == 0:
        return False
    o = causal_overlap(a, b, c, causal_matrix)
    if not np.isfinite(o):
        return False
    z = abs(part.n_ac - part.n_bc)
    rhs = kappa * np.sqrt(part.n_ac + part.n_bc) * np.sqrt(max(0.0, 1.0 - o))
    return bool(z < rhs)


def filter1_passes(
    a: int,
    b: int,
    c: int,
    causal_matrix: np.ndarray,
    rho: float,
    *,
    d: int = 1,
    beta_d: float = BETA_1_UNVALIDATED,
) -> bool:
    """Filter 1 shortcut (eq. 32): ``|n_C(c,a) - n_C(c,b)| < rho^{beta_d/(d+1)}``.

    Provided for completeness only. ``beta_d`` is NOT pinned by the source text we
    could access (see ``BETA_1_UNVALIDATED``); the acceptance test selects ``c``
    with Filter 2, never this. Treat any Filter-1 output as illustrative.
    """
    if rho <= 0:
        raise ValueError(f"rho must be positive, got {rho}")
    n_ca = chain_count(c, a, causal_matrix)
    n_cb = chain_count(c, b, causal_matrix)
    threshold = rho ** (beta_d / (d + 1))
    return bool(abs(n_ca - n_cb) < threshold)


# ---------------------------------------------------------------------------
# Full estimator: average the eq.-24 distance over Filter-2-admissible events c.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OverlapDistanceResult:
    """Outcome of the causal-overlap distance estimate for one pair on one causet.

    Attributes
    ----------
    distance:
        Mean of the per-``c`` eq.-24 estimates (nan if no admissible ``c``).
    sem:
        Standard error of the mean over admissible ``c`` (nan if < 2).
    n_c:
        Number of admissible common events used (report this: a small count means
        the estimate is unreliable -- an honest finding, not something to hide).
    per_c_distance, per_c_overlap, per_c_tau:
        Arrays over the admissible events (for diagnostics / plotting).
    n_candidates:
        Number of common-past events examined before filtering.
    """

    distance: float
    sem: float
    n_c: int
    per_c_distance: np.ndarray = field(default_factory=lambda: np.empty(0))
    per_c_overlap: np.ndarray = field(default_factory=lambda: np.empty(0))
    per_c_tau: np.ndarray = field(default_factory=lambda: np.empty(0))
    n_candidates: int = 0


def distance_causal_overlap(
    a: int,
    b: int,
    causal_matrix: np.ndarray,
    rho: float,
    *,
    d: int = 1,
    alpha_d: float = ALPHA_1,
    kappa: float = KAPPA_FILTER2,
    require_overlap_in_open_unit: bool = True,
    chain_count_fn=None,
) -> OverlapDistanceResult:
    """Estimate the spacelike distance between ``a`` and ``b`` (full B--K pipeline).

    Steps (their Section IV.A, intrinsic Filter 2):
      1. Enumerate common past events ``c`` (``c prec a`` and ``c prec b``).
      2. Keep those passing Filter 2 (eq. 34) -- symmetric vantage points.
      3. For each, measure ``O_C`` (eq. 28), estimate ``tau_c`` (eq. 38), and map
         to a distance via eq. 24.
      4. Average over admissible ``c``; report the mean, its standard error, and
         the count actually used.

    ``require_overlap_in_open_unit`` drops events with ``O`` not in ``(0, 1)``:
    ``O = 1`` (comparable pair -- not spacelike) gives distance 0 and would bias
    the mean, and ``O = 0`` gives an infinite distance; both are excluded from the
    average and would signal that ``c`` is a poor vantage.
    """
    cm = np.asarray(causal_matrix, dtype=bool)
    candidates = common_past(a, b, cm)
    dists: list[float] = []
    overlaps: list[float] = []
    taus: list[float] = []
    for c in candidates.tolist():
        if not filter2_passes(a, b, c, cm, kappa=kappa):
            continue
        o = causal_overlap(a, b, c, cm)
        if not np.isfinite(o):
            continue
        if require_overlap_in_open_unit and not (0.0 < o < 1.0):
            continue
        tau = estimate_tau_c(
            a, b, c, cm, rho, d=d, alpha_d=alpha_d, chain_count_fn=chain_count_fn
        )
        if tau <= 0:
            continue
        dists.append(distance_from_overlap(o, tau))
        overlaps.append(o)
        taus.append(tau)

    arr = np.asarray(dists, dtype=float)
    n_c = int(arr.size)
    if n_c == 0:
        return OverlapDistanceResult(
            distance=float("nan"),
            sem=float("nan"),
            n_c=0,
            n_candidates=int(candidates.size),
        )
    mean = float(arr.mean())
    sem = float(arr.std(ddof=1) / np.sqrt(n_c)) if n_c > 1 else float("nan")
    return OverlapDistanceResult(
        distance=mean,
        sem=sem,
        n_c=n_c,
        per_c_distance=arr,
        per_c_overlap=np.asarray(overlaps, dtype=float),
        per_c_tau=np.asarray(taus, dtype=float),
        n_candidates=int(candidates.size),
    )
