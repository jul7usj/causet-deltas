"""Boguna--Krioukov causal overlap in 2+1 D (d = 2) -- arXiv:2401.17376.

Phase 2b Part B1. Companion to ``causal_overlap.py`` (1+1 D, Phase 2a, accepted
and **unmodified**), in the same relation that ``order3d.py`` bears to
``order.py``: this module re-exports the parts of Phase 2a that are already
dimension-agnostic and supplies only what genuinely changes with dimension.

Which piece is dimension-agnostic, and which is not
---------------------------------------------------
* **The overlap RATIO is dimension-agnostic.** Boguna--Krioukov eqs. 16/28,

      O_C(a, b) = N[C] / ( min(N[A], N[B]) + N[C] ) ,

  is built purely from Alexandrov-interval cardinalities. It needs a causal
  matrix and nothing else, so ``alexandrov_interval``, ``overlap_partition``,
  ``causal_overlap``, ``common_past``, ``filter2_passes`` and ``chain_count``
  are re-exported from Phase 2a verbatim and work on
  ``order3d.causal_matrix_3d`` output unchanged.
* **The DISTANCE formula is NOT.** Phase 2a uses their exact closed form
  (eqs. 23--24), ``d = tau_c (1 - O)/sqrt(O)``, which is derived for d = 1 and
  is d = 1 ONLY. In d = 2 the corresponding statement is the *asymptotic* form
  of eqs. 25--27 (see below).
* **The proper-time calibration is NOT.** eq. 38 needs ``alpha_d``, and
  ``alpha_1 = 1/sqrt(2)`` does not carry over.

NOTATION CLASH -- read this before touching any exponent
--------------------------------------------------------
Boguna--Krioukov's ``d`` is the number of **spatial** dimensions: d = 1 for
M^2 (Phase 2a), d = 2 for M^3 (here). Rideout--Wallden's ``d``, used throughout
``order3d.py``, ``sprinkle3d.py`` and ``rideout_wallden.py``, is the
**spacetime** dimension: 2 for M^2, 3 for M^3. Both papers write the letter
``d``. Throughout this module ``d`` means the B--K spatial dimension and
``spacetime_dim = d + 1`` means the R--W one; every call across the boundary
goes through ``spacetime_dim_from_spatial`` so the conversion is never done
inline. Getting this wrong swaps a cube root for a square root inside
``rho^{-1/(d+1)}`` and corrupts every distance with no failing test.

The d >= 2 distance: asymptotic, eqs. 25--27
---------------------------------------------
For general d the overlap is not invertible in closed form; B--K give instead
the leading behaviour for a pair whose separation is small compared with the
depth of the common event ``c``:

    d(a, b) = (2 / c_d) * tau_c * (1 - O(a, b)) ,     valid for tau_c >> d(a,b)

    c_d = (d + 1)/sqrt(pi) * Gamma(d/2) / Gamma((d+1)/2)                 (eq. 27)

DERIVATION of the value used here (Integrity Rule 4 -- not hardcoded; computed
by ``overlap_coefficient_c`` and cross-checked against ``scipy.special.gamma``
by ``tests/test_causal_overlap_3d.py``):

    c_2 = 3/sqrt(pi) * Gamma(1) / Gamma(3/2)
        = 3/sqrt(pi) * 1 / (sqrt(pi)/2)
        = 3 * 2 / (sqrt(pi) * sqrt(pi))
        = 6/pi
        = 1.9098593171027443...

so the 2+1 D prefactor is  2/c_2 = pi/3 = 1.0471975511965976.

INTERNAL CONSISTENCY CHECK, independent of the paper's text. In d = 1 the same
formula gives

    c_1 = 2/sqrt(pi) * Gamma(1/2) / Gamma(1) = 2/sqrt(pi) * sqrt(pi) = 2 ,

so the asymptotic form reads ``d = tau_c (1 - O)``, which is exactly the
``O -> 1`` limit of Phase 2a's exact eq. 24 ``d = tau_c (1 - O)/sqrt(O)``
(they differ by the factor ``1/sqrt(O) -> 1``). The general ``c_d`` therefore
reproduces the independently validated d = 1 normalisation in the regime where
it is supposed to, which is a check on the *factor* that no test of the d = 2
code alone could provide. ``tests/test_causal_overlap_3d.py`` asserts it.

WEAKER ACCURACY CLAIM THAN PHASE 2a -- state this wherever a 2+1 D B--K number
is quoted. Phase 2a's d = 1 result rests on an exact inversion valid at every
overlap. The 2+1 D number here is a leading-order estimate whose error grows as
the pair separation approaches ``tau_c``. The validity of the regime is not
assumed: ``OverlapDistanceResultND`` records the realised ``tau_c / d_est``
ratio for every admissible ``c``, and the experiments report its distribution so
a reader can see how far into the asymptotic regime the measurement actually
sat. ``d = 1`` inputs are dispatched to the exact eq. 24 and are unaffected.

Proper-time calibration in 2+1 D: alpha_2 from the measured m_3
----------------------------------------------------------------
eq. 38 estimates the depth of the common event ``c`` from chain counts:

    tau_hat_c = 0.5 * alpha_d * rho^{-1/(d+1)} * ( n_C(c,a) + n_C(c,b) ) .

``alpha_d`` is fixed by the same chain law Phase 1 and Part 1 measured, written
in the other paper's normalisation. Rideout--Wallden eq. (1) with eq. (2), for
an Alexandrov interval whose endpoints are separated by proper time ``l`` in
spacetime dimension ``D``:

    L (rho V)^{-1/D} -> m_D ,   V = eta(D) l^D
      =>  L = m_D * l * (rho eta(D))^{1/D}
      =>  l = [ 1 / ( m_D eta(D)^{1/D} ) ] * rho^{-1/D} * L .

Comparing with B--K's ``tau = alpha_d rho^{-1/(d+1)} L`` and using D = d + 1:

    alpha_d = 1 / ( m_D * eta(D)^{1/D} ) ,        D = d + 1.           (*)

``alpha_d_from_m`` implements (*). It reproduces Phase 2a exactly: with the
Brightwell--Gregory constant ``m_2 = 2`` and ``eta(2) = 1/2``,

    alpha_1 = 1/(2 * (1/2)^{1/2}) = sqrt(2)/2 = 1/sqrt(2) ,

which is the ``ALPHA_1`` Phase 2a hardcodes, and is the *same* identity its
docstring records as ``alpha_1 = sqrt(2)/m_2``. (In IEEE-754 the two spellings
differ in the last ulp -- 1.1e-16 relative -- so the d = 1 path keeps Phase 2a's
literal ``ALPHA_1`` and ``tests/test_causal_overlap_3d.py`` asserts the general
formula agrees with it to 1 ulp. Nothing physical hangs on this; it is what lets
the 1+1 D regression test demand *bit* identity rather than a tolerance.)

WHICH m_3 -- and why not 2.296
-------------------------------
The published ``m_3 = 2.296 +/- 0.012`` is the **asymptote** of Rideout--Wallden's
Fig.-4 fit, supported only over ``rho V = 2^10 .. 2^18``. Part 1 (Finding 4,
2026-08-10) recorded that ``m_3 + a N^c`` is an effective description over that
finite window, not a true asymptotic expansion; Part A's diagnostic then measured
the cost of ignoring that -- reading the fit at ``rho V ~ 37`` understates
``m_3^eff`` by 0.09--0.14 (2026-08-17 entry). eq. 38 is applied to Alexandrov
intervals of whatever size the sprinkling supplies, which is emphatically not the
asymptotic regime, so this module calibrates against the **measured**
``m_3^eff(rho V)`` curve and never against 2.296.

``M3_EFF_MEASURED`` below is that curve, assembled from the two committed
datasets of this repository and from nothing else:

  * ``data/exp03b_diagnostic.npz`` (Part A diagnostic, 4000 plain M^3 diamonds
    per point, link convention, unconditioned) at ``rho V = 16, 24, 32, 40, 48,
    64``;
  * ``data/exp02_measurements.npz`` (Part 1 gate) at ``rho V = 2^10 .. 2^17``,
    re-reduced here with ``L = elements - 1``.

``tests/test_causal_overlap_3d.py`` re-derives the table from those files and
asserts the literals match, so the constants cannot silently drift from the data
they claim to come from.

There is a genuine, unmeasured gap between ``rho V = 64`` and ``rho V = 1024``
(four octaves). ``m3_effective`` interpolates log-linearly in ``log2(rho V)``
across it and *says so* in its returned status; outside [16, 131072] it clamps
and says so. A caller that lands in the gap or outside the range has been told.

How alpha enters the answer -- and why the gate survives it
------------------------------------------------------------
``tau_hat_c`` is linear in ``alpha_d`` and the distance is linear in
``tau_hat_c``, so a **fixed** ``alpha_d`` is a pure global scale factor: it
cannot change the rank ordering of a set of pairs at all. A **size-dependent**
``alpha_d`` (policy ``"measured_m3"``, reading ``m_3^eff`` at each interval's own
cardinality) is pair-dependent and therefore *can* move the ordering. Both are
implemented; ``experiments/exp04_rw_vs_bk_2p1d.py`` reports whether the choice
changes any rank, which is the only way the question is settled rather than
assumed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .causal_overlap import (
    ALPHA_1,
    KAPPA_FILTER2,
    OverlapPartition,
    alexandrov_interval,
    causal_overlap,
    chain_count,
    common_past,
    distance_from_overlap,
    filter2_passes,
    future_of,
    hyperbolic_distance_from_overlap,
    overlap_partition,
    overlap_predicted_from_distance,
    past_of,
)
from .sprinkle3d import interval_volume_constant

__all__ = [
    # re-exported, dimension-agnostic (Phase 2a, unchanged)
    "ALPHA_1",
    "KAPPA_FILTER2",
    "OverlapPartition",
    "alexandrov_interval",
    "causal_overlap",
    "chain_count",
    "common_past",
    "distance_from_overlap",
    "filter2_passes",
    "future_of",
    "hyperbolic_distance_from_overlap",
    "overlap_partition",
    "overlap_predicted_from_distance",
    "past_of",
    # new in Part B1
    "SPATIAL_DIM_2P1",
    "spacetime_dim_from_spatial",
    "overlap_coefficient_c",
    "C_D_1",
    "C_D_2",
    "distance_from_overlap_asymptotic",
    "distance_from_overlap_nd",
    "alpha_d_from_m",
    "M3_EFF_MEASURED",
    "M3_EFF_RHO_V_MIN",
    "M3_EFF_RHO_V_MAX",
    "M3_EFF_GAP_LO",
    "M3_EFF_GAP_HI",
    "M3Effective",
    "m3_effective",
    "alpha_2_from_interval_size",
    "ALPHA_2_AT_RHO_V_64",
    "ALPHA_2_FROM_PUBLISHED_M3",
    "estimate_tau_c_nd",
    "OverlapDistanceResultND",
    "distance_causal_overlap_nd",
]

#: Number of *spatial* dimensions of M^3 in Boguna--Krioukov's convention.
#: (Rideout--Wallden would call this spacetime "d = 3"; see the notation clash
#: note in the module docstring.)
SPATIAL_DIM_2P1 = 2


def spacetime_dim_from_spatial(d: int) -> int:
    """Rideout--Wallden spacetime dimension ``D = d + 1`` from B--K spatial ``d``.

    The single place the two papers' ``d`` are converted. d = 1 (M^2) -> D = 2;
    d = 2 (M^3) -> D = 3.
    """
    if d < 1:
        raise ValueError(f"spatial dimension d must be >= 1, got {d}")
    return d + 1


# ---------------------------------------------------------------------------
# c_d: the overlap-to-distance coefficient of eqs. 25--27.
# ---------------------------------------------------------------------------


def overlap_coefficient_c(d: int = SPATIAL_DIM_2P1) -> float:
    """Boguna--Krioukov eq. 27 coefficient ``c_d``.

        c_d = (d + 1)/sqrt(pi) * Gamma(d/2) / Gamma((d+1)/2)

    ``d`` is the number of **spatial** dimensions. Computed from
    ``math.gamma``, never hardcoded (Integrity Rule 4); the closed forms
    ``c_1 = 2`` and ``c_2 = 6/pi`` derived in the module docstring are asserted
    against this function, and this function against ``scipy.special.gamma``, in
    ``tests/test_causal_overlap_3d.py``.
    """
    if d < 1:
        raise ValueError(f"spatial dimension d must be >= 1, got {d}")
    return (d + 1.0) / math.sqrt(math.pi) * math.gamma(d / 2.0) / math.gamma((d + 1.0) / 2.0)


#: ``c_1 = 2`` exactly. Only used to state the consistency check with eq. 24.
C_D_1 = overlap_coefficient_c(1)

#: ``c_2 = 6/pi = 1.90986...`` -- the 2+1 D coefficient (derivation in docstring).
C_D_2 = overlap_coefficient_c(SPATIAL_DIM_2P1)


def distance_from_overlap_asymptotic(
    overlap: float, tau_c: float, d: int = SPATIAL_DIM_2P1
) -> float:
    """Asymptotic B--K spacelike distance, eqs. 25--27.

        d(a, b) = (2 / c_d) * tau_c * (1 - O(a, b)) ,   valid for tau_c >> d(a,b).

    In d = 2 the prefactor is ``2/c_2 = pi/3``.

    This is a **leading-order** expression: unlike Phase 2a's eq. 24 it is not an
    exact inversion, and its error grows as the separation approaches ``tau_c``.
    Callers must record the realised ``tau_c / d`` ratio;
    ``distance_causal_overlap_nd`` does so per admissible ``c``.

    ``overlap`` is not clipped: ``O = 1`` (comparable pair) gives 0 and ``O = 0``
    gives ``(2/c_d) tau_c``, which -- unlike eq. 24's ``+inf`` -- is finite
    precisely because the linearisation has broken down there. That is a property
    of the approximation, not a repair of it, and is why the pipeline still drops
    ``O`` outside ``(0, 1)``.
    """
    if tau_c <= 0:
        raise ValueError(f"tau_c must be positive, got {tau_c}")
    if not np.isfinite(overlap):
        return float("nan")
    return (2.0 / overlap_coefficient_c(d)) * float(tau_c) * (1.0 - float(overlap))


def distance_from_overlap_nd(
    overlap: float, tau_c: float, d: int = SPATIAL_DIM_2P1
) -> float:
    """Overlap -> spacelike distance, dispatching on the spatial dimension.

    * ``d == 1``: Phase 2a's **exact** closed form, eq. 24
      (``causal_overlap.distance_from_overlap``) -- called, not reimplemented, so
      1+1 D results are bit-identical to Phase 2a's.
    * ``d >= 2``: the asymptotic eqs. 25--27 above.

    The exact form is preferred wherever it exists; the approximation is used
    only where B--K supply nothing better.
    """
    if d == 1:
        return distance_from_overlap(overlap, tau_c)
    return distance_from_overlap_asymptotic(overlap, tau_c, d)


# ---------------------------------------------------------------------------
# alpha_d from the chain law (derivation (*) in the module docstring).
# ---------------------------------------------------------------------------


def alpha_d_from_m(m_spacetime: float, d: int = SPATIAL_DIM_2P1) -> float:
    """``alpha_d = 1 / ( m_D * eta(D)^{1/D} )`` with ``D = d + 1`` -- eq. (*).

    Converts a Rideout--Wallden chain constant ``m_D`` (their eq. (1), spacetime
    dimension ``D``) into the Boguna--Krioukov proper-time-per-chain constant
    ``alpha_d`` of their eq. 38 (spatial dimension ``d``). Full derivation in the
    module docstring; ``eta(D)`` comes from
    ``sprinkle3d.interval_volume_constant`` (computed, not hardcoded).

    Sanity values asserted by the tests:
        ``alpha_d_from_m(2.0, d=1)   = 1/sqrt(2)``  (Phase 2a's ALPHA_1, to 1 ulp)
        ``alpha_d_from_m(2.296, d=2) = 0.68083...`` (published m_3 -- NOT used;
        see "WHICH m_3" in the module docstring).
    """
    if m_spacetime <= 0:
        raise ValueError(f"m_spacetime must be positive, got {m_spacetime}")
    big_d = spacetime_dim_from_spatial(d)
    eta = interval_volume_constant(big_d)
    return 1.0 / (m_spacetime * eta ** (1.0 / big_d))


# ---------------------------------------------------------------------------
# The MEASURED m_3^eff(rho V) curve (this repository's own data, not a fit).
# ---------------------------------------------------------------------------

#: Measured effective chain constant ``m_3^eff = L (rho V)^{-1/3}`` for M^3
#: Alexandrov intervals, link convention, endpoints included, unconditioned.
#: Rows are ``(rho V, m_3^eff, standard error)``, sorted by ``rho V``.
#:
#: Provenance (Integrity Rule 4) -- both files are committed to this repository
#: and ``tests/test_causal_overlap_3d.py`` re-derives these numbers from them:
#:   rho V =  16 .. 64      ``data/exp03b_diagnostic.npz`` keys ``base_rho_v``,
#:                          ``base_m_eff``, ``base_m_eff_se`` (Part A diagnostic,
#:                          4000 diamonds per point, seed 30820262).
#:   rho V = 2^10 .. 2^17   ``data/exp02_measurements.npz`` keys ``elements_K``
#:                          (Part 1 gate, seeds 20260810 + 1000*log2N + k),
#:                          reduced as ``m = (elements - 1) / (2^K)^(1/3)`` with
#:                          the mean and standard error taken over realisations.
#: This is deliberately NOT Rideout--Wallden's fitted asymptote 2.296, nor their
#: fit evaluated anywhere; see "WHICH m_3" in the module docstring.
#: Literals are the full-precision reductions of those files (the test compares
#: them at 1e-12), so the table cannot drift from its data by rounding either.
M3_EFF_MEASURED: np.ndarray = np.array(
    [
        (16.0, 1.7592372158437575, 0.005195868897689598),
        (24.0, 1.7808984331696052, 0.004748444723689415),
        (32.0, 1.790032831638141, 0.0043465605006579805),
        (40.0, 1.8118675914836027, 0.004041281228676691),
        (48.0, 1.8238332739571508, 0.003922257636574629),
        (64.0, 1.8403125000000005, 0.0036805340534584352),
        (1024.0, 1.9884678490045404, 0.004260133661762763),
        (2048.0, 2.0144300202954595, 0.003944910734035001),
        (4096.0, 2.04234375, 0.004017327354726193),
        (8192.0, 2.0713103414041303, 0.004776451695968452),
        (16384.0, 2.096193646762595, 0.005845786267986711),
        (32768.0, 2.12578125, 0.00802885515027133),
        (65536.0, 2.1277552047029107, 0.01196559570178517),
        (131072.0, 2.1507246047033584, 0.020292138940061235),
    ],
    dtype=float,
)

#: Smallest / largest ``rho V`` at which ``m_3^eff`` was actually measured.
M3_EFF_RHO_V_MIN = float(M3_EFF_MEASURED[0, 0])
M3_EFF_RHO_V_MAX = float(M3_EFF_MEASURED[-1, 0])

#: The unmeasured interior gap: no point was taken between these two.
M3_EFF_GAP_LO = 64.0
M3_EFF_GAP_HI = 1024.0


@dataclass(frozen=True)
class M3Effective:
    """A read of the measured ``m_3^eff`` curve, with its provenance attached.

    Attributes
    ----------
    value, sem:
        Interpolated ``m_3^eff`` and the (linearly interpolated) standard error
        of the bracketing measurements. ``sem`` is *not* a full uncertainty on
        the interpolation -- inside the four-octave gap the interpolation error
        dominates it, which is what ``status`` is for.
    rho_v:
        The ``rho V`` requested.
    status:
        ``"measured"``          -- inside a measured sub-interval;
        ``"interpolated_gap"``  -- inside the unmeasured 64 .. 1024 gap;
        ``"extrapolated_low"``  -- below 16, value clamped to the 16 measurement;
        ``"extrapolated_high"`` -- above 131072, clamped to the top measurement.
    """

    value: float
    sem: float
    rho_v: float
    status: str

    @property
    def is_supported(self) -> bool:
        """True only for ``status == "measured"``."""
        return self.status == "measured"


def m3_effective(rho_v: float) -> M3Effective:
    """Read the MEASURED ``m_3^eff`` curve at expected interval count ``rho V``.

    Log-linear interpolation in ``log2(rho V)`` between neighbouring measured
    points of ``M3_EFF_MEASURED``; clamped outside the measured range. The
    returned :class:`M3Effective` carries a ``status`` saying which of those
    happened, so no caller can silently consume an extrapolation -- the exact
    failure Part A's diagnostic traced the bogus 0.748 prediction to (RESULTS.md,
    2026-08-17, "The 0.748 prediction was itself wrong").

    Clamping direction, stated because it is a choice: ``m_3^eff`` increases
    monotonically with ``rho V`` over the measured range, so clamping *below*
    16 returns a value that is too large, which makes ``alpha_2`` too small and
    therefore **under**-estimates ``tau_c`` and the distance.
    """
    rv = float(rho_v)
    if not np.isfinite(rv) or rv <= 0:
        raise ValueError(f"rho_v must be positive and finite, got {rho_v}")

    grid = M3_EFF_MEASURED
    if rv <= M3_EFF_RHO_V_MIN:
        status = "measured" if rv == M3_EFF_RHO_V_MIN else "extrapolated_low"
        return M3Effective(float(grid[0, 1]), float(grid[0, 2]), rv, status)
    if rv >= M3_EFF_RHO_V_MAX:
        status = "measured" if rv == M3_EFF_RHO_V_MAX else "extrapolated_high"
        return M3Effective(float(grid[-1, 1]), float(grid[-1, 2]), rv, status)

    # side="right" so that an rv landing exactly ON a measured node makes that
    # node the LOWER bracket (w = 0, value reproduced exactly, and the node is
    # correctly reported as measured rather than as an endpoint of the gap it
    # happens to border). With side="left" the nodes at rho V = 64 and 1024 would
    # both be labelled "interpolated_gap" despite being measurements.
    j = int(np.searchsorted(grid[:, 0], rv, side="right"))
    lo, hi = grid[j - 1], grid[j]
    w = (math.log2(rv) - math.log2(lo[0])) / (math.log2(hi[0]) - math.log2(lo[0]))
    value = float(lo[1] + w * (hi[1] - lo[1]))
    sem = float(lo[2] + w * (hi[2] - lo[2]))
    in_gap = lo[0] == M3_EFF_GAP_LO and hi[0] == M3_EFF_GAP_HI and lo[0] < rv < hi[0]
    return M3Effective(value, sem, rv, "interpolated_gap" if in_gap else "measured")


def alpha_2_from_interval_size(n_interval: float) -> tuple[float, M3Effective]:
    """``alpha_2`` calibrated at the size of the interval the chain was counted in.

    The realised element count ``|I(x, c)|`` is the natural intrinsic estimator
    of that interval's ``rho V`` (Poisson: ``E[N] = rho V``), so ``m_3^eff`` is
    read there and converted by :func:`alpha_d_from_m`. Uses no embedding
    information whatsoever -- only the causal-set cardinality -- which is what
    keeps the estimator intrinsic in B--K's sense.

    Returns ``(alpha_2, m3_read)`` so the caller can record the status of the
    curve read alongside the value it used.

    Caveat, recorded not hidden: the curve was measured on *unconditioned*
    diamonds, while the intervals here have passed the eq.-34 Filter 2. Part A's
    diagnostic measured a comparable conditioning (Step-2 selection) to shift the
    interval cardinality by ~6% and the *chain* by ~16% (RESULTS.md, 2026-08-17),
    so this is a real though second-order effect. Filter 2 conditions on the two
    intervals being of *similar* size, not on emptiness, so the bias should be
    weaker still -- but it is not zero and has not been measured here.
    """
    read = m3_effective(max(float(n_interval), np.nextafter(0.0, 1.0)))
    return alpha_d_from_m(read.value, d=SPATIAL_DIM_2P1), read


#: ``alpha_2`` at the top of the Part-A diagnostic's small-interval baseline
#: (``rho V = 64``, ``m_3^eff = 1.8403``). Provided as the documented default for
#: the ``"fixed"`` policy: a single constant, so it acts as a pure global scale
#: and cannot perturb a rank ordering.
ALPHA_2_AT_RHO_V_64 = alpha_d_from_m(float(M3_EFF_MEASURED[5, 1]), d=SPATIAL_DIM_2P1)

#: ``alpha_2`` implied by Rideout--Wallden's **published asymptote** m_3 = 2.296.
#: Provided ONLY so experiments can quote the size of the difference; it is not
#: a default anywhere, because eq. 38 is applied far below the fit's support.
ALPHA_2_FROM_PUBLISHED_M3 = alpha_d_from_m(2.296, d=SPATIAL_DIM_2P1)


# ---------------------------------------------------------------------------
# Depth estimator (eq. 38), dimension-parametrised.
# ---------------------------------------------------------------------------


def estimate_tau_c_nd(
    a: int,
    b: int,
    c: int,
    causal_matrix: np.ndarray,
    rho: float,
    *,
    d: int = SPATIAL_DIM_2P1,
    alpha_d: float | None = None,
    alpha_policy: str | None = None,
    chain_count_fn=None,
) -> tuple[float, dict]:
    """Depth ``tau_c`` of common event ``c`` below the pair -- B--K eq. 38, any ``d``.

        tau_hat_c = 0.5 * alpha_d * rho^{-1/(d+1)} * ( n_C(c,a) + n_C(c,b) ) .

    With ``alpha_policy="measured_m3"`` the constant is read per chain rather
    than shared, which is the faithful generalisation: eq. 38 is the average of
    the two proper-time estimates ``tau(c,a)`` and ``tau(c,b)``, each of which is
    ``alpha_d rho^{-1/(d+1)} L`` for *its own* interval, so

        tau_hat_c = 0.5 * rho^{-1/(d+1)} * ( alpha(|I(a,c)|) n_C(c,a)
                                           + alpha(|I(b,c)|) n_C(c,b) ) ,

    collapsing to eq. 38 verbatim when the two alphas coincide.

    Parameters
    ----------
    alpha_d:
        Explicit constant. Overrides ``alpha_policy``.
    alpha_policy:
        ``"fixed"`` -- use ``ALPHA_1`` (d = 1) or ``ALPHA_2_AT_RHO_V_64`` (d >= 2).
        ``"measured_m3"`` -- read the measured curve per interval (d >= 2 only).
        ``None`` resolves to ``"fixed"`` for d = 1 (where ``m_2 = 2`` is exact,
        so there is nothing to interpolate and Phase 2a's exact ``ALPHA_1``
        applies) and ``"measured_m3"`` for d >= 2.

    Returns
    -------
    (tau_hat_c, diagnostics) where ``diagnostics`` records the chain counts, the
    interval cardinalities, the alpha actually used for each chain and the
    ``m_3^eff`` curve status -- everything needed to audit the calibration after
    the fact.
    """
    if rho <= 0:
        raise ValueError(f"rho must be positive, got {rho}")
    cm = np.asarray(causal_matrix, dtype=bool)

    if chain_count_fn is None:
        n_ca = chain_count(c, a, cm)
        n_cb = chain_count(c, b, cm)
    else:
        n_ca = chain_count_fn(c, a)
        n_cb = chain_count_fn(c, b)

    policy = alpha_policy
    if alpha_d is not None:
        policy = "explicit"
    elif policy is None:
        policy = "fixed" if d == 1 else "measured_m3"

    part = overlap_partition(a, b, c, cm)
    status_a = status_b = "n/a"
    if policy == "explicit":
        alpha_a = alpha_b = float(alpha_d)  # type: ignore[arg-type]
    elif policy == "fixed":
        alpha_a = alpha_b = ALPHA_1 if d == 1 else ALPHA_2_AT_RHO_V_64
    elif policy == "measured_m3":
        if d == 1:
            raise ValueError(
                'alpha_policy="measured_m3" is a d >= 2 calibration; in d = 1 the '
                "chain constant m_2 = 2 is exact (Brightwell--Gregory) and Phase 2a's "
                "ALPHA_1 applies -- use alpha_policy='fixed'."
            )
        alpha_a, read_a = alpha_2_from_interval_size(part.n_ac)
        alpha_b, read_b = alpha_2_from_interval_size(part.n_bc)
        status_a, status_b = read_a.status, read_b.status
    else:
        raise ValueError(f"unknown alpha_policy {alpha_policy!r}")

    if alpha_a == alpha_b:
        # eq. 38 verbatim, with the shared alpha factored out exactly as written.
        # This spelling (not the equivalent per-chain sum below) is what makes the
        # d = 1 path reproduce Phase 2a's `estimate_tau_c` BIT for bit: in IEEE-754
        # `a*(x+y)` and `a*x + a*y` may differ in the last ulp, and the regression
        # test asserts exact equality rather than a tolerance.
        tau = 0.5 * alpha_a * rho ** (-1.0 / (d + 1)) * (n_ca + n_cb)
    else:
        tau = 0.5 * rho ** (-1.0 / (d + 1)) * (alpha_a * n_ca + alpha_b * n_cb)
    diagnostics = {
        "n_ca": int(n_ca),
        "n_cb": int(n_cb),
        "n_ac": int(part.n_ac),
        "n_bc": int(part.n_bc),
        "alpha_a": float(alpha_a),
        "alpha_b": float(alpha_b),
        "m3_status_a": status_a,
        "m3_status_b": status_b,
        "policy": policy,
    }
    return float(tau), diagnostics


# ---------------------------------------------------------------------------
# Full pipeline, dimension-parametrised.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OverlapDistanceResultND:
    """B--K causal-overlap distance for one pair on one causet, any spatial ``d``.

    Mirrors Phase 2a's ``OverlapDistanceResult`` field for field (so the 1+1 D
    regression test can compare them directly) and adds the fields the 2+1 D
    asymptotic claim has to be auditable by.

    Attributes
    ----------
    distance, sem, n_c, n_candidates:
        As Phase 2a: mean over admissible common events ``c``, its standard
        error, the number used, and the number examined. A small ``n_c`` means
        the estimate rests on few vantage points -- reported, never hidden.
    per_c_distance, per_c_overlap, per_c_tau:
        As Phase 2a.
    d_spatial:
        Boguna--Krioukov spatial dimension used (1 for M^2, 2 for M^3).
    distance_formula:
        ``"exact_eq24"`` (d = 1) or ``"asymptotic_eq25_27"`` (d >= 2). Any quoted
        2+1 D number must carry the weaker accuracy claim this records.
    per_c_asymptotic_ratio:
        ``tau_c / d_est`` for each admissible ``c``. The asymptotic form of
        eqs. 25--27 requires this to be LARGE; the distribution is the audit of
        whether the measurement sat in the regime it assumed. Meaningful only
        when ``distance_formula == "asymptotic_eq25_27"``, but recorded always.
    per_c_interval_size:
        ``(|I(a,c)| + |I(b,c)|)/2`` per admissible ``c`` -- the ``rho V`` at
        which the ``m_3^eff`` calibration was read.
    per_c_alpha:
        The ``alpha_d`` actually applied (mean of the two chains' values when the
        measured-m_3 policy makes them differ).
    alpha_policy, m3_status_counts:
        Which calibration policy ran, and a tally of the ``m3_effective`` read
        statuses over all chains used (e.g. how many landed in the unmeasured
        gap). Empty tally for the ``fixed``/``explicit`` policies.
    """

    distance: float
    sem: float
    n_c: int
    per_c_distance: np.ndarray = field(default_factory=lambda: np.empty(0))
    per_c_overlap: np.ndarray = field(default_factory=lambda: np.empty(0))
    per_c_tau: np.ndarray = field(default_factory=lambda: np.empty(0))
    n_candidates: int = 0
    d_spatial: int = SPATIAL_DIM_2P1
    distance_formula: str = "asymptotic_eq25_27"
    per_c_asymptotic_ratio: np.ndarray = field(default_factory=lambda: np.empty(0))
    per_c_interval_size: np.ndarray = field(default_factory=lambda: np.empty(0))
    per_c_alpha: np.ndarray = field(default_factory=lambda: np.empty(0))
    alpha_policy: str = ""
    m3_status_counts: dict = field(default_factory=dict)

    @property
    def min_asymptotic_ratio(self) -> float:
        """Worst (smallest) ``tau_c / d_est`` over the admissible ``c``.

        The single number that says how safe the asymptotic form was: it is the
        closest any vantage point came to violating ``tau_c >> separation``.
        """
        if self.per_c_asymptotic_ratio.size == 0:
            return float("nan")
        return float(np.min(self.per_c_asymptotic_ratio))

    @property
    def median_asymptotic_ratio(self) -> float:
        """Median ``tau_c / d_est`` over the admissible ``c``."""
        if self.per_c_asymptotic_ratio.size == 0:
            return float("nan")
        return float(np.median(self.per_c_asymptotic_ratio))


def distance_causal_overlap_nd(
    a: int,
    b: int,
    causal_matrix: np.ndarray,
    rho: float,
    *,
    d: int = SPATIAL_DIM_2P1,
    alpha_d: float | None = None,
    alpha_policy: str | None = None,
    kappa: float = KAPPA_FILTER2,
    require_overlap_in_open_unit: bool = True,
    chain_count_fn=None,
) -> OverlapDistanceResultND:
    """Boguna--Krioukov causal-overlap spacelike distance in ``d`` spatial dimensions.

    Same four steps as Phase 2a's ``distance_causal_overlap`` (their Sec. IV.A,
    intrinsic Filter 2), with the two dimension-dependent pieces dispatched:

      1. Enumerate common past events ``c`` (``c prec a`` and ``c prec b``).
      2. Keep those passing Filter 2, eq. 34 -- symmetric vantage points.
      3. Per ``c``: measure ``O_C`` (eq. 28, dimension-agnostic), estimate
         ``tau_c`` (eq. 38 with ``alpha_d``), map to a distance with the **exact**
         eq. 24 when ``d = 1`` and the **asymptotic** eqs. 25--27 when ``d >= 2``.
      4. Average over admissible ``c``; report mean, standard error, and count.

    Passing ``d = 1`` reproduces Phase 2a bit for bit -- it calls the same
    ``distance_from_overlap`` with the same ``ALPHA_1``, and
    ``tests/test_causal_overlap_3d.py`` asserts exact equality of every returned
    array against ``causal_overlap.distance_causal_overlap``. That regression is
    the reason the Phase 2a module is not edited.

    ``rho`` is the sprinkling density; ``chain_count_fn`` is the same optional
    speed injection Phase 2a documents (it changes no physics).
    """
    if d < 1:
        raise ValueError(f"spatial dimension d must be >= 1, got {d}")
    cm = np.asarray(causal_matrix, dtype=bool)
    candidates = common_past(a, b, cm)

    dists: list[float] = []
    overlaps: list[float] = []
    taus: list[float] = []
    ratios: list[float] = []
    sizes: list[float] = []
    alphas: list[float] = []
    status_counts: dict[str, int] = {}
    policy_used = ""

    for c in candidates.tolist():
        if not filter2_passes(a, b, c, cm, kappa=kappa):
            continue
        o = causal_overlap(a, b, c, cm)
        if not np.isfinite(o):
            continue
        if require_overlap_in_open_unit and not (0.0 < o < 1.0):
            continue
        tau, diag = estimate_tau_c_nd(
            a,
            b,
            c,
            cm,
            rho,
            d=d,
            alpha_d=alpha_d,
            alpha_policy=alpha_policy,
            chain_count_fn=chain_count_fn,
        )
        if tau <= 0:
            continue
        dist = distance_from_overlap_nd(o, tau, d)
        dists.append(dist)
        overlaps.append(o)
        taus.append(tau)
        ratios.append(tau / dist if dist > 0 else float("inf"))
        sizes.append(0.5 * (diag["n_ac"] + diag["n_bc"]))
        alphas.append(0.5 * (diag["alpha_a"] + diag["alpha_b"]))
        policy_used = diag["policy"]
        for key in ("m3_status_a", "m3_status_b"):
            st = diag[key]
            if st != "n/a":
                status_counts[st] = status_counts.get(st, 0) + 1

    formula = "exact_eq24" if d == 1 else "asymptotic_eq25_27"
    arr = np.asarray(dists, dtype=float)
    n_c = int(arr.size)
    if n_c == 0:
        return OverlapDistanceResultND(
            distance=float("nan"),
            sem=float("nan"),
            n_c=0,
            n_candidates=int(candidates.size),
            d_spatial=int(d),
            distance_formula=formula,
            alpha_policy=policy_used,
            m3_status_counts=status_counts,
        )
    mean = float(arr.mean())
    sem = float(arr.std(ddof=1) / np.sqrt(n_c)) if n_c > 1 else float("nan")
    return OverlapDistanceResultND(
        distance=mean,
        sem=sem,
        n_c=n_c,
        per_c_distance=arr,
        per_c_overlap=np.asarray(overlaps, dtype=float),
        per_c_tau=np.asarray(taus, dtype=float),
        n_candidates=int(candidates.size),
        d_spatial=int(d),
        distance_formula=formula,
        per_c_asymptotic_ratio=np.asarray(ratios, dtype=float),
        per_c_interval_size=np.asarray(sizes, dtype=float),
        per_c_alpha=np.asarray(alphas, dtype=float),
        alpha_policy=policy_used,
        m3_status_counts=status_counts,
    )
