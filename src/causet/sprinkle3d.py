"""Poisson sprinkling into finite regions of 2+1 D Minkowski spacetime (M^3).

Phase 2b infrastructure. The 1+1 D module (``sprinkle.py``) is a closed,
accepted dependency and is not touched here; this module is its M^3 analogue.

Why 2+1 D is required
---------------------
Rideout & Wallden, "Spacelike distance from discrete causal order"
(arXiv:0810.1768), Section II.B, note that the *naive* spatial-distance
construction already works in 1+1 D: there is a unique minimising pair, no
degeneracy. Their 2-link construction exists to repair a failure that only
appears for spacetime dimension d >= 3, where Lorentz boost freedom supplies
infinitely many independent minimising pairs. Testing that construction in
1+1 D would therefore validate nothing.

Coordinates and signature
-------------------------
Coordinates ``(t, x, y)`` with signature ``(-,+,+)``: the squared interval
between two events is ``ds^2 = -(dt)^2 + (dx)^2 + (dy)^2``, so events are
timelike-separated iff ``(dt)^2 - (dx)^2 - (dy)^2 > 0`` (see ``order3d.py``).
There is **no** null-coordinate change of variables that turns the M^3 causal
order into a coordinate/dominance order: in 1+1 D the light cone has two flat
faces (hence ``u, v``), whereas in 2+1 D it is a round cone with infinitely
many supporting hyperplanes. The ``(u, v)`` trick of Phase 1 does not
generalise and is not attempted anywhere in Phase 2b.

Regions provided
----------------
1. ``sprinkle_diamond_3d`` -- the Alexandrov interval (causal diamond)
   ``I(p, q)`` between ``p = (0, 0, 0)`` and ``q = (T, 0, 0)``. **This is the
   region used by the Part-1 gate experiment** (``exp02_m3_validation.py``),
   because the constant ``m_d`` of Rideout--Wallden eq. (1) is defined by the
   longest chain *in an Alexandrov interval*; it is not a property of an
   arbitrary region, so a box could not be used to measure it.
2. ``sprinkle_box_3d`` -- a rectangular cuboid ``[0,T] x [-Lx/2,Lx/2] x
   [-Ly/2,Ly/2]``, volume ``V = T * Lx * Ly`` by inspection. Provided because
   (a) its volume formula is unambiguous, which makes it an independent anchor
   for the Poisson-count tests that would otherwise all rest on this module's
   own derivation of the diamond volume, and (b) Parts 2--3 need a region that
   can be grown while a chosen pair of *fixed* target events stays inside it
   (the Rideout--Wallden Fig. 14 stability protocol).

Diamond geometry and volume
---------------------------
With ``p = (0,0,0)``, ``q = (T,0,0)``, an event ``(t,x,y)`` lies in
``I(p,q) = J^+(p) ∩ J^-(q)`` iff it is in the future of ``p`` and the past of
``q``, i.e. with ``r = sqrt(x^2 + y^2)``::

    r < t          (inside p's future light cone)
    r < T - t      (inside q's past light cone)

so ``r < R(t) := min(t, T - t)`` for ``0 < t < T``. The region is a bicone: two
right circular cones of height ``T/2`` and base radius ``T/2`` glued base to
base at ``t = T/2``. Its volume is

    V = 2 * (1/3) * pi * (T/2)^2 * (T/2) = pi * T^3 / 12,

reproducing the general Rideout--Wallden eq. (2) constant

    eta(d) = 2 * V^s_{d-1} / (2^d * d),      V_xy = eta(d) * l_xy^d,

with ``V^s_k = pi^{k/2} / Gamma(k/2 + 1)`` the volume of the unit ``k``-ball:
``eta(3) = 2*pi/(8*3) = pi/12`` and, as a cross-check against Phase 1,
``eta(2) = 2*2/(4*2) = 1/2``, which is the 1+1 D value ``V = tau^2/2`` already
validated in Phase 1. ``interval_volume_constant`` computes ``eta(d)`` from the
formula rather than hardcoding either number (Integrity Rule 4).

Exact uniform sampling in the bicone
------------------------------------
Rejection sampling in the bounding box would accept only ``eta(3) = pi/12
~ 26%`` of proposals. We sample exactly instead. The cross-section at time
``t`` is a disk of area ``pi R(t)^2``, so the marginal density of ``t`` is
``p(t) = 12 t^2 / T^3`` on ``(0, T/2]`` and its mirror image on ``[T/2, T)``.
Equivalently, writing ``w`` for the distance in ``t`` from the nearer apex
(``w = t`` in the lower cone, ``w = T - t`` in the upper cone), ``w`` has
density ``3 w^2 / (T/2)^3`` on ``[0, T/2]`` and each cone carries half the
volume. Hence

    coin ~ Bernoulli(1/2),   w = (T/2) * U1^(1/3),
    t = w if coin else T - w,     R(t) = w  in both cases,
    r = w * sqrt(U2),   theta = 2*pi*U3,   x = r cos theta,  y = r sin theta,

with ``U1, U2, U3`` i.i.d. uniform on [0,1). ``r = R sqrt(U)`` is the standard
uniform-in-a-disk radial law. The resulting cumulative distributions, used as
independent test oracles in ``tests/test_sprinkle3d.py``, are

    F_t(t)  = 4 t^3 / T^3                 for 0 <= t <= T/2
            = 1 - 4 (T - t)^3 / T^3       for T/2 <= t <= T
    F_r(r)  = 12 r^2 / T^2 - 16 r^3 / T^3 for 0 <= r <= T/2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# ---- Physically named, documented constants (Integrity Rule 4) -------------
SPACETIME_DIM_2P1 = 3  # 1 time + 2 space dimensions: "d = 3" in Rideout--Wallden.


def unit_ball_volume(k: int) -> float:
    """Volume of the unit ball in ``k`` Euclidean dimensions.

    ``V^s_k = pi^(k/2) / Gamma(k/2 + 1)``. Used only to build ``eta(d)`` below.
    Sanity values: ``V^s_1 = 2`` (the interval [-1,1]), ``V^s_2 = pi`` (unit
    disk), ``V^s_3 = 4 pi / 3``.
    """
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    return math.pi ** (k / 2.0) / math.gamma(k / 2.0 + 1.0)


def interval_volume_constant(d: int = SPACETIME_DIM_2P1) -> float:
    """Alexandrov-interval volume constant ``eta(d)`` of Rideout--Wallden eq. (2).

    Their eq. (2), Section II.1:
        ``V_xy = eta(d) * l_xy^d``  with  ``eta(d) = 2 V^s_{d-1} / (2^d d)``,
    where ``l_xy`` is the proper-time separation of the interval's endpoints and
    ``d`` is the **spacetime** dimension.

    Computed, not hardcoded (Integrity Rule 4). Verified values:
        ``eta(2) = 1/2``    -- agrees with the Phase-1 1+1 D diamond V = tau^2/2,
        ``eta(3) = pi/12``  -- the 2+1 D value quoted by Rideout--Wallden,
        ``eta(4) = pi/24``  -- the standard 3+1 D causal-diamond volume.
    """
    if d < 2:
        raise ValueError(f"spacetime dimension d must be >= 2, got {d}")
    return 2.0 * unit_ball_volume(d - 1) / (2.0**d * d)


#: ``eta(3) = pi/12``, the 2+1 D diamond-volume constant (Rideout--Wallden eq. 2).
DIAMOND_VOLUME_CONSTANT_3D = interval_volume_constant(SPACETIME_DIM_2P1)


def diamond_volume_3d(tau: float) -> float:
    """Spacetime volume of the M^3 causal diamond of proper time ``tau``.

    ``V = eta(3) * tau^3 = pi tau^3 / 12`` (Rideout--Wallden eq. 2; derived
    independently as a bicone volume in the module docstring).
    """
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    return DIAMOND_VOLUME_CONSTANT_3D * tau**3


def box_volume_3d(t_extent: float, x_extent: float, y_extent: float) -> float:
    """Spacetime volume of the M^3 cuboid: ``V = t_extent * x_extent * y_extent``.

    No derivation needed -- that is the point of offering the box as an
    independent anchor for the Poisson-count tests.
    """
    for name, val in (("t_extent", t_extent), ("x_extent", x_extent), ("y_extent", y_extent)):
        if val <= 0:
            raise ValueError(f"{name} must be positive, got {val}")
    return t_extent * x_extent * y_extent


def expected_count_diamond_3d(rho: float, tau: float) -> float:
    """``E[N] = rho * V`` for the M^3 diamond. This is the ``rho V`` of eq. (1)."""
    if rho <= 0:
        raise ValueError(f"rho must be positive, got {rho}")
    return rho * diamond_volume_3d(tau)


@dataclass(frozen=True)
class Sprinkling3D:
    """A realised Poisson sprinkling into a finite region of M^3.

    Attributes
    ----------
    t, x, y:
        Minkowski coordinates of the ``N`` events, shape ``(N,)``. Unlike the
        1+1 D ``Sprinkling`` there are no null coordinates: they carry no
        advantage in 2+1 D (see module docstring).
    rho:
        Sprinkling density used (events per unit spacetime volume).
    volume:
        Spacetime volume of the sprinkling region.
    n_expected:
        ``E[N] = rho * volume``, i.e. the ``rho V`` appearing in
        Rideout--Wallden eq. (1). Deterministic given the parameters.
    seed:
        Integer seed that produced this realisation (Integrity Rule 1).
    region:
        ``"diamond"`` or ``"box"``.
    tau:
        Proper time between the diamond endpoints; ``None`` for a box.
    extent:
        ``(t_extent, x_extent, y_extent)`` for a box; ``None`` for a diamond.
    endpoint_indices:
        ``(i_past, i_future)`` indices of the two diamond endpoints ``p``, ``q``
        if they were included, else ``None``. Following Phase 1's convention
        ``p`` is index 0 and ``q`` is the last index.
    """

    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    rho: float
    volume: float
    n_expected: float
    seed: int
    region: str
    tau: float | None = None
    extent: tuple[float, float, float] | None = None
    endpoint_indices: tuple[int, int] | None = None

    @property
    def n(self) -> int:
        """Number of events actually realised (including endpoints if present)."""
        return int(self.t.shape[0])

    @property
    def n_interior(self) -> int:
        """Number of Poisson-sprinkled events, excluding any added endpoints."""
        return self.n - (2 if self.endpoint_indices is not None else 0)

    def coords(self) -> np.ndarray:
        """Return the events as an ``(N, 3)`` array of ``(t, x, y)`` rows."""
        return np.column_stack((self.t, self.x, self.y))


def sprinkle_diamond_3d(
    rho: float,
    tau: float,
    seed: int,
    *,
    include_endpoints: bool = False,
) -> Sprinkling3D:
    """Poisson-sprinkle into the M^3 causal diamond ``I(p, q)`` of proper time ``tau``.

    ``p = (0, 0, 0)`` and ``q = (tau, 0, 0)``, so the geodesic joining them is
    purely timelike with proper time ``tau``. The interior count is drawn from
    ``Poisson(rho * V)`` with ``V = pi tau^3 / 12`` and the events are placed by
    the exact bicone sampler derived in the module docstring (no rejection).

    Parameters
    ----------
    rho:
        Sprinkling density, > 0.
    tau:
        Proper time between the endpoints, > 0.
    seed:
        Seed for ``numpy.random.Generator`` (PCG64); recorded in the result so
        every downstream figure is regenerable (Integrity Rule 1).
    include_endpoints:
        If True, prepend ``p`` and append ``q``. They are the interval's unique
        minimum and maximum, and every interior event lies causally between
        them, so the longest chain of the full set runs from ``p`` to ``q``.
        Required for the ``m_d`` measurement, whose ``L`` is defined as the
        number of links in the longest chain *between and including* the two
        endpoints (Rideout--Wallden, Section II.1). Adding two events is a
        measure-zero modification of the Poisson process.
    """
    if rho <= 0:
        raise ValueError(f"rho must be positive, got {rho}")
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")

    rng = np.random.default_rng(seed)
    volume = diamond_volume_3d(tau)
    n_expected = rho * volume
    n_interior = int(rng.poisson(n_expected))

    half = 0.5 * tau
    # w: distance in t from the nearer cone apex; density 3 w^2 / half^3.
    w = half * np.cbrt(rng.random(n_interior))
    upper = rng.random(n_interior) < 0.5
    t = np.where(upper, tau - w, w)
    # Cross-section radius at that time is exactly w in either cone.
    r = w * np.sqrt(rng.random(n_interior))
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n_interior)
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    endpoint_indices: tuple[int, int] | None = None
    if include_endpoints:
        t = np.concatenate(([0.0], t, [tau]))
        x = np.concatenate(([0.0], x, [0.0]))
        y = np.concatenate(([0.0], y, [0.0]))
        endpoint_indices = (0, t.shape[0] - 1)

    return Sprinkling3D(
        t=t,
        x=x,
        y=y,
        rho=float(rho),
        volume=float(volume),
        n_expected=float(n_expected),
        seed=int(seed),
        region="diamond",
        tau=float(tau),
        endpoint_indices=endpoint_indices,
    )


def sprinkle_box_3d(
    rho: float,
    t_extent: float,
    x_extent: float,
    y_extent: float,
    seed: int,
) -> Sprinkling3D:
    """Poisson-sprinkle into the M^3 cuboid ``[0,T] x [-Lx/2,Lx/2] x [-Ly/2,Ly/2]``.

    ``V = T * Lx * Ly`` by inspection, and uniformity is a product of three
    independent uniforms, so this region introduces no geometric derivation of
    its own. That is exactly why it serves as the independent anchor for the
    Poisson-count tests and, in Parts 2--3, as a region that can be enlarged
    while chosen target events stay fixed inside it.
    """
    if rho <= 0:
        raise ValueError(f"rho must be positive, got {rho}")
    volume = box_volume_3d(t_extent, x_extent, y_extent)

    rng = np.random.default_rng(seed)
    n_expected = rho * volume
    n = int(rng.poisson(n_expected))

    t = rng.uniform(0.0, t_extent, size=n)
    x = rng.uniform(-0.5 * x_extent, 0.5 * x_extent, size=n)
    y = rng.uniform(-0.5 * y_extent, 0.5 * y_extent, size=n)

    return Sprinkling3D(
        t=t,
        x=x,
        y=y,
        rho=float(rho),
        volume=float(volume),
        n_expected=float(n_expected),
        seed=int(seed),
        region="box",
        extent=(float(t_extent), float(x_extent), float(y_extent)),
    )


def in_diamond_3d(t: np.ndarray, x: np.ndarray, y: np.ndarray, tau: float) -> np.ndarray:
    """Boolean mask: which events lie inside ``I((0,0,0), (tau,0,0))``.

    Condition ``sqrt(x^2 + y^2) < min(t, tau - t)`` from the module docstring.
    Used by tests (to confirm the sampler never leaves the bicone) and by
    Parts 2--3 (to restrict a box sprinkling to an interval).
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r = np.hypot(x, y)
    return (t > 0.0) & (t < tau) & (r < np.minimum(t, tau - t))
