"""Poisson sprinkling into Alexandrov intervals of 1+1 D Minkowski spacetime.

A causal set ("causet") is obtained from a continuum spacetime by a Poisson
process: points are sprinkled at density ``rho`` so that the expected number of
points in any region of spacetime volume ``V`` is ``rho * V``, and the actual
count is Poisson-distributed. The induced partial order is the causal order of
the sprinkled points. Poisson sprinkling is the standard kinematically-Lorentz-
invariant discretisation used in causal set theory (see Sorkin, "Causal Sets:
Discrete Gravity", gr-qc/0309009, and the review by Surya, Living Rev. Relativ.
22, 5 (2019)).

Geometry, 1+1 D (this module).
--------------------------------
We sprinkle into the Alexandrov interval (causal diamond) ``I(p, q)`` between a
past endpoint ``p`` and a future endpoint ``q`` separated by a purely timelike
interval of proper time ``tau``. We place
    p = (t=0,   x=0),
    q = (t=tau, x=0),
so the geodesic separating them is purely timelike with proper time ``tau``.

Null (light-cone) coordinates make the diamond and the causal order trivial:
    u = t + x,      v = t - x,
    t = (u + v)/2,  x = (u - v)/2.
The Jacobian is  dt dx = (1/2) du dv. In null coordinates the diamond ``I(p,q)``
is exactly the axis-aligned square ``[0, tau] x [0, tau]``, and the causal order
``x_i precedes x_j`` becomes the 2-D coordinate (dominance) order
``u_i <= u_j AND v_i <= v_j`` (see order.py). Because the Jacobian is constant,
a uniform distribution in ``(u, v)`` is uniform in ``(t, x)``.

Spacetime volume of the diamond:
    V(tau) = (1/2) * tau^2         [area of the tau x tau square times |J| = 1/2].
Hence the expected sprinkled count is  E[N] = rho * V = rho * tau^2 / 2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Physically named, documented parameters (Scientific Integrity Rule 4).
SPACETIME_DIM_1P1 = 2  # 1 time + 1 space dimension.


def diamond_volume_1d(tau: float) -> float:
    """Spacetime volume (area) of the 1+1 D causal diamond of proper time ``tau``.

    V(tau) = tau^2 / 2. Derived in the module docstring from the null-coordinate
    square [0, tau]^2 and the Jacobian |d(t,x)/d(u,v)| = 1/2.
    """
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    return 0.5 * tau * tau


def expected_count_1d(rho: float, tau: float) -> float:
    """Expected number of sprinkled points, E[N] = rho * V(tau)."""
    if rho <= 0:
        raise ValueError(f"rho must be positive, got {rho}")
    return rho * diamond_volume_1d(tau)


@dataclass(frozen=True)
class Sprinkling:
    """A realised sprinkling into a 1+1 D causal diamond.

    Attributes
    ----------
    t, x:
        Minkowski coordinates of the ``N`` sprinkled points, shape ``(N,)``.
    u, v:
        Null coordinates ``u = t + x``, ``v = t - x``, shape ``(N,)``.
        Provided because the causal order is cheapest to compute from them.
    tau:
        Proper time between the diamond endpoints ``p`` and ``q``.
    rho:
        Sprinkling density used.
    seed:
        The integer seed that produced this realisation (Rule 1: reproducible).
    n_expected:
        E[N] = rho * V(tau) for this realisation's parameters.
    """

    t: np.ndarray
    x: np.ndarray
    u: np.ndarray
    v: np.ndarray
    tau: float
    rho: float
    seed: int
    n_expected: float

    @property
    def n(self) -> int:
        """Number of sprinkled points actually realised."""
        return int(self.t.shape[0])


def sprinkle_diamond_1d(
    rho: float,
    tau: float,
    seed: int,
    *,
    include_endpoints: bool = False,
) -> Sprinkling:
    """Poisson-sprinkle points into the 1+1 D causal diamond of proper time ``tau``.

    The number of interior points is drawn from ``Poisson(rho * V(tau))`` and each
    point is placed uniformly in the null-coordinate square ``[0, tau]^2`` (which
    is uniform in ``(t, x)`` because the Jacobian is constant). This is the exact
    Poisson process on the diamond.

    Parameters
    ----------
    rho:
        Sprinkling density (points per unit spacetime volume). Must be > 0.
    tau:
        Proper time between the past endpoint ``p=(0,0)`` and future endpoint
        ``q=(tau,0)``. Must be > 0.
    seed:
        Seed for a ``numpy.random.Generator`` (PCG64). Recorded in the result so
        every figure is regenerable (Scientific Integrity Rule 1).
    include_endpoints:
        If True, prepend ``p`` and append ``q`` to the point set. The endpoints
        are the diamond's unique minimum and maximum. They are a measure-zero
        addition and do not change Poisson statistics, but they are the natural
        chain endpoints for the Brightwell--Gregory longest-chain check. Default
        False (pure Poisson process, for statistics tests).

    Returns
    -------
    Sprinkling
        The realised point set with both Minkowski and null coordinates.
    """
    if rho <= 0:
        raise ValueError(f"rho must be positive, got {rho}")
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")

    rng = np.random.default_rng(seed)
    n_expected = expected_count_1d(rho, tau)
    n_interior = int(rng.poisson(n_expected))

    # Uniform in the null-coordinate square => uniform in (t, x).
    u = rng.uniform(0.0, tau, size=n_interior)
    v = rng.uniform(0.0, tau, size=n_interior)

    if include_endpoints:
        # p = (0,0) -> (u,v)=(0,0);  q = (tau,tau in null) i.e. (t,x)=(tau,0).
        u = np.concatenate(([0.0], u, [tau]))
        v = np.concatenate(([0.0], v, [tau]))

    t = 0.5 * (u + v)
    x = 0.5 * (u - v)

    return Sprinkling(
        t=t,
        x=x,
        u=u,
        v=v,
        tau=float(tau),
        rho=float(rho),
        seed=int(seed),
        n_expected=float(n_expected),
    )
