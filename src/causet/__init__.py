"""causet: numerical tools for the Delta-s spacelike-distance estimator project.

Phase 1 public surface: sprinkling, causal order, antichains (1+1 D).
Phase 2a adds: causal_overlap (Boguñá--Krioukov spacelike distance baseline).
Phase 2b Part 1 adds: sprinkle3d, order3d (2+1 D / M^3 infrastructure).
Phase 2b Part 2 adds: rideout_wallden (2-link spacelike distance).
"""

from __future__ import annotations

__all__ = [
    "sprinkle",
    "order",
    "antichain",
    "causal_overlap",
    "sprinkle3d",
    "order3d",
    "rideout_wallden",
]
