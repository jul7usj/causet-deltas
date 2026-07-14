"""Antichains: mutually-incomparable sets, and maximal (inextendible) antichains.

An antichain is a set of pairwise causally-unrelated events -- a discrete analogue
of a spacelike slice. The Delta-s estimator (later phases) lives on *maximal*
(equivalently, inextendible) antichains near the mid-slice of the diamond.

Definitions
-----------
* Antichain ``A`` : for all distinct ``a, b`` in ``A``, neither ``a prec b`` nor
  ``b prec a``.
* Maximal / inextendible antichain ``A`` : an antichain such that every element
  ``x not in A`` is comparable to at least one ``a in A`` (i.e. ``A`` cannot be
  enlarged and still be an antichain). Every element of the causet is therefore
  either in ``A`` or causally related to some element of ``A``; ``A`` "cuts" the
  order. This is the standard causal-set notion of an inextendible antichain
  (cf. Major, Rideout & Surya, "Spatial hypersurfaces in causal set cosmology",
  gr-qc/0506133).
"""

from __future__ import annotations

import numpy as np


def is_antichain(c: np.ndarray, indices: np.ndarray | list[int]) -> bool:
    """Return True iff ``indices`` are pairwise causally unrelated under ``C``."""
    c = np.asarray(c, dtype=bool)
    idx = np.asarray(list(indices), dtype=int)
    if idx.size <= 1:
        return True
    sub = c[np.ix_(idx, idx)]
    # No relation in either direction among the chosen elements.
    return not bool(np.any(sub | sub.T))


def is_maximal_antichain(c: np.ndarray, indices: np.ndarray | list[int]) -> bool:
    """Return True iff ``indices`` form an inextendible (maximal) antichain.

    Checks (i) it is an antichain, and (ii) every element not in the set is
    comparable (in either direction) to at least one member -- so no element can
    be added without breaking the antichain property.
    """
    c = np.asarray(c, dtype=bool)
    idx = np.asarray(list(indices), dtype=int)
    if not is_antichain(c, idx):
        return False

    n = c.shape[0]
    in_set = np.zeros(n, dtype=bool)
    in_set[idx] = True
    others = np.nonzero(~in_set)[0]
    if others.size == 0:
        return True

    # comparable[o] True iff o relates to some member a (o prec a or a prec o).
    # C[others][:, idx] gives o prec a ; C[idx][:, others].T gives a prec o.
    forward = c[np.ix_(others, idx)]  # o prec a
    backward = c[np.ix_(idx, others)].T  # a prec o
    comparable = np.any(forward | backward, axis=1)
    return bool(np.all(comparable))


def maximal_antichain_near_midslice(
    c: np.ndarray,
    t: np.ndarray,
    *,
    t_mid: float | None = None,
) -> np.ndarray:
    """Greedily build a maximal antichain hugging the time slice ``t = t_mid``.

    Elements are considered in order of increasing ``|t_i - t_mid|`` and each is
    added to the antichain if it is incomparable to every element already chosen.
    Because *every* element is offered, the result is inextendible (maximal): any
    element not chosen was rejected precisely because it is comparable to a chosen
    one. Ordering by proximity to the mid-slice makes the antichain concentrate
    near ``t_mid`` (the natural place to read off a spacelike separation), which is
    where the Delta-s estimator will operate.

    Parameters
    ----------
    c:
        Causal matrix, shape ``(N, N)``.
    t:
        Time coordinate of each element, shape ``(N,)``.
    t_mid:
        Slice time. Defaults to the midpoint ``(min(t) + max(t)) / 2``.

    Returns
    -------
    np.ndarray
        Sorted array of element indices forming a maximal antichain.
    """
    c = np.asarray(c, dtype=bool)
    t = np.asarray(t, dtype=float)
    n = c.shape[0]
    if n == 0:
        return np.empty(0, dtype=int)
    if t.shape[0] != n:
        raise ValueError("t must have one entry per element of C")

    if t_mid is None:
        t_mid = 0.5 * (float(t.min()) + float(t.max()))

    order = np.argsort(np.abs(t - t_mid), kind="stable")

    chosen: list[int] = []
    for j in order:
        j = int(j)
        if not chosen:
            chosen.append(j)
            continue
        chosen_arr = np.asarray(chosen, dtype=int)
        # j comparable to any chosen member?
        related = np.any(c[chosen_arr, j] | c[j, chosen_arr])
        if not related:
            chosen.append(j)

    return np.sort(np.asarray(chosen, dtype=int))
