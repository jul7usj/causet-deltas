"""Tests for the Rideout--Wallden 2-link spacelike distance (arXiv:0810.1768).

The centrepiece is a hand-placed 13-element M^3 causet in which every set the
construction depends on can be enumerated on paper. It is built deliberately so
that the two *wrong* implementations one is tempted to write both fail it:

* using the minimal elements of ``fut(x) n fut(y)`` instead of the 2-links --
  the causet contains two elements (``fbad_x``, ``fbad_y``) that are minimal in
  the common future yet are *not* 2-links, each blocked by an intervening element
  that lies in only one of the two futures (Rideout--Wallden's Section V.A
  warning; the negative controls the brief demands);
* counting chain *elements* instead of *links* -- the hand answer is 2 links,
  which is 3 elements (Part-1 Constraint 1).

Layout of the hand causet (signature (-,+,+), coordinates ``(t, x, y)``)
------------------------------------------------------------------------
``x = (0,-1,0)`` and ``y = (0,+1,0)`` are spacelike at separation 2. Their two
future light cones first meet on the hyperbola ``X = 0, t = sqrt(1 + Y^2)``;
elements placed just outside it are in both futures. Every coordinate below was
chosen from that geometry and each claimed relation is asserted explicitly, so a
failure points at a specific pair rather than at "the causet".
"""

from __future__ import annotations

import numpy as np
import pytest

from causet import order, rideout_wallden as rw, sprinkle3d
from causet.order3d import causal_matrix_3d

# --------------------------------------------------------------------------- #
# The hand-built causet.
# --------------------------------------------------------------------------- #

#: index -> name, in the order the coordinates are listed below.
NAMES = [
    "x", "y", "p0", "p1", "f0", "f1", "f2",
    "fbad_x", "z_x", "fbad_y", "z_y", "s", "q",
]
IDX = {name: i for i, name in enumerate(NAMES)}

#: (t, x, y) of each element. Roles:
#:   x, y      -- the spacelike target pair, separation 2 at t = 0.
#:   p0, p1    -- common past; p1 prec p0, so p0 is the unique maximal one.
#:   f0, f2    -- the two genuine future 2-links (both cones empty below them).
#:   f1        -- common future but NOT minimal: f0 lies below it.
#:   fbad_x    -- minimal in fut(x) n fut(y) but blocked from x by z_x.
#:   z_x       -- in fut(x) only (misses fut(y)), and x prec z_x prec fbad_x.
#:   fbad_y    -- mirror image: minimal, blocked from y by z_y.
#:   z_y       -- in fut(y) only, and y prec z_y prec fbad_y.
#:   s         -- in fut(x) only, below nothing that matters (filler).
#:   q         -- far past off-axis: precedes y but not x, so it is NOT in the
#:                common past (filler that a sloppy past-set would pick up).
COORDS = np.array(
    [
        (0.0, -1.0, 0.0),    # x
        (0.0, 1.0, 0.0),     # y
        (-1.2, 0.0, 0.0),    # p0
        (-2.5, 0.0, 0.0),    # p1
        (1.2, 0.0, 0.0),     # f0
        (2.5, 0.0, 0.0),     # f1
        (1.6, 0.0, 1.0),     # f2
        (2.4, 0.0, 2.0),     # fbad_x
        (1.2, -0.5, 1.0),    # z_x
        (2.4, 0.0, -2.0),    # fbad_y
        (1.2, 0.5, -1.0),    # z_y
        (0.6, -1.2, 0.0),    # s
        (-3.0, 3.0, 0.0),    # q
    ]
)


@pytest.fixture(scope="module")
def hand_causet() -> np.ndarray:
    return causal_matrix_3d(COORDS[:, 0], COORDS[:, 1], COORDS[:, 2])


def names_of(indices) -> set[str]:
    return {NAMES[int(i)] for i in np.asarray(indices).tolist()}


# --------------------------------------------------------------------------- #
# 0. The causet is what the docstring says it is.
# --------------------------------------------------------------------------- #


def test_hand_causet_relations_are_as_designed(hand_causet):
    """Assert every relation the later tests rely on, so failures are localised."""
    c = hand_causet
    i = IDX

    # The target pair is spacelike.
    assert not c[i["x"], i["y"]] and not c[i["y"], i["x"]]

    # Common past: p1 prec p0 prec both targets; q precedes y only.
    for p in ("p0", "p1"):
        assert c[i[p], i["x"]] and c[i[p], i["y"]]
    assert c[i["p1"], i["p0"]]
    assert c[i["q"], i["y"]] and not c[i["q"], i["x"]]

    # Common future.
    for f in ("f0", "f1", "f2", "fbad_x", "fbad_y"):
        assert c[i["x"], i[f]] and c[i["y"], i[f]], f
    assert c[i["f0"], i["f1"]]  # f1 is not minimal
    assert not c[i["f0"], i["f2"]] and not c[i["f2"], i["f0"]]  # f0, f2 incomparable

    # The two blocking elements sit in exactly one future each.
    assert c[i["x"], i["z_x"]] and not c[i["y"], i["z_x"]]
    assert c[i["y"], i["z_y"]] and not c[i["x"], i["z_y"]]
    assert c[i["z_x"], i["fbad_x"]]
    assert c[i["z_y"], i["fbad_y"]]
    # ... and nothing from the *other* side reaches the bad element.
    assert not c[i["z_y"], i["fbad_x"]]
    assert not c[i["z_x"], i["fbad_y"]]
    # Neither genuine 2-link is above the blockers, so they stay clean.
    for f in ("f0", "f2"):
        assert not c[i["z_x"], i[f]] and not c[i["z_y"], i[f]], f


def test_hand_causet_is_transitive(hand_causet):
    """Sanity: it is a genuine partial order (slow int64 path is fine at N=13)."""
    assert order.is_transitive(hand_causet)


# --------------------------------------------------------------------------- #
# 1. future_2links -- the hand-countable test the brief asks for FIRST.
# --------------------------------------------------------------------------- #


def test_future_2links_hand_countable(hand_causet):
    """``future_2links`` returns exactly ``{f0, f2}``.

    By hand: the common future is ``{f0, f1, f2, fbad_x, fbad_y}``.
      * ``f1`` has ``f0`` below it, so ``[x, f1]`` is non-empty.
      * ``fbad_x`` has ``x prec z_x prec fbad_x``.
      * ``fbad_y`` has ``y prec z_y prec fbad_y``.
      * ``f0`` and ``f2`` have ``past(.) = {p0, p1, x, y}``, and neither ``p``
        is in ``fut(x)`` or ``fut(y)``, so both intervals are empty.
    """
    got = rw.future_2links(IDX["x"], IDX["y"], hand_causet)
    assert names_of(got) == {"f0", "f2"}


def test_negative_control_minimal_in_common_future_is_not_a_2link(hand_causet):
    """The exact failure Rideout--Wallden warn about in Section V.A.

    ``fbad_x`` and ``fbad_y`` ARE minimal elements of ``fut(x) n fut(y)`` -- the
    weaker condition -- yet each has a non-empty interval to one target. An
    implementation that used minimal elements would admit them; ours must not.
    """
    x, y, c = IDX["x"], IDX["y"], hand_causet

    minimal = names_of(rw.minimal_common_future(x, y, c))
    assert minimal == {"f0", "f2", "fbad_x", "fbad_y"}

    two_links = names_of(rw.future_2links(x, y, c))
    assert {"fbad_x", "fbad_y"} <= minimal
    assert {"fbad_x", "fbad_y"} & two_links == set()
    assert two_links < minimal  # strict subset: the whole point

    # And the blocking elements are exactly the ones named, each missing the
    # other target's future -- which is why minimality did not catch them.
    assert c[x, IDX["z_x"]] and c[IDX["z_x"], IDX["fbad_x"]]
    assert not c[y, IDX["z_x"]]
    assert c[y, IDX["z_y"]] and c[IDX["z_y"], IDX["fbad_y"]]
    assert not c[x, IDX["z_y"]]


def test_future_2links_rejects_identical_or_bad_input(hand_causet):
    with pytest.raises(ValueError):
        rw.future_2links(0, 0, hand_causet)
    with pytest.raises(ValueError):
        rw.future_2links(0, 1, hand_causet[:, :3])


def test_future_2links_of_a_causally_related_pair_is_empty(hand_causet):
    """Degenerate but well-defined: ``p1 prec p0`` leaves no 2-link."""
    assert rw.future_2links(IDX["p1"], IDX["p0"], hand_causet).size == 0


# --------------------------------------------------------------------------- #
# 2. two_link_distance -- hand-built sanity check (Steps 1-5).
# --------------------------------------------------------------------------- #


def test_two_link_distance_hand_built(hand_causet):
    """The minimising pair is identifiable by hand; the answer is 2 links.

    Step 1: 2-links are ``{f0, f2}``.
    Step 2: the common past is ``{p0, p1}`` with ``p1 prec p0``, so ``p0`` is the
        only maximal element and the only possible minimiser.
        ``[p0, f0] = {x, y}`` (two incomparable elements), so the longest chain
        ``p0 prec x prec f0`` has 3 ELEMENTS = **2 LINKS**. Identically for
        ``f2``. For contrast ``[p1, f0] = {p0, x, y}`` gives 4 elements = 3
        links, confirming the non-maximal element is never the minimiser.
    Steps 3-5: both 2-links contribute 2, so mean 2, standard error 0.
    """
    x, y, c = IDX["x"], IDX["y"], hand_causet

    assert names_of(rw.maximal_common_past(x, y, c)) == {"p0"}
    assert rw.chain_links_between(IDX["p0"], IDX["f0"], c) == 2
    assert rw.chain_links_between(IDX["p0"], IDX["f2"], c) == 2
    assert rw.chain_links_between(IDX["p1"], IDX["f0"], c) == 3

    res = rw.two_link_distance(x, y, c)
    mean, sem, n_two_links = res.as_tuple()
    assert (mean, sem, n_two_links) == (2.0, 0.0, 2)
    assert res.units == "links"
    assert res.per_link_chain_links.tolist() == [2, 2]
    assert res.per_link_interval_size.tolist() == [2, 2]  # {x, y} in both
    assert res.n_past_candidates == 2 and res.n_past_scanned == 1


def test_two_link_distance_reports_the_selected_pair(hand_causet):
    """The ``(p, f_i)`` pairs are exposed, and they reproduce the reported values.

    Needed by the offset diagnostic: only the coordinates of the selected pair
    can separate a calibration bias from a geometric one.
    """
    x, y, c = IDX["x"], IDX["y"], hand_causet
    res = rw.two_link_distance(x, y, c)
    assert names_of(res.per_link_future) == {"f0", "f2"}
    assert names_of(res.per_link_past) == {"p0"}  # the sole maximal common past
    for p, f, links in zip(
        res.per_link_past, res.per_link_future, res.per_link_chain_links
    ):
        assert rw.chain_links_between(int(p), int(f), c) == int(links)


def test_selected_pair_reproduces_the_reported_links_on_sprinklings(random_causets):
    """Same check on real causets: the recorded pair *is* the reported minimum."""
    rng = np.random.default_rng(31)
    checked = 0
    for c in random_causets:
        for x, y in spacelike_pairs(c, rng, 6):
            res = rw.two_link_distance(x, y, c)
            assert res.per_link_past.size == res.n_two_links
            assert res.per_link_future.size == res.n_two_links
            for p, f, links in zip(
                res.per_link_past, res.per_link_future, res.per_link_chain_links
            ):
                assert rw.chain_links_between(int(p), int(f), c) == int(links)
                # and it really is minimal over the whole common past
                whole = np.flatnonzero(c[:, x] & c[:, y])
                assert int(links) == int(rw.chain_links_to_target(int(f), whole, c).min())
                checked += 1
    assert checked > 0


def test_two_link_distance_maximal_past_reduction_is_exact(hand_causet):
    """Scanning only maximal common-past elements must not change the answer."""
    x, y, c = IDX["x"], IDX["y"], hand_causet
    fast = rw.two_link_distance(x, y, c)
    slow = rw.two_link_distance(x, y, c, exhaustive_past=True)
    assert fast.as_tuple() == slow.as_tuple()
    assert slow.n_past_scanned == 2 and fast.n_past_scanned == 1


def test_two_link_distance_calibration_is_the_link_count_times_a_scale(hand_causet):
    """With ``rho`` given, every distance is eqs. (1)-(2) applied to the links."""
    rho = 500.0
    res = rw.two_link_distance(IDX["x"], IDX["y"], hand_causet, rw.RW_M3, rho=rho)
    assert res.units == "length"
    scale = rw.proper_time_from_chain_links(1.0, rho, rw.RW_M3)
    assert res.mean == pytest.approx(2.0 * scale)
    assert res.per_link_distance == pytest.approx(np.array([2.0, 2.0]) * scale)


def test_two_link_distance_refuses_a_timelike_pair(hand_causet):
    with pytest.raises(ValueError, match="causally related"):
        rw.two_link_distance(IDX["p1"], IDX["p0"], hand_causet)


def test_two_link_distance_reports_zero_when_no_2links_exist():
    """Two isolated events: nothing found, and that is reported, not crashed on."""
    c = np.zeros((2, 2), dtype=bool)
    res = rw.two_link_distance(0, 1, c)
    assert np.isnan(res.mean) and np.isnan(res.std_error)
    assert res.n_two_links == 0 and res.n_two_links_found == 0


# --------------------------------------------------------------------------- #
# 2b. naive_distance -- the Section II.B control.
# --------------------------------------------------------------------------- #


def test_naive_distance_hand_built(hand_causet):
    """Naive minimises over the whole common future, 2-links and non-2-links alike.

    By hand: candidates are ``p0`` (the only maximal common-past element) crossed
    with the four minimal common-future elements, so 4 pairs. ``[p0, fbad_x] =
    {x, y, z_x}`` gives the chain ``p0 prec x prec z_x prec fbad_x`` = 4 elements
    = 3 links, and ``fbad_y`` likewise; ``f0`` and ``f2`` give 2 links. The
    minimum is therefore 2, attained at ``(p0, f0)``.
    """
    x, y, c = IDX["x"], IDX["y"], hand_causet
    assert rw.chain_links_between(IDX["p0"], IDX["fbad_x"], c) == 3
    assert rw.chain_links_between(IDX["p0"], IDX["fbad_y"], c) == 3

    res = rw.naive_distance(x, y, c)
    assert res.chain_links == 2
    assert res.n_pairs == 4  # 1 maximal past x 4 minimal future
    assert res.argmin == (IDX["p0"], IDX["f0"])
    assert res.units == "links"


def test_naive_distance_never_exceeds_the_2link_average(random_causets):
    """``naive <= min_i d^i <= mean_i d^i``, since 2-links are minimal-future elements.

    A structural inequality, and the reason the two estimators can be plotted on
    shared axes: any gap between them is the averaging, not a different geometry.
    """
    rng = np.random.default_rng(23)
    compared = 0
    for c in random_causets:
        for x, y in spacelike_pairs(c, rng, 6):
            two = rw.two_link_distance(x, y, c)
            if two.n_two_links == 0:
                continue
            naive = rw.naive_distance(x, y, c)
            assert naive.chain_links <= two.per_link_chain_links.min()
            assert naive.chain_links <= two.mean
            compared += 1
    assert compared > 0


def test_naive_distance_refuses_a_timelike_pair(hand_causet):
    with pytest.raises(ValueError, match="causally related"):
        rw.naive_distance(IDX["p1"], IDX["p0"], hand_causet)


# --------------------------------------------------------------------------- #
# 3. Calibration arithmetic (eqs. 1-2).
# --------------------------------------------------------------------------- #


def test_proper_time_from_chain_links_inverts_equation_1():
    """Round-trip: feed in the ``L`` that eq. (1) predicts for a known interval.

    For a diamond of proper time ``tau`` at density ``rho``, eq. (2) gives
    ``V = eta(3) tau^3`` and eq. (1) gives ``L = m_3 (rho V)^{1/3}``. Feeding
    that ``L`` back must return ``tau``.
    """
    rho, tau, m3 = 2000.0, 0.37, rw.RW_M3
    volume = sprinkle3d.diamond_volume_3d(tau)
    n_links = m3 * (rho * volume) ** (1.0 / 3.0)
    assert rw.proper_time_from_chain_links(n_links, rho, m3) == pytest.approx(tau)


def test_proper_time_scales_as_rho_to_the_minus_one_third():
    """Doubling the density shrinks the length a chain link represents by 2^(1/3)."""
    a = rw.proper_time_from_chain_links(10.0, 1000.0)
    b = rw.proper_time_from_chain_links(10.0, 8000.0)
    assert a / b == pytest.approx(2.0)


def test_proper_time_rejects_nonsense():
    with pytest.raises(ValueError):
        rw.proper_time_from_chain_links(3.0, 0.0)
    with pytest.raises(ValueError):
        rw.proper_time_from_chain_links(3.0, 10.0, m_d=0.0)


# --------------------------------------------------------------------------- #
# 4. chain_links_between -- the Constraint-1 call site.
# --------------------------------------------------------------------------- #


def test_chain_links_between_uses_the_link_convention():
    """A bare covering relation is 1 link (2 elements), a 3-chain is 2 links."""
    # 0 prec 1 prec 2 on the time axis; 0 prec 2 transitively.
    t = np.array([0.0, 1.0, 2.0])
    zeros = np.zeros(3)
    c = causal_matrix_3d(t, zeros, zeros)
    assert rw.chain_links_between(0, 1, c) == 1  # elements 2 -> links 1
    assert rw.chain_links_between(0, 2, c) == 2  # elements 3 -> links 2


def test_chain_links_between_requires_a_related_pair(hand_causet):
    with pytest.raises(ValueError, match="prec"):
        rw.chain_links_between(IDX["x"], IDX["y"], hand_causet)


def test_batched_chain_links_agree_with_the_single_pair_oracle(hand_causet):
    """``chain_links_to_target`` is an optimisation, so it must match exactly.

    Both estimators run on the batched dynamic program; ``chain_links_between``
    is the straightforward per-pair implementation and stays the oracle.
    """
    c = hand_causet
    sources = np.array([IDX["p0"], IDX["p1"]])
    for f in ("f0", "f2", "fbad_x", "fbad_y"):
        batched = rw.chain_links_to_target(IDX[f], sources, c)
        oracle = [rw.chain_links_between(int(p), IDX[f], c) for p in sources]
        assert batched.tolist() == oracle, f


def test_batched_chain_links_rejects_unrelated_sources(hand_causet):
    with pytest.raises(ValueError, match="do not precede"):
        rw.chain_links_to_target(IDX["f0"], np.array([IDX["f1"]]), hand_causet)


def test_batched_chain_links_agree_with_oracle_on_sprinklings(random_causets):
    """Same check on real 2+1 D causets, over every admissible (p, f) pair."""
    rng = np.random.default_rng(29)
    checked = 0
    for c in random_causets:
        for x, y in spacelike_pairs(c, rng, 4):
            past = rw.maximal_common_past(x, y, c)
            for f in rw.minimal_common_future(x, y, c).tolist():
                batched = rw.chain_links_to_target(f, past, c)
                oracle = [rw.chain_links_between(int(p), f, c) for p in past]
                assert batched.tolist() == oracle
                checked += len(oracle)
    assert checked > 50, f"only {checked} pairs exercised"


# --------------------------------------------------------------------------- #
# 5. Randomised checks against brute force on real 2+1 D sprinklings.
# --------------------------------------------------------------------------- #


def brute_force_future_2links(x: int, y: int, c: np.ndarray) -> np.ndarray:
    """Definition 2b transcribed literally with Python loops: the oracle."""
    n = c.shape[0]
    out = []
    for f in range(n):
        if not (c[x, f] and c[y, f]):
            continue
        if any(c[x, z] and c[z, f] for z in range(n)):
            continue
        if any(c[y, z] and c[z, f] for z in range(n)):
            continue
        out.append(f)
    return np.array(out, dtype=int)


def spacelike_pairs(c: np.ndarray, rng, k: int) -> list[tuple[int, int]]:
    """Sample up to ``k`` causally unrelated pairs that have a non-empty common past."""
    n = c.shape[0]
    pairs = []
    for _ in range(20 * k):
        i, j = rng.integers(0, n, size=2)
        if i == j or c[i, j] or c[j, i]:
            continue
        if not (c[:, i] & c[:, j]).any():
            continue
        pairs.append((int(i), int(j)))
        if len(pairs) == k:
            break
    return pairs


@pytest.fixture(scope="module")
def random_causets() -> list[np.ndarray]:
    """Five modest M^3 box sprinklings; small enough for the O(N^3) brute force."""
    out = []
    for seed in range(5):
        s = sprinkle3d.sprinkle_box_3d(
            rho=60.0, t_extent=2.0, x_extent=2.0, y_extent=2.0, seed=1000 + seed
        )
        out.append(causal_matrix_3d(s.t, s.x, s.y))
    return out


def test_future_2links_matches_brute_force_definition(random_causets):
    rng = np.random.default_rng(7)
    checked = 0
    for c in random_causets:
        for x, y in spacelike_pairs(c, rng, 6):
            assert np.array_equal(
                rw.future_2links(x, y, c), brute_force_future_2links(x, y, c)
            )
            checked += 1
    assert checked >= 20, f"only {checked} pairs exercised"


def test_every_2link_is_minimal_in_the_common_future(random_causets):
    """The containment that makes the negative control meaningful.

    If ``g`` in ``fut(x) n fut(y)`` had ``g prec f`` then ``x prec g prec f``, so
    ``f`` would not be a 2-link. Hence 2-links are always a subset of the minimal
    elements -- and the hand causet shows the inclusion is strict.
    """
    rng = np.random.default_rng(11)
    for c in random_causets:
        for x, y in spacelike_pairs(c, rng, 6):
            two = set(rw.future_2links(x, y, c).tolist())
            minimal = set(rw.minimal_common_future(x, y, c).tolist())
            assert two <= minimal


def test_maximal_past_reduction_matches_exhaustive_scan(random_causets):
    """The Step-2 speed reduction is exact, not an approximation."""
    rng = np.random.default_rng(13)
    compared = 0
    for c in random_causets:
        for x, y in spacelike_pairs(c, rng, 5):
            fast = rw.two_link_distance(x, y, c)
            slow = rw.two_link_distance(x, y, c, exhaustive_past=True)
            assert fast.per_link_chain_links.tolist() == slow.per_link_chain_links.tolist()
            assert fast.n_past_scanned <= slow.n_past_scanned == fast.n_past_candidates
            compared += 1
    assert compared >= 15


def test_two_link_distances_respect_the_two_link_resolution_floor(random_causets):
    """Every minimising interval contains both targets, so ``L >= 2`` always.

    ``p prec x prec f`` for any admissible ``p`` and any 2-link ``f``, so the
    longest chain from ``p`` to ``f`` has at least 3 elements. This is the hard
    quantisation floor documented in the module docstring; if it were ever
    violated the interval restriction in ``chain_links_between`` would be wrong.
    """
    rng = np.random.default_rng(17)
    seen = 0
    for c in random_causets:
        for x, y in spacelike_pairs(c, rng, 6):
            res = rw.two_link_distance(x, y, c)
            if res.n_two_links == 0:
                continue
            assert res.per_link_chain_links.min() >= 2
            assert res.per_link_interval_size.min() >= 2
            seen += res.n_two_links
    assert seen > 0
