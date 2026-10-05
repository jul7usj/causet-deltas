"""Structural tests for the Sorkin--Johnston construction (src/causet/sj.py).

Three groups:

1. The index convention of ``G_R`` on a hand-placed 5-element causet, where every
   relation is asserted explicitly from the coordinates.
2. The algebraic identities that define the SJ state (Sorkin 1703.00610 eq. (15);
   ABDRSY 1207.7101 Sec. 2), checked on sprinklings: Hermiticity, +/- pairing,
   W >= 0, W - conj(W) = iDelta, W + conj(W) = |iDelta| (|iDelta| computed by an
   INDEPENDENT route -- real symmetric eigh of -Delta^2), W conj(W) = 0.
3. Exact hand results (2-chain, 2-antichain, 3-chain, disjoint unions) and the
   convention-catching physics check: the top eigenvector of iDelta on a
   sprinkled diamond is the continuum POSITIVE-frequency mode g_k of ABDRSY eq.
   (SJfunctions2), not its conjugate. A transpose (or sign) error in G_R swaps
   these two overlaps, so this test fails loudly under exactly the error the
   module docstring warns about.

"conj" below is always the ELEMENTWISE complex conjugate.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import brentq

from causet import sj
from causet.order import causal_matrix_1d
from causet.sprinkle import sprinkle_diamond_1d

# Tolerances for floating-point identities on matrices with O(1) entries and
# spectral norm O(N). ATOL is per-entry; scaled by N where a sum over N terms
# accumulates rounding.
ATOL = 1e-12

SPRINKLE_SEEDS = (11, 12, 13)
SPRINKLE_RHO = 600.0  # E[N] = 300 in the tau = 1 diamond
SPRINKLE_TAU = 1.0


def causet_from_tx(t, x):
    """Causal matrix of explicit 1+1 D points via the Phase-1 substrate."""
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    return causal_matrix_1d(t + x, t - x)


def sj_pipeline(c):
    g = sj.retarded_green_2d(c)
    i_delta = sj.pauli_jordan(g)
    w, (evals, evecs, tol) = sj.sj_wightman(i_delta, return_spectrum=True)
    return g, i_delta, w, evals, tol


@pytest.fixture(scope="module", params=SPRINKLE_SEEDS)
def sprinkled(request):
    s = sprinkle_diamond_1d(SPRINKLE_RHO, SPRINKLE_TAU, request.param)
    c = causal_matrix_1d(s.u, s.v)
    return (s, c) + sj_pipeline(c)


def abs_hermitian_independent(i_delta):
    """|iDelta| = sqrt(-Delta^2) by real-symmetric eigh of -Delta^2 (Sorkin eq. 15).

    Independent of ``sj_wightman``: different matrix (real, PSD), no pairing of
    complex eigenvectors, no positive/negative selection.
    """
    delta = (i_delta / 1j).real
    m = -(delta @ delta)
    m = 0.5 * (m + m.T)
    mu, q = np.linalg.eigh(m)
    return (q * np.sqrt(np.clip(mu, 0.0, None))) @ q.T


# --------------------------------------------------------------------------- #
# 1. G_R index convention on a hand-placed causet.
# --------------------------------------------------------------------------- #

# Names and (t, x). p is the bottom, q the top; a, b are spacelike to each other;
# s is spacelike to everything (|dx| > |dt| against every other point).
HAND_NAMES = ["p", "a", "b", "s", "q"]
HAND_T = [0.0, 1.0, 1.0, 0.5, 2.0]
HAND_X = [0.0, 0.5, -0.5, 3.0, 0.0]
# Every "earlier prec later" relation, read off from the coordinates.
HAND_RELATIONS = {("p", "a"), ("p", "b"), ("p", "q"), ("a", "q"), ("b", "q")}


def test_hand_causet_relations_are_as_designed():
    c = causet_from_tx(HAND_T, HAND_X)
    idx = {n: i for i, n in enumerate(HAND_NAMES)}
    for x in HAND_NAMES:
        for y in HAND_NAMES:
            assert bool(c[idx[x], idx[y]]) == ((x, y) in HAND_RELATIONS), (x, y)


def test_green_is_half_on_retarded_direction_only():
    """G_R[x, y] = 1/2 iff x prec y (first index = earlier element), else 0."""
    c = causet_from_tx(HAND_T, HAND_X)
    g = sj.retarded_green_2d(c)
    idx = {n: i for i, n in enumerate(HAND_NAMES)}
    for x in HAND_NAMES:
        for y in HAND_NAMES:
            want = sj.GREEN_2D_AMPLITUDE if (x, y) in HAND_RELATIONS else 0.0
            assert g[idx[x], idx[y]] == want, (x, y)
    # spot checks spelled out: p -> q propagates, q -> p does not; spacelike 0.
    assert g[idx["p"], idx["q"]] == 0.5 and g[idx["q"], idx["p"]] == 0.0
    assert g[idx["a"], idx["b"]] == 0.0 and g[idx["s"], idx["p"]] == 0.0


def test_pauli_jordan_sign_matches_abdrsy_eq36():
    """iDelta[x, y] = +i/2 for x prec y, -i/2 for y prec x, 0 if spacelike.

    ABDRSY 1207.7101 eq. (36): iDelta(X,X') = -(i/2)[theta(u-u') + theta(v-v') - 1].
    """
    t, x = np.array(HAND_T), np.array(HAND_X)
    i_delta = sj.pauli_jordan(sj.retarded_green_2d(causet_from_tx(t, x)))
    u, v = t + x, t - x
    du = u[:, None] - u[None, :]
    dv = v[:, None] - v[None, :]
    timelike = (du * dv) > 0  # strictly inside the light cone (no nulls here)
    eq36 = -0.5j * ((du > 0).astype(float) + (dv > 0).astype(float) - 1.0)
    eq36 = np.where(timelike, eq36, 0.0)
    assert np.array_equal(i_delta, eq36)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((2, 3), dtype=bool),
        np.eye(3, dtype=bool),
        np.array([[0, 1], [1, 0]], dtype=bool),
    ],
    ids=["non-square", "reflexive", "symmetric"],
)
def test_green_rejects_non_orders(bad):
    with pytest.raises(ValueError):
        sj.retarded_green_2d(bad)


# --------------------------------------------------------------------------- #
# 2. Structural identities on sprinklings.
# --------------------------------------------------------------------------- #


def test_sprinkled_green_is_strictly_retarded(sprinkled):
    s, c, g, *_ = sprinkled
    # x prec y in the continuum <=> t_x < t_y and |dx| < dt.
    dt = s.t[None, :] - s.t[:, None]
    dx = np.abs(s.x[None, :] - s.x[:, None])
    future = (dt > 0) & (dx < dt)
    assert np.array_equal(g != 0, future)
    assert np.all(g[future] == 0.5)


def test_i_delta_hermitian_and_purely_imaginary(sprinkled):
    _, _, _, i_delta, *_ = sprinkled
    assert np.all(i_delta.real == 0.0)
    assert np.array_equal(i_delta, i_delta.conj().T)


def test_i_delta_eigenvalues_pair(sprinkled):
    """Spectrum is symmetric: conj(iDelta) = -iDelta, so lambda <-> -lambda."""
    _, _, _, i_delta, w, evals, tol = sprinkled
    n = evals.size
    lam_max = np.max(np.abs(evals))
    assert np.allclose(evals, -evals[::-1], rtol=0, atol=1e-12 * n * lam_max)
    assert np.sum(evals > tol) == np.sum(evals < -tol)


def test_w_hermitian_and_psd(sprinkled):
    _, _, _, _, w, evals, _ = sprinkled
    n = w.shape[0]
    assert np.allclose(w, w.conj().T, rtol=0, atol=ATOL * n)
    mu = np.linalg.eigvalsh(w)
    assert mu.min() >= -1e-12 * n * np.max(np.abs(evals))


def test_commutator_recovered(sprinkled):
    """(a) W - conj(W) = iDelta."""
    _, _, _, i_delta, w, *_ = sprinkled
    n = w.shape[0]
    assert np.allclose(w - w.conj(), i_delta, rtol=0, atol=ATOL * n)


def test_w_is_positive_part_independent_route(sprinkled):
    """W + conj(W) = |iDelta|, with |iDelta| from eigh of the REAL matrix -Delta^2."""
    _, _, _, i_delta, w, *_ = sprinkled
    n = w.shape[0]
    abs_id = abs_hermitian_independent(i_delta)
    # sqrt of near-zero eigenvalues amplifies rounding: sqrt(N eps lam) ~ 1e-6.
    assert np.allclose(w + w.conj(), abs_id, rtol=0, atol=1e-6)
    # and the full positive-part formula, Sorkin 1703.00610 eq. (15)
    assert np.allclose(w, 0.5 * (i_delta + abs_id), rtol=0, atol=1e-6)


def test_orthogonal_supports(sprinkled):
    """(c) W conj(W) = 0  (ABDRSY 1207.7101 Sec. 2, ground-state condition)."""
    _, _, _, _, w, evals, _ = sprinkled
    n = w.shape[0]
    lam_max = np.max(np.abs(evals))
    assert np.max(np.abs(w @ w.conj())) <= 1e-12 * n * lam_max**2


def test_imag_w_is_half_delta(sprinkled):
    """Im W = Delta/2, so Im W[x, y] = +1/4 exactly when x prec y."""
    _, c, _, _, w, *_ = sprinkled
    n = w.shape[0]
    want = 0.25 * (c.astype(float) - c.T.astype(float))
    assert np.allclose(w.imag, want, rtol=0, atol=ATOL * n)


def test_relabelling_covariance(sprinkled):
    """W depends on the causet, not on the labelling: W[P][:, P] = W(P-relabelled)."""
    _, c, _, _, w, *_ = sprinkled
    perm = np.random.default_rng(0).permutation(c.shape[0])
    w_perm = sj.sj_wightman(sj.pauli_jordan(sj.retarded_green_2d(c[np.ix_(perm, perm)])))
    assert np.allclose(w_perm, w[np.ix_(perm, perm)], rtol=0, atol=1e-9)


# --------------------------------------------------------------------------- #
# 3. Exact hand results.
# --------------------------------------------------------------------------- #


def chain_matrix(n):
    return np.triu(np.ones((n, n), dtype=bool), k=1)


def test_two_chain_exact():
    """1 prec 2: iDelta = (i/2)[[0,1],[-1,0]], eigenvalues +-1/2,
    v+ = (1, -i)/sqrt2, so W = (1/2) v+ v+^dagger = (1/4)[[1, i], [-i, 1]]."""
    _, _, w, evals, _ = sj_pipeline(chain_matrix(2))
    assert np.allclose(evals, [-0.5, 0.5], rtol=0, atol=ATOL)
    assert np.allclose(w, 0.25 * np.array([[1, 1j], [-1j, 1]]), rtol=0, atol=ATOL)


def test_two_antichain_is_zero():
    _, i_delta, w, _, _ = sj_pipeline(np.zeros((2, 2), dtype=bool))
    assert np.all(i_delta == 0) and np.all(w == 0)


def test_three_chain_closed_form():
    """1 prec 2 prec 3. Delta = A/2, A = [[0,1,1],[-1,0,1],[-1,-1,0]].

    -A^2 = 3 P with P = I - w w^T / 3 the projector off the kernel w = (1,-1,1),
    so |iDelta| = (sqrt3/2) P and W = (sqrt3/4) P + (i/4) A. Spectrum {-sqrt3/2, 0,
    +sqrt3/2}.
    """
    a = np.array([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]], dtype=float)
    k = np.array([1.0, -1.0, 1.0])
    p = np.eye(3) - np.outer(k, k) / 3.0
    _, i_delta, w, evals, _ = sj_pipeline(chain_matrix(3))
    assert np.allclose(i_delta, 0.5j * a, rtol=0, atol=ATOL)
    assert np.allclose(evals, [-np.sqrt(3) / 2, 0.0, np.sqrt(3) / 2], rtol=0, atol=ATOL)
    assert np.allclose(w, np.sqrt(3) / 4 * p + 0.25j * a, rtol=0, atol=ATOL)
    # the kernel vector is annihilated by W (excluded zero mode)
    assert np.allclose(w @ k, 0.0, rtol=0, atol=ATOL)


def test_disjoint_union_is_block_diagonal():
    """Mutually spacelike components do not talk: W(A u B) = W(A) (+) W(B)."""
    ca, cb = chain_matrix(2), chain_matrix(3)
    c = np.zeros((5, 5), dtype=bool)
    c[:2, :2], c[2:, 2:] = ca, cb
    w = sj_pipeline(c)[2]
    assert np.allclose(w[:2, :2], sj_pipeline(ca)[2], rtol=0, atol=ATOL)
    assert np.allclose(w[2:, 2:], sj_pipeline(cb)[2], rtol=0, atol=ATOL)
    assert np.allclose(w[:2, 2:], 0.0, rtol=0, atol=ATOL)


# --------------------------------------------------------------------------- #
# Convention-catching physics check: positive frequency.
# --------------------------------------------------------------------------- #

# Smallest positive root of tan(x) = 2x: ABDRSY eq. (SJfunctions2), the
# g-family with the largest continuum eigenvalue lambda = L / k = L^2 / x1.
TAN_2X_ROOT_1 = brentq(lambda z: np.tan(z) - 2.0 * z, 1.0, 1.5)

# Measured during development (seeds 0-4, N = 367-442): overlap with g 0.994-0.999,
# with conj(g) <= 0.001. Thresholds are set well outside that spread; a
# transposed G_R would swap the two numbers.
POS_FREQ_MIN_OVERLAP = 0.95
NEG_FREQ_MAX_OVERLAP = 0.02


def test_top_mode_is_positive_frequency(sprinkled):
    """Top eigenvector of iDelta ~ g_k = e^{-iku} + e^{-ikv} - 2cos(kL), k = x1/L.

    ABDRSY coordinates: u = (t+x)/sqrt2, v = (t-x)/sqrt2, diamond u,v in (-L, L).
    Our diamond is (t+x, t-x) in [0, tau]^2, so u_A = (u - tau/2)/sqrt2 and
    L = tau / (2 sqrt2).
    """
    s, *_ = sprinkled
    _, _, _, i_delta, w, evals, _ = sprinkled
    _, evecs = sj.pauli_jordan_spectrum(i_delta)
    top = evecs[:, -1]
    tau = s.tau
    big_l = tau / (2.0 * np.sqrt(2.0))
    ua = (s.u - tau / 2.0) / np.sqrt(2.0)
    va = (s.v - tau / 2.0) / np.sqrt(2.0)
    k = TAN_2X_ROOT_1 / big_l
    g = np.exp(-1j * k * ua) + np.exp(-1j * k * va) - 2.0 * np.cos(k * big_l)
    g /= np.linalg.norm(g)
    assert abs(np.vdot(g, top)) ** 2 > POS_FREQ_MIN_OVERLAP
    assert abs(np.vdot(g.conj(), top)) ** 2 < NEG_FREQ_MAX_OVERLAP
