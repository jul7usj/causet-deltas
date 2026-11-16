"""Structural tests for the SSEE (src/causet/ssee.py), run before any Gate 3 data.

1. Known answer on a SINGULAR iDelta: two "thermal" mode pairs plus a 3-dim kernel
   (and a W component living in the kernel, which the support projection must
   ignore). Each pair with occupation n gives lambda = 1+n and -n, and contributes
   the bosonic entropy (1+n) ln(1+n) - n ln n.
2. Purity: the SJ state is pure, so the SSEE of the whole causet vanishes --
   untruncated, and after the double truncation with the inner region = everything.
3. The low-rank inner spectrum equals a full eigensolve of the restricted
   truncated iDelta (non-zero part), and SY's (lambda, 1 - lambda) pairing holds.
"""

from __future__ import annotations

import numpy as np
import pytest

from causet import sj, ssee
from causet.order import causal_matrix_1d
from causet.sprinkle import sprinkle_diamond_1d


def _thermal_case(seed=0):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((7, 7)))
    sig, occ = [1.7, 0.4], [0.3, 2.5]
    i_delta = np.zeros((7, 7), complex)
    w = np.zeros((7, 7), complex)
    for k, (s, n) in enumerate(zip(sig, occ)):
        u = (q[:, 2 * k] + 1j * q[:, 2 * k + 1]) / np.sqrt(2)
        uu, bb = np.outer(u, u.conj()), np.outer(u.conj(), u)
        i_delta += s * (uu - bb)
        w += s * ((1 + n) * uu + n * bb)
    w += 0.9 * np.outer(q[:, 5], q[:, 5])        # lives in ker(iDelta): must be ignored
    expected = sum((1 + n) * np.log(1 + n) - n * np.log(n) for n in occ)
    return w, i_delta, occ, expected


def test_entropy_zero_convention():
    assert ssee.ssee_entropy([0.0, 1.0, 0.0]) == 0.0


def test_known_answer_singular_delta():
    w, i_delta, occ, expected = _thermal_case()
    out = ssee.untruncated_ssee(w, i_delta, zero_tol=1e-10)
    assert out["n_inner"] == 2                     # kernel (3 dims) excluded
    want = sorted([1 + occ[0], -occ[0], 1 + occ[1], -occ[1]])
    assert np.allclose(np.sort(out["lam"]), want, atol=1e-12)
    assert abs(out["S"] - expected) < 1e-12
    assert out["max_imag"] < 1e-12


def test_commutator_identity_of_known_case():
    w, i_delta, _, _ = _thermal_case(3)
    # the kernel term is real symmetric, so it drops out of W - conj(W)
    assert np.allclose(w - w.conj(), i_delta, atol=1e-14)


@pytest.fixture(scope="module", params=[11, 12])
def causet(request):
    s = sprinkle_diamond_1d(300 / 0.5, 1.0, 7000 + request.param)
    i_delta = sj.pauli_jordan(sj.retarded_green_2d(causal_matrix_1d(s.u, s.v)))
    ev, vec = np.linalg.eigh(i_delta)
    pos = np.nonzero(ev > sj.zero_tolerance(ev, s.n))[0][::-1]
    return s, i_delta, ev[pos], vec[:, pos]


def test_purity_untruncated(causet):
    s, i_delta, lam, vec = causet
    w = (vec * lam) @ vec.conj().T
    out = ssee.untruncated_ssee(w, i_delta, sj.zero_tolerance(np.r_[lam, -lam], s.n))
    assert abs(out["S"]) < 1e-8
    assert np.allclose(np.sort(out["lam"]), np.r_[np.zeros(lam.size), np.ones(lam.size)],
                       atol=1e-8)


def test_purity_double_truncated(causet):
    s, _, lam, vec = causet
    k = 25
    out = ssee.double_truncated_ssee(lam, vec, np.arange(s.n), k, 0.5 * (lam[k - 1] + lam[k]))
    assert out["n_inner"] == k
    assert abs(out["S"]) < 1e-9


def test_lowrank_inner_spectrum_matches_full(causet):
    s, _, lam, vec = causet
    k = 30
    inner = np.nonzero((np.abs(s.u - 0.5) <= 0.25) & (np.abs(s.v - 0.5) <= 0.25))[0]
    vp = vec[inner, :k]
    w_r = (vp * lam[:k]) @ vp.conj().T
    full = w_r - w_r.conj()                          # iDelta^t restricted = W^R - conj(W^R)
    v_both = np.concatenate([vp, vp.conj()], axis=1)
    _, mu = ssee.inner_support_lowrank(v_both, np.r_[lam[:k], -lam[:k]])
    ref = np.linalg.eigvalsh(full)
    ref = np.sort(ref[np.abs(ref) > 1e-9 * np.abs(ref).max()])
    assert np.allclose(np.sort(mu[np.abs(mu) > 1e-9 * np.abs(mu).max()]), ref, atol=1e-9)


def test_nested_pairs_and_positive_entropy(causet):
    s, _, lam, vec = causet
    inner = np.nonzero((np.abs(s.u - 0.5) <= 0.25) & (np.abs(s.v - 0.5) <= 0.25))[0]
    n_out = int(np.sum(lam >= np.sqrt(s.n) / (4 * np.pi)))
    out = ssee.double_truncated_ssee(lam, vec, inner, n_out,
                                     np.sqrt(inner.size) / (4 * np.pi))
    lg = np.sort(out["lam"])
    assert np.allclose(lg, np.sort(1.0 - lg), atol=1e-8)      # SY: pairs (lambda, 1 - lambda)
    assert out["S"] > 0 and np.isfinite(out["S"])
    assert out["max_imag"] < 1e-8
