"""Sorkin's spacetime entanglement entropy (SSEE) on a causal set (Phase 3a Gate 3).

Source: Sorkin--Yazdi (SY), CQG 35 (2018) 074004, arXiv:1611.10281, Sec. 2--3.
Equations are quoted by their LaTeX ``\\label``.

Definition (SY eq. (gee), (s4))
-------------------------------
For a region O with W and Delta restricted to O,

    W v = i lambda Delta v ,     Delta v != 0 ,          S = sum_lambda lambda ln|lambda| ,

with v "in the image of Delta" (SY Sec. 2). The lambda are the eigenvalues of
``-i Delta^{-1} W`` on that image and come in pairs (lambda, 1 - lambda) (SY Sec. 3).

How it is solved here -- never by inverting a singular matrix
--------------------------------------------------------------
``iDelta_O`` is Hermitian. Let ``U`` (orthonormal columns) span the eigenvectors of
``iDelta_O`` that are KEPT (|sigma| >= cut; the cut is the numerical zero for the
untruncated case, or a physical truncation), with eigenvalues ``sigma`` (diagonal D).
On that support ``iDelta = U D U^dagger`` is invertible, and the problem becomes the
r x r problem ``W' x = lambda D x`` with ``W' = U^dagger W U`` -- W is projected onto
the same support, which is exactly SY's "projected out from W and Delta". Writing
``A = |D|^{-1/2} W' |D|^{-1/2}`` (Hermitian, PSD) and ``J = sign(D)``, the lambda are the
eigenvalues of ``J A`` (similar to ``D^{-1} W'``); D is diagonal with |sigma| >= cut > 0,
so no ill-conditioned inverse appears. ``J A`` is J-Hermitian, so its eigenvalues are
real for a physical W; the largest imaginary part is returned as a diagnostic.

Double truncation (SY Sec. 3, text around Fig. "trunc")
------------------------------------------------------
"We first truncate both Delta and W in the larger diamond ... We then restrict both
matrices to the smaller diamond. Call these W^R and Delta^R. We then do a second
truncation on them, based on the spectrum of Delta^R." Cuts lambda_min ~ sqrt(N_L)/4pi
and sqrt(N_l)/4pi. After the first truncation, iDelta^R = V D V^dagger with V the
inner rows of the kept outer eigenvectors, so its non-zero spectrum is obtained
EXACTLY from a QR factorisation of V (``inner_support_lowrank``), not from an N_l x N_l
eigensolve whose exact zeros would be indistinguishable from round-off.
"""

from __future__ import annotations

import numpy as np

from causet import sj

#: |lambda| below this contributes lambda ln|lambda| -> 0 (the 0 ln 0 = 0 convention).
ENTROPY_ZERO = 1e-300


def ssee_entropy(lam: np.ndarray) -> float:
    """SY eq. (s4): ``S = sum lambda ln|lambda|`` with ``0 ln 0 = 0``."""
    lam = np.asarray(lam, float)
    nz = np.abs(lam) > ENTROPY_ZERO
    return float(np.sum(lam[nz] * np.log(np.abs(lam[nz]))))


def generalized_lambdas(w: np.ndarray, u: np.ndarray, sigma: np.ndarray):
    """Solutions lambda of SY eq. (gee) on the support ``span(u)``.

    ``u``: orthonormal columns (eigenvectors of iDelta_O kept), ``sigma``: their
    iDelta eigenvalues (all non-zero). Returns ``(lambda, max |Im lambda|)``.
    """
    sigma = np.asarray(sigma, float)
    if sigma.size == 0:
        return np.zeros(0), 0.0
    wp = u.conj().T @ w @ u
    s = 1.0 / np.sqrt(np.abs(sigma))
    a = (s[:, None] * wp) * s[None, :]
    a = 0.5 * (a + a.conj().T)
    ev = np.linalg.eigvals(np.sign(sigma)[:, None] * a)
    return ev.real, float(np.max(np.abs(ev.imag)))


def inner_support_full(i_delta_o: np.ndarray, cut: float):
    """Eigenvectors of a (full-rank-ish) restricted iDelta with |sigma| >= cut."""
    sig, vec = np.linalg.eigh(i_delta_o)
    keep = np.abs(sig) >= cut
    return vec[:, keep], sig[keep], sig


def inner_support_lowrank(v_rows: np.ndarray, d: np.ndarray):
    """Non-zero spectrum of ``V diag(d) V^dagger`` exactly, via QR of V.

    ``V = Q R`` gives ``V D V^dagger = Q (R D R^dagger) Q^dagger``; the eigenpairs of
    the small Hermitian ``R D R^dagger`` (Q y, mu) are all eigenpairs of the
    restriction with non-zero eigenvalue. Returns ``(Q Y, mu)``, mu ascending.
    """
    q, r = np.linalg.qr(v_rows)
    m = (r * d[None, :]) @ r.conj().T
    mu, y = np.linalg.eigh(0.5 * (m + m.conj().T))
    return q @ y, mu


def double_truncated_ssee(lam_pos: np.ndarray, vec_pos: np.ndarray, inner: np.ndarray,
                          n_outer: int, inner_cut):
    """SSEE of the inner region after SY's double truncation.

    ``lam_pos`` (descending) / ``vec_pos``: positive eigenpairs of the outer iDelta.
    The first truncation keeps the top ``n_outer`` positive modes and their
    negative partners (iDelta is purely imaginary, so the partner of (lam, v) is
    (-lam, conj v)). ``inner_cut``: a float magnitude cut, or a callable taking the
    descending positive inner spectrum and returning the number of positive modes
    to keep (the linear truncation). Returns a dict with S, the lambdas and counts.
    """
    lam = lam_pos[:n_outer]
    vp = vec_pos[inner, :n_outer]
    v_both = np.concatenate([vp, vp.conj()], axis=1)
    d = np.concatenate([lam, -lam])
    q, mu = inner_support_lowrank(v_both, d)
    tol = sj.zero_tolerance(mu, len(inner))              # round-off zeros of the QR route
    q, mu = q[:, np.abs(mu) > tol], mu[np.abs(mu) > tol]
    w_r = (vp * lam) @ vp.conj().T                       # W^R: restriction of truncated W
    mu_pos = np.sort(mu[mu > 0])[::-1]
    if callable(inner_cut):
        k = int(inner_cut(mu_pos))
        cut2 = mu_pos[k - 1] if k >= 1 else np.inf
    else:
        cut2 = float(inner_cut)
    keep = np.abs(mu) >= cut2
    lam_g, im = generalized_lambdas(w_r, q[:, keep], mu[keep])
    return {"S": ssee_entropy(lam_g), "lam": lam_g, "max_imag": im,
            "n_outer": int(n_outer), "n_inner": int(np.sum(mu[keep] > 0)), "cut2": cut2}


def untruncated_ssee(w_o: np.ndarray, i_delta_o: np.ndarray, zero_tol: float | None = None):
    """SSEE with only the numerical kernel of the restricted iDelta excluded.

    Default ``zero_tol``: ``sj.zero_tolerance`` of the restricted spectrum (the
    numerical-rank threshold; NOT a physical truncation).
    """
    sig, vec = np.linalg.eigh(i_delta_o)
    tol = sj.zero_tolerance(sig, sig.size) if zero_tol is None else float(zero_tol)
    keep = np.abs(sig) > tol
    u, sig = vec[:, keep], sig[keep]
    lam_g, im = generalized_lambdas(w_o, u, sig)
    return {"S": ssee_entropy(lam_g), "lam": lam_g, "max_imag": im,
            "n_inner": int(np.sum(sig > 0))}
