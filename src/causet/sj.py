"""Sorkin--Johnston (SJ) two-point function on a 1+1 D causal set (massless).

Phase 3a. Builds on the Phase-1 substrate (``sprinkle``, ``order``), which is a
closed, validated dependency and is not modified or re-derived here. This module
builds the SJ Wightman function ``W`` and nothing downstream of it.

The construction, G_R -> iDelta -> W
------------------------------------
1. Retarded Green function (Johnston, CQG 25 (2008) 202001, arXiv:0806.3083,
   eq. (1+1Amplitudes): chain amplitudes ``a = 1/2``, ``b = -m^2/rho``; at
   ``m = 0`` the chain sum terminates after one hop). For the massless field on
   a sprinkling into M^2,

       G_R = (1/2) C ,          C_xy = 1  iff  x prec y ,

   exactly as stated in Sorkin--Yazdi, CQG 35 (2018) 074004, arXiv:1611.10281,
   Sec. 3 (unnumbered display), and Johnston, PhD thesis, arXiv:1010.5514,
   Ch. "Free Quantum Field Theory" ("For sprinklings into M^2 ... K_R = 1/2 C").
   Its sprinkling expectation equals the continuum retarded Green function
   ``(1/2) theta`` at every density (Johnston 2008, Sec. "Fluctuations").

2. Pauli--Jordan (commutator) function (Sorkin--Yazdi Sec. 3; Johnston thesis
   eq. (CommutationCondition)):

       Delta = G_R - G_R^T ,     [phi_x, phi_y] = i Delta_xy .

   ``iDelta`` is purely imaginary and antisymmetric, hence Hermitian.

3. SJ Wightman function (Sorkin, arXiv:1703.00610, eq. (15); Afshordi, Buck,
   Dowker, Rideout, Sorkin, Yazdi, JHEP 10 (2012) 088, arXiv:1207.7101,
   eq. (SJwightman)): the positive spectral part of ``iDelta``,

       W = Pos(iDelta) = sum_{lambda_k > 0} lambda_k v_k v_k^dagger
         = (iDelta + |iDelta|) / 2 .

   Equivalently (1207.7101 Sec. 2) W is fixed uniquely by
       (a) W - conj(W) = iDelta       (commutator; conj is ELEMENTWISE),
       (b) W >= 0                     (positivity),
       (c) W conj(W) = 0              (orthogonal supports: the "ground state").
   Because W is Hermitian, conj(W) = W^T; note W - W^dagger = 0, not iDelta.

INDEX CONVENTION -- verified against the sources, read before editing
--------------------------------------------------------------------
``G_R[x, y]`` is nonzero iff ``x prec y``: **first index = the earlier (source)
element**. This is Johnston's convention (0806.3083 Sec. 2.1: ``K_ij`` is the
amplitude *from* ``v_i`` *to* ``v_j``) and Sorkin--Yazdi's (``C_xy = 1`` iff
``x prec y``), and it coincides with ``order.causal_matrix_1d`` (``C[i, j]``
True iff ``x_i prec x_j``), so **no transpose is applied**.

The continuum papers (Sorkin 1703.00610 Sec. 2: "G(x,y)=0 unless x succ y";
AAS 1205.1296; ABDRSY 1207.7101 eq. (pjdef)) use the OPPOSITE index order
*and* the opposite sign, because they solve ``(Box - m^2) G = delta`` in
signature (-,+): Sorkin 1703.00610 Sec. 2 gives ``G = -(1/2) theta(u) theta(v)``
in M^2. The two conventions agree on the one physical object,

    iDelta_xy = +i/2   if  x prec y ,    -i/2  if  y prec x ,    0 otherwise,

which is ABDRSY eq. (36), ``iDelta(X,X') = -(i/2)[theta(u-u') + theta(v-v') - 1]``.
Mixing them (the continuum index order with Johnston's +1/2 sign) conjugates
iDelta and so silently swaps W with conj(W), i.e. retarded with advanced. ABDRSY
footnote to eq. (SJfunctions2) records exactly this error in Johnston's thesis
continuum section. Physical consequence used as a test: ``Im W_xy = +1/4`` for
``x prec y`` (Im W = Delta/2, ABDRSY Sec. 3.1), and the positive-eigenvalue
eigenvectors are positive-frequency, ~ ``exp(-i k u)`` (ABDRSY eqs.
(SJfunctions1)-(SJfunctions2)).

Normalisation
-------------
The matrices here carry no density factor: ``iDelta`` is the bare matrix of
Sorkin--Yazdi. As an integral operator, ``(iDelta f)(x) = int dV' iDelta(x,x')
f(x')`` is approximated by ``(1/rho) sum_y``, so matrix eigenvalues are ``rho``
times continuum eigenvalues (Sorkin--Yazdi Sec. 3: ``lambda^cs = rho
lambda^cont``), while matrix entries ``W_xy`` approximate the continuum kernel
``W(X_x, X_y)`` directly (the ``rho`` in the eigenvalue cancels the ``1/rho``
in the unit-normalised eigenvector's outer product).
"""

from __future__ import annotations

import numpy as np

#: Retarded Green function amplitude for the massless scalar in 1+1 D:
#: ``G_R = GREEN_2D_AMPLITUDE * C``. Johnston 0806.3083 eq. (1+1Amplitudes),
#: ``a = 1/2``; Sorkin--Yazdi 1611.10281 Sec. 3, ``G_R = (1/2) C``.
GREEN_2D_AMPLITUDE = 0.5

#: An eigenvalue of ``iDelta`` with ``|lambda| <= ZERO_EIG_RTOL_PER_N * N *
#: max|lambda|`` is treated as an exact zero (kernel of Delta) and excluded from
#: ``W``. ``N * eps * ||A||_2`` is the standard numerical-rank threshold (same
#: convention as ``numpy.linalg.matrix_rank``); it is the scale of the backward
#: error of a Hermitian eigensolver. A kernel eigenvalue that slips past it
#: contributes at most ``tol`` to any entry of ``W``, so the choice is benign.
#: This is a NUMERICAL zero only -- it is NOT the Sorkin--Yazdi physical
#: truncation ``lambda_min ~ sqrt(N)/(4 pi)``, which belongs to Gate 3.
ZERO_EIG_RTOL_PER_N = float(np.finfo(np.float64).eps)


def retarded_green_2d(causal_matrix: np.ndarray, *, check: bool = True) -> np.ndarray:
    """Massless retarded Green function on a 1+1 D causet: ``G_R = (1/2) C``.

    Parameters
    ----------
    causal_matrix:
        Strict causal matrix with ``C[x, y]`` True iff ``x prec y`` (the
        convention of ``order.causal_matrix_1d``). See the module docstring for
        why no transpose is taken.
    check:
        If True, verify that ``C`` is square, irreflexive and antisymmetric
        (a strict order's matrix). Transitivity is not checked here (O(N^3));
        ``order.is_transitive`` does that.

    Returns
    -------
    G_R : float64 array, shape (N, N), with ``G_R[x, y] = 1/2`` iff ``x prec y``.
    """
    c = np.asarray(causal_matrix)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError(f"causal matrix must be square, got shape {c.shape}")
    c = c.astype(bool)
    if check:
        if np.any(np.diag(c)):
            raise ValueError("causal matrix must be irreflexive (zero diagonal)")
        if np.any(c & c.T):
            raise ValueError("causal matrix must be antisymmetric (no x<y and y<x)")
    return GREEN_2D_AMPLITUDE * c.astype(np.float64)


def pauli_jordan(g_r: np.ndarray) -> np.ndarray:
    """Pauli--Jordan matrix ``iDelta = i (G_R - G_R^T)`` (complex Hermitian).

    With ``G_R`` from ``retarded_green_2d``, ``iDelta[x, y] = +i/2`` for
    ``x prec y`` -- the sign of ABDRSY 1207.7101 eq. (36).
    """
    g = np.asarray(g_r, dtype=np.float64)
    if g.ndim != 2 or g.shape[0] != g.shape[1]:
        raise ValueError(f"G_R must be square, got shape {g.shape}")
    return 1j * (g - g.T)


def zero_tolerance(eigenvalues: np.ndarray, n: int) -> float:
    """Absolute threshold below which an ``iDelta`` eigenvalue counts as zero.

    ``ZERO_EIG_RTOL_PER_N * n * max|lambda|`` (see the constant's docstring).
    """
    lam = np.asarray(eigenvalues)
    scale = float(np.max(np.abs(lam))) if lam.size else 0.0
    return ZERO_EIG_RTOL_PER_N * n * scale


def pauli_jordan_spectrum(i_delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigen-decomposition of the Hermitian ``iDelta`` via ``numpy.linalg.eigh``.

    Returns ``(eigenvalues, eigenvectors)``, eigenvalues ascending (real),
    eigenvectors as columns, unit-normalised. A general (non-Hermitian) solver
    is never used: it would not guarantee real eigenvalues or orthonormal
    eigenvectors.
    """
    a = np.asarray(i_delta)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"iDelta must be square, got shape {a.shape}")
    return np.linalg.eigh(a.astype(np.complex128))


def sj_wightman(
    i_delta: np.ndarray,
    *,
    zero_tol: float | None = None,
    return_spectrum: bool = False,
):
    """SJ Wightman function ``W = sum_{lambda_k > tol} lambda_k v_k v_k^dagger``.

    Parameters
    ----------
    i_delta:
        Hermitian Pauli--Jordan matrix from ``pauli_jordan``.
    zero_tol:
        Absolute eigenvalue threshold; eigenvalues ``<= zero_tol`` are treated as
        zero/negative and excluded. Default ``zero_tolerance(evals, N)``.
    return_spectrum:
        If True also return ``(eigenvalues, eigenvectors, tol_used)`` so callers
        (spectrum gate) need not diagonalise twice.

    Returns
    -------
    W : complex128 array (N, N), Hermitian positive semidefinite.
    """
    evals, evecs = pauli_jordan_spectrum(i_delta)
    n = evals.shape[0]
    tol = zero_tolerance(evals, n) if zero_tol is None else float(zero_tol)
    pos = evals > tol
    v = evecs[:, pos]
    w = (v * evals[pos]) @ v.conj().T
    if return_spectrum:
        return w, (evals, evecs, tol)
    return w


# --------------------------------------------------------------------------- #
# Spectral truncations of W (Phase 3a Gate 2 follow-up; NOT used by Gates 1-2)
# --------------------------------------------------------------------------- #
# Both rules act on the positive spectrum of iDelta on the FULL causet and keep
# its top ``n_keep`` modes (single truncation: no subregion is involved, so the
# "double truncation" of SY / SXY for the SSEE does not arise here).

#: Sorkin--Yazdi 1611.10281 Sec. 3 (also KMY 2111.05879 eq. (mineig)):
#: keep eigenvalues ``lambda >= sqrt(N)/(4 pi)``, N = elements of the causet.
SY_TRUNCATION_PREFACTOR = 1.0 / (4.0 * np.pi)

#: "Linear truncation", Surya--X--Yazdi 2008.07697 Sec. "Generalised Truncation
#: Schemes" (cited for its definition by Mathur--Surya--X 2207.01080 Sec. 4):
#: the knee is where the binned, smoothed log-log slope of the spectrum drops
#: from the plateau slope ``m`` to ``m' = m (1 + delta)``. SXY's d = 2 choice is
#: delta ~ 0.1 ("most compatible with the area law", dS_2).
LIN_DELTA = 0.1
#: Plateau slope m: the continuum diamond spectrum is lambda ~ 1/n, i.e. a = 1
#: in SXY eq. (csmagtrunc) ("For D_L^2, a = 1 and b = 1/(4 pi)").
LIN_PLATEAU_SLOPE = -1.0
#: Log-spaced slope bins per decade of rank. SXY bin and average the pair
#: slopes but do not state the bin width; this value is OUR choice, fixed
#: before any data were generated (pre-registration, RESULTS.md 2026-11-02).
LIN_BINS_PER_DECADE = 10
#: First rank pair used. The continuum diamond's own 1->2 pair slope is -1.43
#: (below the threshold -1.1 by itself); every later continuum pair slope is
#: >= -1.081. Excluding pair 1->2 is fixed from the continuum alone.
LIN_FIRST_RANK = 2
#: Evaluation grid for the interpolant, points per decade of rank.
LIN_GRID_PER_DECADE = 1000


def sy_truncation_rank(lam_desc: np.ndarray, n_elements: int) -> int:
    """Number of positive eigenvalues ``>= sqrt(N)/(4 pi)`` (Sorkin--Yazdi cut)."""
    lam = np.asarray(lam_desc, float)
    return int(np.sum(lam >= SY_TRUNCATION_PREFACTOR * np.sqrt(n_elements)))


def linear_truncation_rank(
    lam_desc: np.ndarray,
    *,
    delta: float = LIN_DELTA,
    plateau: float = LIN_PLATEAU_SLOPE,
    bins_per_decade: int = LIN_BINS_PER_DECADE,
    first_rank: int = LIN_FIRST_RANK,
    return_diagnostics: bool = False,
):
    """Knee rank ``n_lin`` of the SXY linear truncation; keep ranks ``1..n_lin``.

    Implements SXY 2008.07697 Sec. "Generalised Truncation Schemes" step by step:
    (1) ``ln lambda_n`` against ``ln n`` for the positive spectrum, descending;
    (2) the slope between each nearest-neighbour pair, placed at
    ``n_mid = sqrt(n (n+1))``; (3) slopes binned (``bins_per_decade`` log bins)
    and averaged; (4) a smooth interpolating function (monotone cubic PCHIP in
    ``log10 n`` through the bin averages); (5) the knee is where that function
    drops below ``m' = plateau * (1 + delta)``.

    Which crossing: the LAST rank at which the smoothed slope is still
    ``>= m'`` (one-sided, so an isolated dip inside the plateau cannot end it;
    past the knee the slope only steepens). The FIRST downward crossing is
    returned as a diagnostic. ``n_lin`` is rounded to the nearest integer, as
    SXY do. If the slope never falls below ``m'`` nothing is truncated
    (``n_lin = M``).
    """
    from scipy.interpolate import PchipInterpolator

    lam = np.asarray(lam_desc, float)
    m_tot = lam.size
    n = np.arange(1, m_tot + 1, dtype=float)
    s = np.diff(np.log(lam)) / np.diff(np.log(n))
    nmid = np.sqrt(n[:-1] * n[1:])
    use = n[:-1] >= first_rank
    s, lx = s[use], np.log10(nmid[use])
    k = np.floor(lx * bins_per_decade).astype(int)
    keys = np.unique(k)
    cen = np.array([lx[k == b].mean() for b in keys])
    avg = np.array([s[k == b].mean() for b in keys])
    thr = plateau * (1.0 + delta)
    f = PchipInterpolator(cen, avg, extrapolate=False)
    npts = max(2, int(np.ceil((cen[-1] - cen[0]) * LIN_GRID_PER_DECADE)) + 1)
    g = np.linspace(cen[0], cen[-1], npts)
    above = f(g) >= thr
    if above.all():
        n_lin, n_first = m_tot, m_tot
    elif not above.any():
        n_lin, n_first = 1, 1
    else:
        n_lin = int(np.clip(round(10 ** g[np.nonzero(above)[0].max()]), 1, m_tot))
        n_first = int(np.clip(round(10 ** g[np.nonzero(~above)[0].min()]), 1, m_tot))
    if return_diagnostics:
        return n_lin, {"n_first_crossing": n_first, "bin_log10n": cen, "bin_slope": avg,
                       "threshold": thr}
    return n_lin


def wightman_block(evals_pos: np.ndarray, evecs_pos: np.ndarray, rows: np.ndarray,
                   n_keep: int | None = None) -> np.ndarray:
    """``W`` restricted to ``rows x rows`` from the top ``n_keep`` positive modes.

    ``evals_pos`` descending with matching eigenvector columns ``evecs_pos``.
    ``n_keep=None`` keeps all (the untruncated W of ``sj_wightman``).
    """
    k = evals_pos.size if n_keep is None else int(n_keep)
    v = evecs_pos[rows, :k]
    return (v * evals_pos[:k]) @ v.conj().T
