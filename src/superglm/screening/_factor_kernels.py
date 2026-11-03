"""Row-space factor kernels shared by the structured and dense screening paths.

Merging two weighted-row factors without squaring either, rooting an assembled
penalty on a bar the caller can see, and the round-off floor every rank
decision in the package is taken at. A leaf module, so that both
:mod:`superglm.screening._structured` and :mod:`superglm.screening._pair_factor`
can import it without a cycle.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from superglm.solvers.rank import _symmetric_part


def _rank_floor(n: int) -> float:
    """Share of the largest eigenvalue below which a direction is round-off.

    ``max(n, 1) * eps``, LAPACK's convention and the ``numpy.linalg.matrix_rank``
    default. It scales with the dimension because round-off accumulates with it;
    a fixed constant either deletes real curvature (too large) or keeps
    subtraction dust as a degree of freedom (too small).

    Rank cuts are taken on a factor at :func:`_factor_rank_floor`, this
    expression's square root, which is the same cut on the Gram the factor squares to.
    """
    return max(int(n), 1) * float(np.finfo(np.float64).eps)


def _factor_rank_floor(n: int) -> float:
    """:func:`_rank_floor`'s cut expressed for a factor rather than a Gram.

    Singular values are the Gram eigenvalues' square roots, so
    ``sigma > sqrt(n eps) * sigma_max`` and ``w > n eps * w_max`` are the same
    statement about the same direction. Cutting on the factor avoids squaring its
    conditioning.
    """
    return float(np.sqrt(_rank_floor(n)))


def _combine_row_factors(left: NDArray, right: NDArray) -> NDArray:
    """Compact two weighted-row factors without squaring either one.

    The stack is built Fortran-ordered and handed to LAPACK ``dgeqrf`` to
    overwrite, which saves one ``(m, n)`` copy; the buffer is local, so nothing a
    caller holds is destroyed. The leading rows of ``R`` are copied so the full
    buffer is released. Agrees with ``numpy.linalg.qr`` to round-off, not bitwise.
    """
    rows_left, width = left.shape
    stacked = np.empty((rows_left + right.shape[0], width), dtype=np.float64, order="F")
    stacked[:rows_left] = left
    stacked[rows_left:] = right
    (factor,) = scipy.linalg.qr(stacked, mode="r", overwrite_a=True)
    return factor[: min(factor.shape)].copy()


def _penalty_root(S_a: NDArray) -> tuple[NDArray, float, float]:
    """``rootS`` with ``rootS' rootS`` a PSD matrix within ``eigh``'s bar of ``S_a``.

    Returns the factor, the spectral distance the drop branch moved ``S_a`` and its cut, so the
    caller decides on the same bar. An eigenvalue inside ``n eps ||S||_2`` (*LAPACK Users'
    Guide*, 3rd ed., sec. 4.7) has an unresolved sign and is taken at its magnitude, so a
    published ``edf0`` does not depend on rounding (at most ``2 n eps ||S||_2``, not Higham's
    1988 nearest PSD matrix). One below ``-n eps ||S||_2`` is a real negative direction: it is
    dropped and its largest magnitude returned so ``_profile`` refuses on the same cut. The
    distance is spectral (Goulart, Nakatsukasa and Rontsis, arXiv:1908.01606, sec. 2.2).
    Dropping frees a direction (filter factor 1), so ``edf`` can only go up.
    """
    lifted, Q, dropped, unresolved = _penalty_spectrum(S_a)
    keep = lifted > 0.0
    return (Q[:, keep] * np.sqrt(lifted[keep])).T, dropped, unresolved


def _penalty_spectrum(S_a: NDArray) -> tuple[NDArray, NDArray, float, float]:
    """:func:`_penalty_root`'s policy before it is collapsed into a factor.

    Returns ``(lifted, Q, dropped, unresolved)``: the admitted non-negative
    eigenvalues, their eigenvectors, the spectral-norm distance the drop branch
    moved ``S_a``, and the cut it used. Split out so a Kronecker sum can be rooted
    from its margins' spectra: ``eigh`` is accurate to ``n eps ||S||_2``, so rooting
    the assembled sum resolves a null direction only to the assembled dimension's
    round-off. See :func:`superglm.screening._overlap.tensor_penalty_root`.
    """
    n = S_a.shape[0]
    if n == 0:
        return (
            np.zeros(0, dtype=np.float64),
            np.zeros((0, 0), dtype=np.float64),
            0.0,
            0.0,
        )
    symmetric = _symmetric_part(S_a)
    diagonal = np.diag(symmetric)
    if np.count_nonzero(symmetric) == np.count_nonzero(diagonal):
        # The diagonal spectrum is known exactly. Eigh may internally scale
        # the whole matrix and erase a subnormal entry beside a large one.
        # Preserve the same sorted spectral convention without that scaling.
        order = np.argsort(diagonal)
        w = diagonal[order]
        Q = np.eye(n)[:, order]
    else:
        w, Q = np.linalg.eigh(symmetric)
    unresolved = float(float(n) * np.finfo(np.float64).eps * float(np.max(np.abs(w), initial=0.0)))
    lifted = np.where(w >= -unresolved, np.abs(w), 0.0)
    # Distance to the cone from certified negatives only (w < -unresolved); zeros
    # and unresolved eigenvalues are not counted. ``unresolved`` is returned so the
    # caller refuses on the same cut.
    dropped = float(np.max(-w, initial=0.0, where=w < -unresolved))
    return lifted, Q, dropped, unresolved
