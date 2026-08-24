"""Factorization of a symmetric PSD block-arrow matrix.

    K = [[G_0                E_0'],
         [     G_1           E_1'],
         [          ...       ...],
         [E_0  E_1   ...      O  ]]

``L`` diagonal blocks ``G_q`` of size ``g``, each coupled only to a shared
border ``O`` of size ``r`` and never to each other.  Factoring, solving and
reading back the diagonal blocks of the inverse are all
``O(L * (g^3 + g^2 r + g r^2))`` — LINEAR in the block count, where
densifying ``K`` and factoring it would be ``O((L g + r)^3)`` and would
allocate ``(L g + r)^2`` doubles it never needed.

Every operation is batched over the block axis; nothing loops in Python over
``L``.  That is the whole point: the caller's ``L`` is a categorical level
count, which is the thing that gets large.

**Rank deficiency is routine here, not an edge case.** A level carrying one
observation, or none, makes its block singular, and at high cardinality some
level always does.  Blocks are therefore inverted through a batched
eigendecomposition with an ``rcond`` cut rather than a Cholesky: a stacked
Cholesky reports only THAT some block failed, never which, so recovering
would mean a Python loop over exactly the axis this module exists to keep
out of Python.  For ``K`` symmetric PSD the generalized Schur complement
``O - E G^+ E'`` is exact, because PSD forces ``range(E') <= range(G)``.

**This module counts nothing.**  It used to return a rank alongside the
inverse, because its one caller wrote ``edf`` as ``rank(A) - lambda
tr(A^-1 S)`` and needed the two halves to agree.  That difference is gone —
:mod:`superglm.screening._structured` now evaluates the same ``edf`` as a sum
of Tikhonov filter factors, every term in ``[0, 1]``, with no rank anywhere —
so the only cut left is :func:`_solve_floor`, which decides what an inverse
may RESOLVE.  A cut chosen to make a rank robust was always the wrong cut for
an inverse and vice versa; with the rank gone there is one question and one
answer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from superglm.screening._factor_kernels import _rank_floor


def _solve_floor(n: int) -> float:
    """Relative cut for what an inverse may RESOLVE.

    ``max(n, 1) * eps`` — round-off, LAPACK's convention and
    ``numpy.linalg.matrix_rank``'s own tolerance.  Dropping a direction here
    does not round the answer, it deletes the direction's whole contribution,
    so the cut is set at the point below which the arithmetic is meaningless
    rather than at the point below which an answer is "small".

    **THE ARITHMETIC IS NOT WRITTEN OUT HERE, AND THAT IS DELIBERATE.**  It
    was a third transcription of the expression
    :func:`superglm.screening._factor_kernels._rank_floor` derives and
    :func:`superglm.screening._factor_kernels._factor_rank_floor` takes the
    square root of, and "having them in one place is the point" is the reason
    that module exists.  It is a leaf -- it imports nothing from this package
    -- so reaching it from here adds no cycle.  The NAME stays, because what
    it means at this site is not what ``_rank_floor`` means at that one: there
    the question is whether a direction is round-off, here it is whether an
    inverse may resolve one, and :mod:`superglm.screening._score_stat`'s
    threshold taxonomy names this function as a Type 1 cut in its own right.
    Same number, two questions, one derivation.

    ``rcond`` is RELATIVE to each matrix's own largest eigenvalue, which is
    the only scale a batched routine has, and a sum of two PSD terms on wildly
    different scales defeats any such cut: at the ladder's high edge the block
    is ``V_q + lambda S_a`` with ``lambda`` at ``1e10`` times the pair's
    scale, so the directions the penalty does not reach sit around ``1e-10``
    of the block's largest eigenvalue for a level carrying its share of the
    weight — but around ``1e-10`` TIMES that level's weight share for one that
    does not, and a rare level's share is bounded only by the data.

    That used to matter because a direction the caller's RANK counted and this
    inverse dropped contributed ``1 - 0`` to ``edf``: a whole degree of freedom
    with no penalty offset.  Nothing subtracts a rank from this inverse any
    more, so a dropped direction now contributes its filter factor of zero and
    nothing else — the error is bounded by the direction's own share rather
    than by a whole count.
    """
    return _rank_floor(n)


def _psd_pinv(A: NDArray) -> NDArray:
    """Batched PSD (pseudo-)inverse.

    ``A`` is ``(..., n, n)`` and symmetric.  Directions below the cut are
    dropped rather than inverted, which is the pseudo-inverse.  This used to
    add "and is what the dense path's own ``pinv`` fallback does for the same
    matrices".  **THAT MECHANISM IS GONE AND THE SENTENCE IS PAST TENSE**:
    issue #257 removed the ``pinv`` fallback with the rest of the moment
    route, and the dense ladder now takes its one rank decision on a design
    FACTOR, through a pivoted QR at
    :func:`superglm.screening._factor_kernels._factor_rank_floor`'s
    ``sqrt(n eps)``.  That is the same cut as this one, since a factor's
    singular values are the Gram's eigenvalues' square roots -- so what
    survives the correction is the agreement, not the mechanism.  A
    present-tense claim by one module about another module's behaviour is
    exactly what :mod:`superglm.screening._factor_kernels`' own header warns
    is how a true sentence becomes a false one; this one outlived its subject
    by a branch because ``_arrow`` was outside that branch's changed set.
    The cut is
    :func:`_solve_floor`, and it is NOT selectable: this took an ``rcond``
    override while the caller counted a rank, so that the count and the
    inverse could be given different cuts.  Nothing counts any more -- there
    is one question about this matrix and one answer -- so a second cut is
    surface with no meaning behind it.
    """
    w, Q = np.linalg.eigh(A)
    floor = _solve_floor(A.shape[-1])
    scale = np.maximum(w[..., -1:], np.finfo(np.float64).tiny)
    keep = w > floor * scale
    inv = np.where(keep, 1.0 / np.where(keep, w, 1.0), 0.0)
    return (Q * inv[..., None, :]) @ np.swapaxes(Q, -1, -2)


@dataclass(frozen=True)
class ArrowFactor:
    """One factorization of an arrow matrix, reusable for every read below."""

    Ginv: NDArray  # (L, g, g) — block (pseudo-)inverses
    Y: NDArray  # (L, g, r) — G_q^{-1} E_q'
    Sinv: NDArray  # (r, r)   — inverse of the border Schur complement

    def solve(self, b_blocks: NDArray, b_border: NDArray) -> tuple[NDArray, NDArray]:
        """Solve ``K [x; z] = [b_blocks; b_border]``.

        Returns ``(x, z)`` with ``x`` shaped ``(L, g)`` and ``z`` shaped
        ``(r,)``.  ``Y' b`` is exactly ``E G^{-1} b`` because ``G^{-1}`` is
        symmetric, so the border reduction needs no separate copy of ``E``.
        """
        Gb = np.einsum("lgh,lh->lg", self.Ginv, b_blocks, optimize=True)
        z = self.Sinv @ (b_border - np.einsum("lgr,lg->r", self.Y, b_blocks, optimize=True))
        return Gb - self.Y @ z, z

    def diag_blocks(self) -> NDArray:
        """The ``(L, g, g)`` diagonal blocks of ``K^{-1}``.

        ``[K^-1]_qq = G_q^-1 + G_q^-1 E_q' Sigma^-1 E_q G_q^-1``, formed
        without forming the inverse — which would cost the quadratic memory
        this module exists to avoid.  The OFF-diagonal blocks are
        ``Y_q Sigma^-1 Y_q'``, which a caller that needs them can contract
        itself out of :attr:`Y` and :attr:`Sinv` without ever forming one.

        **NO PRODUCTION CALLER, DELIBERATELY KEPT.**  This used to be how
        ``tr(K^-1 X)`` was taken for a block-diagonal ``X``; the ``edf`` it
        served is a sum of squared norms now and reads nothing back from the
        inverse.  What is left is the one statement about this factorization
        that the rest of the class cannot make -- :meth:`solve` exercises the
        inverse only through a right-hand side -- so the module's own
        contract test compares these blocks against a dense ``K^-1``.  It is
        surface with a test behind it rather than surface with nothing.
        """
        return self.Ginv + self.Y @ self.Sinv @ np.swapaxes(self.Y, -1, -2)


def factor_arrow(
    G: NDArray,
    E: NDArray,
    border: NDArray,
) -> ArrowFactor:
    """Factor the arrow matrix with blocks ``G`` (L, g, g), coupling ``E``
    (L, r, g) and border (r, r) — ``O`` in the layout at the top of the
    module.
    """
    Ginv = _psd_pinv(G)
    Y = Ginv @ np.swapaxes(E, -1, -2)
    Sigma = border - np.einsum("lrg,lgs->rs", E, Y, optimize=True)
    return ArrowFactor(Ginv=Ginv, Y=Y, Sinv=_psd_pinv(0.5 * (Sigma + Sigma.T)))
