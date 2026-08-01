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

That ``rcond`` cut is relative to each block's own largest eigenvalue, which
is the only scale it has, and it is therefore NOT a reliable rank test when
the caller's blocks are sums of PSD terms on far-apart scales.  Ranks a
caller cares about should be counted on a better-conditioned member of the
same family and handed in — see ``block_ranks`` on :func:`factor_arrow`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

_RCOND = 1e-12


def psd_ranks(A: NDArray, rcond: float = _RCOND) -> NDArray:
    """Per-matrix ranks of a batch of symmetric PSD matrices, no inverse.

    Eigenvalues only, so this costs a fraction of :func:`_psd_pinv` and can be
    afforded once per pair on a matrix chosen for CONDITIONING rather than for
    the quantity being solved — which is the whole reason it is separate.  See
    :func:`factor_arrow`'s ``block_ranks``.
    """
    w = np.linalg.eigvalsh(A)
    scale = np.maximum(w[..., -1:], np.finfo(np.float64).tiny)
    return (w > rcond * scale).sum(axis=-1)


def _psd_pinv(A: NDArray, rcond: float) -> tuple[NDArray, NDArray]:
    """Batched PSD (pseudo-)inverse and per-matrix rank.

    ``A`` is ``(..., n, n)`` and symmetric.  Directions below the cut are
    dropped rather than inverted, which is the pseudo-inverse and is what the
    dense path's own ``pinv`` fallback does for the same matrices.

    ``rcond`` is RELATIVE to each matrix's own largest eigenvalue, and a sum
    of two PSD terms on wildly different scales defeats any such cut: at the
    ladder's high edge the block is ``V_q + lambda S_a`` with ``lambda`` at
    ``1e10`` times the pair's scale, so the directions the penalty does not
    reach sit around ``1e-10`` of the block's largest eigenvalue for a level
    carrying its share of the weight — but around ``1e-10`` TIMES that
    level's weight share for one that does not, and a rare level's share is
    bounded only by the data.  Measured on a 20-level pair, one level
    weighted 0.01 against the rest: the block-rank sum reads 227 at the high
    edge and 227 at the low one where the true value is 228.

    Nothing chosen for ``rcond`` fixes that, because on this matrix the two
    populations are not separated at all.  So the cut here is left to govern
    only what it can — which directions are safe to INVERT, where dropping a
    direction the penalty has already flattened moved the statistic by a
    relative 2e-5 on that same pair — and rank, where a whole degree of
    freedom rides on the answer, is counted elsewhere: see ``block_ranks`` on
    :func:`factor_arrow`.
    """
    w, Q = np.linalg.eigh(A)
    scale = np.maximum(w[..., -1:], np.finfo(np.float64).tiny)
    keep = w > rcond * scale
    inv = np.where(keep, 1.0 / np.where(keep, w, 1.0), 0.0)
    return (Q * inv[..., None, :]) @ np.swapaxes(Q, -1, -2), keep.sum(axis=-1)


@dataclass(frozen=True)
class ArrowFactor:
    """One factorization of an arrow matrix, reusable for every read below."""

    Ginv: NDArray  # (L, g, g) — block (pseudo-)inverses
    Y: NDArray  # (L, g, r) — G_q^{-1} E_q'
    Sinv: NDArray  # (r, r)   — inverse of the border Schur complement
    rank: int  # rank of the whole arrow matrix

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

        ``[K^-1]_qq = G_q^-1 + G_q^-1 E_q' Sigma^-1 E_q G_q^-1``.  Reading
        these back is what makes ``tr(K^-1 S)`` affordable for a block-
        diagonal ``S`` — no other entry of the inverse is ever needed, and
        forming the whole inverse would cost the quadratic memory this module
        exists to avoid.
        """
        return self.Ginv + self.Y @ self.Sinv @ np.swapaxes(self.Y, -1, -2)


def factor_arrow(
    G: NDArray,
    E: NDArray,
    border: NDArray,
    rcond: float = _RCOND,
    block_ranks: NDArray | None = None,
) -> ArrowFactor:
    """Factor the arrow matrix with blocks ``G`` (L, g, g), coupling ``E``
    (L, r, g) and border (r, r) — ``O`` in the layout at the top of the
    module.

    ``block_ranks`` overrides the per-block ranks counted here.  Supply it
    when ``G`` is a sum of PSD terms whose scales are far apart and a
    better-conditioned matrix with the SAME null space is available: rank is
    the same for both, but a relative eigenvalue cut is not, and the
    reasoning is in :func:`_psd_pinv`.  The border's own rank is still
    counted from the factorization, which is where it becomes available.
    """
    Ginv, counted = _psd_pinv(G, rcond)
    if block_ranks is None:
        block_ranks = counted
    Y = Ginv @ np.swapaxes(E, -1, -2)
    Sigma = border - np.einsum("lrg,lgs->rs", E, Y, optimize=True)
    Sinv, border_rank = _psd_pinv(0.5 * (Sigma + Sigma.T), rcond)
    return ArrowFactor(
        Ginv=Ginv,
        Y=Y,
        Sinv=Sinv,
        rank=int(np.sum(block_ranks)) + int(border_rank),
    )
