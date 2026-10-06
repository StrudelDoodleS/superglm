"""Column-local centring: repair a rejected raw-moment build at its failing columns only."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ._group_matrix_centered import (
    RawMomentRejection,
    _compact_support,
    _compact_support_rows,
    _raw_centering_admitted,
)
from ._group_matrix_kernels import _fused_bincount_2

# Support rows anchored at a time: bounds the (rows, raw width) transient.
_SUPPORT_CHUNK_BYTES = 8 << 20
# Ceiling on the centred support columns the repair holds at once, over every
# failing column: 128 columns of a 256 x 256 tensor grid.
_MAX_CENTRED_COLUMN_BYTES = 64 << 20


def _anchor_centred_columns(
    gm, local: NDArray, W: NDArray, weighted_z: NDArray, sum_w: float
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    """``(centred, codes, mean, mass, response)`` for ``gm``'s columns ``local``.

    ``_anchor_center_support``'s arithmetic on those columns alone: the
    support rows less the heaviest one, less their weighted mean, then
    projected, accumulated over chunks of support rows so a tensor grid's
    raw rows are never copied at once.  A categorical's identity support is
    formed on the failing levels only, so the heavy level is the anchor and
    its column becomes the centred complement indicator, never a ``(K+1, K)``
    array.  ``centred`` is ``(support rows, len(local))``.
    """
    from superglm.group_matrix import CategoricalGroupMatrix

    rows = _compact_support_rows(gm)
    if isinstance(gm, CategoricalGroupMatrix):
        values = np.zeros((rows, len(local)), dtype=np.float64)
        values[local, np.arange(len(local))] = 1.0
        codes, transform = gm.codes, None
    else:
        values, codes, transform = _compact_support(gm)
        if transform is None:
            values = values[:, local]
        else:
            transform = transform[:, local]
    mass, response = _fused_bincount_2(codes, W, weighted_z, rows)
    anchor = values[int(np.argmax(mass))]
    step = max(1, _SUPPORT_CHUNK_BYTES // (8 * max(values.shape[1], 1)))
    shift = np.zeros(values.shape[1], dtype=np.float64)
    for start in range(0, rows, step):
        shift += mass[start : start + step] @ (values[start : start + step] - anchor)
    shift /= sum_w
    centred = np.empty((rows, len(local)), dtype=np.float64)
    for start in range(0, rows, step):
        block = (values[start : start + step] - anchor) - shift
        centred[start : start + step] = block if transform is None else block @ transform
    mean = anchor + shift
    if transform is not None:
        mean = mean @ transform
    return centred, codes, mean, mass, response


def column_local_centering(
    *,
    dm,
    W: NDArray,
    rejected: RawMomentRejection,
    sum_w: float,
) -> tuple[NDArray, NDArray, NDArray, int] | None:
    """``(mean_x, data_gram, rhs, columns)`` from rejected raw moments, recentring only the failing columns.

    The raw-moment certificate (``_raw_centering_admitted``) is per column:
    subtracting raw moments keeps column ``j`` in the rounding envelope of a
    Gram when ``kappa_j^2 = 1 + m_j^2 / s_j^2 <= 2``, ``m_j`` its weighted
    mean and ``s_j`` its centred RMS (Chan, Golub & LeVeque 1983, eqs.
    3.2-3.3).  A 0/1 indicator whose level carries a weight share ``p > 1/2``
    fails it, ``kappa^2 = 1 / (1 - p)``, and one such column used to send
    every column of the design to the chunked ``O(n p^2)`` pass.  Here each
    failing column ``j`` alone is recentred inside its own group: its
    compact support is anchored at the group's heaviest support row and
    centred about its weighted mean before any product
    (``_anchor_centred_columns``; an indicator becomes its centred
    complement), giving centred rows ``c_j``, and its row of the Gram is

    - against an admitted column ``k``, any group's: that group's transpose
      product of ``W c_j`` less ``k``'s raw mean times ``e_j = sum W c_j``;
    - against a failing column of its own group: ``sum_b mass_b c_j c_k``
      over support rows; of another group: ``W c_j`` summed by that group's
      codes against its centred support.

    Every entry between two admitted columns, their means and their
    right-hand sides are bitwise the raw rung's subtraction, which is what
    the certificate would have returned for them on its own.  The cost is
    one gather and one transpose product of the design per failing column,
    ``O(n G)`` for ``G`` groups, against the chunked pass's ``O(n p^2)``.
    ``None`` when a failing column's group has no compact support, when the
    failing columns' centred supports exceed ``_MAX_CENTRED_COLUMN_BYTES``,
    or when an entry between admitted columns is not finite (a rejection
    that is not column-local).

    **Error.**  Each recentred entry is a sum over rows of ``W_r c_j(r)
    y_k(r)``: ``y_k`` is ``c_k`` (a failing partner), ``x_k - m^_k`` (an
    admitted partner, ``m^_k`` its raw mean, applied to ``e_j``) or ``z``
    (the right-hand side).  It is evaluated with at most ``K = n + n_g +
    n_h + q_g + q_h + 5`` roundings a term (``n_g``, ``n_h`` the two
    groups' support rows and ``q_g``, ``q_h`` their raw widths: the row
    sums, the support sums, the two projections, the anchoring, the
    weighting and the correction), so it lies within ``gamma_K sum_r W_r
    A_j(r) (A_k(r) + |m^_k|)`` of the exact sum over the computed centre
    (Higham 2002, sec. 3.1), ``A`` the absolute product of the factors each
    kernel multiplies: ``(|v - v_h| + |shift|) |T_j|`` for a recentred
    column, ``|v| |T_k|`` for an admitted one, and ``|c|`` itself for a
    categorical level, whose anchoring is exact.  By Cauchy-Schwarz
    ``sum W A_j A_k <= S a_j a_k`` (``S = sum W``, ``a`` the weighted RMS of
    ``A``), and an admitted column has ``a_k <= kappa_k s_k <= sqrt(2) s_k``
    and ``|m^_k| <= s_k`` where its projection does not cancel, so a
    recentred entry lies within ``(1 + sqrt(2)) gamma_K S a_j a_k``; raw
    subtraction between two admitted columns lies within ``gamma_K S s_k
    s_l (kappa_k kappa_l + kappa_k + kappa_l + 1) <= (3 + 2 sqrt(2))
    gamma_K S s_k s_l``.  Every recentred entry is inside the envelope a
    certified column already has.  The computed centre ``mu~_j = (v_h +
    shift) T_j`` differs from the exact weighted mean by a constant
    ``d_j``, which enters an entry only as ``S d_j d_k`` or ``S d_j (m_k -
    m^_k)``: second order in ``u``.
    """
    weighted_z = rejected.weighted_z
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        sum_weighted_z = rejected.sum_weighted_z
        if sum_weighted_z is None:
            sum_weighted_z = float(np.sum(weighted_z, dtype=np.float64))
        # The raw rung's subtraction, operation for operation.
        mean_x = rejected.xtw / sum_w
        gram = rejected.raw_gram - np.outer(rejected.xtw, mean_x)
        gram = 0.5 * (gram + gram.T)
        scale = np.sqrt(np.diag(gram) / sum_w)
        rhs = rejected.raw_rhs - mean_x * sum_weighted_z
    admitted = _raw_centering_admitted(mean_x, scale)
    kept = np.flatnonzero(admitted)
    if (
        admitted.all()
        or not np.isfinite(sum_weighted_z)
        or not np.all(np.isfinite(gram[np.ix_(kept, kept)]))
        or not np.all(np.isfinite(rhs[kept]))
    ):
        return None
    widths = [gm.shape[1] for gm in dm.group_matrices]
    starts = np.concatenate([np.zeros(1, dtype=np.intp), np.cumsum(widths, dtype=np.intp)])
    failing = np.flatnonzero(~admitted)
    owners = np.searchsorted(starts, failing, side="right") - 1
    by_group = {int(g): failing[owners == g] for g in np.unique(owners)}
    held = 0
    for g, columns in by_group.items():
        rows = _compact_support_rows(dm.group_matrices[g])
        if rows is None:
            return None
        held += 8 * rows * len(columns)
    if held > _MAX_CENTRED_COLUMN_BYTES:
        return None

    centred = {
        g: (
            columns,
            *_anchor_centred_columns(
                dm.group_matrices[g], columns - starts[g], W, weighted_z, sum_w
            ),
        )
        for g, columns in by_group.items()
    }
    for g, (columns, values, codes, mean, mass, response) in centred.items():
        first = mass @ values
        for i, column in enumerate(columns):
            weighted = W * values[:, i][codes]
            cross = dm.rmatvec(weighted)[kept] - mean_x[kept] * first[i]
            gram[column, kept] = cross
            gram[kept, column] = cross
            for h, (partner, partner_values, partner_codes, *_) in centred.items():
                if h <= g:
                    continue
                summed = np.bincount(
                    partner_codes, weights=weighted, minlength=partner_values.shape[0]
                )
                block = summed @ partner_values
                gram[column, partner] = block
                gram[partner, column] = block
        own = values.T @ (mass[:, None] * values)
        gram[np.ix_(columns, columns)] = 0.5 * (own + own.T)
        rhs[columns] = values.T @ response
        mean_x[columns] = mean
    if not (np.all(np.isfinite(gram)) and np.all(np.isfinite(rhs)) and np.all(np.isfinite(mean_x))):
        return None
    return mean_x, gram, rhs, len(failing)
