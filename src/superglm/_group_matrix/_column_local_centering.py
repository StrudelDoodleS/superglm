"""Column-local centring: repair a rejected raw-moment build at its failing columns only."""

from __future__ import annotations

from collections.abc import Iterator
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from ._group_matrix_centered import (
    RawMomentRejection,
    _compact_support,
    _compact_support_rows,
    _raw_centering_admitted,
    _reads_as_float64,
)
from ._group_matrix_kernels import _fused_bincount_2, _native, _ssp_gram_needs_exact

# Support rows anchored at a time: bounds the (rows, raw width) transient.
_SUPPORT_CHUNK_BYTES = 8 << 20
# Ceiling on the centred support columns the repair holds at once over every
# failing column, and on the weighted rows of the failing columns it forms at
# once (at least one).  A categorical or SCOP support's value copy and the own
# block's mass-weighted temporary take the transient to about four times it,
# beside the kept projections (``_MAX_KEPT_PROJECTION_BYTES``) and one more
# projected support (``_admitted_products``), no larger than that group's
# ``B_unique``.
_MAX_CENTRED_COLUMN_BYTES = 64 << 20
# Ceiling on the projections the repair forms itself and keeps for later
# chunks of failing columns; past it a projection is formed again per chunk.
_MAX_KEPT_PROJECTION_BYTES = _MAX_CENTRED_COLUMN_BYTES


class RecentredColumns(NamedTuple):
    """One group's recentred columns: their design indices, centred support and row codes.

    Row ``r`` of column ``columns[i]`` is ``values[codes[r], i]``, centred
    about the column's ``mean_x`` entry.  Owner: the build that recentred
    them; it lives as long as that build's system is assembled.
    """

    columns: NDArray
    values: NDArray
    codes: NDArray


def recentred_products(
    recentred: tuple[RecentredColumns, ...], weighted: NDArray
) -> Iterator[tuple[NDArray, NDArray]]:
    """``(columns, sum_r weighted_r c_j(r))`` for each group of recentred columns.

    ``weighted`` summed by the group's support codes, then against its centred
    support: the repair's own product with a failing partner of another
    group, so a recentred column's raw values, whose ``kappa`` the repair
    removed, never meet ``weighted``.  ``n + n_s`` roundings a term beyond
    those that formed ``c_j`` and ``weighted``, ``n_s`` the support rows.
    """
    for group in recentred:
        summed = np.bincount(group.codes, weights=weighted, minlength=group.values.shape[0])
        yield group.columns, summed @ group.values


def _project(gm) -> NDArray:
    """``B_unique @ R_inv``: the one projection the repair forms itself."""
    return gm.B_unique @ gm.R_inv


def _admitted_readers(group_matrices, supports) -> list | None:
    """How the repair reads each group's admitted columns, so it multiplies their values alone.

    A categorical, SCOP, sparse or dense group: ``None`` (its ``rmatvec`` multiplies
    its values alone).  A discretized SSP group or spline-by-category level:
    ``(support, codes, rows)``, its support projected before any product, because
    ``rmatvec`` would sum ``B_unique`` first and multiply the sum's rounding by a
    projection that cancels.  ``support`` is the projection the rejected build
    formed (``supports``), else ``None`` for ``_admitted_products`` to form from
    float64 factors in the range in which the raw Gram projects
    (``_ssp_gram_needs_exact``).  ``None`` overall for any other group or factors.
    """
    from superglm.group_matrix import (
        CategoricalGroupMatrix,
        DenseGroupMatrix,
        DiscretizedSCOPGroupMatrix,
        DiscretizedSplineCategoricalGroupMatrix,
        DiscretizedSSPGroupMatrix,
        SparseGroupMatrix,
    )

    values_alone = (
        CategoricalGroupMatrix,
        DenseGroupMatrix,
        DiscretizedSCOPGroupMatrix,
        SparseGroupMatrix,
    )
    projected = (DiscretizedSSPGroupMatrix, DiscretizedSplineCategoricalGroupMatrix)
    readers = []
    for gm in group_matrices:
        if isinstance(gm, values_alone):
            readers.append(None)
            continue
        if not isinstance(gm, projected):
            return None
        support = None if supports is None else supports.get(gm)
        if support is None and (
            gm.B_unique.dtype != np.float64
            or gm.R_inv.dtype != np.float64
            or _ssp_gram_needs_exact(gm.B_unique, gm.R_inv)
        ):
            return None
        level = isinstance(gm, DiscretizedSplineCategoricalGroupMatrix)
        codes, rows = (gm.bin_idx_level, gm.row_idx) if level else (gm.bin_idx, None)
        readers.append((support, codes, rows))
    return readers


def _admitted_products(group_matrices, readers, weighted: NDArray, formed: dict) -> NDArray:
    """``weighted @ X``, a row a failing column, each group read as ``_admitted_readers`` says.

    Groups outside, failing columns inside.  A projection this repair forms
    is kept in ``formed`` for the later chunks of failing columns while the
    kept ones total at most ``_MAX_KEPT_PROJECTION_BYTES``; past that it is
    released at the next group's and formed again by the next chunk.
    """
    parts = []
    for gm, reader in zip(group_matrices, readers, strict=True):
        if reader is None:
            parts.append(np.stack([gm.rmatvec(row) for row in weighted]))
            continue
        support, codes, rows = reader
        if support is None:
            support = formed.get(gm)
        if support is None:
            support = _project(gm)
            if (
                sum(v.nbytes for v in formed.values()) + support.nbytes
                <= _MAX_KEPT_PROJECTION_BYTES
            ):
                formed[gm] = support
        summed = np.stack(
            [
                np.bincount(
                    codes, weights=row if rows is None else row[rows], minlength=len(support)
                )
                for row in weighted
            ]
        )
        parts.append(summed @ support)
    return np.concatenate(parts, axis=1)


def _anchor_centred_columns(
    gm, local: NDArray, W: NDArray, weighted_z: NDArray, sum_w: float
) -> tuple[NDArray, NDArray, tuple[NDArray, NDArray], NDArray, NDArray] | None:
    """``(centred, codes, (anchor, shift), mass, response)`` for ``gm``'s columns ``local``.

    ``_anchor_center_support``'s anchoring on those columns alone: the support rows
    less the heaviest one, less their weighted mean, then projected, accumulated
    over chunks of support rows so a tensor grid's raw rows are never copied at
    once.  A categorical's identity support is formed on the failing levels only
    (the heavy level is the anchor), never as a ``(K+1, K)`` array.  ``centred`` is
    ``(support rows, len(local))``.  Support and transform are read as float64 chunk
    by chunk before any arithmetic (an integer difference would wrap); ``None`` for
    one ``_reads_as_float64`` declines.  The centre is the projected anchor and
    shift, apart (``column_local_centering``, **Centre**): their sum is ``mu~``.
    """
    from superglm.group_matrix import CategoricalGroupMatrix

    rows = _compact_support_rows(gm)
    if isinstance(gm, CategoricalGroupMatrix):
        values = np.zeros((rows, len(local)), dtype=np.float64)
        values[local, np.arange(len(local))] = 1.0
        codes, transform = gm.codes, None
    else:
        values, codes, transform = _compact_support(gm)
        operands = (values,) if transform is None else (values, transform)
        if any(not _reads_as_float64(array.dtype) for array in operands):
            return None
        if transform is None:
            values = values[:, local]
        else:
            transform = np.asarray(transform[:, local], dtype=np.float64)
    mass, response = _fused_bincount_2(codes, _native(W), _native(weighted_z), rows)
    anchor = np.asarray(values[int(np.argmax(mass))], dtype=np.float64)
    step = max(1, _SUPPORT_CHUNK_BYTES // (8 * max(values.shape[1], 1)))
    shift = np.zeros(values.shape[1], dtype=np.float64)
    for start in range(0, rows, step):
        block = np.asarray(values[start : start + step], dtype=np.float64)
        shift += mass[start : start + step] @ (block - anchor)
    shift /= sum_w
    centred = np.empty((rows, len(local)), dtype=np.float64)
    for start in range(0, rows, step):
        block = (np.asarray(values[start : start + step], dtype=np.float64) - anchor) - shift
        centred[start : start + step] = block if transform is None else block @ transform
    parts = (anchor, shift) if transform is None else (anchor @ transform, shift @ transform)
    return centred, codes, parts, mass, response


def column_local_centering(
    *,
    dm,
    W: NDArray,
    rejected: RawMomentRejection,
    sum_w: float,
) -> tuple[NDArray, NDArray, NDArray, tuple[RecentredColumns, ...]] | None:
    """``(mean_x, data_gram, rhs, recentred)`` from rejected raw moments, recentring only the failing columns.

    The raw-moment certificate (``_raw_centering_admitted``) is per column:
    subtracting raw moments keeps column ``j`` in the rounding envelope of a Gram
    when ``kappa_j^2 = 1 + m_j^2 / s_j^2 <= 2``, ``m_j`` its weighted mean and
    ``s_j`` its centred RMS (Chan, Golub & LeVeque 1983, eqs. 3.2-3.3).  A 0/1
    indicator whose level carries a weight share ``p > 1/2`` fails it, ``kappa^2 =
    1 / (1 - p)``.  Each failing column alone is recentred inside its own group: its
    compact support is anchored at the group's heaviest support row and centred
    about its weighted mean before any product (``_anchor_centred_columns``),
    giving centred rows ``c_j``, and its row of the Gram is

    - against an admitted column ``k``: ``W c_j`` against ``k``'s values ``x_k``
      themselves (an SSP support projected before any product,
      ``_admitted_readers``), less ``k``'s raw mean times ``e_j = sum W c_j``;
    - against a failing column of its own group: ``sum_b mass_b c_j c_k`` over
      support rows; of another group: ``W c_j`` summed by that group's codes
      against its centred support.

    Entries between two admitted columns, their means and right-hand sides are
    bitwise the raw rung's.  The cost is one gather and one transpose product of
    the design per failing column, ``O(n G)`` for ``G`` groups, against the chunked
    pass's ``O(n p^2)``.  ``recentred`` holds each group's recentred columns and
    centred support (``RecentredColumns``); callers meet them with other rows
    through ``recentred_products``.  ``None`` when a failing column's group has no
    compact support or one ``_reads_as_float64`` declines, when the centred supports
    exceed ``_MAX_CENTRED_COLUMN_BYTES``, when a group's admitted columns cannot be
    read as their own values, or when an entry between admitted columns is not
    finite.

    **Error.**  Each recentred entry is a sum over rows of ``W_r c_j(r) y_k(r)``
    with ``y_k`` the partner's centred (or admitted-minus-raw-mean) values, evaluated
    with at most ``K = n + n_g + n_h + q_g + q_h + 5`` roundings a term, so it lies
    within ``gamma_K sum_r W_r A_j(r) (A_k(r) + |m^_k|)`` of the exact sum over the
    computed centre (Higham 2002, sec. 3.1), ``A`` the absolute product of the
    factors each kernel multiplies.  By Cauchy-Schwarz ``sum W A_j A_k <= S a_j a_k``
    (``S = sum W``, ``a`` the weighted RMS of ``A``), and an admitted column has
    ``a_k = kappa_k s_k <= sqrt(2) s_k`` and ``|m^_k| <= s_k``, so a recentred entry
    lies within ``(1 + sqrt(2)) gamma_K S a_j s_k``.  That holds where ``x_k`` is
    the projection the raw Gram reads (an exact ``DiscretizedSSPGroupMatrix`` or
    ``SupportCompressedSSPGroupMatrix``).  A tensor and a spline-by-category level
    apply ``R_inv`` after contracting ``B_unique``, so the projection's ``q_h``
    roundings add ``gamma_(q_h) sum_r W_r |c_j(r)| (|B_unique| |T_k|)(code_r)``.
    Raw subtraction between two admitted columns lies within ``(3 + 2 sqrt(2))
    gamma_K S s_k s_l``.

    **Centre.**  The returned centre ``mu~_j = fl(v_h T_j) + fl(shift T_j)`` is the
    counterpart of ``c_j``: the projected support ``fl(v_b T_j)`` lies within
    ``gamma_q |v_b| |T_j| + gamma_(q+1) (|v_h| + |shift|) |T_j| + gamma_(q+2)
    (|v_b - v_h| + |shift|) |T_j|`` of ``mu~_j + c_j(b)``.  Projecting ``fl(v_h +
    shift)`` instead would erase ``shift`` where ``|shift| < u |v_h|``.  ``mu~_j``
    differs from the exact weighted mean by a constant ``d_j``, which enters an
    entry only at second order in ``u``.
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
    repair = _recentre_columns(dm, admitted, W, weighted_z, sum_w, rejected.supports)
    if repair is None:
        return None
    centred, readers = repair
    recentred = tuple(
        RecentredColumns(columns, values, codes) for columns, values, codes, *_ in centred.values()
    )
    masses = [mass for *_, mass, _ in centred.values()]
    recentred_gram_rows(dm.group_matrices, readers, W, gram, mean_x, kept, recentred, masses)
    for columns, values, _, (anchor, shift), _, response in centred.values():
        rhs[columns] = values.T @ response
        mean_x[columns] = anchor + shift
    if not (np.all(np.isfinite(gram)) and np.all(np.isfinite(rhs)) and np.all(np.isfinite(mean_x))):
        return None
    return mean_x, gram, rhs, recentred


def _recentre_columns(
    dm, admitted: NDArray, W: NDArray, weighted_z: NDArray, sum_w: float, supports
):
    """``(centred, readers)`` for the columns ``admitted`` rejects, else ``None``.

    ``centred`` maps each group holding a failing column to ``(columns, centred,
    codes, mean, mass, response)``, anchored and centred inside the group about the
    weights ``W`` (``_anchor_centred_columns``); ``readers`` reads every group's
    admitted columns (``_admitted_readers``).  ``None`` where the repair declines.
    """
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
    readers = _admitted_readers(dm.group_matrices, supports)
    if readers is None:
        return None
    centred = {}
    for g, columns in by_group.items():
        anchored = _anchor_centred_columns(
            dm.group_matrices[g], columns - starts[g], W, weighted_z, sum_w
        )
        if anchored is None:
            return None
        centred[g] = (columns, *anchored)
    return centred, readers


def recentred_gram_rows(
    group_matrices,
    readers,
    weights: NDArray,
    gram: NDArray,
    mean_x: NDArray,
    kept: NDArray,
    recentred: tuple[RecentredColumns, ...],
    masses: list[NDArray] | None = None,
    formed: dict | None = None,
) -> None:
    """Write ``gram``'s rows and columns at the recentred columns, from their centred supports.

    A recentred column ``j`` (centred rows ``c_j``) against an admitted column
    ``k`` (``kept``, read as ``readers`` says, ``_admitted_products``) is
    ``sum w c_j x_k - mean_x[k] sum w c_j``; against a failing column of
    another group, ``w c_j`` summed by that group's codes against its centred
    support (``recentred_products``); within its group, ``sum_b mass_b c_j
    c_k`` over support rows, ``mass`` the group's ``weights`` summed by its
    codes (``masses``, formed here when ``None``).  The arithmetic holds for
    weights of any sign.  ``formed`` holds the projections the readers form,
    for a caller that reads the same factors again; without it they are
    released before the own blocks.
    """
    if masses is None:
        masses = [
            np.bincount(group.codes, weights=weights, minlength=group.values.shape[0])
            for group in recentred
        ]
    if formed is None:
        formed = {}
    first = [mass @ group.values for group, mass in zip(recentred, masses, strict=True)]
    pending = [(g, i) for g, group in enumerate(recentred) for i in range(len(group.columns))]
    step = max(1, _MAX_CENTRED_COLUMN_BYTES // (8 * len(weights)))
    for start in range(0, len(pending), step):
        chunk = pending[start : start + step]
        weighted = np.stack(
            [weights * recentred[g].values[:, i][recentred[g].codes] for g, i in chunk]
        )
        products = _admitted_products(group_matrices, readers, weighted, formed)
        for row, (g, i) in enumerate(chunk):
            column = recentred[g].columns[i]
            cross = products[row, kept] - mean_x[kept] * first[g][i]
            gram[column, kept] = cross
            gram[kept, column] = cross
            for partner, block in recentred_products(recentred[g + 1 :], weighted[row]):
                gram[column, partner] = block
                gram[partner, column] = block
    del formed, weighted, products  # the own blocks below run without them
    for group, mass in zip(recentred, masses, strict=True):
        own = group.values.T @ (mass[:, None] * group.values)
        gram[np.ix_(group.columns, group.columns)] = 0.5 * (own + own.T)


def recentred_about(
    dm, admitted: NDArray, W: NDArray, sum_w: float, centre: NDArray
) -> tuple[tuple[RecentredColumns, ...], list] | None:
    """``(recentred, readers)``: the columns ``admitted`` rejects, centred about a given ``centre``.

    The repair of ``column_local_centering`` for a caller that holds its centre
    already (the weighted mean at the weights ``W``, sum ``sum_w``) and forms
    products with other weights from it (``recentred_gram_rows``): each failing
    column is anchored and centred inside its group about ``W``, then moved to
    ``centre`` by ``c_j + ((a_j - centre_j) + s_j)``.  ``a_j - centre_j`` is exact
    within a factor two (Sterbenz), as for a rejected column whose spread is below
    its mean, so the move adds one rounding of ``c_j``'s own size; adding the
    rounded ``mu~_j`` instead would carry ``u |mu~_j|`` into every row.  ``None``
    where the repair declines.
    """
    repair = _recentre_columns(dm, admitted, W, np.zeros_like(W), sum_w, None)
    if repair is None:
        return None
    centred, readers = repair
    recentred = tuple(
        RecentredColumns(columns, values + ((anchor - centre[columns]) + shift), codes)
        for columns, values, codes, (anchor, shift), *_ in centred.values()
    )
    return recentred, readers
