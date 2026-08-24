"""One triangular factor of a candidate pair's weighted joint design.

The dense screening path used to hand :mod:`superglm.screening._score_stat`
five assembled moment matrices -- ``U``, ``V``, the overlap ``C``/``M`` and a
nuisance score.  A Gram's spectrum is its design's SQUARED, so on a pair with
a starved level the deciding direction falls under the noise floor of the
operator it is read from and no arrangement of the arithmetic downstream can
put it back (issue #257; the measurement is in :mod:`._score_stat`).  This
module builds a factor instead: ONE upper triangular ``R`` such that

    [ X_o | X_T | yhat ] = Q R ,        Q' Q = I

with ``X_o`` the overlap span the pair's own mains already fit, ``X_T`` the
candidate tensor block, and ``yhat`` the working response that reproduces the
pair's score.  Frisch-Waugh-Lovell then reads the two quantities the ladder
needs off ``R``'s trailing blocks, with no difference of two Grams anywhere:

    V_eff = R_eff' R_eff ,     U_eff = R_eff' z_t

where ``R_eff`` is ``R``'s tensor-by-tensor block and ``z_t`` the response
column beside it.  ``R_eff``'s spectrum is ``V_eff``'s square root, which is
the whole content of #257's ``sqrt(cond) = 1.67e+10 against 2.78e+20``.

**THE DESIGN HERE IS THE CELL DESIGN, NOT THE ROW DESIGN, AND THAT IS EXACT
RATHER THAN AN APPROXIMATION.**  Both margins are constant within a joint
cell, so a cell's rows contribute ``W_cell`` to every curvature entry and
``S_cell`` to every score entry -- the sufficient-statistic identity
:mod:`superglm.screening._pair_moments` states and pins.  Weighting cell
``(i, j)``'s single design row by ``sqrt(W_cell[i, j])`` and giving it the
working response ``S_cell[i, j] / sqrt(W_cell[i, j])`` reproduces the row
design's Gram and its score exactly:

    sum_ij  sqrt(w) x_ij  * (s_ij / sqrt(w))  =  sum_ij s_ij x_ij .

``W_cell == 0`` forces ``S_cell == 0`` and the response is 0 there.  That is
not a convention: ``screening_ops`` builds working weights as ``weights *
dmu_deta**2 / var_mu`` from weights validated finite and non-negative, and
``working_score`` carries the same ``weights * dmu_deta`` factor, so a cell
whose weight sums to zero had every row's weight zero and every row's score
exactly zero.  Both cell tables are sums of non-negative-weighted terms, so
nothing cancels into a false zero.

**WHY THE REDUCTION IS CHUNKED AND SEQUENTIAL RATHER THAN A BINARY TREE.**
:func:`superglm.screening._structured._reduce_row_factors` exists to keep a
LEVEL COUNT out of a matrix dimension, because on the structured path ``L``
reaches fifty thousand and stacking the whole batch makes the temporary
``L``-sized.  Its objection does not transfer here and the shape of the
operands is why.  That reducer pads every block up to ``(w, w)``; the blocks
here are ``(k_inner + 2, w)`` with ``w`` of order ``k_a * k_b``, so padding
would allocate ``n_outer * w**2`` doubles where the whole point is to stay at
the ``(k, k)`` the dense path already allocates.  What answers the objection
is that the outer count is a SUPPORT size bounded by a chunk width chosen
here, not a level count arriving from the data: the chunk is sized so each
merge sees about ``w`` new rows, which caps the temporary at the shape of the
accumulator itself.

**COST CLASS, STATED AND NOT MEASURED HERE.**  The dominant term is
``O(n_outer * k_inner * w**2)`` against the moment route's ``O(n_outer *
k**2)``, so the reduction costs a factor of the INNER margin's width.  The
builder therefore reduces the NARROWER margin inside and loops over the other,
which makes that factor ``min(k_a, k_b)``.  No timing is taken on this branch
and none is claimed; the budget gates in :mod:`superglm.model.screening_ops`
are DIMENSIONAL (``k**3 <= _CUBIC_BUDGET_FACTOR * max_cells``), so which pairs
are admitted is unchanged, and re-fitting their calibration is a separate
decision taken with a machine to itself.

Column order out is ``[ 1 | inner menu | outer menu | tensor | yhat ]``.  The
overlap's internal order is the FACTOR's and differs from
:func:`superglm.screening._overlap.pair_overlap_moments`' ``[1 | A | B]``:
the overlap is only ever consumed as a SPAN, so nothing downstream can see it,
but a reader who slices ``overlap_width`` assuming the old layout will get the
wrong columns.  The TENSOR order is C-order ``p * k_b + q`` -- unchanged, and
load-bearing, because ``tensor_penalty(S_a, S_b)`` is aligned to it with no
permutation in between.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from superglm.screening._factor_kernels import _combine_row_factors

# Peak doubles allowed in one reduction temporary, matching
# ``_structured._TRACE_CHUNK_DOUBLES``.  It bounds the inner batched QR's
# operand only; the merge chunk is sized by ``w`` instead, because a merge that
# sees fewer than about ``w`` new rows pays a full ``O(w**3)`` factorization for
# them and the reduction stops being linear in the support.
_FACTOR_CHUNK_DOUBLES = 262_144


@dataclass(frozen=True)
class PairFactor:
    """A candidate pair's weighted joint design, reduced to one factor.

    ``joint`` is the ``(w, w)`` upper triangular ``R`` of
    ``[X_o | X_T | yhat]`` with ``w = overlap_width + tensor_width + 1``.  The
    widths are carried rather than inferred: a caller that guessed them from
    the shape would slice a transposed pair silently, and ``edf0`` is the only
    place it would show.
    """

    joint: NDArray
    overlap_width: int
    tensor_width: int

    def __post_init__(self) -> None:
        width = int(self.overlap_width) + int(self.tensor_width) + 1
        if self.joint.shape != (width, width):
            raise ValueError(
                f"joint must be ({width}, {width}) for overlap {self.overlap_width} and "
                f"tensor {self.tensor_width}; got {self.joint.shape}"
            )


def _square(factor: NDArray, width: int) -> NDArray:
    """Pad a short ``R`` up to ``(width, width)``.

    ``numpy.linalg.qr(mode="r")`` returns ``(m, width)`` when the reduction saw
    fewer rows than columns, which is the ordinary case for a pair with more
    tensor columns than occupied cells.  The missing rows are exactly zero, so
    padding changes no Gram and lets the widths above be CHECKED rather than
    re-derived at every consumer.
    """
    if factor.shape[0] == width:
        return factor
    padded = np.zeros((width, width), dtype=np.float64)
    padded[: factor.shape[0]] = factor
    return padded


def _working_response(S_cell: NDArray, root_w: NDArray) -> NDArray:
    """``S / sqrt(W)``, exactly zero where the weight is."""
    return np.divide(
        S_cell,
        root_w,
        out=np.zeros(S_cell.shape, dtype=np.float64),
        where=root_w > 0.0,
    )


def _merge_chunk(n_outer: int, block_rows: int, width: int) -> int:
    """Outer points per merge, sized so each merge sees about ``width`` new rows.

    A merge is a ``(width + rows, width)`` QR and costs ``O(width**3)`` however
    few rows it is given, so a chunk much shorter than ``width`` makes the
    reduction cubic in the SUPPORT rather than linear in it.  A chunk much
    longer buys nothing and grows the temporary past the accumulator.
    """
    return max(1, min(int(n_outer), -(-int(width) // max(int(block_rows), 1))))


def _inner_batch(inner_rows: int, inner_cols: int) -> int:
    """Outer points per batched inner QR, bounded by :data:`_FACTOR_CHUNK_DOUBLES`."""
    per_point = max(1, int(inner_rows) * int(inner_cols))
    return max(1, _FACTOR_CHUNK_DOUBLES // per_point)


def pair_design_factor(
    B_a: NDArray,
    B_b: NDArray,
    S_cell: NDArray,
    W_cell: NDArray,
) -> PairFactor:
    """Reduce a gridded pair's weighted joint design to one triangular factor.

    Serves ``ti``, ``spline_cat`` and ``cat_cat`` -- every kind whose margins
    both grid, and whose moments :func:`~superglm.screening._pair_moments.
    pair_score_curvature` and :func:`~superglm.screening._overlap.
    pair_overlap_moments` used to assemble.  ``B_a`` is ``(n_a, k_a)``, ``B_b``
    is ``(n_b, k_b)``, and the cell tables are ``(n_a, n_b)``.

    **THE REDUCTION IS ONE ORTHOGONAL TRANSFORMATION PER OUTER SUPPORT POINT,
    AND THE ALGEBRA IS WHY IT IS AFFORDABLE.**  Write the augmented margins
    ``a_i = [1, B_a[i]]`` and ``t_j = [1, B_b[j]]``.  The joint design row of
    cell ``(i, j)`` is ``sqrt(W[i, j]) * kron(a_i, t_j)``, because the
    Kronecker product of the two AUGMENTED margins IS the joint span: its
    first column is the intercept, its ``[1, k_b]`` block is ``B_b``, its
    ``p * (1 + k_b)`` columns are ``B_a``, and what is left is the tensor.
    For fixed ``i`` every such row is ``t_j`` mapped through the same
    ``kron(a_i', I)``, so with ``Z_i`` the ``(n_b, k_b + 2)`` matrix of rows
    ``[sqrt(W[i, j]) t_j , yhat_ij]`` and ``Z_i = Q_i R_i``,

        rows(i) = Q_i [ kron(a_i', R_i[:, :1+k_b]) | R_i[:, -1] ] ,

    so ``R_i`` -- ``k_b + 2`` rows, whatever the inner support -- stands in for
    the whole of level ``i`` exactly.  ``Q_i`` is never formed.

    Which margin is reduced inside is chosen by width, so the emitted blocks
    are as short as the pair allows; when that puts ``B_a`` inside, the tensor
    columns come out transposed and are permuted back before the factor is
    returned, because :func:`~superglm.screening._overlap.tensor_penalty` is
    aligned to ``p * k_b + q`` and nothing downstream permutes.
    """
    B_a = np.asarray(B_a, dtype=np.float64)
    B_b = np.asarray(B_b, dtype=np.float64)
    S_cell = np.asarray(S_cell, dtype=np.float64)
    W_cell = np.asarray(W_cell, dtype=np.float64)
    n_a, k_a = B_a.shape
    n_b, k_b = B_b.shape
    if S_cell.shape != (n_a, n_b) or W_cell.shape != (n_a, n_b):
        raise ValueError(
            f"cell tables must be ({n_a}, {n_b}) for menus of {k_a} and {k_b} columns; "
            f"got {S_cell.shape} and {W_cell.shape}"
        )
    overlap_width = 1 + k_a + k_b
    tensor_width = k_a * k_b
    width = overlap_width + tensor_width + 1

    # Reduce the NARROWER margin inside: the emitted block is
    # ``(k_inner + 2, width)`` and the reduction pays for every one of its
    # rows, so the inner width is the multiplier on the whole pass.  Ties go to
    # ``B_b``, which keeps the tensor order natural and skips the permutation.
    swapped = k_a < k_b
    outer, inner = (B_b, B_a) if swapped else (B_a, B_b)
    root_w = np.sqrt(W_cell if not swapped else W_cell.T)
    response = _working_response(S_cell if not swapped else S_cell.T, root_w)

    n_outer, k_outer = outer.shape
    n_inner, k_inner = inner.shape
    block_rows = min(n_inner, k_inner + 2)
    merge = _merge_chunk(n_outer, block_rows, width)
    batch = _inner_batch(n_inner, k_inner + 2)

    joint = np.zeros((0, width), dtype=np.float64)
    for start in range(0, n_outer, merge):
        stop = min(start + merge, n_outer)
        factors = np.empty((stop - start, block_rows, k_inner + 2), dtype=np.float64)
        for lo in range(start, stop, batch):
            hi = min(lo + batch, stop)
            local = np.empty((hi - lo, n_inner, k_inner + 2), dtype=np.float64)
            local[:, :, 0] = 1.0
            local[:, :, 1 : 1 + k_inner] = inner
            local[:, :, : 1 + k_inner] *= root_w[lo:hi, :, None]
            local[:, :, -1] = response[lo:hi]
            factors[lo - start : hi - start] = np.linalg.qr(local, mode="r")
        rows = stop - start
        block = np.empty((rows, block_rows, width), dtype=np.float64)
        # ``[1 | inner]`` rides through untouched -- it is ``a_i``'s leading 1.
        block[:, :, : 1 + k_inner] = factors[:, :, : 1 + k_inner]
        # the outer margin's own main effect: ``a_i[1:]`` against the intercept
        block[:, :, 1 + k_inner : overlap_width] = outer[start:stop, None, :] * factors[:, :, 0:1]
        # and the tensor, ``a_i[1:]`` against ``t_j[1:]``
        np.multiply(
            outer[start:stop, None, :, None],
            factors[:, :, None, 1 : 1 + k_inner],
            out=block[:, :, overlap_width : overlap_width + tensor_width].reshape(
                rows, block_rows, k_outer, k_inner
            ),
        )
        block[:, :, -1] = factors[:, :, -1]
        joint = _combine_row_factors(joint, block.reshape(rows * block_rows, width))

    joint = _square(joint, width)
    if swapped and tensor_width:
        # The loop emitted ``q * k_a + p``; the penalty is assembled on
        # ``p * k_b + q``.  One permutation of the tensor columns and one
        # re-triangularization -- an orthogonal transformation, so the Gram is
        # unchanged -- rather than a permutation every consumer would have to
        # know about.
        order = np.arange(width)
        order[overlap_width : overlap_width + tensor_width] = overlap_width + np.arange(
            tensor_width
        ).reshape(k_outer, k_inner).T.reshape(-1)
        joint = _square(np.linalg.qr(joint[:, order], mode="r"), width)
    return PairFactor(joint=joint, overlap_width=overlap_width, tensor_width=tensor_width)


def numeric_cat_factor(
    codes_g: NDArray,
    n_g: int,
    menu_g: NDArray,
    z: NDArray,
    score: NDArray,
    working_weights: NDArray,
) -> PairFactor:
    """Factor the ``menu_g[codes] * z`` probe with overlap ``[1 | z | menu]``.

    A numeric margin has no grid, so the pair's cells ARE the factor margin's
    levels and the numeric side enters each of them as a two-column geometry
    ``[1, z]``.  Level ``g``'s rows therefore reduce to a ``(3, 3)`` factor of
    ``[sqrt(w), sqrt(w) z, s / sqrt(w)]``, and the same Kronecker identity
    :func:`pair_design_factor` uses carries it to the joint block:
    ``kron([1, menu_g[g]]', R_g[:, :2])`` beside ``R_g[:, -1]``.

    **ONE GRAM SURVIVES HERE AND IT IS A STATED COMPROMISE.**  ``R_g`` is
    rooted from the level's ``3 x 3`` moment block, which the bincount pass
    already accumulates -- six channels of one O(n) sweep -- where taking it
    from the rows would need them grouped by level, a sort of ``n``.  What is
    squared is a two-dimensional geometry INSIDE a cell, whose conditioning is
    a property of the covariate's scale and centring within that level and not
    of the starved-level geometry #257 is about; for a treatment-coded menu the
    probe's own block is diagonal in any case.  The root is taken through
    ``eigh`` rather than a Cholesky, because a level carrying one row has a
    rank-1 block and a Cholesky is entitled to refuse it.  Graded against a
    row-space QR of the same levels in
    ``test_the_numeric_cat_cell_gram_is_not_what_limits_the_pair``, including
    on a level starved to a single row.
    """
    codes_g = np.asarray(codes_g, dtype=np.intp)
    menu_g = np.asarray(menu_g, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    w = np.asarray(working_weights, dtype=np.float64)
    if not (codes_g.shape == z.shape == score.shape == w.shape):
        raise ValueError("codes, z, score, and working weights must share one row dimension")
    if menu_g.ndim != 2 or menu_g.shape[0] != n_g:
        raise ValueError(
            f"menu_g must be (n_g, k) with one row per gridded cell; got shape "
            f"{menu_g.shape} for n_g={n_g}"
        )
    if codes_g.size and (int(codes_g.min()) < 0 or int(codes_g.max()) >= n_g):
        raise ValueError("codes_g fall outside [0, n_g)")

    def cell(v):
        return np.bincount(codes_g, weights=v, minlength=n_g)

    # The response channel ``s**2 / w`` is the working-response energy inside
    # the level.  It is bounded by the level's own residual energy (Cauchy-
    # Schwarz on ``S_cell <= sqrt(W_cell) * sqrt(sum s**2 / w)``), and a
    # zero-weight row contributes an exactly zero score, so the guarded divide
    # loses nothing.
    safe = w > 0.0
    q0 = cell(np.divide(score * score, w, out=np.zeros_like(score), where=safe))
    s0, s1 = cell(score), cell(score * z)
    w0, w1, w2 = cell(w), cell(w * z), cell(w * z * z)

    grams = np.empty((n_g, 3, 3), dtype=np.float64)
    grams[:, 0, 0] = w0
    grams[:, 0, 1] = grams[:, 1, 0] = w1
    grams[:, 1, 1] = w2
    grams[:, 0, 2] = grams[:, 2, 0] = s0
    grams[:, 1, 2] = grams[:, 2, 1] = s1
    grams[:, 2, 2] = q0
    values, vectors = np.linalg.eigh(grams)
    factors = np.swapaxes(vectors * np.sqrt(np.clip(values, 0.0, None))[:, None, :], -1, -2)

    k = menu_g.shape[1]
    overlap_width, tensor_width = 2 + k, k
    width = overlap_width + tensor_width + 1
    block = np.empty((n_g, 3, width), dtype=np.float64)
    block[:, :, :2] = factors[:, :, :2]
    block[:, :, 2:overlap_width] = menu_g[:, None, :] * factors[:, :, 0:1]
    block[:, :, overlap_width : overlap_width + tensor_width] = (
        menu_g[:, None, :] * factors[:, :, 1:2]
    )
    block[:, :, -1] = factors[:, :, -1]
    joint = _square(np.linalg.qr(block.reshape(n_g * 3, width), mode="r"), width)
    return PairFactor(joint=joint, overlap_width=overlap_width, tensor_width=tensor_width)


def numeric_numeric_factor(
    z1: NDArray,
    z2: NDArray,
    score: NDArray,
    working_weights: NDArray,
) -> PairFactor:
    """Factor the ``z1 * z2`` probe with overlap span ``[1 | z2 | z1]``.

    Two numerics contract to a five-column design whatever their supports, so
    this one reduces the RAW ROWS in chunks -- there is no cell table to stand
    in for them and none is wanted: five columns of ``n`` rows is the smallest
    exact object the pair has, and reducing it squares nothing at all.
    """
    z1 = np.asarray(z1, dtype=np.float64)
    z2 = np.asarray(z2, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    w = np.asarray(working_weights, dtype=np.float64)
    if not (z1.shape == z2.shape == score.shape == w.shape):
        raise ValueError("z1, z2, score, and working weights must share one row dimension")
    width = 5
    root = np.sqrt(w)
    chunk = max(width, _FACTOR_CHUNK_DOUBLES // width)
    joint = np.zeros((0, width), dtype=np.float64)
    for start in range(0, z1.size, chunk):
        stop = min(start + chunk, z1.size)
        r = root[start:stop]
        rows = np.empty((stop - start, width), dtype=np.float64)
        rows[:, 0] = r
        rows[:, 1] = r * z2[start:stop]
        rows[:, 2] = r * z1[start:stop]
        rows[:, 3] = rows[:, 2] * z2[start:stop]
        rows[:, 4] = np.divide(
            score[start:stop], r, out=np.zeros(stop - start, dtype=np.float64), where=r > 0.0
        )
        joint = _combine_row_factors(joint, rows)
    return PairFactor(joint=_square(joint, width), overlap_width=3, tensor_width=1)


def _pair_scale(factor: PairFactor) -> float:
    """``tr(V_eff)`` from the factor: ``||R_eff||_F**2``, never a difference."""
    q, k = factor.overlap_width, factor.tensor_width
    return float(np.sum(factor.joint[q : q + k, q : q + k] ** 2))


def _profiled_factor(factor: PairFactor) -> tuple[NDArray, NDArray]:
    """``(R_eff, z_t)``: the tensor block residualized on the overlap, and its score.

    Frisch-Waugh-Lovell on a triangular factor is a slice.  ``[X_o | X_T |
    yhat] = Q R`` puts ``X_T``'s component orthogonal to ``X_o`` in ``R``'s
    ``(tensor, tensor)`` block and the response's in the column beside it, so
    ``R_eff' R_eff`` is ``V - C' M^-1 C`` and ``R_eff' z_t`` is
    ``U - C' M^-1 u_m`` -- both without forming ``M``, inverting it, or
    subtracting anything.
    """
    q, k = factor.overlap_width, factor.tensor_width
    return factor.joint[q : q + k, q : q + k], factor.joint[q : q + k, -1]
