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
(That slice is the whole story only while the overlap has full rank;
:func:`_profiled_factor` states what a rank-deficient one adds back, and
:func:`_pair_scale` takes its block rather than re-slicing so the two cannot
part company.)

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

**COST CLASS, AND THE MULTIPLIER MEASURED FOR IT.**  The dominant term is
``O(n_outer * k_inner * w**2)`` against the moment route's ``O(n_outer *
k**2)``, so the reduction costs a factor of the INNER margin's width.  The
builder therefore reduces the NARROWER margin inside and loops over the other,
which makes that factor ``min(k_a, k_b)``.  Against the moment route at
``16e7f810``, as 18-rep interleaved A/B/B/A medians taken in a dedicated
exclusive phase with all seven thread pools pinned to one, the ladder phase
costs 0.95x, 3.10x, 3.94x and 4.09x at tensor widths 35, 133, 1043 and 1349
(``ps(8)`` on 6, 20 and 150 levels, then ``ps(20)`` on 72), and 1.10x, 1.67x,
2.31x and 4.17x end to end through ``screen_interactions``.  The budget gates
in :mod:`superglm.model.screening_ops` are DIMENSIONAL (``k**3 <=
_CUBIC_BUDGET_FACTOR * max_cells``), so which pairs are admitted is unchanged;
what those multipliers move is the ~1.5 s per-pair target the constants were
fitted to, and re-fitting them is a separate decision taken with a machine to
itself.

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

from superglm.screening._factor_kernels import (
    _combine_row_factors,
    _factor_rank_floor,
)

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
        # and the tensor, ``a_i[1:]`` against ``t_j[1:]``, in the C-order the
        # penalty is assembled on.  Written through a temporary rather than
        # with ``out=`` on a reshaped slice: that reshape does return a view
        # here (the axis being split is contiguous), but a reader should not
        # have to know numpy's rule for when it does, and the temporary is
        # smaller than ``block`` itself.
        block[:, :, overlap_width : overlap_width + tensor_width] = (
            outer[start:stop, None, :, None] * factors[:, :, None, 1 : 1 + k_inner]
        ).reshape(rows, block_rows, tensor_width)
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

    **THE PER-LEVEL FACTOR IS BUILT FROM CENTERED MOMENTS, AND THE RAW ONES
    WERE MEASURED AND REFUSED.**  ``R_g`` could be rooted from the level's raw
    ``3 x 3`` moment block ``[[w0, w1, s0], [w1, w2, s1], [s0, s1, q0]]``,
    which one bincount sweep already accumulates.  That block is a GRAM, and a
    level whose ``z`` is constant makes it exactly rank 1 -- so its root's
    second row comes back at ``sqrt(eps)`` of the design rather than at ``eps``,
    which is the square-root loss this whole change exists to remove, reappearing
    two dimensions wide inside a cell.  It is not academic.  Measured over the
    20 seeds of the wholly absorbed ``numeric_cat`` geometry the unpenalized
    rung is graded on -- a numeric constant within each level, so the true
    profiled rank is 0 -- with everything relative to
    :func:`_profiled_rank_scale`'s joint-design reference and a round-off cut
    of 4.7122e-08 there:

        raw per-level root       ``||R_eff||_2`` 1.03e-16 .. 2.50e-08,
                                 worst seed 1.89x below the cut
        centered per-level root  ``||R_eff||_2`` 4.23e-17 .. 1.84e-16,
                                 worst seed 2.57e+08x below the cut

    Both forms come back rank 0 on all 20, so this is not the difference
    between right and wrong on that geometry -- it is the difference between a
    margin of 1.89x and one of eight orders, and a 1.89x margin on a rank
    decision is what this repo has twice recorded going the wrong way on
    another machine.

    So the level's mean is subtracted BEFORE the second moment is formed, in a
    second bincount pass over the rows, exactly as
    :func:`superglm.screening._structured._centered_level_factors` does on the
    structured path and for the reason its docstring gives -- the raw-moment
    identity ``w2 - w1**2 / w0`` is unusable when the answer is many orders
    below both terms.  The triangular factor is then written down rather than
    factorized:

        R_g = [[sqrt(w0), sqrt(w0) zbar, s0 / sqrt(w0)],
               [0,        sqrt(m2),      c1 / sqrt(m2)],
               [0,        0,             sqrt(rho)    ]]

    with ``zbar = w1 / w0`` the level's weighted mean, ``m2 = sum w (z -
    zbar)**2`` and ``c1 = sum s (z - zbar)`` its centered moments, and ``rho``
    the response's own residual energy.  Its Gram is the raw block entry for
    entry, so nothing is approximated; what changes is where the cancellation
    happens -- on the ROWS, where the quantity is a sum of squares, instead of
    on two assembled moments that nearly agree.  ``rho`` is the one entry still
    formed by subtraction, and it is the one entry nothing downstream reads:
    it is the reduction's residual, not part of ``V_eff`` or ``U_eff``.

    A level with no mass, or one whose ``z`` carries none, contributes an exact
    zero row rather than a division.  Graded against a row-space QR of the same
    levels in ``test_the_numeric_cat_cell_gram_is_not_what_limits_the_pair``,
    including on a level starved to a single row.

    **THE REFERENCE IS TAKEN FROM A POSITIVELY-WEIGHTED ROW, AND A ROW WITH NO
    WEIGHT MAY NOT SUPPLY IT.**  The scatter has duplicate indices, so what
    survives for a level is whichever of its rows comes LAST.  A zero-weight
    row reaches the arithmetic NOWHERE else -- its weight drops out of every
    accumulation and ``screening_ops`` forces its score to exactly zero -- so
    an unmasked scatter is the single channel through which a row that is not
    in the fit can move a published number.  Masking to ``w > 0`` keeps
    ``shifted`` inside the level's own spread, which is what every exactness
    claim above rests on.

    **THE MEASUREMENT THAT SETTLED IT IS UNBOUNDED, NOT THE BOUNDED ONE, AND
    THE DIFFERENCE IS WHY THIS CHANGED.**  Sweeping MODERATE outliers found
    only a margin: over twelve random seven-row levels with unequal weights, a
    zero-weight reference put ``m2`` in 2.34e-33 .. 2.14e-30 on 6 of 12 where
    a weighted one puts it at exactly 0.0 on 12 of 12, carrying a response
    entry of up to 1.9099 -- 29.0% of that level's own ``sum(s**2 / w)`` --
    against a direction of norm at most 1.461e-15.  That is more than seven
    orders under :func:`._factor_kernels._factor_rank_floor`'s ``sqrt(k eps)``,
    so the pencil dropped it and nothing published moved.  That sweep stands
    and it is not the case that matters, because the outlier's magnitude is
    bounded by nothing: the row is not in the fit.

    Taken to the scale the input boundary actually admits, at UNIT weights and
    one appended row, the same channel erases and then crashes:

        appended zero-weight row   published (statistic, edf0)
        none                       (0.1203333333, 1.0)
        z = 1e+20                  (0.0, 0.0)          level erased
        z = 1e+308                 ValueError          non-finite reduction

    ``1 - 1e20`` and ``2 - 1e20`` round to the SAME double, so ``m2`` came
    back exactly zero and a real per-level slope went with it; at 1e308 the
    level's four shifted values summed past the largest double, ``offset``
    came back infinite and ``_combine_row_factors``' finiteness guard raised.
    Both were decided by the row's POSITION -- the same rows with that row
    first published correctly -- which is the shape of a defect rather than of
    a tolerance.

    The fix costs nothing anyone had: with no zero-weight row present the mask
    is all-true and the scatter is the one it replaces, so the emitted factor
    is bit-identical.  Checked on 300 random all-positive-weight geometries
    and on all eight of the suite's own ``_numeric_cat_case`` arms, 0 of 308
    differing in any bit.

    **THE ROW-LENGTH TEMPORARIES HERE ARE NOT CHUNKED, UNLIKE THE ONES BESIDE
    THEM.**  ``reference[codes_g]``, ``shifted``, ``offset[codes_g]``,
    ``centered``, ``root_w``, ``residual`` and the two products inside it are
    each ``n`` long and about eight are live at once, where
    :func:`numeric_numeric_factor` below reduces its raw rows in chunks of
    ``_FACTOR_CHUNK_DOUBLES`` and the level-emission loop further down chunks
    too.  The moment route this replaced held one or two at a time.  Nothing
    gates it: ``_numeric_margin_within_budget`` bounds ``(k_g + 2)**2`` and
    says nothing about ``n``, so on a ten-million-row frame this is a few
    hundred megabytes of transients on a path documented as one O(n) pass.
    Chunking it is a recorded follow-up and not a defect: every quantity here
    is a bincount accumulation or a row-local map, so the pass splits without
    changing an emitted bit.
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
    s0 = cell(score)
    w0 = cell(w)
    positive = w0 > 0.0
    # SHIFT, THEN CENTRE, THEN ACCUMULATE -- AND THE SHIFT IS WHAT MAKES A
    # CONSTANT LEVEL EXACT RATHER THAN NEARLY SO.  ``zbar = sum(w z) / sum(w)``
    # is two roundings away from ``z`` even when every row of the level carries
    # the SAME ``z``, so ``z - zbar`` comes back at one ulp instead of at zero,
    # ``m2`` at ``eps**2`` instead of at zero -- and ``c1 / sqrt(m2)`` is then
    # the level's WHOLE response energy rather than none of it, because the
    # round-off cancels out of that ratio exactly.  Measured on a level starved
    # to a single row: the level's energy came back doubled, 0.4567 against a
    # true 0.2284.
    #
    # Referencing every row to one of the level's own values first makes the
    # difference EXACT for a constant level (and exact to the last bit for a
    # nearly constant one, since subtracting nearby floats is), so ``m2`` is
    # exactly zero there and ``m2 > 0`` is a rank test with no constant in it.
    # Same device, and the same reason, as
    # :func:`superglm.screening._structured._centered_level_factors`.
    #
    # AND THE VALUE IS TAKEN FROM A POSITIVELY-WEIGHTED ROW, WHICH IS THE
    # WHOLE OF THE GUARANTEE.  This scatter has duplicate indices, so what
    # survives for a level is whichever of its rows comes LAST -- and a
    # zero-weight row is a row.  Such a row is not in the fit, so its ``z`` is
    # bounded by nothing; masking to ``w > 0`` is what keeps ``shifted``
    # inside the level's OWN spread, which is the property every claim above
    # rests on.  A level with no positively-weighted row is left at 0.0 and
    # that reference is inert rather than arbitrary: its ``w0`` and ``s0`` are
    # both exactly zero, so ``positive`` and ``spread`` are both False, every
    # divide below is guarded to zero, and the level emits an exact zero
    # block whatever its ``z`` holds.
    reference = np.zeros(n_g, dtype=np.float64)
    weighted = w > 0.0
    reference[codes_g[weighted]] = z[weighted]
    shifted = z - reference[codes_g]
    offset = np.divide(cell(w * shifted), w0, out=np.zeros_like(w0), where=positive)
    centered = shifted - offset[codes_g]
    m2 = cell(w * centered * centered)
    c1 = cell(score * centered)
    zbar = reference + offset

    root_w0 = np.sqrt(w0)
    root_m2 = np.sqrt(m2)
    spread = m2 > 0.0
    factors = np.zeros((n_g, 3, 3), dtype=np.float64)
    factors[:, 0, 0] = root_w0
    factors[:, 0, 1] = root_w0 * zbar
    factors[:, 0, 2] = np.divide(s0, root_w0, out=np.zeros_like(s0), where=positive)
    factors[:, 1, 1] = root_m2
    factors[:, 1, 2] = np.divide(c1, root_m2, out=np.zeros_like(c1), where=spread)
    # THE RESIDUAL, ACCUMULATED RATHER THAN SUBTRACTED, AND IT COSTS NO EXTRA
    # PASS.  ``rho`` closes the level's factor: the level's response energy
    # splits by Pythagoras into the two projected terms above and this one.
    # Taking it as ``q0 - (s0**2 / w0) - (c1**2 / m2)`` needs one bincount
    # channel too -- ``q0 = sum s**2 / w`` -- and CANCELS, where accumulating
    # the residual is a sum of squares.  The fit is evaluated in the SHIFTED
    # coordinate for the same reason the moments are.
    slope = np.divide(c1, m2, out=np.zeros_like(c1), where=spread)
    level_mean = np.divide(s0, w0, out=np.zeros_like(s0), where=positive)
    root_w = np.sqrt(w)
    residual = np.divide(score, root_w, out=np.zeros_like(score), where=root_w > 0.0) - root_w * (
        level_mean[codes_g] + slope[codes_g] * centered
    )
    factors[:, 2, 2] = np.sqrt(cell(residual * residual))

    k = menu_g.shape[1]
    overlap_width, tensor_width = 2 + k, k
    width = overlap_width + tensor_width + 1
    # CHUNKED FOR THE SAME REASON THE GRIDDED BUILDER IS, AND THE CEILING IS
    # WHY IT MATTERS HERE.  Emitting every level at once costs
    # ``n_g * 3 * width`` doubles, and ``width`` is ``2k + 3``, so at the
    # widest factor the default ``max_cells`` admits -- 1709 contrasts, where
    # ``_within_cubic_budget`` refuses -- that one temporary is 134 MB on top
    # of the 385 MB the whole pair measures there, and 229 MB on top of 616 MB
    # at the ``(k_g + 2)**2 <= max_cells`` gate's own frontier of 2234.  (The
    # second figure read 240 MB before, which is the same byte count taken as
    # a decimal megabyte; every other size in this package is the binary one
    # the kernel reports, so both are stated that way.)  A
    # chunk sized so each merge sees about ``width`` new rows caps it at the
    # accumulator's own shape instead, which is the ``(width, width)`` this
    # pair has to hold in any case.
    merge = _merge_chunk(n_g, 3, width)
    joint = np.zeros((0, width), dtype=np.float64)
    for start in range(0, n_g, merge):
        stop = min(start + merge, n_g)
        rows = stop - start
        local = factors[start:stop]
        menu = menu_g[start:stop, None, :]
        block = np.empty((rows, 3, width), dtype=np.float64)
        block[:, :, :2] = local[:, :, :2]
        block[:, :, 2:overlap_width] = menu * local[:, :, 0:1]
        block[:, :, overlap_width : overlap_width + tensor_width] = menu * local[:, :, 1:2]
        block[:, :, -1] = local[:, :, -1]
        joint = _combine_row_factors(joint, block.reshape(rows * 3, width))
    return PairFactor(
        joint=_square(joint, width),
        overlap_width=overlap_width,
        tensor_width=tensor_width,
    )


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


def _pair_scale(profiled: NDArray) -> float:
    """``tr(V_eff)`` from the factor: ``||R_eff||_F**2``, never a difference.

    **IT TAKES :func:`_profiled_factor`'S BLOCK RATHER THAN THE
    :class:`PairFactor`, AND THAT SIGNATURE IS THE POINT.**  It used to slice
    ``joint[q:q+k, q:q+k]`` itself, which is the same block only while the
    overlap has full rank; where it does not, :func:`_profiled_factor` returns
    the slice STACKED with ``outside' R_ot`` and the two functions in this
    module then disagreed about what ``V_eff`` is, with the ladder scoring one
    and bracketing on the other.  Passing the block in makes the disagreement
    unrepresentable.

    Measured on the mixed suite's ``band x power`` pair -- an
    ``OrderedCategorical`` margin whose 7-column inner menu sits on 5 levels,
    so the overlap is 17 wide with singular values 9.8658e-15 .. 6.8369e+01 and
    three of its directions fall outside :func:`._factor_kernels.
    _factor_rank_floor`'s cut of 4.2005e-06, giving ``R_eff`` 66 rows where the
    slice has 63::

        tr(V - C' pinv(M) C)   538.7617246636    the moment route's own V_eff
        ||R_aug||_F**2         538.7617246636    what the ladder scores
        ||slice||_F**2         441.4506031909    18.06% low

    That number reaches two published fields.  It is
    :func:`~superglm.screening._score_stat.penalized_score_statistic_ladder`'s
    bracket, ``tr(V_eff) / tr(S)``, so a rung CLAMPED to an edge publishes
    ``lambda0`` 18% off what the moment route published; and it is
    :func:`~superglm.screening._score_stat._pair_pencil`'s balance, which is
    accuracy only.  On this fixture every rung searches, so the published move
    is at 1e-7 of a degree of freedom -- the slice-only bracket reads
    ``edf(lo)`` 35.999999895 and ``edf(hi)`` 0.999999344 where the block's
    reads 35.999999872 and 0.999999169 -- and it is the clamped rung on such a
    pair that the slice-only form got wrong outright.
    """
    return float(np.sum(np.asarray(profiled, dtype=np.float64) ** 2))


def _profiled_rank_scale(factor: PairFactor) -> float:
    """The scale a PROFILED direction's rank has to be decided against.

    ``rank(V_eff)`` is the unpenalized rung's ``edf``, and the one thing that
    must not decide it is ``R_eff``'s own largest direction.  On a wholly
    absorbed block -- every probe column a multiple of a column the overlap
    already carries -- ``R_eff`` is round-off in its entirety, so a cut taken
    relative to its own top is a cut taken against the noise it must reject and
    counts all of it.  Measured: 20 of 20 seeds of the reachable absorbed
    ``numeric_cat`` path reported a nonzero ``edf0`` that way, which is the
    same 37-of-40 failure :func:`superglm.screening._score_stat`'s deleted
    ``_profiled_rank`` recorded for the same cut on ``V_eff``.

    So the reference is the JOINT design's, where nothing has been residualized
    away -- ``[X_o | X_T]``, the object the reduction was applied to.  That is
    the factor-space form of the Guttman argument the moment route used:
    ``rank([[V, C'], [C, M]]) - rank(M)`` counted both operands against the
    JOINT's scale rather than against the difference's.  Here it is one count
    rather than two, because the factor already carries the difference as a
    block instead of leaving it to be formed.

    BALANCED FIRST, for the reason :func:`_rank_floor`'s SCALE DISCIPLINE note
    gives: the tensor block carries a numeric margin's scale SQUARED, so at 1e4
    units it would set the reference for the whole joint and a real profiled
    direction would fall under a cut the overlap's own scale never justified.
    Scaling the tensor columns to the overlap's Frobenius mass is a column
    scaling, so it preserves rank exactly, and it is undone on the way out so
    the returned number applies to ``R_eff`` as it stands.
    """
    q, k = int(factor.overlap_width), int(factor.tensor_width)
    if k == 0:
        return 0.0
    overlap = factor.joint[:, :q]
    tensor = factor.joint[:, q : q + k]
    mass_overlap = float(np.sum(overlap**2))
    mass_tensor = float(np.sum(tensor**2))
    balance = (
        np.sqrt(mass_overlap / mass_tensor) if mass_overlap > 0.0 and mass_tensor > 0.0 else 1.0
    )
    balanced = np.concatenate((overlap, balance * tensor), axis=1)
    top = float(np.linalg.norm(balanced, 2)) if balanced.size else 0.0
    return top / balance


def _profiled_factor(factor: PairFactor) -> tuple[NDArray, NDArray]:
    """``(R_eff, z_t)``: the tensor block residualized on the overlap, and its score.

    Frisch-Waugh-Lovell on a triangular factor is a slice.  ``[X_o | X_T |
    yhat] = Q R`` puts ``X_T``'s component orthogonal to ``X_o`` in ``R``'s
    ``(tensor, tensor)`` block and the response's in the column beside it, so
    ``R_eff' R_eff`` is ``V - C' M^-1 C`` and ``R_eff' z_t`` is
    ``U - C' M^-1 u_m`` -- both without forming ``M``, inverting it, or
    subtracting anything.

    **A RANK-DEFICIENT OVERLAP MAKES THAT SLICE WRONG ON ITS OWN, AND THE
    REGIME IS REACHABLE RATHER THAN EXOTIC.**  The reduction is unpivoted, so a
    column of ``X_o`` that its predecessors already span leaves a zero pivot,
    and the direction ``Q`` puts there is an arbitrary unit vector orthogonal
    to what came before -- NOT in ``range(X_o)``.  Residualizing on the whole
    leading block then removes a component the overlap never explained, and the
    profiled block loses curvature the pair genuinely carries.  An
    ``OrderedCategorical`` margin reaches this by construction: its inner
    spline menu can be wider than its level count, so the menu is rank
    deficient before the pair is even formed.  Measured on the mixed suite's
    ``band x power`` pair -- a 7-column menu on 5 levels, whose overlap is rank
    14 of 17 -- taking the slice alone reports a statistic of 13.28 where the
    exact design carries 22.31, and ``z`` moves from 10.07 to 5.67.

    So the directions the overlap does NOT span are returned to the profiled
    block, which is what the moment route's pseudo-inverse did implicitly:
    with ``P`` the projector onto ``range(R_o)``,

        X_T - P_{X_o} X_T = Q_o (I - P) R_ot + Q_T R_eff ,

    so ``[(I - P) R_ot ; R_eff]`` is the residualized tensor block and
    ``[(I - P) z_o ; z_t]`` its response.  A full-rank overlap makes ``I - P``
    exactly zero and the returned factor is the bare slice, bit for bit.  The
    rank is decided on ``R_o`` -- a factor formed by reduction, where nothing
    has cancelled -- at :func:`._factor_kernels._factor_rank_floor`, the same cut
    every other rank decision here takes.  The moment route took the same
    decision and did not state it: ``cho_factor`` or, on refusal,
    ``numpy.linalg.pinv``'s shape-derived default ``rcond``.

    **THE RETURNED BLOCK IS THE ONLY ``V_eff`` IN THIS MODULE.**  Its Frobenius
    mass IS ``tr(V_eff)``, and :func:`_pair_scale` is handed it rather than
    re-slicing ``joint`` for itself: on the fixture above the two differ by
    18.06%, and the smaller one used to set the ladder's bracket.
    """
    q, k = int(factor.overlap_width), int(factor.tensor_width)
    R_eff = factor.joint[q : q + k, q : q + k]
    z_t = factor.joint[q : q + k, -1]
    if q == 0 or k == 0:
        return R_eff, z_t
    R_o = factor.joint[:q, :q]
    left, values, _ = np.linalg.svd(R_o)
    top = float(values[0]) if values.size else 0.0
    if top <= 0.0:
        outside = left
    else:
        outside = left[:, values <= _factor_rank_floor(q) * top]
    if outside.shape[1] == 0:
        return R_eff, z_t
    return (
        np.concatenate((R_eff, outside.T @ factor.joint[:q, q : q + k]), axis=0),
        np.concatenate((z_t, outside.T @ factor.joint[:q, -1])),
    )
