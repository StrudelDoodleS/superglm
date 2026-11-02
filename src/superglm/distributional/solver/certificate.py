"""The terminal rank of the LSS penalised curvature, certified against its formation.

The terminal decision keeps the eigenvalues of the formed, Jacobi-equilibrated
penalised curvature ``G`` above ``tau = max(eps, n_a eps) lambda_max``.
Forming ``G`` from the rows moves it by more than ``tau`` (Higham 2002,
eq. 20.12 and eq. 3.13), so directions within that distance of the cut are
decided by rounding.  :func:`rank.decompose_symmetric_deflated` decides the
rank of the exact matrix ``H`` instead: Weyl's inequality outside a band
``formation_bound`` wide, and inside it one pass over the rows that evaluates
the grey block (Li and Li, *Linear Algebra Appl.* 395, 2005, Thm 2).  This
module supplies that routine's two inputs at the terminal state:

* :func:`formation_bound`, a bound on ``||D^-1 (G - H) D^-1||_2`` from the
  statistics each assembly records (``_formation.CurvatureFormation``);
* a ``project`` that re-evaluates the accepted curvature source's channels at
  the terminal coefficients, chunk by chunk as the assembly did, and forms
  ``B' H B`` and ``H B`` with their entrywise error bounds.

``H`` is the exact matrix of the stored design (its stored factors multiplied
exactly; on the dense route its dense predictor matrices), the computed
channels and the stored penalty.  Errors in evaluating the channels are
outside the certificate.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator
from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

import superglm.distributional.solver.chunks as chunking
from superglm.distributional.solver._formation import AbsDesign, CurvatureFormation, gamma
from superglm.distributional.solver.curvature import CurvatureDecision
from superglm.distributional.solver.packing import packed_pairs
from superglm.solvers.rank import SymmetricGreyProjection, decompose_symmetric_deflated


def _up(value: float) -> float:
    return float(np.nextafter(value, np.inf))


def _column_sum_bound(values: NDArray) -> float:
    """An upper bound on ``max_j sum_i values_ij`` for nonnegative ``values``."""
    if values.size == 0:
        return 0.0
    rows = values.shape[0]
    return _up(float(np.max(np.sum(values, axis=0), initial=0.0)) * (1.0 + gamma(rows + 2)))


def formation_bound(
    formation: CurvatureFormation,
    penalized: NDArray,
    penalty: NDArray,
    blocks: Iterable[slice],
    *,
    face=None,
    reduced: NDArray | None = None,
) -> float:
    """``>= ||D^-1 (G - H) D^-1||_2``, ``D = diag(sqrt(diag G))``, for the matrix the policy decides.

    ``G`` is ``penalized`` (``reduced = N' G N`` under an active face) and
    ``H`` its exact counterpart (module docstring).  The steps, with ``u =
    2**-53`` and ``gamma_m = m u / (1 - m u)``:

    1. Entrywise, in any order of summation, ``|G - H| <= gamma_m M`` with
       ``M = sum_r X^abs_r' |C_r| X^abs_r + |S|`` (Higham 2002, eq. 3.13).
       ``m`` counts the roundings on any term's path. Chunked route: the
       factor chain's inner sums (at most ``inner`` each, ``4 inner`` for
       the two sides and a congruence ``R' (B' W B) R``), the products with
       the channel (2), a bin, level or bin-pair sum and then the sum over
       its cells (``n`` together: every nonempty cell holds a row, and an
       empty one adds an exact zero), BLAS row sums of a dense block
       (``n``), the intercept sums, one add per chunk or block into the
       accumulator (``chunks``), the triangle copy and mirrors (exact), the
       symmetric average and the penalty's add (2): ``m = 2 n + chunks + 4
       inner + 16``. Dense route: ``M' (w M)`` over ``n`` rows, its two
       products, the symmetric average and the penalty: ``m = n + 8``.
    2. For nonnegative ``v``, ``t_r = X^abs_r v``: ``t_r' |C_r| t_r <= sum_k
       w_rk t_rk^2`` with the row weights of ``_formation.channel_weights``.
       The spectral norm of a nonnegative symmetric matrix is attained on a
       nonnegative vector (Perron-Frobenius, as in Rump, *BIT* 46, 2006, Thm
       2.3), and the right side is block diagonal over the predictors with
       positive semidefinite blocks, each bounded by its trace.  So ``||D^-1
       M_data D^-1||_2 <= max_k sum_(i in k) majorant_i / d_i``, the
       majorant taking those weights.  ``||D^-1 |S| D^-1||_2`` is at most its
       largest column sum.
    3. Under a face, ``G_red - N' H N = (G_red - N' G N) + N' (G - H) N``.
       The first is the reduction's rounding, ``gamma_(2p + 2) |N|' |G| |N|``
       (two products of length ``p`` and the symmetric average); the second
       is ``L' (D^-1 (G - H) D^-1) L`` with ``L = D N D_r^-1``, and
       ``||L||_2^2 <= ||L||_1 ||L||_inf``.

    Every bound is accumulated outward (``nextafter``, ``gamma`` inflations
    counting its own roundings; terms below float64's normal range are not
    counted).  Returns ``inf`` when a column the bound needs has no positive
    diagonal or a weight overflowed; the caller then keeps the formed
    decision.
    """
    matrix = np.asarray(penalized, dtype=np.float64)
    diagonal = np.diag(matrix)
    n_rows = int(formation.rows)
    if formation.route == "dense":
        roundings = n_rows + 8
        statistic_roundings = n_rows + 4
    else:
        roundings = 2 * n_rows + formation.chunks + 4 * formation.inner + 16
        # Each statistic: an absolute table entry (``inner`` sums), its square
        # and weight (2), a cell sum and the sum over cells (``n``), the chunks.
        statistic_roundings = 2 * n_rows + formation.chunks + 2 * formation.inner + 8
    # The row weights carry ``gamma_(K + 4)`` (``channel_weights``).
    blocks = list(blocks)
    predictors = len(blocks)
    majorant = np.asarray(formation.majorant, dtype=np.float64) * (
        1.0 + 2.0 * gamma(statistic_roundings + predictors + 4)
    )
    if not np.all(np.isfinite(majorant)):
        return math.inf
    absolute_penalty = np.abs(np.asarray(penalty, dtype=np.float64))
    active = diagonal > 0.0
    if np.any(diagonal < 0.0) or np.any(
        ~active & ((majorant > 0.0) | np.any(absolute_penalty != 0.0, axis=0))
    ):
        return math.inf
    traces = []
    for columns in blocks:
        local = active[columns]
        ratio = np.nextafter(majorant[columns][local] / diagonal[columns][local], np.inf)
        traces.append(_up(math.fsum(ratio) * (1.0 + gamma(2))))
    data = max(traces, default=0.0)
    reciprocal = np.zeros_like(diagonal)
    reciprocal[active] = 1.0 / np.sqrt(diagonal[active])
    scaled_penalty = reciprocal[:, None] * absolute_penalty * reciprocal[None, :]
    penalty_norm = _up(_column_sum_bound(scaled_penalty) * (1.0 + gamma(4)))
    full = _up(gamma(roundings) * _up(data + penalty_norm) * (1.0 + gamma(2)))
    if face is None:
        return full
    reduced_matrix = np.asarray(reduced, dtype=np.float64)
    reduced_diagonal = np.diag(reduced_matrix)
    if np.any(reduced_diagonal <= 0.0):
        return math.inf
    null = np.asarray(face.null_basis, dtype=np.float64)
    reduced_reciprocal = 1.0 / np.sqrt(reduced_diagonal)
    lift = np.abs(np.sqrt(diagonal)[:, None] * null * reduced_reciprocal[None, :])
    lift_norm = _up(_column_sum_bound(lift) * _column_sum_bound(lift.T) * (1.0 + gamma(8)))
    weighted = np.abs(null) * reduced_reciprocal[None, :]
    reduction = weighted.T @ (np.abs(matrix) @ weighted)
    rounding = _up(
        gamma(2 * matrix.shape[0] + 2)
        * _column_sum_bound(reduction)
        * (1.0 + gamma(2 * matrix.shape[0] + 4))
    )
    return _up(_up(full * lift_norm) + rounding)


# Rows per block of the dense route's row pass, so its absolute rows and
# products are block-sized rather than the size of the design.
_DENSE_BLOCK_ROWS = 2**15


class _DenseRows:
    """One block of rows of the dense predictor matrices."""

    def __init__(self, layout, matrices, curvature, start, stop):
        self.start, self.stop = start, stop
        self.curvature = curvature[start:stop]
        self._matrices = [matrix[start:stop] for matrix in matrices]
        self._absolute = [np.abs(matrix) for matrix in self._matrices]
        self._slices = [state.coefficient_slice for state in layout.predictors]

    def products(self, k, values):
        return self._matrices[k] @ values[self._slices[k]]

    def abs_products(self, k, values):
        return self._absolute[k] @ values[self._slices[k]]

    def add_rmatvec(self, k, rows, out):
        out[self._slices[k]] += self._matrices[k].T @ rows

    def add_abs_rmatvec(self, k, rows, out):
        out[self._slices[k]] += self._absolute[k].T @ rows


class _ChunkRows:
    """One likelihood chunk's rows, through each predictor design's own products."""

    def __init__(self, layout, design: AbsDesign, chunk):
        rows = chunk.rows
        self.start, self.stop = rows.start, rows.stop
        self.curvature = chunk.curvature_packed
        self._layout = layout
        self._design = design
        self._designs = [
            chunking._row_design(state, rows, copy=False) for state in layout.predictors
        ]

    def products(self, k, values):
        state = self._layout.predictors[k]
        local = values[state.coefficient_slice]
        start = int(state.intercept_index is not None)
        out = np.zeros((self.stop - self.start, values.shape[1]))
        design = self._designs[k]
        if design.p:
            for column in range(values.shape[1]):
                out[:, column] = design.matvec(local[start:, column])
        if start:
            out += local[0]
        return out

    def abs_products(self, k, values):
        return self._design.matvec(k, self.start, self.stop, values)

    def add_rmatvec(self, k, rows, out):
        state = self._layout.predictors[k]
        columns = state.coefficient_slice
        start = int(state.intercept_index is not None)
        if start:
            out[columns.start] += np.sum(rows, axis=0)
        design = self._designs[k]
        if design.p:
            for column in range(rows.shape[1]):
                out[columns.start + start : columns.stop, column] += design.rmatvec(rows[:, column])

    def add_abs_rmatvec(self, k, rows, out):
        self._design.rmatvec(k, self.start, self.stop, rows, out)


def _chunk_rows(context, coefficients, source, formation, design) -> Iterator[_ChunkRows]:
    iterator = chunking.iter_likelihood_chunks(
        context.family,
        context.layout,
        context.response,
        context.likelihood_plan,
        coefficients,
        chunk_size=context.chunk_size,
        curvature_source=source,
        _range_geometry=formation.range_geometry,
        **(
            {"likelihood_cache": context.likelihood_cache}
            if context.likelihood_cache is not None
            else {}
        ),
    )
    try:
        for chunk in iterator:
            yield _ChunkRows(context.layout, design, chunk)
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()


def project_terminal_curvature(
    basis: NDArray,
    rows: Iterable,
    *,
    layout,
    penalty: NDArray,
    formation: CurvatureFormation,
    face=None,
    product_roundings: int,
) -> SymmetricGreyProjection:
    """``B' H B`` and ``H B`` from the rows, with entrywise error bounds (``project``'s contract).

    ``basis`` is in the decided matrix's coordinates; under a face it is
    lifted to ``N B`` first, with ``|fl(N B) - N B| <= gamma_(p_r) |N| |B|``
    carried as ``lift``.  Per row block ``T_k = X_k B_k`` through the
    design's own products, ``|fl(T_k) - X_k (N B)_k| <= Delta_k = X^abs_k
    (gamma_q |B_k| + lift_k)`` with ``q = product_roundings`` (the factor
    chain, the coefficient sum, the groups' sum and the intercept).  Then,
    channel by channel (Higham 2002, eq. 3.13, with ``T = T_hat + delta``):

    * ``block += T_k' diag(c_kl) T_l`` (and its transpose for ``k != l``),
      error ``gamma_qH |T_k|' |c| |T_l| + (1 + gamma_qH) (Delta_k' |c| |T_l|
      + |T_k|' |c| Delta_l + Delta_k' |c| Delta_l)``, ``qH = n + chunks + 2
      channels + 4`` (a row sum, the chunks, the channels into one array,
      two products);
    * ``cross_k += X_k' (sum_l c_kl T_l)``, error ``gamma_qF X^abs_k' (sum_l
      |c_kl| |T_l|) + (1 + gamma_qF) X^abs_k' (sum_l |c_kl| Delta_l)``, ``qF
      = 2 n + chunks + inner + 2 K + 8`` (the row combination, the design's
      transpose product, the chunks);
    * the penalty, ``B' S B`` within ``gamma_(2p + 1) |B|' |S| |B|`` and ``S
      B`` within ``gamma_(p + 1) |S| |B|``, plus the lift's terms;
    * under a face, ``cross = N' (H N B)`` within ``gamma_p |N|' |cross| +
      |N|' cross_bound``.

    The bounds' own arithmetic (sums and products of nonnegative terms) is
    covered by a final ``1 + 2 gamma_(2n + 2p + 64)``.
    """
    values = np.asarray(basis, dtype=np.float64)
    width = layout.n_coefficients
    grey = values.shape[1]
    if face is None:
        lifted, lift = values, np.zeros_like(values)
    else:
        null = np.asarray(face.null_basis, dtype=np.float64)
        lifted = null @ values
        lift = (np.abs(null) @ np.abs(values)) * (
            gamma(null.shape[1]) * (1.0 + gamma(null.shape[1] + 2))
        )
    absolute = np.abs(lifted)
    pairs = packed_pairs(len(layout.predictors))
    channel_of = {}
    for index, (k, j) in enumerate(pairs):
        channel_of[(k, j)] = channel_of[(j, k)] = index
    k_parameters = len(layout.predictors)
    block = np.zeros((grey, grey))
    block_gamma = np.zeros((grey, grey))
    block_delta = np.zeros((grey, grey))
    cross = np.zeros((width, grey))
    cross_gamma = np.zeros((width, grey))
    cross_delta = np.zeros((width, grey))
    product_gamma = gamma(product_roundings)
    inflation = 1.0 + gamma(product_roundings + 2)
    perturbation = product_gamma * absolute + lift
    total_rows = 0
    chunks = 0
    for row_block in rows:
        chunks += 1
        total_rows += row_block.stop - row_block.start
        curvature = np.asarray(row_block.curvature, dtype=np.float64)
        products = [row_block.products(k, lifted) for k in range(k_parameters)]
        magnitudes = [np.abs(product) for product in products]
        deltas = [row_block.abs_products(k, perturbation) * inflation for k in range(k_parameters)]
        for index, (k, j) in enumerate(pairs):
            weights = curvature[:, index][:, None]
            absolute_weights = np.abs(weights)
            contribution = products[k].T @ (weights * products[j])
            magnitude = magnitudes[k].T @ (absolute_weights * magnitudes[j])
            weighted_delta = absolute_weights * deltas[j]
            delta = (
                deltas[k].T @ (absolute_weights * magnitudes[j])
                + magnitudes[k].T @ weighted_delta
                + deltas[k].T @ weighted_delta
            )
            block += contribution
            block_gamma += magnitude
            block_delta += delta
            if k != j:
                block += contribution.T
                block_gamma += magnitude.T
                block_delta += delta.T
        for k in range(k_parameters):
            combined = np.zeros_like(products[k])
            combined_magnitude = np.zeros_like(products[k])
            combined_delta = np.zeros_like(products[k])
            for j in range(k_parameters):
                weights = curvature[:, channel_of[(k, j)]][:, None]
                combined += weights * products[j]
                combined_magnitude += np.abs(weights) * magnitudes[j]
                combined_delta += np.abs(weights) * deltas[j]
            row_block.add_rmatvec(k, combined, cross)
            row_block.add_abs_rmatvec(k, combined_magnitude, cross_gamma)
            row_block.add_abs_rmatvec(k, combined_delta, cross_delta)
    block_roundings = total_rows + chunks + 2 * len(pairs) + 4
    cross_roundings = 2 * total_rows + chunks + formation.inner + 2 * k_parameters + 8
    block_bound = (
        gamma(block_roundings) * block_gamma + (1.0 + gamma(block_roundings)) * block_delta
    )
    cross_bound = (
        gamma(cross_roundings) * cross_gamma + (1.0 + gamma(cross_roundings)) * cross_delta
    )

    penalty_matrix = np.asarray(penalty, dtype=np.float64)
    absolute_penalty = np.abs(penalty_matrix)
    penalty_product = penalty_matrix @ lifted
    penalty_magnitude = absolute_penalty @ absolute
    penalty_lift = absolute_penalty @ lift
    block += lifted.T @ penalty_product
    cross += penalty_product
    block_bound = (
        block_bound
        + gamma(2 * width + 2) * (absolute.T @ penalty_magnitude)
        + 2.0 * (lift.T @ penalty_magnitude)
        + lift.T @ penalty_lift
    )
    cross_bound = cross_bound + gamma(width + 2) * penalty_magnitude + penalty_lift
    if face is not None:
        null = np.asarray(face.null_basis, dtype=np.float64)
        reduced_cross = null.T @ cross
        cross_bound = gamma(width + 2) * (np.abs(null).T @ np.abs(cross)) + np.abs(null).T @ (
            cross_bound
        )
        cross = reduced_cross
    final = 1.0 + 2.0 * gamma(2 * total_rows + 2 * width + 64)
    return SymmetricGreyProjection(block, cross, block_bound * final, cross_bound * final)


def certify_terminal_rank(
    decision: CurvatureDecision,
    formation: CurvatureFormation | None,
    *,
    layout,
    penalized: NDArray,
    penalty: NDArray,
    face=None,
    rows: Callable[[], Iterable],
    product_roundings: int,
) -> CurvatureDecision:
    """The accepted curvature decision with its rank certified against the formation bound.

    ``decision`` is the curvature policy's accepted result; its acceptance
    (and so ``actual_source``) is unchanged.  ``rows`` returns the row blocks
    of the accepted source at the terminal coefficients and is called only
    when the grey set is non-empty, once.  Grey directions resolved negative
    are dropped (rank module, ``indefinite``) and do not reopen the policy:
    they lie within the formation bound of a matrix the policy accepted,
    the scale of the channels' own evaluation error, which the certificate
    does not cover.  A decision the row pass cannot improve on (the grey set
    past its cap, a non-finite projection) is the formed one, unchanged.
    Without formation statistics (an assembly route or group type that
    records none) the decision is returned as it was, ``certified=None``.
    """
    matrix = decision.matrix
    if formation is None or matrix is None or decision.decomposition is None:
        return decision
    blocks = [state.coefficient_slice for state in layout.predictors]
    bound = formation_bound(
        formation, penalized, penalty, blocks, face=face, reduced=None if face is None else matrix
    )
    if not math.isfinite(bound):
        return decision

    def project(basis: NDArray) -> SymmetricGreyProjection:
        return project_terminal_curvature(
            basis,
            rows(),
            layout=layout,
            penalty=penalty,
            formation=formation,
            face=face,
            product_roundings=product_roundings,
        )

    certified = decompose_symmetric_deflated(
        matrix, project, formation_bound=bound, spectrum=decision.spectrum
    )
    info = dict(certified.grey_subspace or {})
    if info.get("capped") or info.get("projection_failed"):
        certified = replace(decision.decomposition, grey_subspace=info)
    telemetry = replace(
        decision.telemetry,
        rank=int(certified.rank),
        grey_k=int(info.get("k", 0.0)),
        certified=bool(info.get("certified", 0.0)),
    )
    return replace(decision, decomposition=certified, telemetry=telemetry)


def dense_rows(layout, matrices, channels: Callable[[], NDArray]) -> Callable[[], Iterable]:
    """The dense route's rows in blocks, their channels evaluated on demand."""

    def rows():
        curvature = np.asarray(channels(), dtype=np.float64)
        n = int(curvature.shape[0])
        for start in range(0, n, _DENSE_BLOCK_ROWS):
            yield _DenseRows(layout, matrices, curvature, start, min(start + _DENSE_BLOCK_ROWS, n))

    return rows


def chunked_rows(context, coefficients, source, formation) -> Callable[[], Iterable]:
    """The chunked route's row blocks, replayed as the assembly streamed them."""

    def rows():
        return _chunk_rows(context, coefficients, source, formation, AbsDesign(context.layout))

    return rows


def product_roundings(layout, formation: CurvatureFormation) -> int:
    """``q`` for ``T = X B``: the factor chain, the coefficient sum, the groups' sum, the intercept."""
    widest = max(
        (state.coefficient_slice.stop - state.coefficient_slice.start)
        for state in layout.predictors
    )
    if formation.route == "dense":
        return widest + 4
    groups = max(len(state.design.group_matrices) for state in layout.predictors)
    return formation.inner + widest + groups + 4


__all__ = [
    "certify_terminal_rank",
    "chunked_rows",
    "dense_rows",
    "formation_bound",
    "product_roundings",
    "project_terminal_curvature",
]
