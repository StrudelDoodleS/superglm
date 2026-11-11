"""W(rho) derivatives and correction for REML.

Computes dW/deta, d2W/deta2 (analytic + FD fallback), and the
W(rho) correction terms for the REML gradient and Hessian.

References
----------
- Wood (2011) Section 3.4 / Appendix C.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from superglm._blas_threads import release_for_serial_row_passes
from superglm._group_matrix._block_queue import _side, block_cost, diagonal_cost
from superglm._group_matrix._column_local_centering import (
    recentred_about,
    recentred_gram_rows,
    recentred_products,
)
from superglm._group_matrix._group_matrix_centered import (
    _raw_centering_admitted,
    centered_gram_rhs,
    centered_signed_grams,
)
from superglm._group_matrix._lookup_tables import (
    _P_SPACE_WEIGHT,
    lookup_plan,
    table_leverages,
    table_rmatvec,
)
from superglm._parallel import pooled_map
from superglm.distributions import _VARIANCE_FLOOR, Gamma, clip_mu
from superglm.group_matrix import DesignMatrix
from superglm.links import LogLink, stabilize_eta
from superglm.reml._leverage_rows import solved_row_blocks
from superglm.reml.observed_geometry import ObservedREMLGeometry
from superglm.reml.penalty_algebra import (
    coerce_reml_penalties,
    penalty_component_dense_matrix,
    penalty_component_matvec,
)
from superglm.solvers._structured.balance_tree import (
    ProfiledSumToZeroTreeFactor,
)
from superglm.solvers._structured.block_leaves import (
    ProfiledFactorSmoothLeafFactor,
    factor_smooth_moment_operators,
)
from superglm.solvers._structured.layout import (
    FactorSmoothLeafLayout,
    get_nested_structured_layout,
    get_structured_layout,
    structured_design_matvec,
    structured_design_rmatvec,
)
from superglm.solvers._structured.moments import (
    build_block_structured_system,
    build_nested_structured_system,
)
from superglm.solvers._structured.nested import (
    NestedStructuredLayout,
    ProfiledNestedSchurFactor,
)
from superglm.solvers._structured.operators import (
    CenteredBlockOperator,
    CompactSymmetricOperator,
    LowRankSymmetricOperator,
    SumBlockOperator,
    compact_operator_diagonal,
)
from superglm.solvers._structured.state import (
    centred_data_operator,
)
from superglm.solvers.centered_system import (
    _dense_split,
    _DenseSplit,
    anchored_dense_moments,
    dense_mean_pair,
    dense_split_cross,
    dense_split_gram,
    grouped_augmented_row_quadratic_forms,
    two_pass_centred_gram,
)
from superglm.solvers.hessian_factor import (
    DenseHessianFactor,
    HessianFactor,
    as_hessian_factor,
)
from superglm.solvers.mode_score import (
    dense_centred_matvec,
    dense_centred_rmatvec,
    dense_columns,
    linear_predictor,
)
from superglm.solvers.pirls import PIRLSResult
from superglm.solvers.working_rows import fisher_working_weights
from superglm.types import GroupSlice, PenaltyComponent

_SIGNED_GRAM_BATCH_BYTES = 64 << 20


def _penalty_rhs(
    penalties: list[PenaltyComponent], lambdas: dict[str, float], beta: NDArray, gms, p: int
) -> NDArray:
    """The ``(p, m)`` penalty directions ``s_j = S_j beta``, each on its own block."""
    rhs = np.zeros((p, len(penalties)), dtype=np.float64)
    for i, pc in enumerate(penalties):
        rhs[pc.group_sl, i] = lambdas[pc.name] * penalty_component_matvec(
            pc, beta[pc.group_sl], gms[pc.group_index]
        )
    return rhs


def _structured_weight_cross(
    factor: ProfiledNestedSchurFactor,
    penalties: list[PenaltyComponent],
    lambdas: dict[str, float],
    operators: dict[int, NDArray | CompactSymmetricOperator],
) -> NDArray:
    """The leverage route's cross terms ``-(P + P') / 2`` from the nested factor.

    ``P_ij = tr(H^-1 S_i H^-1 C_j)``, ``S_i = lambda_i Omega_i`` and ``C_j`` the
    weight-derivative operator, as ``_leverage_derivative_terms`` forms them
    from solved rows.  The factor's derivative directions may carry an operator
    without a penalty, so one ``derivative_cross_traces`` call over the ``m``
    penalty and ``m`` operator directions forms every ``P_ij``.
    """
    m = len(penalties)
    # a structured factor's derivative operators are compact, never dense
    compact = [cast(CompactSymmetricOperator, operators[j]) for j in range(m)]
    directions: list = [(pc, lambdas[pc.name], None) for pc in penalties]
    directions += [(None, 0.0, operator) for operator in compact]
    cross = factor.derivative_cross_traces(directions)[:m, m:]
    return -0.5 * (cross + cross.T)


def _leverage_table_gradient(
    plan,
    factor: DenseHessianFactor,
    centre_hi: NDArray,
    centre_lo: NDArray | None,
    row_weights: NDArray,
    sum_w: float,
    penalty_rhs: NDArray,
) -> NDArray:
    """The W(rho) gradient correction from table read-back leverages (``_lookup_tables``).

    ``grad_j = sum_r w'_r (x_r' dbeta_j) (h_r + 1/sum_w) / 2`` is ``dbeta_j'
    X_c' v`` with the row channel ``v_r = w'_r (h_r + 1/sum_w) / 2``: one
    leverage gather and one scatter, never a solved row.  Rows without a
    weight derivative are never read.
    """
    active = row_weights != 0.0
    leverage = table_leverages(plan, factor.inverse, centre_hi, centre_lo, active)
    channel = np.zeros_like(leverage)
    channel[active] = row_weights[active] * (0.5 * (leverage[active] + 1.0 / sum_w))
    rhs = table_rmatvec(plan, channel, centre_hi, centre_lo)
    return rhs @ -factor.solve(penalty_rhs)


@dataclass(frozen=True)
class _PenaltyBatch:
    """Adjacent penalty blocks of one shape, traced together (``_leverage_cross_batches``).

    Block ``b`` is columns ``start + b * width`` onward, ``width`` of them, and
    ``members[b]`` are its penalties, the ``dense`` matrix penalties first
    (``stacked[b]``, lambda-scaled and side by side), then identity penalties
    (``identity_scales[b]``, their lambdas).
    """

    start: int
    width: int
    members: NDArray
    dense: int
    stacked: NDArray | None
    identity_scales: NDArray

    @property
    def stop(self) -> int:
        return self.start + self.width * self.members.shape[0]


def _leverage_derivative_terms(
    dm: DesignMatrix,
    inverse: NDArray,
    mean_x: NDArray,
    row_weights: NDArray,
    sum_w: float,
    batches: list[_PenaltyBatch],
    penalty_rhs: NDArray,
    mean_lo: NDArray | None = None,
    cross_terms: bool = True,
) -> tuple[NDArray, NDArray | None]:
    """The W(rho) gradient correction and the Hessian's penalty-by-weight cross traces, in one pass.

    Each centred row ``x_r`` with ``w'_r != 0`` gives ``y_r = H^-1 x_r`` and
    ``h_r = x_r' y_r``; with ``a_jr = -w'_r y_r' s_j``, ``s_j = S_j beta``:

    - ``grad_j = sum_r a_jr (h_r + 1/sum_w) / 2 = tr(H^-1 C_j) / 2 + d log(sum
      W)/(2 drho_j)``, ``C_j = X_c' diag(a_j) X_c`` (Wood 2011, sec. 3.5.1);
    - ``P_ij = sum_r (y_r' S_i y_r) a_jr = tr(H^-1 S_i H^-1 C_j)``.

    One solved row per data row whatever ``m`` is, never a ``p x p`` Gram per
    penalty. ``cross_terms=False`` returns ``None`` for ``P``.
    """
    m = penalty_rhs.shape[1]
    block_rhs = [
        np.stack(
            [
                penalty_rhs[batch.start + b * batch.width : batch.start + (b + 1) * batch.width][
                    :, members
                ]
                for b, members in enumerate(batch.members)
            ]
        )
        for batch in batches
    ]
    batch_ones = {batch.width: np.ones(batch.width, dtype=np.float64) for batch in batches}
    grad = np.zeros(m, dtype=np.float64)
    grad_compensation = np.zeros(m, dtype=np.float64)
    cross = np.zeros((m, m), dtype=np.float64)
    cross_compensation = np.zeros((m, m), dtype=np.float64)

    def contributions(block):
        """One block's gradient and cross-trace terms (``None`` for a block with no active row)."""
        if block is None:
            return None
        rows, pieces = block
        quadratic = np.empty((rows.size, m), dtype=np.float64) if cross_terms else None
        directions = np.empty((rows.size, m), dtype=np.float64)
        leverage = np.empty(rows.size, dtype=np.float64)
        # Each row's terms read only its own solved row, so they are formed
        # piece by piece; the block's sums below run over all its rows.
        for span, solved, piece_leverage in pieces:
            leverage[span] = piece_leverage
            count = solved.shape[0]
            for batch, rhs in zip(batches, block_rhs, strict=True):
                # (blocks, rows, width): a view of each block's solved rows, for matmul
                n_blocks, width = batch.members.shape[0], batch.width
                solved_blocks = (
                    solved[:, batch.start : batch.stop]
                    .reshape(count, n_blocks, width)
                    .transpose(1, 0, 2)
                )
                directions[span, batch.members] = -np.matmul(solved_blocks, rhs).transpose(1, 0, 2)
                if quadratic is None:
                    continue
                # Row sums over a block's width as one matrix-vector product.
                ones = batch_ones[width]
                if batch.stacked is not None:
                    products = np.matmul(solved_blocks, batch.stacked).reshape(
                        n_blocks, count, batch.dense, width
                    )
                    products *= solved_blocks[:, :, None, :]
                    quadratic[span, batch.members[:, : batch.dense]] = (
                        (products.reshape(-1, width) @ ones)
                        .reshape(n_blocks, count, batch.dense)
                        .transpose(1, 0, 2)
                    )
                    products = None  # freed before the next batch's temporaries
                if batch.members.shape[1] > batch.dense:
                    squares = ((solved_blocks * solved_blocks).reshape(-1, width) @ ones).reshape(
                        n_blocks, count
                    )
                    quadratic[span, batch.members[:, batch.dense :]] = (
                        squares.T[:, :, None] * batch.identity_scales[None, :, :]
                    )
            # Drop this piece and its products before the next piece is formed.
            solved = solved_blocks = piece_leverage = products = squares = None
        weighted = row_weights[rows, None] * directions
        gradient = weighted.T @ (0.5 * (leverage + 1.0 / sum_w))
        return (gradient,) if quadratic is None else (gradient, quadratic.T @ weighted)

    # Blocks are formed and reduced on the pool when they are independent;
    # their terms are summed here in row order either way, so the totals are
    # the same at every worker count (``_parallel.pooled_map``).
    sums = [(grad, grad_compensation)] + ([(cross, cross_compensation)] if cross_terms else [])
    # A block's rows keep their leverage and directions, and their quadratic
    # forms and weighted directions with the cross terms.
    row_bytes = 8 * (1 + (3 if cross_terms else 2) * m)
    # A piece's widest products: one batch's dense penalties, side by side.
    piece_row_bytes = 8 * max(
        (
            batch.dense * (batch.stop - batch.start)
            for batch in batches
            if cross_terms and batch.stacked is not None
        ),
        default=0,
    )
    blocks = solved_row_blocks(
        dm, inverse, mean_x, mean_lo, row_weights != 0.0, row_bytes, piece_row_bytes
    )
    if blocks.block_bytes is None:
        # The chunked route runs on this thread under the fit's BLAS policy; a
        # generator, so each chunk is reduced before the next is formed.
        parts = (contributions(job()) for job in blocks.jobs)
    else:
        parts = pooled_map(lambda job: contributions(job()), list(blocks.jobs), blocks.block_bytes)
    try:
        for part in parts:
            if part is None:
                continue
            for (total, compensation), contribution in zip(sums, part, strict=True):
                corrected = contribution - compensation
                updated = total + corrected
                compensation[...] = (updated - total) - corrected
                total[...] = updated
    finally:
        # A raise in the loop leaves the generator parked inside the pool's
        # BLAS scope; closing it restores the BLAS policy and stops the pool.
        parts.close()
    return grad, cross if cross_terms else None


def _leverage_cross_batches(
    penalties: list[PenaltyComponent],
    lambdas: dict[str, float],
    group_matrices,
    dense_matrices: bool = True,
) -> list[_PenaltyBatch]:
    """The penalties by coefficient block, lambda-scaled, blocks of one shape batched together.

    Each dense member is the matrix ``reml_direct_hessian`` traces against,
    so both read the same ``S_i``. Adjacent blocks of one shape (a
    spline-by-category term's levels) share a batch, so a row block costs a
    few products per term, not per level. ``dense_matrices=False`` leaves
    ``stacked`` unset: only the cross traces read it.
    """
    members: dict[tuple[int, int], list[int]] = {}
    for index, pc in enumerate(penalties):
        members.setdefault((pc.group_sl.start, pc.group_sl.stop), []).append(index)
    # Runs of adjacent blocks of one shape: (start, width, n_dense, n_identity) and blocks.
    runs: list[tuple[tuple[int, int, int, int], list[tuple[list[int], list[int]]]]] = []
    for (start, stop), indices in sorted(members.items()):
        dense = [i for i in indices if penalties[i].penalty_kind != "identity"]
        identity = [i for i in indices if penalties[i].penalty_kind == "identity"]
        shape = (stop - start, len(dense), len(identity))
        if runs:
            (run_start, width, n_dense, n_identity), blocks = runs[-1]
            if (width, n_dense, n_identity) == shape and run_start + width * len(blocks) == start:
                blocks.append((dense, identity))
                continue
        runs.append(((start, *shape), [(dense, identity)]))
    batches = []
    for (start, width, n_dense, _), blocks in runs:
        stacked = None
        if n_dense and dense_matrices:
            stacked = np.stack(
                [
                    np.hstack(
                        [
                            float(lambdas[penalties[i].name])
                            * penalty_component_dense_matrix(
                                penalties[i], group_matrices[penalties[i].group_index]
                            )
                            for i in dense
                        ]
                    )
                    for dense, _ in blocks
                ]
            )
        batches.append(
            _PenaltyBatch(
                start=start,
                width=width,
                members=np.array([dense + identity for dense, identity in blocks], dtype=np.intp),
                dense=n_dense,
                stacked=stacked,
                identity_scales=np.array(
                    [
                        [float(lambdas[penalties[i].name]) for i in identity]
                        for _, identity in blocks
                    ],
                    dtype=np.float64,
                ).reshape(len(blocks), -1),
            )
        )
    return batches


def _centered_signed_moments(moments, row_weights: NDArray, mean_x: NDArray) -> NDArray:
    """``X_c' diag(row_weights) X_c`` from raw signed moments about ``mean_x``.

    The mass meets one mean first, ``(S m_j) m_k``, so no intermediate leaves
    float64's range where the raw moments do not, as ``m_j m_k`` could.
    """
    if moments.xtw is None:
        raise RuntimeError("centered signed moments omitted X'W1")
    sum_weights = float(np.sum(row_weights, dtype=np.float64))
    result = moments.gram
    result = result - np.outer(moments.xtw, mean_x)
    result = result - np.outer(mean_x, moments.xtw)
    result = result + np.outer(sum_weights * mean_x, mean_x)
    return 0.5 * (result + result.T)


def _centred_scale(geometry, pirls_result, sum_w: float | None, p: int) -> NDArray | None:
    """Per column ``sqrt(diag(X_c' W X_c) / sum_w)`` at the weights ``W`` the centre is the mean of.

    Read from the record ``mean_x`` comes from: the geometry's centred Gram,
    the rank record's ``column_scale`` or the geometry summary's scales.
    ``None`` without a profiled intercept or such a record.
    """
    if sum_w is None:
        return None
    if geometry is not None:
        gram = geometry.centered_data_gram
        diagonal = (
            np.diag(gram) if isinstance(gram, np.ndarray) else compact_operator_diagonal(gram)
        )
        scale = np.sqrt(np.abs(diagonal))
    elif pirls_result.rank_info is not None:
        record = pirls_result.rank_info
        selected = np.asarray(record.selected_columns, dtype=np.intp)
        if record.data.column_scale.shape != selected.shape:
            return None
        scale = np.zeros(p, dtype=np.float64)
        scale[selected] = record.data.column_scale
    elif pirls_result.reml_geometry is not None:
        carried = getattr(pirls_result.reml_geometry, "column_scale", None)
        if carried is None:
            return None
        scale = np.asarray(carried, dtype=np.float64)
    else:
        return None
    if scale.shape != (p,):
        return None
    with np.errstate(invalid="ignore", over="ignore"):
        scale = scale / np.sqrt(sum_w)
    return np.where(np.isfinite(scale), scale, 0.0)


@dataclass(frozen=True)
class _ColumnRepair:
    """The moment design's columns the raw-moment certificate rejected, recentred about the fixed centre.

    ``recentred`` and ``readers`` as ``recentred_about`` returns them,
    ``kept`` the admitted columns.  Owner: one ``reml_w_correction`` call,
    whose directions share ``formed``, the projections the readers form.
    """

    recentred: tuple
    readers: list
    kept: NDArray
    formed: dict

    def apply(
        self, group_matrices, gram: NDArray, row_weights: NDArray, centre: NDArray
    ) -> NDArray:
        """``gram``, raw-moment centred, with the recentred columns' rows and columns formed again."""
        recentred_gram_rows(
            group_matrices,
            self.readers,
            row_weights,
            gram,
            centre,
            self.kept,
            self.recentred,
            formed=self.formed,
        )
        return gram


def _split_pays(split: _DenseSplit) -> bool:
    """Whether a signed Gram costs less through the dense split than through centred rows.

    Centred rows cost one ``n p^2`` BLAS product per direction. The split
    costs the bounded design's raw moments (the block queue's estimate), the
    dense block's ``n p_d^2`` and the cross block's ``n p_d (1 + r)``, ``r``
    a group's row work; a BLAS multiply-add counts ``_P_SPACE_WEIGHT`` of a
    row visit. The split must estimate at most half the rows' cost, the margin
    its kernels' measured overrun on one core.
    """
    n = split.dense.n
    groups = split.bounded.group_matrices
    moments = sum(diagonal_cost(gm, n) for gm in groups) + sum(
        block_cost(left, right, n)
        for index, left in enumerate(groups)
        for right in groups[index + 1 :]
    )
    p_d, p = split.dense.p, split.dense.p + split.bounded.p
    cross = n * p_d * sum(1.0 + _side(gm, n)[0] for gm in groups)
    return 2.0 * (moments + cross + _P_SPACE_WEIGHT * n * p_d * p_d) < _P_SPACE_WEIGHT * n * p * p


def _dense_split_signed_gram(
    split: _DenseSplit,
    moments,
    row_weights: NDArray,
    centre_hi: NDArray,
    centre_lo: NDArray,
    repair: _ColumnRepair | None = None,
    formed: dict | None = None,
) -> NDArray:
    """``X_c' diag(a) X_c`` beside dense columns, ``moments`` the bounded design's raw signed moments.

    The IRLS Gram's split (``centered_system._attach_dense_split``): bounded columns from raw
    moments, centred after; dense rows centred as ``r = x - c`` about the exact pair ``(c, d)``,
    block ``G - e d' - d e' + d d' sum a``, cross ``N'(a r) - m_N e' - (N'a - m_N sum a) d'``.
    Linear in ``a``, and summation bounds hold for any sign (Higham 2002, eq. 3.5), so the
    corrected two-pass bound (Chan, Golub & LeVeque 1983) holds with ``|a|``: per entry
    ``gamma_(n+p+8) sum |a| (|x~_j| + |l_j|)(|x~_k| + |l_k|)``, ``x~`` a column's values (``r``
    if dense), ``l`` its centre or ``d``; a ``repair`` column enters by its centred support,
    ``x~ = c_j``, ``l_j = 0``.
    """
    dense_index, bounded_index = split.dense_index, split.bounded_index
    lo = centre_lo[dense_index]
    mean_bounded = centre_hi[bounded_index]
    anchor = centre_hi[dense_index]
    total = float(np.sum(row_weights, dtype=np.float64))
    first, gram, _ = anchored_dense_moments(split.dense, row_weights, anchor)
    bounded = _centered_signed_moments(moments, row_weights, mean_bounded)
    centred_xtw = moments.xtw - mean_bounded * total
    recentred: tuple = ()
    if repair is not None:
        bounded = repair.apply(split.bounded.group_matrices, bounded, row_weights, mean_bounded)
        recentred = repair.recentred
        for columns, products in recentred_products(recentred, row_weights):
            centred_xtw[columns] = products
    cross = dense_split_cross(split, row_weights, anchor, mean_bounded, first, recentred, formed)
    cross -= np.outer(centred_xtw, lo)
    return dense_split_gram(split, bounded, two_pass_centred_gram(first, gram, lo, total), cross)


def _missing_w_derivative_methods(link: Any, distribution: Any) -> list[str]:
    """Second-order methods the W(rho) correction needs and this pair lacks."""
    missing: list[str] = []
    if not hasattr(link, "deriv2_inverse"):
        missing.append(f"{type(link).__name__}.deriv2_inverse")
    if not hasattr(distribution, "variance_derivative"):
        missing.append(f"{type(distribution).__name__}.variance_derivative")
    return missing


def compute_dW_deta(
    link: Any,
    distribution: Any,
    mu: NDArray,
    eta: NDArray,
    sample_weight: NDArray,
) -> NDArray | None:
    """Derivative of IRLS weights w.r.t. the linear predictor.

    W_i = exposure_i * (dmu/deta)^2 / V(mu)

    dW_i/deta = exposure_i * (dmu/deta / V(mu)) * [2(d2mu/deta2) - (dmu/deta)^2 V'(mu)/V(mu)]

    For log link: dW/deta = W*(2 - mu V'(mu)/V(mu)).
    Poisson/log: dW/deta = W. Gamma/log: dW/deta = 0 identically.

    Returns None if the link or distribution does not provide the
    required second-order methods (deriv2_inverse, variance_derivative),
    which skips the W(rho) correction for custom objects.
    """
    if isinstance(distribution, Gamma) and isinstance(link, LogLink):
        return np.zeros_like(mu, dtype=np.float64)
    if _missing_w_derivative_methods(link, distribution):
        return None
    g1 = link.deriv_inverse(eta)  # dmu/deta
    g2 = link.deriv2_inverse(eta)  # d2mu/deta2
    V = np.maximum(distribution.variance(mu), _VARIANCE_FLOOR)
    Vp = distribution.variance_derivative(mu)
    return sample_weight * (g1 / V) * (2.0 * g2 - g1**2 * Vp / V)


def compute_d2W_deta2(
    link: Any,
    distribution: Any,
    mu: NDArray,
    eta: NDArray,
    sample_weight: NDArray,
) -> NDArray | None:
    """Second derivative of IRLS weights w.r.t. the linear predictor.

    Analytic formula from differentiating the Appendix D expression
    dW/deta = sw * (g1/V) * [2g2 - g1^2 Vp/V].

    Let A = g1/V,  B = 2g2 - g1^2 Vp/V.
    Then dW/deta = sw * A * B, and
    d2W/deta2 = sw * (A' B + A B').

    Requires ``link.deriv3_inverse`` (d3mu/deta3) and
    ``distribution.variance_second_derivative`` (V''(mu)).
    Falls back to central finite differences of ``compute_dW_deta``
    when those methods are absent.
    """
    if isinstance(distribution, Gamma) and isinstance(link, LogLink):
        return np.zeros_like(mu, dtype=np.float64)
    has_analytic = hasattr(link, "deriv3_inverse") and hasattr(
        distribution, "variance_second_derivative"
    )
    if has_analytic:
        return _compute_d2W_deta2_analytic(link, distribution, mu, eta, sample_weight)
    return _compute_d2W_deta2_fd(link, distribution, eta, sample_weight)


def _compute_d2W_deta2_analytic(
    link: Any,
    distribution: Any,
    mu: NDArray,
    eta: NDArray,
    sample_weight: NDArray,
) -> NDArray:
    """Analytic d2W/deta2 using third-order link and second-order variance."""
    g1 = link.deriv_inverse(eta)  # dmu/deta
    g2 = link.deriv2_inverse(eta)  # d2mu/deta2
    g3 = link.deriv3_inverse(eta)  # d3mu/deta3
    V = np.maximum(distribution.variance(mu), _VARIANCE_FLOOR)
    Vp = distribution.variance_derivative(mu)
    Vpp = distribution.variance_second_derivative(mu)

    # A = g1 / V
    # A' = dA/deta = (g2 V - g1 Vp g1) / V^2  = (g2 - g1^2 Vp/V) / V
    #    using chain rule: dV/deta = Vp * g1
    inv_V = 1.0 / V
    A = g1 * inv_V
    A_prime = (g2 - g1**2 * Vp * inv_V) * inv_V

    # B = 2g2 - g1^2 Vp / V
    # B' = dB/deta = 2g3 - d/deta[g1^2 Vp / V]
    #    d/deta[g1^2 Vp / V] = (2g1 g2 Vp + g1^2 Vpp g1) / V - g1^2 Vp * Vp g1 / V^2
    #                          = g1 (2g2 Vp + g1^2 Vpp) / V - g1^3 Vp^2 / V^2
    B = 2.0 * g2 - g1**2 * Vp * inv_V
    d_g1sq_Vp_over_V = g1 * (2.0 * g2 * Vp + g1**2 * Vpp) * inv_V - g1**3 * Vp**2 * inv_V**2
    B_prime = 2.0 * g3 - d_g1sq_Vp_over_V

    return sample_weight * (A_prime * B + A * B_prime)


def _compute_d2W_deta2_fd(
    link: Any,
    distribution: Any,
    eta: NDArray,
    sample_weight: NDArray,
) -> NDArray | None:
    """Finite-difference fallback for d2W/deta2.

    Central FD of ``compute_dW_deta``, used when the link or distribution
    does not provide ``deriv3_inverse`` or ``variance_second_derivative``.
    """
    eps = 1e-5
    mu_base = clip_mu(link.inverse(eta), distribution)
    dW_base = compute_dW_deta(link, distribution, mu_base, eta, sample_weight)
    if dW_base is None:
        return None

    eta_plus = eta + eps
    mu_plus = clip_mu(link.inverse(eta_plus), distribution)
    dW_plus = compute_dW_deta(link, distribution, mu_plus, eta_plus, sample_weight)

    eta_minus = eta - eps
    mu_minus = clip_mu(link.inverse(eta_minus), distribution)
    dW_minus = compute_dW_deta(link, distribution, mu_minus, eta_minus, sample_weight)

    if dW_plus is None or dW_minus is None:
        return None

    return (dW_plus - dW_minus) / (2.0 * eps)


def _qualified_class_name(obj: Any) -> str:
    """Module-qualified class identity, e.g. ``superglm.links.LogLink``.

    ``__qualname__`` rather than ``__name__`` so nested and locally defined
    classes stay distinguishable from a module-level class of the same name.
    """
    cls = type(obj)
    return f"{cls.__module__}.{cls.__qualname__}"


def _warn_w_correction_unavailable(link: Any, distribution: Any) -> None:
    """Report the capability gap that silently drops the W(rho) correction.

    Called only after ``compute_dW_deta`` returned None, so the missing list
    is never empty. The message opens with the module-qualified
    link/distribution pair because the default warning filter deduplicates on
    the message: one warning per pair, not one per REML iteration.
    """
    missing = _missing_w_derivative_methods(link, distribution)
    detail = f"{' and '.join(missing)} is not implemented"
    pair = f"{_qualified_class_name(link)}/{_qualified_class_name(distribution)}"
    warnings.warn(
        f"REML W(rho) correction skipped for {pair}: {detail}. "
        "Smoothing-parameter gradients omit the weight-derivative term, so "
        "REML may converge slowly or select slightly different smoothing "
        "parameters. Implement the missing method to restore the correction.",
        UserWarning,
        stacklevel=3,
    )


def reml_w_correction(
    dm: DesignMatrix,
    link: Any,
    groups: list[GroupSlice],
    pirls_result: PIRLSResult,
    XtWX_S_inv: NDArray | HessianFactor,
    lambdas: dict[str, float],
    reml_groups=None,
    penalty_caches: dict | None = None,
    sample_weight: NDArray | None = None,
    offset_arr: NDArray | None = None,
    distribution: Any = None,
    w_correction_order: int = 1,
    *,
    reml_penalties: list[PenaltyComponent] | None = None,
    geometry: ObservedREMLGeometry | None = None,
    leverage_hessian: bool = False,
    lookup_tables: bool = True,
    gradient_terms_only: bool = False,
) -> (
    tuple[NDArray, dict[int, NDArray | CompactSymmetricOperator]]
    | tuple[NDArray, dict[int, NDArray | CompactSymmetricOperator] | None, NDArray | None]
    | None
):
    """W(rho) correction for REML derivatives (first- or second-order).

    Implicit differentiation of beta_hat(rho) through W(eta(rho)), with
    ``dbeta_hat/drho_j = -H^{-1} S_j beta_hat`` (Wood 2011, sec. 3.4 and
    App. C). Adds ``X_c' diag(dW/drho_j) X_c`` and the profiled determinant
    ``log(sum(W))``, which the fixed-W Laplace approximation drops. The
    gradient correction is exact; second derivatives enter only with
    ``w_correction_order=2`` (Wood 2011, sec. 3.5.1), where in the
    profiled-intercept coordinates ``d2eta_jk = X_c d2beta_jk - dmean_x_k'
    dbeta_j`` and both weighted-mean outer products enter.

    Parameters
    ----------
    w_correction_order : int, default 1
        1 = first order only; 2 = add the second-order Hessian cross terms.
    geometry
        Observed-information rows and profiled-intercept metadata for Wood's
        LAML criterion; omitted, the Fisher working geometry is used.
        ``XtWX_S_inv`` must belong to the same geometry.
    leverage_hessian
        First order with a profiled intercept, on a dense or nested
        (``ProfiledNestedSchurFactor``) factor: one row pass
        (``_leverage_derivative_terms``) replaces the ``m`` derivative Grams
        and returns ``(grad_correction, None, w_cross)``, ``w_cross =
        -(P + P') / 2`` with ``P_ij = tr(H^-1 S_i H^-1 C_j)``. The
        ``tr(H^-1 C_i H^-1 C_j)`` term is left out, as order 1 leaves out
        second weight derivatives. Callers add ``w_cross`` as ``dH2_cross``.
        Otherwise the derivative-Gram result is returned.
    lookup_tables
        With ``leverage_hessian``, when every group is a lookup or dense
        column (``_lookup_tables``), or on the nested factor, the leverages
        are read back without solving rows and the result is
        ``(grad_correction, None, None)``. ``False`` forces the solved-row
        ``w_cross`` (the REML stop's second-order check needs it).
    gradient_terms_only
        With ``leverage_hessian``, return ``(grad_correction, None, None)``
        from the solved rows, forming no ``P_ij``.

    Returns
    -------
    tuple or None
        ``(grad_correction, dH_extra)`` at order 1, ``(grad_correction,
        dH_extra, dH2_cross)`` at order 2, ``dH2_cross`` the ``(m, m)``
        second-order corrections, including the curvature of
        ``0.5 * log(sum(W))``. None when the correction vanishes or the
        link/distribution lacks the required methods.
    """
    w_correction_order = validate_w_correction_order(w_correction_order)
    if leverage_hessian and w_correction_order != 1:
        raise ValueError("leverage_hessian requires w_correction_order=1")
    penalties = coerce_reml_penalties(
        reml_groups=reml_groups,
        reml_penalties=reml_penalties,
        group_matrices=dm.group_matrices,
        penalty_caches=penalty_caches,
    )
    if geometry is None:
        eta = stabilize_eta(linear_predictor(dm, pirls_result, offset_arr), link)
        mu = clip_mu(link.inverse(eta), distribution)
        dW_deta = compute_dW_deta(link, distribution, mu, eta, sample_weight)
    else:
        if geometry.eta.shape != (dm.n,) or geometry.mu.shape != (dm.n,):
            raise ValueError("observed REML geometry does not match the design rows")
        if geometry.hessian_inverse is None:
            raise ValueError("observed REML geometry omitted the slope inverse")
        if geometry.weight_derivative is None:
            raise ValueError("observed REML geometry omitted first weight derivatives")
        eta = geometry.eta
        mu = geometry.mu
        dW_deta = geometry.weight_derivative
        # Geometry is one determinant/inverse/centering payload: a separate
        # Fisher inverse would differentiate no coherent LAML criterion.
        XtWX_S_inv = geometry.hessian_inverse

    if dW_deta is None:
        # Custom link/distribution w/o 2nd-order.  Distinct from the
        # structurally-zero branch below, which is silent by design.
        _warn_w_correction_unavailable(link, distribution)
        return None

    if not np.any(dW_deta):
        return None  # Structurally constant working curvature.

    factor = as_hessian_factor(XtWX_S_inv)
    p = factor.shape[0]
    m = len(penalties)
    grad_correction = np.zeros(m)
    dH_extra: dict[int, NDArray | CompactSymmetricOperator] = {}

    gms = dm.group_matrices
    if geometry is not None:
        mean_x = np.asarray(geometry.mean_x, dtype=np.float64)
        sum_w: float | None = float(geometry.sum_w)
        if mean_x.shape != (p,):
            raise ValueError("observed REML geometry does not match the coefficient space")
    elif pirls_result.rank_info is not None:
        mean_x = np.asarray(pirls_result.rank_info.mean_x, dtype=np.float64)
        if mean_x.shape != (p,):
            raise ValueError("PIRLS rank metadata does not match the REML coefficient space")
        sum_w = float(pirls_result.rank_info.sum_w)
        if not np.isfinite(sum_w) or sum_w <= 0.0:
            raise ValueError("PIRLS rank metadata has an invalid working-weight sum")
    elif pirls_result.reml_geometry is not None:
        # In-loop fits skip rank metadata; the geometry summary carries the
        # same centered moments.
        summary = pirls_result.reml_geometry
        mean_x = np.asarray(summary.mean_x, dtype=np.float64)
        if mean_x.shape != (p,):
            raise ValueError("PIRLS geometry summary does not match the REML coefficient space")
        sum_w = float(summary.sum_w)
        if not np.isfinite(sum_w) or sum_w <= 0.0:
            raise ValueError("PIRLS geometry summary has an invalid working-weight sum")
    else:
        factor_mean = getattr(factor, "mean_x", None)
        factor_sum_w = getattr(factor, "sum_w", None)
        if factor_mean is None or factor_sum_w is None:
            mean_x = np.zeros(p)
            sum_w = None
        else:
            mean_x = np.asarray(factor_mean, dtype=np.float64)
            sum_w = float(factor_sum_w)
    centred_scale = _centred_scale(geometry, pirls_result, sum_w, p)

    def centre_weights() -> NDArray:
        """The weights ``mean_x`` is the mean of: the geometry's, else the Fisher weights at the mode."""
        if geometry is not None:
            return np.asarray(geometry.weights, dtype=np.float64)
        return fisher_working_weights(
            distribution=distribution,
            link=link,
            mu=mu,
            eta=eta,
            sample_weight=np.ones(dm.n) if sample_weight is None else np.asarray(sample_weight),
        )

    # Every dense column is centred about an exact pair ``(x - c) - d``
    # (``centered_system.dense_mean_pair``, the corrected two-pass algorithm),
    # since a rounded centre alone cancels ``c' dbeta`` at a large offset.
    # Without a profiled intercept (``sum_w`` None) nothing is centred.
    dense = dense_columns(dm)
    centre_hi = mean_x
    centre_lo: NDArray | None = None
    # Without a geometry, read the final system's pair from the summary
    # (``irls_direct.centered_final``); older pickled summaries carry none.
    summary = pirls_result.reml_geometry if geometry is None else None
    carried_hi = None if summary is None else getattr(summary, "mean_hi", None)
    carried_lo = None if summary is None else getattr(summary, "mean_lo", None)
    if carried_hi is not None and carried_lo is not None and sum_w is not None and np.any(dense):
        centre_hi = np.where(dense, np.asarray(carried_hi, dtype=np.float64), mean_x)
        centre_lo = np.where(dense, np.asarray(carried_lo, dtype=np.float64), 0.0)
    elif sum_w is not None and np.any(dense):
        pair = dense_mean_pair(dm, centre_weights(), sum_w)
        if pair is None:  # pragma: no cover - a dense column is present
            raise RuntimeError("A dense column formed no centre pair.")
        centre_hi = np.where(dense, pair[0], mean_x)
        centre_lo = np.where(dense, pair[1], 0.0)

    def centered_matvec(values: NDArray) -> NDArray:
        """Apply the profiled-intercept design ``X - 1 mean_x'``."""
        rest = values if centre_lo is None else np.where(dense, 0.0, values)
        design_values = (
            dm.matvec(rest)
            if structured_layout is None
            else structured_design_matvec(
                structured_layout,
                dm.group_matrices,
                rest,
            )
        )
        product = design_values - float(mean_x @ rest)
        if centre_lo is None:
            return product
        return product + dense_centred_matvec(dm, values, centre_hi, centre_lo)

    def centered_rmatvec(values: NDArray) -> NDArray:
        """Apply the transpose of the profiled-intercept design."""
        transpose_values = (
            dm.rmatvec(values)
            if structured_layout is None
            else structured_design_rmatvec(
                structured_layout,
                dm.group_matrices,
                values,
            )
        )
        total = float(np.sum(values, dtype=np.float64))
        product = transpose_values - mean_x * total
        if centre_lo is None:
            return product
        return np.where(dense, dense_centred_rmatvec(dm, values, centre_hi, centre_lo), product)

    structured_group_index: int | None = None
    structured_layout = None
    signed_leaf_mean: NDArray | None = None
    signed_leaf_center: NDArray | None = None
    signed_center = mean_x
    if isinstance(factor, ProfiledNestedSchurFactor):
        # A nested factor is found by its chain, never its leaf name; its
        # signed operators are centred on its own leaf means and mean_x.
        structured_group_index = factor.chain_group_indices[-1]
        structured_layout = get_nested_structured_layout(
            dm,
            groups,
            chain_group_indices=factor.chain_group_indices,
        )
        signed_leaf_mean = factor.data_operator.leaf.mean
        signed_leaf_center = factor.data_operator.leaf.center
        signed_center = factor.mean_x
    elif not isinstance(factor, DenseHessianFactor):
        dominant_name = getattr(factor, "dominant_group_name", None)
        structured_group_index = next(
            (index for index, group in enumerate(groups) if group.name == dominant_name),
            None,
        )
        if structured_group_index is None:
            raise ValueError("Structured Hessian factor has no matching dominant group.")
        structured_layout = get_structured_layout(
            dm,
            groups,
            dominant_group_index=structured_group_index,
        )

    leverage_route = leverage_hessian and sum_w is not None
    if leverage_route and isinstance(factor, DenseHessianFactor):
        penalty_rhs = _penalty_rhs(penalties, lambdas, pirls_result.beta, gms, p)
        plan = lookup_plan(dm) if lookup_tables else None
        if plan is not None:
            # Every group a lookup or a dense column: leverages read back from
            # term-pair tables, the gradient exact, no per-row cross terms.
            grad_correction[:] = _leverage_table_gradient(
                plan, factor, centre_hi, centre_lo, dW_deta, sum_w, penalty_rhs
            )
            return grad_correction, None, None
        cross_terms = not (gradient_terms_only and lookup_tables)
        grad_correction[:], penalty_weight_cross = _leverage_derivative_terms(
            dm,
            factor.inverse,
            centre_hi,
            dW_deta,
            sum_w,
            _leverage_cross_batches(penalties, lambdas, gms, dense_matrices=cross_terms),
            penalty_rhs,
            mean_lo=centre_lo,
            cross_terms=cross_terms,
        )
        if penalty_weight_cross is None:
            return grad_correction, None, None
        return grad_correction, None, -0.5 * (penalty_weight_cross + penalty_weight_cross.T)

    # The nested factor's leverage route: its augmented factor gives each row's
    # ``1/sum_w + x_c' H_c^-1 x_c`` (Bates et al. 2015, eqs. 63-65), so the row
    # channel ``w'(h + 1/sum_w) / 2`` forms the exact gradient.
    structured_leverage = leverage_route and isinstance(factor, ProfiledNestedSchurFactor)
    if structured_leverage and lookup_tables:
        leverage = grouped_augmented_row_quadratic_forms(
            dm, factor.augmented_factor.row_quadratic_forms, active=dW_deta != 0.0
        )
        penalty_rhs = _penalty_rhs(penalties, lambdas, pirls_result.beta, gms, p)
        grad_correction[:] = centered_rmatvec(0.5 * dW_deta * leverage) @ -factor.solve(penalty_rhs)
        return grad_correction, None, None

    # Beside a dense column the Grams take the split where ``_split_pays``,
    # else the rows centred about the pair.
    split = _dense_split(dm) if centre_lo is not None and structured_group_index is None else None
    stable_rows = split is not None and not _split_pays(split)
    if stable_rows:
        split = None
    elif split is not None:
        release_for_serial_row_passes(split.dense.p)  # the dense block, n p_d^2 per direction
    moment_dm = dm if split is None else split.bounded
    # Raw moments are certified per column as the IRLS Gram's are: with
    # ``|m_j| <= s_j`` and ``|a| <= rho W`` an entry lies within
    # ``6 gamma rho S s_j s_k``, the centred rows' bound up to the constant.
    # A rejected column is recentred (``recentred_about``), else centred rows.
    repair: _ColumnRepair | None = None
    # the bounded supports the cross blocks project, formed once for every direction
    cross_supports: dict = {}
    if structured_group_index is None and sum_w is not None and not stable_rows:
        moment_centre = mean_x if split is None else centre_hi[split.bounded_index]
        admitted = (
            moment_centre == 0.0
            if centred_scale is None
            else _raw_centering_admitted(
                moment_centre,
                centred_scale if split is None else centred_scale[split.bounded_index],
            )
        )
        if not admitted.all():
            recentred = recentred_about(moment_dm, admitted, centre_weights(), sum_w, moment_centre)
            if recentred is None:
                stable_rows = True
            else:
                repair = _ColumnRepair(*recentred, np.flatnonzero(admitted), {})
    batch_moments = (
        type(dm) is DesignMatrix
        and type(factor) is DenseHessianFactor
        and w_correction_order == 1
        and m > 1
        and not stable_rows
    )
    # Factors cannot change between these first-order directions.
    fixed_support = moment_dm.execution_plan._fixed_support_cache() if batch_moments else None
    # Batch only where no direction keeps its own support projections, which
    # the memory budget below does not count.
    batch_moments = batch_moments and (
        fixed_support is not None or moment_dm.execution_plan._projects_no_support()
    )
    batch_size = 0
    if stable_rows and isinstance(factor, DenseHessianFactor) and w_correction_order == 1:
        # Retained weights, Gram accumulators, compensation and outputs.
        bytes_per_direction = np.dtype(np.float64).itemsize * (dm.n + 3 * p * p)
        batch_size = min(m, _SIGNED_GRAM_BATCH_BYTES // max(1, bytes_per_direction))
        if batch_size < 2:
            batch_size = 0
    elif batch_moments:
        # Retained weights and their stacked copy, raw moments and outputs.
        bytes_per_direction = np.dtype(np.float64).itemsize * (2 * dm.n + 3 * p * p)
        batch_size = min(m, _SIGNED_GRAM_BATCH_BYTES // max(1, bytes_per_direction))
        if batch_size < 2:
            batch_size = 0
    pending: list[tuple[int, NDArray, float]] = []
    fs_pending: list[tuple[int, NDArray]] = []
    fs_layout = structured_layout if isinstance(structured_layout, FactorSmoothLeafLayout) else None

    def signed_gram(moments, row_weights: NDArray) -> NDArray:
        """``X_c' diag(row_weights) X_c`` from ``moment_dm``'s raw signed moments."""
        if split is None:
            gram = _centered_signed_moments(moments, row_weights, mean_x)
            if repair is not None:
                repair.apply(dm.group_matrices, gram, row_weights, mean_x)
            return gram
        # a split is formed only beside the pair
        lo = cast(NDArray, centre_lo)
        return _dense_split_signed_gram(
            split, moments, row_weights, centre_hi, lo, repair, cross_supports
        )

    def flush_signed_grams() -> None:
        weights = [row_weights for _, row_weights, _ in pending]
        if stable_rows:
            grams = centered_signed_grams(
                dm=dm, weights=weights, mean_x=centre_hi, mean_lo=centre_lo
            )
        else:
            moments = moment_dm.execution_plan._signed_moments_channels(weights, fixed_support)
            grams = [
                signed_gram(channel, row_weights)
                for channel, row_weights in zip(moments, weights, strict=True)
            ]
        for (i, _, dsum_w_j), C_j in zip(pending, grams, strict=True):
            grad_correction[i] = 0.5 * float(np.sum(factor.inverse * C_j))
            if sum_w is not None:
                grad_correction[i] += 0.5 * dsum_w_j / sum_w
            dH_extra[i] = C_j
        pending.clear()

    def centered_signed_gram(
        row_weights: NDArray,
    ) -> NDArray | CompactSymmetricOperator:
        """Return ``X_c' diag(row_weights) X_c`` for fixed ``mean_x``."""
        if structured_group_index is not None:
            system = (
                build_nested_structured_system(
                    list(dm.group_matrices),
                    groups,
                    row_weights,
                    None,
                    layout=structured_layout,
                    mean=signed_leaf_mean,
                    center=signed_leaf_center,
                )
                if isinstance(structured_layout, NestedStructuredLayout)
                # A FactorSmooth term's weight-derivative operators only enter
                # traces: the moment kernel, never the leaf factorization.
                else build_block_structured_system(
                    list(dm.group_matrices),
                    groups,
                    row_weights,
                    np.zeros_like(row_weights),
                    dominant_group_index=structured_group_index,
                    layout=structured_layout,
                )
            )
            cross = np.empty(p, dtype=np.float64)
            cross[system.operator.small_indices] = system.xtw_small
            cross[system.operator.structured_indices] = system.xtw_structured
            return CenteredBlockOperator(
                raw=system.operator,
                cross=cross,
                total=system.sum_w,
                center=signed_center,
            )
        if stable_rows:
            gram, _ = centered_gram_rhs(
                dm=dm,
                W=row_weights,
                mean_x=centre_hi,
                z_centered=np.zeros(dm.n),
                mean_lo=centre_lo,
            )
            return gram
        moments = (
            moment_dm.execution_plan.moments(row_weights, include_xtw=True, signed=True)
            if fixed_support is None
            else moment_dm.execution_plan._signed_moments_fixed_support(row_weights, fixed_support)
        )
        return signed_gram(moments, row_weights)

    # Pre-compute d2W/deta2 for second-order path
    d2W_deta2: NDArray | None = None
    if w_correction_order >= 2:
        if geometry is not None:
            d2W_deta2 = geometry.weight_second_derivative
            if d2W_deta2 is None:
                raise ValueError("observed REML geometry omitted second weight derivatives")
        else:
            d2W_deta2 = compute_d2W_deta2(link, distribution, mu, eta, sample_weight)

    # Store per-group quantities for second-order cross-terms
    deta_vectors: list[NDArray] = []
    dbeta_vectors: list[NDArray] = []
    dmean_vectors: list[NDArray] = []
    dsum_w_values: list[float] = []
    lam_list: list[float] = []

    for i, pc in enumerate(penalties):
        gm = gms[pc.group_index]
        lam = lambdas[pc.name]
        beta_g = pirls_result.beta[pc.group_sl]

        # S_j beta (p-vector, nonzero only in pc.group_sl block)
        s_beta = np.zeros(p)
        s_beta[pc.group_sl] = lam * penalty_component_matvec(pc, beta_g, gm)

        # dbeta/drho_j = -H^{-1} S_j beta  (IFT)
        dbeta_j = -factor.solve(s_beta)

        # The intercept is unpenalized and profiled out, so
        # dalpha/drho_j = -mean_x' dbeta/drho_j and hence
        # deta/drho_j = (X - 1 mean_x') dbeta/drho_j.
        deta_j = centered_matvec(dbeta_j)

        # a_j = (dW/deta) * deta_j  -- weight changes per obs
        a_j = dW_deta * deta_j

        if batch_size:
            pending.append((i, a_j, float(np.sum(a_j, dtype=np.float64))))
            if len(pending) == batch_size:
                flush_signed_grams()
            continue

        if fs_layout is not None:
            # every direction's operator from one multi-weight moment pass (below)
            fs_pending.append((i, a_j))
        else:
            # C_j = X_c'diag(a_j)X_c -- dW contribution to the
            # profiled-intercept Hessian.
            C_j = centered_signed_gram(a_j)

            # Profiled-determinant gradient: centered-Hessian trace plus the
            # scalar 0.5 * log(sum(W)) derivative below.
            if isinstance(C_j, np.ndarray):
                if not isinstance(factor, DenseHessianFactor):  # pragma: no cover - invariant
                    raise RuntimeError("Structured derivative Gram unexpectedly materialized.")
                grad_correction[i] = 0.5 * float(np.sum(factor.inverse * C_j))
            else:
                grad_correction[i] = 0.5 * factor.trace_inverse_operator(C_j)
            dH_extra[i] = C_j
        dsum_w_j = float(np.sum(a_j, dtype=np.float64))
        if sum_w is not None:
            # The fitted determinant is log(sum(W)) + log|H_c| after
            # profiling the intercept, so its scalar factor varies with W.
            grad_correction[i] += 0.5 * dsum_w_j / sum_w

        if w_correction_order >= 2:
            # Mean-derivative outer products enter only at second order;
            # the first derivative has no such term because X_c'W1=0.
            dmean_j = centered_rmatvec(a_j) / sum_w if sum_w is not None else np.zeros_like(mean_x)
            deta_vectors.append(deta_j)
            dbeta_vectors.append(dbeta_j)
            dmean_vectors.append(dmean_j)
            dsum_w_values.append(dsum_w_j)
            lam_list.append(lam)

    if pending:
        flush_signed_grams()
    if fs_pending and fs_layout is not None:
        # One row pass forms every direction's moments, on rows shifted by the
        # fit's c0 and centred on its shifted mean, so a large-offset column
        # does not cancel to noise (the centring is shift-invariant).
        if not isinstance(factor, ProfiledFactorSmoothLeafFactor | ProfiledSumToZeroTreeFactor):
            raise RuntimeError("An fs or sz weight derivative needs its profiled leaf factor.")
        fit_system = factor.augmented_factor.system
        fit_centre = centred_data_operator(fit_system).center
        # An sz operator also carries its per-level cross, which its factor's
        # traces read (``ProfiledSumToZeroTreeFactor._trace_form``).
        moment_sets = factor_smooth_moment_operators(
            fs_layout, [a for _, a in fs_pending], center=fit_system.leaf.center, level_cross=True
        )
        for (i, _), (raw, cross, total, level) in zip(fs_pending, moment_sets, strict=True):
            C_j = CenteredBlockOperator(
                raw=raw, cross=cross, total=total, center=fit_centre, raw_structured_cross=level
            )
            grad_correction[i] += 0.5 * factor.trace_inverse_operator(C_j)
            dH_extra[i] = C_j

    # Second-order Hessian cross terms (Wood 2011, sec. 3.5.1): m^2/2 signed
    # centred Grams plus m^2/2 matvec/rmatvec pairs.
    dH2_cross: NDArray | None = None
    if w_correction_order >= 2 and d2W_deta2 is not None:
        dH2_cross = np.zeros((m, m))
        for i in range(m):
            pc_i = penalties[i]
            for j in range(i, m):
                pc_j = penalties[j]

                # Differentiating H dbeta_i = -S_i beta contributes
                # X_c'[(dW/deta) deta_j deta_i] through dH_j dbeta_i.
                f_jk = deta_vectors[i] * deta_vectors[j] * dW_deta

                # X_c^T f^{jk}
                Xt_f = centered_rmatvec(f_jk)

                # lam_i S_i dbeta/drho_j  (nonzero in pc_i block)
                lam_i_S_i_dbeta_j = np.zeros(p)
                lam_i_S_i_dbeta_j[pc_i.group_sl] = lam_list[i] * penalty_component_matvec(
                    pc_i,
                    dbeta_vectors[j][pc_i.group_sl],
                    gms[pc_i.group_index],
                )

                # lam_j S_j dbeta/drho_i  (nonzero in pc_j block)
                lam_j_S_j_dbeta_i = np.zeros(p)
                lam_j_S_j_dbeta_i[pc_j.group_sl] = lam_list[j] * penalty_component_matvec(
                    pc_j,
                    dbeta_vectors[i][pc_j.group_sl],
                    gms[pc_j.group_index],
                )

                # rhs = X_c^T f^{jk} + lam_i S_i dbeta_j + lam_j S_j dbeta_i
                rhs = Xt_f + lam_i_S_i_dbeta_j + lam_j_S_j_dbeta_i

                # d2beta_hat/(drho_i drho_j) = delta_ij * dbeta_hat/drho_j - H^{-1} rhs
                d2beta_ij = -factor.solve(rhs)
                if i == j:
                    d2beta_ij += dbeta_vectors[j]

                # Differentiating eta_i = X_c dbeta_i also differentiates
                # the weighted center: d2eta_ij = X_c d2beta_ij - dm_j' dbeta_i.
                d2eta_ij = centered_matvec(d2beta_ij) - float(dmean_vectors[j] @ dbeta_vectors[i])

                # Full d2w/(drho_i drho_j) (Section 3.5.1 T_{jk} derivation):
                d2w_drho_ij = d2W_deta2 * deta_vectors[i] * deta_vectors[j] + dW_deta * d2eta_ij

                # Differentiate X_c' diag(dw_i) X_c: the centered d2w Gram
                # plus the two negative weighted-mean outer products, the mass
                # on one mean first (``_centered_signed_moments``).
                C_ij = centered_signed_gram(d2w_drho_ij)
                if sum_w is not None:
                    if isinstance(C_ij, np.ndarray):
                        C_ij -= np.outer(sum_w * dmean_vectors[i], dmean_vectors[j]) + np.outer(
                            sum_w * dmean_vectors[j], dmean_vectors[i]
                        )
                    else:
                        C_ij = SumBlockOperator(
                            (
                                C_ij,
                                LowRankSymmetricOperator(
                                    basis=np.column_stack((dmean_vectors[i], dmean_vectors[j])),
                                    core=np.array(
                                        [
                                            [0.0, -sum_w],
                                            [-sum_w, 0.0],
                                        ]
                                    ),
                                ),
                            )
                        )
                if isinstance(C_ij, np.ndarray):
                    if not isinstance(
                        factor, DenseHessianFactor
                    ):  # pragma: no cover - branch invariant
                        raise RuntimeError(
                            "Structured second derivative Gram unexpectedly materialized."
                        )
                    val = 0.5 * float(np.sum(factor.inverse * C_ij))
                else:
                    val = 0.5 * factor.trace_inverse_operator(C_ij)
                if sum_w is not None:
                    d2sum_w_ij = float(np.sum(d2w_drho_ij, dtype=np.float64))
                    val += 0.5 * (
                        d2sum_w_ij / sum_w - (dsum_w_values[i] / sum_w) * (dsum_w_values[j] / sum_w)
                    )
                dH2_cross[i, j] = val
                dH2_cross[j, i] = val

    if structured_leverage:
        return grad_correction, None, _structured_weight_cross(factor, penalties, lambdas, dH_extra)
    if w_correction_order >= 2:
        return grad_correction, dH_extra, dH2_cross
    return grad_correction, dH_extra


def validate_w_correction_order(value: object) -> int:
    """Return a supported Wood derivative order or fail before fitting."""
    if type(value) is not int or value not in (1, 2):
        raise ValueError("w_correction_order must be 1 or 2")
    return value
