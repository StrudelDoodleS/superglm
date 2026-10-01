"""Rank-aware estimability geometry for compact structured operators."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
from numpy.typing import NDArray

from superglm.solvers._structured.operators import (
    BlockSymmetricOperator,
    CenteredBlockOperator,
    SymmetricBlockOperator,
    compact_operator_diagonal,
    materialize_compact_operator,
)
from superglm.solvers.rank import (
    SHARED_RANK_POLICY,
    RankDecomposition,
    decompose_gram,
    needs_factor_certification,
)

if TYPE_CHECKING:
    from superglm.solvers._structured.nested import NestedDataOperator

_MAX_DENSE_CENTERED_ESTIMABILITY_WIDTH = 512


def _bounded_centered_estimability(operator: CenteredBlockOperator) -> NDArray:
    """Use exact dense rank only below a fixed inference-memory bound."""
    if operator.shape[0] > _MAX_DENSE_CENTERED_ESTIMABILITY_WIDTH:
        # Large constrained systems with deficient local data can have a
        # border as wide as the structured term. Refusing to claim any
        # individual coordinate is safer than allocating a global p-by-p Gram.
        return np.zeros(operator.shape[0], dtype=bool)
    return decompose_gram(materialize_compact_operator(operator)).coefficient_estimable()


def _augmented_small_data_block(operator: CenteredBlockOperator) -> NDArray:
    """Return the intercept-plus-small raw data Gram."""
    raw = operator.raw
    cross_small = operator.cross[raw.small_indices]
    q = len(raw.small_indices)
    augmented = np.empty((q + 1, q + 1), dtype=np.float64)
    augmented[0, 0] = operator.total
    augmented[0, 1:] = cross_small
    augmented[1:, 0] = cross_small
    augmented[1:, 1:] = raw.A
    return augmented


def _moment_column_squared_norms(
    operator: CenteredBlockOperator,
) -> tuple[NDArray, NDArray, NDArray]:
    """Centred squared column norms from the moments, their round-off bounds, and
    the columns whose norm cancels to within its bound."""
    raw_diagonal = compact_operator_diagonal(operator.raw)
    centered_diagonal = compact_operator_diagonal(operator)
    roundoff_bound = (
        SHARED_RANK_POLICY.certification_band
        * np.finfo(np.float64).eps
        * (
            np.abs(raw_diagonal)
            + 2.0 * np.abs(operator.cross * operator.center)
            + abs(operator.total) * operator.center**2
        )
    )
    cancellation_limited = (raw_diagonal > 0.0) & (np.abs(centered_diagonal) <= roundoff_bound)
    return centered_diagonal, roundoff_bound, cancellation_limited


def _centered_operator_column_scale(operator: CenteredBlockOperator) -> NDArray:
    """Return cancellation-certified centered public-design column norms.

    A column whose moments cancel takes its norm from the rows when the terminal
    build supplied it (``operator.row_column_norm``), and its round-off bound
    otherwise.  The bound keeps a large-mean column such as ``1e9 + i``
    estimable, but it also invents a scale for a truly constant column, which
    can lift another coefficient's rounding-level null entry above
    ``factor_rcond`` and mark it non-estimable.
    """
    centered_diagonal, roundoff_bound, limited = _moment_column_squared_norms(operator)
    centered_diagonal[limited] = roundoff_bound[limited]
    scale = np.sqrt(np.maximum(centered_diagonal, 0.0))
    if operator.row_column_norm is None:
        return scale
    return np.where(np.isnan(operator.row_column_norm), scale, operator.row_column_norm)


def cancelled_column_row_norms(
    operator: CenteredBlockOperator,
    design,
    weights: NDArray,
) -> NDArray:
    """Centred norms, from the rows, of the columns whose moments cancel; NaN elsewhere.

    ``design`` applies the public design (``matvec``) and ``weights`` are the
    rows' weights in ``operator``.  Each such column, usually none, is one O(n)
    shifted two-pass: about the first weighted row, which leaves a constant
    column exactly zero, then about the weighted mean of the shifted values
    (Chan, Golub & LeVeque 1983).
    """
    width = operator.shape[0]
    norms = np.full(width, np.nan)
    first = np.flatnonzero(weights)[0]
    for index in np.flatnonzero(_moment_column_squared_norms(operator)[2]):
        unit = np.zeros(width)
        unit[index] = 1.0
        column = np.asarray(design.matvec(unit), dtype=np.float64)
        shifted = column - column[first]
        mean = float(weights @ shifted) / float(np.sum(weights))
        norms[index] = np.sqrt(float(weights @ (shifted - mean) ** 2))
    return norms


def _lifted_null_row_norms(
    small_null: NDArray,
    structured_lift: NDArray,
    *,
    small_column_scale: NDArray,
    structured_column_scale: NDArray,
) -> tuple[NDArray, NDArray]:
    """Return lifted-null leverage in centered design-column coordinates."""
    if small_null.shape[1] == 0:
        return (
            np.zeros(small_null.shape[0], dtype=np.float64),
            np.zeros(structured_lift.shape[:-1], dtype=np.float64),
        )

    small_scale = np.asarray(small_column_scale, dtype=np.float64)
    structured_scale = np.asarray(structured_column_scale, dtype=np.float64)
    if small_scale.shape != small_null.shape[:1]:
        raise ValueError("small lifted-null scale does not match its rows")
    if structured_scale.shape != structured_lift.shape[:2]:
        raise ValueError("structured lifted-null scale does not match its rows")

    # Parameter-null leverage is defined after multiplying each parameter row
    # by its centered design-column norm. A reduced-Schur basis is also free to
    # scale its columns independently, so equilibrate those lifted directions
    # before the Gram cutoff as decompose_factor does.
    scaled_null_gram = small_null.T @ ((small_scale**2)[:, None] * small_null)
    scaled_null_gram += np.einsum(
        "kir,ki,kis->rs",
        structured_lift,
        structured_scale**2,
        structured_lift,
        optimize=True,
    )
    raw_squared_norm = np.sum(small_null * small_null, axis=0)
    raw_squared_norm += np.einsum(
        "kir,kir->r",
        structured_lift,
        structured_lift,
        optimize=True,
    )
    active_raw_squared_norm = np.sum(
        small_null[small_scale > 0.0] ** 2,
        axis=0,
    )
    active_raw_squared_norm += np.einsum(
        "kir,ki,kir->r",
        structured_lift,
        structured_scale > 0.0,
        structured_lift,
        optimize=True,
    )
    squared_column_norm = np.maximum(np.diag(scaled_null_gram), 0.0)
    meaningful_support = np.sqrt(active_raw_squared_norm) > (
        SHARED_RANK_POLICY.factor_rcond * np.sqrt(raw_squared_norm)
    )
    active = (squared_column_norm > 0.0) & meaningful_support
    if not np.any(active):
        return (
            np.zeros(small_null.shape[0], dtype=np.float64),
            np.zeros(structured_lift.shape[:-1], dtype=np.float64),
        )
    column_scale = np.sqrt(squared_column_norm[active])
    null_gram = scaled_null_gram[np.ix_(active, active)] / np.outer(
        column_scale,
        column_scale,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (null_gram + null_gram.T))
    scale = max(float(np.max(eigenvalues, initial=0.0)), 1.0)
    positive = eigenvalues > SHARED_RANK_POLICY.gram_rcond * scale
    if not np.any(positive):
        return (
            np.zeros(small_null.shape[0], dtype=np.float64),
            np.zeros(structured_lift.shape[:-1], dtype=np.float64),
        )
    whitening = (eigenvectors[:, positive] / np.sqrt(eigenvalues[positive])) @ (
        eigenvectors[:, positive].T
    )
    lifted_transform = np.zeros(
        (small_null.shape[1], whitening.shape[1]),
        dtype=np.float64,
    )
    lifted_transform[active] = whitening / column_scale[:, None]
    orthogonal_small = (small_null @ lifted_transform) * small_scale[:, None]
    orthogonal_structured = (structured_lift @ lifted_transform) * structured_scale[:, :, None]
    return (
        np.linalg.norm(orthogonal_small, axis=1),
        np.linalg.norm(orthogonal_structured, axis=2),
    )


def _independent_block_centered_estimability(
    operator: CenteredBlockOperator,
) -> NDArray:
    """Exact compact centered-data estimability for RE and FS blocks."""
    raw = operator.raw
    if isinstance(raw, SymmetricBlockOperator):
        D = raw.d[:, None, None]
        C = raw.C[:, None, :]
        structured_indices = raw.structured_indices[:, None]
    elif isinstance(raw, BlockSymmetricOperator):
        D = raw.D
        C = raw.C
        structured_indices = raw.structured_indices
    else:  # pragma: no cover - caller dispatch
        raise TypeError("independent block rank requires RE or FS geometry")

    q = len(raw.small_indices)
    # A column whose norm from the rows is exactly zero is constant on the
    # weighted rows, the intercept's exact alias: non-estimable, and its null
    # direction has no other slope entry, so the rest is decided without it.
    # Kept, its rounding-level null direction is all a zero scale leaves.
    kept = np.ones(q + 1, dtype=bool)
    if operator.row_column_norm is not None:
        kept[1:] = operator.row_column_norm[raw.small_indices] != 0.0
    C_augmented = np.empty((D.shape[0], D.shape[1], q + 1), dtype=np.float64)
    C_augmented[:, :, 0] = operator.cross[structured_indices]
    C_augmented[:, :, 1:] = C
    C_augmented = C_augmented[:, :, kept]
    local_estimable = np.empty(D.shape[:2], dtype=bool)
    inverses = np.empty(D.shape, dtype=np.float64)
    for level, block in enumerate(D):
        decomposition = decompose_gram(block)
        null_basis = _certified_local_null_basis(block, decomposition)
        inverses[level], _null_projector = _local_range_inverse_and_null_projector(
            block,
            decomposition,
            null_basis=null_basis,
        )
        local_estimable[level] = _coefficient_estimable_from_scaled_null_basis(
            null_basis,
            decomposition.column_scale,
        )

    structured_map, kkt_residual = _refined_local_solves(D, inverses, C_augmented)
    kept_null = _certified_reduced_schur_null_basis(
        source=_augmented_small_data_block(operator)[np.ix_(kept, kept)],
        structured_cross=C_augmented,
        structured_map=structured_map,
        kkt_residual=kkt_residual,
    )
    structured_lift = np.einsum(
        "kiq,qr->kir",
        structured_map,
        kept_null,
        optimize=True,
    )
    small_null = np.zeros((q + 1, kept_null.shape[1]), dtype=np.float64)
    small_null[kept] = kept_null
    public_column_scale = _centered_operator_column_scale(operator)
    small_column_scale = np.zeros(q + 1, dtype=np.float64)
    small_column_scale[1:] = public_column_scale[raw.small_indices]
    structured_column_scale = public_column_scale[structured_indices]
    small_null_norm, lifted_null_norm = _lifted_null_row_norms(
        small_null,
        structured_lift,
        small_column_scale=small_column_scale,
        structured_column_scale=structured_column_scale,
    )
    result = np.empty(operator.shape[0], dtype=bool)
    result[raw.small_indices] = (small_column_scale[1:] > 0.0) & (
        small_null_norm[1:] <= SHARED_RANK_POLICY.factor_rcond
    )
    result[structured_indices] = (
        local_estimable
        & (structured_column_scale > 0.0)
        & (lifted_null_norm <= SHARED_RANK_POLICY.factor_rcond)
    )
    return result


def _refined_local_solves(D: NDArray, inverses: NDArray, cross: NDArray) -> tuple[NDArray, NDArray]:
    """``X_k`` with ``D_k X_k = -C_k`` on each level's range, and the residual ``D_k X_k + C_k``.

    Multiplying by a formed inverse is not backward stable (Higham 2002,
    §14.1): on a level whose data block is near singular -- a level of a few
    rows whose basis columns its rows barely separate -- the residual grows
    like ``kappa(D_k) u |C_k|`` instead of ``u (|D_k||X_k| + |C_k|)``.  The
    reduced Schur complement's a posteriori bound ``|X|'|R|``
    (``_certified_reduced_schur_null_basis``) carries that residual, so it
    would declare every small coordinate null and blank every standard error.
    Fixed-precision iterative refinement with the same inverses restores a
    componentwise backward-stable residual (Skeel 1980; Higham 2002 §12.2);
    it stops as LAPACK's ``DPORFS`` does: once the componentwise backward error
    (Oettli & Prager 1964) is at most ``eps``, once it fails to halve, or after
    ``ITMAX = 5`` corrections.  Every residual it returns is the computed one,
    so the certificate stays an a posteriori bound whatever the refinement did.
    """
    eps = np.finfo(np.float64).eps
    solution = -np.einsum("kij,kjq->kiq", inverses, cross, optimize=True)
    last_error = 3.0
    for count in range(6):
        residual = np.einsum("kij,kjq->kiq", D, solution, optimize=True) + cross
        scale = np.einsum("kij,kjq->kiq", np.abs(D), np.abs(solution), optimize=True)
        scale += np.abs(cross)
        ratio = np.divide(np.abs(residual), scale, out=np.zeros_like(residual), where=scale > 0.0)
        ratio[(scale == 0.0) & (residual != 0.0)] = np.inf
        error = float(ratio.max(initial=0.0))
        if error <= eps or 2.0 * error > last_error or count == 5:
            return solution, residual
        solution = solution - np.einsum("kij,kjq->kiq", inverses, residual, optimize=True)
        last_error = error
    raise AssertionError("unreachable")  # pragma: no cover


def _nested_centered_estimability(
    operator: CenteredBlockOperator,
    raw: NestedDataOperator,
) -> NDArray:
    """Estimability of a nested chain's data geometry by the §6 reduction.

    The tree design ``Z_leaf M`` spans exactly the columns of ``Z_leaf``
    (``M = [M_parents | I]``), so a border coordinate is estimable exactly when
    it is in the single-level geometry with the leaf level dominant.  Every
    chain coordinate is non-estimable in the data sense: each node is a parent
    or a child in some ``z = e_p - sum_{c in ch(p)} e_c`` with ``M z = 0``, and
    the roots alias the intercept.
    """
    q, leaves = len(raw.small_indices), raw.tree.sizes[-1]
    leaf_indices = raw.structured_indices[raw.tree.offsets[-2] :]
    # Reduce on the rows x - c the leaf statistics were accumulated on (§3.4).
    # The intercept shear [1, X] -> [1, X - 1 c'] changes neither the slope
    # part of a null vector nor a centred column norm, so the decision is the
    # same, but raw moments of a column with |mean| >> sd cancel to nothing in
    # the Schur complement; the shift replaces that condition number
    # |mean| / sd with |mean - c| / sd (Chan, Golub & LeVeque 1983, §3).  The
    # border's X'w - (sum w) c is the sum of the shifted leaf crosses, and its
    # centre is that over sum w, as the operator's own centre is xtw / sum_w.
    shifted = replace(raw, leaf=replace(raw.leaf, center=np.zeros(q)))
    border_cross = shifted.leaf.cross.sum(axis=0)
    leaf_operator = CenteredBlockOperator(
        raw=SymmetricBlockOperator(
            A=shifted.A,
            C=shifted.leaf.cross,
            d=raw.leaf.weight,
            small_indices=np.arange(q),
            structured_indices=np.arange(q, q + leaves),
        ),
        cross=np.concatenate((border_cross, operator.cross[leaf_indices])),
        total=operator.total,
        center=np.concatenate((border_cross / operator.total, operator.center[leaf_indices])),
    )
    result = np.zeros(operator.shape[0], dtype=bool)
    result[raw.small_indices] = _independent_block_centered_estimability(leaf_operator)[:q]
    return result


def _orthonormal_column_span(values: NDArray) -> NDArray:
    """Return an orthonormal basis for independent input columns."""
    basis = np.asarray(values, dtype=np.float64)
    if basis.shape[1] == 0:
        return np.empty((basis.shape[0], 0), dtype=np.float64)
    column_norm = np.linalg.norm(basis, axis=0)
    nonzero = column_norm > 0.0
    if not np.any(nonzero):
        return np.empty((basis.shape[0], 0), dtype=np.float64)
    basis = basis[:, nonzero] / column_norm[nonzero]
    return np.asarray(
        scipy.linalg.orth(
            basis,
            rcond=SHARED_RANK_POLICY.factor_rcond,
        ),
        dtype=np.float64,
    )


def _coefficient_estimable_from_scaled_null_basis(
    null_basis: NDArray,
    column_scale: NDArray,
) -> NDArray:
    """Apply the shared rank policy in equilibrated coefficient coordinates."""
    null = np.asarray(null_basis, dtype=np.float64)
    scale = np.asarray(column_scale, dtype=np.float64)
    result = np.zeros(len(scale), dtype=bool)
    active = scale > 0.0
    if null.shape[1] == 0:
        result[active] = True
        return result
    equilibrated_null = null[active] * scale[active, None]
    null_norm = np.linalg.norm(equilibrated_null, axis=0)
    retained = null_norm > np.finfo(float).eps
    if not np.any(retained):
        result[active] = True
        return result
    normalized_null = equilibrated_null[:, retained] / null_norm[retained]
    result[active] = np.linalg.norm(normalized_null, axis=1) <= SHARED_RANK_POLICY.factor_rcond
    return result


def _orthonormal_scaled_parameter_null_span(
    values: NDArray,
    column_scale: NDArray,
) -> NDArray:
    """Build a parameter-null span through the rank policy's scaled coordinates."""
    candidates = np.asarray(values, dtype=np.float64)
    scale = np.asarray(column_scale, dtype=np.float64)
    width = len(scale)
    active = np.flatnonzero(scale > 0.0)
    inactive = np.flatnonzero(scale == 0.0)
    pieces: list[NDArray] = []
    if inactive.size:
        structural_null = np.zeros((width, len(inactive)), dtype=np.float64)
        structural_null[inactive, np.arange(len(inactive))] = 1.0
        pieces.append(structural_null)
    if active.size and candidates.shape[1]:
        equilibrated = candidates[active] * scale[active, None]
        equilibrated_span = _orthonormal_column_span(equilibrated)
        if equilibrated_span.shape[1]:
            # Rows below the factor-policy threshold are estimable coordinates,
            # not support to be magnified again when returning to parameter
            # coordinates. Restricting before the second orthogonalization also
            # prevents raw-coordinate SVD leakage into those rows.
            supported = np.linalg.norm(equilibrated_span, axis=1) > SHARED_RANK_POLICY.factor_rcond
            if np.any(supported):
                supported_span = _orthonormal_column_span(equilibrated_span[supported])
                supported_indices = active[supported]
                raw_values = supported_span / scale[supported_indices, None]
                # The span width was already certified in equilibrated
                # coordinates. Raw rescaling is invertible, so re-rank-testing
                # it can only discard a valid direction. QR preserves that
                # width while restoring a Euclidean parameter-space projector.
                raw_orthogonal, _triangular = scipy.linalg.qr(
                    raw_values,
                    mode="economic",
                    check_finite=False,
                )
                raw_span = np.zeros((width, raw_orthogonal.shape[1]), dtype=np.float64)
                raw_span[supported_indices] = raw_orthogonal
                pieces.append(raw_span)
    if not pieces:
        return np.empty((width, 0), dtype=np.float64)
    return np.column_stack(pieces)


def _certified_local_null_basis(
    block: NDArray,
    decomposition: RankDecomposition,
) -> NDArray:
    """Augment a local Gram null basis when factor certification is unavailable."""
    candidates = decomposition.null_basis()
    if decomposition.rank < decomposition.width or needs_factor_certification(decomposition):
        inherited_null = _null_basis_with_inherited_gram_scale(
            block,
            coordinate_gram=block,
            roundoff_reference=np.abs(block),
        )
        candidates = np.column_stack((candidates, inherited_null))
    return _orthonormal_scaled_parameter_null_span(
        candidates,
        decomposition.column_scale,
    )


def _local_range_inverse_and_null_projector(
    block: NDArray,
    decomposition: RankDecomposition,
    *,
    null_basis: NDArray | None = None,
) -> tuple[NDArray, NDArray]:
    """Return the inverse on a local PSD range and its Euclidean null projector."""
    width = block.shape[0]
    null = (
        _certified_local_null_basis(block, decomposition)
        if null_basis is None
        else np.asarray(null_basis, dtype=np.float64)
    )
    if null.shape[1] == 0:
        return decomposition.pseudo_inverse(), np.zeros_like(block)

    # A formed local moment can retain a roundoff eigenvalue that its first
    # Gram decomposition cannot distinguish from data information.  Re-test
    # the proposed range and fold any residual null directions back into the
    # Euclidean null space.  Each pass strictly shrinks the range, so this
    # bounded loop costs only small block-size decompositions.
    for _pass in range(width + 1):
        null_width = null.shape[1]
        if null_width == 0:
            range_basis = np.eye(width)
        else:
            complete_basis, _triangular = scipy.linalg.qr(
                null,
                mode="full",
                check_finite=False,
            )
            null = np.asarray(complete_basis[:, :null_width], dtype=np.float64)
            range_basis = np.asarray(complete_basis[:, null_width:], dtype=np.float64)
        null_projector = null @ null.T
        if range_basis.shape[1] == 0:
            return np.zeros_like(block), null_projector

        reduced = range_basis.T @ block @ range_basis
        reduced_decomposition = decompose_gram(0.5 * (reduced + reduced.T))
        residual_null = _certified_local_null_basis(reduced, reduced_decomposition)
        if residual_null.shape[1] == 0:
            inverse = range_basis @ reduced_decomposition.pseudo_inverse() @ range_basis.T
            return 0.5 * (inverse + inverse.T), null_projector

        expanded_null = range_basis @ residual_null
        null = _orthonormal_scaled_parameter_null_span(
            np.column_stack((null, expanded_null)),
            decomposition.column_scale,
        )

    raise np.linalg.LinAlgError("local null-space refinement did not converge")


def _null_basis_with_inherited_gram_scale(
    residual: NDArray,
    *,
    coordinate_gram: NDArray,
    roundoff_reference: NDArray,
    absolute_error: NDArray | None = None,
) -> NDArray:
    """Rank a residual against its source scale and a posteriori error bound."""
    residual = 0.5 * (np.asarray(residual, dtype=np.float64) + residual.T)
    coordinate_gram = np.asarray(coordinate_gram, dtype=np.float64)
    width = residual.shape[0]
    coordinate_scale = np.sqrt(np.maximum(np.diag(coordinate_gram), 0.0))
    active = np.flatnonzero(coordinate_scale > 0.0)
    inactive = np.flatnonzero(coordinate_scale == 0.0)

    pieces: list[NDArray] = []
    if inactive.size:
        structural_null = np.zeros((width, len(inactive)), dtype=np.float64)
        structural_null[inactive, np.arange(len(inactive))] = 1.0
        pieces.append(structural_null)
    if active.size:
        active_scale = coordinate_scale[active]
        scale_outer = np.outer(active_scale, active_scale)
        active_residual = residual[np.ix_(active, active)] / scale_outer
        active_reference = (
            np.asarray(roundoff_reference, dtype=np.float64)[np.ix_(active, active)] / scale_outer
        )
        reference_scale = max(float(np.linalg.norm(active_reference, ord=2)), 1.0)
        absolute_error_scale = 0.0
        if absolute_error is not None:
            active_error = (
                np.asarray(absolute_error, dtype=np.float64)[np.ix_(active, active)] / scale_outer
            )
            absolute_error_scale = float(np.linalg.norm(active_error, ord=2))
        cutoff = (
            SHARED_RANK_POLICY.certification_band * SHARED_RANK_POLICY.gram_rcond * reference_scale
        ) + absolute_error_scale
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (active_residual + active_residual.T))
        if eigenvalues[0] < -100.0 * cutoff:
            raise np.linalg.LinAlgError("reduced Schur complement is materially indefinite")
        discarded = eigenvectors[:, eigenvalues <= cutoff]
        if discarded.shape[1]:
            numerical_null = np.zeros((width, discarded.shape[1]), dtype=np.float64)
            numerical_null[active] = discarded / active_scale[:, None]
            pieces.append(numerical_null)

    if not pieces:
        return np.empty((width, 0), dtype=np.float64)
    return np.column_stack(pieces)


def _certified_reduced_schur_null_basis(
    *,
    source: NDArray,
    structured_cross: NDArray,
    structured_map: NDArray,
    kkt_residual: NDArray,
) -> NDArray:
    """Certify a reduced Schur null space from its source and KKT residual."""
    schur = np.asarray(source, dtype=np.float64) + np.einsum(
        "kiq,kir->qr",
        structured_cross,
        structured_map,
        optimize=True,
    )
    roundoff_reference = np.abs(source) + np.einsum(
        "kiq,kir->qr",
        np.abs(structured_cross),
        np.abs(structured_map),
        optimize=True,
    )
    absolute_error = np.einsum(
        "kiq,kir->qr",
        np.abs(structured_map),
        np.abs(kkt_residual),
        optimize=True,
    )
    return _null_basis_with_inherited_gram_scale(
        schur,
        coordinate_gram=source,
        roundoff_reference=roundoff_reference,
        absolute_error=absolute_error,
    )


def centered_operator_coefficient_estimable(
    operator: CenteredBlockOperator,
) -> NDArray:
    """Return coefficient estimability from compact centered data geometry."""
    from superglm.solvers._structured.nested import NestedDataOperator

    try:
        if isinstance(operator.raw, NestedDataOperator):
            return _nested_centered_estimability(operator, operator.raw)
        return _independent_block_centered_estimability(operator)
    except (
        np.linalg.LinAlgError,
        scipy.sparse.linalg.ArpackError,
        scipy.sparse.linalg.ArpackNoConvergence,
    ) as error:
        if operator.shape[0] <= _MAX_DENSE_CENTERED_ESTIMABILITY_WIDTH:
            return _bounded_centered_estimability(operator)
        raise RuntimeError(
            "Compact structured estimability certification failed for a system "
            "wider than the bounded dense fallback; coefficient standard errors "
            "cannot be reported safely."
        ) from error


def _coefficient_estimable_from_null_basis(
    width: int,
    null_basis: NDArray,
) -> NDArray:
    """Mark coordinates orthogonal to a retained parameter null space."""
    null = np.asarray(null_basis, dtype=np.float64)
    if null.shape[0] != width:
        raise ValueError("Null basis must match the coefficient width.")
    if null.shape[1] == 0:
        return np.ones(width, dtype=bool)
    orthonormal, _ = np.linalg.qr(null, mode="reduced")
    return np.linalg.norm(orthonormal, axis=1) <= SHARED_RANK_POLICY.factor_rcond
