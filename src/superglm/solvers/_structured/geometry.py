"""Rank-aware estimability geometry for compact structured operators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
from numpy.typing import NDArray

from superglm.solvers._structured.operators import (
    BlockSymmetricOperator,
    CenteredBlockOperator,
    SumToZeroBlockOperator,
    SymmetricBlockOperator,
    compact_operator_diagonal,
    materialize_compact_operator,
)
from superglm.solvers.rank import (
    SHARED_RANK_POLICY,
    RankDecomposition,
    _eigensolver_relative_bar,
    decompose_gram,
    needs_factor_certification,
)

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


def _centered_operator_column_scale(operator: CenteredBlockOperator) -> NDArray:
    """Return cancellation-certified centered public-design column norms."""
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
    centered_diagonal[cancellation_limited] = roundoff_bound[cancellation_limited]
    return np.sqrt(np.maximum(centered_diagonal, 0.0))


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
    C_augmented = np.empty((D.shape[0], D.shape[1], q + 1), dtype=np.float64)
    C_augmented[:, :, 0] = operator.cross[structured_indices]
    C_augmented[:, :, 1:] = C
    local_estimable = np.empty(D.shape[:2], dtype=bool)
    inverse_cross = np.empty_like(C_augmented)
    for level, block in enumerate(D):
        decomposition = decompose_gram(block)
        null_basis = _certified_local_null_basis(block, decomposition)
        inverse, _null_projector = _local_range_inverse_and_null_projector(
            block,
            decomposition,
            null_basis=null_basis,
        )
        inverse_cross[level] = inverse @ C_augmented[level]
        local_estimable[level] = _coefficient_estimable_from_scaled_null_basis(
            null_basis,
            decomposition.column_scale,
        )

    structured_map = -inverse_cross.copy()
    kkt_residual = inverse_cross
    np.einsum(
        "kij,kjq->kiq",
        D,
        structured_map,
        out=kkt_residual,
        optimize=True,
    )
    kkt_residual += C_augmented
    small_null = _certified_reduced_schur_null_basis(
        source=_augmented_small_data_block(operator),
        structured_cross=C_augmented,
        structured_map=structured_map,
        kkt_residual=kkt_residual,
    )
    structured_lift = np.einsum(
        "kiq,qr->kir",
        structured_map,
        small_null,
        optimize=True,
    )
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
            raise np.linalg.LinAlgError("reduced SZ Schur complement is materially indefinite")
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
    constraint_multiplier: NDArray | None = None,
    constraint_residual: NDArray | None = None,
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
    if (constraint_multiplier is None) != (constraint_residual is None):
        raise ValueError("constraint multiplier and residual must be supplied together")
    if constraint_multiplier is not None and constraint_residual is not None:
        absolute_error += np.abs(constraint_multiplier).T @ np.abs(constraint_residual)
    return _null_basis_with_inherited_gram_scale(
        schur,
        coordinate_gram=source,
        roundoff_reference=roundoff_reference,
        absolute_error=absolute_error,
    )


def _sum_to_zero_retained_constraint_row_space(
    scaled_bases: tuple[NDArray, ...],
    final_range_projector: NDArray,
    structured_column_scale: NDArray,
) -> NDArray:
    """Rank a concatenated SZ constraint map in scaled parameter coordinates."""
    constraint_maps: list[NDArray] = []
    for scaled_basis, column_scale in zip(
        scaled_bases,
        structured_column_scale,
        strict=True,
    ):
        unscaled_basis = np.zeros_like(scaled_basis)
        active = column_scale > 0.0
        unscaled_basis[active] = scaled_basis[active] / column_scale[active, None]
        constraint_maps.append(final_range_projector @ unscaled_basis)

    constraint_factor = np.column_stack(constraint_maps)
    if constraint_factor.shape[1] == 0:
        return np.empty((0, 0), dtype=np.float64)

    # The map columns are orthonormal coordinates in the scaled public
    # parameter space. Rank that factor directly: forming M M.T and then
    # equilibrating its equation rows can retain a direction below the shared
    # factor cutoff. The thin SVD retains O(K k²) storage for a wide map.
    _left_vectors, singular_values, right_vectors = scipy.linalg.svd(
        constraint_factor,
        full_matrices=False,
        check_finite=False,
    )
    retained = singular_values > SHARED_RANK_POLICY.factor_rcond * singular_values[0]
    return right_vectors[retained]


def _sum_to_zero_scaled_null_constraint_geometry(
    local_null_projector: NDArray,
    structured_column_scale: NDArray,
) -> tuple[tuple[NDArray, ...], NDArray]:
    """Return scaled local-null bases and the retained constraint row space."""
    _n_free, block_size = structured_column_scale.shape
    scaled_bases: list[NDArray] = []
    for projector, column_scale in zip(
        local_null_projector[:-1],
        structured_column_scale,
        strict=True,
    ):
        active = np.flatnonzero(column_scale > 0.0)
        if active.size == 0:
            scaled_bases.append(np.empty((block_size, 0), dtype=np.float64))
            continue
        active_projector = projector[np.ix_(active, active)]
        _eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (active_projector + active_projector.T))
        null_width = int(np.rint(np.trace(active_projector)))
        if null_width == 0:
            scaled_bases.append(np.empty((block_size, 0), dtype=np.float64))
            continue
        active_basis = _orthonormal_column_span(
            column_scale[active, None] * eigenvectors[:, -null_width:]
        )
        scaled_basis = np.zeros((block_size, active_basis.shape[1]), dtype=np.float64)
        scaled_basis[active] = active_basis
        scaled_bases.append(scaled_basis)
    bases = tuple(scaled_bases)
    retained_row_space = _sum_to_zero_retained_constraint_row_space(
        bases,
        np.eye(block_size) - local_null_projector[-1],
        structured_column_scale,
    )
    return bases, retained_row_space


def _sum_to_zero_scaled_basis_null_row_norms(
    scaled_bases: tuple[NDArray, ...],
    retained_constraint_row_space: NDArray,
) -> tuple[NDArray, int, NDArray]:
    """Return leverage and rank-boundary uncertainty for a constrained null space."""
    n_free = len(scaled_bases)
    block_size = scaled_bases[0].shape[0] if n_free else 0
    result = np.zeros((n_free, block_size), dtype=np.float64)
    ambiguous = np.zeros((n_free, block_size), dtype=bool)
    total_width = sum(basis.shape[1] for basis in scaled_bases)
    null_dimension = total_width - retained_constraint_row_space.shape[0]
    offset = 0
    for level, scaled_basis in enumerate(scaled_bases):
        width = scaled_basis.shape[1]
        if width == 0:
            continue
        local_row_space = retained_constraint_row_space[:, offset : offset + width]
        offset += width
        removed = local_row_space.T @ local_row_space
        alpha_projector = np.eye(width) - removed
        scaled_projector = scaled_basis @ alpha_projector @ scaled_basis.T
        constrained_diagonal = np.diag(scaled_projector).copy()

        local_projector = scaled_basis @ scaled_basis.T
        removed_projector = scaled_basis @ removed @ scaled_basis.T
        projector_scale = np.abs(np.diag(local_projector)) + np.abs(np.diag(removed_projector))
        # Two width-sized contractions form this diagonal.  Bound their
        # first-order dot-product error without applying the much wider rank
        # certification band: that band can erase a genuine null leverage
        # which is already resolved above the factor-policy cutoff.
        projector_noise = 2.0 * max(width, 1) * SHARED_RANK_POLICY.gram_rcond * projector_scale
        rank_uncertainty = (
            SHARED_RANK_POLICY.certification_band * SHARED_RANK_POLICY.gram_rcond * projector_scale
        )
        # Below `projector_noise` the diagonal is not AMBIGUOUS, it is
        # UNRESOLVED, and the two need opposite treatment -- so the noise floor
        # is tested first and ambiguity is only asked above it.  Issue #356.
        #
        # Routing an unresolved entry to the certificate instead would not save
        # it, but NOT because a certificate is powerless in principle.  Two of
        # the three consumers re-read this same floored quantity, so they are
        # circular; the third, the dense branch, does independent arithmetic --
        # a sum of squares over discarded modes, with no cancelling subtraction
        # -- and only failed because ITS selection cut was a bare `gram_rcond`
        # times the largest eigenvalue, with no residual interval.
        #
        # THAT CUT IS NOW FLOORED, and this note is the disposition rather than
        # the anticipation.  The precedence STAYS as it is, on two measured
        # grounds.  First, "genuinely resolving" turns out to be half true: the
        # floor makes the dense branch's mode SELECTION a function of the data,
        # but inside `(gram_rcond, bar]` that branch DROPS a direction rather
        # than resolving it, so asking it about a sub-`projector_noise`
        # diagonal returns "not estimable" by policy, not by resolution -- the
        # same floor one level up, in other coordinates.  Second, `ambiguous`
        # is not a per-entry route: its consumer replaces the WHOLE block's
        # verdict with the certificate's, so marking `stable_zero` ambiguous
        # would send every SZ fit carrying any local null through the public
        # spectral certificate and leave this function unable to decide
        # anything.  That is a branch deletion, not an adjustment, and it is
        # recorded as a follow-up on #356 rather than taken here.
        #
        # The old order asked ambiguity first and gated `stable_zero` on it.
        # `rank_uncertainty` carries `projector_scale` and the `gram_rcond` it
        # is compared against does not, so the interval straddles the cutoff
        # on width alone: `stable_zero` was unreachable for `width <= 15` and
        # `np.sqrt(np.maximum(., 0.0))` below then decided the outcome on the
        # SIGN of a round-off residue -- a negative one clipped to exactly 0.0
        # and always estimable, a positive one of the same magnitude reaching
        # `sqrt` as ~1e-8 and failing `factor_rcond`.  Same clip, same defect,
        # as the Gram rank gate.
        #
        # THE TWO BARS CROSS AT WIDTH 16, and the fix moves the inert branch
        # across it rather than removing it: `projector_noise` is `2 width eps
        # scale` and tracks the order, `rank_uncertainty` is `32 eps scale`
        # with `p` frozen at `certification_band`, so for `width >= 17`
        # `ambiguous` is now identically False and this predicate never
        # requests the certificate.  That is the same frozen-`p` observation
        # `rank.py`'s policy docstring makes, unfixed here because no constant
        # in this function moved.  Width is a per-level local null dimension,
        # so 17 needs a fat structured block; `test_..._at_a_width_above_the_
        # two_bars_crossing` pins the regime rather than leaving it derived.
        stable_zero = np.abs(constrained_diagonal) <= projector_noise
        # A diagonal resolved BELOW zero is not a Higham projection case: `R`
        # is a column slice of an orthonormal row basis, so `0 <= R'R <= I` and
        # `B (I - R'R) B'` is PSD by construction.  Past the floor a negative
        # entry is evidence the construction broke, and letting the clip answer
        # it returns "estimable" silently.  Flagging it routes it to the
        # certificate instead, which leaves the surviving `maximum(., 0.0)`
        # provably inert -- the point of the whole change.
        resolved_negative = constrained_diagonal < -projector_noise
        ambiguous[level] = resolved_negative | (
            ~stable_zero
            & (constrained_diagonal - rank_uncertainty <= SHARED_RANK_POLICY.gram_rcond)
            & (constrained_diagonal + rank_uncertainty > SHARED_RANK_POLICY.gram_rcond)
        )
        constrained_diagonal[stable_zero] = 0.0
        # Measured on the wide-deficient SZ fixture over 7 OPENBLAS_CORETYPE
        # microkernels: the diagonal runs -0.124x to +0.323x of
        # `projector_noise` -- unresolved on every configuration and of BOTH
        # signs -- so zero is the correct answer everywhere, and the floor
        # already sized here clears the worst reading by 3.1x.
        result[level] = np.sqrt(np.maximum(constrained_diagonal, 0.0))
    return result, null_dimension, ambiguous


@dataclass(frozen=True)
class _SumToZeroPublicNullGeometry:
    """Reusable exact-null geometry in active, scaled public coordinates."""

    local_projector: NDArray
    scaled_bases: tuple[NDArray, ...]
    retained_constraint_row_space: NDArray
    row_norm: NDArray
    ambiguous: NDArray


def _sum_to_zero_public_null_geometry(
    local_null_projector: NDArray,
    structured_column_scale: NDArray,
) -> _SumToZeroPublicNullGeometry:
    """Build the compact exact-null geometry shared by SZ rank certificates."""
    scaled_bases, retained_row_space = _sum_to_zero_scaled_null_constraint_geometry(
        local_null_projector,
        structured_column_scale,
    )
    if any(basis.shape[1] for basis in scaled_bases):
        row_norm, _null_dimension, ambiguous = _sum_to_zero_scaled_basis_null_row_norms(
            scaled_bases,
            retained_row_space,
        )
    else:
        row_norm = np.zeros_like(structured_column_scale)
        ambiguous = np.zeros_like(structured_column_scale, dtype=bool)
    return _SumToZeroPublicNullGeometry(
        local_projector=local_null_projector,
        scaled_bases=scaled_bases,
        retained_constraint_row_space=retained_row_space,
        row_norm=row_norm,
        ambiguous=ambiguous,
    )


def _sum_to_zero_inherent_null_row_norms(
    geometry: _SumToZeroPublicNullGeometry,
) -> tuple[NDArray, float, NDArray]:
    """Return constrained local-null leverage in public SZ coordinates."""
    return (
        geometry.row_norm,
        SHARED_RANK_POLICY.factor_rcond,
        geometry.ambiguous,
    )


def _sum_to_zero_public_spectral_bound(
    operator: CenteredBlockOperator,
    structured_column_scale: NDArray,
) -> float:
    """Bound the normalized public SZ Gram by its absolute row sums."""
    raw = operator.raw
    if not isinstance(raw, SumToZeroBlockOperator):  # pragma: no cover - caller dispatch
        raise TypeError("public SZ rank certification requires SZ geometry")
    n_free, block_size = structured_column_scale.shape
    if n_free == 0:
        return 0.0

    # The off-diagonal final-level block and the centering outer products
    # factor across levels, so this remains O(K k²) rather than constructing
    # the public p-by-p matrix.
    inverse_scale = np.zeros_like(structured_column_scale)
    np.divide(
        1.0,
        structured_column_scale,
        out=inverse_scale,
        where=structured_column_scale > 0.0,
    )
    inverse_scale_sum = np.sum(inverse_scale, axis=0)
    structured_cross = operator.cross[raw.structured_indices]
    structured_center = operator.center[raw.structured_indices]
    absolute_cross_scale = np.abs(structured_cross) * inverse_scale
    absolute_center_scale = np.abs(structured_center) * inverse_scale
    absolute_cross_scale_sum = np.sum(absolute_cross_scale, axis=0)
    absolute_center_scale_sum = np.sum(absolute_center_scale, axis=0)
    final_block = raw.D[-1]
    absolute_final_block = np.abs(final_block)
    spectral_bound = 0.0
    for level in range(n_free):
        scale_inverse = inverse_scale[level]
        cross = structured_cross[level]
        center = structured_center[level]
        diagonal_block = (
            raw.D[level]
            + final_block
            - np.outer(cross, center)
            - np.outer(center, cross)
            + operator.total * np.outer(center, center)
        )
        diagonal_row_sum = np.sum(
            np.abs(diagonal_block) * scale_inverse[:, None] * scale_inverse[None, :],
            axis=1,
        )
        other_inverse_scale = inverse_scale_sum - scale_inverse
        off_diagonal_row_sum = scale_inverse * (absolute_final_block @ other_inverse_scale)
        other_center_sum = float(np.sum(absolute_center_scale_sum - absolute_center_scale[level]))
        other_cross_sum = float(np.sum(absolute_cross_scale_sum - absolute_cross_scale[level]))
        off_diagonal_row_sum += (
            np.abs(cross) * scale_inverse * other_center_sum
            + np.abs(center) * scale_inverse * other_cross_sum
            + abs(operator.total) * np.abs(center) * scale_inverse * other_center_sum
        )
        spectral_bound = max(
            spectral_bound,
            float(np.max(diagonal_row_sum + off_diagonal_row_sum)),
        )
    return spectral_bound


def _sum_to_zero_public_weak_bases(
    operator: CenteredBlockOperator,
    structured_column_scale: NDArray,
    local_decompositions: tuple[RankDecomposition, ...],
    spectral_bound: float,
) -> tuple[NDArray, ...]:
    """Return locally retained SZ directions weak in public coordinates."""
    raw = operator.raw
    if not isinstance(raw, SumToZeroBlockOperator):  # pragma: no cover - caller dispatch
        raise TypeError("public SZ rank certification requires SZ geometry")
    n_free, block_size = structured_column_scale.shape
    empty = tuple(np.empty((block_size, 0), dtype=np.float64) for _ in range(n_free))
    if n_free == 0:
        return empty

    # A retained local direction can still participate in a globally
    # unresolved cancellation through the shared final level.  Use the
    # factor-scale warning boundary here; the spectral certificate below
    # makes the actual rank decision at the stricter Gram cutoff.
    cutoff = SHARED_RANK_POLICY.factor_rcond * max(spectral_bound, 1.0)
    weak_bases: list[NDArray] = []
    for block, column_scale, decomposition in zip(
        raw.D[:-1],
        structured_column_scale,
        local_decompositions[:-1],
        strict=True,
    ):
        active = np.flatnonzero(column_scale > 0.0)
        if active.size == 0:
            weak_bases.append(np.empty((block_size, 0), dtype=np.float64))
            continue
        active_scale = column_scale[active]
        scaled_block = block[np.ix_(active, active)] / np.outer(
            active_scale,
            active_scale,
        )
        raw_null = _certified_local_null_basis(block, decomposition)
        scaled_null = _orthonormal_column_span(active_scale[:, None] * raw_null[active])
        if scaled_null.shape[1] == active.size:
            weak_bases.append(np.empty((block_size, 0), dtype=np.float64))
            continue
        if scaled_null.shape[1]:
            complete_basis, _triangular = scipy.linalg.qr(
                scaled_null,
                mode="full",
                check_finite=False,
            )
            retained_basis = complete_basis[:, scaled_null.shape[1] :]
        else:
            retained_basis = np.eye(active.size)
        retained_block = retained_basis.T @ scaled_block @ retained_basis
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (retained_block + retained_block.T))
        active_weak_basis = retained_basis @ eigenvectors[:, eigenvalues <= cutoff]
        weak_basis = np.zeros((block_size, active_weak_basis.shape[1]), dtype=np.float64)
        weak_basis[active] = active_weak_basis
        weak_bases.append(weak_basis)
    return tuple(weak_bases)


def _sum_to_zero_normalized_structured_matvec(
    operator: CenteredBlockOperator,
    structured_column_scale: NDArray,
    rhs: NDArray,
) -> NDArray:
    """Apply the centered public SZ Gram in column-normalized coordinates."""
    raw = operator.raw
    if not isinstance(raw, SumToZeroBlockOperator):  # pragma: no cover - caller dispatch
        raise TypeError("normalized public SZ geometry requires an SZ operator")
    values = np.asarray(rhs, dtype=np.float64)
    if values.shape != (structured_column_scale.size,):
        raise ValueError("normalized SZ rhs has the wrong width")
    active = structured_column_scale > 0.0
    structured_values = np.zeros_like(structured_column_scale)
    reshaped_values = values.reshape(structured_column_scale.shape)
    structured_values[active] = reshaped_values[active] / structured_column_scale[active]
    public = np.zeros(operator.shape[0], dtype=np.float64)
    public[raw.structured_indices] = structured_values
    product = operator.matvec(public)
    normalized_product = np.zeros_like(structured_column_scale)
    public_product = product[raw.structured_indices]
    normalized_product[active] = public_product[active] / structured_column_scale[active]
    return normalized_product.ravel()


def _ritz_residual_norms(
    operator: scipy.sparse.linalg.LinearOperator,
    eigenvalues: NDArray,
    eigenvectors: NDArray,
) -> NDArray:
    """Return absolute residual bounds for symmetric Ritz pairs."""
    applied = np.column_stack(
        [operator.matvec(eigenvectors[:, index]) for index in range(eigenvectors.shape[1])]
    )
    return np.linalg.norm(applied - eigenvectors * eigenvalues, axis=0)


def _ritz_rank_masks(
    operator: scipy.sparse.linalg.LinearOperator,
    eigenvalues: NDArray,
    eigenvectors: NDArray,
    discarded_cutoff: float,
    retained_cutoff: float | None = None,
) -> tuple[NDArray, NDArray, NDArray]:
    """Partition Ritz pairs by residual intervals around the rank cutoff."""
    residual_norm = _ritz_residual_norms(operator, eigenvalues, eigenvectors)
    retained_boundary = discarded_cutoff if retained_cutoff is None else retained_cutoff
    discarded = eigenvalues + residual_norm <= discarded_cutoff
    retained = eigenvalues - residual_norm > retained_boundary
    return discarded, retained, ~(discarded | retained)


def _certified_ritz_discarded(
    operator: scipy.sparse.linalg.LinearOperator,
    eigenvalues: NDArray,
    eigenvectors: NDArray,
    cutoff: float,
) -> NDArray:
    """Classify Ritz pairs only when their residual intervals miss the cutoff."""
    discarded, _retained, ambiguous = _ritz_rank_masks(
        operator,
        eigenvalues,
        eigenvectors,
        cutoff,
    )
    if np.any(ambiguous):
        raise np.linalg.LinAlgError("public SZ Ritz residual crosses the rank cutoff")
    return discarded


def _sum_to_zero_public_spectral_estimability(
    operator: CenteredBlockOperator,
    structured_column_scale: NDArray,
    local_decompositions: tuple[RankDecomposition, ...],
    public_weak_bases: tuple[NDArray, ...],
    spectral_bound: float,
    exact_null_geometry: _SumToZeroPublicNullGeometry | None = None,
) -> NDArray:
    """Certify public SZ rank with block solves and a thin spectral subspace."""
    raw = operator.raw
    if not isinstance(raw, SumToZeroBlockOperator):  # pragma: no cover - caller dispatch
        raise TypeError("public SZ spectral certification requires SZ geometry")
    n_free, block_size = structured_column_scale.shape
    result = np.zeros((n_free, block_size), dtype=bool)
    active_columns = structured_column_scale.ravel() > 0.0
    width = int(np.count_nonzero(active_columns))
    if width == 0:
        return result

    if exact_null_geometry is None:
        if all(decomposition.rank == block_size for decomposition in local_decompositions):
            local_null_projector = np.zeros_like(raw.D)
        else:
            local_null_projector = np.empty_like(raw.D)
            for level, (block, decomposition) in enumerate(
                zip(raw.D, local_decompositions, strict=True)
            ):
                _inverse, local_null_projector[level] = _local_range_inverse_and_null_projector(
                    block,
                    decomposition,
                )
        exact_null_geometry = _sum_to_zero_public_null_geometry(
            local_null_projector,
            structured_column_scale,
        )
    local_null_projector = exact_null_geometry.local_projector
    scaled_null_bases = exact_null_geometry.scaled_bases
    retained_constraint_row_space = exact_null_geometry.retained_constraint_row_space
    exact_null_row_norm = exact_null_geometry.row_norm
    combined_bases = tuple(
        _orthonormal_column_span(np.column_stack((exact_basis, weak_basis)))
        for exact_basis, weak_basis in zip(
            scaled_null_bases,
            public_weak_bases,
            strict=True,
        )
    )
    combined_row_space = _sum_to_zero_retained_constraint_row_space(
        combined_bases,
        np.eye(block_size) - local_null_projector[-1],
        structured_column_scale,
    )
    (
        combined_null_row_norm,
        _combined_null_dimension,
        _combined_ambiguous,
    ) = _sum_to_zero_scaled_basis_null_row_norms(
        combined_bases,
        combined_row_space,
    )

    active_null_basis_map = scipy.sparse.block_diag(
        scaled_null_bases,
        format="csr",
    )[active_columns]

    def project_exact_null(vectors: NDArray) -> NDArray:
        values = np.asarray(vectors, dtype=np.float64)
        was_vector = values.ndim == 1
        matrix = values[:, None] if was_vector else values
        local_coordinates = active_null_basis_map.T @ matrix
        null_coordinates = local_coordinates - retained_constraint_row_space.T @ (
            retained_constraint_row_space @ local_coordinates
        )
        active_projection = np.asarray(active_null_basis_map @ null_coordinates)
        return active_projection[:, 0] if was_vector else active_projection

    def normalized_matvec(rhs: NDArray) -> NDArray:
        full_rhs = np.zeros(active_columns.size, dtype=np.float64)
        full_rhs[active_columns] = rhs
        return _sum_to_zero_normalized_structured_matvec(
            operator,
            structured_column_scale,
            full_rhs,
        )[active_columns]

    normalized_operator = scipy.sparse.linalg.LinearOperator(
        (width, width),
        matvec=normalized_matvec,
        dtype=np.float64,
    )
    if width == 1:
        largest_eigenvalue_lower = max(float(normalized_matvec(np.ones(1))[0]), 0.0)
    else:
        largest_values, largest_vectors = scipy.sparse.linalg.eigsh(
            normalized_operator,
            k=1,
            which="LA",
            return_eigenvectors=True,
            v0=np.cos(np.arange(width, dtype=np.float64) + 0.5),
            tol=0.0,
            maxiter=max(1000, 20 * block_size),
        )
        largest_residual = float(
            _ritz_residual_norms(
                normalized_operator,
                largest_values,
                largest_vectors,
            )[0]
        )
        largest_eigenvalue_lower = max(float(largest_values[0]) - largest_residual, 0.0)
    largest_eigenvalue_upper = max(
        float(spectral_bound)
        * (1.0 + SHARED_RANK_POLICY.certification_band * SHARED_RANK_POLICY.gram_rcond),
        largest_eigenvalue_lower,
    )
    if not np.isfinite(largest_eigenvalue_upper) or largest_eigenvalue_upper <= 0.0:
        raise np.linalg.LinAlgError("centered public SZ Gram has no positive spectral scale")

    discarded_cutoff = SHARED_RANK_POLICY.gram_rcond * largest_eigenvalue_lower
    retained_cutoff = SHARED_RANK_POLICY.gram_rcond * largest_eigenvalue_upper
    max_null_modes = min(2 * block_size + 1, width)
    # Small public systems are cheaper and more reliable to certify directly.
    # The fixed floor keeps this allocation bounded even when block_size is 1.
    use_dense_spectrum = width <= max(2 * block_size + 1, 32)
    if use_dense_spectrum:
        factor = np.column_stack([normalized_matvec(column) for column in np.eye(width)])
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (factor + factor.T))
    else:

        def deflated_matvec(rhs: NDArray) -> NDArray:
            exact_component = project_exact_null(rhs)
            complement = np.asarray(rhs, dtype=np.float64) - exact_component
            applied = normalized_matvec(complement)
            applied -= project_exact_null(applied)
            return applied + largest_eigenvalue_upper * exact_component

        deflated_operator = scipy.sparse.linalg.LinearOperator(
            (width, width),
            matvec=deflated_matvec,
            dtype=np.float64,
        )
        # In normalized public coordinates the SZ structured Gram is a block
        # diagonal local term plus a rank-(k + 2) update: k final-level rows
        # and the two centering vectors.  A modest negative shift makes every
        # local block safely positive definite, while preserving the ordering
        # of the eigenvalues nearest zero.
        shift = -SHARED_RANK_POLICY.factor_rcond * largest_eigenvalue_upper
        local_factors: list[tuple[slice, tuple[NDArray, bool]]] = []
        update = np.zeros((width, block_size + 2), dtype=np.float64)
        offset = 0
        for block, column_scale in zip(
            raw.D[:-1],
            structured_column_scale,
            strict=True,
        ):
            active_coordinates = np.flatnonzero(column_scale > 0.0)
            local_width = len(active_coordinates)
            if local_width == 0:
                continue
            row_slice = slice(offset, offset + local_width)
            active_scale = column_scale[active_coordinates]
            normalized_block = block[np.ix_(active_coordinates, active_coordinates)] / np.outer(
                active_scale,
                active_scale,
            )
            shifted_block = normalized_block - shift * np.eye(local_width)
            local_factor = scipy.linalg.cho_factor(
                0.5 * (shifted_block + shifted_block.T),
                lower=True,
                check_finite=False,
            )
            local_factors.append((row_slice, local_factor))
            update[
                offset + np.arange(local_width),
                active_coordinates,
            ] = 1.0 / active_scale
            offset += local_width
        if offset != width:  # pragma: no cover - construction invariant
            raise np.linalg.LinAlgError("active SZ block partition has the wrong width")

        structured_indices = raw.structured_indices.ravel()[active_columns]
        flat_scale = structured_column_scale.ravel()[active_columns]
        update[:, block_size] = operator.cross[structured_indices] / flat_scale
        update[:, block_size + 1] = operator.center[structured_indices] / flat_scale
        core = np.zeros((block_size + 2, block_size + 2), dtype=np.float64)
        core[:block_size, :block_size] = raw.D[-1]
        core[block_size:, block_size:] = np.array(
            [[0.0, -1.0], [-1.0, operator.total]],
            dtype=np.float64,
        )

        inverse_update = np.empty_like(update)
        for row_slice, local_factor in local_factors:
            inverse_update[row_slice] = scipy.linalg.cho_solve(
                local_factor,
                update[row_slice],
                check_finite=False,
            )
        border = np.eye(block_size + 2) + core @ (update.T @ inverse_update)
        border_factor = scipy.linalg.lu_factor(border, check_finite=False)

        def original_shifted_inverse_matvec(rhs: NDArray) -> NDArray:
            local_solution = np.empty(width, dtype=np.float64)
            values = np.asarray(rhs, dtype=np.float64)
            for row_slice, local_factor in local_factors:
                local_solution[row_slice] = scipy.linalg.cho_solve(
                    local_factor,
                    values[row_slice],
                    check_finite=False,
                )
            multiplier = scipy.linalg.lu_solve(
                border_factor,
                core @ (update.T @ local_solution),
                check_finite=False,
            )
            return local_solution - inverse_update @ multiplier

        def approximate_shifted_inverse_matvec(rhs: NDArray) -> NDArray:
            exact_component = project_exact_null(rhs)
            complement = np.asarray(rhs, dtype=np.float64) - exact_component
            complement_solution = original_shifted_inverse_matvec(complement)
            complement_solution -= project_exact_null(complement_solution)
            return complement_solution + exact_component / (largest_eigenvalue_upper - shift)

        def deflated_shifted_matvec(rhs: NDArray) -> NDArray:
            values = np.asarray(rhs, dtype=np.float64)
            return deflated_matvec(values) - shift * values

        deflated_shifted_operator = scipy.sparse.linalg.LinearOperator(
            (width, width),
            matvec=deflated_shifted_matvec,
            dtype=np.float64,
        )
        approximate_shifted_inverse = scipy.sparse.linalg.LinearOperator(
            (width, width),
            matvec=approximate_shifted_inverse_matvec,
            dtype=np.float64,
        )
        # Since |shift| is factor_rcond times the spectral scale, this relative
        # solve tolerance keeps inverse error below one quarter of the Gram
        # rank cutoff while usually accepting the Woodbury predictor directly.
        shifted_solve_rtol = 0.25 * SHARED_RANK_POLICY.factor_rcond

        def shifted_inverse_matvec(rhs: NDArray) -> NDArray:
            values = np.asarray(rhs, dtype=np.float64)
            solution, info = scipy.sparse.linalg.cg(
                deflated_shifted_operator,
                values,
                x0=approximate_shifted_inverse_matvec(values),
                M=approximate_shifted_inverse,
                rtol=shifted_solve_rtol,
                atol=0.0,
                maxiter=min(width, max(20, 8 * block_size)),
            )
            if info != 0:
                raise np.linalg.LinAlgError("deflated public SZ shifted solve did not converge")
            return solution

        shifted_inverse = scipy.sparse.linalg.LinearOperator(
            (width, width),
            matvec=shifted_inverse_matvec,
            dtype=np.float64,
        )
        probe = np.sin(np.arange(width, dtype=np.float64) + 0.25)
        probe_solution = shifted_inverse_matvec(probe)
        relative_residual = np.linalg.norm(deflated_shifted_matvec(probe_solution) - probe) / max(
            float(np.linalg.norm(probe)), np.finfo(np.float64).tiny
        )
        # This catches catastrophic preconditioner failure only.  Rank
        # certification uses the residual interval of every returned Ritz
        # pair below, rather than treating this single probe as an error bound.
        if not np.isfinite(relative_residual) or relative_residual > 1e-5:
            raise np.linalg.LinAlgError("shifted public SZ inverse failed its residual check")

        eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
            deflated_operator,
            k=max_null_modes,
            sigma=shift,
            which="LM",
            OPinv=shifted_inverse,
            v0=np.sin(np.arange(width, dtype=np.float64) + 0.75),
            tol=0.0,
            maxiter=max(1000, 20 * block_size),
        )
        order = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

    if use_dense_spectrum:
        # Floored at the eigensolver's own bar, exactly as ``decompose_gram``'s
        # cut is and for the same reason.  ``factor`` above is an assembled
        # DENSE Gram and these eigenvalues come from ``np.linalg.eigh`` on it,
        # so by the *LAPACK Users' Guide* bound cited in
        # :func:`_eigensolver_relative_bar` nothing under ``width eps max|w|``
        # is a property of the matrix.  Against a bare ``gram_rcond`` a
        # computed zero is therefore discarded or retained on the SIGN of its
        # round-off -- measured on the shared-constant SZ fixture over 7
        # ``OPENBLAS_CORETYPE`` microkernels, an exactly null direction reads
        # -3.032e-16 to +1.494e-16 relative, both signs, at 0.32x to 1.37x of
        # that cut.  Issue #356; the sibling clip one function up was the same
        # defect and ``rank.py``'s Gram cut was the original.
        #
        # THIS IS NOT A RANK-POLICY VERSION BUMP.  ``RankPolicy.version``
        # documents itself as a claim about the selection rule inside
        # ``rank.py`` -- "bump when the deficient path can return a different
        # ``active_columns`` for the same input" -- and explicitly excludes "a
        # change outside this module".  This cut is this module's own, applied
        # to this module's own ``eigh`` output; no ``decompose_gram`` result
        # moves, and version 3 already carries the identical argument.
        #
        # WHAT IT COSTS, stated because the direction is not free.  The
        # boundary left behind is the FACTOR's: ``factor_rcond`` on singular
        # values IS ``gram_rcond`` on eigenvalues, so the bare cut sat exactly
        # on ``decompose_factor``'s answer and this one sits ``width`` times
        # above it.  Inside that band the certificate now withholds
        # estimability where the factor route grants it --
        # ``test_public_sz_certificate_is_conservative_below_its_bar_against_
        # the_factor`` pins 18 coordinates against 34 at a residue of 2.1e-15
        # relative.  That is the price of having formed the Gram, the same one
        # ``decompose_gram`` has paid since ``0d7d51b0``, and withholding a
        # standard error is the conservative direction.  What it buys: swept
        # monotonically in the perturbation, the bare cut answers 16, 32, 16,
        # 32 because the computed eigenvalue is not monotone there
        # (2.035e-16, 3.078e-16, 1.108e-16, 2.687e-16) while the exact one is.
        #
        # NOTHING IN THE SUITE CHANGES BRANCH.  This branch runs five times in
        # the whole test suite, and in every one the nearest retained
        # eigenvalue clears the floored cut by at least 60.06x (widths 4, 4,
        # 10, 25, 30; margins 6.0e+01 to 1.1e+15).  The two tests above
        # construct the geometry that does sit in the band, because none
        # existed.
        #
        # THE PRECEDENCE IN ``_sum_to_zero_scaled_basis_null_row_norms`` IS
        # UNCHANGED, and its note there is now measured rather than
        # anticipated.  See that function.
        dense_cutoff = max(
            SHARED_RANK_POLICY.gram_rcond,
            _eigensolver_relative_bar(width),
        ) * max(float(eigenvalues[-1]), 0.0)
        discarded = eigenvalues <= dense_cutoff
    else:
        discarded, _retained, ambiguous = _ritz_rank_masks(
            deflated_operator,
            eigenvalues,
            eigenvectors,
            discarded_cutoff,
            retained_cutoff,
        )
        if np.any(ambiguous):
            # Exact nulls were lifted out of this spectrum, so every crossing
            # interval is genuinely additional and uses the compact
            # conservative candidate certificate.
            result.ravel()[active_columns] = (
                combined_null_row_norm.ravel()[active_columns] <= SHARED_RANK_POLICY.factor_rcond
            )
            return result
    discarded_dimension = int(np.count_nonzero(discarded))
    if not use_dense_spectrum and discarded_dimension == max_null_modes:
        result.ravel()[active_columns] = (
            combined_null_row_norm.ravel()[active_columns] <= SHARED_RANK_POLICY.factor_rcond
        )
        return result
    if not use_dense_spectrum:
        additional_candidates = eigenvectors[:, discarded]
        additional_candidates -= project_exact_null(additional_candidates)
        candidate_norm = np.linalg.norm(additional_candidates, axis=0)
        resolved = candidate_norm > SHARED_RANK_POLICY.factor_rcond
        additional_basis = _orthonormal_column_span(additional_candidates[:, resolved])
        null_row_norm = np.sqrt(
            exact_null_row_norm.ravel()[active_columns] ** 2 + np.sum(additional_basis**2, axis=1)
        )
    else:
        null_leverage = np.sum(eigenvectors[:, discarded] ** 2, axis=1)
        # Leverage is the squared null-space row norm, so its policy boundary
        # is gram_rcond.  A value inside the certification band can be mere
        # Gram-eigenspace leakage; resolve only that bounded case against the
        # complete centered Gram, where the small/structured geometry is joint.
        ambiguous_leverage = (null_leverage > SHARED_RANK_POLICY.gram_rcond) & (
            null_leverage <= SHARED_RANK_POLICY.certification_band * SHARED_RANK_POLICY.gram_rcond
        )
        if np.any(ambiguous_leverage):
            if operator.shape[0] <= _MAX_DENSE_CENTERED_ESTIMABILITY_WIDTH:
                dense_estimable = _bounded_centered_estimability(operator)
                return dense_estimable[raw.structured_indices]
            return result
        null_row_norm = np.sqrt(null_leverage)
    result.ravel()[active_columns] = null_row_norm <= SHARED_RANK_POLICY.factor_rcond
    return result


def _certified_sum_to_zero_centered_estimability(
    operator: CenteredBlockOperator,
    local_decompositions: tuple[RankDecomposition, ...],
    public_weak_bases: tuple[NDArray, ...],
    public_spectral_bound: float,
) -> NDArray:
    """Resolve certification-limited SZ null geometry through its constraint space."""
    raw = operator.raw
    if not isinstance(raw, SumToZeroBlockOperator):  # pragma: no cover - caller dispatch
        raise TypeError("certified sum-to-zero rank requires SZ geometry")
    if operator.raw_structured_cross is None:  # pragma: no cover - caller validation
        raise ValueError("certified SZ rank requires all-level structured cross moments")

    local_inverse = np.empty_like(raw.D)
    local_null_projector = np.empty_like(raw.D)
    total_local_null_dimension = 0
    for level, (block, decomposition) in enumerate(zip(raw.D, local_decompositions, strict=True)):
        inverse, null_projector = _local_range_inverse_and_null_projector(
            block,
            decomposition,
        )
        local_inverse[level] = inverse
        local_null_projector[level] = null_projector
        total_local_null_dimension += int(np.rint(np.trace(null_projector)))

    # Zero local energy requires b_k in null(D_k).  The sum-to-zero
    # constraint couples those directions through only the k-by-k normal
    # matrix sum(P_k), regardless of the number of levels.
    constraint_null_gram = np.sum(local_null_projector, axis=0)
    constraint_null_rank = decompose_gram(0.5 * (constraint_null_gram + constraint_null_gram.T))
    constraint_null_inverse, constraint_common_range_projector = (
        _local_range_inverse_and_null_projector(
            constraint_null_gram,
            constraint_null_rank,
        )
    )
    multiplier_basis = _orthonormal_column_span(constraint_common_range_projector)
    constraint_rank = raw.block_size - multiplier_basis.shape[1]
    constrained_null_dimension = total_local_null_dimension - constraint_rank
    if constrained_null_dimension < 0:  # pragma: no cover - numerical invariant
        raise np.linalg.LinAlgError("SZ local-null constraint rank exceeds its domain")

    q = len(raw.small_indices)
    C_augmented = np.empty((raw.n_levels, raw.block_size, q + 1), dtype=np.float64)
    C_augmented[:, :, 0] = operator.raw_structured_cross
    C_augmented[:, :, 1:] = raw.C
    inverse_cross = np.einsum(
        "kij,kjq->kiq",
        local_inverse,
        C_augmented,
        optimize=True,
    )
    constraint_covariance = np.sum(local_inverse, axis=0)
    constraint_cross = np.sum(inverse_cross, axis=0)

    # The constraint multiplier lies in the common range of every D_k,
    # which is null(sum(P_k)).  Eliminate it in that at-most-k-dimensional
    # space, then satisfy the remaining constraint through local nulls.
    multiplier_map = np.zeros((raw.block_size, q + 1), dtype=np.float64)
    if multiplier_basis.shape[1]:
        restricted_covariance = multiplier_basis.T @ constraint_covariance @ multiplier_basis
        restricted_rank = decompose_gram(0.5 * (restricted_covariance + restricted_covariance.T))
        if restricted_rank.rank != multiplier_basis.shape[1]:
            raise np.linalg.LinAlgError(
                "SZ constraint multiplier is singular on the common local range"
            )
        restricted_inverse = restricted_rank.pseudo_inverse()
        multiplier_map = (
            -multiplier_basis @ restricted_inverse @ multiplier_basis.T @ constraint_cross
        )

    constraint_residual = constraint_cross + constraint_covariance @ multiplier_map
    null_dual = constraint_null_inverse @ constraint_residual
    unresolved_constraint = constraint_residual - constraint_null_gram @ null_dual
    residual_scale = max(
        float(np.linalg.norm(constraint_cross)),
        float(np.linalg.norm(constraint_covariance @ multiplier_map)),
        1.0,
    )
    if (
        np.linalg.norm(unresolved_constraint)
        > 10.0 * SHARED_RANK_POLICY.factor_rcond * residual_scale
    ):
        raise np.linalg.LinAlgError(
            "SZ cross moments are incompatible with the constrained local ranges"
        )

    structured_map = -inverse_cross
    structured_map -= np.einsum(
        "kij,jq->kiq",
        local_inverse,
        multiplier_map,
        optimize=True,
    )
    structured_map += np.einsum(
        "kij,jq->kiq",
        local_null_projector,
        null_dual,
        optimize=True,
    )
    # The reduced Schur block is a subtraction of fitted structured energy
    # from the raw small Gram.  Its KKT residual provides an a posteriori
    # floating-point error bound: exact aliases can otherwise leave positive
    # cancellation dust larger than a plain eps-scaled cutoff, while genuine
    # weak residual directions remain certifiable when this bound is small.
    kkt_residual = inverse_cross
    np.einsum(
        "kij,kjq->kiq",
        raw.D,
        structured_map,
        out=kkt_residual,
        optimize=True,
    )
    kkt_residual += C_augmented
    kkt_residual += multiplier_map[None, :, :]
    constraint_map_residual = np.sum(structured_map, axis=0)
    small_null = _certified_reduced_schur_null_basis(
        source=_augmented_small_data_block(operator),
        structured_cross=C_augmented,
        structured_map=structured_map,
        kkt_residual=kkt_residual,
        constraint_multiplier=multiplier_map,
        constraint_residual=constraint_map_residual,
    )

    structured_lift = np.einsum(
        "kiq,qr->kir",
        structured_map,
        small_null,
        optimize=True,
    )
    public_column_scale = _centered_operator_column_scale(operator)
    small_column_scale = np.zeros(q + 1, dtype=np.float64)
    small_column_scale[1:] = public_column_scale[raw.small_indices]
    structured_column_scale = public_column_scale[raw.structured_indices]
    small_null_norm, lifted_null_norm = _lifted_null_row_norms(
        small_null,
        structured_lift[:-1],
        small_column_scale=small_column_scale,
        structured_column_scale=structured_column_scale,
    )

    exact_null_geometry = _sum_to_zero_public_null_geometry(
        local_null_projector,
        structured_column_scale,
    )
    (
        inherent_null_norm,
        inherent_null_cutoff,
        inherent_rank_ambiguity,
    ) = _sum_to_zero_inherent_null_row_norms(exact_null_geometry)
    inherent_estimable = inherent_null_norm <= inherent_null_cutoff
    needs_public_certificate = any(basis.shape[1] for basis in public_weak_bases)
    if np.any(inherent_rank_ambiguity) or needs_public_certificate:
        public_estimable = _sum_to_zero_public_spectral_estimability(
            operator,
            structured_column_scale,
            local_decompositions,
            public_weak_bases,
            public_spectral_bound,
            exact_null_geometry,
        )
        inherent_estimable = public_estimable

    result = np.empty(operator.shape[0], dtype=bool)
    result[raw.small_indices] = (small_column_scale[1:] > 0.0) & (
        small_null_norm[1:] <= SHARED_RANK_POLICY.factor_rcond
    )
    result[raw.structured_indices] = (
        (structured_column_scale > 0.0)
        & inherent_estimable
        & (lifted_null_norm <= SHARED_RANK_POLICY.factor_rcond)
    )
    return result


def _sum_to_zero_centered_estimability(
    operator: CenteredBlockOperator,
) -> NDArray:
    """Compact centered-data estimability for constrained SZ blocks."""
    raw = operator.raw
    if not isinstance(raw, SumToZeroBlockOperator):  # pragma: no cover - caller dispatch
        raise TypeError("sum-to-zero rank requires SZ geometry")
    raw_structured_cross = operator.raw_structured_cross
    if raw_structured_cross is None:
        return _bounded_centered_estimability(operator)

    local_decompositions = tuple(decompose_gram(block) for block in raw.D)
    public_column_scale = _centered_operator_column_scale(operator)
    structured_column_scale = public_column_scale[raw.structured_indices]
    public_spectral_bound = _sum_to_zero_public_spectral_bound(
        operator,
        structured_column_scale,
    )
    public_weak_bases = _sum_to_zero_public_weak_bases(
        operator,
        structured_column_scale,
        local_decompositions,
        public_spectral_bound,
    )
    needs_public_certificate = any(basis.shape[1] for basis in public_weak_bases)
    if any(
        decomposition.rank < raw.block_size or needs_factor_certification(decomposition)
        for decomposition in local_decompositions
    ):
        result = _certified_sum_to_zero_centered_estimability(
            operator,
            local_decompositions,
            public_weak_bases,
            public_spectral_bound,
        )
        return result
    local_inverse = np.stack(
        [decomposition.pseudo_inverse() for decomposition in local_decompositions]
    )
    q = len(raw.small_indices)
    C_augmented = np.empty((raw.n_levels, raw.block_size, q + 1), dtype=np.float64)
    C_augmented[:, :, 0] = raw_structured_cross
    C_augmented[:, :, 1:] = raw.C
    inverse_cross = np.einsum(
        "kij,kjq->kiq",
        local_inverse,
        C_augmented,
        optimize=True,
    )
    constraint_covariance = np.sum(local_inverse, axis=0)
    constraint_rank = decompose_gram(constraint_covariance)
    if constraint_rank.rank < raw.block_size:
        return _bounded_centered_estimability(operator)
    constraint_inverse = constraint_rank.pseudo_inverse()
    constraint_cross = np.sum(inverse_cross, axis=0)
    multiplier_map = constraint_inverse @ constraint_cross
    structured_map = -inverse_cross.copy()
    structured_map += np.einsum(
        "kij,jq->kiq",
        local_inverse,
        multiplier_map,
        optimize=True,
    )
    kkt_residual = inverse_cross
    np.einsum(
        "kij,kjq->kiq",
        raw.D,
        structured_map,
        out=kkt_residual,
        optimize=True,
    )
    kkt_residual += C_augmented
    kkt_residual -= multiplier_map[None, :, :]
    constraint_map_residual = np.sum(structured_map, axis=0)
    small_null = _certified_reduced_schur_null_basis(
        source=_augmented_small_data_block(operator),
        structured_cross=C_augmented,
        structured_map=structured_map,
        kkt_residual=kkt_residual,
        constraint_multiplier=multiplier_map,
        constraint_residual=constraint_map_residual,
    )
    structured_lift = np.einsum(
        "kiq,qr->kir",
        structured_map,
        small_null,
        optimize=True,
    )
    public_column_scale = _centered_operator_column_scale(operator)
    small_column_scale = np.zeros(q + 1, dtype=np.float64)
    small_column_scale[1:] = public_column_scale[raw.small_indices]
    structured_column_scale = public_column_scale[raw.structured_indices]
    small_null_norm, lifted_null_norm = _lifted_null_row_norms(
        small_null,
        structured_lift[:-1],
        small_column_scale=small_column_scale,
        structured_column_scale=structured_column_scale,
    )
    result = np.empty(operator.shape[0], dtype=bool)
    result[raw.small_indices] = (small_column_scale[1:] > 0.0) & (
        small_null_norm[1:] <= SHARED_RANK_POLICY.factor_rcond
    )
    result[raw.structured_indices] = (structured_column_scale > 0.0) & (
        lifted_null_norm <= SHARED_RANK_POLICY.factor_rcond
    )
    if needs_public_certificate:
        result[raw.structured_indices] &= _sum_to_zero_public_spectral_estimability(
            operator,
            structured_column_scale,
            local_decompositions,
            public_weak_bases,
            public_spectral_bound,
        )
    return result


def centered_operator_coefficient_estimable(
    operator: CenteredBlockOperator,
) -> NDArray:
    """Return coefficient estimability from compact centered data geometry."""
    try:
        if isinstance(operator.raw, SumToZeroBlockOperator):
            return _sum_to_zero_centered_estimability(operator)
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
