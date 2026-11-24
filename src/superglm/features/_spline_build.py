"""Shared build-path helpers for spline feature specs."""

from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from superglm._group_matrix import _group_matrix_support as row_supports
from superglm.features._spline_config import (
    clear_scop_state,
    fit_time_select_message,
    warn_scop_retired,
)
from superglm.features._spline_identifiability import (
    apply_identifiability_for_spec,
    build_identifiability_projection_for_spec,
    term_null_basis,
)
from superglm.features._spline_multi_penalty import build_multi_m_components
from superglm.features._spline_penalties import _gapped_rank, structural_penalty_ranks
from superglm.features._spline_select import build_select
from superglm.features._spline_subclass_ops import prepare_shape_rules
from superglm.features.constraint import postfit_token, rules_of
from superglm.types import GroupInfo, LinearConstraintSet, NullCoordinates, TensorMarginalInfo


def _uses_fit_time_shape_constraints(spec: Any) -> bool:
    """Whether the spec requests any fit-time shape-constrained build path."""
    return any(rule.mode == "fit" for rule in rules_of(spec))


def shape_engine(spec: Any) -> Literal["qp"] | None:
    """The engine that fits ``spec``'s fit-time shape constraint, or None.

    ``"qp"`` for every basis that states its shape as rows on the raw
    coefficients. None without a fit-time constraint, or when no engine can
    build it.
    """
    if not _uses_fit_time_shape_constraints(spec):
        return None
    if hasattr(spec, "_build_monotone_constraints_raw"):
        return "qp"
    return None


def saved_on_scop(spec: Any) -> bool:
    """Whether ``spec`` is a spline saved fitted on SCOP, not yet built since.

    A P-spline pickled by 0.41.0 or earlier, before ``shape_engine`` existed,
    reads the class default None; one pickled by a later build that still had
    SCOP keeps the ``"scop"`` it was declared with, which a spline constructed
    now never stores (it warns and stores ``"qp"``).  A spline declared with
    no kind that such a fit built as one is marked None when it loads
    (``mark_kindless_saved_on_scop``).
    """
    return _uses_fit_time_shape_constraints(spec) and getattr(spec, "shape_engine", "qp") in (
        None,
        "scop",
    )


def held_splines(spec: Any) -> list[Any]:
    """The splines ``spec`` holds: itself, or an ordered term's declared basis and its built copy.

    An ordered term builds its ``_spline``, a copy of the declared
    ``_spline_obj`` clamped to its levels, which the editor clones again;
    both carry the basis's ``shape_engine``.
    """
    from superglm.features.ordered_categorical import OrderedCategorical

    if not isinstance(spec, OrderedCategorical):
        return [spec]
    held = (getattr(spec, "_spline_obj", None), getattr(spec, "_spline", None))
    return [spline for spline in held if spline is not None]


def mark_kindless_saved_on_scop(model) -> None:
    """Mark each kindless declaration that ``model``'s saved fit built as a SCOP P-spline.

    A spline declared with no kind beside a constrained P-spline was once
    fitted as a ``ps`` on SCOP. Every refit now builds the declaration as it
    reads (a ``cr`` on the QP), so its first build warns. An ordered term's
    declared basis is marked alike and its fitted spec takes the declaration
    back as ``_spline_obj``; its built ``_spline`` stays the ``ps``, which
    predicts as saved.
    """
    import copy

    from superglm.features.ordered_categorical import OrderedCategorical
    from superglm.features.spline import PSpline

    specs = getattr(model, "_specs", None) or {}
    templates = getattr(getattr(model, "_config", None), "feature_templates", ())
    for name, template in templates:
        fitted = specs.get(name)
        ordered = isinstance(fitted, OrderedCategorical) and isinstance(
            template, OrderedCategorical
        )
        fitted_basis = getattr(fitted, "_spline_obj", None) if ordered else fitted
        declared = getattr(template, "_spline_obj", None) if ordered else template
        if not (
            isinstance(fitted_basis, PSpline)
            and saved_on_scop(fitted_basis)
            and not isinstance(declared, PSpline)
            and getattr(declared, "_kindless_args", None) is not None
            and _uses_fit_time_shape_constraints(declared)
        ):
            continue
        for spline in held_splines(template):
            spline.shape_engine = None
        if ordered:
            fitted._spline_obj = copy.deepcopy(declared)


def _warn_if_saved_on_scop(spec: Any, feature_name: str | None) -> None:
    """Warn once that a spline saved fitted on SCOP (``saved_on_scop``) now fits on the QP.

    The build marks the spec ``"qp"``, which stops a second build of the same
    spec warning again, and a fit that commits marks the model's template
    (``fit_state._scop_templates_marked_qp``), so a saved model warns once,
    not on every refit.
    """
    if saved_on_scop(spec):
        warn_scop_retired(saved=True, feature_name=feature_name)
        spec.shape_engine = "qp"


def _raise_if_unsupported_fit_shape_constraint(spec: Any) -> None:
    """Reject fit-time shape constraints that no current engine can build, or under ``select``."""
    if not _uses_fit_time_shape_constraints(spec):
        return
    if shape_engine(spec) is not None:
        if spec.select:
            raise NotImplementedError(fit_time_select_message(rules_of(spec)))
        return
    rules = rules_of(spec)
    token = postfit_token(rules)
    if token:
        raise NotImplementedError(
            f"Fit-time {rules.describe()} constraints are not implemented yet. Use {token} instead."
        )
    raise NotImplementedError(
        f"Fit-time {rules.describe()} constraints are not implemented for this spline kind yet. "
        "Constraint.postfit repairs one whole-curve shape and cannot take them; "
        "use kind='cr', kind='ps' or kind='bs', or drop the rules."
    )


def qp_geometry(
    spec: Any, engine: str | None, projection: NDArray | None
) -> tuple[LinearConstraintSet | None, str | None, NDArray | None]:
    """The QP rows, engine tag and raw-to-solver map a built group carries, or three Nones.

    A group with no inequality row has no work for the QP and takes the ordinary
    route: a term whose rules all went into its basis (a flat stretch) holds them
    there, and a constrained group would be refused under a selection penalty and
    fitted by the direct solver for nothing.
    """
    if engine != "qp":
        return None, None, None
    raw = spec._build_monotone_constraints_raw()
    if raw.A.shape[0] == 0:
        return None, None, None
    return (raw if projection is None else raw.compose(projection)), "qp", projection


def training_basis_mean(spec: Any, x: NDArray) -> NDArray:
    """The unweighted mean of the raw basis over the training rows ``x``.

    ``m @ c`` is the shift ``model/runtime_canonicalize.py`` removes from the
    fitted term to print it: the mean of ``spec.transform`` over every stored
    training row, zero-weight rows included, each distinct value counted by
    its rows. Evaluated once per distinct value.
    """
    support, counts = np.unique(np.asarray(x, dtype=np.float64).ravel(), return_counts=True)
    basis = spec._basis_matrix(support)
    total = np.asarray(basis.T @ counts.astype(np.float64), dtype=np.float64).ravel()
    return total / float(counts.sum())


def _store_training_basis_mean(spec: Any, x: NDArray) -> None:
    """Store the basis mean a fit-time cap is stated against, or clear it without one.

    Every later rebuild of the cap rows (the REML restore, the design
    compiler) reads it from the spec, so they are the build's rows.
    """
    capped = _uses_fit_time_shape_constraints(spec) and any(
        rule.kind == "cap" for rule in rules_of(spec)
    )
    spec._training_basis_mean = training_basis_mean(spec, x) if capped else None


def _is_unpenalised(omega: NDArray) -> bool:
    """Whether a built penalty is identically zero.

    Only polynomial ranges covering every knot interval produce one. The term
    is then an unpenalised polynomial and is built like a parametric term,
    with no penalty matrix, so REML has no smoothing parameter to estimate.
    """
    return not np.any(omega)


def _with_row_support(info: GroupInfo, value_rows, index: NDArray | None, basis) -> GroupInfo:
    """``info`` with the row support of ``basis``, found on its rows per value (``value_rows``).

    Only while ``info`` holds ``basis``, which ``value_rows[index]`` is.
    """
    if index is not None and info.columns is basis:
        info.row_support = row_supports.detect_cell_support(
            value_rows,
            index,
            nnz=int(basis.nnz),
            replaced_bytes=row_supports.stored_csr_bytes(basis),
        )
    return info


def _interaction_projection(
    spec: Any,
    x: NDArray,
    omega_raw: NDArray,
    projection: NDArray | None,
    sample_weight: NDArray | None,
) -> NDArray | None:
    """The projection interactions inherit: the main effect's, less its flat stretches.

    ``projection`` is the main effect's constraint and identifiability
    projection. Flat stretches bind the main-effect curve only, so with flat
    rows the interaction projection is rebuilt from the other equality rows.
    """
    if spec._flat_rows is None or not spec._flat_rows.shape[0]:
        return projection
    _, _, _, constraint = spec._apply_constraints(None, omega_raw, main_effect=False)
    return build_identifiability_projection_for_spec(spec, x, constraint, sample_weight)


def build_group_info(
    spec: Any,
    x: NDArray,
    sample_weight: NDArray | None = None,
    *,
    find_row_support: bool = False,
    precision_weight: NDArray | None = None,
    feature_name: str | None = None,
) -> GroupInfo | list[GroupInfo]:
    """Build the main GroupInfo payload for a spline spec.

    The basis is evaluated once per distinct value of ``x`` and gathered to
    its rows (``distinct_values``).  ``find_row_support`` also finds its row
    support on those rows (``GroupInfo.row_support``), for the design
    builder to store without scanning the basis.
    """
    _raise_if_unsupported_fit_shape_constraint(spec)
    _warn_if_saved_on_scop(spec, feature_name)
    clear_scop_state(spec)
    engine = shape_engine(spec)

    x = np.asarray(x, dtype=np.float64).ravel()
    spec._place_knots(x, sample_weight)
    spec._validate_m_orders_build()
    prepare_shape_rules(spec, stacklevel=3, feature_name=feature_name)
    _store_training_basis_mean(spec, x)
    distinct = row_supports.distinct_values(x)
    if distinct is None:
        value_rows = index = None
        basis = spec._basis_matrix(x).tocsr()
    else:
        value_rows = spec._basis_matrix(distinct[0]).tocsr()
        index = distinct[1] if find_row_support else None
        basis = row_supports.gather_rows(value_rows, distinct[1])
    full = basis

    if spec.select:
        info = build_select(spec, x, basis, sample_weight)
        info.structural_ranks = structural_penalty_ranks(spec, info)
        return _with_row_support(info, value_rows, index, full)

    omega_raw = spec._build_penalty()
    basis, omega, n_cols, projection = spec._apply_constraints(basis, omega_raw)
    omega, n_cols, projection = apply_identifiability_for_spec(
        spec,
        x,
        omega,
        projection,
        sample_weight,
    )
    spec._interaction_projection = _interaction_projection(
        spec, x, omega_raw, projection, sample_weight
    )

    penalty_components = None
    if len(spec._m_orders) > 1:
        penalty_components = build_multi_m_components(
            x=x,
            m_orders=spec._m_orders,
            build_penalty_for_order=spec._build_penalty_for_order,
            apply_constraints=spec._apply_constraints,
            apply_identifiability=lambda values, penalty, constraint: (
                apply_identifiability_for_spec(
                    spec,
                    values,
                    penalty,
                    constraint,
                    sample_weight,
                )
            ),
        )
        omega = cast(
            NDArray,
            sum(component_omega for _, component_omega in penalty_components),
        )
    if _is_unpenalised(omega):
        omega, penalty_components = None, None

    constraints, monotone_engine, raw_to_solver_map = qp_geometry(spec, engine, projection)

    info = GroupInfo(
        columns=basis,
        n_cols=n_cols,
        penalty_matrix=omega,
        reparametrize=(spec.penalty == "ssp"),
        projection=projection,
        penalty_components=penalty_components,
        constraints=constraints,
        monotone_engine=monotone_engine,
        raw_to_solver_map=raw_to_solver_map,
    )
    # Knots and centring use geometry weights. Visibility asks whether the
    # solver's weighted observations identify the unpenalised columns.
    weights = sample_weight if precision_weight is None else precision_weight
    if distinct is None:
        finish_penalties(spec, info, full, weights)
    else:
        counts = np.bincount(distinct[1], weights=weights, minlength=len(distinct[0]))
        finish_penalties(spec, info, value_rows, counts)
    return _with_row_support(info, value_rows, index, full)


def finish_penalties(spec: Any, info: GroupInfo, basis, weights: NDArray | None) -> GroupInfo:
    """``info`` with the spec's lambda policies, the rank of each named penalty and its null basis.

    A ``lambda_policy`` names a one-penalty term's penalty ``"wiggle"``, and the
    ranks are keyed by penalty name, so they are found after it. Each raw design
    row in ``basis`` carries the combined weight of the observations it represents.
    """
    if (
        spec._lambda_policy is not None
        and info.penalty_components is None
        and info.penalty_matrix is not None
    ):
        info.penalty_components = [("wiggle", info.penalty_matrix)]
        info.component_types = {"wiggle": "difference"}
    info.lambda_policies = spec._resolve_lambda_policies(info)
    info.structural_ranks = structural_penalty_ranks(spec, info)
    info.null_coordinates = declared_null_coordinates(spec, info, basis, weights)
    return info


def declared_null_coordinates(
    spec: Any, info: GroupInfo, basis, weights: NDArray | None
) -> NullCoordinates | None:
    """The term's exact penalty null spaces and penalty roots in ``info``'s coordinates, if stated.

    A built-in kind states, for each penalty order, the raw polynomials the
    penalty annihilates (``_penalty_null_basis``) and its square root
    (``_penalty_root_for_order``). ``term_null_basis`` keeps the polynomials
    the constraints and centring leave, which must account for the whole
    null space each penalty's rank leaves: the structural rank where the
    kind declares one, else the penalty's own rank across a gap (a
    difference penalty's spectrum resolves it). A multi-order term's null
    spaces nest, the polynomials of lower degree inside those of higher, so
    its penalties share one basis (``_nested_null_basis``). A subclass may
    build other penalties, so it states none.

    The null directions are unpenalised, so only the rows can identify them:
    ``d`` centred polynomial directions need ``d + 1`` distinct values among
    the rows of positive weight. With fewer, a null coordinate's column is
    zero but for rounding, and an exactly unpenalised one would be fitted to
    that rounding; such a term states none.

    A basis function that no row of positive weight reaches has an exactly
    zero data column, and where the penalty does not reach it either (a zero
    lambda) the fit's rank counts it exactly. No orthogonal ``T`` keeps both
    that column and the null basis on coordinate axes, because the
    polynomials have coefficients inside and outside the empty support (their
    projectors do not commute), so a rotation would leave that count to
    rounding; such a term states none and keeps the data's coordinates.

    Centring cancels a null direction whose polynomial is nearly constant on
    the rows (``x`` clustered within a small share of the knot span). Each
    basis value is a rounded float, so a unit coefficient vector's column
    ``B m`` carries rounding of order ``u ||W^(1/2) B||_2``, which centring
    (an exact or once-rounded subtraction, Higham 2002, sec. 2.2 and 3.1)
    does not reduce. When the centred null columns' smallest weighted norm,
    ``sigma_min(W^(1/2) B M)`` for the orthonormal null members ``M``, falls
    below ``sqrt(u) ||W^(1/2) B||_2``, that rounding exceeds ``sqrt(u)`` of
    the column: more than half its digits are lost, the order of the fit's
    ``sqrt(eps) = sqrt(2 u)`` rank cut, so the fit's rank would count the
    direction's rounding. Such a term states none.
    """
    from superglm.features.spline import _BUILT_IN_SPLINES

    if type(spec) not in _BUILT_IN_SPLINES or info.penalty_matrix is None:
        return None
    named = info.penalty_components or [("", info.penalty_matrix)]
    orders = spec._m_orders if len(named) > 1 else spec._m_orders[:1]
    thetas, roots = [], []
    for (suffix, omega), order in zip(named, orders, strict=True):
        null_raw, root = spec._penalty_null_basis(order), spec._penalty_root_for_order(order)
        rank = (info.structural_ranks or {}).get(suffix)
        if rank is None:
            rank = _gapped_rank(np.linalg.eigvalsh(omega))
        if null_raw is None or root is None or rank is None:
            return None
        theta = term_null_basis(null_raw, info.projection, info.n_cols - rank)
        if theta is None:
            return None
        thetas.append(theta)
        roots.append(root if info.projection is None else root @ info.projection)
    null_basis = _nested_null_basis(thetas)
    if null_basis is None:
        return None
    nullity = null_basis.shape[1]
    occupied = basis.shape[0] if weights is None else np.count_nonzero(weights > 0)
    if occupied <= nullity:
        return None
    rows = basis
    if weights is not None:
        scale = (np.sqrt(weights) / np.sqrt(np.max(weights)))[:, None]
        rows = basis.multiply(scale) if sparse.issparse(basis) else scale * basis
    if sparse.issparse(rows):
        column_squares = np.asarray(rows.multiply(rows).sum(axis=0)).ravel()
    else:
        column_squares = np.einsum("ij,ij->j", rows, rows)
    if not np.all(column_squares > 0.0):
        return None
    if nullity:
        members = null_basis if info.projection is None else info.projection @ null_basis
        smallest = np.linalg.svd(rows @ members, compute_uv=False)[-1]
        # The Frobenius norm bounds the spectral norm from above, so most
        # terms pass without a Gram. Only its largest eigenvalue is needed
        # for the original cutoff; the small singular value uses B M itself.
        resolution = np.sqrt(2.0**-53)
        if smallest < resolution * np.sqrt(column_squares.sum()):
            gram = rows.T @ rows
            if sparse.issparse(gram):
                gram = gram.toarray()
            if smallest < resolution * np.sqrt(np.linalg.eigvalsh(gram)[-1]):
                return None
    return NullCoordinates(null_basis, tuple(t.shape[1] for t in thetas), tuple(roots))


def _nested_null_basis(thetas: list[NDArray]) -> NDArray | None:
    """One orthonormal basis whose leading columns span each of the null bases ``thetas``.

    Taken from the narrowest up, each must hold the basis so far, which is
    then extended by its own directions orthogonal to it. A narrower null
    space inside a wider one leaves it a residual of rounding, ``O(u)``; one
    outside leaves at least the share of a polynomial the wider kind does
    not annihilate, so the shared factor resolution ``sqrt(eps)`` decides, as
    in ``term_null_basis``. Null spaces that do not nest share no such
    basis, and the term declares none.
    """
    from superglm.solvers.rank import SHARED_RANK_POLICY

    basis = thetas[0][:, :0]
    for theta in sorted(thetas, key=lambda t: t.shape[1]):
        width = basis.shape[1]
        if width:
            inside = theta.T @ basis
            if np.linalg.norm(basis - theta @ inside, 2) > SHARED_RANK_POLICY.factor_rcond:
                return None
            theta = theta @ np.linalg.qr(inside, mode="complete")[0][:, width:]
        basis = np.hstack([basis, theta])
    return basis


def build_knots_and_penalty(
    spec: Any,
    x: NDArray,
    sample_weight: NDArray | None = None,
    n_bins: int | None = None,
    *,
    feature_name: str | None = None,
) -> tuple[NDArray | None, int, NDArray | None]:
    """Place knots and return projected penalty info without building the full basis."""
    _raise_if_unsupported_fit_shape_constraint(spec)
    _warn_if_saved_on_scop(spec, feature_name)
    clear_scop_state(spec)

    x = np.asarray(x, dtype=np.float64).ravel()
    spec._place_knots(x, sample_weight, n_bins)
    spec._validate_m_orders_build()
    prepare_shape_rules(spec, stacklevel=3, feature_name=feature_name)
    _store_training_basis_mean(spec, x)
    omega = spec._build_penalty()
    _, omega_constrained, n_cols, projection = spec._apply_constraints(None, omega)

    if spec.select:
        if len(spec._m_orders) > 1:
            max_order = max(spec._m_orders)
            omega_for_eigen = spec._build_penalty_for_order(max_order)
            _, omega_for_select, _, _ = spec._apply_constraints(None, omega_for_eigen)
            spec._eigendecompose_select(omega_for_select, projection)
        else:
            spec._eigendecompose_select(omega_constrained, projection)
        spec._interaction_projection = build_identifiability_projection_for_spec(
            spec,
            x,
            projection,
            sample_weight,
        )
        return omega_constrained, n_cols, projection

    omega_ident, n_cols, projection = apply_identifiability_for_spec(
        spec,
        x,
        omega_constrained,
        projection,
        sample_weight,
    )
    spec._interaction_projection = _interaction_projection(
        spec, x, omega, projection, sample_weight
    )

    if len(spec._m_orders) > 1:
        spec._penalty_components = build_multi_m_components(
            x=x,
            m_orders=spec._m_orders,
            build_penalty_for_order=spec._build_penalty_for_order,
            apply_constraints=spec._apply_constraints,
            apply_identifiability=lambda values, penalty, constraint: (
                apply_identifiability_for_spec(
                    spec,
                    values,
                    penalty,
                    constraint,
                    sample_weight,
                )
            ),
        )
        omega_ident = cast(
            NDArray,
            sum(component_omega for _, component_omega in spec._penalty_components),
        )
    else:
        spec._penalty_components = None
    if _is_unpenalised(omega_ident):
        omega_ident, spec._penalty_components = None, None

    return omega_ident, n_cols, projection


def tensor_marginal_info(
    spec: Any,
    x: NDArray,
    *,
    support: NDArray | None = None,
    counts: NDArray | None = None,
) -> TensorMarginalInfo:
    """Compute tensor-product marginal ingredients for an already-built spline spec."""
    if not spec._tensor_supported:
        raise NotImplementedError(
            f"{type(spec).__name__} does not support tensor marginal ingredients. "
            f"Use kind='cr' or kind='ps' for tensor product interactions."
        )

    reasons: list[str] = []
    if spec.select:
        reasons.append("select=True")
    if len(spec._m_orders) > 1:
        reasons.append(f"m={spec._m_orders}")
    if reasons:
        detail = " and ".join(reasons)
        raise NotImplementedError(
            f"Tensor interactions require single-penalty parent smooths, but "
            f"{type(spec).__name__} was configured with {detail}. "
            "This matches the mgcv te()/ti() marginal-smooth contract."
        )

    x = np.asarray(x, dtype=np.float64).ravel()
    if (support is None) != (counts is None):
        raise ValueError("support and counts must be provided together")
    evaluation_points = x if support is None else np.asarray(support, dtype=np.float64).ravel()
    support_counts = None if counts is None else np.asarray(counts, dtype=np.float64).ravel()
    if support_counts is not None and support_counts.shape != evaluation_points.shape:
        raise ValueError("counts must match the tensor marginal support")

    from superglm.features.spline import _BUILT_IN_SPLINES

    # A margin of superglm's own spline reads that spline's basis, so past the training range
    # it continues, holds or refuses as the spline does; a subclass's reads its raw basis.
    follows_policy = type(spec) in _BUILT_IN_SPLINES

    def raw_rows(points: NDArray) -> NDArray:
        if follows_policy:
            return spec._basis_matrix(points).toarray()
        return spec._raw_basis_matrix(points)

    distinct = row_supports.distinct_values(evaluation_points)
    basis_raw = (
        raw_rows(evaluation_points)
        if distinct is None
        else np.asarray(raw_rows(distinct[0]))[distinct[1]]
    )
    omega = spec._build_penalty()
    _, omega_constrained, _, projection_constraints = spec._apply_constraints(
        None, omega, main_effect=False
    )

    if projection_constraints is not None:
        basis_constrained = basis_raw @ projection_constraints
    else:
        basis_constrained = basis_raw

    centered_direction = (
        basis_constrained.sum(axis=0)
        if support_counts is None
        else support_counts @ basis_constrained
    )
    centered_norm = np.linalg.norm(centered_direction)
    if centered_norm < 1e-12:
        projection_ident = np.eye(basis_constrained.shape[1])
    else:
        centered_direction = centered_direction / centered_norm
        q, _ = np.linalg.qr(centered_direction[:, None], mode="complete")
        projection_ident = q[:, 1:]

    basis = basis_constrained @ projection_ident
    penalty = projection_ident.T @ omega_constrained @ projection_ident
    if projection_constraints is not None:
        projection = projection_constraints @ projection_ident
    else:
        projection = projection_ident
    structural = spec._structural_penalty_for_order(spec._m_orders[0])

    info = TensorMarginalInfo(
        basis=basis,
        penalty=penalty,
        knots=spec._knots.copy(),
        lo=spec._lo,
        hi=spec._hi,
        projection=projection,
        K_eff=projection.shape[1],
        degree=spec.degree,
        raw_basis_eval=spec._raw_basis_matrix,
        normalize_penalty=False,
        structural_penalty=None if structural is None else projection.T @ structural @ projection,
        follows_policy=follows_policy,
    )
    if distinct is not None:
        info.distinct = distinct
    return info
