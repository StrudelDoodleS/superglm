"""Shared build-path helpers for spline feature specs."""

from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from superglm._group_matrix import _group_matrix_support as row_supports
from superglm.features._spline_config import clear_scop_state, warn_scop_retired
from superglm.features._spline_identifiability import (
    apply_identifiability_for_spec,
    build_identifiability_projection_for_spec,
)
from superglm.features._spline_multi_penalty import build_multi_m_components
from superglm.features._spline_penalties import structural_penalty_ranks
from superglm.features._spline_select import build_select
from superglm.features._spline_subclass_ops import prepare_shape_rules
from superglm.types import GroupInfo, TensorMarginalInfo


def _constraint_kind(spec: Any) -> str | None:
    """Return the canonical shape token, falling back to legacy fields if needed."""
    return getattr(spec, "constraint_kind", getattr(spec, "monotone", None))


def _constraint_mode(spec: Any) -> str:
    """Return the canonical shape mode, falling back to legacy fields if needed."""
    return getattr(spec, "constraint_mode", getattr(spec, "monotone_mode", "postfit"))


def _uses_fit_time_shape_constraints(spec: Any) -> bool:
    """Whether the spec requests any fit-time shape-constrained build path."""
    return _constraint_kind(spec) is not None and _constraint_mode(spec) == "fit"


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


def _warn_if_saved_on_scop(spec: Any) -> None:
    """Warn once that a spline saved fitted on SCOP (``saved_on_scop``) now fits on the QP.

    The build marks the spec ``"qp"``, which stops a second build of the same
    spec warning again, and a fit that commits marks the model's template
    (``fit_state._scop_templates_marked_qp``), so a saved model warns once,
    not on every refit.
    """
    if saved_on_scop(spec):
        warn_scop_retired(saved=True)
        spec.shape_engine = "qp"


def _raise_if_unsupported_fit_shape_constraint(spec: Any) -> None:
    """Reject fit-time shape constraints that no current engine can build."""
    kind = _constraint_kind(spec)
    if _uses_fit_time_shape_constraints(spec) and shape_engine(spec) is None:
        raise NotImplementedError(
            f"Fit-time {kind} constraints are not implemented yet. "
            f"Use Constraint.postfit.{kind} instead."
        )


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
) -> GroupInfo | list[GroupInfo]:
    """Build the main GroupInfo payload for a spline spec.

    The basis is evaluated once per distinct value of ``x`` and gathered to
    its rows (``distinct_values``).  ``find_row_support`` also finds its row
    support on those rows (``GroupInfo.row_support``), for the design
    builder to store without scanning the basis.
    """
    _raise_if_unsupported_fit_shape_constraint(spec)
    _warn_if_saved_on_scop(spec)
    clear_scop_state(spec)
    engine = shape_engine(spec)

    if _uses_fit_time_shape_constraints(spec):
        if spec.select:
            raise NotImplementedError(
                "Fit-time shape constraints are not supported with "
                "select=True. Use select=False or Constraint.postfit.*."
            )

    x = np.asarray(x, dtype=np.float64).ravel()
    spec._place_knots(x, sample_weight)
    spec._validate_m_orders_build()
    prepare_shape_rules(spec, stacklevel=3)
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

    constraints = None
    monotone_engine = None
    raw_to_solver_map = None
    if engine == "qp":
        constraints = spec._build_monotone_constraints_raw()
        if projection is not None:
            constraints = constraints.compose(projection)
        monotone_engine = "qp"
        raw_to_solver_map = projection

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
    if (
        spec._lambda_policy is not None
        and info.penalty_components is None
        and info.penalty_matrix is not None
    ):
        info.penalty_components = [("wiggle", info.penalty_matrix)]
        info.component_types = {"wiggle": "difference"}
    info.lambda_policies = spec._resolve_lambda_policies(info)
    info.structural_ranks = structural_penalty_ranks(spec, info)
    return _with_row_support(info, value_rows, index, full)


def build_knots_and_penalty(
    spec: Any,
    x: NDArray,
    sample_weight: NDArray | None = None,
    n_bins: int | None = None,
) -> tuple[NDArray | None, int, NDArray | None]:
    """Place knots and return projected penalty info without building the full basis."""
    _raise_if_unsupported_fit_shape_constraint(spec)
    _warn_if_saved_on_scop(spec)
    clear_scop_state(spec)

    x = np.asarray(x, dtype=np.float64).ravel()
    spec._place_knots(x, sample_weight, n_bins)
    spec._validate_m_orders_build()
    prepare_shape_rules(spec, stacklevel=3)
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

    distinct = row_supports.distinct_values(evaluation_points)
    basis_raw = (
        spec._raw_basis_matrix(evaluation_points)
        if distinct is None
        else np.asarray(spec._raw_basis_matrix(distinct[0]))[distinct[1]]
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
    )
    if distinct is not None:
        info.distinct = distinct
    return info
