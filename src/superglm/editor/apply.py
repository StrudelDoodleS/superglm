"""Apply editor session state to copied SuperGLM models."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from superglm.editor.terms import native_log_effect_values
from superglm.features.categorical import Categorical
from superglm.features.numeric import Numeric
from superglm.features.ordered_categorical import OrderedCategorical
from superglm.features.piecewise import Piecewise
from superglm.features.polynomial import Polynomial
from superglm.features.spline import _SplineBase
from superglm.model.fit_state import FittedStateRevision, invalidate_revised_coefficient_mode

if TYPE_CHECKING:
    from superglm._frame import FrameLike
    from superglm.editor.session import EditableTerm
    from superglm.types import GroupSlice


_EDITOR_SHARED_ROW_INPUTS = (
    "_dm",
    "_fit_X_ref",
    "_fit_y_ref",
    "_fit_sample_weight_ref",
    "_fit_offset_ref",
    "_fit_weights",
    "_fit_offset",
    "_fit_data_guard",
)

_EDITOR_EDIT_ONLY_MEMO_STATE = (
    "_fit_mu",
    "_fit_null_mu",
    "_fit_metrics_cache",
    "_fit_stats",
    "_prediction_plan",
    "_runtime_canonical_state",
    "_fast_prediction_state",
)


def _copy_model_for_editor_edits(model, *, share_transient_state: bool = False):
    """Copy a fitted model without duplicating its row-scale fit inputs."""
    shared_names: tuple[str, ...] = _EDITOR_SHARED_ROW_INPUTS
    if share_transient_state:
        shared_names += _EDITOR_EDIT_ONLY_MEMO_STATE
    shared = {name: getattr(model, name) for name in shared_names if hasattr(model, name)}
    memo = {id(value): value for value in shared.values()}
    edited_model = copy.deepcopy(model, memo)
    for name, value in shared.items():
        setattr(edited_model, name, value)
    return edited_model


def apply_edits_to_model_copy(model, terms: dict[str, EditableTerm]):
    """Return a deep-copied model with changed editor terms applied."""
    return apply_edits_to_model_copy_with_data(model, terms)


def apply_edits_to_model_copy_with_data(
    model,
    terms: dict[str, EditableTerm],
    *,
    X: FrameLike | None = None,
    y=None,
    sample_weight=None,
    offset=None,
):
    """Return a deep-copied model and refresh scalar fit stats if data is available."""
    changed_terms = [
        term
        for term in terms.values()
        if not np.allclose(
            term.edited_log_effect,
            term.original_log_effect,
            rtol=0.0,
            atol=1e-14,
        )
    ]
    has_explicit_rows = X is not None or y is not None
    has_retained_rows = (
        getattr(model, "_fit_X_ref", None) is not None
        and getattr(model, "_fit_y_ref", None) is not None
    )
    if changed_terms and not has_explicit_rows and not has_retained_rows:
        raise RuntimeError(
            "coefficient edits require scoring data; supply both X and y or refit with "
            "retain_fit_state=True"
        )
    copied_model = _copy_model_for_editor_edits(
        model,
        share_transient_state=bool(changed_terms),
    )
    has_scoring_data = (
        X is not None or y is not None or sample_weight is not None or offset is not None
    )
    if not changed_terms and not has_scoring_data:
        return copied_model
    revision = FittedStateRevision.start(
        copied_model,
        increment=bool(changed_terms or has_scoring_data),
        freeze_auxiliary_arrays=bool(changed_terms or has_scoring_data),
    )
    edited_model = revision.model
    for term in changed_terms:
        _apply_term_edit(edited_model, term)
    if changed_terms:
        invalidate_revised_coefficient_mode(edited_model)
    edited_terms = [term.name for term in changed_terms]
    if edited_terms:
        _stamp_stale_inference(edited_model, edited_terms)
        _invalidate_model_caches(edited_model, keep_inference=True)
    if edited_terms or has_scoring_data:
        _refresh_fit_statistics(
            edited_model,
            X=X,
            y=y,
            sample_weight=sample_weight,
            offset=offset,
        )
    if not getattr(edited_model, "_retain_fit_state", True):
        from superglm.model.fit_ops import _maybe_release_fit_state

        _maybe_release_fit_state(edited_model)
    return revision.commit()


def materialize_edit_request(request):
    """Construct an edited model from an immutable session snapshot."""
    kwargs = {}
    if request.dataset is not None:
        kwargs = {
            "X": request.dataset.X,
            "y": request.dataset.y,
            "sample_weight": request.dataset.sample_weight,
            "offset": request.dataset.offset,
        }
    return apply_edits_to_model_copy_with_data(
        request.base_model,
        request.terms,
        **kwargs,
    )


def _apply_term_edit(model, term: EditableTerm) -> None:
    spec = model._specs[term.name]
    groups = _feature_groups(model, term.name)

    if isinstance(spec, OrderedCategorical):
        if spec.basis == "spline":
            _apply_ordered_spline_term(model, spec, groups, term)
        else:
            _apply_ordered_step_term(model, spec, groups, term)
        return

    if isinstance(spec, Categorical):
        _apply_categorical_term(model, spec, groups, term)
        return

    if isinstance(spec, Piecewise):
        _apply_piecewise_term(model, spec, groups, term)
        return

    if isinstance(spec, Numeric):
        _patch_beta_block(model, groups, native_log_effect_values(term))
        return

    if isinstance(spec, Polynomial | _SplineBase):
        if term.x is None:
            raise NotImplementedError(f"Term {term.name!r} does not expose an editable x grid.")
        _apply_projected_term(model, spec, groups, term, term.x)
        return

    raise NotImplementedError(f"Editing is not implemented for term {term.name!r}.")


def _apply_projected_term(
    model,
    spec,
    groups: list[GroupSlice],
    term: EditableTerm,
    x_values: NDArray,
) -> None:
    B = _as_dense(spec.transform(x_values))
    weights = _term_weights(term)
    intercept_delta, beta_new = _solve_with_intercept(
        B,
        native_log_effect_values(term),
        weights,
    )
    _adjust_intercept(model, intercept_delta)
    _patch_beta_block(model, groups, beta_new)


def _ordered_spline_base_shift(model, spec: OrderedCategorical, groups: list[GroupSlice]) -> float:
    """Return f(base) under the term's PRE-edit coefficients.

    Read before the block is patched: this is the constant the base-relative
    targets have already had removed, and it has to go back into the intercept so
    an edit moves only what the user selected.

    A model with neither result attribute raises rather than returning 0.0.
    Unreachable today -- the caller only runs on a fitted model -- but 0.0 is
    exactly the value this function exists to stop being used: it is what the
    bug looked like before the shift was added, and it moves every prediction by
    f(base) while leaving the relativities right, so nothing downstream notices.
    """
    for result_name in ("_result", "_solver_result"):
        result = getattr(model, result_name, None)
        if result is None:
            continue
        beta = np.concatenate([np.asarray(result.beta)[group.sl] for group in groups])
        return spec._base_log_effect(beta)
    raise RuntimeError(
        "Cannot read the pre-edit base shift: the model has neither `_result` nor "
        "`_solver_result`. Editing an ordered spline term requires fitted coefficients."
    )


def _apply_ordered_spline_term(
    model,
    spec: OrderedCategorical,
    groups: list[GroupSlice],
    term: EditableTerm,
) -> None:
    """Project ordered levels onto the spline and assign special levels exactly.

    A special contributes exactly one design row whose spline columns are zero,
    so its edited effect determines its coefficient outright.  Feeding that row
    to the least-squares solve instead leaves the intercept and the special
    coefficient jointly under-determined whenever the constant vector lies in
    the span of the ordered rows' spline columns -- which is exactly what an
    unpenalized null-space column (``Spline(select=True)``) puts there -- and the
    min-norm solution then splits the edit between them, moving the reported
    intercept and the fitted curve.
    """
    x_values = _ordered_spline_x(term)
    # The editable targets are BASE-RELATIVE (`f(level) - f(base)`), so the
    # least-squares intercept they imply is short by exactly f(base): a null edit
    # solves to a = -f(base) and `_adjust_intercept` then shifts every prediction
    # down by f(base), on top of whatever the user asked for. The relativities stay
    # correct, which is why only an absolute assertion catches it. Add the base
    # effect of the PRE-edit coefficients back, mirroring what the Categorical path
    # gets for free from `target[base] = 0`.
    base_shift = _ordered_spline_base_shift(model, spec, groups)
    if not spec.has_specials:
        _apply_projected_term(model, spec, groups, term, x_values)
        _adjust_intercept(model, base_shift)
        return

    B = _as_dense(spec.transform(x_values))
    targets = native_log_effect_values(term)
    weights = _term_weights(term)
    labels = [str(level) for level in x_values]
    # Match on the DISPLAY namespace, which is what `x_values` carries. `_specials`
    # holds the str-coerced declaration, so on a float domain
    # (`order=[1.0, 2.0, 9.0]`, `specials=[9]`) it spells the level "9" while the
    # term's own rows spell it "9.0" -- the level is then reported missing and the
    # edit is refused on data that plainly contains it.
    specials = [str(level) for level in spec._special_display]
    missing = [level for level in specials if level not in labels]
    if missing:
        raise ValueError(f"Editable term {term.name!r} has no row for special level(s) {missing}.")
    row_of = {label: index for index, label in enumerate(labels)}
    special_rows = np.array([row_of[level] for level in specials], dtype=np.intp)
    smooth_rows = np.setdiff1d(np.arange(len(labels), dtype=np.intp), special_rows)
    n_spline = spec._split_beta(np.zeros(B.shape[1], dtype=np.float64))[0].size

    intercept_delta, spline_beta = _solve_with_intercept(
        B[smooth_rows, :n_spline],
        targets[smooth_rows],
        weights[smooth_rows],
    )
    special_beta = targets[special_rows] - intercept_delta
    _adjust_intercept(model, intercept_delta + base_shift)
    _patch_beta_block(model, groups, np.concatenate([spline_beta, special_beta]))


def _apply_categorical_term(
    model,
    spec: Categorical,
    groups: list[GroupSlice],
    term: EditableTerm,
) -> None:
    if term.levels is None:
        raise NotImplementedError(f"Term {term.name!r} has no editable levels.")
    target = _level_target_map(term, spec)
    base_value = float(target[str(spec._base_level)])
    beta_new = np.array(
        [float(target[str(level)]) - base_value for level in spec._non_base],
        dtype=np.float64,
    )
    _adjust_intercept(model, base_value)
    _patch_beta_block(model, groups, beta_new)


def _apply_piecewise_term(
    model,
    spec: Piecewise,
    groups: list[GroupSlice],
    term: EditableTerm,
) -> None:
    """Assign knot values straight into the coefficient block -- no least squares.

    ``term.x`` is the knot vector, so the editable targets are already one
    value per coefficient and the projection every other continuous term needs
    would be an identity solve computed the slow, inexact way.

    It also keeps the ``Categorical`` path's freedom from #236 for free.  The
    targets are base-relative, so a projection's implied intercept is short by
    exactly ``f(base)``; here ``target[base] == 0`` holds by construction, a
    null edit therefore yields ``base_value == 0`` and ``_adjust_intercept``
    returns without touching anything.  Dragging the base handle to ``d`` is
    the one edit that moves every coefficient: it re-bases the term, shifting
    each coefficient by ``-d`` and the intercept by ``+d``.  Predictions stay
    local even then, because the hats are a partition of unity and the two
    shifts cancel to ``d * h_base(x)``.
    """
    target = native_log_effect_values(term)
    knots = spec._knots
    if target.size != knots.size:
        raise ValueError(
            f"Editable term {term.name!r} carries {target.size} value(s) but its spec has "
            f"{knots.size} knots; the editor grid must be the knot vector."
        )
    base_value = float(target[spec._base_index])
    beta_new = np.asarray(target[spec._non_base_indices] - base_value, dtype=np.float64)
    _adjust_intercept(model, base_value)
    _patch_beta_block(model, groups, beta_new)


def _apply_ordered_step_term(
    model,
    spec: OrderedCategorical,
    groups: list[GroupSlice],
    term: EditableTerm,
) -> None:
    if term.levels is None:
        raise NotImplementedError(f"Term {term.name!r} has no editable levels.")
    target = _level_target_map(term, spec)
    base_value = float(target[str(spec._base_level)])
    beta_orig = np.array(
        [float(target[str(level)]) - base_value for level in spec._non_base],
        dtype=np.float64,
    )
    if spec._R_inv is not None:
        beta_new = np.linalg.lstsq(spec._R_inv, beta_orig, rcond=None)[0]
    else:
        beta_new = beta_orig
    _adjust_intercept(model, base_value)
    _patch_beta_block(model, groups, beta_new)


def _ordered_spline_x(term: EditableTerm) -> NDArray:
    if term.levels is None:
        raise NotImplementedError(f"Term {term.name!r} has no editable levels.")
    native_levels = term.metadata.get("native_levels")
    if native_levels is not None and len(native_levels) == len(term.levels):
        return np.asarray(native_levels, dtype=object)
    return np.asarray(term.levels, dtype=object)


def _level_target_map(term: EditableTerm, spec) -> dict[str, float]:
    assert term.levels is not None
    effects = native_log_effect_values(term)
    target = {str(level): float(effects[i]) for i, level in enumerate(term.levels)}
    grouping = getattr(spec, "_grouping", None)
    if grouping is None:
        return target

    weights = (
        np.ones(len(term.levels), dtype=np.float64)
        if term.weights is None
        else np.asarray(term.weights, dtype=np.float64).ravel()
    )
    weight_by_level = {str(level): float(weights[i]) for i, level in enumerate(term.levels)}
    for group_label in grouping.grouped_levels:
        members = [str(level) for level in grouping.group_to_originals.get(group_label, [])]
        present = [level for level in members if level in target]
        if not present:
            continue
        member_values = np.asarray([target[level] for level in present], dtype=np.float64)
        member_weights = np.asarray(
            [max(weight_by_level.get(level, 0.0), 0.0) for level in present],
            dtype=np.float64,
        )
        if float(np.sum(member_weights)) <= 0.0:
            target[str(group_label)] = float(np.mean(member_values))
        else:
            target[str(group_label)] = float(np.average(member_values, weights=member_weights))
    return target


def _feature_groups(model, name: str) -> list[GroupSlice]:
    groups = [group for group in model._groups if group.feature_name == name]
    if not groups:
        raise KeyError(f"No fitted groups found for editable term {name!r}.")
    return groups


def _term_weights(term: EditableTerm) -> NDArray:
    if term.weights is None:
        return np.ones(term.edited_log_effect.size, dtype=np.float64)
    weights = np.asarray(term.weights, dtype=np.float64)
    return np.maximum(weights, 1e-12)


def _solve_with_intercept(B: NDArray, y: NDArray, weights: NDArray) -> tuple[float, NDArray]:
    design = np.column_stack([np.ones(B.shape[0], dtype=np.float64), B])
    sqrt_w = np.sqrt(weights)
    coef = np.linalg.lstsq(design * sqrt_w[:, None], y * sqrt_w, rcond=None)[0]
    return float(coef[0]), np.asarray(coef[1:], dtype=np.float64)


def _as_dense(matrix) -> NDArray:
    if hasattr(matrix, "toarray"):
        return np.asarray(matrix.toarray(), dtype=np.float64)
    return np.asarray(matrix, dtype=np.float64)


def _patch_beta_block(model, groups: list[GroupSlice], beta_new: NDArray) -> None:
    expected = sum(group.size for group in groups)
    if beta_new.size != expected:
        raise ValueError(
            f"Projected beta has size {beta_new.size}, expected {expected} for "
            f"{[group.name for group in groups]}"
        )
    for result_name in ("_result", "_solver_result"):
        result = getattr(model, result_name, None)
        if result is None:
            continue
        offset = 0
        for group in groups:
            result.beta[group.sl] = beta_new[offset : offset + group.size]
            offset += group.size


def _adjust_intercept(model, delta: float) -> None:
    if abs(delta) < 1e-15:
        return
    for result_name in ("_result", "_solver_result"):
        result = getattr(model, result_name, None)
        if result is not None:
            result.intercept = float(result.intercept + delta)


def _invalidate_model_caches(model, *, keep_inference: bool = False) -> None:
    if not keep_inference:
        for attr in (
            "_coef_covariance",
            "_fit_active_info",
            "_fit_inference_info",
            "_group_edf",
        ):
            try:
                delattr(model, attr)
            except AttributeError:
                pass
    model._prediction_plan = None
    model._runtime_canonical_state = copy.deepcopy(model._runtime_canonical_state)
    model._fast_prediction_state = None
    model._fit_metrics_cache = None
    model._fit_metrics_cache_signature = None
    model._summary_cache = None
    model._fit_mu = None
    model._fit_null_mu = None
    model._fit_stats = None


def _refresh_fit_statistics(
    model,
    *,
    X: FrameLike | None = None,
    y=None,
    sample_weight=None,
    offset=None,
    use_fitted_design: bool = False,
) -> None:
    from superglm.distributions import clip_mu
    from superglm.links import stabilize_eta
    from superglm.model.fit_ops import (
        _compute_fit_stats,
        _compute_null_mu,
        _required_fit_columns,
    )
    from superglm.model.input_validation import validate_fit_input

    if (X is None) != (y is None):
        raise ValueError("Explicit scoring data requires both X and y.")

    retained_X_ref = getattr(model, "_fit_X_ref", None)
    retained_y_ref = getattr(model, "_fit_y_ref", None)
    X_ref = retained_X_ref if X is None else X
    y_ref = retained_y_ref if y is None else y
    uses_retained_data = (X is None and y is None) or (X is retained_X_ref and y is retained_y_ref)
    if uses_retained_data and X_ref is not None and y_ref is not None:
        from superglm.model.fit_data_guard import require_unchanged_fit_data

        require_unchanged_fit_data(model, X_ref, y_ref)
    if y_ref is None:
        model._fit_stats = None
        return

    explicit_scoring_data = X is not None or y is not None
    validation_weight = sample_weight
    validation_offset = offset
    if not explicit_scoring_data:
        if validation_weight is None:
            validation_weight = getattr(model, "_fit_weights", None)
        if validation_offset is None:
            validation_offset = getattr(model, "_fit_offset", None)
    validated_override = None
    if X_ref is not None and (
        explicit_scoring_data or sample_weight is not None or offset is not None
    ):
        validated_override = validate_fit_input(
            X_ref,
            y_ref,
            validation_weight,
            validation_offset,
            family=model._distribution,
            required_columns=_required_fit_columns(model),
        )
        X_ref = validated_override.X
        y_arr = validated_override.y
    else:
        y_arr = np.asarray(y_ref, dtype=np.float64).ravel()
    retained_weights = validation_weight is not None and validation_weight is getattr(
        model, "_fit_weights", None
    )
    if validated_override is not None:
        weights = (
            validation_weight
            if retained_weights
            else np.array(validated_override.sample_weight, dtype=np.float64, copy=True)
        )
    elif sample_weight is None:
        weights = None if explicit_scoring_data else getattr(model, "_fit_weights", None)
        if weights is None:
            weights = np.ones(y_arr.size, dtype=np.float64)
    elif retained_weights:
        weights = sample_weight
    else:
        weights = np.array(sample_weight, dtype=np.float64, copy=True).ravel()
    if weights.size != y_arr.size:
        raise ValueError(f"sample_weight has length {weights.size}, expected {y_arr.size}.")

    retained_offset = validation_offset is not None and validation_offset is getattr(
        model, "_fit_offset", None
    )
    if validated_override is not None:
        offset_arr = (
            validation_offset
            if retained_offset
            else (
                None
                if validated_override.offset is None
                else np.array(validated_override.offset, dtype=np.float64, copy=True)
            )
        )
        offset_ref = (
            getattr(model, "_fit_offset_ref", None)
            if offset is None and not explicit_scoring_data
            else offset
        )
    elif offset is None:
        offset_arr = None if explicit_scoring_data else getattr(model, "_fit_offset", None)
        offset_ref = None if explicit_scoring_data else getattr(model, "_fit_offset_ref", None)
    elif retained_offset:
        offset_arr = offset
        offset_ref = getattr(model, "_fit_offset_ref", None)
    else:
        offset_arr = np.array(offset, dtype=np.float64, copy=True).ravel()
        offset_ref = offset
    if offset_arr is not None and offset_arr.size != y_arr.size:
        raise ValueError(f"offset has length {offset_arr.size}, expected {y_arr.size}.")

    can_use_fitted_design = (
        use_fitted_design
        and getattr(model, "_dm", None) is not None
        and model._dm.n == y_arr.size
        and (X_ref is retained_X_ref or X_ref is None)
    )
    if X_ref is not None and not can_use_fitted_design:
        mu = model.predict(X_ref, offset=offset_arr)
    elif getattr(model, "_dm", None) is not None and model._dm.n == y_arr.size:
        solver_result = (
            model._solver_pirls_result() if model._solver_result is not None else model.result
        )
        eta = model._dm.matvec(solver_result.beta) + solver_result.intercept
        if offset_arr is not None:
            eta = eta + offset_arr
        eta = stabilize_eta(eta, model._link)
        mu = clip_mu(model._link.inverse(eta), model._distribution)
    else:
        model._fit_stats = None
        return

    mu = np.asarray(mu, dtype=np.float64).ravel()
    if mu.size != y_arr.size:
        raise ValueError(f"Predictions have length {mu.size}, expected {y_arr.size}.")

    null_mu = _compute_null_mu(y_arr, weights, offset_arr, model._distribution, model._link)
    deviance = float(np.sum(weights * model._distribution.deviance_unit(y_arr, mu)))
    model._fit_stats = _compute_fit_stats(
        y_arr,
        mu,
        weights,
        offset_arr,
        model._distribution,
        model._link,
        model.result.phi,
        null_mu=null_mu,
    )
    for result_name in ("_result", "_solver_result"):
        result = getattr(model, result_name, None)
        if result is not None:
            result.deviance = deviance
    model._fit_mu = mu
    model._fit_null_mu = null_mu
    if X is not None:
        model._fit_X_ref = X
    if y is not None:
        model._fit_y_ref = y
    if sample_weight is not None or explicit_scoring_data:
        model._fit_weights = weights
        if not retained_weights:
            model._fit_sample_weight_ref = sample_weight
    if offset is not None or explicit_scoring_data:
        model._fit_offset = offset_arr
        if not retained_offset:
            model._fit_offset_ref = offset_ref
    model._fit_metrics_cache = None
    model._fit_metrics_cache_signature = None
    model._summary_cache = None
    if X_ref is not None and (
        not uses_retained_data or getattr(model, "_fit_data_guard", None) is None
    ):
        from superglm.model.fit_data_guard import FitDataGuard

        model._fit_data_guard = FitDataGuard.capture(
            X_ref,
            y_arr,
            columns=tuple(model._feature_order),
        )


def _stamp_stale_inference(model, edited_terms: list[str]) -> None:
    model._editor_inference_stale = True
    model._editor_edits = {
        "format": "superglm.editor.v1",
        "terms": list(edited_terms),
        "inference": "stale",
        "message": (
            "Manual editor coefficient edits were applied. Standard errors, "
            "confidence intervals, and p-values from the original fit are "
            "reference-only until the model is refit."
        ),
    }
