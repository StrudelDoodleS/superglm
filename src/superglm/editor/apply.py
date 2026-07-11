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
from superglm.features.polynomial import Polynomial
from superglm.features.spline import _SplineBase

if TYPE_CHECKING:
    from superglm.editor.session import EditableTerm
    from superglm.types import GroupSlice


_EDITOR_SHARED_FIT_INPUTS = (
    "_dm",
    "_fit_X_ref",
    "_fit_y_ref",
    "_fit_sample_weight_ref",
    "_fit_offset_ref",
    "_fit_weights",
    "_fit_offset",
    "_fit_mu",
    "_fit_null_mu",
    "_fit_metrics_cache",
    "_fit_stats",
    "_prediction_plan",
    "_runtime_canonical_state",
    "_fast_prediction_state",
)


def _copy_model_for_editor_edits(model):
    """Copy a fitted model without duplicating its row-scale fit inputs."""
    shared = {
        name: getattr(model, name) for name in _EDITOR_SHARED_FIT_INPUTS if hasattr(model, name)
    }
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
    X=None,
    y=None,
    sample_weight=None,
    offset=None,
):
    """Return a deep-copied model and refresh scalar fit stats if data is available."""
    edited_model = _copy_model_for_editor_edits(model)
    edited_terms: list[str] = []
    for term in terms.values():
        if np.allclose(term.edited_log_effect, term.original_log_effect, rtol=0.0, atol=1e-14):
            continue
        _apply_term_edit(edited_model, term)
        edited_terms.append(term.name)
    has_scoring_data = (
        X is not None or y is not None or sample_weight is not None or offset is not None
    )
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
    return edited_model


def _apply_term_edit(model, term: EditableTerm) -> None:
    spec = model._specs[term.name]
    groups = _feature_groups(model, term.name)

    if isinstance(spec, OrderedCategorical):
        if spec.basis == "spline":
            _apply_projected_term(model, spec, groups, term, _ordered_spline_x(term))
        else:
            _apply_ordered_step_term(model, spec, groups, term)
        return

    if isinstance(spec, Categorical):
        _apply_categorical_term(model, spec, groups, term)
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


def _refresh_fit_statistics(
    model,
    *,
    X=None,
    y=None,
    sample_weight=None,
    offset=None,
) -> None:
    from superglm.distributions import clip_mu
    from superglm.links import stabilize_eta
    from superglm.model.fit_ops import _compute_fit_stats, _compute_null_mu

    if (X is None) != (y is None):
        raise ValueError("Explicit scoring data requires both X and y.")

    X_ref = getattr(model, "_fit_X_ref", None) if X is None else X
    y_ref = getattr(model, "_fit_y_ref", None) if y is None else y
    if y_ref is None:
        model._fit_stats = None
        return

    explicit_scoring_data = X is not None or y is not None
    y_arr = np.asarray(y_ref, dtype=np.float64).ravel()
    retained_weights = sample_weight is not None and sample_weight is getattr(
        model, "_fit_weights", None
    )
    if sample_weight is None:
        weights = None if explicit_scoring_data else getattr(model, "_fit_weights", None)
        if weights is None:
            weights = np.ones(y_arr.size, dtype=np.float64)
    elif retained_weights:
        weights = sample_weight
    else:
        weights = np.asarray(sample_weight, dtype=np.float64).ravel()
    if weights.size != y_arr.size:
        raise ValueError(f"sample_weight has length {weights.size}, expected {y_arr.size}.")

    retained_offset = offset is not None and offset is getattr(model, "_fit_offset", None)
    if offset is None:
        offset_arr = None if explicit_scoring_data else getattr(model, "_fit_offset", None)
        offset_ref = None if explicit_scoring_data else getattr(model, "_fit_offset_ref", None)
    elif retained_offset:
        offset_arr = offset
        offset_ref = getattr(model, "_fit_offset_ref", None)
    else:
        offset_arr = np.asarray(offset, dtype=np.float64).ravel()
        offset_ref = offset
    if offset_arr is not None and offset_arr.size != y_arr.size:
        raise ValueError(f"offset has length {offset_arr.size}, expected {y_arr.size}.")

    if X_ref is not None:
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
