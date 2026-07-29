"""Post-fit shape repair operations for SuperGLM."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from superglm._frame import as_eager_frame
from superglm.group_matrix import GroupMatrix
from superglm.model.fit_state import FittedStateRevision
from superglm.solvers.dispersion import pearson_residual_degrees_of_freedom
from superglm.types import PenaltyComponent


def _constraint_kind(spec) -> str | None:
    return getattr(spec, "constraint_kind", getattr(spec, "monotone", None))


def _constraint_mode(spec) -> str:
    return getattr(spec, "constraint_mode", getattr(spec, "monotone_mode", "postfit"))


def _grid_weights(spec, x_col, sample_weight, n_grid: int):
    hist_counts, bin_edges = np.histogram(x_col, bins=n_grid, weights=sample_weight)
    x_grid = np.linspace(spec._lo, spec._hi, n_grid)
    grid_weights = np.interp(
        x_grid,
        0.5 * (bin_edges[:-1] + bin_edges[1:]),
        hist_counts.astype(np.float64) + 1.0,
    )
    return np.maximum(grid_weights, 1e-6)


def _repairer(kind: str):
    from superglm.constraints import CurvatureRepairer, MonotoneRepairer

    if kind in {"increasing", "decreasing"}:
        return MonotoneRepairer(direction=kind)
    if kind in {"convex", "concave"}:
        return CurvatureRepairer(kind=kind)
    raise ValueError(f"Unsupported postfit shape kind: {kind!r}")


def _invalidate_repair_caches(model) -> None:
    for attr in ("_coef_covariance", "_fit_active_info", "_fit_inference_info", "_group_edf"):
        try:
            delattr(model, attr)
        except AttributeError:
            pass
    model._fit_metrics_cache = None
    model._fit_metrics_cache_signature = None
    model._summary_cache = None
    model._fit_mu = None
    model._fit_null_mu = None
    model._fit_stats = None


def _replace_result_beta(model, beta) -> None:
    """Update public, solver, and aliased REML results in private revision state."""
    updated: set[int] = set()
    for result_name in ("_result", "_solver_result"):
        result = getattr(model, result_name, None)
        if result is None or id(result) in updated:
            continue
        result.beta = np.asarray(beta, dtype=np.float64).copy()
        updated.add(id(result))


def _canonical_intercept_shift(model, beta) -> float:
    """Return the solver-to-public intercept shift implied by ``beta``."""
    runtime_state = getattr(model, "_runtime_canonical_state", None)
    if not isinstance(runtime_state, dict):
        return 0.0
    shift = 0.0
    for term_state in runtime_state.get("terms", {}).values():
        if not term_state.get("applied_to_public_model", False):
            continue
        for group_state in term_state.get("groups", []):
            start, end = group_state["solver_slice"]
            means = np.asarray(group_state["column_means"], dtype=np.float64)
            shift += float(means @ np.asarray(beta[start:end], dtype=np.float64))
    return shift


def _synchronize_repaired_intercept_state(model) -> None:
    """Keep solver/public intercepts coherent after a public-basis coefficient repair."""
    public_result = model._result
    solver_result = model._solver_result
    if public_result is None or solver_result is None:
        raise RuntimeError("Shape repair cannot synchronize missing fitted results")
    shift = _canonical_intercept_shift(model, public_result.beta)
    if solver_result is public_result:
        if abs(shift) > 1e-13:
            raise RuntimeError("Shape repair cannot apply a nonzero canonical intercept shift")
    else:
        solver_result.intercept = float(public_result.intercept) - shift

    runtime_state = model._runtime_canonical_state
    runtime_state["intercept_shift"] = float(shift)
    diagnostics = runtime_state.get("diagnostics")
    if isinstance(diagnostics, dict):
        diagnostics["intercept_shift"] = float(shift)
        diagnostics["coefficients_revised"] = True
    beta = np.asarray(public_result.beta, dtype=np.float64)
    for term_state in runtime_state.get("terms", {}).values():
        term_shift = 0.0
        for group_state in term_state.get("groups", []):
            start, end = group_state["solver_slice"]
            means = np.asarray(group_state["column_means"], dtype=np.float64)
            term_shift += float(means @ beta[start:end])
        term_state["intercept_shift"] = term_shift


@dataclass(frozen=True)
class _CompactPenaltyTerms:
    """Owned compact penalty geometry for one fitted REML result."""

    lambdas: float | dict[str, float]
    penalties: tuple[PenaltyComponent, ...]
    group_matrices: tuple[GroupMatrix, ...]


_LegacyPenaltyTerms = tuple[tuple[slice, float, object, object], ...]


def _build_smooth_penalty_terms(model) -> _CompactPenaltyTerms | _LegacyPenaltyTerms:
    """Prepare block/component quadratic terms without allocating a global ``p x p`` S."""
    from superglm.group_matrix import (
        DiscretizedSplineCategoricalGroupMatrix,
        DiscretizedSSPGroupMatrix,
        SparseSSPGroupMatrix,
        SplineCategoricalGroupMatrix,
    )
    from superglm.model.fit_state import fitted_lambda2

    lambda2 = fitted_lambda2(model)
    group_matrices = model._dm.group_matrices
    groups = model._groups
    reml_penalties = getattr(model, "_reml_penalties", None)
    if reml_penalties is not None:
        return _CompactPenaltyTerms(
            lambdas=lambda2,
            penalties=tuple(reml_penalties),
            group_matrices=tuple(group_matrices),
        )

    terms: list[tuple[slice, float, object, object]] = []
    penalty_matrix_types = (
        SparseSSPGroupMatrix,
        SplineCategoricalGroupMatrix,
        DiscretizedSplineCategoricalGroupMatrix,
        DiscretizedSSPGroupMatrix,
    )
    for gm, group in zip(group_matrices, groups, strict=True):
        if not group.penalized:
            continue
        if isinstance(gm, penalty_matrix_types):
            omega_components = getattr(gm, "omega_components", None)
            if omega_components is not None:
                from superglm.reml.penalty_algebra import resolve_component_lambda

                for suffix, omega_j in omega_components:
                    lam_j = float(resolve_component_lambda(lambda2, group.name, suffix))
                    if lam_j == 0.0:
                        continue
                    terms.append((group.sl, lam_j, omega_j, gm.R_inv))
                continue
            lam = lambda2.get(group.name, 0.0) if isinstance(lambda2, dict) else lambda2
            lam_float = float(lam)
            if lam_float == 0.0:
                continue
            omega = gm.omega
            if omega is not None:
                terms.append((group.sl, lam_float, omega, gm.R_inv))
            continue
        lam = lambda2.get(group.name, 0.0) if isinstance(lambda2, dict) else lambda2
        lam_float = float(lam)
        if lam_float == 0.0:
            continue
        if group.scop_reparameterization is not None:
            terms.append(
                (
                    group.sl,
                    lam_float,
                    group.scop_reparameterization.penalty_matrix(),
                    None,
                )
            )
    return tuple(terms)


def _smooth_penalty_value(
    beta,
    terms: _CompactPenaltyTerms | _LegacyPenaltyTerms,
) -> float:
    """Evaluate exact ``beta' S beta`` from compact block/component terms."""
    beta_arr = np.asarray(beta, dtype=np.float64)
    if isinstance(terms, _CompactPenaltyTerms):
        from superglm.reml.penalty_algebra import total_penalty_quadratic

        return float(
            total_penalty_quadratic(
                beta_arr,
                terms.lambdas,
                list(terms.penalties),
                list(terms.group_matrices),
            )
        )
    value = 0.0
    for group_slice, lam, omega, raw_projection in terms:
        group_beta = beta_arr[group_slice]
        penalty_beta = (
            group_beta
            if raw_projection is None
            else np.asarray(raw_projection @ group_beta, dtype=np.float64)
        )
        omega_arr = np.asarray(omega, dtype=np.float64)
        value += lam * float(penalty_beta @ (omega_arr @ penalty_beta))
    return float(value)


def _profile_repaired_intercept(
    model,
    eta,
    *,
    y,
    weights,
) -> tuple[np.ndarray, float]:
    """Minimize deviance over the unpenalized intercept for fixed coefficients."""
    from superglm.distributions import _VARIANCE_FLOOR, Gaussian, clip_mu
    from superglm.links import IdentityLink, stabilize_eta

    eta_base = np.asarray(eta, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    weights_arr = np.asarray(weights, dtype=np.float64)
    if eta_base.shape != y_arr.shape or weights_arr.shape != y_arr.shape:
        raise RuntimeError(
            "Unsafe shape repair rejected before publication: incoherent intercept geometry"
        )
    sum_weights = float(np.sum(weights_arr, dtype=np.float64))
    if not np.isfinite(sum_weights) or sum_weights <= 0.0:
        raise RuntimeError(
            "Unsafe shape repair rejected before publication: invalid scoring weights"
        )

    # Gaussian/identity has an exact conditional solution.  Keeping this
    # algebraic also preserves frequency-weight/physical-row replication to
    # floating-point precision when a repair leaves only a constant fit.
    if type(model._distribution) is Gaussian and type(model._link) is IdentityLink:
        weighted_residuals = weights_arr * (y_arr - eta_base)
        residual_sum = float(np.sum(weighted_residuals, dtype=np.float64))
        roundoff = (
            64.0
            * np.finfo(np.float64).eps
            * max(
                1.0,
                float(np.sum(np.abs(weighted_residuals), dtype=np.float64)),
            )
        )
        shift = 0.0 if abs(residual_sum) <= roundoff else residual_sum / sum_weights
        profiled_eta = eta_base + shift
        if not np.isfinite(shift) or not np.all(np.isfinite(profiled_eta)):
            raise RuntimeError(
                "Unsafe shape repair rejected before publication: invalid profiled intercept"
            )
        return profiled_eta, shift

    def evaluate(shift: float) -> tuple[float, float, float, float]:
        eta_safe = stabilize_eta(eta_base + shift, model._link)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            mu = clip_mu(model._link.inverse(eta_safe), model._distribution)
            deviance_units = np.asarray(
                model._distribution.deviance_unit(y_arr, mu),
                dtype=np.float64,
            )
            variance = np.maximum(
                np.asarray(model._distribution.variance(mu), dtype=np.float64),
                _VARIANCE_FLOOR,
            )
            dmu_deta = np.asarray(model._link.deriv_inverse(eta_safe), dtype=np.float64)
            score_rows = weights_arr * (y_arr - mu) * dmu_deta / variance
            information_rows = weights_arr * dmu_deta**2 / variance
            deviance = float(np.sum(weights_arr * deviance_units, dtype=np.float64))
            score = float(np.sum(score_rows, dtype=np.float64))
            information = float(np.sum(information_rows, dtype=np.float64))
        if not (
            np.isfinite(deviance)
            and np.isfinite(score)
            and np.isfinite(information)
            and information > 0.0
        ):
            raise RuntimeError(
                "Unsafe shape repair rejected before publication: invalid profiled intercept"
            )
        score_scale = 1.0 + float(np.sum(np.abs(score_rows), dtype=np.float64))
        return deviance, score, score_scale, information

    shift = 0.0
    deviance, score, score_scale, information = evaluate(shift)
    for _ in range(50):
        if abs(score) <= 1e-11 * score_scale:
            return eta_base + shift, shift

        proposal = score / information
        if not np.isfinite(proposal):
            break

        accepted = False
        for halving in range(21):
            trial_shift = shift + proposal * (2.0**-halving)
            (
                trial_deviance,
                trial_score,
                trial_score_scale,
                trial_information,
            ) = evaluate(trial_shift)
            roundoff = (
                64.0
                * np.finfo(np.float64).eps
                * max(
                    1.0,
                    abs(deviance),
                    abs(trial_deviance),
                )
            )
            if trial_deviance <= deviance + roundoff:
                shift = trial_shift
                deviance = trial_deviance
                score = trial_score
                score_scale = trial_score_scale
                information = trial_information
                accepted = True
                break
        if not accepted:
            break

    if abs(score) > 1e-9 * score_scale:
        raise RuntimeError(
            "Unsafe shape repair rejected before publication: conditional intercept "
            "optimization did not converge"
        )
    return eta_base + shift, shift


def _shape_candidate_objective(
    model,
    beta,
    *,
    eta,
    y,
    weights,
    smooth_penalty_terms,
    selection_penalty,
) -> tuple[float, float]:
    """Return weighted deviance and the fitted penalized merit for a coefficient state."""
    beta_arr = np.asarray(beta, dtype=np.float64)
    if beta_arr.shape != model.result.beta.shape or not np.all(np.isfinite(beta_arr)):
        raise RuntimeError("Unsafe shape repair rejected before publication: invalid coefficients")

    eta = np.asarray(eta, dtype=np.float64)
    if eta.shape != (model._dm.n,):
        raise RuntimeError("Unsafe shape repair rejected before publication: invalid predictor")
    if not np.all(np.isfinite(eta)):
        raise RuntimeError(
            "Unsafe shape repair rejected before publication: non-finite linear predictor"
        )
    from superglm.distributions import clip_mu
    from superglm.links import stabilize_eta

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        eta_safe = stabilize_eta(eta, model._link)
        mu = clip_mu(model._link.inverse(eta_safe), model._distribution)
        deviance_units = np.asarray(model._distribution.deviance_unit(y, mu), dtype=np.float64)
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(deviance_units)):
        raise RuntimeError(
            "Unsafe shape repair rejected before publication: invalid link-domain predictions"
        )

    deviance = float(np.sum(weights * deviance_units))
    smooth_value = _smooth_penalty_value(beta_arr, smooth_penalty_terms)
    selection_value = 2.0 * float(selection_penalty.eval(beta_arr, model._groups))
    merit = deviance + smooth_value + selection_value
    if not np.isfinite(deviance) or not np.isfinite(merit):
        raise RuntimeError(
            "Unsafe shape repair rejected before publication: non-finite fitted objective"
        )
    return deviance, merit


def _shape_term_eta_delta(model, current_beta, candidate_beta, groups) -> np.ndarray:
    """Evaluate one public-term coefficient change through retained group kernels."""
    current = np.asarray(current_beta, dtype=np.float64)
    candidate = np.asarray(candidate_beta, dtype=np.float64)
    delta = candidate - current
    group_keys = {(group.name, group.start, group.end) for group in groups}
    contribution = np.zeros(model._dm.n, dtype=np.float64)
    for group_matrix, group in zip(
        model._dm.group_matrices,
        model._groups,
        strict=True,
    ):
        if (group.name, group.start, group.end) in group_keys:
            contribution += np.asarray(group_matrix.matvec(delta[group.sl]), dtype=np.float64)
    contribution -= _canonical_intercept_shift(model, delta)
    if not np.all(np.isfinite(contribution)):
        raise RuntimeError(
            "Unsafe shape repair rejected before publication: non-finite term predictor"
        )
    return contribution


def _validate_repair_for_publication(
    model,
    *,
    spec,
    kind,
    groups,
    repair_result,
    y,
    scoring_weights,
    smooth_penalty_terms,
    selection_penalty,
    current_eta,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Certify fitted geometry and objective quality before changing private state."""
    current_beta = np.asarray(model.result.beta, dtype=np.float64)
    candidate_beta = np.asarray(repair_result.repaired_beta_reparam, dtype=np.float64)
    if candidate_beta.shape != current_beta.shape or not np.all(np.isfinite(candidate_beta)):
        raise RuntimeError("Unsafe shape repair rejected before publication: invalid coefficients")

    grid = np.asarray(repair_result.grid, dtype=np.float64)
    if grid.ndim != 1 or grid.size < 3 or not np.all(np.isfinite(grid)):
        raise RuntimeError("Unsafe shape repair rejected before publication: invalid repair grid")
    term_beta = np.concatenate([candidate_beta[group.sl] for group in groups])
    fitted_basis = np.asarray(spec.transform(grid), dtype=np.float64)
    if fitted_basis.shape != (grid.size, term_beta.size):
        raise RuntimeError(
            "Unsafe shape repair rejected before publication: incoherent fitted-basis geometry"
        )
    repaired_curve = fitted_basis @ term_beta
    if not np.all(np.isfinite(repaired_curve)):
        raise RuntimeError(
            "Unsafe shape repair rejected before publication: non-finite repair curve"
        )

    if kind in {"increasing", "decreasing", "convex", "concave"}:
        from superglm.constraints import shape_constraint_certificate

        certificate = shape_constraint_certificate(spec, term_beta, kind)
        feasibility_tolerance = 2e-12 * (1.0 + float(np.linalg.norm(term_beta)))
        if certificate.minimum_scaled_slack < -feasibility_tolerance:
            constraint_name = (
                "monotonicity" if kind in {"increasing", "decreasing"} else "curvature"
            )
            raise RuntimeError(
                f"Unsafe shape repair rejected before publication: infeasible {constraint_name}"
            )

    candidate_eta_delta = _shape_term_eta_delta(
        model,
        current_beta,
        candidate_beta,
        groups,
    )
    mean_shift = float(np.mean(candidate_eta_delta))
    centering_tolerance = 2e-10 * (1.0 + float(np.max(np.abs(candidate_eta_delta))))
    if not np.isfinite(mean_shift) or abs(mean_shift) > centering_tolerance:
        raise RuntimeError(
            "Unsafe shape repair rejected before publication: fitted centering changed"
        )

    fallback_beta = current_beta.copy()
    for group in groups:
        fallback_beta[group.sl] = 0.0
    fallback_eta_delta = _shape_term_eta_delta(
        model,
        current_beta,
        fallback_beta,
        groups,
    )
    candidate_eta, candidate_intercept_shift = _profile_repaired_intercept(
        model,
        np.asarray(current_eta, dtype=np.float64) + candidate_eta_delta,
        y=y,
        weights=scoring_weights,
    )
    fallback_eta, _ = _profile_repaired_intercept(
        model,
        np.asarray(current_eta, dtype=np.float64) + fallback_eta_delta,
        y=y,
        weights=scoring_weights,
    )
    candidate_deviance, candidate_merit = _shape_candidate_objective(
        model,
        candidate_beta,
        eta=candidate_eta,
        y=y,
        weights=scoring_weights,
        smooth_penalty_terms=smooth_penalty_terms,
        selection_penalty=selection_penalty,
    )
    fallback_deviance, fallback_merit = _shape_candidate_objective(
        model,
        fallback_beta,
        eta=fallback_eta,
        y=y,
        weights=scoring_weights,
        smooth_penalty_terms=smooth_penalty_terms,
        selection_penalty=selection_penalty,
    )
    deviance_tolerance = 1e-8 * (1.0 + abs(fallback_deviance))
    merit_tolerance = 1e-8 * (1.0 + abs(fallback_merit))
    if (
        candidate_deviance > fallback_deviance + deviance_tolerance
        or candidate_merit > fallback_merit + merit_tolerance
    ):
        raise RuntimeError(
            "Unsafe shape repair rejected before publication: candidate objective is worse "
            "than the feasible zero-term fallback"
        )

    repair_result.repaired_log_effect = repaired_curve.copy()
    if kind in {"increasing", "decreasing", "convex", "concave"}:
        repair_result.max_violation_after = certificate.violation
    return candidate_beta, candidate_eta, candidate_intercept_shift


def _pending_repairs(model, spline_type) -> list[tuple[str, object, str, list[object]]]:
    """Return repairs that can change coefficients without cloning fitted state."""
    beta = model.result.beta
    pending: list[tuple[str, object, str, list[object]]] = []
    for name, spec in model._specs.items():
        if not isinstance(spec, spline_type):
            continue
        kind = _constraint_kind(spec)
        if kind is None or _constraint_mode(spec) != "postfit":
            continue
        groups = [group for group in model._groups if group.feature_name == name]
        if groups and any(np.any(beta[group.sl] != 0.0) for group in groups):
            pending.append((name, spec, kind, groups))
    return pending


def _repair_changes_coefficients(model, pending_repair) -> bool:
    """Return whether span-wise certification requires a coefficient projection."""
    from superglm.constraints import shape_constraint_is_roundoff_feasible

    _, spec, kind, groups = pending_repair
    term_beta = np.concatenate([model.result.beta[group.sl] for group in groups])
    return not shape_constraint_is_roundoff_feasible(spec, term_beta, kind)


def _refresh_repaired_geometry(model, prior_working_weights, prior_selected_names) -> None:
    """Rebuild exact fitted geometry after any published coefficient revision."""
    from types import SimpleNamespace

    from superglm.model import state_ops
    from superglm.model.fit_state import fitted_penalty
    from superglm.solvers.pirls import _selection_local_curvature_depends_on_beta
    from superglm.solvers.rank import selected_group_name_set

    current_working_weights = state_ops._solver_space_working_weights(model)
    coefficient_state = SimpleNamespace(beta=model._solver_result.beta, rank_info=None)
    current_selected_names = selected_group_name_set(
        coefficient_state,
        model._groups,
        penalty=fitted_penalty(model),
    )
    weights_unchanged = np.array_equal(current_working_weights, prior_working_weights)
    selection_unchanged = current_selected_names == prior_selected_names
    curvature_changes = _selection_local_curvature_depends_on_beta(fitted_penalty(model))
    if (
        weights_unchanged
        and selection_unchanged
        and not curvature_changes
        and model._solver_result.rank_info is not None
    ):
        return

    updated: set[int] = set()
    for result_name in ("_result", "_solver_result"):
        result = getattr(model, result_name, None)
        if result is None or id(result) in updated:
            continue
        result.rank_info = None
        updated.add(id(result))

    inference = state_ops.fit_inference_info(model)
    effective_df = 1.0 + float(np.sum(inference["edf"]))
    updated.clear()
    for result_name in ("_result", "_solver_result"):
        result = getattr(model, result_name, None)
        if result is None or id(result) in updated:
            continue
        result.effective_df = effective_df
        updated.add(id(result))
    model.__dict__["_fit_inference_info"] = inference


def _refresh_repaired_scale_and_statistics(model) -> None:
    """Pair the repaired coefficients and final EDF with a coherent scale."""
    from superglm.distributions import _VARIANCE_FLOOR
    from superglm.model.fit_ops import _compute_fit_stats

    y = np.asarray(model._fit_y_ref, dtype=np.float64)
    weights = np.asarray(model._fit_weights, dtype=np.float64)
    mu = np.asarray(model._fit_mu, dtype=np.float64)
    if getattr(model._distribution, "scale_known", True):
        phi = 1.0
    elif getattr(model, "_last_fit_meta", {}).get("method") == "fit_reml":
        from superglm.model.reml_finalize import compute_profiled_phi

        phi = compute_profiled_phi(
            model,
            y=y,
            sample_weight=weights,
            lambdas=model._reml_lambdas,
            reml_penalties=model._reml_penalties,
            pirls_result=model._solver_result,
        )
    else:
        variance = np.maximum(model._distribution.variance(mu), _VARIANCE_FLOOR)
        pearson = float(np.sum(weights * (y - mu) ** 2 / variance))
        residual_df = pearson_residual_degrees_of_freedom(
            model._distribution,
            weights,
            model._solver_result.effective_df,
        )
        phi = pearson / residual_df

    updated: set[int] = set()
    for result_name in ("_result", "_solver_result"):
        result = getattr(model, result_name, None)
        if result is None or id(result) in updated:
            continue
        result.phi = float(phi)
        updated.add(id(result))
    model._fit_stats = _compute_fit_stats(
        y,
        mu,
        weights,
        model._fit_offset,
        model._distribution,
        model._link,
        float(phi),
        null_mu=model._fit_null_mu,
    )
    model._summary_cache = None


def apply_shape_postfit(model, X, sample_weight=None, offset=None, *, n_grid: int = 500):
    from superglm.features.spline import _SplineBase

    if model._result is None:
        raise RuntimeError("Model must be fitted before calling apply_shape_postfit().")
    if not getattr(model, "_retain_fit_state", True):
        raise RuntimeError(
            "Post-fit shape repair requires row-scale fitted state; refit with "
            "retain_fit_state=True before calling apply_shape_postfit()."
        )

    pending_repairs = _pending_repairs(model, _SplineBase)
    if not pending_repairs:
        return model

    y_ref = getattr(model, "_fit_y_ref", None)
    if y_ref is None:
        raise RuntimeError("Post-fit shape repair requires retained fit features and response.")
    from superglm.model.fit_data_guard import require_unchanged_fit_data

    require_unchanged_fit_data(model, X, y_ref)
    frame = as_eager_frame(X)
    fit_data_guard = model._fit_data_guard
    if not fit_data_guard.matches(
        X,
        y_ref,
        sample_weight,
        offset,
        fit_weights=getattr(model, "_fit_weights", None),
        fit_offset=getattr(model, "_fit_offset", None),
    ):
        raise RuntimeError(
            "post-fit sample_weight and offset must match the fitted data; "
            "refit before repairing with different scoring geometry"
        )
    if not any(_repair_changes_coefficients(model, repair) for repair in pending_repairs):
        # Shape-feasible terms are an exact publication no-op: no metadata,
        # revision, convergence flag, accepted REML state, or result identity changes.
        return model
    if any(group.monotone_engine == "scop" for group in model._groups):
        raise RuntimeError(
            "Post-fit shape repair cannot be combined with fitted SCOP terms until "
            "the repaired mode can rebuild one coherent joint SCOP inference geometry."
        )

    from superglm.model import state_ops
    from superglm.model.fit_state import fitted_penalty
    from superglm.solvers.rank import selected_group_name_set

    prior_working_weights = state_ops._solver_space_working_weights(model)
    prior_selected_names = selected_group_name_set(
        model._solver_result,
        model._groups,
        penalty=fitted_penalty(model),
    )

    revision = FittedStateRevision.start(model)
    work_model = revision.model
    # FittedStateRevision intentionally shares immutable/heavy fit projections.
    # Canonicalization metadata is nested and is revised term-by-term below, so
    # give this transaction its own copy before the first mutation.
    work_model._runtime_canonical_state = copy.deepcopy(model._runtime_canonical_state)
    work_model._shape_repairs = dict(getattr(model, "_shape_repairs", {}))
    work_model._monotone_repairs = dict(getattr(model, "_monotone_repairs", {}))

    scoring_weight = (
        sample_weight if sample_weight is not None else getattr(work_model, "_fit_weights", None)
    )
    scoring_weights_arr = np.asarray(scoring_weight, dtype=np.float64)
    scoring_offset = offset if offset is not None else getattr(work_model, "_fit_offset", None)

    selection_penalty = fitted_penalty(work_model)
    smooth_penalty_terms = _build_smooth_penalty_terms(work_model)
    current_eta = work_model._dm.matvec(work_model._solver_result.beta) + float(
        work_model._solver_result.intercept
    )
    if scoring_offset is not None:
        current_eta = current_eta + np.asarray(scoring_offset, dtype=np.float64)

    for name, spec, kind, groups in pending_repairs:
        beta = work_model.result.beta
        x_col = frame.column_array(name, dtype=np.float64)
        grid_weights = _grid_weights(spec, x_col, scoring_weights_arr, n_grid)

        repair_result = _repairer(kind).repair(
            spec,
            beta,
            groups,
            weights=grid_weights,
            n_grid=n_grid,
        )
        repair_result.feature_name = name

        candidate_beta, current_eta, intercept_shift = _validate_repair_for_publication(
            work_model,
            spec=spec,
            kind=kind,
            groups=groups,
            repair_result=repair_result,
            y=np.asarray(y_ref, dtype=np.float64),
            scoring_weights=scoring_weights_arr,
            smooth_penalty_terms=smooth_penalty_terms,
            selection_penalty=selection_penalty,
            current_eta=current_eta,
        )
        _replace_result_beta(work_model, candidate_beta)
        work_model._result.intercept = float(work_model._result.intercept) + intercept_shift
        _synchronize_repaired_intercept_state(work_model)
        work_model._shape_repairs[name] = repair_result
        if kind in {"increasing", "decreasing"}:
            work_model._monotone_repairs[name] = repair_result

    from superglm.model.fit_state import invalidate_revised_coefficient_mode

    invalidate_revised_coefficient_mode(work_model)
    _invalidate_repair_caches(work_model)
    from superglm.editor.apply import _refresh_fit_statistics

    _refresh_fit_statistics(
        work_model,
        X=X,
        y=y_ref,
        sample_weight=scoring_weight,
        offset=scoring_offset,
        use_fitted_design=True,
    )
    _refresh_repaired_geometry(work_model, prior_working_weights, prior_selected_names)
    _refresh_repaired_scale_and_statistics(work_model)
    return revision.commit()
