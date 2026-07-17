"""Fitting logic: fit(), fit_path(), fit_reml(), and REML helpers."""

from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from superglm.distributions import (
    Distribution,
    NegativeBinomial,
    clip_mu,
    resolve_distribution,
)
from superglm.links import Link, stabilize_eta
from superglm.model import path_ops, runtime_canonicalize
from superglm.model.input_validation import validate_fit_input
from superglm.model.reml_execute import (
    optimize_reml_best,
    run_fixed_monotone_reml,
    run_scop_efs_reml,
)
from superglm.model.reml_finalize import finalize_reml_fit
from superglm.model.reml_ops import (
    model_compute_dW_deta,
    model_optimize_direct_reml,
    model_optimize_discrete_reml_cached_w,
    model_optimize_efs_reml,
    model_reml_direct_gradient,
    model_reml_direct_hessian,
    model_reml_laml_objective,
    model_reml_w_correction,
    model_run_reml_once,
)
from superglm.model.reml_setup import (
    collect_reml_groups,
    constraint_engine_flags,
    initialize_component_lambdas,
    inject_fixed_scop_lambdas,
    restore_qp_constraints,
    strip_qp_constraints,
)
from superglm.solvers.irls_direct import fit_irls_direct
from superglm.solvers.pirls import fit_pirls
from superglm.types import FitStats

logger = logging.getLogger(__name__)

_VALID_REML_INTERACTION_MODES = {"full", "fast_candidate"}
_VALID_RUNTIME_VALIDATION_MODES = {"auto", "full", "skip"}
_FAST_CANDIDATE_REML_ITER_CAP = 5
_AUTO_RUNTIME_VALIDATION_MAX_ROWS = 100_000

__all__ = [
    "PathResult",
    "fit",
    "fit_path",
    "fit_reml",
    "model_compute_dW_deta",
    "model_optimize_direct_reml",
    "model_optimize_discrete_reml_cached_w",
    "model_optimize_efs_reml",
    "model_reml_direct_gradient",
    "model_reml_direct_hessian",
    "model_reml_laml_objective",
    "model_reml_w_correction",
    "model_run_reml_once",
]


@dataclass
class PathResult:
    """Container for regularization path results."""

    lambda_seq: NDArray  # shape (n_lambda,)
    coef_path: NDArray  # shape (n_lambda, p)
    intercept_path: NDArray  # shape (n_lambda,)
    deviance_path: NDArray  # shape (n_lambda,)
    n_iter_path: NDArray  # shape (n_lambda,) — PIRLS iters per lambda
    converged_path: NDArray  # shape (n_lambda,) — bool
    edf_path: NDArray | None = None  # shape (n_lambda,) — effective df

    def to_frame(self):
        """Return path telemetry as a pandas DataFrame."""
        import pandas as pd

        data = {
            "step": np.arange(len(self.lambda_seq), dtype=int),
            "lambda": np.asarray(self.lambda_seq, dtype=np.float64),
            "deviance": np.asarray(self.deviance_path, dtype=np.float64),
            "n_iter": np.asarray(self.n_iter_path, dtype=int),
            "converged": np.asarray(self.converged_path, dtype=bool),
        }
        if self.edf_path is not None:
            data["edf"] = np.asarray(self.edf_path, dtype=np.float64)
        return pd.DataFrame(data)

    def to_telemetry(self) -> dict[str, list]:
        """Return path telemetry as JSON-serializable lists."""
        telemetry = {
            "lambda_seq": np.asarray(self.lambda_seq, dtype=np.float64).tolist(),
            "deviance_path": np.asarray(self.deviance_path, dtype=np.float64).tolist(),
            "n_iter_path": np.asarray(self.n_iter_path, dtype=int).tolist(),
            "converged_path": [bool(v) for v in self.converged_path],
            "intercept_path": np.asarray(self.intercept_path, dtype=np.float64).tolist(),
        }
        if self.edf_path is not None:
            telemetry["edf_path"] = np.asarray(self.edf_path, dtype=np.float64).tolist()
        return telemetry


def _compute_null_mu(
    y: NDArray,
    weights: NDArray,
    offset: NDArray | None,
    distribution: Distribution,
    link: Link,
) -> NDArray:
    """Null model prediction: intercept-only MLE, offset-aware."""
    from superglm.distributions import Binomial, Gaussian, clip_mu

    y_bar = float(np.average(y, weights=weights))
    if isinstance(distribution, Binomial):
        y_bar = np.clip(y_bar, 1e-3, 1 - 1e-3)
    elif isinstance(distribution, Gaussian):
        y_bar = float(y_bar)
    else:
        y_bar = max(y_bar, 1e-10)

    if offset is None or np.all(offset == 0):
        return np.full(len(y), y_bar)

    b0 = float(link.link(np.atleast_1d(y_bar))[0]) - float(np.average(offset, weights=weights))
    for _ in range(25):
        eta_null = stabilize_eta(b0 + offset, link)
        mu_null = clip_mu(link.inverse(eta_null), distribution)
        dmu = link.deriv_inverse(eta_null)
        V = distribution.variance(mu_null)
        score = float(np.sum(weights * (y - mu_null) * dmu / V))
        info = float(np.sum(weights * dmu**2 / V))
        step = score / max(info, 1e-10)
        b0 += step
        if abs(step) < 1e-8:
            break

    eta_null = stabilize_eta(b0 + offset, link)
    return clip_mu(link.inverse(eta_null), distribution)


def _compute_fit_stats(
    y: NDArray,
    mu: NDArray,
    weights: NDArray,
    offset: NDArray | None,
    distribution: Distribution,
    link: Link,
    phi: float,
    null_mu: NDArray | None = None,
) -> FitStats:
    """Compute scalar fit statistics from training arrays."""

    ll = distribution.log_likelihood(y, mu, weights, phi)
    if null_mu is None:
        null_mu = _compute_null_mu(y, weights, offset, distribution, link)

    null_ll = distribution.log_likelihood(y, null_mu, weights, phi)
    null_dev = float(np.sum(weights * distribution.deviance_unit(y, null_mu)))
    dev = float(np.sum(weights * distribution.deviance_unit(y, mu)))
    expl_dev = 1.0 - dev / null_dev if null_dev > 0 else 0.0

    V = distribution.variance(mu)
    pearson = float(np.sum(weights * (y - mu) ** 2 / V))

    return FitStats(
        log_likelihood=ll,
        null_log_likelihood=null_ll,
        null_deviance=null_dev,
        explained_deviance=expl_dev,
        pearson_chi2=pearson,
        n_obs=len(y),
    )


def _auto_detect_specs_if_needed(model, X, sample_weight) -> None:
    """Populate feature specs for spline shorthand configs before fitting."""
    if model._splines is not None and not model._specs:
        from superglm.model.base import auto_detect

        auto_detect(model, X, sample_weight)


def _required_fit_columns(model) -> tuple[str, ...]:
    """Return configured columns that must exist before feature construction."""
    names = [*model._feature_order, *(model._splines or [])]
    for interaction_name in model._interaction_order:
        names.extend(model._interaction_specs[interaction_name].parent_names)
    # Preserve the interaction API's more specific configuration error before
    # treating a parent name as merely a missing input column.  In fully
    # auto-detected models there is no configured feature universe yet, so the
    # builder remains responsible for resolving parents after detection.
    configured_features = set(model._feature_order) | set(model._splines or ())
    for left, right in model._pending_interactions:
        if configured_features:
            if left not in configured_features:
                raise ValueError(f"Parent feature not found: {left}")
            if right not in configured_features:
                raise ValueError(f"Parent feature not found: {right}")
        names.extend((left, right))
    return tuple(dict.fromkeys(names))


def _validate_entrypoint_input(model, X, y, sample_weight, offset):
    distribution = resolve_distribution(model.family)
    validated = validate_fit_input(
        X,
        y,
        sample_weight,
        offset,
        family=distribution,
        required_columns=_required_fit_columns(model),
        check_all_columns=model._splines is not None and not model._specs,
    )
    return validated.X, validated.y, validated.sample_weight, validated.offset


def _clear_profile_results(model) -> None:
    """Clear stale profile-estimation results from previous fits."""
    model._nb_profile_result = None
    model._tweedie_profile_result = None


def _clear_fit_inference_caches(model) -> None:
    """Drop cached post-fit inference state invalidated by a new fit."""
    model.__dict__.pop("_coef_covariance", None)
    model.__dict__.pop("_fit_active_info", None)
    model.__dict__.pop("_fit_inference_info", None)
    model.__dict__.pop("_group_edf", None)
    model._solver_result = None
    model._runtime_canonical_state = None
    model._fast_prediction_state = None
    model._prediction_plan = None
    model._fit_mu = None
    model._fit_null_mu = None
    model._fit_X_ref = None
    model._fit_y_ref = None
    model._fit_sample_weight_ref = None
    model._fit_offset_ref = None
    model._fit_metrics_cache = None
    model._fit_metrics_cache_signature = None
    model._summary_cache = None


def _clear_reml_state(model) -> None:
    """Clear stale REML state from previous fit_reml calls."""
    model._reml_lambdas = None
    model._reml_penalties = None
    model._reml_result = None
    model._reml_profile = None


def _store_fit_arrays(model, sample_weight, offset):
    """Persist training weights/offset arrays on the model and return them."""
    model._fit_weights = np.array(sample_weight)
    model._fit_offset = np.array(offset) if offset is not None else None
    model._fit_used_offset = offset is not None
    return model._fit_weights, model._fit_offset


def _make_reml_debug_recorder(
    model,
    *,
    y: NDArray,
    reml_groups,
    has_constraints: bool,
    has_qp_constraints: bool,
    has_scop_constraints: bool,
    max_reml_iter: int,
    reml_tol: float,
    pirls_tol: float,
    max_pirls_iter: int,
):
    """Create the private REML debug recorder when tracing is enabled."""
    from superglm._debug import get_debug_level
    from superglm.model.reml_debug import REMLDebugRecorder

    debug_level = get_debug_level()
    if debug_level <= 0:
        return None

    base_dir = Path(os.environ.get("SUPERGLM_DEBUG_DIR", ".superglm-debug"))
    run_id = f"fit_reml_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
    recorder = REMLDebugRecorder(debug_level, base_dir, run_id)
    recorder.write_run_metadata(
        {
            "run_id": run_id,
            "debug_level": debug_level,
            "method": "fit_reml",
            "family": type(model._distribution).__name__,
            "link": type(model._link).__name__,
            "discrete": bool(model._discrete),
            "n_obs": int(len(y)),
            "n_columns": int(model._dm.p),
            "n_groups": int(len(model._groups)),
            "reml_group_names": [group.name for _, group in reml_groups],
            "has_constraints": bool(has_constraints),
            "has_qp_constraints": bool(has_qp_constraints),
            "has_scop_constraints": bool(has_scop_constraints),
            "max_reml_iter": int(max_reml_iter),
            "reml_tol": float(reml_tol),
            "pirls_tol": float(pirls_tol),
            "max_pirls_iter": int(max_pirls_iter),
        }
    )
    return recorder


def _resolve_interaction_reml_mode(model, interaction_mode: str, max_reml_iter: int) -> dict:
    """Return profile metadata and the effective REML iteration limit."""
    if interaction_mode not in _VALID_REML_INTERACTION_MODES:
        valid = ", ".join(sorted(_VALID_REML_INTERACTION_MODES))
        raise ValueError(f"interaction_mode must be one of {{{valid}}}, got {interaction_mode!r}")

    requested = int(max_reml_iter)
    interaction_order = getattr(model, "_interaction_order", ()) or ()
    pending_interactions = getattr(model, "_pending_interactions", ()) or ()
    n_interactions = len(interaction_order) if interaction_order else len(pending_interactions)
    active = interaction_mode == "fast_candidate" and n_interactions > 0
    effective = min(requested, _FAST_CANDIDATE_REML_ITER_CAP) if active else requested
    return {
        "interaction_mode": interaction_mode,
        "interaction_candidate_active": bool(active),
        "n_interactions": int(n_interactions),
        "requested_max_reml_iter": requested,
        "effective_max_reml_iter": int(effective),
    }


def _validate_runtime_validation_mode(runtime_validation: str | bool) -> None:
    """Validate the runtime-validation mode before expensive fit work starts."""
    if isinstance(runtime_validation, bool):
        return
    if runtime_validation not in _VALID_RUNTIME_VALIDATION_MODES:
        valid = ", ".join(sorted(_VALID_RUNTIME_VALIDATION_MODES))
        raise ValueError(
            f"runtime_validation must be one of {{{valid}}} or a bool, got {runtime_validation!r}"
        )


def _resolve_runtime_validation(
    runtime_validation: str | bool,
    *,
    n_rows: int,
    interaction_candidate_active: bool,
) -> tuple[bool, str]:
    """Resolve runtime parity validation for post-fit canonicalization."""
    _validate_runtime_validation_mode(runtime_validation)
    if isinstance(runtime_validation, bool):
        return bool(runtime_validation), "explicit_full" if runtime_validation else "explicit_skip"

    if runtime_validation == "full":
        return True, "explicit_full"
    if runtime_validation == "skip":
        return False, "explicit_skip"

    if interaction_candidate_active:
        return False, "fast_candidate"
    if n_rows > _AUTO_RUNTIME_VALIDATION_MAX_ROWS:
        return False, "large_fit"
    return True, "auto_full"


def _canonicalize_fitted_model(
    model,
    profile: dict,
    *,
    runtime_validation: str | bool,
    n_rows: int,
) -> None:
    """Canonicalize the public runtime state and profile validation cost."""
    validate_runtime, reason = _resolve_runtime_validation(
        runtime_validation,
        n_rows=n_rows,
        interaction_candidate_active=bool(profile.get("interaction_candidate_active", False)),
    )
    profile["fit_runtime_canonicalize_validate"] = bool(validate_runtime)
    profile["fit_runtime_canonicalize_validate_reason"] = reason
    t0 = time.perf_counter()
    runtime_canonicalize.canonicalize_fitted_model(model, validate=validate_runtime)
    profile["fit_runtime_canonicalize_s"] = time.perf_counter() - t0


def _prime_fit_caches(
    model,
    *,
    X_ref,
    y_ref,
    sample_weight_ref,
    offset_ref,
    y_arr: NDArray,
) -> None:
    """Store fit-data caches for summary/metrics fast paths."""
    fit_space_result = (
        model._solver_pirls_result() if model._solver_result is not None else model.result
    )
    eta = model._dm.matvec(fit_space_result.beta) + fit_space_result.intercept
    if model._fit_offset is not None:
        eta = eta + model._fit_offset
    eta = stabilize_eta(eta, model._link)
    mu = clip_mu(model._link.inverse(eta), model._distribution)
    null_mu = _compute_null_mu(
        y_arr,
        model._fit_weights,
        model._fit_offset,
        model._distribution,
        model._link,
    )
    model._fit_mu = mu
    model._fit_null_mu = null_mu
    model._fit_X_ref = X_ref
    model._fit_y_ref = y_ref
    model._fit_sample_weight_ref = sample_weight_ref
    model._fit_offset_ref = offset_ref
    model._fit_metrics_cache = None
    model._fit_metrics_cache_signature = None
    model._summary_cache = None


def _maybe_release_fit_state(model) -> None:
    """Optionally retain only compact post-fit inference state."""
    if getattr(model, "_retain_fit_state", True):
        return

    # Distill the fit-time design into coefficient-space inference caches before
    # releasing row-scale state. These cached_property values are then returned
    # directly without needing model._dm.
    inf = model._fit_inference_info
    model.__dict__["_group_edf"] = dict(inf["group_edf_map"])
    model.__dict__["_coef_covariance"] = (
        model.result.phi * inf["XtWX_inv_aug"][1:, 1:],
        inf["active_groups"],
    )
    inf["W"] = np.empty(0, dtype=np.float64)

    model.__dict__.pop("_fit_active_info", None)
    model._dm = None
    model._fit_weights = None
    model._fit_offset = None
    model._fit_mu = None
    model._fit_null_mu = None
    model._fit_X_ref = None
    model._fit_y_ref = None
    model._fit_sample_weight_ref = None
    model._fit_offset_ref = None
    model._fit_metrics_cache = None
    model._fit_metrics_cache_signature = None
    model._summary_cache = None


def _maybe_estimate_nb_theta(model, X, y, sample_weight=None, offset=None) -> None:
    """Resolve auto-theta negative-binomial fits before building the design matrix."""
    if isinstance(model.family, NegativeBinomial) and model.family.theta == "auto":
        from superglm.profiling.nb import estimate_nb_theta

        nb_result = estimate_nb_theta(model, X, y, sample_weight=sample_weight, offset=offset)
        model.family = NegativeBinomial(theta=nb_result.theta_hat)
        model._nb_profile_result = nb_result
        logger.info(f"NB theta estimated: {nb_result.theta_hat:.4f}")


def fit(
    model,
    X,
    y,
    sample_weight=None,
    offset=None,
    *,
    tol=None,
    max_iter=None,
    convergence=None,
    record_diagnostics=False,
):
    """Fit the model to data."""
    X_ref = X
    y_ref = y
    sample_weight_ref = sample_weight
    offset_ref = offset
    X, y, sample_weight, offset = _validate_entrypoint_input(model, X, y, sample_weight, offset)
    # Resolve fit controls: explicit kwargs > constructor fallback
    tol = tol if tol is not None else model._tol
    max_iter = max_iter if max_iter is not None else model._max_iter
    convergence = convergence if convergence is not None else model._convergence

    # lambda_policy is only supported in fit_reml(); reject here.
    for name, spec in model._specs.items():
        lp = getattr(spec, "_lambda_policy", None)
        if lp is not None:
            raise NotImplementedError(
                f"lambda_policy on feature '{name}' is only supported with "
                f"fit_reml(), not fit(). Use fit_reml() or remove lambda_policy."
            )

    _auto_detect_specs_if_needed(model, X, sample_weight_ref)
    _clear_profile_results(model)

    _clear_reml_state(model)

    _maybe_estimate_nb_theta(model, X, y, sample_weight=sample_weight, offset=offset)

    from superglm.model.base import (
        compute_lambda_max,
        model_build_design_matrix,
        model_has_lambda1_targets,
    )

    y, sample_weight, offset = model_build_design_matrix(model, X, y, sample_weight, offset)

    sample_weight, offset = _store_fit_arrays(model, sample_weight, offset)

    # Auto-calibrate lambda1 if not set
    if model.penalty.lambda1 is None:
        model.penalty.lambda1 = compute_lambda_max(model, y, sample_weight) * 0.1
    has_lambda1_targets = model_has_lambda1_targets(model)

    # Invalidate cached properties from previous fit
    _clear_fit_inference_caches(model)

    # Monotone fit-time constraints are incompatible with selection_penalty (lambda1).
    # The constrained QP solver path ignores lambda1 — reject explicitly.
    if (
        any(g.monotone_engine is not None for g in model._groups)
        and model.penalty.lambda1 is not None
        and model.penalty.lambda1 > 0
        and has_lambda1_targets
    ):
        raise NotImplementedError(
            "Monotone fit-time constraints are not supported with selection_penalty > 0. "
            "Set selection_penalty=0 or fit unconstrained and call model.monotonize()."
        )

    # Guard: SCOP + QP monotone engines cannot coexist in the same model.
    _monotone_engines = {g.monotone_engine for g in model._groups if g.monotone_engine is not None}
    if len(_monotone_engines) > 1:
        raise NotImplementedError("SCOP + QP monotone terms in the same model are not supported.")

    # Direct IRLS when lambda1=0 (no L1 penalty → no BCD needed),
    # or when any group has monotone constraints (constrained QP / SCOP Newton).
    _has_constraints = any(g.constraints is not None for g in model._groups)
    _has_scop = any(g.monotone_engine == "scop" for g in model._groups)
    if (
        _has_constraints
        or _has_scop
        or (
            model.penalty.lambda1 is not None
            and (model.penalty.lambda1 == 0 or not has_lambda1_targets)
        )
    ):
        model._result, _ = fit_irls_direct(
            X=model._dm,
            y=y,
            weights=sample_weight,
            family=model._distribution,
            link=model._link,
            groups=model._groups,
            lambda2=model.lambda2,
            offset=offset,
            max_iter=max_iter,
            tol=tol,
            record_diagnostics=record_diagnostics,
            direct_solve=model._direct_solve,
            convergence=convergence,
        )
    else:
        model._result = fit_pirls(
            X=model._dm,
            y=y,
            weights=sample_weight,
            family=model._distribution,
            link=model._link,
            groups=model._groups,
            penalty=model.penalty,
            offset=offset,
            max_iter_outer=max_iter,
            tol=tol,
            active_set=model._active_set,
            lambda2=model.lambda2,
            record_diagnostics=record_diagnostics,
            convergence=convergence,
        )

    # Fix phi for known-scale families (Poisson): phi is always 1.0.
    scale_known = getattr(model._distribution, "scale_known", True)
    if scale_known and model._result.phi != 1.0:
        model._result = replace(model._result, phi=1.0)

    eta = model._dm.matvec(model._result.beta) + model._result.intercept
    if offset is not None:
        eta = eta + offset
    eta = stabilize_eta(eta, model._link)
    mu = clip_mu(model._link.inverse(eta), model._distribution)

    null_mu = _compute_null_mu(y, sample_weight, offset, model._distribution, model._link)
    model._fit_stats = _compute_fit_stats(
        y,
        mu,
        sample_weight,
        offset,
        model._distribution,
        model._link,
        model._result.phi,
        null_mu=null_mu,
    )
    model._solver_result = model._result
    _prime_fit_caches(
        model,
        X_ref=X_ref,
        y_ref=y_ref,
        sample_weight_ref=sample_weight_ref,
        offset_ref=offset_ref,
        y_arr=y,
    )
    runtime_canonicalize.canonicalize_fitted_model(model)

    model._last_fit_meta = {"method": "fit", "discrete": model._discrete}
    _maybe_release_fit_state(model)
    return model


def fit_path(
    model,
    X,
    y,
    sample_weight=None,
    offset=None,
    *,
    n_lambda=50,
    lambda_ratio=1e-3,
    lambda_seq=None,
):
    """Fit a regularization path from lambda_max down to lambda_min."""
    X_ref = X
    y_ref = y
    sample_weight_ref = sample_weight
    offset_ref = offset
    X, y, sample_weight, offset = _validate_entrypoint_input(model, X, y, sample_weight, offset)
    from superglm.model.base import (
        compute_lambda_max,
        model_build_design_matrix,
        model_has_lambda1_targets,
    )

    _auto_detect_specs_if_needed(model, X, sample_weight_ref)
    y, sample_weight, offset = model_build_design_matrix(model, X, y, sample_weight, offset)
    sample_weight, offset = _store_fit_arrays(model, sample_weight, offset)
    _clear_fit_inference_caches(model)
    _clear_reml_state(model)

    if not model_has_lambda1_targets(model):
        raise ValueError(
            "fit_path() requires at least one group targeted by the penalty. "
            "Adjust penalty.features or use fit() / fit_reml() instead."
        )
    lambda_max = compute_lambda_max(model, y, sample_weight)

    lambda_seq = path_ops.resolve_lambda_sequence(
        lambda_max,
        n_lambda=n_lambda,
        lambda_ratio=lambda_ratio,
        lambda_seq=lambda_seq,
    )
    n_lambda = len(lambda_seq)
    path_data = path_ops.run_lambda_path(
        model,
        y=y,
        sample_weight=sample_weight,
        offset=offset,
        lambda_seq=lambda_seq,
    )
    result = path_data["result"]

    # Set model state to the last (least-regularized) fit
    model._result = result

    eta = model._dm.matvec(result.beta) + result.intercept
    if offset is not None:
        eta = eta + offset
    eta = stabilize_eta(eta, model._link)
    mu = clip_mu(model._link.inverse(eta), model._distribution)
    null_mu = _compute_null_mu(y, sample_weight, offset, model._distribution, model._link)
    model._fit_stats = _compute_fit_stats(
        y,
        mu,
        sample_weight,
        offset,
        model._distribution,
        model._link,
        result.phi,
        null_mu=null_mu,
    )
    model._solver_result = result
    _prime_fit_caches(
        model,
        X_ref=X_ref,
        y_ref=y_ref,
        sample_weight_ref=sample_weight_ref,
        offset_ref=offset_ref,
        y_arr=y,
    )
    runtime_canonicalize.canonicalize_fitted_model(model)
    model._last_fit_meta = {"method": "fit_path", "discrete": model._discrete}
    intercept_path = runtime_canonicalize.canonicalize_intercept_path(
        model,
        path_data["coef_path"],
        path_data["intercept_path"],
    )
    intercept_path[-1] = model.result.intercept
    _maybe_release_fit_state(model)

    return PathResult(
        lambda_seq=lambda_seq,
        coef_path=path_data["coef_path"],
        intercept_path=intercept_path,
        deviance_path=path_data["deviance_path"],
        n_iter_path=path_data["n_iter_path"],
        converged_path=path_data["converged_path"],
        edf_path=path_data["edf_path"],
    )


def fit_reml(
    model,
    X,
    y,
    sample_weight=None,
    offset=None,
    *,
    max_reml_iter=20,
    reml_tol=1e-6,
    pirls_tol=1e-6,
    max_pirls_iter=100,
    lambda2_init=None,
    interaction_mode="full",
    runtime_validation="auto",
    verbose=False,
    w_correction_order=1,
):
    """Fit with REML estimation of per-term smoothing parameters."""
    X_ref = X
    y_ref = y
    sample_weight_ref = sample_weight
    offset_ref = offset
    X, y, sample_weight, offset = _validate_entrypoint_input(model, X, y, sample_weight, offset)
    from superglm.model.base import (
        model_build_design_matrix,
        model_has_lambda1_targets,
    )

    _auto_detect_specs_if_needed(model, X, sample_weight_ref)

    # Clear stale results from previous fit
    _clear_profile_results(model)
    model._reml_result = None
    model._reml_profile = None

    _maybe_estimate_nb_theta(model, X, y, sample_weight=sample_weight, offset=offset)

    import time as _time

    _t_total_start = _time.perf_counter()
    _profile: dict = {}
    _profile.update(_resolve_interaction_reml_mode(model, interaction_mode, max_reml_iter))
    _validate_runtime_validation_mode(runtime_validation)
    effective_max_reml_iter = _profile["effective_max_reml_iter"]

    _t0 = _time.perf_counter()
    y, sample_weight, offset = model_build_design_matrix(model, X, y, sample_weight, offset)
    _profile["dm_build_s"] = _time.perf_counter() - _t0

    sample_weight, offset = _store_fit_arrays(model, sample_weight, offset)
    _clear_fit_inference_caches(model)

    # lambda1=None means "no L1 selection" in the REML path — default to 0
    # so the direct IRLS optimizer (Newton REML) is used instead of BCD+EFS.
    if model.penalty.lambda1 is None:
        model.penalty.lambda1 = 0.0

    if model.penalty.lambda1 > 0 and model_has_lambda1_targets(model):
        raise ValueError(
            "fit_reml() requires selection_penalty=0. "
            "Use fit() / fit_path() for sparse selection, or use select=True on spline terms "
            "when you want REML-managed shrinkage."
        )

    reml_groups = collect_reml_groups(model._groups, model._dm.group_matrices)
    _has_monotone, _has_qp_monotone, _has_scop_monotone = constraint_engine_flags(model._groups)
    debug_recorder = _make_reml_debug_recorder(
        model,
        y=y,
        reml_groups=reml_groups,
        has_constraints=_has_monotone,
        has_qp_constraints=_has_qp_monotone,
        has_scop_constraints=_has_scop_monotone,
        max_reml_iter=effective_max_reml_iter,
        reml_tol=reml_tol,
        pirls_tol=pirls_tol,
        max_pirls_iter=max_pirls_iter,
    )

    if not reml_groups and not _has_monotone:
        logger.warning("fit_reml: no REML-eligible groups found, falling back to fit()")
        model._result = fit_pirls(
            X=model._dm,
            y=y,
            weights=sample_weight,
            family=model._distribution,
            link=model._link,
            groups=model._groups,
            penalty=model.penalty,
            offset=offset,
            active_set=model._active_set,
            lambda2=model.lambda2,
            tol=pirls_tol,
            max_iter_outer=max_pirls_iter,
        )
        eta = model._dm.matvec(model._result.beta) + model._result.intercept
        if offset is not None:
            eta = eta + offset
        eta = stabilize_eta(eta, model._link)
        mu = clip_mu(model._link.inverse(eta), model._distribution)
        null_mu = _compute_null_mu(y, sample_weight, offset, model._distribution, model._link)
        model._fit_stats = _compute_fit_stats(
            y,
            mu,
            sample_weight,
            offset,
            model._distribution,
            model._link,
            model._result.phi,
            null_mu=null_mu,
        )
        model._solver_result = model._result
        _prime_fit_caches(
            model,
            X_ref=X_ref,
            y_ref=y_ref,
            sample_weight_ref=sample_weight_ref,
            offset_ref=offset_ref,
            y_arr=y,
        )
        _canonicalize_fitted_model(
            model,
            _profile,
            runtime_validation=runtime_validation,
            n_rows=len(y),
        )
        model._last_fit_meta = {"method": "fit_reml", "discrete": model._discrete}
        _maybe_release_fit_state(model)
        return model

    # Build penalty components and caches (eigenstructure computed once)
    from superglm.reml.penalty_algebra import build_penalty_context

    reml_penalties, penalty_caches, penalty_ranks = build_penalty_context(
        model._dm.group_matrices,
        reml_groups,
    )

    # Initialize per-component lambdas (penalty-indexed, not term-indexed)
    # Partition into fixed (policy.mode == "fixed") and estimated components.
    lam_init = lambda2_init if lambda2_init is not None else model.lambda2
    lambdas, estimated_names = initialize_component_lambdas(reml_penalties, lam_init)
    _any_unfixed_scop = inject_fixed_scop_lambdas(model._groups, model._specs, lambdas)

    # QP monotone with auto lambda → two-stage passthrough heuristic:
    # Stage 1: run unconstrained REML (temporarily strip QP constraints)
    # Stage 2: constrained refit at estimated lambdas
    # This is a heuristic, not exact joint REML for constrained terms.
    _qp_passthrough = _has_qp_monotone and bool(estimated_names)

    # Stage 1 setup: temporarily disable QP constraints so REML runs fully
    # unconstrained. Save the original state to restore for stage 2.
    _qp_saved_state: list[tuple[int, object, object]] = []
    _qp_stripped = False
    if _qp_passthrough:
        _qp_saved_state = strip_qp_constraints(model._groups)
        _qp_stripped = True
        _has_monotone, _has_qp_monotone, _has_scop_monotone = constraint_engine_flags(model._groups)

    try:
        # Direct IRLS when lambda1=0 or unset (no L1 penalty -> no BCD needed)
        offset_arr = offset if offset is not None else np.zeros(len(y))
        lam1 = model.penalty.lambda1
        use_direct = lam1 is None or lam1 == 0 or not model_has_lambda1_targets(model)

        if _has_monotone and not _any_unfixed_scop and not estimated_names:
            run_fixed_monotone_reml(
                model,
                y=y,
                sample_weight=sample_weight,
                offset=offset,
                pirls_tol=pirls_tol,
                max_pirls_iter=max_pirls_iter,
                lambdas=lambdas,
                reml_penalties=reml_penalties,
                compute_fit_stats=_compute_fit_stats,
                debug_recorder=debug_recorder,
            )
            _prime_fit_caches(
                model,
                X_ref=X_ref,
                y_ref=y_ref,
                sample_weight_ref=sample_weight_ref,
                offset_ref=offset_ref,
                y_arr=y,
            )
            _canonicalize_fitted_model(
                model,
                _profile,
                runtime_validation=runtime_validation,
                n_rows=len(y),
            )
            logger.info(f"fit_reml (monotone, fixed lambdas): lambdas={lambdas}")
            _maybe_release_fit_state(model)
            return model

        if _any_unfixed_scop or (_has_scop_monotone and estimated_names):
            best = run_scop_efs_reml(
                model,
                y=y,
                sample_weight=sample_weight,
                offset=offset,
                offset_arr=offset_arr,
                lambdas=lambdas,
                estimated_names=estimated_names,
                lam_init=lam_init,
                reml_penalties=reml_penalties,
                max_reml_iter=effective_max_reml_iter,
                reml_tol=reml_tol,
                pirls_tol=pirls_tol,
                max_pirls_iter=max_pirls_iter,
                verbose=verbose,
                profile=_profile,
                total_start=_t_total_start,
                compute_fit_stats=_compute_fit_stats,
                debug_recorder=debug_recorder,
            )
            _prime_fit_caches(
                model,
                X_ref=X_ref,
                y_ref=y_ref,
                sample_weight_ref=sample_weight_ref,
                offset_ref=offset_ref,
                y_arr=y,
            )
            _canonicalize_fitted_model(
                model,
                _profile,
                runtime_validation=runtime_validation,
                n_rows=len(y),
            )
            logger.info(
                f"REML SCOP EFS converged={best.converged} in {best.n_reml_iter} iters, "
                f"lambdas={best.lambdas}"
            )
            _maybe_release_fit_state(model)
            return model

        best = optimize_reml_best(
            model,
            use_direct=use_direct,
            y=y,
            sample_weight=sample_weight,
            offset_arr=offset_arr,
            reml_groups=reml_groups,
            penalty_ranks=penalty_ranks,
            lambdas=lambdas,
            max_reml_iter=effective_max_reml_iter,
            reml_tol=reml_tol,
            verbose=verbose,
            penalty_caches=penalty_caches,
            profile=_profile,
            w_correction_order=w_correction_order,
            reml_penalties=reml_penalties,
            estimated_names=estimated_names,
            pirls_tol=pirls_tol,
            max_pirls_iter=max_pirls_iter,
            model_optimize_direct_reml=model_optimize_direct_reml,
            model_optimize_efs_reml=model_optimize_efs_reml,
            debug_recorder=debug_recorder,
        )
        lambdas, n_reml_iter, converged = finalize_reml_fit(
            model,
            best=best,
            use_direct=use_direct,
            reml_groups=reml_groups,
            reml_penalties=reml_penalties,
            y=y,
            sample_weight=sample_weight,
            offset=offset,
            offset_arr=offset_arr,
            max_pirls_iter=max_pirls_iter,
            pirls_tol=pirls_tol,
            qp_passthrough=_qp_passthrough,
            qp_saved_state=_qp_saved_state,
            profile=_profile,
            total_start=_t_total_start,
            compute_fit_stats=_compute_fit_stats,
        )
        _t0 = _time.perf_counter()
        _prime_fit_caches(
            model,
            X_ref=X_ref,
            y_ref=y_ref,
            sample_weight_ref=sample_weight_ref,
            offset_ref=offset_ref,
            y_arr=y,
        )
        _profile["fit_prime_caches_s"] = _time.perf_counter() - _t0

        _canonicalize_fitted_model(
            model,
            _profile,
            runtime_validation=runtime_validation,
            n_rows=len(y),
        )

        logger.info(f"REML converged={converged} in {n_reml_iter} iters, lambdas={lambdas}")
        _t0 = _time.perf_counter()
        _maybe_release_fit_state(model)
        _profile["fit_release_state_s"] = _time.perf_counter() - _t0
        _profile["total_s"] = _time.perf_counter() - _t_total_start
        return model
    finally:
        # Always restore QP constraints if stripped
        if _qp_stripped:
            restore_qp_constraints(model._groups, _qp_saved_state)
