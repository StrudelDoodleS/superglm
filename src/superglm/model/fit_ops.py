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

from superglm._utils import _explained_deviance
from superglm.distributions import (
    Distribution,
    NegativeBinomial,
    Tweedie,
    clip_mu,
    resolve_distribution,
)
from superglm.links import Link, stabilize_eta
from superglm.model import path_ops, runtime_canonicalize
from superglm.model.fit_state import (
    _install_fit_state,
    capture_fit_state,
    configured_family,
    configured_lambda2,
    configured_penalty,
)
from superglm.model.fit_workspace import FitWorkspace
from superglm.model.input_validation import validate_fit_input
from superglm.model.reml_execute import (
    optimize_reml_best,
    record_reml_terminal,
    run_fixed_monotone_reml,
    run_scop_efs_reml,
)
from superglm.model.reml_finalize import (
    _build_reml_reporting_support_state,
    finalize_reml_fit,
)
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
# Screening candidates are ranked, never published; under the 5-iteration cap
# the tight publication default cannot buy determination -- it only burns an
# extra Newton iteration and flips converged flags in screening logs.
_FAST_CANDIDATE_REML_TOL = 1e-6
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


def _immutable_path_array(values, *, dtype) -> NDArray:
    """Return a contiguous array whose immutable bytes backing cannot be reopened."""
    array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    # A bytes owner is intentional: write=False alone can be reversed with
    # setflags(write=True), while a bytes-backed view cannot be reopened.
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class PathResult:
    """Immutable container for regularization path results."""

    lambda_seq: NDArray  # shape (n_lambda,)
    coef_path: NDArray  # shape (n_lambda, p)
    intercept_path: NDArray  # shape (n_lambda,)
    deviance_path: NDArray  # shape (n_lambda,)
    n_iter_path: NDArray  # shape (n_lambda,) — PIRLS iters per lambda
    converged_path: NDArray  # shape (n_lambda,) — bool
    edf_path: NDArray | None = None  # shape (n_lambda,) — effective df

    def __post_init__(self) -> None:
        normalized = {
            "lambda_seq": _immutable_path_array(self.lambda_seq, dtype=np.float64),
            "coef_path": _immutable_path_array(self.coef_path, dtype=np.float64),
            "intercept_path": _immutable_path_array(self.intercept_path, dtype=np.float64),
            "deviance_path": _immutable_path_array(self.deviance_path, dtype=np.float64),
            "n_iter_path": _immutable_path_array(self.n_iter_path, dtype=np.int64),
            "converged_path": _immutable_path_array(self.converged_path, dtype=np.bool_),
            "edf_path": (
                None
                if self.edf_path is None
                else _immutable_path_array(self.edf_path, dtype=np.float64)
            ),
        }
        for name, value in normalized.items():
            object.__setattr__(self, name, value)

    def __reduce__(self):
        """Route pickle restoration through the immutable constructor boundary."""
        return (
            type(self),
            (
                self.lambda_seq,
                self.coef_path,
                self.intercept_path,
                self.deviance_path,
                self.n_iter_path,
                self.converged_path,
                self.edf_path,
            ),
        )

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


def _reject_lambda_policy_fit(model, method: str = "fit") -> None:
    """lambda_policy is only supported in fit_reml(); reject ordinary fits."""
    for name, spec in model._specs.items():
        if getattr(spec, "_lambda_policy", None) is not None:
            raise NotImplementedError(
                f"lambda_policy on feature '{name}' is only supported with "
                f"fit_reml(), not {method}(). Use fit_reml() or remove lambda_policy."
            )


def _reject_random_effect_selection_fit(model, method: str) -> None:
    """Keep structured variance components out of selection-penalty fit paths."""
    from superglm.features.factor_smooth import FactorSmooth
    from superglm.features.random_effect import RandomEffect

    random_effect_names = [
        name for name, spec in model._specs.items() if isinstance(spec, RandomEffect)
    ]
    factor_smooth_names = [
        name for name, spec in model._interaction_specs.items() if isinstance(spec, FactorSmooth)
    ]
    if random_effect_names:
        joined = ", ".join(repr(name) for name in random_effect_names)
        raise NotImplementedError(
            f"RandomEffect feature(s) {joined} are only supported with fit_reml(), not {method}()."
        )
    if factor_smooth_names:
        joined = ", ".join(repr(name) for name in factor_smooth_names)
        raise NotImplementedError(
            f"FactorSmooth interaction(s) {joined} are only supported with "
            f"fit_reml(), not {method}()."
        )


def _reject_structured_fit_constraints(model) -> None:
    """Refuse the one structured/constrained pairing that does not reach a mode.

    This used to refuse every fit-time shape constraint that shared a model with
    a ``RandomEffect`` or a ``FactorSmooth``. That was scope containment from the
    release which introduced both terms, not a derived result, and it was far
    wider than the algebra requires. A variance component and a smoothing
    parameter are the same object to the extended Fellner-Schall update (Wood &
    Fasiolo, 2017, Biometrics 73(4):1071-1081), whose founding case in Fellner
    (1986) is a penalty assembled from identity blocks -- exactly what a random
    effect contributes. Shape-constrained additive *mixed* models are the
    documented, supported combination in the reference implementation (Pya
    Arnqvist, 2024, arXiv:2403.09438 section 3).

    What the refusal was really standing in for was a plumbing defect: the EFS
    lambda step read ``PenaltyComponent.omega_ssp`` directly, which is ``None``
    for an identity penalty, so a random effect arrived at a matmul as a 0-d
    operand. That is fixed at the read: the EFS step now reduces through
    ``penalty_algebra.penalty_component_quadratic`` and ``penalty_component_trace``,
    which are kind-aware and never materialize a structured penalty.

    What survives is one specification, not one pair of term types: a SCOP
    constraint on a column that a ``basis="fs"`` factor smooth also spans. An
    ``fs`` factor smooth carries its own main effect, so such a model states that
    effect twice -- once confined to a shape cone and once free. The free copy
    absorbs whatever the constrained copy is forbidden to do, the two compensate
    for each other, and no coefficient mode is reached.

    A factor smooth spans BOTH its parents, and the duplication is reachable on
    either. Along ``variable`` the free copy is the marginal smooth (measured
    0/7); along ``group`` it is the per-level null-space blocks, which are a
    per-level effect of the grouping column, reachable when that column carries
    an ``OrderedCategorical`` SCOP constraint (measured 0/6). Testing only
    ``variable`` would let the second through.

    Everything adjacent converges (7/7 each): ``basis="sz"``, which excludes the
    main effect and so states it once; and an ``fs`` factor smooth sharing
    neither parent with a constrained term. The unconstrained version of the
    duplicated model converges too, which is what localises this to constrained
    duplication rather than to duplication or to SCOP-with-FactorSmooth.

    Selection stays on the DECLARATION rather than the spec's type, which is what
    lets a constraint declared on a spline inside a wrapper such as
    ``OrderedCategorical`` be seen at all.
    """
    from superglm.features._spline_build import (
        _uses_fit_time_scop_constraints,
        _uses_fit_time_shape_constraints,
    )
    from superglm.features.factor_smooth import FactorSmooth
    from superglm.features.ordered_categorical import OrderedCategorical

    def _engine_spec(spec):
        """The spec that answers the engine question, not merely the declaration.

        ``OrderedCategorical`` delegates its build to an inner spline, and it is
        the inner basis that picks the engine: ``ps`` routes through SCOP. The
        wrapper defines ``_build_monotone_constraints_raw`` unconditionally --
        deliberately, so the builder can find raw geometry through it -- and
        defines no SCOP method, so asking the wrapper reports QP for every basis
        including the SCOP ones. Resolve to the inner spline before asking.

        A pre-0.24 step pickle never reaches this function: the declaration test
        above reads ``constraint_kind`` through ``getattr(spec, name, default)``,
        which is the one spelling that swallows the ``AttributeError``
        ``_basis_spline`` raises, so such a spec answers ``None`` and is filtered
        out before the engine question is asked. The direct read here is
        therefore not what protects that case -- do not read it as a refusal of
        step pickles, which the guard does not perform. It is written directly
        only because there is no default that would be correct if one did arrive.
        """
        return spec._basis_spline if isinstance(spec, OrderedCategorical) else spec

    # SCOP wins engine precedence in the build path, so classify it first.
    scop_constrained = [
        name
        for name, spec in model._specs.items()
        if _uses_fit_time_shape_constraints(spec)
        and _uses_fit_time_scop_constraints(_engine_spec(spec))
    ]
    if not scop_constrained:
        return

    # A main-effect-carrying factor smooth duplicates a constrained effect on
    # EITHER of its two axes, so test both parents. Along ``variable`` the
    # duplicate is the marginal smooth; along ``group`` it is the per-level
    # null-space blocks, which are a per-level effect of the grouping column.
    # Both were measured non-convergent (0/7 and 0/6). ``sz`` excludes the main
    # effect, and a factor smooth sharing neither parent states nothing twice.
    constrained_set = set(scop_constrained)
    duplicated = [
        (sorted(set(spec.parent_names) & constrained_set), name)
        for name, spec in model._interaction_specs.items()
        if isinstance(spec, FactorSmooth)
        and spec.basis == "fs"
        and set(spec.parent_names) & constrained_set
    ]
    if not duplicated:
        return

    joined = ", ".join(
        f"{iname!r} (sharing shape-constrained {', '.join(repr(p) for p in parents)})"
        for parents, iname in duplicated
    )
    raise NotImplementedError(
        f"fit-time SCOP shape constraints are not supported with FactorSmooth "
        f"term(s) {joined}; a basis='fs' factor smooth carries its own main "
        "effect, so the constrained variable's effect is stated twice -- once "
        "confined to the shape cone and once free -- and no coefficient mode is "
        "reached. Use basis='sz', which excludes the main effect, or a QP-engine "
        "spline (BSplineSmooth or CubicRegressionSpline), or a post-fit shape "
        "constraint. A factor smooth of any other variable is unaffected, and a "
        "RandomEffect carries no such restriction."
    )


def _compute_null_mu(
    y: NDArray,
    weights: NDArray,
    offset: NDArray | None,
    distribution: Distribution,
    link: Link,
) -> NDArray:
    """Null model prediction: intercept-only MLE, offset-aware."""
    from superglm.distributions import Binomial, Gaussian, clip_mu
    from superglm.links import SqrtLink

    has_offset = offset is not None and not np.all(offset == 0)
    if has_offset and isinstance(link, SqrtLink):
        assert offset is not None
        # The inverse sqrt link has two eta branches. A single Fisher-scoring
        # start can converge to the wrong offset-dependent local solution, so
        # use the same line-searched direct IRLS solver as a public
        # intercept-only fit.
        from superglm.group_matrix import DesignMatrix

        null_design = DesignMatrix([], n=len(y), p=0)
        null_result, _ = fit_irls_direct(
            X=null_design,
            y=y,
            weights=weights,
            family=distribution,
            link=link,
            groups=[],
            lambda2=0.0,
            offset=offset,
            max_iter=100,
            tol=1e-10,
            direct_solve="gram",
            convergence="deviance",
            compute_rank_info=False,
            _return_working_system=True,
            _compute_fit_statistics=False,
            _compute_reml_geometry=False,
            _compute_scop_postfit_inference=False,
        )
        eta_null = stabilize_eta(null_result.intercept + offset, link)
        return clip_mu(link.inverse(eta_null), distribution)

    y_bar = float(np.average(y, weights=weights))
    if isinstance(distribution, Binomial):
        y_bar = np.clip(y_bar, 1e-3, 1 - 1e-3)
    elif isinstance(distribution, Gaussian):
        y_bar = float(y_bar)
    else:
        y_bar = max(y_bar, 1e-10)

    if not has_offset:
        return np.full(len(y), y_bar)

    assert offset is not None
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

    if null_mu is None:
        null_mu = _compute_null_mu(y, weights, offset, distribution, link)

    if isinstance(distribution, Tweedie):
        from superglm.profiling.tweedie import (
            _tweedie_logpdf_pair,
            _tweedie_pearson_contributions,
        )

        fitted_logpdf, null_logpdf = _tweedie_logpdf_pair(
            y,
            mu,
            null_mu,
            phi,
            distribution.p,
            weights=weights,
        )
        ll = float(np.sum(fitted_logpdf))
        null_ll = float(np.sum(null_logpdf))
        pearson = float(np.sum(weights * _tweedie_pearson_contributions(y, mu, distribution.p)))
    else:
        ll = distribution.log_likelihood(y, mu, weights, phi)
        null_ll = distribution.log_likelihood(y, null_mu, weights, phi)
        V = distribution.variance(mu)
        pearson = float(np.sum(weights * (y - mu) ** 2 / V))
    null_dev = float(np.sum(weights * distribution.deviance_unit(y, null_mu)))
    dev = float(np.sum(weights * distribution.deviance_unit(y, mu)))
    expl_dev = _explained_deviance(dev, null_dev, y, null_mu, weights)

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
    config = model._config
    configured_feature_names = [name for name, _ in config.feature_templates]
    names = [*configured_feature_names, *(config.splines or ())]
    for _, interaction in config.interaction_templates:
        names.extend(interaction.parent_names)
    # Preserve the interaction API's more specific configuration error for an
    # explicit feature mapping, which defines a closed feature universe.  The
    # spline shorthand only names columns to smooth; every other X column is
    # still eligible for auto-detection when the fit workspace is built.
    configured_features = set(configured_feature_names) if config.splines is None else set()
    for left, right in config.interactions:
        if configured_features:
            if left not in configured_features:
                raise ValueError(f"Parent feature not found: {left}")
            if right not in configured_features:
                raise ValueError(f"Parent feature not found: {right}")
        names.extend((left, right))
    return tuple(dict.fromkeys(names))


def _validate_entrypoint_input(model, X, y, sample_weight, offset):
    distribution = resolve_distribution(configured_family(model))
    config = model._config
    validated = validate_fit_input(
        X,
        y,
        sample_weight,
        offset,
        family=distribution,
        required_columns=_required_fit_columns(model),
        check_all_columns=config.splines is not None and not config.feature_templates,
    )
    if (
        not config.features_explicit
        and config.splines is None
        and not config.feature_templates
        and not config.interaction_templates
        and not config.interactions
        and validated.X.columns
    ):
        raise ValueError(
            "X has columns but no features were configured; pass features={...}, "
            "use splines=[] for legacy auto-detection, or pass features={} for "
            "an explicit intercept-only model"
        )
    return validated.X, validated.y, validated.sample_weight, validated.offset


def _validate_pirls_iteration_limit(max_iter) -> None:
    """Preserve the public solver-control error before model-shape validation."""
    if max_iter < 1:
        raise ValueError(f"max_iter must be at least 1, got {max_iter}")


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
    model._linear_system_state = None
    model._reporting_support_state = None
    model._runtime_canonical_state = None
    model._fast_prediction_state = None
    model._prediction_plan = None
    model._fit_mu = None
    model._fit_null_mu = None
    model._fit_X_ref = None
    model._fit_y_ref = None
    model._fit_sample_weight_ref = None
    model._fit_offset_ref = None
    model._fit_data_guard = None
    model._fit_geometry_guard = None
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
    # Survives fit-state release (unlike the arrays), so consumers that
    # default to the fit's weights can tell "unweighted fit" apart from
    # "weighted fit whose arrays were released" and refuse to silently
    # substitute unit weights in the latter case.
    model._fit_used_weights = bool(np.any(model._fit_weights != 1.0))
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
    reml_tol: float | None,
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
            # None = engine default; the resolved value is an engine concern.
            "reml_tol": None if reml_tol is None else float(reml_tol),
            "pirls_tol": float(pirls_tol),
            "max_pirls_iter": int(max_pirls_iter),
        }
    )
    return recorder


def _fetch_or_build_design(model, X, y, sample_weight, offset, cache: dict):
    """Serve the profile-search design from its cache, or build and fill it.

    The design matrix depends on the frame, the feature specs, and the
    weights -- never on the family's power -- so candidate fits at different
    powers share one build. Distribution and link DO depend on the family and
    are re-resolved on every hit. The cache self-verifies with a fixed-probe
    matvec: a served design that no longer reproduces its stored probe answer
    is dropped and rebuilt rather than trusted. Arrays are handed out as
    copies, matching the fresh arrays an uncached build returns per fit.
    """
    from superglm._blas_threads import allow_wide_design
    from superglm.distributions import resolve_distribution
    from superglm.links import resolve_link
    from superglm.model.base import model_build_design_matrix
    from superglm.model.fit_state import configured_family, configured_link

    if cache:
        dm = cache["dm"]
        if np.array_equal(dm.matvec(cache["probe"]), cache["expected_probe"]):
            distribution = resolve_distribution(configured_family(model))
            model._distribution = distribution
            model._link = resolve_link(configured_link(model), distribution)
            model._groups = cache["groups"]
            model._dm = dm
            # The build also teaches the spec objects (seen levels, spline
            # reparametrisations, tensor marginal geometry) and resolves
            # pending interactions; a fresh workspace materializes untaught
            # constructor templates, so restore the taught specs -- main
            # effects AND interactions -- alongside the design they describe.
            model._specs = cache["specs"]
            model._interaction_specs = cache["interaction_specs"]
            model._interaction_order = cache["interaction_order"]
            model._pending_interactions = ()
            allow_wide_design(dm.p)
            return (
                cache["y"].copy(),
                cache["sample_weight"].copy(),
                None if cache["offset"] is None else cache["offset"].copy(),
            )
        cache.clear()

    y_out, w_out, off_out = model_build_design_matrix(model, X, y, sample_weight, offset)
    has_constrained_group, _, _ = constraint_engine_flags(model._groups)
    if has_constrained_group:
        # The constrained REML path mutates GroupSlice objects in place
        # (strip_qp_constraints / restore_qp_constraints recompose against
        # the current group matrix), and the fixed-probe check certifies only
        # the design's matvec -- it cannot see a mutated group. Until the
        # bitwise-equivalence evidence covers constrained fixtures, the cache
        # stands down and every candidate builds fresh.
        return y_out, w_out, off_out
    probe = np.random.default_rng(0x5D).standard_normal(model._dm.p)
    cache.update(
        dm=model._dm,
        groups=model._groups,
        specs=model._specs,
        interaction_specs=model._interaction_specs,
        interaction_order=model._interaction_order,
        y=np.array(y_out, copy=True),
        sample_weight=np.array(w_out, copy=True),
        offset=None if off_out is None else np.array(off_out, copy=True),
        probe=probe,
        expected_probe=model._dm.matvec(probe),
    )
    return y_out, w_out, off_out


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
    mu: NDArray | None = None,
    null_mu: NDArray | None = None,
) -> None:
    """Store fit-data caches for summary/metrics fast paths."""
    from superglm.model.fit_data_guard import FitDataGuard, FitGeometryGuard

    guard_columns = list(model._feature_order)
    for name in model._interaction_order:
        guard_columns.extend(model._interaction_specs[name].parent_names)
    guard_columns = list(dict.fromkeys(guard_columns))

    if mu is None:
        fit_space_result = (
            model._solver_pirls_result() if model._solver_result is not None else model.result
        )
        eta = model._dm.matvec(fit_space_result.beta) + fit_space_result.intercept
        if model._fit_offset is not None:
            eta = eta + model._fit_offset
        eta = stabilize_eta(eta, model._link)
        mu = clip_mu(model._link.inverse(eta), model._distribution)
    if null_mu is None:
        null_mu = _compute_null_mu(
            y_arr,
            model._fit_weights,
            model._fit_offset,
            model._distribution,
            model._link,
        )
    model._fit_mu = mu
    model._fit_null_mu = null_mu
    nb_profile_result = getattr(model, "_nb_profile_result", None)
    if nb_profile_result is not None:
        model._nb_profile_result = nb_profile_result._published_with_data(
            y_arr,
            mu,
            model._fit_weights,
        )
    model._fit_X_ref = X_ref
    model._fit_y_ref = y_ref
    model._fit_sample_weight_ref = sample_weight_ref
    model._fit_offset_ref = offset_ref
    model._fit_data_guard = (
        FitDataGuard.capture(X_ref, y_arr, columns=tuple(guard_columns))
        if getattr(model, "_retain_fit_state", True)
        else None
    )
    model._fit_geometry_guard = FitGeometryGuard.capture(
        X_ref,
        y_arr,
        model._fit_weights,
        (
            np.zeros(len(y_arr), dtype=np.float64)
            if model._fit_offset is None
            else model._fit_offset
        ),
        columns=tuple(guard_columns),
    )
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
    augmented = inf["XtWX_inv_aug"]
    if hasattr(augmented, "scaled") and hasattr(augmented, "slopes"):
        coefficient_covariance = augmented.scaled(model.result.phi).slopes
    else:
        coefficient_covariance = model.result.phi * augmented[1:, 1:]
    model.__dict__["_coef_covariance"] = (
        coefficient_covariance,
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
    model._fit_data_guard = None
    model._fit_metrics_cache = None
    model._fit_metrics_cache_signature = None
    model._summary_cache = None


def _maybe_estimate_nb_theta(model, X, y, sample_weight=None, offset=None) -> None:
    """Resolve auto-theta negative-binomial fits before building the design matrix."""
    family = configured_family(model)
    if isinstance(family, NegativeBinomial) and family.theta == "auto":
        from superglm.profiling.nb import estimate_nb_theta

        nb_result = estimate_nb_theta(model, X, y, sample_weight=sample_weight, offset=offset)
        model.family = NegativeBinomial(theta=nb_result.theta_hat)
        model._nb_profile_result = nb_result
        logger.info(f"NB theta estimated: {nb_result.theta_hat:.4f}")


def _solve_coefficients(
    model,
    y,
    sample_weight,
    offset,
    *,
    penalty,
    lambda2,
    has_lambda1_targets,
    max_iter,
    tol,
    record_diagnostics,
    convergence,
):
    """Apply the ordinary fit policy for selecting the coefficient solver."""
    has_constraints = any(group.constraints is not None for group in model._groups)
    has_scop = any(group.monotone_engine == "scop" for group in model._groups)
    uses_direct_solver = (
        has_constraints
        or has_scop
        or (penalty.lambda1 is not None and (penalty.lambda1 == 0 or not has_lambda1_targets))
    )

    if uses_direct_solver:
        result, _ = fit_irls_direct(
            X=model._dm,
            y=y,
            weights=sample_weight,
            family=model._distribution,
            link=model._link,
            groups=model._groups,
            lambda2=lambda2,
            offset=offset,
            max_iter=max_iter,
            tol=tol,
            record_diagnostics=record_diagnostics,
            direct_solve=model._direct_solve,
            convergence=convergence,
        )
        return result

    return fit_pirls(
        X=model._dm,
        y=y,
        weights=sample_weight,
        family=model._distribution,
        link=model._link,
        groups=model._groups,
        penalty=penalty,
        offset=offset,
        max_iter_outer=max_iter,
        tol=tol,
        active_set=model._active_set,
        lambda2=lambda2,
        record_diagnostics=record_diagnostics,
        convergence=convergence,
    )


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
    """Fit through an attempt-local workspace and publish one complete state."""
    _validate_pirls_iteration_limit(max_iter)
    X_ref = X
    y_ref = y
    sample_weight_ref = sample_weight
    offset_ref = offset
    X, y, sample_weight, offset = _validate_entrypoint_input(model, X, y, sample_weight, offset)
    validated_inputs = (X, y, sample_weight, offset)
    workspace = FitWorkspace.start(model, mode="fit", validated_inputs=validated_inputs)
    _fit_in_workspace(
        workspace.model,
        X,
        y,
        sample_weight,
        offset,
        X_ref=X_ref,
        y_ref=y_ref,
        sample_weight_ref=sample_weight_ref,
        offset_ref=offset_ref,
        tol=tol,
        max_iter=max_iter,
        convergence=convergence,
        record_diagnostics=record_diagnostics,
    )
    candidate = capture_fit_state(
        workspace,
        model,
        revision=model._fit_revision + 1,
    )
    _install_fit_state(model, candidate)
    return model


def _fit_in_workspace(
    model,
    X,
    y,
    sample_weight,
    offset,
    *,
    X_ref,
    y_ref,
    sample_weight_ref,
    offset_ref,
    tol=None,
    max_iter=None,
    convergence=None,
    record_diagnostics=False,
):
    """Build, solve, and finalize an ordinary fit on private mutable state."""
    # Resolve fit controls: explicit kwargs > constructor fallback
    tol = tol if tol is not None else model._tol
    max_iter = max_iter if max_iter is not None else model._max_iter
    convergence = convergence if convergence is not None else model._convergence
    penalty = configured_penalty(model)
    lambda2 = configured_lambda2(model)

    _reject_random_effect_selection_fit(model, "fit")
    _reject_lambda_policy_fit(model, "fit")

    _auto_detect_specs_if_needed(model, X, sample_weight_ref)
    _clear_profile_results(model)

    _clear_reml_state(model)

    _maybe_estimate_nb_theta(model, X, y, sample_weight=sample_weight, offset=offset)

    from superglm.model.base import (
        model_build_design_matrix,
        model_has_lambda1_targets,
        resolve_selection_penalty_for_fit,
    )

    y, sample_weight, offset = model_build_design_matrix(model, X, y, sample_weight, offset)

    sample_weight, offset = _store_fit_arrays(model, sample_weight, offset)

    resolve_selection_penalty_for_fit(model, penalty, y, sample_weight)
    has_lambda1_targets = model_has_lambda1_targets(model)

    # Invalidate cached properties from previous fit
    _clear_fit_inference_caches(model)

    # Monotone fit-time constraints are incompatible with selection_penalty (lambda1).
    # The constrained QP solver path ignores lambda1 — reject explicitly.
    if (
        any(g.monotone_engine is not None for g in model._groups)
        and penalty.lambda1 is not None
        and penalty.lambda1 > 0
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

    model._result = _solve_coefficients(
        model,
        y,
        sample_weight,
        offset,
        penalty=penalty,
        lambda2=lambda2,
        has_lambda1_targets=has_lambda1_targets,
        max_iter=max_iter,
        tol=tol,
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
        mu=mu,
        null_mu=null_mu,
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
    """Fit a regularization path and atomically publish its final solution."""
    _validate_pirls_iteration_limit(model._config.max_iter)
    lambda_seq = path_ops.validate_lambda_path_controls(
        n_lambda=n_lambda,
        lambda_ratio=lambda_ratio,
        lambda_seq=lambda_seq,
    )
    X_ref = X
    y_ref = y
    sample_weight_ref = sample_weight
    offset_ref = offset
    X, y, sample_weight, offset = _validate_entrypoint_input(model, X, y, sample_weight, offset)
    validated_inputs = (X, y, sample_weight, offset)
    workspace = FitWorkspace.start(model, mode="fit_path", validated_inputs=validated_inputs)
    path_result = _fit_path_in_workspace(
        workspace.model,
        X,
        y,
        sample_weight,
        offset,
        X_ref=X_ref,
        y_ref=y_ref,
        sample_weight_ref=sample_weight_ref,
        offset_ref=offset_ref,
        n_lambda=n_lambda,
        lambda_ratio=lambda_ratio,
        lambda_seq=lambda_seq,
    )
    candidate = capture_fit_state(
        workspace,
        model,
        revision=model._fit_revision + 1,
    )
    _install_fit_state(model, candidate)
    return path_result


def _fit_path_in_workspace(
    model,
    X,
    y,
    sample_weight,
    offset,
    *,
    X_ref,
    y_ref,
    sample_weight_ref,
    offset_ref,
    n_lambda=50,
    lambda_ratio=1e-3,
    lambda_seq=None,
):
    """Build and solve a regularization path on private mutable state."""
    _reject_random_effect_selection_fit(model, "fit_path")
    from superglm.model.base import (
        compute_lambda_max,
        model_build_design_matrix,
        model_has_lambda1_targets,
    )

    _auto_detect_specs_if_needed(model, X, sample_weight_ref)
    # A path sweeps lambda1 over a positive grid on ONE design, so the design
    # has to be the one a positive lambda1 requires, whatever the model was
    # configured with.
    y, sample_weight, offset = model_build_design_matrix(
        model, X, y, sample_weight, offset, selection_active=True
    )
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
        mu=mu,
        null_mu=null_mu,
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

    path_result = PathResult(
        lambda_seq=lambda_seq,
        coef_path=path_data["coef_path"],
        intercept_path=intercept_path,
        deviance_path=path_data["deviance_path"],
        n_iter_path=path_data["n_iter_path"],
        converged_path=path_data["converged_path"],
        edf_path=path_data["edf_path"],
    )
    return path_result


def fit_reml(
    model,
    X,
    y,
    sample_weight=None,
    offset=None,
    *,
    max_reml_iter=20,
    reml_tol=None,
    pirls_tol=1e-6,
    max_pirls_iter=100,
    lambda2_init=None,
    interaction_mode="full",
    runtime_validation="auto",
    verbose=False,
    w_correction_order=1,
):
    """Fit REML in a private workspace and atomically publish success."""
    from superglm.reml.w_derivatives import validate_w_correction_order

    _validate_pirls_iteration_limit(max_pirls_iter)
    w_correction_order = validate_w_correction_order(w_correction_order)
    X_ref = X
    y_ref = y
    sample_weight_ref = sample_weight
    offset_ref = offset
    X, y, sample_weight, offset = _validate_entrypoint_input(model, X, y, sample_weight, offset)
    validated_inputs = (X, y, sample_weight, offset)
    workspace = FitWorkspace.start(model, mode="fit_reml", validated_inputs=validated_inputs)
    debug_recorder = _fit_reml_in_workspace(
        workspace.model,
        X,
        y,
        sample_weight,
        offset,
        X_ref=X_ref,
        y_ref=y_ref,
        sample_weight_ref=sample_weight_ref,
        offset_ref=offset_ref,
        max_reml_iter=max_reml_iter,
        reml_tol=reml_tol,
        pirls_tol=pirls_tol,
        max_pirls_iter=max_pirls_iter,
        lambda2_init=lambda2_init,
        interaction_mode=interaction_mode,
        runtime_validation=runtime_validation,
        verbose=verbose,
        w_correction_order=w_correction_order,
    )
    candidate = capture_fit_state(
        workspace,
        model,
        revision=model._fit_revision + 1,
    )
    _install_fit_state(model, candidate)
    _record_reml_terminal_best_effort(model, debug_recorder)
    return model


def _record_reml_terminal_best_effort(model, debug_recorder) -> None:
    """Emit a post-install terminal without letting diagnostic I/O escape."""
    try:
        # The dictionary swap above is the sole public commit point.  Trace
        # persistence is external diagnostic I/O and cannot be part of that
        # transaction, so a sink failure must not make a successful fit look
        # like a failed operation after its state has already been installed.
        record_reml_terminal(model, debug_recorder)
    except Exception:
        try:
            logger.warning(
                "fit_reml completed but terminal trace emission failed",
                exc_info=True,
            )
        except Exception:
            # A custom logging handler is external I/O too. The fitted state
            # has already been installed and must remain a successful return.
            pass


def _fit_reml_in_workspace(
    model,
    X,
    y,
    sample_weight,
    offset,
    *,
    X_ref,
    y_ref,
    sample_weight_ref,
    offset_ref,
    max_reml_iter=20,
    reml_tol=None,
    pirls_tol=1e-6,
    max_pirls_iter=100,
    lambda2_init=None,
    interaction_mode="full",
    runtime_validation="auto",
    verbose=False,
    w_correction_order=1,
    durable_retain_fit_state: bool | None = None,
):
    """Run the complete REML attempt on private mutable model state."""
    from superglm.model.base import (
        model_build_design_matrix,
        model_has_lambda1_targets,
        resolve_selection_penalty_for_reml,
    )

    penalty = configured_penalty(model)
    resolve_selection_penalty_for_reml(penalty)

    # Clear stale results from previous fit
    _clear_profile_results(model)
    model._reml_result = None
    model._reml_profile = None

    _auto_detect_specs_if_needed(model, X, sample_weight_ref)
    _reject_structured_fit_constraints(model)
    _maybe_estimate_nb_theta(model, X, y, sample_weight=sample_weight, offset=offset)
    configured_smoothing = configured_lambda2(model)

    import time as _time

    _t_total_start = _time.perf_counter()
    _profile: dict = {}
    _profile.update(_resolve_interaction_reml_mode(model, interaction_mode, max_reml_iter))
    _validate_runtime_validation_mode(runtime_validation)
    effective_max_reml_iter = _profile["effective_max_reml_iter"]
    if _profile["interaction_candidate_active"] and reml_tol is None:
        reml_tol = _FAST_CANDIDATE_REML_TOL

    _t0 = _time.perf_counter()
    _design_cache = getattr(model, "_profile_design_cache", None)
    if isinstance(_design_cache, dict):
        y, sample_weight, offset = _fetch_or_build_design(
            model, X, y, sample_weight, offset, _design_cache
        )
    else:
        y, sample_weight, offset = model_build_design_matrix(model, X, y, sample_weight, offset)
    _profile["dm_build_s"] = _time.perf_counter() - _t0

    sample_weight, offset = _store_fit_arrays(model, sample_weight, offset)
    _clear_fit_inference_caches(model)

    reml_groups = collect_reml_groups(model._groups, model._dm.group_matrices)
    _has_monotone, _has_qp_monotone, _has_scop_monotone = constraint_engine_flags(model._groups)
    if _has_qp_monotone and _has_scop_monotone:
        raise NotImplementedError("SCOP + QP monotone terms in the same model are not supported.")
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
        model._result = _solve_coefficients(
            model,
            y,
            sample_weight,
            offset,
            penalty=penalty,
            lambda2=configured_smoothing,
            has_lambda1_targets=model_has_lambda1_targets(model),
            tol=pirls_tol,
            max_iter=max_pirls_iter,
            record_diagnostics=False,
            convergence=model._convergence,
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
            mu=mu,
            null_mu=null_mu,
        )
        _canonicalize_fitted_model(
            model,
            _profile,
            runtime_validation=runtime_validation,
            n_rows=len(y),
        )
        model._last_fit_meta = {"method": "fit_reml", "discrete": model._discrete}
        _maybe_release_fit_state(model)
        return debug_recorder

    # Build penalty components and caches (eigenstructure computed once)
    from superglm.reml.penalty_algebra import build_penalty_context

    reml_penalties, penalty_caches, penalty_ranks = build_penalty_context(
        model._dm.group_matrices,
        reml_groups,
    )

    # Initialize per-component lambdas (penalty-indexed, not term-indexed)
    # Partition into fixed (policy.mode == "fixed") and estimated components.
    lam_init = lambda2_init if lambda2_init is not None else configured_smoothing
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
        lam1 = penalty.lambda1
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
                profile=_profile,
                total_start=_t_total_start,
                debug_recorder=debug_recorder,
            )
            model._reporting_support_state = _build_reml_reporting_support_state(
                model,
                result=model._solver_result,
                y=y,
                sample_weight=sample_weight,
                offset_arr=offset_arr,
                durable_retain_fit_state=durable_retain_fit_state,
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
            return debug_recorder

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
            model._reporting_support_state = _build_reml_reporting_support_state(
                model,
                result=model._solver_result,
                y=y,
                sample_weight=sample_weight,
                offset_arr=offset_arr,
                durable_retain_fit_state=durable_retain_fit_state,
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
            return debug_recorder

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
            trace_run=getattr(debug_recorder, "trace_run", None),
            durable_retain_fit_state=durable_retain_fit_state,
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
        return debug_recorder
    finally:
        # Always restore QP constraints if stripped
        if _qp_stripped:
            restore_qp_constraints(model, _qp_saved_state)
