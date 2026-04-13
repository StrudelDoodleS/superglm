"""Inference, plotting delegation, and feature diagnostics."""

from __future__ import annotations

from superglm.inference._term_covariance import (
    feature_se_from_cov,
)
from superglm.inference._term_covariance import (
    simultaneous_bands as _simultaneous_bands,
)
from superglm.inference._term_model_ops import (
    drop1 as _drop1,
)
from superglm.inference._term_model_ops import (
    refit_unpenalised as _refit_unpenalised,
)
from superglm.inference._term_model_ops import (
    relativities as _relativities,
)
from superglm.inference._term_ops import (
    term_inference as _term_inference,
)


def metrics(model, X, y, sample_weight=None, offset=None):
    """Compute comprehensive diagnostics for the fitted model."""
    from superglm.inference.metrics import ModelMetrics

    cache_signature = (id(model.result), id(getattr(model, "_reml_penalties", None)))
    same_fit_refs = (
        X is getattr(model, "_fit_X_ref", None)
        and y is getattr(model, "_fit_y_ref", None)
        and sample_weight is getattr(model, "_fit_sample_weight_ref", None)
        and offset is getattr(model, "_fit_offset_ref", None)
    )
    if (
        same_fit_refs
        and model._fit_metrics_cache is not None
        and model._fit_metrics_cache_signature == cache_signature
    ):
        return model._fit_metrics_cache

    use_fit_mu = X is getattr(model, "_fit_X_ref", None) and (
        (offset is None and model._fit_offset is None)
        or offset is getattr(model, "_fit_offset_ref", None)
    )

    metrics_obj = ModelMetrics(
        model,
        X,
        y,
        sample_weight,
        offset,
        _mu=model._fit_mu if use_fit_mu else None,
        _null_mu=model._fit_null_mu if same_fit_refs else None,
        _fit_stats=model._fit_stats if same_fit_refs else None,
    )
    if same_fit_refs:
        model._fit_metrics_cache = metrics_obj
        model._fit_metrics_cache_signature = cache_signature
    return metrics_obj


def drop1(model, X, y, sample_weight=None, offset=None, test="Chisq"):
    """Drop-one deviance analysis for each feature."""
    return _drop1(model, X, y, offset=offset, test=test)


def refit_unpenalised(model, X, y, sample_weight=None, offset=None, keep_smoothing=True):
    """Refit with only active features and no selection penalty."""
    return _refit_unpenalised(
        model,
        X,
        y,
        sample_weight=sample_weight,
        offset=offset,
        keep_smoothing=keep_smoothing,
    )


def relativities(model, with_se=False, centering="native"):
    """Extract plot-ready relativity DataFrames for all features.

    By default returns the canonical fitted term contributions under
    the model's identifiability constraint (``centering="native"``).
    Pass ``centering="mean"`` to shift so geometric mean of
    relativities = 1 — a reporting convenience, not the fitted term.
    """
    return _relativities(
        model._feature_order,
        model._interaction_order,
        model._specs,
        model._interaction_specs,
        model._groups,
        model.result,
        with_se=with_se,
        covariance_fn=(lambda: model._coef_covariance) if with_se else None,
        centering=centering,
    )


def model_feature_se_from_cov(model, name, Cov_active, active_groups, n_points=200):
    """Compute feature-level SEs from the precomputed covariance matrix."""
    return feature_se_from_cov(
        name,
        Cov_active,
        active_groups,
        model.result,
        model._groups,
        model._specs,
        model._interaction_specs,
        n_points=n_points,
    )


def simultaneous_bands(model, feature, *, alpha=0.05, n_sim=10_000, n_points=200, seed=42):
    """Simultaneous confidence bands for a spline feature."""
    if model._result is None:
        raise RuntimeError("Model must be fitted before calling simultaneous_bands().")
    return _simultaneous_bands(
        feature,
        result=model.result,
        groups=model._groups,
        specs=model._specs,
        covariance_fn=lambda: model._coef_covariance,
        alpha=alpha,
        n_sim=n_sim,
        n_points=n_points,
        seed=seed,
    )


def term_inference(
    model,
    name,
    *,
    with_se=True,
    simultaneous=False,
    n_points=200,
    alpha=0.05,
    n_sim=10_000,
    seed=42,
    centering="native",
):
    """Per-term inference: curve, uncertainty, and metadata in one object."""
    if model._result is None:
        raise RuntimeError("Model must be fitted before calling term_inference().")
    return _term_inference(
        name,
        result=model.result,
        groups=model._groups,
        specs=model._specs,
        interaction_specs=model._interaction_specs,
        covariance_fn=lambda: model._coef_covariance,
        reml_lambdas=getattr(model, "_reml_lambdas", None),
        lambda2=model.lambda2,
        group_edf=model._group_edf,
        with_se=with_se,
        simultaneous=simultaneous,
        n_points=n_points,
        alpha=alpha,
        n_sim=n_sim,
        seed=seed,
        centering=centering,
    )


def term_importance(model, X, sample_weight=None):
    """Weighted variance of each term's contribution to eta."""
    from superglm.diagnostics.term_diagnostics import term_importance as _term_importance

    return _term_importance(model, X, sample_weight)


def term_drop_diagnostics(
    model,
    X,
    y,
    sample_weight=None,
    offset=None,
    *,
    mode="refit",
    X_val=None,
    y_val=None,
):
    """Drop-term diagnostics wrapper."""
    from superglm.diagnostics.term_diagnostics import (
        term_drop_diagnostics as _term_drop_diagnostics,
    )

    return _term_drop_diagnostics(
        model,
        X,
        y,
        sample_weight,
        offset,
        mode=mode,
        X_val=X_val,
        y_val=y_val,
    )


def spline_redundancy(model, X, sample_weight=None):
    """Spline redundancy diagnostics."""
    from superglm.diagnostics.spline_checks import spline_redundancy as _spline_redundancy

    return _spline_redundancy(model, X, sample_weight)


def discretization_impact(model, X, y, sample_weight=None, **kwargs):
    """Analyse the impact of discretizing spline/polynomial curves."""
    from superglm.diagnostics.discretize import discretization_impact as _disc_impact

    return _disc_impact(model, X, y, sample_weight, **kwargs)
