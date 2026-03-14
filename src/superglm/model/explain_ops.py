"""Inference, plotting delegation, and feature diagnostics."""

from __future__ import annotations

from superglm.inference import drop1 as _drop1
from superglm.inference import (
    feature_se_from_cov,
)
from superglm.inference import refit_unpenalised as _refit_unpenalised
from superglm.inference import relativities as _relativities
from superglm.inference import simultaneous_bands as _simultaneous_bands
from superglm.inference import term_inference as _term_inference


def metrics(model, X, y, exposure=None, offset=None, *, sample_weight=None):
    """Compute comprehensive diagnostics for the fitted model."""
    from superglm.metrics import ModelMetrics
    from superglm.model.base import resolve_sample_weight_alias

    exposure = resolve_sample_weight_alias(exposure, sample_weight, method_name="metrics()")
    return ModelMetrics(model, X, y, exposure, offset)


def drop1(model, X, y, exposure=None, offset=None, *, sample_weight=None, test="Chisq"):
    """Drop-one deviance analysis for each feature."""
    from superglm.model.base import resolve_sample_weight_alias

    exposure = resolve_sample_weight_alias(exposure, sample_weight, method_name="drop1()")
    return _drop1(model, X, y, exposure=exposure, offset=offset, test=test)


def refit_unpenalised(
    model, X, y, exposure=None, offset=None, *, sample_weight=None, keep_smoothing=True
):
    """Refit with only active features and no selection penalty."""
    from superglm.model.base import resolve_sample_weight_alias

    exposure = resolve_sample_weight_alias(
        exposure, sample_weight, method_name="refit_unpenalised()"
    )
    return _refit_unpenalised(
        model,
        X,
        y,
        exposure=exposure,
        offset=offset,
        keep_smoothing=keep_smoothing,
    )


def relativities(model, with_se=False):
    """Extract plot-ready relativity DataFrames for all features."""
    return _relativities(
        model._feature_order,
        model._interaction_order,
        model._specs,
        model._interaction_specs,
        model._groups,
        model.result,
        with_se=with_se,
        covariance_fn=(lambda: model._coef_covariance) if with_se else None,
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
    )


def resolve_ci(ci):
    """Normalize the ``ci`` parameter to an interval string or None."""
    if ci is None or ci is False:
        return None
    if ci is True:
        return "pointwise"
    valid = {"pointwise", "simultaneous", "both"}
    if ci not in valid:
        raise ValueError(
            f"ci={ci!r} is not valid. Expected one of {sorted(valid)}, None, or False."
        )
    return ci


def plot(
    model,
    terms=None,
    *,
    ci="pointwise",
    X=None,
    sample_weight=None,
    show_density=True,
    show_knots=False,
    engine="matplotlib",
    n_points=200,
    figsize=None,
    title=None,
    subtitle=None,
    alpha=0.05,
    n_sim=10_000,
    seed=42,
    **kwargs,
):
    """Plot model terms."""
    from superglm.plotting import plot_interaction, plot_relativities, plot_term

    if model._result is None:
        raise RuntimeError("Model must be fitted before calling plot().")

    # ── Normalize ci ────────────────────────────────────────
    interval = resolve_ci(ci)

    # ── Resolve terms ───────────────────────────────────────
    if terms is None:
        names = list(model._feature_order)
        mode = "all_main"
    elif isinstance(terms, str):
        names = [terms]
        in_main = terms in model._specs
        in_inter = terms in model._interaction_specs
        if in_main and in_inter:
            raise ValueError(
                f"Ambiguous term {terms!r}: exists as both a main effect "
                f"and an interaction. Use the feature or interaction spec "
                f"directly to disambiguate."
            )
        if in_inter:
            mode = "interaction"
        elif in_main:
            mode = "single_main"
        else:
            raise KeyError(f"Term not found: {terms!r}")
    else:
        names = list(terms)
        ambiguous = [n for n in names if n in model._specs and n in model._interaction_specs]
        if ambiguous:
            raise ValueError(
                f"Ambiguous term(s) {ambiguous}: exist as both main effects "
                f"and interactions. Use the feature or interaction spec "
                f"directly to disambiguate."
            )
        interactions = [n for n in names if n in model._interaction_specs]
        mains = [n for n in names if n in model._specs]
        unknown = [n for n in names if n not in model._specs and n not in model._interaction_specs]
        if unknown:
            raise KeyError(f"Term(s) not found: {unknown}")
        if interactions and mains:
            raise ValueError(
                "Cannot mix main effects and interactions in one plot() call. "
                f"Got main effects {mains} and interactions {interactions}."
            )
        mode = "interaction" if interactions else "multi_main"

    # ── Validate interaction count ──────────────────────────
    if mode == "interaction":
        if len(names) != 1:
            raise ValueError(
                f"plot() supports one interaction at a time. Got {len(names)}: {names}."
            )
        iname = names[0]

    # ── Dispatch ────────────────────────────────────────────
    if mode == "interaction":
        if engine not in ("matplotlib", "plotly"):
            raise ValueError(f"Unknown engine {engine!r}. Expected 'matplotlib' or 'plotly'.")
        return plot_interaction(
            model,
            iname,
            engine=engine,
            with_ci=(interval is not None),
            figsize=figsize,
            X=X,
            exposure=sample_weight,
            **kwargs,
        )

    if engine != "matplotlib":
        raise ValueError(
            f"engine={engine!r} is only supported for single interactions. "
            "Use engine='matplotlib' for main effects."
        )

    need_sim = interval in ("simultaneous", "both")
    ti_list = [
        model.term_inference(
            n,
            with_se=(interval is not None),
            simultaneous=need_sim,
            n_points=n_points,
            alpha=alpha,
            n_sim=n_sim,
            seed=seed,
        )
        for n in names
    ]

    if mode == "single_main":
        return plot_term(
            ti_list[0],
            X=X,
            exposure=sample_weight,
            interval=interval,
            show_exposure=show_density,
            show_knots=show_knots,
            figsize=figsize,
            title=title,
            subtitle=subtitle,
        )

    # all_main or multi_main → grid
    return plot_relativities(
        ti_list,
        X=X,
        exposure=sample_weight,
        interval=interval,
        show_exposure=show_density,
        show_knots=show_knots,
        title=title,
        subtitle=subtitle,
        figsize=figsize,
        **kwargs,
    )


def discretization_impact(model, X, y, exposure=None, *, sample_weight=None, **kwargs):
    """Analyse the impact of discretizing spline/polynomial curves."""
    from superglm.discretize import discretization_impact as _disc_impact
    from superglm.model.base import resolve_sample_weight_alias

    exposure = resolve_sample_weight_alias(
        exposure, sample_weight, method_name="discretization_impact()"
    )
    return _disc_impact(model, X, y, exposure, **kwargs)
