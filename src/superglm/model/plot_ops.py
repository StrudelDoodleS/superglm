"""Plotting and plot-data helpers for SuperGLM."""

from __future__ import annotations


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
    kind="global",
    ci="pointwise",
    X=None,
    sample_weight=None,
    show_density=True,
    show_knots=False,
    show_bases=False,
    scale="response",
    ci_style="band",
    categorical_display="auto",
    engine="matplotlib",
    n_points=200,
    figsize=None,
    title=None,
    subtitle=None,
    plotly_style=None,
    alpha=0.05,
    n_sim=10_000,
    seed=42,
    centering="native",
    **kwargs,
):
    """Plot model terms."""
    from superglm.plotting import plot_interaction, plot_relativities, plot_term

    if model._result is None:
        raise RuntimeError("Model must be fitted before calling plot().")

    if kind != "global":
        raise NotImplementedError(
            f"kind={kind!r} is not yet implemented. Only kind='global' is supported."
        )

    valid_scales = {"response", "link"}
    if scale not in valid_scales:
        raise ValueError(f"scale={scale!r} is not valid, expected one of {sorted(valid_scales)}")

    interval = resolve_ci(ci)

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

    if mode == "interaction":
        if len(names) != 1:
            raise ValueError(
                f"plot() supports one interaction at a time. Got {len(names)}: {names}."
            )
        interaction_name = names[0]

    if mode == "interaction":
        if engine not in ("matplotlib", "plotly"):
            raise ValueError(f"Unknown engine {engine!r}. Expected 'matplotlib' or 'plotly'.")
        return plot_interaction(
            model,
            interaction_name,
            engine=engine,
            with_ci=(interval is not None),
            figsize=figsize,
            n_points=n_points,
            X=X,
            sample_weight=sample_weight,
            **kwargs,
        )

    if engine not in ("matplotlib", "plotly"):
        raise ValueError(f"Unknown engine {engine!r}. Expected 'matplotlib' or 'plotly'.")

    if engine == "plotly" and mode in ("all_main", "single_main", "multi_main") and len(names) < 2:
        raise ValueError(
            "engine='plotly' is reserved for the multi-term main-effect explorer. "
            "Use terms=None or pass at least two main effects, or use "
            "engine='matplotlib' for a single-term chart."
        )

    need_simultaneous = interval in ("simultaneous", "both")
    term_inferences = [
        model.term_inference(
            name,
            with_se=(interval is not None),
            simultaneous=need_simultaneous,
            n_points=n_points,
            alpha=alpha,
            n_sim=n_sim,
            seed=seed,
            centering=centering,
        )
        for name in names
    ]

    if engine == "plotly":
        from superglm.plotting.main_effects_plotly import plot_main_effects_plotly

        return plot_main_effects_plotly(
            model,
            term_inferences,
            X=X,
            sample_weight=sample_weight,
            interval=interval,
            show_exposure=show_density,
            show_knots=show_knots,
            show_bases=show_bases,
            ci_style=ci_style,
            categorical_display=categorical_display,
            scale=scale,
            title=title,
            subtitle=subtitle,
            style=plotly_style,
        )

    if mode == "single_main":
        return plot_term(
            term_inferences[0],
            X=X,
            sample_weight=sample_weight,
            interval=interval,
            show_exposure=show_density,
            show_knots=show_knots,
            figsize=figsize,
            title=title,
            subtitle=subtitle,
        )

    return plot_relativities(
        term_inferences,
        X=X,
        sample_weight=sample_weight,
        interval=interval,
        show_exposure=show_density,
        show_knots=show_knots,
        title=title,
        subtitle=subtitle,
        figsize=figsize,
        **kwargs,
    )


def plot_data(
    model,
    terms=None,
    *,
    kind="global",
    ci="pointwise",
    X=None,
    sample_weight=None,
    show_density=True,
    show_knots=False,
    show_bases=False,
    n_points=200,
    alpha=0.05,
    n_sim=10_000,
    seed=42,
    centering="native",
):
    """Return plain plot-ready data for one or more terms."""
    from superglm.plotting.data import build_interaction_plot_data, build_main_effect_plot_data

    if model._result is None:
        raise RuntimeError("Model must be fitted before calling plot_data().")

    if kind != "global":
        raise NotImplementedError(
            f"kind={kind!r} is not yet implemented. Only kind='global' is supported."
        )

    interval = resolve_ci(ci)

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
                "Cannot mix main effects and interactions in one plot_data() call. "
                f"Got main effects {mains} and interactions {interactions}."
            )
        mode = "interaction" if interactions else "multi_main"

    if mode == "interaction":
        if len(names) != 1:
            raise ValueError(
                f"plot_data() supports one interaction at a time. Got {len(names)}: {names}."
            )
        return build_interaction_plot_data(
            model,
            names[0],
            n_points=n_points,
            X=X,
            sample_weight=sample_weight,
        )

    need_simultaneous = interval in ("simultaneous", "both")
    term_inferences = [
        model.term_inference(
            name,
            with_se=(interval is not None),
            simultaneous=need_simultaneous,
            n_points=n_points,
            alpha=alpha,
            n_sim=n_sim,
            seed=seed,
            centering=centering,
        )
        for name in names
    ]
    return build_main_effect_plot_data(
        model,
        term_inferences,
        X=X,
        sample_weight=sample_weight,
        show_density=show_density,
        show_knots=show_knots,
        show_bases=show_bases,
    )
