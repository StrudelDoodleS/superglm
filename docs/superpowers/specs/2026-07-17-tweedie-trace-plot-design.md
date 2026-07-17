# Cached Tweedie Trace Plot Design

## Goal

Add a fast plot of the evaluations already performed by Tweedie `estimate_p()` without
triggering any additional GLM fits or dispersion optimizations.

## Public API

`TweedieProfileResult.trace_plot(*, ax=None)` returns a Matplotlib `Figure`. When `ax` is
provided, it draws on that axis and returns its owning figure. This first version is
Matplotlib-only, matching the default `model.plot()` backend and the existing NB and Tweedie
profile plots. It adds no Seaborn or Plotly dependency and does not change `profile_plot()`.

## Data and Rendering

The method reads only the immutable `search_trace` snapshot. It sorts finite `p`/`nll` rows by
`p`, converts mean NLL to `2 * n * (nll - result.nll)`, and connects the actual evaluated points
with straight line segments. It performs no interpolation and does not imply that intermediate
points were evaluated. The winning `p_hat` is marked separately.

For exact-MLE dispersion profiles, the vertical axis is labelled profile deviance. Pearson
plug-in profiles use neutral profile-objective wording because likelihood-ratio interpretation
is not valid. Colours, grid weight, and reference-line styling reuse the existing SuperGLM
Matplotlib visual language from `superglm.plotting.common`.

## Errors and Side Effects

An empty trace or a trace without finite `p`/`nll` pairs raises a descriptive `RuntimeError`.
The method never calls `_objective`, `ci()`, or any fitting code. Calling it must leave
`n_total_evaluations`, `n_post_search_evaluations`, the CI caches, and `search_trace` unchanged.

## Tests and Documentation

Regression tests will verify the return type, supplied-axis behaviour, sorted plotted values,
MLE versus Pearson labels, finite-row handling, and the zero-evaluation invariant. The profile
likelihood documentation will show `result.trace_plot()` as the cheap diagnostic and retain
`result.profile_plot()` as the explicitly more expensive dense-profile plot.
