# Plotting

The primary plotting entry point is [`SuperGLM.plot()`](model.md). The standalone
functions below are the underlying renderers and can be used directly with
`TermInference` objects for advanced customization.

For the public API, `engine="matplotlib"` is the chart/export path, while
`engine="plotly"` is the interactive multi-term main-effect explorer path.
Single-term main effects should use matplotlib.

Use `SuperGLM.plot_data()` when you want the plain effect / density / grid data
needed to recreate charts in your own plotting stack.

## Grouped categorical levels

Grouped categorical levels can be drawn as one visual group without changing
the model rows used for scoring, inference tables, or exports:

```python
model.plot("age_band", grouped_level_display="auto")
model.plot("age_band", grouped_level_display="expanded")
model.plot("age_band", grouped_level_display="collapsed")
```

`"auto"` collapses grouped ordered-categorical terms and leaves unordered
categoricals expanded. Collapsed exposure bars sum the original member levels.

::: superglm.plotting.plot_term

::: superglm.plotting.plot_relativities

::: superglm.plotting.plot_interaction

## Diagnostic plots

See [Diagnostics](diagnostics.md) for `plot_diagnostics()` — the GLM/GAM
4-panel diagnostic figure with simulation-based Q-Q envelopes.
