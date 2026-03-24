# Plotting

The primary plotting entry point is [`SuperGLM.plot()`](model.md). The standalone
functions below are the underlying renderers and can be used directly with
`TermInference` objects for advanced customization.

For the public API, `engine="matplotlib"` is the chart/export path, while
`engine="plotly"` is the interactive multi-term main-effect explorer path.
Single-term main effects should use matplotlib.

Use `SuperGLM.plot_data()` when you want the plain effect / density / grid data
needed to recreate charts in your own plotting stack.

::: superglm.plotting.plot_term

::: superglm.plotting.plot_relativities

::: superglm.plotting.plot_interaction
