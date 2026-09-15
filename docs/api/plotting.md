# Plotting

Term comparison across models. The per-model plotting methods (`plot`,
`plot_data`, `plot_diagnostics`) live on `SuperGLM` and are documented on
the [model page](model.md).

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   superglm.plot_term_comparison
```

## Appearance and observation support

The default plots use blue effects, muted uncertainty bands, grey support and
orange markers for free levels. Matplotlib main-effect plots place support in
a separate strip below each effect, with shared x limits and tick labels on the
bottom axis. Unordered categories use points and error bars; ordered spline
terms also show their fitted curve.

Pass `X` to show the observation distribution. With `sample_weight`, the strips
show weighted density or weight per level. Without it, they show observation
density or counts. Use `show_density=False` on `model.plot()` to hide them.

The returned Matplotlib figure remains editable, and plotting does not change
global Matplotlib settings. Plotly uses the same palette; `plotly_style` can
override its colours and sizes.
