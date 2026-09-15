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

The default plots use blue effects, muted uncertainty bands, yellow support and
orange markers for free levels. Matplotlib main-effect plots place support in
a separate strip below each effect, with shared x limits and tick labels on the
bottom axis. Unordered categories use points and error bars; ordered spline
terms also show their fitted curve.

Pass `X` to show the observation distribution. With `sample_weight`, the strips
show weighted density or weight per level. Without it, they show observation
density or counts. Use `show_density=False` on `model.plot()` to hide them.
With frequency weights, bar heights are replication totals. With prior weights,
they are totals of the supplied precisions. The strips describe these supplied
weights; they do not change the fitted model's weight contract.

The returned Matplotlib figure remains editable, and plotting does not change
global Matplotlib settings. Plotly uses the same palette; `plotly_style` can
override its colours and sizes.

Main-effect figures keep constrained layout active so panels remain aligned
when resized. Adjust its padding with
`fig.get_layout_engine().set(h_pad=0.1, w_pad=0.1)`; `fig.subplots_adjust(...)`
is ignored while that engine is active. Manual `ax.set_position(...)` calls
remove the chosen axis from automatic layout and survive subsequent draws.
