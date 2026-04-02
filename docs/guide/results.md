# Inspecting Results

## Summary table

Statsmodels-style summary with SEs, p-values, and smooth tests:

```python
m = model.metrics(df, y, sample_weight=exposure)
print(m.summary())
```

## Per-term inference

The `TermInference` dataclass holds everything about a single term: grid values, relativities, confidence intervals, spline metadata.

```python
ti = model.term_inference("DrivAge")

ti.x                        # evaluation grid (spline/polynomial) or levels (categorical)
ti.relativity               # exp(f(x)) relativity curve
ti.ci_lower, ti.ci_upper    # pointwise CI bounds
ti.edf                      # effective degrees of freedom
ti.spline                   # SplineMetadata (interior_knots, boundary_knots, basis_dim, ...)
```

By default this is the canonical fitted term contribution under the model's
identifiability constraint. If you want a rebased reporting view where the
geometric mean of relativities is 1, pass `centering="mean"` explicitly:

```python
ti = model.term_inference("DrivAge", centering="mean")
```

## Plotting

All plotting goes through `model.plot()`:

```python
# Single-term chart
model.plot("DrivAge", X=df, sample_weight=exposure)

# All terms in a grid
model.plot(X=df, sample_weight=exposure)

# Subset of terms
model.plot(["DrivAge", "VehAge"], X=df, sample_weight=exposure)

# Interactive Plotly main-effect explorer
model.plot(engine="plotly", X=df, sample_weight=exposure)

# Plotly subset explorer
model.plot(["DrivAge", "VehAge"], engine="plotly", X=df, sample_weight=exposure)

# Plotly interaction contour + exposure HDR view
model.plot(
    "DrivAge:VehAge",
    engine="plotly",
    interaction_view="contour_pair",
    X=df,
    sample_weight=exposure,
)

# Interaction
model.plot("DrivAge:Area")
```

`engine="matplotlib"` is the chart/export path. `engine="plotly"` is the multi-term main-effect explorer path and requires at least two main effects (or `terms=None`).

Options: `ci` (`"pointwise"`, `"simultaneous"`, `"both"`, `None`, `False`), `show_knots`, `show_density`, `title`, `subtitle`, `engine`.

## Plot data export

Use `model.plot_data()` when you need the underlying x/y/grid data to rebuild a
plot outside SuperGLM:

```python
# Main-effect data
payload = model.plot_data("DrivAge", X=df, sample_weight=exposure, show_knots=True)
curve_df = payload["terms"][0]["effect"]
density_df = payload["terms"][0]["density"]
knots_df = payload["terms"][0]["knots"]

# Continuous x continuous interaction grid
payload = model.plot_data("DrivAge:VehAge", X=df, sample_weight=exposure, n_points=220)
surface_df = payload["effect"]
hdr_df = payload["density"]  # includes density + hdr_mass columns
```

## Relativity DataFrames

For manual access or export:

```python
rels = model.relativities(with_se=True)  # canonical fitted-term view
# dict of {feature_name: DataFrame}
```

Use `centering="mean"` only when you explicitly want to rebase each term for
reporting or cross-feature comparison:

```python
rels = model.relativities(with_se=True, centering="mean")
```

## Families

| Family | Variance function | Use case |
|--------|------------------|----------|
| `Poisson()` | V(μ) = μ | Claim frequency |
| `NegativeBinomial(theta=1.0)` | V(μ) = μ + μ²/θ | Overdispersed frequency |
| `Gamma()` | V(μ) = μ² | Claim severity |
| `Tweedie(p=1.5)` | V(μ) = μᵖ | Pure premium (frequency × severity) |
