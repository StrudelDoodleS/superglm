# Inspecting Results

## Summary table

Statsmodels-style summary with SEs, p-values, and smooth tests:

```python
print(model.summary())                          # exact original levels, expanded
print(model.summary(level_display="grouped"))  # one fitted row per group + legend
```

Categorical levels are expanded by default, including reference levels. If fitted levels have been
collapsed together, each exact original label remains visible and a separate `Level group` column
shows IDs such as `G1`. Grouped mode shows one row per fitted group and a membership legend instead.
These IDs restart within each feature, exist only for presentation, and never rename level labels.

The same `level_display="expanded"` or `"grouped"` option is available from a metrics summary:

```python
m = model.metrics(df, y, sample_weight=exposure)
print(m.summary(level_display="grouped"))
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

For a spline-backed `OrderedCategorical`, inference is a single whole-smooth Wood test. Its ordered
level rows report effect estimates, standard errors, and confidence intervals for interpretation;
they intentionally do not present separate level p-values or significance stars.

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

## Rating table export

Use `export_rating_tables()` to create deployment-oriented Excel tables with
binned spline effects:

```python
model.export_rating_tables(
    "rating_tables.xlsx",
    X_train,
    y_train,
    sample_weight=exposure_train,
    n_bins=150,
)
```

The workbook includes selected-bin rating tables, a discretization impact sweep for
`20, 50, 100, 200, 250` bins, and a structured Model Summary sheet. The `ModelOverview` and
`TermInference` Excel tables use typed cells for metrics, estimates, intervals, and p-values instead
of storing the formatted console summary as one text column. The editor exposes the same workbook
through **Export > Excel rating workbook** when explicit training or retained fit data is available.

Excel export remains expanded over the exact original categorical levels, regardless of the most
recent summary view. `export_rating_tables()` does not accept `level_display` and does not emit the
summary-only `Level group` column or `G1` metadata.

## Families

| Family | Variance function | Use case |
|--------|------------------|----------|
| `Poisson()` | V(μ) = μ | Claim frequency |
| `NegativeBinomial(theta=1.0)` | V(μ) = μ + μ²/θ | Overdispersed frequency |
| `Gamma()` | V(μ) = μ² | Claim severity |
| `Tweedie(p=1.5)` | V(μ) = μᵖ | Pure premium (frequency × severity) |
