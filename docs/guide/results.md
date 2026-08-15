# Inspecting Results

The examples on this page assume the Poisson rate workflow, where `exposure`
is a case/frequency weight. Weight semantics differ by family; see
[Families & Dispersion](families.md#weight-semantics) before carrying this
spelling into Gaussian, Gamma, or Tweedie diagnostics.

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

## Diagnostic weight semantics

Quantile residuals and diagnostic simulations follow the same family-specific
contract as fitting:

- for non-Tweedie case/frequency weights, the row's response distribution is
  unchanged by its weight; the weight represents repeated likelihood
  contribution rather than smaller row variance
- for Tweedie EDM prior weights, row \(i\) has observation-specific dispersion
  \(\phi / w_i\), which is used by both its CDF residual and simulation

For exact discrete Poisson quantile residuals, diagnose raw claim counts with
`log(exposure)` as an offset. A fractional rate response with exposure as a
frequency weight is useful for fitting, but the case/frequency API cannot
reconstruct that row's count distribution from `sample_weight`.

The diagnostic Pearson denominator follows the same split:
`sum(sample_weight) - edf` for non-Tweedie families and `n - edf` for
Tweedie.

```python
fig = model.plot_diagnostics(df, y, sample_weight=exposure)
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

## Credibility reports

Random effects expose the fitted variance component and one row per level:

```python
brand = model.random_effects("VehBrand", exposure=train["Exposure"])
brand.tau_squared
brand.effective_df
brand.table[["level", "effect", "posterior_se", "credibility", "shrinkage"]]
```

Factor smooths expose the shared smoothing parameters, normalized
block-credibility summaries, and posterior curves:

```python
region_age = model.factor_smooth(
    "DrivAge:Region:fs",
    grid=80,
    levels=["R11", "R24"],
)
region_age.lambdas
region_age.table[["level", "effective_df", "credibility", "sufficient_support"]]
region_age.curves[["level", "DrivAge", "effect", "posterior_se", "lower", "upper"]]
```

See [Credibility terms](credibility.md) for the definitions and interpretation.

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

The exported table is **multiplicative**: a base relativity times one relativity per block
reproduces `model.predict` row by row. That only holds under a **log link**, because only then is
the mean a product of per-term factors — so the export is restricted to log-link models and raises
`ValueError` for anything else. Under an identity link the same arithmetic would return
`exp(linear predictor)` rather than the prediction, and under a logit link it would return the
odds. Poisson, Gamma, Tweedie and negative-binomial fits all take the log link by default and are
unaffected; a Gaussian fit left on its own default link is refused.

`Binomial` is refused outright, whatever its link. `model.predict` finishes by clamping a binomial
mean into `[1e-7, 1 - 1e-7]`, and a clamp is not a factor, so no multiplicative table can carry it.
Even on a log link the two agree only for `-16.118 <= eta <= -1e-7` — 20.1% of the permitted range —
and outside it the workbook returns a "probability" above one, or below the clamp, while the model
returns the clamped value. Predicting above one out of sample is the characteristic behaviour of
log-binomial regression, so this is refused by family rather than checked per frame: the exported
table is applied to risks the export never saw.

For the same reason the export refuses a fit that **saturates**. `model.predict` clips a log-link
predictor to `[-80, 80]`, and a clip is not a factor either; a frame that already reaches it would
produce a workbook that silently disagrees with the model it came from. Such a fit is
quasi-separated or mis-scaled — refit or rescale rather than export it. The same applies to every
**individual relativity**, on a main-effect block and on an interaction cell alike: a factor a
consumer cannot multiply by — `inf`, `nan`, `0.0`, negative, subnormal, or at or beyond the
`exp(±500)` range `superglm` clips confidence bounds to — is refused rather than exported. Two
blocks whose contributions cancel can leave the prediction healthy while neither factor is the
model's, so this is checked per block rather than on the product. A factor of exactly `0.0` is
refused for the same reason an infinite base is: it zeroes every premium it touches while every
relativity *ratio* on the sheet still reads correctly.

Bin boundaries and interaction axis values are printed at **round-trip precision** — the string in
the workbook converts back to exactly the number the model binned on, so applying the printed table
puts every risk in the same bin the model did.

The **offset multiplier** block is exact only while the fit carries fewer than 20 distinct offset
multipliers. Above that — the normal case for a continuous exposure — it is binned like a spline
block, with rows keyed on interval strings and each carrying its bin's exposure-weighted average, so
it is a summary rather than a per-row lookup. A bin with no exposure — reachable with
`bin_strategy="uniform"` on a skewed exposure — reports the midpoint of its own interval at weight
zero. Pass `offset_source=` to export the exact form, keyed on a raw column of the frame.

The workbook includes selected-bin rating tables, a discretization impact sweep, and a structured
Model Summary sheet. The sweep runs the `impact_bins` ladder — `20, 50, 100, 200, 250` by default —
**plus the `n_bins` the workbook was actually exported at**, marked by the sheet's `exported` column.
That row is the one that describes the table in hand: the error falls as bins rise, so without it a
reader taking the smallest number on the sheet as their bound got one below their own. Passing
`impact_bins=()` still skips the sheet entirely.

The sweep covers every block the workbook approximates: main-effect spline and polynomial terms, and
a continuous-by-continuous interaction, which is sampled onto an `n_bins`-per-axis grid rather than
binned into intervals. Its row is keyed on the interaction's name and its `actual_bins` counts the
grid's *cells*, so an interaction swept at 20 reports 400 where a main effect reports 20. The metric
columns are **joint** over everything discretized at that resolution — within one `n_bins` group only
`feature` and `actual_bins` vary — which is what makes them the bound for a consumer applying the
whole table.

The `ModelOverview` and
`TermInference` Excel tables use typed cells for metrics, estimates, intervals, and p-values instead
of storing the formatted console summary as one text column. The editor exposes the same workbook
through **Export > Excel rating workbook** when explicit training or retained fit data is available
and the session's model meets the log-link requirement above.

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
