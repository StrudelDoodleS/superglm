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

### Exact continuous blocks with `continuous_kind="ppform"`

A binned continuous block is an approximation, and the default `n_bins=150` is a **budget** rather
than a target: the exporter emits at most that many intervals, and a covariate with fewer distinct
values than the budget is still binned — 30 distinct ages export as 29 interval rows, not 30 exact
ones. On a motor book with 81 distinct ages the worst row came out **mis-rated by 60%**,
concentrated in the wide intervals the quantile strategy opens in the sparse tails, which is where
exposure is thinnest and a single large risk absorbs all of it.

`continuous_kind="ppform"` declines that approximation and exports the fitted spline as the exact
piecewise cubic it already is — one row per knot interval, four coefficients:

```python
model.export_rating_tables(
    "rating_tables.xlsx",
    X_train,
    y_train,
    sample_weight=exposure_train,
    continuous_kind="ppform",
)
```

A consumer **reads both bounds back out of the interval key** — `"[18.0, 25.363636363636363)"` — and
evaluates `exp(a + b*u + c*u**2 + d*u**3)` with `u = (x - lower) / (upper - lower)`. The result
reproduces `model.predict` to machine precision — 2.4e-15 against 6.0e-01 for the binned block it
replaces — in usually an order of magnitude fewer rows. `u` is **normalised** onto `[0, 1]`; a raw
`x - lower` on a covariate ranging to 1e5 loses enough precision in a fixed-scale decimal column to
produce a 3.3× relativity error, which is worse than the binning it replaces.

The bounds are not repeated as their own columns. The key already holds them *exactly*: it is
printed through `repr`, the shortest string that reads back as the same binary64, so parsing it
recovers both bounds bit for bit and each row's upper bound is identically the next row's lower
bound. A separate pair of float columns could only ever disagree with the key — and, being numbers,
could not carry the infinite bound of a tail row through a spreadsheet at all.

The block is **seven columns rather than three**, and a superset rather than a new shape. The
familiar `<feature>`, `Relativity` and `Weight` stay in front of it unchanged, with `a`, `b`, `c`,
`d` appended behind them, so a loader that cannot evaluate a polynomial still finds the block by the
same header signature, slices the same three columns, and scores it as the step function it scores
today, while an upgraded one reads the coefficients and is exact. `Relativity` is the curve's value
at the row's lower bound rather than the interval's average, so the two readings of one row agree at
its left edge. Because the blocks are laid out at their own widths, a seven-column block moves every
block to its right — read the header row rather than assuming the three-column stride.

If you store the coefficients, include them in any **content digest** you fingerprint a published
package with. Two models differing only in their coefficients otherwise fingerprint identically,
and the second is silently deduplicated into the first.

Extrapolation is carried in the table rather than described beside it, because a cubic continued
past its last knot is unbounded — 1581× the correct factor twenty-one years past the boundary of a
real age curve. Under the default `extrapolation="clip"` the block emits a constant leading and a
constant trailing row, so "match an interval and evaluate it" is correct outside the training range
too. Under `extrapolation="error"` there are no unbounded rows at all and a value outside the knot
range matches nothing, which is the answer the model itself gives. `extrapolation="extend"` is
refused unless you pass `allow_unbounded_extrapolation=True`, because it exports a tariff with no
upper bound; exported with that acknowledgement its tails clip where the model extends, since an
unbounded interval has no width and the normalised `u` does not exist there.

The tail rows read `[-inf, 18.0)` and `[99.0, inf)` in the key, and every numeric cell in the block
is a real number. A spreadsheet cell cannot hold an infinity, so keeping the bounds in the key —
which is text — is what lets those rows survive the workbook at all. Their coefficients are
`b = c = d = 0`, so a consumer that recognises an infinite bound and skips `u` there and one that
clamps `u` to `[0, 1]` arrive at the same factor.

Two term kinds are not converted. Terms carrying a `Constraint.postfit` repair are refused by name,
because that path's repaired curve has never been verified to be piecewise polynomial on the term's
own knots; `Constraint.fit` constraints convert unchanged. And `Polynomial` terms stay binned, as
does the continuous-by-continuous interaction grid — the block is fixed at four coefficients while a
polynomial's degree is not. One workbook therefore carries both kinds of block at once, and the
discretization impact sweep still describes the ones that stayed binned.

The **offset multiplier** block is exact whatever the fit's cardinality. Up to
`offset_max_exact_levels` distinct multipliers it lists them; above that it is a single `per_unit`
row. Binning it is opt-in, and a binned block is a summary rather than a per-row lookup — its rows
are keyed on interval strings and each carries its bin's exposure-weighted average, and a bin with
no exposure, reachable with `bin_strategy="uniform"` on a skewed exposure, reports the midpoint of
its own interval at weight zero. Pass `offset_source=` to key the block on a raw column of the frame
instead of on the multiplier.

### Continuous offsets: `offset_kind="per_unit"`

An offset is not an estimated surface. Its coefficient is fixed at 1 by construction — no standard
error, no lambda, nothing fitted — so a continuous offset is not a curve to approximate but a column
to multiply by. What decides the exported shape is therefore not how many distinct values the source
takes, but that the consumer multiplies rather than looks up.

One rule covers both paths: `offset_max_exact_levels` governs both paths — the declared one reached
through `offset_source=`, and the undeclared one that has nothing but the fitted offset vector.
`offset_kind` chooses what each emits:

| value | behaviour |
|---|---|
| `"auto"` (default) | one row per level up to `offset_max_exact_levels`, a single `per_unit` row above it |
| `"discrete"` | levels only; a declared source refuses above the cap, an undeclared one falls through to `per_unit` |
| `"per_unit"` | always the relation |
| `"binned"` | the exposure summary below; undeclared offsets only, a declared source rejects it |

A per-unit block is one row, applied by multiplying the named column by its `Relativity`:

```
SumInsured | Relativity | Weight
per_unit   | 1.0        | 6000.0
```

`log(Exposure)` gives a scale of exactly `1.0`; `log(Term/12)` gives `1/12`. It is the same shape
`Numeric` terms already export, for the same reason.

**It has no bounds and no extrapolation rule.** Nothing in it is estimated, so nothing is known only
over a training range — the relation holds at a sum insured of 10m on a book that saw 442k for
exactly the reason it holds at 10k. The binned alternative could not: its top bin capped every risk
above the fitted range at that bin's average. Measured on a continuous sum-insured offset, 6,000
rows with 5,998 distinct values — binned, all 6,000 receive a factor that is not their own and the
worst is out by **1.022e+00**, more than double; per-unit, the worst row is out by **9.99e-16**.

The relation is derived on the log scale and verified on the multiplier scale against
`offset_mapping_rtol`/`offset_mapping_atol`, on every row. A column the offset is not proportional
to is refused by name rather than approximated.

Low cardinality is left alone: `Term ∈ {12, 36}` still exports as the two-row lookup `12 → 1`,
`36 → 3`, because for a handful of levels a lookup already *is* the exact answer and asks nothing
of the consumer.

**An undeclared offset follows that same rule.** Exported with no `offset_source=`, an offset above
the cap is one `per_unit` row as well — keyed on the offset multiplier rather than on a column,
because the exporter is not told which column produced the offset and does not guess one from the
frame. Its `Relativity` is exactly `1.0`: the consumer computes `exp(offset)` and the block hands
that number back unchanged. That reads as though it says nothing, and it says exactly as much as the
binned block it replaces — which was *also* keyed on the multiplier, so a consumer had to compute the
same number before it could look anything up, and then received its bin's exposure-weighted average
instead of the value it had just computed. Nothing bins an offset now unless a caller asks for it by
name with `offset_kind="binned"`.

**This changes the exported block for any model with a continuous offset and no `offset_source=`.**
Where such a model used to ship a block of `n_bins` rows keyed on interval strings, it now ships one
row keyed `per_unit`, carried on the new `offset_per_unit` block kind. A consumer that stages the
binned form — matching a computed multiplier against an interval key — needs the per-unit kind before
it can load such a workbook; exporting with `offset_kind="binned"` reproduces the old shape in the
meantime. What it reproduces was never rateable: the factor it returns is the bin average of a number
the consumer had already computed exactly.

The workbook includes selected-bin rating tables, a discretization impact sweep, and a structured
Model Summary sheet. The sweep runs the `impact_bins` ladder — `20, 50, 100, 200, 250` by default —
**plus the `n_bins` the workbook was actually exported at**, marked by the sheet's `exported` column.
That row is the one that describes the table in hand. Take it rather than the smallest number on the
sheet: each row is an independently measured error at its own resolution, not a point on a curve
guaranteed to fall — a finer grid shrinks the worst-case bound, but successive nearest-node grids are
not nested, so a nearer node can carry a value further from a given row. Passing `impact_bins=()`
skips the sweep, leaving the worksheet present with its headers and no rows.

The sweep covers the binned and sampled model terms: main-effect spline and polynomial terms, and a
continuous-by-continuous interaction, which is sampled onto an `n_bins`-per-axis grid rather than
binned into intervals. Its row is keyed on the interaction's name and its `actual_bins` counts the
grid's *cells*, so an interaction swept at 20 reports 400 where a main effect reports 20. The metric
columns are **joint** over everything discretized at that resolution — within one `n_bins` group only
`feature` and `actual_bins` vary.

Two approximations the workbook can carry are outside that measurement, so the sheet is a bound on
the terms it lists rather than on every factor in the book. An **offset multiplier** block asked for
as `offset_kind="binned"` is not swept
([#314](https://github.com/StrudelDoodleS/superglm/issues/314)) — under the default there is no
offset approximation left to sweep — and the sheet's own discretized
predictor is stabilized where the workbook's product is not
([#313](https://github.com/StrudelDoodleS/superglm/issues/313)).

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
