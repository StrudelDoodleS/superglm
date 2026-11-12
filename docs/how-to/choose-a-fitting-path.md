# Choosing A Fitting Path

The most important decision is whether you are fitting a REML-selected pricing
model or a fixed-penalty sparse model.

| Situation | Recommended path | Why |
|---|---|---|
| Standard spline pricing model | `fit_reml()` with `selection_penalty=0` | Automatic smoothness selection and clean GAM-style inference |
| Large-`n` spline pricing model | `fit_reml(discrete=True)` | Same modeling story, cheaper outer iterations |
| High-cardinality random effect or factor smooth | `fit_reml()` with `direct_solve="auto"` | A compact solver for large terms, chosen from the model's size |
| Smooth shrinkage inside REML | `fit_reml()` with `select=True` on spline terms | mgcv-style double-penalty shrinkage |
| Sparse screening / compression | `fit()` with `selection_penalty > 0` | Fixed-penalty sparse model rather than REML smoothness selection |
| Regularisation path analysis | `fit_path()` | Warm-started lambda path for fixed-penalty models |

## Selection Penalty Intent

Selection calibration is never implicit:

```python
SuperGLM()                                  # no sparse selection
SuperGLM(selection_penalty="auto")         # calibrate from the fit data
SuperGLM(selection_penalty=0.05)           # fixed selection strength
```

`None` and `0.0` disable sparse selection. The string `"auto"` is the only
automatic-calibration setting. REML accepts only `None` or `0.0`; its outer
optimizer owns spline smoothing, while `select=True` supplies REML-native term
shrinkage.

## Default REML Path

This is the intended path for spline-based GAM-style pricing models.

```python
model = SuperGLM(
    family="poisson",
    selection_penalty=0.0,
    features=features,
)
model.fit_reml(df, y, sample_weight=exposure, max_reml_iter=30)
```

Use this when:

- you want automatic smoothness selection
- you care about interpretable smooth terms
- you want statsmodels-style summaries and smooth-term inference

## Large-`n` REML

Turn on `discrete=True` when the model is still a REML pricing model but the
data is large enough that exact REML is too expensive.

```python
model = SuperGLM(
    family="poisson",
    selection_penalty=0.0,
    discrete=True,
    n_bins=256,
    features=features,
)
model.fit_reml(df, y, sample_weight=exposure, max_reml_iter=30)
```

This is the preferred production path for large spline-heavy frequency models.
`RandomEffect` and `FactorSmooth` retain their exact factor levels on this
path; only the continuous spline support is binned.

A column with more distinct values than `n_bins` is binned. The fit sees each
row at the centre of an equal-width bin between the column's smallest and
largest value when those bins already follow the pieces of the curve between
the spline's knots and no bin holds more than eight times its share of the
rows. It does so too when `n_bins` is below four, or when at least `n_bins`
pieces hold rows, since no `n_bins` bins could then give each piece one.
Otherwise, as with a skewed column or a pile of equal values, the bins follow
the knots and the data instead, within the same `n_bins`: half of them divide
the pieces of the curve that hold rows (one bin each when more pieces than
that hold rows, the rows then getting the rest), the others cut the rows into
groups of equal size, and each bin is represented by the average of its rows.
A pile of equal values is then seen at its own value, and every row inside
its own piece of the curve. Without that, the fit would read many rows at one
value, and `predict` would read the curve at values the fit did not check. On
a spline with polynomial ranges, each range and each stretch beside one is
also split at its own values into as many pieces as it needs values, so these
bins give it enough whenever its rows below the column's largest value have
them.

Rows count by their frequency weight, so a fit with positive integer weights
gets the same bins as the replicated rows it stands for; under prior weights
each row with a positive weight counts once. Rows of zero weight are left out
of a main effect's bins. In a tensor interaction, a spline by a factor or a
factor smooth they still help place the bins, as they did before, so there
integer weights that include zeros do not give the replicated rows' bins. A
column with at most `n_bins` distinct values is used exactly.

An exact fit (`discrete=False`, the default) that will be slow says so
before it starts, with a `SlowFitWarning`. It fires when rows x coefficients
squared reaches 1e11, because every coefficient update forms a weighted
coefficients-by-coefficients product over all rows, and choosing the smoothing
takes dozens of them. A Gaussian fit with the identity link does not warn,
since its weights never change and the product is formed once. Nor does a fit
that uses the structured solver for a random effect or factor smooth, since it
never forms that product.

On the 542,000-row freMTPL2 frequency book, a model with three interaction
pairs (about 480 coefficients) took 48 seconds exactly and 17 seconds with
`discrete=True`; with ten pairs (about 1,540 coefficients) it took 3.6 minutes
and 76 seconds. With one pair (about 190 coefficients) it took 15 seconds and
does not warn.

## When a REML fit counts as converged

A REML fit stops only when the solver can check that it has reached the best
coefficients for the chosen smoothing, to the accuracy the smoothing search
needs. Earlier releases could stop a few steps early and still report the fit
as converged. This trades some speed for results you can rely on.

- **Some numbers move slightly.** The smoothing of a nearly flat smooth, and a
  term's effective degrees of freedom, can differ a little from earlier
  releases. The new values are the ones the check confirms.
- **Some fits take longer.** On a claim-frequency model with 77,000 rows and
  a random effect for 942 vehicle models nested in 87 makes, the check made
  the fit about a quarter slower.
- **A fit that cannot be checked says so.** It is returned, not refused. It
  warns with a `ConvergenceWarning` that names the reason, the summary repeats
  it, and `model.reml_diagnostics()["converged"]` reads `False`.
- **Barely identified coefficients are left out of the check.** A
  `WeakIdentificationWarning` names them.

### Smoothing search with a monotone term

A shape-constrained term chooses its smoothing by the same Newton search as
every other term: the search runs without the constraint, and the
coefficients are then refitted with it at the chosen smoothing (see
[Constrain a smooth](constrain-a-smooth.md)). Releases up to 0.41.0 searched
a shape-constrained `PSpline` on the SCOP engine instead, which is retired; a
model they fitted still reports its own search in `reml_diagnostics()`.

### Refitting on similar data

A refit on similar data, such as a cross-validation fold, a bootstrap sample
or next year's book, can start its smoothing search where an earlier fit
ended. That usually takes fewer iterations.

```python
first = model.fit_reml(train_df, y_train, sample_weight=exposure_train)
start = first.reml_diagnostics()["lambdas"]
refit = model.clone_unfitted().fit_reml(df, y, sample_weight=exposure, lambda2_init=start)
```

- **The answer is the refit's own.** The search still runs to its own
  convergence test on the new data.
- **Leave out terms the earlier fit switched off.** A term smoothed to its
  limit, for example a spline reduced to a straight line, sits where the
  search cannot tell which way to move. Started there, it stays there even if
  the new data supports a curve. Drop such terms from `start`.
- **A tensor interaction starts from `start` only as a whole.** If `start`
  names some parts of a tensor interaction but not all, the whole interaction
  starts afresh, and the fit warns. A name that matches nothing in the model
  is ignored, also with a warning.
- **`cross_validate(..., fit_mode="fit_reml")` does this for you.** Every
  fold after the first starts from the first converged fold, minus the terms
  it switched off. A tensor interaction with any part switched off starts
  afresh as a whole. Pass `warm_start=False` to start every fold afresh.

### When a coefficient is barely identified

`WeakIdentificationWarning` means some coefficients carry information only at
the noise level of the data. Each belongs to a factor level or column with
little or no weight, few rows, or little information.

- **What it means.** The data pin these coefficients down no better than the
  rounding of the calculation, so their estimates and standard errors carry
  little information.
- **What the fit does.** It keeps them in the model and names them in the
  warning and in `model.diagnostics()`.
- **What it leaves out.** Coefficients at the noise level do not take part in
  choosing the smoothing.
- **What to do.** Drop the column or merge the level, or give its rows more
  data or weight. If those coefficients do not matter to you, you can leave
  them.

## Structured credibility terms

`RandomEffect` and `FactorSmooth` are REML-only terms. With
`direct_solve="auto"`, a model whose random effect or factor smooth is large
enough is fitted with a compact solver, which never builds the full matrix for
that term. A smaller model uses the ordinary Gram solver. The choice depends
only on the model's terms, their sizes, and which grouping levels sit inside
which others. It never depends on the response, the weights, or how the fit is
going, so the same model on the same grouping columns always takes the same
solver.

```python
model = SuperGLM(
    family="poisson",
    terms=[
        re("VehBrand"),
        interaction(FactorSmooth("DrivAge", group="Region", basis="fs", k=6)),
    ],
    discrete=True,
    n_bins=256,
    direct_solve="auto",
    selection_penalty=0.0,
)
model.fit_reml(df, y, offset=np.log(df["Exposure"]))
```

`Region` needs no term of its own: a term on the group is redundant, because
the factor smooth carries each level's intercept.

The compact solver handles one large credibility term beside narrow dense
features, global splines and other random effects. It works with every family
and link, and with both `basis="fs"` and `basis="sz"` factor smooths. See
[Credibility terms](../explanation/credibility-as-smoothing.md) for model
semantics and the French motor example.

- **The reason for Gram is recorded.** When `direct_solve="auto"` uses the
  Gram solver, `model.result.direct_fallback_reason` says why.
- **Small models use Gram.** Below a size set by the number of rows and
  columns, the compact solver does not pay off.
- **Shape constraints use Gram.** A model with a monotone or other
  constrained term is fitted with the Gram solver.
- **Two factor smooths use Gram.** The compact solver takes one factor smooth
  per model.
- **A fit never switches solver.** If the compact solver cannot finish a fit,
  the fit stops with an error that says what went wrong. This should not
  happen for a model the data can identify.
- **`direct_solve="gram"` fits the same model with the ordinary solver.** Use
  it if the compact solver ever stops with that error.
- **`direct_solve="structured"` forces the compact solver.** It stops with an
  error on a model the compact solver does not handle.

### Nested grouping factors

- **Nesting is found from the data.** If every `region` level appears under a
  single `country` level, `region` is nested in `country`.
- **The compact solver uses it for its main random effect.** It builds its
  hierarchy around the random effect with the most levels, and fits that
  effect together with the effects it is nested in.
- **Other nested pairs are fitted as usual.** A pair that does not include
  the compact solver's main random effect is fitted like any other pair of
  random effects. So is every pair when the Gram solver fits the model.
- **You can declare it.** `RandomEffect(nested_in="country")` on `region`
  states the hierarchy. The fit checks it and stops with an error that names
  any row that breaks it.
- **A declaration fixes the hierarchy.** When `region` is the compact solver's
  main random effect, the declared parent is always part of its hierarchy.
- **A near miss is reported.** If all but a few rows nest, a warning names
  those rows and the fit treats the two random effects as crossed. Fixing
  those rows makes the pair nested again.

### Factor smooths with `basis="sz"`

For `basis="sz"`, configure the matching global spline explicitly:

```python
model = SuperGLM(
    family="poisson",
    terms=[
        s("DrivAge", kind="ps", k=7, m=2),
        interaction(FactorSmooth("DrivAge", group="Region", basis="sz", kind="ps", k=6, m=2)),
    ],
    direct_solve="auto",
    selection_penalty=0.0,
)
model.fit_reml(df, y, offset=np.log(exposure))
```

The factor smooth stays in its compact form, the level codes and one shared
basis, so no matrix with a column per level and basis function is built.

## `select=True` Versus `selection_penalty > 0`

These are different tools and should not be documented as interchangeable.

- `select=True` keeps you in the REML story and adds mgcv-style double-penalty
  shrinkage to the spline term.
- `selection_penalty > 0` activates sparse/group penalties and moves you toward
  a sparse additive model workflow.

If your question is "should this smooth shrink toward linear or zero while I
stay in REML?", use `select=True`.

If your question is "which groups should survive a fixed-penalty sparse fit?",
use `selection_penalty > 0`.

## Fixed-Penalty Sparse Models

Use `fit()` when you want a fixed `spline_penalty` and sparse or shrinkage
regularisation.

```python
model = SuperGLM(
    family="poisson",
    penalty="group_elastic_net",
    selection_penalty=0.01,
    spline_penalty=0.1,
    features=features,
)
model.fit(df, y, sample_weight=exposure)
```

This is a good fit for:

- feature screening
- model compression
- fixed-penalty challenger models
- lambda-path experiments

## Multi-Order Spline Penalties

Spline specs can emit multiple derivative-order penalties on one term, each
with its own REML smoothing parameter.

```python
features = {
    "DrivAge": Spline(kind="cr", k=14, m=(1, 2)),
    "VehAge": Spline(kind="ps", k=10, m=(2, 3)),
}
model = SuperGLM(
    family="poisson",
    selection_penalty=0.0,
    features=features,
)
model.fit_reml(df, y, sample_weight=exposure)
```

Current guard rails:

- `select=True + m=(...)` is supported for tuples compatible with the selected
  spline class and produces a null-space penalty plus one component per
  derivative order; per-class order limits still apply
- tensor interactions with a multi-order spline parent are not yet supported
- `kind="cr_cardinal"` currently supports only the default `m=2`
- shared-block multi-penalty terms are accepted by the fixed
  `selection_penalty > 0` path

## Regularisation Path

`fit_path()` is for fixed-penalty models, not the main REML path.

```python
from superglm import Categorical, GroupLasso, Poisson, Spline, SuperGLM

model = SuperGLM(
    family=Poisson(),
    penalty=GroupLasso(),
    features={
        "DrivAge": Spline(kind="ps", k=14),
        "Area": Categorical(base="most_exposed"),
    },
)
result = model.fit_path(df, y, sample_weight=exposure, n_lambda=50, lambda_ratio=1e-3)

result.lambda_seq
result.coef_path
result.deviance_path
result.n_iter_path
```

After `fit_path()` the model is fitted at the last lambda, so `model.predict()`
predicts at that point. To predict at another point, fit with
`selection_penalty=result.lambda_seq[i]` and predict with that model. The
refit matches point `i` to the solver's tolerance, not exactly, because it
starts from scratch instead of from the previous point.

You can also rebuild predictions as `X @ coef_path[i] + intercept_path[i]`.
When a numeric column's values sit far from zero, its term and the intercept
cancel. Each rebuilt prediction is then off by about $2^{-52}$ times the
column's offset times its coefficient:

- **Years, and epoch times in seconds that span months or more,** keep 12
  or more significant digits of the column's effect.
- **A column whose offset is many orders of magnitude larger than its
  spread**, such as a raw ID or a nanosecond timestamp spanning a few
  seconds, loses visible accuracy. Predict with a refitted model instead.

Next:

- [Recommended workflows](recommended-workflows.md)
- [Feature types](specify-features.md)
- [Monotone splines](constrain-a-smooth.md)
- [REML and solvers](../explanation/solvers-and-internals.md)
