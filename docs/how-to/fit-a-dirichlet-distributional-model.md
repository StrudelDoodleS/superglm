# Fit a Dirichlet distributional model

`DirichletLSS` models complete continuous compositions: every component is
strictly positive and each row sums to one. Give the components explicit names
and one predictor for each positive concentration parameter, `alpha`.
The default link is logarithmic. A change in one log-alpha predictor generally
changes both the mean composition and total concentration.

This example uses synthetic compositions with three known components:

```python
import numpy as np
import pandas as pd
from superglm import DirichletLSS, SuperLSS, s

rng = np.random.default_rng(6021)
X = pd.DataFrame({"x": np.linspace(-1, 1, 300)})
alpha = np.exp(
    np.column_stack([
        0.5 + 0.7 * np.cos(3 * X.x),
        0.8 - 0.9 * np.sin(4 * X.x),
        np.full(len(X), 1.1),
    ])
)
shares = pd.DataFrame(
    [rng.dirichlet(a) for a in alpha],
    columns=["red", "green", "blue"],
    index=X.index,
)
family = DirichletLSS(components=("red", "green", "blue"))
model = SuperLSS(
    family,
    family.alpha("red", s("x", k=5)),
    family.alpha("green", s("x", k=6)),
    family.alpha("blue"),  # a constant alpha, estimated through an intercept
    weight_semantics="frequency",
)
model.fit_reml(X, shares)
mean_shares = model.predict(X)
parameters = model.predict_parameters(X)
```

`mean_shares` is a DataFrame with columns `red`, `green`, `blue`, in the
declared order, and the index of `X`. Its values are `alpha / alpha.sum(axis=1)`.
`parameters` has columns `alpha_0`, `alpha_1`, `alpha_2` in that same order.
`family.component_parameters` maps component names to these parameter names.
The total concentration is `parameters.sum(axis=1)`; it is derived from the
fitted alphas, not an additional predictor.

For fixed smoothing strengths, call `fit` with each penalty's qualified name:

```python
model.fit(
    X, shares,
    lambdas={"alpha_0:x#wiggle": 0.3, "alpha_1:x#wiggle": 0.8},
)
```

`fit_reml` actually estimates smoothing strengths. Read
`model.smoothing_convergence_reason_`, `model.smoothing_certified_`, and
`model.diagnose()` before treating its endpoint as stationary. A returned fit
may have stopped at an iteration limit or practical plateau; convergence
warnings and diagnostics remain part of its result. Constant observed
compositions can have an unbounded concentration likelihood, so a finite
accepted iterate is not evidence of a finite MLE.

## Response and weight contract

Components must be an ordered tuple of at least two distinct nonempty strings.
A labeled response must contain every component exactly once, with no extra
columns. Columns are reordered by name. When `X` and the response have pandas
indices, those indices must agree positionally; rows are never silently joined
or sorted. An unlabeled `(n, K)` array is positional in the declared component
order.

The model checks original shapes and labels before removing zero-weight rows.
It then validates retained values, refusing booleans, complex values, strings,
missing values, infinities, zeros, negative components, and materially nonunit
row sums. Invalid values in a dropped row do not enter the likelihood.
Integer frequency weights replicate the entire joint observation. Unit prior
weights are supported; nonunit prior weights have no specified Dirichlet law
and are refused.

Only roundoff-sized deviations from unit sum are admitted. With
`u = 2**-53` and `gamma_m = m*u/(1-m*u)`, the allowance is
`8*gamma_(K+1)` times row magnitude. Such a row denotes its mathematically
normalized supplied positive shares. Larger errors are not normalized away.
Counts, percentages, unknown totals, structural or rounded zeros, incomplete
triangles, and component censoring require other observation models. This API
does not supply those laws or a pseudocount repair.

Mathematical support and numerical range differ. Evaluation refuses an alpha
configuration whose required float64 channels cannot be represented; prediction
also refuses a positive mean share that underflows to zero. There is no hidden
alpha cap, zero substitution, or added ridge.

## Fixed offsets

Declare offsets on each component helper. Column lists and array lists add
their contributions with coefficient one on the link scale:

```python
X["known_log_factor"] = np.log1p(0.2 * (X.x + 1))
X["adjustment"] = 0.03
array_a = 0.1 * np.sin(X.x.to_numpy())
array_b = np.full(len(X), 0.02)
array_c = 0.05 * X.x.to_numpy()
offset_model = SuperLSS(
    family,
    family.alpha("red", "x", offset=["known_log_factor", "adjustment"]),
    family.alpha("green", "x", offset=[array_a, array_b]),
    family.alpha("blue", offset=["known_log_factor", array_c]),
).fit(X, shares)
```

`fit` and `fit_reml` have no offset argument. For predictions, array declarations
require replacements on **every** call, including a call on the training frame:

```python
new_X = X.iloc[:5].copy()
new_a, new_b = array_a[:5] + 0.01, array_b[:5]
new_c = array_c[:5] - 0.01
predicted = offset_model.predict(
    new_X,
    offsets={"alpha_1": new_a + new_b, "alpha_2": new_c},
)
```

The replacement is the sum of the **array** contributions. Declared columns
are read from `new_X` and added once. Column-only declarations reject a second
mapping. The same rule applies to scoring and posterior bounds. See
[Specify offsets](specify-distributional-offsets.md) for source ownership,
Series-index checks, retained-row validation, and clone/refit details.

## Joint scoring and cross-validation

There is one normalized joint density and one log score per composition:

```python
scores = model.scores(X, shares, which=("log",))
```

Frequency-weighted scores contain `count * -log_density`, once per row;
zero-weight rows read NaN. To average frequency scores, divide their sum by
the retained frequency mass. The existing scalar `scores()` default still asks
for log and CRPS. For a Dirichlet fit that default refuses CRPS; explicitly
request log-only scoring.

```python
from sklearn.model_selection import KFold
from superglm import cross_validate

cv = cross_validate(
    model, X, shares,
    cv=KFold(3, shuffle=True, random_state=42),
    fit_mode="fit",
    fit_kwargs={"lambdas": model.smoothing_parameters_},
    return_oof=True,
)
print(cv.pooled_scores["nll"])
assert cv.oof_predictions.shape == shares.shape
```

The Dirichlet CV default is joint NLL only. `fit_mode="fit_reml"` estimates
smoothing separately inside each fold. OOF columns follow `family.components`.
Unpriceable zero-weight rows remain NaN as whole rows, without discarding other
predictions in that fold. Declared array offsets are sliced automatically;
CV accepts no separate SuperLSS training offset.

Custom scalar scorers use
`scorer(model, X, y, *, sample_weight, offset) -> float` (or a dict of scalar
scores). `y` is in canonical component order; `offset` holds prediction
replacements and is passed to `model.predict(X, offsets=offset)`. A custom
scorer is responsible for respecting zero-weight rows and its chosen weight
interpretation.

`model.compare(other, X, shares, which="log")` gives paired joint log-score
differences. Labeled inputs allow different declared orders of the same
component set; each candidate is canonicalized separately. Positional responses
require identical declared component order. Candidate-specific offset maps are
accepted through `a_offsets` and `b_offsets`.

## Uncertainty and saved models

`model.covariance_` is the joint **coefficient** covariance. It retains the
cross-component blocks. `predict_link(X, se_fit=True)` reports per-alpha
link-scale standard errors; term inference and summaries use the same joint
fit geometry.

Posterior scalar quantities use one shared coefficient draw set:

```python
draws = model.posterior_draws(2000, seed=42)
red_share = model.posterior_bounds(
    X, lambda a: a[:, 0] / a.sum(axis=1), draws=draws,
)
red_green_share = model.posterior_bounds(
    X, lambda a: a[:, :2].sum(axis=1) / a.sum(axis=1), draws=draws,
)
log_ratio = model.posterior_bounds(
    X, lambda a: np.log(a[:, 0] / a[:, 2]), draws=draws,
)
concentration = model.posterior_bounds(X, lambda a: a.sum(axis=1), draws=draws)
alpha_red = model.posterior_bounds(X, ("parameter", "alpha_0"), draws=draws)
```

Each callable returns one value per row. The named scalar quantity `"mean"`
refuses vector responses and directs you to a share callable. Independent
marginal draws would lose the fitted cross-component dependence.

`covariance="corrected"` requests the existing smoothing-uncertainty
correction. It uses authenticated fitted derivative evidence or retained-row
replay. When unavailable, a `ConvergenceWarning` explains why conditional
covariance is returned, and the draw object's `covariance_kind` is `"fixed"`.
A nonstationary correction is also disclosed. Compact smoothing fits may need
`retain_rows=True` to replay a Hessian that was not published at the endpoint.

The separate `family.response_covariance(parameters.to_numpy())` returns
`(n, K, K)` response covariance, `(diag(mu) - mu*mu.T)/(sum(alpha)+1)`.
It is singular in the all-ones direction because shares sum to one. This is
neither coefficient covariance nor posterior uncertainty in mean shares.

Standard pickle and joblib round trips preserve predictions, parameters,
component order, offset declarations, and inference. Prediction evaluates
saved fitted coefficients without rerunning the training optimizer.
`retain_rows=False` retains prediction and conditional coefficient inference.
After loading, give `X_train=` to term inference and summaries, since their
training-frame reference is not serialized.
The existing clone/refit and hand-edit machinery is supported: edits change
predictions while covariance still describes the fitted model, with the usual
warning; corrected covariance requires the fitted coefficients or a refit.

Scalar CDFs, quantiles, PIT/quantile residuals and their plots, CRPS and
threshold/tail scores, Gini, actual/expected totals, calibration, Murphy
diagrams, scalar distribution surfaces, quantile-based posterior predictive
draws, portfolio simulation, and the response editor explicitly refuse vector
responses. Per-parameter effects and scalar posterior callables remain available.
Dense and discrete/chunked fitting are supported; inspect the recorded backend
in diagnostics. Discretization remains a basis approximation, and small
synthetic agreement does not establish a general speed or accuracy guarantee.
