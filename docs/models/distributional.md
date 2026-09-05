# Distributional Location–Scale Models

`SuperLSS` fits several linked predictors jointly. `GaussianLS` has one predictor
for conditional location and another for conditional standard deviation. Use it
when a continuous response is reasonably Gaussian after any documented
transformation but its spread changes with risk characteristics. The
distributional surface also includes `GammaLS`, with predictors for conditional
mean and coefficient of variation, for strictly positive responses. `TweedieLSS`
adds a dense three-predictor model for nonnegative responses with a zero atom.
`NegativeBinomialLS` jointly models the mean and NB2 size of count data.

Do not use `GaussianLS` for raw claim counts. Use `NegativeBinomialLS` when both
the count mean and overdispersion vary; use a Poisson or scalar
negative-binomial `SuperGLM` when a second predictor is not needed.

## First Gaussian model

Predictors are ordered and their names must exactly match the family parameters:

```python
from superglm import Spline, SuperLSS
from superglm.distributional import GaussianLS, Predictor

model = SuperLSS(
    family=GaussianLS(scale_floor=0.05),
    predictors=(
        Predictor(
            "location",
            {
                "DrivAge": Spline(kind="cr", k=10),
                "VehAge": Spline(kind="cr", k=8),
            },
        ),
        Predictor(
            "scale",
            {
                "DrivAge": Spline(kind="cr", k=8, select=True),
            },
        ),
    ),
)

model.fit_reml(train_df, y_train, method="efs")
parameters = model.predict_parameters(holdout_df)
```

`parameters` is a pandas DataFrame with columns `location` and `scale`.
`predict_link()` returns the corresponding linear predictors. For Gaussian LS,
`predict()` returns conditional location, not scale and not a transformed-response
mean.

## Gamma mean–CV model

Gamma predictors are ordered `mean`, then `scale`, and `fit_reml()` estimates
their smoothing parameters jointly:

```python
from superglm import Spline, SuperLSS
from superglm.distributional import GammaLS, Predictor

model = SuperLSS(
    family=GammaLS(),
    predictors=(
        Predictor(
            "mean",
            {"DrivAge": Spline(kind="cr", k=10)},
        ),
        Predictor(
            "scale",
            {"DrivAge": Spline(kind="cr", k=8, select=True)},
        ),
    ),
).fit_reml(train_df, y_train)

parameters = model.predict_parameters(holdout_df)
```

`parameters` has columns `mean` and `scale`; `predict()` returns `mean`. Here
`scale` is the coefficient of variation (CV), not variance and not Gamma shape.
For a unit prior weight,

```text
Var(Y | x) = mean² × scale²
```

and under prior precision weight `w`,

```text
Var(Y | x, w) = mean² × scale² / w.
```

This is the parameterization used internally. Comparators such as mgcv and MSSM
usually report dispersion `φ`, with the exact mapping `φ = scale²`.

Gamma support is strictly positive. A zero response is not a small positive
Gamma observation and is rejected; use a model whose law has mass at zero. Prior
precision is the default weight semantics. Explicit
`weight_semantics="frequency"` instead treats each integer weight as literal row
replication.

**Functionals.** `predict_cdf(X, y)` and `predict_quantile(X, p)` use the gamma
distribution functions in this mean–CV parametrisation. Certified expected
shortfall is available for both the unit law and the prior-weighted row law.

## Negative-binomial mean–theta model

`NegativeBinomialLS` is the NB2 family with predictors in the exact order
`mean`, then `theta`. Both use log links. For conditional mean \(\mu\) and size
\(\theta\),

\[
E(Y)=\mu, \qquad \operatorname{Var}(Y)=\mu+\frac{\mu^2}{\theta}.
\]

Larger `theta` means less overdispersion. `predict_parameters()` returns columns
`mean` and `theta`, while `predict()` returns the conditional mean in the
observation coordinates supplied to the fit or prediction call.

The primary actuarial representation models a rate and supplies exposure as the
default prior weight:

```python
import numpy as np

from superglm import Numeric, SuperLSS
from superglm.distributional import NegativeBinomialLS, Predictor


def nb2_model():
    return SuperLSS(
        family=NegativeBinomialLS(),
        predictors=(
            Predictor("mean", {"DrivAge": Numeric(), "VehAge": Numeric()}),
            Predictor("theta", {"DrivAge": Numeric(), "VehAge": Numeric()}),
        ),
    )


rate_model = nb2_model().fit(
    train_df,
    claim_count / exposure,
    sample_weight=exposure,
)
```

`claim_count` must be a non-negative integer count; the prior route verifies
that `exposure * (claim_count / exposure)` returns to that count lattice.

If \(\lambda\) is mean rate, \(\kappa\) is size per unit exposure, and \(e\)
is exposure, this prior-weight construction is

\[
\text{ClaimCount}\mid x,e \sim \operatorname{NB2}(e\lambda,e\kappa).
\]

The exactly equivalent raw-count fit uses unit likelihood weights and adds
`log(exposure)` to both predictors:

```python
log_exposure = np.log(exposure)
raw_equivalent = nb2_model().fit(
    train_df,
    claim_count,
    offsets={
        "mean": log_exposure,
        "theta": log_exposure,
    },
)
```

Adding exposure only to the mean is supported, but it is a different model:

```python
persistent_frailty = nb2_model().fit(
    train_df,
    claim_count,
    offsets={"mean": log_exposure},
)
```

This specifies \(\operatorname{NB2}(e\lambda,\theta)\): heterogeneity persists
as exposure grows instead of the NB2 size accumulating with exposure. Choose
between these laws as a modeling assumption; the API does not infer one from a
column name.

Future raw-count predictions must receive offsets consistent with the fitted
law. For the primary reproductive-exposure model, offset both parameters:

```python
future_log_exposure = np.log(future_exposure)
future_offsets = {
    "mean": future_log_exposure,
    "theta": future_log_exposure,
}
future_count_parameters = rate_model.predict_parameters(
    future_df,
    offsets=future_offsets,
)
future_count_mean = rate_model.predict(future_df, offsets=future_offsets)
```

Omit those offsets for unit-exposure rate predictions. A persistent-frailty fit
instead receives only the future `mean` offset.

The likelihood's size terms are evaluated by an exact recurrence with one
cell per count unit, so an evaluation costs `sum(claim_count)` recurrence
cells in addition to its ordinary per-row work. The kernel refuses an
evaluation above a budget of 2e8 cells. Both standard remedies keep the
likelihood exact: aggregate identical rows and fit them with
`weight_semantics="frequency"`, or move exposure into an offset so that the
counts themselves stay small.

The supported route is dense fixed-smoothing or dense EFS with observed
curvature. Public `SuperLSS` currently refuses `discrete=True` for every
family. This family is not zero-inflated and does not claim an exact Poisson
active face, CDF or quantile methods, random generation, or complete-fit speed.

## Generalized gamma mean–scale–shape model

`GeneralizedGammaLSS` is Prentice's generalized gamma for a strictly positive
severity, with three predictors in the exact order `mean` (log link), `scale`
(log link with a floor, default 0.01) and `shape` (identity link, any real
value). `shape = 0` is the log-normal, `shape = 1` the Weibull and
`shape = scale` the gamma; a negative `shape` gives a power-law right tail with
index `1/(scale·|shape|)`.

```python
from superglm import Categorical, Spline, SuperLSS
from superglm.distributional import GeneralizedGammaLSS, Predictor

model = SuperLSS(
    family=GeneralizedGammaLSS(),
    predictors=(
        Predictor("mean", {"DrivAge": Spline(kind="cr", k=8), "Region": Categorical()}),
        Predictor("scale", {"Region": Categorical()}),
        Predictor("shape", {}),
    ),
).fit_reml(frame, severity)
```

**Mean form (default).** The first predictor is `E[Y]`, so its relativities
multiply the mean exactly as for `GammaLS`; the scale and shape predictors
only redistribute mass within a cell and never touch the mean table. The mean
must exist: cells with `shape < 0` and `scale·|shape| ≥ 1` are outside the
model, the solver's line search backs away from them, and a repeated curvature
failure is diagnosed with a message naming that boundary. A book whose tail
genuinely has no mean drives the intercept-only null fit into that boundary
and the fit stops with `NullModelFitError`; use the location form for it.

**Location form.** `GeneralizedGammaLSS(parametrisation="location")` puts the
identity-linked log-scale location first (`log Y = location + scale·W`); its
relativities multiply every quantile, and an infinite mean is a legitimate
fit. `predict()` returns the conditional mean in both forms (`inf` where it
does not exist).

**Weights.** Frequency weights replicate rows. Non-unit prior weights are
refused: the family is not reproductive, so there is no averaging law. Fit
claim-level rows and carry exposure through offsets.

**Functionals.** `predict_cdf(X, y)` and `predict_quantile(X, p)` are closed
form in the regularised incomplete gamma; `predict()` is the mean. Certified
expected shortfall is available for the unit law. Non-unit prior weights are
already refused by this family, so it has no prior-weighted tail functional.

**Oracles.** `scipy.stats.gengamma(a=shape⁻², c=shape/scale,
scale=exp(location − scale·log(shape⁻²)/shape))`; gamlss `GG` with
`mu = exp(location)`, `sigma = scale`, `nu = shape/scale`.

## Log-normal mean–scale model

`LogNormalLS` is the two-parameter log-normal for a strictly positive severity:
`log Y ~ N(mu, sigma^2)`, with predictors `mean` (log link) and `scale` (log
link with a floor, default 0.01). It is the `shape = 0` member of
`GeneralizedGammaLSS` and the `skew = 0` member of `TwoPieceLogNormalLSS`, and
the tests pin both identities to 1e-14, so it is the right choice when the
extra shape parameter is not paying for itself.

```python
from superglm import Categorical, Spline, SuperLSS
from superglm.distributional import LogNormalLS, Predictor

model = SuperLSS(
    family=LogNormalLS(),
    predictors=(
        Predictor("mean", {"vehicle_age": Spline(kind="cr", k=8)}),
        Predictor("scale", {"region": Categorical()}),
    ),
).fit_reml(frame, severity)
```

**Mean form (default).** The first predictor is `E[Y] = exp(mu + sigma^2 / 2)`,
so its relativities multiply the mean exactly as for `GammaLS`, and the scale
predictor redistributes mass within a cell without moving the mean table. The
mean always exists, so -- unlike the generalized gamma -- neither form has an
invalid region.

**Location form.** `LogNormalLS(parametrisation="location")` puts the
identity-linked log-scale location first; its relativities multiply every
quantile. Fitting this form on `y` is the same likelihood as fitting
`GaussianLS` on `log y`, but `predict()` returns the mean on the `y` scale
instead of the location on the log scale, which is the transformation the
current-limits list warns about doing by hand.

**Weights.** Frequency weights replicate rows. Non-unit prior weights are
refused: the family is not reproductive, so there is no averaging law. Fit
claim-level rows and carry exposure through offsets.

**Functionals.** `predict_cdf(X, y)` and `predict_quantile(X, p)` are the
normal cdf and its inverse on the log scale; `predict()` is the mean. Certified
expected shortfall is available for the unit law, but not for a non-unit prior
law that the family cannot represent.

**Oracles.** `scipy.stats.lognorm(s=scale, scale=exp(location))`; gamlss
`LOGNO` with `mu = location` and `sigma = scale`.

## Two-piece skew models

`TwoPieceLogNormalLSS` and `TwoPieceNormalLSS` are the two carriers of one
epsilon-skew two-piece normal law. Writing `W` for the standard two-piece
variate,

    f_W(w) = phi(w / (1 - eps))   for w <  0
             phi(w / (1 + eps))   for w >= 0

so the **right piece is the wide one for a positive skew**, and the density
needs no extra normalising constant: each half contributes `(1 -/+ eps)/2`.
`TwoPieceLogNormalLSS` puts `log Y = mu + scale * W` on `y > 0`;
`TwoPieceNormalLSS` puts `Y = location + scale * W` on the whole line. Positive
`skew` therefore means a heavier right tail on the **log** scale for the
log-normal family and on the **response** scale for the real-line one.

```python
from superglm import Spline, SuperLSS
from superglm.distributional import Predictor, TwoPieceLogNormalLSS

severity = SuperLSS(
    family=TwoPieceLogNormalLSS(),
    predictors=(
        Predictor("mean", {"DrivAge": Spline(kind="cr", k=8)}),
        Predictor("scale", {"VehAge": Spline(kind="cr", k=6)}),
        Predictor("skew", {}),
    ),
).fit_reml(claims, claims["loss"])
```

**Naming.** Both classes end in `LSS`, including `TwoPieceNormalLSS`: the
suffix records that the family carries a *shape* predictor, not that its
response leaves the real line. `LogNormalLS`, which has no skew, is the `LS`
sibling.

**Parametrisation.** `TwoPieceLogNormalLSS` is mean-parametrised by default:
its first parameter is `E[Y]` under a log link, so a relativity on the `mean`
predictor multiplies the fitted mean and the `scale` and `skew` predictors
leave the mean table alone. That works because the mean loading

    K(scale, skew) = (1 + eps) e^{s^2 (1+eps)^2 / 2} Phi(s (1+eps))
                   + (1 - eps) e^{s^2 (1-eps)^2 / 2} Phi(-s (1-eps))

is finite everywhere in the support, so `E[Y] = e^mu K` never diverges; there
is no infinite-mean region to diagnose. Pass
`TwoPieceLogNormalLSS(parametrisation="location")` for the location form
(identity link on `mu`), which reaches the same fitted law.
`TwoPieceNormalLSS` has no mean form: `E[Y] = location + 2 * skew * scale *
sqrt(2/pi)` is a functional that `predict()` returns, not a natural parameter.

**The skew wall.** `skew` carries a two-wall logit on `(-skew_bound,
skew_bound)` with `skew_bound = 0.9`. Both parametrisations are positive
definite over the whole grid, and at the symmetric point the mean form is the
better conditioned of the two (condition number 21.4 against the location
form's 29.7 at `scale = 0.5`). Where the mean form degrades is the skew wall:
at `scale = 1.5`, `skew = 0.9` its condition number is 3.2e3 against the
location form's 56.8, which is why the wall stops at `0.9` rather than at
`0.99`. Sample skewness is also bounded: the family reaches only `+/-0.9655`
skewness at `|eps| = 0.899`, and the moment initialiser clamps beyond that
with a warning rather than failing.

**Weights.** Frequency weights replicate rows exactly. Non-unit prior weights
are refused: the family is not reproductive, so averaging has no likelihood
law. Fit claim-level rows and carry exposure through offsets, or declare
`weight_semantics="frequency"` for integer replication.

**The kink.** The log-likelihood is C1 and its Hessian jumps at `w = 0`. This
is not smoothed away. Kimber (1985) proves the MLE remains asymptotically
normal despite the break, and a refusal at the kink surfaces as a named
`smoothing_convergence_reason_` rather than a silent flag.

**Functionals.** `predict_cdf(X, y)` and `predict_quantile(X, p)` are
piecewise closed form, split at the mode; `predict()` is the conditional mean.
The two-piece families do not currently certify expected shortfall, so the
named posterior quantity refuses.

**Oracles.** At `skew = 0`, `TwoPieceLogNormalLSS` equals `GeneralizedGammaLSS`
at shape 0 (the log-normal) and `TwoPieceNormalLSS` equals `GaussianLS`, both
to 1e-14. Against gamlss `SN2` the mapping is exact, normalising constant
included: `nu^2 = (1 + eps) / (1 - eps)` and `sigma_SN2 = scale * sqrt(1 -
eps^2)`.

**References.** Mudholkar and Hutson (2000), *Journal of Statistical Planning
and Inference* 83:291-309, for the epsilon-skew form; Arellano-Valle, Gomez and
Quintana (2005), *Journal of Statistical Planning and Inference* 128:427-443;
Rubio and Steel (2014), *Bayesian Analysis* 9:1-22, whose Theorem 3 and
Corollary 2 give the two exact zeros in the expected information; Wallis (2014),
*Statistical Science* 29:106-112, for the two-piece variance; Kimber (1985),
*Journal of the Royal Statistical Society B* 47:16-19, for the MLE's asymptotic
normality across the kink.

## Generalized Pareto excess scale–shape model

`GeneralizedParetoLSS` is the generalized Pareto distribution of **excesses
over a threshold**, with two predictors: `scale` (log link) and `shape` (a
two-wall logit, default walls `(0, 1)`). The response is `y - u` for the rows
above a threshold `u` that you choose; the threshold is not a family argument.

```python
from superglm import Spline, SuperLSS
from superglm.distributional import GeneralizedParetoLSS, Predictor

threshold = float(claims["loss"].quantile(0.9))
above = claims["loss"] > threshold
tail = SuperLSS(
    family=GeneralizedParetoLSS(),
    predictors=(
        Predictor("scale", {"DrivAge": Spline(kind="cr", k=8)}),
        Predictor("shape", {}),
    ),
).fit_reml(claims.loc[above], claims.loc[above, "loss"] - threshold)
```

**Why the walls.** A non-negative shape keeps the support `[0, inf)` for every
row, so no row's support depends on its fitted parameters; a shape below one
keeps the mean `scale / (1 - shape)` finite, which is what `predict()` returns.
A negative lower wall is refused: it needs a response-dependent support slot the
engine does not have yet. Narrow the walls with
`GeneralizedParetoLSS(shape_lower=..., shape_upper=...)` when the tail index is
known from elsewhere.

**Weights.** Frequency weights replicate rows. Non-unit prior weights are
refused: the family is not reproductive, so there is no averaging law. Fit
claim-level excess rows and carry exposure through offsets.

**Functionals.** `predict_cdf(X, y)` and `predict_quantile(X, p)` are closed
form; `predict()` is the conditional mean of the excess. Certified expected
shortfall is available for the unit excess law.

**Splice recipe.** A severity model for the whole book is three fits, not one:

1. a body model on `y` for the rows at or below `u` (`GammaLS`,
   `GeneralizedGammaLSS`, or a log-scale Gaussian);
2. the exceedance probability `P(Y > u | x)` from a binary GLM;
3. `GeneralizedParetoLSS` on `y - u` for the rows above `u`.

The spliced survival above the threshold is
`P(Y > u | x) * (1 - predict_cdf(X, y - u))`, and a spliced quantile above `u`
is `u + predict_quantile(X, 1 - q / P(Y > u | x))` for `q < P(Y > u | x)`.
Choosing `u` is a modelling decision (mean-residual-life or stability plots),
not a fitted parameter, and the tail fit is conditional on it.

**Oracles.** `scipy.stats.genpareto(c=shape, scale=scale)`; gamlss `GP` with
`mu = scale/shape`, `sigma = 1/shape`.

## Tweedie mean–dispersion–power model

`TweedieLSS` has response support `[0, ∞)`, including a point mass at zero. Its
predictors must appear in the exact order `mean`, `dispersion`, `power`:

```python
from superglm import LambdaPolicy, Spline, SuperLSS
from superglm.distributional import Predictor, TweedieLSS

estimate = LambdaPolicy.estimate()
model = SuperLSS(
    family=TweedieLSS(power_lower=1.08, power_upper=1.92),
    predictors=(
        Predictor("mean", {"DrivAge": Spline(kind="cr", k=10, lambda_policy=estimate)}),
        Predictor(
            "dispersion",
            {"DrivAge": Spline(kind="cr", k=8, lambda_policy=estimate)},
        ),
        Predictor("power", {"DrivAge": Spline(kind="cr", k=8, lambda_policy=estimate)}),
    ),
).fit_reml(
    train_df,
    y_train,
    lambdas={
        "mean:DrivAge#wiggle": 1.0,
        "dispersion:DrivAge#wiggle": 1.0,
        "power:DrivAge#wiggle": 1.0,
    },
    max_reml_iter=120,
    reml_tol=1.0e-4,
    max_log_step=1.0,
    practical_reml=False,
)

parameters = model.predict_parameters(holdout_df)
```

`parameters` has columns `mean`, `dispersion`, and `power`; `predict()` returns
the conditional mean. The configured power walls are part of the fitted family
and artifact. They must satisfy `1 < power_lower < power_upper < 2`, and the
power link keeps fitted values strictly inside those walls.

The supported fitting route is dense with observed coefficient curvature for
both fixed smoothing parameters and automatic EFS smoothing. There is no Fisher
fallback for `TweedieLSS`. Public `SuperLSS` currently refuses `discrete=True`
for every family.

The example sets the outer iteration policy explicitly for reproducibility.
Always inspect reported convergence, the terminal residual, and curvature
provenance before relying on a fitted distributional model.

Prior precision is the default weight semantics: a row weight `w` acts through
Tweedie dispersion divided by `w`; it is not a row count. With explicit
`weight_semantics="frequency"`, integer weights instead reproduce the complete
normalized row law and learned geometry of literal replication.

**Functionals.** `predict_cdf(X, y)` and `predict_quantile(X, p)` evaluate the
unit compound Poisson–gamma law, including its atom at zero. Weight-aware
residual, scoring and posterior APIs use the separate prior-weighted CDF and
quantile when prior weights change the row law. The Poisson sum has a
20,000-order safety cap. At that cap the evaluator checks the actual omitted
tail mass and continues only when it is at most 10⁻¹²; an uncertified tail is
refused with the mass in the error. `TweedieLSS` does not currently certify
expected shortfall.

## Gaussian parameterization and scale floor

For response \(y_i\),

\[
y_i \mid x_i \sim N(\mu_i, \sigma_i^2), \qquad
\mu_i = \eta_{\mu i}, \qquad
\sigma_i = b + \exp(\eta_{\sigma i}).
\]

`GaussianLS(scale_floor=b)` serializes `b` as family configuration. The floor is
part of the model, not a post-fit clip. Consequently,

\[
\frac{\partial \sigma}{\partial \eta_\sigma}
= \frac{\partial^2 \sigma}{\partial \eta_\sigma^2}
= \exp(\eta_\sigma)
= \sigma-b.
\]

A scale coefficient therefore acts on \(\log(\sigma-b)\). For example, a
coefficient of `-0.07` means a one-unit increase multiplies the excess scale
\(\sigma-b\) by \(\exp(-0.07)\), about `0.932`. It does not directly subtract
`0.07` from standard deviation.

## Weights and offsets

`weight_semantics` declares what a `sample_weight` entry says about a row, on the
same contract the scalar path uses. Under `"prior"` — the default — a weight says
how precisely that row was measured; under `"frequency"` it says how many
identical rows it stands for. The proper prior-precision action is family
specific: Gaussian variance and Gamma dispersion are divided by `w`, while an
NB2 rate row obeys `wY ~ NB2(wμ, wθ)`. Frequency semantics instead repeats the
complete normalized row law `w` times.

The declaration decides one thing beyond bookkeeping: whether learned geometry —
spline knot placement, discretized-bin geometry — follows weight mass or physical
rows. Frequency weights are replication mass, so geometry follows them. Prior
weights are not row counts, so under the default geometry follows physical rows
instead, which is the geometry unweighted rows would have produced. Knot
strategies that read the weight (`quantile_rows`, `quantile_tempered`) move under
the declaration; `uniform` and plain `quantile` do not depend on it.

Scores follow the same declaration. Prior weights change each row's predictive
law, while score comparisons average over retained physical rows. Frequency
weights leave the row law at unit weight and contribute literal replication
mass to log score, CRPS, threshold-weighted CRPS, Murphy curves, paired standard
errors and default Murphy thresholds. Zero-weight rows are omitted. Candidates
with different declarations may be compared only when every retained weight is
exactly one.

Offsets are explicit and predictor-keyed:

```python
model.fit(
    X_train,
    y_train,
    sample_weight=case_weights,
    offsets={
        "location": location_offset,
        "scale": scale_offset,
    },
)
```

Offsets enter their named linear predictors. An ambiguous scalar `offset=` keyword
is deliberately unavailable.

## Separated cells

A categorical level that carries exposure but whose responses all sit on the
response boundary (every burn cost zero for Tweedie, every count zero for
NB2) has no finite effect: the likelihood keeps increasing as the level's
predictor walks to infinity. `SuperLSS(separation="warn")` (the default)
scans every `Categorical` term and `CategoricalInteraction` on every
predictor the family says can escape (mean and dispersion for `TweedieLSS`,
mean and theta for `NegativeBinomialLS`) before any coefficient is fitted and
emits a `SeparationWarning` naming the levels and the remedies; `"error"`
refuses the design; `"ignore"` skips the scan. `GaussianLS` and `GammaLS`
have no reachable boundary and are never scanned. Ordered-categorical smooths
are not scanned. The scan does not cover all separation patterns, and
`fit.curvature_indefinite` is a curvature diagnostic, not a separation detector.
The absence of either warning does not rule out separation.

## Fixed penalties and EFS

`fit()` performs a fixed-smoothing-parameter fit. Fully qualified names prevent
terms on different predictors from colliding:

```python
model.fit(
    X_train,
    y_train,
    lambdas={
        "location:DrivAge#wiggle": 0.4,
        "location:VehAge#wiggle": 0.7,
        "scale:DrivAge#wiggle": 1.2,
    },
)
```

A `LambdaPolicy.fixed(value)` on a feature is authoritative and need not be
duplicated in `lambdas`. Every other penalty component needs a supplied value for
`fit()`.

### How smoothing parameters are chosen

`fit_reml()` estimates every component whose policy is `LambdaPolicy.estimate()`
by minimising the negative Laplace-approximate marginal likelihood (LAML) in
log λ over the box `[1e-6, max_lambda]`. The default `outer="efs"` uses the
generalised Fellner–Schall fixed-point iteration, safeguarded by objective
backtracking. Set `outer="efs+newton"` to opt into a two-stage search:

1. **Warm-up.** The generalised Fellner–Schall (EFS) fixed-point iteration of
   Wood and Fasiolo (2017), safeguarded by objective backtracking, moves λ from
   its start while its steps are large. It hands over once its largest accepted
   step falls to 0.5 in log λ, or after ten iterations, whichever comes first.
   In this opt-in mode, `practical_reml=True` only shortens the warm-up.
2. **Endgame.** Newton on the exact gradient and Hessian of the LAML in log λ,
   the construction of Wood, Pya and Säfken (2016): the implicit derivative of
   the coefficient mode through the observed penalised Hessian, the derivative
   of `log|H|` through the third derivatives of the row log-likelihood
   (analytic for `GaussianLS` and `GammaLS`, finite differences of the packed
   row curvature otherwise, each carrying a certificate), and the fourth
   derivatives by second differences of the same rows. The Hessian is made
   positive definite by a growing ridge, the step is capped at `max_log_step`
   and accepted by Armijo backtracking; when the Hessian's certificate exceeds
   a tenth of its smallest active diagonal the iteration steps by damped BFGS
   instead. The search stops at a stationary point of the box-constrained
   problem: the largest projected gradient component below
   `reml_tol * (1 + |LAML|)`, the objective change below the same bar, and the
   Newton step it would still take below `reml_tol` in log λ.

With the default `outer="efs"`, `practical_reml=True` permits a sustained
objective-and-parameter plateau to stop the fit. `smoothing_convergence_reason_`
reports how the search ended. `stationary` is the optional endgame's converged
stop; `lambda_change` and `objective_plateau` are the Fellner–Schall fixed
point's. `gradient_unresolved` means a component's gradient certificate exceeded
the stationarity bar, so the optimum could not be certified; the certificate is
published in `training_telemetry()`.

A component at `max_lambda` whose gradient still points outward beyond the bar
is assessed at the exact face: the endpoint LAML derivative at τ = 1/λ = 0
decides whether the optimum is infinite (the term is linear and becomes an
exact face, `exact_face_components_`, `converged=True`) or finite. A finite
verdict with the cap still under pressure starts a bracketed search on
dF/dτ between the cap and the conditioning limit `1e14`; a root there
releases the component above `max_lambda` (`beyond_cap_components` on the
smoothing result) and the endgame resumes. A cap never certifies an infinite
optimum: without a root or an endpoint certificate the fit reports
`lambda_cap_unresolved`, and `smoothing_unresolved_upper_bound_` names the
component. A component at the cap whose gradient is within the bar is a
stationary point of the box and is reported as such.

`smoothing_certified_` (`matched_certified`) means: the terminal stop carries
its own authority — the projected gradient and every gradient certificate
within the bar for `stationary`, the Fellner–Schall residual within `reml_tol`
otherwise — no curvature fallback occurred, no endpoint assessment was refused,
and every exact face carries analytic endpoint evidence. When the Newton
endgame computes the exact Hessian of the LAML in log λ, it publishes that
matrix as `smoothing_hessian`, with its certificate, for smoothing-parameter
uncertainty. A stationary fit can stop before that Hessian pass. If it retained
the authenticated training rows, corrected-covariance inference can replay the
Hessian once without mutating or republishing fit state; a compact fit without
a published Hessian refuses corrected covariance.

`outer="efs"` runs the Fellner–Schall loop alone and keeps its stopping rules:
`practical_reml` stops after three accepted updates whose relative LAML change
is at most `reml_plateau_tol` and whose largest relative fitted-parameter change
is at most `practical_reml_parameter_tol` (for source value `a` and candidate
`b` the rowwise change is `abs(b - a) / (1 + max(abs(a), abs(b)))`), and
`practical_reml=False` requires the fixed point's own stationarity. That mode
is start-dependent on some problems; vary `initial_lambda` when comparing fits.

```python
model.fit_reml(X_train, y_train)

# The Fellner--Schall fixed-point route without the exact-LAML endgame:
efs_model.fit_reml(X_train, y_train, outer="efs", practical_reml=False)
```

The coefficient engine is established IRLS/PIRLS/Fisher–Newton: repeated
penalized weighted least-squares solves using the chosen joint curvature. The
outer EFS/LAML layer chooses smoothing parameters around those coefficient fits.
The engineering work is in normalized family laws and derivatives, weight and
carrier contracts, joint curvature, automatic smoothing, assembly, inference,
and certification. IRLS itself is old, standard machinery and is not an
originality claim.

## Prediction and inference

The complete joint covariance and effective degrees of freedom use one terminal
curvature source:

```python
import numpy as np
import pandas as pd

result = model.result_

coefficient_table = pd.DataFrame(
    {
        "estimate": pd.Series(model.coef_),
        "standard_error": np.sqrt(np.diag(model.covariance_)),
    },
)

print(result.total_effective_df)
print(dict(result.predictor_edf))
print(dict(result.intercept_edf))
print(dict(result.term_edf))
print(result.curvature_telemetry.to_dict())
```

The covariance is in the same qualified global coordinates as `model.coef_` and
retains location–scale cross-blocks. EDF is computed by solves from the joint
influence matrix, with predictor intercepts and qualified terms separately
attributed. Negative local EDF contributions, if present, are retained rather
than clipped.

Coefficient Wald calculations are possible from this state, but smooth-basis
coefficients are not usually the scientific estimand. Prefer parameter surfaces,
out-of-sample likelihood, interval calibration, and whole-term EDF when
interpreting smooth distributional models.

Distribution functions answer query points outside the response support. For a
positive continuous law, every finite threshold `y ≤ 0` has CDF zero; this does
not relax likelihood admission, which continues to enforce the family's
declared response support. Tweedie's CDF instead includes its atom at zero.

Expected shortfall is owned by `ExpectedShortfallFamily`, not synthesised by
generic quantile quadrature. The certified unit-law implementations are
`GaussianLS`, `GammaLS`, `LogNormalLS`, `GeneralizedGammaLSS` and
`GeneralizedParetoLSS`. Only Gaussian and gamma also certify non-unit
prior-weighted expected shortfall. `TweedieLSS`, both two-piece families and
`NegativeBinomialLS` refuse it. Structural support does not promise that every
float64 tail is resolvable: Gamma, log-normal and generalized-gamma rows refuse
when their numerical tail certificate fails. Generalized-gamma location form
returns `+∞` when the first moment does not exist, while mean form rejects that
cell as outside its model. Named posterior intervals refuse a non-finite plug-in
value or any non-finite posterior draw rather than reducing it.

### Inference suite

`SuperLSS` carries checking, inference, and story-telling tools on top of the
fitted state. Parameter and term inference use the fitted coefficient state.
Quantile residuals, calibration, tail surfaces, predictive simulation, and
non-log scores additionally require a family implementing
`DistributionFunctionFamily`; methods refuse when that capability is absent.
Available tools include the six-panel diagnostic, binned checks and calibration
tables, per-parameter term effects with Wood (2013) tests and a summary table,
posterior bounds on derived quantities, risk curves, the spread among
identically priced rows, portfolio totals, and proper-score comparison between
candidates.

```python
r = model.residuals(X_holdout, y_holdout)          # randomised quantile residuals
figure = model.plot_diagnostics(X_holdout, y_holdout)
table = model.summary()                            # edf, statistic, p per (parameter, term)
effect = model.term_inference("scale", "x")        # bands on one term of one parameter
bounds = model.posterior_bounds(X_new, ("quantile", 0.99), level=0.9)
```

By default the suite holds smoothing parameters fixed. With
`covariance="corrected"`, it uses a trusted smoothing Hessian published by a
stationary fit or performs one authenticated, on-demand replay from retained
training rows. That replay is ephemeral. A compact fit without a published
Hessian refuses, as does a fit whose terminal state cannot be authenticated.
Term effects, plots and summaries validate their requested work before
resolving the covariance, and resolve it at most once per top-level call. The
simultaneous critical value is floored at the pointwise normal critical value,
so simulation cannot produce a narrower simultaneous band.

Every builder returns a payload with a JSON-clean `to_json()`, and
`model.plot_data(kind, ...)` returns that dictionary directly.
[Checking a distributional fit](distributional-inference.md) walks the whole
suite in the order a review asks its questions, with a "how to read it" box per
figure and the papers each method comes from.

## Worked FreMTPL2 severity comparison

This example asks a concrete question: does policy information explain
conditional variation in claim severity after it has already entered the mean?
It aggregates multiple claim rows to one claimant policy, models
`log(total claim amount)`, and compares:

1. a conventional Gaussian location model with one constant scale; and
2. the same location predictor plus policy features in the scale predictor.

This is a severity example. Exposure is not used as a weight because the sample
is already conditional on a claimant policy.

```python
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

from superglm import Numeric, SuperLSS
from superglm.distributional import GaussianLS, Predictor

data_dir = Path("data")
frequency = pd.read_csv(data_dir / "freMTPL2freq.csv")
severity = pd.read_csv(data_dir / "freMTPL2sev.csv")

severity = severity.groupby("IDpol", as_index=False)["ClaimAmount"].sum()
data = severity.merge(frequency, on="IDpol", validate="one_to_one")
data["LogDensity"] = np.log1p(data["Density"])
data["LogClaimAmount"] = np.log(data["ClaimAmount"])

rng = np.random.default_rng(20260723)
order = rng.permutation(len(data))
train = data.iloc[order[:20_000]].copy()
holdout = data.iloc[order[20_000:]].copy()

columns = ["DrivAge", "VehAge", "BonusMalus", "LogDensity"]
for column in columns:
    centre = train[column].mean()
    scale = train[column].std()
    train[column] = (train[column] - centre) / scale
    holdout[column] = (holdout[column] - centre) / scale


def linear_features():
    return {column: Numeric() for column in columns}


def fit_candidate(scale_features):
    return SuperLSS(
        family=GaussianLS(scale_floor=0.05),
        predictors=(
            Predictor("location", linear_features()),
            Predictor("scale", scale_features),
        ),
    ).fit(
        train[columns],
        train["LogClaimAmount"].to_numpy(),
        retain_rows=False,
    )


constant_scale = fit_candidate({})
location_scale = fit_candidate(linear_features())


def holdout_metrics(fitted):
    predicted = fitted.predict_parameters(holdout[columns])
    y = holdout["LogClaimAmount"].to_numpy()
    z = (y - predicted["location"].to_numpy()) / predicted["scale"].to_numpy()
    nll = np.mean(
        np.log(predicted["scale"].to_numpy())
        + 0.5 * np.log(2.0 * np.pi)
        + 0.5 * z**2,
    )
    coverage_90 = np.mean(np.abs(z) <= norm.ppf(0.95))
    return float(nll), float(coverage_90)


likelihood_ratio = 2.0 * (
    location_scale.result_.log_likelihood
    - constant_scale.result_.log_likelihood
)
p_value = chi2.sf(likelihood_ratio, df=4)
```

Using local FreMTPL2 CSVs with SHA-256
`0f363eca0bd43d3ab83c0f54bf99e7d0a328814cd6c0a1a3d1380de3032f471d`
(frequency) and
`7328d2ac07bfbb72aec7723982b8c60cfa626dbfd4946a10f75832d5d65a6f84`
(severity), the deterministic 20,000/4,944 split produced:

| Measure | Constant scale | Location–scale |
| --- | ---: | ---: |
| Holdout mean Gaussian NLL | 1.53048 | 1.52252 |
| Holdout nominal 90% interval coverage | 88.49% | 88.69% |
| Predicted scale, 1st percentile | 1.13492 | 0.89507 |
| Predicted scale, median | 1.13492 | 1.14083 |
| Predicted scale, 99th percentile | 1.13492 | 1.27924 |
| Training AIC | 61,831.92 | 61,622.56 |

The nested likelihood-ratio statistic is `217.35` on four added scale
coefficients (`p = 6.96e-46` under the ordinary model assumptions). The fitted
scale-link effects and joint observed-curvature standard errors were:

| Scale term | Estimate | Standard error | 95% interval |
| --- | ---: | ---: | ---: |
| `DrivAge` | -0.07223 | 0.00576 | [-0.08351, -0.06095] |
| `VehAge` | 0.00656 | 0.00543 | [-0.00407, 0.01720] |
| `BonusMalus` | -0.07761 | 0.00587 | [-0.08911, -0.06611] |
| `LogDensity` | 0.01073 | 0.00534 | [0.00027, 0.02120] |

On this standardized specification, older-driver and higher-bonus-malus rows
have lower conditional variation in log severity after controlling for the
location predictor. `VehAge` has no clear scale effect in this linear
specification. These are associations, not causal effects.

The terminal inference used observed curvature, rank `10/10`, condition estimate
`2.20`, and zero fallback. The holdout log score improves and the model discovers
material scale variation, but nominal coverage remains below 90%. That is useful
model-risk evidence: location–scale structure helps, while a Gaussian model on
log severity still does not fully describe the tail.

This worked example is descriptive, not release-certification evidence. Its
ordinary Wald and likelihood-ratio calculations assume independent policy rows
and an adequate Gaussian log-severity likelihood. Model specification was
explored on this dataset, so the displayed `p` value must not be presented as a
preregistered confirmatory result.

## Discrete fitting

Discrete fitting is not currently available through `SuperLSS`.
Constructing `SuperLSS(..., discrete=True)` raises `NotImplementedError`. The
internal multi-parameter prototype remains for development, but it is neither a
public correctness nor performance claim. Scalar `SuperGLM` discrete fitting
is unchanged and remains available.

## Curvature choice

The dense coefficient solve is Newton's method on the observed Hessian for
every family. `SuperLSS(coefficient_curvature="fisher")` asks for Fisher
scoring instead and is accepted only for a family that supplies expected
information. Every current built-in except `NegativeBinomialLS` and
`TweedieLSS` supplies it; a family implementing that capability does not change
the default solve. Under either policy the accepted terminal point is
re-examined with the penalized observed Hessian `A + S`, restricted to the
active coefficient subspace when needed. If it is materially indefinite the
solver retries with a tighter tolerance, then falls back to penalized expected
information `F + S` when the family supplies it and refuses the fit otherwise.
The requested policy
is `training_telemetry().curvature_policy`;
the terminal source and any fallback are in `training_telemetry().curvature`
(`actual_source`, `fallback_count`, `matrix_kind`).

```python
newton = SuperLSS(family=GammaLS(), predictors=predictors)
scoring = SuperLSS(family=GammaLS(), predictors=predictors, coefficient_curvature="fisher")
```

## Derivative orders, and which families supply them

Each level of the fit asks the row log-likelihood for two more derivative
orders than the level below it, and there are only two levels, so the ladder
ends at four:

| Level | What is solved | Orders of the row log-likelihood |
|---|---|---|
| Coefficient solve at fixed λ (Newton) | the penalised mode `β̂(λ)` | 0, 1 and 2: value-only backtrack screens, then score and observed Hessian with every cross term included |
| Fellner–Schall fixed point (`outer="efs"`) | λ from `β̂(λ)` and `H` | 2 only: it reuses the Hessian and never differentiates it |
| LAML gradient in log λ (the Newton endgame's step direction, and the stationarity certificate) | `dV/dρ` through `dβ̂/dρ` | 3: the derivative of `H` along the direction the mode moves, which is the third derivative contracted with `X dβ̂/dρ` |
| LAML Hessian in log λ (the Newton endgame's step, and `smoothing_hessian`) | `d²V/dρ²` through `d²β̂/dρ²` | 4: the fourth derivative contracted with two such directions, plus the third contracted with `d²β̂/dρ²` |

The third order appears because the coefficient mode is itself a solution of
the inner problem: by the implicit function theorem its derivative in ρ is
`−(H + S)⁻¹ S_j λ_j β̂`, and differentiating `log|H + S|` along that path
differentiates `H`. The fourth appears the same way one level up. Nothing
needs a fifth: the outer problem is the last one, and its Newton step is the
most derivative-hungry thing anyone runs. Wood, Pya and Säfken (2016) is the
reference for the whole construction.

Every family supports orders 0, 1 and 2. Order 0 supplies likelihood values
without derivatives for inexpensive rejection of repeated backtracks; every
accepted point still receives a full order-2 evaluation. Wherever order 2 is
valid, order 0 must also be valid and return the same optimizing values and
carrier terms.
Built-in families supply orders 1 and 2 analytically as one fused row kernel
(`evaluate_natural`: log density, score and the packed cross-derivative
Hessian in its natural parameters, chained through the links by the solver).
Order 3 is optional (`PredictorCurvatureDirectionalFamily`, the directional
derivative of the packed curvature along a predictor direction); when a family
does not implement it the endgame differences the packed rows with a
four-point stencil per axis and carries a certificate. Order 4 is always
obtained by second differences of the packed rows today, on top of an analytic
or differenced third. So the analytic hook makes the gradient exact and saves
the third-order stencils, but the fourth-order stencils remain.

| Family | Row log-likelihood | Fused order-2 kernel | Analytic order 3 | Order 4 |
|---|---|---|---|---|
| `GaussianLS` | smooth to every order | yes | yes | differenced |
| `GammaLS` | smooth to every order | yes | yes (built-in log links only) | differenced |
| `NegativeBinomialLS` | smooth to every order in mean and size (log-gamma and its derivatives) | yes | not implemented; possible | differenced |
| `TweedieLSS` | smooth in mean, dispersion and power on `1 < p < 2` (Wright function series) | yes | not implemented; possible but the power derivatives are series | differenced |
| `LogNormalLS` | smooth to every order | yes | not implemented; short | differenced |
| `GeneralizedGammaLSS` | smooth to every order; `Q = 0` is removable and the kernel's series cover it | yes | not implemented; needs the tetragamma remainder and one more derivative of each series function, and the mean form's third-order chain | differenced |
| `GeneralizedParetoLSS` | smooth to every order on `0 < ξ < 1` | yes | not implemented; short | differenced |
| `TwoPieceNormalLSS`, `TwoPieceLogNormalLSS` | `C¹` only: the curvature jumps at the mode `w = 0` | yes, away from the kink | **not possible**: the third derivative at the kink is a point mass, and the LAML gradient's true value includes the density at the kink times the curvature jump, which a pointwise formula drops | differenced |

For the two-piece families the differenced route with its certificate is the
correct method, not a shortcut: the engine sees rows cross the kink and the
certificate reports when it cannot resolve the gradient (`gradient_unresolved`,
seen at small `n`). The same holds for any likelihood with a kink or a
boundary, the Laplace, quantile and Huber losses among them: they fit on the
Fellner–Schall route and are certified by differences, and no analytic
endgame exists for them. A family is a candidate for the analytic hook exactly
when its row log-likelihood is `C³` in every natural parameter over the whole
support the links can reach.

## Telemetry and serialization

`training_telemetry()` returns compact immutable fit metadata without requiring
retained training rows:

```python
telemetry = model.training_telemetry()
print(telemetry.curvature_policy)
print(telemetry.curvature.actual_source)
print(telemetry.rank, telemetry.converged)
print(telemetry.discrete, telemetry.n_bins)
```

Serialize the complete fitted revision with the versioned public artifact:

```python
from pathlib import Path

Path("severity.superlss").write_bytes(model.to_bytes())
restored = SuperLSS.from_bytes(Path("severity.superlss").read_bytes())
restored_parameters = restored.predict_parameters(scoring_data)
```

The artifact preserves family configuration, links, compiled predictor state,
penalties, coefficients, joint covariance, EDF, convergence and fallback
telemetry, null model, and discrete configuration. Its JSON envelope and payload
digests detect corruption. The payload contains Python pickle data for learned
feature objects, so load artifacts only from trusted sources; a digest does not
make hostile pickle safe.

Artifacts carry two independent version numbers. The envelope's
`schema_version` versions the fitted state itself, and **MAJOR is a read
barrier unless explicitly supported**. This build writes envelope `9.0.0`
and also reads `8.0.0`. New fits record whether curvature diagnostics assessed
the data matrix or the penalized matrix; older fits keep their original
family-dependent interpretation. Unsupported majors receive a version error
naming both versions. MINOR and PATCH promise readability. Because the loader
recomputes the manifest and compares it for
equality, any change to what the manifest describes must be a MAJOR bump — so a
manifest mismatch always means corruption or tampering, never age. The
`public_api` block carries its own `schema_version` for the SuperLSS wrapper
metadata alone; it moves independently of the envelope's, and both are written
into every artifact so that whichever layer refuses a stale artifact names its
own version. The wrapper remains at `2.0.0`. Envelope majors below 8 identify
development-only shapes that are refused, not compatibility promises.

### Diagnosing a fit

`diagnose()` explains solver and smoothing behaviour from the accepted fitted
revision without reading any training rows:

```python
report = model.diagnose()
print(report)  # compact: the work profile, then finding headlines
print(report.render(detail="full"))  # adds evidence, caveats, and limitations
payload = report.to_dict()  # plain JSON containers; report schema_version 2
```

The report leads with the fit's work profile (`report.profile`):

- rows, coefficients, and the wall time of the fit this object ran;
- the work done: outer EFS iterations, coefficient fits, inner iterations,
  and how many outer proposals were rejected or backtracked;
- the time distribution over the recorded phases (likelihood evaluation,
  curvature and gradient assembly, decomposition solves, EFS updates and
  backtracking, terminal inference, ...) with each phase's share of the fit
  and its call count; time the phases do not cover is reported as
  orchestration and unmeasured;
- one row per smoothing component: initial and final lambda, accepted moves,
  how often it led the largest accepted move, iterations spent at the cap, the
  term's terminal EDF, and its outcome: `finite`, `fixed` by the caller,
  `upper_bound`, `unresolved_cap`, or `exact_face` with the iteration that
  activated the face and what the face left (`linear_only` for a plain cubic
  regression spline, `null_space_only`, `fully_suppressed`, or `unresolved`).

Findings follow the profile, ranked as before. Phase timings are measurements
of this fit on this machine, not benchmark claims or per-feature attribution.
A model restored with `from_bytes` has no machine timing: its report renders
the same work counts and smoothing metrics, says that timing is unavailable,
and the artifact schema is unchanged.

## Current limits

- `TweedieLSS` is dense and observed-curvature only; Fisher fallback and
  discrete execution are unavailable.
- `NegativeBinomialLS` is dense and observed-curvature only. It has no
  zero-inflation component, discrete execution, or exact Poisson active face.
- `GeneralizedGammaLSS`, `GeneralizedParetoLSS`, `TwoPieceLogNormalLSS` and
  `TwoPieceNormalLSS` are dense-path only and are certified through the generic
  finite-difference endpoint authority, so a certified face is converged but
  not `matched_certified`.
- Shape constraints (`Constraint.fit.*` and `Constraint.postfit.*` on a
  feature) are not applied on the distributional path: the smooth is fitted
  unconstrained and `ShapeConstraintIgnoredWarning` is emitted at compile time.
  Constrained smooths on this path are a planned feature with their own design.
- `LogNormalLS` is dense-path only and is certified through the generic
  finite-difference endpoint authority, so a certified face is converged but
  not `matched_certified`.
- Cross-predictor penalties are not supported.
- A scalar offset is intentionally rejected.
- `predict()` returns Gaussian location; transformations such as a lognormal
  original-scale mean are the caller's explicit modeling decision. `LogNormalLS`
  is the supported way to model that directly: it fits the same likelihood on
  `y` and returns the original-scale mean from `predict()`.
- Dedicated higher-dimensional interaction-surface confidence-band helpers and
  broader count LSS families are future work; the joint covariance needed for
  downstream inference is retained.

The scale predictor is not a second price. For an ordinary expected-loss tariff,
the selected technical severity remains the conditional mean. Scale becomes
decision input when an explicitly chosen business functional needs conditional
quantiles, prediction intervals, layer costs, capital, or a risk margin.
