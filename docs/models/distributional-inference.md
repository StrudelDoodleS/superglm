# Checking and Explaining a Distributional Fit

A `SuperLSS` fit states a whole conditional distribution per row, so the
questions you can ask of it are wider than the ones a mean model answers. This
page walks the questions in the order a review actually asks them:

1. **Is the family right?** — Q-Q, worm and PIT.
2. **Where is it wrong, and in which moment?** — binned checks, Q-statistics,
   actual against expected, calibration.
3. **What drives each parameter?** — term effects and the summary table.
4. **What does it mean for a policy and for a book?** — risk curves, the
   density fan, the spread among identically priced rows, the portfolio total.
5. **How do candidates compare?** — proper scores, the Murphy diagram and the
   tail tables.

Every method below is a thin call on the fitted model. Underneath, one
primitive does the work: draws from the Bayesian posterior of the coefficients,
pushed through the family. Every builder returns a frozen payload with a
`to_json()`, so a figure can be redrawn from its payload alone —
[`plot_data`](#payloads-without-figures) hands you exactly that.

## The example

Everything on this page runs on simulated data with a known truth. Here `x` and
`z` are covariates, `band` is a level factor, and `exposure` is a prior weight.

```python
import numpy as np
import pandas as pd

from superglm import Categorical, Spline, SuperLSS
from superglm.distributional import Predictor
from superglm.distributional.families.gamma import GammaLS

rng = np.random.default_rng(20260903)
n = 20_000
frame = pd.DataFrame(
    {
        "x": rng.uniform(-1.0, 1.0, n),
        "z": rng.uniform(-1.0, 1.0, n),
        "band": rng.choice(["low", "mid", "high"], n),
    }
)
exposure = rng.uniform(0.2, 1.0, n)

mean = np.exp(0.8 + 0.6 * np.sin(np.pi * frame["x"]) + 0.3 * frame["z"] ** 2)
cv = np.exp(-0.4 + 0.3 * frame["x"])
y = rng.gamma(exposure / cv**2, mean * cv**2 / exposure)

model = SuperLSS(
    family=GammaLS(),
    predictors=[
        Predictor("mean", {"x": Spline("cr", k=8), "z": Spline("cr", k=8), "band": Categorical()}),
        Predictor("scale", {"x": Spline("cr", k=8)}),
    ],
).fit_reml(frame, y, exposure)
```

The response is a rate: each row's gamma has an exposure inside its own law, so
`exposure` is a **prior** weight and not a count of replicated rows. That single
fact decides how every weight on this page is read; the
[weights section](#weights-rates-are-ratios-of-sums) says how.

!!! note "Capabilities differ by family"
    Methods that need a fitted CDF or quantile refuse a family that does not
    implement `DistributionFunctionFamily`; predictive simulation also draws
    through that quantile. In particular, `NegativeBinomialLS` supports log
    scores and parameter-based inference but not distributional residuals,
    calibration, tail surfaces, or predictive simulation. Check the family
    guide or the method's documented requirement before choosing a diagnostic.

Hold out a sample and check on that, exactly as you would for a mean model. The
methods take any frame the model can predict on.

## 1. Is the family right?

For a family implementing `DistributionFunctionFamily`, the residual that
makes this question answerable is the **randomised quantile residual** of Dunn
and Smyth (1996): put the response through its own fitted distribution
function, `u = F(y | θ̂)`, and then through the inverse normal. If the family
and fitted parameters are right, `u` is uniform and the residual is standard
normal.

```python
r = model.residuals(frame, y, sample_weight=exposure)      # Φ⁻¹(u), one per row
u = model.residuals(frame, y, kind="pit", sample_weight=exposure)
```

One figure carries the whole answer:

```python
figure = model.plot_diagnostics(frame, y, sample_weight=exposure)
```

Six panels: the Q-Q plot with a simulated envelope, the worm plot, the PIT
histogram, the residual density against the standard normal, the residuals
against the first parameter's linear predictor, and the residual standard
deviation in bins of the second's.

!!! note "How to read the Q-Q panel"
    The points are the sorted quantile residuals against their theoretical
    order statistics; the shaded band is the envelope of Fasiolo, Nedellec,
    Goude and Wood (2020), simulated from the fitted model itself, so it is a
    statement about *this* fit at *this* sample size rather than a generic
    reference line. Points inside the envelope mean the family is not
    contradicted. A run of points above the band at the right-hand end is a
    tail heavier than the family allows; below it, a tail that is too light.
    Curvature through the middle is a shape problem, not a tail problem.

!!! note "How to read the worm"
    The worm is the Q-Q plot with the diagonal subtracted (van Buuren and
    Fredriks 2001), which is what makes small departures visible at all. A
    flat worm inside its band is a fit with nothing to explain. The shape
    names the defect: a rising line means the residuals are shifted, a U means
    they are too spread out (or too tight, inverted), and an S means the
    skewness is wrong. Passing a covariate splits it into one panel per
    interval of that covariate, so the defect can be located:

    ```python
    worm = model.plot_data("worm", X=frame, y=y, covariate="x", sample_weight=exposure)
    ```

!!! note "How to read the PIT histogram"
    The bars are the PIT values in twenty bins with a binomial band
    (Gneiting, Balabdaoui and Raftery 2007). Flat is calibrated. A U shape
    means the predictive distributions are too narrow — reality lands in the
    tails more often than the model says. A hump in the middle means they are
    too wide. A slope means a systematic bias in location.

For a family with a point mass — `TweedieLSS` has one at zero — the transform
is *randomised* across the jump, which is what keeps it uniform; the payload
reports how many rows were randomised, and how many had to be clipped away from
0 or 1.

A CDF query is not likelihood admission. Positive continuous families return
zero at every finite threshold `y ≤ 0`, including in a vector that also has
interior thresholds, while fitting still enforces the family's declared
response support. Tweedie is different at zero: its CDF includes the atom.

## 2. Where is it wrong, and in which moment?

A Q-Q plot says *that* the residuals are wrong; it does not say where. Binned
checks do (Fasiolo, Nedellec, Goude and Wood 2020): cut the rows by a covariate
and look at the residuals' mean, standard deviation and skewness per bin, each
with a bootstrap band.

```python
check = model.check(frame, y, "x", sample_weight=exposure)     # column name or array
pair = model.check_2d(frame, y, "x", "z", sample_weight=exposure)
```

!!! note "How to read a binned check"
    Three rows of panels against the covariate: the mean should sit at zero,
    the standard deviation at one, the skewness at zero, each within its
    bootstrap band. A band clear of its reference marks both *where* the fit is
    wrong and *in which moment*. A mean away from zero over part of the range
    is a location term that has not been given enough freedom; a standard
    deviation away from one is the scale predictor missing that covariate; a
    skewness away from zero is a family that cannot bend that far, which is an
    argument for a three-parameter family rather than for another knot. The
    two-dimensional version shows drift in a joint region that neither
    one-dimensional check reveals.

The same question, asked in the money units rather than in residual units, is
the actual-versus-expected table:

```python
table = model.actual_expected(frame, y, "band", sample_weight=exposure)
```

!!! note "How to read actual against expected"
    Per bin or level: the realised total `Σ w y`, the predicted total
    `Σ w μ̂`, their ratio, and the standard error of that ratio under the
    fitted law. Every number is a ratio of **sums**, never a mean of per-row
    ratios. A ratio more than about three standard errors from one is a bin
    the model is not paying for correctly. The `variance_law` field on the
    payload names which law the standard error was read from, so a table built
    on simulated variance is never mistaken for one built on a closed form.

Calibration asks the four questions a review asks of the *whole* distribution
rather than of its mean:

```python
calibration = model.calibration(
    frame, y, thresholds=(5.0, 10.0), sample_weight=exposure
)
calibration.coverage      # realised coverage per interval level, overall and by decile
calibration.tails         # expected against realised exceedances per threshold
calibration.quantiles     # realised exceedance rate of each predicted quantile
calibration.reliability   # CORP reliability curve per threshold
```

!!! note "How to read the calibration panels"
    *Coverage*: the realised share of rows inside each central predictive
    interval against its nominal level — a 90 % interval that holds 78 % of
    rows is a model that is too confident, and the per-decile rows say for
    which kinds of row. *Tails*: `Σ P(Y > t)` against the realised count, with
    a Poisson-binomial standard error; this is the number a reinsurance
    conversation actually turns on. *Quantiles*: each predicted `p`-quantile
    should leave `1 - p` of the rows above it. On a family with a point mass
    (a Tweedie burn-cost model is mostly zeros) both of these are read on the
    randomised PIT rather than on the response: a zero response would
    otherwise count as inside every central interval whose quantiles sit on
    the atom, and the 50 % interval would read 0.85 on an 83 %-zero book.
    `calibration.calibration_law` says which reading was used
    (`"response"` or `"randomised_pit"`). *Reliability*: the CORP diagram
    of Dimitriadis, Gneiting and Jordan (2021) recalibrates the exceedance
    forecast by isotonic regression, which chooses its own binning rather than
    taking one from you; the consistency bands show how far a diagram can
    stray while the forecast is in fact calibrated. A curve below the diagonal
    at high forecast probabilities means the model over-predicts that event.

The worm plot's numeric companion is the set of Q-statistics of Royston and
Wright (2000): per interval, the standardised mean, variance, skewness and
kurtosis of the residuals, each ~ N(0, 1) under the null. `|z| > 2` in the
variance column of one interval and nowhere else is a scale term that needs a
covariate, and it says so without anyone squinting at a curve.

## 3. What drives each parameter?

```python
figures = model.plot()                      # one figure per parameter, keyed by name
scale_figure = model.plot(parameter="scale")
effect = model.term_inference("scale", "x")
```

`term_inference` sweeps one term over its training range with every other
covariate held at its training centre, and reports the effect on that
predictor's own link scale.

!!! note "How to read a term panel"
    The curve is the term's contribution to its parameter's linear predictor;
    the filled band is the Bayesian pointwise interval of Marra and Wood
    (2012), and the outlined one is the max-deviation simultaneous band of
    Ruppert, Wand and Carroll (2003). Use the pointwise band to read one point
    and the simultaneous band to make a claim about the whole curve — "the
    effect is not flat" is a whole-curve claim, and the pointwise band will
    tell you so too often. Where the link is a log-type link the payload also
    carries `multiplier = exp(effect)`, which is the relativity a rating table
    would print. A categorical term reports one entry per fitted level with the
    reference level reading back as an exact zero. An ordered band declared
    with `specials=` carries its special levels as extra entries, flagged in
    `effect.special` and drawn in the flagged-point style, because their
    coefficients live in a separate `<term>:special` block of the layout.
    The simultaneous critical value is floored at the pointwise normal critical value,
    so Monte Carlo variation can never make the
    simultaneous band narrower than the pointwise band.
    In `model.summary()` those rows are labelled `"<term> (special level)"`,
    and the `note` column says `"absorbed by <interaction>"` when a
    categorical's coefficients are aliased by a factor-smooth deviation on the
    same feature: the coefficient is then a by-product to rebase, never a
    number to read.

The table over all of them:

```python
summary = model.summary()
```

One row per intercept and per term of every parameter, carrying the effective
degrees of freedom, the smoothing parameter, and the Wood (2013) test that the
term is flat, with its rank and p-value. Because the scale is itself modelled,
the reference distribution is a chi-squared and not an F.

!!! note "The training frame"
    Term grids need the frame the model was fitted on. It is kept **by
    reference** at `fit`/`fit_reml` — not copied — and it does not survive
    `to_bytes()`. A model restored with `from_bytes` must be given the frame
    explicitly, on any of the three methods:

    ```python
    restored = SuperLSS.from_bytes(model.to_bytes())
    restored.summary(X_train=frame)
    restored.term_inference("mean", "x", X_train=frame)
    ```

Bands and tests treat the smoothing parameters as fixed at their estimates.
Passing `covariance="corrected"` asks instead for the first-order correction of
Wood, Pya and Säfken (2016). A stationary fit can use a trusted smoothing
Hessian published by the Newton endgame, or authenticate the terminal fit and
replay that Hessian once from retained training rows. The replay is ephemeral:
it does not mutate, cache, or republish fit state. A compact fit without a
published Hessian refuses, as does a non-stationary or unauthenticated fit.
`term_inference()`, `term_test()`, `plot()` and `summary()` validate their term,
grid and frame inputs first, then resolve corrected covariance at most once per
top-level call; a multi-panel plot or summary cannot replay it term by term.

## 4. What does it mean for a policy and for a book?

The parameters are not the story a business reads. These four views turn a
fitted law into statements about a policy and about a book.

```python
reference = {"z": 0.0, "band": "mid"}
curves = model.risk_curves(reference, "x", quantiles=(0.5, 0.9, 0.99))
fan = model.density_fan(reference, "x")
```

!!! note "How to read the risk curves and the density fan"
    The curves are predicted quantiles of the **response** along one covariate,
    with everything else held at a reference row, each with a posterior band
    from one shared draw set (so the curves are coherent with one another, not
    independently simulated). A median that barely moves while the 99th
    percentile doubles is the whole argument for modelling the scale: the
    average policy is unchanged and the tail is not. The density fan shows the
    same sweep as a whole conditional density per point, which is where a shape
    change — mass moving into the tail, a mode splitting — becomes visible at
    all.

```python
spread = model.parameter_spread(frame, threshold=25.0, sample_weight=exposure)
spread.identically_priced      # per price bin: n, the mean price, and the tail-risk spread
```

!!! note "How to read the spread"
    Rows are binned by predicted mean, so every row in a bin is priced alike.
    Within each bin the table reports the 5th and 95th percentiles of
    `P(Y > threshold)` and their ratio. A ratio of 20 says that among rows a
    mean model prices identically, one is twenty times likelier to breach the
    threshold than another. That number is invisible to a mean model by
    construction: it is the quantity the second predictor exists to see.

```python
book = model.portfolio(frame, by="band", sample_weight=exposure)
book.total_quantiles       # median, 90th and 99th percentile of the simulated book total
book.by_segment            # the same per segment, means summing to the book mean
```

!!! note "How to read the portfolio total"
    Each row is simulated on its own predictive law and the draws are summed
    across rows, so the reported quantiles are of the **book** total and carry
    the dependence the shared coefficient draws induce — which is why the book
    99th percentile is not the sum of the rows' 99th percentiles. With prior
    weights, what the book pays is `Σ w y`: the rate times the exposure that
    bought it.

Under all four sits one primitive, and you can call it directly for any
quantity:

```python
bounds = model.posterior_bounds(
    frame, ("quantile", 0.99), level=0.9, n_draws=1000, sample_weight=exposure
)
draws = model.posterior_predictive(frame, 200, sample_weight=exposure)
total = model.posterior_predictive(frame, 500, reduce="sum", sample_weight=exposure)
```

`quantity` may be `("parameter", name)`, `("quantile", p)`,
`("exceedance", t)`, `("expected_shortfall", p)` or any callable of the `(m, k)`
parameter matrix, so a derived quantity gets a posterior interval without any
new machinery.

Expected shortfall is nevertheless a family-owned capability, not generic
quantile quadrature. `GaussianLS`, `GammaLS`, `LogNormalLS`,
`GeneralizedGammaLSS` and `GeneralizedParetoLSS` implement the certified unit
law through `ExpectedShortfallFamily`; only Gaussian and gamma also implement
the non-unit prior-weighted law. `TweedieLSS`, both two-piece families and
`NegativeBinomialLS` refuse the named quantity. Structural support does not
promise that every float64 tail is resolvable: Gamma, log-normal and
generalized-gamma rows refuse when their numerical tail certificate fails.
Generalized-gamma location form returns `+∞` when the first moment does not
exist, while mean form rejects that cell as outside its model. Named posterior
bounds require finite plug-in values and finite values on every posterior draw,
and refuse instead of reducing an interval containing infinities.

## 5. How do candidates compare?

Proper scores, and only proper scores: a rule is proper when the true
distribution is the one that optimises its expectation, which is what makes a
lower average evidence rather than an artefact (Gneiting and Raftery 2007).

```python
scores = model.scores(frame, y, which=("log", "crps"), sample_weight=exposure)
comparison = model.compare(other_model, frame, y, which="log", by="band")
comparison.overall     # mean difference, its standard error, the paired t and n
comparison.by_segment  # the same per segment
```

!!! note "How to read a score comparison"
    The comparison is paired row by row, so its standard error is that of the
    mean *difference* and not of two independent means — this is the difference
    between a comparison that resolves and one that never leaves the noise. A
    negative mean difference favours the model the method was called on. The
    per-segment table says whether the win is broad or comes from one corner of
    the book; a candidate that wins overall and loses on the largest segment is
    a candidate to look at again.

The continuous ranked probability score is available in closed form for the
Gaussian, gamma and log-normal families (the catalogue of Jordan, Krüger and
Lerch 2019) and by quantile-score integration for other families implementing
`DistributionFunctionFamily`. A family without a CDF and quantile, including
`NegativeBinomialLS`, supports log score only. Every available score follows the
fitted weight contract. Under prior semantics the weight changes the row's
predictive law and comparisons aggregate the retained physical rows. Under
frequency semantics the predictive law is the unit-weight law and `w` is
literal replication mass. This replication rule covers CRPS, threshold-weighted
CRPS, Murphy curves, paired standard errors and the quantiles that choose the
default Murphy threshold grid, not only the log score.

When the decision is about the tail, score the tail:

```python
tail = model.scores(frame, y, which=("crps",), thresholds=(25.0,))
```

which is the threshold-weighted CRPS of Gneiting and Ranjan (2011). Choose the
threshold before looking at the data; it is a weight function, not a knob.

!!! note "How to read the Murphy diagram"
    Ask for it with `murphy_quantile=`. Every consistent scoring rule for a
    quantile is a mixture of elementary scores, one per threshold (Ehm,
    Gneiting, Jordan and Krüger 2016), so plotting the two candidates'
    elementary scores against the threshold shows *where* one wins. A curve
    below the other everywhere means the win holds for every user of that
    functional, whatever their loss; curves that cross mean the ranking depends
    on the threshold you care about, which is a finding and not a failure.

## Weights: rates are ratios of sums

Whenever the suite reports the mean of a rate over a group of rows — a decile, a
bin, a level, a segment, an identically-priced band, a portfolio segment — it
computes

$$\frac{\sum_i w_i y_i}{\sum_i w_i} \quad\text{and}\quad \frac{\sum_i w_i \hat\mu_i}{\sum_i w_i},$$

never the mean of per-row ratios. For a burn-cost target with exposure weights
this is total cost over total exposure; for claim-level severity with unit
weights it reduces to the plain mean; for a gamma fitted on per-claim averages
with count weights it is total cost over total claims. One helper does this for
every builder, so no table can take the wrong mean by accident. Quantities that
are not rates — quantile residuals, PIT values, coverage indicators — are
averaged with the replication weights only.

Which slot a `sample_weight` reaches depends on the model's declared
`weight_semantics`:

| Method | Under `"prior"` | Under `"frequency"` |
|---|---|---|
| `residuals`, `check`, `check_2d`, `actual_expected`, `calibration` | inside each row's own law, and as the aggregation weight of the tables | replication: the row counts `w` times |
| `scores`, `compare` | every requested score that the family supports reads the prior-weighted row law; comparisons give each retained physical row one observation | every requested score that the family supports reads the unit law and uses `w` as literal replication mass, including tail scores, Murphy diagrams, paired standard errors and default thresholds; unsupported score or quantile requests refuse |
| `posterior_bounds`, `posterior_predictive`, `portfolio` | inside each row's own law — a policy at a fifth of a year's exposure is simulated on its own law | refused: a replication count is not part of a row's law, so expand the rows or declare prior semantics |
| `parameter_spread` | both: it weighs the ratio of sums *and* it is part of the law | the aggregation weight only |

A family that cannot express its weighted law refuses rather than quietly
inverting the unit-weight one. A comparison between candidates with different
weight semantics likewise refuses whenever any retained weight is not exactly
one. Zero-weight rows are omitted from every calculation; input-aligned score
tables mark their positions as `NaN`, while comparison summaries remove them.
They also leave the fit and residual payload, and a covariate passed beside `X`
is cut with them.

## Payloads without figures

Every payload is JSON-clean, so a front end can draw what the model computed
without holding the model:

```python
import json

payload = model.plot_data("qq", X=frame, y=y, sample_weight=exposure)
json.dumps(payload)          # succeeds: lists, floats and nulls only
```

`kind` is one of `"qq"`, `"worm"`, `"pit"`, `"binned"`, `"actual_expected"`,
`"calibration"`, `"scores"`, `"comparison"`, `"term"`, `"risk_curves"`,
`"density_fan"`, `"spread"` and `"portfolio"`, and the keyword arguments are
those of the method that builds it. Each payload carries what was asked of it —
levels, seeds, bin edges, draw counts — so the figure is reproducible from the
payload alone.

Both rendering engines are available and only the one asked for is imported:

```python
figure = model.plot_diagnostics(frame, y, engine="plotly")
grid = model.plot(parameter="mean", engine="plotly")
```

## Cost

Residual and calibration costs are approximately linear in the number of rows.
Posterior bounds, portfolio simulations, and Q-Q envelopes additionally scale
with the requested draw count. Tweedie quantiles require numerical inversion of
the compound Poisson–gamma distribution and are consequently more expensive
than closed-form Gaussian or Gamma functionals. Reduce `n_draws`, or evaluate
bounds on a representative row sample, when a draw-based diagnostic is too
expensive. The Tweedie evaluator refuses a result when its omitted Poisson-tail
mass cannot be certified below its numerical tolerance.

## Method provenance

| Question | Method | Source |
|---|---|---|
| A residual for a family with a CDF | Randomised quantile residual, `r = Φ⁻¹(F(y \| θ̂))` | Dunn and Smyth (1996), *JCGS* 5(3) |
| Is the predictive distribution calibrated? | PIT histogram; calibration and sharpness | Gneiting, Balabdaoui and Raftery (2007), *JRSS-B* 69(2) |
| Detrended Q-Q; misfit by covariate region | Worm plot with pointwise bands; Q-statistics per interval | van Buuren and Fredriks (2001), *Statistics in Medicine* 20; Royston and Wright (2000), *Statistics in Medicine* 19 |
| Checking a GAM at scale | Binned residual checks with bootstrap bands; subsampled simulated envelopes | Fasiolo, Nedellec, Goude and Wood (2020), *JCGS* 29(1) |
| Bands on a smooth effect | Bayesian posterior `N(β̂, V)`; across-the-function coverage | Marra and Wood (2012), *Scandinavian Journal of Statistics* 39(1); Wood (2017), *GAMs*, 2nd ed. |
| Smoothing-parameter uncertainty | First-order corrected covariance | Wood, Pya and Säfken (2016), *JASA* 111(516) |
| Simultaneous bands | Max-deviation quantile over posterior draws | Ruppert, Wand and Carroll (2003), *Semiparametric Regression* §6.5 |
| Is a term flat? | Wald test on a rank-truncated pseudo-inverse, rank tied to the EDF | Wood (2013), *Biometrika* 100(1) |
| Which model is better? | Log score; CRPS with closed forms per family; CRPS as the quantile-score integral | Gneiting and Raftery (2007), *JASA* 102; Jordan, Krüger and Lerch (2019), *JSS* 90(12); Laio and Tamea (2007), *HESS* 11 |
| Which model is better *in the tail*? | Threshold-weighted CRPS | Gneiting and Ranjan (2011), *JBES* 29(3) |
| Where does one model win? | Murphy diagram of elementary scores | Ehm, Gneiting, Jordan and Krüger (2016), *JRSS-B* 78(3) |
| Is an exceedance forecast reliable? | CORP reliability diagram with consistency bands | Dimitriadis, Gneiting and Jordan (2021), *PNAS* 118(8) |
| The framework itself | GAMLSS | Rigby and Stasinopoulos (2005), *JRSS-C* 54(3) |
| A Tweedie quantile | Compound Poisson–gamma series summed where its terms live; gamma-initialized, bracketed third-order Householder iteration on `log q`, with bisection whenever a step leaves the bracket or stalls | Dunn and Smyth (2005), *Statistics and Computing* 15(4); Giner and Smyth (2016), *The R Journal* 8(1); Press, Teukolsky, Vetterling and Flannery (2007), *Numerical Recipes*, 3rd ed. §9.4 |

The spread among identically priced rows (§4) is the one construction on this
page without a citation: it is ours.
