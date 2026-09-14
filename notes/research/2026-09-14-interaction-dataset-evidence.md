# External evidence for interaction trials

The next priority is Bike Sharing's `hr × workingday`, followed by Ames
`log(Gr Liv Area) × Bldg Type` under a log-price target. Both have original
author examples that fit interactions and evaluate held-out predictions.
The Ames improvement is small. Neither result guarantees an improvement in
SuperGLM or under the repository's temporal splits.

This is a bounded source review on 2026-09-14. It adds no fit results and
changes no dataset, solver, or benchmark. Local context is the
[four-table pilot](2026-09-13-real-interaction-trials.md) and the
[dataset registry](../../benchmarks/interaction_datasets.json), inspected in
the adaptive-interactions worktree at the supplied head `052c1e67`.
Source pages below were opened during this review. Displayed external
metrics are transcribed from those pages, not reproduced locally.

The pilot does not establish that useful interactions are absent. Three
fits reached the REML iteration limit. The bike candidate generator admitted
only four weather splines and their six numeric pairs. It never considered
calendar-by-weather or calendar-by-calendar pairs. Ames used raw-price
Gaussian loss. These facts limit the experiment's answer to its actual
representations, candidate set, convergence, and split.

## Bike Sharing

UCI defines `cnt` as total hourly rentals, including `casual` and
`registered`; those component counts must stay excluded from predictors.
`workingday` distinguishes ordinary working days from weekends and holidays.
The source covers Capital Bikeshare in 2011 and 2012, with contemporaneous
weather. The local verified hourly table has 17,379 rows. UCI's catalog
headline currently says 17,389, so the verified file and registry determine
the trial row count. [UCI Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)

The scikit-learn developers' time-feature example fits squared-error models
to `count / 977` from OpenML `Bike_Sharing_Demand`, version 2. It uses five
`TimeSeriesSplit` folds, `gap=48`, `max_train_size=10000`, and
`test_size=1000`, all measured in rows. Reported mean errors are:

| Model | MAE | RMSE |
| --- | ---: | ---: |
| Additive cyclic-spline ridge | 0.097 | 0.132 |
| Ridge with the hour/working-day polynomial block | 0.078 | 0.104 |
| Unrestricted histogram gradient boosting | 0.044 | 0.068 |

The added block combines eight cyclic hour splines with `workingday` and
degree-two products. It also adds products between hour splines. Therefore
the improvement supports that joint pipeline, while a clean one-pair
ablation remains necessary. The example relates working-day rush-hour peaks
and broader weekend demand to this pair. Its units and folds differ from
the pilot's Poisson deviance and final chronological 20% block.
[Time-related feature engineering](https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html)

A separate official example takes every fifth hourly row, trains on
`year=0`, tests on `year=1`, and drops year as a predictor. This gives
1,729 training and 1,747 test rows, corresponding to 2011 and 2012 in UCI.
Its plot captions label the years incorrectly. With raw-count squared-error
loss and 50 boosting iterations, HGBT reports test R² 0.62; the cloned model
with all feature interactions prohibited reports 0.38. Its two-dimensional
partial dependence plot identifies `temp × humidity`, and it also displays
`season × weather`. These plots describe the fitted model. They do not
establish the held-out benefit of either pair separately, and correlated
weather features can make partial-dependence combinations unrealistic.
[Partial dependence and individual conditional expectation](https://scikit-learn.org/stable/auto_examples/inspection/plot_partial_dependence.html)

| Proposed candidate | Evidence and proposed test |
| --- | --- |
| `hr × workingday` | Highest priority from the fitted time-feature example. Refit all main effects with only this extra group first. Use all 24 hour levels or a cyclic hour basis with enough resolution for commute peaks. |
| `temp × hum` | The HGBT partial-dependence example supports a fitted interaction. It was already available to the old screen, so test it alone before attributing the old six-pair failure to this pair. |
| `season × weathersit` | A displayed joint HGBT effect, with no isolated loss result. Pool unsupported weather categories using training counts. |
| `hr × temp`, `hr × hum`, `hr × season` | Domain hypotheses for weather sensitivity and daily demand shape. The reviewed sources do not report separate held-out gains for these pairs. |
| `hr × weekday`, `workingday × temp` | Follow-up hypotheses. Check overlap with the first candidate's contribution before adding redundant groups. |

The primary new trial should retain count-Poisson loss and its log link.
A log-link additive model already gives multiplicative effects on expected
counts. A raw-response GBM interaction can consequently represent something
that is additive on the Poisson predictor scale. Keep the link and loss
matched when diagnosing the remaining interaction benefit. A separate
squared-error reproduction can test the published example without replacing
the primary task. Deriving lagged counts would define another prediction
problem and needs a separately frozen forecast horizon and feature contract.

## Ames Housing

Dean De Cock's original paper describes 2,930 residential sales in Ames
between 2006 and 2010, with property size, quality, age, and other attributes.
It presents the dataset and teaching tasks, but it does not provide a
validated ranking of named Ames interaction pairs. Its sample interaction
equation must not be read as such a result.
[De Cock, 2011](https://jse.amstat.org/v19n3/decock.pdf)

Kuhn and Silge's original modeling example uses `log10(Sale_Price)` because
the response is right-skewed and expensive properties can dominate raw-price
errors. Their version also includes latitude and longitude as well as
neighborhood. Those coordinate columns are absent from our current raw Ames
predictor contract. [Ames data and target transformation](https://www.tmwr.org/ames)

Their recipe chapter shows different log-area/log-price slopes by building
type and fits `log10(Gr_Liv_Area) × Bldg_Type` alongside the main effects.
This directly motivates a mixed categorical/numeric candidate. It supports
a particular transformation and pair, not every area-by-quality hypothesis.
[Feature engineering with recipes](https://www.tmwr.org/recipes)

The comparison chapter adds that pair to a baseline with neighborhood,
log living area, year built, building type, and linear coordinates. Its
reported log10-price RMSE is 0.0803 for the baseline, 0.0799 after adding the
interaction, and 0.0785 after also adding coordinate main-effect splines.
The interaction result is an actual joint-refit comparison, with a small
average improvement. The coordinate spline improvement does not establish
a latitude-by-longitude interaction. [Comparing models](https://www.tmwr.org/compare)

The book first creates an 80% random training split stratified by price,
seed 502, with 2,342 training rows. It then creates ten random CV folds
inside that training set, seed 1001. These are the resamples used above.
The feature set, random folds, and transformed target differ from our
2006-2008 training, 2009 validation, and 2010 test protocol. The reported
RMSEs cannot be compared with our raw-price MSE or Kaggle leaderboard scores.
[Data splitting](https://www.tmwr.org/splitting),
[Resampling](https://www.tmwr.org/resampling)

| Proposed candidate or transformation | Evidence and proposed test |
| --- | --- |
| `log(Gr Liv Area) × Bldg Type` | First pair, supported by the book's fitted comparison. Keep the transformed area main effect in both arms. |
| `log(Gr Liv Area) × Neighborhood` | Location may modify the price contribution of size. This is a domain hypothesis here, not a claimed replicated external gain. Start with pooled neighborhood slopes before a large tensor. |
| `log(Gr Liv Area) × Overall Qual` | Quantity and quality may combine beyond their main effects. Treat as a hypothesis, and compare against a sufficiently flexible quality main effect. |
| `Year Built × Overall Qual`, `Year Built × Neighborhood` | Hypotheses about age, condition, and location. Year built is available in the frozen contract; sale age needs a deliberate addition of sale-date information. |
| `log(SalePrice)` target and `log(Gr Liv Area)` margin | Source-supported initial transformations. Freeze them before interaction discovery; otherwise a transformation gain can be credited to the interaction. |

Use natural-log price Gaussian loss as a declared new task, with log-price
RMSE as the primary metric. A different logarithm base changes units, so
name the base in every receipt. Keep raw-dollar MAE/RMSE secondary and
declare the retransformation rule. Exponentiating a fitted log mean does
not generally yield the conditional mean price; estimate any correction
from training data. Compare the additive and interaction models under the
same response transformation. Keep the prior raw-price experiment as a
separate reference. Do not remove expensive or large houses merely because
doing so improves the new score.

The user's recalled California housing geography result motivates a future
spatial pair trial, but it is not independently verified by this memo.
It also cannot supply coordinates missing from this Ames artifact.

## Credit default and diagnostic breast cancer

These are lower-priority interaction-discovery datasets for this follow-up.
This bounded search did not locate a primary, isolated pair-ablation result
on the exact four-table credit or diagnostic contracts. That is a limit of
the search, not evidence that the pairs have no value.

UCI credit default has 30,000 clients and 23 predictors. The registry uses
`X1` for credit limit, `X6`/`X7` for September/August repayment status,
`X12` for the September bill, and `X18` for the September payment. UCI defines
April-September 2005 histories and binary default. Its original research
compares six prediction methods, without identifying a held-out benefit for
a particular feature pair in the repository description. A neural-network
advantage would not isolate interactions from univariate flexibility or
tuning. [UCI credit default](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)

Proposed hypotheses are `X6 × X7` for recent payment-history patterns,
`X6 × X1` for status conditional on credit limit, and `X12 × X18` for bills
and payment amounts. A bill-to-limit or payment-to-bill feature is itself a
multivariable transformation. Label it as such rather than adding it only
to an allegedly additive control. Check zeros and signs before considering
ratios or log transforms; use a declared signed transform for signed
amounts. Keep categorical repayment semantics instead of treating status
codes as arbitrary continuous measurements. First obtain a converged,
regularized additive fit, then score probabilities with log loss and
secondary Brier score/AUC on the existing client-group splits. This single
historical cohort does not supply independent future calendar cohorts.

UCI diagnostic breast cancer has 569 cases and 30 nuclear-image measurements.
The target is malignant versus benign diagnosis, not a future cancer event.
The source defines radius, texture, perimeter, area, and shape measurements;
compactness already combines perimeter and area. Its original construction
used a small-feature separating-plane search. This does not establish a
need for a large spline-interaction model.
[UCI diagnostic breast cancer](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

Scikit-learn's original example on this dataset shows why correlated
predictors can dilute individual permutation importance. It demonstrates
feature clustering using Spearman correlations and retaining one feature
per cluster. This is evidence about redundant measurements, not useful
interaction pairs. [Correlated-feature importance example](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html)

Candidate hypotheses are `radius1 × texture1` and
`radius3 × concave_points3`. They combine different measurement concepts;
they have no isolated predictive evidence in this review. If used, retain
an all-feature regularized additive control and perform any clustering on
training predictors only. Limit candidate complexity and evaluate log loss,
Brier score, and calibration as well as discrimination. The old fit's
near-zero training deviance and worse held-out log loss make regularization
a more immediate question than adding eight more tensors. These remain
historical statistical benchmarks, without a clinical deployment claim.

## A trial that separates interactions from baseline choices

The following is a proposed protocol, not an executed experiment.

1. Freeze data bytes, grouping, splits, target/link, predictor exclusions,
   and the metric before new screening. The pilot's final test scores have
   already been viewed. Further tuning informed by them makes these four
   datasets development evidence; a new split alone does not undo that
   history. Use an untouched corpus for confirmation.
2. Establish a converged additive baseline. Give its main effects enough
   resolution and regularization for calendar peaks or skewed housing
   quantities. Use training diagnostics and validation to choose among a
   small declared set of additive representations. Freeze that choice for
   the pair additions.
3. Fit the externally motivated first pair, then any validation-selected
   small set, with all main effects refitted jointly. Preserve the same
   margins. Record the chosen pair count, rejected candidates, convergence,
   and the cost of every attempt. A selected pair's name does not establish
   its incremental value after other pairs enter.
4. Run the same GBM implementation in additive, pairwise, and unrestricted
   modes. HGBT provides `interaction_cst="no_interactions"`, `"pairwise"`,
   and unconstrained `None`. For a named-pair control, explicitly include
   singleton groups for all other features; omitted features otherwise form
   another allowed interaction group. Verify the installed implementation
   before execution. [HGBT interaction constraints](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)
5. Match the target, link, preprocessing, splits, and tuning budget across
   GBM modes. Provide both a shared-hyperparameter comparison and a bounded
   validation-tuned comparison per mode. Prohibiting interactions can need
   more boosting iterations. Avoid an undeclared random early-stopping
   partition in a temporal task. Pairwise GBM gains are a stronger diagnostic
   than comparing one untuned additive model with an unrestricted GBM, but
   still measure these algorithms under finite budgets.
6. Evaluate interaction plots on the linear predictor scale used by the
   model. Use training support to avoid implausible correlated feature
   combinations. Group redundant proposals and perform joint held-out
   ablations; marginal importance and partial dependence are candidate
   generators, not proof that a pair helps.
7. Charge complete-fit wall time, peak RSS, retained storage, candidate
   generation, tuning, failed fits, and final joint refits. Record numerical
   outputs and actual backend dispatch separately. A fast screen that needs
   repeated expensive refits has not demonstrated cheap discovery.

A GBM advantage can result from better univariate approximation, coding of
categories, regularization, tuning, or interactions. Conversely, failure of
a particular tensor representation can coexist with useful interactions.
The comparisons above distinguish those possibilities empirically; they do
not prove a true interaction count or statistical certification.

## Untouched related confirmation data

No model test scores for either proposed confirmation dataset were inspected
in this review, and no fit was run. Their suitability below follows from
metadata and the existing registry, not performance selection.

| Corpus | Proposed confirmation task and limitations |
| --- | --- |
| Seoul Bike Sharing Demand | UCI provides 8,760 hourly observations with weather, holidays, season, and operating status. Transfer the hour/calendar/weather candidate policy after freezing it. The existing `uci_seoul_bike` registry splits whole days 60/20/20, then restricts an operating-demand estimand to functioning hours. Keep that declared policy, or define a separate all-hours estimand with structural closure zeros. It is a new city with different operating conditions, not a second sample from Capital Bikeshare. [UCI Seoul Bike Sharing](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand) |
| King County house sales | The University of Chicago GeoDa distribution describes 21,613 sales in 2014-2015, with area, location, quality, and sale date. Transfer a frozen area/type-or-quality policy and consider latitude-by-longitude as a separately declared spatial candidate. Build and verify the adapter first, group repeated property IDs, and choose chronological or spatial blocking to match the question. The distributor traces the sales file to Kaggle and the ZIP polygons to county GIS; record that provenance rather than calling the sales file a directly audited county extract. [GeoDa dataset description](https://geodacenter.github.io/data-and-lab/KingCounty-HouseSales2015/) |

Seoul is already registered and is the nearer next confirmation step. Check
the repository's experiment ledger before calling it untouched across the
whole project. King County requires a new data contract. A successful result
on either would be a new empirical observation under its frozen protocol.
