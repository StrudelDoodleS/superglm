# Every family in the editor: design

Date: 2026-11-12. Base: master `0fbf84073`. Owner request (Max): "I want all current
and future families for superglm and superlss to work in the editor", plus, separately,
"the superglm family [should] have the preview of the feature in the sidebar like
superlss has" (built already on `claude/superglm-sidebar-curves`, its own small PR).

## 1. Intent

Any model superglm can fit opens in the editor and works there: open, edit, refit, final
fit, export, history, the Cross-validation and Validation tabs, and, for SuperLSS, the
distribution panel and the Explain tab. That holds for every family and link exported
today and for every family added later. Nothing in the editor's code decides what to do
by a family's class or name. It reads what the family declares (its parameters, their
links, its distribution functions), so a new family, hand-written or from the family
factory (`tools/generate_severity_families.py`, see the family-factory plan), arrives in
the editor without editor changes, and CI fails if one does not.

## 2. What exists today (measured on master `0fbf84073`)

A probe fitted every family on 800 synthetic rows (`s(age)` + `cat(region)` on the first
parameter, `s(age)` on the second, intercept on the rest) and drove the editor through
open, payload, shift, `to_model`, the widget's state, rating table, distribution panel,
risk row and joblib export.

- **SuperGLM.** Poisson/log, Gaussian/identity, Gaussian/log, Gamma/log, Gamma/inverse,
  Tweedie(1.5)/log, NB2/log, Binomial/logit and Binomial/log all open and run every step.
  But every curve is drawn as `exp(effect)` and labelled "relativity", whatever the link
  (`payloads.py` sets `y_label: "relativity"`, `chart/display.js` exponentiates). That is
  a multiplier only on a log link. On Gaussian/identity it multiplies nothing; on
  Binomial/logit it is an odds ratio called a relativity; on Gamma/inverse it is
  meaningless. The library's own rating-table export already refuses non-log links for
  exactly this reason (`export/rating_tables.py`, `_require_log_link_export`: "the product
  is exp(linear_predictor), measured 9.19 maximum relative error against model.predict").
- **SuperLSS.** Opens 4 of 12 families: LogNormalLS (mean form only), GammaLS,
  NegativeBinomialLS, TweedieLSS (power an intercept only). `editor/lss/open.py` refuses
  GaussianLS, GeneralizedGammaLSS, GeneralizedParetoLSS, LogLogisticLSS, LomaxLSS,
  TwoPieceLogNormalLSS and TwoPieceNormalLSS through the allow-list `SPREAD_FAMILIES`
  (`editor/lss/display.py`); it also refuses LogNormalLS in another parametrisation, a
  Tweedie power with terms, and any parameter fitted on a link other than its default.
  DirichletLSS (a vector response) is refused by `require_scalar_response`.
- **Where family knowledge lives.** Python: `lss/display.py` (labels keyed by parameter
  name: mean, scale→"CV", theta, dispersion, power; anything else raises), `lss/panel.py`
  (`_FAMILIES`, `link_spec` accepts only log, floored log and bounded power links),
  `lss/risk.py` (`_family_name`, which raises `StopIteration` for an unlisted family),
  `lss/open.py`. Browser: `distribution/families.js` `lawOf` (log-normal, gamma, Tweedie),
  `views/distribution_panel.js` and `views/risk_panel.js` (dispatch on family strings,
  log10 amount axis), `views/risk_drivers.js` (inverts links by kind, `name === "theta"`),
  `views/risk_counts.js` (the NB's count bars).
- Saved sessions and history store edits on the link scale (`edited_log_effect`), so a
  change of display cannot change a saved model.

## 3. Prior art (searched, 2026-11-12)

- GAM Changer (Wang et al. 2021, arXiv 2112.03245), the editor's stated inspiration, draws
  and edits each shape function on the link scale (log-odds for a classifier), re-centred
  so that 0 is "no effect", with a thick dashed 0 baseline.
- mgcv `plot.gam` draws partial effects on the linear-predictor scale; its `trans`
  argument (a monotone function applied after `shift`) is the documented way to get
  another scale, e.g. `exp` for a log link.
- Odds ratios, `exp` of a logit coefficient, are the standard reading of a logistic
  model's effects; relativities, `exp` of a log-link effect, are the actuarial one.

So the established practice is: an additive effect on the link scale by default, and a
multiplicative reading exactly where the link makes one true. The editor's relativity
display is mgcv's `trans = exp`, correct for log links and only there.

## 4. Goals, non-goals, anti-goals

Goals
- G1. Every scalar-response SuperGLM family × link and every scalar-response SuperLSS
  family opens and runs every editor step, in every parametrisation and on any link a fit
  accepts.
- G2. Every curve shows a quantity that is true for its link (section 5).
- G3. The distribution panel and the Explain tab draw every SuperLSS family that has a
  distribution function and quantile (section 6).
- G4. A family added later works without editor changes; CI proves it (section 8).
- G5. DirichletLSS opens with the tools the library offers vector responses (section 7).

Non-goals
- Rating tables for non-log links. The library refuses them on stated grounds (section 2);
  the editor shows that refusal's sentence for such a term or parameter, as it does today
  for a SuperGLM.
- A count distribution panel. Max dropped it on 2026-11-03 ("not really sure what the
  utility is"); a count family keeps the panel's "not drawn" sentence and the Explain
  tab's count bars.
- Truncated or censored fits; non-string column names. Those refusals are about the fit,
  not the family, and stay.

Anti-goals (a failure even if it "works")
- A family table anywhere a new family would have to be added by hand, other than the
  optional browser fast paths of section 6.
- A drag that stutters (the editor-smoothness hard limit): no new per-frame server work.
- A number on screen that is not what its label says, including an interpolated readout
  with no measured bound.
- Changing what a log-link model shows today. Every log-link curve, label and number on
  a SuperGLM or one of today's four SuperLSS families stays as it is.

## 5. The display follows the link

Each term's curve is its additive contribution `f` to one linear predictor. What the
editor draws is decided by the fitted link of that predictor (`model._link` for a
SuperGLM, the parameter's `state.link` for a SuperLSS), never by the family:

| Fitted link | Curve shows | No effect at | Edits act on |
|---|---|---|---|
| log | relativity, `exp(f)` | 1 | the relativity (today's behaviour) |
| floored log, `floor + exp(η)` | relativity of the parameter above its floor | 1 | the same (today's log-normal σ) |
| logit | relativity of the odds, `exp(f)` | 1 | the odds relativity |
| identity | the effect `f` itself, in the parameter's units | 0 | the added amount |
| any other (inverse, inverse squared, probit, cloglog, cauchit, sqrt, power, NB link, bounded logit, bounded power, any future link) | the effect `f` on the link scale | 0 | the link-scale effect |

The one exception is kept as decided on 2026-11-09/10: a precision parameter, declared
with role `size` (the negative binomial's θ), shows the relativity of its reciprocal
(1/θ, overdispersion), `exp(-f)`.

Display kinds. Python `DisplayTransform` (`lss/display.py`, moved to a module both model
kinds use) and browser `chart/display.js` gain one kind, `additive`, beside `relativity`
and `inverse_theta`: `to_display(f) = f`, `from_display(v) = v`, shift adds, no floor.
Every tool already works through these maps (Move, handles, Average, Straighten, Smooth,
Make increasing/decreasing, snaps, undo), so the tools work unchanged on an additive
curve; the builder checks each one against it. Things that assume a baseline of 1 read it
from the display: the chart's no-effect line, the sidebar mini curve's bar baseline
(`chart/sparkline.js`), the Contrib/Build overlays (shown only on a relativity today, which
stays), the impact summary's weighted mean, and the Cross-validation tab's fold anchor
(today the exposure-weighted mean log; for an additive curve the exposure-weighted mean).

Increase/Decrease. Today ±log(1.05) on the effect, ×1.05 on a relativity, which stays for
every multiplicative kind. On an additive curve the step is 5% of the weighted mean
absolute value of the term's linear predictor over the training rows, so on an identity
link it moves an average row by about what ×1.05 would; where that mean is 0 it is 5% of
the curve's own range. *(Builder's call; revisit if it reads oddly.)*

Labels. A parameter's name a user reads comes from the family, not from an editor table.
`ParameterSpec` (`distributional/family.py`) gains three optional fields, all defaulting
to `None`: `label` (the name a user reads: "CV", "Overdispersion", "Standard deviation",
"Skew"), `description` (one sentence: what the parameter is), and `quantity` (what the
link acts on, when it is not the label: the log-normal's scale is labelled "CV" and its
link acts on σ). With no `label` the editor shows the parameter's name, capitalised. The
built-in families set them so that today's four show exactly today's text (Mean, CV,
Overdispersion with chip 1/θ, Dispersion, Power; "relativity of σ" for the log-normal).
The y-axis label is built from the kind: "relativity of the {quantity}", "relativity of
the odds of the {quantity}", "effect on the {quantity}", "effect on the {link} scale of
the {quantity}". A SuperGLM's quantity is "mean". Defaults on a frozen dataclass keep old
pickles loading (a missing attribute falls back to the class default); a test loads a
pickle made on master to prove it.

Role and colour. One parameter per family is `average` (blue): the one declared with role
`mean`, else the one with role `location`, else the first declared parameter (families
declare their headline parameter first: the scale of GeneralizedParetoLSS, LomaxLSS and
LogLogisticLSS). Every other parameter is `spread` (orange). The editor derives this from
`ParameterSpec.role` and declaration order, never from the parameter's name. *(Designer's
call, from the builder check.)*

What opening refuses afterwards: an unfitted model, truncated or censored rows, a
non-string column, missing training rows, a centering other than native, missing offsets
or weights, rows that are not the fit's. Not: a family, a parametrisation, a link, terms
on a power or shape.

## 6. Distribution panel and Explain tab for every family

Today the browser rebuilds each law from link-scale predictors with hand-written maths,
which is what lets the panel move during a drag. That stays, as an optional fast path,
for the families that have it (log-normal mean form, gamma, Tweedie; NB counts in
Explain). Every other family gets a **tabulated law** from the server:

- For each panel or risk row, the server sends the family's own quantiles at a fixed grid
  of probabilities (dense in both tails on a logit scale), the mass at zero where the
  family has an atom (`AtomFamily`), the mean (`default_prediction`), and the density on
  the panel's amount grid, evaluated from the family's own log likelihood (the kernels the
  fit uses, one definition per family).
- The browser builds a `Law` (the interface in `distribution/families.js`: mean, zero,
  median, mode, cdf, sf, quantile, log10Density) from that table by monotone
  piecewise-cubic interpolation (Fritsch and Carlson 1980, SIAM J. Numer. Anal. 17(2))
  of the quantile in logit probability, and of the density on the grid.
- Interpolated readouts (the chance above the line, a quantile between grid points)
  carry an error the builder measures against `family.cdf`/`family.quantile` on every
  family across the probability range, and the grid is sized so that error stays below
  the readout's printed precision. The bound and its test live with the code.
- A tabulated family's panel redraws when an edit lands (one server reply), not during
  the drag. No per-frame request.
- The amount axis is log10 when the family's response support is bounded below by 0, and
  linear otherwise (GaussianLS, TwoPieceNormalLSS); the panel's drawing reads an axis
  object (forward and inverse transform) instead of assuming log10.
- A count response (integer support) keeps the panel's "not drawn" sentence (non-goal);
  the Explain tab's count bars read a tabulated probability mass for any count family,
  with the NB's browser maths kept as its fast path.
- A family without `cdf` and `quantile` capabilities gets one sentence in the panel and
  in Explain saying the family has no distribution function to draw; everything else in
  the editor works.
- The Explain tab's drivers (`risk_drivers.js`) read each parameter's natural value from
  the payload (computed on the server, fitted and edited) instead of inverting the link
  in the browser, and show a multiplicative move as "×" and an additive one as "+".

The fast-path table (`lss/panel.py`) is the only place a family is named, and it only
selects browser maths; a family missing from it is drawn by the generic path.

## 7. DirichletLSS (vector response)

The library offers a vector response the joint log score and refuses CRPS, A/E, Gini,
`check` and calibration (`require_scalar_response` call sites). The editor matches it:
each component's concentration parameter is a parameter group with its curves (log link,
relativities), edits, refit, final fit, export and history; the Validation and
Cross-validation tabs score by joint log score only and say why the others are absent;
the distribution panel and Explain tab show one sentence saying they draw scalar
responses. This is the last stage; it may need evaluation-data and metrics plumbing for
a matrix `y`, which the builder maps first.

## 8. The guarantee for future families

- **The walker test.** One test module enumerates every exported SuperLSS family
  (`superglm.distributional.families.__all__`, the classes that are families) and every
  SuperGLM distribution class (`superglm.distributions`) at its default link, plus each
  kind of link of section 5 once, on a family where it is conventional: Gaussian/log,
  Gamma/inverse, Binomial/log, Binomial/probit, Binomial/cloglog, Poisson/sqrt. The
  library declares no per-family list of valid links (`resolve_link` accepts any shortcut
  for any family), and the display depends on the link kind alone, so one case per kind
  covers every display path; a new SuperGLM distribution is picked up at its default link.
  Each case fits a small model and drives
  every editor step: open, state payload, an edit on each term, undo, refit, final fit,
  joblib export, CV fold items, validation scores, the panel and Explain payloads, and,
  where the link is log, the rating table. Data come from inverse-CDF sampling with the
  family's own `quantile` at example natural parameters held in one table in the test;
  a family without `quantile` brings its own sampler in that table entry instead. An
  exported family missing from the table fails with a sentence naming the table. The
  walker runs in the quick suite, so its cases stay small (a few hundred rows, final fit
  once per family, CV folds on one family per response kind); the builder reports its
  wall time on the xdist run. *(Designer's calls, from the builder check.)*
- **A source test** that no editor module (Python or JS) names a family class or family
  string outside the fast-path table.
- Both live in the stage that introduces them and stay.

## 9. Stages (one PR each)

0. SuperGLM sidebar mini curves. Built on `claude/superglm-sidebar-curves`; front-end
   only; a node test and a browser test; mutation-checked.
1. Display follows the link; every scalar family opens. Sections 5 and 8 (walker covers
   open, edits, refit, export, CV, validation; panel and Explain show "drawn in the next
   release" for families without a fast path until stage 2). `ParameterSpec` labels.
   Docs: the editor tutorial's family and relativity text.
2. Panel and Explain for every family. Section 6; the walker extends to panel and Explain.
3. DirichletLSS. Section 7.

Each stage runs the focused tests, the editor browser suites (two separate pytest
commands, `--run-browser`), `npm run check:frontend`, and the quick suite before its
push; the full suite once per PR. Performance: stage 1 adds no fit work; stage 2's panel
request on the generic path is timed on a 12-row panel for every family (target: the
same order as today's panel request), and a drag on a fast-path family is shown to cost
what it does on master.

## 10. What changes for users

- Every family opens. Today's four SuperLSS families and every log-link SuperGLM look
  exactly as they do now.
- SuperGLM Gaussian/identity curves show added amounts around 0 instead of `exp` numbers
  around 1; Binomial/logit shows the same numbers, now labelled "relativity of the odds";
  other links show the link-scale effect around 0. Saved sessions reopen to the same
  model: only the drawing changes.
- New SuperLSS families' panels follow an edit when it lands rather than during the drag.

## 11. Calls made by the designer (check these)

1. Additive display for identity and every non-log, non-logit link (section 5), instead
   of `exp` everywhere. This is the one place the editor departs from "every curve is a
   relativity" (2026-11-10), and only where no relativity exists.
2. Logit shows odds relativities rather than the link-scale log odds.
3. Increase/Decrease step on an additive curve (section 5).
4. `ParameterSpec` gains `label`, `description`, `quantity` (library change, optional,
   default `None`).
5. Tabulated laws from the server for families without browser maths; no live panel
   during a drag for those.
6. Dirichlet included, as the last stage, with log score only.
7. Rating tables stay log-link only (the library's rule).

## 12. Left to the builder

Module layout; the probability and amount grids; the fast-path registry's shape; how the
walker's example parameters are chosen per family; test fixture moves (the shared
`editor_model` fixture is Gaussian/identity and many tests assert relativity mechanics on
it: moving it to a log link, or asserting through the display maps, is the builder's
choice, with no loss of coverage).
