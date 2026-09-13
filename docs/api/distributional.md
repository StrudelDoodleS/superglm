# Distributional models

`SuperLSS` fits several parameters of a response distribution together, one
predictor per parameter. Pass a family first, then one predictor declaration
per parameter using the family's helper methods; build the terms inside each
declaration with `s`, `cat`, `re`, `ti`, `term` and `interaction`. Start
with the [tutorial](../tutorials/distributional-model.md). The members of
`SuperLSS` are grouped below by what you do with them; each has its own page.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/class-no-members-distributional.rst

   superglm.SuperLSS
```

## Declare and fit

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   ~superglm.SuperLSS.fit
   ~superglm.SuperLSS.fit_reml
   ~superglm.SuperLSS.diagnose
   ~superglm.SuperLSS.predictors
   ~superglm.SuperLSS.family
```

## Predict

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   ~superglm.SuperLSS.predict
   ~superglm.SuperLSS.predict_parameters
   ~superglm.SuperLSS.predict_link
   ~superglm.SuperLSS.predict_cdf
   ~superglm.SuperLSS.predict_quantile
   ~superglm.SuperLSS.posterior_predictive
   ~superglm.SuperLSS.posterior_draws
   ~superglm.SuperLSS.posterior_bounds
```

## Read the fit

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   ~superglm.SuperLSS.summary
   ~superglm.SuperLSS.term_inference
   ~superglm.SuperLSS.term_test
   ~superglm.SuperLSS.parameter_names_
   ~superglm.SuperLSS.family_
   ~superglm.SuperLSS.predictors_
   ~superglm.SuperLSS.coef_
   ~superglm.SuperLSS.coef_by_predictor_
   ~superglm.SuperLSS.covariance_
   ~superglm.SuperLSS.result_
   ~superglm.SuperLSS.smoothing_parameters_
```

## Check the fit

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   ~superglm.SuperLSS.residuals
   ~superglm.SuperLSS.residual_set
   ~superglm.SuperLSS.check
   ~superglm.SuperLSS.check_2d
   ~superglm.SuperLSS.actual_expected
   ~superglm.SuperLSS.calibration
   ~superglm.SuperLSS.scores
   ~superglm.SuperLSS.compare
```

## Price and portfolio views

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   ~superglm.SuperLSS.risk_curves
   ~superglm.SuperLSS.density_fan
   ~superglm.SuperLSS.parameter_spread
   ~superglm.SuperLSS.portfolio
```

## Plot

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   ~superglm.SuperLSS.plot
   ~superglm.SuperLSS.plot_data
   ~superglm.SuperLSS.plot_diagnostics
```

## Smoothing certification and telemetry

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   ~superglm.SuperLSS.smoothing_certified_
   ~superglm.SuperLSS.smoothing_convergence_reason_
   ~superglm.SuperLSS.smoothing_unresolved_upper_bound_
   ~superglm.SuperLSS.exact_face_components_
   ~superglm.SuperLSS.coefficient_curvature
   ~superglm.SuperLSS.training_telemetry
```

## Save and load

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   ~superglm.SuperLSS.to_bytes
   ~superglm.SuperLSS.from_bytes
```

## Configuration

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   ~superglm.SuperLSS.discrete
   ~superglm.SuperLSS.n_bins
   ~superglm.SuperLSS.separation
   ~superglm.SuperLSS.weight_semantics
```

## Declarations and families

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   superglm.Predictor
   superglm.BoundPredictor
   superglm.bind_predictor
   superglm.BoundTerm
   superglm.BoundInteraction
   superglm.term
   superglm.s
   superglm.cat
   superglm.re
   superglm.ti
   superglm.interaction
   superglm.GaussianLS
   superglm.GammaLS
   superglm.LogNormalLS
   superglm.NegativeBinomialLS
   superglm.GeneralizedGammaLSS
   superglm.GeneralizedParetoLSS
   superglm.TweedieLSS
   superglm.TwoPieceLogNormalLSS
   superglm.TwoPieceNormalLSS
```
