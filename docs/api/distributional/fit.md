# Fit

Construct the model with a family and one declaration per family parameter,
made with that family's helper methods. {py:meth}`~superglm.SuperLSS.fit_reml`
fits the coefficients and estimates the smoothing parameters jointly, by
generalised Fellner-Schall updates with optional Newton refinement, and is the
normal path; {py:meth}`~superglm.SuperLSS.fit` holds the smoothing parameters
fixed at the `lambdas` you pass. {py:meth}`~superglm.SuperLSS.diagnose`
returns a {py:class}`~superglm.FitDiagnosticReport` that explains how the fit
ran and how smoothing stopped: phase timings, iteration and refit counts, and
the terminal state of every smoothing component.
{py:attr}`~superglm.SuperLSS.predictors` and
{py:attr}`~superglm.SuperLSS.family` echo the declaration as independent
copies. The [how-to on fitting a distributional
model](../../how-to/fit-a-distributional-model.md) covers the declaration
syntax.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:
   :template: autosummary/class-no-members-distributional.rst

   superglm.SuperLSS
```

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   ~superglm.SuperLSS.fit
   ~superglm.SuperLSS.fit_reml
   ~superglm.SuperLSS.diagnose
   ~superglm.SuperLSS.predictors
   ~superglm.SuperLSS.family
   superglm.FitDiagnosticReport
```
