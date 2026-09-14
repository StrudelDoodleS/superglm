# Internals

Objects the library builds on your behalf. You do not write them, but they
appear in signatures, inside the records a fit returns, and in tracebacks, so
each has a page. Nothing here is needed to fit, read or deploy a model.

## Interaction types

Pass a pair of column names as `interactions=[("age", "region")]` and the
constructor picks one of these from the two parent feature specifications;
pass one explicitly only when you need its options. The
[how-to on interactions](../how-to/specify-interactions.md) says which pair
produces which type.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   superglm.TensorInteraction
   superglm.SplineCategorical
   superglm.NumericCategorical
   superglm.PolynomialCategorical
   superglm.CategoricalInteraction
   superglm.NumericInteraction
   superglm.PolynomialInteraction
```

## SuperLSS predictor plumbing

The [declaration helpers](distributional/declarations.md) return a
{py:class}`~superglm.BoundTerm` or {py:class}`~superglm.BoundInteraction`, a
feature specification attached to a named column; a family's helper method
wraps them in a {py:class}`~superglm.BoundPredictor` tied to that family, and
{py:class}`~superglm.Predictor` is the immutable configuration the estimator
reads underneath.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   superglm.BoundTerm
   superglm.BoundInteraction
   superglm.BoundPredictor
   superglm.Predictor
```

## Constraint machinery

{py:class}`~superglm.Constraint` is what you write; it compiles to a
{py:class}`~superglm.ConstraintSpec`, one shape constraint with when it
applies and what it requires, and to the linear inequalities in a
{py:class}`~superglm.LinearConstraintSet`. The post-fit repair behind
{py:meth}`~superglm.SuperGLM.apply_shape_postfit` is a
{py:class}`~superglm.MonotoneRepairer`, and it records what it did in a
{py:class}`~superglm.MonotoneRepairResult`.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   superglm.ConstraintSpec
   superglm.LinearConstraintSet
   superglm.MonotoneRepairer
   superglm.MonotoneRepairResult
```

## Parts of a term inference

{py:class}`~superglm.TermInference`, which
{py:meth}`~superglm.SuperGLM.term_inference` returns, is built from these: the
continuous fitted curve for plotting, the knot and basis metadata of a spline
term, and the lighter per-interaction result.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   superglm.SmoothCurve
   superglm.SplineMetadata
   superglm.InteractionInference
```

## Tweedie profile records

{py:class}`~superglm.TweedieProfileResult` carries the evidence for its
confidence interval in these: the details and diagnostics of the interval,
each endpoint and how it was obtained, each likelihood-ratio evaluation, and
the density method used at each evaluated point.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   superglm.TweedieProfileCIDetails
   superglm.TweedieProfileCIEndpoint
   superglm.TweedieProfileCIEvaluation
   superglm.TweedieProfileCIDensityProvenance
```
