# Internals

Objects the library builds on your behalf. You rarely write them, but they
appear in signatures, inside the records a fit returns, and in tracebacks, so
each has a page. None is needed to fit, read or deploy a model the documented
way; a few carry options you reach for only when the default construction is
not enough.

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
   superglm.PolynomialCategorical
   superglm.NumericCategorical
   superglm.CategoricalInteraction
   superglm.NumericInteraction
   superglm.PolynomialInteraction
```

## SuperLSS predictor plumbing

The [declaration helpers](distributional/declarations.md) return a
{py:class}`~superglm.BoundTerm` or {py:class}`~superglm.BoundInteraction`, a
feature specification attached to a named column; a family's helper method
wraps them in a {py:class}`~superglm.BoundPredictor`, which is what `SuperLSS`
takes and so stays on the Declarations page, and
{py:class}`~superglm.Predictor` is the immutable configuration the estimator
reads underneath.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   superglm.BoundTerm
   superglm.BoundInteraction
   superglm.Predictor
```

## Constraint machinery

{py:obj}`~superglm.Constraint` is what you write, and `Constraint.fit.increasing`
and its siblings are {py:class}`~superglm.ConstraintSpec` values: one shape
constraint, with when it applies and what it requires. Where a fit-time
constraint is enforced by quadratic programming it is expressed as the linear
inequalities of a {py:class}`~superglm.LinearConstraintSet`. Behind
{py:meth}`~superglm.SuperGLM.apply_shape_postfit`, monotone constraints are
repaired by a {py:class}`~superglm.MonotoneRepairer` and curvature constraints
by a repairer of their own that is not exported; both record what they did in
a {py:class}`~superglm.MonotoneRepairResult`.

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
{py:meth}`~superglm.SuperGLM.term_inference` returns for a main effect, carries
these two: the continuous fitted curve for plotting, and the knot and basis
metadata of a spline term. For an interaction the method returns an
{py:class}`~superglm.InteractionInference` instead, listed with the other
[inference results](inference.md).

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   superglm.SmoothCurve
   superglm.SplineMetadata
```

## Tweedie profile records

{py:class}`~superglm.TweedieProfileCIDetails`, which
{py:meth}`~superglm.TweedieProfileResult.ci_details` returns and which is
listed with the [families](families-and-links.md), holds the evidence for a
Tweedie profile confidence interval in these: each endpoint and how it was
obtained, each finite likelihood-ratio evaluation, and the density method
retained for the evaluated points inside the connected likelihood-ratio
region, which is not every evaluation.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   superglm.TweedieProfileCIEndpoint
   superglm.TweedieProfileCIEvaluation
   superglm.TweedieProfileCIDensityProvenance
```
