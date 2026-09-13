# Features

Feature specifications turn raw columns into model terms. `Spline` is the
public factory for smooth terms; the concrete spline classes are what it
returns. Categorical, ordered-categorical, numeric, polynomial and piecewise
terms cover the rest of a rating structure; `FactorSmooth` and
`RandomEffect` add credibility-style shrinkage; the interaction classes
combine terms; `Constraint` requests monotone or curvature constraints.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   superglm.Spline
   superglm.PSpline
   superglm.BSplineSmooth
   superglm.NaturalSpline
   superglm.CubicRegressionSpline
   superglm.n_knots_from_k
   superglm.Categorical
   superglm.OrderedCategorical
   superglm.Numeric
   superglm.Piecewise
   superglm.Polynomial
   superglm.FactorSmooth
   superglm.RandomEffect
   superglm.LevelGrouping
   superglm.collapse_levels
   superglm.Constraint
   superglm.ConstraintSpec
   superglm.LinearConstraintSet
   superglm.SplineCategorical
   superglm.PolynomialCategorical
   superglm.NumericCategorical
   superglm.CategoricalInteraction
   superglm.NumericInteraction
   superglm.PolynomialInteraction
   superglm.TensorInteraction
```
