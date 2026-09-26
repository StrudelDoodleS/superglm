# Features

Feature specifications turn raw columns into model terms. `Spline` is the
public factory for smooth terms; the concrete spline classes are what it
returns. Categorical, ordered-categorical, numeric, polynomial and piecewise
terms cover the rest of a rating structure; `FactorSmooth` and
`RandomEffect` add credibility-style shrinkage; `Constraint` requests
monotone or curvature constraints. The interaction types the constructor
builds from a pair of columns are on the [Internals](internals.md) page.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   superglm.Spline
   superglm.PSpline
   superglm.BSplineSmooth
   superglm.NaturalSpline
   superglm.CubicRegressionSpline
   superglm.PolynomialRange
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
```
