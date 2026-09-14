# Declarations

Inside a predictor, {py:func}`~superglm.s` declares a smooth of one numeric
column, {py:func}`~superglm.cat` a categorical effect with a reference level,
{py:func}`~superglm.re` a random effect with a coefficient for every level,
{py:func}`~superglm.term` attaches any other feature specification to a
column, and {py:func}`~superglm.ti` and {py:func}`~superglm.interaction`
declare interactions between terms already in that predictor. Those helpers
return {py:class}`~superglm.BoundTerm` and
{py:class}`~superglm.BoundInteraction`; the family helper wraps them in a
{py:class}`~superglm.BoundPredictor`, which
{py:func}`~superglm.bind_predictor` also creates by parameter name for custom
families, and {py:class}`~superglm.Predictor` is the immutable configuration
underneath; those four are described on the [Internals](../internals.md)
page. The families and their helper methods are on the
[Families](families.md) page.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   superglm.bind_predictor
   superglm.term
   superglm.s
   superglm.cat
   superglm.re
   superglm.ti
   superglm.interaction
```
