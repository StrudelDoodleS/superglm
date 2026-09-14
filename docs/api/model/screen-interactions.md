# Screen interactions

{py:meth}`~superglm.SuperGLM.screen_interactions` ranks candidate pairs of
fitted features by PSST, the penalised smooth score test: one O(n) pass per
pair and no refits, asking how much of the model's leftover working signal
each interaction could absorb once the pair's own main effects are profiled
out. Run it before adding interactions to the spec; the [how-to on screening
interactions](../../how-to/screen-interactions.md) covers reading the `z` and
`kind` columns.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   ~superglm.SuperGLM.screen_interactions
```
