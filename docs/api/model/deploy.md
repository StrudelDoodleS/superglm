# Deploy

{py:meth}`~superglm.SuperGLM.export_rating_tables` writes the deployment
rating tables for the fitted model;
{py:meth}`~superglm.SuperGLM.rating_table_payload` builds the
renderer-independent payload behind them, for when you need the tables as
objects rather than files. The [how-to on deploying a fitted
model](../../how-to/deploy-a-fitted-model.md) shows the export end to end.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   ~superglm.SuperGLM.export_rating_tables
   ~superglm.SuperGLM.rating_table_payload
```
