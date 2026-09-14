# Configuration

{py:attr}`~superglm.SuperLSS.discrete` and
{py:attr}`~superglm.SuperLSS.n_bins` echo the discrete-fitting setting:
grouped marginal designs and row chunks, which can reduce design memory, and
the bin count per feature. {py:attr}`~superglm.SuperLSS.separation` is the
build-time policy for categorical cells whose responses all sit on a boundary
the family declares: warn, error or ignore.
{py:attr}`~superglm.SuperLSS.weight_semantics` says what a `sample_weight`
entry means, how precisely a row was measured (`"prior"`, the default) or how
many identical rows it stands for (`"frequency"`).

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   ~superglm.SuperLSS.discrete
   ~superglm.SuperLSS.n_bins
   ~superglm.SuperLSS.separation
   ~superglm.SuperLSS.weight_semantics
```
