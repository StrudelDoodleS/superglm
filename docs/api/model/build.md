# Build

{py:class}`~superglm.SuperGLM` is built from a family, a feature specification
and a penalty policy, with every constructor argument described on its page.
Once the model exists, the properties below hold that configuration, and the
trailing-underscore attributes are filled in by the fit.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:
   :template: autosummary/class-no-members-model.rst

   superglm.SuperGLM
```

## Configuration and fitted attributes

{py:attr}`~superglm.SuperGLM.family`, {py:attr}`~superglm.SuperGLM.link`,
{py:attr}`~superglm.SuperGLM.penalty` and
{py:attr}`~superglm.SuperGLM.selection_penalty` hold the configuration and can
be assigned to change it before the next fit;
{py:attr}`~superglm.SuperGLM.lambda2` is the fixed smoothing penalty the
constructor takes as `spline_penalty`, assignable the same way; and
{py:attr}`~superglm.SuperGLM.features` is read-only. The trailing-underscore
attributes follow the scikit-learn convention and are resolved by the latest
successful fit: {py:attr}`~superglm.SuperGLM.selection_penalty_`,
{py:attr}`~superglm.SuperGLM.distribution_` and
{py:attr}`~superglm.SuperGLM.theta_`, the NB2 dispersion.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   ~superglm.SuperGLM.family
   ~superglm.SuperGLM.link
   ~superglm.SuperGLM.features
   ~superglm.SuperGLM.penalty
   ~superglm.SuperGLM.lambda2
   ~superglm.SuperGLM.selection_penalty
   ~superglm.SuperGLM.selection_penalty_
   ~superglm.SuperGLM.distribution_
   ~superglm.SuperGLM.theta_
```
