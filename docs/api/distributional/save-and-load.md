# Save and load

{py:meth}`~superglm.SuperLSS.to_bytes` serialises one fitted revision and
{py:meth}`~superglm.SuperLSS.from_bytes` restores it after schema and
integrity checks. Load artifacts only from trusted sources. A restored model
carries no training frame or machine timing, so the reading methods need
`X_train=` and {py:meth}`~superglm.SuperLSS.diagnose` says the timing is
absent.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   ~superglm.SuperLSS.to_bytes
   ~superglm.SuperLSS.from_bytes
```
