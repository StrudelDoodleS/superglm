# Save and load

Save a fitted {py:class}`~superglm.SuperLSS` with joblib and load it back with
joblib:

```python
import joblib

joblib.dump(model, "severity.joblib")
restored = joblib.load("severity.joblib")
```

A restored model predicts exactly what the saved one predicted. It carries no
training frame or machine timing, so the reading methods need `X_train=` and
{py:meth}`~superglm.SuperLSS.diagnose` says the timing is absent. Load files
only from sources you trust: joblib files are pickles, and a hostile pickle
can run code.

`SuperLSS.to_bytes()` and `SuperLSS.from_bytes()` were removed; both raise an
`AttributeError` that says so. To convert a file that `to_bytes()` wrote, open
it with superglm 0.41 (`SuperLSS.from_bytes(...)`) and save the model again
with `joblib.dump`.
