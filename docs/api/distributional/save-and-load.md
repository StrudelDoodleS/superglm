# Save and load

Save a fitted {py:class}`~superglm.SuperLSS` with joblib and load it back with
joblib:

```python
import joblib

joblib.dump(model, "severity.joblib")
restored = joblib.load("severity.joblib")
```

A restored model predicts exactly what the saved one predicted. It carries no
training frame, so the reading methods need `X_train=`.

The file does hold row-level fit state, one value per training row. With the
default `retain_rows=True` that is the response, the likelihood weights, the
offsets, and the fitted and null linear predictors and parameters. With any
setting it is the fitted linear predictors and parameters of the final fit and
of the last few fits in the smoothing history, and each predictor's offsets.
Treat the file as you treat the training data. Fitting with
`retain_rows=False` leaves out the first group; no option leaves out the
second.

Load files only from sources you trust: joblib files are pickles, and a hostile
pickle can run code.

`SuperLSS.to_bytes()` and `SuperLSS.from_bytes()` were removed; both raise an
`AttributeError` that says so. To convert a file that `to_bytes()` wrote, open
it with superglm 0.41 (`SuperLSS.from_bytes(...)`) and save the model again
with `joblib.dump`.
