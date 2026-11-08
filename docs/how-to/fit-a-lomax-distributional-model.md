# Fit a Lomax distributional model

Choose what the response means before fitting: ground-up severity, recorded
payment, or excess above a fixed threshold are different observation laws.
Start with a constant tail exponent and a modest scale predictor:

```python
from superglm import LomaxLSS, SuperLSS, s

family = LomaxLSS()
model = SuperLSS(
    family,
    family.scale(s("age", k=8)),
    family.shape(),
).fit(X, severity)

parameters = model.predict_parameters(X_new)
q99 = model.predict_quantile(X_new, 0.99)
```

Both natural parameters are positive and use log links. Shape is the power-tail
exponent, not the GPD index. Mean is `scale/(shape-1)` only above shape one;
`predict()` returns infinity otherwise. Variance needs shape above two. Use
parameters, quantiles or bounded layers when the required moment does not exist.
Do not impose a finite-mean constraint by clipping fitted shapes.

For a cap, supply the paid amount and `censored=paid >= cap` if that flag
correctly describes how the data were recorded. For reporting thresholds,
supply `truncation=(reporting_threshold, float("inf"))`. Bounds can vary by row
and can be combined with left/right censoring. The fitted law is the parent
severity law, normalized for the actual observation region during fitting.
General interval censoring is not part of this API. Integer replication belongs
in `SuperLSS(..., weight_semantics="frequency")` with `sample_weight` at fit;
exposure is not an interchangeable severity prior weight.

A bounded layer is finite even below shape one. The following callback computes
`E[min((Y-attachment)+, width)]` per claim and can be passed to the existing
`model.posterior_bounds(X_new, layer)` method:

```python
import numpy as np

def layer(theta, attachment=1_000.0, width=10_000.0):
    scale, shape = theta.T
    log_start = np.log1p(attachment / scale)
    log_span = np.log1p(width / (scale + attachment))
    v = (1.0 - shape) * log_span
    ratio = np.ones_like(v)
    np.divide(np.expm1(v), v, out=ratio, where=v != 0.0)
    return np.exp(np.log(scale) + (1.0 - shape) * log_start) * log_span * ratio
```

This avoids direct subtraction of powers near shape one. The recipe assumes
representable monetary inputs and intermediates; it is not a general extreme-
range integration API. Posterior intervals propagate the fitted coefficient
approximation, conditional on the family and threshold. Named expected-shortfall
summaries refuse draws with an infinite tail mean rather than dropping them.
Per-claim layer expectations do not include claim frequency or portfolio
dependence.

Keep training and validation seeds/splits separate, and fix thresholds and
layers before inspecting validation outcomes. Compare observed-law log score,
exceedance calibration and counts, quantile loss, and bounded-layer calibration.
Censored observations cannot validate unobserved latent payments. Inspect
`result_` warnings and `diagnose()` for convergence, rank and derivative authority;
try more than one smoothing start before interpreting a flexible shape curve.
Synthetic engineering comparisons do not establish superiority on a real book.
