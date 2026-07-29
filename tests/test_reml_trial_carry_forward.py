"""Nonconverged accepted line-search trials must not be carried forward.

Review finding: an Armijo-accepted expected/Fisher trial that exhausted
``max_pirls_iter`` was stored as the next iteration's candidate state, so
REML derivatives were evaluated at nonstationary coefficients instead of
warm-start refitting them.  With a tight per-trial PIRLS budget the outer
loop must still land where the unconstrained fit lands.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from superglm import SuperGLM
from superglm.features.spline import Spline


def _fit(frame, y, **kwargs):
    model = SuperGLM(
        family="poisson",
        selection_penalty=None,
        discrete=False,
        features={c: Spline(kind="ps", k=8) for c in frame.columns},
    )
    return model.fit_reml(frame, y, **kwargs)


def test_iteration_limited_trials_are_refit_not_reused():
    rng = np.random.default_rng(0)
    n = 1500
    frame = pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(3)})
    eta = -1.0 + 0.8 * np.sin(frame["x0"]) + 0.5 * frame["x1"]
    y = rng.poisson(np.exp(eta)).astype(float)

    free = _fit(frame, y)
    tight = _fit(frame, y, max_pirls_iter=2)

    assert np.isfinite(tight._result.deviance)
    np.testing.assert_allclose(
        tight._result.deviance, free._result.deviance, rtol=1e-5
    )
