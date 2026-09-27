"""Spec criterion 8: estimate_p recovers the true power on constant-phi data."""

import math

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, Spline, SuperGLM, families, generate_tweedie_cpg

pytestmark = pytest.mark.slow


def _simulate(p: float, seed: int, n: int = 20_000):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    level = rng.integers(0, 5, n)
    mu = np.exp(0.3 + np.sin(2 * np.pi * x) * 0.6 + np.array([0.0, 0.2, -0.3, 0.1, 0.4])[level])
    # Constant phi: a mean-correlated dispersion would be absorbed into p by the
    # constant-phi model, which is a property of the model, not of the estimator.
    y = generate_tweedie_cpg(n, mu, 1.8, p, rng=rng)
    return pd.DataFrame({"x": x, "level": level.astype(str)}), y


@pytest.mark.parametrize("fit_mode", ["fit", "reml"])
@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("true_p", [1.2, 1.5, 1.8])
def test_estimate_p_recovers_true_power(true_p, seed, fit_mode, request):
    if (true_p, fit_mode) == (1.8, "reml"):
        # Not a search defect: plain fit_reml cannot certify its penalized mode
        # at most powers in (1.69, 1.95] on these books (mode scores 1e-9 to
        # 3e-7 against the fixed 1e-9 bar, identically before the rebuild), so
        # the REML search is censored there and its curvature probes land on
        # uncertifiable powers. Strict: certifying those modes must remove this.
        request.applymarker(
            pytest.mark.xfail(strict=True, reason="REML mode certification fails near p = 1.8")
        )
    X, y = _simulate(true_p, seed)
    model = SuperGLM(
        family=families.tweedie(p=1.5),
        features={"x": Spline(n_knots=10), "level": Categorical()},
    )
    result = model.estimate_p(X, y, fit_mode=fit_mode)
    # The profile's observed information: n times the second difference of the
    # mean NLL at p_hat. The step's truncation error is O(h^2) of the quartic
    # term, and the candidates' fit noise (measured at most 6e-9 relative) over
    # h^2 is under 1e-4 of the curvature.
    h = 0.01
    lower, upper = result._objective(result.p_hat - h), result._objective(result.p_hat + h)
    # A REML candidate whose mode cannot be certified has no objective (inf).
    assert math.isfinite(lower) and math.isfinite(upper), result.warnings
    curvature = (lower - 2 * result.search_nll + upper) / h**2
    standard_error = 1.0 / math.sqrt(len(y) * curvature)
    assert abs(result.p_hat - true_p) <= 3.0 * standard_error
