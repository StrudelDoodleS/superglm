"""Canary tests for Tweedie IRLS convergence.

Tweedie p=1.5 with log link and high zero-inflation (~80%) stresses the IRLS
solver: V(mu)=mu^1.5 creates extreme working weight ranges, and the non-canonical
log link means the Newton step can overshoot near convergence.  These tests verify
that the condition-aware SVD fallback in _robust_solve() prevents divergence.
"""

import numpy as np
import pandas as pd
import pytest

from superglm import SuperGLM
from superglm.distributions import Tweedie
from superglm.features.numeric import Numeric
from superglm.features.spline import Spline
from superglm.penalties.group_lasso import GroupLasso
from superglm.profiling.tweedie import generate_tweedie_cpg


class TestTweedieConvergence:
    """Verify IRLS convergence on synthetic Tweedie data with zeros."""

    @pytest.fixture()
    def tweedie_data(self):
        """Synthetic Tweedie dataset: 5000 obs, p=1.5, ~80% zeros."""
        rng = np.random.default_rng(2026)
        n = 5000
        x1 = rng.uniform(0, 1, n)
        x2 = rng.uniform(0, 1, n)
        # True linear predictor: intercept + effects
        eta_true = -1.0 + 0.5 * x1 - 0.3 * x2
        mu_true = np.exp(eta_true)
        y = generate_tweedie_cpg(n, mu=mu_true, phi=2.0, p=1.5, rng=rng)
        zero_frac = np.mean(y == 0)
        assert zero_frac > 0.5, f"Expected >50% zeros, got {zero_frac:.0%}"
        df = pd.DataFrame({"x1": x1, "x2": x2})
        return df, y

    def test_direct_solver_converges(self, tweedie_data):
        """Direct IRLS (lambda1=0) must converge on Tweedie with zeros."""
        df, y = tweedie_data
        model = SuperGLM(
            family=Tweedie(p=1.5),
            penalty=GroupLasso(lambda1=0.0),
            features={"x1": Numeric(), "x2": Numeric()},
        )
        model.fit(df, y)
        assert model._result.converged
        assert np.isfinite(model._result.deviance)
        assert model._result.deviance > 0

    def test_bcd_solver_converges(self, tweedie_data):
        """BCD PIRLS (lambda1>0) must converge on Tweedie with zeros."""
        df, y = tweedie_data
        model = SuperGLM(
            family=Tweedie(p=1.5),
            penalty=GroupLasso(lambda1=0.01),
            features={"x1": Numeric(), "x2": Numeric()},
        )
        model.fit(df, y)
        assert model._result.converged
        assert np.isfinite(model._result.deviance)

    def test_spline_model_converges(self, tweedie_data):
        """Tweedie with splines (higher p) must converge."""
        df, y = tweedie_data
        model = SuperGLM(
            family=Tweedie(p=1.5),
            penalty=GroupLasso(lambda1=0.0),
            features={"x1": Spline(n_knots=8), "x2": Spline(n_knots=8)},
        )
        model.fit(df, y)
        assert model._result.converged
        assert np.isfinite(model._result.deviance)

    def test_predictions_finite(self, tweedie_data):
        """Predictions from converged Tweedie model must be finite and positive."""
        df, y = tweedie_data
        model = SuperGLM(
            family=Tweedie(p=1.5),
            penalty=GroupLasso(lambda1=0.0),
            features={"x1": Numeric(), "x2": Numeric()},
        )
        model.fit(df, y)
        preds = model.predict(df)
        assert np.all(np.isfinite(preds))
        assert np.all(preds > 0)

    def test_qr_direct_solver_converges(self, tweedie_data):
        """QR-based direct IRLS converges on Tweedie with zeros."""
        df, y = tweedie_data
        model = SuperGLM(
            family=Tweedie(p=1.5),
            penalty=GroupLasso(lambda1=0.0),
            features={"x1": Numeric(), "x2": Numeric()},
            direct_solve="qr",
        )
        model.fit(df, y)
        assert model._result.converged
        assert np.isfinite(model._result.deviance)
        assert model._result.deviance > 0
