"""Numeric relativity reports preserve exp's range and confidence intervals."""

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from superglm import Numeric, SuperGLM


@pytest.mark.parametrize("coef, expected", [(-800.0, 0.0), (800.0, np.inf)])
def test_numeric_reconstruction_accepts_exp_range_limits_with_strict_errors(coef, expected):
    with np.errstate(all="raise"):
        report = Numeric().reconstruct(np.array([coef]))

    assert report["coef"] == coef
    assert report["relativity_per_unit"] == expected


@pytest.mark.parametrize("slope", [-800.0, -600.0, 600.0, 800.0])
def test_fitted_numeric_intervals_use_the_unclipped_exponential(slope):
    """Clipping at +/-500 moves both CI bounds outside an ordinary fitted point.

    An independent intercept-and-slope least-squares fit supplies the Gaussian
    coefficient and its standard error. The +/-800 cases also check the
    correctly rounded zero/infinity reports under strict NumPy error settings.
    These assertions fail on the original clipped intervals; -800 additionally
    exposes reconstruct's unsuppressed underflow.
    """
    rng = np.random.default_rng(541)
    x = rng.uniform(-1.0, 1.0, 200)
    y = 3.0 + slope * x + rng.normal(0.0, 0.1, len(x))
    model = SuperGLM(family="gaussian", features={"x": Numeric()}, selection_penalty=0.0).fit(
        pd.DataFrame({"x": x}), y
    )
    assert model.result.converged

    design = np.column_stack((np.ones_like(x), x))
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ beta
    variance = (residual @ residual) / (len(x) - 2)
    se = math.sqrt(variance * np.linalg.inv(design.T @ design)[1, 1])
    critical_value = norm.ppf(0.975)
    if slope == 800.0:
        expected_point = expected_lower = expected_upper = np.inf
    else:
        expected_point = math.exp(beta[1])
        expected_lower = math.exp(beta[1] - critical_value * se)
        expected_upper = math.exp(beta[1] + critical_value * se)

    with np.errstate(all="raise"):
        report = model.term_inference("x", with_se=True)

    np.testing.assert_allclose(report.log_relativity, beta[1], rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(report.se_log_relativity, se, rtol=1e-9, atol=0.0)
    # The fixture is well conditioned; allow rounding in the independent fits
    # and the amplification by exp of the coefficient's absolute error.
    np.testing.assert_allclose(report.relativity, expected_point, rtol=1e-10, atol=0.0)
    np.testing.assert_allclose(report.ci_lower, expected_lower, rtol=1e-10, atol=0.0)
    np.testing.assert_allclose(report.ci_upper, expected_upper, rtol=1e-10, atol=0.0)
    assert report.ci_lower[0] <= report.relativity[0] <= report.ci_upper[0]
