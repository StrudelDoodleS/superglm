"""Deterministic black-box reference checks for Tweedie REML fits."""

import numpy as np
import pandas as pd
import pytest

from superglm import SuperGLM, Tweedie
from superglm.features.spline import Spline


def test_fixed_power_terminal_scale_matches_black_box_reference() -> None:
    """The reported scale must describe the terminal fit, not its REML criterion."""
    n = 300
    x = np.linspace(0.0, 1.0, n)
    row = np.arange(1, n + 1)
    mean = np.exp(0.4 + np.sin(2.0 * np.pi * x))
    y = mean * (0.65 + 0.7 * ((row % 11) / 10.0))
    y[row % 5 == 0] = 0.0
    frame = pd.DataFrame({"x": x})

    model = SuperGLM(
        features={"x": Spline(n_knots=6)},
        family=Tweedie(p=1.5),
        selection_penalty=0,
    )
    model.fit_reml(frame, y, max_reml_iter=30)

    # Independent black-box reference generated with R 4.5.3/package 1.9-4.
    # Spline bases use different identifiability parameterizations, so compare
    # fit outcomes instead of raw smoothing parameters.
    assert model._reml_result.converged
    assert model.result.phi == pytest.approx(0.3741648, rel=0.02)
    assert model.result.deviance == pytest.approx(309.7028, rel=0.001)
    assert model.result.effective_df == pytest.approx(5.87704, abs=0.5)
    assert float(np.mean(model.predict(frame))) == pytest.approx(1.496676, rel=0.002)
