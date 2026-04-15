"""Tests for the fit_reml() selection-penalty contract.

Historically this file covered the old EFS REML path that mixed sparse
selection_penalty handling into fit_reml(). The current contract is simpler:
fit_reml() is the smoothness-selection path and requires selection_penalty=0.
"""

import numpy as np
import pandas as pd
import pytest

from superglm import SuperGLM
from superglm.features.spline import Spline


def _poisson_data(seed: int = 42, n: int = 800) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0, 10, n)
    x2 = rng.uniform(0, 10, n)
    mu = np.exp(0.5 + 0.3 * np.sin(x1) + 0.2 * np.cos(x2))
    y = rng.poisson(mu).astype(float)
    return pd.DataFrame({"x1": x1, "x2": x2}), y


def _gamma_data(seed: int = 123, n: int = 600) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0, 10, n)
    x2 = rng.uniform(0, 10, n)
    mu = np.exp(0.3 + 0.35 * np.sin(x1) + 0.15 * np.cos(x2))
    y = rng.gamma(shape=5.0, scale=mu / 5.0)
    y = np.maximum(y, 1e-4)
    return pd.DataFrame({"x1": x1, "x2": x2}), y


class TestREMLSelectionPenaltyContract:
    @pytest.mark.parametrize("selection_penalty", [1e-8, 0.01])
    def test_poisson_rejects_positive_selection_penalty(self, selection_penalty):
        X, y = _poisson_data()
        model = SuperGLM(
            family="poisson",
            selection_penalty=selection_penalty,
            features={
                "x1": Spline(n_knots=8, penalty="ssp"),
                "x2": Spline(n_knots=8, penalty="ssp"),
            },
        )

        with pytest.raises(ValueError, match="selection_penalty=0"):
            model.fit_reml(X, y, max_reml_iter=20)

    def test_gamma_rejects_positive_selection_penalty(self):
        X, y = _gamma_data()
        model = SuperGLM(
            family="gamma",
            selection_penalty=0.01,
            features={
                "x1": Spline(n_knots=6, penalty="ssp"),
                "x2": Spline(n_knots=6, penalty="ssp"),
            },
        )

        with pytest.raises(ValueError, match="selection_penalty=0"):
            model.fit_reml(X, y, max_reml_iter=12, verbose=True)

    def test_select_true_still_rejects_if_selection_penalty_positive(self):
        X, y = _poisson_data(n=500)
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.01,
            features={"x1": Spline(n_knots=8, penalty="ssp", select=True)},
        )

        with pytest.raises(ValueError, match="selection_penalty=0"):
            model.fit_reml(X[["x1"]], y, max_reml_iter=20)
