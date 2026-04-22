"""Runtime canonicalization regressions for spline-backed public terms."""

import numpy as np
import pandas as pd

from superglm import SuperGLM
from superglm.features.spline import PSpline, Spline


def _feature_beta(model: SuperGLM, feature_name: str) -> np.ndarray:
    groups = model._feature_groups(feature_name)
    return np.concatenate([model.result.beta[g.sl] for g in groups])


def _runtime_canonicalization_diagnostics(model: SuperGLM) -> dict:
    state = getattr(model, "_runtime_canonical_state", None)
    assert state is not None
    diagnostics = state.get("diagnostics")
    assert diagnostics is not None
    return diagnostics


def _assert_zero_mean_and_before_after_parity(
    model: SuperGLM,
    feature_name: str,
    x: np.ndarray,
) -> None:
    spec = model._specs[feature_name]
    beta = _feature_beta(model, feature_name)
    contribution = spec.score(x, beta)
    assert abs(np.mean(contribution)) < 1e-10

    diagnostics = _runtime_canonicalization_diagnostics(model)
    assert diagnostics["max_abs_eta_delta"] < 1e-10
    assert diagnostics["max_abs_mu_delta"] < 1e-10


def _make_monotone_poisson_data(
    n: int = 800,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0.0, 1.0, n))
    sample_weight = rng.uniform(0.3, 1.2, n)
    eta = -1.2 + 1.1 * x
    y = rng.poisson(sample_weight * np.exp(eta)).astype(float)
    X = pd.DataFrame({"x": x})
    return X, y, sample_weight


class TestRuntimeCanonicalization:
    def test_standalone_spline_public_term_is_zero_mean_and_prediction_parity_is_locked(self):
        X, y, sample_weight = _make_monotone_poisson_data(seed=0)
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            features={"x": Spline(n_knots=10, penalty="ssp")},
        )
        model.fit_reml(X, y, sample_weight=sample_weight, max_reml_iter=10)

        _assert_zero_mean_and_before_after_parity(
            model,
            "x",
            X["x"].to_numpy(dtype=np.float64),
        )

    def test_monotone_fit_spline_public_term_is_zero_mean_and_prediction_parity_is_locked(self):
        X, y, sample_weight = _make_monotone_poisson_data(seed=1)
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            features={
                "x": PSpline(
                    n_knots=10,
                    penalty="ssp",
                    monotone="increasing",
                    monotone_mode="fit",
                )
            },
        )
        model.fit(X, y, sample_weight=sample_weight)

        _assert_zero_mean_and_before_after_parity(
            model,
            "x",
            X["x"].to_numpy(dtype=np.float64),
        )
