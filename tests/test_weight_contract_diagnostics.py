"""Family-specific weight-contract regressions for model diagnostics."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from superglm import Numeric, SuperGLM
from superglm._frame import as_eager_frame
from superglm.diagnostics.term_diagnostics import (
    _drop_term_holdout,
    _drop_term_refit,
    term_importance,
)
from superglm.distributions import Gaussian, NegativeBinomial, Tweedie
from superglm.profiling.tweedie import tweedie_logpdf
from superglm.stats.model_tests import vuong_test, zero_inflation_index


@pytest.mark.parametrize(
    ("family", "theta"),
    [
        ("poisson", None),
        ("nb2", 2.3),
    ],
)
def test_weighted_zero_inflation_index_matches_literal_row_replication(
    family,
    theta,
) -> None:
    y = np.array([0.0, 1.0, 0.0, 3.0, 2.0, 0.0])
    mu = np.array([0.4, 1.2, 0.8, 2.7, 1.9, 1.5])
    weights = np.array([3, 1, 4, 2, 1, 2])

    weighted = zero_inflation_index(
        y,
        mu,
        sample_weight=weights,
        family=family,
        theta=theta,
    )
    rows = np.repeat(np.arange(len(y)), weights)
    repeated = zero_inflation_index(
        y[rows],
        mu[rows],
        family=family,
        theta=theta,
    )

    assert weighted.observed_zeros == pytest.approx(repeated.observed_zeros)
    assert weighted.expected_zeros == pytest.approx(repeated.expected_zeros)
    assert weighted.zero_inflation_index == pytest.approx(repeated.zero_inflation_index)
    assert weighted.ratio == pytest.approx(repeated.ratio)
    assert weighted.observed_zeros == pytest.approx(np.sum(weights * (y == 0.0)))


class _Drop1ICModel:
    def __init__(self, family, *, log_likelihood: float, edf: float, reduced_bic: float):
        self._distribution = family
        self._fit_stats = SimpleNamespace(log_likelihood=log_likelihood)
        self.result = SimpleNamespace(effective_df=edf)
        self._reduced_bic = reduced_bic

    def drop1(self, X, y, sample_weight=None, offset=None):
        del X, y, sample_weight, offset
        return pd.DataFrame(
            {
                "feature": ["x"],
                "aic": [47.0],
                "bic": [self._reduced_bic],
            }
        )


@pytest.mark.parametrize(
    ("family", "likelihood_size"),
    [
        (Gaussian(), 10.0),
        (Tweedie(p=1.5), 4.0),
    ],
)
def test_term_refit_bic_baseline_uses_family_likelihood_size(
    family,
    likelihood_size,
) -> None:
    y = np.array([0.2, 0.7, 1.1, 2.0])
    weights = np.array([1.0, 2.0, 3.0, 4.0])
    full_ll = -18.0
    full_edf = 3.25
    reduced_bic = 52.0
    model = _Drop1ICModel(
        family,
        log_likelihood=full_ll,
        edf=full_edf,
        reduced_bic=reduced_bic,
    )

    result = _drop_term_refit(model, object(), y, weights, None)
    expected_full_bic = -2.0 * full_ll + np.log(likelihood_size) * full_edf

    assert result.loc[0, "delta_bic"] == pytest.approx(reduced_bic - expected_full_bic)


def _family_for_frequency_test(name: str):
    if name == "nb2":
        return NegativeBinomial(theta=2.4)
    return name


def _fit_frequency_comparison(
    family_name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    sample_weight=None,
):
    model_a = SuperGLM(
        family=_family_for_frequency_test(family_name),
        features={"x": Numeric(), "z": Numeric()},
        selection_penalty=0.0,
    ).fit(X, y, sample_weight=sample_weight)
    model_b = SuperGLM(
        family=_family_for_frequency_test(family_name),
        features={"x": Numeric()},
        selection_penalty=0.0,
    ).fit(X, y, sample_weight=sample_weight)
    return model_a, model_b


@pytest.mark.parametrize("family_name", ["gaussian", "poisson", "nb2"])
def test_frequency_weighted_vuong_matches_literal_row_replication(
    family_name: str,
) -> None:
    rng = np.random.default_rng(21901)
    n = 55
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    eta = 0.25 + 0.45 * x - 0.55 * z
    if family_name == "gaussian":
        y = eta + rng.normal(scale=0.7, size=n)
    elif family_name == "poisson":
        y = rng.poisson(np.exp(eta)).astype(float)
    else:
        theta = 2.4
        mu = np.exp(eta)
        y = rng.negative_binomial(theta, theta / (theta + mu)).astype(float)
    X = pd.DataFrame({"x": x, "z": z})
    weights = rng.integers(1, 5, size=n).astype(float)

    weighted_a, weighted_b = _fit_frequency_comparison(
        family_name,
        X,
        y,
        weights,
    )
    rows = np.repeat(np.arange(n), weights.astype(int))
    repeated_X = X.iloc[rows].reset_index(drop=True)
    repeated_y = y[rows]
    repeated_a, repeated_b = _fit_frequency_comparison(
        family_name,
        repeated_X,
        repeated_y,
    )

    for correction in ("none", "aic", "bic"):
        weighted = vuong_test(
            weighted_a,
            weighted_b,
            X,
            y,
            sample_weight=weights,
            correction=correction,
        )
        repeated = vuong_test(
            repeated_a,
            repeated_b,
            repeated_X,
            repeated_y,
            correction=correction,
        )

        assert weighted.mean_diff == pytest.approx(repeated.mean_diff, rel=5e-12, abs=5e-12)
        assert weighted.omega == pytest.approx(repeated.omega, rel=5e-12, abs=5e-12)
        assert weighted.statistic == pytest.approx(repeated.statistic, rel=5e-12, abs=5e-12)
        assert weighted.p_value == pytest.approx(repeated.p_value, rel=5e-12, abs=5e-12)
        assert weighted.preferred == repeated.preferred


class _PredictionModel:
    def __init__(
        self,
        family,
        mu,
        *,
        phi: float,
        effective_df: float,
    ):
        self._distribution = family
        self._mu = np.asarray(mu, dtype=np.float64)
        self.result = SimpleNamespace(phi=phi, effective_df=effective_df)

    def predict(self, X, offset=None):
        del X
        if offset is None:
            return self._mu
        return self._mu * np.exp(np.asarray(offset, dtype=np.float64))


def test_tweedie_vuong_uses_phi_over_prior_weight_density_and_physical_n() -> None:
    y = np.array([0.0, 0.15, 0.5, 1.1, 2.4, 4.0])
    X = pd.DataFrame({"x": np.arange(len(y), dtype=float)})
    weights = np.array([0.4, 1.2, 2.5, 0.7, 3.1, 1.8])
    model_a = _PredictionModel(
        Tweedie(p=1.45),
        [0.2, 0.3, 0.7, 1.0, 2.1, 3.3],
        phi=0.8,
        effective_df=3.4,
    )
    model_b = _PredictionModel(
        Tweedie(p=1.65),
        [0.35, 0.4, 0.6, 1.4, 1.7, 3.9],
        phi=1.1,
        effective_df=2.2,
    )

    actual = vuong_test(
        cast(SuperGLM, model_a),
        cast(SuperGLM, model_b),
        X,
        y,
        sample_weight=weights,
        correction="bic",
    )
    density_diff = tweedie_logpdf(
        y,
        model_a._mu,
        model_a.result.phi,
        model_a._distribution.p,
        weights=weights,
    ) - tweedie_logpdf(
        y,
        model_b._mu,
        model_b.result.phi,
        model_b._distribution.p,
        weights=weights,
    )
    n = len(y)
    expected_omega = float(np.std(density_diff, ddof=1))
    expected_mean = float(
        np.mean(density_diff)
        - (model_a.result.effective_df - model_b.result.effective_df) * np.log(n) / (2 * n)
    )
    expected_statistic = np.sqrt(n) * expected_mean / expected_omega

    assert actual.mean_diff == pytest.approx(expected_mean)
    assert actual.omega == pytest.approx(expected_omega)
    assert actual.statistic == pytest.approx(expected_statistic)
    assert actual.p_value == pytest.approx(2.0 * stats.norm.sf(abs(expected_statistic)))


@pytest.mark.parametrize("invalid_weight", [0.0, -0.5, np.nan])
def test_tweedie_vuong_rejects_invalid_prior_weights(invalid_weight: float) -> None:
    y = np.array([0.0, 0.4, 1.2])
    X = pd.DataFrame({"x": np.arange(len(y), dtype=float)})
    model_a = _PredictionModel(
        Tweedie(p=1.4),
        [0.2, 0.5, 1.0],
        phi=0.7,
        effective_df=2.0,
    )
    model_b = _PredictionModel(
        Tweedie(p=1.6),
        [0.3, 0.6, 1.1],
        phi=0.9,
        effective_df=1.5,
    )
    weights = np.ones(len(y))
    weights[1] = invalid_weight

    with pytest.raises(ValueError, match="strictly positive"):
        vuong_test(
            cast(SuperGLM, model_a),
            cast(SuperGLM, model_b),
            X,
            y,
            sample_weight=weights,
        )


def test_weighted_mixed_tweedie_vuong_rejects_incompatible_semantics() -> None:
    y = np.array([0.0, 0.4, 1.2])
    X = pd.DataFrame({"x": np.arange(len(y), dtype=float)})
    tweedie_model = _PredictionModel(
        Tweedie(p=1.5),
        [0.2, 0.5, 1.0],
        phi=0.7,
        effective_df=2.0,
    )
    gaussian_model = _PredictionModel(
        Gaussian(),
        [0.1, 0.4, 1.1],
        phi=0.8,
        effective_df=1.5,
    )

    with pytest.raises(ValueError, match="incompatible semantics"):
        vuong_test(
            cast(SuperGLM, tweedie_model),
            cast(SuperGLM, gaussian_model),
            X,
            y,
            sample_weight=np.ones(len(y)),
        )


def test_tweedie_term_diagnostics_reject_nonpositive_prior_weights() -> None:
    n = 3
    X = pd.DataFrame(index=np.arange(n))
    model = SimpleNamespace(
        _result=object(),
        _distribution=Tweedie(p=1.5),
        result=SimpleNamespace(beta=np.array([], dtype=np.float64)),
    )
    invalid_weights = np.array([1.0, 0.0, 1.0])

    with pytest.raises(ValueError, match="strictly positive"):
        term_importance(model, X, sample_weight=invalid_weights)
    with pytest.raises(ValueError, match="strictly positive"):
        _drop_term_holdout(
            model,
            as_eager_frame(X),
            np.array([0.0, 0.5, 1.0]),
            sample_weight_val=invalid_weights,
        )
