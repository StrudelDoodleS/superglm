"""Public regressions for the two defects confirmed by the PR 611 debate."""

import mpmath as mp
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

from superglm import (
    DirichletLSS,
    GammaLS,
    GaussianLS,
    NegativeBinomialLS,
    SuperLSS,
    TwoPieceNormalLSS,
    bind_predictor,
    cross_validate,
)
from superglm.distributional.kernels import _negative_binomial_windows


def _intercept_model(family, *, semantics="prior"):
    return SuperLSS(
        family,
        *(bind_predictor(family, parameter.name) for parameter in family.parameters),
        weight_semantics=semantics,
    )


@pytest.mark.parametrize("family_type", [GaussianLS, GammaLS])
@pytest.mark.parametrize("entry", ["fit", "fit_reml", "cross_validate"])
def test_scalar_response_series_keeps_positional_pairing(family_type, entry):
    rng = np.random.default_rng(719)
    frame = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 120)})
    response = pd.Series(rng.gamma(4.0, 0.5, len(frame)), index=np.arange(len(frame)) + 700)
    series_model = _intercept_model(family_type())
    array_model = _intercept_model(family_type())

    if entry == "cross_validate":
        options = dict(
            cv=KFold(3),
            fit_kwargs={"lambdas": {}},
            scoring="nll",
            return_oof=True,
            error_score="raise",
        )
        series = cross_validate(series_model, frame, response, **options)
        array = cross_validate(array_model, frame, response.to_numpy(), **options)
        np.testing.assert_array_equal(series.oof_predictions, array.oof_predictions)
        assert series.mean_scores == array.mean_scores
    else:
        getattr(series_model, entry)(frame, response)
        getattr(array_model, entry)(frame, response.to_numpy())
        np.testing.assert_array_equal(
            series_model.result_.coefficients, array_model.result_.coefficients
        )
        np.testing.assert_array_equal(series_model.covariance_, array_model.covariance_)


@pytest.mark.parametrize("entry", ["fit", "fit_reml", "cross_validate"])
def test_vector_response_still_requires_matching_row_indices(entry):
    family = DirichletLSS(components=("a", "b"))
    model = SuperLSS(family, family.alpha("a"), family.alpha("b"))
    frame = pd.DataFrame({"x": np.arange(12.0)})
    response = pd.DataFrame(
        np.random.default_rng(12).dirichlet([2.0, 3.0], len(frame)),
        columns=["a", "b"],
        index=np.arange(len(frame)) + 700,
    )
    with pytest.raises(ValueError, match="response index must match"):
        if entry == "cross_validate":
            cross_validate(model, frame, response, cv=KFold(2), error_score="raise")
        else:
            getattr(model, entry)(frame, response)


def test_bounded_numerical_refusal_cannot_publish_infinite_scores():
    family = TwoPieceNormalLSS()
    training = pd.DataFrame({"x": np.arange(80.0)})
    model = _intercept_model(family, semantics="frequency").fit(
        training, np.random.default_rng(719).normal(size=len(training))
    )
    frame = pd.DataFrame({"x": np.arange(3.0)}, index=[80, 10, 50])
    target = np.array([[0.0, 1.0, 0.3], [-1e8, 1.0, 0.3], [0.0, 1.0, 0.3]])
    eta = model.predict_link(frame).to_numpy()
    offsets = {
        parameter.name: parameter.default_link.link(target[:, j]) - eta[:, j]
        for j, parameter in enumerate(family.parameters)
    }
    theta = model.predict_parameters(frame, offsets=offsets).to_numpy()
    # An independent positive-density conditional law has a finite,
    # representable score. The production kernel deliberately declines its
    # cancellation; the scorer must preserve that refusal, not invent zero mass.
    with mp.workdps(100):
        location, scale, skew = map(mp.mpf, theta[1])
        z0 = -location / (scale * (1 + skew))
        z1 = (1 - location) / (scale * (1 + skew))
        log_density = -(z1**2) / 2 - mp.log(2 * mp.pi) / 2 - mp.log(scale)
        log_window = mp.log(1 + skew) + mp.log(mp.erfc(z0 / mp.sqrt(2)) / 2)
        reference = float(-3 * (log_density - log_window))
    assert np.isfinite(reference)
    with pytest.raises(ValueError, match=r"invalid original rows \[1\]"):
        model.scores(
            frame,
            np.array([np.nan, 1.0, 1.0]),
            sample_weight=np.array([0.0, 3.0, 1.0]),
            which=("log",),
            offsets=offsets,
            truncation=(0.0, None),
        )


def test_bounded_nb_work_exhaustion_refuses_instead_of_scoring_infinity(monkeypatch):
    training = pd.DataFrame({"x": np.arange(120.0)})
    response = np.tile([0.0, 0.0, 1.0, 2.0, 4.0, 7.0], 20)
    model = _intercept_model(NegativeBinomialLS(), semantics="frequency").fit(training, response)
    frame = pd.DataFrame({"x": [0.0, 1.0, 2.0]}, index=[80, 10, 50])
    monkeypatch.setattr(_negative_binomial_windows, "_MAX_RECURRENCE_CELLS", 0)
    with pytest.raises(ValueError, match=r"invalid original rows \[1, 2\]"):
        model.scores(
            frame,
            np.array([np.nan, 2.0, 1.0]),
            sample_weight=np.array([0.0, 3.0, 1.0]),
            censored=[False, True, False],
            which=("log",),
        )
