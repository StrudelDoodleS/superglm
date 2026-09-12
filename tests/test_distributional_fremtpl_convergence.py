"""Regressions for isolated-tail Gamma fits on uncapped freMTPL claims."""

from __future__ import annotations

import numpy as np
import pytest

from superglm import Categorical, GammaLS, Predictor, Spline
from tests.bound_predictor_fixtures import model_from_templates

from . import _datasets

_NUMERIC = ["VehAge", "DrivAge", "BonusMalus", "LogDensity", "VehPower"]
_CATEGORICAL = ["Area", "VehBrand", "VehGas", "Region"]
_MISSING = _datasets.skip_reason("freMTPL2freq.parquet") or _datasets.skip_reason(
    "freMTPL2sev.parquet"
)
pytestmark = pytest.mark.skipif(_MISSING is not None, reason=_MISSING or "")


@pytest.fixture(scope="module")
def uncapped_claims():
    claims = _datasets.load_sev().merge(
        _datasets.load_freq(), on="IDpol", how="inner", validate="many_to_one"
    )
    assert (claims.ClaimAmount > 0).all()
    claims["LogDensity"] = np.log(claims.Density)
    claims["y"] = claims.ClaimAmount / 1000.0
    rng = np.random.default_rng(20260909)
    policies = np.sort(claims.IDpol.unique())
    rng.shuffle(policies)
    split = {
        policy: "train"
        if i < int(0.6 * len(policies))
        else "valid"
        if i < int(0.8 * len(policies))
        else "test"
        for i, policy in enumerate(policies)
    }
    claims["split"] = claims.IDpol.map(split)
    claims["budget_order"] = rng.permutation(len(claims))
    for name in _CATEGORICAL:
        claims[name] = claims[name].astype(str)
    train = claims.loc[claims.split == "train"].sort_values("budget_order").reset_index(drop=True)
    full = claims.loc[claims.split != "test"].reset_index(drop=True)
    return train, full


@pytest.mark.parametrize("case", ["all_train", "all_full", "selected_full"])
def test_uncapped_gamma_mean_and_scale_converge(uncapped_claims, case):
    """Kills weak automatic starts and bitwise-only exact-face revalidation."""
    train, full = uncapped_claims
    data = train if case == "all_train" else full
    levels = {name: sorted(train[name].unique()) for name in _CATEGORICAL}

    def features(names):
        return {
            name: Spline("cr", k=8) if name in _NUMERIC else Categorical(levels=levels[name])
            for name in names
        }

    all_features = _NUMERIC + _CATEGORICAL
    scale_features = ["DrivAge", "VehPower"] if case == "selected_full" else all_features
    options = {"initial_lambda": 0.1} if case == "selected_full" else {}
    model = model_from_templates(
        family=GammaLS(),
        predictors=[
            Predictor("mean", features(all_features)),
            Predictor("scale", features(scale_features)),
        ],
    ).fit_reml(data[all_features], data.y.to_numpy(), **options)
    state = model._require_fitted().fit_state
    fit = state.solver_result
    assert len(fit.theta) == len(data)
    assert model.result_.converged
    assert fit.converged
    assert fit.score_relative <= fit.config.tolerance
    retained_width = (
        len(fit.coefficients)
        if fit.coefficient_face is None
        else fit.coefficient_face.reduced_width
    )
    assert fit.terminal_rank.rank == retained_width
    assert fit.terminal_curvature.actual_source == "observed"
    assert all(
        item.terminal_curvature.fallback_count == 0 for item in state.smoothing.coefficient_fits
    )
    if case == "selected_full":
        assert state.exact_face_components == ("mean:VehAge#wiggle",)


def test_uncapped_gamma_explicit_weak_start_refuses_unresolved_stationarity(uncapped_claims):
    """The isolated claim stays in training; an unresolved score cannot certify a fit."""
    from scipy.special import digamma

    train, _ = uncapped_claims
    levels = {name: sorted(train[name].unique()) for name in _CATEGORICAL}
    all_features = _NUMERIC + _CATEGORICAL

    def features():
        return {
            name: Spline("cr", k=8) if name in _NUMERIC else Categorical(levels=levels[name])
            for name in all_features
        }

    model = model_from_templates(
        family=GammaLS(),
        predictors=[Predictor("mean", features()), Predictor("scale", features())],
    ).fit_reml(train[all_features], train.y.to_numpy(), initial_lambda=0.1)
    state = model._require_fitted().fit_state
    fit = state.solver_result
    np.testing.assert_array_equal(state.retained_rows.response, train.y.to_numpy())
    assert len(fit.theta) == len(train)
    assert all(value == 0.1 for value in state.smoothing.initial_lambdas.values())
    mean, scale = fit.theta.T
    shape = 1.0 / scale**2
    ratio = train.y.to_numpy() / mean
    bracket = np.log(shape) + 1 - digamma(shape) + np.log(ratio) - ratio
    row_score = (shape * (ratio - 1), -2 * shape * bracket)
    score, magnitudes = [], []
    for k, predictor in enumerate(state.layout.predictors):
        matrix = np.column_stack((np.ones(len(train)), predictor.design.toarray()))
        score.append(matrix.T @ row_score[k])
        # Include the terms before cancellation in the Gamma shape derivative.
        row_magnitude = (
            shape * (ratio + 1)
            if k == 0
            else 2
            * shape
            * (abs(np.log(shape)) + 1 + abs(digamma(shape)) + abs(np.log(ratio)) + ratio)
        )
        magnitudes.append(abs(matrix).T @ row_magnitude)
    independent = np.concatenate(score) - fit.penalty @ fit.coefficients
    magnitude = np.concatenate(magnitudes) + abs(fit.penalty) @ abs(fit.coefficients)
    operations = 64 * (len(train) + len(fit.coefficients))
    arithmetic = operations * np.finfo(float).eps / (1 - operations * np.finfo(float).eps)
    error = arithmetic * (1 + magnitude)
    assert np.all(np.isfinite(independent))
    assert np.linalg.norm(independent - fit.terminal_score, np.inf) <= max(error)
    denominator = 1 + abs(fit.penalized_optimizing_log_likelihood)
    unresolved = np.max(np.maximum(abs(independent) - error, 0.0)) / denominator
    assert unresolved > fit.config.tolerance
    assert not fit.converged
    assert not model.result_.coefficient_converged
    assert not model.result_.converged
    report = model.diagnose()
    assert any(finding.identifier == "fit.not_converged:coefficient" for finding in report.findings)
