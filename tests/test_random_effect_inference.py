"""Compact covariance and retained-state tests for random effects."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.optimize

import superglm.model.state_ops as state_ops
from superglm import LambdaPolicy, Numeric, RandomEffect, SuperGLM
from superglm.distributions import _VARIANCE_FLOOR, Binomial, Gamma, Gaussian, clip_mu
from superglm.inference.covariance import StructuredCovarianceAccessor
from superglm.inference.random_effects import (
    RandomEffectResult,
    vectorized_conditional_unpooled_effect,
)
from superglm.links import IdentityLink, LogitLink, LogLink, stabilize_eta
from superglm.solvers.structured import StructuredLinearSystemState


def _fit_pair(
    *,
    retain_fit_state: bool = True,
    n_levels: int = 18,
    fit_dense: bool = True,
    max_reml_iter: int = 7,
) -> tuple[SuperGLM | None, SuperGLM, pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(862)
    repeats = 18
    codes = np.repeat(np.arange(n_levels), repeats)
    rng.shuffle(codes)
    x = rng.normal(size=len(codes))
    exposure = rng.uniform(0.5, 1.8, size=len(codes))
    effects = rng.normal(scale=0.3, size=n_levels)
    y = rng.poisson(exposure * np.exp(-0.3 + 0.2 * x + effects[codes])).astype(float)
    X = pd.DataFrame(
        {
            "x": x,
            "broker": np.array([f"b{i}" for i in codes], dtype=object),
        }
    )
    common = {
        "family": "poisson",
        "features": {"x": Numeric(), "broker": RandomEffect()},
        "selection_penalty": 0,
        "retain_fit_state": retain_fit_state,
    }
    dense = SuperGLM(**common, direct_solve="gram") if fit_dense else None
    structured = SuperGLM(**common, direct_solve="structured")
    if dense is not None:
        dense.fit_reml(X, y, offset=np.log(exposure), max_reml_iter=max_reml_iter)
    structured.fit_reml(
        X,
        y,
        offset=np.log(exposure),
        max_reml_iter=max_reml_iter,
    )
    return dense, structured, X, y, exposure


def test_structured_selected_covariance_matches_dense_augmented_inverse(monkeypatch):
    dense, structured, _, _, _ = _fit_pair()
    assert dense is not None

    def fail_dense_legacy(*_args, **_kwargs):
        raise AssertionError("structured inference rebuilt a dense coefficient system")

    monkeypatch.setattr(state_ops, "_legacy_active_state", fail_dense_legacy)

    dense_augmented = dense._fit_inference_info["XtWX_inv_aug"]
    compact_info = structured._fit_inference_info
    compact = compact_info["XtWX_inv_aug"]

    assert isinstance(structured._linear_system_state, StructuredLinearSystemState)
    assert isinstance(compact, StructuredCovarianceAccessor)

    # Intercept, the numeric slope, and a few random-effect levels exercise
    # every augmented covariance block without asking for the full K x K block.
    selected = np.array([0, 1, 2, 5, 9], dtype=np.intp)
    np.testing.assert_allclose(
        compact.selected_block(selected),
        dense_augmented[np.ix_(selected, selected)],
        rtol=3e-8,
        atol=3e-9,
    )
    np.testing.assert_allclose(
        compact.selected_diagonal(selected),
        np.diag(dense_augmented)[selected],
        rtol=3e-8,
        atol=3e-9,
    )

    slope_indices = np.array([0, 1, 4, 8], dtype=np.intp)
    np.testing.assert_allclose(
        compact.slope_selected_block(slope_indices),
        dense_augmented[1:, 1:][np.ix_(slope_indices, slope_indices)],
        rtol=3e-8,
        atol=3e-9,
    )
    np.testing.assert_allclose(
        compact.intercept_cross(slope_indices),
        dense_augmented[0, 1:][slope_indices],
        rtol=3e-8,
        atol=3e-9,
    )
    assert compact.intercept_variance() == pytest.approx(
        dense_augmented[0, 0],
        rel=3e-8,
        abs=3e-9,
    )
    assert compact_info["group_edf_map"]["broker"] == pytest.approx(
        dense._fit_inference_info["group_edf_map"]["broker"],
        rel=3e-8,
        abs=3e-9,
    )


def test_structured_summary_uses_selected_covariance_only(monkeypatch):
    _, structured, X, y, _ = _fit_pair()

    def fail_dense_legacy(*_args, **_kwargs):
        raise AssertionError("summary requested the legacy dense covariance path")

    monkeypatch.setattr(state_ops, "_legacy_active_state", fail_dense_legacy)

    summary = structured.summary()
    assert summary["fit"]["n_obs"] > 0
    assert np.isfinite(structured.metrics(X, y).coefficient_se["x"][0])


def test_released_structured_state_keeps_compact_factors_and_support():
    _, structured, X, _, exposure = _fit_pair(
        retain_fit_state=False,
        n_levels=48,
        fit_dense=False,
    )

    state = structured._linear_system_state
    assert isinstance(state, StructuredLinearSystemState)
    assert structured._dm is None
    assert structured._fit_weights is None
    assert structured._fit_X_ref is None
    assert isinstance(
        structured.__dict__["_fit_inference_info"]["XtWX_inv_aug"],
        StructuredCovarianceAccessor,
    )
    assert state.coefficient_factor.shape == (len(structured.result.beta),) * 2
    assert state.augmented_factor.shape == (len(structured.result.beta) + 1,) * 2
    assert state.backend == "structured"
    assert "broker" in state.support_totals
    support = state.support_totals["broker"]
    assert int(np.sum(support.count)) == len(X)
    assert support.information.shape == (48,)

    predictions = structured.predict(X.head(8), offset=np.log(exposure[:8]))
    assert np.all(np.isfinite(predictions))
    assert structured.summary()["fit"]["n_obs"] == len(X)


def test_structured_state_has_no_dominant_square_array():
    _, structured, _, _, _ = _fit_pair(
        n_levels=270,
        fit_dense=False,
        max_reml_iter=2,
    )
    state = structured._linear_system_state
    assert isinstance(state, StructuredLinearSystemState)
    dominant_size = len(state.system.operator.structured_indices)

    arrays = [
        value
        for owner in (
            state,
            state.system,
            state.system.operator,
            state.coefficient_factor,
            state.augmented_factor,
        )
        for value in vars(owner).values()
        if isinstance(value, np.ndarray)
    ]
    assert all(array.shape != (dominant_size, dominant_size) for array in arrays)


def test_random_effect_report_matches_dense_and_poisson_actual_expected():
    dense, structured, X, y, exposure = _fit_pair()
    assert dense is not None

    dense_report = dense.random_effects("broker", exposure=exposure)
    report = structured.random_effects("broker", exposure=exposure)

    assert isinstance(report, RandomEffectResult)
    assert report.name == "broker"
    assert report.lambda_value == structured._reml_lambdas["broker"]
    assert report.tau_squared == pytest.approx(report.phi / report.lambda_value)
    assert report.variance_component == report.tau_squared
    assert report.standard_deviation == pytest.approx(np.sqrt(report.tau_squared))
    assert report.effective_df == pytest.approx(structured._group_edf["broker"])
    assert list(report.table.columns) == [
        "level",
        "count",
        "fit_weight",
        "exposure",
        "unpooled_effect",
        "effect",
        "relativity",
        "posterior_se",
        "credibility",
        "shrinkage",
        "finite",
        "has_information",
        "collapsed",
    ]

    spec = structured._specs["broker"]
    codes = spec._prediction_codes(X["broker"].to_numpy())
    population_mean = structured.predict(
        X,
        offset=np.log(exposure),
        random_effects="population",
    )
    actual = np.bincount(codes, weights=y, minlength=len(spec._levels))
    expected = np.bincount(
        codes,
        weights=population_mean,
        minlength=len(spec._levels),
    )
    np.testing.assert_allclose(
        report.table["unpooled_effect"],
        np.log(actual / expected),
        rtol=3e-10,
        atol=3e-10,
    )
    information = structured._linear_system_state.support_totals["broker"].information
    np.testing.assert_allclose(
        report.table["credibility"],
        information / (information + report.lambda_value),
        rtol=2e-13,
        atol=2e-13,
    )
    np.testing.assert_allclose(
        report.table["shrinkage"],
        1.0 - report.table["credibility"],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        report.table[["unpooled_effect", "effect", "posterior_se", "credibility"]].to_numpy(),
        dense_report.table[["unpooled_effect", "effect", "posterior_se", "credibility"]].to_numpy(),
        rtol=4e-8,
        atol=4e-9,
    )

    local_se = np.sqrt(
        report.phi / (information + report.lambda_value),
    )
    assert np.max(np.abs(report.table["posterior_se"].to_numpy() - local_se)) > 1e-6


def test_random_effect_report_never_infers_exposure_from_offset():
    _, structured, X, y, exposure = _fit_pair(fit_dense=False)

    without_exposure = structured.random_effects("broker")
    assert without_exposure.table["exposure"].isna().all()

    with_exposure = structured.random_effects("broker", exposure=exposure)
    assert np.isclose(with_exposure.table["exposure"].sum(), exposure.sum())

    with pytest.raises(ValueError, match="exposure"):
        structured.random_effects("broker", exposure=exposure[:-1])
    with pytest.raises(ValueError, match="X and y"):
        structured.random_effects(
            "broker",
            X=X,
            y=None,
            offset=np.zeros(len(y)),
        )


def test_random_effect_report_aggregates_fit_weight_and_explicit_exposure():
    rng = np.random.default_rng(581)
    codes = np.repeat(np.arange(4), 30)
    sample_weight = rng.uniform(0.25, 2.0, size=len(codes))
    exposure = rng.uniform(0.4, 1.6, size=len(codes))
    truth = np.array([-0.3, 0.1, 0.25, -0.05])
    y = rng.poisson(exposure * np.exp(-0.2 + truth[codes])).astype(float)
    X = pd.DataFrame({"broker": np.array([f"b{i}" for i in codes], dtype=object)})
    model = SuperGLM(
        family="poisson",
        features={"broker": RandomEffect()},
        selection_penalty=0,
        direct_solve="structured",
    )
    model.fit_reml(
        X,
        y,
        sample_weight=sample_weight,
        offset=np.log(exposure),
        max_reml_iter=5,
    )

    report = model.random_effects("broker", exposure=exposure)

    np.testing.assert_allclose(
        report.table["fit_weight"],
        np.bincount(codes, weights=sample_weight),
    )
    np.testing.assert_allclose(
        report.table["exposure"],
        np.bincount(codes, weights=exposure),
    )


def test_released_random_effect_report_uses_precomputed_unpooled_effects():
    _, structured, _, _, _ = _fit_pair(
        retain_fit_state=False,
        n_levels=32,
        fit_dense=False,
    )

    report = structured.random_effects("broker")

    assert np.all(np.isfinite(report.table["unpooled_effect"]))
    assert np.all(np.isfinite(report.table["posterior_se"]))
    assert report.table["exposure"].isna().all()


@pytest.mark.parametrize(
    ("distribution", "link", "response"),
    [
        pytest.param(
            Gaussian(),
            IdentityLink(),
            lambda rng, eta: eta + rng.normal(scale=0.35, size=len(eta)),
            id="gaussian",
        ),
        pytest.param(
            Binomial(),
            LogitLink(),
            lambda rng, eta: rng.binomial(1, 1.0 / (1.0 + np.exp(-eta))).astype(float),
            id="binomial",
        ),
        pytest.param(
            Gamma(),
            LogLink(),
            lambda rng, eta: rng.gamma(shape=4.0, scale=np.exp(eta) / 4.0),
            id="gamma",
        ),
    ],
)
def test_vectorized_unpooled_fisher_matches_scalar_score_roots(
    distribution,
    link,
    response,
):
    rng = np.random.default_rng(4008)
    n_levels = 3
    codes = np.repeat(np.arange(n_levels), 80)
    base_eta = rng.normal(scale=0.25, size=len(codes))
    truth = np.array([-0.4, 0.15, 0.55])
    y = response(rng, base_eta + truth[codes])
    # Guarantee finite logit roots without changing the vectorized contract.
    if isinstance(distribution, Binomial):
        for level in range(n_levels):
            rows = np.flatnonzero(codes == level)
            y[rows[0]] = 0.0
            y[rows[1]] = 1.0
    weights = rng.uniform(0.4, 1.8, size=len(codes))

    actual = vectorized_conditional_unpooled_effect(
        codes=codes,
        n_levels=n_levels,
        y=y,
        sample_weight=weights,
        base_eta=base_eta,
        distribution=distribution,
        link=link,
    )

    expected = np.empty(n_levels)
    for level in range(n_levels):
        rows = codes == level

        def score(effect):
            eta = stabilize_eta(base_eta[rows] + effect, link)
            mu = clip_mu(link.inverse(eta), distribution)
            variance = np.maximum(distribution.variance(mu), _VARIANCE_FLOOR)
            derivative = link.deriv_inverse(eta)
            return float(np.sum(weights[rows] * (y[rows] - mu) * derivative / variance))

        expected[level] = scipy.optimize.brentq(score, -30.0, 30.0)

    np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=2e-9)


def test_random_effect_upper_boundary_is_reported_as_collapsed():
    X = pd.DataFrame({"broker": ["a"] * 24 + ["b"] * 24})
    y = np.concatenate((np.ones(24), np.full(24, 2.0)))
    model = SuperGLM(
        family="gaussian",
        features={
            "broker": RandomEffect(
                lambda_policy=LambdaPolicy.fixed(1.0e10),
            )
        },
        selection_penalty=0,
        direct_solve="structured",
    )
    model.fit_reml(X, y, max_reml_iter=2)

    with pytest.warns(UserWarning, match="collapsed"):
        report = model.random_effects("broker")

    assert report.collapsed
    assert report.at_upper_boundary
    assert not report.at_lower_boundary
    assert report.table["collapsed"].all()
    assert report.table["relativity"].isna().all()
