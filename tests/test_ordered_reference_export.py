"""Regression tests for ordered-spline reference-level reporting."""

import numpy as np
import pandas as pd

from superglm import OrderedCategorical, Spline, SuperGLM
from superglm.export.rating_tables import build_rating_table_payload


def _fit_two_ordered_splines():
    rng = np.random.default_rng(20260710)
    age_levels = ["18-25", "26-35", "36-45", "46-55", "56+"]
    score_levels = ["low", "medium", "high", "very_high"]
    combinations = pd.MultiIndex.from_product(
        [age_levels, score_levels], names=["age_band", "score_band"]
    ).to_frame(index=False)
    X = pd.concat([combinations] * 20, ignore_index=True)

    age_effect = dict(zip(age_levels, [-0.35, -0.1, 0.15, 0.4, 0.65]))
    score_effect = dict(zip(score_levels, [0.3, 0.05, -0.1, -0.25]))
    eta = (
        -0.7
        + X["age_band"].map(age_effect).to_numpy(dtype=np.float64)
        + X["score_band"].map(score_effect).to_numpy(dtype=np.float64)
    )
    y = rng.poisson(np.exp(eta)).astype(np.float64)

    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        spline_penalty=0.05,
        features={
            "age_band": OrderedCategorical(
                order=age_levels,
                basis=Spline(kind="ps", k=5),
                base="36-45",
            ),
            "score_band": OrderedCategorical(
                order=score_levels,
                basis=Spline(kind="ps", k=5),
                base="high",
            ),
        },
    )
    model.fit(X, y)
    return model, X, y, combinations


def _block_relativities(payload, feature: str) -> dict[str, float]:
    block = next(item for item in payload.main_effects if item.name == feature)
    return dict(zip(block.table[feature], block.table["Relativity"], strict=True))


def test_rating_payload_reference_factorization_matches_predictions():
    model, X, y, combinations = _fit_two_ordered_splines()

    payload = build_rating_table_payload(model, X, y)
    age_relativities = _block_relativities(payload, "age_band")
    score_relativities = _block_relativities(payload, "score_band")
    factored_predictions = np.array(
        [
            payload.base_relativity
            * age_relativities[row.age_band]
            * score_relativities[row.score_band]
            for row in combinations.itertuples(index=False)
        ]
    )

    np.testing.assert_allclose(
        factored_predictions,
        model.predict(combinations),
        rtol=1e-10,
        atol=1e-12,
    )


def test_ordered_base_effect_and_model_helper_use_fitted_reference_levels():
    model, _, _, _ = _fit_two_ordered_splines()
    from superglm.inference._ordered_reference import (
        ordered_reference_beta_contrast,
        ordered_reference_intercept,
    )

    expected_adjustment = 0.0
    for name in model._feature_order:
        spec = model._specs[name]
        groups = [group for group in model._groups if group.feature_name == name]
        beta = np.concatenate([model.result.beta[group.sl] for group in groups])
        base_value = np.array([spec._level_to_value[spec._base_level]])
        expected = float(spec._spline.score(base_value, beta)[0])

        assert np.isclose(spec._base_log_effect(beta), expected)
        expected_adjustment += expected

    adjusted = ordered_reference_intercept(
        model.result.intercept,
        model.result.beta,
        model._feature_order,
        model._specs,
        model._groups,
    )
    contrast = ordered_reference_beta_contrast(
        len(model.result.beta),
        model._feature_order,
        model._specs,
        model._groups,
    )
    assert np.isclose(adjusted, model.result.intercept + expected_adjustment)
    assert np.isclose(contrast @ model.result.beta, expected_adjustment)
