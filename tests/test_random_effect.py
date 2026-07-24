"""Public contract tests for all-level random effects."""

import warnings

import numpy as np
import pandas as pd
import pytest

import superglm
from superglm import LambdaPolicy, Numeric, RandomEffect, SuperGLM


def _fit_prediction_model(*, unseen="population"):
    X = pd.DataFrame(
        {
            "x": np.array([-1.0, 0.0, 1.0, 2.0, -0.5, 1.5]),
            "broker": np.array(["a", "a", "b", "b", "c", "c"], dtype=object),
        }
    )
    y = np.array([0.2, 0.8, 1.9, 2.8, -0.1, 1.1])
    model = SuperGLM(
        family="gaussian",
        features={
            "x": Numeric(),
            "broker": RandomEffect(
                unseen=unseen,
                lambda_policy=LambdaPolicy.fixed(1.5),
            ),
        },
        selection_penalty=0,
        direct_solve="gram",
    )
    model.fit_reml(X, y, max_reml_iter=2)
    return model, X


def test_random_effect_is_exported_from_top_level():
    assert hasattr(superglm, "RandomEffect")


@pytest.mark.parametrize("unseen", ["population", "error"])
def test_random_effect_accepts_supported_unseen_policies(unseen):
    policy = LambdaPolicy.fixed(2.5)

    spec = RandomEffect(unseen=unseen, missing="error", lambda_policy=policy)

    assert spec.unseen == unseen
    assert spec.missing == "error"
    assert spec._lambda_policy is policy
    assert spec._levels == []
    assert spec._level_to_code == {}


def test_random_effect_rejects_unknown_unseen_policy():
    with pytest.raises(ValueError, match="unseen"):
        RandomEffect(unseen="new_level")


def test_random_effect_rejects_unsupported_missing_policy():
    with pytest.raises(ValueError, match="missing"):
        RandomEffect(missing="level")


def test_random_effect_build_codes_every_fitted_level_without_a_base():
    policy = LambdaPolicy.fixed(3.0)
    spec = RandomEffect(lambda_policy=policy)

    info = spec.build(np.array(["b", "a", "b", "c"], dtype=object))

    assert info.n_cols == 3
    assert info.columns is None
    assert info.cat_codes.tolist() == [1, 0, 1, 2]
    assert info.structured_kind == "random_effect"
    assert info.lambda_policies == {"_default": policy}
    assert spec._levels == ["a", "b", "c"]
    assert spec._level_to_code == {"a": 0, "b": 1, "c": 2}


@pytest.mark.parametrize(
    "values",
    [
        np.array(["a", np.nan], dtype=object),
        np.array(["a", None], dtype=object),
    ],
)
def test_random_effect_build_rejects_missing_values(values):
    with pytest.raises(ValueError, match="missing values"):
        RandomEffect().build(values)


def test_random_effect_scores_known_levels_and_population_unknown_as_zero():
    spec = RandomEffect(unseen="population")
    spec.build(np.array(["b", "a", "c", "b"], dtype=object))
    beta = np.array([0.1, -0.2, 0.3])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scored = spec.score(np.array(["c", "new", "a", "b"], dtype=object), beta)

    np.testing.assert_allclose(scored, [0.3, 0.0, 0.1, -0.2])


def test_random_effect_error_policy_rejects_unknown_level():
    spec = RandomEffect(unseen="error")
    spec.build(np.array(["a", "b"], dtype=object))

    with pytest.raises(ValueError, match="unseen.*new"):
        spec.score(np.array(["a", "new"], dtype=object), np.array([0.1, 0.2]))


@pytest.mark.parametrize("unseen", ["population", "error"])
def test_random_effect_score_rejects_missing_before_unseen_policy(unseen):
    spec = RandomEffect(unseen=unseen)
    spec.build(np.array(["a", "b"], dtype=object))

    with pytest.raises(ValueError, match="missing values"):
        spec.score(np.array(["a", None], dtype=object), np.array([0.1, 0.2]))


def test_random_effect_validate_prediction_values_allows_unseen_but_not_missing():
    spec = RandomEffect(unseen="error")
    spec.build(np.array(["a", "b"], dtype=object))

    spec.validate_prediction_values(np.array(["new"], dtype=object))
    with pytest.raises(ValueError, match="missing values"):
        spec.validate_prediction_values(np.array([np.nan], dtype=object))


def test_random_effect_transform_and_reconstruct_keep_all_levels():
    spec = RandomEffect(unseen="population")
    spec.build(np.array(["b", "a", "c", "b"], dtype=object))
    beta = np.array([0.1, -0.2, 0.3])

    transformed = spec.transform(np.array(["c", "new", "a"], dtype=object))

    np.testing.assert_array_equal(
        transformed,
        np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        ),
    )
    assert spec.reconstruct(beta) == {
        "levels": ["a", "b", "c"],
        "effects": {"a": 0.1, "b": -0.2, "c": 0.3},
    }


def test_superglm_accepts_structured_direct_solver_mode():
    model = SuperGLM(
        features={"broker": RandomEffect()},
        selection_penalty=0,
        direct_solve="structured",
    )

    assert model._direct_solve == "structured"


@pytest.mark.parametrize("method", ["fit", "fit_path"])
def test_random_effect_rejects_selection_fit_paths(method):
    X = pd.DataFrame({"broker": np.array(["a", "b", "a", "c"], dtype=object)})
    y = np.array([1.0, 2.0, 1.5, 0.5])
    model = SuperGLM(
        family="gaussian",
        features={"broker": RandomEffect()},
        selection_penalty=0.1,
    )

    with pytest.raises(NotImplementedError, match="RandomEffect.*fit_reml"):
        if method == "fit":
            model.fit(X, y)
        else:
            model.fit_path(X, y, n_lambda=2)


def test_random_effect_prediction_defaults_to_conditional_and_supports_population_mode():
    model, X = _fit_prediction_model()

    default = model.predict(X)
    conditional = model.predict(X, random_effects="conditional")
    population = model.predict(X, random_effects="population")

    np.testing.assert_allclose(default, conditional, rtol=0.0, atol=0.0)
    expected_population = model.result.intercept + X["x"].to_numpy() * model.result.beta[0]
    np.testing.assert_allclose(population, expected_population)
    assert not np.allclose(conditional, population)


def test_random_effect_population_unseen_matches_population_prediction():
    model, _ = _fit_prediction_model(unseen="population")
    X_new = pd.DataFrame({"x": [0.25], "broker": ["new"]})

    conditional = model.predict(X_new, random_effects="conditional")
    population = model.predict(X_new, random_effects="population")

    np.testing.assert_allclose(conditional, population)


def test_random_effect_error_policy_applies_only_to_conditional_unseen_levels():
    model, _ = _fit_prediction_model(unseen="error")
    X_new = pd.DataFrame({"x": [0.25], "broker": ["new"]})

    with pytest.raises(ValueError, match="unseen"):
        model.predict(X_new, random_effects="conditional")
    population = model.predict(X_new, random_effects="population")

    assert np.isfinite(population[0])


@pytest.mark.parametrize("mode", ["conditional", "population"])
def test_random_effect_prediction_rejects_missing_values_in_every_mode(mode):
    model, _ = _fit_prediction_model()
    X_new = pd.DataFrame({"x": [0.25], "broker": [None]})

    with pytest.raises(ValueError, match="missing values"):
        model.predict(X_new, random_effects=mode)


def test_random_effect_prediction_rejects_unknown_mode():
    model, X = _fit_prediction_model()

    with pytest.raises(ValueError, match="random_effects"):
        model.predict(X, random_effects="marginal")
