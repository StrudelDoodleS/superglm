"""Public contract tests for mgcv-style factor smooth interactions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from superglm import FactorSmooth, LambdaPolicy, Numeric, RandomEffect, SuperGLM


def test_factor_smooth_constructor_contract() -> None:
    smooth = FactorSmooth("age", group="broker")

    assert smooth.variable == "age"
    assert smooth.group == "broker"
    assert smooth.parent_names == ("age", "broker")
    assert smooth.name == "age:broker:fs"
    assert smooth.kind == "ps"
    assert smooth.k == 6
    assert smooth.m == 2
    assert smooth.unseen == "population"
    assert smooth.missing == "error"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"kind": "cr"}, "kind='ps'"),
        ({"k": 4}, "k must be at least 5"),
        ({"m": 0}, "m must be"),
        ({"m": 6}, "m must be"),
        ({"unseen": "base"}, "unseen"),
        ({"missing": "population"}, "missing"),
        ({"name": ""}, "name"),
    ],
)
def test_factor_smooth_rejects_invalid_configuration(kwargs, message) -> None:
    with pytest.raises((TypeError, ValueError, NotImplementedError), match=message):
        FactorSmooth("age", group="broker", **kwargs)


def test_factor_smooth_rejects_unknown_lambda_component() -> None:
    with pytest.raises(ValueError, match="unknown component"):
        FactorSmooth(
            "age",
            group="broker",
            lambda_policy={"wiggle": LambdaPolicy.estimate(), "mystery": LambdaPolicy.off()},
        )


def test_constructor_accepts_explicit_factor_smooth_without_parent_main_effects() -> None:
    smooth = FactorSmooth("age", group="broker")
    model = SuperGLM(
        family="gaussian",
        features={"intercept_trend": Numeric()},
        interactions=[smooth],
        selection_penalty=0.0,
    )

    assert model._pending_interactions == ()
    assert model._interaction_order == ["age:broker:fs"]
    stored = model._interaction_specs["age:broker:fs"]
    assert isinstance(stored, FactorSmooth)
    assert stored is not smooth
    assert stored.parent_names == ("age", "broker")


def test_explicit_factor_smooth_name_cannot_collide_with_main_feature():
    with pytest.raises(ValueError, match="risk.*main feature.*interaction"):
        SuperGLM(
            family="gaussian",
            features={"risk": Numeric()},
            interactions=[FactorSmooth("age", group="broker", name="risk")],
        )


def test_clone_unfitted_owns_factor_smooth_configuration() -> None:
    model = SuperGLM(
        family="gaussian",
        interactions=[
            FactorSmooth(
                "age",
                group="broker",
                k=7,
                lambda_policy=LambdaPolicy.fixed(2.5),
            )
        ],
    )

    cloned = model.clone_unfitted()
    original = model._interaction_specs["age:broker:fs"]
    copied = cloned._interaction_specs["age:broker:fs"]

    assert isinstance(copied, FactorSmooth)
    assert copied is not original
    assert copied.k == 7
    copied._levels.append("mutated")
    assert original._levels == []


def test_clone_without_features_preserves_explicit_factor_smooth() -> None:
    model = SuperGLM(
        family="gaussian",
        features={"trend": Numeric()},
        interactions=[FactorSmooth("age", group="broker", k=7)],
    )

    cloned = model._clone_without_features(set())

    assert cloned._pending_interactions == ()
    assert cloned._interaction_order == ["age:broker:fs"]
    copied = cloned._interaction_specs["age:broker:fs"]
    assert isinstance(copied, FactorSmooth)
    assert copied.k == 7


def test_constructor_rejects_duplicate_explicit_interaction_names() -> None:
    with pytest.raises(ValueError, match="Interaction already added"):
        SuperGLM(
            interactions=[
                FactorSmooth("age", group="broker"),
                FactorSmooth("age", group="broker"),
            ]
        )


def test_factor_smooth_rejects_duplicate_random_intercept_geometry() -> None:
    with pytest.raises(ValueError, match="duplicates the constant null-space"):
        SuperGLM(
            features={"broker": RandomEffect()},
            interactions=[FactorSmooth("age", group="broker")],
        )


@pytest.mark.parametrize("method", ["fit", "fit_path"])
def test_selection_fit_paths_reject_factor_smooth(method: str) -> None:
    X = pd.DataFrame(
        {
            "age": np.linspace(18.0, 80.0, 20),
            "broker": np.repeat(["a", "b"], 10),
        }
    )
    y = np.linspace(0.0, 1.0, len(X))
    model = SuperGLM(
        family="gaussian",
        interactions=[FactorSmooth("age", group="broker")],
    )

    with pytest.raises(NotImplementedError, match=r"FactorSmooth.*fit_reml"):
        getattr(model, method)(X, y)


def test_required_column_validation_includes_both_factor_smooth_columns() -> None:
    model = SuperGLM(
        family="gaussian",
        interactions=[FactorSmooth("age", group="broker")],
    )
    X = pd.DataFrame({"age": np.linspace(18.0, 80.0, 20)})
    y = np.linspace(0.0, 1.0, len(X))

    with pytest.raises(ValueError, match="broker"):
        model.fit_reml(X, y)
