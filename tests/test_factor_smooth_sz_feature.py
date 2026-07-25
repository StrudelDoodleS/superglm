"""Public contracts for unified FS and sum-to-zero factor smooths."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from superglm import (
    Categorical,
    FactorSmooth,
    LambdaPolicy,
    RandomEffect,
    Spline,
    SuperGLM,
)


def test_factor_smooth_basis_defaults_and_names() -> None:
    fs = FactorSmooth("age", group="region")
    sz = FactorSmooth("age", group="region", basis="sz")

    assert (fs.basis, fs.kind, fs.name) == ("fs", "ps", "age:region:fs")
    assert (sz.basis, sz.kind, sz.name) == ("sz", "ps", "age:region:sz")


def test_factor_smooth_rejects_invalid_basis() -> None:
    with pytest.raises(ValueError, match=r"basis must be 'fs' or 'sz'"):
        FactorSmooth("age", group="region", basis="reference")


def test_sz_rejects_fs_only_lambda_components() -> None:
    with pytest.raises(ValueError, match=r"null_0.*valid names.*wiggle"):
        FactorSmooth(
            "age",
            group="region",
            basis="sz",
            lambda_policy={"null_0": LambdaPolicy.fixed(1.0)},
        )


def test_clone_unfitted_preserves_sz_constructor_intent() -> None:
    model = SuperGLM(
        features={"age": Spline()},
        interactions=[
            FactorSmooth(
                "age",
                group="region",
                basis="sz",
                kind="ps",
                k=7,
                m=2,
                unseen="error",
                lambda_policy={"wiggle": LambdaPolicy.fixed(2.5)},
            )
        ],
    )

    cloned = model.clone_unfitted()
    original = model._interaction_specs["age:region:sz"]
    copied = cloned._interaction_specs["age:region:sz"]

    assert copied is not original
    assert copied.basis == "sz"
    assert copied.kind == "ps"
    assert copied.k == 7
    assert copied.m == 2
    assert copied.unseen == "error"
    assert copied._lambda_policy == {"wiggle": LambdaPolicy.fixed(2.5)}


def test_sz_requires_explicit_global_spline() -> None:
    with pytest.raises(ValueError, match=r"basis='sz'.*features=.*Spline"):
        SuperGLM(
            interactions=[FactorSmooth("age", group="region", basis="sz")],
        )

    model = SuperGLM(
        features={"age": Spline(kind="ps", k=7, m=2)},
        interactions=[FactorSmooth("age", group="region", basis="sz")],
    )

    assert model._interaction_order == ["age:region:sz"]


@pytest.mark.parametrize("group_spec", [Categorical(), RandomEffect()])
@pytest.mark.parametrize("basis", ["fs", "sz"])
def test_factor_smooth_rejects_duplicate_group_geometry(group_spec, basis) -> None:
    with pytest.raises(ValueError, match=r"region.*duplicates.*group-intercept"):
        SuperGLM(
            features={"age": Spline(), "region": group_spec},
            interactions=[FactorSmooth("age", group="region", basis=basis)],
        )


def test_factor_smooth_rejects_duplicate_pair_despite_custom_names() -> None:
    with pytest.raises(ValueError, match=r"\('age', 'region'\).*more than once"):
        SuperGLM(
            features={"age": Spline()},
            interactions=[
                FactorSmooth("age", group="region", name="first"),
                FactorSmooth("age", group="region", basis="sz", name="second"),
            ],
        )


def test_fs_remains_valid_with_or_without_global_spline() -> None:
    standalone = SuperGLM(
        interactions=[FactorSmooth("age", group="region")],
    )
    with_global = SuperGLM(
        features={"age": Spline()},
        interactions=[FactorSmooth("age", group="region")],
    )

    assert standalone._interaction_order == ["age:region:fs"]
    assert with_global._interaction_order == ["age:region:fs"]


@pytest.mark.parametrize("basis", ["fs", "sz"])
def test_auto_detection_cannot_bypass_group_geometry_validation(basis) -> None:
    model = SuperGLM(
        splines=["age"],
        interactions=[FactorSmooth("age", group="region", basis=basis)],
    )
    X = pd.DataFrame(
        {
            "age": np.linspace(18.0, 80.0, 20),
            "region": np.repeat(["north", "south"], 10),
        }
    )
    y = np.linspace(0.2, 1.4, len(X))

    with pytest.raises(ValueError, match=r"region.*duplicates.*group-intercept"):
        model.fit_reml(
            X,
            y,
            max_reml_iter=1,
            runtime_validation="skip",
        )
