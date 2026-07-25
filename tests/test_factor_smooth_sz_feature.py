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


def test_sum_to_zero_geometry_matches_explicit_contrast() -> None:
    from superglm.factor_smooth_geometry import (
        adjoint_sum_to_zero_blocks,
        expand_sum_to_zero_blocks,
        sum_to_zero_contrast,
        sum_to_zero_penalty,
    )

    contrast = sum_to_zero_contrast(4)
    expected_contrast = np.vstack((np.eye(3), -np.ones((1, 3))))
    np.testing.assert_array_equal(contrast, expected_contrast)

    free = np.arange(18, dtype=np.float64).reshape(3, 6) / 10.0
    raw = expand_sum_to_zero_blocks(free)
    np.testing.assert_allclose(raw, contrast @ free, atol=0.0)
    np.testing.assert_allclose(raw.sum(axis=0), 0.0, atol=0.0)

    raw_probe = np.arange(24, dtype=np.float64).reshape(4, 6) / 7.0
    np.testing.assert_allclose(
        adjoint_sum_to_zero_blocks(raw_probe),
        contrast.T @ raw_probe,
        atol=0.0,
    )

    local = np.diag(np.arange(1.0, 7.0))
    np.testing.assert_allclose(
        sum_to_zero_penalty(local, 4),
        np.kron(contrast.T @ contrast, local),
        atol=0.0,
    )


def test_sum_to_zero_geometry_rejects_invalid_shapes() -> None:
    from superglm.factor_smooth_geometry import (
        adjoint_sum_to_zero_blocks,
        expand_sum_to_zero_blocks,
        sum_to_zero_contrast,
        sum_to_zero_penalty,
    )

    with pytest.raises(ValueError, match="at least two"):
        sum_to_zero_contrast(1)
    with pytest.raises(TypeError, match="integer"):
        sum_to_zero_contrast(True)
    with pytest.raises(ValueError, match=r"shape \(K-1, k"):
        expand_sum_to_zero_blocks(np.ones(3))
    with pytest.raises(ValueError, match=r"shape \(K, k"):
        adjoint_sum_to_zero_blocks(np.ones((1, 3)))
    with pytest.raises(ValueError, match="square"):
        sum_to_zero_penalty(np.ones((2, 3)), 3)


def test_sz_build_has_k_minus_one_blocks_and_one_wiggle_component() -> None:
    x = np.tile(np.linspace(-1.0, 1.0, 12), 3)
    group = np.repeat(["a", "b", "c"], 12)
    spec = FactorSmooth("x", group="g", basis="sz", k=6, m=2)

    info = spec.build(x, group, {})

    assert info.n_cols == 12
    assert info.factor_smooth_factor_basis == "sz"
    assert info.factor_smooth_n_levels == 3
    np.testing.assert_array_equal(info.factor_smooth_transform, np.eye(6))
    assert info.repeated_penalty_components is not None
    assert [name for name, _ in info.repeated_penalty_components] == ["wiggle"]
    assert info.repeated_penalty_components[0][1].shape == (6, 6)


def test_sz_transform_and_score_sum_to_zero_over_levels() -> None:
    x = np.tile(np.linspace(-1.0, 1.0, 12), 3)
    group = np.repeat(["a", "b", "c"], 12)
    spec = FactorSmooth("x", group="g", basis="sz", k=6)
    spec.build(x, group, {})
    beta = np.linspace(-0.4, 0.7, 12)
    grid = np.linspace(-0.7, 0.7, 11)

    level_scores = np.stack(
        [spec.score(grid, np.repeat(level, len(grid)), beta) for level in spec._levels]
    )
    np.testing.assert_allclose(level_scores.sum(axis=0), 0.0, atol=1e-13)

    rows = np.array(["a", "b", "c", "unseen"])
    matrix = spec.transform(np.repeat(0.2, len(rows)), rows)
    assert matrix.shape == (4, 12)
    np.testing.assert_allclose(
        matrix @ beta,
        spec.score(np.repeat(0.2, len(rows)), rows, beta),
        atol=1e-13,
    )
    np.testing.assert_allclose(matrix[-1], 0.0, atol=0.0)
    np.testing.assert_allclose(matrix[2, :6], matrix[2, 6:], atol=0.0)


def test_sz_reconstruct_returns_all_levels_with_exact_zero_sum() -> None:
    x = np.tile(np.linspace(-1.0, 1.0, 12), 3)
    group = np.repeat(["a", "b", "c"], 12)
    spec = FactorSmooth("x", group="g", basis="sz", k=6, m=2)
    spec.build(x, group, {})
    beta = np.linspace(-0.4, 0.7, 12)

    reconstructed = spec.reconstruct(beta)
    blocks = np.stack(list(reconstructed["coefficients"].values()))

    assert reconstructed["basis"] == "sz"
    assert reconstructed["levels"] == ["a", "b", "c"]
    np.testing.assert_allclose(blocks.sum(axis=0), 0.0, atol=0.0)


def test_sz_requires_two_fitted_levels() -> None:
    with pytest.raises(ValueError, match="at least two"):
        FactorSmooth("x", group="g", basis="sz").build(
            np.linspace(0.0, 1.0, 20),
            np.repeat("only", 20),
            {},
        )


@pytest.mark.parametrize("basis", ["fs", "sz"])
def test_factor_smooth_reports_clear_marginal_rank_error(basis) -> None:
    with pytest.raises(ValueError, match=r"smaller k.*non-smooth"):
        FactorSmooth("x", group="g", basis=basis, k=6).build(
            np.resize([0.0, 0.5, 1.0], 30),
            np.repeat(["a", "b"], 15),
            {},
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
