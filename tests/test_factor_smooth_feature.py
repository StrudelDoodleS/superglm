"""Public contract tests for mgcv-style factor smooth interactions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import superglm.features.factor_smooth as factor_smooth_module
from superglm import FactorSmooth, LambdaPolicy, Numeric, RandomEffect, SuperGLM
from superglm.features.spline import PSpline


def _built_discrete_spec(*, basis="fs", m=2, lambda_policy=None):
    x = np.linspace(-2.0, 2.0, 5000)
    group = np.array([f"g-{index % 20}" for index in range(len(x))], dtype=object)
    spec = FactorSmooth(
        "x",
        group="group",
        basis=basis,
        k=max(6, m + 4),
        m=m,
        lambda_policy=lambda_policy,
    )
    info = spec.build_discrete(x, group, {}, 256)
    return spec, info


def _legacy_natural_parameterization(basis, penalty, *, rank):
    import scipy.linalg as la

    X = np.asarray(basis, dtype=np.float64)
    _Q, R = np.linalg.qr(X, mode="reduced")
    R_inv = la.solve_triangular(R, np.eye(R.shape[0]), lower=False)
    transformed = R_inv.T @ penalty @ R_inv
    eigenvalues, eigenvectors = la.eigh(
        0.5 * (transformed + transformed.T),
        driver="evr",
    )
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    natural_map = R_inv @ eigenvectors
    natural_map[:, :rank] /= np.sqrt(eigenvalues[:rank])
    natural_basis = X @ natural_map
    penalized_scale = 1.0 / np.sqrt(np.mean(natural_basis[:, :rank] ** 2))
    natural_map[:, :rank] *= penalized_scale
    null_dim = X.shape[1] - rank
    if null_dim:
        null_scale = 1.0 / np.sqrt(np.mean(natural_basis[:, rank:] ** 2))
        natural_map[:, rank:] *= null_scale
    wiggle = np.zeros((X.shape[1], X.shape[1]))
    wiggle[np.arange(rank), np.arange(rank)] = penalized_scale**2
    components = [("wiggle", wiggle)]
    for null_index in range(null_dim):
        component = np.zeros_like(wiggle)
        coordinate = rank + null_index
        component[coordinate, coordinate] = 1.0
        components.append((f"null_{null_index}", component))
    return natural_map, tuple(components)


def test_default_fs_and_sz_use_streamed_marginal_qr():
    fs, fs_info = _built_discrete_spec()
    sz, sz_info = _built_discrete_spec(basis="sz")

    assert fs._marginal_build_backend == "streamed_tsqr"
    assert sz._marginal_build_backend == "streamed_tsqr"
    assert fs_info.factor_smooth_basis is None
    assert fs_info.factor_smooth_basis_unique.shape == (256, fs.k)
    assert sz_info.factor_smooth_basis_unique.shape == (256, sz.k)


def test_streamed_discrete_marginal_bounds_basis_evaluation(monkeypatch):
    chunk_rows = 64
    monkeypatch.setattr(
        factor_smooth_module,
        "_MARGINAL_QR_CHUNK_ROWS",
        chunk_rows,
    )
    original_basis = PSpline._basis_matrix
    original_raw_basis = PSpline._raw_basis_matrix
    basis_rows = []
    raw_basis_rows = []

    def bounded_basis(self, values):
        basis_rows.append(len(values))
        return original_basis(self, values)

    def bounded_raw_basis(self, values):
        raw_basis_rows.append(len(values))
        return original_raw_basis(self, values)

    monkeypatch.setattr(PSpline, "_basis_matrix", bounded_basis)
    monkeypatch.setattr(PSpline, "_raw_basis_matrix", bounded_raw_basis)

    spec, info = _built_discrete_spec()

    assert spec._marginal_build_backend == "streamed_tsqr"
    assert len(basis_rows) > 1
    assert max(basis_rows) <= chunk_rows
    assert raw_basis_rows == [256]
    assert info.factor_smooth_basis is None


def test_streamed_tsqr_matches_legacy_geometry_up_to_null_permutation(monkeypatch):
    monkeypatch.setattr(
        factor_smooth_module,
        "_MARGINAL_QR_CHUNK_ROWS",
        64,
    )
    x = np.linspace(-2.0, 2.0, 5000)
    group = np.arange(len(x), dtype=np.intp) % 20
    spec = FactorSmooth("x", group="group", k=6)
    spec.build_discrete(x, group, {}, 256)
    raw = np.asarray(spec._spline._raw_basis_matrix(x), dtype=np.float64)
    penalty = np.asarray(spec._spline._build_penalty(), dtype=np.float64)
    legacy_map, _components = _legacy_natural_parameterization(
        raw,
        penalty,
        rank=spec.k - spec.m,
    )
    streamed_basis = raw @ spec._natural_map
    legacy_basis = raw @ legacy_map
    rank = spec.k - spec.m

    streamed_penalized = streamed_basis[:, :rank]
    streamed_penalized /= np.linalg.norm(streamed_penalized, axis=0)
    legacy_penalized = legacy_basis[:, :rank]
    legacy_penalized /= np.linalg.norm(legacy_penalized, axis=0)
    np.testing.assert_allclose(
        np.abs(streamed_penalized.T @ legacy_penalized),
        np.eye(rank),
        atol=2e-11,
    )

    streamed_null, _ = np.linalg.qr(streamed_basis[:, rank:], mode="reduced")
    legacy_null, _ = np.linalg.qr(legacy_basis[:, rank:], mode="reduced")
    null_alignment = np.abs(streamed_null.T @ legacy_null)
    np.testing.assert_allclose(null_alignment.max(axis=0), 1.0, atol=2e-11)
    np.testing.assert_allclose(null_alignment.max(axis=1), 1.0, atol=2e-11)
    np.testing.assert_allclose(null_alignment.min(axis=0), 0.0, atol=2e-11)
    np.testing.assert_allclose(null_alignment.min(axis=1), 0.0, atol=2e-11)


def test_asymmetric_or_high_order_fs_uses_dense_compatibility_qr():
    asymmetric = {
        "wiggle": LambdaPolicy.fixed(1.0),
        "null_0": LambdaPolicy.fixed(0.7),
        "null_1": LambdaPolicy.fixed(1.3),
    }

    custom, _ = _built_discrete_spec(lambda_policy=asymmetric)
    high_order, _ = _built_discrete_spec(m=3)

    assert custom._marginal_build_backend == "dense_qr_compat"
    assert high_order._marginal_build_backend == "dense_qr_compat"


def test_dense_qr_compat_matches_legacy_transform_and_penalties():
    x = np.linspace(-2.0, 2.0, 5000)
    group = np.arange(len(x), dtype=np.intp) % 20
    policies = {
        "wiggle": LambdaPolicy.fixed(1.0),
        "null_0": LambdaPolicy.fixed(0.7),
        "null_1": LambdaPolicy.fixed(1.3),
    }
    spec = FactorSmooth(
        "x",
        group="group",
        k=6,
        m=2,
        lambda_policy=policies,
    )
    spec.build_discrete(x, group, {}, 256)
    raw = np.asarray(spec._spline._raw_basis_matrix(x), dtype=np.float64)
    penalty = np.asarray(spec._spline._build_penalty(), dtype=np.float64)
    expected_map, expected_components = _legacy_natural_parameterization(
        raw,
        penalty,
        rank=spec.k - spec.m,
    )

    np.testing.assert_allclose(spec._natural_map, expected_map, atol=2e-12)
    assert [name for name, _ in spec._base_penalty_components] == [
        name for name, _ in expected_components
    ]
    for (_, actual), (_, expected) in zip(
        spec._base_penalty_components,
        expected_components,
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, atol=2e-12)


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


# ── Bound level universes (spec 2026-08-11, §3.1/§3.2) ──────────


def _two_level_columns(n: int = 240) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-2.0, 2.0, n)
    group = np.array(["a", "b"] * (n // 2), dtype=object)
    return x, group


def test_declared_universe_keeps_a_block_for_an_unobserved_level() -> None:
    x, group = _two_level_columns()
    spec = FactorSmooth("x", group="group", levels=["a", "b", "ghost"])

    info = spec.build(x, group, {})

    assert spec._levels == ["a", "b", "ghost"]
    assert spec._level_to_code == {"a": 0, "b": 1, "ghost": 2}
    assert spec._level_source == "declared"
    assert info.factor_smooth_n_levels == 3
    assert info.factor_smooth_levels == ("a", "b", "ghost")
    assert info.n_cols == 3 * spec.k


def test_declared_universe_rejects_group_values_outside_it() -> None:
    x, group = _two_level_columns()
    spec = FactorSmooth("x", group="group", levels=["a", "b"])
    rogue = group.copy()
    rogue[0] = "ROGUE"

    with pytest.raises(ValueError, match="outside the declared level universe"):
        spec.build(x, rogue, {})


def test_declared_universe_still_rejects_missing_group_values() -> None:
    x, group = _two_level_columns()
    spec = FactorSmooth("x", group="group", levels=["a", "b"])
    broken = group.copy()
    broken[0] = None

    with pytest.raises(ValueError, match="missing values"):
        spec.build(x, broken, {})


def test_score_on_a_declared_unobserved_level_uses_its_own_block() -> None:
    x, group = _two_level_columns()
    spec = FactorSmooth("x", group="group", levels=["a", "b", "ghost"])
    spec.build(x, group, {})
    beta = np.zeros(3 * spec.k, dtype=np.float64)
    beta[2 * spec.k :] = 1.0

    scored = spec.score(
        np.array([0.5]),
        np.array(["ghost"], dtype=object),
        beta,
    )

    expected = spec.marginal_basis(np.array([0.5])).sum(axis=1)
    np.testing.assert_allclose(scored, expected)
    assert scored[0] != 0.0


@pytest.mark.parametrize("declared", [["a", "b", "ghost"], ["ghost", "a", "b"]])
def test_sz_basis_rejects_a_declared_level_without_rows(declared) -> None:
    # Measured, not assumed: one empty level takes the penalized system's
    # smallest eigenvalue to 4e-10 and max|beta| to 4.9e3 under sz, wherever
    # the empty level sits in the contrast.
    x, group = _two_level_columns()
    spec = FactorSmooth("x", group="group", basis="sz", levels=declared)

    with pytest.raises(ValueError, match=r"basis='sz'.*no training rows"):
        spec.build(x, group, {})


def _three_level_columns(n: int = 240) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-2.0, 2.0, n)
    group = np.array(["a", "b", "ghost"] * (n // 3), dtype=object)
    return x, group


@pytest.mark.parametrize("build", ["build", "build_discrete"])
def test_sz_basis_rejects_a_level_present_only_on_zero_weight_rows(build) -> None:
    # A physical `bincount` counts a zero-weight row as occupancy, so the level
    # walks past the guard and rebuilds exactly the near-singularity the guard
    # exists to refuse: a row with weight 0 contributes nothing to the weighted
    # system, which is the same thing as having no row at all. Both build paths,
    # because both were discarding the weight.
    x, group = _three_level_columns()
    weight = np.where(group == "ghost", 0.0, 1.0)
    spec = FactorSmooth("x", group="group", basis="sz", levels=["a", "b", "ghost"])

    with pytest.raises(ValueError, match=r"basis='sz'.*ghost"):
        if build == "build":
            spec.build(x, group, {}, sample_weight=weight)
        else:
            spec.build_discrete(x, group, {}, 64, sample_weight=weight)


def test_sz_basis_still_accepts_a_level_carrying_positive_weight() -> None:
    # The guard reads EFFECTIVE occupancy, so a level whose rows are merely
    # light is not empty; refusing it would make the fix a different bug.
    x, group = _three_level_columns()
    weight = np.where(group == "ghost", 1e-3, 1.0)
    spec = FactorSmooth("x", group="group", basis="sz", levels=["a", "b", "ghost"])

    info = spec.build(x, group, {}, sample_weight=weight)

    assert info.factor_smooth_n_levels == 3


def test_sz_basis_keeps_physical_counts_when_no_weight_is_given() -> None:
    x, group = _three_level_columns()
    spec = FactorSmooth("x", group="group", basis="sz", levels=["a", "b", "ghost"])

    info = spec.build(x, group, {})

    assert info.factor_smooth_n_levels == 3


def test_fs_basis_tolerates_a_zero_weight_level() -> None:
    # The refusal is sz-specific: under fs every coordinate carries a penalty,
    # so an empty block sits at its own lambda and the observed levels do not
    # move. Widening the guard to fs would remove a supported configuration.
    x, group = _three_level_columns()
    weight = np.where(group == "ghost", 0.0, 1.0)
    spec = FactorSmooth("x", group="group", basis="fs", levels=["a", "b", "ghost"])

    info = spec.build(x, group, {}, sample_weight=weight)

    assert info.factor_smooth_n_levels == 3


def test_sz_basis_binds_a_fully_observed_declared_universe() -> None:
    x, group = _two_level_columns()
    spec = FactorSmooth("x", group="group", basis="sz", levels=["b", "a"])

    info = spec.build(x, group, {})

    assert spec._levels == ["b", "a"]
    assert info.factor_smooth_n_levels == 2
    assert info.n_cols == spec.k
    assert info.factor_smooth_codes[:2].tolist() == [1, 0]


def test_dtype_and_full_frame_sources_bind_the_group_universe() -> None:
    from superglm.types import LevelBinding

    x, group = _two_level_columns()
    dtype_bound = FactorSmooth("x", group="group")
    dtype_bound.adopt_dtype_categories(["b", "a", "ghost"])
    dtype_bound.build(x, group, {})

    frame_bound = FactorSmooth("x", group="group")
    frame_bound.apply_level_binding(LevelBinding(levels=("a", "b", "ghost")))
    frame_bound.build(x, group, {})

    declared = FactorSmooth("x", group="group", levels=["a", "b"])
    declared.apply_level_binding(LevelBinding(levels=("a", "b", "ghost")))

    assert dtype_bound._levels == ["b", "a", "ghost"]
    assert dtype_bound._level_source == "dtype"
    assert frame_bound._levels == ["a", "b", "ghost"]
    assert frame_bound._level_source == "full-frame"
    assert declared._declared_levels == ["a", "b"]
    assert declared._level_source == "declared"


def test_resolve_binding_reads_the_group_column_without_mutating_the_spec() -> None:
    spec = FactorSmooth("x", group="group")

    binding = spec.resolve_binding(np.array(["b", "a", "b"], dtype=object))

    assert binding.levels == ("a", "b")
    assert binding.base is None
    assert spec._levels == []


def test_reml_fit_shrinks_a_declared_unobserved_level_to_the_population() -> None:
    x_level = np.linspace(-1.0, 1.0, 24)
    levels = np.array(["a", "b"], dtype=object)
    x = np.tile(x_level, len(levels))
    group = np.repeat(levels, len(x_level))
    curves = {
        "a": 0.8 * np.sin(2.2 * x_level) + 0.2 * x_level,
        "b": -0.4 * np.cos(1.7 * x_level) - 0.3 * x_level,
    }
    y = 1.1 + np.concatenate([curves[level] for level in levels])
    X = pd.DataFrame({"x": x, "group": group})
    model = SuperGLM(
        family="gaussian",
        interactions=[
            FactorSmooth(
                "x",
                group="group",
                k=6,
                levels=["a", "b", "ghost"],
                unseen="error",
                lambda_policy=LambdaPolicy.fixed(1.4),
            )
        ],
        selection_penalty=0.0,
        direct_solve="gram",
    )

    model.fit_reml(X, y, max_reml_iter=2, runtime_validation="skip")

    assert len(model.result.beta) == 3 * 6
    assert np.all(np.isfinite(model.result.beta))
    X_ghost = pd.DataFrame({"x": [0.25], "group": ["ghost"]})
    conditional = model.predict(X_ghost, random_effects="conditional")
    population = model.predict(X_ghost, random_effects="population")
    np.testing.assert_allclose(conditional, population)
    X_known = pd.DataFrame({"x": [0.25], "group": ["a"]})
    assert not np.allclose(
        model.predict(X_known, random_effects="conditional"),
        model.predict(X_known, random_effects="population"),
    )
