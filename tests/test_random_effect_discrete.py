"""Discrete/fREML coverage for compact random-effect systems."""

import numpy as np
import pandas as pd
import pytest

import superglm.solvers._structured.moments as structured_moments
from superglm import RandomEffect, Spline, SuperGLM
from superglm._frame import as_eager_frame
from superglm.dm_builder import build_design_matrix, should_discretize
from superglm.group_matrix import (
    DenseGroupMatrix,
    RandomEffectGroupMatrix,
)
from superglm.model.reml_setup import collect_reml_groups
from superglm.reml.discrete import _solve_cached_profiled_system
from superglm.reml.penalty_algebra import build_penalty_context
from superglm.solvers.irls_direct import fit_irls_direct
from superglm.solvers.structured import (
    ScalarStructuredSystem,
    build_scalar_structured_system,
    solve_cached_scalar_structured,
)
from superglm.types import GroupSlice, PenaltyComponent


def _build_random_effect_design(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    discrete: bool,
):
    spec = RandomEffect()
    result = build_design_matrix(
        as_eager_frame(X),
        y,
        None,
        None,
        family="gaussian",
        link_spec=None,
        specs={"broker": spec},
        feature_order=["broker"],
        interaction_specs={},
        interaction_order=[],
        pending_interactions=[],
        model_discrete=discrete,
        n_bins_config=32,
        lambda2=0.1,
    )
    return result.dm.group_matrices[0]


def _cached_system_fixture(
    *,
    n_levels: int = 31,
) -> tuple[
    ScalarStructuredSystem,
    list,
    list[GroupSlice],
    list[PenaltyComponent],
    np.ndarray,
    np.ndarray,
    float,
    float,
]:
    rng = np.random.default_rng(177)
    n = n_levels * 7
    codes = rng.integers(0, n_levels, size=n, dtype=np.intp)
    numeric = rng.normal(size=(n, 3))
    matrices = [
        DenseGroupMatrix(numeric),
        RandomEffectGroupMatrix(codes, n_levels),
    ]
    groups = [
        GroupSlice(name="numeric", start=0, end=3, penalized=False),
        GroupSlice(
            name="broker",
            start=3,
            end=3 + n_levels,
            penalized=True,
        ),
    ]
    penalties = [
        PenaltyComponent(
            name="broker",
            group_name="broker",
            group_index=1,
            group_sl=groups[1].sl,
            omega_raw=None,
            penalty_kind="identity",
        )
    ]
    W = rng.uniform(0.4, 2.0, size=n)
    z = rng.normal(size=n)
    Wz = W * z
    system = build_scalar_structured_system(
        matrices,
        groups,
        W,
        Wz,
        dominant_group_index=1,
    )

    dense = np.column_stack((numeric, np.eye(n_levels)[codes]))
    xtw = dense.T @ W
    sum_w = float(np.sum(W))
    sum_wz = float(np.sum(Wz))
    mean_z = sum_wz / sum_w
    centered_gram = dense.T @ (W[:, None] * dense) - np.outer(xtw, xtw) / sum_w
    centered_rhs = dense.T @ Wz - xtw * mean_z
    return (
        system,
        matrices,
        groups,
        penalties,
        centered_gram,
        centered_rhs,
        mean_z,
        sum_w,
    )


def test_random_effect_codes_are_never_discretized():
    assert should_discretize(RandomEffect(), True) is False

    X = pd.DataFrame({"broker": np.array(["b", "a", "d", "b", "c", "a"], dtype=object)})
    y = np.arange(len(X), dtype=float)

    exact = _build_random_effect_design(X, y, discrete=False)
    discrete = _build_random_effect_design(X, y, discrete=True)

    assert isinstance(exact, RandomEffectGroupMatrix)
    assert isinstance(discrete, RandomEffectGroupMatrix)
    np.testing.assert_array_equal(discrete.codes, exact.codes)
    assert discrete.n_levels == exact.n_levels


def test_discrete_spline_is_exact_on_observed_support_with_random_effect():
    rng = np.random.default_rng(915)
    n = 270
    n_levels = 13
    codes = rng.integers(0, n_levels, size=n)
    x = np.resize(np.linspace(-2.0, 2.0, 9), n)
    rng.shuffle(x)
    exposure = rng.uniform(0.6, 1.7, size=n)
    truth = rng.normal(scale=0.24, size=n_levels)
    y = rng.poisson(exposure * np.exp(-0.2 + 0.3 * np.sin(x) + truth[codes])).astype(float)
    X = pd.DataFrame(
        {
            "x": x,
            "broker": np.array([f"b{i}" for i in codes], dtype=object),
        }
    )

    def build(discrete: bool):
        return build_design_matrix(
            as_eager_frame(X),
            y,
            None,
            np.log(exposure),
            family="poisson",
            link_spec=None,
            specs={
                "x": Spline(n_knots=6, penalty="ssp"),
                "broker": RandomEffect(),
            },
            feature_order=["x", "broker"],
            interaction_specs={},
            interaction_order=[],
            pending_interactions=[],
            model_discrete=discrete,
            n_bins_config=32,
            lambda2={"x": 2.0, "broker": 3.0},
        )

    exact = build(False)
    discrete = build(True)
    exact_reml_groups = collect_reml_groups(exact.groups, list(exact.dm.group_matrices))
    discrete_reml_groups = collect_reml_groups(
        discrete.groups,
        list(discrete.dm.group_matrices),
    )
    exact_penalties, _, _ = build_penalty_context(
        list(exact.dm.group_matrices),
        exact_reml_groups,
    )
    discrete_penalties, _, _ = build_penalty_context(
        list(discrete.dm.group_matrices),
        discrete_reml_groups,
    )
    common = {
        "y": y,
        "weights": np.ones(n),
        "family": exact.distribution,
        "link": exact.link,
        "lambda2": {"x": 2.0, "broker": 3.0},
        "offset": np.log(exposure),
        "max_iter": 100,
        "tol": 1e-10,
        "direct_solve": "structured",
    }

    exact_result, _ = fit_irls_direct(
        X=exact.dm,
        groups=exact.groups,
        reml_penalties=exact_penalties,
        **common,
    )
    discrete_result, _ = fit_irls_direct(
        X=discrete.dm,
        groups=discrete.groups,
        reml_penalties=discrete_penalties,
        **common,
    )

    np.testing.assert_allclose(
        discrete.dm.toarray(),
        exact.dm.toarray(),
        rtol=0.0,
        atol=4e-15,
    )
    np.testing.assert_allclose(discrete_result.beta, exact_result.beta, rtol=2e-10, atol=2e-11)
    assert discrete_result.intercept == pytest.approx(
        exact_result.intercept,
        rel=2e-10,
        abs=2e-11,
    )


@pytest.mark.parametrize("lambda_value", [1e-4, 0.03, 0.7, 8.0, 1e4])
def test_cached_structured_lambda_solve_matches_dense_profiled_oracle(
    lambda_value: float,
):
    (
        system,
        matrices,
        groups,
        penalties,
        centered_gram,
        centered_rhs,
        mean_z,
        sum_w,
    ) = _cached_system_fixture()
    p = centered_gram.shape[0]
    penalty = np.zeros((p, p))
    penalty[groups[1].sl, groups[1].sl] = np.eye(groups[1].size) * lambda_value

    # The dense helper expects globally ordered means, whereas the structured
    # cache stores its two partitions separately.
    mean_x = np.empty(p)
    mean_x[system.operator.small_indices] = system.xtw_small / sum_w
    mean_x[system.operator.structured_indices] = system.xtw_structured / sum_w
    expected_beta, expected_intercept, expected_logdet, expected_rank = (
        _solve_cached_profiled_system(
            centered_gram,
            penalty,
            centered_rhs,
            mean_x,
            sum_w,
            mean_z,
        )
    )

    actual = solve_cached_scalar_structured(
        system,
        matrices,
        groups,
        {"broker": lambda_value},
        reml_penalties=penalties,
    )

    np.testing.assert_allclose(actual.beta, expected_beta, rtol=2e-11, atol=2e-11)
    assert actual.intercept == pytest.approx(expected_intercept, rel=2e-11, abs=2e-11)
    assert actual.log_det_H == pytest.approx(expected_logdet, rel=2e-11, abs=2e-11)
    assert actual.hessian_rank == expected_rank


def test_cached_structured_trial_has_no_data_pass_or_dense_p_squared_state(monkeypatch):
    system, matrices, groups, penalties, *_ = _cached_system_fixture(n_levels=257)
    p = system.operator.shape[0]

    cached_arrays = (
        system.operator.A,
        system.operator.C,
        system.operator.d,
        system.xtw_small,
        system.xtw_structured,
        system.xtwz_small,
        system.xtwz_structured,
    )
    assert all(array.shape != (p, p) for array in cached_arrays)
    assert sum(array.nbytes for array in cached_arrays) < p * p * np.dtype(float).itemsize

    def fail_data_pass(*_args, **_kwargs):
        raise AssertionError("cached lambda solve revisited row-scale design data")

    monkeypatch.setattr(DenseGroupMatrix, "matvec", fail_data_pass)
    monkeypatch.setattr(DenseGroupMatrix, "rmatvec", fail_data_pass)
    monkeypatch.setattr(RandomEffectGroupMatrix, "matvec", fail_data_pass)
    monkeypatch.setattr(RandomEffectGroupMatrix, "rmatvec", fail_data_pass)
    monkeypatch.setattr(
        structured_moments,
        "_random_effect_cross_gram",
        fail_data_pass,
    )
    monkeypatch.setattr(
        structured_moments,
        "_random_effect_sufficient_stats",
        fail_data_pass,
    )

    solution = solve_cached_scalar_structured(
        system,
        matrices,
        groups,
        {"broker": 2.5},
        reml_penalties=penalties,
    )

    assert np.all(np.isfinite(solution.beta))
    assert np.isfinite(solution.intercept)


def test_discrete_structured_random_effect_matches_dense_freml():
    rng = np.random.default_rng(2219)
    n_levels = 16
    repeats = 24
    codes = np.repeat(np.arange(n_levels), repeats)
    effects = rng.normal(scale=0.36, size=n_levels)
    exposure = rng.uniform(0.45, 2.0, size=len(codes))
    y = rng.poisson(exposure * np.exp(-0.25 + effects[codes])).astype(float)
    X = pd.DataFrame({"broker": np.array([f"b{i}" for i in codes], dtype=object)})
    common = {
        "family": "poisson",
        "features": {"broker": RandomEffect()},
        "selection_penalty": 0,
        "discrete": True,
    }

    dense = SuperGLM(**common, direct_solve="gram")
    structured = SuperGLM(**common, direct_solve="structured")
    dense.fit_reml(X, y, offset=np.log(exposure), max_reml_iter=8)
    structured.fit_reml(X, y, offset=np.log(exposure), max_reml_iter=8)

    np.testing.assert_allclose(structured.result.beta, dense.result.beta, atol=3e-8)
    np.testing.assert_allclose(structured.result.intercept, dense.result.intercept, atol=3e-8)
    np.testing.assert_allclose(
        structured._reml_lambdas["broker"],
        dense._reml_lambdas["broker"],
        rtol=3e-7,
    )
    np.testing.assert_allclose(
        structured._reml_result.objective,
        dense._reml_result.objective,
        atol=3e-8,
    )
    assert structured._reml_profile["reml_n_structured_cache_solves"] > 0
    assert structured._reml_profile["reml_structured_cache_solve_s"] >= 0.0
    assert structured._reml_profile["reml_structured_cache_data_passes"] == 0


@pytest.mark.parametrize("companion", ["spline", "tensor"])
def test_discrete_structured_random_effect_with_smooth_companion_matches_dense(
    companion: str,
):
    rng = np.random.default_rng(514)
    n = 420
    n_levels = 18
    codes = rng.integers(0, n_levels, size=n)
    x1 = rng.uniform(-2.0, 2.0, size=n)
    x2 = rng.uniform(-1.5, 1.5, size=n)
    random_truth = rng.normal(scale=0.28, size=n_levels)
    eta = -0.35 + 0.22 * np.sin(1.7 * x1) - 0.16 * np.cos(1.3 * x2) + random_truth[codes]
    if companion == "tensor":
        eta += 0.12 * np.sin(x1 * x2)
    exposure = rng.uniform(0.6, 1.8, size=n)
    y = rng.poisson(exposure * np.exp(eta)).astype(float)
    X = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "broker": np.array([f"b{i}" for i in codes], dtype=object),
        }
    )
    features = {
        "x1": Spline(n_knots=6, penalty="ssp"),
        "broker": RandomEffect(),
    }
    interactions = None
    if companion == "tensor":
        features["x2"] = Spline(n_knots=5, penalty="ssp")
        interactions = [("x1", "x2")]

    common = {
        "family": "poisson",
        "features": features,
        "interactions": interactions,
        "selection_penalty": 0,
        "discrete": True,
        "n_bins": 32,
    }
    dense = SuperGLM(**common, direct_solve="gram")
    structured = SuperGLM(**common, direct_solve="structured")

    dense.fit_reml(
        X,
        y,
        offset=np.log(exposure),
        max_reml_iter=5,
        pirls_tol=1e-8,
    )
    structured.fit_reml(
        X,
        y,
        offset=np.log(exposure),
        max_reml_iter=5,
        pirls_tol=1e-8,
    )

    np.testing.assert_allclose(structured.result.beta, dense.result.beta, rtol=2e-7, atol=2e-8)
    assert structured.result.intercept == pytest.approx(
        dense.result.intercept,
        rel=2e-7,
        abs=2e-8,
    )
    np.testing.assert_allclose(
        structured.predict(X, offset=np.log(exposure)),
        dense.predict(X, offset=np.log(exposure)),
        rtol=2e-7,
        atol=2e-8,
    )
    for name, dense_lambda in dense._reml_lambdas.items():
        assert structured._reml_lambdas[name] == pytest.approx(dense_lambda, rel=2e-6)
    assert structured._reml_result.objective == pytest.approx(
        dense._reml_result.objective,
        rel=2e-8,
        abs=2e-8,
    )
