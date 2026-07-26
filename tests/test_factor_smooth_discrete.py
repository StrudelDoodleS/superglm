"""Discrete cached-fREML coverage for compact factor smooths."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import superglm._group_matrix._group_matrix_core as group_matrix_core
import superglm.reml.discrete as discrete_module
from superglm import FactorSmooth, LambdaPolicy, Numeric, RandomEffect, Spline, SuperGLM
from superglm.group_matrix import FactorSmoothGroupMatrix
from superglm.reml.penalty_algebra import build_penalty_matrix
from superglm.solvers.structured import (
    BlockStructuredSystem,
    CachedBlockStructuredSolution,
    materialize_compact_operator,
    solve_cached_structured,
)


def _data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1031)
    x_support = np.linspace(-1.0, 1.0, 60)
    x = np.tile(x_support, 8)
    n = len(x)
    segment_code = np.repeat(np.arange(8), len(x_support))
    permutation = rng.permutation(n)
    x = x[permutation]
    segment_code = segment_code[permutation]
    branch_code = rng.integers(0, 4, size=n)
    z = rng.normal(size=n)
    offset = np.log(rng.uniform(0.55, 1.9, size=n))
    weights = rng.uniform(0.45, 2.0, size=n)
    amplitudes = np.array([0.42, -0.3, 0.24, -0.38, 0.18, 0.33, -0.12, 0.27])
    eta = (
        -0.3
        + 0.17 * z
        + 0.19 * np.sin(2.2 * x)
        + amplitudes[segment_code] * (x + 0.3 * x**2)
        + np.array([0.1, -0.08, 0.04, 0.13])[branch_code]
        + offset
    )
    y = rng.poisson(np.exp(eta)).astype(np.float64)
    X = pd.DataFrame(
        {
            "x": x,
            "z": z,
            "segment": np.array([f"s-{code}" for code in segment_code], dtype=object),
            "branch": np.array([f"b-{code}" for code in branch_code], dtype=object),
        }
    )
    return X, y, weights, offset


def _model(*, discrete: bool, direct_solve: str) -> SuperGLM:
    return SuperGLM(
        family="poisson",
        features={
            "x": Spline(n_knots=5, lambda_policy=LambdaPolicy.fixed(1.5)),
            "z": Numeric(),
            "branch": RandomEffect(),
        },
        interactions=[FactorSmooth("x", group="segment", k=6)],
        selection_penalty=0.0,
        discrete=discrete,
        n_bins=256,
        direct_solve=direct_solve,
    )


def _fixed_policy_model(*, discrete: bool, lambda_policy) -> SuperGLM:
    return SuperGLM(
        family="poisson",
        features={},
        interactions=[
            FactorSmooth(
                "x",
                group="segment",
                k=6,
                lambda_policy=lambda_policy,
            )
        ],
        selection_penalty=0.0,
        discrete=discrete,
        n_bins=256,
        direct_solve="gram",
    )


def _fit(model: SuperGLM, X, y, weights, offset) -> SuperGLM:
    return model.fit_reml(
        X,
        y,
        sample_weight=weights,
        offset=offset,
        max_reml_iter=12,
        reml_tol=1e-5,
        pirls_tol=1e-9,
        runtime_validation="skip",
    )


def _training_eta(model: SuperGLM, offset: np.ndarray) -> np.ndarray:
    return model._dm.matvec(model.result.beta) + model.result.intercept + offset


def _cell_test_matrix(
    *,
    factor_basis: str,
    discrete: bool = True,
    n: int = 23,
) -> FactorSmoothGroupMatrix:
    support = np.array(
        [
            [1.0, 0.0, 0.2, 0.0, -0.1],
            [0.7, 0.3, 0.1, -0.2, 0.0],
            [0.2, 0.7, 0.0, 0.2, 0.1],
            [0.0, 0.5, 0.5, 0.0, 0.2],
            [0.0, 0.1, 0.6, 0.3, -0.1],
            [0.0, 0.0, 0.2, 0.8, 0.3],
        ],
        dtype=np.float64,
    )
    natural_map = np.array(
        [
            [1.0, 0.2, 0.0, -0.1],
            [0.1, 0.8, -0.2, 0.0],
            [0.0, 0.15, 1.1, 0.05],
            [-0.1, 0.0, 0.25, 0.9],
            [0.2, -0.1, 0.0, 0.3],
        ],
        dtype=np.float64,
    )
    base_codes = np.array(
        [0, 0, 1, 2, 3, 1, 2, 0, 3, 3, 1, 2, 0, 1, 2, 3, 0, 2, 1, 3, 2, 0, 1],
        dtype=np.intp,
    )
    base_bins = np.array(
        [0, 1, 1, 2, 3, 4, 0, 2, 5, 1, 3, 4, 4, 0, 5, 2, 1, 3, 5, 0, 1, 5, 2],
        dtype=np.intp,
    )
    codes = np.resize(base_codes, n)
    bin_idx = np.resize(base_bins, n)
    kwargs = {
        "codes": codes,
        "n_levels": 5,
        "natural_map": natural_map,
        "levels": ("alpha", "beta", "gamma", "delta", "empty"),
        "repeated_penalty_components": (
            ("wiggle", np.eye(natural_map.shape[1], dtype=np.float64)),
        ),
        "factor_basis": factor_basis,
    }
    if discrete:
        return FactorSmoothGroupMatrix(
            support,
            bin_idx=bin_idx,
            **kwargs,
        )
    return FactorSmoothGroupMatrix(
        sp.csr_matrix(support[bin_idx]),
        **kwargs,
    )


@pytest.mark.parametrize("factor_basis", ["fs", "sz"])
def test_discrete_cell_moments_match_explicit_row_level_reference(
    factor_basis: str,
) -> None:
    gm = _cell_test_matrix(factor_basis=factor_basis)
    row_count = gm.shape[0]
    weights = 0.17 + (np.arange(row_count, dtype=np.float64) * 0.37) % 1.83
    rhs = np.cos(np.arange(row_count, dtype=np.float64) * 0.61) - 0.23

    cell_weights, local_gram, xtw_nat, rhs_nat = gm.factor_smooth_discrete_cell_moments(
        weights, rhs
    )

    expected_cell_weights = np.zeros((gm.n_levels, gm.B_unique.shape[0]))
    for row in range(row_count):
        expected_cell_weights[gm.codes[row], gm.bin_idx[row]] += weights[row]
    natural_rows = gm.B_unique[gm.bin_idx] @ gm.natural_map
    expected_gram = np.zeros((gm.n_levels, gm.block_size, gm.block_size))
    expected_xtw = np.zeros((gm.n_levels, gm.block_size))
    expected_rhs = np.zeros((gm.n_levels, gm.block_size))
    for level in range(gm.n_levels):
        rows = gm.codes == level
        level_basis = natural_rows[rows]
        expected_gram[level] = level_basis.T @ (weights[rows, None] * level_basis)
        expected_xtw[level] = level_basis.T @ weights[rows]
        expected_rhs[level] = level_basis.T @ rhs[rows]

    assert cell_weights.shape == (gm.n_levels, gm.B_unique.shape[0])
    assert local_gram.shape == (gm.n_levels, gm.block_size, gm.block_size)
    assert xtw_nat.shape == (gm.n_levels, gm.block_size)
    assert rhs_nat.shape == (gm.n_levels, gm.block_size)
    assert all(result.flags.c_contiguous for result in (cell_weights, local_gram, xtw_nat, rhs_nat))
    assert all(
        result.dtype == np.float64 for result in (cell_weights, local_gram, xtw_nat, rhs_nat)
    )
    assert np.count_nonzero(expected_cell_weights == 0.0) > 0
    np.testing.assert_array_equal(cell_weights[-1], 0.0)
    np.testing.assert_array_equal(local_gram[-1], 0.0)
    np.testing.assert_array_equal(xtw_nat[-1], 0.0)
    np.testing.assert_array_equal(rhs_nat[-1], 0.0)
    np.testing.assert_array_equal(cell_weights, expected_cell_weights)
    np.testing.assert_allclose(local_gram, expected_gram, rtol=2.0e-12, atol=2.0e-12)
    np.testing.assert_allclose(xtw_nat, expected_xtw, rtol=2.0e-12, atol=2.0e-12)
    np.testing.assert_allclose(rhs_nat, expected_rhs, rtol=2.0e-12, atol=2.0e-12)
    np.testing.assert_array_equal(local_gram, local_gram.transpose(0, 2, 1))

    if factor_basis == "sz":
        assert gm.coefficient_levels == gm.n_levels - 1
        public_design = gm.toarray()
        public_gram, public_xtw, public_rhs = gm.gram_rmatvec(weights, rhs)
        np.testing.assert_allclose(
            public_gram,
            public_design.T @ (weights[:, None] * public_design),
            rtol=2.0e-12,
            atol=2.0e-12,
        )
        np.testing.assert_allclose(public_xtw, public_design.T @ weights, rtol=2.0e-12)
        np.testing.assert_allclose(public_rhs, public_design.T @ rhs, rtol=2.0e-12)


def test_discrete_cell_moments_do_not_use_raw_gram_einsum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gm = _cell_test_matrix(factor_basis="fs")
    weights = np.linspace(-0.4, 1.7, gm.shape[0])
    rhs = np.linspace(-1.2, 0.9, gm.shape[0])

    def forbidden_einsum(*_args, **_kwargs):
        raise AssertionError("discrete cell moments used raw-Gram einsum")

    monkeypatch.setattr(np, "einsum", forbidden_einsum)
    cell_weights, local_gram, xtw_nat, rhs_nat = gm.factor_smooth_discrete_cell_moments(
        weights,
        rhs,
    )

    assert cell_weights.shape == (gm.n_levels, gm.B_unique.shape[0])
    assert local_gram.shape == (gm.n_levels, gm.block_size, gm.block_size)
    assert xtw_nat.shape == rhs_nat.shape == (gm.n_levels, gm.block_size)


def test_discrete_sufficient_stats_delegate_to_cell_moments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gm = _cell_test_matrix(factor_basis="sz")
    weights = np.linspace(0.2, 1.9, gm.shape[0])
    rhs = np.linspace(-1.3, 0.8, gm.shape[0])
    cell_method = FactorSmoothGroupMatrix.factor_smooth_discrete_cell_moments
    expected = cell_method(gm, weights, rhs)[1:]
    calls = 0

    def counted(self, W, values):
        nonlocal calls
        calls += 1
        return cell_method(self, W, values)

    monkeypatch.setattr(
        FactorSmoothGroupMatrix,
        "factor_smooth_discrete_cell_moments",
        counted,
    )

    actual = gm.factor_smooth_sufficient_stats(weights, rhs)

    assert calls == 1
    for actual_moment, expected_moment in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(actual_moment, expected_moment)


def test_exact_sufficient_stats_remain_on_csr_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gm = _cell_test_matrix(factor_basis="fs", discrete=False)
    weights = np.linspace(0.3, 1.7, gm.shape[0])
    rhs = np.linspace(-0.8, 1.2, gm.shape[0])
    csr_kernel = group_matrix_core._factor_smooth_csr_sufficient_stats
    calls = 0

    def counted(*args):
        nonlocal calls
        calls += 1
        return csr_kernel(*args)

    def forbidden_cell_route(*_args, **_kwargs):
        raise AssertionError("exact FactorSmooth entered the discrete cell route")

    monkeypatch.setattr(
        group_matrix_core,
        "_factor_smooth_csr_sufficient_stats",
        counted,
    )
    monkeypatch.setattr(
        FactorSmoothGroupMatrix,
        "factor_smooth_discrete_cell_moments",
        forbidden_cell_route,
        raising=False,
    )

    local_gram, xtw_nat, rhs_nat = gm.factor_smooth_sufficient_stats(weights, rhs)

    natural_rows = np.asarray(gm.B @ gm.natural_map)
    for level in range(gm.n_levels):
        rows = gm.codes == level
        level_basis = natural_rows[rows]
        np.testing.assert_allclose(
            local_gram[level],
            level_basis.T @ (weights[rows, None] * level_basis),
        )
        np.testing.assert_allclose(xtw_nat[level], level_basis.T @ weights[rows])
        np.testing.assert_allclose(rhs_nat[level], level_basis.T @ rhs[rows])
    assert calls == 1


def test_discrete_cell_moments_validate_rows_and_discrete_geometry() -> None:
    gm = _cell_test_matrix(factor_basis="fs")
    weights = np.ones(gm.shape[0])
    rhs = np.ones(gm.shape[0])

    with pytest.raises(ValueError, match="weights and rhs must match"):
        gm.factor_smooth_discrete_cell_moments(weights[:-1], rhs)
    with pytest.raises(ValueError, match="weights and rhs must match"):
        gm.factor_smooth_discrete_cell_moments(weights, rhs[:, None])

    exact = _cell_test_matrix(factor_basis="fs", discrete=False)
    with pytest.raises(ValueError, match="require a discrete FactorSmooth"):
        exact.factor_smooth_discrete_cell_moments(weights, rhs)


@pytest.mark.parametrize("factor_basis", ["fs", "sz"])
def test_discrete_cell_moments_never_allocate_observation_level_designs(
    monkeypatch: pytest.MonkeyPatch,
    factor_basis: str,
) -> None:
    gm = _cell_test_matrix(factor_basis=factor_basis, n=257)
    weights = np.linspace(0.2, 1.8, gm.shape[0])
    rhs = np.linspace(-0.9, 1.1, gm.shape[0])
    gm.factor_smooth_discrete_cell_moments(weights, rhs)
    forbidden_shapes = {
        gm.shape,
        (gm.shape[0], gm.raw_width),
        (gm.shape[0], gm.block_size),
        (gm.shape[0], gm.n_levels * gm.raw_width),
        (gm.shape[0], gm.n_levels * gm.block_size),
    }
    original_zeros = np.zeros
    original_empty = np.empty

    def guarded_zeros(shape, *args, **kwargs):
        if tuple(np.atleast_1d(shape)) in forbidden_shapes:
            raise AssertionError("allocated an observation-level FactorSmooth design")
        return original_zeros(shape, *args, **kwargs)

    def guarded_empty(shape, *args, **kwargs):
        if tuple(np.atleast_1d(shape)) in forbidden_shapes:
            raise AssertionError("allocated an observation-level FactorSmooth design")
        return original_empty(shape, *args, **kwargs)

    def forbidden_materialization(_self):
        raise AssertionError("materialized an observation-level FactorSmooth design")

    monkeypatch.setattr(np, "zeros", guarded_zeros)
    monkeypatch.setattr(np, "empty", guarded_empty)
    monkeypatch.setattr(FactorSmoothGroupMatrix, "toarray", forbidden_materialization)

    cell_weights, local_gram, xtw_nat, rhs_nat = gm.factor_smooth_discrete_cell_moments(
        weights, rhs
    )

    assert cell_weights.shape == (gm.n_levels, gm.B_unique.shape[0])
    assert local_gram.shape == (gm.n_levels, gm.block_size, gm.block_size)
    assert xtw_nat.shape == rhs_nat.shape == (gm.n_levels, gm.block_size)


def test_discrete_factor_smooth_matches_exact_at_full_support_resolution() -> None:
    X, y, weights, offset = _data()
    exact = _fit(_model(discrete=False, direct_solve="structured"), X, y, weights, offset)
    discrete = _fit(_model(discrete=True, direct_solve="structured"), X, y, weights, offset)

    assert exact._reml_result.converged
    assert discrete._reml_result.converged
    eta_delta = _training_eta(discrete, offset) - _training_eta(exact, offset)
    assert np.sqrt(np.mean(eta_delta**2)) < 5e-3
    assert np.max(np.abs(eta_delta)) < 2e-2
    for name in exact._reml_lambdas:
        assert discrete._reml_lambdas[name] == pytest.approx(
            exact._reml_lambdas[name],
            rel=5e-2,
            abs=2e-6,
        )
    assert discrete._reml_result.objective == pytest.approx(
        exact._reml_result.objective,
        abs=3e-2,
    )


@pytest.mark.parametrize(
    ("lambda_policy", "backend"),
    [
        (LambdaPolicy.fixed(1.0), "streamed_tsqr"),
        (
            {
                "wiggle": LambdaPolicy.fixed(1.0),
                "null_0": LambdaPolicy.fixed(0.7),
                "null_1": LambdaPolicy.fixed(1.3),
            },
            "dense_qr_compat",
        ),
    ],
    ids=["symmetric", "asymmetric"],
)
def test_factor_smooth_marginal_backend_preserves_exact_discrete_fit(
    lambda_policy,
    backend: str,
) -> None:
    X, y, weights, offset = _data()
    exact = _fixed_policy_model(
        discrete=False,
        lambda_policy=lambda_policy,
    ).fit_reml(
        X,
        y,
        sample_weight=weights,
        offset=offset,
        runtime_validation="skip",
    )
    discrete = _fixed_policy_model(
        discrete=True,
        lambda_policy=lambda_policy,
    ).fit_reml(
        X,
        y,
        sample_weight=weights,
        offset=offset,
        runtime_validation="skip",
    )

    assert exact._interaction_specs["x:segment:fs"]._marginal_build_backend == backend
    assert discrete._interaction_specs["x:segment:fs"]._marginal_build_backend == backend
    np.testing.assert_allclose(
        discrete.predict(X),
        exact.predict(X),
        rtol=2e-5,
        atol=2e-6,
    )
    assert discrete.result.deviance == pytest.approx(exact.result.deviance, rel=2e-6)


def test_forced_block_structured_discrete_matches_gram_and_uses_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X, y, weights, offset = _data()
    dense = _fit(_model(discrete=True, direct_solve="gram"), X, y, weights, offset)

    def forbidden_dense_cached_solve(*_args, **_kwargs):
        raise AssertionError("block structured fREML entered the dense cached solver")

    def forbidden_materialization(_self):
        raise AssertionError("factor smooth rows were materialized")

    monkeypatch.setattr(
        discrete_module,
        "_solve_cached_profiled_system",
        forbidden_dense_cached_solve,
    )
    monkeypatch.setattr(FactorSmoothGroupMatrix, "toarray", forbidden_materialization)
    structured = _fit(
        _model(discrete=True, direct_solve="structured"),
        X,
        y,
        weights,
        offset,
    )

    np.testing.assert_allclose(structured.result.beta, dense.result.beta, atol=6e-8)
    assert structured.result.intercept == pytest.approx(dense.result.intercept, abs=6e-8)
    for name in dense._reml_lambdas:
        assert structured._reml_lambdas[name] == pytest.approx(
            dense._reml_lambdas[name],
            rel=6e-7,
            abs=3e-8,
        )
    assert structured._reml_result.objective == pytest.approx(
        dense._reml_result.objective,
        abs=8e-8,
    )
    assert structured._reml_profile["reml_n_structured_cache_solves"] > 0
    assert structured._reml_profile["reml_structured_cache_solve_s"] >= 0.0
    assert structured._reml_profile["reml_structured_cache_data_passes"] == 0
    assert structured._reml_profile["reml_n_block_structured_cache_solves"] > 0
    assert structured._reml_profile["reml_block_structured_cache_solve_s"] >= 0.0
    assert structured._reml_profile["reml_block_structured_cache_data_passes"] == 0


def test_cached_block_lambda_trial_uses_only_retained_moments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X, y, weights, offset = _data()
    model = _fit(
        _model(discrete=True, direct_solve="structured"),
        X,
        y,
        weights,
        offset,
    )
    state = model._linear_system_state
    assert state is not None
    system = state.system
    assert isinstance(system, BlockStructuredSystem)
    trial_lambdas = {
        name: value * (1.25 if name.endswith("wiggle") else 0.85)
        for name, value in model._reml_lambdas.items()
    }
    penalty = build_penalty_matrix(
        list(model._dm.group_matrices),
        model._groups,
        trial_lambdas,
        model._dm.p,
        reml_penalties=model._reml_penalties,
    )
    xtw = np.empty(model._dm.p)
    xtw[system.operator.small_indices] = system.xtw_small
    xtw[system.operator.structured_indices] = system.xtw_structured
    xtwz = np.empty(model._dm.p)
    xtwz[system.operator.small_indices] = system.xtwz_small
    xtwz[system.operator.structured_indices] = system.xtwz_structured
    dense_augmented = np.empty((model._dm.p + 1, model._dm.p + 1))
    dense_augmented[0, 0] = system.sum_w
    dense_augmented[0, 1:] = xtw
    dense_augmented[1:, 0] = xtw
    dense_augmented[1:, 1:] = materialize_compact_operator(system.operator) + penalty
    rhs = np.concatenate(([system.sum_wz], xtwz))
    expected = np.linalg.solve(dense_augmented, rhs)

    def forbidden_rows(*_args, **_kwargs):
        raise AssertionError("cached lambda trial touched observation rows")

    monkeypatch.setattr(
        FactorSmoothGroupMatrix,
        "factor_smooth_sufficient_stats",
        forbidden_rows,
    )
    monkeypatch.setattr(FactorSmoothGroupMatrix, "matvec", forbidden_rows)
    monkeypatch.setattr(FactorSmoothGroupMatrix, "rmatvec", forbidden_rows)
    monkeypatch.setattr(FactorSmoothGroupMatrix, "toarray", forbidden_rows)
    solution = solve_cached_structured(
        system,
        list(model._dm.group_matrices),
        model._groups,
        trial_lambdas,
        reml_penalties=model._reml_penalties,
    )

    assert isinstance(solution, CachedBlockStructuredSolution)
    np.testing.assert_allclose(solution.intercept, expected[0], atol=2e-9)
    np.testing.assert_allclose(solution.beta, expected[1:], atol=2e-9)
    assert solution.log_det_H == pytest.approx(
        np.linalg.slogdet(dense_augmented)[1],
        abs=2e-9,
    )
