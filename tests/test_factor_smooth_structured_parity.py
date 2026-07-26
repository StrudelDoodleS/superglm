"""Exact dense-versus-block parity for factor-smooth structured solves."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import superglm.solvers.irls_direct as irls_direct
from superglm import FactorSmooth, LambdaPolicy, Numeric, RandomEffect, Spline, SuperGLM
from superglm.distributions import Gamma, Gaussian, Poisson
from superglm.group_matrix import (
    DenseGroupMatrix,
    DesignMatrix,
    FactorSmoothGroupMatrix,
    RandomEffectGroupMatrix,
    SparseSSPGroupMatrix,
)
from superglm.links import IdentityLink, LogLink
from superglm.reml.gradient import reml_direct_gradient, reml_direct_hessian
from superglm.reml.objective import REMLObjectiveEvaluation, reml_laml_objective
from superglm.reml.w_derivatives import reml_w_correction
from superglm.solvers.structured import (
    BlockSymmetricOperator,
    ProfiledBlockSchurFactor,
    StructuredLinearSystemState,
    materialize_compact_operator,
)
from superglm.types import GroupSlice, PenaltyComponent


def _factor_smooth_problem(
    response_factory: Callable[[np.random.Generator, np.ndarray], np.ndarray],
):
    rng = np.random.default_rng(819)
    n = 480
    n_levels = 8
    block_size = 4
    x = rng.uniform(-1.0, 1.0, size=n)
    codes = rng.integers(0, n_levels, size=n, dtype=np.intp)
    secondary_codes = rng.integers(0, 3, size=n, dtype=np.intp)
    numeric = rng.normal(size=(n, 2))
    main_basis = np.column_stack((x, x**2, x**3, x**4))
    local_basis = np.column_stack((np.ones(n), x, x**2, x**3))

    main = SparseSSPGroupMatrix(sp.csr_matrix(main_basis), np.eye(4))
    main.omega = np.diag([0.2, 0.7, 1.4, 2.1])
    wiggle = np.diag([0.0, 0.0, 0.8, 1.7])
    null_0 = np.diag([1.0, 0.0, 0.0, 0.0])
    null_1 = np.diag([0.0, 1.0, 0.0, 0.0])
    factor_smooth = FactorSmoothGroupMatrix(
        sp.csr_matrix(local_basis),
        codes,
        n_levels,
        natural_map=np.eye(block_size),
        levels=tuple(f"segment-{level}" for level in range(n_levels)),
        repeated_penalty_components=(
            ("wiggle", wiggle),
            ("null_0", null_0),
            ("null_1", null_1),
        ),
    )
    matrices = [
        DenseGroupMatrix(numeric),
        main,
        RandomEffectGroupMatrix(secondary_codes, n_levels=3),
        factor_smooth,
    ]
    groups: list[GroupSlice] = []
    start = 0
    for name, matrix, penalized in zip(
        ("numeric", "x:main", "branch", "x:segment:fs"),
        matrices,
        (False, True, True, True),
        strict=True,
    ):
        groups.append(
            GroupSlice(
                name=name,
                start=start,
                end=start + matrix.shape[1],
                penalized=penalized,
            )
        )
        start += matrix.shape[1]

    penalties = [
        PenaltyComponent(
            name="x:main",
            group_name="x:main",
            group_index=1,
            group_sl=groups[1].sl,
            omega_raw=main.omega,
            omega_ssp=main.omega,
            rank=4.0,
        ),
        PenaltyComponent(
            name="branch",
            group_name="branch",
            group_index=2,
            group_sl=groups[2].sl,
            omega_raw=None,
            rank=3.0,
            penalty_kind="identity",
        ),
    ]
    for suffix, omega in factor_smooth.repeated_penalty_components:
        penalties.append(
            PenaltyComponent(
                name=f"x:segment:fs:{suffix}",
                group_name="x:segment:fs",
                group_index=3,
                group_sl=groups[3].sl,
                omega_raw=omega,
                omega_ssp=omega,
                rank=float(n_levels * np.linalg.matrix_rank(omega)),
                penalty_kind="repeated",
                repeat_count=n_levels,
                block_width=block_size,
            )
        )

    offset = rng.normal(scale=0.09, size=n)
    local_truth = rng.normal(scale=0.18, size=(n_levels, block_size))
    eta = (
        -0.35
        + numeric @ np.array([0.22, -0.14])
        + 0.16 * np.sin(2.0 * x)
        + np.einsum("ij,ij->i", local_basis, local_truth[codes])
        + np.array([0.12, -0.08, 0.03])[secondary_codes]
        + offset
    )
    y = response_factory(rng, eta)
    weights = rng.uniform(0.35, 2.4, size=n)
    lambdas = {
        "x:main": 1.3,
        "branch": 2.2,
        "x:segment:fs:wiggle": 1.7,
        "x:segment:fs:null_0": 0.65,
        "x:segment:fs:null_1": 0.9,
    }
    return DesignMatrix(matrices, n=n, p=start), groups, penalties, y, weights, offset, lambdas


def _gaussian_response(rng: np.random.Generator, eta: np.ndarray) -> np.ndarray:
    return eta + rng.normal(scale=0.13, size=len(eta))


def _poisson_response(rng: np.random.Generator, eta: np.ndarray) -> np.ndarray:
    return rng.poisson(np.exp(eta)).astype(np.float64)


def _gamma_response(rng: np.random.Generator, eta: np.ndarray) -> np.ndarray:
    mean = np.exp(eta)
    return rng.gamma(shape=4.0, scale=mean / 4.0)


@pytest.mark.parametrize(
    ("family", "link", "response_factory"),
    [
        pytest.param(Gaussian(), IdentityLink(), _gaussian_response, id="gaussian"),
        pytest.param(Poisson(), LogLink(), _poisson_response, id="poisson"),
        pytest.param(Gamma(), LogLink(), _gamma_response, id="gamma"),
    ],
)
def test_forced_factor_smooth_structured_irls_matches_dense(
    family,
    link,
    response_factory,
) -> None:
    dm, groups, penalties, y, weights, offset, lambdas = _factor_smooth_problem(response_factory)
    dense = irls_direct.fit_irls_direct(
        X=dm,
        y=y,
        weights=weights,
        family=family,
        link=link,
        groups=groups,
        lambda2=lambdas,
        offset=offset,
        max_iter=100,
        tol=1e-10,
        return_xtwx=True,
        direct_solve="gram",
        reml_penalties=penalties,
    )
    structured = irls_direct.fit_irls_direct(
        X=dm,
        y=y,
        weights=weights,
        family=family,
        link=link,
        groups=groups,
        lambda2=lambdas,
        offset=offset,
        max_iter=100,
        tol=1e-10,
        return_xtwx=True,
        direct_solve="structured",
        reml_penalties=penalties,
    )
    dense_result, dense_inverse, dense_data = dense
    result, factor, data_operator = structured

    np.testing.assert_allclose(result.beta, dense_result.beta, rtol=3e-8, atol=3e-9)
    assert result.intercept == pytest.approx(dense_result.intercept, rel=3e-8, abs=3e-9)
    np.testing.assert_allclose(
        dm.matvec(result.beta) + result.intercept + offset,
        dm.matvec(dense_result.beta) + dense_result.intercept + offset,
        rtol=3e-8,
        atol=3e-9,
    )
    assert result.deviance == pytest.approx(dense_result.deviance, rel=3e-9, abs=3e-9)
    assert result.effective_df == pytest.approx(
        dense_result.effective_df,
        rel=3e-9,
        abs=3e-9,
    )
    assert result.log_det_H == pytest.approx(dense_result.log_det_H, rel=3e-9, abs=3e-9)
    assert result.n_iter == dense_result.n_iter
    assert result.converged == dense_result.converged
    assert isinstance(factor, ProfiledBlockSchurFactor)
    assert isinstance(data_operator, BlockSymmetricOperator)
    np.testing.assert_allclose(
        factor.solve(np.eye(dm.p)),
        dense_inverse,
        rtol=3e-8,
        atol=3e-9,
    )
    np.testing.assert_allclose(
        materialize_compact_operator(data_operator),
        dense_data,
        rtol=3e-9,
        atol=3e-9,
    )


def test_factor_smooth_fixed_weight_reml_derivatives_match_dense() -> None:
    dm, groups, penalties, y, weights, offset, lambdas = _factor_smooth_problem(_poisson_response)
    dense_result, dense_inverse = irls_direct.fit_irls_direct(
        X=dm,
        y=y,
        weights=weights,
        family=Poisson(),
        link=LogLink(),
        groups=groups,
        lambda2=lambdas,
        offset=offset,
        direct_solve="gram",
        reml_penalties=penalties,
        tol=1e-10,
    )
    result, factor = irls_direct.fit_irls_direct(
        X=dm,
        y=y,
        weights=weights,
        family=Poisson(),
        link=LogLink(),
        groups=groups,
        lambda2=lambdas,
        offset=offset,
        direct_solve="structured",
        reml_penalties=penalties,
        tol=1e-10,
    )

    dense_gradient = reml_direct_gradient(
        list(dm.group_matrices),
        dense_result,
        dense_inverse,
        lambdas,
        reml_penalties=penalties,
    )
    gradient = reml_direct_gradient(
        list(dm.group_matrices),
        result,
        factor,
        lambdas,
        reml_penalties=penalties,
    )
    np.testing.assert_allclose(gradient, dense_gradient, atol=3e-9)

    dense_hessian = reml_direct_hessian(
        list(dm.group_matrices),
        Poisson(),
        dense_inverse,
        lambdas,
        gradient=dense_gradient,
        reml_penalties=penalties,
    )
    hessian = reml_direct_hessian(
        list(dm.group_matrices),
        Poisson(),
        factor,
        lambdas,
        gradient=gradient,
        reml_penalties=penalties,
    )
    np.testing.assert_allclose(hessian, dense_hessian, atol=3e-8)


def test_factor_smooth_w_derivatives_match_dense() -> None:
    dm, groups, penalties, y, weights, offset, lambdas = _factor_smooth_problem(_poisson_response)
    dense_result, dense_inverse = irls_direct.fit_irls_direct(
        X=dm,
        y=y,
        weights=weights,
        family=Poisson(),
        link=LogLink(),
        groups=groups,
        lambda2=lambdas,
        offset=offset,
        direct_solve="gram",
        reml_penalties=penalties,
        tol=1e-10,
    )
    result, factor = irls_direct.fit_irls_direct(
        X=dm,
        y=y,
        weights=weights,
        family=Poisson(),
        link=LogLink(),
        groups=groups,
        lambda2=lambdas,
        offset=offset,
        direct_solve="structured",
        reml_penalties=penalties,
        tol=1e-10,
    )
    dense_correction = reml_w_correction(
        dm,
        LogLink(),
        groups,
        dense_result,
        dense_inverse,
        lambdas,
        sample_weight=weights,
        offset_arr=offset,
        distribution=Poisson(),
        w_correction_order=2,
        reml_penalties=penalties,
    )
    correction = reml_w_correction(
        dm,
        LogLink(),
        groups,
        result,
        factor,
        lambdas,
        sample_weight=weights,
        offset_arr=offset,
        distribution=Poisson(),
        w_correction_order=2,
        reml_penalties=penalties,
    )

    assert dense_correction is not None
    assert correction is not None
    dense_gradient, dense_operators, dense_second = dense_correction
    gradient, operators, second = correction
    np.testing.assert_allclose(gradient, dense_gradient, atol=4e-9)
    for index, dense_operator in dense_operators.items():
        np.testing.assert_allclose(
            materialize_compact_operator(operators[index]),
            dense_operator,
            atol=3e-9,
        )
    np.testing.assert_allclose(second, dense_second, atol=4e-8)


def test_all_factor_smooth_lambda_derivatives_match_finite_differences() -> None:
    dm, groups, penalties, y, weights, offset, base_lambdas = _factor_smooth_problem(
        _gaussian_response
    )
    factor_names = [
        "x:segment:fs:wiggle",
        "x:segment:fs:null_0",
        "x:segment:fs:null_1",
    ]
    all_names = [component.name for component in penalties]
    factor_columns = np.array([all_names.index(name) for name in factor_names], dtype=np.intp)
    base_rho = np.log([base_lambdas[name] for name in factor_names])

    def evaluate(rho: np.ndarray, *, derivatives: bool):
        lambdas = dict(base_lambdas)
        lambdas.update(
            {name: float(np.exp(value)) for name, value in zip(factor_names, rho, strict=True)}
        )
        result, factor, data_operator = irls_direct.fit_irls_direct(
            X=dm,
            y=y,
            weights=weights,
            family=Gaussian(),
            link=IdentityLink(),
            groups=groups,
            lambda2=lambdas,
            offset=offset,
            direct_solve="structured",
            reml_penalties=penalties,
            return_xtwx=True,
            tol=1e-11,
        )
        objective = reml_laml_objective(
            dm,
            Gaussian(),
            IdentityLink(),
            groups,
            y,
            result,
            lambdas,
            weights,
            offset,
            XtWX=data_operator,
            log_det_H=result.log_det_H,
            hessian_rank=result.reml_hessian_rank,
            reml_penalties=penalties,
            return_evaluation=True,
        )
        assert isinstance(objective, REMLObjectiveEvaluation)
        if not derivatives:
            return objective.value
        assert objective.profiled_scale is not None
        scale = objective.profiled_scale
        gradient = reml_direct_gradient(
            list(dm.group_matrices),
            result,
            factor,
            lambdas,
            inverse_phi=scale.inverse_phi,
            reml_penalties=penalties,
        )
        hessian = reml_direct_hessian(
            list(dm.group_matrices),
            Gaussian(),
            factor,
            lambdas,
            gradient=gradient,
            pirls_result=result,
            n_obs=dm.n,
            inverse_phi=scale.inverse_phi,
            d_inverse_phi_d_penalized_deviance=(scale.d_inverse_phi_d_penalized_deviance),
            penalty_nullity=objective.penalty_nullity,
            reml_penalties=penalties,
        )
        return objective.value, gradient, hessian

    value, gradient, hessian = evaluate(base_rho, derivatives=True)
    assert np.isfinite(value)
    eps = 2e-5
    finite_gradient = np.empty(len(factor_names))
    finite_hessian_columns = np.empty((len(penalties), len(factor_names)))
    for column in range(len(factor_names)):
        step = np.zeros_like(base_rho)
        step[column] = eps
        finite_gradient[column] = (
            evaluate(base_rho + step, derivatives=False)
            - evaluate(base_rho - step, derivatives=False)
        ) / (2.0 * eps)
        _, gradient_plus, _ = evaluate(base_rho + step, derivatives=True)
        _, gradient_minus, _ = evaluate(base_rho - step, derivatives=True)
        finite_hessian_columns[:, column] = (gradient_plus - gradient_minus) / (2.0 * eps)

    np.testing.assert_allclose(
        gradient[factor_columns],
        finite_gradient,
        rtol=2e-5,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        hessian[:, factor_columns],
        finite_hessian_columns,
        rtol=2e-4,
        atol=2e-5,
    )


def _public_factor_smooth_data(
    family: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(947)
    n = 420
    x = rng.uniform(-1.0, 1.0, size=n)
    z = rng.normal(size=n)
    segment_code = rng.integers(0, 7, size=n)
    branch_code = rng.integers(0, 4, size=n)
    segment = np.array([f"segment-{code}" for code in segment_code], dtype=object)
    branch = np.array([f"branch-{code}" for code in branch_code], dtype=object)
    offset = rng.normal(scale=0.08, size=n)
    weights = rng.uniform(0.4, 2.1, size=n)
    amplitudes = np.array([0.45, -0.25, 0.3, -0.4, 0.2, 0.35, -0.15])
    branch_effect = np.array([0.08, -0.12, 0.04, 0.15])
    eta = (
        -0.2
        + 0.18 * z
        + 0.22 * np.sin(2.4 * x)
        + amplitudes[segment_code] * (x + 0.35 * x**2)
        + branch_effect[branch_code]
        + offset
    )
    if family == "gaussian":
        y = eta + rng.normal(scale=0.16, size=n)
    elif family == "poisson":
        y = rng.poisson(np.exp(eta)).astype(np.float64)
    else:
        mean = np.exp(eta)
        y = rng.gamma(shape=4.5, scale=mean / 4.5)
    X = pd.DataFrame(
        {
            "x": x,
            "z": z,
            "segment": segment,
            "branch": branch,
        }
    )
    return X, y, weights, offset


def _public_factor_smooth_model(family: str, direct_solve: str) -> SuperGLM:
    return SuperGLM(
        family=family,
        features={
            "x": Spline(n_knots=5),
            "z": Numeric(),
            "branch": RandomEffect(),
        },
        interactions=[FactorSmooth("x", group="segment", k=6)],
        selection_penalty=0.0,
        direct_solve=direct_solve,
    )


@pytest.mark.parametrize("family", ["gaussian", "poisson", "gamma"])
def test_factor_smooth_exact_reml_matches_dense_end_to_end(family: str) -> None:
    X, y, weights, offset = _public_factor_smooth_data(family)
    dense = _public_factor_smooth_model(family, "gram")
    structured = _public_factor_smooth_model(family, "structured")

    dense.fit_reml(
        X,
        y,
        sample_weight=weights,
        offset=offset,
        max_reml_iter=6,
        reml_tol=1e-5,
        pirls_tol=1e-9,
        runtime_validation="skip",
    )
    structured.fit_reml(
        X,
        y,
        sample_weight=weights,
        offset=offset,
        max_reml_iter=6,
        reml_tol=1e-5,
        pirls_tol=1e-9,
        runtime_validation="skip",
    )

    np.testing.assert_allclose(structured.result.beta, dense.result.beta, atol=5e-8)
    assert structured.result.intercept == pytest.approx(
        dense.result.intercept,
        abs=5e-8,
    )
    np.testing.assert_allclose(
        structured._dm.matvec(structured.result.beta) + structured.result.intercept + offset,
        dense._dm.matvec(dense.result.beta) + dense.result.intercept + offset,
        atol=5e-8,
    )
    assert structured.result.deviance == pytest.approx(dense.result.deviance, abs=5e-8)
    assert structured.result.effective_df == pytest.approx(
        dense.result.effective_df,
        abs=5e-8,
    )
    assert structured.result.log_det_H == pytest.approx(
        dense.result.log_det_H,
        abs=5e-8,
    )
    assert structured._reml_result.objective == pytest.approx(
        dense._reml_result.objective,
        abs=8e-8,
    )
    assert structured._reml_result.converged == dense._reml_result.converged
    assert structured._reml_result.n_reml_iter == dense._reml_result.n_reml_iter
    assert structured._reml_lambdas.keys() == dense._reml_lambdas.keys()
    for name in structured._reml_lambdas:
        assert structured._reml_lambdas[name] == pytest.approx(
            dense._reml_lambdas[name],
            rel=5e-7,
            abs=2e-8,
        )
    assert isinstance(structured._linear_system_state, StructuredLinearSystemState)
    assert isinstance(
        structured._linear_system_state.profiled_factor,
        ProfiledBlockSchurFactor,
    )


def test_factor_smooth_estimability_and_summary_match_dense_centered_geometry():
    rng = np.random.default_rng(20260726)
    n_levels = 10
    repeats = 15
    codes = np.repeat(np.arange(n_levels), repeats)
    x = np.tile(np.linspace(0.0, 1.0, repeats), n_levels)
    z = rng.normal(size=len(x))
    X = pd.DataFrame(
        {
            "x": x,
            "z": z,
            "group": np.array([f"g{code}" for code in codes], dtype=object),
        }
    )
    y = np.sin(3.0 * x) + 0.1 * z + rng.normal(scale=0.05, size=len(x))
    policies = {
        "wiggle": LambdaPolicy.fixed(1.0),
        "null_0": LambdaPolicy.fixed(1.0),
        "null_1": LambdaPolicy.fixed(1.0),
    }
    common = {
        "family": "gaussian",
        "features": {"z": Numeric()},
        "interactions": [
            FactorSmooth("x", group="group", k=5, lambda_policy=policies),
        ],
        "selection_penalty": 0.0,
    }
    dense = SuperGLM(**common, direct_solve="gram").fit_reml(
        X,
        y,
        runtime_validation="skip",
    )
    structured = SuperGLM(**common, direct_solve="structured").fit_reml(
        X,
        y,
        runtime_validation="skip",
    )

    np.testing.assert_array_equal(
        structured._fit_inference_info["coefficient_estimable"],
        dense._fit_inference_info["coefficient_estimable"],
    )
    factor_group = next(group for group in structured._groups if group.name == "x:group:fs")
    assert np.any(~structured._fit_inference_info["coefficient_estimable"][factor_group.sl])
    row = next(row for row in structured.summary()._coef_rows if row.name == "x:group:fs")
    assert row.coef is None
    assert row.structured_kind == "factor_smooth_fs"
    assert row.n_levels == n_levels
    assert row.n_params == n_levels * 5
    assert {name for name, _value in row.smoothing_lambdas} == {
        "wiggle",
        "null_0",
        "null_1",
    }


@pytest.mark.parametrize("basis", ["fs", "sz"])
@pytest.mark.parametrize("discrete", [False, True])
def test_auto_factor_smooth_falls_back_for_singular_local_blocks(
    basis: str,
    discrete: bool,
) -> None:
    rng = np.random.default_rng(20260726)
    n_levels = 10
    repeats = 12
    codes = np.repeat(np.arange(n_levels), repeats)
    x = np.tile(np.linspace(0.0, 1.0, repeats), n_levels)
    z = rng.normal(size=len(x))
    X = pd.DataFrame(
        {
            "x": x,
            "z": z,
            "group": np.array([f"g{code}" for code in codes], dtype=object),
        }
    )
    y = np.sin(4.0 * x) + 0.1 * z + rng.normal(scale=0.05, size=len(x))
    sample_weight = np.ones(len(x))
    sample_weight[codes == n_levels - 1] = 0.0
    policies = {"wiggle": LambdaPolicy.off()}
    features = {"z": Numeric()}
    if basis == "fs":
        policies.update(
            null_0=LambdaPolicy.off(),
            null_1=LambdaPolicy.off(),
        )
    else:
        features["x"] = Spline(k=5, lambda_policy=LambdaPolicy.fixed(1.0))
    model = SuperGLM(
        family="gaussian",
        features=features,
        interactions=[
            FactorSmooth(
                "x",
                group="group",
                basis=basis,
                k=5,
                lambda_policy=policies,
            ),
        ],
        selection_penalty=0.0,
        direct_solve="auto",
        discrete=discrete,
        n_bins=64,
    ).fit_reml(
        X,
        y,
        sample_weight=sample_weight,
        runtime_validation="skip",
    )

    assert model.result.direct_backend == "gram"
    assert "singular local block" in model.result.direct_fallback_reason
