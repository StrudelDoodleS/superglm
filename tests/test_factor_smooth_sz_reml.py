"""Dense-versus-structured REML parity for sum-to-zero factor smooths."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import superglm.reml.penalty_algebra as penalty_algebra
import superglm.solvers.irls_direct as irls_direct
from superglm import FactorSmooth, LambdaPolicy, Numeric, Spline, SuperGLM
from superglm.group_matrix import FactorSmoothGroupMatrix
from superglm.reml.gradient import reml_direct_gradient, reml_direct_hessian
from superglm.reml.objective import REMLObjectiveEvaluation, reml_laml_objective
from superglm.reml.observed_geometry import build_observed_reml_geometry
from superglm.reml.w_derivatives import reml_w_correction
from superglm.solvers.structured import BlockStructuredLayout, materialize_compact_operator
from superglm.solvers.sum_to_zero import ProfiledSumToZeroBlockFactor


def _data(family: str):
    rng = np.random.default_rng(2027)
    n = 360
    n_levels = 6
    x = rng.uniform(-1.0, 1.0, size=n)
    z = rng.normal(size=n)
    codes = rng.integers(0, n_levels, size=n)
    group = np.array([f"level-{code}" for code in codes], dtype=object)
    deviations = rng.normal(scale=0.18, size=(n_levels, 3))
    deviations -= deviations.mean(axis=0)
    local = deviations[codes, 0] + deviations[codes, 1] * x + deviations[codes, 2] * x**2
    eta = -0.25 + 0.45 * np.sin(2.1 * x) + 0.17 * z + local
    if family == "gaussian":
        y = eta + rng.normal(scale=0.15, size=n)
    else:
        y = rng.poisson(np.exp(eta)).astype(np.float64)
    weights = rng.uniform(0.5, 1.8, size=n)
    offset = rng.normal(scale=0.04, size=n)
    X = pd.DataFrame({"x": x, "z": z, "group": group})
    return X, y, weights, offset


def _model(
    *,
    family: str,
    discrete: bool,
    direct_solve: str,
    estimate_wiggle: bool = False,
) -> SuperGLM:
    factor_smooth = FactorSmooth(
        "x",
        group="group",
        basis="sz",
        k=6,
        lambda_policy=(None if estimate_wiggle else {"wiggle": LambdaPolicy.fixed(1.7)}),
    )
    return SuperGLM(
        family=family,
        features={
            "x": Spline(n_knots=5, lambda_policy=LambdaPolicy.fixed(1.3)),
            "z": Numeric(),
        },
        interactions=[factor_smooth],
        selection_penalty=0.0,
        discrete=discrete,
        n_bins=192,
        direct_solve=direct_solve,
    )


@pytest.mark.parametrize("family", ["gaussian", "poisson"])
@pytest.mark.parametrize("discrete", [False, True])
def test_fixed_lambda_sz_structured_fit_matches_dense(
    family: str,
    discrete: bool,
) -> None:
    X, y, weights, offset = _data(family)
    dense = _model(
        family=family,
        discrete=discrete,
        direct_solve="gram",
    ).fit_reml(
        X,
        y,
        sample_weight=weights,
        offset=offset,
        max_reml_iter=2,
        pirls_tol=1e-10,
        runtime_validation="skip",
    )
    structured = _model(
        family=family,
        discrete=discrete,
        direct_solve="structured",
    ).fit_reml(
        X,
        y,
        sample_weight=weights,
        offset=offset,
        max_reml_iter=2,
        pirls_tol=1e-10,
        runtime_validation="skip",
    )

    np.testing.assert_allclose(structured.result.beta, dense.result.beta, atol=6e-8)
    assert structured.result.intercept == pytest.approx(dense.result.intercept, abs=6e-8)
    np.testing.assert_allclose(
        structured.predict(X),
        dense.predict(X),
        atol=6e-8,
    )
    assert structured.result.deviance == pytest.approx(dense.result.deviance, abs=8e-8)
    assert structured.result.effective_df == pytest.approx(
        dense.result.effective_df,
        abs=8e-8,
    )
    assert structured.result.log_det_H == pytest.approx(dense.result.log_det_H, abs=8e-8)
    assert structured._reml_result.objective == pytest.approx(
        dense._reml_result.objective,
        abs=1e-7,
    )
    assert structured.result.direct_backend == "structured"
    assert set(structured._reml_result.lambdas) == {
        "x:wiggle",
        "x:group:sz:wiggle",
    }
    assert isinstance(
        structured._linear_system_state.profiled_factor,
        ProfiledSumToZeroBlockFactor,
    )


def test_sz_structured_full_runtime_geometry_validation() -> None:
    X, y, weights, offset = _data("poisson")
    model = _model(
        family="poisson",
        discrete=False,
        direct_solve="structured",
    )

    model.fit_reml(
        X,
        y,
        sample_weight=weights,
        offset=offset,
        max_reml_iter=2,
        pirls_tol=1e-10,
        runtime_validation="full",
    )

    assert model.result.direct_backend == "structured"
    assert np.isfinite(model._reml_result.objective)


def test_sz_compact_observed_geometry_uses_public_positive_definiteness() -> None:
    X, y, weights, offset = _data("poisson")
    model = _model(
        family="poisson",
        discrete=False,
        direct_solve="structured",
    ).fit_reml(
        X,
        y,
        sample_weight=weights,
        offset=offset,
        max_reml_iter=2,
        pirls_tol=1e-10,
        runtime_validation="skip",
    )
    structured_index = next(
        index
        for index, matrix in enumerate(model._dm.group_matrices)
        if getattr(matrix, "factor_basis", None) == "sz"
    )

    geometry = build_observed_reml_geometry(
        dm=model._dm,
        distribution=model._distribution,
        link=model._link,
        y=y,
        sample_weight=weights,
        offset_arr=offset,
        result=model.result,
        penalty=None,
        derivative_order=2,
        groups=model._groups,
        lambdas=model._reml_lambdas,
        reml_penalties=model._reml_penalties,
        structured_group_index=structured_index,
    )

    assert geometry.hessian_rank == model._dm.p + 1
    assert geometry.log_det_H == pytest.approx(model.result.log_det_H, rel=2e-8)


@pytest.mark.parametrize("discrete", [False, True])
def test_estimated_sz_lambda_structured_reml_matches_dense(discrete: bool) -> None:
    X, y, weights, offset = _data("poisson")
    dense = _model(
        family="poisson",
        discrete=discrete,
        direct_solve="gram",
        estimate_wiggle=True,
    ).fit_reml(
        X,
        y,
        sample_weight=weights,
        offset=offset,
        max_reml_iter=7,
        reml_tol=1e-6,
        pirls_tol=1e-10,
        runtime_validation="skip",
    )
    structured = _model(
        family="poisson",
        discrete=discrete,
        direct_solve="structured",
        estimate_wiggle=True,
    ).fit_reml(
        X,
        y,
        sample_weight=weights,
        offset=offset,
        max_reml_iter=7,
        reml_tol=1e-6,
        pirls_tol=1e-10,
        runtime_validation="skip",
    )
    name = "x:group:sz:wiggle"

    assert set(
        component.name
        for component in structured._reml_penalties
        if component.group_name == "x:group:sz"
    ) == {name}
    assert structured._reml_lambdas[name] == pytest.approx(
        dense._reml_lambdas[name],
        rel=2e-5,
        abs=2e-7,
    )
    np.testing.assert_allclose(structured.result.beta, dense.result.beta, atol=2e-7)
    assert structured.result.intercept == pytest.approx(dense.result.intercept, abs=2e-7)
    assert structured._reml_result.objective == pytest.approx(
        dense._reml_result.objective,
        abs=3e-7,
    )


@pytest.mark.parametrize("discrete", [False, True])
def test_sz_structured_reml_derivatives_and_working_weight_correction_match_dense(
    discrete: bool,
) -> None:
    X, y, weights, offset = _data("poisson")
    model = _model(
        family="poisson",
        discrete=discrete,
        direct_solve="structured",
    ).fit_reml(
        X,
        y,
        sample_weight=weights,
        offset=offset,
        max_reml_iter=2,
        pirls_tol=1e-10,
        runtime_validation="skip",
    )
    common = dict(
        X=model._dm,
        y=y,
        weights=weights,
        family=model._distribution,
        link=model._link,
        groups=model._groups,
        lambda2=model._reml_lambdas,
        offset=offset,
        reml_penalties=model._reml_penalties,
        tol=1e-10,
    )
    dense_result, dense_inverse = irls_direct.fit_irls_direct(
        **common,
        direct_solve="gram",
        weight_semantics="frequency",
    )
    result, factor = irls_direct.fit_irls_direct(
        **common,
        direct_solve="structured",
        weight_semantics="frequency",
    )

    dense_gradient = reml_direct_gradient(
        list(model._dm.group_matrices),
        dense_result,
        dense_inverse,
        model._reml_lambdas,
        reml_penalties=model._reml_penalties,
    )
    gradient = reml_direct_gradient(
        list(model._dm.group_matrices),
        result,
        factor,
        model._reml_lambdas,
        reml_penalties=model._reml_penalties,
    )
    np.testing.assert_allclose(gradient, dense_gradient, atol=4e-9)

    dense_hessian = reml_direct_hessian(
        list(model._dm.group_matrices),
        model._distribution,
        dense_inverse,
        model._reml_lambdas,
        gradient=dense_gradient,
        reml_penalties=model._reml_penalties,
    )
    hessian = reml_direct_hessian(
        list(model._dm.group_matrices),
        model._distribution,
        factor,
        model._reml_lambdas,
        gradient=gradient,
        reml_penalties=model._reml_penalties,
    )
    np.testing.assert_allclose(hessian, dense_hessian, atol=4e-8)

    correction_kwargs = dict(
        dm=model._dm,
        link=model._link,
        groups=model._groups,
        lambdas=model._reml_lambdas,
        sample_weight=weights,
        offset_arr=offset,
        distribution=model._distribution,
        w_correction_order=2,
        reml_penalties=model._reml_penalties,
    )
    dense_correction = reml_w_correction(
        pirls_result=dense_result,
        XtWX_S_inv=dense_inverse,
        **correction_kwargs,
    )
    correction = reml_w_correction(
        pirls_result=result,
        XtWX_S_inv=factor,
        **correction_kwargs,
    )

    assert dense_correction is not None
    assert correction is not None
    dense_w_gradient, dense_operators, dense_second = dense_correction
    w_gradient, operators, second = correction
    np.testing.assert_allclose(w_gradient, dense_w_gradient, atol=5e-9)
    for index, dense_operator in dense_operators.items():
        np.testing.assert_allclose(
            materialize_compact_operator(operators[index]),
            dense_operator,
            atol=5e-9,
        )
    np.testing.assert_allclose(second, dense_second, atol=5e-8)


def test_sz_lambda_derivatives_match_central_finite_differences() -> None:
    X, y, weights, offset = _data("gaussian")
    model = _model(
        family="gaussian",
        discrete=False,
        direct_solve="structured",
    ).fit_reml(
        X,
        y,
        sample_weight=weights,
        offset=offset,
        max_reml_iter=2,
        pirls_tol=1e-11,
        runtime_validation="skip",
    )
    factor_name = "x:group:sz:wiggle"
    penalty_names = [component.name for component in model._reml_penalties]
    factor_column = penalty_names.index(factor_name)
    base_rho = float(np.log(model._reml_lambdas[factor_name]))

    def evaluate(rho: float, *, derivatives: bool):
        lambdas = dict(model._reml_lambdas)
        lambdas[factor_name] = float(np.exp(rho))
        result, factor, data_operator = irls_direct.fit_irls_direct(
            X=model._dm,
            y=y,
            weights=weights,
            family=model._distribution,
            link=model._link,
            groups=model._groups,
            lambda2=lambdas,
            offset=offset,
            direct_solve="structured",
            reml_penalties=model._reml_penalties,
            return_xtwx=True,
            tol=1e-11,
            weight_semantics="frequency",
        )
        objective = reml_laml_objective(
            model._dm,
            model._distribution,
            model._link,
            model._groups,
            y,
            result,
            lambdas,
            weights,
            offset,
            XtWX=data_operator,
            log_det_H=result.log_det_H,
            hessian_rank=result.reml_hessian_rank,
            reml_penalties=model._reml_penalties,
            return_evaluation=True,
            weight_semantics="frequency",
        )
        assert isinstance(objective, REMLObjectiveEvaluation)
        if not derivatives:
            return objective.value
        assert objective.profiled_scale is not None
        scale = objective.profiled_scale
        gradient = reml_direct_gradient(
            list(model._dm.group_matrices),
            result,
            factor,
            lambdas,
            inverse_phi=scale.inverse_phi,
            reml_penalties=model._reml_penalties,
        )
        hessian = reml_direct_hessian(
            list(model._dm.group_matrices),
            model._distribution,
            factor,
            lambdas,
            gradient=gradient,
            pirls_result=result,
            n_obs=model._dm.n,
            inverse_phi=scale.inverse_phi,
            d_inverse_phi_d_penalized_deviance=scale.d_inverse_phi_d_penalized_deviance,
            penalty_nullity=objective.penalty_nullity,
            reml_penalties=model._reml_penalties,
        )
        return objective.value, gradient, hessian

    value, gradient, hessian = evaluate(base_rho, derivatives=True)
    assert np.isfinite(value)
    eps = 2e-5
    objective_plus = evaluate(base_rho + eps, derivatives=False)
    objective_minus = evaluate(base_rho - eps, derivatives=False)
    finite_gradient = (objective_plus - objective_minus) / (2.0 * eps)
    _, gradient_plus, _ = evaluate(base_rho + eps, derivatives=True)
    _, gradient_minus, _ = evaluate(base_rho - eps, derivatives=True)
    finite_hessian_column = (gradient_plus - gradient_minus) / (2.0 * eps)

    assert gradient[factor_column] == pytest.approx(finite_gradient, rel=2e-5, abs=2e-6)
    np.testing.assert_allclose(
        hessian[:, factor_column],
        finite_hessian_column,
        rtol=2e-4,
        atol=2e-5,
    )


@pytest.mark.parametrize("discrete", [False, True])
def test_sz_structured_fit_uses_tabmat_small_partition_without_dense_dominant(
    monkeypatch: pytest.MonkeyPatch,
    discrete: bool,
) -> None:
    X, y, weights, offset = _data("gaussian")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("structured SZ path materialized dominant geometry")

    monkeypatch.setattr(FactorSmoothGroupMatrix, "toarray", forbidden)
    monkeypatch.setattr(FactorSmoothGroupMatrix, "gram", forbidden)
    monkeypatch.setattr(penalty_algebra, "penalty_component_dense_matrix", forbidden)
    model = _model(
        family="gaussian",
        discrete=discrete,
        direct_solve="structured",
    ).fit_reml(
        X,
        y,
        sample_weight=weights,
        offset=offset,
        max_reml_iter=2,
        runtime_validation="skip",
    )

    layout = next(
        layout
        for layout in model._dm._scalar_structured_layout_cache.values()
        if isinstance(layout, BlockStructuredLayout)
    )
    assert layout.small_execution_plan is not None
    assert layout.small_execution_plan.ordinary_indices
    assert layout.small_execution_plan._ordinary_split_built
    assert model.result.direct_backend == "structured"
