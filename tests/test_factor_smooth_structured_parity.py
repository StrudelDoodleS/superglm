"""Exact dense-versus-block parity for factor-smooth structured solves."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import superglm.reml.direct as direct_reml
import superglm.reml.discrete as discrete_reml
import superglm.solvers._structured.selection as structured_selection
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
from superglm.reml.penalty_algebra import build_penalty_matrix
from superglm.reml.w_derivatives import reml_w_correction
from superglm.solvers.structured import (
    BlockSymmetricOperator,
    ProfiledBlockSchurFactor,
    StructuredLinearSystemState,
    materialize_compact_operator,
    resolve_structured_backend,
)
from superglm.types import GroupSlice, PenaltyComponent


def _factor_smooth_problem(
    response_factory: Callable[[np.random.Generator, np.ndarray], np.ndarray],
    *,
    factor_basis: str = "fs",
):
    rng = np.random.default_rng(819)
    n = 480
    n_levels = 16
    block_size = 4
    x = rng.uniform(-1.0, 1.0, size=n)
    codes = rng.integers(0, n_levels, size=n, dtype=np.intp)
    secondary_codes = rng.integers(0, 3, size=n, dtype=np.intp)
    numeric = rng.normal(size=(n, 2))
    main_basis = np.column_stack((x, x**2, x**3, x**4))
    local_basis = np.column_stack((np.ones(n), x, x**2, x**3))

    main = SparseSSPGroupMatrix(sp.csr_matrix(main_basis), np.eye(4))
    main.omega = np.diag([0.2, 0.7, 1.4, 2.1])
    wiggle = np.diag([0.0, 0.0, 0.8, 1.7]) if factor_basis == "fs" else np.eye(block_size)
    null_0 = np.diag([1.0, 0.0, 0.0, 0.0])
    null_1 = np.diag([0.0, 1.0, 0.0, 0.0])
    repeated_components = (
        (
            ("wiggle", wiggle),
            ("null_0", null_0),
            ("null_1", null_1),
        )
        if factor_basis == "fs"
        else (("wiggle", wiggle),)
    )
    factor_smooth = FactorSmoothGroupMatrix(
        sp.csr_matrix(local_basis),
        codes,
        n_levels,
        natural_map=np.eye(block_size),
        levels=tuple(f"segment-{level}" for level in range(n_levels)),
        repeated_penalty_components=repeated_components,
        factor_basis=factor_basis,
    )
    matrices = [
        DenseGroupMatrix(numeric),
        main,
        RandomEffectGroupMatrix(secondary_codes, n_levels=3),
        factor_smooth,
    ]
    groups: list[GroupSlice] = []
    start = 0
    factor_group_name = f"x:segment:{factor_basis}"
    for name, matrix, penalized in zip(
        ("numeric", "x:main", "branch", factor_group_name),
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
        coefficient_levels = n_levels if factor_basis == "fs" else n_levels - 1
        penalties.append(
            PenaltyComponent(
                name=f"{factor_group_name}:{suffix}",
                group_name=factor_group_name,
                group_index=3,
                group_sl=groups[3].sl,
                omega_raw=omega,
                omega_ssp=omega,
                rank=float(coefficient_levels * np.linalg.matrix_rank(omega)),
                penalty_kind=("repeated" if factor_basis == "fs" else "sum_to_zero"),
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
        f"{factor_group_name}:wiggle": 1.7,
    }
    if factor_basis == "fs":
        lambdas.update(
            {
                f"{factor_group_name}:null_0": 0.65,
                f"{factor_group_name}:null_1": 0.9,
            }
        )
    return DesignMatrix(matrices, n=n, p=start), groups, penalties, y, weights, offset, lambdas


def _gaussian_response(rng: np.random.Generator, eta: np.ndarray) -> np.ndarray:
    return eta + rng.normal(scale=0.13, size=len(eta))


def _poisson_response(rng: np.random.Generator, eta: np.ndarray) -> np.ndarray:
    return rng.poisson(np.exp(eta)).astype(np.float64)


def _gamma_response(rng: np.random.Generator, eta: np.ndarray) -> np.ndarray:
    mean = np.exp(eta)
    return rng.gamma(shape=4.0, scale=mean / 4.0)


def _factor_smooth_override(
    dm: DesignMatrix,
    groups: list[GroupSlice],
    penalties: list[PenaltyComponent],
    lambdas: dict[str, float],
    *,
    local_penalty: np.ndarray | None,
) -> np.ndarray:
    override = build_penalty_matrix(
        dm.group_matrices,
        groups,
        lambdas,
        dm.p,
        reml_penalties=penalties,
    )
    matrix = dm.group_matrices[3]
    assert isinstance(matrix, FactorSmoothGroupMatrix)
    group = groups[3]
    override[group.sl, group.sl] = 0.0
    if local_penalty is None:
        return override
    if matrix.factor_basis == "fs":
        override[group.sl, group.sl] = np.kron(
            np.eye(matrix.n_levels),
            local_penalty,
        )
        return override
    free_levels = matrix.n_levels - 1
    sum_to_zero_level_geometry = np.ones((free_levels, free_levels)) + np.eye(free_levels)
    override[group.sl, group.sl] = np.kron(
        sum_to_zero_level_geometry,
        local_penalty,
    )
    return override


def _selection_factor_smooth_matrix(
    *,
    factor_basis: str,
    n_levels: int,
    local_basis: np.ndarray,
    repeated_penalty_components: tuple[tuple[str, np.ndarray], ...],
) -> tuple[FactorSmoothGroupMatrix, GroupSlice]:
    rows_per_level = local_basis.shape[0] // n_levels
    codes = np.repeat(np.arange(n_levels, dtype=np.intp), rows_per_level)
    block_size = local_basis.shape[1]
    matrix = FactorSmoothGroupMatrix(
        sp.csr_matrix(local_basis),
        codes,
        n_levels,
        natural_map=np.eye(block_size),
        levels=tuple(f"level-{level}" for level in range(n_levels)),
        repeated_penalty_components=repeated_penalty_components,
        factor_basis=factor_basis,
    )
    return matrix, GroupSlice(
        name=f"x:group:{factor_basis}",
        start=0,
        end=matrix.shape[1],
        penalized=True,
    )


def test_sz_resolution_never_runs_local_feasibility_scans(monkeypatch) -> None:
    n_levels = 4
    x = np.tile(np.linspace(-1.0, 1.0, 6), n_levels)
    matrix, group = _selection_factor_smooth_matrix(
        factor_basis="sz",
        n_levels=n_levels,
        local_basis=np.column_stack((np.ones_like(x), x)),
        repeated_penalty_components=(("wiggle", np.eye(2)),),
    )
    weights = np.ones(matrix.shape[0])
    weights[matrix.codes == n_levels - 1] = 0.0
    override = np.zeros((matrix.shape[1], matrix.shape[1]))
    monkeypatch.setattr(
        FactorSmoothGroupMatrix,
        "factor_smooth_sufficient_stats",
        lambda *_args, **_kwargs: pytest.fail("SZ resolution scanned all training rows"),
    )
    monkeypatch.setattr(
        structured_selection,
        "_first_singular_factor_smooth_block",
        lambda *_args, **_kwargs: pytest.fail("SZ resolution eigendecomposed local penalties"),
    )
    monkeypatch.setattr(
        structured_selection,
        "_factor_smooth_override_local_blocks",
        lambda *_args, **_kwargs: pytest.fail("SZ resolution expanded local override blocks"),
    )
    automatic = resolve_structured_backend(
        [matrix],
        [group],
        direct_solve="auto",
        coefficient_width=matrix.shape[1],
        row_weights=weights,
        lambda2={f"{group.name}:wiggle": 1.0},
        S_override=override,
    )
    assert not automatic.use_structured
    assert "crossover" in automatic.fallback_reason

    forced = resolve_structured_backend(
        [matrix],
        [group],
        direct_solve="structured",
        coefficient_width=matrix.shape[1],
        row_weights=weights,
        lambda2={f"{group.name}:wiggle": 1.0},
        S_override=override,
    )
    assert forced.use_structured


def test_sz_lambda_resolution_never_runs_local_feasibility_scan(monkeypatch) -> None:
    n_levels = 20
    x = np.tile(np.linspace(-1.0, 1.0, 6), n_levels)
    matrix, group = _selection_factor_smooth_matrix(
        factor_basis="sz",
        n_levels=n_levels,
        local_basis=np.column_stack((np.ones_like(x), x)),
        repeated_penalty_components=(("wiggle", np.diag([0.0, 1.0])),),
    )

    monkeypatch.setattr(
        FactorSmoothGroupMatrix,
        "factor_smooth_sufficient_stats",
        lambda *_args, **_kwargs: pytest.fail("SZ resolution scanned all training rows"),
    )
    decision = resolve_structured_backend(
        [matrix],
        [group],
        direct_solve="auto",
        coefficient_width=matrix.shape[1],
        row_weights=np.ones(matrix.shape[0]),
        lambda2={f"{group.name}:wiggle": 1.0},
    )

    assert decision.use_structured


def test_authoritative_sz_override_uses_global_tiny_weight_rank() -> None:
    dm, groups, penalties, y, weights, offset, lambdas = _factor_smooth_problem(
        _gaussian_response,
        factor_basis="sz",
    )
    matrix = dm.group_matrices[3]
    assert isinstance(matrix, FactorSmoothGroupMatrix)
    weights = np.array(weights, copy=True)
    weights[matrix.codes == matrix.n_levels - 1] = 1.0e-20
    override = _factor_smooth_override(
        dm,
        groups,
        penalties,
        lambdas,
        local_penalty=None,
    )

    automatic, _ = irls_direct.fit_irls_direct(
        X=dm,
        y=y,
        weights=weights,
        family=Gaussian(),
        link=IdentityLink(),
        groups=groups,
        lambda2=lambdas,
        offset=offset,
        direct_solve="auto",
        S_override=override,
        tol=1.0e-10,
    )
    gram, _ = irls_direct.fit_irls_direct(
        X=dm,
        y=y,
        weights=weights,
        family=Gaussian(),
        link=IdentityLink(),
        groups=groups,
        lambda2=lambdas,
        offset=offset,
        direct_solve="gram",
        S_override=override,
        tol=1.0e-10,
    )

    assert automatic.direct_backend == "gram"
    assert "globally unidentifiable" in automatic.direct_fallback_reason
    np.testing.assert_allclose(automatic.beta, gram.beta, atol=3.0e-8)
    with pytest.raises(np.linalg.LinAlgError, match="globally unidentifiable"):
        irls_direct.fit_irls_direct(
            X=dm,
            y=y,
            weights=weights,
            family=Gaussian(),
            link=IdentityLink(),
            groups=groups,
            lambda2=lambdas,
            offset=offset,
            direct_solve="structured",
            S_override=override,
            tol=1.0e-10,
        )


def test_factor_smooth_feasibility_cache_includes_lambda_scales() -> None:
    n_levels = 20
    local_basis = np.tile(
        np.array([[0.0, 1.0], [0.0, 2.0]]),
        (n_levels, 1),
    )
    matrix, group = _selection_factor_smooth_matrix(
        factor_basis="fs",
        n_levels=n_levels,
        local_basis=local_basis,
        repeated_penalty_components=(("wiggle", np.diag([1.0, 0.0])),),
    )
    weights = np.ones(matrix.shape[0])
    moderate = resolve_structured_backend(
        [matrix],
        [group],
        direct_solve="auto",
        coefficient_width=matrix.shape[1],
        row_weights=weights,
        lambda2={f"{group.name}:wiggle": 1.0},
    )
    tiny = resolve_structured_backend(
        [matrix],
        [group],
        direct_solve="auto",
        coefficient_width=matrix.shape[1],
        row_weights=weights,
        lambda2={f"{group.name}:wiggle": 1.0e-20},
    )

    assert moderate.use_structured
    assert not tiny.use_structured
    assert "singular local block" in tiny.fallback_reason
    with pytest.raises(ValueError, match="singular local block"):
        resolve_structured_backend(
            [matrix],
            [group],
            direct_solve="structured",
            coefficient_width=matrix.shape[1],
            row_weights=weights,
            lambda2={f"{group.name}:wiggle": 1.0e-20},
        )


@pytest.mark.parametrize("factor_basis", ["fs", "sz"])
def test_authoritative_factor_smooth_override_supersedes_stale_zero_lambdas(
    factor_basis: str,
) -> None:
    dm, groups, penalties, y, weights, offset, lambdas = _factor_smooth_problem(
        _gaussian_response,
        factor_basis=factor_basis,
    )
    matrix = dm.group_matrices[3]
    assert isinstance(matrix, FactorSmoothGroupMatrix)
    weights = np.array(weights, copy=True)
    weights[matrix.codes == matrix.n_levels - 1] = 0.0
    override = _factor_smooth_override(
        dm,
        groups,
        penalties,
        lambdas,
        local_penalty=np.eye(matrix.block_size),
    )
    stale_lambdas = dict(lambdas)
    for component in penalties:
        if component.group_name == groups[3].name:
            stale_lambdas[component.name] = 0.0

    automatic, _ = irls_direct.fit_irls_direct(
        X=dm,
        y=y,
        weights=weights,
        family=Gaussian(),
        link=IdentityLink(),
        groups=groups,
        lambda2=stale_lambdas,
        offset=offset,
        direct_solve="auto",
        S_override=override,
        tol=1.0e-10,
    )
    gram, _ = irls_direct.fit_irls_direct(
        X=dm,
        y=y,
        weights=weights,
        family=Gaussian(),
        link=IdentityLink(),
        groups=groups,
        lambda2=stale_lambdas,
        offset=offset,
        direct_solve="gram",
        S_override=override,
        tol=1.0e-10,
    )

    assert automatic.direct_backend == "structured"
    assert automatic.direct_fallback_reason is None
    np.testing.assert_allclose(automatic.beta, gram.beta, atol=3.0e-8)


def test_authoritative_singular_factor_smooth_override_falls_back_to_gram() -> None:
    dm, groups, penalties, y, weights, offset, lambdas = _factor_smooth_problem(
        _gaussian_response,
        factor_basis="fs",
    )
    matrix = dm.group_matrices[3]
    assert isinstance(matrix, FactorSmoothGroupMatrix)
    weights = np.array(weights, copy=True)
    weights[matrix.codes == matrix.n_levels - 1] = 0.0
    override = _factor_smooth_override(
        dm,
        groups,
        penalties,
        lambdas,
        local_penalty=None,
    )

    automatic, _ = irls_direct.fit_irls_direct(
        X=dm,
        y=y,
        weights=weights,
        family=Gaussian(),
        link=IdentityLink(),
        groups=groups,
        lambda2=lambdas,
        offset=offset,
        direct_solve="auto",
        S_override=override,
        tol=1.0e-10,
    )
    gram, _ = irls_direct.fit_irls_direct(
        X=dm,
        y=y,
        weights=weights,
        family=Gaussian(),
        link=IdentityLink(),
        groups=groups,
        lambda2=lambdas,
        offset=offset,
        direct_solve="gram",
        S_override=override,
        tol=1.0e-10,
    )

    assert automatic.direct_backend == "gram"
    assert "authoritative S_override" in automatic.direct_fallback_reason
    np.testing.assert_allclose(automatic.beta, gram.beta, atol=3.0e-8)
    with pytest.raises(
        ValueError,
        match=r"direct_solve='structured'.*authoritative S_override",
    ):
        irls_direct.fit_irls_direct(
            X=dm,
            y=y,
            weights=weights,
            family=Gaussian(),
            link=IdentityLink(),
            groups=groups,
            lambda2=lambdas,
            offset=offset,
            direct_solve="structured",
            S_override=override,
            tol=1.0e-10,
        )


def test_sz_locally_singular_but_globally_identifiable_remains_structured() -> None:
    n_levels = 20
    support = [np.array([0.0, 0.0])] + [np.array([-1.0, 0.0, 1.0]) for _ in range(n_levels - 1)]
    x = np.concatenate(support)
    codes = np.concatenate(
        [
            np.full(len(level_support), level, dtype=np.intp)
            for level, level_support in enumerate(support)
        ]
    )
    basis = np.column_stack((np.ones_like(x), x))
    matrix = FactorSmoothGroupMatrix(
        sp.csr_matrix(basis),
        codes,
        n_levels,
        natural_map=np.eye(2),
        levels=tuple(f"level-{level}" for level in range(n_levels)),
        repeated_penalty_components=(("wiggle", np.zeros((2, 2))),),
        factor_basis="sz",
    )
    group = GroupSlice(
        name="x:group:sz",
        start=0,
        end=matrix.shape[1],
        penalized=True,
    )
    dm = DesignMatrix([matrix], n=len(x), p=matrix.shape[1])
    weights = np.ones(len(x))
    offset = 0.08 * np.cos(np.linspace(0.0, 2.0 * np.pi, len(x)))
    y = 0.3 + 0.2 * x + np.linspace(-0.1, 0.1, len(x)) + offset
    override = np.zeros((dm.p, dm.p))
    lambdas = {f"{group.name}:wiggle": 0.0}

    automatic, _ = irls_direct.fit_irls_direct(
        X=dm,
        y=y,
        weights=weights,
        family=Gaussian(),
        link=IdentityLink(),
        groups=[group],
        lambda2=lambdas,
        offset=offset,
        direct_solve="auto",
        S_override=override,
        tol=1.0e-10,
    )
    gram, _ = irls_direct.fit_irls_direct(
        X=dm,
        y=y,
        weights=weights,
        family=Gaussian(),
        link=IdentityLink(),
        groups=[group],
        lambda2=lambdas,
        offset=offset,
        direct_solve="gram",
        S_override=override,
        tol=1.0e-10,
    )

    assert automatic.direct_backend == "structured"
    assert automatic.direct_fallback_reason is None
    np.testing.assert_allclose(automatic.beta, gram.beta, atol=3.0e-8)
    np.testing.assert_allclose(automatic.intercept, gram.intercept, atol=3.0e-8)


@pytest.mark.parametrize("factor_basis", ["fs", "sz"])
def test_authoritative_factor_smooth_override_roundoff_asymmetry_matches_gram(
    factor_basis: str,
) -> None:
    dm, groups, penalties, y, weights, offset, lambdas = _factor_smooth_problem(
        _gaussian_response,
        factor_basis=factor_basis,
    )
    matrix = dm.group_matrices[3]
    assert isinstance(matrix, FactorSmoothGroupMatrix)
    local_penalty = 2.0 * np.eye(matrix.block_size)
    local_penalty[0, 1] = 0.2
    local_penalty[1, 0] = 0.2 + 1.0e-10
    override = _factor_smooth_override(
        dm,
        groups,
        penalties,
        lambdas,
        local_penalty=local_penalty,
    )

    automatic, _ = irls_direct.fit_irls_direct(
        X=dm,
        y=y,
        weights=weights,
        family=Gaussian(),
        link=IdentityLink(),
        groups=groups,
        lambda2=lambdas,
        offset=offset,
        direct_solve="auto",
        S_override=override,
        tol=1.0e-10,
    )
    gram, _ = irls_direct.fit_irls_direct(
        X=dm,
        y=y,
        weights=weights,
        family=Gaussian(),
        link=IdentityLink(),
        groups=groups,
        lambda2=lambdas,
        offset=offset,
        direct_solve="gram",
        S_override=override,
        tol=1.0e-10,
    )

    assert automatic.direct_backend == "structured"
    assert automatic.direct_fallback_reason is None
    np.testing.assert_allclose(automatic.beta, gram.beta, atol=3.0e-8)


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


@pytest.mark.parametrize(
    ("basis", "fallback_reason"),
    [
        ("fs", "zero penalty component"),
        ("sz", "globally unidentifiable"),
    ],
)
@pytest.mark.parametrize("discrete", [False, True])
def test_auto_factor_smooth_falls_back_for_unsupported_local_geometry(
    basis: str,
    fallback_reason: str,
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
    offset = 0.07 * np.cos(np.linspace(0.0, 3.0 * np.pi, len(x)))
    y += offset
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
    model_kwargs = dict(
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
        discrete=discrete,
        n_bins=64,
    )
    model = SuperGLM(**model_kwargs, direct_solve="auto").fit_reml(
        X,
        y,
        sample_weight=sample_weight,
        offset=offset,
        runtime_validation="skip",
    )

    assert model.result.direct_backend == "gram"
    assert fallback_reason in model.result.direct_fallback_reason
    if basis == "sz":
        gram = SuperGLM(**model_kwargs, direct_solve="gram").fit_reml(
            X,
            y,
            sample_weight=sample_weight,
            offset=offset,
            runtime_validation="skip",
        )
        np.testing.assert_allclose(model.predict(X), gram.predict(X), atol=2.0e-8)
        with pytest.raises(np.linalg.LinAlgError, match="globally unidentifiable"):
            SuperGLM(**model_kwargs, direct_solve="structured").fit_reml(
                X,
                y,
                sample_weight=sample_weight,
                offset=offset,
                runtime_validation="skip",
            )


@pytest.mark.parametrize("discrete", [False, True])
def test_reml_latches_runtime_sz_fallback_after_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
    discrete: bool,
) -> None:
    dm, groups, penalties, y, weights, offset, lambdas = _factor_smooth_problem(
        _gaussian_response,
        factor_basis="sz",
    )
    reml_groups = [
        (group_index, group) for group_index, group in enumerate(groups) if group.penalized
    ]
    penalty_ranks = {penalty.name: penalty.rank for penalty in penalties}
    optimizer_module = discrete_reml if discrete else direct_reml
    original_fit = optimizer_module.fit_irls_direct
    direct_modes: list[str] = []
    fallback_reason = "synthetic globally unidentifiable SZ candidate"

    def delayed_runtime_fallback(*args, **kwargs):
        direct_modes.append(kwargs["direct_solve"])
        np.testing.assert_allclose(kwargs["offset"], offset)
        if len(direct_modes) == 2:
            gram_kwargs = dict(kwargs)
            gram_kwargs["direct_solve"] = "gram"
            gram_kwargs["S_override"] = None
            result = original_fit(*args, **gram_kwargs)
            result[0].direct_fallback_reason = fallback_reason
            return result
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(
        optimizer_module,
        "fit_irls_direct",
        delayed_runtime_fallback,
    )
    result = direct_reml.optimize_direct_reml(
        dm=dm,
        distribution=Gaussian(),
        link=IdentityLink(),
        groups=groups,
        discrete=discrete,
        y=y,
        sample_weight=weights,
        offset_arr=offset,
        reml_groups=reml_groups,
        penalty_ranks=penalty_ranks,
        lambdas=lambdas,
        max_reml_iter=1,
        reml_tol=1.0e-5,
        verbose=False,
        direct_solve="auto",
        reml_penalties=penalties,
        pirls_tol=1.0e-9,
    )

    assert direct_modes[:2] == ["auto", "auto"]
    assert all(mode == "gram" for mode in direct_modes[2:])
    assert result.pirls_result.direct_backend == "gram"
    assert result.pirls_result.direct_fallback_reason == fallback_reason


def test_zero_penalty_fs_falls_back_or_rejects_but_gram_fits() -> None:
    rng = np.random.default_rng(20260727)
    n_levels = 10
    repeats = 16
    codes = np.repeat(np.arange(n_levels), repeats)
    x = np.tile(np.linspace(-1.0, 1.0, repeats), n_levels)
    X = pd.DataFrame(
        {
            "x": x,
            "group": np.array([f"g{code}" for code in codes], dtype=object),
        }
    )
    y = np.sin(2.5 * x) + rng.normal(scale=0.05, size=len(x))
    policies = {
        "wiggle": LambdaPolicy.off(),
        "null_0": LambdaPolicy.off(),
        "null_1": LambdaPolicy.off(),
    }
    common = {
        "family": "gaussian",
        "features": {"x": Spline(k=5, lambda_policy=LambdaPolicy.fixed(1.0))},
        "interactions": [
            FactorSmooth(
                "x",
                group="group",
                basis="fs",
                k=5,
                lambda_policy=policies,
            )
        ],
        "selection_penalty": 0.0,
    }

    automatic = SuperGLM(**common, direct_solve="auto").fit_reml(
        X,
        y,
        runtime_validation="skip",
    )
    gram = SuperGLM(**common, direct_solve="gram").fit_reml(
        X,
        y,
        runtime_validation="skip",
    )

    assert automatic.result.direct_backend == "gram"
    assert "zero penalty component" in automatic.result.direct_fallback_reason
    np.testing.assert_allclose(automatic.predict(X), gram.predict(X), atol=1.0e-8)
    with pytest.raises(
        ValueError,
        match=r"direct_solve='structured'.*zero penalty component",
    ):
        SuperGLM(**common, direct_solve="structured").fit_reml(
            X,
            y,
            runtime_validation="skip",
        )


def test_zero_penalty_sz_wiggle_remains_structured_and_matches_gram() -> None:
    rng = np.random.default_rng(20260727)
    n_levels = 10
    repeats = 16
    codes = np.repeat(np.arange(n_levels), repeats)
    x = np.tile(np.linspace(-1.0, 1.0, repeats), n_levels)
    deviations = rng.normal(scale=0.15, size=n_levels)
    deviations -= np.mean(deviations)
    X = pd.DataFrame(
        {
            "x": x,
            "group": np.array([f"g{code}" for code in codes], dtype=object),
        }
    )
    y = np.sin(2.5 * x) + deviations[codes] * x + rng.normal(scale=0.05, size=len(x))
    common = {
        "family": "gaussian",
        "features": {"x": Spline(k=5, lambda_policy=LambdaPolicy.fixed(1.0))},
        "interactions": [
            FactorSmooth(
                "x",
                group="group",
                basis="sz",
                k=5,
                lambda_policy={"wiggle": LambdaPolicy.off()},
            )
        ],
        "selection_penalty": 0.0,
    }

    structured = SuperGLM(**common, direct_solve="structured").fit_reml(
        X,
        y,
        runtime_validation="skip",
    )
    gram = SuperGLM(**common, direct_solve="gram").fit_reml(
        X,
        y,
        runtime_validation="skip",
    )

    assert structured.result.direct_backend == "structured"
    np.testing.assert_allclose(
        structured.predict(X),
        gram.predict(X),
        rtol=1.0e-7,
        atol=1.0e-8,
    )


def _multi_structured_data() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(20260726)
    n = 600
    g1 = np.arange(n) % 8
    g2 = np.arange(n) % 6
    re = np.arange(n) % 50
    x1 = rng.uniform(-1.0, 1.0, n)
    x2 = rng.uniform(-1.0, 1.0, n)
    y = 0.4 + 0.3 * x1 - 0.2 * x2 + rng.normal(scale=0.15, size=n)
    X = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "g1": np.array([f"a-{value}" for value in g1], dtype=object),
            "g2": np.array([f"b-{value}" for value in g2], dtype=object),
            "re": np.array([f"r-{value}" for value in re], dtype=object),
        }
    )
    return X, y


def test_two_factor_smooths_auto_fall_back_and_forced_structured_rejects():
    X, y = _multi_structured_data()

    def model(mode: str) -> SuperGLM:
        return SuperGLM(
            family="gaussian",
            features={},
            interactions=[
                FactorSmooth(
                    "x1",
                    group="g1",
                    k=5,
                    lambda_policy=LambdaPolicy.fixed(1.0),
                ),
                FactorSmooth(
                    "x2",
                    group="g2",
                    k=5,
                    lambda_policy=LambdaPolicy.fixed(1.0),
                ),
            ],
            selection_penalty=0.0,
            direct_solve=mode,
        )

    gram = model("gram").fit_reml(X, y, runtime_validation="skip")
    auto = model("auto").fit_reml(X, y, runtime_validation="skip")

    assert auto.result.direct_backend == "gram"
    assert "at most one FactorSmooth" in auto.result.direct_fallback_reason
    np.testing.assert_allclose(auto.predict(X), gram.predict(X), atol=2e-8)

    with pytest.raises(ValueError, match="at most one FactorSmooth"):
        model("structured").fit_reml(X, y, runtime_validation="skip")


@pytest.mark.parametrize("basis", ["fs", "sz"])
def test_single_factor_smooth_dominates_wider_random_effect(basis: str):
    X, y = _multi_structured_data()

    def model(mode: str) -> SuperGLM:
        features = {
            "re": RandomEffect(lambda_policy=LambdaPolicy.fixed(1.1)),
        }
        if basis == "sz":
            features["x1"] = Spline(
                n_knots=5,
                lambda_policy=LambdaPolicy.fixed(1.2),
            )
        return SuperGLM(
            family="gaussian",
            features=features,
            interactions=[
                FactorSmooth(
                    "x1",
                    group="g1",
                    basis=basis,
                    k=5,
                    lambda_policy=LambdaPolicy.fixed(0.9),
                )
            ],
            selection_penalty=0.0,
            direct_solve=mode,
        )

    gram = model("gram").fit_reml(X, y, runtime_validation="skip")
    auto = model("auto").fit_reml(X, y, runtime_validation="skip")
    structured = model("structured").fit_reml(X, y, runtime_validation="skip")

    # Selection still nominates the single FactorSmooth as the dominant block
    # over the wider RandomEffect -- forced structured factors around it -- but
    # for `auto` this shape is now on the dense side of the measured crossover
    # (issue #343): the whole RandomEffect sits in the dense border, so the
    # elimination removes a minority of the width and was measured slower than
    # the dense path on shapes like it.  The recorded reason proves both
    # halves: FactorSmooth won the dominance choice, and cost declined it.
    assert auto.result.direct_backend == "gram"
    assert "FactorSmooth" in auto.result.direct_fallback_reason
    assert "crossover" in auto.result.direct_fallback_reason
    assert structured.result.direct_backend == "structured"
    np.testing.assert_allclose(auto.predict(X), gram.predict(X), atol=3e-8)
    np.testing.assert_allclose(structured.predict(X), gram.predict(X), atol=3e-8)
    assert auto.result.deviance == pytest.approx(gram.result.deviance, abs=2e-8)
    assert structured.result.deviance == pytest.approx(gram.result.deviance, abs=2e-8)
