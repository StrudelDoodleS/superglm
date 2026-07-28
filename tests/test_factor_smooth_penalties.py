"""Repeated-penalty algebra for compact factor smooths."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from superglm.group_matrix import FactorSmoothGroupMatrix
from superglm.model.reml_setup import collect_reml_groups
from superglm.reml.penalty_algebra import (
    build_penalty_components,
    build_penalty_matrix,
    compute_logdet_s_derivatives,
    compute_logdet_s_plus,
    compute_total_penalty_rank,
    penalty_component_matvec,
    penalty_component_quadratic,
    penalty_component_trace,
)
from superglm.solvers.hessian_factor import DenseHessianFactor
from superglm.types import GroupSlice


def _penalty_fixture():
    n_levels = 4
    block_size = 5
    basis = sp.csr_matrix(np.eye(block_size)[np.arange(20) % block_size])
    codes = np.arange(20, dtype=np.intp) % n_levels
    wiggle = np.diag([2.0, 0.8, 0.3, 0.0, 0.0])
    null_0 = np.diag([0.0, 0.0, 0.0, 1.0, 0.0])
    null_1 = np.diag([0.0, 0.0, 0.0, 0.0, 1.0])
    gm = FactorSmoothGroupMatrix(
        basis,
        codes,
        n_levels,
        natural_map=np.eye(block_size),
        levels=("a", "b", "c", "d"),
        repeated_penalty_components=(
            ("wiggle", wiggle),
            ("null_0", null_0),
            ("null_1", null_1),
        ),
    )
    group = GroupSlice(name="x:group:fs", start=0, end=n_levels * block_size)
    components = build_penalty_components(
        [gm],
        collect_reml_groups([group], [gm]),
    )
    return gm, group, components


def _expanded(component) -> np.ndarray:
    return np.kron(np.eye(component.repeat_count), component.omega_ssp)


def test_factor_smooth_builds_repeated_components_without_full_penalty_matrices() -> None:
    gm, group, components = _penalty_fixture()

    assert collect_reml_groups([group], [gm]) == [(0, group)]
    assert [component.name for component in components] == [
        "x:group:fs:wiggle",
        "x:group:fs:null_0",
        "x:group:fs:null_1",
    ]
    assert [component.rank for component in components] == [12.0, 4.0, 4.0]
    for component in components:
        assert component.penalty_kind == "repeated"
        assert component.repeat_count == 4
        assert component.block_width == 5
        assert component.omega_raw.shape == (5, 5)
        assert component.omega_ssp.shape == (5, 5)
        assert component.omega_ssp.shape != (group.size, group.size)


def test_repeated_penalty_vector_algebra_matches_explicit_kronecker_reference() -> None:
    gm, _group, components = _penalty_fixture()
    rng = np.random.default_rng(811)
    beta = rng.normal(size=gm.shape[1])
    inverse_block = rng.normal(size=(gm.shape[1], gm.shape[1]))

    for component in components:
        omega = _expanded(component)
        assert penalty_component_quadratic(component, beta, gm) == pytest.approx(
            beta @ omega @ beta
        )
        np.testing.assert_allclose(
            penalty_component_matvec(component, beta, gm),
            omega @ beta,
        )
        assert penalty_component_trace(component, inverse_block, gm) == pytest.approx(
            np.trace(inverse_block @ omega)
        )


def test_repeated_dense_penalty_oracle_matches_explicit_kronecker_sum() -> None:
    gm, group, components = _penalty_fixture()
    lambdas = {
        "x:group:fs:wiggle": 1.7,
        "x:group:fs:null_0": 0.4,
        "x:group:fs:null_1": 2.2,
    }

    actual = build_penalty_matrix(
        [gm],
        [group],
        lambdas,
        gm.shape[1],
        reml_penalties=components,
    )
    expected = sum(lambdas[component.name] * _expanded(component) for component in components)

    np.testing.assert_allclose(actual, expected)


def test_repeated_joint_logdet_rank_gradient_and_hessian_match_dense_reference() -> None:
    _gm, _group, components = _penalty_fixture()
    lambdas = {
        "x:group:fs:wiggle": 1.7,
        "x:group:fs:null_0": 0.4,
        "x:group:fs:null_1": 2.2,
    }
    full_components = [_expanded(component) for component in components]
    penalty = sum(
        lambdas[component.name] * omega for component, omega in zip(components, full_components)
    )
    sign, expected_logdet = np.linalg.slogdet(penalty)
    assert sign > 0

    actual_logdet = compute_logdet_s_plus(lambdas, components)
    gradient, hessian = compute_logdet_s_derivatives(lambdas, components)

    inverse = np.linalg.inv(penalty)
    expected_gradient: dict[str, float] = {}
    expected_hessian: dict[tuple[str, str], float] = {}
    scaled = []
    for component, omega in zip(components, full_components):
        derivative = lambdas[component.name] * omega
        scaled.append(derivative)
        expected_gradient[component.name] = float(np.trace(inverse @ derivative))
    for left, left_derivative in zip(components, scaled):
        for right, right_derivative in zip(components, scaled):
            value = -float(np.trace(inverse @ left_derivative @ inverse @ right_derivative))
            if left.name == right.name:
                value += expected_gradient[left.name]
            expected_hessian[(left.name, right.name)] = value

    assert actual_logdet == pytest.approx(expected_logdet)
    assert compute_total_penalty_rank(components) == float(penalty.shape[0])
    assert gradient == pytest.approx(expected_gradient)
    assert hessian == pytest.approx(expected_hessian)


def test_dense_hessian_factor_repeated_traces_match_explicit_reference() -> None:
    gm, _group, components = _penalty_fixture()
    rng = np.random.default_rng(929)
    root = rng.normal(size=(gm.shape[1], gm.shape[1]))
    hessian = root.T @ root + np.eye(gm.shape[1])
    inverse = np.linalg.inv(hessian)
    factor = DenseHessianFactor(inverse=inverse, log_det=np.linalg.slogdet(hessian)[1])

    for component in components:
        assert factor.trace_inverse_penalty(component) == pytest.approx(
            np.trace(inverse @ _expanded(component))
        )
    for left in components:
        for right in components:
            expected = 1.3 * 0.7 * np.trace(inverse @ _expanded(left) @ inverse @ _expanded(right))
            assert factor.penalty_cross_trace(left, right, 1.3, 0.7) == pytest.approx(expected)
