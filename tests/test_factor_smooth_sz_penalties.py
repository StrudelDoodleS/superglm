"""Shared-penalty algebra for sum-to-zero factor smooths."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from superglm.factor_smooth_geometry import sum_to_zero_contrast
from superglm.group_matrix import FactorSmoothGroupMatrix
from superglm.model.reml_setup import collect_reml_groups
from superglm.reml.penalty_algebra import (
    build_penalty_components,
    build_penalty_matrix,
    compute_logdet_s_derivatives,
    compute_logdet_s_plus,
    penalty_component_dense_matrix,
    penalty_component_matvec,
    penalty_component_quadratic,
    penalty_component_trace,
)
from superglm.solvers.hessian_factor import DenseHessianFactor
from superglm.types import GroupSlice


def _sz_penalty_fixture():
    n_levels = 4
    block_size = 6
    basis = sp.csr_matrix(np.eye(block_size)[np.arange(24) % block_size])
    codes = np.arange(24, dtype=np.intp) % n_levels
    wiggle = np.diag([2.0, 0.8, 0.3, 0.1, 0.0, 0.0])
    gm = FactorSmoothGroupMatrix(
        basis,
        codes,
        n_levels,
        natural_map=np.eye(block_size),
        levels=("a", "b", "c", "d"),
        repeated_penalty_components=(("wiggle", wiggle),),
        factor_basis="sz",
    )
    group = GroupSlice(
        name="x:group:sz",
        start=0,
        end=(n_levels - 1) * block_size,
    )
    components = build_penalty_components(
        [gm],
        collect_reml_groups([group], [gm]),
    )
    return gm, group, wiggle, components


def _dense_oracle(gm: FactorSmoothGroupMatrix, wiggle: np.ndarray) -> np.ndarray:
    contrast = sum_to_zero_contrast(gm.n_levels)
    return np.kron(contrast.T @ contrast, wiggle)


def test_sz_builds_one_shared_compact_penalty_with_exact_spectrum() -> None:
    gm, group, wiggle, components = _sz_penalty_fixture()

    assert len(components) == 1
    component = components[0]
    assert component.name == "x:group:sz:wiggle"
    assert component.group_sl == group.sl
    assert component.penalty_kind == "sum_to_zero"
    assert component.repeat_count == gm.n_levels
    assert component.block_width == gm.block_size
    assert component.omega_raw.shape == (gm.block_size, gm.block_size)
    assert component.omega_ssp.shape == (gm.block_size, gm.block_size)
    assert component.omega_ssp.shape != (group.size, group.size)

    local_positive = np.linalg.eigvalsh(wiggle)
    local_positive = local_positive[local_positive > 0]
    expected_eigenvalues = np.sort(
        np.concatenate(
            (
                np.tile(local_positive, gm.n_levels - 2),
                gm.n_levels * local_positive,
            )
        )
    )[::-1]
    assert component.rank == float((gm.n_levels - 1) * len(local_positive))
    assert component.log_det_omega_plus == pytest.approx(
        (gm.n_levels - 1) * np.log(local_positive).sum() + len(local_positive) * np.log(gm.n_levels)
    )
    np.testing.assert_allclose(component.eigvals_omega, expected_eigenvalues)

    dense = penalty_component_dense_matrix(component, gm)
    dense_positive = np.linalg.eigvalsh(dense)
    dense_positive = dense_positive[dense_positive > 1e-12]
    np.testing.assert_allclose(dense_positive[::-1], expected_eigenvalues)
    assert np.log(dense_positive).sum() == pytest.approx(component.log_det_omega_plus)


def test_sz_compact_vector_and_trace_algebra_matches_dense_oracle() -> None:
    gm, _group, wiggle, components = _sz_penalty_fixture()
    component = components[0]
    omega = _dense_oracle(gm, wiggle)
    rng = np.random.default_rng(423)
    beta = rng.normal(size=gm.shape[1])
    inverse_block = rng.normal(size=(gm.shape[1], gm.shape[1]))

    assert penalty_component_quadratic(component, beta, gm) == pytest.approx(beta @ omega @ beta)
    np.testing.assert_allclose(
        penalty_component_matvec(component, beta, gm),
        omega @ beta,
    )
    assert penalty_component_trace(component, inverse_block, gm) == pytest.approx(
        np.trace(inverse_block @ omega)
    )


def test_sz_dense_penalty_and_single_lambda_reml_algebra_match_oracle() -> None:
    gm, group, wiggle, components = _sz_penalty_fixture()
    component = components[0]
    lam = 1.7
    lambdas = {component.name: lam}
    omega = _dense_oracle(gm, wiggle)

    actual = build_penalty_matrix(
        [gm],
        [group],
        lambdas,
        gm.shape[1],
        reml_penalties=components,
    )
    np.testing.assert_allclose(actual, lam * omega)
    assert compute_logdet_s_plus(lambdas, components) == pytest.approx(
        component.rank * np.log(lam) + component.log_det_omega_plus
    )
    gradient, hessian = compute_logdet_s_derivatives(lambdas, components)
    assert gradient == {component.name: component.rank}
    assert hessian == {(component.name, component.name): 0.0}


def test_dense_hessian_factor_sz_traces_match_dense_oracle() -> None:
    gm, _group, wiggle, components = _sz_penalty_fixture()
    component = components[0]
    omega = _dense_oracle(gm, wiggle)
    rng = np.random.default_rng(823)
    root = rng.normal(size=(gm.shape[1], gm.shape[1]))
    hessian = root.T @ root + np.eye(gm.shape[1])
    inverse = np.linalg.inv(hessian)
    factor = DenseHessianFactor(inverse=inverse, log_det=np.linalg.slogdet(hessian)[1])

    assert factor.trace_inverse_penalty(component) == pytest.approx(np.trace(inverse @ omega))
    assert factor.penalty_cross_trace(component, component, 1.3, 0.7) == pytest.approx(
        1.3 * 0.7 * np.trace(inverse @ omega @ inverse @ omega)
    )


@pytest.mark.parametrize(
    "components",
    [
        (("null", np.eye(4)),),
        (("wiggle", np.eye(4)), ("extra", np.eye(4))),
    ],
)
def test_sz_rejects_noncanonical_penalty_component_geometry(components) -> None:
    gm = FactorSmoothGroupMatrix(
        sp.eye(8, 4, format="csr"),
        np.arange(8, dtype=np.intp) % 2,
        2,
        natural_map=np.eye(4),
        levels=("a", "b"),
        repeated_penalty_components=components,
        factor_basis="sz",
    )
    group = GroupSlice(name="x:g:sz", start=0, end=4)

    with pytest.raises(ValueError, match="exactly one 'wiggle' component"):
        build_penalty_components([gm], collect_reml_groups([group], [gm]))
