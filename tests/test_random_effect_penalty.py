"""Implicit identity-penalty tests for random effects."""

import numpy as np
import pytest

from superglm import LambdaPolicy
from superglm.group_matrix import RandomEffectGroupMatrix
from superglm.model.reml_setup import collect_reml_groups
from superglm.reml import penalty_algebra
from superglm.reml.penalty_algebra import (
    build_penalty_components,
    build_penalty_matrix,
)
from superglm.types import GroupSlice


def test_random_effect_builds_full_rank_implicit_identity_component():
    policy = LambdaPolicy.fixed(2.5)
    gm = RandomEffectGroupMatrix(
        np.array([0, 1, 2, 1], dtype=np.intp),
        n_levels=3,
        lambda_policies={"_default": policy},
    )
    group = GroupSlice(name="broker", start=0, end=3)

    reml_groups = collect_reml_groups([group], [gm])
    components = build_penalty_components([gm], reml_groups)

    assert reml_groups == [(0, group)]
    assert len(components) == 1
    component = components[0]
    assert component.name == "broker"
    assert component.penalty_kind == "identity"
    assert component.omega_raw is None
    assert component.omega_ssp is None
    assert component.rank == 3.0
    assert component.log_det_omega_plus == 0.0
    assert component.eigvals_omega is None
    assert component.lambda_policy is policy


def test_dense_oracle_fills_random_effect_penalty_diagonal_without_identity_matrix(monkeypatch):
    gm = RandomEffectGroupMatrix(np.array([0, 1, 2, 1], dtype=np.intp), n_levels=3)
    group = GroupSlice(name="broker", start=0, end=3)
    components = build_penalty_components([gm], collect_reml_groups([group], [gm]))
    original_eye = np.eye

    def reject_random_effect_identity(n, *args, **kwargs):
        if n == 3:
            raise AssertionError("random-effect identity must stay implicit")
        return original_eye(n, *args, **kwargs)

    monkeypatch.setattr(np, "eye", reject_random_effect_identity)

    penalty = build_penalty_matrix(
        [gm],
        [group],
        {"broker": 2.0},
        p=3,
        reml_penalties=components,
    )

    np.testing.assert_array_equal(penalty, np.diag([2.0, 2.0, 2.0]))


def test_identity_penalty_helpers_use_vector_and_diagonal_algebra():
    gm = RandomEffectGroupMatrix(np.array([0, 1, 2, 1], dtype=np.intp), n_levels=3)
    group = GroupSlice(name="broker", start=0, end=3)
    component = build_penalty_components(
        [gm],
        collect_reml_groups([group], [gm]),
    )[0]
    beta = np.array([0.5, -0.25, 1.0])
    inverse_block = np.array(
        [
            [0.4, 0.01, -0.02],
            [0.01, 0.3, 0.03],
            [-0.02, 0.03, 0.2],
        ]
    )

    assert penalty_algebra.penalty_component_quadratic(component, beta) == pytest.approx(
        beta @ beta
    )
    np.testing.assert_array_equal(
        penalty_algebra.penalty_component_matvec(component, beta),
        beta,
    )
    assert penalty_algebra.penalty_component_trace(component, inverse_block) == pytest.approx(
        np.trace(inverse_block)
    )
    assert penalty_algebra.penalty_component_trace(
        component, np.diag(inverse_block)
    ) == pytest.approx(np.trace(inverse_block))
    assert penalty_algebra.total_penalty_quadratic(
        beta,
        {"broker": 2.0},
        [component],
        [gm],
    ) == pytest.approx(2.0 * (beta @ beta))


def test_identity_penalty_metadata_never_eigendecomposes_identity(monkeypatch):
    gm = RandomEffectGroupMatrix(np.array([0, 1, 2, 1], dtype=np.intp), n_levels=3)
    group = GroupSlice(name="broker", start=0, end=3)

    def reject_eigendecomposition(*args, **kwargs):
        raise AssertionError("random-effect identity must not be eigendecomposed")

    monkeypatch.setattr(np.linalg, "eigvalsh", reject_eigendecomposition)

    component = build_penalty_components(
        [gm],
        collect_reml_groups([group], [gm]),
    )[0]

    assert component.rank == 3.0
    assert component.log_det_omega_plus == 0.0
