"""Regression tests for factor-authoritative cutoff-boundary geometry."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from superglm.distributions import Gamma, Gaussian
from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
from superglm.inference._metrics_design import iter_dense_chunks
from superglm.inference.metrics import (
    _certified_coefficient_rank,
    _certified_data_rank,
    _certified_profile_rank,
)
from superglm.links import IdentityLink, LogLink
from superglm.model.state_ops import _legacy_active_state
from superglm.penalties.group_lasso import GroupLasso
from superglm.penalties.ridge import Ridge
from superglm.reml.observed_geometry import build_observed_reml_geometry
from superglm.reml.scop_efs import _certified_terminal_rank
from superglm.reml.scop_geometry import build_cached_scop_joint_geometry
from superglm.solvers.centered_system import (
    grouped_augmented_factor,
    grouped_weighted_factor,
    penalty_factor,
)
from superglm.solvers.pirls import fit_pirls
from superglm.solvers.rank import (
    decompose_factor,
    decompose_gram,
    needs_factor_certification,
    streamed_weighted_factor,
)
from superglm.types import GroupSlice


def _paired_boundary_design(
    singular_values: tuple[float, float, float] = (1.0, 1.55e-8, 1.30e-8),
    *,
    seed: int = 4274,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the deterministic centered design whose Gram loses its factor subspace."""
    rng = np.random.default_rng(seed)
    right, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    half_left, _ = np.linalg.qr(rng.normal(size=(8, 3)))
    left = np.vstack((half_left, -half_left)) / np.sqrt(2.0)
    return left @ np.diag(singular_values) @ right.T, right


def _grouped_design(values: np.ndarray) -> DesignMatrix:
    return DesignMatrix([DenseGroupMatrix(values)], n=len(values), p=values.shape[1])


def _null_projector(decomposition) -> np.ndarray:
    null = np.asarray(decomposition.parameter_null_basis, dtype=np.float64)
    return null @ np.linalg.pinv(null)


def _assert_equal_rank_different_subspaces(gram, factor) -> None:
    assert gram.rank == factor.rank == 2
    assert needs_factor_certification(gram)
    assert np.linalg.norm(_null_projector(gram) - _null_projector(factor), ord=2) > 0.1


def _assert_factor_certificate(actual, expected) -> None:
    assert actual.rank == expected.rank
    assert actual.method == "qr_svd"
    np.testing.assert_allclose(
        _null_projector(actual),
        _null_projector(expected),
        rtol=0.0,
        atol=2e-14,
    )
    np.testing.assert_allclose(
        actual.pseudo_inverse(),
        expected.pseudo_inverse(),
        rtol=2e-14,
        atol=0.0,
    )
    assert actual.log_pdet == pytest.approx(expected.log_pdet, rel=2e-14, abs=0.0)


def test_metrics_rank_helpers_use_equal_rank_factor_subspaces() -> None:
    design, _ = _paired_boundary_design()
    weights = np.ones(len(design))
    xtw1 = design.T @ weights
    mean_x = xtw1 / float(np.sum(weights))
    centered = design - mean_x
    centered_gram = centered.T @ centered
    raw_gram = design.T @ design

    centered_factor = streamed_weighted_factor(
        iter_dense_chunks(design),
        weights,
        center=mean_x,
    )
    raw_factor = streamed_weighted_factor(iter_dense_chunks(design), weights)
    expected_data = decompose_factor(centered_factor)
    expected_coefficient = decompose_factor(raw_factor)
    gram_data = decompose_gram(centered_gram)
    gram_coefficient = decompose_gram(raw_gram)
    _assert_equal_rank_different_subspaces(gram_data, expected_data)
    _assert_equal_rank_different_subspaces(gram_coefficient, expected_coefficient)

    actual_data = _certified_data_rank(design, weights, centered_gram, xtw1)
    actual_coefficient = _certified_coefficient_rank(
        design,
        weights,
        raw_gram,
        np.zeros((3, 3)),
    )

    # A nonzero penalty exercises the profile-specific helper while remaining
    # far below the deliberately chosen cutoff-boundary data directions.
    penalty = 1e-40 * np.eye(3)
    smooth_factor = penalty_factor(penalty)
    expected_profile = decompose_factor(np.vstack((centered_factor, smooth_factor)))
    gram_profile = decompose_gram(centered_gram + penalty)
    _assert_equal_rank_different_subspaces(gram_profile, expected_profile)
    actual_profile = _certified_profile_rank(
        design,
        weights,
        centered_gram,
        xtw1,
        penalty,
        actual_data,
    )

    _assert_factor_certificate(actual_data, expected_data)
    _assert_factor_certificate(actual_coefficient, expected_coefficient)
    _assert_factor_certificate(actual_profile, expected_profile)


def test_legacy_state_covariance_uses_equal_rank_factor_subspaces() -> None:
    design, _ = _paired_boundary_design()
    dm = _grouped_design(design)
    weights = np.ones(len(design))
    groups = [GroupSlice(name="x", start=0, end=3)]
    solver = SimpleNamespace(beta=np.ones(3), rank_info=None)
    model = SimpleNamespace(
        _dm=dm,
        _groups=groups,
        _fit_state=None,
        _reml_lambdas=None,
        _lambda2_config=0.0,
        _resolved_penalty=None,
        _penalty_config=Ridge(lambda1=0.0),
        _reml_penalties=None,
        _runtime_canonical_state=None,
    )

    (
        _,
        _,
        coefficient_inverse,
        augmented_inverse,
        centered_gram,
        profile_inverse,
        data_rank,
    ) = _legacy_active_state(model, solver, weights)

    mean_x = np.average(design, axis=0, weights=weights)
    expected_data = decompose_factor(grouped_weighted_factor(dm, weights, center=mean_x))
    expected_coefficient = decompose_factor(grouped_weighted_factor(dm, weights))
    gram_data = decompose_gram(centered_gram)
    _assert_equal_rank_different_subspaces(gram_data, expected_data)

    _assert_factor_certificate(data_rank, expected_data)
    np.testing.assert_allclose(
        coefficient_inverse,
        expected_coefficient.pseudo_inverse(),
        rtol=2e-14,
        atol=0.0,
    )
    np.testing.assert_allclose(
        profile_inverse,
        expected_data.pseudo_inverse(),
        rtol=2e-14,
        atol=0.0,
    )
    np.testing.assert_allclose(
        augmented_inverse[1:, 1:],
        expected_data.pseudo_inverse(),
        rtol=2e-14,
        atol=0.0,
    )


def test_observed_reml_geometry_uses_equal_rank_factor_subspace() -> None:
    design, _ = _paired_boundary_design()
    dm = _grouped_design(design)
    weights = np.ones(len(design))
    mean_x = np.average(design, axis=0, weights=weights)
    penalty = np.zeros((3, 3))
    expected = decompose_factor(grouped_augmented_factor(dm, weights, penalty, center=mean_x))
    gram = decompose_gram((design - mean_x).T @ (design - mean_x))
    _assert_equal_rank_different_subspaces(gram, expected)

    geometry = build_observed_reml_geometry(
        dm=dm,
        distribution=Gamma(),
        link=LogLink(),
        y=np.ones(len(design)),
        sample_weight=weights,
        offset_arr=np.zeros(len(design)),
        result=SimpleNamespace(beta=np.zeros(3), intercept=0.0),
        penalty=penalty,
    )

    assert geometry.hessian_rank == 1 + expected.rank
    np.testing.assert_allclose(
        geometry.hessian_inverse,
        expected.pseudo_inverse(),
        rtol=2e-14,
        atol=0.0,
    )
    assert geometry.log_det_H == pytest.approx(
        float(np.log(np.sum(weights)) + expected.log_pdet),
        rel=2e-14,
        abs=0.0,
    )


def test_scop_efs_terminal_rank_uses_equal_rank_factor_subspace() -> None:
    design, _ = _paired_boundary_design()
    gram = decompose_gram(design.T @ design)
    expected = decompose_factor(design)
    _assert_equal_rank_different_subspaces(gram, expected)

    actual = _certified_terminal_rank(design.T @ design, lambda: design)

    _assert_factor_certificate(actual, expected)


def test_scop_fisher_geometry_uses_equal_full_rank_factor_certificate() -> None:
    # This construction leaves a wide cross-platform margin: both routes are
    # full rank, while their inverses differ by roughly 60% on the reference
    # LAPACK build rather than sitting near the assertion boundary.
    design, _ = _paired_boundary_design((1.0, 5.5e-8, 3.025e-8), seed=4256)
    dm = _grouped_design(design)
    weights = np.ones(len(design))
    mean_x = np.average(design, axis=0, weights=weights)
    centered = design - mean_x
    centered_gram = centered.T @ centered
    raw_gram = design.T @ design
    penalty = np.zeros((3, 3))
    gram = decompose_gram(centered_gram)
    expected = decompose_factor(grouped_augmented_factor(dm, weights, penalty, center=mean_x))
    assert gram.rank == expected.rank == 3
    assert needs_factor_certification(gram)
    assert (
        np.linalg.norm(gram.pseudo_inverse() - expected.pseudo_inverse())
        / np.linalg.norm(expected.pseudo_inverse())
        > 0.1
    )
    states = {
        0: {
            "group_sl": slice(0, 3),
            "gamma_eff": np.ones(3),
            "last_fisher_fallback": True,
        }
    }

    geometry = build_cached_scop_joint_geometry(
        raw_fisher_gram=raw_gram,
        fisher_xtw=design.T @ weights,
        fisher_sum_w=float(np.sum(weights)),
        latent_penalty=penalty,
        scop_states=states,
        centered_fisher_gram=centered_gram,
        fisher_mean_x=mean_x,
        dm=dm,
        fisher_weights=weights,
    )

    assert geometry.curvature_source == "fisher"
    assert geometry.hessian_rank == 1 + expected.rank
    np.testing.assert_allclose(
        geometry.hessian_inverse,
        expected.pseudo_inverse(),
        rtol=2e-14,
        atol=0.0,
    )
    assert geometry.log_det_H == pytest.approx(
        float(np.log(np.sum(weights)) + expected.log_pdet),
        rel=2e-14,
        abs=0.0,
    )


def test_scop_equal_full_rank_certificate_preserves_observed_curvature() -> None:
    """A full Fisher range certifies estimability, not observed curvature values."""
    design, _ = _paired_boundary_design((1.0, 4e-8, 3e-8))
    dm = _grouped_design(design)
    weights = np.ones(len(design))
    mean_x = np.average(design, axis=0, weights=weights)
    centered_gram = (design - mean_x).T @ (design - mean_x)
    raw_gram = design.T @ design
    penalty = np.zeros((3, 3))
    observed_centered = centered_gram + 5.0e-16 * np.eye(3)
    observed = decompose_gram(observed_centered)
    certified = decompose_factor(grouped_augmented_factor(dm, weights, penalty, center=mean_x))
    assert observed.rank == certified.rank == observed.width == 3
    assert needs_factor_certification(observed)
    transformed_cross = float(np.sum(weights)) * mean_x
    states = {
        0: {
            "group_sl": slice(0, 3),
            "gamma_eff": np.ones(3),
            "H_scop_penalized": observed_centered
            + np.outer(transformed_cross, transformed_cross) / float(np.sum(weights)),
            "last_fisher_fallback": False,
        }
    }

    geometry = build_cached_scop_joint_geometry(
        raw_fisher_gram=raw_gram,
        fisher_xtw=design.T @ weights,
        fisher_sum_w=float(np.sum(weights)),
        latent_penalty=penalty,
        scop_states=states,
        centered_fisher_gram=centered_gram,
        fisher_mean_x=mean_x,
        dm=dm,
        fisher_weights=weights,
    )

    assert geometry.curvature_source == "observed"
    assert geometry.hessian_rank == 1 + observed.rank
    np.testing.assert_allclose(geometry.centered_hessian, observed_centered, rtol=0.0, atol=2e-30)
    np.testing.assert_allclose(
        geometry.hessian_inverse,
        observed.pseudo_inverse(),
        rtol=2e-14,
        atol=0.0,
    )
    assert geometry.log_det_H == pytest.approx(
        float(np.log(np.sum(weights)) + observed.log_pdet),
        rel=2e-14,
        abs=0.0,
    )


def test_scop_observed_geometry_uses_equal_rank_certified_range() -> None:
    design, _ = _paired_boundary_design()
    dm = _grouped_design(design)
    weights = np.ones(len(design))
    mean_x = np.average(design, axis=0, weights=weights)
    centered = design - mean_x
    centered_gram = centered.T @ centered
    raw_gram = design.T @ design
    penalty = np.zeros((3, 3))
    gram = decompose_gram(centered_gram)
    certified = decompose_factor(grouped_augmented_factor(dm, weights, penalty, center=mean_x))
    _assert_equal_rank_different_subspaces(gram, certified)
    states = {
        0: {
            "group_sl": slice(0, 3),
            "gamma_eff": np.ones(3),
            "H_scop_penalized": centered_gram,
            "last_fisher_fallback": False,
        }
    }

    geometry = build_cached_scop_joint_geometry(
        raw_fisher_gram=raw_gram,
        fisher_xtw=design.T @ weights,
        fisher_sum_w=float(np.sum(weights)),
        latent_penalty=penalty,
        scop_states=states,
        centered_fisher_gram=centered_gram,
        fisher_mean_x=mean_x,
        dm=dm,
        fisher_weights=weights,
    )

    # Fisher rows certify estimability, while the retained observed matrix
    # still supplies curvature and determinant on that certified range.
    nullity = certified.width - certified.rank
    orthogonal, _ = np.linalg.qr(certified.parameter_null_basis, mode="complete")
    estimable_basis = orthogonal[:, nullity:]
    reduced = estimable_basis.T @ centered_gram @ estimable_basis
    reduced = 0.5 * (reduced + reduced.T)
    observed_on_range = decompose_gram(reduced)
    expected_centered = estimable_basis @ reduced @ estimable_basis.T
    expected_inverse = estimable_basis @ observed_on_range.pseudo_inverse() @ estimable_basis.T
    expected_log_det = float(np.log(np.sum(weights)) + observed_on_range.log_pdet)

    assert geometry.curvature_source == "observed"
    assert geometry.hessian_rank == 1 + certified.rank
    np.testing.assert_allclose(
        geometry.centered_hessian,
        expected_centered,
        rtol=2e-14,
        atol=2e-30,
    )
    np.testing.assert_allclose(
        geometry.hessian_inverse,
        expected_inverse,
        rtol=2e-14,
        atol=0.0,
    )
    assert geometry.log_det_H == pytest.approx(expected_log_det, rel=2e-14, abs=0.0)
    assert (
        np.linalg.norm(geometry.hessian_inverse - gram.pseudo_inverse())
        / np.linalg.norm(expected_inverse)
        > 0.1
    )


def test_pirls_rank_metadata_uses_equal_rank_factor_subspaces() -> None:
    design, right = _paired_boundary_design()
    dm = _grouped_design(design)
    weights = np.ones(len(design))
    groups = [GroupSlice(name="x", start=0, end=3)]
    y = 0.4 + design @ right[:, 0]

    result = fit_pirls(
        dm,
        y,
        weights,
        Gaussian(),
        IdentityLink(),
        groups,
        GroupLasso(lambda1=0.0),
        tol=1e-12,
    )

    assert result.converged
    assert result.rank_info is not None
    mean_x = np.average(design, axis=0, weights=weights)
    expected_centered = decompose_factor(grouped_weighted_factor(dm, weights, center=mean_x))
    expected_raw = decompose_factor(grouped_weighted_factor(dm, weights))
    gram = decompose_gram((design - mean_x).T @ (design - mean_x))
    _assert_equal_rank_different_subspaces(gram, expected_centered)

    _assert_factor_certificate(result.rank_info.data, expected_centered)
    _assert_factor_certificate(result.rank_info.augmented, expected_centered)
    _assert_factor_certificate(result.rank_info.coefficient, expected_raw)
