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
    SHARED_RANK_POLICY,
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


def _roundoff_gamma(operation_count: int) -> float:
    eps = np.finfo(np.float64).eps
    return operation_count * eps / (1.0 - operation_count * eps)


def _factor_space_reference(
    factor: np.ndarray,
) -> SimpleNamespace:
    """Independent retained-factor geometry with resolved perturbation bounds."""
    factor = np.asarray(factor, dtype=np.float64)
    width = factor.shape[1]
    column_scale = np.linalg.norm(factor, axis=0)
    active = np.flatnonzero(column_scale > 0.0)
    assert active.size

    equilibrated = factor[:, active] / column_scale[active]
    _left, singular_values, right_t = np.linalg.svd(equilibrated, full_matrices=True)
    cutoff = SHARED_RANK_POLICY.factor_rcond * singular_values[0]
    rank = int(np.count_nonzero(singular_values > cutoff))
    assert rank
    lower_gap = singular_values[rank - 1] - cutoff if rank else np.inf
    upper_gap = cutoff - singular_values[rank] if rank < len(singular_values) else np.inf
    gap = float(min(lower_gap, upper_gap))
    eta_factor = (
        64.0 * _roundoff_gamma(max(factor.shape)) * float(np.linalg.norm(equilibrated, ord=2))
    )
    assert gap > 2.0 * eta_factor
    assert singular_values[rank - 1] - eta_factor > (
        SHARED_RANK_POLICY.factor_rcond * (singular_values[0] + eta_factor)
    )
    if rank < len(singular_values):
        assert singular_values[rank] + eta_factor < (
            SHARED_RANK_POLICY.factor_rcond * (singular_values[0] - eta_factor)
        )
    projector_bound = 2.0 * eta_factor / (gap - 2.0 * eta_factor)

    selected_local: list[int] = []
    for candidate in range(len(active)):
        trial = selected_local + [candidate]
        trial_values = np.linalg.svd(equilibrated[:, trial], compute_uv=False)
        if np.count_nonzero(trial_values > cutoff) > len(selected_local):
            selected_local.append(candidate)
        if len(selected_local) == rank:
            break
    assert len(selected_local) == rank
    representative_columns = active[np.asarray(selected_local, dtype=np.intp)]

    discarded = right_t[rank:].T
    null_pieces: list[np.ndarray] = []
    if discarded.shape[1]:
        spectral_null = np.zeros((width, discarded.shape[1]))
        spectral_null[active] = discarded / column_scale[active, None]
        null_pieces.append(spectral_null)
    inactive = np.setdiff1d(np.arange(width), active, assume_unique=True)
    if inactive.size:
        structural_null = np.zeros((width, len(inactive)))
        structural_null[inactive, np.arange(len(inactive))] = 1.0
        null_pieces.append(structural_null)
    null = np.column_stack(null_pieces) if null_pieces else np.empty((width, 0))
    if null.shape[1]:
        orthogonal, _ = np.linalg.qr(null, mode="complete")
        null_basis = orthogonal[:, : width - rank]
        range_basis = orthogonal[:, width - rank :]
        null_projector = null_basis @ null_basis.T
    else:
        range_basis = np.eye(width)
        null_projector = np.zeros((width, width))

    # Reconstruct the factor after applying the equilibrated factor cutoff.
    # Projecting the untruncated input factor would leak the deliberately
    # discarded singular row back into this determinant.
    certified_factor = np.zeros((rank, width))
    certified_factor[:, active] = (
        singular_values[:rank, None] * right_t[:rank] * column_scale[active][None, :]
    )
    retained_singular_values = np.linalg.svd(
        certified_factor @ range_basis,
        compute_uv=False,
    )
    log_pdet = 2.0 * float(np.sum(np.log(retained_singular_values)))
    log_summation_bound = (
        2.0 * _roundoff_gamma(2 * width) * float(np.sum(np.abs(np.log(retained_singular_values))))
    )
    log_pdet_bound = (
        2.0 * rank * eta_factor / (singular_values[rank - 1] - eta_factor) + log_summation_bound
    )
    return SimpleNamespace(
        rank=rank,
        null_projector=null_projector,
        representative_columns=representative_columns,
        log_pdet=log_pdet,
        projector_bound=projector_bound,
        log_pdet_bound=log_pdet_bound,
    )


def _assert_well_conditioned_inverse_action(
    inverse: np.ndarray,
    factor: np.ndarray,
) -> SimpleNamespace:
    """Bound inverse and fitted action from an independent factor QR."""
    factor = np.asarray(factor, dtype=np.float64)
    inverse = np.asarray(inverse, dtype=np.float64)
    assert factor.shape[0] >= factor.shape[1]
    assert inverse.shape == (factor.shape[1], factor.shape[1])
    assert np.all(np.isfinite(inverse))

    orthogonal, triangular = np.linalg.qr(factor, mode="reduced")
    width = factor.shape[1]
    triangular_inverse = np.linalg.solve(triangular, np.eye(width))
    reference_inverse = triangular_inverse @ triangular_inverse.T
    gram = factor.T @ factor
    gram_norm = float(np.linalg.norm(gram, ord=2))
    inverse_norm = float(np.linalg.norm(inverse, ord=2))
    backward = np.linalg.norm(np.eye(width) - gram @ inverse, ord=2) / (
        gram_norm * inverse_norm + 1.0
    )
    operation_count = factor.shape[0] + 8 * width
    beta = 64.0 * _roundoff_gamma(operation_count)
    assert backward <= beta
    condition = float(np.linalg.cond(gram, p=2))
    conditioned_beta = condition * beta
    assert conditioned_beta < 1.0
    forward_bound = 2.0 * conditioned_beta / (1.0 - conditioned_beta)
    relative_inverse_error = np.linalg.norm(inverse - reference_inverse, ord=2) / np.linalg.norm(
        reference_inverse,
        ord=2,
    )
    assert relative_inverse_error <= forward_bound

    actual_action = factor @ inverse @ factor.T
    reference_action = orthogonal @ orthogonal.T
    action_roundoff = (
        8.0
        * _roundoff_gamma(operation_count)
        * (
            np.linalg.norm(factor, ord=2) ** 2
            * (inverse_norm + np.linalg.norm(reference_inverse, ord=2))
            + 1.0
        )
    )
    action_bound = (
        np.linalg.norm(factor, ord=2) ** 2
        * np.linalg.norm(reference_inverse, ord=2)
        * forward_bound
        + action_roundoff
    )
    assert np.linalg.norm(actual_action - reference_action, ord=2) <= action_bound
    return SimpleNamespace(
        backward=backward,
        beta=beta,
        condition=condition,
        conditioned_beta=conditioned_beta,
        forward_bound=forward_bound,
        relative_inverse_error=relative_inverse_error,
        action_error=float(np.linalg.norm(actual_action - reference_action, ord=2)),
        action_bound=action_bound,
    )


def _assert_factor_certificate(actual, factor: np.ndarray) -> None:
    reference = _factor_space_reference(factor)

    assert actual.rank == reference.rank
    assert actual.method == "qr_svd"
    np.testing.assert_array_equal(actual.active_columns, reference.representative_columns)
    assert (
        np.linalg.norm(_null_projector(actual) - reference.null_projector, ord=2)
        <= reference.projector_bound
    )
    null = np.asarray(actual.parameter_null_basis, dtype=np.float64)
    null_action = np.linalg.norm(np.asarray(factor) @ null, ord=2)
    null_scale = np.linalg.norm(factor, ord=2) * max(
        np.linalg.norm(null, ord=2),
        np.finfo(np.float64).tiny,
    )
    assert null_action <= (SHARED_RANK_POLICY.factor_rcond + reference.projector_bound) * null_scale
    assert abs(actual.log_pdet - reference.log_pdet) <= reference.log_pdet_bound


def _assert_inverse_representative_support(
    inverse: np.ndarray,
    factor: np.ndarray,
) -> None:
    """Check only the representative support resolvable at the cutoff."""
    reference = _factor_space_reference(factor)
    inverse = np.asarray(inverse, dtype=np.float64)
    assert inverse.shape == (factor.shape[1], factor.shape[1])
    assert np.all(np.isfinite(inverse))

    selected = reference.representative_columns
    inactive = np.setdiff1d(np.arange(factor.shape[1]), selected, assume_unique=True)
    assert not np.any(inverse[inactive, :]), "inverse escaped certified representative support"
    assert not np.any(inverse[:, inactive]), "inverse escaped certified representative support"


def _assert_log_pdet(actual: float, factor: np.ndarray) -> None:
    reference = _factor_space_reference(factor)
    assert abs(actual - reference.log_pdet) <= reference.log_pdet_bound


def _extended_precision_gram(factor: np.ndarray) -> np.ndarray:
    extended = np.asarray(factor, dtype=np.longdouble)
    products = extended[:, :, None] * extended[:, None, :]
    return np.sum(products, axis=0, dtype=np.longdouble).astype(np.float64)


def test_factor_certificate_is_invariant_to_row_order_and_gram_accumulation() -> None:
    design, right = _paired_boundary_design()
    reversed_design = design[::-1]
    variants = (
        (design, design.T @ design),
        (reversed_design, reversed_design.T @ reversed_design),
        (design, np.einsum("ni,nj->ij", design, design, optimize=False)),
        (design, _extended_precision_gram(design)),
    )
    reference_prediction = design @ right[:, 0]

    for factor, gram in variants:
        preliminary = decompose_gram(gram)
        assert needs_factor_certification(preliminary)
        actual = _certified_terminal_rank(gram, lambda factor=factor: factor)
        _assert_factor_certificate(actual, factor)

        solve = decompose_factor(factor, retain_factor_solve=True)
        fitted = design @ solve.solve_factor_rhs(factor @ right[:, 0])
        action_scale = np.linalg.norm(design, ord=2) * np.linalg.norm(right[:, 0])
        action_bound = (
            2.0 * SHARED_RANK_POLICY.factor_rcond
            + _roundoff_gamma(design.shape[0] + design.shape[1])
        ) * action_scale
        assert np.linalg.norm(fitted - reference_prediction) <= action_bound


def test_factor_certificate_checker_rejects_preliminary_gram_mutation() -> None:
    design, _ = _paired_boundary_design()
    preliminary = decompose_gram(design.T @ design)

    with pytest.raises(AssertionError):
        _assert_factor_certificate(preliminary, design)


def test_inverse_checker_rejects_preliminary_gram_inverse_hybrid() -> None:
    design, _ = _paired_boundary_design()
    dm = _grouped_design(design)
    weights = np.ones(len(design))
    mean_x = np.average(design, axis=0, weights=weights)
    factor = grouped_weighted_factor(dm, weights, center=mean_x)
    certificate = decompose_factor(factor)
    preliminary_inverse = decompose_gram(factor.T @ factor).pseudo_inverse()

    _assert_factor_certificate(certificate, factor)
    with pytest.raises(AssertionError, match="inverse escaped certified"):
        _assert_inverse_representative_support(preliminary_inverse, factor)


def test_well_conditioned_factor_inverse_action_rejects_scale_mutations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A forced certificate branch has a resolvable covariance magnitude."""
    import superglm.reml.scop_efs as scop_efs

    factor = np.array(
        [
            [1.5, 0.2, -0.1],
            [0.1, 1.2, 0.3],
            [0.2, -0.1, 1.1],
            [0.5, 0.4, -0.2],
            [-0.3, 0.2, 0.6],
            [0.4, -0.5, 0.3],
        ]
    )
    factor_calls = 0

    def factor_factory() -> np.ndarray:
        nonlocal factor_calls
        factor_calls += 1
        return factor

    monkeypatch.setattr(
        scop_efs,
        "decompose_gram_if_authoritative",
        lambda _matrix: None,
    )
    certificate = scop_efs._certified_terminal_rank(factor.T @ factor, factor_factory)
    assert factor_calls == 1
    _assert_factor_certificate(certificate, factor)

    inverse = certificate.pseudo_inverse()
    metrics = _assert_well_conditioned_inverse_action(inverse, factor)
    assert metrics.conditioned_beta < 1.0
    for scale in (0.5, 2.0):
        with pytest.raises(AssertionError):
            _assert_well_conditioned_inverse_action(scale * inverse, factor)


def test_metrics_rank_helpers_use_factor_certified_subspaces() -> None:
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
    gram_data = decompose_gram(centered_gram)
    gram_coefficient = decompose_gram(raw_gram)
    assert needs_factor_certification(gram_data)
    assert needs_factor_certification(gram_coefficient)

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
    profile_factor = np.vstack((centered_factor, smooth_factor))
    gram_profile = decompose_gram(centered_gram + penalty)
    assert needs_factor_certification(gram_profile)
    actual_profile = _certified_profile_rank(
        design,
        weights,
        centered_gram,
        xtw1,
        penalty,
        actual_data,
    )

    _assert_factor_certificate(actual_data, centered_factor)
    _assert_factor_certificate(actual_coefficient, raw_factor)
    _assert_factor_certificate(actual_profile, profile_factor)


def test_legacy_state_covariance_uses_factor_certified_subspaces() -> None:
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
    data_factor = grouped_weighted_factor(dm, weights, center=mean_x)
    coefficient_factor = grouped_weighted_factor(dm, weights)
    gram_data = decompose_gram(centered_gram)
    assert needs_factor_certification(gram_data)

    _assert_factor_certificate(data_rank, data_factor)
    _assert_inverse_representative_support(
        coefficient_inverse,
        coefficient_factor,
    )
    _assert_inverse_representative_support(
        profile_inverse,
        data_factor,
    )
    _assert_inverse_representative_support(
        augmented_inverse[1:, 1:],
        data_factor,
    )


def test_observed_reml_geometry_uses_factor_certified_subspace() -> None:
    design, _ = _paired_boundary_design()
    dm = _grouped_design(design)
    weights = np.ones(len(design))
    mean_x = np.average(design, axis=0, weights=weights)
    penalty = np.zeros((3, 3))
    factor = grouped_augmented_factor(dm, weights, penalty, center=mean_x)
    expected = decompose_factor(factor)
    gram = decompose_gram((design - mean_x).T @ (design - mean_x))
    assert needs_factor_certification(gram)

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
    _assert_inverse_representative_support(
        geometry.hessian_inverse,
        factor,
    )
    _assert_log_pdet(
        geometry.log_det_H - float(np.log(np.sum(weights))),
        factor,
    )


def test_scop_efs_terminal_rank_uses_factor_certified_subspace() -> None:
    design, _ = _paired_boundary_design()
    gram = decompose_gram(design.T @ design)
    assert needs_factor_certification(gram)

    actual = _certified_terminal_rank(design.T @ design, lambda: design)

    _assert_factor_certificate(actual, design)


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
    factor = grouped_augmented_factor(dm, weights, penalty, center=mean_x)
    expected = decompose_factor(factor)
    assert expected.rank == expected.width == 3
    assert needs_factor_certification(gram)
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
    _assert_inverse_representative_support(
        geometry.hessian_inverse,
        factor,
    )
    _assert_log_pdet(
        geometry.log_det_H - float(np.log(np.sum(weights))),
        factor,
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


def test_scop_observed_geometry_uses_factor_certified_range() -> None:
    design, _ = _paired_boundary_design()
    dm = _grouped_design(design)
    weights = np.ones(len(design))
    mean_x = np.average(design, axis=0, weights=weights)
    centered = design - mean_x
    centered_gram = centered.T @ centered
    raw_gram = design.T @ design
    penalty = np.zeros((3, 3))
    gram = decompose_gram(centered_gram)
    factor = grouped_augmented_factor(dm, weights, penalty, center=mean_x)
    certified = decompose_factor(factor)
    assert needs_factor_certification(gram)
    _assert_factor_certificate(certified, factor)
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


def test_pirls_rank_metadata_uses_factor_certified_subspaces() -> None:
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
    centered_factor = grouped_weighted_factor(dm, weights, center=mean_x)
    raw_factor = grouped_weighted_factor(dm, weights)
    gram = decompose_gram((design - mean_x).T @ (design - mean_x))
    assert needs_factor_certification(gram)

    _assert_factor_certificate(result.rank_info.data, centered_factor)
    _assert_factor_certificate(result.rank_info.augmented, centered_factor)
    _assert_factor_certificate(result.rank_info.coefficient, raw_factor)
