"""Dense Pya--Wood oracles for post-fit SCOP inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from superglm.solvers.rank import (
    SHARED_RANK_POLICY,
    decompose_factor,
    decompose_gram,
    needs_factor_certification,
)
from superglm.types import GroupSlice


def _observed_geometry_from_augmented(hessian: np.ndarray):
    from superglm.reml.scop_geometry import SCOPJointGeometry

    sum_w = float(hessian[0, 0])
    cross = np.asarray(hessian[1:, 0], dtype=np.float64)
    centered = hessian[1:, 1:] - np.outer(cross, cross) / sum_w
    decomposition = decompose_gram(centered)
    return SCOPJointGeometry(
        centered_hessian=centered,
        hessian_inverse=decomposition.pseudo_inverse(),
        transformed_intercept_cross=cross,
        sum_w=sum_w,
        log_det_H=float(np.linalg.slogdet(hessian)[1]),
        hessian_rank=hessian.shape[0],
        curvature_source="observed",
    )


def _observed_geometry_from_centered(
    *,
    centered_hessian: np.ndarray,
    mean_x: np.ndarray,
    sum_w: float,
):
    """Build an independently centered observed geometry without raw moments."""
    from superglm.reml.scop_geometry import SCOPJointGeometry

    decomposition = decompose_gram(centered_hessian)
    return SCOPJointGeometry(
        centered_hessian=centered_hessian,
        hessian_inverse=decomposition.pseudo_inverse(),
        transformed_intercept_cross=sum_w * mean_x,
        sum_w=sum_w,
        log_det_H=float(np.log(sum_w) + decomposition.log_pdet),
        hessian_rank=1 + decomposition.rank,
        curvature_source="observed",
        transformed_mean_x=mean_x,
    )


def test_scop_covariance_uses_expected_geometry_but_edf_uses_observed_geometry():
    """Pya--Wood S.5 covariance and Eq. 16 EDF intentionally use different Hessians."""
    import superglm.reml.scop_geometry as scop_geometry

    assert hasattr(scop_geometry, "build_scop_postfit_inference")

    rng = np.random.default_rng(20260722)
    n = 90
    raw_design = np.column_stack(
        (
            rng.normal(size=n),
            rng.normal(size=n),
            rng.normal(size=n),
        )
    )
    fisher_weights = rng.uniform(0.4, 2.1, size=n)
    raw_fisher_gram = raw_design.T @ (fisher_weights[:, None] * raw_design)
    fisher_xtw = raw_design.T @ fisher_weights
    fisher_sum_w = float(np.sum(fisher_weights))

    beta_eff = np.log(np.array([0.35, 1.4]))
    jacobian = np.array([1.0, *np.exp(beta_eff)])
    jacobian_aug = np.diag(np.array([1.0, *jacobian]))
    latent_penalty = np.array(
        [
            [0.25, 0.0, 0.0],
            [0.0, 1.2, -0.45],
            [0.0, -0.45, 0.9],
        ]
    )

    raw_fisher_aug = np.empty((4, 4), dtype=np.float64)
    raw_fisher_aug[0, 0] = fisher_sum_w
    raw_fisher_aug[0, 1:] = fisher_xtw
    raw_fisher_aug[1:, 0] = fisher_xtw
    raw_fisher_aug[1:, 1:] = raw_fisher_gram
    expected_data_latent = jacobian_aug @ raw_fisher_aug @ jacobian_aug
    penalty_aug = np.zeros((4, 4), dtype=np.float64)
    penalty_aug[1:, 1:] = latent_penalty
    expected_penalized = expected_data_latent + penalty_aug

    # Deliberately make the retained full-Newton geometry differ materially
    # from Fisher geometry while preserving positive definiteness.
    observed_penalized = expected_penalized.copy()
    observed_penalized += np.diag([7.0, 1.1, 2.4, 0.7])
    observed_penalized[0, 2] += 0.35
    observed_penalized[2, 0] += 0.35
    observed_geometry = _observed_geometry_from_augmented(observed_penalized)

    states = {
        1: {
            "group_sl": slice(1, 3),
            "group_name": "mono",
            "beta_eff": beta_eff,
            "gamma_eff": np.exp(beta_eff),
        }
    }
    groups = [
        GroupSlice(name="ordinary", start=0, end=1, feature_name="ordinary"),
        GroupSlice(
            name="mono",
            start=1,
            end=3,
            feature_name="mono",
            monotone_engine="scop",
        ),
    ]

    actual = scop_geometry.build_scop_postfit_inference(
        raw_fisher_gram=raw_fisher_gram,
        fisher_xtw=fisher_xtw,
        fisher_sum_w=fisher_sum_w,
        latent_penalty=latent_penalty,
        scop_states=states,
        groups=groups,
        observed_geometry=observed_geometry,
    )

    expected_augmented_inverse = jacobian_aug @ np.linalg.inv(expected_penalized) @ jacobian_aug
    expected_coefficient_inverse = (
        np.diag(jacobian)
        @ np.linalg.inv(expected_data_latent[1:, 1:] + latent_penalty)
        @ np.diag(jacobian)
    )
    observed_mean = observed_penalized[1:, 0] / observed_penalized[0, 0]
    fisher_mean = expected_data_latent[1:, 0] / expected_data_latent[0, 0]
    delta_mean = fisher_mean - observed_mean
    centered_expected_data = expected_data_latent[1:, 1:] - fisher_sum_w * np.outer(
        fisher_mean, fisher_mean
    )
    centered_observed_inverse = np.linalg.inv(observed_geometry.centered_hessian)
    influence = np.empty((4, 4))
    influence[0, 0] = fisher_sum_w / observed_penalized[0, 0]
    influence[0, 1:] = fisher_sum_w * delta_mean / observed_penalized[0, 0]
    influence[1:, 0] = centered_observed_inverse @ (fisher_sum_w * delta_mean)
    influence[1:, 1:] = centered_observed_inverse @ (
        centered_expected_data + fisher_sum_w * np.outer(delta_mean, delta_mean)
    )
    expected_edf = np.diag(influence)[1:]
    expected_edf1 = 2.0 * expected_edf - np.diag(influence @ influence)[1:]

    np.testing.assert_allclose(
        actual.augmented_inverse,
        expected_augmented_inverse,
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        actual.coefficient_inverse,
        expected_coefficient_inverse,
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(actual.feature_edf, expected_edf, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(actual.feature_edf1, expected_edf1, rtol=2e-12, atol=2e-12)
    assert actual.intercept_edf == pytest.approx(influence[0, 0], rel=2e-12, abs=2e-12)
    assert actual.total_edf == pytest.approx(np.trace(influence), rel=2e-12, abs=2e-12)
    assert actual.group_edf["ordinary"] == pytest.approx(expected_edf[0])
    assert actual.group_edf["mono"] == pytest.approx(np.sum(expected_edf[1:]))

    # The test must detect the tempting but incorrect reuse of observed REML
    # geometry for the Bayesian covariance.
    wrong_observed_covariance = jacobian_aug @ np.linalg.inv(observed_penalized) @ jacobian_aug
    assert not np.allclose(actual.augmented_inverse, wrong_observed_covariance, rtol=1e-3)


def test_cached_scop_geometry_falls_back_to_coherent_expected_hessian():
    """An indefinite retained Newton block falls back as one complete geometry."""
    from superglm.reml.scop_geometry import build_cached_scop_joint_geometry

    raw_fisher_gram = np.array([[7.0, 1.2], [1.2, 5.5]])
    fisher_xtw = np.array([1.5, -0.8])
    fisher_sum_w = 12.0
    latent_penalty = np.array([[0.8, -0.2], [-0.2, 0.6]])
    jacobian = np.array([0.4, 1.7])
    states = {
        0: {
            "group_sl": slice(0, 2),
            "group_name": "mono",
            "beta_eff": np.log(jacobian),
            "gamma_eff": jacobian,
            "H_scop_penalized": -10.0 * np.eye(2),
            "last_fisher_fallback": False,
        }
    }

    actual = build_cached_scop_joint_geometry(
        raw_fisher_gram=raw_fisher_gram,
        fisher_xtw=fisher_xtw,
        fisher_sum_w=fisher_sum_w,
        latent_penalty=latent_penalty,
        scop_states=states,
    )

    transformed_gram = raw_fisher_gram * jacobian[:, None] * jacobian[None, :]
    transformed_cross = fisher_xtw * jacobian
    expected_centered = (
        transformed_gram
        + latent_penalty
        - np.outer(transformed_cross, transformed_cross) / fisher_sum_w
    )
    assert actual.curvature_source == "fisher"
    np.testing.assert_allclose(actual.centered_hessian, expected_centered)
    np.testing.assert_allclose(actual.hessian_inverse, np.linalg.inv(expected_centered))
    np.testing.assert_allclose(actual.transformed_intercept_cross, transformed_cross)


def test_cached_scop_geometry_discards_optimizer_ridge_after_fisher_fallback():
    """A Newton rescue ridge is not part of Wood's likelihood Hessian."""
    from superglm.reml.scop_geometry import build_cached_scop_joint_geometry

    actual = build_cached_scop_joint_geometry(
        raw_fisher_gram=np.array([[6.0]]),
        centered_fisher_gram=np.array([[6.0]]),
        fisher_xtw=np.array([0.0]),
        fisher_mean_x=np.array([0.0]),
        fisher_sum_w=1.0,
        latent_penalty=np.zeros((1, 1)),
        scop_states={
            0: {
                "group_sl": slice(0, 1),
                "group_name": "mono",
                "beta_eff": np.array([0.0]),
                "gamma_eff": np.array([1.0]),
                "H_scop_penalized": np.array([[6.0001]]),
                "last_fisher_fallback": True,
            }
        },
    )

    assert actual.curvature_source == "fisher"
    assert actual.centered_hessian[0, 0] == pytest.approx(6.0, rel=0.0, abs=0.0)
    assert actual.hessian_inverse[0, 0] == pytest.approx(1.0 / 6.0, rel=2e-15)
    assert actual.log_det_H == pytest.approx(np.log(6.0), rel=2e-15)


def test_cached_scop_geometry_uses_stable_centered_fisher_input_under_translation():
    """A large column origin must not cancel the canonical SCOP Hessian."""
    from superglm.reml.scop_geometry import build_cached_scop_joint_geometry

    sum_w = 100.0
    mean_x = np.array([1.0e10, 0.25])
    centered = np.array([[100.0, 2.0], [2.0, 8.0]])
    raw = centered + sum_w * np.outer(mean_x, mean_x)
    penalty = np.diag([0.0, 0.7])
    jacobian = np.array([1.0, 1.4])
    states = {
        1: {
            "group_sl": slice(1, 2),
            "group_name": "mono",
            "beta_eff": np.log(jacobian[1:]),
            "gamma_eff": jacobian[1:],
            "H_scop_penalized": np.array([[raw[1, 1] * jacobian[1] ** 2 + penalty[1, 1]]]),
            "last_fisher_fallback": False,
        }
    }

    actual = build_cached_scop_joint_geometry(
        raw_fisher_gram=raw,
        centered_fisher_gram=centered,
        fisher_xtw=sum_w * mean_x,
        fisher_mean_x=mean_x,
        fisher_sum_w=sum_w,
        latent_penalty=penalty,
        scop_states=states,
    )

    expected = centered * jacobian[:, None] * jacobian[None, :] + penalty
    np.testing.assert_allclose(actual.centered_hessian, expected, rtol=2e-15, atol=2e-15)
    np.testing.assert_allclose(actual.hessian_inverse, np.linalg.inv(expected), rtol=2e-15)
    assert actual.hessian_rank == 3


def test_canonical_scop_fit_is_invariant_to_large_ordinary_column_translation():
    """The reduced solve and retained SCOP Schur geometry share stable centering."""
    from superglm import Constraint, SuperGLM
    from superglm.features import Numeric, PSpline

    rng = np.random.default_rng(20260729)
    n = 180
    x = np.sort(rng.uniform(0.0, 1.0, size=n))
    z = rng.normal(size=n)
    y = 0.3 + 0.7 * z + 1.2 * x + rng.normal(scale=0.03, size=n)

    fitted = []
    for shift in (0.0, 1.0e10):
        frame = pd.DataFrame({"z": z + shift, "x": x})
        model = SuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            discrete=True,
            features={
                "z": Numeric(),
                "x": PSpline(n_knots=7, constraint=Constraint.fit.increasing),
            },
        )
        model.fit(frame, y)
        fitted.append((model, frame, model.predict(frame)))

    translated_values = z + 1.0e10
    stable_frame = pd.DataFrame({"z": translated_values - translated_values[0], "x": x})
    stable_reference = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        discrete=True,
        features={
            "z": Numeric(),
            "x": PSpline(n_knots=7, constraint=Constraint.fit.increasing),
        },
    )
    stable_reference.fit(stable_frame, y)
    stable_prediction = stable_reference.predict(stable_frame)

    baseline, _, prediction = fitted[0]
    translated, _, translated_prediction = fitted[1]
    baseline_result = baseline._solver_result
    translated_result = translated._solver_result
    stable_result = stable_reference._solver_result

    np.testing.assert_allclose(translated_prediction, prediction, rtol=0.0, atol=2e-5)
    input_ulp = float(np.spacing(1.0e10))
    np.testing.assert_allclose(
        translated_prediction,
        stable_prediction,
        rtol=0.0,
        atol=4.0 * input_ulp,
    )
    assert translated_result.deviance == pytest.approx(
        baseline_result.deviance,
        rel=2e-5,
        abs=2e-7,
    )
    assert translated_result.effective_df == pytest.approx(
        baseline_result.effective_df,
        rel=2e-6,
        abs=2e-7,
    )
    assert translated_result.scop_geometry.hessian_rank == (
        baseline_result.scop_geometry.hessian_rank
    )
    np.testing.assert_allclose(
        translated_result.scop_geometry.centered_hessian,
        baseline_result.scop_geometry.centered_hessian,
        rtol=2e-5,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        translated_result.scop_geometry.centered_hessian,
        stable_result.scop_geometry.centered_hessian,
        rtol=2e-6,
        atol=2e-6,
    )


def test_postfit_mapping_accepts_numerically_underflowed_scop_jacobian():
    """Inference never forms C^-1, so an exp-map boundary at zero stays usable."""
    from superglm.reml.scop_geometry import build_scop_postfit_inference

    observed = _observed_geometry_from_augmented(np.diag([9.0, 2.0]))
    actual = build_scop_postfit_inference(
        raw_fisher_gram=np.array([[4.0]]),
        fisher_xtw=np.array([0.0]),
        fisher_sum_w=9.0,
        latent_penalty=np.array([[2.0]]),
        scop_states={
            0: {
                "group_sl": slice(0, 1),
                "group_name": "mono",
                "beta_eff": np.array([-1000.0]),
                "gamma_eff": np.array([0.0]),
            }
        },
        groups=[
            GroupSlice(
                name="mono",
                start=0,
                end=1,
                feature_name="mono",
                monotone_engine="scop",
            )
        ],
        observed_geometry=observed,
    )

    assert actual.coefficient_inverse[0, 0] == 0.0
    assert actual.augmented_inverse[1, 1] == 0.0


def _mapped_factor_inverses(
    design: np.ndarray,
    weights: np.ndarray,
    jacobian: np.ndarray,
    penalty: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent factor-space covariance oracle in latent SCOP coordinates."""
    sum_w = float(np.sum(weights))
    mean_x = np.average(design, axis=0, weights=weights)
    centered_factor = np.sqrt(weights)[:, None] * (design - mean_x) * jacobian
    raw_factor = np.sqrt(weights)[:, None] * design * jacobian
    if np.any(penalty):
        eigenvalues, eigenvectors = np.linalg.eigh(penalty)
        positive = eigenvalues > 0.0
        penalty_factor = np.sqrt(eigenvalues[positive])[:, None] * eigenvectors[:, positive].T
        centered_factor = np.vstack((centered_factor, penalty_factor))
        raw_factor = np.vstack((raw_factor, penalty_factor))
    centered_inverse = _independent_factor_inverse(centered_factor)
    coefficient_inverse = _independent_factor_inverse(raw_factor)
    latent_mean = mean_x * jacobian
    inverse_mean = centered_inverse @ latent_mean
    augmented = np.empty((len(jacobian) + 1, len(jacobian) + 1))
    augmented[0, 0] = 1.0 / sum_w + float(latent_mean @ inverse_mean)
    augmented[0, 1:] = -inverse_mean
    augmented[1:, 0] = -inverse_mean
    augmented[1:, 1:] = centered_inverse
    augmented_jacobian = np.concatenate(([1.0], jacobian))
    return (
        coefficient_inverse * jacobian[:, None] * jacobian[None, :],
        augmented * augmented_jacobian[:, None] * augmented_jacobian[None, :],
    )


def _roundoff_gamma(operation_count: int) -> float:
    eps = np.finfo(np.float64).eps
    return operation_count * eps / (1.0 - operation_count * eps)


def _independent_factor_inverse(factor: np.ndarray) -> np.ndarray:
    """Direct factor-SVD rank policy with an independent QR representative solve."""
    factor = np.asarray(factor, dtype=np.float64)
    width = factor.shape[1]
    column_scale = np.linalg.norm(factor, axis=0)
    active = np.flatnonzero(column_scale > 0.0)
    if not active.size:
        return np.zeros((width, width))

    equilibrated = factor[:, active] / column_scale[active]
    singular_values = np.linalg.svd(equilibrated, compute_uv=False)
    cutoff = np.sqrt(np.finfo(np.float64).eps) * singular_values[0]
    rank = int(np.count_nonzero(singular_values > cutoff))
    lower_gap = singular_values[rank - 1] - cutoff if rank else np.inf
    upper_gap = cutoff - singular_values[rank] if rank < len(singular_values) else np.inf
    gap = min(lower_gap, upper_gap)
    eta_factor = (
        64.0 * _roundoff_gamma(max(factor.shape)) * float(np.linalg.norm(equilibrated, ord=2))
    )
    assert gap > 2.0 * eta_factor

    selected_local: list[int] = []
    for candidate in range(len(active)):
        trial = selected_local + [candidate]
        trial_values = np.linalg.svd(equilibrated[:, trial], compute_uv=False)
        trial_rank = int(np.count_nonzero(trial_values > cutoff))
        if trial_rank > len(selected_local):
            selected_local.append(candidate)
        if len(selected_local) == rank:
            break
    assert len(selected_local) == rank
    selected = active[np.asarray(selected_local, dtype=np.intp)]
    _orthogonal, triangular = np.linalg.qr(factor[:, selected], mode="reduced")
    triangular_inverse = np.linalg.solve(triangular, np.eye(rank))
    inverse = np.zeros((width, width))
    inverse[np.ix_(selected, selected)] = triangular_inverse @ triangular_inverse.T
    return inverse


def _assert_covariance_factor_geometry(
    actual: np.ndarray,
    expected: np.ndarray,
    factor: np.ndarray,
) -> None:
    """Check covariance size, fitted action, and reflexive Penrose residuals."""
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    factor = np.asarray(factor, dtype=np.float64)
    gram = factor.T @ factor
    actual_norm = float(np.linalg.norm(actual, ord=2))
    expected_norm = float(np.linalg.norm(expected, ord=2))
    gram_norm = float(np.linalg.norm(gram, ord=2))
    tiny = np.finfo(np.float64).tiny
    gamma = _roundoff_gamma(factor.shape[0] + 4 * factor.shape[1])

    assert np.all(np.isfinite(actual))
    assert np.linalg.norm(actual - actual.T, ord=2) <= gamma * max(actual_norm, tiny)
    assert actual_norm <= expected_norm + gamma * max(expected_norm, tiny)

    action = factor @ actual @ factor.T
    expected_action = factor @ expected @ factor.T
    action_scale = np.linalg.norm(factor, ord=2) ** 2 * (actual_norm + expected_norm)
    assert np.linalg.norm(action - expected_action, ord=2) <= gamma * action_scale

    first_scale = gram_norm * gram_norm * actual_norm + gram_norm
    second_scale = actual_norm * actual_norm * gram_norm + actual_norm
    first = np.linalg.norm(gram @ actual @ gram - gram, ord=2) / max(first_scale, tiny)
    second = np.linalg.norm(actual @ gram @ actual - actual, ord=2) / max(
        second_scale,
        tiny,
    )
    assert first <= gamma
    assert second <= gamma


def _rank_boundary_inputs(jacobian: np.ndarray):
    from superglm.group_matrix import DenseGroupMatrix, DesignMatrix

    rng = np.random.default_rng(7)
    n = 500
    primary = rng.normal(size=n)
    orthogonal = rng.normal(size=n)
    design = np.column_stack((primary, primary + 1.0e-12 * orthogonal))
    weights = np.ones(n)
    mean_x = np.average(design, axis=0, weights=weights)
    centered = design - mean_x
    centered_gram = centered.T @ centered
    raw_gram = centered_gram + n * np.outer(mean_x, mean_x)
    latent_gram = centered_gram * jacobian[:, None] * jacobian[None, :]
    preliminary = decompose_gram(latent_gram)
    certified = decompose_factor(centered * jacobian)
    dm = DesignMatrix([DenseGroupMatrix(design)], n=n, p=2)
    states = {
        0: {
            "group_sl": slice(0, 2),
            "group_name": "mono",
            "beta_eff": np.log(jacobian, where=jacobian > 0.0, out=np.full(2, -1000.0)),
            "gamma_eff": jacobian,
        }
    }
    groups = [
        GroupSlice(
            name="mono",
            start=0,
            end=2,
            feature_name="mono",
            monotone_engine="scop",
        )
    ]
    observed = _observed_geometry_from_centered(
        centered_hessian=np.eye(2),
        mean_x=np.zeros(2),
        sum_w=float(n),
    )
    return (
        design,
        dm,
        weights,
        mean_x,
        centered_gram,
        raw_gram,
        preliminary,
        certified,
        states,
        groups,
        observed,
    )


def test_scop_postfit_covariance_certifies_near_aliases_from_factor():
    """Normal-equation round-off cannot publish a spurious 1e12 covariance."""
    from superglm.reml.scop_geometry import build_scop_postfit_inference

    jacobian = np.ones(2)
    (
        design,
        dm,
        weights,
        mean_x,
        centered_gram,
        raw_gram,
        preliminary,
        certified,
        states,
        groups,
        observed,
    ) = _rank_boundary_inputs(jacobian)
    assert certified.rank == 1
    assert needs_factor_certification(preliminary)
    actual = build_scop_postfit_inference(
        raw_fisher_gram=raw_gram,
        centered_fisher_gram=centered_gram,
        fisher_xtw=float(len(weights)) * mean_x,
        fisher_mean_x=mean_x,
        fisher_sum_w=float(len(weights)),
        latent_penalty=np.zeros((2, 2)),
        scop_states=states,
        groups=groups,
        observed_geometry=observed,
        dm=dm,
        fisher_weights=weights,
    )

    expected_coefficient, expected_augmented = _mapped_factor_inverses(
        design,
        weights,
        jacobian,
        np.zeros((2, 2)),
    )
    raw_factor = np.sqrt(weights)[:, None] * design
    augmented_factor = np.column_stack((np.sqrt(weights), raw_factor))
    _assert_covariance_factor_geometry(
        actual.coefficient_inverse,
        expected_coefficient,
        raw_factor,
    )
    _assert_covariance_factor_geometry(
        actual.augmented_inverse,
        expected_augmented,
        augmented_factor,
    )


def test_scop_postfit_full_rank_preliminary_dispatches_factor_certificate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exact dispatch coverage does not depend on a positive roundoff eigenvalue."""
    import superglm.reml.scop_geometry as scop_geometry

    jacobian = np.ones(2)
    (
        design,
        dm,
        weights,
        mean_x,
        centered_gram,
        raw_gram,
        _preliminary,
        _certified,
        states,
        groups,
        observed,
    ) = _rank_boundary_inputs(jacobian)
    eps = SHARED_RANK_POLICY.gram_rcond
    injected = decompose_gram(np.array([[1.0, 1.0 - 8.0 * eps], [1.0 - 8.0 * eps, 1.0]]))
    assert injected.rank == injected.width
    assert needs_factor_certification(injected)
    factor_calls = 0
    original_factor = scop_geometry.decompose_factor

    def counted_factor(factor):
        nonlocal factor_calls
        factor_calls += 1
        return original_factor(factor)

    monkeypatch.setattr(scop_geometry, "decompose_gram", lambda _matrix: injected)
    monkeypatch.setattr(scop_geometry, "decompose_factor", counted_factor)
    actual = scop_geometry.build_scop_postfit_inference(
        raw_fisher_gram=raw_gram,
        centered_fisher_gram=centered_gram,
        fisher_xtw=float(len(weights)) * mean_x,
        fisher_mean_x=mean_x,
        fisher_sum_w=float(len(weights)),
        latent_penalty=np.zeros((2, 2)),
        scop_states=states,
        groups=groups,
        observed_geometry=observed,
        dm=dm,
        fisher_weights=weights,
    )

    assert factor_calls == 2
    expected_coefficient, expected_augmented = _mapped_factor_inverses(
        design,
        weights,
        jacobian,
        np.zeros((2, 2)),
    )
    raw_factor = np.sqrt(weights)[:, None] * design
    augmented_factor = np.column_stack((np.sqrt(weights), raw_factor))
    _assert_covariance_factor_geometry(
        actual.coefficient_inverse,
        expected_coefficient,
        raw_factor,
    )
    _assert_covariance_factor_geometry(
        actual.augmented_inverse,
        expected_augmented,
        augmented_factor,
    )
    with pytest.raises(AssertionError):
        _assert_covariance_factor_geometry(
            injected.pseudo_inverse(),
            expected_coefficient,
            raw_factor,
        )


def test_scop_postfit_covariance_preserves_exact_alias_rank_convention():
    """Exact aliases retain the deterministic shared representative-column policy."""
    from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
    from superglm.reml.scop_geometry import build_scop_postfit_inference

    primary = np.linspace(-2.0, 2.0, 101)
    design = np.column_stack((primary, primary))
    weights = np.linspace(0.4, 1.7, len(primary))
    jacobian = np.array([0.5, 2.0])
    mean_x = np.average(design, axis=0, weights=weights)
    centered = design - mean_x
    centered_gram = centered.T @ (weights[:, None] * centered)
    raw_gram = centered_gram + np.sum(weights) * np.outer(mean_x, mean_x)
    states = {
        0: {
            "group_sl": slice(0, 2),
            "group_name": "mono",
            "gamma_eff": jacobian,
            "beta_eff": np.log(jacobian),
        }
    }
    groups = [GroupSlice(name="mono", start=0, end=2, monotone_engine="scop")]
    dm = DesignMatrix([DenseGroupMatrix(design)], n=len(design), p=2)

    actual = build_scop_postfit_inference(
        raw_fisher_gram=raw_gram,
        centered_fisher_gram=centered_gram,
        fisher_xtw=np.sum(weights) * mean_x,
        fisher_mean_x=mean_x,
        fisher_sum_w=float(np.sum(weights)),
        latent_penalty=np.zeros((2, 2)),
        scop_states=states,
        groups=groups,
        observed_geometry=_observed_geometry_from_centered(
            centered_hessian=np.eye(2),
            mean_x=np.zeros(2),
            sum_w=float(np.sum(weights)),
        ),
        dm=dm,
        fisher_weights=weights,
    )
    expected_coefficient, expected_augmented = _mapped_factor_inverses(
        design,
        weights,
        jacobian,
        np.zeros((2, 2)),
    )

    np.testing.assert_allclose(actual.coefficient_inverse, expected_coefficient, rtol=3e-15)
    np.testing.assert_allclose(actual.augmented_inverse, expected_augmented, rtol=3e-15)


@pytest.mark.parametrize(
    "jacobian",
    [np.array([1.0e-120, 1.0e120]), np.array([0.0, 1.0])],
    ids=["extreme-scale", "underflow-boundary"],
)
def test_scop_factor_certification_respects_jacobian_scaling_boundaries(jacobian):
    """Certification operates on XJ and maps back without ever forming J^-1."""
    from superglm.reml.scop_geometry import build_scop_postfit_inference

    (
        design,
        dm,
        weights,
        mean_x,
        centered_gram,
        raw_gram,
        _preliminary,
        _certified,
        states,
        groups,
        observed,
    ) = _rank_boundary_inputs(jacobian)
    actual = build_scop_postfit_inference(
        raw_fisher_gram=raw_gram,
        centered_fisher_gram=centered_gram,
        fisher_xtw=float(len(weights)) * mean_x,
        fisher_mean_x=mean_x,
        fisher_sum_w=float(len(weights)),
        latent_penalty=np.zeros((2, 2)),
        scop_states=states,
        groups=groups,
        observed_geometry=observed,
        dm=dm,
        fisher_weights=weights,
    )
    expected_coefficient, expected_augmented = _mapped_factor_inverses(
        design,
        weights,
        jacobian,
        np.zeros((2, 2)),
    )

    np.testing.assert_allclose(actual.coefficient_inverse, expected_coefficient, rtol=5e-14)
    np.testing.assert_allclose(actual.augmented_inverse, expected_augmented, rtol=5e-14)
    assert np.all(np.isfinite(actual.augmented_inverse))


def test_scop_postfit_well_conditioned_geometry_never_resolves_factor_rows():
    """The usual O(p^3) path does not materialize or revisit design rows."""
    from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
    from superglm.reml.scop_geometry import build_scop_postfit_inference

    rng = np.random.default_rng(20260727)
    design = rng.normal(size=(120, 2))
    weights = rng.uniform(0.5, 1.8, size=len(design))
    mean_x = np.average(design, axis=0, weights=weights)
    centered = design - mean_x
    centered_gram = centered.T @ (weights[:, None] * centered)
    raw_gram = centered_gram + np.sum(weights) * np.outer(mean_x, mean_x)
    calls = 0

    def weights_provider():
        nonlocal calls
        calls += 1
        return weights

    build_scop_postfit_inference(
        raw_fisher_gram=raw_gram,
        centered_fisher_gram=centered_gram,
        fisher_xtw=np.sum(weights) * mean_x,
        fisher_mean_x=mean_x,
        fisher_sum_w=float(np.sum(weights)),
        latent_penalty=np.eye(2),
        scop_states={},
        groups=[GroupSlice(name="x", start=0, end=2)],
        observed_geometry=_observed_geometry_from_centered(
            centered_hessian=centered_gram + np.eye(2),
            mean_x=mean_x,
            sum_w=float(np.sum(weights)),
        ),
        dm=DesignMatrix([DenseGroupMatrix(design)], n=len(design), p=2),
        fisher_weights=weights_provider,
    )

    assert calls == 0


def test_scop_centered_eq16_and_covariance_are_translation_invariant():
    """Eq. 16 is evaluated in the profiled coordinate system, without raw cancellation."""
    from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
    from superglm.reml.scop_geometry import build_scop_postfit_inference
    from superglm.solvers.rank import diagonal_of_square

    rng = np.random.default_rng(20260728)
    n = 180
    local_design = rng.normal(size=(n, 3))
    fisher_weights = rng.uniform(0.5, 1.7, size=n)
    observed_weights = rng.uniform(0.3, 2.2, size=n)
    jacobian = np.array([1.0, 0.4, 1.7])
    latent_penalty = np.array([[0.2, 0.0, 0.0], [0.0, 0.9, -0.25], [0.0, -0.25, 0.7]])
    states = {
        1: {
            "group_sl": slice(1, 3),
            "group_name": "mono",
            "gamma_eff": jacobian[1:],
            "beta_eff": np.log(jacobian[1:]),
        }
    }
    groups = [
        GroupSlice(name="ordinary", start=0, end=1),
        GroupSlice(name="mono", start=1, end=3, monotone_engine="scop"),
    ]
    results = []
    for translation in (0.0, 1.0e10):
        design = local_design.copy()
        design[:, 0] += translation
        fisher_mean = np.average(local_design, axis=0, weights=fisher_weights)
        observed_mean = np.average(local_design, axis=0, weights=observed_weights)
        fisher_mean[0] += translation
        observed_mean[0] += translation
        fisher_centered = design - fisher_mean
        observed_centered = design - observed_mean
        centered_fisher_gram = fisher_centered.T @ (fisher_weights[:, None] * fisher_centered)
        latent_observed_centered = observed_centered * jacobian
        observed_hessian = (
            latent_observed_centered.T @ (observed_weights[:, None] * latent_observed_centered)
            + latent_penalty
        )
        fisher_sum_w = float(np.sum(fisher_weights))
        observed_sum_w = float(np.sum(observed_weights))
        raw_fisher_gram = centered_fisher_gram + fisher_sum_w * np.outer(fisher_mean, fisher_mean)
        observed = _observed_geometry_from_centered(
            centered_hessian=observed_hessian,
            mean_x=observed_mean * jacobian,
            sum_w=observed_sum_w,
        )
        actual = build_scop_postfit_inference(
            raw_fisher_gram=raw_fisher_gram,
            centered_fisher_gram=centered_fisher_gram,
            fisher_xtw=fisher_sum_w * fisher_mean,
            fisher_mean_x=fisher_mean,
            fisher_sum_w=fisher_sum_w,
            latent_penalty=latent_penalty,
            scop_states=states,
            groups=groups,
            observed_geometry=observed,
            dm=DesignMatrix([DenseGroupMatrix(design)], n=n, p=3),
            fisher_weights=fisher_weights,
        )

        centered_expected_data = centered_fisher_gram * jacobian[:, None] * jacobian[None, :]
        delta_mean = fisher_mean * jacobian - observed_mean * jacobian
        expected_data_in_observed_coordinates = centered_expected_data + fisher_sum_w * np.outer(
            delta_mean, delta_mean
        )
        observed_inverse = decompose_gram(observed_hessian).pseudo_inverse()
        influence = np.empty((4, 4))
        influence[0, 0] = fisher_sum_w / observed_sum_w
        influence[0, 1:] = fisher_sum_w * delta_mean / observed_sum_w
        influence[1:, 0] = observed_inverse @ (fisher_sum_w * delta_mean)
        influence[1:, 1:] = observed_inverse @ expected_data_in_observed_coordinates
        expected_edf = np.diag(influence)
        expected_edf1 = 2.0 * expected_edf - diagonal_of_square(influence)
        np.testing.assert_allclose(actual.intercept_edf, expected_edf[0], rtol=3e-13)
        np.testing.assert_allclose(actual.feature_edf, expected_edf[1:], rtol=3e-13)
        np.testing.assert_allclose(actual.feature_edf1, expected_edf1[1:], rtol=3e-13)
        results.append(actual)

    np.testing.assert_allclose(results[1].feature_edf, results[0].feature_edf, rtol=3e-6)
    np.testing.assert_allclose(results[1].feature_edf1, results[0].feature_edf1, rtol=3e-6)
    np.testing.assert_allclose(
        results[1].augmented_inverse[1:, 1:],
        results[0].augmented_inverse[1:, 1:],
        rtol=3e-6,
    )
    assert results[1].intercept_edf == pytest.approx(results[0].intercept_edf, rel=3e-6)
    assert results[1].total_edf == pytest.approx(results[0].total_edf, rel=3e-6)


@pytest.mark.slow
def test_gamma_log_reml_publishes_pya_wood_covariance_and_edf():
    """The terminal public state exposes Fisher covariance and observed-Hessian EDF."""
    from superglm import Constraint, SuperGLM
    from superglm.distributions import Gamma, clip_mu
    from superglm.features.spline import PSpline
    from superglm.links import stabilize_eta
    from superglm.reml.penalty_algebra import build_penalty_matrix
    from superglm.reml.scop_geometry import (
        build_observed_scop_joint_geometry,
        build_scop_postfit_inference,
    )

    rng = np.random.default_rng(20260723)
    n = 240
    x = np.sort(rng.uniform(0.0, 1.0, size=n))
    mean = np.exp(-0.3 + 1.05 * x)
    y = mean * rng.gamma(shape=8.0, scale=1.0 / 8.0, size=n)
    frame = pd.DataFrame({"x": x})
    model = SuperGLM(
        family=Gamma(),
        selection_penalty=0.0,
        discrete=True,
        features={"x": PSpline(n_knots=7, constraint=Constraint.fit.increasing)},
    )

    model.fit_reml(frame, y, max_reml_iter=3, max_pirls_iter=100)

    result = model._solver_result
    reml_result = model._reml_result
    states = reml_result.scop_states
    assert states
    published = getattr(result, "scop_inference", None)
    published_geometry = getattr(result, "scop_geometry", None)
    assert published is not None
    assert published_geometry is not None

    latent_penalty = build_penalty_matrix(
        model._dm.group_matrices,
        model._groups,
        reml_result.lambdas,
        model._dm.p,
        reml_penalties=reml_result.reml_penalties,
    )
    eta = model._dm.matvec(result.beta) + result.intercept
    if model._fit_offset is not None:
        eta = eta + model._fit_offset
    eta = stabilize_eta(eta, model._link)
    mu = clip_mu(model._link.inverse(eta), model._distribution)
    fisher_weights = (
        model._fit_weights
        * np.asarray(model._link.deriv_inverse(eta), dtype=np.float64) ** 2
        / np.asarray(model._distribution.variance(mu), dtype=np.float64)
    )
    moments = model._dm.execution_plan.moments(fisher_weights, include_xtw=True)
    assert moments.xtw is not None
    observed = build_observed_scop_joint_geometry(
        dm=model._dm,
        distribution=model._distribution,
        link=model._link,
        y=y,
        sample_weight=model._fit_weights,
        offset_arr=np.zeros(n) if model._fit_offset is None else model._fit_offset,
        result=result,
        penalty=latent_penalty,
        scop_states=states,
        fisher_XtWX=moments.gram,
        fisher_XtW1=moments.xtw,
        fisher_sum_W=float(np.sum(fisher_weights)),
    )
    expected = build_scop_postfit_inference(
        raw_fisher_gram=moments.gram,
        fisher_xtw=moments.xtw,
        fisher_sum_w=float(np.sum(fisher_weights)),
        latent_penalty=latent_penalty,
        scop_states=states,
        groups=model._groups,
        observed_geometry=observed,
    )

    np.testing.assert_allclose(
        published.augmented_inverse,
        expected.augmented_inverse,
        rtol=2e-9,
        atol=2e-9,
    )
    np.testing.assert_allclose(published.feature_edf, expected.feature_edf, rtol=2e-9, atol=2e-9)
    assert published.intercept_edf == pytest.approx(
        expected.intercept_edf,
        rel=2e-9,
        abs=2e-9,
    )
    assert result.effective_df == pytest.approx(expected.total_edf, rel=2e-9, abs=2e-9)
    assert result.rank_info is not None
    np.testing.assert_allclose(result.rank_info.feature_edf, expected.feature_edf)
    assert result.rank_info.intercept_edf == pytest.approx(expected.intercept_edf)
    assert dict(result.rank_info.group_edf) == pytest.approx(dict(expected.group_edf))
    covariance, _ = model._coef_covariance
    np.testing.assert_allclose(
        covariance,
        result.phi * expected.augmented_inverse[1:, 1:],
        rtol=2e-9,
        atol=2e-9,
    )


@pytest.mark.slow
@pytest.mark.parametrize("fit_kind", ["fit", "fixed_reml"])
def test_fixed_scop_fit_lifecycle_builds_postfit_inference_once(monkeypatch, fit_kind):
    """Ordinary and fixed-REML fits install one terminal inference object."""
    import superglm.reml.scop_geometry as scop_geometry
    from superglm import Constraint, LambdaPolicy, SuperGLM
    from superglm.features.spline import PSpline

    calls = 0
    original = scop_geometry.build_scop_postfit_inference

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(scop_geometry, "build_scop_postfit_inference", counted)
    rng = np.random.default_rng(20260724)
    n = 260
    x = np.sort(rng.uniform(size=n))
    y = 0.2 + 1.7 * x + rng.normal(0.0, 0.18, size=n)
    frame = pd.DataFrame({"x": x})
    if fit_kind == "fit":
        spline = PSpline(n_knots=7, constraint=Constraint.fit.increasing)
        model = SuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            spline_penalty=1.7,
            features={"x": spline},
        )
        model.fit(frame, y)
    else:
        spline = PSpline(
            n_knots=7,
            constraint=Constraint.fit.increasing,
            lambda_policy=LambdaPolicy(mode="fixed", value=1.7),
        )
        model = SuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            features={"x": spline},
        )
        model.fit_reml(frame, y)

    result = model._solver_result
    inference = getattr(result, "scop_inference", None)
    assert inference is not None
    assert getattr(result, "scop_geometry", None) is not None
    assert calls == 1
    assert result.effective_df == pytest.approx(inference.total_edf)
    assert result.rank_info is not None
    np.testing.assert_allclose(result.rank_info.feature_edf, inference.feature_edf)
    covariance, _ = model._coef_covariance
    np.testing.assert_allclose(
        covariance,
        result.phi * inference.augmented_inverse[1:, 1:],
        rtol=2e-10,
        atol=2e-10,
    )
    _, _, coefficient_inverse, augmented_inverse, _ = model._fit_active_info
    np.testing.assert_allclose(
        coefficient_inverse,
        inference.coefficient_inverse,
        rtol=2e-10,
        atol=2e-10,
    )
    np.testing.assert_allclose(
        augmented_inverse[1:, 1:],
        inference.augmented_inverse[1:, 1:],
        rtol=2e-10,
        atol=2e-10,
    )
    summary_inference = model._fit_inference_info
    np.testing.assert_allclose(
        summary_inference["XtWX_inv"],
        inference.coefficient_inverse,
        rtol=2e-10,
        atol=2e-10,
    )
    np.testing.assert_allclose(summary_inference["edf"], inference.feature_edf)
    np.testing.assert_allclose(summary_inference["edf1"], inference.feature_edf1)
    assert summary_inference["group_edf_map"] == pytest.approx(dict(inference.group_edf))
    metrics = model.metrics(frame, y)
    _, _, metrics_inverse, metrics_augmented, _ = metrics._active_info
    np.testing.assert_allclose(metrics_inverse, inference.coefficient_inverse)
    np.testing.assert_allclose(
        metrics_augmented[1:, 1:],
        inference.augmented_inverse[1:, 1:],
    )
    metrics_edf, metrics_edf1 = metrics._influence_edf
    np.testing.assert_allclose(metrics_edf, inference.feature_edf)
    np.testing.assert_allclose(metrics_edf1, inference.feature_edf1)

    # A value-identical evaluation frame must not switch SCOP inference back
    # to ordinary coefficient coordinates merely because its object identity
    # differs from the frame retained at fit time.
    copied_metrics = model.metrics(frame.copy(), y.copy())
    _, _, copied_inverse, copied_augmented, _ = copied_metrics._active_info
    np.testing.assert_allclose(copied_inverse, inference.coefficient_inverse)
    np.testing.assert_allclose(
        copied_augmented[1:, 1:],
        inference.augmented_inverse[1:, 1:],
    )
    copied_edf, copied_edf1 = copied_metrics._influence_edf
    np.testing.assert_allclose(copied_edf, inference.feature_edf)
    np.testing.assert_allclose(copied_edf1, inference.feature_edf1)
    np.testing.assert_allclose(copied_metrics.leverage, metrics.leverage)

    evaluation_frame = pd.DataFrame({"x": np.linspace(0.05, 0.95, 31)})
    evaluation_metrics = model.metrics(
        evaluation_frame,
        model.predict(evaluation_frame),
    )
    evaluation_design, evaluation_weights, evaluation_inverse, _, _ = (
        evaluation_metrics._active_info
    )
    np.testing.assert_allclose(evaluation_inverse, inference.coefficient_inverse)
    evaluation_dense = evaluation_design.toarray()
    expected_leverage = evaluation_weights * np.sum(
        (evaluation_dense @ inference.coefficient_inverse) * evaluation_dense,
        axis=1,
    )
    np.testing.assert_allclose(
        evaluation_metrics.leverage,
        np.clip(expected_leverage, 0.0, 1.0),
    )


@pytest.mark.slow
def test_estimated_scop_reml_builds_postfit_inference_only_for_terminal_mode(monkeypatch):
    """Outer coefficient candidates do not pay for unused post-fit decompositions."""
    import superglm.reml.scop_geometry as scop_geometry
    from superglm import Constraint, SuperGLM
    from superglm.features.spline import PSpline

    calls = 0
    original = scop_geometry.build_scop_postfit_inference

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(scop_geometry, "build_scop_postfit_inference", counted)
    rng = np.random.default_rng(20260725)
    n = 280
    x = np.sort(rng.uniform(size=n))
    y = 0.4 + 1.3 * x + rng.normal(0.0, 0.22, size=n)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        discrete=True,
        features={"x": PSpline(n_knots=7, constraint=Constraint.fit.increasing)},
    )

    model.fit_reml(pd.DataFrame({"x": x}), y, max_reml_iter=3, max_pirls_iter=100)

    assert calls == 1
    assert model._solver_result.scop_inference is not None
