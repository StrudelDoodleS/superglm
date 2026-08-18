"""Observed-information geometry for Wood's LAML criterion."""

from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest

from superglm.distributions import (
    Binomial,
    Gamma,
    Gaussian,
    NegativeBinomial,
    Poisson,
    Tweedie,
    clip_mu,
)
from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
from superglm.links import (
    CauchitLink,
    CloglogLink,
    IdentityLink,
    LogLink,
    ProbitLink,
    SqrtLink,
)
from superglm.reml.gradient import reml_direct_gradient, reml_direct_hessian
from superglm.reml.objective import REMLObjectiveEvaluation, reml_laml_objective
from superglm.reml.observed_geometry import (
    build_observed_reml_geometry,
    compute_observed_d2W_deta2,
    compute_observed_dW_deta,
    compute_observed_information_weights,
    observed_penalized_mode_score,
)
from superglm.reml.scale import prepare_gamma_reml_scale_data
from superglm.reml.w_derivatives import reml_w_correction
from superglm.solvers.centered_system import grouped_augmented_factor
from superglm.solvers.irls_direct import fit_irls_direct
from superglm.solvers.pirls import PIRLSResult
from superglm.solvers.rank import (
    SHARED_RANK_POLICY,
    decompose_factor,
    decompose_gram,
    needs_factor_certification,
)
from superglm.types import GroupSlice, PenaltyComponent


def test_ordinary_reml_curvature_classifier_requires_exact_proof() -> None:
    from superglm.distributions import NegativeBinomial, Tweedie
    from superglm.links import (
        InverseLink,
        LogitLink,
        NegativeBinomialLink,
        PowerLink,
        ProbitLink,
    )
    from superglm.reml.observed_geometry import classify_reml_curvature

    fisher_pairs = [
        (Gaussian(), IdentityLink()),
        (Poisson(), LogLink()),
        (Binomial(), LogitLink()),
        (Gamma(), InverseLink()),
        (NegativeBinomial(theta=2.5), NegativeBinomialLink(theta=2.5)),
        (Tweedie(p=1.6), PowerLink(power=-0.6)),
    ]
    for distribution, link in fisher_pairs:
        assert classify_reml_curvature(distribution, link) == "fisher"

    assert classify_reml_curvature(NegativeBinomial(theta=2.5), LogLink()) == "observed"
    assert classify_reml_curvature(Tweedie(p=1.6), LogLink()) == "observed"
    assert classify_reml_curvature(Binomial(), ProbitLink()) == "observed"

    class CustomGaussian(Gaussian):
        pass

    with pytest.raises(NotImplementedError, match="explicit ordinary REML curvature"):
        classify_reml_curvature(CustomGaussian(), IdentityLink())


def test_ordinary_reml_curvature_classifier_requires_consistent_custom_protocol() -> None:
    from superglm.reml.observed_geometry import classify_reml_curvature

    class CustomDistribution:
        def reml_curvature(self, link):
            return "observed"

    class CustomLink:
        pass

    assert classify_reml_curvature(CustomDistribution(), CustomLink()) == "observed"

    class ConflictingLink:
        def reml_curvature(self, distribution):
            return "fisher"

    with pytest.raises(ValueError, match="protocols disagree"):
        classify_reml_curvature(CustomDistribution(), ConflictingLink())

    class InvalidDistribution:
        def reml_curvature(self, link):
            return "approximate"

    with pytest.raises(ValueError, match="must return 'fisher' or 'observed'"):
        classify_reml_curvature(InvalidDistribution(), CustomLink())


def test_gamma_log_observed_rows_match_glum_and_likelihood_finite_difference() -> None:
    """Gamma/log has observed rows w*y/mu, not constant Fisher rows w."""
    distribution = Gamma()
    link = LogLink()
    y = np.array([0.31, 0.8, 1.7, 4.2], dtype=np.float64)
    eta = np.array([-1.1, -0.2, 0.4, 1.0], dtype=np.float64)
    mu = link.inverse(eta)
    sample_weight = np.array([0.4, 1.0, 2.5, 3.0], dtype=np.float64)

    observed = compute_observed_information_weights(
        distribution,
        link,
        y,
        mu,
        eta,
        sample_weight,
    )
    expected = sample_weight * y / mu
    np.testing.assert_allclose(observed, expected, rtol=2e-15, atol=0.0)

    eps = 2e-5
    nll_base = -sample_weight * (np.log(y / mu) - y / mu - np.log(y))
    mu_plus = link.inverse(eta + eps)
    mu_minus = link.inverse(eta - eps)
    nll_plus = -sample_weight * (np.log(y / mu_plus) - y / mu_plus - np.log(y))
    nll_minus = -sample_weight * (np.log(y / mu_minus) - y / mu_minus - np.log(y))
    fd_hessian = (nll_plus - 2.0 * nll_base + nll_minus) / eps**2
    np.testing.assert_allclose(observed, fd_hessian, rtol=3e-6, atol=2e-6)


def test_observed_weight_derivatives_are_exact_for_gamma_and_poisson_log() -> None:
    link = LogLink()
    eta = np.array([-0.7, 0.1, 0.9], dtype=np.float64)
    mu = link.inverse(eta)
    y = np.array([0.4, 1.3, 3.7], dtype=np.float64)
    sample_weight = np.array([0.5, 1.7, 2.2], dtype=np.float64)

    gamma_w = compute_observed_information_weights(Gamma(), link, y, mu, eta, sample_weight)
    gamma_d1 = compute_observed_dW_deta(Gamma(), link, y, mu, eta, sample_weight)
    gamma_d2 = compute_observed_d2W_deta2(Gamma(), link, y, mu, eta, sample_weight)
    np.testing.assert_allclose(gamma_d1, -gamma_w, rtol=2e-15, atol=0.0)
    np.testing.assert_allclose(gamma_d2, gamma_w, rtol=2e-9, atol=2e-10)

    poisson_w = compute_observed_information_weights(Poisson(), link, y, mu, eta, sample_weight)
    poisson_d1 = compute_observed_dW_deta(Poisson(), link, y, mu, eta, sample_weight)
    poisson_d2 = compute_observed_d2W_deta2(Poisson(), link, y, mu, eta, sample_weight)
    fisher = sample_weight * mu
    np.testing.assert_allclose(poisson_w, fisher, rtol=2e-15, atol=0.0)
    np.testing.assert_allclose(poisson_d1, fisher, rtol=2e-15, atol=0.0)
    np.testing.assert_allclose(poisson_d2, fisher, rtol=2e-9, atol=2e-10)


@pytest.mark.parametrize(
    ("distribution", "link", "eta", "y"),
    [
        pytest.param(
            NegativeBinomial(theta=2.3),
            LogLink(),
            np.log(np.array([0.45, 1.4, 3.2])),
            np.array([0.0, 2.0, 5.0]),
            id="nb2-log",
        ),
        pytest.param(
            Tweedie(p=1.55),
            LogLink(),
            np.log(np.array([0.35, 1.1, 2.7])),
            np.array([0.0, 0.8, 3.4]),
            id="tweedie-log",
        ),
        pytest.param(
            Binomial(),
            ProbitLink(),
            np.array([-0.7, 0.1, 0.8]),
            np.array([0.0, 1.0, 0.0]),
            id="binomial-probit",
        ),
        pytest.param(
            Binomial(),
            CloglogLink(),
            np.array([-0.8, -0.1, 0.6]),
            np.array([0.0, 1.0, 1.0]),
            id="binomial-cloglog",
        ),
        pytest.param(
            Binomial(),
            CauchitLink(),
            np.array([-0.7, 0.2, 0.8]),
            np.array([0.0, 1.0, 0.0]),
            id="binomial-cauchit",
        ),
        pytest.param(
            Poisson(),
            SqrtLink(),
            np.array([0.7, 1.1, 1.6]),
            np.array([0.0, 1.0, 4.0]),
            id="poisson-sqrt",
        ),
        pytest.param(
            Poisson(),
            IdentityLink(),
            np.array([0.7, 1.2, 2.2]),
            np.array([0.0, 1.0, 4.0]),
            id="poisson-identity",
        ),
        pytest.param(
            Gamma(),
            IdentityLink(),
            np.array([0.6, 1.1, 2.0]),
            np.array([0.4, 1.4, 2.8]),
            id="gamma-identity",
        ),
        pytest.param(
            Gamma(),
            LogLink(),
            np.log(np.array([0.6, 1.1, 2.0])),
            np.array([0.4, 1.4, 2.8]),
            id="gamma-log",
        ),
        pytest.param(
            Gaussian(),
            LogLink(),
            np.log(np.array([0.7, 1.3, 2.1])),
            np.array([0.3, 1.6, 2.8]),
            id="gaussian-log",
        ),
    ],
)
def test_observed_row_derivative_oracle_matches_likelihood_finite_differences(
    distribution, link, eta, y
) -> None:
    sample_weight = np.array([0.6, 1.2, 1.9])

    def observed(delta: float) -> np.ndarray:
        shifted_eta = eta + delta
        shifted_mu = clip_mu(link.inverse(shifted_eta), distribution)
        return compute_observed_information_weights(
            distribution,
            link,
            y,
            shifted_mu,
            shifted_eta,
            sample_weight,
        )

    def negative_log_likelihood_rows(delta: float) -> np.ndarray:
        shifted_mu = clip_mu(link.inverse(eta + delta), distribution)
        return 0.5 * sample_weight * distribution.deviance_unit(y, shifted_mu)

    mu = clip_mu(link.inverse(eta), distribution)
    actual_w = observed(0.0)
    actual_d1 = compute_observed_dW_deta(
        distribution,
        link,
        y,
        mu,
        eta,
        sample_weight,
    )
    actual_d2 = compute_observed_d2W_deta2(
        distribution,
        link,
        y,
        mu,
        eta,
        sample_weight,
    )

    likelihood_step = 4.0e-4
    expected_w = (
        -negative_log_likelihood_rows(2.0 * likelihood_step)
        + 16.0 * negative_log_likelihood_rows(likelihood_step)
        - 30.0 * negative_log_likelihood_rows(0.0)
        + 16.0 * negative_log_likelihood_rows(-likelihood_step)
        - negative_log_likelihood_rows(-2.0 * likelihood_step)
    ) / (12.0 * likelihood_step**2)
    derivative_step = 2.0e-4
    expected_d1 = (
        observed(-2.0 * derivative_step)
        - 8.0 * observed(-derivative_step)
        + 8.0 * observed(derivative_step)
        - observed(2.0 * derivative_step)
    ) / (12.0 * derivative_step)
    expected_d2 = (
        -observed(2.0 * derivative_step)
        + 16.0 * observed(derivative_step)
        - 30.0 * observed(0.0)
        + 16.0 * observed(-derivative_step)
        - observed(-2.0 * derivative_step)
    ) / (12.0 * derivative_step**2)

    np.testing.assert_allclose(actual_w, expected_w, rtol=3e-6, atol=2e-6)
    np.testing.assert_allclose(actual_d1, expected_d1, rtol=2e-7, atol=2e-8)
    np.testing.assert_allclose(actual_d2, expected_d2, rtol=3e-6, atol=3e-7)


def test_custom_second_observed_derivative_requires_exact_fourth_order_protocol() -> None:
    class CustomGaussian(Gaussian):
        pass

    class CustomIdentity(IdentityLink):
        pass

    eta = np.array([-0.7, 0.2, 0.8])
    link = CustomIdentity()
    mu = link.inverse(eta)
    y = np.array([0.0, 1.0, 0.0])
    weights = np.ones_like(y)

    with pytest.raises(NotImplementedError, match="deriv4_inverse"):
        compute_observed_d2W_deta2(CustomGaussian(), link, y, mu, eta, weights)


def _result(beta: np.ndarray, intercept: float, *, phi: float = 1.0) -> PIRLSResult:
    return PIRLSResult(
        beta=beta,
        intercept=intercept,
        n_iter=1,
        deviance=0.0,
        converged=True,
        phi=phi,
        effective_df=0.0,
    )


def _roundoff_gamma(operation_count: int) -> float:
    eps = np.finfo(np.float64).eps
    return operation_count * eps / (1.0 - operation_count * eps)


def _assert_observed_cutoff_geometry(
    inverse: np.ndarray,
    slope_log_pdet: float,
    certificate,
    factor: np.ndarray,
) -> None:
    """Check only resolved rank, support, projector, and log-pdet at the cutoff."""
    factor = np.asarray(factor, dtype=np.float64)
    inverse = np.asarray(inverse, dtype=np.float64)
    width = factor.shape[1]
    column_scale = np.linalg.norm(factor, axis=0)
    active = np.flatnonzero(column_scale > 0.0)
    assert len(active) == width
    equilibrated = factor / column_scale
    equilibrated_singular_values = np.linalg.svd(equilibrated, compute_uv=False)
    cutoff = SHARED_RANK_POLICY.factor_rcond * equilibrated_singular_values[0]
    rank = int(np.count_nonzero(equilibrated_singular_values > cutoff))
    assert rank == width
    gap = float(equilibrated_singular_values[rank - 1] - cutoff)
    eta_factor = (
        64.0 * _roundoff_gamma(max(factor.shape)) * float(np.linalg.norm(equilibrated, ord=2))
    )
    assert gap > 2.0 * eta_factor
    assert equilibrated_singular_values[rank - 1] - eta_factor > (
        SHARED_RANK_POLICY.factor_rcond * (equilibrated_singular_values[0] + eta_factor)
    )
    projector_bound = 2.0 * eta_factor / (gap - 2.0 * eta_factor)

    assert certificate.rank == rank
    assert certificate.method == "qr_svd"
    null = np.asarray(certificate.parameter_null_basis, dtype=np.float64)
    null_projector = null @ np.linalg.pinv(null)
    assert np.linalg.norm(null_projector, ord=2) <= projector_bound

    selected = np.asarray(certificate.active_columns, dtype=np.intp)
    inactive = np.setdiff1d(np.arange(width), selected, assume_unique=True)
    assert inverse.shape == (width, width)
    assert np.all(np.isfinite(inverse))
    assert not np.any(inverse[inactive, :])
    assert not np.any(inverse[:, inactive])

    singular_values = np.linalg.svd(factor, compute_uv=False)[:rank]
    factor_error = eta_factor * float(np.max(column_scale))
    smallest = float(singular_values[-1])
    assert smallest > factor_error

    expected_log_pdet = 2.0 * float(np.sum(np.log(singular_values)))
    log_summation_error = (
        2.0 * _roundoff_gamma(rank) * float(np.sum(np.abs(np.log(singular_values))))
    )
    log_bound = 2.0 * rank * factor_error / (smallest - factor_error) + log_summation_error
    assert abs(slope_log_pdet - expected_log_pdet) <= log_bound


def _assert_well_conditioned_observed_inverse(
    inverse: np.ndarray,
    factor: np.ndarray,
) -> SimpleNamespace:
    """Bound public covariance and fitted prediction from an independent QR."""
    factor = np.asarray(factor, dtype=np.float64)
    inverse = np.asarray(inverse, dtype=np.float64)
    width = factor.shape[1]
    orthogonal, triangular = np.linalg.qr(factor, mode="reduced")
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
    action_error = float(np.linalg.norm(actual_action - reference_action, ord=2))
    assert action_error <= action_bound

    probe = np.linspace(-1.0, 1.0, factor.shape[0])
    prediction_error = float(np.linalg.norm((actual_action - reference_action) @ probe))
    prediction_bound = action_bound * float(np.linalg.norm(probe))
    assert prediction_error <= prediction_bound
    return SimpleNamespace(
        backward=backward,
        beta=beta,
        condition=condition,
        conditioned_beta=conditioned_beta,
        forward_bound=forward_bound,
        relative_inverse_error=relative_inverse_error,
        action_error=action_error,
        action_bound=action_bound,
        prediction_error=prediction_error,
        prediction_bound=prediction_bound,
    )


class _ExponentialVarianceGaussian(Gaussian):
    """Identity-link test family with controllable signed observed rows."""

    def variance(self, mu: np.ndarray) -> np.ndarray:
        return np.exp(mu)

    def variance_derivative(self, mu: np.ndarray) -> np.ndarray:
        return np.exp(mu)

    def variance_second_derivative(self, mu: np.ndarray) -> np.ndarray:
        return np.exp(mu)


def test_intercept_only_signed_geometry_preserves_near_canceling_positive_sum() -> None:
    dm = DesignMatrix([], n=3, p=0)
    # At eta=mu=0 with V(mu)=exp(mu), observed rows are exactly 1 + y.
    # Ordinary summation loses the central unit between the two 1e16 rows,
    # while the mathematical intercept curvature is positive and equal to 1.
    y = np.array([1e16, 0.0, -1e16])
    geometry = build_observed_reml_geometry(
        dm=dm,
        distribution=_ExponentialVarianceGaussian(),
        link=IdentityLink(),
        y=y,
        sample_weight=np.ones(3),
        offset_arr=np.zeros(3),
        result=_result(np.zeros(0), 0.0),
        penalty=np.zeros((0, 0)),
    )

    assert geometry.sum_w == 1.0
    assert geometry.hessian_rank == 1
    assert geometry.mean_x.shape == (0,)
    assert geometry.centered_hessian.shape == (0, 0)
    assert geometry.hessian_inverse is not None
    assert geometry.hessian_inverse.shape == (0, 0)
    assert geometry.log_det_H == pytest.approx(0.0, abs=0.0)


def test_intercept_only_signed_geometry_avoids_compensated_sum_overflow() -> None:
    dm = DesignMatrix([], n=3, p=0)
    geometry = build_observed_reml_geometry(
        dm=dm,
        distribution=_ExponentialVarianceGaussian(),
        link=IdentityLink(),
        y=np.array([1e308, 1e308, -1e308]),
        sample_weight=np.ones(3),
        offset_arr=np.zeros(3),
        result=_result(np.zeros(0), 0.0),
        penalty=np.zeros((0, 0)),
    )

    assert geometry.sum_w == 1e308
    assert geometry.log_det_H == pytest.approx(float(np.log(1e308)))
    assert geometry.hessian_rank == 1


@pytest.mark.parametrize("translation", [0.0, 1e10])
def test_signed_geometry_is_stable_under_large_feature_translation(
    translation: float,
) -> None:
    X = np.array([translation, translation + 1.0, translation])[:, None]
    dm = DesignMatrix([DenseGroupMatrix(X)], n=3, p=1)
    geometry = build_observed_reml_geometry(
        dm=dm,
        distribution=_ExponentialVarianceGaussian(),
        link=IdentityLink(),
        y=np.array([1e16, 0.0, -1e16]),
        sample_weight=np.ones(3),
        offset_arr=np.zeros(3),
        result=_result(np.zeros(1), 0.0),
        penalty=np.ones((1, 1)),
    )

    assert geometry.sum_w == 1.0
    np.testing.assert_allclose(geometry.mean_x, [translation + 1.0], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(geometry.centered_data_gram, [[0.0]], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(geometry.centered_hessian, [[1.0]], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(geometry.hessian_inverse, [[1.0]], rtol=0.0, atol=0.0)
    assert geometry.log_det_H == pytest.approx(0.0, abs=0.0)
    assert geometry.hessian_rank == 2


@pytest.mark.parametrize("y", [np.array([0.0, -2.0]), np.array([0.0, -3.0])])
def test_signed_geometry_rejects_nonpositive_intercept_curvature(y: np.ndarray) -> None:
    dm = DesignMatrix([], n=2, p=0)
    with pytest.raises(ValueError, match="positive finite sum"):
        build_observed_reml_geometry(
            dm=dm,
            distribution=_ExponentialVarianceGaussian(),
            link=IdentityLink(),
            y=y,
            sample_weight=np.ones(2),
            offset_arr=np.zeros(2),
            result=_result(np.zeros(0), 0.0),
            penalty=np.zeros((0, 0)),
        )


def test_gamma_observed_geometry_is_invariant_to_fitted_dispersion() -> None:
    x = np.linspace(-1.0, 1.0, 15)
    X = np.column_stack((x, x**2))
    dm = DesignMatrix([DenseGroupMatrix(X)], n=len(x), p=2)
    beta = np.array([0.25, -0.1])
    intercept = 0.3
    mu = np.exp(intercept + X @ beta)
    y = mu * np.linspace(0.4, 1.8, len(mu))
    kwargs = dict(
        dm=dm,
        distribution=Gamma(),
        link=LogLink(),
        y=y,
        sample_weight=np.linspace(0.5, 1.7, len(mu)),
        offset_arr=np.zeros(len(mu)),
        penalty=np.array([[0.8, 0.1], [0.1, 1.2]]),
        derivative_order=2,
    )

    low_phi = build_observed_reml_geometry(
        **kwargs,
        result=_result(beta, intercept, phi=0.2),
    )
    high_phi = build_observed_reml_geometry(
        **kwargs,
        result=_result(beta, intercept, phi=7.5),
    )

    for field in (
        "eta",
        "mu",
        "weights",
        "weight_derivative",
        "weight_second_derivative",
        "mean_x",
        "centered_data_gram",
        "centered_hessian",
        "hessian_inverse",
    ):
        np.testing.assert_array_equal(getattr(low_phi, field), getattr(high_phi, field))
    assert low_phi.sum_w == high_phi.sum_w
    assert low_phi.log_det_H == high_phi.log_det_H
    assert low_phi.hessian_rank == high_phi.hessian_rank


def test_gamma_intercept_only_geometry_has_scalar_determinant() -> None:
    dm = DesignMatrix([], n=4, p=0)
    y = np.array([0.5, 1.0, 2.0, 4.0])
    sample_weight = np.array([0.3, 0.7, 1.2, 2.0])
    mu = np.full(4, 2.0)
    geometry = build_observed_reml_geometry(
        dm=dm,
        distribution=Gamma(),
        link=LogLink(),
        y=y,
        sample_weight=sample_weight,
        offset_arr=np.zeros(4),
        result=_result(np.zeros(0), float(np.log(2.0)), phi=3.0),
        penalty=np.zeros((0, 0)),
        derivative_order=2,
    )

    expected_w = sample_weight * y / mu
    np.testing.assert_array_equal(geometry.weights, expected_w)
    assert geometry.sum_w == pytest.approx(float(np.sum(expected_w)))
    assert geometry.log_det_H == pytest.approx(float(np.log(np.sum(expected_w))))
    assert geometry.hessian_rank == 1
    assert geometry.mean_x.shape == (0,)
    assert geometry.centered_hessian.shape == (0, 0)
    assert geometry.hessian_inverse is not None
    assert geometry.hessian_inverse.shape == (0, 0)


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (_result(np.zeros(2), 0.0), "result.beta"),
        (_result(np.array([np.nan]), 0.0), "result.beta"),
        (_result(np.zeros(1), np.inf), "result.intercept"),
    ],
)
def test_observed_geometry_validates_fitted_coefficient_state(
    result: PIRLSResult,
    message: str,
) -> None:
    dm = DesignMatrix([DenseGroupMatrix(np.arange(4.0)[:, None])], n=4, p=1)
    with pytest.raises(ValueError, match=message):
        build_observed_reml_geometry(
            dm=dm,
            distribution=Gamma(),
            link=LogLink(),
            y=np.ones(4),
            sample_weight=np.ones(4),
            offset_arr=np.zeros(4),
            result=result,
            penalty=np.ones((1, 1)),
        )


def test_gamma_observed_geometry_includes_nonzero_offsets() -> None:
    x = np.linspace(-1.0, 1.0, 9)
    X = x[:, None]
    dm = DesignMatrix([DenseGroupMatrix(X)], n=len(x), p=1)
    result = _result(np.array([0.3]), -0.2)
    offset = np.linspace(-0.4, 0.5, len(x))
    expected_eta = result.intercept + X[:, 0] * result.beta[0] + offset
    expected_mu = np.exp(expected_eta)
    sample_weight = np.linspace(0.5, 1.3, len(x))
    y = expected_mu * np.linspace(0.6, 1.5, len(x))
    geometry = build_observed_reml_geometry(
        dm=dm,
        distribution=Gamma(),
        link=LogLink(),
        y=y,
        sample_weight=sample_weight,
        offset_arr=offset,
        result=result,
        penalty=np.array([[0.7]]),
    )

    np.testing.assert_allclose(geometry.eta, expected_eta, rtol=0.0, atol=2e-16)
    np.testing.assert_allclose(geometry.mu, expected_mu, rtol=2e-15, atol=0.0)
    np.testing.assert_allclose(
        geometry.weights,
        sample_weight * y / expected_mu,
        rtol=2e-15,
        atol=0.0,
    )


def test_observed_geometry_retains_exact_alias_rank() -> None:
    x = np.where(np.arange(128) % 2, 1.0, -1.0)
    X = np.column_stack((x, x))
    dm = DesignMatrix([DenseGroupMatrix(X)], n=len(x), p=2)
    geometry = build_observed_reml_geometry(
        dm=dm,
        distribution=Gamma(),
        link=LogLink(),
        y=np.ones(len(x)),
        sample_weight=np.ones(len(x)),
        offset_arr=np.zeros(len(x)),
        result=_result(np.zeros(2), 0.0),
        penalty=np.zeros((2, 2)),
    )

    assert geometry.hessian_rank == 2  # intercept plus one identified slope
    expected = decompose_factor(X)
    assert expected.rank == 1
    np.testing.assert_allclose(
        geometry.hessian_inverse,
        expected.pseudo_inverse(),
        rtol=2e-15,
        atol=0.0,
    )
    assert geometry.log_det_H == pytest.approx(
        float(np.log(len(x)) + expected.log_pdet),
        rel=2e-15,
        abs=0.0,
    )


def test_observed_geometry_uses_factor_certification_at_rank_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import superglm.reml.observed_geometry as observed_geometry

    rows = np.arange(128)
    primary = np.where(rows % 2, 1.0, -1.0)
    orthogonal = np.where(rows % 4 < 2, 1.0, -1.0)
    X = np.column_stack((primary, primary + 3.03e-8 * orthogonal))
    weights = np.ones(len(rows))
    dm = DesignMatrix([DenseGroupMatrix(X)], n=len(rows), p=2)
    factor = grouped_augmented_factor(
        dm,
        weights,
        np.zeros((2, 2)),
        center=np.zeros(2),
    )
    preliminary = decompose_gram(X.T @ X)
    assert needs_factor_certification(preliminary)
    original_factor = observed_geometry.decompose_factor
    factor_calls = 0
    factor_certificates = []

    def counted_factor(factor, *args, **kwargs):
        nonlocal factor_calls
        factor_calls += 1
        certificate = original_factor(factor, *args, **kwargs)
        factor_certificates.append(certificate)
        return certificate

    monkeypatch.setattr(observed_geometry, "decompose_factor", counted_factor)
    geometry = observed_geometry.build_observed_reml_geometry(
        dm=dm,
        distribution=Gamma(),
        link=LogLink(),
        y=np.ones(len(rows)),
        sample_weight=weights,
        offset_arr=np.zeros(len(rows)),
        result=_result(np.zeros(2), 0.0),
        penalty=np.zeros((2, 2)),
    )
    assert factor_calls == 1
    assert geometry.hessian_rank == 3
    _assert_observed_cutoff_geometry(
        geometry.hessian_inverse,
        geometry.log_det_H - float(np.log(len(rows))),
        factor_certificates[0],
        factor,
    )


def test_observed_well_conditioned_inverse_rejects_scale_mutations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A forced factor branch has resolvable covariance and fitted action."""
    import superglm.reml.observed_geometry as observed_geometry

    rows = np.arange(8)
    primary = np.where(rows % 2, 1.0, -1.0)
    orthogonal = np.where(rows % 4 < 2, 1.0, -1.0)
    X = np.column_stack((1.5 * primary, 0.75 * orthogonal))
    dm = DesignMatrix([DenseGroupMatrix(X)], n=len(rows), p=2)
    weights = np.ones(len(rows))
    eps = SHARED_RANK_POLICY.gram_rcond
    injected = decompose_gram(np.array([[1.0, 1.0 - 8.0 * eps], [1.0 - 8.0 * eps, 1.0]]))
    assert injected.rank == injected.width
    assert needs_factor_certification(injected)
    original_gram = observed_geometry.decompose_gram
    original_factor = observed_geometry.decompose_factor
    factor_calls = 0

    def injected_gram(matrix, *args, **kwargs):
        if np.asarray(matrix).shape == (2, 2) and np.any(matrix):
            return injected
        return original_gram(matrix, *args, **kwargs)

    def counted_factor(factor, *args, **kwargs):
        nonlocal factor_calls
        factor_calls += 1
        return original_factor(factor, *args, **kwargs)

    monkeypatch.setattr(observed_geometry, "decompose_gram", injected_gram)
    monkeypatch.setattr(observed_geometry, "decompose_factor", counted_factor)
    geometry = observed_geometry.build_observed_reml_geometry(
        dm=dm,
        distribution=Gamma(),
        link=LogLink(),
        y=np.ones(len(rows)),
        sample_weight=weights,
        offset_arr=np.zeros(len(rows)),
        result=_result(np.zeros(2), 0.0),
        penalty=np.zeros((2, 2)),
    )

    assert factor_calls == 1
    assert geometry.hessian_rank == 3
    metrics = _assert_well_conditioned_observed_inverse(geometry.hessian_inverse, X)
    assert metrics.conditioned_beta < 1.0
    for scale in (0.5, 2.0):
        with pytest.raises(AssertionError):
            _assert_well_conditioned_observed_inverse(
                scale * geometry.hessian_inverse,
                X,
            )


@pytest.mark.parametrize("translation", [0.0, 1e10])
def test_gamma_observed_geometry_matches_full_augmented_hessian(translation: float) -> None:
    """The profiled geometry is exact and stable under intercept-absorbed shifts."""
    x = np.linspace(-1.4, 1.2, 17)
    X_base = np.column_stack((x, np.sin(1.7 * x)))
    shift = np.array([translation, -0.75 * translation])
    X = X_base + shift
    dm = DesignMatrix([DenseGroupMatrix(X)], n=len(x), p=X.shape[1])
    beta = np.array([0.35, -0.2])
    base_intercept = 0.4
    intercept = base_intercept - float(shift @ beta)
    # Use the predictor represented by the shifted floating-point design.  At
    # 1e10 the stored columns themselves are quantized at roughly 1e-6; the
    # centering oracle below must avoid adding raw-moment cancellation on top
    # of that unavoidable input representation error.
    eta = intercept + X @ beta
    mu = np.exp(eta)
    y = mu * np.linspace(0.45, 1.8, len(mu))
    sample_weight = np.linspace(0.3, 2.1, len(mu))
    penalty = np.array([[1.2, 0.15], [0.15, 0.8]])

    geometry = build_observed_reml_geometry(
        dm=dm,
        distribution=Gamma(),
        link=LogLink(),
        y=y,
        sample_weight=sample_weight,
        offset_arr=np.zeros_like(y),
        result=_result(beta, intercept),
        penalty=penalty,
    )

    observed_w = sample_weight * y / mu
    anchored = X - X[0]
    centered = anchored - np.average(anchored, axis=0, weights=observed_w)
    expected_hessian = centered.T @ (observed_w[:, None] * centered) + penalty
    expected_sign, expected_slope_logdet = np.linalg.slogdet(expected_hessian)
    assert expected_sign == 1.0
    expected_logdet = np.log(np.sum(observed_w)) + expected_slope_logdet

    np.testing.assert_allclose(
        geometry.centered_hessian,
        expected_hessian,
        rtol=2e-7 if translation else 3e-14,
        atol=2e-6 if translation else 3e-14,
    )
    np.testing.assert_allclose(
        geometry.hessian_inverse,
        np.linalg.inv(expected_hessian),
        rtol=2e-7 if translation else 3e-14,
        atol=2e-7 if translation else 3e-14,
    )
    assert geometry.log_det_H == pytest.approx(
        expected_logdet,
        rel=2e-8 if translation else 3e-14,
        abs=2e-8 if translation else 3e-14,
    )
    assert geometry.hessian_rank == 1 + X.shape[1]


def test_gamma_observed_geometry_differs_materially_from_fisher_geometry() -> None:
    x = np.linspace(-1.0, 1.0, 12)
    X = np.column_stack((x, x**2))
    dm = DesignMatrix([DenseGroupMatrix(X)], n=len(x), p=2)
    beta = np.array([0.2, -0.1])
    intercept = 0.3
    mu = np.exp(intercept + X @ beta)
    y = mu * np.geomspace(0.2, 3.0, len(mu))
    sample_weight = np.ones_like(y)
    penalty = np.diag([0.7, 1.1])

    geometry = build_observed_reml_geometry(
        dm=dm,
        distribution=Gamma(),
        link=LogLink(),
        y=y,
        sample_weight=sample_weight,
        offset_arr=np.zeros_like(y),
        result=_result(beta, intercept),
        penalty=penalty,
    )

    fisher_center = X - np.mean(X, axis=0)
    fisher_hessian = fisher_center.T @ fisher_center + penalty
    assert np.linalg.norm(geometry.centered_hessian - fisher_hessian) > 1.0


def test_noncanonical_observed_rows_can_be_signed_and_match_likelihood_fd() -> None:
    """Wood's Newton rows are not generally positive for noncanonical links."""
    distribution = Binomial()
    link = CauchitLink()
    eta = np.arange(-3.0, 4.0)
    mu = link.inverse(eta)
    y = np.zeros_like(eta)
    sample_weight = np.ones_like(eta)

    observed = compute_observed_information_weights(
        distribution,
        link,
        y,
        mu,
        eta,
        sample_weight,
    )
    assert np.any(observed < 0.0)
    assert np.sum(observed) > 0.0

    eps = 2e-5
    nll = -np.log1p(-mu)
    nll_plus = -np.log1p(-link.inverse(eta + eps))
    nll_minus = -np.log1p(-link.inverse(eta - eps))
    fd_hessian = (nll_plus - 2.0 * nll + nll_minus) / eps**2
    np.testing.assert_allclose(observed, fd_hessian, rtol=3e-5, atol=2e-6)


def test_signed_observed_geometry_matches_augmented_hessian() -> None:
    distribution = Binomial()
    link = CauchitLink()
    X = np.arange(-3.0, 4.0)[:, None]
    dm = DesignMatrix([DenseGroupMatrix(X)], n=len(X), p=1)
    beta = np.array([1.0])
    intercept = 0.0
    y = np.zeros(len(X))
    sample_weight = np.ones(len(X))
    penalty = np.array([[100.0]])

    geometry = build_observed_reml_geometry(
        dm=dm,
        distribution=distribution,
        link=link,
        y=y,
        sample_weight=sample_weight,
        offset_arr=np.zeros_like(y),
        result=_result(beta, intercept),
        penalty=penalty,
    )

    observed_w = compute_observed_information_weights(
        distribution,
        link,
        y,
        link.inverse(X[:, 0]),
        X[:, 0],
        sample_weight,
    )
    assert np.any(observed_w < 0.0)
    np.testing.assert_array_equal(geometry.weights, observed_w)
    augmented = np.column_stack((np.ones(len(X)), X))
    expected = augmented.T @ (observed_w[:, None] * augmented)
    expected[1:, 1:] += penalty
    sign, logdet = np.linalg.slogdet(expected)
    assert sign == 1.0
    assert geometry.log_det_H == pytest.approx(logdet, rel=2e-13, abs=2e-13)
    expected_slope_inverse = np.linalg.inv(expected)[1:, 1:]
    np.testing.assert_allclose(
        geometry.hessian_inverse,
        expected_slope_inverse,
        rtol=2e-13,
        atol=2e-13,
    )


def test_observed_geometry_rejects_indefinite_penalty_and_total_curvature() -> None:
    X = np.arange(-3.0, 4.0)[:, None]
    dm = DesignMatrix([DenseGroupMatrix(X)], n=len(X), p=1)

    eta = X[:, 0]
    observed_w = compute_observed_information_weights(
        Binomial(),
        CauchitLink(),
        np.zeros(len(X)),
        CauchitLink().inverse(eta),
        eta,
        np.ones(len(X)),
    )
    assert np.sum(observed_w) > 0.0
    mean_x = np.sum(observed_w[:, None] * X, axis=0) / np.sum(observed_w)
    centered = X - mean_x
    terminal_hessian = centered.T @ (observed_w[:, None] * centered)
    assert np.linalg.eigvalsh(terminal_hessian)[0] < -1.0

    with pytest.raises(ValueError, match="negative|indefinite"):
        build_observed_reml_geometry(
            dm=dm,
            distribution=Gamma(),
            link=LogLink(),
            y=np.exp(X[:, 0]),
            sample_weight=np.ones(len(X)),
            offset_arr=np.zeros(len(X)),
            result=_result(np.array([1.0]), 0.0),
            penalty=np.array([[-100.0]]),
        )

    with pytest.raises(
        ValueError, match="terminal observed REML coefficient Hessian is indefinite"
    ):
        build_observed_reml_geometry(
            dm=dm,
            distribution=Binomial(),
            link=CauchitLink(),
            y=np.zeros(len(X)),
            sample_weight=np.ones(len(X)),
            offset_arr=np.zeros(len(X)),
            result=_result(np.array([1.0]), 0.0),
            penalty=np.zeros((1, 1)),
        )


def test_objective_only_geometry_skips_the_slope_inverse() -> None:
    X = np.linspace(-1.0, 1.0, 10)[:, None]
    dm = DesignMatrix([DenseGroupMatrix(X)], n=len(X), p=1)
    result = _result(np.array([0.2]), 0.1)
    mu = np.exp(result.intercept + X[:, 0] * result.beta[0])
    geometry = build_observed_reml_geometry(
        dm=dm,
        distribution=Gamma(),
        link=LogLink(),
        y=mu * np.linspace(0.6, 1.4, len(mu)),
        sample_weight=np.ones(len(mu)),
        offset_arr=np.zeros(len(mu)),
        result=result,
        penalty=np.array([[0.7]]),
        compute_inverse=False,
    )

    assert geometry.hessian_inverse is None
    assert np.isfinite(geometry.log_det_H)
    assert geometry.hessian_rank == 2


def test_observed_correction_does_not_disappear_for_tiny_weight_units() -> None:
    X = np.linspace(-1.0, 1.0, 20)[:, None]
    dm = DesignMatrix([DenseGroupMatrix(X)], n=len(X), p=1)
    result = _result(np.array([0.25]), 0.2)
    mu = np.exp(result.intercept + X[:, 0] * result.beta[0])
    y = mu * np.linspace(0.5, 1.8, len(mu))
    unit_scale = 1e-20
    sample_weight = np.full(len(mu), unit_scale)
    lam = 0.7 * unit_scale
    component = PenaltyComponent(
        name="x",
        group_name="x",
        group_index=0,
        group_sl=slice(0, 1),
        omega_raw=np.ones((1, 1)),
        omega_ssp=np.ones((1, 1)),
        rank=1.0,
    )
    geometry = build_observed_reml_geometry(
        dm=dm,
        distribution=Gamma(),
        link=LogLink(),
        y=y,
        sample_weight=sample_weight,
        offset_arr=np.zeros(len(mu)),
        result=result,
        penalty=np.array([[lam]]),
        derivative_order=1,
    )
    assert geometry.hessian_inverse is not None
    correction = reml_w_correction(
        dm,
        LogLink(),
        [GroupSlice(name="x", start=0, end=1)],
        result,
        geometry.hessian_inverse,
        {"x": lam},
        sample_weight=sample_weight,
        offset_arr=np.zeros(len(mu)),
        distribution=Gamma(),
        reml_penalties=[component],
        geometry=geometry,
    )

    assert correction is not None
    assert np.all(np.isfinite(correction[0]))
    assert abs(correction[0][0]) > 1e-6


def test_observed_mode_score_is_stable_under_large_feature_translation() -> None:
    x = np.linspace(-1.2, 1.3, 220)
    beta = np.array([0.35])
    base_intercept = 0.2
    penalty = np.zeros((1, 1))

    scores: list[float] = []
    for shift, unit_scale in ((0.0, 1.0), (1e10, 1.0), (1e10, 1e-20)):
        X = (x + shift)[:, None]
        dm = DesignMatrix(
            [DenseGroupMatrix(X)],
            n=len(x),
            p=1,
        )
        result = _result(beta, base_intercept - shift * beta[0])
        eta = result.intercept + X[:, 0] * beta[0]
        mu = np.exp(eta)
        weights = np.full(len(x), unit_scale)
        # Construct a positive response whose Gamma/log score is orthogonal to
        # both the intercept and represented slope.  This isolates the
        # translation and weight-unit behavior of the score evaluation itself
        # from coefficient-solver predictor representation.
        q = np.cos(np.linspace(0.0, 5.0 * np.pi, len(x)))
        anchored = X[:, 0] - X[0, 0]
        constraints = np.column_stack((np.ones(len(x)), anchored))
        q = q - constraints @ np.linalg.lstsq(constraints, q, rcond=None)[0]
        q *= 0.1 * unit_scale / max(float(np.max(np.abs(q))), 1e-300)
        y = mu * (1.0 + q / weights)
        geometry = build_observed_reml_geometry(
            dm=dm,
            distribution=Gamma(),
            link=LogLink(),
            y=y,
            sample_weight=weights,
            offset_arr=np.zeros(len(x)),
            result=result,
            penalty=penalty,
        )
        scores.append(
            observed_penalized_mode_score(
                dm=dm,
                distribution=Gamma(),
                link=LogLink(),
                y=y,
                sample_weight=weights,
                result=result,
                penalty=penalty,
                geometry=geometry,
            ).relative_max
        )

    assert max(scores) < 2e-12
    assert scores[1] == pytest.approx(scores[0], abs=2e-12)
    assert scores[2] == pytest.approx(scores[1], abs=2e-12)


def test_an_overflowing_mode_score_is_scored_not_raised() -> None:
    """A score that overflows describes these coefficients, not a malformed call.

    Nothing the build certifies bounds the score's quotient, and the two
    clippers that would have to bound it both decline by design: ``stabilize_eta``
    holds a sqrt eta only back from where squaring overflows, at about 6.7e153,
    and ``clip_mu`` returns Gaussian means uncapped. So a Gaussian/sqrt row at
    eta 1e150 carries observed curvature ``w * (6 eta^2 - 2 y)`` -- 6e300, which
    is finite -- while ``w * (y - mu) * dmu/deta`` is 1e300 times 2e150, which is
    not. The geometry is built and asserted on first, so what follows is a
    statement about the score alone rather than about an iterate the build would
    have refused anyway.
    """
    from superglm.reml.observed_geometry import (
        ObservedGeometryInfeasibleError,
        classify_reml_curvature,
    )

    # Gaussian's canonical link is the identity, so sqrt takes the observed
    # branch -- the only branch that evaluates this score at all.
    assert classify_reml_curvature(Gaussian(), SqrtLink()) == "observed"

    n = 6
    X = np.zeros((n, 1))
    X[0, 0] = 1.0
    dm = DesignMatrix([DenseGroupMatrix(X)], n=n, p=1)
    result = _result(np.array([1e150]), 1.0)
    y = np.linspace(1.0, 2.0, n)
    weights = np.ones(n)
    penalty = np.zeros((1, 1))
    geometry = build_observed_reml_geometry(
        dm=dm,
        distribution=Gaussian(),
        link=SqrtLink(),
        y=y,
        sample_weight=weights,
        offset_arr=np.zeros(n),
        result=result,
        penalty=penalty,
    )

    # Every guard the build owns passes on this iterate: the observed rows are
    # finite, their sum is positive, and the determinant it hands on is real.
    assert geometry.eta[0] == 1e150, "stabilize_eta clipped the iterate this test needs"
    assert geometry.mu[0] == geometry.eta[0] ** 2, "clip_mu capped a Gaussian mean"
    assert geometry.mu[0] > 1e299
    assert np.all(np.isfinite(geometry.mu))
    assert np.isfinite(geometry.sum_w) and geometry.sum_w > 0.0
    assert np.isfinite(geometry.log_det_H)

    with pytest.raises(ObservedGeometryInfeasibleError) as excinfo:
        observed_penalized_mode_score(
            dm=dm,
            distribution=Gaussian(),
            link=SqrtLink(),
            y=y,
            sample_weight=weights,
            result=result,
            penalty=penalty,
            geometry=geometry,
        )

    assert "penalized mode score is not finite" in str(excinfo.value)


def test_a_contract_bug_in_the_mode_score_stays_a_bare_value_error() -> None:
    """The other half of the retype, and a guard rather than a fix.

    A penalty in the wrong coordinates is a bug in the call at every iterate, so
    it must not join the retype above and invite a power search to route quietly
    around it. This cannot fail against the unfixed code -- both sides raise a
    bare ValueError here -- but it is what stops a later widening of the retype
    from swallowing the argument checks with it.
    """
    from superglm.reml.observed_geometry import ObservedGeometryInfeasibleError

    n = 6
    X = np.linspace(-1.0, 1.0, n)[:, None]
    dm = DesignMatrix([DenseGroupMatrix(X)], n=n, p=1)
    result = _result(np.array([0.2]), 0.1)
    y = np.linspace(1.0, 2.0, n)
    weights = np.ones(n)
    geometry = build_observed_reml_geometry(
        dm=dm,
        distribution=Gamma(),
        link=LogLink(),
        y=y,
        sample_weight=weights,
        offset_arr=np.zeros(n),
        result=result,
        penalty=np.zeros((1, 1)),
    )

    with pytest.raises(ValueError) as excinfo:
        observed_penalized_mode_score(
            dm=dm,
            distribution=Gamma(),
            link=LogLink(),
            y=y,
            sample_weight=weights,
            result=result,
            penalty=np.zeros((2, 2)),
            geometry=geometry,
        )

    assert not isinstance(excinfo.value, ObservedGeometryInfeasibleError)
    assert "slope coordinates" in str(excinfo.value)


def test_gamma_observed_total_gradient_matches_refitted_laml_finite_difference() -> None:
    """Direct REML must use observed inverse, determinant, rank, and dW rows together."""
    rng = np.random.default_rng(20260718)
    n = 350
    x = np.linspace(-1.3, 1.4, n)
    X = np.column_stack((x, np.sin(1.8 * x)))
    dm = DesignMatrix([DenseGroupMatrix(X)], n=n, p=2)
    group = GroupSlice(name="smooth", start=0, end=2)
    omega = np.array([[1.0, 0.15], [0.15, 1.7]])
    component = PenaltyComponent(
        name="smooth",
        group_name="smooth",
        group_index=0,
        group_sl=slice(0, 2),
        omega_raw=omega,
        omega_ssp=omega,
        rank=2.0,
    )
    family = Gamma()
    link = LogLink()
    sample_weight = rng.uniform(0.4, 1.8, n)
    offset = np.zeros(n)
    true_mu = np.exp(0.35 + 0.55 * x - 0.3 * np.sin(1.8 * x))
    y = rng.gamma(shape=3.5, scale=true_mu / 3.5)
    scale_data = prepare_gamma_reml_scale_data(y, sample_weight)

    def evaluate(log_lambda: float, *, need_gradient: bool):
        lam = float(np.exp(log_lambda))
        lambdas = {"smooth": lam}
        penalty = lam * omega
        result, fisher_inverse, fisher_gram = fit_irls_direct(
            X=dm,
            y=y,
            weights=sample_weight,
            family=family,
            link=link,
            groups=[group],
            lambda2=lambdas,
            offset=offset,
            return_xtwx=True,
            S_override=penalty,
            reml_penalties=[component],
            tol=1e-10,
        )
        geometry = build_observed_reml_geometry(
            dm=dm,
            distribution=family,
            link=link,
            y=y,
            sample_weight=sample_weight,
            offset_arr=offset,
            result=result,
            penalty=penalty,
            compute_inverse=need_gradient,
            derivative_order=1 if need_gradient else 0,
        )
        objective = reml_laml_objective(
            dm,
            family,
            link,
            [group],
            y,
            result,
            lambdas,
            sample_weight,
            offset,
            XtWX=fisher_gram,
            log_det_H=geometry.log_det_H,
            hessian_rank=geometry.hessian_rank,
            S_override=penalty,
            reml_penalties=[component],
            gamma_scale_data=scale_data,
            return_evaluation=True,
        )
        assert isinstance(objective, REMLObjectiveEvaluation)
        if not need_gradient:
            return objective.value

        assert geometry.hessian_inverse is not None
        assert objective.profiled_scale is not None
        partial = reml_direct_gradient(
            dm.group_matrices,
            result,
            geometry.hessian_inverse,
            lambdas,
            reml_penalties=[component],
            inverse_phi=objective.profiled_scale.inverse_phi,
        )
        correction = reml_w_correction(
            dm,
            link,
            [group],
            result,
            # Deliberately pass the incompatible Fisher inverse: geometry mode
            # must bind its own inverse atomically and make this hybrid inert.
            fisher_inverse,
            lambdas,
            sample_weight=sample_weight,
            offset_arr=offset,
            distribution=family,
            reml_penalties=[component],
            geometry=geometry,
        )
        assert correction is not None
        return (
            objective.value,
            partial + correction[0],
            partial,
            fisher_inverse,
            geometry.hessian_inverse,
        )

    rho = np.log(1.4)
    value, total_gradient, partial_gradient, fisher_inverse, observed_inverse = evaluate(
        rho,
        need_gradient=True,
    )
    assert np.isfinite(value)
    eps = 2e-4
    finite_difference = (
        evaluate(rho + eps, need_gradient=False) - evaluate(rho - eps, need_gradient=False)
    ) / (2.0 * eps)
    np.testing.assert_allclose(total_gradient[0], finite_difference, rtol=2e-4, atol=2e-5)
    assert abs(partial_gradient[0] - finite_difference) > 5e-4
    assert np.linalg.norm(fisher_inverse - observed_inverse) > 1e-3


def test_gamma_observed_order2_hessian_matches_refitted_gradient_finite_difference() -> None:
    """Order-2 includes observed rows, centered means, and profiled Gamma scale."""
    rng = np.random.default_rng(20260719)
    n = 260
    x = np.linspace(-1.2, 1.3, n)
    X = np.column_stack((x, np.sin(1.6 * x)))
    dm = DesignMatrix([DenseGroupMatrix(X)], n=n, p=2)
    group = GroupSlice(name="smooth", start=0, end=2)
    omega = np.array([[1.0, 0.12], [0.12, 1.6]])
    component = PenaltyComponent(
        name="smooth",
        group_name="smooth",
        group_index=0,
        group_sl=slice(0, 2),
        omega_raw=omega,
        omega_ssp=omega,
        rank=2.0,
    )
    family = Gamma()
    link = LogLink()
    sample_weight = rng.uniform(0.5, 1.6, n)
    offset = np.linspace(-0.15, 0.1, n)
    true_mu = np.exp(0.25 + 0.5 * x - 0.25 * np.sin(1.6 * x) + offset)
    y = rng.gamma(shape=4.0, scale=true_mu / 4.0)
    scale_data = prepare_gamma_reml_scale_data(y, sample_weight)

    def evaluate(log_lambda: float, *, derivative_order: int):
        lam = float(np.exp(log_lambda))
        lambdas = {"smooth": lam}
        penalty = lam * omega
        result, _fisher_inverse, fisher_gram = fit_irls_direct(
            X=dm,
            y=y,
            weights=sample_weight,
            family=family,
            link=link,
            groups=[group],
            lambda2=lambdas,
            offset=offset,
            return_xtwx=True,
            S_override=penalty,
            reml_penalties=[component],
            tol=1e-12,
            convergence="coefficients",
        )
        geometry = build_observed_reml_geometry(
            dm=dm,
            distribution=family,
            link=link,
            y=y,
            sample_weight=sample_weight,
            offset_arr=offset,
            result=result,
            penalty=penalty,
            derivative_order=derivative_order,
        )
        objective = reml_laml_objective(
            dm,
            family,
            link,
            [group],
            y,
            result,
            lambdas,
            sample_weight,
            offset,
            XtWX=fisher_gram,
            log_det_H=geometry.log_det_H,
            hessian_rank=geometry.hessian_rank,
            S_override=penalty,
            reml_penalties=[component],
            gamma_scale_data=scale_data,
            return_evaluation=True,
        )
        assert isinstance(objective, REMLObjectiveEvaluation)
        assert objective.profiled_scale is not None
        assert geometry.hessian_inverse is not None
        partial = reml_direct_gradient(
            dm.group_matrices,
            result,
            geometry.hessian_inverse,
            lambdas,
            reml_penalties=[component],
            inverse_phi=objective.profiled_scale.inverse_phi,
        )
        correction = reml_w_correction(
            dm,
            link,
            [group],
            result,
            geometry.hessian_inverse,
            lambdas,
            sample_weight=sample_weight,
            offset_arr=offset,
            distribution=family,
            w_correction_order=derivative_order,
            reml_penalties=[component],
            geometry=geometry,
        )
        assert correction is not None
        gradient = partial + correction[0]
        if derivative_order == 1:
            return gradient, None

        assert len(correction) == 3
        hessian = reml_direct_hessian(
            dm.group_matrices,
            family,
            geometry.hessian_inverse,
            lambdas,
            gradient=partial,
            pirls_result=result,
            n_obs=n,
            inverse_phi=objective.profiled_scale.inverse_phi,
            d_inverse_phi_d_penalized_deviance=(
                objective.profiled_scale.d_inverse_phi_d_penalized_deviance
            ),
            penalty_nullity=objective.penalty_nullity,
            dH_extra=correction[1],
            dH2_cross=correction[2],
            reml_penalties=[component],
        )
        return gradient, hessian

    rho = float(np.log(1.3))
    _gradient, analytic_hessian = evaluate(rho, derivative_order=2)
    assert analytic_hessian is not None
    eps = 2e-4
    gradient_plus, _ = evaluate(rho + eps, derivative_order=1)
    gradient_minus, _ = evaluate(rho - eps, derivative_order=1)
    finite_difference = (gradient_plus[0] - gradient_minus[0]) / (2.0 * eps)

    np.testing.assert_allclose(
        analytic_hessian[0, 0],
        finite_difference,
        rtol=2e-7,
        atol=2e-8,
    )


@dataclass(frozen=True)
class _ObservedLAMLProblem:
    dm: DesignMatrix
    distribution: object
    link: object
    group: GroupSlice
    component: PenaltyComponent
    omega: np.ndarray
    y: np.ndarray
    sample_weight: np.ndarray
    offset: np.ndarray


def _make_observed_laml_problem(case: str) -> _ObservedLAMLProblem:
    from scipy.special import ndtr

    case_index = {
        "nb2-log": 1,
        "tweedie-log": 2,
        "binomial-probit": 3,
        "binomial-cloglog": 4,
        "binomial-cauchit": 5,
        "poisson-sqrt": 6,
        "poisson-identity": 7,
        "gamma-identity": 8,
        "gamma-log": 9,
        "gaussian-log": 10,
    }[case]
    rng = np.random.default_rng(20260720 + case_index)
    n = 320 if case.startswith("binomial") else 240
    x = np.linspace(-0.9, 0.9, n)
    harmonic = np.sin(1.7 * x)
    X = np.column_stack((x, harmonic))
    dm = DesignMatrix([DenseGroupMatrix(X)], n=n, p=2)
    group = GroupSlice(name="smooth", start=0, end=2)
    omega = np.array([[1.0, 0.11], [0.11, 1.55]])
    component = PenaltyComponent(
        name="smooth",
        group_name="smooth",
        group_index=0,
        group_sl=slice(0, 2),
        omega_raw=omega,
        omega_ssp=omega,
        rank=2.0,
    )
    sample_weight = rng.uniform(0.7, 1.4, n)
    offset = np.linspace(-0.08, 0.06, n)

    if case == "nb2-log":
        distribution = NegativeBinomial(theta=2.7)
        link = LogLink()
        mu = np.exp(0.35 + 0.35 * x - 0.22 * harmonic + offset)
        y = rng.negative_binomial(2.7, 2.7 / (2.7 + mu)).astype(float)
    elif case == "tweedie-log":
        distribution = Tweedie(p=1.5)
        link = LogLink()
        mu = np.exp(0.25 + 0.3 * x - 0.18 * harmonic + offset)
        phi = 0.55
        count_mean = mu**0.5 / (phi * 0.5)
        counts = rng.poisson(count_mean)
        y = np.zeros(n)
        positive = counts > 0
        y[positive] = rng.gamma(
            shape=counts[positive],
            scale=phi * 0.5 * mu[positive] ** 0.5,
        )
    elif case.startswith("binomial"):
        distribution = Binomial()
        if case == "binomial-probit":
            link = ProbitLink()
            linear_predictor = 0.12 + 0.48 * x - 0.24 * harmonic + offset
            probability = ndtr(linear_predictor)
        elif case == "binomial-cloglog":
            link = CloglogLink()
            linear_predictor = 0.12 + 0.48 * x - 0.24 * harmonic + offset
            probability = 1.0 - np.exp(-np.exp(linear_predictor))
        else:
            link = CauchitLink()
            # A wider fitted predictor plus contrary tail outcomes exercises
            # Wood's valid signed Newton rows while the penalized total
            # curvature remains positive definite.
            linear_predictor = 0.1 + 1.6 * x - 0.3 * harmonic + offset
            probability = 0.5 + np.arctan(linear_predictor) / np.pi
        y = rng.binomial(1, probability).astype(float)
    elif case == "poisson-sqrt":
        distribution = Poisson()
        link = SqrtLink()
        eta = 1.35 + 0.16 * x - 0.1 * harmonic + offset
        y = rng.poisson(eta**2).astype(float)
    elif case == "poisson-identity":
        distribution = Poisson()
        link = IdentityLink()
        mu = 2.1 + 0.34 * x - 0.18 * harmonic + offset
        y = rng.poisson(mu).astype(float)
    elif case == "gamma-identity":
        distribution = Gamma()
        link = IdentityLink()
        mu = 1.5 + 0.28 * x - 0.14 * harmonic + offset
        y = rng.gamma(shape=5.0, scale=mu / 5.0)
    elif case == "gamma-log":
        distribution = Gamma()
        link = LogLink()
        mu = np.exp(0.3 + 0.3 * x - 0.16 * harmonic + offset)
        y = rng.gamma(shape=5.0, scale=mu / 5.0)
    else:
        distribution = Gaussian()
        link = LogLink()
        mu = np.exp(0.4 + 0.22 * x - 0.12 * harmonic + offset)
        y = mu + rng.normal(scale=0.16, size=n)

    return _ObservedLAMLProblem(
        dm=dm,
        distribution=distribution,
        link=link,
        group=group,
        component=component,
        omega=omega,
        y=y,
        sample_weight=sample_weight,
        offset=offset,
    )


def _evaluate_refitted_observed_laml(
    problem: _ObservedLAMLProblem,
    log_lambda: float,
    *,
    derivative_order: int,
) -> tuple[float, np.ndarray | None, np.ndarray | None]:
    lam = float(np.exp(log_lambda))
    lambdas = {problem.component.name: lam}
    penalty = lam * problem.omega
    result, _fisher_inverse, fisher_gram = fit_irls_direct(
        X=problem.dm,
        y=problem.y,
        weights=problem.sample_weight,
        family=problem.distribution,
        link=problem.link,
        groups=[problem.group],
        lambda2=lambdas,
        offset=problem.offset,
        return_xtwx=True,
        S_override=penalty,
        reml_penalties=[problem.component],
        tol=1e-11,
        max_iter=200,
        convergence="coefficients",
    )
    assert result.converged
    geometry = build_observed_reml_geometry(
        dm=problem.dm,
        distribution=problem.distribution,
        link=problem.link,
        y=problem.y,
        sample_weight=problem.sample_weight,
        offset_arr=problem.offset,
        result=result,
        penalty=penalty,
        compute_inverse=derivative_order >= 1,
        derivative_order=derivative_order,
    )
    gamma_scale_data = (
        prepare_gamma_reml_scale_data(problem.y, problem.sample_weight)
        if type(problem.distribution) is Gamma
        else None
    )
    objective = reml_laml_objective(
        problem.dm,
        problem.distribution,
        problem.link,
        [problem.group],
        problem.y,
        result,
        lambdas,
        problem.sample_weight,
        problem.offset,
        XtWX=fisher_gram,
        log_det_H=geometry.log_det_H,
        hessian_rank=geometry.hessian_rank,
        S_override=penalty,
        reml_penalties=[problem.component],
        gamma_scale_data=gamma_scale_data,
        return_evaluation=True,
    )
    assert isinstance(objective, REMLObjectiveEvaluation)
    if derivative_order == 0:
        return objective.value, None, None

    assert geometry.hessian_inverse is not None
    if objective.profiled_scale is not None:
        inverse_phi = objective.profiled_scale.inverse_phi
        inverse_phi_derivative = objective.profiled_scale.d_inverse_phi_d_penalized_deviance
    elif not getattr(problem.distribution, "scale_known", True):
        assert objective.penalty_nullity is not None
        inverse_phi = max(len(problem.y) - objective.penalty_nullity, 1.0) / max(
            objective.penalized_deviance,
            1e-300,
        )
        inverse_phi_derivative = None
    else:
        inverse_phi = 1.0
        inverse_phi_derivative = None
    partial = reml_direct_gradient(
        problem.dm.group_matrices,
        result,
        geometry.hessian_inverse,
        lambdas,
        reml_penalties=[problem.component],
        inverse_phi=inverse_phi,
    )
    correction = reml_w_correction(
        problem.dm,
        problem.link,
        [problem.group],
        result,
        geometry.hessian_inverse,
        lambdas,
        sample_weight=problem.sample_weight,
        offset_arr=problem.offset,
        distribution=problem.distribution,
        w_correction_order=derivative_order,
        reml_penalties=[problem.component],
        geometry=geometry,
    )
    assert correction is not None
    gradient = partial + correction[0]
    if derivative_order == 1:
        return objective.value, gradient, None

    assert len(correction) == 3
    hessian = reml_direct_hessian(
        problem.dm.group_matrices,
        problem.distribution,
        geometry.hessian_inverse,
        lambdas,
        gradient=partial,
        pirls_result=result,
        n_obs=len(problem.y),
        inverse_phi=inverse_phi,
        d_inverse_phi_d_penalized_deviance=inverse_phi_derivative,
        penalty_nullity=(
            objective.penalty_nullity
            if not getattr(problem.distribution, "scale_known", True)
            else None
        ),
        dH_extra=correction[1],
        dH2_cross=correction[2],
        reml_penalties=[problem.component],
    )
    return objective.value, gradient, hessian


@pytest.mark.parametrize(
    "case",
    [
        "nb2-log",
        "tweedie-log",
        "binomial-probit",
        "binomial-cloglog",
        "binomial-cauchit",
        "poisson-sqrt",
        "poisson-identity",
        "gamma-identity",
        "gamma-log",
        "gaussian-log",
    ],
)
def test_refitted_noncanonical_laml_gradient_matches_objective_finite_difference(
    case: str,
) -> None:
    problem = _make_observed_laml_problem(case)
    rho = float(np.log(1.25))
    _value, gradient, _hessian = _evaluate_refitted_observed_laml(
        problem,
        rho,
        derivative_order=1,
    )
    assert gradient is not None
    eps = 2e-4
    value_plus, _, _ = _evaluate_refitted_observed_laml(
        problem,
        rho + eps,
        derivative_order=0,
    )
    value_minus, _, _ = _evaluate_refitted_observed_laml(
        problem,
        rho - eps,
        derivative_order=0,
    )
    finite_difference = (value_plus - value_minus) / (2.0 * eps)
    np.testing.assert_allclose(gradient[0], finite_difference, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("case", ["nb2-log", "tweedie-log", "binomial-cauchit"])
def test_refitted_noncanonical_order2_hessian_matches_gradient_finite_difference(
    case: str,
) -> None:
    problem = _make_observed_laml_problem(case)
    rho = float(np.log(1.25))
    _value, _gradient, hessian = _evaluate_refitted_observed_laml(
        problem,
        rho,
        derivative_order=2,
    )
    assert hessian is not None
    eps = 2e-4
    _, gradient_plus, _ = _evaluate_refitted_observed_laml(
        problem,
        rho + eps,
        derivative_order=1,
    )
    _, gradient_minus, _ = _evaluate_refitted_observed_laml(
        problem,
        rho - eps,
        derivative_order=1,
    )
    assert gradient_plus is not None and gradient_minus is not None
    finite_difference = (gradient_plus[0] - gradient_minus[0]) / (2.0 * eps)
    np.testing.assert_allclose(hessian[0, 0], finite_difference, rtol=5e-6, atol=2e-7)


def test_fit_reml_gamma_uses_observed_main_and_objective_only_trial_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pandas as pd

    import superglm.reml.direct as direct
    from superglm import SuperGLM
    from superglm.features.spline import Spline

    rng = np.random.default_rng(99)
    n = 400
    x = rng.uniform(1.0, 10.0, n)
    mu = np.exp(0.3 + 0.1 * np.sin(x))
    y = rng.gamma(shape=5.0, scale=mu / 5.0)
    calls: list[tuple[bool, int]] = []
    original = direct.build_observed_reml_geometry

    def observed_geometry_spy(**kwargs):
        calls.append(
            (
                bool(kwargs.get("compute_inverse", True)),
                int(kwargs.get("derivative_order", 0)),
            )
        )
        return original(**kwargs)

    monkeypatch.setattr(direct, "build_observed_reml_geometry", observed_geometry_spy)
    model = SuperGLM(
        family="gamma",
        selection_penalty=0,
        features={"x": Spline(n_knots=6, penalty="ssp")},
    )
    model.fit_reml(pd.DataFrame({"x": x}), y)

    assert model._reml_result.converged
    assert any(
        compute_inverse and derivative_order == 1 for compute_inverse, derivative_order in calls
    )
    assert any(
        not compute_inverse and derivative_order == 0 for compute_inverse, derivative_order in calls
    )
    assert model._reml_profile["reml_observed_mode_residual_accepted_max"] < 1e-9
    assert model._reml_profile["reml_observed_mode_rejected_trial_count"] >= 0
    assert model._reml_profile["reml_observed_mode_residual_rejected_trial_max"] >= 0.0
    assert model._reml_profile["reml_w_correction_order"] == 1
    assert model._reml_profile["reml_observed_geometry_s"] > 0.0


def test_fit_reml_terminal_observed_state_owns_objective_rank_and_scale() -> None:
    """The published final refit must own its observed LAML evaluation."""
    import pandas as pd

    from superglm import SuperGLM
    from superglm.features.spline import Spline
    from superglm.reml.penalty_algebra import build_penalty_matrix

    rng = np.random.default_rng(991)
    n = 240
    x = rng.uniform(0.0, 6.0, n)
    mu = np.exp(0.25 + 0.18 * np.sin(1.3 * x))
    y = rng.gamma(shape=4.0, scale=mu / 4.0)
    sample_weight = np.resize(np.array([1.0, 2.0, 3.0]), n)
    X = pd.DataFrame({"x": x})
    model = SuperGLM(
        family="gamma",
        selection_penalty=0,
        features={"x": Spline(n_knots=6, penalty="ssp")},
    )

    model.fit_reml(X, y, sample_weight=sample_weight)

    result = model._solver_result
    penalty = build_penalty_matrix(
        model._dm.group_matrices,
        model._groups,
        model._reml_lambdas,
        model._dm.p,
        reml_penalties=model._reml_penalties,
    )
    geometry = build_observed_reml_geometry(
        dm=model._dm,
        distribution=model._distribution,
        link=model._link,
        y=y,
        sample_weight=sample_weight,
        offset_arr=np.zeros(n),
        result=result,
        penalty=penalty,
        derivative_order=0,
        compute_inverse=False,
    )
    mode_score = observed_penalized_mode_score(
        dm=model._dm,
        distribution=model._distribution,
        link=model._link,
        y=y,
        sample_weight=sample_weight,
        result=result,
        penalty=penalty,
        geometry=geometry,
    )
    evaluation = reml_laml_objective(
        model._dm,
        model._distribution,
        model._link,
        model._groups,
        y,
        result,
        model._reml_lambdas,
        sample_weight,
        np.zeros(n),
        log_det_H=geometry.log_det_H,
        hessian_rank=geometry.hessian_rank,
        S_override=penalty,
        reml_penalties=model._reml_penalties,
        return_evaluation=True,
    )

    assert isinstance(evaluation, REMLObjectiveEvaluation)
    assert mode_score.relative_max < 1e-9
    assert model._reml_result.curvature_source == "observed"
    assert result.reml_hessian_rank == geometry.hessian_rank
    assert result.log_det_H == geometry.log_det_H
    assert model._reml_result.objective == evaluation.value
    assert evaluation.profiled_scale is not None
    assert result.phi == evaluation.profiled_scale.phi


@pytest.mark.parametrize("discrete", [False, True])
def test_fit_reml_terminal_fisher_state_owns_objective(discrete: bool) -> None:
    """The published Fisher refit must own the reported LAML objective."""
    import pandas as pd

    from superglm import SuperGLM
    from superglm.features.spline import Spline
    from superglm.reml.penalty_algebra import build_penalty_matrix

    rng = np.random.default_rng(7302)
    n = 300
    x = np.sort(rng.uniform(-2.0, 2.0, n))
    y = rng.poisson(np.exp(-0.4 + 3.0 * np.sin(2.3 * x))).astype(float)
    sample_weight = np.ones(n)
    model = SuperGLM(
        family="poisson",
        selection_penalty=0,
        discrete=discrete,
        features={"x": Spline(n_knots=12, penalty="ssp")},
    )

    model.fit_reml(
        pd.DataFrame({"x": x}),
        y,
        sample_weight=sample_weight,
        max_reml_iter=10,
        reml_tol=1e-2,
        max_pirls_iter=100,
        pirls_tol=1e-3,
    )

    result = model._solver_result
    penalty = build_penalty_matrix(
        model._dm.group_matrices,
        model._groups,
        model._reml_lambdas,
        model._dm.p,
        reml_penalties=model._reml_penalties,
    )
    evaluation = reml_laml_objective(
        model._dm,
        model._distribution,
        model._link,
        model._groups,
        y,
        result,
        model._reml_lambdas,
        sample_weight,
        np.zeros(n),
        log_det_H=result.log_det_H,
        hessian_rank=result.reml_hessian_rank,
        S_override=penalty,
        reml_penalties=model._reml_penalties,
        return_evaluation=True,
    )

    assert isinstance(evaluation, REMLObjectiveEvaluation)
    assert model._reml_result.converged
    assert result.converged
    assert model._reml_result.curvature_source == "fisher"
    assert model._reml_result.objective == pytest.approx(evaluation.value, rel=2e-12)


def test_fit_reml_terminal_discrete_tensor_keeps_closed_form_penalty_logdet(
    monkeypatch,
) -> None:
    """Terminal publication must not rematerialize a discrete tensor log-determinant."""
    import pandas as pd

    import superglm.reml.multi_penalty as multi_penalty
    from superglm import SuperGLM
    from superglm.features.spline import Spline

    dense_logdet = multi_penalty.similarity_transform_logdet
    dense_logdet_calls = 0

    def count_dense_logdet(*args, **kwargs):
        nonlocal dense_logdet_calls
        dense_logdet_calls += 1
        return dense_logdet(*args, **kwargs)

    monkeypatch.setattr(multi_penalty, "similarity_transform_logdet", count_dense_logdet)

    rng = np.random.default_rng(77)
    n = 260
    x1 = rng.uniform(0.0, 1.0, n)
    x2 = rng.uniform(0.0, 1.0, n)
    eta = 0.2 + np.sin(2.0 * np.pi * x1) + 0.3 * np.cos(2.0 * np.pi * x2)
    y = rng.poisson(np.exp(eta)).astype(float)
    model = SuperGLM(
        family="poisson",
        selection_penalty=0,
        discrete=True,
        features={"x1": Spline(n_knots=5), "x2": Spline(n_knots=5)},
        interactions=[("x1", "x2")],
    )

    model.fit_reml(
        pd.DataFrame({"x1": x1, "x2": x2}),
        y,
        max_reml_iter=3,
        reml_tol=1e-12,
    )

    assert dense_logdet_calls == 0


def test_qp_passthrough_terminal_state_owns_fisher_objective_and_scale() -> None:
    """A constrained terminal refit must replace unconstrained LAML provenance."""
    import pandas as pd

    from superglm import Constraint, SuperGLM
    from superglm.features.spline import BSplineSmooth
    from superglm.reml.penalty_algebra import build_penalty_matrix

    rng = np.random.default_rng(303)
    n = 160
    x = np.sort(rng.uniform(0.1, 1.0, n))
    mu = np.exp(0.2 + 0.7 * x)
    y = rng.gamma(shape=4.0, scale=mu / 4.0)
    frame = pd.DataFrame({"x": x})
    model = SuperGLM(
        family="gamma",
        selection_penalty=0,
        features={
            "x": BSplineSmooth(
                n_knots=7,
                constraint=Constraint.fit.increasing,
            )
        },
    )

    model.fit_reml(frame, y, max_reml_iter=5)

    result = model._solver_result
    penalty = build_penalty_matrix(
        model._dm.group_matrices,
        model._groups,
        model._reml_lambdas,
        model._dm.p,
        reml_penalties=model._reml_penalties,
    )
    evaluation = reml_laml_objective(
        model._dm,
        model._distribution,
        model._link,
        model._groups,
        y,
        result,
        model._reml_lambdas,
        np.ones(n),
        np.zeros(n),
        log_det_H=result.log_det_H,
        hessian_rank=result.reml_hessian_rank,
        S_override=penalty,
        reml_penalties=model._reml_penalties,
        return_evaluation=True,
    )

    assert model._last_fit_meta["lambda_strategy"] == "qp_passthrough"
    assert model._reml_result.curvature_source == "fisher"
    assert model._reml_result.objective == pytest.approx(evaluation.value, rel=2e-12)
    assert evaluation.profiled_scale is not None
    assert result.phi == pytest.approx(evaluation.profiled_scale.phi, rel=2e-12)


def test_qp_passthrough_tweedie_scale_uses_terminal_penalty_nullity() -> None:
    """Reduced Tweedie phi must use the same identified rank as terminal LAML."""
    import pandas as pd

    from superglm import Constraint, SuperGLM
    from superglm.features.spline import BSplineSmooth
    from superglm.reml.penalty_algebra import build_penalty_matrix

    rng = np.random.default_rng(914)
    n = 120
    x = np.sort(rng.uniform(0.1, 1.0, n))
    offset = 0.1 * np.sin(np.arange(n) / 9.0)
    mu = np.exp(0.2 + 0.6 * x + offset)
    y = np.maximum(mu * (1.0 + 0.08 * rng.normal(size=n)), 0.03)
    sample_weight = np.resize(np.array([1.0, 3.0, 2.0, 4.0]), n)
    frame = pd.DataFrame({"x": x})
    model = SuperGLM(
        family=Tweedie(p=1.5),
        selection_penalty=0,
        features={
            "x": BSplineSmooth(
                n_knots=7,
                constraint=Constraint.fit.increasing,
            )
        },
    )

    model.fit_reml(
        frame,
        y,
        sample_weight=sample_weight,
        offset=offset,
        max_reml_iter=5,
    )

    result = model._solver_result
    penalty = build_penalty_matrix(
        model._dm.group_matrices,
        model._groups,
        model._reml_lambdas,
        model._dm.p,
        reml_penalties=model._reml_penalties,
    )
    evaluation = reml_laml_objective(
        model._dm,
        model._distribution,
        model._link,
        model._groups,
        y,
        result,
        model._reml_lambdas,
        sample_weight,
        offset,
        log_det_H=result.log_det_H,
        hessian_rank=result.reml_hessian_rank,
        S_override=penalty,
        reml_penalties=model._reml_penalties,
        return_evaluation=True,
    )
    assert isinstance(evaluation, REMLObjectiveEvaluation)
    assert evaluation.profiled_scale is None
    assert evaluation.penalty_nullity is not None
    expected_phi = evaluation.penalized_deviance / (n - evaluation.penalty_nullity)

    assert model._last_fit_meta["lambda_strategy"] == "qp_passthrough"
    assert model._reml_result.curvature_source == "fisher"
    assert model._reml_result.objective == pytest.approx(evaluation.value, rel=2e-12)
    assert result.phi == pytest.approx(expected_phi, rel=2e-12)


def test_qp_passthrough_poisson_fast_objective_keeps_deviance_scale() -> None:
    """Retained-geometry fast evaluation must keep Poisson's deviance convention."""
    import pandas as pd

    from superglm import Constraint, SuperGLM
    from superglm.features.spline import BSplineSmooth
    from superglm.reml.penalty_algebra import build_penalty_matrix

    rng = np.random.default_rng(812)
    n = 100
    x = np.sort(rng.uniform(0.1, 1.0, n))
    offset = 0.12 * np.sin(np.arange(n) / 9.0)
    mu = np.exp(0.2 + 0.7 * x + offset)
    y = rng.poisson(mu).astype(float)
    sample_weight = np.resize(np.array([1.0, 2.0, 3.0]), n)
    frame = pd.DataFrame({"x": x})
    model = SuperGLM(
        family="poisson",
        selection_penalty=0,
        features={
            "x": BSplineSmooth(
                n_knots=7,
                constraint=Constraint.fit.increasing,
            )
        },
    )

    model.fit_reml(
        frame,
        y,
        sample_weight=sample_weight,
        offset=offset,
        max_reml_iter=4,
    )

    result = model._solver_result
    penalty = build_penalty_matrix(
        model._dm.group_matrices,
        model._groups,
        model._reml_lambdas,
        model._dm.p,
        reml_penalties=model._reml_penalties,
    )
    shared = dict(
        dm=model._dm,
        distribution=model._distribution,
        link=model._link,
        groups=model._groups,
        y=y,
        result=result,
        lambdas=model._reml_lambdas,
        sample_weight=sample_weight,
        offset_arr=offset,
        log_det_H=result.log_det_H,
        hessian_rank=result.reml_hessian_rank,
        reml_penalties=model._reml_penalties,
        return_evaluation=True,
    )
    fast = reml_laml_objective(**shared, S_override=penalty)
    full = reml_laml_objective(**shared)

    assert isinstance(fast, REMLObjectiveEvaluation)
    assert isinstance(full, REMLObjectiveEvaluation)
    assert fast.value == pytest.approx(full.value, rel=2e-12, abs=2e-12)
    assert model._reml_result.objective == pytest.approx(fast.value, rel=2e-12)


def test_observed_laml_backtracks_after_invalid_trial_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pandas as pd

    import superglm.reml.direct as direct
    from superglm import SuperGLM
    from superglm.features.spline import Spline
    from superglm.reml.observed_geometry import ObservedGeometryInfeasibleError

    rng = np.random.default_rng(99)
    n = 400
    x = rng.uniform(1.0, 10.0, n)
    mu = np.exp(0.3 + 0.1 * np.sin(x))
    y = rng.gamma(shape=5.0, scale=mu / 5.0)
    objective_only_calls = 0
    original = direct.build_observed_reml_geometry

    def reject_first_long_trial(**kwargs):
        nonlocal objective_only_calls
        if not kwargs.get("compute_inverse", True):
            objective_only_calls += 1
            if objective_only_calls == 1:
                # The refusal a shorter step can answer: this iterate carries no
                # usable observed geometry. A bare ValueError here would be a
                # bug in the call, which the trial gate deliberately no longer
                # backtracks on -- see the companion test below.
                raise ObservedGeometryInfeasibleError("synthetic invalid signed trial geometry")
        return original(**kwargs)

    monkeypatch.setattr(direct, "build_observed_reml_geometry", reject_first_long_trial)
    model = SuperGLM(
        family="gamma",
        selection_penalty=0,
        features={"x": Spline(n_knots=6, penalty="ssp")},
    )

    model.fit_reml(pd.DataFrame({"x": x}), y)

    assert model._reml_result.converged
    assert objective_only_calls >= 2
    assert model._reml_profile["reml_observed_mode_rejected_trial_count"] >= 1


def test_a_contract_bug_in_a_trial_is_not_answered_by_a_shorter_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Halving the step retries the same malformed call, and hides why it failed.

    The line-search half of the narrowing above. A caller-contract violation
    does not improve at half the step, so backtracking on one burns the trial
    budget and then reports the bug as a conditioning failure at this point.
    """
    import pandas as pd

    import superglm.reml.direct as direct
    from superglm import SuperGLM
    from superglm.features.spline import Spline

    rng = np.random.default_rng(99)
    n = 400
    x = rng.uniform(1.0, 10.0, n)
    mu = np.exp(0.3 + 0.1 * np.sin(x))
    y = rng.gamma(shape=5.0, scale=mu / 5.0)
    objective_only_calls = 0
    original = direct.build_observed_reml_geometry

    def break_the_first_long_trial(**kwargs):
        nonlocal objective_only_calls
        if not kwargs.get("compute_inverse", True):
            objective_only_calls += 1
            if objective_only_calls == 1:
                raise ValueError("derivative_order must be 0, 1, or 2")
        return original(**kwargs)

    monkeypatch.setattr(direct, "build_observed_reml_geometry", break_the_first_long_trial)
    model = SuperGLM(
        family="gamma",
        selection_penalty=0,
        features={"x": Spline(n_knots=6, penalty="ssp")},
    )

    with pytest.raises(ValueError) as excinfo:
        model.fit_reml(pd.DataFrame({"x": x}), y)

    assert objective_only_calls == 1, "the line search retried the malformed call"
    assert type(excinfo.value) is ValueError
    assert "derivative_order" in str(excinfo.value)


def test_a_trial_whose_mode_score_refuses_is_answered_by_a_shorter_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backtracking was already the right answer one call earlier in the same trial.

    The trial geometry built immediately above this score answers a refusal by
    halving the step and counting a rejection. The score reads that geometry,
    refuses for the same reason about the same iterate, and used to end the fit
    instead -- two consecutive statements about one trial, answered opposite
    ways. Only the fit surviving proves it: the counter alone would advance on
    a refusal the geometry raised.
    """
    import pandas as pd

    import superglm.reml.direct as direct
    from superglm import SuperGLM
    from superglm.features.spline import Spline
    from superglm.reml.observed_geometry import ObservedGeometryInfeasibleError

    rng = np.random.default_rng(99)
    n = 400
    x = rng.uniform(1.0, 10.0, n)
    mu = np.exp(0.3 + 0.1 * np.sin(x))
    y = rng.gamma(shape=5.0, scale=mu / 5.0)
    trial_scores = 0
    original = direct.observed_penalized_mode_score

    def refuse_the_first_trial_score(**kwargs):
        nonlocal trial_scores
        # Only the line-search trial asks for an objective-only geometry, so a
        # missing slope inverse is what separates a trial from the candidate
        # gate, which has no step left to halve and retypes instead.
        if kwargs["geometry"].hessian_inverse is None:
            trial_scores += 1
            if trial_scores == 1:
                raise ObservedGeometryInfeasibleError("penalized mode score is not finite")
        return original(**kwargs)

    monkeypatch.setattr(direct, "observed_penalized_mode_score", refuse_the_first_trial_score)
    model = SuperGLM(
        family="gamma",
        selection_penalty=0,
        features={"x": Spline(n_knots=6, penalty="ssp")},
    )

    model.fit_reml(pd.DataFrame({"x": x}), y)

    assert model._reml_result.converged
    assert trial_scores >= 2, "the line search never came back with a shorter step"
    assert model._reml_profile["reml_observed_mode_rejected_trial_count"] >= 1


def test_a_structured_factor_that_refuses_an_iterate_is_scored_not_raised() -> None:
    """The structured branch refuses during construction, and numpy speaks ValueError.

    ``numpy.linalg.LinAlgError`` subclasses ``ValueError``, so the sum-to-zero
    and Schur factors' refusal of a level with negative local curvature used to
    be absorbed by the blanket catch in ``optimize_direct_reml`` along with
    everything else. Narrowing that catch to ``ObservedGeometryInfeasibleError``
    would have let it escape ``fit_reml`` raw -- and the structured path has no
    other indefiniteness signal, because ``public_positive_definite`` is never
    set False.

    The iterate here is not exotic. Under a non-canonical link the observed
    weights carry a signed residual term, so a halving line search walks
    straight through scaled betas like this one on its way back to the mode.
    """
    import pandas as pd

    from superglm import FactorSmooth, LambdaPolicy, Numeric, Spline, SuperGLM
    from superglm.reml.observed_geometry import ObservedGeometryInfeasibleError

    rng = np.random.default_rng(2027)
    n, n_levels = 360, 6
    x = rng.uniform(-1.0, 1.0, size=n)
    z = rng.normal(size=n)
    codes = rng.integers(0, n_levels, size=n)
    deviation = rng.normal(scale=0.18, size=(n_levels, 3))
    deviation -= deviation.mean(axis=0)
    eta = (
        -0.25
        + 0.45 * np.sin(2.1 * x)
        + 0.17 * z
        + deviation[codes, 0]
        + deviation[codes, 1] * x
        + deviation[codes, 2] * x**2
    )
    y = rng.poisson(np.exp(eta)).astype(np.float64)
    weight = rng.uniform(0.5, 1.8, size=n)
    offset = rng.normal(scale=0.04, size=n)
    frame = pd.DataFrame({"x": x, "z": z, "group": [f"level-{c}" for c in codes]})

    model = SuperGLM(
        family="poisson",
        features={
            "x": Spline(n_knots=5, lambda_policy=LambdaPolicy.fixed(1.3)),
            "z": Numeric(),
        },
        interactions=[
            FactorSmooth(
                "x",
                group="group",
                basis="sz",
                k=6,
                lambda_policy={"wiggle": LambdaPolicy.fixed(1.7)},
            )
        ],
        selection_penalty=0.0,
        discrete=False,
        direct_solve="structured",
    ).fit_reml(
        frame,
        y,
        sample_weight=weight,
        offset=offset,
        max_reml_iter=2,
        pirls_tol=1e-10,
        runtime_validation="skip",
    )
    structured_index = next(
        i
        for i, matrix in enumerate(model._dm.group_matrices)
        if getattr(matrix, "factor_basis", None) == "sz"
    )

    def build_at(scale: float):
        return build_observed_reml_geometry(
            dm=model._dm,
            distribution=Gaussian(),
            link=LogLink(),
            y=y,
            sample_weight=weight,
            offset_arr=offset,
            result=replace(
                model.result,
                beta=model.result.beta * scale,
                intercept=model.result.intercept * scale,
            ),
            penalty=None,
            derivative_order=0,
            groups=model._groups,
            lambdas=model._reml_lambdas,
            reml_penalties=model._reml_penalties,
            structured_group_index=structured_index,
            compute_inverse=False,
        )

    # The mode itself still builds, so what follows is a statement about the
    # iterate rather than a geometry broken for every input.
    assert build_at(1.0) is not None

    with pytest.raises(ObservedGeometryInfeasibleError) as excinfo:
        build_at(0.2)

    # Retyped, not replaced: the factor's own diagnostic (which level, what
    # curvature) still hangs off the cause for anyone reading the traceback.
    assert isinstance(excinfo.value.__cause__, np.linalg.LinAlgError)


def test_a_structural_refusal_from_the_structured_factor_is_not_relabelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same call raises in two dialects, and only one of them is this iterate.

    The companion above pins the dialect that IS the iterate: a level whose local
    curvature the factor rejects, raised as ``LinAlgError``. This pins the other.
    ``LinAlgError`` subclasses ``ValueError``, so a catch written for the first
    also caught partitions that disagree, a cross block of the wrong shape, level
    labels that do not match K -- and reported a misassembled operator as "the
    fitted coefficients do not define a valid Laplace mode", burying an assembly
    bug inside a diagnostic about the data.

    Both arms inject, and both inject a diagnostic the callee really raises,
    because every public route into that call is already gated: the layout
    builder rejects group slices that do not cover or do not match their matrix
    widths, and the penalized operator re-checks its own symmetry and shapes
    before it is handed over. A misassembled operator is reachable only from
    inside the assembly, which is exactly why the refusal must not be laundered
    when it happens. Injecting both arms through one seam also makes the
    exception type the only difference between them.
    """
    import pandas as pd

    import superglm.reml.observed_geometry as observed_geometry
    from superglm import Numeric, RandomEffect, SuperGLM
    from superglm.group_matrix import RandomEffectGroupMatrix
    from superglm.reml.observed_geometry import ObservedGeometryInfeasibleError

    rng = np.random.default_rng(2028)
    n, n_levels = 240, 5
    z = rng.normal(size=n)
    codes = rng.integers(0, n_levels, size=n)
    deviation = rng.normal(scale=0.3, size=n_levels)
    y = rng.gamma(shape=6.0, scale=np.exp(0.4 + 0.25 * z + deviation[codes]) / 6.0)
    frame = pd.DataFrame({"z": z, "lvl": [f"L{c}" for c in codes]})

    model = SuperGLM(
        family="gamma",
        link="log",
        selection_penalty=0.0,
        features={
            "z": Numeric(),
            "lvl": RandomEffect(levels=[f"L{i}" for i in range(n_levels)]),
        },
        direct_solve="structured",
    ).fit_reml(frame, y, max_reml_iter=2, runtime_validation="skip")
    structured_index = next(
        i
        for i, matrix in enumerate(model._dm.group_matrices)
        if isinstance(matrix, RandomEffectGroupMatrix)
    )

    def build():
        return build_observed_reml_geometry(
            dm=model._dm,
            distribution=Gamma(),
            link=LogLink(),
            y=y,
            sample_weight=np.ones(n),
            offset_arr=np.zeros(n),
            result=model.result,
            penalty=None,
            derivative_order=0,
            groups=model._groups,
            lambdas=model._reml_lambdas,
            reml_penalties=model._reml_penalties,
            structured_group_index=structured_index,
            compute_inverse=False,
        )

    # The unpatched build reaches the factor and comes back, so each arm below
    # is a statement about the exception it injects rather than about a geometry
    # that was never going to assemble.
    assert build() is not None

    def refuse(error):
        def refusing(system, penalized):
            raise error

        return refusing

    monkeypatch.setattr(
        observed_geometry,
        "build_augmented_structured_factor",
        # The scalar builder's own diagnostic for an operator whose partitions
        # do not agree with the system it augments.
        refuse(ValueError("Penalized and unpenalized operators must use identical partitions.")),
    )
    with pytest.raises(ValueError) as structural:
        build()

    assert not isinstance(structural.value, ObservedGeometryInfeasibleError)
    assert "identical partitions" in str(structural.value)
    assert "Laplace mode" not in str(structural.value)

    monkeypatch.setattr(
        observed_geometry,
        "build_augmented_structured_factor",
        refuse(
            np.linalg.LinAlgError("Structured term 'lvl' has an invalid minimum local diagonal")
        ),
    )
    with pytest.raises(ObservedGeometryInfeasibleError) as conditioning:
        build()

    assert isinstance(conditioning.value.__cause__, np.linalg.LinAlgError)


@pytest.mark.parametrize(
    "case",
    ["nb2-log", "binomial-probit", "poisson-sqrt", "gamma-log", "tweedie-log"],
)
def test_exact_reml_routes_all_noncanonical_builtins_through_observed_geometry(
    case: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pandas as pd
    from scipy.special import ndtr

    import superglm.reml.direct as direct
    from superglm import SuperGLM
    from superglm.features.spline import Spline

    rng = np.random.default_rng(104)
    n = 140
    x = np.linspace(-1.0, 1.0, n)
    if case == "nb2-log":
        distribution = NegativeBinomial(theta=2.5)
        link = LogLink()
        mu = np.exp(0.3 + 0.2 * np.sin(2.0 * x))
        y = rng.negative_binomial(2.5, 2.5 / (2.5 + mu)).astype(float)
    elif case == "binomial-probit":
        distribution = Binomial()
        link = ProbitLink()
        y = rng.binomial(1, ndtr(0.15 + 0.4 * x)).astype(float)
    elif case == "poisson-sqrt":
        distribution = Poisson()
        link = SqrtLink()
        y = rng.poisson((1.1 + 0.25 * np.sin(2.0 * x)) ** 2).astype(float)
    elif case == "gamma-log":
        distribution = Gamma()
        link = LogLink()
        mu = np.exp(0.2 + 0.2 * np.sin(2.0 * x))
        y = rng.gamma(shape=6.0, scale=mu / 6.0)
    else:
        distribution = Tweedie(p=1.5)
        link = LogLink()
        mu = np.exp(0.1 + 0.2 * np.sin(2.0 * x))
        y = np.where(rng.random(n) < 0.25, 0.0, rng.gamma(shape=3.0, scale=mu / 3.0))

    calls: list[tuple[bool, int]] = []
    original = direct.build_observed_reml_geometry

    def observed_geometry_spy(**kwargs):
        calls.append(
            (
                bool(kwargs.get("compute_inverse", True)),
                int(kwargs.get("derivative_order", 0)),
            )
        )
        return original(**kwargs)

    monkeypatch.setattr(direct, "build_observed_reml_geometry", observed_geometry_spy)
    model = SuperGLM(
        family=distribution,
        link=link,
        selection_penalty=0,
        features={"x": Spline(n_knots=5, penalty="ssp")},
    )
    model.fit_reml(
        pd.DataFrame({"x": x}),
        y,
        max_reml_iter=2,
        reml_tol=1e-12,
        pirls_tol=1e-9,
        runtime_validation="skip",
        w_correction_order=2,
    )

    assert any(compute_inverse and order == 2 for compute_inverse, order in calls)
    assert any(not compute_inverse and order == 0 for compute_inverse, order in calls)


def test_discrete_bam_path_bypasses_ordinary_observed_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pandas as pd

    import superglm.reml.direct as direct
    from superglm import SuperGLM
    from superglm.features.spline import Spline

    rng = np.random.default_rng(105)
    x = np.linspace(-1.0, 1.0, 120)
    mu = np.exp(0.2 + 0.2 * np.sin(2.0 * x))
    y = rng.gamma(shape=6.0, scale=mu / 6.0)

    def unexpected_geometry(**_kwargs):  # pragma: no cover - failure sentinel
        raise AssertionError("discrete BAM approximation must bypass ordinary observed geometry")

    monkeypatch.setattr(direct, "build_observed_reml_geometry", unexpected_geometry)
    model = SuperGLM(
        family=Gamma(),
        link=LogLink(),
        selection_penalty=0,
        discrete=True,
        features={"x": Spline(n_knots=5, penalty="ssp")},
    )
    model.fit_reml(
        pd.DataFrame({"x": x}),
        y,
        max_reml_iter=2,
        runtime_validation="skip",
        w_correction_order=2,
    )


def test_custom_order2_observed_capability_fails_before_first_pirls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pandas as pd

    import superglm.reml.direct as direct
    from superglm import SuperGLM
    from superglm.features.spline import Spline

    class CustomGaussian(Gaussian):
        def reml_curvature(self, link):
            return "observed"

    class CustomIdentity(IdentityLink):
        def reml_curvature(self, distribution):
            return "observed"

    def unexpected_pirls(*_args, **_kwargs):  # pragma: no cover - failure sentinel
        raise AssertionError("observed derivative capability must be checked before PIRLS")

    monkeypatch.setattr(direct, "fit_irls_direct", unexpected_pirls)
    x = np.linspace(-1.0, 1.0, 80)
    y = 0.2 + 0.3 * x
    model = SuperGLM(
        family=CustomGaussian(),
        link=CustomIdentity(),
        selection_penalty=0,
        features={"x": Spline(n_knots=5, penalty="ssp")},
    )

    with pytest.raises(NotImplementedError, match="link.deriv4_inverse"):
        model.fit_reml(
            pd.DataFrame({"x": x}),
            y,
            max_reml_iter=2,
            runtime_validation="skip",
            w_correction_order=2,
        )


@pytest.mark.parametrize("family", ["poisson", "gaussian"])
def test_canonical_direct_reml_does_not_build_redundant_observed_geometry(
    family: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pandas as pd

    import superglm.reml.direct as direct
    from superglm import SuperGLM
    from superglm.features.spline import Spline

    rng = np.random.default_rng(123)
    x = rng.uniform(-1.0, 1.0, 240)
    mean = np.exp(0.2 + 0.3 * np.sin(2.0 * x))
    if family == "poisson":
        y = rng.poisson(mean).astype(float)
    else:
        y = 0.4 + 0.3 * np.sin(2.0 * x) + rng.normal(scale=0.2, size=len(x))

    def unexpected_geometry(**_kwargs):  # pragma: no cover - failure sentinel
        raise AssertionError("canonical/equal-curvature fits must reuse Fisher geometry")

    monkeypatch.setattr(direct, "build_observed_reml_geometry", unexpected_geometry)
    model = SuperGLM(
        family=family,
        selection_penalty=0,
        features={"x": Spline(n_knots=6, penalty="ssp")},
    )
    model.fit_reml(pd.DataFrame({"x": x}), y)
    assert model._reml_result.converged
    assert model._reml_profile["reml_observed_geometry_s"] == 0.0


@pytest.mark.parametrize("order", [0, 3, -1, True, 1.5])
def test_fit_reml_rejects_invalid_w_correction_order_before_fitting(order) -> None:
    import pandas as pd

    from superglm import SuperGLM
    from superglm.features.numeric import Numeric

    X = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 20)})
    y = 0.2 + 0.4 * X["x"].to_numpy()
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x": Numeric()},
    )

    with pytest.raises(ValueError, match="w_correction_order.*1 or 2"):
        model.fit_reml(X, y, w_correction_order=order)

    assert model._fit_state is None


class TestModeCertificationRecovery:
    """The mode-score gate must not be self-defeating, nor fatal to a search."""

    @staticmethod
    def _tweedie_fixture(n=600, seed=20260807):
        import pandas as pd

        rng = np.random.default_rng(seed)
        x = np.linspace(0.0, 1.0, n)
        band = np.array([f"{j:02d}" for j in range(12)])[rng.integers(0, 12, n)]
        eta = -0.5 + 0.8 * np.sin(2.0 * np.pi * x)
        y = np.where(rng.random(n) < 0.7, 0.0, rng.gamma(1.4, np.exp(eta) * 40.0, n))
        weights = rng.uniform(1e-3, 1.0, n)
        return pd.DataFrame({"x": x, "band": band}), y, weights

    @staticmethod
    def _model(p=1.5, **kwargs):
        from superglm import SuperGLM
        from superglm.features.spline import Spline

        return SuperGLM(
            family=Tweedie(p=p),
            selection_penalty=0,
            features={"x": Spline(kind="cr", n_knots=4)},
            **kwargs,
        )

    def test_certification_bar_does_not_tighten_with_solver_tolerance(self, monkeypatch):
        """Asking PIRLS to work harder must not raise the bar it is judged against."""
        import superglm.reml.direct as direct_module

        X, y, weights = self._tweedie_fixture()
        # A mode that is good enough for the documented 1e-9 bar, but far above
        # the 1e-13 bar that a tight `tol` used to impose.
        monkeypatch.setattr(
            direct_module,
            "observed_penalized_mode_score",
            lambda **kwargs: SimpleNamespace(relative_max=5.0e-10),
        )

        loose = self._model()
        loose.fit_reml(X, y, sample_weight=weights)

        tight = self._model(tol=1.0e-14)
        tight.fit_reml(X, y, sample_weight=weights)

        assert loose.reml_diagnostics()["converged"] is True
        assert tight.reml_diagnostics()["converged"] is True

    def test_uncertifiable_candidate_is_infeasible_not_fatal(self, monkeypatch):
        """One bad candidate p must not abort the whole power search."""
        import superglm.reml.direct as direct_module

        X, y, weights = self._tweedie_fixture()
        seen = []

        def flaky_score(**kwargs):
            # Certification fails only in the upper power region, exactly the
            # shape observed on real data: feasible below a ceiling, not above.
            failing = any(getattr(g, "p", 0.0) > 1.6 for g in (kwargs.get("distribution"),))
            seen.append(failing)
            return SimpleNamespace(relative_max=1.0e-3 if failing else 1.0e-12)

        monkeypatch.setattr(direct_module, "observed_penalized_mode_score", flaky_score)

        result = self._model().estimate_p(
            X, y, sample_weight=weights, fit_mode="reml", p_bounds=(1.05, 1.95)
        )

        assert any(seen), "the fixture never entered the uncertifiable region"
        assert 1.05 <= result.p_hat <= 1.6
        assert np.isfinite(result.nll)

    def test_infeasible_powers_do_not_poison_the_optimizer_arithmetic(self, monkeypatch):
        """An infeasible score must stay finite: inf-inf is NaN inside Brent."""
        import warnings as _warnings

        import superglm.reml.direct as direct_module

        X, y, weights = self._tweedie_fixture()
        monkeypatch.setattr(
            direct_module,
            "observed_penalized_mode_score",
            lambda **kwargs: SimpleNamespace(
                relative_max=(
                    1.0e-3
                    if any(getattr(g, "p", 0.0) > 1.6 for g in (kwargs.get("distribution"),))
                    else 1.0e-12
                )
            ),
        )

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            self._model().estimate_p(
                X, y, sample_weight=weights, fit_mode="reml", p_bounds=(1.05, 1.95)
            )

        numeric = [w for w in caught if "invalid value" in str(w.message)]
        assert not numeric, f"infeasible scores reached the optimizer as non-finite: {numeric}"

    def test_all_powers_infeasible_reports_the_range_not_a_keyerror(self, monkeypatch):
        """If nothing is certifiable, say so; do not fail looking up a cache."""
        import superglm.reml.direct as direct_module

        X, y, weights = self._tweedie_fixture()
        monkeypatch.setattr(
            direct_module,
            "observed_penalized_mode_score",
            lambda **kwargs: SimpleNamespace(relative_max=1.0e-3),
        )

        with pytest.raises(RuntimeError, match="could not certify"):
            self._model().estimate_p(
                X, y, sample_weight=weights, fit_mode="reml", p_bounds=(1.05, 1.95)
            )

    def test_a_caller_contract_bug_is_not_reported_as_an_unreachable_mode(self, monkeypatch):
        """Only the geometry's own refusal describes a point worth routing around.

        The gate caught every ValueError the geometry build could raise, so a
        violated caller contract -- a misshapen design, a penalty that is not
        PSD -- reached the user as "PIRLS reached no penalized mode at this
        point": the data's conditioning blamed for a bug in the call, and a
        power search invited to score the point infeasible and quietly carry on
        with it.
        """
        import superglm.reml.direct as direct_module

        X, y, weights = self._tweedie_fixture()
        built = []

        def contract_violation(**kwargs):
            built.append(1)
            raise ValueError("penalty must be a finite square matrix in slope coordinates")

        monkeypatch.setattr(direct_module, "build_observed_reml_geometry", contract_violation)

        with pytest.raises(ValueError) as excinfo:
            self._model().fit_reml(X, y, sample_weight=weights)

        assert built, "the gate never reached the geometry build"
        # ObservedModeNotConvergedError is a RuntimeError, so raises(ValueError)
        # already fails against the blanket catch. Typed exactly all the same: a
        # retype into ObservedGeometryInfeasibleError would satisfy ValueError
        # too, and would report the same wrong thing about the same bug.
        assert type(excinfo.value) is ValueError
        assert "slope coordinates" in str(excinfo.value)

    def test_an_infeasible_geometry_is_still_scored_not_raised(self, monkeypatch):
        """Narrowing the catch must not close the door it exists to hold open."""
        import superglm.reml.direct as direct_module
        from superglm.reml.observed_geometry import (
            ObservedGeometryInfeasibleError,
            ObservedModeNotConvergedError,
        )

        X, y, weights = self._tweedie_fixture()

        def infeasible_here(**kwargs):
            raise ObservedGeometryInfeasibleError(
                "observed intercept curvature must have a positive finite sum"
            )

        monkeypatch.setattr(direct_module, "build_observed_reml_geometry", infeasible_here)

        with pytest.raises(ObservedModeNotConvergedError) as excinfo:
            self._model().fit_reml(X, y, sample_weight=weights)

        # The type every handler in this class routes around, carrying the
        # refusal it retyped rather than replacing it.
        assert isinstance(excinfo.value.__cause__, ObservedGeometryInfeasibleError)
        assert "positive finite sum" in str(excinfo.value)

    def test_an_infeasible_mode_score_is_still_scored_not_raised(self, monkeypatch):
        """The candidate gate's score has the same exposure as its geometry.

        The twin of the test above, one call later. Both describe the accepted
        candidate at this point, both are answered by a point scored infeasible
        rather than a failed fit, and a bare ValueError from either walks past
        every ``except ObservedModeNotCertifiedError`` on the way out.
        """
        import superglm.reml.direct as direct_module
        from superglm.reml.observed_geometry import (
            ObservedGeometryInfeasibleError,
            ObservedModeNotConvergedError,
        )

        X, y, weights = self._tweedie_fixture()

        def unscorable_here(**kwargs):
            raise ObservedGeometryInfeasibleError("penalized mode score is not finite")

        monkeypatch.setattr(direct_module, "observed_penalized_mode_score", unscorable_here)

        with pytest.raises(ObservedModeNotConvergedError) as excinfo:
            self._model().fit_reml(X, y, sample_weight=weights)

        assert isinstance(excinfo.value.__cause__, ObservedGeometryInfeasibleError)
        assert "not finite" in str(excinfo.value)

    def test_an_infeasible_terminal_refit_is_infeasible_not_fatal(self, monkeypatch):
        """The terminal refit is a place the power search names, so it must speak the type.

        Publication runs one more observed build after the search has accepted a
        lambda, through a different module's binding, and it can refuse an
        iterate for the reasons every other build can. Left a bare ValueError it
        sails past the handler in ``model/profile_ops.py`` that exists to score
        this power infeasible -- so one power near a bound kills a search whose
        optimum sits well inside the feasible region.
        """
        import superglm.model.reml_finalize as finalize_module
        from superglm.reml.observed_geometry import ObservedGeometryInfeasibleError

        X, y, weights = self._tweedie_fixture()
        original = finalize_module.build_observed_reml_geometry
        refused: list[float] = []

        def infeasible_in_the_upper_region(**kwargs):
            # The shape observed on real data, and the shape the sibling
            # candidate-gate test uses: feasible below a ceiling, not above.
            power = getattr(kwargs.get("distribution"), "p", 0.0)
            if power > 1.6:
                refused.append(float(power))
                raise ObservedGeometryInfeasibleError(
                    "observed intercept curvature must have a positive finite sum"
                )
            return original(**kwargs)

        monkeypatch.setattr(
            finalize_module, "build_observed_reml_geometry", infeasible_in_the_upper_region
        )

        result = self._model().estimate_p(
            X, y, sample_weight=weights, fit_mode="reml", p_bounds=(1.05, 1.95)
        )

        assert refused, "the fixture never reached the infeasible region"
        assert 1.05 <= result.p_hat <= 1.6
        assert np.isfinite(result.nll)

    def test_message_does_not_offer_one_family_s_remedy_to_another(self, monkeypatch):
        """Observed geometry is the default branch, so this reaches many families."""
        import pandas as pd

        import superglm.reml.direct as direct_module
        from superglm import SuperGLM
        from superglm.features.spline import Spline

        monkeypatch.setattr(
            direct_module,
            "observed_penalized_mode_score",
            lambda **kwargs: SimpleNamespace(relative_max=1.0e-3),
        )
        rng = np.random.default_rng(20260807)
        x = np.linspace(0.1, 1.0, 300)
        X = pd.DataFrame({"x": x})
        y = rng.gamma(3.0, np.exp(0.4 + 0.6 * x) / 3.0, 300)

        # Gamma/log is non-canonical, so it takes the observed branch too.
        gamma_model = SuperGLM(
            family=Gamma(), link="log", selection_penalty=0, features={"x": Spline(n_knots=4)}
        )
        with pytest.raises(RuntimeError) as gamma_error:
            gamma_model.fit_reml(X, y)

        message = str(gamma_error.value)
        assert "could not certify" in message
        assert "does not move it" in message
        assert "Tweedie" not in message
        assert "estimate_p" not in message

    def test_tweedie_message_names_the_parameter_the_caller_can_change(self, monkeypatch):
        import superglm.reml.direct as direct_module

        X, y, weights = self._tweedie_fixture()
        monkeypatch.setattr(
            direct_module,
            "observed_penalized_mode_score",
            lambda **kwargs: SimpleNamespace(relative_max=1.0e-3),
        )

        with pytest.raises(RuntimeError, match="p approaches 2"):
            self._model(p=1.7).fit_reml(X, y, sample_weight=weights)


class TestModeCertifiesAtTheRoundOffFloor:
    """PIRLS's step-length flag must not veto a certifiable mode.

    Under observed geometry PIRLS is asked to drive
    ``max|dbeta| / max(1, |beta|)`` below ``OBSERVED_PIRLS_TOL_CEILING``
    (1e-10). On a burn-cost-scale Tweedie problem that threshold sits BELOW
    the attainable round-off floor of the iteration map: the iterate reaches
    the mode, enters a period-2 round-off limit cycle between two adjacent
    floating-point states, and the step test can never fire. Measured floors
    on the fixture below run 9x to 646x above the ceiling, and the pass/fail
    outcome was decided entirely by whether a given draw's floor happened to
    land under 1e-10 -- two draws a factor of 1.4 apart in floating-point
    noise separated a published model from a hard error.

    A step-length test cannot tell convergence from stagnation, which is why
    it is the secondary criterion everywhere it appears (SAS/IML ships its
    gradient tests active and its parameter-change tests disabled by
    default). The authoritative instrument is the KKT residual this module
    already computes, and it fails closed: a mid-descent mode scores 1e-1 to
    1e-4 against the 1e-9 bar. So the gate defers to the certificate rather
    than pre-empting it.
    """

    @staticmethod
    def _burn_cost_fixture(n=20_000, k=3, seed=0, phi=150.0, p=1.5):
        """A small imitation of a burn-cost book, at the scale that matters.

        The round-off floor scales with the magnitude of the weighted sums,
        so the response scale is load-bearing: shrink ``phi`` and the mean
        and the floor drops back under the ceiling and the bug vanishes.
        """
        import pandas as pd

        from superglm import generate_tweedie_cpg

        rng = np.random.default_rng(seed)
        level = rng.integers(0, k, size=n)
        level_effect = rng.normal(0.0, 0.35, size=k)
        term_months = rng.choice([12.0, 36.0], size=n)
        offset = np.log(term_months / 12.0)
        weight = rng.uniform(0.01, 1.0, size=n)
        eta = np.log(650.0) + level_effect[level] + offset
        y = generate_tweedie_cpg(
            n,
            np.exp(eta),
            phi / np.maximum(weight, 1e-6),
            p,
            rng=np.random.default_rng(seed + 1),
        )
        frame = pd.DataFrame({"lvl": [f"L{i:02d}" for i in level]})
        return frame, y, weight, offset

    @staticmethod
    def _model(k=3, p=1.5):
        from superglm import RandomEffect, SuperGLM

        return SuperGLM(
            family=Tweedie(p=p),
            link="log",
            selection_penalty=0.0,
            features={
                "lvl": RandomEffect(levels=[f"L{i:02d}" for i in range(k)]),
            },
        )

    @staticmethod
    @contextlib.contextmanager
    def _recording_pirls():
        """Record ``(gate, termination reason)`` for every observed-geometry PIRLS call.

        A fixture that stops stalling stops testing anything, and it can stop
        silently: the round-off floor moves with the numeric stack, so a draw
        that exercised the defect on one numpy can certify on the next. The
        tests below assert the stall was actually reached, so a fixture that
        goes vacuous fails loudly instead of passing for the wrong reason.

        Three gates defer to the certificate and each reads a DIFFERENT
        module-level ``fit_irls_direct``: ``reml/direct.py`` holds the
        candidate gate and the line-search trial, and ``model/reml_finalize.py``
        imported the name for itself and holds the terminal publication refit.
        Rebinding one module reaches two gates and leaves the third
        unwatched -- which is how the terminal refit went unrecorded here, with
        nothing in the guard able to say so. Both bindings are patched, and
        each record carries the call's own ``trace_purpose`` so a test can name
        the gate it means rather than a tag invented for the occasion.
        """
        import superglm.model.reml_finalize as finalize_module
        import superglm.reml.direct as direct_module

        records: list[tuple[str | None, str | None]] = []
        patched = [
            (direct_module, direct_module.fit_irls_direct),
            (finalize_module, finalize_module.fit_irls_direct),
        ]

        def recorder(original):
            def recording(*args, **kwargs):
                output = original(*args, **kwargs)
                result = output[0] if isinstance(output, tuple) else output
                if kwargs.get("convergence") == "coefficients":
                    records.append((kwargs.get("trace_purpose"), result.termination_reason))
                return output

            return recording

        for module, original in patched:
            module.fit_irls_direct = recorder(original)
        try:
            yield records
        finally:
            for module, original in patched:
                module.fit_irls_direct = original

    @pytest.mark.parametrize("seed", range(8))
    def test_every_draw_fits_not_just_the_lucky_ones(self, seed):
        """The defining symptom: identical design, outcome decided by the draw.

        Held to a sweep rather than one fixture because any single draw
        passes most of the time -- which is what made this look stochastic.
        """
        frame, y, weight, offset = self._burn_cost_fixture(seed=seed)
        model = self._model()
        model.fit_reml(frame, y, sample_weight=weight, offset=offset, max_reml_iter=30)

        assert model.reml_diagnostics()["converged"] is True
        assert np.all(np.isfinite(model.predict(frame, offset=offset)))

    def test_the_sweep_still_reaches_the_stall_it_regresses(self):
        """Guard against the fixture going vacuous -- see ``_recording_pirls``.

        Counted per seed and held to a majority. A sum over the whole sweep --
        over draws, and over every PIRLS call within a draw -- lets one stall
        anywhere stand in for all eight: seeds 1 and 3 never stall here and
        both pass against the unfixed code, so two of the eight cases above
        already regress nothing while a summed count reports the fixture
        healthy. The threshold is a majority rather than a named set of seeds
        because WHICH draw stalls is decided by a round-off floor that moves
        with numpy and the BLAS. Pinning individual seeds would red on a stack
        where the defect is merely differently distributed; a majority still
        catches the failure that matters, the fixture going vacuous everywhere.
        """
        stalled = []
        for seed in range(8):
            frame, y, weight, offset = self._burn_cost_fixture(seed=seed)
            with self._recording_pirls() as records:
                self._model().fit_reml(
                    frame, y, sample_weight=weight, offset=offset, max_reml_iter=30
                )
            if any(reason == "max_iter" for _, reason in records):
                stalled.append(seed)
        assert len(stalled) >= 5, (
            f"MEASURED NOW: {len(stalled)} of 8 draws exhausted a PIRLS budget "
            f"(seeds {stalled}), against a threshold of 5. MEASURED THEN: 6 of "
            "8, on numpy 2.4.2 / scipy 1.18.0, 2026-08-18 -- a baseline, not a "
            "set this run should have reproduced, because WHICH draw stalls "
            "moves with the numeric stack. Below the threshold most of the "
            "sweep no longer reaches the round-off floor it exists to regress, "
            "and the cases above are passing on draws that would pass against "
            "the unfixed gate too."
        )

    def test_the_recorder_watches_every_gate_that_defers(self):
        """The terminal publication refit is a gate too, with its own binding.

        A claim about the guard above rather than about the fix: a recorder
        blind to a gate cannot report that gate going vacuous. Rebinding only
        ``reml/direct.py`` -- the shape this helper had -- watches the candidate
        gate and the line-search trial and records no terminal refit at all, on
        any draw.
        """
        frame, y, weight, offset = self._burn_cost_fixture(seed=0)
        with self._recording_pirls() as records:
            self._model().fit_reml(frame, y, sample_weight=weight, offset=offset, max_reml_iter=30)

        gates = {purpose for purpose, _ in records}
        assert gates >= {"reml_candidate", "reml_line_search", "reml_final"}, (
            f"a gate that defers to the certificate went unwatched; recorded {sorted(gates)}"
        )

    def test_the_published_mode_is_certified_not_merely_accepted(self):
        """Deferring to the certificate must mean the certificate ran and passed."""
        from superglm.reml.observed_geometry import observed_mode_certification_bar

        frame, y, weight, offset = self._burn_cost_fixture(seed=0)
        model = self._model()
        model.fit_reml(frame, y, sample_weight=weight, offset=offset, max_reml_iter=30)

        residual = model._reml_profile["reml_terminal_observed_mode_residual"]
        assert residual <= observed_mode_certification_bar()

    def test_an_uncertifiable_mode_is_still_refused(self, monkeypatch):
        """Fail-closed: the certificate, not the step test, is what guards the gate."""
        import superglm.reml.direct as direct_module
        from superglm.reml.observed_geometry import (
            ObservedModeNotCertifiedError,
            ObservedModeNotConvergedError,
        )

        frame, y, weight, offset = self._burn_cost_fixture(seed=0)
        monkeypatch.setattr(
            direct_module,
            "observed_penalized_mode_score",
            lambda **kwargs: SimpleNamespace(relative_max=1.0e-3),
        )
        with pytest.raises(ObservedModeNotCertifiedError) as excinfo:
            self._model().fit_reml(frame, y, sample_weight=weight, offset=offset, max_reml_iter=30)
        # ObservedModeNotConvergedError SUBCLASSES the error asserted above, so
        # a bare raises() cannot tell the certificate's refusal from the gate's
        # own, and passes against the unfixed code.
        assert not isinstance(excinfo.value, ObservedModeNotConvergedError)
        assert "certify the penalized coefficient mode" in str(excinfo.value)

    def test_only_budget_exhaustion_defers_to_the_certificate(self):
        """Every other termination reason names something the score cannot judge."""
        from superglm.reml.observed_geometry import stopped_on_iteration_budget
        from superglm.solvers.pirls import PIRLS_TERMINATION_REASONS

        def ended(reason):
            return stopped_on_iteration_budget(SimpleNamespace(termination_reason=reason))

        assert "max_iter" in PIRLS_TERMINATION_REASONS
        assert ended("max_iter")
        # Read out of the solver's own exported vocabulary rather than copied
        # into a tuple here. The copy claimed a reason added to PIRLS could not
        # silently change which door it takes, and it could: a scratch build
        # carrying a real new literal left this green. What is pinned is only
        # that every OTHER declared reason takes the non-deferring door --
        # that the vocabulary is complete is the solver's own assertion.
        for reason in (*sorted(PIRLS_TERMINATION_REASONS - {"max_iter"}), None):
            assert not ended(reason), reason

    def test_no_solver_writes_a_termination_reason_the_vocabulary_omits(self):
        """Read the literals out of the source, because a set cannot list its own gaps.

        The test above drives its sweep from ``PIRLS_TERMINATION_REASONS``, which
        by construction cannot notice a literal that was never declared -- and
        the type annotation that would is checked by a gate carrying a backlog
        ceiling, so one new diagnostic does not red the build. This parses both
        loops instead and asks the opposite question: does every string that
        reaches ``termination_reason`` appear in the vocabulary the consumers
        switch on? A guard rather than a fix, and green today by construction;
        it earns its place by reddening on a literal nobody declared.

        The walk collects string constants anywhere inside the assigned
        expression, so a reason hidden in a conditional is still seen. That is a
        deliberate superset: a literal that reaches the field by any expression
        shape still has to be declared.

        It has to read more than plain assignment to make that claim. The field
        is also written as a constructor keyword and by ``replace()``, and an
        earlier version of this guard saw neither -- so a reason introduced as
        ``PIRLSResult(..., termination_reason="...")`` would have shipped
        undeclared, which is the one case the guard exists for.
        """
        import ast
        from pathlib import Path

        import superglm.solvers.irls_direct as irls_direct_module
        import superglm.solvers.pirls as pirls_module
        from superglm.solvers.pirls import PIRLS_TERMINATION_REASONS

        def names_the_field(target) -> bool:
            """Does this assignment target write ``termination_reason``?

            Three shapes reach the field: the loop local, an attribute on a
            result object, and a dict entry in the trace payloads.
            """
            if isinstance(target, ast.Name):
                return target.id == "termination_reason"
            if isinstance(target, ast.Attribute):
                return target.attr == "termination_reason"
            if isinstance(target, ast.Subscript):
                key = target.slice
                return isinstance(key, ast.Constant) and key.value == "termination_reason"
            return False

        def assigned_literals(module) -> set[str]:
            tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
            literals: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.AnnAssign):
                    # A bare ``termination_reason: TerminationReason`` field
                    # declaration carries no value to read.
                    written = [node.target] if node.value is not None else []
                    value = node.value
                elif isinstance(node, ast.Assign):
                    written, value = node.targets, node.value
                elif isinstance(node, ast.keyword):
                    # ``PIRLSResult(..., termination_reason="converged")`` and
                    # ``replace(result, termination_reason=...)`` never bind a
                    # name, so a target walk alone cannot see them.
                    written, value = [], node.value
                    if node.arg != "termination_reason":
                        continue
                else:
                    continue
                if value is None:
                    continue
                if not isinstance(node, ast.keyword) and not any(
                    names_the_field(target) for target in written
                ):
                    continue
                literals.update(
                    child.value
                    for child in ast.walk(value)
                    if isinstance(child, ast.Constant) and isinstance(child.value, str)
                )
            return literals

        written = assigned_literals(pirls_module) | assigned_literals(irls_direct_module)
        # Without this the test passes by parsing nothing -- a renamed local, a
        # moved loop, a file that no longer holds either.
        assert written, "no termination_reason literal was found in either solver"
        assert written <= set(PIRLS_TERMINATION_REASONS), sorted(
            written - set(PIRLS_TERMINATION_REASONS)
        )

    @staticmethod
    def _stamped_terminal_refit(monkeypatch, *, converged, termination_reason):
        """Make the terminal publication refit report a chosen verdict.

        No draw produces a budget-exhausted terminal refit. It warm starts from
        the accepted candidate's coefficients at the final lambda, so its first
        step is already under the 1e-10 bar and it returns after one iteration
        reporting ``converged`` -- measured on every one of 18 draws (n=20k to
        200k, k=3 to 40, means 650 to 650000) and on the factor-smooth fixture
        below. A test that waited for the natural case would assert nothing at
        all. Only the verdict is stamped: the coefficients, the geometry built
        from them and the certificate that scores them are the real refit's, so
        the label the relabel publishes is still earned by a real KKT residual.

        Returns the verdicts the refit actually reached, so a caller can say
        whether the gate ran rather than assume it did.
        """
        import superglm.model.reml_finalize as finalize_module

        original = finalize_module.fit_irls_direct
        reached: list[tuple[bool, str | None]] = []

        def stamping(*args, **kwargs):
            output = original(*args, **kwargs)
            if kwargs.get("trace_purpose") != "reml_final":
                return output
            result = output[0] if isinstance(output, tuple) else output
            reached.append((result.converged, result.termination_reason))
            stamped = replace(result, converged=converged, termination_reason=termination_reason)
            return (stamped, *output[1:]) if isinstance(output, tuple) else stamped

        monkeypatch.setattr(finalize_module, "fit_irls_direct", stamping)
        return reached

    @staticmethod
    def _reported_convergence(summary):
        """The Converged value as a reader of the rendered summary sees it."""
        rendered = str(summary)
        match = re.search(r"Converged:\s*(True|False)", rendered)
        assert match is not None, f"no Converged row in the rendered summary:\n{rendered}"
        return match.group(1)

    def test_a_certified_budget_exhausted_publication_reads_converged(self, monkeypatch):
        """What the certificate admitted, every reader of the fit must report.

        The gate lets a budget-exhausted terminal refit publish, and the
        published result then carried PIRLS's step-length verdict on a mode the
        certificate had just passed. ``summary()``, ``metrics().summary()`` and
        the telemetry payload all read that flag, so a fit admitted on its KKT
        residual reported itself unconverged while its own REML diagnostics
        said the opposite.
        """
        from superglm.reml.observed_geometry import observed_mode_certification_bar

        frame, y, weight, offset = self._burn_cost_fixture(seed=0)
        reached = self._stamped_terminal_refit(
            monkeypatch, converged=False, termination_reason="max_iter"
        )
        model = self._model()
        model.fit_reml(frame, y, sample_weight=weight, offset=offset, max_reml_iter=30)

        assert reached, "the terminal publication refit never ran, so nothing was relabelled"
        # Pinned to the certificate, not merely to the branch: the relabelled
        # claim is defensible only because this residual cleared the fixed bar.
        residual = model._reml_profile["reml_terminal_observed_mode_residual"]
        assert residual <= observed_mode_certification_bar()

        assert model.result.converged is True
        assert model.result.termination_reason == "mode_certified"
        # A candidate whose public and solver copies disagree on this flag is
        # refused at fit-state validation, so the relabel has to reach both.
        assert bool(model._result.converged) is bool(model._solver_result.converged)
        assert model._result.termination_reason == model._solver_result.termination_reason

        assert model.training_telemetry()["fit"]["converged"] is True
        metrics = model.metrics(frame, y, sample_weight=weight, offset=offset)
        # The two summaries reach the flag by different routes -- one through
        # the REML loop's verdict, one through the published PIRLS result --
        # which is how they came to print opposite answers for one fit.
        assert self._reported_convergence(model.summary()) == "True"
        assert self._reported_convergence(metrics.summary()) == "True"

    def test_a_terminal_refit_that_converged_keeps_its_own_reason(self, monkeypatch):
        """The relabel is conditioned on the step test having failed, not on the gate.

        ``mode_certified`` names one situation -- a mode the step test never
        passed, published because the certificate did -- and it stops naming it
        the moment it is stamped on refits that converged normally. The verdict
        here is pinned rather than changed: every terminal refit measured on
        this fixture already reported exactly this, so the stamp only fixes the
        branch's input on a stack where that stops being true.
        """
        frame, y, weight, offset = self._burn_cost_fixture(seed=0)
        reached = self._stamped_terminal_refit(
            monkeypatch, converged=True, termination_reason="converged"
        )
        model = self._model()
        model.fit_reml(frame, y, sample_weight=weight, offset=offset, max_reml_iter=30)

        assert reached, "the terminal publication refit never ran"
        assert model.result.converged is True
        assert model.result.termination_reason == "converged"

    @staticmethod
    def _factor_smooth_fixture(n=60_000, seed=1, phi=150.0, p=1.5, n_cat=3, cat_levels=6):
        """A fully penalised ``fs`` factor smooth beside unpenalised factors.

        The second symptom of the same defect. ``fs`` penalises its null space
        too, so its lambda bootstraps near zero and the candidate solve warm
        starts already at its own mode -- the first step is at the round-off
        floor with no descent to cross the ceiling on. ``sz`` keeps an
        unpenalised null space, its lambda moves, and it never stalls.
        """
        import pandas as pd

        from superglm import generate_tweedie_cpg

        rng = np.random.default_rng(seed)
        x = rng.uniform(0.0, 100.0, size=n)
        grp = rng.integers(0, 2, size=n)
        offset = np.log(rng.choice([12.0, 36.0], size=n) / 12.0)
        weight = rng.uniform(0.01, 1.0, size=n)
        eta = (
            np.log(650.0) + 0.010 * x + np.where(grp == 0, 1.0, -1.0) * 0.004 * (x - 50.0) + offset
        )
        columns = {"x": x, "grp": pd.Categorical(grp.astype(str))}
        for j in range(n_cat):
            level = rng.integers(0, cat_levels, size=n)
            eta = eta + 0.03 * level
            columns[f"cat{j}"] = pd.Categorical([f"C{i}" for i in level])
        y = generate_tweedie_cpg(
            n,
            np.exp(eta),
            phi / np.maximum(weight, 1e-6),
            p,
            rng=np.random.default_rng(seed + 1),
        )
        return pd.DataFrame(columns), y, weight, offset, n_cat

    @pytest.mark.parametrize("basis", ["fs", "sz"])
    def test_a_factor_smooth_certifies_on_either_basis(self, basis):
        """``fs`` failed where ``sz`` fit, on identical data through one gate.

        Reported separately from the RandomEffect symptom; the same predicate
        closes both, so both are pinned here.
        """
        from superglm import Categorical, FactorSmooth, Spline, SuperGLM

        frame, y, weight, offset, n_cat = self._factor_smooth_fixture()
        features: dict = {"x": Spline(kind="ps", k=10)}
        for j in range(n_cat):
            features[f"cat{j}"] = Categorical(base="most_exposed")
        model = SuperGLM(
            family=Tweedie(p=1.5),
            link="log",
            selection_penalty=0.0,
            features=features,
            interactions=[FactorSmooth("x", group="grp", basis=basis, kind="ps", k=6)],
        )
        columns = ["x", "grp"] + [f"cat{j}" for j in range(n_cat)]

        with self._recording_pirls() as records:
            model.fit_reml(frame[columns], y, sample_weight=weight, offset=offset)

        assert model.reml_diagnostics()["converged"] is True
        if basis == "fs":
            assert any(reason == "max_iter" for _, reason in records), (
                "the fs fixture no longer reaches the stall it exists to regress"
            )
