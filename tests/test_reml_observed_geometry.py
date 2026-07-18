"""Observed-information geometry for Wood's LAML criterion."""

from __future__ import annotations

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
from superglm.solvers.rank import decompose_factor, decompose_gram, needs_factor_certification
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


def test_observed_geometry_uses_factor_certification_above_rank_boundary() -> None:
    rows = np.arange(128)
    primary = np.where(rows % 2, 1.0, -1.0)
    orthogonal = np.where(rows % 4 < 2, 1.0, -1.0)
    X = np.column_stack((primary, primary + 3.03e-8 * orthogonal))
    dm = DesignMatrix([DenseGroupMatrix(X)], n=len(rows), p=2)

    preliminary = decompose_gram(X.T @ X)
    certified = decompose_factor(
        grouped_augmented_factor(
            dm,
            np.ones(len(rows)),
            np.zeros((2, 2)),
            center=np.zeros(2),
        )
    )
    assert preliminary.rank == 1
    assert preliminary.resolution_limited
    assert needs_factor_certification(preliminary)
    assert certified.rank == 2

    geometry = build_observed_reml_geometry(
        dm=dm,
        distribution=Gamma(),
        link=LogLink(),
        y=np.ones(len(rows)),
        sample_weight=np.ones(len(rows)),
        offset_arr=np.zeros(len(rows)),
        result=_result(np.zeros(2), 0.0),
        penalty=np.zeros((2, 2)),
    )

    assert geometry.hessian_rank == 3
    np.testing.assert_allclose(
        geometry.hessian_inverse,
        certified.pseudo_inverse(),
        rtol=2e-15,
        atol=0.0,
    )
    assert geometry.log_det_H == pytest.approx(
        float(np.log(len(rows)) + certified.log_pdet),
        rel=2e-15,
        abs=0.0,
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

    with pytest.raises(ValueError, match="negative|indefinite"):
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
