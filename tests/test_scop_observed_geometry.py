"""Dense oracles for shape-constrained observed REML geometry."""

import numpy as np
import pandas as pd
import pytest


def _curvature_solver_reparam(q_raw: int, kind: str):
    from superglm.solvers.scop import build_scop_solver_reparam

    degree = 3
    n_interior = q_raw - degree - 1
    knots = np.concatenate(
        (
            np.zeros(degree + 1),
            np.linspace(0.0, 1.0, n_interior + 2)[1:-1] ** 1.7,
            np.ones(degree + 1),
        )
    )
    return build_scop_solver_reparam(
        q_raw,
        kind=kind,
        knots=knots,
        degree=degree,
        domain=(0.0, 1.0),
    )


def test_scop_curvature_classifier_covers_proven_builtin_pairs():
    """Only analytically equal Fisher/observed pairs may reuse Fisher geometry."""
    from superglm.distributions import (
        Binomial,
        Gamma,
        Gaussian,
        NegativeBinomial,
        Poisson,
        Tweedie,
    )
    from superglm.links import (
        IdentityLink,
        InverseLink,
        LogitLink,
        LogLink,
        NegativeBinomialLink,
        PowerLink,
    )
    from superglm.reml.observed_geometry import classify_scop_reml_curvature

    fisher_pairs = [
        (Gaussian(), IdentityLink()),
        (Poisson(), LogLink()),
        (Binomial(), LogitLink()),
        (Gamma(), InverseLink()),
        (NegativeBinomial(theta=2.5), NegativeBinomialLink(theta=2.5)),
        (Tweedie(p=1.6), PowerLink(power=-0.6)),
    ]
    for distribution, link in fisher_pairs:
        assert classify_scop_reml_curvature(distribution, link) == "fisher"

    assert classify_scop_reml_curvature(NegativeBinomial(theta=2.5), LogLink()) == "observed"
    assert classify_scop_reml_curvature(Tweedie(p=1.6), LogLink()) == "observed"
    assert (
        classify_scop_reml_curvature(
            NegativeBinomial(theta=2.5),
            NegativeBinomialLink(theta=3.0),
        )
        == "observed"
    )


def test_scop_curvature_classifier_rejects_implicit_custom_geometry():
    """Custom family/link pairs must explicitly certify their SCOP curvature."""
    from superglm.reml.observed_geometry import classify_scop_reml_curvature

    class CustomDistribution:
        pass

    class CustomLink:
        pass

    with pytest.raises(NotImplementedError, match="explicit SCOP REML curvature protocol"):
        classify_scop_reml_curvature(CustomDistribution(), CustomLink())

    class CertifiedDistribution:
        def scop_reml_curvature(self, link):
            return "observed"

    assert classify_scop_reml_curvature(CertifiedDistribution(), CustomLink()) == "observed"


def test_scop_observed_rows_reject_signed_curvature_before_moments():
    """Indefinite row curvature needs signed stable kernels, so it must not fall through."""
    from superglm.distributions import Binomial
    from superglm.links import CauchitLink
    from superglm.reml.observed_geometry import compute_scop_observed_information_weights

    eta = np.array([-1.0, 0.0, 1.0])
    link = CauchitLink()
    mu = link.inverse(eta)
    y = np.array([1.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="signed observed-information rows"):
        compute_scop_observed_information_weights(
            Binomial(),
            link,
            y,
            mu,
            eta,
            np.ones_like(y),
        )


@pytest.mark.parametrize(
    ("distribution", "link", "y", "mu", "expected"),
    [
        pytest.param(
            "nb",
            "log",
            np.array([0.0, 2.0, 7.0]),
            np.array([0.4, 1.7, 5.0]),
            None,
            id="nb2-log",
        ),
        pytest.param(
            "tweedie",
            "log",
            np.array([0.0, 1.5, 4.0]),
            np.array([0.3, 1.2, 3.5]),
            None,
            id="tweedie-log",
        ),
    ],
)
def test_positive_noncanonical_scop_rows_use_exact_observed_formula(
    distribution, link, y, mu, expected
):
    """NB2/log and Tweedie/log have positive exact Newton rows."""
    from superglm.distributions import NegativeBinomial, Tweedie
    from superglm.links import LogLink
    from superglm.reml.observed_geometry import compute_scop_observed_information_weights

    weights = np.array([0.7, 1.1, 1.8])
    eta = np.log(mu)
    if distribution == "nb":
        family = NegativeBinomial(theta=2.3)
        expected = weights * 2.3 * mu * (2.3 + y) / (2.3 + mu) ** 2
    else:
        family = Tweedie(p=1.55)
        expected = weights * mu ** (1.0 - 1.55) * ((2.0 - 1.55) * mu + 0.55 * y)

    def unexpected_generic_variance(_mu):
        raise AssertionError("the log-link specialization should avoid generic derivatives")

    family.variance = unexpected_generic_variance

    actual = compute_scop_observed_information_weights(
        family,
        LogLink(),
        y,
        mu,
        eta,
        weights,
    )

    assert np.all(actual >= 0.0)
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)


def test_observed_scop_geometry_uses_retained_eta_under_large_translation():
    """Observed rows must not reconstruct eta through catastrophic cancellation."""
    from types import SimpleNamespace

    from superglm.distributions import Gamma
    from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
    from superglm.links import LogLink
    from superglm.reml.scop_geometry import build_observed_scop_joint_geometry

    x = np.linspace(-0.8, 1.1, 41)
    translated = x + 1.0e10
    centered = translated - translated[0]
    retained_eta = -0.2 + 0.6 * centered
    mu = np.exp(retained_eta)
    y = mu * (1.0 + 0.12 * np.sin(np.arange(x.size)))
    sample_weight = np.linspace(0.7, 1.4, x.size)
    penalty = np.array([[0.3]])

    translated_dm = DesignMatrix(
        [DenseGroupMatrix(translated[:, None])],
        n=x.size,
        p=1,
    )
    centered_dm = DesignMatrix(
        [DenseGroupMatrix(centered[:, None])],
        n=x.size,
        p=1,
    )
    common = {
        "distribution": Gamma(),
        "link": LogLink(),
        "y": y,
        "sample_weight": sample_weight,
        "offset_arr": np.zeros_like(y),
        "penalty": penalty,
        "scop_states": {},
    }

    expected = build_observed_scop_joint_geometry(
        dm=centered_dm,
        result=SimpleNamespace(beta=np.array([0.6]), intercept=-0.2),
        eta_unclipped=retained_eta,
        **common,
    )
    actual = build_observed_scop_joint_geometry(
        dm=translated_dm,
        result=SimpleNamespace(
            beta=np.array([0.6]),
            intercept=-0.2 - 0.6 * translated[0],
        ),
        eta_unclipped=retained_eta,
        **common,
    )

    np.testing.assert_allclose(
        actual.centered_hessian,
        expected.centered_hessian,
        rtol=5e-13,
        atol=5e-13,
    )
    assert actual.log_det_H == pytest.approx(expected.log_det_H, rel=5e-13, abs=5e-13)


def test_scop_latent_mode_score_certifies_the_penalized_root():
    """The terminal certificate is in latent, intercept-profiled coordinates."""
    from types import SimpleNamespace

    from superglm.distributions import Gaussian
    from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
    from superglm.links import IdentityLink
    from superglm.reml.scop_geometry import scop_penalized_mode_score

    x = np.array([-1.0, 0.0, 1.0])
    dm = DesignMatrix([DenseGroupMatrix(x[:, None])], n=3, p=1)
    beta_eff = np.array([np.log(2.0)])
    mapped = np.exp(beta_eff)
    penalty = np.array([[1.5]])
    result = SimpleNamespace(beta=mapped.copy(), intercept=0.4)
    mu = result.intercept + x * mapped[0]
    fisher_mean = np.array([0.0])
    centered_fisher = np.array([[2.0]])
    states = {
        0: {
            "group_sl": slice(0, 1),
            "group_name": "mono",
            "beta_eff": beta_eff,
            "gamma_eff": mapped,
        }
    }
    common = {
        "dm": dm,
        "distribution": Gaussian(),
        "link": IdentityLink(),
        "sample_weight": np.ones(3),
        "offset_arr": np.zeros(3),
        "result": result,
        "latent_penalty": penalty,
        "scop_states": states,
        "centered_fisher_gram": centered_fisher,
        "fisher_mean_x": fisher_mean,
        "fisher_sum_w": 3.0,
    }

    nonstationary = scop_penalized_mode_score(y=mu, **common)
    assert nonstationary.relative_max > 0.1

    # Choose a zero-sum likelihood score whose centered slope component
    # exactly balances S beta in latent coordinates.
    target_raw_score = float((penalty @ beta_eff)[0] / mapped[0])
    residual = 0.5 * target_raw_score * x
    stationary = scop_penalized_mode_score(y=mu + residual, **common)
    assert stationary.relative_max < 2e-15


def test_scop_latent_mode_score_treats_nullspace_roundoff_as_zero():
    """A suppressed SCOP null-space must not fail KKT certification on cancellation."""
    from types import SimpleNamespace

    from superglm.distributions import Gaussian
    from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
    from superglm.links import IdentityLink
    from superglm.reml.scop_geometry import scop_penalized_mode_score

    # Wider than the former fixed 64-epsilon guard so the regression exercises
    # the dimension-aware dot-product bound.
    q = 384
    n = 12
    rng = np.random.default_rng(4)
    X = rng.normal(size=(n, q))
    dm = DesignMatrix([DenseGroupMatrix(X)], n=n, p=q)

    null = np.ones(q) / np.sqrt(q)
    complement = rng.normal(size=(q, q - 1))
    complement -= null[:, None] * (null @ complement)[None, :]
    complement = np.linalg.qr(complement)[0][:, : q - 1]
    penalty = complement @ np.diag(np.arange(1.0, q)) @ complement.T * 1.0e-4

    beta_eff = np.full(q, -49.0)
    mapped = np.exp(beta_eff)
    result = SimpleNamespace(beta=mapped.copy(), intercept=0.0)
    mu = X @ mapped
    fisher_mean = X.mean(axis=0)
    centered = X - fisher_mean
    score = scop_penalized_mode_score(
        dm=dm,
        distribution=Gaussian(),
        link=IdentityLink(),
        y=mu,
        sample_weight=np.ones(n),
        offset_arr=np.zeros(n),
        result=result,
        latent_penalty=penalty,
        scop_states={
            0: {
                "group_sl": slice(0, q),
                "group_name": "mono",
                "beta_eff": beta_eff,
                "gamma_eff": mapped,
            }
        },
        centered_fisher_gram=centered.T @ centered,
        fisher_mean_x=fisher_mean,
        fisher_sum_w=float(n),
    )

    assert score.max_abs < 1.0e-10
    assert score.relative_max < 1.0e-9


def test_scop_latent_mode_score_does_not_hide_real_data_score_with_null_penalty():
    """Penalty-matvec uncertainty must not erase an independent data residual."""
    from types import SimpleNamespace

    from superglm.distributions import Gaussian
    from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
    from superglm.links import IdentityLink
    from superglm.reml.scop_geometry import scop_penalized_mode_score

    X = np.eye(2)
    dm = DesignMatrix([DenseGroupMatrix(X)], n=2, p=2)
    beta_eff = np.ones(2)
    gamma_eff = np.exp(beta_eff)
    penalty = 1.0e10 * np.array([[1.0, -1.0], [-1.0, 1.0]])
    residual = np.array([1.0e-6, -1.0e-6])
    mu = X @ gamma_eff
    fisher_mean = X.mean(axis=0)
    centered = X - fisher_mean

    score = scop_penalized_mode_score(
        dm=dm,
        distribution=Gaussian(),
        link=IdentityLink(),
        y=mu + residual,
        sample_weight=np.ones(2),
        offset_arr=np.zeros(2),
        result=SimpleNamespace(beta=gamma_eff.copy(), intercept=0.0),
        latent_penalty=penalty,
        scop_states={
            0: {
                "group_sl": slice(0, 2),
                "group_name": "mono",
                "beta_eff": beta_eff,
                "gamma_eff": gamma_eff,
            }
        },
        centered_fisher_gram=centered.T @ centered,
        fisher_mean_x=fisher_mean,
        fisher_sum_w=2.0,
    )

    assert score.max_abs == pytest.approx(np.e * 1.0e-6)
    assert score.relative_max == pytest.approx(1.0)


@pytest.mark.parametrize("family_name", ["nb2", "tweedie"])
def test_noncanonical_log_scop_fit_installs_observed_geometry(family_name):
    """The production direct path must not silently label Fisher rows as Newton rows."""
    from superglm import Constraint, SuperGLM
    from superglm.distributions import NegativeBinomial, Tweedie
    from superglm.features.spline import PSpline
    from superglm.model.base import model_build_design_matrix
    from superglm.solvers.irls_direct import fit_irls_direct

    rng = np.random.default_rng(20260730)
    n = 150
    x = np.sort(rng.uniform(0.0, 1.0, size=n))
    mean = np.exp(-0.2 + x)
    if family_name == "nb2":
        family = NegativeBinomial(theta=2.5)
        y = rng.negative_binomial(2.5, 2.5 / (2.5 + mean)).astype(float)
    else:
        family = Tweedie(p=1.5)
        y = np.maximum(0.0, mean + rng.normal(scale=0.4 * mean**0.75, size=n))
    frame = pd.DataFrame({"x": x})
    model = SuperGLM(
        family=family,
        selection_penalty=0.0,
        discrete=True,
        features={"x": PSpline(n_knots=6, constraint=Constraint.fit.increasing)},
    )
    y_fit, sample_weight, offset = model_build_design_matrix(
        model,
        frame,
        y,
        np.ones(n),
        None,
    )

    cache = {}
    result, _, fisher_gram, states = fit_irls_direct(
        model._dm,
        y_fit,
        sample_weight,
        model._distribution,
        model._link,
        model._groups,
        {"x": 1.0},
        offset=offset,
        convergence="coefficients",
        tol=1.0e-9,
        return_xtwx=True,
        return_scop_state=True,
        cache_out=cache,
    )

    assert result.converged
    assert result.scop_geometry.curvature_source == "observed"
    from superglm.reml.penalty_algebra import build_penalty_matrix
    from superglm.reml.scop_efs import build_scop_penalty_components
    from superglm.reml.scop_geometry import (
        build_cached_scop_joint_geometry,
        build_observed_scop_joint_geometry,
    )

    penalty = build_penalty_matrix(
        model._dm.group_matrices,
        model._groups,
        {"x": 1.0},
        model._dm.p,
        reml_penalties=build_scop_penalty_components(states),
    )
    offset_arr = np.zeros(n) if offset is None else np.asarray(offset, dtype=float)
    expected = build_observed_scop_joint_geometry(
        dm=model._dm,
        distribution=model._distribution,
        link=model._link,
        y=y_fit,
        sample_weight=sample_weight,
        offset_arr=offset_arr,
        result=result,
        penalty=penalty,
        scop_states=states,
        fisher_XtWX=fisher_gram,
        fisher_XtW1=result.rank_info.sum_w * result.rank_info.mean_x,
        fisher_sum_W=result.rank_info.sum_w,
        centered_fisher_gram=cache["centered_XtWX"],
        fisher_mean_x=cache["mean_x"],
    )
    cached_fisher = build_cached_scop_joint_geometry(
        raw_fisher_gram=fisher_gram,
        centered_fisher_gram=cache["centered_XtWX"],
        fisher_xtw=result.rank_info.sum_w * result.rank_info.mean_x,
        fisher_mean_x=cache["mean_x"],
        fisher_sum_w=result.rank_info.sum_w,
        latent_penalty=penalty,
        scop_states=states,
    )
    np.testing.assert_allclose(
        result.scop_geometry.centered_hessian,
        expected.centered_hessian,
        rtol=2e-12,
        atol=2e-12,
    )
    assert not np.allclose(
        result.scop_geometry.centered_hessian,
        cached_fisher.centered_hessian,
        rtol=1e-5,
        atol=1e-7,
    )


def test_gamma_log_latent_hessian_matches_finite_difference_with_intercept():
    """Pya--Wood's chain rule and intercept Schur term hold for Gamma/log."""
    from superglm.reml.scop_geometry import assemble_observed_scop_hessian

    rng = np.random.default_rng(20260718)
    n = 80
    ordinary = rng.normal(size=n)
    basis = np.abs(rng.normal(size=(n, 2))) + 0.2
    weights = rng.uniform(0.5, 2.0, size=n)
    theta = np.array([0.3, -0.25, np.log(0.8), np.log(1.2)])
    penalty_block = 0.7 * np.array([[1.0, -1.0], [-1.0, 1.0]])
    latent_slice = slice(1, 3)

    def gradient(parameters):
        intercept, coefficient = parameters[:2]
        beta_eff = parameters[2:]
        jacobian = np.exp(beta_eff)
        eta = intercept + ordinary * coefficient + basis @ jacobian
        mu = np.exp(eta)
        # Negative unit-dispersion Gamma/log score with frequency weights.
        negative_score_eta = weights * (1.0 - y / mu)
        return np.concatenate(
            (
                [np.sum(negative_score_eta)],
                [ordinary @ negative_score_eta],
                jacobian * (basis.T @ negative_score_eta) + penalty_block @ beta_eff,
            )
        )

    eta = theta[0] + ordinary * theta[1] + basis @ np.exp(theta[2:])
    mu = np.exp(eta)
    y = mu * rng.lognormal(mean=-0.03, sigma=0.18, size=n)
    slopes = np.column_stack((ordinary, basis))
    observed_weights = weights * y / mu
    negative_score_eta = weights * (1.0 - y / mu)
    raw_observed_gram = slopes.T @ (observed_weights[:, None] * slopes)
    raw_negative_score = slopes.T @ negative_score_eta
    penalty = np.zeros((3, 3))
    penalty[latent_slice, latent_slice] = penalty_block
    states = {
        0: {
            "group_sl": latent_slice,
            "group_name": "mono",
            "beta_eff": theta[2:],
        }
    }

    actual, transformed_cross = assemble_observed_scop_hessian(
        raw_observed_gram=raw_observed_gram,
        raw_negative_score=raw_negative_score,
        penalty=penalty,
        scop_states=states,
        XtW1=slopes.T @ observed_weights,
        sum_W=float(np.sum(observed_weights)),
    )

    epsilon = 2.0e-5
    numerical = np.empty((theta.size, theta.size))
    for column in range(theta.size):
        step = np.zeros_like(theta)
        step[column] = epsilon
        numerical[:, column] = (gradient(theta + step) - gradient(theta - step)) / (2.0 * epsilon)
    numerical = 0.5 * (numerical + numerical.T)
    expected_cross = numerical[1:, 0]
    expected_centered = (
        numerical[1:, 1:] - np.outer(expected_cross, expected_cross) / numerical[0, 0]
    )

    np.testing.assert_allclose(transformed_cross, expected_cross, rtol=2e-9, atol=2e-9)
    np.testing.assert_allclose(actual, expected_centered, rtol=2e-8, atol=2e-8)


def test_curvature_mixed_map_observed_hessian_matches_finite_difference():
    """Observed REML keeps zero map curvature on the free affine slope."""
    from superglm.reml.scop_geometry import assemble_observed_scop_hessian

    rng = np.random.default_rng(217)
    n = 90
    reparam = _curvature_solver_reparam(7, "concave")
    basis = rng.normal(size=(n, reparam.q))
    weights = rng.uniform(0.5, 2.0, size=n)
    beta_eff = np.array([-0.45, np.log(0.7), np.log(1.1), -0.2, 0.15, -0.35])
    intercept = 0.2
    penalty = 0.3 * reparam.penalty_matrix()
    theta = np.concatenate(([intercept], beta_eff))

    retained_eta = intercept + basis @ reparam.forward(beta_eff)
    retained_mu = np.exp(retained_eta)
    y = retained_mu * rng.lognormal(mean=-0.03, sigma=0.18, size=n)

    def gradient(parameters):
        latent = parameters[1:]
        mapped = reparam.forward(latent)
        eta = parameters[0] + basis @ mapped
        mu = np.exp(eta)
        negative_score_eta = weights * (1.0 - y / mu)
        return np.concatenate(
            (
                [np.sum(negative_score_eta)],
                reparam.jacobian_diagonal(latent) * (basis.T @ negative_score_eta)
                + penalty @ latent,
            )
        )

    observed_weights = weights * y / retained_mu
    negative_score_eta = weights * (1.0 - y / retained_mu)
    raw_observed_gram = basis.T @ (observed_weights[:, None] * basis)
    raw_negative_score = basis.T @ negative_score_eta
    states = {
        0: {
            "group_sl": slice(0, reparam.q),
            "group_name": "concave_x",
            "beta_eff": beta_eff,
            "reparam": reparam,
        }
    }
    actual, transformed_cross = assemble_observed_scop_hessian(
        raw_observed_gram=raw_observed_gram,
        raw_negative_score=raw_negative_score,
        penalty=penalty,
        scop_states=states,
        XtW1=basis.T @ observed_weights,
        sum_W=float(np.sum(observed_weights)),
    )

    epsilon = 2e-5
    eye = np.eye(theta.size)
    numerical = np.column_stack(
        [
            (gradient(theta + epsilon * step) - gradient(theta - epsilon * step)) / (2.0 * epsilon)
            for step in eye
        ]
    )
    numerical = 0.5 * (numerical + numerical.T)
    expected_cross = numerical[1:, 0]
    expected_centered = (
        numerical[1:, 1:] - np.outer(expected_cross, expected_cross) / numerical[0, 0]
    )

    assert abs(raw_negative_score[0]) > 0.1
    np.testing.assert_allclose(transformed_cross, expected_cross, rtol=2e-8, atol=1e-8)
    np.testing.assert_allclose(actual, expected_centered, rtol=3e-8, atol=3e-8)


@pytest.mark.slow
def test_gamma_log_model_geometry_matches_dense_latent_oracle():
    """The production row reductions reproduce the full dense Gamma/log Hessian."""
    from superglm import Constraint, SuperGLM
    from superglm.distributions import Gamma
    from superglm.features.spline import PSpline
    from superglm.model.base import model_build_design_matrix
    from superglm.reml.penalty_algebra import build_penalty_matrix
    from superglm.reml.scop_efs import build_scop_penalty_components
    from superglm.reml.scop_geometry import build_observed_scop_joint_geometry
    from superglm.solvers.irls_direct import fit_irls_direct

    rng = np.random.default_rng(20260719)
    n = 180
    x = np.sort(rng.uniform(0.0, 1.0, size=n))
    mean = np.exp(-0.4 + 1.1 * x)
    y = mean * rng.gamma(shape=8.0, scale=1.0 / 8.0, size=n)
    frame = pd.DataFrame({"x": x})
    model = SuperGLM(
        family=Gamma(),
        selection_penalty=0.0,
        discrete=True,
        features={"x": PSpline(n_knots=7, constraint=Constraint.fit.increasing)},
    )
    y_fit, sample_weight, offset = model_build_design_matrix(
        model,
        frame,
        y,
        np.ones(n),
        None,
    )
    offset_arr = np.zeros(n) if offset is None else np.asarray(offset, dtype=np.float64)
    lambdas = {"x": 1.3}
    result, _, fisher_xtwx, states = fit_irls_direct(
        X=model._dm,
        y=y_fit,
        weights=sample_weight,
        family=model._distribution,
        link=model._link,
        groups=model._groups,
        lambda2=lambdas,
        offset=offset_arr,
        return_xtwx=True,
        return_scop_state=True,
    )
    penalties = build_scop_penalty_components(states)
    penalty = build_penalty_matrix(
        model._dm.group_matrices,
        model._groups,
        lambdas,
        model._dm.p,
        reml_penalties=penalties,
    )
    rank_info = result.rank_info
    assert rank_info is not None
    geometry = build_observed_scop_joint_geometry(
        dm=model._dm,
        distribution=model._distribution,
        link=model._link,
        y=y_fit,
        sample_weight=sample_weight,
        offset_arr=offset_arr,
        result=result,
        penalty=penalty,
        scop_states=states,
        fisher_XtWX=fisher_xtwx,
        fisher_XtW1=rank_info.sum_w * rank_info.mean_x,
        fisher_sum_W=rank_info.sum_w,
    )

    state = next(iter(states.values()))
    group_slice = state["group_sl"]
    assert group_slice == slice(0, model._dm.p)
    beta_eff = np.asarray(state["beta_eff"], dtype=np.float64)
    theta = np.concatenate(([result.intercept], beta_eff))

    def gradient(parameters):
        intercept = parameters[0]
        latent = parameters[1:]
        mapped = np.exp(latent)
        eta = intercept + model._dm.matvec(mapped) + offset_arr
        mu = np.exp(eta)
        negative_score_eta = sample_weight * (1.0 - y_fit / mu)
        raw_score = model._dm.rmatvec(negative_score_eta)
        return np.concatenate(([np.sum(negative_score_eta)], mapped * raw_score + penalty @ latent))

    epsilon = 1.0e-5
    numerical = np.empty((theta.size, theta.size))
    for column in range(theta.size):
        step = np.zeros_like(theta)
        step[column] = epsilon
        numerical[:, column] = (gradient(theta + step) - gradient(theta - step)) / (2.0 * epsilon)
    numerical = 0.5 * (numerical + numerical.T)
    expected = numerical[1:, 1:] - np.outer(numerical[1:, 0], numerical[0, 1:]) / numerical[0, 0]

    assert geometry.curvature_source == "observed"
    np.testing.assert_allclose(geometry.centered_hessian, expected, rtol=2e-7, atol=2e-7)
    assert geometry.log_det_H == pytest.approx(
        np.linalg.slogdet(numerical)[1],
        rel=2e-7,
        abs=2e-7,
    )


@pytest.mark.slow
def test_gamma_log_scop_reml_installs_observed_latent_geometry():
    """The public SCOP optimizer must retain the observed, not Fisher, determinant."""
    from superglm import Constraint, SuperGLM
    from superglm.distributions import Gamma
    from superglm.features.spline import PSpline
    from superglm.reml.penalty_algebra import build_penalty_matrix
    from superglm.reml.scop_geometry import build_observed_scop_joint_geometry

    rng = np.random.default_rng(20260720)
    n = 220
    x = np.sort(rng.uniform(0.0, 1.0, size=n))
    mean = np.exp(-0.25 + 0.9 * x)
    y = mean * rng.gamma(shape=10.0, scale=0.1, size=n)
    frame = pd.DataFrame({"x": x})
    model = SuperGLM(
        family=Gamma(),
        selection_penalty=0.0,
        discrete=True,
        features={"x": PSpline(n_knots=7, constraint=Constraint.fit.increasing)},
    )

    model.fit_reml(frame, y, max_reml_iter=3, max_pirls_iter=100)

    reml_result = model._reml_result
    result = model._solver_result
    states = reml_result.scop_states
    assert states
    assert reml_result.curvature_source == "observed"
    penalty = build_penalty_matrix(
        model._dm.group_matrices,
        model._groups,
        reml_result.lambdas,
        model._dm.p,
        reml_penalties=reml_result.reml_penalties,
    )
    rank_info = result.rank_info
    assert rank_info is not None
    geometry = build_observed_scop_joint_geometry(
        dm=model._dm,
        distribution=model._distribution,
        link=model._link,
        y=y,
        sample_weight=np.ones(n),
        offset_arr=np.zeros(n),
        result=result,
        penalty=penalty,
        scop_states=states,
        fisher_XtWX=model._dm.execution_plan.moments(np.ones(n)).gram,
        fisher_XtW1=rank_info.sum_w * rank_info.mean_x,
        fisher_sum_W=rank_info.sum_w,
    )
    assert result.log_det_H == pytest.approx(geometry.log_det_H, rel=2e-9, abs=2e-9)
    assert result.reml_hessian_rank == geometry.hessian_rank
