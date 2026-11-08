from importlib.util import find_spec

import numpy as np
import pandas as pd
import pytest

from superglm.distributional.family import COMPLETE_OBSERVATION, VarianceFamily
from superglm.distributional.solver.derivatives import transform_natural_derivatives
from superglm.distributional.weights import (
    LikelihoodWeightError,
    UnsupportedLikelihoodContractError,
    WeightContract,
    resolve_likelihood_weights,
)
from superglm.links import IdentityLink, LogLink

from ._dirichlet_oracle import mp_curvature_direction, mp_row


def family(k=3, components=None):
    assert find_spec("superglm.distributional.families.dirichlet") is not None, (
        "Task 5 requires the concrete Dirichlet family"
    )
    from superglm.distributional.families.dirichlet import DirichletLSS

    return DirichletLSS(
        components=tuple(f"share {j}" for j in range(k)) if components is None else components
    )


def bind(fam, y, weights=None, semantics="frequency"):
    weights = resolve_likelihood_weights(
        weights, n_observations=len(y), contract=WeightContract(semantics=semantics)
    )
    return fam.bind_likelihood(y, weights, COMPLETE_OBSERVATION)


@pytest.mark.parametrize("k", [2, 3, 5])
def test_component_order_plan_identity_and_takes(k):
    fam = family(k)
    y = np.random.default_rng(k).dirichlet(np.arange(1, k + 1), 5)
    frame = pd.DataFrame(y, columns=fam.components)
    plan = bind(fam, frame[frame.columns[::-1]])
    canonical = fam.validate_likelihood_plan(y, plan)
    np.testing.assert_array_equal(canonical, y)
    assert not canonical.flags.writeable
    with pytest.raises(ValueError):
        canonical.setflags(write=True)
    changed = y.copy()
    changed[0, 0] = np.nextafter(changed[0, 0], 1)
    with pytest.raises(UnsupportedLikelihoodContractError, match="response|identity"):
        fam.validate_likelihood_plan(changed, plan)
    other = family(k, tuple(reversed(fam.components)))
    with pytest.raises(
        UnsupportedLikelihoodContractError, match="component|configuration|identity"
    ):
        other.validate_likelihood_plan(y, plan)
    indices = np.array([4, 1, 0])
    child = plan.take(indices)
    np.testing.assert_array_equal(child.weights.input_positions, indices)
    np.testing.assert_array_equal(fam.validate_likelihood_plan(y[indices], child), y[indices])
    assert plan.plan_identifier == bind(fam, y.copy()).plan_identifier
    assert plan.plan_identifier != bind(fam, changed).plan_identifier
    assert not isinstance(fam, VarianceFamily)
    assert list(fam.component_parameters.values()) == [f"alpha_{j}" for j in range(k)]


def test_duplicate_take_preserves_shared_nonoverlap_contract():
    fam = family()
    y = np.array([[0.2, 0.3, 0.5], [0.6, 0.25, 0.15]])
    plan = bind(fam, y)
    indices = np.array([1, 0, 1])
    identity = plan.plan_identifier
    with pytest.raises(LikelihoodWeightError, match="duplicates"):
        plan.take(indices)
    assert plan.plan_identifier == identity
    np.testing.assert_array_equal(fam.validate_likelihood_plan(y, plan), y)


@pytest.mark.parametrize("identity", [False, True])
def test_linked_scores_fisher_and_both_curvature_directions(identity):
    fam = family()
    y = np.array([[0.2, 0.3, 0.5], [0.6, 0.25, 0.15]])
    alpha = np.array([[0.7, 2.1, 4.2], [4.0, 1.1, 0.5]])
    mass = np.array([2.0, 3.0])
    plan = bind(fam, y, mass)
    links = [IdentityLink() if identity else LogLink() for _ in range(3)]
    eta = alpha if identity else np.log(alpha)
    alpha = np.column_stack([link.inverse(eta[:, j]) for j, link in enumerate(links)])
    first = np.array([[0.2, -0.3, 0.5], [-0.1, 0.7, -0.2]])
    second = np.array([[-0.5, 0.1, 0.4], [0.8, -0.2, 0.1]])
    natural = fam.evaluate_natural(y, alpha, plan)
    linked = transform_natural_derivatives(natural, eta, links)
    fisher = fam.expected_information_natural(alpha, plan)
    for i in range(2):
        _, score, hessian = mp_row(y[i], alpha[i])
        np.testing.assert_allclose(
            linked.score_eta[i], score * mass[i] * (1 if identity else alpha[i]), rtol=3e-12
        )
        np.testing.assert_allclose(fisher[i], -hessian * mass[i], rtol=3e-12)
    actual1 = fam.predictor_curvature_directional_derivative(y, eta, first, links, plan)
    actual2 = fam.predictor_curvature_second_directional_derivative(
        y, eta, first, second, links, plan
    )
    for i in range(2):
        np.testing.assert_allclose(
            actual1[i],
            mass[i] * mp_curvature_direction(y[i], eta[i], first[i], identity=identity),
            rtol=3e-11,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            actual2[i],
            mass[i] * mp_curvature_direction(y[i], eta[i], first[i], second[i], identity=identity),
            rtol=3e-11,
            atol=1e-12,
        )


def test_frequency_replication_and_response_covariance():
    fam = family()
    y = np.array([[0.2, 0.3, 0.5], [0.6, 0.25, 0.15]])
    alpha = np.array([[2.0, 3.0, 5.0], [4.0, 1.0, 2.0]])
    counts = np.array([2, 3])
    weighted = fam.evaluate_natural(y, alpha, bind(fam, y, counts))
    expanded_y, expanded_a = np.repeat(y, counts, axis=0), np.repeat(alpha, counts, axis=0)
    expanded = fam.evaluate_natural(expanded_y, expanded_a, bind(fam, expanded_y))
    for field in ("reported_log_likelihood", "score", "hessian_packed"):
        np.testing.assert_allclose(
            getattr(weighted, field).sum(axis=0), getattr(expanded, field).sum(axis=0)
        )
    mean = fam.default_prediction(alpha)
    covariance = fam.response_covariance(alpha)
    for i in range(2):
        mu = alpha[i] / alpha[i].sum()
        np.testing.assert_allclose(mean[i], mu)
        np.testing.assert_allclose(
            covariance[i], (np.diag(mu) - np.outer(mu, mu)) / (alpha[i].sum() + 1)
        )
        np.testing.assert_allclose(covariance[i] @ np.ones(3), 0, atol=2e-17)
    with pytest.raises(UnsupportedLikelihoodContractError, match="prior"):
        bind(fam, y, np.array([1.0, 2.0]), semantics="prior")


@pytest.mark.parametrize(
    "bad",
    [
        [[0.2, 0.3, 0.6]],
        [[0.0, 0.3, 0.7]],
        [[True, 0.2, 0.8]],
        [["0.2", 0.3, 0.5]],
        [[0.2 + 0j, 0.3, 0.5]],
        [[np.nan, 0.3, 0.7]],
    ],
)
def test_invalid_simplex_is_not_coerced_or_repaired(bad):
    with pytest.raises(ValueError):
        bind(family(), bad)


@pytest.mark.parametrize("components", [("a",), ["a", "b"], ("a", "a"), ("", "b"), ("a", 2)])
def test_component_configuration_is_explicit_and_immutable(components):
    with pytest.raises(ValueError):
        family(components=components)


@pytest.mark.parametrize("k", [3, 5])
def test_row_varying_likelihood_matches_scipy_joint_density(k):
    from scipy import stats

    fam = family(k)
    rng = np.random.default_rng(190 + k)
    alpha = np.exp(rng.normal(size=(5, k)))
    y = np.array([rng.dirichlet(a) for a in alpha])
    got = fam.evaluate_natural(y, alpha, bind(fam, y))
    want = np.array([stats.dirichlet.logpdf(row, a) for row, a in zip(y, alpha, strict=True)])
    np.testing.assert_allclose(got.reported_log_likelihood, want, rtol=5e-13, atol=3e-14)


@pytest.mark.parametrize(
    "alpha",
    [
        np.array([1e-5, 0.2, 5.0]),
        np.array([2e10, 3e10, 5e10]),
        np.array([4.0, 2.0**-50, 2.0**-50]),
        np.array([0.3, 0.7, 1.2, 2.3, 5.0]),
    ],
)
def test_high_and_unequal_shapes_stacked_curvature_directions(alpha):
    k = len(alpha)
    fam = family(k)
    eta = np.log(alpha)[None]
    theta = np.exp(eta)
    y = theta / theta.sum(axis=1, keepdims=True)
    a = np.linspace(-0.3, 0.5, k)[None]
    b = np.linspace(0.6, -0.2, k)[None]
    plan = bind(fam, y)
    links = [LogLink()] * k
    first = fam.predictor_curvature_directional_derivative(y, eta, np.stack([a, b]), links, plan)
    second = fam.predictor_curvature_second_directional_derivative(
        y, eta, np.stack([a, b]), np.stack([b, a]), links, plan
    )
    expected1 = mp_curvature_direction(y[0], eta[0], a[0])
    expected2 = mp_curvature_direction(y[0], eta[0], a[0], b[0])
    np.testing.assert_allclose(first[0, 0], expected1, rtol=2e-11, atol=1e-25)
    np.testing.assert_allclose(second[0, 0], expected2, rtol=2e-11, atol=1e-25)
    np.testing.assert_allclose(second[0], second[1], rtol=2e-12, atol=1e-25)


def test_optional_directions_refuse_nonrepresentable_movements_and_missing_link_derivatives():
    fam = family()
    y = np.array([[0.2, 0.3, 0.5]])
    eta = np.log([[2.0, 3.0, 5.0]])
    plan = bind(fam, y)
    with pytest.raises(UnsupportedLikelihoodContractError, match="optional"):
        fam.predictor_curvature_second_directional_derivative(
            y, eta, np.full((1, 3), 1e250), np.full((1, 3), 1e250), [LogLink()] * 3, plan
        )

    class IncompleteLog(LogLink):
        deriv4_inverse = None

    with pytest.raises(UnsupportedLikelihoodContractError, match="order four"):
        fam.predictor_curvature_second_directional_derivative(
            y, eta, np.ones((1, 3)), np.ones((1, 3)), [IncompleteLog()] * 3, plan
        )


def test_mean_range_refusals_and_finite_moment_starts():
    fam = family()
    for alpha in ([[1e308, 1e308, 1e308]], [[1e300, 1e-300, 1.0]]):
        with pytest.raises(ValueError, match="range"):
            fam.default_prediction(alpha)
    for y in (
        np.tile([0.2, 0.3, 0.5], (4, 1)),
        np.array([[1e-250, 0.3, 0.7], [0.2, 0.3, 0.5], [0.6, 0.3, 0.1]]),
    ):
        plan = bind(fam, y)
        initial = fam.initialize(y, plan).theta
        assert np.isfinite(initial).all() and (initial > 0).all()
        assert fam.evaluate_natural(y, initial, plan).valid.all()


def test_fisher_does_not_depend_on_response_range():
    fam = family()
    theta = np.array([[1e306, 1e306, 1e306]])
    ordinary = np.array([[0.2, 0.3, 0.5]])
    remote = np.array([[1e-250, 0.4, 0.6]])
    expected = fam.expected_information_natural(theta, bind(fam, ordinary))
    np.testing.assert_array_equal(
        fam.expected_information_natural(theta, bind(fam, remote)), expected
    )


@pytest.mark.parametrize(
    "alpha", [[1e12, 2e12, 3e12], [4.0, 2.0**-50, 2.0**-50], [1.0, 1e-250, 2e-250]]
)
def test_mean_and_response_covariance_projections_at_extreme_compositions(alpha):
    import mpmath as mp

    fam = family()
    alpha = np.array(alpha)
    contrast = np.array([1.0, -2.0, 0.75])
    with mp.workdps(350):
        a = list(map(mp.mpf, map(float, alpha)))
        total = mp.fsum(a)
        mu = [v / total for v in a]
        expected_mean = np.array(mu, dtype=float)
        covariance = mp.matrix(
            [
                [((mu[j] if j == k else 0) - mu[j] * mu[k]) / (total + 1) for k in range(3)]
                for j in range(3)
            ]
        )
        v = mp.matrix(list(map(mp.mpf, map(float, contrast))))
        expected_projection = float((v.T * covariance * v)[0])
    actual_mean = fam.default_prediction(alpha[None])[0]
    actual_covariance = fam.response_covariance(alpha[None])[0]
    np.testing.assert_allclose(actual_mean, expected_mean, rtol=8 * np.finfo(float).eps)
    np.testing.assert_allclose(
        contrast @ actual_covariance @ contrast, expected_projection, rtol=16 * np.finfo(float).eps
    )
