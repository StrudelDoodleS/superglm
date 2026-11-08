"""Core bounded two-piece laws and independent complete-fit references.

The oracle differentiates a high-precision normalized piecewise CDF, not the
implementation's jets. Away from a tiny tail difference, the gate is 8192 u
times each derivative order's scale (the normal primitive's rounding plus
the two coordinate maps). Narrow/remote cases state their conditioning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import optimize

from superglm import Spline
from superglm.distributional import Predictor
from superglm.distributional.families.two_piece import TwoPieceLogNormalLSS, TwoPieceNormalLSS
from superglm.distributional.family import COMPLETE_OBSERVATION, bounded_observation
from superglm.distributional.observation import ResponseBounds
from superglm.distributional.weights import (
    UnsupportedLikelihoodContractError,
    WeightContract,
    resolve_likelihood_weights,
)
from tests._two_piece_observation_oracle import (
    bounded_sample,
    complex_gradient,
    linked_loglik,
    mp_contribution,
    mp_jet,
    mp_location,
    reference_hessian,
)
from tests.bound_predictor_fixtures import model_from_templates

CASES = [("identity", "location"), ("log", "location"), ("log", "mean")]
U = 2.0**-53


def family_for(transform, parametrisation):
    return (
        TwoPieceNormalLSS()
        if transform == "identity"
        else TwoPieceLogNormalLSS(parametrisation=parametrisation)
    )


def weights(mass, semantics="frequency"):
    mass = np.asarray(mass, dtype=float)
    return resolve_likelihood_weights(
        mass, n_observations=len(mass), contract=WeightContract(semantics=semantics)
    )


def bind(family, y, lo, hi, side, mass=None):
    bounds = ResponseBounds.from_arrays(np.asarray(lo), np.asarray(hi), np.asarray(side))
    return family.bind_likelihood(
        y, weights(np.ones(len(y)) if mass is None else mass), bounded_observation(bounds)
    )


def assert_jet(actual, expected, factor=8192):
    for got, want in zip(actual, expected, strict=True):
        scale = max(1.0, float(np.max(np.abs(want))))
        np.testing.assert_allclose(got, want, rtol=0, atol=factor * U * scale)


@pytest.mark.parametrize(("transform", "parametrisation"), CASES)
@pytest.mark.parametrize("skew", [-0.65, 0.55])
def test_bounded_natural_channels_match_normalized_oracle(transform, parametrisation, skew):
    family = family_for(transform, parametrisation)
    theta = np.array([1.6 if parametrisation == "mean" else 0.0, 0.7, skew])
    mu = float(mp_location(*theta, parametrisation))
    # Exact, left, right; one- and two-sided truncation; join endpoints and crossing.
    lo = mu + np.array([-np.inf, -2.0, -0.8, -2.0, 0.0, -2.0, -np.inf, -0.8])
    hi = mu + np.array([np.inf, 0.0, 2.0, 1.0, 2.0, -0.1, 1.5, np.inf])
    y = mu + np.array([0.3, -0.3, 0.0, 0.2, 0.4, -0.8, 0.3, 0.5])
    side = np.array([0, -1, 1, 0, 1, -1, -1, 0])
    if transform == "log":
        lo, hi, y = np.exp(lo), np.exp(hi), np.exp(y)
    mass = np.array([1, 2, 3, 1, 2, 1, 3, 1])
    plan = bind(family, y, lo, hi, side, mass)
    assert np.array_equal(plan.parameter_independent_carrier, np.zeros(len(y)))
    evaluated = family.evaluate_natural(y, np.tile(theta, (len(y), 1)), plan)
    assert np.all(evaluated.valid)
    for row in range(len(y)):
        expected = mp_jet(
            lambda a, b, c: mp_contribution(
                y[row], lo[row], hi[row], side[row], (a, b, c), transform, parametrisation
            ),
            theta,
        )
        assert_jet(
            (
                evaluated.reported_log_likelihood[row],
                evaluated.score[row],
                evaluated.hessian_packed[row],
            ),
            tuple(mass[row] * channel for channel in expected),
        )
    for order in (0, 1):
        lower_order = family.evaluate_natural(
            y, np.tile(theta, (len(y), 1)), plan, derivative_order=order
        )
        assert_jet((lower_order.reported_log_likelihood,), (evaluated.reported_log_likelihood,))
        assert lower_order.hessian_packed is None
        if order:
            assert_jet((lower_order.score,), (evaluated.score,))
        else:
            assert lower_order.score is None


@pytest.mark.parametrize(("transform", "parametrisation"), CASES)
def test_plan_bounds_identity_take_and_complete_compatibility(transform, parametrisation):
    family = family_for(transform, parametrisation)
    y = np.array([0.5, 1.0, 2.0])
    lo, hi, side = np.zeros(3), np.full(3, 3.0), np.array([0, -1, 1])
    plan = bind(family, y, lo, hi, side, [1, 2, 3])
    other = bind(family, y, lo, hi + 1.0, side, [1, 2, 3])
    assert plan.plan_identifier != other.plan_identifier
    take = np.array([2, 0, 1])
    sliced = plan.take(take)
    assert np.array_equal(sliced.observation.bounds.side, side[take])
    assert np.array_equal(sliced.weights.input_positions, plan.weights.input_positions[take])
    assert np.array_equal(family.validate_likelihood_plan(y[take], sliced), y[take])
    assert not sliced.exact_response.flags.writeable
    theta = family.initialize(y, plan).theta
    original = family.evaluate_natural(y, theta, plan)
    subset = family.evaluate_natural(y[take], theta[take], sliced)
    np.testing.assert_allclose(
        subset.reported_log_likelihood, original.reported_log_likelihood[take]
    )
    complete = family.bind_likelihood(y, weights(np.ones(3)), COMPLETE_OBSERVATION)
    assert "/v2:" in complete.plan_identifier
    assert np.any(complete.parameter_independent_carrier != 0.0)
    assert family.capabilities.censored_response
    assert not family.capabilities.continuous_curvature


@pytest.mark.parametrize(("transform", "parametrisation"), CASES)
@pytest.mark.parametrize("skew", [-0.899, -0.5, 0.5, 0.899])
def test_windows_join_crossing_remote_and_adjacent(transform, parametrisation, skew):
    from superglm.distributional.kernels._two_piece_windows import two_piece_window_jets
    from tests._two_piece_observation_oracle import mp_window

    theta = np.array([1.3 if parametrisation == "mean" else 0.0, 0.8, skew])
    mu = float(mp_location(*theta, parametrisation))
    # Include remote tails, fixed endpoints exactly at a location join, and
    # crossings whose half-tail fluxes almost cancel.
    offsets = [
        (-1.2, -0.2),
        (0.0, 0.7),
        (-0.7, 0.0),
        (-0.5, 0.9),
        (-1e-7, 2e-7),
        (12.0, 12.02),
        (-14.0, -13.99),
    ]
    lo = np.array([mu + a for a, _ in offsets])
    hi = np.array([mu + b for _, b in offsets])
    if transform == "log":
        lo, hi = np.exp(lo), np.exp(hi)
    lo = np.append(lo, 3.0)
    hi = np.append(hi, np.nextafter(3.0, np.inf))
    actual = two_piece_window_jets(
        lo,
        hi,
        np.tile(theta, (len(lo), 1)),
        order=2,
        response_transform=transform,
        parametrisation=parametrisation,
    )
    for row in range(len(lo)):
        import mpmath as mp

        reference = mp_jet(
            lambda a, b, c: mp.log(
                mp_window(lo[row], hi[row], (a, b, c), transform, parametrisation)
            ),
            theta,
        )
        # For a far same-side interval the differenced tail score carries
        # normal-tail rounding amplified by inverse log-tail separation.
        assert_jet(tuple(channel[row] for channel in actual[:2]), reference[:2], factor=65536)
        # A crossing integrates a discontinuous density Hessian. An O(u)
        # location/log-endpoint rounding shifts the fraction on either side
        # by O(u / width), amplified by the curvature jump. This is the
        # conditioning of the observable, absent from same-side windows.
        factor = 65536
        if row == 4 and parametrisation == "mean":
            factor = max(factor, 64 * (1 + abs(mu)) / (3e-7))
        assert_jet((actual[2][row],), (reference[2],), factor=factor)
    family = family_for(transform, parametrisation)
    assert_jet((family.log_window_probability(lo, hi, np.tile(theta, (len(lo), 1))),), (actual[0],))


@pytest.mark.parametrize(("transform", "parametrisation"), CASES)
def test_strongly_conditioned_narrow_rows_keep_full_contributions(transform, parametrisation):
    # The log density and normalizing log mass are about -1e16, while the
    # answer and its score are O(1). Subtracting completed jets loses them.
    family = family_for(transform, parametrisation)
    lo = np.array([1.0, 1.0])
    hi = np.array([np.nextafter(1.0, np.inf)] * 2)
    y = hi.copy()
    side = np.array([0, -1])  # the second row observes its whole conditional event
    theta = np.tile([-1e8 if parametrisation == "location" else 1.0, 0.8, 0.4], (2, 1))
    if parametrisation == "mean":
        # A large log mean, still finite, gives a strongly conditioned log law.
        # Keep natural Hessian storage resolvable. The linked-consumer
        # regression below separately covers loss beyond this range.
        theta[:, 0], theta[:, 1] = 1e100, 0.02
    plan = bind(family, y, lo, hi, side)
    actual = family.evaluate_natural(y, theta, plan)
    assert np.all(actual.valid)
    for row in range(2):
        expected = mp_jet(
            lambda a, b, c: mp_contribution(
                y[row], lo[row], hi[row], side[row], (a, b, c), transform, parametrisation
            ),
            theta[row],
            dps=90,
        )
        assert_jet(
            (actual.reported_log_likelihood[row], actual.score[row], actual.hessian_packed[row]),
            expected,
            factor=32768,
        )


@pytest.mark.parametrize(("transform", "parametrisation"), CASES)
def test_centered_partial_censoring_windows_are_normalized_once(transform, parametrisation):
    family = family_for(transform, parametrisation)
    lo, hi = np.ones(3), np.full(3, 1.0 + 1e-7)
    y, side, mass = lo + 0.4 * (hi - lo), np.array([0, -1, 1]), np.array([2, 3, 4])
    theta = np.tile([-1e4 if parametrisation == "location" else 1e20, 0.3, -0.5], (3, 1))
    plan = bind(family, y, lo, hi, side, mass)
    result = family.evaluate_natural(y, theta, plan)
    assert np.all(result.valid)
    for row in range(3):
        reference = mp_jet(
            lambda a, b, c: mp_contribution(
                y[row], lo[row], hi[row], side[row], (a, b, c), transform, parametrisation
            ),
            theta[row],
        )
        assert_jet(
            (result.reported_log_likelihood[row], result.score[row], result.hessian_packed[row]),
            tuple(mass[row] * part for part in reference),
            factor=32768,
        )


@pytest.mark.parametrize("parametrisation", ["location", "mean"])
def test_out_of_support_right_threshold_is_not_evaluated_as_a_density(parametrisation):
    family = TwoPieceLogNormalLSS(parametrisation=parametrisation)
    y = np.array([0.0, -2.0])
    plan = bind(family, y, [-np.inf] * 2, [np.inf] * 2, [1, 1])
    theta = np.tile([1.0, 0.8, -0.4], (2, 1))
    result = family.evaluate_natural(y, theta, plan)
    assert np.all(result.valid)
    for channel in (result.reported_log_likelihood, result.score, result.hessian_packed):
        assert np.count_nonzero(channel) == 0
    assert np.all(np.isfinite(family.initialize(y, plan).theta))
    with pytest.raises(ValueError, match="zero probability"):
        bind(family, y, [-np.inf] * 2, [np.inf] * 2, [-1, -1])


@pytest.mark.parametrize(("transform", "parametrisation"), CASES)
def test_bounded_prior_weights_are_explicitly_refused(transform, parametrisation):
    family = family_for(transform, parametrisation)
    y = np.array([0.5, 1.0])
    obs = bounded_observation(
        ResponseBounds.from_arrays(np.zeros(2), np.ones(2) * 3, np.array([0, 1]))
    )
    with pytest.raises(UnsupportedLikelihoodContractError, match="non-unit prior"):
        family.bind_likelihood(y, weights([1, 2], "prior"), obs)


@pytest.mark.parametrize(("transform", "parametrisation"), CASES)
def test_linked_derivatives_and_optional_refusal_use_bounded_curvature(transform, parametrisation):
    from superglm.distributional.smoothing.endpoint_direction import (
        analytic_curvature_direction,
        analytic_curvature_second_direction,
        finite_difference_curvature_direction,
    )
    from superglm.distributional.solver.derivatives import transform_natural_derivatives

    family = family_for(transform, parametrisation)
    y, lo, hi = np.array([0.2, 0.8, -0.2]), np.full(3, -1.4), np.full(3, 1.8)
    if transform == "log":
        y, lo, hi = np.exp(y), np.exp(lo), np.exp(hi)
    side = np.array([0, 1, -1])
    plan = bind(family, y, lo, hi, side)
    eta = np.tile([0.3, -0.5, -0.7], (3, 1))
    links = tuple(p.default_link for p in family.parameters)
    theta = np.column_stack([link.inverse(eta[:, k]) for k, link in enumerate(links)])
    evaluated = transform_natural_derivatives(family.evaluate_natural(y, theta, plan), eta, links)
    for row in range(3):

        def f(e):
            return linked_loglik(
                e[None],
                y[row : row + 1],
                lo[row : row + 1],
                hi[row : row + 1],
                side[row : row + 1],
                transform=transform,
                parametrisation=parametrisation,
            )[0]

        np.testing.assert_allclose(
            evaluated.score_eta[row], complex_gradient(f, eta[row]), atol=2e-12, rtol=2e-12
        )
        h = reference_hessian(f, eta[row])
        np.testing.assert_allclose(
            evaluated.hessian_eta_packed[row], h[np.triu_indices(3)], atol=2e-9, rtol=2e-9
        )
    direction = np.tile([0.2, -0.3, 0.1], (3, 1))
    calls = [
        lambda: family.expected_information_natural(theta, plan),
        lambda: family.predictor_curvature_directional_derivative(y, eta, direction, links, plan),
        lambda: family.predictor_curvature_second_directional_derivative(
            y, eta, direction, direction, links, plan
        ),
        lambda: family.certified_predictor_curvature_second_directional_derivative(
            y, eta, direction, direction, links, plan
        ),
    ]
    for call in calls:
        with pytest.raises(UnsupportedLikelihoodContractError, match="complete rows"):
            call()
    assert analytic_curvature_direction(family, y, eta, direction, links, plan) is None
    assert (
        analytic_curvature_second_direction(family, y, eta, direction, direction, links, plan)
        is None
    )
    numeric = finite_difference_curvature_direction(family, y, eta, direction, links, plan)
    finer = finite_difference_curvature_direction(
        family, y, eta, direction, links, plan, step=numeric.step / 2
    )
    assert numeric.evaluations == 4
    assert np.max(numeric.certificate) > 0.0
    assert np.all(
        np.abs(numeric.values - finer.values) <= numeric.certificate + finer.certificate + 1e-9
    )


def small_fit(transform, parametrisation, kind, *, mass=None, skew=0.35):
    family = family_for(transform, parametrisation)
    x, y, lo, hi, side = bounded_sample(varying=kind != "intercept", skew=skew)
    if transform == "log":
        y, lo, hi = np.exp(y), np.exp(lo), np.exp(hi)
    frame = pd.DataFrame({"x": x})
    names = [p.name for p in family.parameters]
    predictors = (
        Predictor(names[0], {} if kind == "intercept" else {"x": Spline(kind="cr", k=5)}),
        Predictor("scale", {}),
        Predictor("skew", {}),
    )
    model = model_from_templates(family=family, predictors=predictors, weight_semantics="frequency")
    args = dict(
        truncation=(lo, hi),
        censored=np.where(side > 0, "right", np.where(side < 0, "left", "none")),
        inner_tol=1e-10,
        max_inner_iter=150,
        sample_weight=mass,
    )
    if kind == "smoothing":
        model.fit_reml(frame, y, initial_lambda=1.0, max_reml_iter=30, **args)
    else:
        model.fit(
            frame, y, lambdas=None if kind == "intercept" else {f"{names[0]}:x#wiggle": 0.8}, **args
        )
    return model, frame, y, lo, hi, side


def check_fit_reference(model, frame, y, lo, hi, side, transform, parametrisation, mass=None):
    from superglm.distributional.solver.assembly import dense_predictor_matrices

    internal = model._model
    layout = internal.layout
    matrices = dense_predictor_matrices(layout)
    penalty = layout.penalty_matrix(internal.smoothing_parameters)
    counts = np.ones(len(y)) if mass is None else mass

    def predictors(beta):
        return np.column_stack(
            [
                matrix @ beta[state.coefficient_slice]
                for matrix, state in zip(matrices, layout.predictors, strict=True)
            ]
        )

    def objective(beta):
        ll = linked_loglik(
            predictors(beta), y, lo, hi, side, transform=transform, parametrisation=parametrisation
        )
        return (-np.dot(counts, ll) + 0.5 * beta @ penalty @ beta) / sum(counts)

    start = np.zeros(layout.n_coefficients)
    start[layout.predictors[1].coefficient_slice] = np.log(0.8 - 0.01)
    reference = optimize.minimize(
        objective,
        start,
        jac=lambda b: complex_gradient(objective, b),
        method="BFGS",
        options={"gtol": 1e-10, "maxiter": 400},
    )
    beta = internal.coefficients
    # Endpoint comparisons are conditional on independent stationarity at BOTH
    # points, not on either optimizer's success bit.
    g_ref, g_fit = complex_gradient(objective, reference.x), complex_gradient(objective, beta)
    assert np.linalg.norm(g_ref, np.inf) < 2e-7, (reference.message, g_ref)
    assert np.linalg.norm(g_fit, np.inf) < 2e-7, (
        internal.fit_state.solver_result.convergence_reason,
        g_fit,
    )
    h = reference_hessian(objective, reference.x)
    minimum = np.linalg.eigvalsh(h)[0]
    assert minimum > 0
    # Strong-convexity local endpoint bound, with margin for Hessian variation
    # and independently differenced curvature. Coefficient forward comparisons
    # are only made on this well-conditioned fixture.
    tolerance = 4 * (np.linalg.norm(g_ref) + np.linalg.norm(g_fit)) / minimum + 256 * np.sqrt(U)
    assert np.linalg.norm(beta - reference.x) <= tolerance
    assert abs(objective(beta) - objective(reference.x)) < 1e-10
    ll = np.dot(
        counts,
        linked_loglik(
            predictors(beta), y, lo, hi, side, transform=transform, parametrisation=parametrisation
        ),
    )
    np.testing.assert_allclose(
        internal.fit_state.solver_result.log_likelihood, ll, atol=2e-10, rtol=0
    )
    np.testing.assert_allclose(
        predictors(beta), predictors(reference.x), atol=5 * tolerance, rtol=0
    )
    eta = predictors(reference.x)
    scale, skew = 0.01 + np.exp(eta[:, 1]), 0.9 * np.tanh(eta[:, 2] / 2)
    if parametrisation == "mean":
        prediction = np.exp(eta[:, 0])
    elif transform == "identity":
        prediction = eta[:, 0] + 2 * scale * skew * np.sqrt(2 / np.pi)
    else:
        from scipy import special

        a, b = scale * (1 - skew), scale * (1 + skew)
        prediction = np.exp(eta[:, 0]) * (
            (1 - skew) * np.exp(a * a / 2) * special.ndtr(-a)
            + (1 + skew) * np.exp(b * b / 2) * special.ndtr(b)
        )
    np.testing.assert_allclose(
        model.predict(frame), prediction, atol=10 * tolerance, rtol=5 * tolerance
    )
    covariance = np.linalg.inv(h * sum(counts))
    projection = np.sin(np.arange(len(beta)) + 0.4)
    np.testing.assert_allclose(
        projection @ model.covariance_ @ projection,
        projection @ covariance @ projection,
        rtol=2e-5,
        atol=1e-10,
    )
    return objective, reference.x, h, predictors


@pytest.mark.parametrize(("transform", "parametrisation"), CASES)
@pytest.mark.parametrize("kind", ["intercept", "fixed", "smoothing"])
@pytest.mark.parametrize("skew", [-0.35, 0.35])
def test_small_public_fits_match_independent_normalized_optimizer(
    transform, parametrisation, kind, skew
):
    model, frame, y, lo, hi, side = small_fit(transform, parametrisation, kind, skew=skew)
    check_fit_reference(model, frame, y, lo, hi, side, transform, parametrisation)
    assert np.all(np.isfinite(model.predict(frame)))
    if kind == "smoothing":
        assert model.smoothing_convergence_reason_ is not None
        assert isinstance(model.smoothing_certified_, bool)


@pytest.mark.parametrize(("transform", "parametrisation"), CASES)
def test_integer_frequency_fits_equal_literal_bounded_replication(transform, parametrisation):
    mass = np.resize([1, 3, 2], 192)
    weighted, frame, y, lo, hi, side = small_fit(transform, parametrisation, "fixed", mass=mass)
    family = family_for(transform, parametrisation)
    templates = weighted.predictors_
    repeated = model_from_templates(
        family=family, predictors=templates, weight_semantics="frequency"
    )
    rows = np.repeat(np.arange(len(y)), mass)
    repeated.fit(
        frame.iloc[rows].reset_index(drop=True),
        y[rows],
        truncation=(lo[rows], hi[rows]),
        censored=np.where(side[rows] > 0, "right", np.where(side[rows] < 0, "left", "none")),
        lambdas=dict(weighted.smoothing_parameters_),
        inner_tol=1e-10,
        max_inner_iter=150,
    )
    check_fit_reference(weighted, frame, y, lo, hi, side, transform, parametrisation, mass)
    np.testing.assert_allclose(
        weighted.predict(frame), repeated.predict(frame), rtol=2e-7, atol=2e-8
    )
    np.testing.assert_allclose(weighted.covariance_, repeated.covariance_, rtol=2e-6, atol=2e-8)
    np.testing.assert_allclose(
        weighted._model.fit_state.solver_result.log_likelihood,
        repeated._model.fit_state.solver_result.log_likelihood,
        rtol=0,
        atol=1e-8,
    )


@pytest.mark.parametrize(("transform", "parametrisation"), CASES)
@pytest.mark.parametrize("side", ["left", "right"])
def test_all_one_side_censored_fit_discloses_nonfinite_optimum(transform, parametrisation, side):
    from superglm import SeparationWarning

    family = family_for(transform, parametrisation)
    y = np.linspace(0.5, 1.5, 24)
    names = [p.name for p in family.parameters]
    model = model_from_templates(
        family=family, predictors=tuple(Predictor(name, {}) for name in names)
    )
    with pytest.warns(SeparationWarning, match="maximum-likelihood"):
        model.fit(pd.DataFrame({"one": np.ones(len(y))}), y, censored=side, max_inner_iter=12)


def test_unresolved_remote_conditional_subtraction_is_a_trial_failure():
    family = TwoPieceNormalLSS()
    y = np.array([1.0])
    plan = bind(family, y, [0.0], [np.inf], [0])
    # The exact mathematical log likelihood is representable, but evaluating
    # the two enormous terms separately cannot resolve its small remainder.
    result = family.evaluate_natural(y, np.array([[-1e8, 1.0, 0.3]]), plan)
    assert not result.valid[0]
    assert np.all(np.isfinite(result.reported_log_likelihood))


@pytest.mark.parametrize(
    ("lower", "upper", "response", "side", "scale"),
    [
        (0.5, 2.0, 1.5, -1, 0.8),
        (0.5, 2.0, 1.5, 0, 0.8),
        (1.0, 1.0 + 1e-7, 1.0 + 0.5e-7, -1, 0.8),
        (1.0, np.nextafter(1.0, np.inf), np.nextafter(1.0, np.inf), 0, 0.02),
    ],
)
@pytest.mark.parametrize("mean", [1e100, 1e200])
def test_mean_mapping_preserves_link_curvature_or_declines_storage_loss(
    lower, upper, response, side, scale, mean
):
    from superglm.distributional.kernels.two_piece import location_of_mean
    from superglm.distributional.solver.derivatives import transform_natural_derivatives

    family = TwoPieceLogNormalLSS(parametrisation="mean")
    y, theta = np.array([response]), np.array([[mean, scale, 0.3]])
    plan = bind(family, y, [lower], [upper], [side])
    location_family = TwoPieceLogNormalLSS(parametrisation="location")
    location_theta = theta.copy()
    location_theta[:, 0] = location_of_mean(*theta.T)
    location = location_family.evaluate_natural(
        y, location_theta, bind(location_family, y, [lower], [upper], [side])
    )
    assert location.valid[0]
    for order in (0, 1):
        result = family.evaluate_natural(y, theta, plan, derivative_order=order)
        assert result.valid[0]
        np.testing.assert_allclose(
            result.reported_log_likelihood, location.reported_log_likelihood, atol=1e-10, rtol=0
        )
        if order == 1:
            np.testing.assert_allclose(
                result.score[:, 0] * mean, location.score[:, 0], atol=1e-10, rtol=0
            )
    result = family.evaluate_natural(y, theta, plan)
    links = tuple(parameter.default_link for parameter in family.parameters)
    eta = np.column_stack([link.link(theta[:, k]) for k, link in enumerate(links)])
    linked = transform_natural_derivatives(result, eta, links)
    if mean == 1e200:
        # The needed natural H_mm is below binary64 range, while its linked
        # contribution is meaningful. Zero must not authorize wrong curvature.
        assert not result.valid[0]
        assert not linked.valid[0]
        assert np.all(np.isfinite(result.hessian_packed))
    else:
        assert result.valid[0]
        assert linked.valid[0]
        np.testing.assert_allclose(
            linked.hessian_eta_packed[:, 0],
            location.hessian_packed[:, 0],
            atol=1e-10,
            rtol=0,
        )


def test_adjacent_log_window_with_unresolved_join_fraction_refuses_curvature():
    from superglm.distributional.kernels._two_piece_windows import two_piece_window_jets

    lo, hi = np.array([3.0]), np.array([np.nextafter(3.0, np.inf)])
    theta = np.array([[np.log(3.0), 0.8, 0.6]])
    jet = two_piece_window_jets(
        lo, hi, theta, order=2, response_transform="log", parametrisation="location"
    )
    # The log rounding is comparable to the entire interval; it cannot
    # establish what fraction lies on each side of a nonzero curvature jump.
    assert np.isfinite(jet[0][0])
    assert not np.all(np.isfinite(jet[2][0]))


def test_crossing_with_remote_lower_end_keeps_the_finite_upper_distance():
    import mpmath as mp

    from superglm.distributional.kernels._two_piece_windows import two_piece_window_jets
    from tests._two_piece_observation_oracle import mp_window

    theta = np.array([[0.0, 0.8, 0.4]])
    jet = two_piece_window_jets(
        np.array([-1e308]),
        np.array([2.0]),
        theta,
        order=2,
        response_transform="identity",
        parametrisation="location",
    )
    # The omitted tail below -100 is far below any binary64 channel. Using
    # the enormous span to reconstruct hi instead would round 2.0 to zero.
    expected = mp_jet(lambda a, b, c: mp.log(mp_window(-100.0, 2.0, (a, b, c))), theta[0])
    assert_jet(tuple(channel[0] for channel in jet), expected)


@pytest.mark.parametrize("skew", [-0.6, 0.6])
@pytest.mark.parametrize(("lo", "hi"), [(0.0, 0.7), (-0.7, 0.0)])
def test_zero_length_moving_half_keeps_its_boundary_flux(skew, lo, hi):
    import mpmath as mp

    from superglm.distributional.kernels._two_piece_windows import _crossing
    from tests._two_piece_observation_oracle import mp_window

    # Exercise the half-tail endpoint formula itself. A zero moving half has
    # zero mass but nonzero flux; deleting it from a log mixture is wrong.
    theta = np.array([0.0, 0.8, skew])
    jet = _crossing(
        np.array([lo]),
        np.array([hi]),
        np.array([hi - lo]),
        np.array([0.0]),
        np.array([0.8]),
        np.array([skew]),
        2,
    )
    reference = mp_jet(lambda a, b, c: mp.log(mp_window(lo, hi, (a, b, c))), theta)
    assert_jet(tuple(channel[0] for channel in jet), reference)


@pytest.mark.parametrize(("transform", "parametrisation"), CASES)
def test_smoothing_newton_fallback_matches_independent_local_profile(transform, parametrisation):
    from superglm.distributional.solver.assembly import dense_predictor_matrices

    model, frame, y, lo, hi, side = small_fit(transform, parametrisation, "smoothing")
    model.fit_reml(
        frame,
        y,
        truncation=(lo, hi),
        censored=np.where(side > 0, "right", np.where(side < 0, "left", "none")),
        initial_lambda=1.0,
        outer="efs+newton",
        inner_tol=1e-10,
        reml_tol=1e-9,
        practical_reml=False,
        max_reml_iter=30,
    )
    check_fit_reference(model, frame, y, lo, hi, side, transform, parametrisation)
    layout = model._model.layout
    matrices = dense_predictor_matrices(layout)
    name = layout.penalty_names[0]
    unit_penalty = layout.penalty_matrix({name: 1.0})
    eigenvalues = np.linalg.eigvalsh(unit_penalty)
    positive = eigenvalues[eigenvalues > 64 * U * np.max(eigenvalues)]

    def profile(rho):
        penalty = np.exp(rho) * unit_penalty

        def objective(beta):
            eta = np.column_stack(
                [
                    a @ beta[s.coefficient_slice]
                    for a, s in zip(matrices, layout.predictors, strict=True)
                ]
            )
            return (
                -sum(
                    linked_loglik(
                        eta, y, lo, hi, side, transform=transform, parametrisation=parametrisation
                    )
                )
                + 0.5 * beta @ penalty @ beta
            ) / len(y)

        start = np.zeros(layout.n_coefficients)
        start[layout.predictors[1].coefficient_slice] = np.log(0.7)
        fit = optimize.minimize(
            objective,
            start,
            jac=lambda b: complex_gradient(objective, b),
            method="BFGS",
            options={"gtol": 1e-10, "maxiter": 300},
        )
        # A profile derivative amplifies coefficient and Hessian noise again.
        # Refine the independent score to stationarity and use a wider, still
        # within-piece Richardson stencil instead of differencing its noise.
        beta = fit.x
        for _ in range(4):
            gradient = complex_gradient(objective, beta)
            hessian = reference_hessian(objective, beta, step=0.001)
            if np.linalg.norm(gradient, np.inf) < 1e-12:
                break
            beta = beta - np.linalg.solve(hessian, gradient)
        assert np.linalg.norm(complex_gradient(objective, beta), np.inf) < 1e-12
        information = reference_hessian(objective, beta, step=0.001) * len(y)
        sign, logdet = np.linalg.slogdet(information)
        assert sign == 1
        return (
            objective(beta) * len(y)
            + 0.5 * logdet
            - 0.5 * (np.log(positive).sum() + len(positive) * rho)
        )

    # This is a local smooth basin. Exact-row joins make the Laplace criterion
    # discontinuous farther away; neither EFS nor Newton promises a global MLE
    # of that nonregular criterion. Do not compare across its different basins.
    reference = optimize.minimize_scalar(
        profile, bounds=(-0.05, 0.10), method="bounded", options={"xatol": 2e-6}
    )
    fitted = np.log(model.smoothing_parameters_[name])
    step = 5e-4
    for point in (fitted, reference.x):
        assert abs((profile(point + step) - profile(point - step)) / (2 * step)) < 3e-5
    assert abs(fitted - reference.x) < 5e-4
    assert model.smoothing_convergence_reason_ == "stationary"
