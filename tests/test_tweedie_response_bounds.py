"""Atom-aware compound windows and actual normalized Tweedie fits."""

import numpy as np
import pytest

from superglm.distributional.families.tweedie import TweedieLSS
from superglm.distributional.family import bounded_observation
from superglm.distributional.observation import ResponseBounds
from superglm.distributional.weights import WeightContract, resolve_likelihood_weights
from tests._tweedie_window_oracle import density, gamma_window, log_jet, window


def _plan(y, lower, upper, side, weights=None, semantics="frequency"):
    n = len(y)
    resolved = resolve_likelihood_weights(
        np.ones(n) if weights is None else np.asarray(weights),
        n_observations=n,
        contract=WeightContract(semantics=semantics),
    )
    bounds = ResponseBounds.from_arrays(
        np.broadcast_to(lower, (n,)), np.broadcast_to(upper, (n,)), np.broadcast_to(side, (n,))
    )
    return TweedieLSS().bind_likelihood(np.asarray(y), resolved, bounded_observation(bounds))


def test_left_zero_and_right_zero_read_different_events():
    """Kills continuous support clipping and the complete-only binder."""
    family = TweedieLSS()
    y = np.array([0.0, 0.0])
    theta = np.array([[2.0, 1.3, 1.5]] * 2)
    plan = _plan(y, -np.inf, np.inf, [-1, 1])
    got = family.evaluate_natural(y, theta, plan)
    rate = 2 * np.sqrt(2) / 1.3
    np.testing.assert_allclose(got.optimizing_log_likelihood, [-rate, np.log(-np.expm1(-rate))])
    assert np.all(got.valid)
    assert (
        family.predictor_rows(y, plan, tuple(p.default_link for p in family.parameters), True)
        is None
    )


@pytest.mark.parametrize(
    "shape,x,upper", [(0.003, 0.1, True), (2.3, 3.9, False), (80.0, 90.0, True)]
)
def test_gamma_engine_scaled_partials_and_fixed_ratio_composition(shape, x, upper):
    """Kills factorial confusion and treating fixed-ratio coefficients as fixed-x."""
    import mpmath as mp

    from superglm.distributional.kernels._incomplete_gamma import (
        WORK_SIZE,
        log_gamma_tail_jet_into,
        log_gamma_tail_scaled_into,
    )

    scaled, taylor, along = (np.empty(6) for _ in range(3))
    work = np.empty(WORK_SIZE)
    assert log_gamma_tail_scaled_into(shape, x, upper, 2, scaled, work)
    assert log_gamma_tail_jet_into(shape, x, upper, 2, False, taylor, work)
    assert log_gamma_tail_jet_into(shape, x, upper, 2, True, along, work)
    np.testing.assert_allclose(scaled, taylor * [1, 1, 1, 2, 1, 2], rtol=2e-14)

    def probability(a, e):
        xx = a * mp.mpf(x) / mp.mpf(shape) * (1 + e)
        return (
            mp.gammainc(a, xx, mp.inf, regularized=True)
            if upper
            else mp.gammainc(a, 0, xx, regularized=True)
        )

    reference = np.concatenate([v.ravel() for v in log_jet(probability, (shape, 0.0))])
    np.testing.assert_allclose(along * [1, 1, 1, 2, 1, 2], reference, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize(
    "lower,upper,shape,log_scale",
    [
        (0.0, 0.8, 0.003, 0.2),
        (1.0, 1.0 + 1e-7, 2.3, -0.4),
        (0.0, 12.0, 10.0, 0.0),
        (80.0, 90.0, 80.0, 0.0),
    ],
)
def test_gamma_shape_scale_windows_match_independent_jet(lower, upper, shape, log_scale):
    from superglm.distributional.kernels._gamma_windows import gamma_shape_scale_window_jets

    expected = log_jet(lambda a, b: gamma_window(lower, upper, a, b), (shape, log_scale))
    got = gamma_shape_scale_window_jets([lower], [upper], [shape], [log_scale], order=2)
    for actual, reference in zip(got, expected, strict=True):
        np.testing.assert_allclose(actual, reference, rtol=2e-11, atol=2e-12)


def test_rare_window_grows_past_absolute_cdf_cutoff():
    """Kills reuse of the absolute 1e-12 CDF stopping rule for log windows."""
    from superglm.distributional.kernels._tweedie_windows import tweedie_window_jets

    theta = (1.0, 2.0, 1.5)  # rate=1, gamma shape=j and scale=1
    expected = log_jet(lambda *t: window(120.0, 120.1, *t), theta)
    assert expected[0][0] < np.log(1e-30)
    got = tweedie_window_jets([120.0], [120.1], [theta], [1.0], order=2)
    for actual, reference in zip(got, expected, strict=True):
        np.testing.assert_allclose(actual, reference, rtol=3e-10, atol=3e-10)


@pytest.mark.parametrize("semantics", ["prior", "frequency"])
def test_normalized_values_scores_hessians_match_independent_law(semantics):
    """Kills density dispatch bypass, lost atom and double frequency scaling."""
    family = TweedieLSS()
    y = np.array([0.0, 0.7, 1.1, 0.0])
    lower, upper, side = [-1.0, 0.0, 0.3, -1.0], [2.5, 3.0, 2.5, 0.0], [0, -1, 1, -1]
    weights = (
        np.array([2.0, 1.4, 1.0, 3.0]) if semantics == "prior" else np.array([2.0, 2.0, 1.0, 3.0])
    )
    theta = np.tile([1.2, 1.1, 1.45], (len(y), 1))
    plan = _plan(y, lower, upper, side, weights, semantics)
    got = family.evaluate_natural(y, theta, plan)
    assert np.all(got.valid)
    assert np.array_equal(got.parameter_independent_carrier, np.zeros(len(y)))
    for row in range(len(y)):
        w = weights[row] if semantics == "prior" else 1
        mass = weights[row] if semantics == "frequency" else 1

        def conditional(*t):
            denominator = window(lower[row], upper[row], *t, weight=w)
            if side[row] == 0:
                numerator = density(y[row], *t, weight=w)
            elif side[row] == -1:
                numerator = window(lower[row], y[row], *t, weight=w)
            else:
                numerator = window(y[row], upper[row], *t, weight=w)
            return numerator / denominator

        expected = log_jet(conditional, theta[row])
        for actual, reference in zip(
            (got.optimizing_log_likelihood, got.score, got.hessian_packed), expected, strict=True
        ):
            np.testing.assert_allclose(actual[row], mass * reference[0], rtol=5e-10, atol=5e-11)


def test_public_atomic_windows_and_plan_take():
    family = TweedieLSS()
    theta = np.tile([1.0, 2.0, 1.5], (5, 1))
    got = family.log_window_probability(
        [-np.inf, -2, 0, 0, -np.inf], [0, 0, np.inf, 0, np.inf], theta
    )
    np.testing.assert_allclose(got, [-1, -1, np.log(-np.expm1(-1)), -np.inf, 0])
    y = np.array([0.0, 0.0, 1.0])
    plan = _plan(y, [-1, -1, 0], [2, 3, 4], [-1, 1, 0])
    child = plan.take(np.array([2, 0]))
    np.testing.assert_array_equal(child.observation.bounds.lower, [0, -np.inf])
    assert child.plan_identifier != plan.plan_identifier
    links = tuple(p.default_link for p in family.parameters)
    from superglm.distributional.weights import UnsupportedLikelihoodContractError

    for name, args in (
        (
            "predictor_curvature_directional_derivative",
            (y, np.zeros((3, 3)), np.ones((3, 3)), links, plan),
        ),
        (
            "predictor_curvature_second_directional_derivative",
            (y, np.zeros((3, 3)), np.ones((3, 3)), np.ones((3, 3)), links, plan),
        ),
    ):
        with pytest.raises(UnsupportedLikelihoodContractError, match="complete"):
            getattr(family, name)(*args)


def test_bounded_plan_identity_retains_original_thresholds_and_take():
    y = np.array([0.0, 1.0, 2.0])
    first = _plan(y, -np.inf, np.inf, [-1, 1, 0])
    different = _plan([0.0, 1.1, 2.0], -np.inf, np.inf, [-1, 1, 0])
    assert first.plan_identifier != different.plan_identifier
    child = first.take(np.array([1, 0]))
    np.testing.assert_array_equal(child.response, [1.0, 0.0])
    y[0] = 7.0
    np.testing.assert_array_equal(first.response, [0.0, 1.0, 2.0])
    assert not first.response.flags.writeable
    with pytest.raises(ValueError):
        first.response.setflags(write=True)
    from superglm.distributional.weights import UnsupportedLikelihoodContractError

    with pytest.raises(UnsupportedLikelihoodContractError, match="match"):
        TweedieLSS().evaluate_natural([0.0, 1.1, 2.0], np.tile([1.0, 1.0, 1.5], (3, 1)), first)


@pytest.mark.parametrize("semantics", ["prior", "frequency"])
def test_small_intercept_fit_has_independent_stationary_normalized_endpoint(semantics):
    import pandas as pd
    from scipy import optimize, special

    from superglm import SuperLSS
    from tests._tweedie_window_oracle import fit_fixture, scipy_rows

    x, y, lower, upper, side, weights = fit_fixture(semantics)
    assert np.any(y == 0) and np.any(side == -1) and np.any(side == 1)
    family = TweedieLSS()
    model = SuperLSS(family, family.mu(), family.phi(), family.p(), weight_semantics=semantics)
    frame = pd.DataFrame({"x": x})
    labels = np.where(side == -1, "left", np.where(side == 1, "right", "none"))
    model.fit(
        frame, y, sample_weight=weights, truncation=(lower, upper), censored=labels, inner_tol=1e-9
    )
    fitted = model.predict_parameters(frame).to_numpy()[0]

    def theta(t):
        return np.array([np.exp(t[0]), np.exp(t[1]), 1.05 + 0.9 * special.expit(t[2])])

    def nll(t):
        return -np.sum(scipy_rows(y, lower, upper, side, theta(t), weights, semantics))

    fit_t = np.array(
        [np.log(fitted[0]), np.log(fitted[1]), special.logit((fitted[2] - 1.05) / 0.9)]
    )
    # Check independently before comparing endpoints: the reference must not
    # turn a nonstationary solver stop into an apparently matching fit.
    step = 1e-5
    gradient = np.array(
        [(nll(fit_t + d * step) - nll(fit_t - d * step)) / (2 * step) for d in np.eye(3)]
    )
    assert np.max(np.abs(gradient)) / (1 + nll(fit_t)) < 2e-8
    reference = optimize.minimize(
        nll,
        [np.log(1.8), np.log(1.4), 0],
        method="Nelder-Mead",
        options={"xatol": 2e-9, "fatol": 1e-12, "maxiter": 1200},
    )
    assert reference.success
    np.testing.assert_allclose(fitted, theta(reference.x), rtol=2e-6, atol=2e-7)
    assert model.result_.coefficient_converged
    assert model._require_fitted().result.execution_backend_identifier == "distributional-dense-v1"
    scores = model.scores(
        frame, y, which=("log",), sample_weight=weights, truncation=(lower, upper), censored=labels
    )
    np.testing.assert_allclose(
        scores["log"],
        -scipy_rows(y, lower, upper, side, fitted, weights, semantics),
        rtol=2e-10,
        atol=2e-11,
    )
    with pytest.raises(NotImplementedError, match="atom|point mass|bounded|truncated|censored"):
        model.residual_set(
            frame, y, sample_weight=weights, truncation=(lower, upper), censored=labels
        )


def test_cutoff_mutation_and_forced_work_refusals(monkeypatch):
    from superglm.distributional.kernels import _gamma_windows as gamma
    from superglm.distributional.kernels import _tweedie_windows as engine

    args = ([120.0], [120.1], [[1.0, 2.0, 1.5]], [1.0])
    work = engine.TweedieWindowWorkspace()
    reference = engine.tweedie_window_jets(*args, order=2, _workspace=work)
    assert work.largest_index > work.initial_largest_index
    with monkeypatch.context() as mutation:
        mutation.setattr(engine, "_admitted", lambda *args: True)
        wrong = engine.tweedie_window_jets(*args, order=2)
    assert abs(reference[0][0] - wrong[0][0]) > 1e-4
    assert np.max(np.abs(reference[2] - wrong[2])) > 0.1
    limited = engine.TweedieWindowWorkspace(gamma=gamma.GammaWindowWorkspace(max_evaluations=1))
    refused = engine.tweedie_window_jets(*args, order=2, _workspace=limited)
    assert np.isnan(refused[0][0])
    assert "budget" in limited.refusal_reason
    with monkeypatch.context() as mutation:
        mutation.setattr(gamma, "log_gamma_tail_jet_into", lambda *args: False)
        refused = engine.tweedie_window_jets([0.0], [1.0], [[1.0, 2.0, 1.5]], [1.0], order=2)
        assert np.isnan(refused[0][0])
        with pytest.raises(ValueError, match="refused"):
            TweedieLSS().log_window_probability([0.0], [1.0], [[1.0, 2.0, 1.5]])


@pytest.mark.parametrize(
    "p,rate,lo,hi",
    [(1.05001, 0.04, 0.0, 0.4), (1.94999, 5.0, 0.7, 0.700001), (1.5, 80.0, 0.8, 1.3)],
)
def test_power_walls_small_large_rates_and_order_coherence(p, rate, lo, hi):
    from superglm.distributional.kernels._tweedie_windows import tweedie_window_jets

    phi = 1 / (rate * (2 - p))
    theta = [[1.0, phi, p]]
    second = tweedie_window_jets([lo], [hi], theta, [1.0], order=2)
    assert all(np.isfinite(c).all() for c in second)
    # A left-censored row under a lower truncation gives the window by
    # multiplying back that truncation's mass; here a direct 240-term sum.
    from scipy import special

    j = np.arange(1.0, 241.0)
    a, b = (2 - p) / (p - 1), phi * (p - 1)
    probability = np.sum(
        np.exp(j * np.log(rate) - rate - special.gammaln(j + 1))
        * (special.gammainc(j * a, hi / b) - special.gammainc(j * a, lo / b))
    )
    np.testing.assert_allclose(second[0][0], np.log(probability), atol=3e-9)
    for order in (0, 1):
        lower = tweedie_window_jets([lo], [hi], theta, [1.0], order=order)
        for got, expected in zip(lower, second, strict=False):
            np.testing.assert_allclose(got, expected, atol=1e-10, rtol=1e-10)


def test_zero_weight_malformed_bounds_and_negative_censor_thresholds():
    import pandas as pd

    from superglm import SuperLSS

    family = TweedieLSS()
    y = np.array([-0.5, 0.0, 1.0, 2.0, 3.0, 4.0, np.nan])
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    plan = _plan(y[:-1], -np.inf, np.inf, [1, 0, 0, 0, 0, 0])
    got = family.evaluate_natural(y[:-1], np.tile([2.0, 1.0, 1.5], (6, 1)), plan)
    assert got.optimizing_log_likelihood[0] == 0
    assert np.array_equal(got.score[0], np.zeros(3))
    model = SuperLSS(family, family.mu(), family.phi(), family.p())
    # Tiny fixture can have a wall optimum: exercise binding and retained
    # observation state without claiming coefficient convergence.
    model.fit(
        pd.DataFrame({"x": np.arange(7)}),
        y,
        sample_weight=weights,
        truncation=([-np.inf] * 6 + [np.nan], [np.inf] * 6 + [-np.inf]),
        censored=["right"] + ["none"] * 5 + ["left"],
        max_inner_iter=2,
    )
    assert len(model._require_fitted().result.theta) == 6


@pytest.mark.parametrize("semantics", ["prior", "frequency"])
@pytest.mark.parametrize("smoothing", [False, True])
def test_bounded_fixed_and_smoothing_fits_use_normalized_natural_rows(
    semantics, smoothing, monkeypatch
):
    import pandas as pd
    from scipy import optimize, special

    from superglm import SuperLSS, s
    from superglm.distributional.families import tweedie as family_module
    from superglm.distributional.solver.assembly import dense_predictor_matrices
    from tests._tweedie_window_oracle import fit_fixture, scipy_rows

    x, y, lower, upper, side, weights = fit_fixture(semantics)
    family = TweedieLSS()
    model = SuperLSS(
        family, family.mu(s("x", k=4)), family.phi(), family.p(), weight_semantics=semantics
    )

    def forbidden_fast_path(*args, **kwargs):
        raise AssertionError("complete predictor rows bypassed observation normalization")

    monkeypatch.setattr(family_module, "tweedie_predictor_rows", forbidden_fast_path)
    frame = pd.DataFrame({"x": x})
    kwargs = dict(
        sample_weight=weights,
        truncation=(lower, upper),
        censored=np.where(side == -1, "left", np.where(side == 1, "right", "none")),
        inner_tol=1e-9,
    )
    if smoothing:
        # A deliberately short outer witness; any iteration-limit disclosure
        # is retained. Coefficient stationarity still has to hold independently.
        model.fit_reml(frame, y, initial_lambda=2.0, max_reml_iter=3, outer="efs", **kwargs)
    else:
        model.fit(frame, y, lambdas={"mean:x#wiggle": 2.0}, **kwargs)
    fitted = model._require_fitted()
    matrices = dense_predictor_matrices(fitted.layout)
    slices = [p.coefficient_slice for p in fitted.layout.predictors]

    def parameters(beta):
        eta = np.column_stack(
            [matrix @ beta[sl] for matrix, sl in zip(matrices, slices, strict=True)]
        )
        return np.column_stack(
            (np.exp(eta[:, 0]), np.exp(eta[:, 1]), 1.05 + 0.9 * special.expit(eta[:, 2]))
        )

    penalty = fitted.result.penalty

    def nll(beta):
        return (
            -scipy_rows(y, lower, upper, side, parameters(beta), weights, semantics).sum()
            + 0.5 * beta @ penalty @ beta
        )

    beta = fitted.coefficients
    step = 1e-5
    gradient = np.array(
        [(nll(beta + d * step) - nll(beta - d * step)) / (2 * step) for d in np.eye(len(beta))]
    )
    assert np.max(np.abs(gradient)) / (1 + nll(beta)) < 3e-8
    reference = optimize.minimize(
        nll,
        beta + 0.02 * np.cos(np.arange(len(beta))),
        method="BFGS",
        options={"gtol": 1e-6, "maxiter": 150},
    )
    # BFGS may disclose precision loss after reaching its finite-difference
    # floor. Its independent score and endpoint, not that flag, are authority.
    reference_gradient = np.array(
        [
            (nll(reference.x + d * step) - nll(reference.x - d * step)) / (2 * step)
            for d in np.eye(len(beta))
        ]
    )
    assert np.max(np.abs(reference_gradient)) / (1 + nll(reference.x)) < 2e-7
    np.testing.assert_allclose(parameters(beta), parameters(reference.x), rtol=2e-5, atol=2e-6)
    assert fitted.result.execution_backend_identifier == "distributional-dense-v1"
    assert model.result_.coefficient_converged
    if smoothing:
        assert fitted.smoothing is not None and fitted.smoothing.iterations > 0


@pytest.mark.parametrize("side", [0, -1, 1])
def test_narrow_conditional_law_keeps_small_derivatives(side):
    family = TweedieLSS()
    lo, y, hi = 3.0, 3.000002, 3.000003
    theta = np.array([[1.2, 0.9, 1.4]])
    plan = _plan([y], lo, hi, side, [2.0], "prior")
    evaluated = family.evaluate_natural([y], theta, plan)

    def conditional(*t):
        denominator = window(lo, hi, *t, weight=2)
        numerator = (
            density(y, *t, weight=2)
            if side == 0
            else window(lo if side == -1 else y, y if side == -1 else hi, *t, weight=2)
        )
        return numerator / denominator

    expected = log_jet(conditional, theta[0])
    assert evaluated.valid[0]
    for got, reference in zip(
        (evaluated.optimizing_log_likelihood, evaluated.score, evaluated.hessian_packed),
        expected,
        strict=True,
    ):
        np.testing.assert_allclose(got, reference, rtol=1e-9, atol=1e-11)


def test_row_tiles_reuse_endpoints_and_instrument_actual_gamma_work():
    from superglm.distributional.kernels._gamma_windows import GammaWindowWorkspace
    from superglm.distributional.kernels._tweedie_windows import (
        TweedieWindowWorkspace,
        tweedie_window_jets,
    )
    from tests._tweedie_window_oracle import gamma_iteration_observer

    theta = np.array([[1.2, 1.1, 1.45]])
    scalar = tweedie_window_jets([0.0], [1.0], theta, [2.0], order=2)
    workspace = TweedieWindowWorkspace(
        gamma=GammaWindowWorkspace(iteration_observer=gamma_iteration_observer)
    )
    tiled = tweedie_window_jets(
        np.zeros(129),
        np.ones(129),
        np.repeat(theta, 129, axis=0),
        np.full(129, 2.0),
        order=2,
        _workspace=workspace,
    )
    for got, expected in zip(tiled, scalar, strict=True):
        np.testing.assert_allclose(got, np.repeat(expected, 129, axis=0), atol=2e-13, rtol=2e-13)
    assert workspace.gamma.cache_hits > 0
    assert workspace.gamma.evaluations < workspace.evaluated_terms
    assert 0 < workspace.gamma.gamma_iterations < workspace.gamma.iteration_upper_bound
