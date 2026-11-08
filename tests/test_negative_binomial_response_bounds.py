"""Finite NB observations: count encodings, normalized derivatives and real fits."""

import importlib
import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import optimize, stats

from superglm import ConvergenceWarning, SuperLSS, s
from superglm.distributional import Predictor
from superglm.distributional.families.negative_binomial import NegativeBinomialLS
from superglm.distributional.family import COMPLETE_OBSERVATION, bounded_observation
from superglm.distributional.observation import ResponseBounds
from superglm.distributional.weights import (
    UnsupportedLikelihoodContractError,
    WeightContract,
    resolve_likelihood_weights,
)
from superglm.features import Numeric, Spline
from superglm.links import LogLink
from tests._negative_binomial_window_oracle import derivatives, normalized_jet, window_jet
from tests.bound_predictor_fixtures import model_from_templates


def _windows():
    return importlib.import_module("superglm.distributional.kernels._negative_binomial_windows")


def _plan(y, lo, hi, side, weights=None, semantics="frequency"):
    y = np.asarray(y, dtype=float)
    n = len(y)
    resolved = resolve_likelihood_weights(
        np.ones(n) if weights is None else np.asarray(weights, dtype=float),
        n_observations=n,
        contract=WeightContract(semantics),
    )
    bounds = ResponseBounds.from_arrays(
        np.broadcast_to(lo, (n,)),
        np.broadcast_to(hi, (n,)),
        np.broadcast_to(side, (n,)),
    )
    return NegativeBinomialLS().bind_likelihood(y, resolved, bounded_observation(bounds))


@pytest.mark.parametrize("observation", ["complete", "left", "right", "truncated"])
def test_curvature_disclosure_does_not_diagnose_censor_limits_as_exact_counts(
    observation, monkeypatch
):
    from superglm.distributional.fit_state import _curvature_disclosures

    family = NegativeBinomialLS()
    y = np.array([1.0, 2.0]) if observation in ("complete", "truncated") else np.array([1.8, 2.7])
    if observation == "complete":
        plan = family.bind_likelihood(
            y,
            resolve_likelihood_weights(
                None, n_observations=len(y), contract=WeightContract("frequency")
            ),
            COMPLETE_OBSERVATION,
        )
    else:
        lo, hi = (-0.5, 4.0) if observation == "truncated" else (-np.inf, np.inf)
        side = {"left": -1, "right": 1, "truncated": 0}[observation]
        plan = _plan(y, lo, hi, side)
    diagnosed = []

    def complete_diagnosis(self, supplied, weights):
        diagnosed.append(supplied.copy())
        return ValueError("complete-count diagnosis sentinel")

    monkeypatch.setattr(
        NegativeBinomialLS, "diagnose_repeated_curvature_failure", complete_diagnosis
    )
    result = SimpleNamespace(
        terminal_curvature=SimpleNamespace(actual_source="nearest_semidefinite", fallback_count=1),
        converged=False,
        theta=None,
    )
    messages = _curvature_disclosures(family, result, y, plan)
    # Keep the generic curvature disclosure for every law, but invoke a
    # complete-response diagnostic only when that law was actually observed.
    assert "nearest positive semidefinite" in messages[0]
    if observation == "complete":
        np.testing.assert_array_equal(diagnosed, [y])
        assert messages[-1] == "complete-count diagnosis sentinel"
    else:
        assert not diagnosed
        assert len(messages) == 1


@pytest.mark.parametrize(
    "lo,hi,L,U,empty",
    [
        (-np.inf, 0, 0, 0, False),
        (0, 1, 1, 1, False),
        (0.1, 0.9, 1, 0, True),
        (1.2, 4.8, 2, 4, False),
        (-5, -1, 0, -1, True),
        (2, np.inf, 3, np.inf, False),
        (2.5, np.inf, 3, np.inf, False),
        (2**50, 2**50 + 2, 2**50 + 1, 2**50 + 2, False),
    ],
)
def test_count_windows_are_open_below_closed_above(lo, hi, L, U, empty):
    actual = _windows().count_window_endpoints([lo], [hi], [1.0])
    for got, want in zip(actual, ([L], [U], [empty]), strict=True):
        np.testing.assert_array_equal(got, want)


def test_prior_threshold_neighbors_use_recorded_rate_comparisons():
    c = float(1 / 3)
    limits = np.array([np.nextafter(c, -np.inf), c, np.nextafter(c, np.inf)])
    L, U, empty = _windows().count_window_endpoints(-np.inf, limits, np.full(3, 3.0))
    np.testing.assert_array_equal(U, [0, 1, 1])
    L, U, empty = _windows().count_window_endpoints(limits, np.inf, np.full(3, 3.0))
    np.testing.assert_array_equal(L, [1, 2, 2])
    assert np.isposinf(U).all()


@pytest.mark.parametrize("limit,weight", [(2**53, 1.0), (1e300, 1e300), ((2**53 - 3) / 3, 3.0)])
def test_indistinguishable_finite_boundaries_are_refused(limit, weight):
    with pytest.raises(
        _windows().CountWindowRepresentationError, match="boundary|count|distinguish"
    ):
        _windows().count_window_endpoints([-np.inf], [limit], [weight])


@pytest.mark.parametrize(
    "L,U,mean,size",
    [
        (0, 0, 0.03, 2.0),
        (1, 1, 5.0, 0.2),
        (2, 9, 3.0, 1.2),
        (70, 75, 0.1, 0.2),
        (0, 4, 10000.0, 10.0),
        (10, np.inf, 0.03, 0.7),
        (0, 10, 0.03, 1e8),
        (1, np.inf, 1e-8, 1e12),
        (0, 500, 200.0, 4.0),
        (300, np.inf, 200.0, 4.0),
    ],
)
def test_window_jets_use_finite_nb_law(L, U, mean, size):
    got = _windows().negative_binomial_window_jets([L - 1], [U], [[mean, size]], [1.0], order=2)
    expected = window_jet(L, U, mean, size)
    scales = [1.0, np.array([[mean, size]]), np.array([[mean * mean, mean * size, size * size]])]
    for actual, want, scale in zip(got, expected, scales, strict=True):
        np.testing.assert_allclose(
            actual * scale, want * scale, rtol=0, atol=2e-10 * max(1, np.max(np.abs(want * scale)))
        )


def test_noninteger_censoring_is_bound_and_has_zero_carrier():
    family = NegativeBinomialLS()
    y = np.array([0.3, 2.2, 3.0])
    plan = _plan(y, -np.inf, np.inf, [-1, 1, 0])
    assert family.capabilities.censored_response
    np.testing.assert_array_equal(plan.parameter_independent_carrier, 0)
    evaluated = family.evaluate_natural(y, np.tile([2.4, 1.3], (3, 1)), plan)
    for row, (count, obs) in enumerate([(None, (0, 0)), (None, (3, np.inf)), (3, None)]):
        oracle = normalized_jet(count, obs, (0, np.inf), 2.4, 1.3)
        for got, expected in zip(
            (evaluated.optimizing_log_likelihood, evaluated.score, evaluated.hessian_packed),
            oracle,
            strict=True,
        ):
            np.testing.assert_allclose(got[row : row + 1], expected, rtol=2e-10, atol=2e-12)


def test_huge_count_conditional_keeps_small_value_and_centered_derivatives():
    k = 2**50
    y = np.array([float(k + 1)])
    plan = _plan(y, k - 1, k + 2, 0)
    theta = np.array([[float(k), 2.5]])
    result = NegativeBinomialLS().evaluate_natural(y, theta, plan)
    oracle = normalized_jet(k + 1, None, (k, k + 2), float(k), 2.5)
    scales = [
        1,
        np.array([[k, 2.5]], dtype=float),
        np.array([[k * k, k * 2.5, 2.5**2]], dtype=float),
    ]
    for got, expected, scale in zip(
        (result.optimizing_log_likelihood, result.score, result.hessian_packed),
        oracle,
        scales,
        strict=True,
    ):
        np.testing.assert_allclose(got * scale, expected * scale, rtol=0, atol=2e-13)


def test_singleton_and_identical_event_conditionals_are_exact_zero():
    family = NegativeBinomialLS()
    for y, lo, hi, side in [(3.0, 2.1, 3.8, 0), (2.2, 2.0, 6.0, 1)]:
        plan = _plan([y], lo, hi, side)
        result = family.evaluate_natural([y], [[1.3, 0.7]], plan)
        for channel in (result.optimizing_log_likelihood, result.score, result.hessian_packed):
            np.testing.assert_array_equal(channel, 0.0)


def test_full_pmf_seed_keeps_central_deviance_at_huge_count():
    k, mean, size = 1e12, 1e12 + 1e6, 1e12
    got = _windows().negative_binomial_window_jets([k - 1], [k], [[mean, size]], [1.0], order=0)
    expected = window_jet(k, k, mean, size)[0]
    np.testing.assert_allclose(got[0], expected, rtol=0, atol=2e-12)


def test_conditional_size_information_survives_density_poisson_classification():
    mean, size = 0.01, 1e14
    plan = _plan([1.0], 0.0, np.inf, 0)
    result = NegativeBinomialLS().evaluate_natural([1.0], [[mean, size]], plan)
    oracle = normalized_jet(1, None, (1, np.inf), mean, size)
    # The complete density's surrogate zeros the size channels here. The
    # conditional size score is finite and has the opposite sign to the
    # derivative of the normalizer alone; an absolute O(u) gate misses it.
    assert oracle[1][0, 1] > 0
    np.testing.assert_allclose(result.score[:, 1], oracle[1][:, 1], rtol=2e-10, atol=0)
    np.testing.assert_allclose(result.hessian_packed[:, 2], oracle[2][:, 2], rtol=2e-10, atol=0)


@pytest.mark.parametrize("semantics,weights", [("frequency", [1, 3, 2]), ("prior", [0.7, 3, 2.5])])
def test_whole_normalized_likelihood_and_take(semantics, weights):
    weights = np.array(weights, dtype=float)
    prior = weights if semantics == "prior" else np.ones(3)
    y = np.array([3, 1.4, 5.7]) / prior
    plan = _plan(
        y,
        np.array([0, -1, 0]) / prior,
        np.array([8, 7, 12]) / prior,
        [0, -1, 1],
        weights,
        semantics,
    )
    family = NegativeBinomialLS()
    parameters = np.tile([2.3, 0.8], (3, 1))
    evaluated = family.evaluate_natural(y, parameters, plan)
    for row, (count, obs, trunc) in enumerate(
        [(3, None, (1, 8)), (None, (0, 1), (0, 7)), (None, (6, 12), (1, 12))]
    ):
        oracle = normalized_jet(count, obs, trunc, 2.3, 0.8, prior[row])
        mass = weights[row] if semantics == "frequency" else 1
        for got, want in zip(
            (evaluated.optimizing_log_likelihood, evaluated.score, evaluated.hessian_packed),
            oracle,
            strict=True,
        ):
            np.testing.assert_allclose(got[row : row + 1], mass * want, rtol=2e-10, atol=2e-12)
    rows = np.array([2, 0])
    child = plan.take(rows)
    rebound = family.bind_likelihood(y[rows], plan.weights.take(rows), plan.observation.take(rows))
    assert child.plan_identifier == rebound.plan_identifier
    sliced = family.evaluate_natural(y[rows], parameters[rows], child)
    np.testing.assert_array_equal(sliced.score, evaluated.score[rows])
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(child)).exact_row_mask, [False, True])


def test_thresholds_with_identical_count_sets_keep_distinct_identity():
    a = _plan([3], 1.1, 5.8, 0)
    b = _plan([3], 1.2, 5.8, 0)
    assert a.plan_identifier != b.plan_identifier
    family = NegativeBinomialLS()
    x = family.evaluate_natural([3], [[2, 1]], a)
    y = family.evaluate_natural([3], [[2, 1]], b)
    np.testing.assert_array_equal(x.optimizing_log_likelihood, y.optimizing_log_likelihood)


def test_censoring_threshold_never_enters_exact_count_mapper(monkeypatch):
    module = importlib.import_module("superglm.distributional.families.negative_binomial")
    real = module._certified_count_mapping

    def check(response, *args):
        assert not np.any(response == 2.3)
        return real(response, *args)

    monkeypatch.setattr(module, "_certified_count_mapping", check)
    plan = _plan([2.3, 1.0], -np.inf, np.inf, [1, 0])
    assert plan.exact_row_mask.tolist() == [False, True]


@pytest.mark.parametrize("y,lo,hi,side", [(0.8, 0.1, 0.9, -1), (0.4, -np.inf, np.inf, 1)])
def test_empty_count_sets_are_data_errors_only_when_impossible(y, lo, hi, side):
    if np.isfinite(hi):
        with pytest.raises(ValueError, match="empty count set.*input row 0"):
            _plan([y], lo, hi, side)
    else:
        plan = _plan([y], lo, hi, side)
        assert NegativeBinomialLS().evaluate_natural([y], [[2, 1]], plan).valid.all()


def test_infinite_tail_retains_mass_above_exact_count_encoding_range():
    # Geometric survival is analytic and keeps most mass above 2**53 here.
    k, mean = 2**52, 2.0**54
    value = NegativeBinomialLS().log_window_probability(k, np.inf, [[mean, 1]])[0]
    np.testing.assert_allclose(value, -(k + 1) * np.log1p(1 / mean), rtol=2e-12)


def test_forced_work_exhaustion_marks_requested_channels_invalid(monkeypatch):
    module = _windows()
    plan = _plan([2.2], -np.inf, np.inf, 1)
    monkeypatch.setattr(module, "_MAX_RECURRENCE_CELLS", 0)
    result = NegativeBinomialLS().evaluate_natural([2.2], [[2, 1]], plan)
    np.testing.assert_array_equal(result.valid, False)
    assert np.isfinite(result.score).all()
    assert np.isfinite(result.hessian_packed).all()


def test_bounded_optional_directions_and_fisher_refuse_honestly():
    family = NegativeBinomialLS()
    plan = _plan([2.2], -np.inf, np.inf, 1)
    theta = np.array([[2, 1]])
    with pytest.raises(UnsupportedLikelihoodContractError, match="complete rows"):
        family.expected_information_natural(theta, plan)
    with pytest.raises(UnsupportedLikelihoodContractError, match="complete rows"):
        family.predictor_curvature_directional_derivative(
            [2.2], np.log(theta), np.ones((1, 2)), (LogLink(), LogLink()), plan
        )
    with pytest.raises(UnsupportedLikelihoodContractError, match="complete rows"):
        family.predictor_curvature_second_directional_derivative(
            [2.2], np.log(theta), np.ones((1, 2)), np.ones((1, 2)), (LogLink(), LogLink()), plan
        )


@pytest.mark.parametrize("semantics", ["prior", "frequency"])
def test_exact_channels_of_bounded_plan_match_complete_finite_count_law(semantics):
    weights = np.array([1, 3, 2], dtype=float)
    y = np.array([2, 4, 1]) / (weights if semantics == "prior" else 1)
    bounded = _plan(y, -1, np.inf, [0, 0, 1], weights, semantics)
    family = NegativeBinomialLS()
    complete = family.bind_likelihood(y, bounded.weights, COMPLETE_OBSERVATION)
    theta = np.tile([2.4, 1.2], (3, 1))
    b, c = family.evaluate_natural(y, theta, bounded), family.evaluate_natural(y, theta, complete)
    np.testing.assert_allclose(
        b.optimizing_log_likelihood[:2],
        (c.optimizing_log_likelihood + c.parameter_independent_carrier)[:2],
        rtol=2e-13,
    )
    np.testing.assert_allclose(b.score[:2], c.score[:2], atol=3e-14, rtol=3e-13)
    np.testing.assert_allclose(b.hessian_packed[:2], c.hessian_packed[:2], atol=3e-14, rtol=3e-13)


def _fit_fixture(semantics, n=100):
    rng = np.random.default_rng(80431)
    x = np.linspace(-1, 1, n * 2)
    weights = np.resize([1.0, 3.0, 2.0], len(x))
    prior = weights if semantics == "prior" else np.ones(len(x))
    counts = rng.negative_binomial(1.7 * prior, 1.7 / (1.7 + np.exp(1.2 + 0.25 * x)))
    keep = np.flatnonzero(counts > 0)[:n]
    x, counts, prior, weights = x[keep], counts[keep], prior[keep], weights[keep]
    side = np.where(counts >= 6, "right", "none")
    y = np.where(counts >= 6, 5.4, counts) / prior
    return pd.DataFrame({"x": x}), y, weights, prior, counts, side


def _reference_objective(coefficients, design, counts, prior, mass):
    mean = np.exp(design @ coefficients[:-1])
    size = np.exp(coefficients[-1])
    law = stats.nbinom(prior * size, size / (size + mean))
    numerator = np.where(counts >= 6, law.logsf(5), law.logpmf(counts))
    return -float(np.mean(mass * (numerator - law.logsf(0))))


def _reference_gradient(function, point):
    # Five-point differentiation of the independent *normalized* objective.
    # h=2e-4 balances O(h^4) truncation against O(u/h) roundoff on this scale.
    steps = np.eye(len(point)) * 2e-4
    return np.array(
        [
            (
                function(point - 2 * h)
                - 8 * function(point - h)
                + 8 * function(point + h)
                - function(point + 2 * h)
            )
            / (12 * 2e-4)
            for h in steps
        ]
    )


@pytest.mark.parametrize("semantics", ["prior", "frequency"])
@pytest.mark.parametrize("slope", [False, True])
def test_real_censored_truncated_fit_has_independent_stationary_reference(semantics, slope):
    X, y, weights, prior, counts, side = _fit_fixture(semantics)
    family = NegativeBinomialLS()
    model = model_from_templates(
        family=family,
        predictors=(Predictor("mean", {"x": Numeric()} if slope else {}), Predictor("theta", {})),
        weight_semantics=semantics,
    ).fit(X, y, sample_weight=weights, censored=side, truncation=(0, None), inner_tol=1e-9)
    design = np.column_stack((np.ones(len(X)), X.x)) if slope else np.ones((len(X), 1))
    mass = weights if semantics == "frequency" else np.ones(len(y))

    def objective(c):
        return _reference_objective(c, design, counts, prior, mass)

    start = np.r_[np.log(3), np.zeros(design.shape[1] - 1), np.log(1.5)]
    reference = optimize.minimize(
        objective,
        start,
        method="Nelder-Mead",
        options={"xatol": 2e-10, "fatol": 1e-14, "maxiter": 1500},
    )
    reference_score = _reference_gradient(objective, reference.x)
    assert np.max(np.abs(reference_score)) < 2e-7
    fitted = np.fromiter(model.coef_.values(), dtype=float)
    # Stationarity is checked *before* endpoint comparison. The finite-diff
    # score noise and a well-conditioned 2/3-parameter Hessian allow 3e-5
    # log-parameter agreement, far below the sampling uncertainty.
    assert np.max(np.abs(_reference_gradient(objective, fitted))) < 2e-7
    assert model.result_.coefficient_converged
    np.testing.assert_allclose(fitted, reference.x, rtol=0, atol=3e-5)
    np.testing.assert_allclose(
        model.result_.log_likelihood, -len(y) * objective(fitted), rtol=3e-12
    )


def test_frequency_bounded_fit_matches_literal_replication_and_null_fit():
    X, y, weights, _, _, side = _fit_fixture("frequency", 60)
    family = NegativeBinomialLS()
    model = SuperLSS(family, family.mean(), family.theta(), weight_semantics="frequency")
    model.fit(X, y, sample_weight=weights, censored=side, truncation=(0, None), inner_tol=1e-9)
    rows = np.repeat(np.arange(len(y)), weights.astype(int))
    repeated = SuperLSS(family, family.mean(), family.theta(), weight_semantics="frequency")
    repeated.fit(
        X.iloc[rows].reset_index(drop=True),
        y[rows],
        censored=side[rows],
        truncation=(0, None),
        inner_tol=1e-9,
    )
    np.testing.assert_allclose(
        list(model.coef_.values()), list(repeated.coef_.values()), rtol=0, atol=2e-7
    )
    np.testing.assert_allclose(
        model.result_.log_likelihood, repeated.result_.log_likelihood, rtol=2e-12
    )
    np.testing.assert_allclose(
        model._require_fitted().null_model.log_likelihood, model.result_.log_likelihood, rtol=2e-11
    )


@pytest.mark.parametrize(
    "L,U,mean,size",
    [(0, 10, 1e-8, 2.0), (0, 10, 0.03, 1e8), (1, np.inf, 1e-10, 1e-120), (50, np.inf, 10.0, 1e-8)],
)
def test_small_complement_is_retained_without_clipping_or_spurious_score(L, U, mean, size):
    actual = _windows().negative_binomial_window_jets([L - 1], [U], [[mean, size]], [1.0], order=2)
    reference = window_jet(L, U, mean, size)
    for got, want in zip(actual, reference, strict=True):
        np.testing.assert_allclose(got, want, rtol=2e-9, atol=0)


def test_finite_wide_near_poisson_window_uses_derivative_tail_admission(monkeypatch):
    module = _windows()
    monkeypatch.setattr(module, "_MAX_RECURRENCE_CELLS", 2000)
    jet = module.negative_binomial_window_jets([-1], [2**40], [[2, 1e9]], [1.0], order=2)
    # The omitted probability is far below the binary64 exponent range.
    for channel in jet:
        np.testing.assert_array_equal(channel, 0.0)


@pytest.mark.parametrize("semantics", ["prior", "frequency"])
def test_fixed_smoothing_fit_and_chunk_replay_have_independent_stationarity(semantics):
    from superglm.distributional.solver.solver import fit_dense_fixed_lambda

    X, y, weights, prior, counts, side = _fit_fixture(semantics, 80)
    family = NegativeBinomialLS()
    model = model_from_templates(
        family=family,
        predictors=(Predictor("mean", {"x": Spline(kind="cr", n_knots=4)}), Predictor("theta", {})),
        weight_semantics=semantics,
    ).fit(
        X,
        y,
        sample_weight=weights,
        censored=side,
        truncation=(0, None),
        lambdas={"mean:x#wiggle": 0.7},
        inner_tol=1e-9,
    )
    fitted = model._require_fitted()
    result = fitted.fit_state.solver_result
    design = np.column_stack((np.ones(len(y)), fitted.layout.predictors[0].design.toarray()))
    mass = weights if semantics == "frequency" else np.ones(len(y))

    def objective(c):
        return _reference_objective(
            c, design, counts, prior, mass
        ) + 0.5 * c @ result.penalty @ c / len(y)

    c = fitted.coefficients
    reference = optimize.minimize(
        objective,
        np.r_[np.log(3), np.zeros(len(c) - 2), np.log(1.5)],
        jac=lambda p: _reference_gradient(objective, p),
        method="BFGS",
        options={"gtol": 2e-9, "maxiter": 200},
    )
    assert np.max(np.abs(_reference_gradient(objective, reference.x))) < 3e-7
    assert np.max(np.abs(_reference_gradient(objective, c))) < 3e-7
    np.testing.assert_allclose(design @ c[:-1], design @ reference.x[:-1], rtol=0, atol=3e-5)
    np.testing.assert_allclose(c[-1], reference.x[-1], rtol=0, atol=3e-5)
    assert model.result_.coefficient_converged
    plan = _plan(y, 0, np.inf, np.where(side == "right", 1, 0), weights, semantics)
    chunked = fit_dense_fixed_lambda(
        family, fitted.layout, y, plan, result.penalty, config=result.config, chunk_size=17
    )
    assert chunked.converged
    assert chunked.execution_backend_identifier == "distributional-chunked-v1"
    np.testing.assert_allclose(chunked.theta, result.theta, rtol=2e-6)
    assert chunked.terminal_rank.rank == result.terminal_rank.rank


def test_aliases_preserve_bounded_predictions_and_rank():
    X, y, weights, _, _, side = _fit_fixture("frequency", 80)
    X["alias"] = 2 * X.x
    family = NegativeBinomialLS()
    models = [
        model_from_templates(
            family=family,
            predictors=(Predictor("mean", terms), Predictor("theta", {})),
            weight_semantics="frequency",
        ).fit(X, y, sample_weight=weights, censored=side, truncation=(0, None), inner_tol=1e-9)
        for terms in ({"x": Numeric()}, {"x": Numeric(), "alias": Numeric()})
    ]
    assert all(m._require_fitted().fit_state.solver_result.terminal_rank.rank == 3 for m in models)
    np.testing.assert_allclose(
        models[0].predict_parameters(X), models[1].predict_parameters(X), rtol=3e-6
    )


def test_conditional_anchor_does_not_cast_an_unbounded_mode_to_integer():
    import mpmath as mp

    plan = _plan([3], 1, 3, 0)
    result = NegativeBinomialLS().evaluate_natural([3], [[1e100, 1e100]], plan)
    reference = derivatives(lambda m, r: -mp.log1p(3 * (m + r) / (m * (r + 2))), (1e100, 1e100))
    for got, want in zip(
        (result.optimizing_log_likelihood, result.score, result.hessian_packed),
        reference,
        strict=True,
    ):
        np.testing.assert_allclose(got, want, rtol=2e-12, atol=0)


def test_automatic_smoothing_uses_bounded_core_and_stationary_endpoint():
    X, y, weights, prior, counts, side = _fit_fixture("prior", 100)
    family = NegativeBinomialLS()
    model = SuperLSS(family, family.mean(s("x", kind="cr", k=5)), family.theta())
    model.fit_reml(
        X,
        y,
        sample_weight=weights,
        censored=side,
        truncation=(0, None),
        max_reml_iter=30,
        inner_tol=1e-9,
        outer="efs+newton",
        practical_reml=False,
    )
    fitted = model._require_fitted()
    result = fitted.fit_state.solver_result
    design = np.column_stack((np.ones(len(y)), fitted.layout.predictors[0].design.toarray()))

    def objective(c):
        return _reference_objective(
            c, design, counts, prior, np.ones(len(y))
        ) + 0.5 * c @ result.penalty @ c / len(y)

    assert np.max(np.abs(_reference_gradient(objective, fitted.coefficients))) < 3e-7
    assert model.result_.coefficient_converged
    assert model.result_.smoothing_converged
    assert model.result_.curvature_telemetry.actual_source == "observed"
    assert np.isfinite(model.covariance_).all()


def test_fraction_failure_falls_back_to_independent_count_recurrence(monkeypatch):
    module = _windows()
    monkeypatch.setattr(module.beta, "MAX_TERMS", 0)
    actual = module.negative_binomial_window_jets([299], [np.inf], [[200, 4]], [1.0], order=2)
    reference = window_jet(300, np.inf, 200.0, 4.0)
    for got, want in zip(actual, reference, strict=True):
        np.testing.assert_allclose(got, want, rtol=2e-10, atol=2e-13)


def test_trial_work_failure_backtracks_without_publishing_a_fake_fit(monkeypatch):
    module = importlib.import_module("superglm.distributional.families.negative_binomial")
    windows = _windows()
    real = module.negative_binomial_normalized_jets
    refused = False

    def evaluate(*args, order):
        nonlocal refused
        if order == 0 and not refused:
            refused = True
            with monkeypatch.context() as patch:
                patch.setattr(windows, "_MAX_RECURRENCE_CELLS", 0)
                return real(*args, order=order)
        return real(*args, order=order)

    monkeypatch.setattr(module, "negative_binomial_normalized_jets", evaluate)
    X, y, weights, prior, counts, side = _fit_fixture("prior", 80)
    family = NegativeBinomialLS()
    model = SuperLSS(family, family.mean(), family.theta())
    model.fit(X, y, sample_weight=weights, censored=side, truncation=(0, None), inner_tol=1e-9)
    result = model._require_fitted().fit_state.solver_result
    assert refused and result.backtracking_steps > 0
    assert model.result_.coefficient_converged

    def objective(c):
        return _reference_objective(c, np.ones((len(y), 1)), counts, prior, np.ones(len(y)))

    assert np.max(np.abs(_reference_gradient(objective, result.coefficients))) < 2e-7


def test_forced_nonconvergence_remains_disclosed():
    X, y, weights, _, _, side = _fit_fixture("prior", 80)
    family = NegativeBinomialLS()
    model = SuperLSS(family, family.mean(), family.theta())
    with pytest.warns(ConvergenceWarning):
        model.fit(
            X, y, sample_weight=weights, censored=side, truncation=(0, None), max_inner_iter=1
        )
    assert not model.result_.coefficient_converged
    assert model.result_.warnings
