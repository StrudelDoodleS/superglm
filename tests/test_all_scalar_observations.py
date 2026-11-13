"""All eleven scalar laws actually evaluate each supported observation cell.

The independent oracle catches density bypass, missing truncation denominators,
incorrect carriers, endpoint inequalities and double frequency multiplication.
Tail-engine tolerances are 1e-12 at core orders; the moderate windows here use
an outcome allowance of 1e-10 times each channel's scale for two tail differences
and the natural/predictor coordinate maps. No refusal counts as a passing cell.
"""

from dataclasses import dataclass

import mpmath as mp
import numpy as np
import pandas as pd
import pytest

from superglm.distributional import (
    GammaLS,
    GaussianLS,
    GeneralizedGammaLSS,
    GeneralizedParetoLSS,
    LogLogisticLSS,
    LogNormalLS,
    LomaxLSS,
    NegativeBinomialLS,
    TweedieLSS,
    TwoPieceLogNormalLSS,
    TwoPieceNormalLSS,
)
from superglm.distributional.family import COMPLETE_OBSERVATION, bounded_observation
from superglm.distributional.observation import ResponseBounds
from superglm.distributional.solver.derivatives import transform_natural_derivatives
from superglm.distributional.weights import (
    LikelihoodWeightError,
    UnsupportedLikelihoodContractError,
    WeightContract,
    resolve_likelihood_weights,
)
from tests import (
    _scalar_observation_oracle,
    _tweedie_window_oracle,
    _two_piece_observation_oracle,
)
from tests._oracle_store import OracleStore
from tests._scalar_observation_oracle import contribution, inverse_links, jet


def _predictors(theta, maps):
    """The predictors of ``theta`` under ``maps``, rounded once to binary64 from 60 digits.

    The engine's links give the same point to within an ulp or two, far inside
    the 1e-10 outcome allowance; computing it here keeps the stored answers'
    arguments the same on every platform's libm.
    """
    with mp.workdps(60):
        point = []
        for x, (kind, lower, upper) in zip(map(mp.mpf, theta), maps, strict=True):
            if kind == "identity":
                point.append(x)
            elif kind == "log":
                point.append(mp.log(x - lower))
            else:
                point.append(mp.log((x - lower) / (upper - x)))
        return tuple(float(v) for v in point)


def _row_jet(law, form, y, lo, hi, side, theta, prior=1, mass=1, maps=None):
    """The oracle jet of ``mass`` times one row's contribution.

    ``maps`` None: in the natural parameters at ``theta``; otherwise in the
    predictors, at ``_predictors(theta, maps)``, mapped back by ``inverse_links``.
    """

    def law_at(*point):
        return mass * contribution(law, form, y, lo, hi, side, point, prior)

    if maps is None:
        return jet(law_at, theta)
    return jet(lambda *e: law_at(*inverse_links(e, maps)), _predictors(theta, maps))


# The 60-digit jets take most of these tests' time and depend only on their
# arguments, so their answers are stored (``tests/_oracle_store.py``); every
# engine-against-oracle comparison below still runs.
ROW_JET = OracleStore(
    "all_scalar_observations_row_jet",
    _row_jet,
    depends=(
        _predictors,
        _scalar_observation_oracle,
        _tweedie_window_oracle,
        _two_piece_observation_oracle,
    ),
    module="tests/test_all_scalar_observations.py",
)


def test_the_stored_oracle_answers_are_current():
    ROW_JET.verify()


@dataclass(frozen=True)
class Case:
    law: str
    family: object
    theta: tuple[float, ...]
    maps: tuple[tuple[str, float, float], ...]
    form: str = "location"


IDENTITY, LOG = ("identity", 0.0, 0.0), ("log", 0.0, 0.0)
SCALE = ("log", 0.01, 0.0)
SKEW = ("logit", -0.9, 0.9)
CASES = [
    Case("gaussian", GaussianLS(scale_floor=0.01), (0.7, 1.2), (IDENTITY, SCALE)),
    Case("gamma", GammaLS(), (2.0, 0.8), (LOG, LOG)),
    Case(
        "log_normal",
        LogNormalLS(parametrisation="location", scale_floor=0.01),
        (0.3, 0.7),
        (IDENTITY, SCALE),
    ),
    Case(
        "log_normal",
        LogNormalLS(parametrisation="mean", scale_floor=0.01),
        (2.0, 0.7),
        (LOG, SCALE),
        "mean",
    ),
    Case(
        "gg",
        GeneralizedGammaLSS(parametrisation="location", scale_floor=0.01),
        (0.3, 0.6, 0.7),
        (IDENTITY, SCALE, IDENTITY),
    ),
    Case(
        "gg",
        GeneralizedGammaLSS(parametrisation="mean", scale_floor=0.01),
        (2.0, 0.6, 0.7),
        (LOG, SCALE, IDENTITY),
        "mean",
    ),
    Case(
        "gpd",
        GeneralizedParetoLSS(shape_lower=0.0, shape_upper=0.9),
        (1.7, 0.2),
        (LOG, ("logit", 0.0, 0.9)),
    ),
    Case("lomax", LomaxLSS(), (1.5, 2.2), (LOG, LOG)),
    Case("log_logistic", LogLogisticLSS(), (1.5, 2.2), (LOG, LOG)),
    Case("nb", NegativeBinomialLS(), (2.0, 0.7), (LOG, LOG)),
    Case("tweedie", TweedieLSS(), (2.0, 1.2, 1.5), (LOG, LOG, ("logit", 1.05, 1.95))),
    Case(
        "two_piece_normal",
        TwoPieceNormalLSS(scale_floor=0.01, skew_bound=0.9),
        (0.4, 1.1, 0.25),
        (IDENTITY, SCALE, SKEW),
    ),
    Case(
        "two_piece_log_normal",
        TwoPieceLogNormalLSS(parametrisation="location", scale_floor=0.01, skew_bound=0.9),
        (0.3, 0.7, 0.25),
        (IDENTITY, SCALE, SKEW),
    ),
    Case(
        "two_piece_log_normal",
        TwoPieceLogNormalLSS(parametrisation="mean", scale_floor=0.01, skew_bound=0.9),
        (2.0, 0.7, 0.25),
        (LOG, SCALE, SKEW),
        "mean",
    ),
]
IDS = [f"{case.law}-{case.form}" for case in CASES]
OLDER_POSITIVE = [
    case
    for case in CASES
    if case.law in {"gamma", "log_normal", "gg", "gpd", "lomax", "log_logistic"}
]


def weights(values, semantics="frequency"):
    return resolve_likelihood_weights(
        np.asarray(values, float),
        n_observations=len(values),
        contract=WeightContract(semantics=semantics),
    )


def fixture(case):
    y = np.full(10, 3.0 if case.law == "nb" else 1.2)
    lower, upper = (0.5, 6.5) if case.law == "nb" else (0.35, 3.5)
    lo = np.array([-np.inf, -np.inf, -np.inf, lower, -np.inf, lower, lower, -np.inf, lower, lower])
    hi = np.array([np.inf, np.inf, np.inf, np.inf, upper, upper, np.inf, upper, upper, upper])
    side = np.array([0, -1, 1, 0, 0, 0, 1, -1, 1, -1])
    return y, ResponseBounds.from_arrays(lo, hi, side)


def channels(evaluation):
    return evaluation.reported_log_likelihood, evaluation.score, evaluation.hessian_packed


def assert_channels(actual, expected):
    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_allclose(
            got, want, rtol=0, atol=1e-10 * max(1.0, float(np.max(np.abs(want))))
        )


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_all_observation_cells_natural_and_linked_orders(case):
    y, bounds = fixture(case)
    family = case.family
    mass = np.arange(len(y)) % 3 + 1
    plan = family.bind_likelihood(y, weights(mass), bounded_observation(bounds))
    theta = np.tile(case.theta, (len(y), 1))
    natural = family.evaluate_natural(y, theta, plan)
    assert np.all(natural.valid), "every ordinary-domain support cell must actually execute"
    links = [p.default_link for p in family.parameters]
    # The engine and the oracle read the same predictors, rounded from 60 digits
    # (``_predictors``); the family's own links agree with them to the last bits.
    eta = np.tile(_predictors(case.theta, case.maps), (len(y), 1))
    engine_eta = np.column_stack([link.link(theta[:, i]) for i, link in enumerate(links)])
    np.testing.assert_allclose(engine_eta, eta, rtol=1e-14, atol=1e-15)
    linked = transform_natural_derivatives(natural, eta, links)
    for row in range(len(y)):
        cell = (
            case.law,
            case.form,
            float(y[row]),
            float(bounds.lower[row]),
            float(bounds.upper[row]),
            int(bounds.side[row]),
        )
        assert_channels(
            tuple(c[row] for c in channels(natural)),
            ROW_JET(*cell, case.theta, mass=int(mass[row])),
        )
        expected = ROW_JET(*cell, case.theta, mass=int(mass[row]), maps=case.maps)
        assert_channels(
            (
                linked.reported_log_likelihood[row],
                linked.score_eta[row],
                linked.hessian_eta_packed[row],
            ),
            expected,
        )
    for order in (0, 1):
        evaluated = family.evaluate_natural(y, theta, plan, derivative_order=order)
        assert np.all(evaluated.valid)
        assert_channels(channels(evaluated)[: order + 1], channels(natural)[: order + 1])
        assert evaluated.hessian_packed is None
        if order == 0:
            assert evaluated.score is None


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_weight_laws_replication_and_unit_prior(case):
    y, bounds = fixture(case)
    family = case.family
    theta = np.tile(case.theta, (len(y), 1))
    obs = bounded_observation(bounds)
    mass = np.arange(len(y)) % 3 + 1
    weighted = family.evaluate_natural(y, theta, family.bind_likelihood(y, weights(mass), obs))
    repeated = np.repeat(np.arange(len(y)), mass)
    yr, tr, br = y[repeated], theta[repeated], bounds.take(repeated)
    replicated = family.evaluate_natural(
        yr, tr, family.bind_likelihood(yr, weights(np.ones(len(yr))), bounded_observation(br))
    )
    assert np.all(weighted.valid) and np.all(replicated.valid)
    for got, literal in zip(channels(weighted), channels(replicated), strict=True):
        np.testing.assert_allclose(
            np.sum(got, axis=0), np.sum(literal, axis=0), rtol=1e-12, atol=1e-12
        )
    unit = []
    for semantics in ("frequency", "prior"):
        plan = family.bind_likelihood(y, weights(np.ones(len(y)), semantics), obs)
        unit.append(family.evaluate_natural(y, theta, plan))
    assert_channels(channels(unit[0]), channels(unit[1]))
    if case.law not in {"gaussian", "gamma", "nb", "tweedie"}:
        with pytest.raises(UnsupportedLikelihoodContractError, match="prior"):
            family.bind_likelihood(y, weights(np.full(len(y), 2), "prior"), obs)
        return
    plan = family.bind_likelihood(y, weights(np.full(len(y), 2), "prior"), obs)
    actual = family.evaluate_natural(y, theta, plan)
    assert np.all(actual.valid)
    for row in range(len(y)):
        expected = ROW_JET(
            case.law,
            case.form,
            float(y[row]),
            float(bounds.lower[row]),
            float(bounds.upper[row]),
            int(bounds.side[row]),
            case.theta,
            prior=2,
        )
        assert_channels(tuple(c[row] for c in channels(actual)), expected)


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_plan_take_preserves_law_bounds_provenance_and_identity(case):
    y, bounds = fixture(case)
    family = case.family
    w = weights(np.arange(len(y)) % 3 + 1)
    plan = family.bind_likelihood(y, w, bounded_observation(bounds))
    changed = ResponseBounds.from_arrays(
        bounds.lower,
        np.where(np.isfinite(bounds.upper), bounds.upper + 1, bounds.upper),
        bounds.side,
    )
    other = family.bind_likelihood(y, w, bounded_observation(changed))
    assert plan.plan_identifier != other.plan_identifier
    if case.law != "gaussian":
        changed_response = family.bind_likelihood(y + 1, w, bounded_observation(bounds))
        assert changed_response.plan_identifier != plan.plan_identifier
    take = np.array([8, 2, 0, 5])
    child = plan.take(take)
    np.testing.assert_array_equal(child.weights.input_positions, take)
    assert child.plan_identifier != plan.plan_identifier
    theta = np.tile(case.theta, (len(y), 1))
    parent_result = family.evaluate_natural(y, theta, plan)
    child_result = family.evaluate_natural(y[take], theta[take], child)
    assert np.all(child_result.valid)
    assert_channels(channels(child_result), tuple(c[take] for c in channels(parent_result)))
    with pytest.raises(LikelihoodWeightError, match="duplicate|unique"):
        plan.take(np.array([0, 0]))
    for value in (
        child.observation.bounds.lower,
        child.observation.bounds.upper,
        child.weights.values,
    ):
        assert not value.flags.writeable
    validate = getattr(family, "validate_likelihood_plan", None)
    if validate is not None:
        np.testing.assert_array_equal(validate(y[take], child), y[take])
        with pytest.raises(UnsupportedLikelihoodContractError, match="response|match"):
            validate(y[take] + 1, child)


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_complete_carriers_and_ineffective_bounds_preserve_the_same_density(case):
    y = np.array([1.0, 2.0])
    mass = np.array([2, 3])
    w = weights(mass)
    empty = bounded_observation(
        ResponseBounds.from_arrays(np.full(2, -np.inf), np.full(2, np.inf), np.zeros(2, int))
    )
    complete = case.family.bind_likelihood(y, w, COMPLETE_OBSERVATION)
    inactive = case.family.bind_likelihood(y, w, empty)
    assert (
        complete.plan_identifier
        == case.family.bind_likelihood(y.copy(), w, COMPLETE_OBSERVATION).plan_identifier
    )
    # An explicitly bounded NB plan also records its finite-NB boundary policy;
    # its identity may differ even when these ordinary rows have no active bounds.
    theta = np.tile(case.theta, (2, 1))
    actual = case.family.evaluate_natural(y, theta, complete)
    assert actual.valid.all()
    for row in range(2):
        expected = jet(
            lambda *t: mass[row] * contribution(case.law, case.form, y[row], -np.inf, np.inf, 0, t),
            case.theta,
        )
        assert_channels(tuple(c[row] for c in channels(actual)), expected)
    zero_order = case.family.evaluate_natural(y, theta, inactive, derivative_order=0)
    assert zero_order.valid.all()
    assert_channels((zero_order.reported_log_likelihood,), (actual.reported_log_likelihood,))


@pytest.mark.parametrize("case", OLDER_POSITIVE, ids=lambda c: f"{c.law}-{c.form}")
@pytest.mark.parametrize("threshold", [-0.5, 0.0])
def test_right_limits_below_density_support_are_not_exact_responses(case, threshold):
    y = np.array([threshold, 1.0, 2.0])
    bounds = ResponseBounds.from_arrays(
        np.full(3, -np.inf), np.full(3, np.inf), np.array([1, 0, 0])
    )
    family = case.family
    plan = family.bind_likelihood(y, weights([2, 1, 1]), bounded_observation(bounds))
    np.testing.assert_array_equal(family.validate_likelihood_plan(y, plan), y)
    np.testing.assert_array_equal(plan.exact_response, y)
    assert plan.parameter_independent_carrier[0] == 0
    initial = family.initialize(y, plan).theta
    assert np.all(np.isfinite(initial))
    for order in (0, 1, 2):
        actual = family.evaluate_natural(
            y, np.tile(case.theta, (3, 1)), plan, derivative_order=order
        )
        assert np.all(actual.valid)
        for channel in channels(actual)[: order + 1]:
            np.testing.assert_allclose(channel[0], 0, atol=1e-12)
    with pytest.raises(ValueError):
        family.bind_likelihood(np.array([-0.5, 1, 2]), weights([1, 1, 1]), COMPLETE_OBSERVATION)


@pytest.mark.parametrize("case", OLDER_POSITIVE, ids=lambda c: f"{c.law}-{c.form}")
def test_original_windows_and_impossible_left_limits_keep_input_positions(case):
    w = weights([0, 1, 0, 1, 1]).take(np.array([2, 0, 1]))
    y = np.array([0.0, 1.0, 2.0])
    impossible = ResponseBounds.from_arrays(
        np.full(3, -np.inf), np.full(3, np.inf), np.array([-1, 0, 0])
    )
    with pytest.raises(ValueError, match="input row 4"):
        case.family.bind_likelihood(y, w, bounded_observation(impossible))
    # The raw threshold lies outside its original window; support clipping
    # would erase this lower bound and accidentally admit the row.
    outside = ResponseBounds.from_arrays(
        np.array([-0.25, -np.inf, -np.inf]), np.full(3, np.inf), np.array([1, 0, 0])
    )
    with pytest.raises(ValueError, match="input row 4"):
        case.family.bind_likelihood(np.array([-0.5, 1, 2]), w, bounded_observation(outside))


@pytest.mark.parametrize("case", OLDER_POSITIVE, ids=lambda c: f"{c.law}-{c.form}")
def test_old_optional_directions_do_not_revalidate_censor_limits_as_densities(case):
    y = np.array([-0.5, 1.0, 2.0])
    bounds = ResponseBounds.from_arrays(
        np.full(3, -np.inf), np.full(3, np.inf), np.array([1, 0, 0])
    )
    family = case.family
    plan = family.bind_likelihood(y, weights([2, 1, 1]), bounded_observation(bounds))
    theta = np.tile(case.theta, (3, 1))
    links = [p.default_link for p in family.parameters]
    eta = np.column_stack([link.link(theta[:, j]) for j, link in enumerate(links)])
    direction = np.full_like(eta, 0.1)
    first = family.predictor_curvature_directional_derivative(y, eta, direction, links, plan)
    second = family.predictor_curvature_second_directional_derivative(
        y, eta, direction, direction, links, plan
    )
    for result in (first, second):
        assert np.all(np.isfinite(result))
        np.testing.assert_allclose(result[0], 0, atol=1e-10)


@pytest.mark.parametrize("case", CASES, ids=IDS)
@pytest.mark.parametrize("side", ["left", "right"])
def test_all_censored_actual_fits_disclose_a_forced_stop(case, side, monkeypatch):
    import warnings

    from superglm import SeparationWarning, SuperLSS
    from superglm.distributional import api, bind_predictor
    from superglm.distributional.family import InitialParameterState
    from superglm.distributional.results.solver import DenseSolverConfig

    def one_step_budget(**kwargs):
        return DenseSolverConfig(terminal_retry_iterations=1, max_predictor_step=0.1, **kwargs)

    monkeypatch.setattr(api, "DenseSolverConfig", one_step_budget)
    family = case.family
    response = np.ones(6)
    bounds = ResponseBounds.from_arrays(
        np.full(6, -np.inf), np.full(6, np.inf), np.full(6, -1 if side == "left" else 1)
    )
    plan = family.bind_likelihood(response, weights(np.ones(6)), bounded_observation(bounds))
    initial = family.initialize(response, plan).theta
    assert np.all(np.isfinite(initial))
    assert family.evaluate_natural(response, initial, plan).valid.all()
    # Some heuristic starts already round an all-censored probability to one.
    # Force an ordinary, nonstationary interior start for the stop test instead
    # of depending on a platform's initializer/rounding trajectory.
    monkeypatch.setattr(
        type(family),
        "initialize",
        lambda self, y, plan: InitialParameterState(np.tile(case.theta, (len(y), 1))),
    )
    frame = pd.DataFrame({"x": np.arange(6)})
    model = SuperLSS(
        family,
        *(bind_predictor(family, p.name) for p in family.parameters),
        weight_semantics="frequency",
    )
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", SeparationWarning)
        model.fit(frame, response, censored=side, max_inner_iter=1, inner_tol=1e-12)
    assert not model.result_.coefficient_converged
    assert np.all(np.isfinite(model.predict_parameters(frame)))
    assert all("fit not converged" in note for note in model.summary()["note"])
    if hasattr(family, "censored_recession"):
        assert any(isinstance(w.message, SeparationWarning) for w in recorded)


@pytest.mark.parametrize("law", ["nb", "tweedie"])
@pytest.mark.parametrize("semantics", ["prior", "frequency"])
def test_atoms_keep_exact_left_right_and_open_lower_events_distinct(law, semantics):
    case = next(c for c in CASES if c.law == law)
    family = case.family
    response = np.array([0.0, 0.0, 0.0, 1.0])
    lower = np.array([-np.inf, -np.inf, -np.inf, 0.0])
    upper = np.full(4, np.inf)
    side = np.array([0, -1, 1, 0])
    bounds = ResponseBounds.from_arrays(lower, upper, side)
    plan = family.bind_likelihood(
        response, weights(np.full(4, 2), semantics), bounded_observation(bounds)
    )
    theta = np.tile(case.theta, (4, 1))
    actual = family.evaluate_natural(response, theta, plan)
    assert actual.valid.all()
    prior, mass = (2, 1) if semantics == "prior" else (1, 2)
    for row in range(4):
        expected = ROW_JET(
            law,
            case.form,
            float(response[row]),
            float(lower[row]),
            float(upper[row]),
            int(side[row]),
            case.theta,
            prior=prior,
            mass=mass,
        )
        assert_channels(tuple(c[row] for c in channels(actual)), expected)
    np.testing.assert_allclose(
        actual.reported_log_likelihood[0], actual.reported_log_likelihood[1], atol=1e-12
    )
    assert actual.reported_log_likelihood[1] != actual.reported_log_likelihood[2]
    with pytest.raises(ValueError):
        family.bind_likelihood(
            np.array([0.0]),
            weights([1]),
            bounded_observation(
                ResponseBounds.from_arrays(np.array([0.0]), np.array([np.inf]), np.array([-1]))
            ),
        )


def test_nb_prior_threshold_uses_the_represented_count_rate_lattice():
    case = next(c for c in CASES if c.law == "nb")
    threshold = 2 / 3
    y = np.array([np.nextafter(threshold, -np.inf), threshold, np.nextafter(threshold, np.inf)])
    for side in (-1, 1):
        bounds = ResponseBounds.from_arrays(
            np.full(3, -np.inf), np.full(3, np.inf), np.full(3, side)
        )
        plan = case.family.bind_likelihood(
            y, weights([3, 3, 3], "prior"), bounded_observation(bounds)
        )
        result = case.family.evaluate_natural(y, np.tile(case.theta, (3, 1)), plan)
        assert result.valid.all()
        for row in range(3):
            expected = jet(
                lambda *t: contribution(
                    "nb", "location", y[row], -np.inf, np.inf, side, t, prior=3
                ),
                case.theta,
            )
            assert_channels(tuple(c[row] for c in channels(result)), expected)
        assert result.reported_log_likelihood[0] != result.reported_log_likelihood[1]
        np.testing.assert_array_equal(
            result.reported_log_likelihood[1], result.reported_log_likelihood[2]
        )


@pytest.mark.parametrize("case", [CASES[0], *OLDER_POSITIVE], ids=lambda c: f"{c.law}-{c.form}")
@pytest.mark.parametrize("bad", [True, "0.5", 1j, np.nan, np.inf])
def test_censor_limits_remain_finite_real_raw_values_with_original_provenance(case, bad):
    w = weights([0, 1, 0, 1]).take(np.array([1, 0]))
    response = np.array([bad, 1.0], dtype=object)
    bounds = ResponseBounds.from_arrays(np.full(2, -np.inf), np.full(2, np.inf), np.array([1, 0]))
    with pytest.raises(ValueError, match="input row 3"):
        case.family.bind_likelihood(response, w, bounded_observation(bounds))
