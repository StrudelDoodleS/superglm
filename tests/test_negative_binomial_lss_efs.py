"""Public automatic-smoothing and Poisson-boundary behavior for NB2 LSS."""

from __future__ import annotations

import math
import warnings
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
from scipy.stats import nbinom, poisson

import superglm.distributional.families.negative_binomial as negative_binomial_family
from superglm import SuperLSS, cat, s
from superglm.diagnostics.convergence import ConvergenceWarning
from superglm.distributional import NegativeBinomialLS, Predictor
from superglm.distributional.weights import WeightContract, resolve_likelihood_weights
from superglm.features import Spline
from superglm.types import LambdaPolicy
from tests.bound_predictor_fixtures import model_from_templates

_INITIAL_LAMBDAS = {
    "mean:x_mean#wiggle": 0.8,
    "theta:x_theta#wiggle": 1.1,
}


def _overdispersed_fixture(n_rows: int = 480) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(2026083103)
    axis = np.linspace(-1.0, 1.0, n_rows)
    x_mean = rng.permutation(axis)
    x_theta = rng.permutation(axis)
    mean = np.exp(0.55 + 0.50 * np.sin(np.pi * x_mean) + 0.12 * x_mean)
    theta = np.exp(0.05 + 0.40 * np.cos(np.pi * x_theta) - 0.12 * x_theta)
    response = nbinom.rvs(
        n=theta,
        p=theta / (mean + theta),
        random_state=rng,
    ).astype(np.float64)
    return pd.DataFrame({"x_mean": x_mean, "x_theta": x_theta}), response


def _poisson_like_fixture(n_rows: int = 256) -> tuple[pd.DataFrame, np.ndarray]:
    probabilities = (np.arange(n_rows, dtype=np.float64) + 0.5) / n_rows
    response = poisson.ppf(probabilities, mu=2.0).astype(np.float64)
    rng = np.random.default_rng(2026083106)
    axis = np.linspace(-1.0, 1.0, n_rows)
    frame = pd.DataFrame(
        {
            "x_mean": rng.permutation(axis),
            "x_theta": rng.permutation(axis),
        }
    )
    return frame, response[rng.permutation(n_rows)]


def _model() -> SuperLSS:
    return model_from_templates(
        family=NegativeBinomialLS(),
        predictors=tuple(
            Predictor(
                name,
                {
                    f"x_{name}": Spline(
                        kind="cr",
                        n_knots=5,
                        knot_strategy="quantile_rows",
                        lambda_policy=LambdaPolicy.estimate(),
                    )
                },
            )
            for name in ("mean", "theta")
        ),
    )


def _fit_reml(
    model: SuperLSS,
    frame: pd.DataFrame,
    response: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> SuperLSS:
    return model.fit_reml(
        frame,
        response,
        sample_weight=sample_weight,
        lambdas=_INITIAL_LAMBDAS,
        max_reml_iter=100,
        reml_tol=1.0e-6,
        max_log_step=1.0,
        max_inner_iter=100,
        inner_tol=1.0e-8,
        practical_reml=False,
    )


def test_public_fit_reml_accepts_updates_and_publishes_observed_inference() -> None:
    frame, response = _overdispersed_fixture()
    model = _model()

    assert _fit_reml(model, frame, response) is model

    result = model.result_
    fitted = model._require_fitted()
    smoothing = fitted.smoothing
    assert smoothing is not None
    assert any(iteration.accepted for iteration in smoothing.history)
    assert result.smoothing_converged is True
    assert smoothing.matched_certified is True
    assert result.n_smoothing_iter > 0
    assert np.all(np.isfinite(model.covariance_))
    assert np.isfinite(result.total_effective_df)
    assert all(np.isfinite(value) for value in result.predictor_edf.values())

    terminal = fitted.result
    assert terminal.terminal_curvature.requested_source == "observed"
    assert terminal.terminal_curvature.actual_source == "observed"
    assert terminal.terminal_curvature.fallback_count == 0
    assert np.all(np.isfinite(terminal.terminal_data_curvature))
    assert np.all(np.isfinite(terminal.terminal_penalized_curvature))
    reconstructed = terminal.terminal_data_curvature + terminal.penalty
    residual_norm = float(
        np.linalg.norm(terminal.terminal_penalized_curvature - reconstructed, ord=np.inf)
    )
    dimension = terminal.terminal_penalized_curvature.shape[0]
    scale = max(
        float(np.linalg.norm(terminal.terminal_penalized_curvature, ord=np.inf)),
        float(np.linalg.norm(terminal.terminal_data_curvature, ord=np.inf))
        + float(np.linalg.norm(terminal.penalty, ord=np.inf)),
        np.finfo(np.float64).tiny,
    )
    bound = 8.0 * dimension * np.finfo(np.float64).eps * scale
    assert residual_norm <= bound
    assert result.rank == terminal.terminal_rank.rank == len(result.coefficients)


@pytest.mark.parametrize("dropped_rows", [0, 3])
def test_poisson_like_fit_finishes_and_discloses_the_boundary(dropped_rows: int) -> None:
    """Poisson-like theta non-identification is disclosed, not refused (it used to raise).

    The fit returns, its final curvature records the fallback it took, and the
    family's Poisson-boundary diagnosis is warned and kept in result_.warnings.
    Zero-weight rows are dropped before fitting; the diagnosis still reads the
    retained rows at their input positions.
    """
    frame, response = _poisson_like_fixture()
    assert float(np.var(response)) <= float(np.mean(response))
    weights = None
    if dropped_rows:
        frame = pd.concat([frame.iloc[:dropped_rows], frame], ignore_index=True)
        response = np.r_[np.full(dropped_rows, 40.0), response]
        weights = np.r_[np.zeros(dropped_rows), np.ones(len(response) - dropped_rows)]

    with pytest.warns(ConvergenceWarning, match="Poisson-like at the diagnostic boundary"):
        model = _fit_reml(_model(), frame, response, weights)

    telemetry = model._require_fitted().fit_state.solver_result.terminal_curvature
    assert telemetry.requested_source == "observed"
    assert telemetry.fallback_count >= 1
    assert telemetry.actual_source in ("fisher", "nearest_semidefinite")
    assert any(
        "could not establish a stable finite theta" in message for message in model.result_.warnings
    )
    assert np.all(np.isfinite(model.covariance_))


def _family_diagnosis(family, response, *, sample_weight, contract):
    """The family's own diagnosis, which a fit reaches through ``_curvature_disclosures``."""
    weights = resolve_likelihood_weights(
        sample_weight, n_observations=len(response), contract=contract
    )
    return family.diagnose_repeated_curvature_failure(response, weights)


def test_finite_overdispersion_does_not_classify_a_curvature_failure_as_poisson_like() -> None:
    response = np.array([0.0, 0.0, 1.0, 2.0, 4.0, 8.0])
    assert float(np.var(response)) > float(np.mean(response))

    result = _family_diagnosis(
        NegativeBinomialLS(),
        response,
        sample_weight=None,
        contract=WeightContract("prior"),
    )

    assert result is None


def test_nb2_diagnosis_drops_zero_weight_rows_before_response_validation() -> None:
    response = np.array([0.5, 1.0, 1.0, 1.0])
    sample_weight = np.array([0.0, 1.0, 1.0, 1.0])

    result = _family_diagnosis(
        NegativeBinomialLS(),
        response,
        sample_weight=sample_weight,
        contract=WeightContract("prior"),
    )

    assert type(result) is negative_binomial_family.NegativeBinomialPoissonBoundaryError


def test_nb2_diagnosis_keeps_the_exact_family_guard() -> None:
    class DerivedNegativeBinomialLS(NegativeBinomialLS):
        pass

    result = _family_diagnosis(
        DerivedNegativeBinomialLS(),
        np.array([0.5, 1.0, 1.0, 1.0]),
        sample_weight=np.array([0.0, 1.0, 1.0, 1.0]),
        contract=WeightContract("prior"),
    )

    assert result is None


def test_prior_moment_direction_does_not_override_positive_poisson_face_direction() -> None:
    response = np.array([5.5, 2.0])
    sample_weight = np.array([2.0, 1.0])
    initial_mean = 13.0 / 3.0
    moment_direction = math.fsum(
        float(weight * (value - initial_mean) ** 2 - initial_mean)
        for value, weight in zip(response, sample_weight, strict=True)
    )
    boundary_direction = math.fsum(
        float(weight * (value - initial_mean) ** 2 - value)
        for value, weight in zip(response, sample_weight, strict=True)
    )

    np.testing.assert_array_equal(sample_weight * response, [11.0, 2.0])
    assert moment_direction == pytest.approx(-0.5)
    assert boundary_direction == pytest.approx(2.0 / 3.0)
    result = _family_diagnosis(
        NegativeBinomialLS(),
        response,
        sample_weight=sample_weight,
        contract=WeightContract("prior"),
    )

    assert result is None


def test_cancellation_limited_frequency_moment_direction_fails_closed() -> None:
    large_weight = 2**52
    response = np.array([0.0, 2.0])
    sample_weight = np.array([large_weight, large_weight - 1], dtype=np.float64)
    exact_mean = Fraction(2 * (large_weight - 1), 2 * large_weight - 1)
    exact_direction = large_weight * ((Fraction(0) - exact_mean) ** 2 - exact_mean) + (
        large_weight - 1
    ) * ((Fraction(2) - exact_mean) ** 2 - exact_mean)
    exact_numerator = (2 * large_weight - 1) * exact_mean**2

    assert exact_direction > 0
    assert exact_numerator / exact_direction == 2 * large_weight - 2
    result = _family_diagnosis(
        NegativeBinomialLS(),
        response,
        sample_weight=sample_weight,
        contract=WeightContract("frequency"),
    )
    assert result is None

    result = _family_diagnosis(
        NegativeBinomialLS(),
        response,
        sample_weight=np.ones(2),
        contract=WeightContract("frequency"),
    )
    assert result is None


def _claims_book_fixture(seed: int = 7, n_rows: int = 1500):
    """A small synthetic motor book shaped like freMTPL2: exposure offsets and sparse tails.

    Vehicle age has a geometric tail, exposure is short for new vehicles, and
    claims are rare, so on this seed master's terminal observed curvature stayed
    indefinite after its retry and the fit raised
    RepeatedCurvatureIndefinitenessError.
    """
    rng = np.random.default_rng(seed)
    vehicle_age = np.minimum(rng.geometric(0.12, n_rows) - 1, 99).astype(np.float64)
    driver_age = rng.uniform(18.0, 90.0, n_rows)
    bonus = np.minimum(50 + rng.geometric(0.15, n_rows) - 1, 230).astype(np.float64)
    exposure = np.clip(rng.beta(0.9, 0.7, n_rows), 0.0027, 1.0)
    exposure = np.where(vehicle_age == 0, exposure * 0.4, exposure).clip(0.0027, 1.0)
    brand = rng.choice([f"B{k}" for k in range(11)], n_rows, p=np.r_[np.full(10, 0.0995), 0.005])
    gas = rng.choice(["Diesel", "Regular"], n_rows)
    area = rng.choice(list("ABCDEF"), n_rows)
    eta = (
        -2.3
        + 0.2 * np.sin(driver_age / 15.0)
        - 0.02 * vehicle_age
        + 0.006 * (bonus - 50.0)
        + np.where(gas == "Diesel", 0.1, 0.0)
    )
    mean = exposure * np.exp(eta)
    size = exposure * np.exp(0.3)
    counts = rng.negative_binomial(size, size / (size + mean)).astype(np.float64)
    frame = pd.DataFrame(
        {
            "DrivAge": driver_age,
            "VehAge": vehicle_age,
            "BonusMalus": bonus,
            "VehBrand": brand,
            "VehGas": gas,
            "Area": area,
        }
    )
    return frame, counts, np.log(exposure)


def _claims_book_model() -> SuperLSS:
    family = NegativeBinomialLS()
    return SuperLSS(
        family,
        family.mean(
            s("DrivAge", kind="cr", k=10),
            s("VehAge", kind="cr", k=10),
            s("BonusMalus", kind="cr", k=10),
            cat("VehBrand"),
            cat("VehGas"),
            cat("Area"),
        ),
        family.theta(s("DrivAge", kind="cr", k=5), cat("VehGas"), cat("Area")),
    )


def test_claims_book_shaped_fit_finishes_instead_of_refusing() -> None:
    """Kills the refusal on ordinary-looking data: the fit returns, predicts and discloses."""
    frame, counts, log_exposure = _claims_book_fixture()
    model = _claims_book_model()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit_reml(frame, counts, offsets={"mean": log_exposure, "theta": log_exposure})

    assert np.all(np.isfinite(model.covariance_))
    offsets = {"mean": log_exposure, "theta": log_exposure}
    assert np.all(np.isfinite(np.asarray(model.predict(frame, offsets=offsets))))
    if not model.result_.converged:
        assert any(issubclass(item.category, ConvergenceWarning) for item in caught)
    # Every NB2 final-curvature fallback used to refuse, so each one is disclosed.
    if model._require_fitted().fit_state.solver_result.terminal_curvature.fallback_count > 0:
        assert model.result_.warnings
    for message in model.result_.warnings:
        assert any(str(item.message) == message for item in caught)


def test_offset_start_is_projected_with_the_starting_penalty(monkeypatch) -> None:
    """Kills an unpenalized starting projection of an offset-shifted target.

    Penalized least squares cannot pay more penalty than the intercept-only
    start's residual sum of squares, which is feasible at zero penalty. The
    unpenalized projection let the vehicle-age spline absorb -log(exposure)
    with a starting penalty 7.6 times that bound on this seed (and 2.3e10
    against a log likelihood of 3.8e3 on the 20,000-row freMTPL2 sample).
    """
    import superglm.distributional.solver.solver as solver_module

    starts = []
    original = solver_module._initial_coefficients

    def record(context):
        coefficients = original(context)
        starts.append((context, coefficients))
        return coefficients

    monkeypatch.setattr(solver_module, "_initial_coefficients", record)
    frame, counts, log_exposure = _claims_book_fixture(seed=3, n_rows=4000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _claims_book_model().fit_reml(
            frame, counts, offsets={"mean": log_exposure, "theta": log_exposure}, max_reml_iter=1
        )

    context, coefficients = starts[0]
    initial = context.family.initialize(context.response, context.likelihood_plan).theta
    weights = context.likelihood_plan.weights.values
    state = context.layout.predictors[0]
    target = np.log(initial[:, 0]) - state.offset
    centred = target - np.sum(weights * target) / np.sum(weights)
    block = state.coefficient_slice
    start = coefficients[block]
    penalty = float(start @ context.penalty[block, block] @ start)
    residual = float(np.sum(weights * centred**2))
    assert penalty <= residual * (1.0 + 1.0e-8), (penalty, residual)


def _poisson_group_fixture() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """One group Poisson (its size heads to the Poisson limit), the other NB2 with size 2."""
    rng = np.random.default_rng(2)
    n_rows = 2000
    x, z = rng.uniform(0.0, 1.0, n_rows), rng.uniform(0.0, 1.0, n_rows)
    group = rng.integers(0, 2, n_rows)
    exposure = rng.uniform(0.2, 1.0, n_rows)
    mean = exposure * np.exp(-1.5 + np.sin(2.0 * np.pi * x))
    response = np.where(
        group == 0,
        rng.poisson(mean),
        rng.negative_binomial(2.0, 2.0 / (2.0 + mean)),
    ).astype(np.float64)
    frame = pd.DataFrame({"x": x, "z": z, "g": pd.Categorical(group.astype(str))})
    return frame, response, np.log(exposure)


def test_size_reaching_its_poisson_limit_during_selection_still_converges() -> None:
    """A rank drop from a size reaching its Poisson limit is not an objective increase.

    The Poisson group's size direction loses its curvature as it heads to the
    limit, and the terminal rank drops between smoothing iterations. Compared
    as computed, the pseudo-determinant over the smaller subspace read as a
    worse criterion and every later proposal was rejected (objective_rejected);
    compared over the common subspace, selection converges.
    """
    frame, response, log_exposure = _poisson_group_fixture()
    family = NegativeBinomialLS()
    model = SuperLSS(
        family,
        family.mean(s("x", kind="cr", k=8)),
        family.theta(s("z", kind="cr", k=5), cat("g")),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(frame, response, offsets={"mean": log_exposure})

    smoothing = model._require_fitted().fit_state.smoothing
    ranks = {fit.terminal_rank.rank for fit in smoothing.coefficient_fits}
    if len(ranks) == 1:
        # The drop sits at a rank cutoff that another BLAS build can round past;
        # the forced unit test in test_distributional_efs.py covers the comparison.
        pytest.skip("this build's trajectory never dropped rank")
    assert smoothing.converged, smoothing.convergence_reason


def test_fisher_fallback_on_overdispersed_data_is_disclosed(monkeypatch) -> None:
    """An NB2 fit that ends on Fisher information says so, as master's refusal did.

    The observed size curvature is sign-flipped at every state, so the final
    observed curvature stays indefinite after the retry and the policy accepts
    the expected information. The data are overdispersed, so the Poisson-boundary
    diagnosis does not fire; the parent then disclosed nothing at all.
    """
    from dataclasses import replace

    evaluate = NegativeBinomialLS.evaluate_natural

    def indefinite(self, y, theta, plan, *, derivative_order=2):
        result = evaluate(self, y, theta, plan, derivative_order=derivative_order)
        if result.hessian_packed is None:
            return result
        hessian = np.array(result.hessian_packed, copy=True)
        hessian[:, 2] = -hessian[:, 2]
        return replace(result, hessian_packed=hessian)

    monkeypatch.setattr(NegativeBinomialLS, "evaluate_natural", indefinite)
    frame, response = _overdispersed_fixture()
    assert float(np.var(response)) > float(np.mean(response))
    model = _model()
    with pytest.warns(ConvergenceWarning, match="expected \\(Fisher\\) information"):
        model.fit(frame, response, lambdas=_INITIAL_LAMBDAS, max_inner_iter=20)

    telemetry = model._require_fitted().fit_state.solver_result.terminal_curvature
    assert telemetry.actual_source == "fisher"
    assert telemetry.fallback_count >= 1
    assert any("expected (Fisher) information" in message for message in model.result_.warnings)
    assert not any("Poisson-like" in message for message in model.result_.warnings)


_EPS = float(np.finfo(np.float64).eps)
# (prior weight, count, mean, size relative to the row's Poisson-limit boundary, at the limit).
# A count-dominated row is at the limit when count - 1 < 2**-52 * w * theta, a
# mean-dominated one when w * mean < 2**-52 * w * theta; the sizes straddle each
# boundary so the expected count is exact by construction, and the weights are
# not 1 so that the count, the weight semantics and the weights all matter. The
# rows that each wrong reading moves are unequal in number, so no pair of
# mistakes cancels in the total.
_LIMIT_ROWS = (
    (0.5, 500, 1.0, 1.5, True),
    (0.5, 500, 1.0, 0.7, False),
    (0.5, 500, 1.0, 0.6, False),
    (2.0, 500, 1.0, 1.5, True),
    (2.0, 500, 1.0, 0.7, False),
    (2.0, 500, 1.0, 0.6, False),
    (2.0, 0, 10.0, 2.0, True),
    (2.0, 0, 10.0, 0.5, False),
    (1.0, 0, 0.1, 1.0, False),
)


def _forced_limit_rows():
    """Terminal parameters and a prior-weighted plan whose Poisson-limit rows are known."""
    from superglm.distributional.family import COMPLETE_OBSERVATION

    weights = np.array([row[0] for row in _LIMIT_ROWS])
    counts = np.array([row[1] for row in _LIMIT_ROWS], dtype=np.float64)
    mean = np.array([row[2] for row in _LIMIT_ROWS])
    reach = np.maximum(weights * mean, np.maximum(counts - 1.0, 0.0))
    boundary = reach / (_EPS * weights)
    theta = np.column_stack([mean, boundary * np.array([row[3] for row in _LIMIT_ROWS])])
    # Prior weights: the response is the weighted count divided by the weight.
    response = counts / weights
    resolved = resolve_likelihood_weights(
        weights, n_observations=len(weights), contract=WeightContract(semantics="prior")
    )
    plan = NegativeBinomialLS().bind_likelihood(response, resolved, COMPLETE_OBSERVATION)
    return theta, response, plan, sum(row[4] for row in _LIMIT_ROWS)


def test_forced_poisson_limit_rows_are_counted_under_prior_weights() -> None:
    """The count is exact for terminal parameters placed across the limit.

    Nothing here depends on where a solver stops. The expected count is
    fixed by construction, and the weights are not 1, so reading the response
    in place of the weighted count, a different weight semantics or unit weights
    each moves a row across its boundary.
    """
    theta, _, plan, expected = _forced_limit_rows()
    family = NegativeBinomialLS()
    assert family.limit_row_count(theta, plan) == (1, expected)
    assert (
        negative_binomial_family.poisson_limit_row_count(
            theta[:, 0], theta[:, 1], plan.weights.values, "prior", plan.exact_count
        )
        == expected
    )
    below = theta.copy()
    below[:, 1] = 1.0
    assert family.limit_row_count(below, plan) == (1, 0)


def test_limit_row_message_names_the_remedy_the_size_can_take() -> None:
    """A constant size is offered only when one is identified, with what it must drop."""
    say = NegativeBinomialLS().limit_row_disclosure
    both = say(3, 9, True, True, True)
    terms = say(3, 9, True, True, False)
    offset = say(3, 9, True, False, True)
    none = say(3, 9, False, True, True)
    assert "3 of 9 rows have a size at the Poisson limit" in both
    assert "singular in a size direction" in both and "not identified" in both
    assert "(family.theta() with no terms and no size offset) or the Poisson family" in both
    assert "(family.theta() with no terms) or the Poisson family" in terms
    assert "(family.theta() with no size offset) or the Poisson family" in offset
    assert "Fit a constant size" not in none and "use the Poisson family instead" in none


def _gate_fit(size_terms: tuple, size_offset: np.ndarray | None = None):
    """A real 240-row fit whose layout and coefficient order the gate reads."""
    frame, response = _overdispersed_fixture(240)
    frame["g"] = pd.Categorical(np.where(np.arange(240) % 2 == 0, "a", "b"))
    family = NegativeBinomialLS()
    model = SuperLSS(family, family.mean(s("x_mean", kind="cr", k=5)), family.theta(*size_terms))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(
            frame, response, offsets=None if size_offset is None else {"theta": size_offset}
        )
    return frame, model._require_fitted().fit_state


def _forced_plan(n: int):
    from superglm.distributional.family import COMPLETE_OBSERVATION

    resolved = resolve_likelihood_weights(
        np.ones(n), n_observations=n, contract=WeightContract(semantics="prior")
    )
    return NegativeBinomialLS().bind_likelihood(np.zeros(n), resolved, COMPLETE_OBSERVATION)


def _size_state(fit_state):
    return next(p for p in fit_state.layout.predictors if p.parameter_index == 1)


def _forced_entries(
    fit_state,
    at_limit: np.ndarray,
    *,
    null: tuple[int, ...] = (),
    intercept_curvature: float = 1.0,
) -> list[str]:
    """The disclosure the fit's gate makes for forced limit rows and a forced terminal rank.

    The mean is 1 and the count 0 everywhere, so a row is at the limit when its
    size exceeds ``2**52``. ``null`` lists the coefficient indices the forced
    terminal curvature is singular in, and ``intercept_curvature``
    is the size intercept's terminal data curvature; with ``null`` empty the
    fit's own terminal rank is used.
    """
    from types import SimpleNamespace

    from superglm.distributional.fit_state import _curvature_disclosures
    from superglm.solvers.rank import decompose_gram

    solved = fit_state.solver_result
    n = len(at_limit)
    theta = np.column_stack([np.ones(n), np.where(at_limit, 1e17, 2.0)])
    width = fit_state.layout.n_coefficients
    diagonal = np.ones(width)
    diagonal[list(null)] = 0.0
    data = np.eye(width)
    size = _size_state(fit_state)
    data[size.intercept_index, size.intercept_index] = intercept_curvature
    result = SimpleNamespace(
        terminal_curvature=solved.terminal_curvature,
        theta=theta,
        terminal_rank=decompose_gram(np.diag(diagonal)) if null else solved.terminal_rank,
        terminal_data_curvature=data,
    )
    messages = _curvature_disclosures(
        NegativeBinomialLS(), result, np.zeros(n), _forced_plan(n), fit_state.layout
    )
    return [m for m in messages if "rows have a size at the" in m]


def test_limit_rows_with_a_nonsingular_terminal_curvature_are_not_disclosed() -> None:
    """Rows without information do not make the size unidentified while others identify it.

    The review's case in forced form: half the rows are at the limit (their mean
    is negligible next to the size) and the fit's own terminal curvature is
    nonsingular, because a constant size is identified by the other rows.
    """
    _, fit_state = _gate_fit(())
    at_limit = np.arange(240) < 120
    theta = np.column_stack([np.ones(240), np.where(at_limit, 1e17, 2.0)])
    assert NegativeBinomialLS().limit_row_count(theta, _forced_plan(240)) == (1, 120)
    assert _forced_entries(fit_state, at_limit) == []


def test_negligible_mean_rows_leave_a_real_constant_size_fit_silent() -> None:
    """A real fit: ten rows with a negligible mean trip the kernel's mask, nothing is disclosed.

    The size is constant and the other rows are overdispersed, so the fit's
    terminal curvature is nonsingular; the mask fires regardless of where the
    solver stops, because exp(-60) is below ``2**-52`` times any size above 1e-10.
    """
    rng = np.random.default_rng(42)
    n = 400
    offset = np.zeros(n)
    offset[:10] = -60.0
    response = rng.negative_binomial(1.0, 1.0 / (1.0 + np.exp(offset))).astype(np.float64)
    family = NegativeBinomialLS()
    model = SuperLSS(family, family.mean(), family.theta())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(
            pd.DataFrame({"x": np.arange(n, dtype=np.float64)}), response, offsets={"mean": offset}
        )
    solved = model._require_fitted().fit_state.solver_result
    _, count = family.limit_row_count(solved.theta, _forced_plan(n))
    assert count >= 1
    assert not any("rows have a size at the" in m for m in model.result_.warnings)


def test_every_row_at_the_limit_offers_only_the_poisson_family() -> None:
    """No row carries size curvature: the constant-size remedy would be false."""
    _, fit_state = _gate_fit(())
    (entry,) = _forced_entries(
        fit_state,
        np.ones(240, dtype=bool),
        null=(_size_state(fit_state).intercept_index,),
        intercept_curvature=0.0,
    )
    assert "240 of 240 rows have a size at the Poisson limit" in entry
    assert "use the Poisson family instead" in entry
    assert "Fit a constant size" not in entry


@pytest.mark.parametrize("offset_varies", [False, True])
def test_a_size_coefficient_the_terminal_rank_cannot_estimate_is_disclosed(
    offset_varies: bool,
) -> None:
    """A size indicator the terminal curvature is singular in is reported with its remedy.

    The size has a group indicator and the forced terminal rank leaves it not
    estimable, while the size intercept still has curvature: the remedy names
    the constant size and what it drops.
    """
    offset = np.where(np.arange(240) < 120, 0.0, 0.5) if offset_varies else None
    frame, fit_state = _gate_fit((cat("g"),), offset)
    at_limit = (frame["g"] == "a").to_numpy()
    indicator = _size_state(fit_state).coefficient_slice.start + 1
    (entry,) = _forced_entries(fit_state, at_limit, null=(indicator,))
    assert f"{int(at_limit.sum())} of 240 rows have a size at the Poisson limit" in entry
    assert "singular in a size direction" in entry
    dropped = "no terms and no size offset" if offset_varies else "no terms"
    assert f"Fit a constant size (family.theta() with {dropped}) or the Poisson family" in entry


def test_a_singular_direction_outside_the_size_block_is_not_disclosed() -> None:
    """The gate attributes the null direction: one in the mean block says nothing about the size."""
    frame, fit_state = _gate_fit((cat("g"),))
    at_limit = (frame["g"] == "a").to_numpy()
    mean_coefficient = fit_state.layout.predictors[0].coefficient_slice.start + 1
    assert _forced_entries(fit_state, at_limit, null=(mean_coefficient,)) == []


def test_no_limit_rows_are_not_disclosed_whatever_the_rank() -> None:
    """The count is a precondition: a singular size block without limit rows is not this finding."""
    frame, fit_state = _gate_fit((cat("g"),))
    indicator = _size_state(fit_state).coefficient_slice.start + 1
    assert _forced_entries(fit_state, np.zeros(240, dtype=bool), null=(indicator,)) == []


def test_failed_limit_count_does_not_fail_a_finished_fit(monkeypatch) -> None:
    """A diagnostic count that raises leaves the finished fit and its other warnings."""

    def broken(*args, **kwargs):
        raise RuntimeError("count unavailable")

    monkeypatch.setattr(negative_binomial_family, "poisson_limit_row_count", broken)
    frame, response = _overdispersed_fixture(240)
    model = _fit_reml(_model(), frame, response)
    assert model.result_ is not None
    assert not any("rows have a size at the" in m for m in model.result_.warnings)


def test_size_at_the_poisson_limit_is_disclosed_with_its_row_count() -> None:
    """A fit reports the rows at the Poisson limit exactly when there are any.

    One group is Poisson, so its size tends to the limit and the size
    coefficients are not identified there. This is integration coverage of the
    public fit: whether the solver's last iterate crosses the kernel's
    threshold depends on its trajectory, so the entry is required to agree
    with a recount from the fitted rows (max(mean, count - 1) below 2**-52 times
    the size), present when the recount is positive and absent when it is not.
    Here the entry also needs the size information to be singular, which the
    Poisson group makes so: its rows beyond the recount sit just short of the
    threshold with information far below the policy's rank cutoff. The event
    itself is forced in the tests above.
    """
    frame, response, log_exposure = _poisson_group_fixture()
    family = NegativeBinomialLS()
    model = SuperLSS(
        family,
        family.mean(s("x", kind="cr", k=8)),
        family.theta(s("z", kind="cr", k=5), cat("g")),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit_reml(frame, response, offsets={"mean": log_exposure})

    fitted = model._require_fitted().fit_state.solver_result.theta
    reach = np.maximum(fitted[:, 0], np.maximum(response - 1.0, 0.0))
    expected = int(np.count_nonzero(reach < _EPS * fitted[:, 1]))
    for messages in ([str(w.message) for w in caught], list(model.result_.warnings)):
        entries = [m for m in messages if "rows have a size at the" in m]
        if expected == 0:
            assert not entries
            continue
        (entry,) = entries
        assert f"{expected} of {len(response)} rows have a size at the Poisson limit" in entry
        assert "constant size" in entry and "Poisson family" in entry

    frame, response = _overdispersed_fixture()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        overdispersed = _fit_reml(_model(), frame, response)
    assert not any("rows have a size at the" in m for m in overdispersed.result_.warnings)
