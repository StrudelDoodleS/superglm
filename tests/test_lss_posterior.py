"""Contract tests for the LSS posterior-simulation primitive."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import special, stats

from superglm import Categorical, Spline, SuperLSS
from superglm._frame import as_eager_frame
from superglm.distributional import Predictor
from superglm.distributional.families.gamma import GammaLS
from superglm.distributional.families.gaussian import GaussianLS
from superglm.distributional.model import DenseDistributionalModel
from superglm.distributional.posterior import (
    PosteriorDraws,
    posterior_bounds,
    posterior_covariance,
    posterior_draws,
    posterior_parameters,
    posterior_predictive,
    resolve_quantity,
    simultaneous_critical_value,
)


def _simulated(n: int = 1500, seed: int = 20260903) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    g = rng.choice(["a", "b", "c"], n)
    location = 0.6 * np.sin(2.4 * x) + np.where(g == "a", 0.3, np.where(g == "b", -0.2, 0.0))
    scale = np.exp(-1.0 + 0.5 * np.cos(1.8 * x))
    return pd.DataFrame({"x": x, "g": g}), location + scale * rng.standard_normal(n)


def _predictors() -> list[Predictor]:
    return [
        Predictor("location", {"x": Spline("cr", k=8), "g": Categorical()}),
        Predictor("scale", {"x": Spline("cr", k=6)}),
    ]


@pytest.fixture(scope="module")
def fit_case() -> tuple[DenseDistributionalModel, pd.DataFrame, np.ndarray]:
    X, y = _simulated()
    model = SuperLSS(family=GaussianLS(), predictors=_predictors()).fit_reml(X, y)
    return model._require_fitted(), X, y


@pytest.fixture(scope="module")
def weighted_gamma_case() -> tuple[DenseDistributionalModel, pd.DataFrame, np.ndarray]:
    """A gamma fit whose rows carry a prior weight, as an exposure would.

    The prior weight is part of the row's own law -- shape ``w / cv^2`` and
    scale ``mean cv^2 / w`` -- so the responses are drawn from that law rather
    than from the unit-weight one.
    """
    rng = np.random.default_rng(20260904)
    n = 600
    x = rng.uniform(-1.0, 1.0, n)
    weights = rng.uniform(0.2, 1.0, n)
    mean = np.exp(0.8 + 0.5 * np.sin(2.0 * x))
    cv2 = 0.36
    X = pd.DataFrame({"x": x})
    y = rng.gamma(weights / cv2, mean * cv2 / weights)
    model = SuperLSS(
        family=GammaLS(),
        predictors=[Predictor("mean", {"x": Spline("cr", k=6)}), Predictor("scale", {})],
    ).fit_reml(X, y, sample_weight=weights)
    return model._require_fitted(), X, weights


def _with_smoothing_hessian(fitted, hessian: np.ndarray) -> DenseDistributionalModel:
    """Republish the fit carrying a smoothing Hessian the endgame did not keep."""
    smoothing = dataclasses.replace(
        fitted.smoothing,
        smoothing_hessian=hessian,
        smoothing_hessian_certificate=np.zeros_like(hessian),
    )
    state = dataclasses.replace(fitted.fit_state, smoothing=smoothing)
    return DenseDistributionalModel(family=fitted.family, _fit_state=state)


class _NoDistributionFunctionFamily:
    """A family exposing neither ``cdf`` nor ``quantile``."""

    def __init__(self, parameters) -> None:
        self.parameters = parameters


class _UnitLawOnlyFamily:
    """A family with the unit-weight distribution functions and no weighted pair."""

    def __init__(self, family) -> None:
        self._family = family

    @property
    def parameters(self):
        return self._family.parameters

    def cdf(self, y, theta):
        return self._family.cdf(y, theta)

    def quantile(self, p, theta):
        return self._family.quantile(p, theta)


class _FiniteEstimateInfiniteDrawFamily(_UnitLawOnlyFamily):
    """A named functional that is finite at the plug-in but not on every draw."""

    def __init__(self, family, estimate_rows: int) -> None:
        super().__init__(family)
        self._estimate_rows = estimate_rows

    def expected_shortfall(self, p, theta):
        values = np.asarray(theta, dtype=float)[:, 0].copy()
        if len(values) > self._estimate_rows:
            values[0] = np.inf
        return values

    def quantile(self, p, theta):
        # The unfixed generic quadrature reaches this method and therefore
        # observes the same non-finite draw that the family-owned method does.
        return self.expected_shortfall(p, theta)


def test_draws_come_from_the_fitted_bayesian_posterior(fit_case) -> None:
    fitted, _, _ = fit_case
    width = fitted.layout.n_coefficients
    drawn = posterior_draws(fitted, 500, seed=1)

    assert isinstance(drawn, PosteriorDraws)
    assert drawn.coefficients.shape == (500, width)
    assert drawn.n_draws == 500
    assert drawn.seed == 1
    assert drawn.covariance_kind == "fixed"
    assert drawn.coefficient_names == fitted.layout.coefficient_names
    assert not drawn.coefficients.flags.writeable

    covariance = np.asarray(fitted.inference.covariance)
    standard_error = np.sqrt(np.diag(covariance) / 500.0)
    centred = drawn.coefficients.mean(axis=0) - np.asarray(fitted.coefficients)
    assert np.all(np.abs(centred) < 4.0 * standard_error)
    sample = np.cov(drawn.coefficients, rowvar=False)
    assert np.linalg.norm(sample - covariance) / np.linalg.norm(covariance) < 0.25

    assert np.array_equal(posterior_draws(fitted, 500, seed=1).coefficients, drawn.coefficients)
    assert not np.array_equal(posterior_draws(fitted, 500, seed=2).coefficients, drawn.coefficients)


def test_corrected_covariance_refuses_a_compact_fit_without_a_smoothing_hessian(fit_case) -> None:
    fitted, _, _ = fit_case
    assert fitted.smoothing.smoothing_hessian is None
    compact_state = dataclasses.replace(fitted.fit_state, retained_rows=None)
    compact = DenseDistributionalModel(family=fitted.family, _fit_state=compact_state)

    with pytest.raises(RuntimeError, match="retained training rows"):
        posterior_covariance(compact, kind="corrected")
    with pytest.raises(ValueError, match="'fixed' or 'corrected'"):
        posterior_covariance(fitted, kind="bogus")


def test_corrected_covariance_adds_the_smoothing_parameter_term(fit_case) -> None:
    fitted, X, y = fit_case
    names = tuple(fitted.smoothing.terminal_gradient)
    assert names == tuple(fitted.smoothing.lambdas)

    republished = _with_smoothing_hessian(fitted, np.eye(len(names)))
    fixed = posterior_covariance(fitted)
    corrected = posterior_covariance(republished, kind="corrected")
    assert np.allclose(corrected, corrected.T, atol=0.0)
    assert np.all(np.linalg.eigvalsh(corrected - fixed) > -1.0e-12)
    assert np.all(np.diag(corrected) >= np.diag(fixed))

    # V_rho is the identity here, so the correction is exactly J J' with
    # J = d beta / d rho.  Check J against a central difference of the
    # fixed-lambda refit in log lambda, which never touches the implicit
    # function theorem the implementation uses.
    lambdas = dict(fitted.lambdas)
    delta = 1.0e-4
    columns = []
    for name in names:
        moved = []
        for step in (delta, -delta):
            trial = dict(lambdas)
            trial[name] = lambdas[name] * float(np.exp(step))
            refit = SuperLSS(family=GaussianLS(), predictors=_predictors()).fit(X, y, lambdas=trial)
            moved.append(np.asarray(refit._require_fitted().coefficients))
        columns.append((moved[0] - moved[1]) / (2.0 * delta))
    jacobian = np.column_stack(columns)
    np.testing.assert_allclose(corrected - fixed, jacobian @ jacobian.T, rtol=1e-4, atol=1e-8)

    assert posterior_draws(republished, 8, covariance="corrected").covariance_kind == "corrected"


def test_chunking_never_changes_the_pushforward(fit_case) -> None:
    fitted, X, _ = fit_case
    head = X.iloc[:40]
    drawn = posterior_draws(fitted, 32, seed=3)

    whole = np.concatenate(
        [block for _, block in posterior_parameters(fitted, head, drawn)], axis=1
    )
    chunked = np.concatenate(
        [block for _, block in posterior_parameters(fitted, head, drawn, chunk_rows=7)], axis=1
    )
    assert whole.shape == (32, 40, 2)
    assert np.array_equal(whole, chunked)

    rows = [row_slice for row_slice, _ in posterior_parameters(fitted, head, drawn, chunk_rows=7)]
    assert rows[0] == slice(0, 7)
    assert rows[-1] == slice(35, 40)

    # A draw matrix of repeated ``beta_hat`` must push forward to exactly the
    # plug-in parameters: the pushforward and ``predict_parameters`` are the
    # same map through the same designs, offsets and links.
    plug_in = PosteriorDraws(
        coefficients=np.tile(np.asarray(fitted.coefficients), (4, 1)),
        covariance_kind="fixed",
        seed=0,
        coefficient_names=tuple(fitted.layout.coefficient_names),
    )
    offsets = {"scale": np.linspace(-0.2, 0.2, len(head))}
    pushed = np.concatenate(
        [block for _, block in posterior_parameters(fitted, head, plug_in, offsets=offsets)],
        axis=1,
    )
    expected = fitted.predict_parameters(head, offsets=offsets)
    np.testing.assert_allclose(pushed[0], expected, rtol=0.0, atol=1e-12)
    assert np.array_equal(pushed[0], pushed[3])

    quantity = ("parameter", "scale")
    unchunked = posterior_bounds(fitted, head, quantity, draws=drawn)
    small = posterior_bounds(fitted, head, quantity, draws=drawn, chunk_rows=7)
    assert np.array_equal(unchunked.to_numpy(), small.to_numpy())


def test_parameter_bounds_bracket_the_plug_in_estimate(fit_case) -> None:
    fitted, X, _ = fit_case
    head = X.iloc[:200]
    bounds = posterior_bounds(fitted, head, ("parameter", "scale"), level=0.9, n_draws=400, seed=5)

    assert list(bounds.columns) == ["estimate", "mean", "sd", "lower", "upper"]
    assert bounds.index.equals(head.index)
    theta = fitted.predict_parameters(head)
    np.testing.assert_allclose(bounds["estimate"].to_numpy(), theta[:, 1])
    inside = (bounds["lower"] <= bounds["estimate"]) & (bounds["estimate"] <= bounds["upper"])
    assert float(inside.mean()) >= 0.95
    assert np.all(bounds["sd"].to_numpy() > 0.0)

    shifted, quantity_draws = posterior_bounds(
        fitted,
        head,
        ("parameter", "location"),
        n_draws=64,
        seed=5,
        offsets={"location": np.full(len(head), 0.25)},
        return_draws=True,
    )
    base = posterior_bounds(fitted, head, ("parameter", "location"), n_draws=64, seed=5)
    np.testing.assert_allclose(shifted["estimate"] - base["estimate"], 0.25)
    # The offset moves the drawn predictor too, not only the plug-in estimate.
    np.testing.assert_allclose(shifted["mean"] - base["mean"], 0.25)
    assert quantity_draws.shape == (64, len(head))
    np.testing.assert_allclose(quantity_draws.mean(axis=0), shifted["mean"].to_numpy())
    np.testing.assert_allclose(quantity_draws.std(axis=0, ddof=1), shifted["sd"].to_numpy())


def test_named_quantities_follow_the_family(fit_case) -> None:
    fitted, X, _ = fit_case
    head = X.iloc[:120]
    drawn = posterior_draws(fitted, 200, seed=6)

    median = posterior_bounds(fitted, head, ("quantile", 0.5), draws=drawn)
    extreme = posterior_bounds(fitted, head, ("quantile", 0.99), draws=drawn)
    width = lambda frame: float((frame["upper"] - frame["lower"]).mean())  # noqa: E731
    assert width(extreme) > width(median)

    location = posterior_bounds(fitted, head, ("parameter", "location"), draws=drawn)
    np.testing.assert_allclose(
        posterior_bounds(fitted, head, lambda theta: theta[:, 0], draws=drawn).to_numpy(),
        location.to_numpy(),
    )
    np.testing.assert_allclose(
        posterior_bounds(fitted, head, "mean", draws=drawn).to_numpy(), location.to_numpy()
    )

    probability = np.nextafter(1.0, 0.0)
    shortfall = posterior_bounds(fitted, head, ("expected_shortfall", probability), draws=drawn)
    tail = posterior_bounds(fitted, head, ("quantile", probability), draws=drawn)
    assert np.all(shortfall["estimate"].to_numpy() >= tail["estimate"].to_numpy())
    theta = fitted.predict_parameters(head)
    z = special.ndtri(probability)
    want = theta[:, 0] + theta[:, 1] * np.exp(-0.5 * z * z) / (
        np.sqrt(2.0 * np.pi) * (1.0 - probability)
    )
    np.testing.assert_allclose(
        shortfall["estimate"].to_numpy(),
        want,
        rtol=16.0 * np.finfo(np.float64).eps,
    )

    cdf = posterior_bounds(fitted, head, ("cdf", 0.4), draws=drawn)
    exceedance = posterior_bounds(fitted, head, ("exceedance", 0.4), draws=drawn)
    np.testing.assert_allclose(cdf["estimate"].to_numpy() + exceedance["estimate"].to_numpy(), 1.0)

    with pytest.raises(ValueError, match="quantity"):
        resolve_quantity(fitted.family, ("nonsense", 1.0))
    with pytest.raises(ValueError, match="unknown parameter"):
        resolve_quantity(fitted.family, ("parameter", "shape"))
    with pytest.raises(NotImplementedError, match="quantile"):
        resolve_quantity(_NoDistributionFunctionFamily(fitted.family.parameters), ("quantile", 0.5))
    with pytest.raises(NotImplementedError, match="expected shortfall"):
        resolve_quantity(_UnitLawOnlyFamily(fitted.family), ("expected_shortfall", 0.5))


def test_named_posterior_bounds_refuse_non_finite_functional_draws(fit_case) -> None:
    fitted, X, _ = fit_case
    head = X.iloc[:5]
    family = _FiniteEstimateInfiniteDrawFamily(fitted.family, len(head))
    stub = SimpleNamespace(
        family=family,
        predict_parameters=fitted.predict_parameters,
        layout=fitted.layout,
        compiled_predictors=fitted.compiled_predictors,
    )

    with pytest.raises(ValueError, match="non-finite.*posterior.*quantity"):
        posterior_bounds(
            stub,
            head,
            ("expected_shortfall", 0.9),
            draws=posterior_draws(fitted, 4, seed=13),
        )


def test_predictive_draws_simulate_the_response(fit_case) -> None:
    fitted, X, _ = fit_case
    head = X.iloc[:150]
    plug_in = posterior_predictive(fitted, head, 50, parameter_uncertainty=False, seed=7)
    assert plug_in.shape == (50, 150)

    theta = fitted.predict_parameters(head)
    tolerance = 3.0 * theta[:, 1] / np.sqrt(50.0)
    assert np.mean(np.abs(plug_in.mean(axis=0) - theta[:, 0]) <= tolerance) >= 0.95

    total = posterior_predictive(
        fitted,
        head,
        50,
        parameter_uncertainty=False,
        reduce=lambda block: block.sum(axis=1),
        seed=7,
    )
    assert total.shape == (50,)
    np.testing.assert_allclose(total, plug_in.sum(axis=1))
    np.testing.assert_allclose(
        posterior_predictive(fitted, head, 50, parameter_uncertainty=False, reduce="sum", seed=7),
        total,
    )

    joint = posterior_predictive(fitted, head, 40, seed=7)
    assert joint.shape == (40, 150)
    assert float(joint.std()) > 0.0


def test_a_family_without_a_quantile_refuses_to_simulate(fit_case) -> None:
    fitted, X, _ = fit_case
    stub = SimpleNamespace(family=_NoDistributionFunctionFamily(fitted.family.parameters))
    with pytest.raises(NotImplementedError, match="quantile"):
        posterior_predictive(stub, X.iloc[:5], 4)


def test_prior_weights_put_every_named_quantity_on_the_row_law(weighted_gamma_case) -> None:
    fitted, X, weights = weighted_gamma_case
    head, w = X.iloc[:120], weights[:120]
    drawn = posterior_draws(fitted, 200, seed=6)
    family = fitted.family
    theta = np.asarray(fitted.predict_parameters(head))
    levels = np.full(len(head), 0.9)

    weighted = posterior_bounds(fitted, head, ("quantile", 0.9), draws=drawn, weights=w)
    unit = posterior_bounds(fitted, head, ("quantile", 0.9), draws=drawn)
    np.testing.assert_allclose(
        weighted["estimate"].to_numpy(), family.quantile_prior_weighted(levels, theta, w)
    )
    # An exposure below one spreads the gamma out at an unchanged mean, so the
    # row's own upper quantile sits above the unit-weight one everywhere.
    assert np.all(weighted["estimate"].to_numpy() > unit["estimate"].to_numpy())
    assert np.all(weighted["lower"].to_numpy() <= weighted["estimate"].to_numpy())
    assert np.all(weighted["estimate"].to_numpy() <= weighted["upper"].to_numpy())

    threshold = np.full(len(head), 2.0)
    cdf = posterior_bounds(fitted, head, ("cdf", 2.0), draws=drawn, weights=w)
    exceedance = posterior_bounds(fitted, head, ("exceedance", 2.0), draws=drawn, weights=w)
    np.testing.assert_allclose(
        cdf["estimate"].to_numpy(), family.cdf_prior_weighted(threshold, theta, w)
    )
    np.testing.assert_allclose(cdf["estimate"] + exceedance["estimate"], 1.0)

    shortfall = posterior_bounds(fitted, head, ("expected_shortfall", 0.9), draws=drawn, weights=w)
    assert np.all(shortfall["estimate"].to_numpy() >= weighted["estimate"].to_numpy())

    # The prior weight scales the gamma's shape and its scale together, so the
    # row law keeps the unit-weight mean and ``"mean"`` must not move at all.
    assert np.array_equal(
        posterior_bounds(fitted, head, "mean", draws=drawn, weights=w).to_numpy(),
        posterior_bounds(fitted, head, "mean", draws=drawn).to_numpy(),
    )


def test_unit_prior_weights_leave_the_primitive_bit_identical(weighted_gamma_case) -> None:
    fitted, X, _ = weighted_gamma_case
    head = X.iloc[:60]
    ones = np.ones(len(head))
    drawn = posterior_draws(fitted, 64, seed=8)

    for quantity in (("quantile", 0.9), ("cdf", 2.0), ("expected_shortfall", 0.9), "mean"):
        assert np.array_equal(
            posterior_bounds(fitted, head, quantity, draws=drawn, weights=ones).to_numpy(),
            posterior_bounds(fitted, head, quantity, draws=drawn).to_numpy(),
        ), quantity
    assert np.array_equal(
        posterior_predictive(fitted, head, 20, seed=8, weights=ones),
        posterior_predictive(fitted, head, 20, seed=8),
    )
    assert not np.array_equal(
        posterior_predictive(fitted, head, 20, seed=8, weights=np.full(len(head), 0.5)),
        posterior_predictive(fitted, head, 20, seed=8),
    )


def test_prior_weights_are_chunked_in_step_with_their_rows(weighted_gamma_case) -> None:
    fitted, X, weights = weighted_gamma_case
    head, w = X.iloc[:40], weights[:40]
    reversed_weights = w[::-1].copy()
    drawn = posterior_draws(fitted, 32, seed=3)

    whole = posterior_bounds(fitted, head, ("quantile", 0.75), draws=drawn, weights=w)
    chunked = posterior_bounds(
        fitted, head, ("quantile", 0.75), draws=drawn, weights=w, chunk_rows=7
    )
    for column in ("estimate", "lower", "upper"):
        assert np.array_equal(chunked[column].to_numpy(), whole[column].to_numpy()), column
    # A weight vector read from the head of the frame on every chunk, or read
    # in the wrong row order, agrees on nothing but an accident.
    shuffled = posterior_bounds(
        fitted, head, ("quantile", 0.75), draws=drawn, weights=reversed_weights
    )
    assert not np.array_equal(shuffled["estimate"].to_numpy(), whole["estimate"].to_numpy())

    # The predictive uniforms are drawn per chunk, so a weighted and an
    # unweighted run of the same seed and chunking share their uniforms: the
    # two simulated responses must invert to the same probability row by row,
    # which a misaligned weight vector cannot do.
    family = fitted.family
    theta = np.asarray(fitted.predict_parameters(head))
    for chunk_rows in (None, 7):
        kwargs = {"parameter_uncertainty": False, "seed": 9, "chunk_rows": chunk_rows}
        plain = posterior_predictive(fitted, head, 20, **kwargs)
        row_law = posterior_predictive(fitted, head, 20, weights=w, **kwargs)
        stacked = np.tile(theta, (20, 1))
        np.testing.assert_allclose(
            family.cdf_prior_weighted(row_law.reshape(-1), stacked, np.tile(w, 20)),
            family.cdf(plain.reshape(-1), stacked),
            rtol=1.0e-8,
            atol=1.0e-10,
        )
        misaligned = posterior_predictive(fitted, head, 20, weights=reversed_weights, **kwargs)
        assert not np.allclose(misaligned, row_law)


def test_a_family_without_the_prior_weighted_law_refuses(fit_case) -> None:
    fitted, X, _ = fit_case
    head = X.iloc[:12]
    weights = np.full(len(head), 0.5)
    family = _UnitLawOnlyFamily(fitted.family)
    stub = SimpleNamespace(family=family)

    # Never the unit law: a family that cannot state the weighted row law
    # refuses instead of quietly inverting the unweighted one.
    with pytest.raises(NotImplementedError, match="_UnitLawOnlyFamily"):
        resolve_quantity(family, ("quantile", 0.5), weights=weights)
    with pytest.raises(NotImplementedError, match="_UnitLawOnlyFamily"):
        posterior_predictive(stub, head, 4, weights=weights)
    with pytest.raises(NotImplementedError, match="_UnitLawOnlyFamily"):
        posterior_bounds(
            stub, head, ("cdf", 0.0), draws=posterior_draws(fitted, 4), weights=weights
        )
    with pytest.raises(ValueError, match="one positive prior weight"):
        posterior_predictive(fitted, head, 4, weights=np.ones(len(head) + 1))
    with pytest.raises(ValueError, match="one positive prior weight"):
        posterior_bounds(fitted, head, "mean", n_draws=4, weights=np.zeros(len(head)))


def test_simultaneous_critical_value_exceeds_the_pointwise_one() -> None:
    rng = np.random.default_rng(11)
    width = 4
    beta_hat = np.zeros(width)
    coefficients = rng.standard_normal((20_000, width))
    grid = np.eye(width)
    standard_error = np.ones(width)

    critical = simultaneous_critical_value(
        grid, slice(0, width), coefficients, beta_hat, standard_error, alpha=0.05
    )
    assert critical >= stats.norm.ppf(0.975)
    assert critical == pytest.approx(2.4909, abs=0.1)

    packed = PosteriorDraws(
        coefficients=coefficients,
        covariance_kind="fixed",
        seed=11,
        coefficient_names=tuple(f"b[{index}]" for index in range(width)),
    )
    assert (
        simultaneous_critical_value(grid, slice(0, width), packed, beta_hat, standard_error)
        == critical
    )


def test_simultaneous_critical_value_floors_a_two_draw_undershoot() -> None:
    coefficients = np.array([[-0.25], [0.25]])
    critical = simultaneous_critical_value(
        np.ones((1, 1)),
        slice(0, 1),
        coefficients,
        np.zeros(1),
        np.ones(1),
        alpha=0.05,
    )

    assert critical == stats.norm.isf(0.025)


def test_simultaneous_critical_value_retains_a_larger_simulated_multiplier() -> None:
    coefficients = np.array([[-3.0], [3.0]])
    critical = simultaneous_critical_value(
        np.ones((1, 1)),
        slice(0, 1),
        coefficients,
        np.zeros(1),
        np.ones(1),
        alpha=0.05,
    )

    assert critical == 3.0


def _covariance_shim(covariance: np.ndarray) -> SimpleNamespace:
    width = len(covariance)
    return SimpleNamespace(
        inference=SimpleNamespace(covariance=covariance, reconciliation_tolerance=1.0e-6),
        result=SimpleNamespace(coefficients=np.zeros(width)),
        layout=SimpleNamespace(
            coefficient_names=tuple(f"b[{index}]" for index in range(width)),
            n_coefficients=width,
        ),
        smoothing=None,
    )


def test_a_negative_eigenvalue_is_an_error_and_a_null_direction_is_a_clip() -> None:
    with pytest.raises(ValueError, match="negative"):
        posterior_draws(_covariance_shim(np.diag([1.0, -0.5])), 8)

    rank_deficient = posterior_draws(_covariance_shim(np.diag([1.0, 0.0])), 64, seed=13)
    assert np.all(rank_deficient.coefficients[:, 1] == 0.0)
    assert float(rank_deficient.coefficients[:, 0].std()) > 0.0

    # An eigenvalue inside the tolerance is set to zero, never reflected: a
    # round-off negative direction must carry no posterior spread at all.
    round_off = posterior_draws(_covariance_shim(np.diag([1.0, -1.0e-9])), 64, seed=13)
    assert np.all(round_off.coefficients[:, 1] == 0.0)
    assert np.array_equal(round_off.coefficients, rank_deficient.coefficients)


def _corrected_shim(*, lambdas, gradient, hessian, penalties=()) -> SimpleNamespace:
    shim = _covariance_shim(np.eye(2))
    terminal = shim.result
    terminal.solve_terminal = lambda rhs: np.asarray(rhs, dtype=float)
    terminal.family_likelihood_plan_identifier = "authenticated-shim-plan"
    terminal.coefficient_face = None
    terminal.config = SimpleNamespace(coefficient_curvature="observed")
    terminal.terminal_curvature = SimpleNamespace(
        requested_source="observed",
        actual_source="observed",
        fallback_count=0,
    )
    shim.layout.penalties = tuple(penalties)
    shim.smoothing = SimpleNamespace(
        lambdas=lambdas,
        terminal_gradient=gradient,
        smoothing_hessian=hessian,
        smoothing_hessian_certificate=(
            None if hessian is None else np.zeros_like(hessian, dtype=float)
        ),
        convergence_reason="stationary",
        converged=True,
        terminal_fit=terminal,
        config=SimpleNamespace(hessian_certificate_fraction=0.1),
    )
    shim.fit_state = SimpleNamespace(
        smoothing=shim.smoothing,
        solver_result=terminal,
        family_likelihood_plan_identifier="authenticated-shim-plan",
        lambdas=lambdas,
        exact_face_components=(),
    )
    return shim


def test_the_smoothing_hessian_row_order_is_asserted_not_assumed() -> None:
    lambdas = {"location:x#wiggle": 1.0, "scale:x#wiggle": 2.0}
    names = tuple(lambdas)

    with pytest.raises(RuntimeError, match="no row order"):
        posterior_covariance(
            _corrected_shim(lambdas=lambdas, gradient=None, hessian=np.eye(2)), kind="corrected"
        )
    with pytest.raises(RuntimeError, match="lambda order"):
        posterior_covariance(
            _corrected_shim(
                lambdas=lambdas,
                gradient={names[1]: 0.0, names[0]: 0.0},
                hessian=np.eye(2),
            ),
            kind="corrected",
        )
    with pytest.raises(RuntimeError, match="square"):
        posterior_covariance(
            _corrected_shim(lambdas=lambdas, gradient={names[0]: 0.0}, hessian=np.eye(2)),
            kind="corrected",
        )
    with pytest.raises(RuntimeError, match="no layout penalty"):
        posterior_covariance(
            _corrected_shim(lambdas=lambdas, gradient={names[0]: 0.0}, hessian=np.eye(1)),
            kind="corrected",
        )


def test_posterior_draws_defend_their_own_shape() -> None:
    names = ("a", "b")
    with pytest.raises(ValueError, match="draws, coefficients"):
        PosteriorDraws(np.zeros(2), "fixed", 0, names)
    with pytest.raises(ValueError, match="finite"):
        PosteriorDraws(np.full((2, 2), np.nan), "fixed", 0, names)
    with pytest.raises(ValueError, match="coefficient_names"):
        PosteriorDraws(np.zeros((2, 3)), "fixed", 0, names)
    with pytest.raises(ValueError, match="covariance_kind"):
        PosteriorDraws(np.zeros((2, 2)), "approximate", 0, names)

    mismatched = _covariance_shim(np.eye(3))
    mismatched.result = SimpleNamespace(coefficients=np.zeros(2))
    with pytest.raises(ValueError, match="does not match the fitted coefficients"):
        posterior_draws(mismatched, 4)


def test_quantity_and_reduce_contracts_are_enforced(fit_case) -> None:
    fitted, X, _ = fit_case
    head = X.iloc[:24]
    family = fitted.family

    with pytest.raises(ValueError, match="finite scalar"):
        resolve_quantity(family, ("quantile", "half"))
    with pytest.raises(ValueError, match="finite scalar"):
        resolve_quantity(family, ("quantile", np.inf))
    with pytest.raises(ValueError, match="strictly inside"):
        resolve_quantity(family, ("quantile", 1.5))
    with pytest.raises(ValueError, match="unsupported posterior quantity"):
        resolve_quantity(family, 42)
    with pytest.raises(ValueError, match="unsupported posterior quantity"):
        resolve_quantity(family, ("parameter",))
    with pytest.raises(NotImplementedError, match="default prediction"):
        resolve_quantity(_NoDistributionFunctionFamily(family.parameters), "mean")

    with pytest.raises(ValueError, match="one value per row"):
        posterior_bounds(fitted, head, lambda theta: theta, n_draws=4)

    calls: list[int] = []

    def flaky(theta: np.ndarray) -> np.ndarray:
        calls.append(len(theta))
        values = np.asarray(theta, dtype=float)[:, 0]
        return values if len(calls) == 1 else values[:-1]

    with pytest.raises(ValueError, match="one value per row"):
        posterior_bounds(fitted, head, flaky, n_draws=4)

    with pytest.raises(ValueError, match="n_draws"):
        posterior_predictive(fitted, head, 0)
    with pytest.raises(ValueError, match="additive across row chunks"):
        posterior_predictive(
            fitted, head, 4, parameter_uncertainty=False, reduce=lambda block: block.sum(axis=0)
        )

    empty = X.iloc[:0]
    assert posterior_predictive(fitted, empty, 4, parameter_uncertainty=False).shape == (4, 0)
    with pytest.raises(ValueError, match="at least one row"):
        posterior_predictive(fitted, empty, 4, parameter_uncertainty=False, reduce="sum")


def test_bounds_carry_the_row_index_of_every_supported_frame(fit_case) -> None:
    import polars as pl

    fitted, X, _ = fit_case
    head = X.iloc[3:9]
    assert posterior_bounds(fitted, as_eager_frame(head), "mean", n_draws=4).index.equals(
        head.index
    )
    frame = pl.DataFrame({"x": head["x"].to_numpy(), "g": head["g"].to_list()})
    assert posterior_bounds(fitted, frame, "mean", n_draws=4).index.equals(pd.RangeIndex(len(head)))


def test_simultaneous_critical_value_validates_its_inputs() -> None:
    coefficients = np.zeros((8, 3))
    grid = np.eye(3)
    beta_hat = np.zeros(3)
    errors = np.ones(3)

    with pytest.raises(ValueError, match="draws, coefficients"):
        simultaneous_critical_value(grid, slice(0, 3), np.zeros(3), beta_hat, errors)
    with pytest.raises(ValueError, match="grid points, coefficients"):
        simultaneous_critical_value(np.zeros(3), slice(0, 3), coefficients, beta_hat, errors)
    with pytest.raises(ValueError, match="one standard error per grid point"):
        simultaneous_critical_value(grid, slice(0, 3), coefficients, beta_hat, np.ones(2))
    with pytest.raises(ValueError, match="alpha"):
        simultaneous_critical_value(grid, slice(0, 3), coefficients, beta_hat, errors, alpha=0.0)
    with pytest.raises(ValueError, match="grid_design columns"):
        simultaneous_critical_value(grid, slice(0, 2), coefficients, beta_hat, errors)


def test_the_primitive_validates_its_arguments(fit_case) -> None:
    fitted, X, _ = fit_case
    head = X.iloc[:5]

    with pytest.raises(ValueError, match="n_draws"):
        posterior_draws(fitted, 1)
    with pytest.raises(ValueError, match="level"):
        posterior_bounds(fitted, head, "mean", level=1.5)
    with pytest.raises(ValueError, match="chunk_rows"):
        list(posterior_parameters(fitted, head, posterior_draws(fitted, 4), chunk_rows=0))
    with pytest.raises(TypeError, match="PosteriorDraws"):
        list(posterior_parameters(fitted, head, np.zeros((4, fitted.layout.n_coefficients))))

    mismatched = PosteriorDraws(
        coefficients=np.zeros((4, fitted.layout.n_coefficients)),
        covariance_kind="fixed",
        seed=0,
        coefficient_names=tuple(f"b[{index}]" for index in range(fitted.layout.n_coefficients)),
    )
    with pytest.raises(ValueError, match="coefficient names"):
        list(posterior_parameters(fitted, head, mismatched))
    with pytest.raises(ValueError, match="reduce"):
        posterior_predictive(fitted, head, 4, reduce="mean")
