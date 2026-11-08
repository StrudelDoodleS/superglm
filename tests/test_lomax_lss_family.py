import mpmath as mp
import numpy as np
import pandas as pd
import pytest

from superglm import SuperLSS
from tests.test_declared_scalar_adapter import (
    family,  # noqa: F401
    weights,
)


def test_frequency_intercept_fit_matches_literal_replication(family):  # noqa: F811
    rng = np.random.default_rng(381)
    y = 3 * np.expm1(rng.exponential(size=160) / 2.5)
    mass = rng.integers(1, 6, len(y))
    frame = pd.DataFrame({"unused": np.zeros(len(y))})
    fit = SuperLSS(family, family.scale(), family.shape(), weight_semantics="frequency").fit(
        frame, y, sample_weight=mass
    )
    rows = np.repeat(np.arange(len(y)), mass)
    repeated = SuperLSS(family, family.scale(), family.shape()).fit(frame.iloc[rows], y[rows])
    from tests.test_declared_scalar_vertical_slice import assert_same_optimum, objective

    # Compare on one independently assembled objective. Its measured score
    # residual and smallest curvature determine parameter-forward tolerance.
    evaluate = objective(
        "lomax", y, mass, np.ones((len(y), 1)), np.zeros((len(y), 2)), np.zeros((2, 2))
    )
    assert_same_optimum(
        evaluate,
        np.log(fit.predict_parameters(frame).to_numpy()[0]),
        np.log(repeated.predict_parameters(frame).to_numpy()[0]),
        tolerance=1e-8,  # both fits requested the default inner tolerance
    )


def test_lomax_mixed_observation_likelihood(family):  # noqa: F811
    from superglm.distributional.family import bounded_observation
    from superglm.distributional.observation import ResponseBounds
    from tests._declared_scalar_oracles import assert_scaled_jets, jets, log_density, log_window

    y = np.array([0.2, 2.0, 0.7, 1.5, 0.8, 1.0, 0.5])
    lower = np.array([0.0, 0.0, 0.0, 0.3, 0.0, 0.2, 0.1])
    upper = np.array([np.inf, np.inf, np.inf, np.inf, 2.0, 3.0, 2.0])
    side = np.array([0, 1, -1, 0, 0, 1, -1])
    mass = np.array([1, 2, 5, 2, 1, 5, 2])
    obs = bounded_observation(ResponseBounds.from_arrays(lower, upper, side))
    plan = family.bind_likelihood(y, weights(mass), obs)
    theta = np.tile([2.0, 1.5], (len(y), 1))
    actual = family.evaluate_natural(y, theta, plan)
    for i in range(len(y)):

        def law(s, a):
            lo, hi = mp.mpf(lower[i]), mp.mpf(upper[i])
            z = mp.mpf(y[i])
            observed = (
                log_density("lomax", z, s, a)
                if side[i] == 0
                else log_window("lomax", z if side[i] > 0 else lo, hi if side[i] > 0 else z, s, a)
            )
            return int(mass[i]) * (observed - log_window("lomax", lo, hi, s, a))

        reference = jets(law, theta[i], 2)
        assert_scaled_jets(
            (actual.reported_log_likelihood[i : i + 1], actual.score[i], actual.hessian_packed[i]),
            reference,
            theta[i],
            factor=8192,
        )
        # Omitted denominator and double replication are rejected on affected rows.
        if lower[i] > 0 or np.isfinite(upper[i]):
            with pytest.raises(AssertionError):
                np.testing.assert_allclose(
                    actual.reported_log_likelihood[i],
                    reference[0][0]
                    + mass[i]
                    * float(log_window("lomax", mp.mpf(lower[i]), mp.mpf(upper[i]), *theta[i])),
                )
    assert (
        plan.plan_identifier
        != family.bind_likelihood(
            y,
            weights(mass),
            __import__(
                "superglm.distributional.family", fromlist=["COMPLETE_OBSERVATION"]
            ).COMPLETE_OBSERVATION,
        ).plan_identifier
    )


def test_weighted_bounded_predictor_directions_and_honest_fallback(family):  # noqa: F811
    from itertools import combinations_with_replacement as packed

    from superglm.distributional.family import bounded_observation
    from superglm.distributional.observation import ResponseBounds
    from superglm.distributional.weights import UnsupportedLikelihoodContractError
    from tests._declared_scalar_oracles import jets, log_density, log_window

    y, theta = np.array([0.5, 1.5]), np.array([[2.0, 1.5], [0.8, 0.75]])
    lo, hi, side = np.array([0.1, 0.2]), np.array([3.0, 5.0]), np.array([0, 1])
    plan = family.bind_likelihood(
        y, weights([2, 5]), bounded_observation(ResponseBounds.from_arrays(lo, hi, side))
    )
    eta = np.log(theta)
    a = np.array([[0.3, -0.2], [0.1, 0.4]])
    b = np.array([[-0.1, 0.5], [0.4, 0.2]])
    links = tuple(p.default_link for p in family.parameters)
    first = family.predictor_curvature_directional_derivative(y, eta, a, links, plan)
    second = family.predictor_curvature_second_directional_derivative(y, eta, a, b, links, plan)
    for row in range(2):

        def law(u, v):
            s, shape = mp.exp(u), mp.exp(v)
            observed = (
                log_density("lomax", mp.mpf(y[row]), s, shape)
                if side[row] == 0
                else log_window("lomax", mp.mpf(y[row]), mp.mpf(hi[row]), s, shape)
            )
            return int(plan.multiplier[row]) * (
                observed - log_window("lomax", mp.mpf(lo[row]), mp.mpf(hi[row]), s, shape)
            )

        ref = jets(law, eta[row], 4)
        columns = {d: {p: i for i, p in enumerate(packed(range(2), d))} for d in (3, 4)}
        ref1, ref2 = [], []
        for i, j in packed(range(2), 2):
            ref1.append(
                -sum(ref[3][columns[3][tuple(sorted((i, j, k)))]] * a[row, k] for k in range(2))
            )
            ref2.append(
                -sum(
                    ref[4][columns[4][tuple(sorted((i, j, k, h)))]] * a[row, k] * b[row, h]
                    for k in range(2)
                    for h in range(2)
                )
            )
        for actual, expected in ((first[row], ref1), (second[row], ref2)):
            np.testing.assert_allclose(actual, expected, rtol=8192 * 2.0**-53, atol=8192 * 2.0**-53)
    stacked = family.predictor_curvature_directional_derivative(
        y, eta, np.stack((a, b)), links, plan
    )
    np.testing.assert_allclose(stacked[0], first, rtol=0, atol=0)
    stacked2 = family.predictor_curvature_second_directional_derivative(
        y, eta, np.stack((a, b)), np.stack((b, a)), links, plan
    )
    np.testing.assert_allclose(stacked2[0], second, rtol=0, atol=0)
    tiny = np.array([1e-100])
    bounded = family.bind_likelihood(
        tiny,
        weights([1]),
        bounded_observation(
            ResponseBounds.from_arrays(np.array([0.0]), np.array([np.inf]), np.array([-1]))
        ),
    )
    assert family.evaluate_natural(tiny, theta[:1], bounded).valid.all()
    with pytest.raises(UnsupportedLikelihoodContractError, match="analytic higher.*quadrature"):
        family.predictor_curvature_second_directional_derivative(
            tiny, eta[:1], a[:1], b[:1], links, bounded
        )


def test_fisher_is_complete_only(family):  # noqa: F811
    from superglm.distributional.family import (
        COMPLETE_OBSERVATION,
        bounded_observation,
        expected_information_provider,
    )
    from superglm.distributional.observation import ResponseBounds
    from superglm.distributional.weights import UnsupportedLikelihoodContractError

    y, theta = np.ones(2), np.tile([2.0, 1.5], (2, 1))
    complete = family.bind_likelihood(y, weights([2, 5]), COMPLETE_OBSERVATION)
    assert expected_information_provider(family, complete) is family
    bounded = family.bind_likelihood(
        y,
        weights([2, 5]),
        bounded_observation(
            ResponseBounds.from_arrays(np.zeros(2), np.full(2, 3.0), np.zeros(2, int))
        ),
    )
    assert expected_information_provider(family, bounded) is None
    with pytest.raises(UnsupportedLikelihoodContractError, match="complete"):
        family.expected_information_natural(theta, bounded)
    assert not np.allclose(
        -family.evaluate_natural(y, theta, complete).hessian_packed,
        family.expected_information_natural(theta, complete),
    )


def _layer(theta, attachment=1.0, width=2.0):
    s, a = theta.T
    log_a = np.log1p(attachment / s)
    delta = np.log1p(width / (s + attachment))
    v = (1 - a) * delta
    ratio = np.ones(len(s))
    np.divide(np.expm1(v), v, out=ratio, where=v != 0)
    return np.exp(np.log(s) + (1 - a) * log_a) * delta * ratio


def test_infinite_mean_does_not_preclude_bounded_layer(family):  # noqa: F811
    a = np.array([0.25, np.nextafter(1.0, 0.0), 1.0, np.nextafter(1.0, np.inf), 3.0])
    theta = np.column_stack((np.full(len(a), 2.0), a))
    actual = _layer(theta)
    assert np.all((actual >= 0) & (actual <= 2))
    with mp.workdps(70):
        expected = [float(mp.quad(lambda y: (1 + y / 2) ** (-mp.mpf(float(x))), [1, 3])) for x in a]
    np.testing.assert_allclose(actual, expected, rtol=128 * 2.0**-53)
    # At a neighbour of one, direct powers subtract away the integral.
    j = 1
    naive = 2 * ((1 + 3 / 2) ** (1 - a[j]) - (1 + 1 / 2) ** (1 - a[j])) / (1 - a[j])
    assert abs(naive - expected[j]) > 0.01
    assert np.isinf(family.default_prediction(theta[:3])).all()


def test_nonfinite_tail_posterior_is_disclosed(family):  # noqa: F811
    from superglm.distributional.checks.scores import crps_numeric
    from superglm.distributional.posterior import PosteriorDraws

    p = (np.arange(200) + 0.5) / 200
    y = 2 * np.expm1(-np.log1p(-p) / 2.5)
    frame = pd.DataFrame({"unused": np.zeros(len(y))})
    model = SuperLSS(family, family.scale(), family.shape()).fit(frame, y)
    ordinary = model.posterior_draws(2, seed=1)
    coefficients = ordinary.coefficients.copy()
    shape = ordinary.coefficient_names.index("shape:(intercept)")
    coefficients[0, shape] = np.log(0.75)
    coefficients[1, shape] = np.log(3.0)
    draws = PosteriorDraws(coefficients, "fixed", 1, ordinary.coefficient_names)
    with pytest.raises(ValueError, match="non-finite.*posterior.*quantity"):
        model.posterior_bounds(frame.iloc[:2], ("expected_shortfall", 0.9), draws=draws)
    bounds = model.posterior_bounds(frame.iloc[:2], _layer, draws=draws)
    assert np.isfinite(bounds.to_numpy()).all()
    with pytest.raises((ValueError, NotImplementedError), match="variance|tail|finite"):
        crps_numeric(family, 1.0, np.array([[2.0, 0.75]]))
    # The facade preserves +inf from an explicitly infinite-mean parameter state.
    pred = model.predict(frame.iloc[:2], offsets={"shape": np.full(2, -10.0)})
    assert np.isposinf(pred).all()
