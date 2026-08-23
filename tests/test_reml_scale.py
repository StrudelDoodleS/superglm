"""Independent checks for family-correct REML scale profiling."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from scipy.special import digamma

from superglm.distributions import Gamma
from superglm.reml.scale import (
    GammaScaleProfileData,
    _gamma_inverse_shape_derivative,
    _gamma_saturated_normalizer,
    _shape_times_log_minus_digamma,
    _trigamma_minus_inverse,
    prepare_gamma_reml_scale_data,
    profile_gamma_reml_scale,
    profile_gaussian_reml_scale,
)


def _brute_gamma_profile(
    y: np.ndarray,
    weights: np.ndarray,
    penalized_deviance: float,
    penalty_nullity: float,
) -> tuple[float, float]:
    distribution = Gamma()

    def criterion(log_phi: float) -> float:
        phi = float(np.exp(log_phi))
        return float(
            penalized_deviance / (2.0 * phi)
            - distribution.log_likelihood(y, y, weights, phi=phi)
            - 0.5 * penalty_nullity * np.log(2.0 * np.pi * phi)
        )

    result = minimize_scalar(
        criterion,
        bounds=(-12.0, 8.0),
        method="bounded",
        options={"xatol": 1.0e-12},
    )
    assert result.success
    return float(np.exp(result.x)), float(result.fun)


@pytest.mark.parametrize(
    ("function", "expected"),
    [
        (_shape_times_log_minus_digamma, 0.5008333250003967837377323792877688335),
        (_gamma_saturated_normalizer, 1.382813229233738027554280476585931908),
        (_trigamma_minus_inverse, 0.00005016666333357139524566846570142254),
    ],
)
def test_gamma_asymptotics_match_independent_high_precision_values(function, expected) -> None:
    """Hard-coded 100-digit reference values guard cancellation thresholds."""
    assert function(100.0) == pytest.approx(expected, rel=3.0e-15, abs=3.0e-15)


@pytest.mark.parametrize(
    "function",
    [
        _shape_times_log_minus_digamma,
        _gamma_saturated_normalizer,
        _trigamma_minus_inverse,
    ],
)
def test_gamma_asymptotic_switch_is_continuous_across_adjacent_floats(function) -> None:
    below = function(float(np.nextafter(100.0, 0.0)))
    above = function(float(np.nextafter(100.0, np.inf)))
    assert below == pytest.approx(above, rel=2.0e-13, abs=2.0e-14)


def test_gamma_scale_profile_matches_direct_wood_criterion() -> None:
    y = np.array([0.35, 0.7, 1.1, 1.8, 2.6, 3.2])
    weights = np.ones_like(y)
    expected_phi, expected_criterion = _brute_gamma_profile(y, weights, 3.7, 2.0)

    profile_data = prepare_gamma_reml_scale_data(y, weights, weight_semantics="frequency")
    actual = profile_gamma_reml_scale(profile_data, 3.7, 2.0)

    assert actual.phi == pytest.approx(expected_phi, rel=2.0e-8)
    assert actual.criterion == pytest.approx(expected_criterion, rel=2.0e-10, abs=2.0e-10)

    step = 1.0e-5
    lower = profile_gamma_reml_scale(profile_data, 3.7 - step, 2.0)
    upper = profile_gamma_reml_scale(profile_data, 3.7 + step, 2.0)
    finite_difference = ((1.0 / upper.phi) - (1.0 / lower.phi)) / (2.0 * step)
    assert actual.inverse_phi == pytest.approx(1.0 / actual.phi, rel=2.0e-15)
    assert actual.d_inverse_phi_d_penalized_deviance == pytest.approx(
        finite_difference,
        rel=2.0e-7,
    )


def test_gamma_frequency_weights_match_expanded_rows() -> None:
    y = np.array([0.4, 0.8, 1.7, 3.1])
    weights = np.array([1.0, 3.0, 2.0, 4.0])
    repeated_y = np.repeat(y, weights.astype(int))

    weighted = profile_gamma_reml_scale(
        prepare_gamma_reml_scale_data(y, weights, weight_semantics="frequency"), 5.2, 1.0
    )
    expanded = profile_gamma_reml_scale(
        prepare_gamma_reml_scale_data(
            repeated_y, np.ones_like(repeated_y), weight_semantics="frequency"
        ),
        5.2,
        1.0,
    )

    assert weighted.phi == pytest.approx(expanded.phi, rel=2.0e-13)
    assert weighted.criterion == pytest.approx(expanded.criterion, rel=2.0e-13)


def test_gamma_scale_profile_rejects_nonpositive_effective_likelihood_size() -> None:
    with pytest.raises(ValueError, match="no finite interior optimum"):
        profile_gamma_reml_scale(
            prepare_gamma_reml_scale_data(
                np.array([1.0, 2.0]),
                np.array([0.1, 0.1]),
                weight_semantics="frequency",
            ),
            1.0,
            penalty_nullity=1.0,
        )


@pytest.mark.parametrize(
    ("sum_weight", "sum_weight_log_y"),
    [
        (np.inf, 0.0),
        (1.0, np.nan),
    ],
)
def test_gamma_scale_profile_rejects_invalid_reduced_statistics(
    sum_weight,
    sum_weight_log_y,
) -> None:
    with pytest.raises(ValueError, match="finite"):
        GammaScaleProfileData(
            sum_weight=sum_weight,
            sum_weight_log_y=sum_weight_log_y,
        )


@pytest.mark.parametrize(
    ("y", "weights"),
    [
        (np.ones(2), np.full(2, np.finfo(np.float64).max)),
        (np.array([np.finfo(np.float64).max]), np.array([1.0e307])),
    ],
)
def test_gamma_scale_preparation_rejects_overflowed_reductions(y, weights) -> None:
    with pytest.raises(ValueError, match="finite"):
        prepare_gamma_reml_scale_data(y, weights, weight_semantics="frequency")


def test_gaussian_profile_uses_frequency_weight_likelihood_size() -> None:
    profile = profile_gaussian_reml_scale(
        penalized_deviance=8.5,
        likelihood_size=17.0,
        penalty_nullity=3.0,
    )

    residual_size = 14.0
    assert profile.phi == pytest.approx(8.5 / 14.0)
    assert profile.inverse_phi == pytest.approx(14.0 / 8.5)
    assert profile.criterion == pytest.approx(
        0.5 * residual_size * (1.0 + np.log(2.0 * np.pi * 8.5 / residual_size))
    )
    assert profile.d_inverse_phi_d_penalized_deviance == pytest.approx(-14.0 / 8.5**2)


@pytest.mark.parametrize(
    ("penalized_deviance", "likelihood_size"),
    [
        (1.0e-300, 1.0e100),  # phi underflows while 1 / phi overflows
        (1.0e-309, 1.0e-309),  # phi is finite but its Dp derivative overflows
    ],
)
def test_gaussian_profile_rejects_unrepresentable_outputs(
    penalized_deviance: float,
    likelihood_size: float,
) -> None:
    with pytest.raises(FloatingPointError, match="representable"):
        profile_gaussian_reml_scale(
            penalized_deviance=penalized_deviance,
            likelihood_size=likelihood_size,
            penalty_nullity=0.0,
        )


def _gamma_shape_score(
    shape: float,
    penalized_deviance: float,
    sum_weight: float,
    penalty_nullity: float,
) -> float:
    if shape < 30.0:
        shape_log_minus_digamma = shape * (np.log(shape) - digamma(shape))
    else:
        inverse = 1.0 / shape
        shape_log_minus_digamma = (
            0.5
            + inverse / 12.0
            - inverse**3 / 120.0
            + inverse**5 / 252.0
            - inverse**7 / 240.0
            + inverse**9 / 132.0
            - 691.0 * inverse**11 / 32760.0
        )
    return float(
        0.5 * penalized_deviance * shape
        - sum_weight * shape_log_minus_digamma
        + 0.5 * penalty_nullity
    )


@pytest.mark.parametrize(
    ("penalized_deviance", "penalty_nullity", "inverse_phi_bound", "comparison"),
    [
        (1.0e-20, 0.0, np.exp(30.0), "greater"),
        (1.0, 2.0 - 1.0e-14, np.exp(-30.0), "less"),
    ],
)
def test_gamma_scale_profile_adaptively_brackets_extreme_valid_optima(
    penalized_deviance: float,
    penalty_nullity: float,
    inverse_phi_bound: float,
    comparison: str,
) -> None:
    profile_data = prepare_gamma_reml_scale_data(
        np.array([1.0]), np.array([1.0]), weight_semantics="frequency"
    )

    profile = profile_gamma_reml_scale(
        profile_data,
        penalized_deviance,
        penalty_nullity,
    )

    if comparison == "greater":
        assert profile.inverse_phi > inverse_phi_bound
    else:
        assert profile.inverse_phi < inverse_phi_bound
    score = _gamma_shape_score(
        profile.inverse_phi,
        penalized_deviance,
        profile_data.sum_weight,
        penalty_nullity,
    )
    assert score == pytest.approx(0.0, abs=2.0e-14)


def test_gamma_scale_profile_handles_curvature_below_square_underflow() -> None:
    """A valid tiny shape must not fail after the root has been found."""
    profile_data = prepare_gamma_reml_scale_data(
        np.array([1.0]), np.array([1.0]), weight_semantics="frequency"
    )

    profile = profile_gamma_reml_scale(
        profile_data,
        penalized_deviance=1.0e200,
        penalty_nullity=0.0,
    )

    assert profile.inverse_phi == pytest.approx(2.0e-200, rel=2.0e-12)
    assert profile.phi == pytest.approx(5.0e199, rel=2.0e-12)
    assert profile.d_inverse_phi_d_penalized_deviance == 0.0
    assert np.signbit(profile.d_inverse_phi_d_penalized_deviance)


def test_gamma_scale_profile_accepts_maximum_finite_numpy_deviance_without_warning() -> None:
    profile = profile_gamma_reml_scale(
        GammaScaleProfileData(sum_weight=1.0, sum_weight_log_y=0.0),
        penalized_deviance=np.float64(np.finfo(np.float64).max),
        penalty_nullity=0.0,
    )

    assert 0.0 < profile.inverse_phi < np.finfo(np.float64).tiny
    assert np.isfinite(profile.phi)
    assert np.isfinite(profile.criterion)


@pytest.mark.parametrize(
    ("sum_weight", "penalized_deviance", "expected_shape", "expected_derivative", "atol"),
    [
        (1.0, 1.0e160, 2.0e-160, -2.0e-320, 5.0e-323),
        (1.0e-200, 1.0, 2.0e-200, -2.0e-200, 1.0e-212),
    ],
)
def test_gamma_scale_profile_retains_representable_tiny_shape_derivatives(
    sum_weight: float,
    penalized_deviance: float,
    expected_shape: float,
    expected_derivative: float,
    atol: float,
) -> None:
    profile = profile_gamma_reml_scale(
        GammaScaleProfileData(sum_weight=sum_weight, sum_weight_log_y=0.0),
        penalized_deviance=penalized_deviance,
        penalty_nullity=0.0,
    )

    assert profile.inverse_phi == pytest.approx(expected_shape, rel=2.0e-12)
    assert profile.d_inverse_phi_d_penalized_deviance == pytest.approx(
        expected_derivative,
        rel=2.0e-12,
        abs=atol,
    )


@pytest.mark.parametrize("penalized_deviance", np.logspace(-18.0, 18.0, 13))
def test_gamma_scale_profile_is_stationary_across_wide_deviance_sweep(
    penalized_deviance: float,
) -> None:
    y = np.array([0.4, 0.8, 1.7, 3.1])
    weights = np.array([0.5, 1.25, 2.0, 1.75])
    profile_data = prepare_gamma_reml_scale_data(y, weights, weight_semantics="frequency")

    profile = profile_gamma_reml_scale(
        profile_data,
        penalized_deviance,
        penalty_nullity=2.0,
    )

    assert profile.phi > 0.0
    assert profile.inverse_phi > 0.0
    assert profile.phi * profile.inverse_phi == pytest.approx(1.0, rel=2.0e-15)
    assert profile.d_inverse_phi_d_penalized_deviance < 0.0
    score = _gamma_shape_score(
        profile.inverse_phi,
        penalized_deviance,
        profile_data.sum_weight,
        2.0,
    )
    assert score == pytest.approx(0.0, abs=1.0e-11)


def _distinct_weight_fixture(n: int = 4000, seed: int = 5):
    rng = np.random.default_rng(seed)
    y = rng.gamma(5.0, 1.0, n)
    weights = rng.uniform(0.5, 2.0, n)
    return y, weights


def _term_fields(term) -> np.ndarray:
    return np.array(
        [term.phi, term.inverse_phi, term.criterion, term.d_inverse_phi_d_penalized_deviance]
    )


def test_gamma_prior_profile_memoizes_exact_repeats() -> None:
    """The outer loop re-evaluates accepted points bitwise-identically."""
    y, weights = _distinct_weight_fixture()
    profile_data = prepare_gamma_reml_scale_data(y, weights, weight_semantics="prior")

    first = profile_gamma_reml_scale(profile_data, 4321.0, 3.0)
    repeat = profile_gamma_reml_scale(profile_data, 4321.0, 3.0)
    moved = profile_gamma_reml_scale(profile_data, 4322.0, 3.0)

    assert repeat is first
    assert moved is not first


def test_gamma_prior_warm_start_matches_cold_solves_at_solver_tolerance() -> None:
    """Roots found from the previous solve's bracket equal the cold roots.

    The warm path runs the same ``brentq`` tolerances inside a smaller
    bracket, so its root may differ from the fixed-window root only within
    the solver's own placement freedom (xtol 1e-12 in log shape).
    """
    y, weights = _distinct_weight_fixture()
    warm_data = prepare_gamma_reml_scale_data(y, weights, weight_semantics="prior")

    deviances = [4321.0, 4325.7, 4329.9, 4329.9012, 4329.9012345]
    warm_terms = [profile_gamma_reml_scale(warm_data, dp, 3.0) for dp in deviances]
    for dp, warm in zip(deviances, warm_terms, strict=True):
        cold_data = prepare_gamma_reml_scale_data(y, weights, weight_semantics="prior")
        cold = profile_gamma_reml_scale(cold_data, dp, 3.0)
        assert np.log(warm.inverse_phi) == pytest.approx(np.log(cold.inverse_phi), abs=5.0e-12)
        assert _term_fields(warm) == pytest.approx(_term_fields(cold), rel=1.0e-11)


def test_gamma_prior_warm_start_falls_back_cold_after_a_deviance_jump() -> None:
    """A root far outside the warm ladder still resolves via the cold window."""
    y, weights = _distinct_weight_fixture()
    warm_data = prepare_gamma_reml_scale_data(y, weights, weight_semantics="prior")

    profile_gamma_reml_scale(warm_data, 4321.0, 3.0)
    profile_gamma_reml_scale(warm_data, 4325.0, 3.0)
    jumped = profile_gamma_reml_scale(warm_data, 4321.0e12, 3.0)

    cold_data = prepare_gamma_reml_scale_data(y, weights, weight_semantics="prior")
    cold = profile_gamma_reml_scale(cold_data, 4321.0e12, 3.0)
    assert np.log(jumped.inverse_phi) == pytest.approx(np.log(cold.inverse_phi), abs=5.0e-12)


def test_gamma_prior_derivative_is_deferred_until_read_then_cached(monkeypatch) -> None:
    """The trigamma pass runs only when the derivative is consumed."""
    y, weights = _distinct_weight_fixture()
    profile_data = prepare_gamma_reml_scale_data(y, weights, weight_semantics="prior")

    calls = {"n": 0}
    real = GammaScaleProfileData.scaled_curvature

    def counting(self, shape, penalty_nullity):
        calls["n"] += 1
        return real(self, shape, penalty_nullity)

    monkeypatch.setattr(GammaScaleProfileData, "scaled_curvature", counting)
    term = profile_gamma_reml_scale(profile_data, 4321.0, 3.0)
    assert calls["n"] == 0

    first_read = term.d_inverse_phi_d_penalized_deviance
    assert calls["n"] == 1
    assert term.d_inverse_phi_d_penalized_deviance == first_read
    assert calls["n"] == 1

    monkeypatch.undo()
    expected = _gamma_inverse_shape_derivative(
        term.inverse_phi,
        profile_data.scaled_curvature(term.inverse_phi, 3.0),
    )
    assert first_read == expected


def test_gamma_prior_term_pickles_to_a_plain_eager_term() -> None:
    """Serialization must not drag the retained weight arrays along."""
    import pickle

    from superglm.reml.scale import ProfiledScaleTerm

    y, weights = _distinct_weight_fixture()
    profile_data = prepare_gamma_reml_scale_data(y, weights, weight_semantics="prior")
    term = profile_gamma_reml_scale(profile_data, 4321.0, 3.0)

    restored = pickle.loads(pickle.dumps(term))
    assert type(restored) is ProfiledScaleTerm
    assert np.array_equal(_term_fields(restored), _term_fields(term))


def test_gamma_prior_sorted_dispatch_matches_masked_dispatch_bitwise() -> None:
    """Slice selectors on sorted arguments reproduce the mask path exactly."""
    from superglm.reml.scale import (
        _gamma_saturated_normalizer_array,
        _scaled_trigamma_minus_inverse_array,
        _shape_times_log_minus_digamma_array,
    )

    rng = np.random.default_rng(11)
    argument = np.sort(10.0 ** rng.uniform(-6.0, 4.0, 20_000))
    for helper in (
        _shape_times_log_minus_digamma_array,
        _gamma_saturated_normalizer_array,
        _scaled_trigamma_minus_inverse_array,
    ):
        masked = helper(argument)
        sliced = helper(argument, assume_sorted=True)
        assert np.array_equal(masked.view(np.uint64), sliced.view(np.uint64))


def test_gamma_frequency_arm_never_enters_the_prior_accelerations(monkeypatch) -> None:
    """The load-bearing constraint: `"frequency"` runs the shipped code path.

    The prior arm's memo, warm-start solver and per-solve score stash exist
    because its saturated term is a per-distinct-weight reduction. The
    frequency arm's is a single scalar, has nothing to accelerate, and
    reproduces a shipped release bit for bit -- so it must not merely *agree*
    with the shipped body, it must BE it. Anything else and the next change to
    the accelerations can move a released number.
    """
    from superglm.reml import scale as scale_module

    rng = np.random.default_rng(11)
    n = 400
    weights = rng.uniform(0.5, 4.0, n)
    y = rng.gamma(5.0, np.exp(0.3 + rng.normal(0.0, 0.3, n)) / 5.0)
    data = scale_module.prepare_gamma_reml_scale_data(y, weights, weight_semantics="frequency")

    def _forbidden(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("the frequency arm must not use the prior-arm warm solver")

    monkeypatch.setattr(scale_module, "_solve_gamma_profile_root_warm", _forbidden)

    # Several solves at moving deviances: enough that a warm path would engage.
    terms = [
        scale_module.profile_gamma_reml_scale(
            data,
            penalized_deviance=float(np.sum(weights)) * factor,
            penalty_nullity=4.0,
        )
        for factor in (0.30, 0.31, 0.32, 0.33)
    ]

    assert not data._profile_cache, "the frequency arm must not populate the memo"
    assert not data._warm_history, "the frequency arm must not record warm-start history"
    # Eager terms: the deferred derivative is a prior-arm object.
    for term in terms:
        assert type(term) is scale_module.ProfiledScaleTerm

    # And re-solving the same input reproduces every field exactly, with the
    # caches still empty -- i.e. the repeat came from the solver, not a memo.
    repeat = scale_module.profile_gamma_reml_scale(
        data,
        penalized_deviance=float(np.sum(weights)) * 0.30,
        penalty_nullity=4.0,
    )
    assert repeat.phi == terms[0].phi
    assert repeat.inverse_phi == terms[0].inverse_phi
    assert repeat.criterion == terms[0].criterion
    assert repeat.d_inverse_phi_d_penalized_deviance == terms[0].d_inverse_phi_d_penalized_deviance
    assert not data._profile_cache
