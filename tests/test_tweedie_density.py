from __future__ import annotations

import math
import warnings
from dataclasses import FrozenInstanceError
from decimal import Decimal

import numpy as np
import pytest

import superglm._tweedie_density as density_module
from superglm._tweedie_density import (
    TweedieDensityDiagnostics,
    TweedieDensityError,
    approximate_tweedie_logpdf,
    evaluate_tweedie_density,
)


def _independent_compound_poisson_sum(
    y: float,
    mu: float,
    phi: float,
    p: float,
    *,
    weight: float = 1.0,
    last_j: int = 5_000,
) -> tuple[float, float]:
    """Direct finite sum used only as a test oracle."""
    effective_phi = phi / weight
    alpha = (2.0 - p) / (p - 1.0)
    lam = mu ** (2.0 - p) / (effective_phi * (2.0 - p))
    scale = effective_phi * (p - 1.0) * mu ** (p - 1.0)
    z = y / scale

    indices = np.arange(1, last_j + 1, dtype=np.float64)
    log_terms = np.asarray(
        [
            -lam
            - z
            - math.log(y)
            + j * (math.log(lam) + alpha * math.log(z))
            - math.lgamma(j + 1.0)
            - math.lgamma(j * alpha)
            for j in indices
        ],
        dtype=np.float64,
    )
    largest = float(np.max(log_terms))
    scaled = np.exp(log_terms - largest)
    scaled_total = math.fsum(float(value) for value in scaled)
    logpdf = largest + math.log(scaled_total)
    scores = lam + z - indices * (1.0 + alpha)
    score = (
        math.fsum(float(term * component) for term, component in zip(scaled, scores, strict=True))
        / scaled_total
    )
    return logpdf, score


def _valid_arguments() -> dict[str, object]:
    return {
        "y": np.array([0.0, 0.5, 3.0], dtype=np.float64),
        "mu": np.array([0.3, 1.2, 2.7], dtype=np.float64),
        "phi": 0.7,
        "p": 1.5,
        "weights": np.array([0.8, 1.0, 1.4], dtype=np.float64),
    }


def test_alpha_one_hand_value() -> None:
    result = evaluate_tweedie_density(np.array([1.0]), np.array([1.0]), 2.0, 1.5, rtol=1e-14)

    assert result.logpdf[0] == pytest.approx(-1.5358655264538403, abs=2e-15)
    assert result.log_phi_score[0] == pytest.approx(
        -0.86625485344462351663,
        abs=2e-15,
    )


def test_integer_shape_reference_has_independent_score_oracle() -> None:
    result = evaluate_tweedie_density(
        np.array([0.7]),
        np.array([1.0]),
        0.5,
        1.25,
    )

    assert result.logpdf[0] == pytest.approx(
        -0.51678334701230611579493075833575636135,
        abs=2e-14,
    )
    assert result.log_phi_score[0] == pytest.approx(
        -0.46551727371437731971659437541153359802,
        abs=4e-13,
    )


@pytest.mark.slow
def test_large_mode_bessel_oracle_is_accurate_or_fails_closed() -> None:
    try:
        result = evaluate_tweedie_density(
            np.array([1.0]),
            np.array([1.0]),
            1e-8,
            1.5,
        )
    except TweedieDensityError as error:
        assert error.reason == "arithmetic precision was insufficient for certification"
    else:
        assert result.logpdf[0] == pytest.approx(
            8.2914018378340099931197610797683626198,
            abs=8e-13,
        )
        assert result.log_phi_score[0] == pytest.approx(
            -0.50000000093750000234375000769042972046,
            abs=8e-13,
        )


@pytest.mark.slow
@pytest.mark.parametrize("p", [1.000001, 1.999999])
def test_default_certification_reaches_one_millionth_from_power_boundaries(
    p: float,
) -> None:
    y = np.array([0.02, 0.5, 3.0]) if p > 1.5 else np.array([0.02])
    mu = np.array([0.3, 1.2, 2.7]) if p > 1.5 else np.array([0.3])
    result = evaluate_tweedie_density(
        y,
        mu,
        0.7,
        p,
    )

    assert np.all(np.isfinite(result.logpdf))
    assert np.all(np.isfinite(result.log_phi_score))
    assert result.diagnostics.certified
    assert result.diagnostics.max_relative_tail_error <= 1e-12


def test_hard_near_poisson_boundary_reference() -> None:
    result = evaluate_tweedie_density(
        np.array([0.04564326798684731]),
        np.array([2.859891821890267]),
        0.10602153698295053,
        1.05,
        rtol=1e-13,
    )

    assert result.logpdf[0] == pytest.approx(-25.2177010089, abs=2e-10)
    assert result.log_phi_score[0] == pytest.approx(15.110538041957676, abs=2e-10)


def test_forward_ratio_interval_includes_compound_parameter_rounding() -> None:
    parameters = density_module._compound_parameters(
        36.2680080641078,
        0.24557887232637282,
        0.01795018649220772,
        1.000001,
        5.125698518755229,
        observation_index=0,
        requested_rtol=1e-12,
    )
    _, lower, upper = density_module._forward_ratio_interval(10_356, parameters)
    exact_binary64_oracle = Decimal(
        "-14.592667872649328853075502161521302476976536507251147198992648646212314862"
    )

    assert Decimal(str(lower)) <= exact_binary64_oracle <= Decimal(str(upper))


def test_score_base_interval_includes_compound_parameter_rounding() -> None:
    parameters = density_module._compound_parameters(
        0.003714329311831774,
        5429.239279458794,
        0.6434415608478098,
        1.05,
        45.39168221894365,
        observation_index=0,
        requested_rtol=1e-12,
    )
    value, radius = density_module._score_base_and_radius(parameters, mode=1)
    exact_binary64_oracle = Decimal(
        "262251.585719146993416813247882197759572714209163424739519455794865890010968"
    )

    assert abs(Decimal(str(value)) - exact_binary64_oracle) <= Decimal(str(radius))


def test_direct_log_term_interval_includes_large_shape_centering_rounding() -> None:
    parameters = density_module._compound_parameters(
        225.79043412290116,
        165.50820560474105,
        0.07037373195054972,
        1.000001,
        28.612392826284182,
        observation_index=0,
        requested_rtol=1e-12,
    )
    value, radius = density_module._stable_log_term(91_788, parameters)
    exact_binary64_oracle = Decimal(
        "-4917.825206050816750983078543854533639156646129424958303233118543614665476"
    )

    assert abs(Decimal(str(value)) - exact_binary64_oracle) <= Decimal(str(radius))


def test_tail_tolerance_does_not_misclassify_nearest_float_logpdf_rounding() -> None:
    result = evaluate_tweedie_density(
        np.array([5.0]),
        np.array([0.1]),
        0.3,
        1.8,
        rtol=16.0 * np.finfo(np.float64).eps,
    )
    exact_binary64_oracle = Decimal(
        "-115.006384517331161957922511789644405214682101347006918288811218744748914362"
    )
    output_error = abs(Decimal.from_float(float(result.logpdf[0])) - exact_binary64_oracle)
    half_output_ulp = Decimal.from_float(math.ulp(float(result.logpdf[0]))) / Decimal(2)

    assert output_error <= half_output_ulp
    assert result.diagnostics.max_relative_tail_error <= result.diagnostics.requested_rtol


@pytest.mark.parametrize("p", [1.0001, 1.05, 1.5, 1.95, 1.9999])
@pytest.mark.parametrize(
    ("y", "mu"),
    [(0.02, 0.3), (0.5, 1.2), (3.0, 2.7)],
)
def test_log_phi_score_matches_centered_finite_difference(p: float, y: float, mu: float) -> None:
    phi = 0.7
    step = 2e-6
    y_array = np.array([y])
    mu_array = np.array([mu])

    result = evaluate_tweedie_density(y_array, mu_array, phi, p)
    plus = evaluate_tweedie_density(y_array, mu_array, phi * math.exp(step), p)
    minus = evaluate_tweedie_density(y_array, mu_array, phi * math.exp(-step), p)
    finite_difference = (plus.logpdf[0] - minus.logpdf[0]) / (2.0 * step)

    assert result.log_phi_score[0] == pytest.approx(finite_difference, rel=3e-7, abs=3e-8)


@pytest.mark.parametrize(
    "case",
    [
        (0.4, 0.7, 0.9, 1.2, 1.0),
        (2.0, 1.0, 0.2, 1.5, 1.0),
        (4.0, 3.0, 1.1, 1.8, 2.5),
    ],
)
def test_exact_series_matches_independent_direct_sum(
    case: tuple[float, float, float, float, float],
) -> None:
    y, mu, phi, p, weight = case
    expected_logpdf, expected_score = _independent_compound_poisson_sum(
        y, mu, phi, p, weight=weight
    )

    result = evaluate_tweedie_density(
        np.array([y]),
        np.array([mu]),
        phi,
        p,
        weights=np.array([weight]),
        rtol=1e-13,
    )

    assert result.logpdf[0] == pytest.approx(expected_logpdf, rel=2e-13, abs=2e-13)
    assert result.log_phi_score[0] == pytest.approx(expected_score, rel=2e-12, abs=2e-12)


def test_zero_mass_and_prior_weight_convention() -> None:
    y = np.zeros(3)
    mu = np.array([0.3, 1.2, 2.7])
    weights = np.array([0.5, 1.0, 2.5])
    phi = 0.7
    p = 1.4
    expected_rate = weights * mu ** (2.0 - p) / (phi * (2.0 - p))

    result = evaluate_tweedie_density(y, mu, phi, p, weights=weights)

    np.testing.assert_allclose(result.logpdf, -expected_rate, rtol=2e-15, atol=0.0)
    np.testing.assert_allclose(result.log_phi_score, expected_rate, rtol=2e-15, atol=0.0)
    assert result.diagnostics.n_positive == 0
    assert result.diagnostics.n_exact == 3
    assert result.diagnostics.n_approximate == 0
    assert result.diagnostics.max_terms == 0
    assert result.diagnostics.exact
    assert result.diagnostics.certified


def test_prior_weights_equal_inverse_dispersion_row_by_row() -> None:
    args = _valid_arguments()
    weighted = evaluate_tweedie_density(**args)
    y = args["y"]
    mu = args["mu"]
    phi = float(args["phi"])
    p = float(args["p"])
    weights = args["weights"]
    assert isinstance(y, np.ndarray)
    assert isinstance(mu, np.ndarray)
    assert isinstance(weights, np.ndarray)

    for index, weight in enumerate(weights):
        unweighted = evaluate_tweedie_density(
            y[index : index + 1], mu[index : index + 1], phi / weight, p
        )
        assert weighted.logpdf[index] == pytest.approx(unweighted.logpdf[0], abs=2e-13)
        assert weighted.log_phi_score[index] == pytest.approx(
            unweighted.log_phi_score[0], abs=2e-12
        )


def test_permutation_equivariance_and_single_row_shape() -> None:
    args = _valid_arguments()
    original = evaluate_tweedie_density(**args)
    permutation = np.array([2, 0, 1])
    permuted = evaluate_tweedie_density(
        args["y"][permutation],  # type: ignore[index]
        args["mu"][permutation],  # type: ignore[index]
        args["phi"],
        args["p"],
        weights=args["weights"][permutation],  # type: ignore[index]
    )

    np.testing.assert_array_equal(permuted.logpdf, original.logpdf[permutation])
    np.testing.assert_array_equal(permuted.log_phi_score, original.log_phi_score[permutation])

    single = evaluate_tweedie_density(np.array([0.5]), np.array([1.2]), 0.7, 1.5)
    assert single.logpdf.shape == (1,)
    assert single.log_phi_score.shape == (1,)


def test_response_scaling_identity_preserves_score_and_density_jacobian() -> None:
    y = np.array([0.2, 1.7])
    mu = np.array([0.5, 1.3])
    phi = 0.8
    p = 1.6
    scale = 37.0
    original = evaluate_tweedie_density(y, mu, phi, p)
    transformed = evaluate_tweedie_density(
        scale * y,
        scale * mu,
        phi * scale ** (2.0 - p),
        p,
    )

    np.testing.assert_allclose(
        transformed.logpdf,
        original.logpdf - math.log(scale),
        rtol=0.0,
        atol=3e-12,
    )
    np.testing.assert_allclose(
        transformed.log_phi_score,
        original.log_phi_score,
        rtol=0.0,
        atol=3e-12,
    )


def test_result_is_immutable_owned_and_does_not_alias_inputs() -> None:
    y = np.array([0.0, 0.5])
    mu = np.array([0.8, 1.2])
    weights = np.array([1.0, 2.0])
    result = evaluate_tweedie_density(y, mu, 0.7, 1.5, weights=weights)
    before_logpdf = result.logpdf.copy()
    before_score = result.log_phi_score.copy()

    y[:] = 100.0
    mu[:] = 100.0
    weights[:] = 100.0

    np.testing.assert_array_equal(result.logpdf, before_logpdf)
    np.testing.assert_array_equal(result.log_phi_score, before_score)
    assert result.logpdf.dtype == np.float64
    assert result.log_phi_score.dtype == np.float64
    assert result.logpdf.flags.owndata
    assert result.log_phi_score.flags.owndata
    assert not result.logpdf.flags.writeable
    assert not result.log_phi_score.flags.writeable
    with pytest.raises(ValueError):
        result.logpdf[0] = 0.0
    with pytest.raises(FrozenInstanceError):
        result.diagnostics = result.diagnostics  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.diagnostics.exact = False  # type: ignore[misc]


@pytest.mark.parametrize(
    "bad",
    [
        np.array([True]),
        np.array(["1.0"]),
        np.array([1.0], dtype=object),
        np.array([1.0 + 0.0j]),
    ],
)
@pytest.mark.parametrize("name", ["y", "mu", "weights"])
def test_rejects_non_real_array_dtypes_without_warnings(name: str, bad: np.ndarray) -> None:
    args = _valid_arguments()
    args[name] = bad
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(TypeError):
            evaluate_tweedie_density(**args)


@pytest.mark.parametrize("name", ["y", "mu", "weights"])
@pytest.mark.parametrize("bad", [np.array(1.0), np.ones((1, 1))])
def test_rejects_non_vector_shapes(name: str, bad: np.ndarray) -> None:
    args = _valid_arguments()
    args[name] = bad
    with pytest.raises(ValueError, match="one-dimensional"):
        evaluate_tweedie_density(**args)


def test_rejects_mismatched_array_lengths() -> None:
    args = _valid_arguments()
    args["mu"] = np.ones(2)
    with pytest.raises(ValueError, match="same length"):
        evaluate_tweedie_density(**args)


@pytest.mark.parametrize(
    ("name", "bad"),
    [
        ("y", np.array([0.0, np.nan, 1.0])),
        ("y", np.array([0.0, np.inf, 1.0])),
        ("y", np.array([0.0, -1.0, 1.0])),
        ("mu", np.array([1.0, np.nan, 1.0])),
        ("mu", np.array([1.0, np.inf, 1.0])),
        ("mu", np.array([1.0, 0.0, 1.0])),
        ("mu", np.array([1.0, -1.0, 1.0])),
        ("weights", np.array([1.0, np.nan, 1.0])),
        ("weights", np.array([1.0, np.inf, 1.0])),
        ("weights", np.array([1.0, 0.0, 1.0])),
        ("weights", np.array([1.0, -1.0, 1.0])),
    ],
)
def test_rejects_nonfinite_or_out_of_support_arrays(name: str, bad: np.ndarray) -> None:
    args = _valid_arguments()
    args[name] = bad
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError):
            evaluate_tweedie_density(**args)


@pytest.mark.parametrize("name", ["phi", "p", "rtol"])
@pytest.mark.parametrize("bad", [True, "1.5", [1.5], 1.0 + 0.0j])
def test_rejects_non_real_scalar_controls(name: str, bad: object) -> None:
    args = _valid_arguments()
    args[name] = bad
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(TypeError):
            evaluate_tweedie_density(**args)


@pytest.mark.parametrize("phi", [0.0, -1.0, np.nan, np.inf])
def test_rejects_invalid_dispersion(phi: float) -> None:
    args = _valid_arguments()
    args["phi"] = phi
    with pytest.raises(ValueError):
        evaluate_tweedie_density(**args)


@pytest.mark.parametrize("p", [1.0, 2.0, 0.9, 2.1, np.nan, np.inf])
def test_rejects_invalid_power(p: float) -> None:
    args = _valid_arguments()
    args["p"] = p
    with pytest.raises(ValueError):
        evaluate_tweedie_density(**args)


@pytest.mark.parametrize("rtol", [0.0, -1.0, 1.0, 2.0, np.nan, np.inf])
def test_rejects_invalid_tolerance(rtol: float) -> None:
    args = _valid_arguments()
    args["rtol"] = rtol
    with pytest.raises(ValueError):
        evaluate_tweedie_density(**args)


@pytest.mark.parametrize("max_terms", [True, 1.5, "2", [2]])
def test_rejects_non_integer_term_limit(max_terms: object) -> None:
    args = _valid_arguments()
    args["max_terms"] = max_terms
    with pytest.raises(TypeError):
        evaluate_tweedie_density(**args)


@pytest.mark.parametrize("max_terms", [0, -1])
def test_rejects_nonpositive_term_limit(max_terms: int) -> None:
    args = _valid_arguments()
    args["max_terms"] = max_terms
    with pytest.raises(ValueError):
        evaluate_tweedie_density(**args)


def test_rejects_term_limit_above_hard_safety_cap() -> None:
    args = _valid_arguments()
    args["max_terms"] = 1_000_001
    with pytest.raises(ValueError, match="1000000"):
        evaluate_tweedie_density(**args)


def test_rejects_tolerance_below_float64_certification_floor() -> None:
    args = _valid_arguments()
    args["rtol"] = np.nextafter(16.0 * np.finfo(np.float64).eps, 0.0)
    with pytest.raises(ValueError, match="float64"):
        evaluate_tweedie_density(**args)


def test_forced_certification_failure_has_only_neutral_bounded_metadata() -> None:
    with pytest.raises(TweedieDensityError) as caught:
        evaluate_tweedie_density(
            np.array([2.0]), np.array([1.0]), 0.2, 1.5, rtol=1e-14, max_terms=1
        )

    error = caught.value
    assert error.observation_index == 0
    assert error.power == 1.5
    assert error.dispersion == 0.2
    assert error.term_count == 1
    assert error.requested_rtol == 1e-14
    assert error.reason == "term limit reached before both tails were certified"
    assert set(error.__dict__) == {
        "observation_index",
        "power",
        "dispersion",
        "term_count",
        "requested_rtol",
        "reason",
    }
    message = str(error)
    assert "2.0" not in message
    assert "1.0" not in message
    assert len(message) < 240


def test_first_moment_tail_prevents_mass_only_false_certification() -> None:
    arguments = (
        np.array([0.01]),
        np.array([1.0]),
        2.0,
        1.5,
    )

    # Four terms make the omitted mass small enough, but not its first moment.
    with pytest.raises(TweedieDensityError) as caught:
        evaluate_tweedie_density(*arguments, rtol=5e-12, max_terms=4)
    assert caught.value.reason == "term limit reached before both tails were certified"

    certified = evaluate_tweedie_density(*arguments, rtol=5e-12, max_terms=5)
    assert certified.diagnostics.max_terms == 5
    assert certified.diagnostics.max_relative_tail_error <= 5e-12


def test_far_mode_is_found_without_linear_scanning_and_both_tail_moments_are_accurate() -> None:
    # For alpha=1 this has lambda=10 and z=20, so the mode is far from j=1.
    expected_logpdf, expected_score = _independent_compound_poisson_sum(
        2.0, 1.0, 0.2, 1.5, last_j=1_000
    )
    result = evaluate_tweedie_density(np.array([2.0]), np.array([1.0]), 0.2, 1.5, rtol=1e-14)

    assert result.logpdf[0] == pytest.approx(expected_logpdf, abs=2e-13)
    assert result.log_phi_score[0] == pytest.approx(expected_score, abs=2e-12)
    assert 1 < result.diagnostics.max_terms < 1_000
    assert result.diagnostics.max_relative_tail_error <= 1e-14


def test_mode_at_one_and_large_mode_extremes_either_evaluate_or_fail_closed() -> None:
    mode_one = evaluate_tweedie_density(np.array([1e-8]), np.array([1e-8]), 10.0, 1.5, rtol=1e-12)
    assert np.isfinite(mode_one.logpdf[0])
    assert mode_one.diagnostics.max_terms >= 1

    # p close to two creates a mode in the tens of thousands. It must not be
    # reached by a linear scan, and all returned values must remain finite.
    try:
        far_mode = evaluate_tweedie_density(
            np.array([3.0]), np.array([2.7]), 0.7, 1.9999, rtol=1e-12
        )
    except TweedieDensityError as error:
        assert error.reason in {
            "compound parameters were not representable",
            "series mode could not be bracketed safely",
            "non-finite series arithmetic",
        }
    else:
        assert np.isfinite(far_mode.logpdf[0])
        assert np.isfinite(far_mode.log_phi_score[0])
        assert far_mode.diagnostics.max_terms < 10_000


def test_exact_diagnostics_report_certified_work() -> None:
    result = evaluate_tweedie_density(
        np.array([0.0, 0.4, 2.0]),
        np.array([0.8, 1.0, 1.5]),
        0.6,
        1.6,
        rtol=3e-11,
    )
    diagnostics = result.diagnostics

    assert diagnostics == TweedieDensityDiagnostics(
        n_positive=2,
        n_exact=3,
        n_approximate=0,
        max_terms=diagnostics.max_terms,
        exact=True,
        certified=True,
        requested_rtol=3e-11,
        max_relative_tail_error=diagnostics.max_relative_tail_error,
        method="compound_poisson_series",
    )
    assert diagnostics.max_terms > 0
    assert diagnostics.max_relative_tail_error <= diagnostics.requested_rtol


def test_exact_path_never_calls_saddlepoint(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("exact evaluator called the approximate evaluator")

    monkeypatch.setattr(density_module, "approximate_tweedie_logpdf", fail_if_called)
    result = density_module.evaluate_tweedie_density(np.array([0.5]), np.array([1.0]), 0.7, 1.5)
    assert result.diagnostics.method == "compound_poisson_series"
    assert result.diagnostics.exact


def test_saddlepoint_is_explicitly_approximate_but_zero_rows_remain_exact() -> None:
    y = np.array([0.0, 0.5])
    mu = np.array([0.8, 1.2])
    phi = 0.7
    p = 1.5
    result = approximate_tweedie_logpdf(y, mu, phi, p)

    rate_zero = mu[0] ** (2.0 - p) / (phi * (2.0 - p))
    assert result.logpdf[0] == pytest.approx(-rate_zero)
    assert result.log_phi_score[0] == pytest.approx(rate_zero)
    assert np.isfinite(result.logpdf[1])
    assert np.isnan(result.log_phi_score[1])
    assert result.diagnostics.n_positive == 1
    assert result.diagnostics.n_exact == 1
    assert result.diagnostics.n_approximate == 1
    assert result.diagnostics.max_terms == 0
    assert not result.diagnostics.exact
    assert not result.diagnostics.certified
    assert result.diagnostics.method == "saddlepoint"
    assert result.logpdf.flags.owndata and not result.logpdf.flags.writeable
    assert result.log_phi_score.flags.owndata and not result.log_phi_score.flags.writeable


def test_saddlepoint_positive_formula_is_independently_reproduced() -> None:
    y = 0.7
    mu = 1.3
    phi = 0.4
    p = 1.7
    first = y ** (2.0 - p) / ((1.0 - p) * (2.0 - p))
    second = y * mu ** (1.0 - p) / (1.0 - p)
    third = mu ** (2.0 - p) / (2.0 - p)
    unit_deviance = 2.0 * (first - second + third)
    expected = -0.5 * (
        math.log(2.0 * math.pi) + math.log(phi) + p * math.log(y)
    ) - unit_deviance / (2.0 * phi)

    result = approximate_tweedie_logpdf(
        np.array([y]), np.array([mu]), phi, p, weights=np.array([1.0])
    )

    assert result.logpdf[0] == pytest.approx(expected, rel=2e-14, abs=2e-14)


def test_saddlepoint_extreme_effective_dispersions_are_warning_free_and_finite() -> None:
    largest = np.finfo(np.float64).max
    smallest = np.nextafter(0.0, 1.0)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        diffuse = approximate_tweedie_logpdf(
            np.array([1.0]),
            np.array([1.0]),
            largest,
            1.5,
            weights=np.array([smallest]),
        )
        concentrated = approximate_tweedie_logpdf(
            np.array([1.0]),
            np.array([1.0]),
            smallest,
            1.5,
            weights=np.array([largest]),
        )

    assert np.isfinite(diffuse.logpdf[0])
    assert np.isfinite(concentrated.logpdf[0])
    assert not diffuse.diagnostics.certified
    assert not concentrated.diagnostics.certified


def test_saddlepoint_unrepresentable_zero_mass_fails_without_warnings() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(TweedieDensityError, match="saddlepoint arithmetic"):
            approximate_tweedie_logpdf(
                np.array([0.0]),
                np.array([1.0]),
                np.nextafter(0.0, 1.0),
                1.5,
                weights=np.array([np.finfo(np.float64).max]),
            )
