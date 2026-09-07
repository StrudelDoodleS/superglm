"""Behavioral mathematics for the normalized NB2 LSS row kernel."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests._distributional_family_kernels import negative_binomial as nb_kernel
from tests._negative_binomial_lss_oracles import (
    NEGATIVE_BINOMIAL_LSS_CASES,
    NegativeBinomialLSSOracleCase,
)

NegativeBinomialDerivativeRepresentationError = (
    nb_kernel.NegativeBinomialDerivativeRepresentationError
)
NegativeBinomialNumericalDomainError = nb_kernel.NegativeBinomialNumericalDomainError
NegativeBinomialPoissonBoundaryError = nb_kernel.NegativeBinomialPoissonBoundaryError
evaluate_negative_binomial_rows = nb_kernel.evaluate_negative_binomial_rows


def test_nb2_initializer_and_poisson_evidence_accept_only_primitive_rows() -> None:
    response = np.array([0.0, 1.0, 2.0, 4.0])
    counts = response.astype(np.int64)
    weights = np.ones(4)

    theta = nb_kernel.initialize_negative_binomial(
        response,
        counts,
        weights,
        "frequency",
    )

    assert theta.shape == (4, 2)
    assert np.all(np.isfinite(theta))
    assert np.all(theta > 0.0)
    assert not theta.flags.writeable
    assert isinstance(
        nb_kernel.has_resolved_poisson_boundary(
            response,
            counts,
            weights,
            "frequency",
        ),
        bool,
    )


@pytest.mark.parametrize(
    ("response", "counts", "weights"),
    [
        ([0.0, 1.0, 2.0, 4.0], np.array([0, 1, 2, 4]), np.ones(4)),
        (np.array([0.0, 1.0, 2.0, 4.0], dtype=np.float32), np.array([0, 1, 2, 4]), np.ones(4)),
        (np.array([0.0, 1.0, 2.0, 4.0]), [0, 1, 2, 4], np.ones(4)),
        (
            np.array([0.0, 1.0, 2.0, 4.0]),
            np.array([0.0, 1.0, 2.0, 4.0], dtype=np.float32),
            np.ones(4),
        ),
        (np.array([0.0, 1.0, 2.0, 4.0]), np.array([0, 1, 2, 4]), [1.0, 1.0, 1.0, 1.0]),
        (np.array([0.0, 1.0, 2.0, 4.0]), np.array([0, 1, 2, 4]), np.ones(4, dtype=np.int64)),
    ],
    ids=(
        "response-list",
        "response-float32",
        "count-list",
        "count-float32",
        "weight-list",
        "weight-int64",
    ),
)
def test_nb2_initialization_primitives_require_literal_numpy_rows(
    response: object,
    counts: object,
    weights: object,
) -> None:
    with pytest.raises(ValueError, match="NumPy|float64"):
        nb_kernel.initialize_negative_binomial(response, counts, weights, "frequency")
    with pytest.raises(ValueError, match="NumPy|float64"):
        nb_kernel.has_resolved_poisson_boundary(response, counts, weights, "frequency")


def _evaluate_case(
    case: NegativeBinomialLSSOracleCase,
    *,
    derivative_order: int = 2,
):
    return evaluate_negative_binomial_rows(
        np.array([case.count]),
        np.array([case.mean]),
        np.array([case.theta]),
        np.array([case.weight]),
        case.semantics,
        derivative_order=derivative_order,
    )


def _assert_enclosed(
    actual: np.ndarray,
    expected: tuple[float, ...],
    rtol: tuple[float, ...],
    atol: tuple[float, ...],
) -> None:
    actual_values = np.asarray(actual)
    expected_values = np.asarray(expected)
    error = np.abs(actual_values - expected_values)
    bound = np.asarray(atol) + np.asarray(rtol) * np.abs(expected_values)
    assert np.all(error <= bound), f"error {error!r} exceeds bound {bound!r}"


@pytest.mark.parametrize(
    "case",
    NEGATIVE_BINOMIAL_LSS_CASES,
    ids=lambda case: case.id,
)
def test_row_kernel_matches_independent_normalized_oracle(
    case: NegativeBinomialLSSOracleCase,
) -> None:
    """A wrong NB2 law, exposure transform, or large-theta sign fails here."""

    assert len(NEGATIVE_BINOMIAL_LSS_CASES) == 4
    result = _evaluate_case(case)

    np.testing.assert_allclose(
        result.optimizing_log_likelihood,
        [case.optimizing_log_likelihood],
        rtol=0.0,
        atol=case.value_atol,
    )
    np.testing.assert_allclose(
        result.optimizing_log_likelihood + case.factorial_carrier,
        [case.full_log_likelihood],
        rtol=0.0,
        atol=case.value_atol,
    )
    assert result.score is not None
    _assert_enclosed(
        result.score[0],
        case.natural_score,
        case.score_rtol,
        case.score_atol,
    )
    assert result.hessian_packed is not None
    _assert_enclosed(
        result.hessian_packed[0],
        case.natural_hessian_packed,
        case.hessian_rtol,
        case.hessian_atol,
    )
    assert result.valid.tolist() == [True]


def test_well_conditioned_score_and_hessian_match_finite_differences() -> None:
    """Missing chain factors or the observed cross channel fail this check."""

    point = np.array([2.4, 1.8])
    count = np.array([6])
    weight = np.array([0.75])

    def evaluate(candidate: np.ndarray, order: int):
        return evaluate_negative_binomial_rows(
            count,
            np.array([candidate[0]]),
            np.array([candidate[1]]),
            weight,
            "prior",
            derivative_order=order,
        )

    analytic = evaluate(point, 2)
    assert analytic.score is not None and analytic.hessian_packed is not None
    steps = np.cbrt(np.finfo(np.float64).eps) * np.maximum(1.0, point)
    score_difference = np.empty(2)
    hessian_difference = np.empty((2, 2))
    for coordinate, step in enumerate(steps):
        upper = point.copy()
        lower = point.copy()
        upper[coordinate] += step
        lower[coordinate] -= step
        upper_value = evaluate(upper, 1)
        lower_value = evaluate(lower, 1)
        score_difference[coordinate] = (
            upper_value.optimizing_log_likelihood[0] - lower_value.optimizing_log_likelihood[0]
        ) / (2.0 * step)
        assert upper_value.score is not None and lower_value.score is not None
        hessian_difference[:, coordinate] = (upper_value.score[0] - lower_value.score[0]) / (
            2.0 * step
        )

    expected_hessian = np.array(
        [
            [analytic.hessian_packed[0, 0], analytic.hessian_packed[0, 1]],
            [analytic.hessian_packed[0, 1], analytic.hessian_packed[0, 2]],
        ]
    )
    np.testing.assert_allclose(analytic.score[0], score_difference, rtol=3.0e-7, atol=3.0e-8)
    np.testing.assert_allclose(expected_hessian, hessian_difference, rtol=3.0e-7, atol=3.0e-8)


def test_derivative_order_controls_the_returned_channels() -> None:
    case = NEGATIVE_BINOMIAL_LSS_CASES[2]
    order_zero = _evaluate_case(case, derivative_order=0)
    order_one = _evaluate_case(case, derivative_order=1)
    order_two = _evaluate_case(case, derivative_order=2)

    np.testing.assert_array_equal(
        order_zero.optimizing_log_likelihood, order_two.optimizing_log_likelihood
    )
    assert order_zero.score is None and order_zero.hessian_packed is None
    assert order_one.score is not None and order_one.hessian_packed is None
    np.testing.assert_array_equal(order_one.score, order_two.score)
    assert order_two.hessian_packed is not None


@pytest.mark.parametrize(
    ("counts", "mean", "theta", "weights", "semantics", "message"),
    [
        ([0.5], [1.0], [1.0], [1.0], "prior", "counts"),
        ([0], [0.0], [1.0], [1.0], "prior", "strictly positive"),
        ([0], [1.0], [np.nan], [1.0], "prior", "finite"),
        ([0], [1.0], [1.0], [-1.0], "prior", "strictly positive"),
        ([0], [1.0], [1.0], [1.5], "frequency", "frequency weights"),
    ],
)
def test_invalid_row_states_are_refused_before_evaluation(
    counts: list[float],
    mean: list[float],
    theta: list[float],
    weights: list[float],
    semantics: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_negative_binomial_rows(
            np.array(counts),
            np.array(mean),
            np.array(theta),
            np.array(weights),
            semantics,  # type: ignore[arg-type]
        )


def test_unsupported_exponent_ratio_has_typed_numerical_domain_refusal() -> None:
    """A finite input can still lie outside the supported numerical domain."""

    with pytest.raises(FloatingPointError) as caught:
        evaluate_negative_binomial_rows(
            np.array([0]),
            np.array([1.0e308]),
            np.array([1.0e-308]),
            np.ones(1),
            "prior",
            derivative_order=0,
        )

    assert caught.type.__name__ == "NegativeBinomialNumericalDomainError"
    assert "numerical domain" in str(caught.value).lower()


_DOMAIN_LOW = math.ldexp(1.0, -450)
_DOMAIN_HIGH = math.ldexp(1.0, 450)


@pytest.mark.parametrize(
    ("count", "mean", "theta", "weight", "expected_error"),
    [
        (0, _DOMAIN_LOW, _DOMAIN_LOW, 1.0, None),
        (0, _DOMAIN_HIGH, _DOMAIN_HIGH, 1.0, None),
        (0, 1.0, 1.0, _DOMAIN_LOW, None),
        (0, 1.0, 1.0, _DOMAIN_HIGH, None),
        (0, 1.0, 2.0**26, 1.0, None),
        (0, 2.0**26, 1.0, 1.0, None),
        (1, _DOMAIN_LOW, 16.0 * _DOMAIN_LOW, 1.0, None),
        (0, np.nextafter(_DOMAIN_LOW, 0.0), _DOMAIN_LOW, 1.0, NegativeBinomialNumericalDomainError),
        (0, _DOMAIN_LOW, np.nextafter(_DOMAIN_LOW, 0.0), 1.0, NegativeBinomialNumericalDomainError),
        (
            0,
            np.nextafter(_DOMAIN_HIGH, np.inf),
            _DOMAIN_HIGH,
            1.0,
            NegativeBinomialNumericalDomainError,
        ),
        (
            0,
            _DOMAIN_HIGH,
            np.nextafter(_DOMAIN_HIGH, np.inf),
            1.0,
            NegativeBinomialNumericalDomainError,
        ),
        (0, 1.0, 1.0, np.nextafter(_DOMAIN_LOW, 0.0), NegativeBinomialNumericalDomainError),
        (0, 1.0, 1.0, np.nextafter(_DOMAIN_HIGH, np.inf), NegativeBinomialNumericalDomainError),
        (0, 1.0, np.nextafter(2.0**26, np.inf), 1.0, NegativeBinomialPoissonBoundaryError),
        (0, np.nextafter(2.0**26, np.inf), 1.0, 1.0, NegativeBinomialNumericalDomainError),
    ],
)
def test_numerical_domain_endpoints_and_ratio_boundary(
    count: int,
    mean: float,
    theta: float,
    weight: float,
    expected_error: type[NegativeBinomialNumericalDomainError] | None,
) -> None:
    arguments = (
        np.array([count]),
        np.array([mean]),
        np.array([theta]),
        np.array([weight]),
        "prior",
    )
    with np.errstate(all="raise"):
        if expected_error is None:
            result = evaluate_negative_binomial_rows(*arguments, derivative_order=2)
            assert result.score is not None and result.hessian_packed is not None
            assert all(
                np.all(np.isfinite(values))
                for values in (
                    result.optimizing_log_likelihood,
                    result.score,
                    result.hessian_packed,
                )
            )
            return
        with pytest.raises(expected_error) as caught:
            evaluate_negative_binomial_rows(*arguments, derivative_order=0)
    assert caught.type is expected_error


def test_series_candidate_boundary_is_accurate_and_conditioned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    theta = np.array([np.nextafter(20.0, 0.0), np.nextafter(20.0, np.inf), 25.0])
    counts, means = np.full(3, 2, dtype=np.int64), np.full(3, 1.25)
    _, remainder_error, remainder_scale = nb_kernel._ratio_remainder(means / theta)
    use, _, _ = nb_kernel._theta_series(counts, means, theta, remainder_error, remainder_scale, 2)
    np.testing.assert_array_equal(use, [[False, False], [True, True], [True, False]])

    observed = np.empty((3, 2))
    for row, value in enumerate(theta):
        result = evaluate_negative_binomial_rows(
            np.array([2]), np.array([1.25]), np.array([value]), np.ones(1), "prior"
        )
        assert result.score is not None and result.hessian_packed is not None
        observed[row] = result.score[0, 1], result.hessian_packed[0, 2]

    # Frozen from a 100-digit evaluation of the combined recurrence formulas.
    expected = np.array(
        [
            [0.0017003081555539536, -0.00016549757157765076],
            [0.0017003081555539523, -0.00016549757157765057],
            [0.0010999457206778871, -0.00008609266191683774],
        ]
    )
    tolerance = 8.0 * np.finfo(np.float64).eps
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=tolerance)
    np.testing.assert_allclose(
        np.diff(observed[:2], axis=0),
        np.diff(expected[:2], axis=0),
        rtol=0.0,
        atol=2.0 * tolerance,
    )

    arguments = (
        counts[-1:],
        means[-1:],
        theta[-1:],
        remainder_error[-1:],
        remainder_scale[-1:],
        2,
    )
    scaled_power_sums = nb_kernel._scaled_power_sums

    def omit_conditioning(count: int):
        values, errors = scaled_power_sums(count)
        return values, np.zeros_like(errors)

    monkeypatch.setattr(nb_kernel, "_scaled_power_sums", omit_conditioning)
    np.testing.assert_array_equal(nb_kernel._theta_series(*arguments)[0], [[True, True]])


@pytest.mark.parametrize("raw_count", [2, 7, 100_000])
def test_scaled_power_sums_are_fixed_width_and_enclose_roundoff(raw_count: int) -> None:
    class NoPowerInt(int):
        def __pow__(self, exponent: object, modulo: object = None) -> int:
            del exponent, modulo
            raise AssertionError("integer exponentiation must not be used")

    count = NoPowerInt(raw_count)
    values, errors = nb_kernel._scaled_power_sums(count)

    assert values.dtype == errors.dtype == np.dtype(np.float64)
    assert np.all(np.isfinite(values)) and np.all(np.isfinite(errors))
    expected = np.array(
        [
            math.fsum((float(offset) / count) ** power for offset in range(count)) / count
            for power in range(12)
        ]
    )
    assert np.all(np.abs(values - expected) <= errors)


def test_recurrence_tiles_never_exceed_the_cell_budget() -> None:
    counts = np.full(257, 256, dtype=np.int64)
    tiles = list(nb_kernel._iter_recurrence_tiles(counts))
    cells = [len(rows) * width for rows, _, width in tiles]

    assert max(cells) == 65_536
    assert sum(cells) == int(np.sum(counts, dtype=np.int64))


def test_requested_derivative_orders_are_actually_lazy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("unrequested derivative work was entered")

    monkeypatch.setattr(nb_kernel, "_theta_series", unexpected)
    result = evaluate_negative_binomial_rows(
        np.array([1]),
        np.array([1.0]),
        np.array([1.0]),
        np.ones(1),
        "prior",
        derivative_order=0,
    )
    assert result.score is None and result.hessian_packed is None

    monkeypatch.undo()
    for order in (0, 1):
        _, natural, direct = nb_kernel._recurrences(
            np.array([1]),
            np.ones(1),
            np.ones(1),
            order,
            np.ones(1, dtype=np.bool_),
        )
        assert natural.shape == direct.shape == (1, order)

    monkeypatch.setattr(nb_kernel, "_series_hessian_step", unexpected)
    theta = np.array([np.nextafter(20.0, np.inf)])
    _, remainder_error, remainder_scale = nb_kernel._ratio_remainder(np.array([1.25]) / theta)
    nb_kernel._theta_series(
        np.array([2]), np.array([1.25]), theta, remainder_error, remainder_scale, 1
    )


def test_public_evaluator_wires_direct_log_channel_retention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RetentionReachedError(Exception):
        pass

    def reached(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RetentionReachedError

    monkeypatch.setattr(nb_kernel, "_retain_log_channels", reached)
    with pytest.raises(RetentionReachedError):
        evaluate_negative_binomial_rows(
            np.array([0]), np.ones(1), np.ones(1), np.ones(1), "prior", derivative_order=1
        )


def test_direct_log_channel_adversary_has_typed_representation_refusal() -> None:
    assert issubclass(
        NegativeBinomialDerivativeRepresentationError,
        NegativeBinomialNumericalDomainError,
    )
    with pytest.raises(NegativeBinomialDerivativeRepresentationError, match="log-link"):
        nb_kernel._retain_log_channels(
            np.zeros(1, dtype=np.int64),
            np.ones(1),
            np.ones(1),
            np.zeros((1, 2)),
            None,
            np.array([[0.0, 1.0]]),
            None,
            np.zeros((1, 2)),
            None,
        )


def test_book_shaped_counts_are_accepted() -> None:
    """No clock in this test: wall time is not a property of the code on a shared runner.
    The exact recurrence-cell cost model is asserted instead."""
    rng = np.random.default_rng(2)
    n = 2_000_000
    counts = rng.negative_binomial(0.8, 0.8 / (0.6 + 0.8), size=n).astype(np.float64)
    assert counts.sum() > 1_000_000  # the old ceiling refused this
    assert nb_kernel.recurrence_cells(counts) == int(counts.sum())
    assert nb_kernel.recurrence_cells(counts) <= nb_kernel._MAX_RECURRENCE_CELLS
    mean = np.full(n, 0.6)
    theta = np.full(n, 0.8)
    weights = np.ones(n)
    out = evaluate_negative_binomial_rows(counts, mean, theta, weights, "prior", derivative_order=2)
    assert np.all(np.isfinite(out.optimizing_log_likelihood))


def test_ceiling_message_names_the_budget_and_the_alternatives() -> None:
    counts = np.array([3.0e9])
    ones = np.ones(1)
    with pytest.raises(ValueError) as info:
        evaluate_negative_binomial_rows(counts, ones, ones, ones, "prior", derivative_order=0)
    text = str(info.value)
    assert "recurrence cells" in text and "frequency weights" in text and "offset" in text
    assert f"{nb_kernel._MAX_RECURRENCE_CELLS:.0e}" in text
    assert "1_000_000" not in text and "100_000" not in text


def test_a_single_large_count_is_no_longer_refused_below_the_budget() -> None:
    counts = np.array([250_000.0])
    ones = np.ones(1)
    out = evaluate_negative_binomial_rows(
        counts, ones * 10.0, ones, ones, "prior", derivative_order=2
    )
    assert np.isfinite(out.optimizing_log_likelihood[0])
