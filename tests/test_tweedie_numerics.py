import warnings
from decimal import Decimal, localcontext

import numpy as np
import pytest

from superglm import _tweedie_numerics as tweedie_numerics
from superglm._tweedie_numerics import (
    TweedieNumericalError,
    compound_poisson_gamma_parameters,
    normalize_tweedie_power,
    tweedie_unit_deviance,
)


def _decimal_tweedie_unit_deviance(y: float, mu: float, p: float) -> float:
    """Return an independently evaluated, correctly rounded binary64 oracle."""
    with localcontext() as context:
        context.prec = 250
        decimal_y = Decimal.from_float(y)
        decimal_mu = Decimal.from_float(mu)
        decimal_p = Decimal.from_float(p)
        q = Decimal(2) - decimal_p
        r = decimal_p - Decimal(1)
        if y == 0.0:
            deviance = Decimal(2) * decimal_mu**q / q
        else:
            deviance = Decimal(2) * (
                -(decimal_y**q) / (r * q) + decimal_y * decimal_mu ** (-r) / r + decimal_mu**q / q
            )
    return float(deviance)


def _native_and_forced_float64_deviance(monkeypatch, y: float, mu: float, p: float):
    native = float(tweedie_unit_deviance(y, mu, p))
    monkeypatch.setattr(tweedie_numerics, "_DEVIANCE_DTYPE", np.dtype(np.float64))
    forced_float64 = float(tweedie_unit_deviance(y, mu, p))
    return native, forced_float64


def _assert_few_ulp_agreement(actual: float, expected: float, *, max_ulps: int = 3):
    assert abs(actual - expected) <= max_ulps * abs(np.spacing(expected))


@pytest.mark.parametrize(
    ("value", "exception"),
    [
        (True, TypeError),
        (np.bool_(False), TypeError),
        (np.array([1.5]), TypeError),
        (object(), TypeError),
        (np.nan, ValueError),
        pytest.param(10**400, ValueError, id="unrepresentable-real"),
    ],
)
def test_normalize_tweedie_power_rejects_non_real_scalar(value, exception):
    with pytest.raises(exception):
        normalize_tweedie_power(value)


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(np.array([2.0 + 7.0j]), id="complex"),
        pytest.param(np.array(["2.0"]), id="string"),
    ],
)
@pytest.mark.parametrize("name", ["y", "mu"])
def test_unit_deviance_rejects_non_real_arrays_without_cast_warning(name, bad):
    arguments = {"y": np.array([2.0]), "mu": np.array([1.0]), "p": 1.5}
    arguments[name] = bad

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(TypeError, match=name):
            tweedie_unit_deviance(**arguments)


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        (
            "mu",
            {"mu": np.array([1.0 + 2.0j]), "phi": 1.0},
        ),
        (
            "phi",
            {"mu": np.array([1.0]), "phi": np.array([1.0 + 2.0j])},
        ),
        (
            "weights",
            {
                "mu": np.array([1.0]),
                "phi": 1.0,
                "weights": np.array([1.0 + 2.0j]),
            },
        ),
    ],
)
def test_compound_parameters_reject_complex_arrays_without_cast_warning(name, arguments):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(TypeError, match=name):
            compound_poisson_gamma_parameters(p=1.5, **arguments)


@pytest.mark.parametrize("p", [1.000001, 1.01, 1.1, 1.5, 1.99, 1.999999])
def test_unit_deviance_is_exactly_zero_at_equal_values(p):
    values = np.array([1e-20, 1.0, 1e12])
    result = tweedie_unit_deviance(values, values.copy(), p)
    np.testing.assert_array_equal(result, np.zeros_like(values))


def test_unit_deviance_matches_stable_closed_form_at_p_one_half():
    mu = np.array([1e-12, 1.0, 1e12])
    y = mu * np.array([1.0 + 1e-8, 4.0, 1e-8])
    delta = (y - mu) / mu
    root_difference = delta / (np.sqrt(1.0 + delta) + 1.0)
    expected = 4.0 * np.sqrt(mu) * root_difference**2
    np.testing.assert_allclose(tweedie_unit_deviance(y, mu, 1.5), expected, rtol=2e-12, atol=0.0)


def test_unit_deviance_tends_to_poisson_at_lower_power_boundary():
    y = np.array([2.0])
    mu = np.array([1.0])
    p = np.nextafter(1.0, 2.0)
    expected = 2.0 * (y * np.log(y / mu) - (y - mu))

    np.testing.assert_allclose(
        tweedie_unit_deviance(y, mu, p),
        expected,
        rtol=2e-15,
        atol=0.0,
    )


def test_close_unit_deviance_remains_stable_near_lower_power_boundary():
    y = np.array([np.exp(-1e-3)])
    mu = np.array([1.0])
    expected = np.array([9.993335832699509e-7])

    np.testing.assert_allclose(
        tweedie_unit_deviance(y, mu, 1.0 + 1e-8),
        expected,
        rtol=2e-15,
        atol=0.0,
    )


def test_unit_deviance_combines_representable_extreme_ratio_terms():
    y = np.array([1.0])
    mu = np.array([np.nextafter(0.0, 1.0)])
    p = 1.01
    r = p - 1.0
    q = 2.0 - p
    log_mu_over_y = np.log(mu) - np.log(y)
    dominant = np.exp(np.log(y) - r * np.log(mu) - np.log(r))
    correction = -np.expm1(r * log_mu_over_y - np.log(q)) + np.exp(
        np.log(r) - np.log(q) + log_mu_over_y
    )
    expected = 2.0 * dominant * correction

    assert np.all(np.isfinite(expected))
    np.testing.assert_allclose(
        tweedie_unit_deviance(y, mu, p),
        expected,
        rtol=2e-14,
        atol=0.0,
    )


def test_lower_power_deviance_avoids_representable_ratio_product_overflow():
    y = np.array([1.0])
    mu = np.array([1e-308])
    p = np.nextafter(1.0, 2.0)
    r = p - 1.0
    q = 2.0 - p
    log_mu_over_y = np.log(mu) - np.log(y)
    dominant = np.exp(np.log(y) - r * np.log(mu) - np.log(r))
    correction = -np.expm1(r * log_mu_over_y - np.log(q)) + np.exp(
        np.log(r) - np.log(q) + log_mu_over_y
    )
    expected = 2.0 * dominant * correction

    assert np.all(np.isfinite(expected))
    np.testing.assert_allclose(
        tweedie_unit_deviance(y, mu, p),
        expected,
        rtol=2e-14,
        atol=0.0,
    )


def test_unit_deviance_combines_subnormal_scale_before_rounding():
    actual = tweedie_unit_deviance(
        np.array([1e-20]),
        np.array([np.nextafter(0.0, 1.0)]),
        1.0005,
    )

    np.testing.assert_allclose(
        actual,
        np.array([1.708589252424202e-17]),
        rtol=2e-14,
        atol=0.0,
    )


def test_unit_deviance_combines_near_maximum_scale_before_overflow():
    actual = tweedie_unit_deviance(
        np.array([1e308]),
        np.array([np.finfo(np.float64).max]),
        np.nextafter(1.0, 2.0),
    )

    np.testing.assert_allclose(
        actual,
        np.array([4.2237776728871276e307]),
        rtol=2e-15,
        atol=0.0,
    )


@pytest.mark.parametrize(
    ("y", "mu", "p", "expected", "rtol"),
    [
        (
            1e-20,
            np.nextafter(0.0, 1.0),
            1.0005,
            1.708589252424202e-17,
            2e-14,
        ),
        (
            1e308,
            np.finfo(np.float64).max,
            np.nextafter(1.0, 2.0),
            4.2237776728871276e307,
            2e-15,
        ),
    ],
)
def test_unit_deviance_extremes_do_not_require_extended_precision(
    monkeypatch, y, mu, p, expected, rtol
):
    monkeypatch.setattr(tweedie_numerics, "_DEVIANCE_DTYPE", np.dtype(np.float64))

    actual = tweedie_unit_deviance(y, mu, p)

    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=0.0)


@pytest.mark.parametrize(
    ("y", "mu", "p", "expected"),
    [
        (
            1e-100,
            np.nextafter(0.0, 1.0),
            1.999999,
            4.045036705639803e223,
        ),
        (
            1e300,
            1.0,
            1.999999,
            2.000002000002e300,
        ),
    ],
)
def test_upper_power_extremes_are_float64_authoritative(monkeypatch, y, mu, p, expected):
    native = float(tweedie_unit_deviance(y, mu, p))
    monkeypatch.setattr(tweedie_numerics, "_DEVIANCE_DTYPE", np.dtype(np.float64))

    forced_float64 = float(tweedie_unit_deviance(y, mu, p))
    ulp = abs(np.spacing(expected))

    assert abs(forced_float64 - expected) <= 2.0 * ulp
    assert abs(native - expected) <= 2.0 * ulp
    assert abs(native - forced_float64) <= 3.0 * ulp


@pytest.mark.parametrize("y", [0.0, np.nextafter(0.0, 1.0)], ids=["zero", "minsubnormal"])
def test_lower_power_mean_dominant_extremes_are_float64_authoritative(monkeypatch, y):
    mu = 1e300
    p = 1.000000000001
    expected = float.fromhex("0x1.7e43c87b9ac43p+997")
    assert expected == 1.9999999986203265e300
    assert _decimal_tweedie_unit_deviance(y, mu, p) == expected

    native, forced_float64 = _native_and_forced_float64_deviance(monkeypatch, y, mu, p)

    _assert_few_ulp_agreement(native, expected, max_ulps=2)
    _assert_few_ulp_agreement(forced_float64, expected, max_ulps=2)
    _assert_few_ulp_agreement(native, forced_float64, max_ulps=3)


@pytest.mark.parametrize(
    ("y", "mu", "p"),
    [
        (np.nextafter(0.0, 1.0), 1e100, 1.000000000001),
        (1e-250, 1e200, 1.0000000001),
        (1e-100, 1e300, 1.00000001),
        (1e-200, 1e250, 1.0005),
        (1e-50, 1e150, 1.01),
    ],
)
def test_positive_mean_dominant_decimal_grid_is_float64_authoritative(monkeypatch, y, mu, p):
    expected = _decimal_tweedie_unit_deviance(y, mu, p)

    native, forced_float64 = _native_and_forced_float64_deviance(monkeypatch, y, mu, p)

    _assert_few_ulp_agreement(native, expected)
    _assert_few_ulp_agreement(forced_float64, expected)
    _assert_few_ulp_agreement(native, forced_float64)


@pytest.mark.parametrize("y", [0.0, np.nextafter(0.0, 1.0)], ids=["zero", "minsubnormal"])
@pytest.mark.parametrize("work_dtype", [np.longdouble, np.float64])
def test_unrepresentable_mean_dominant_deviance_raises(monkeypatch, work_dtype, y):
    monkeypatch.setattr(tweedie_numerics, "_DEVIANCE_DTYPE", np.dtype(work_dtype))
    mu = np.finfo(np.float64).max
    p = np.nextafter(1.0, 2.0)
    assert np.isinf(_decimal_tweedie_unit_deviance(y, mu, p))

    with pytest.raises(TweedieNumericalError, match="represented"):
        tweedie_unit_deviance(
            np.array([y]),
            np.array([mu]),
            p,
        )


def test_negative_certification_clamps_only_one_local_ulp():
    reference = np.array([np.finfo(np.float64).tiny, 1.0])
    one_ulp = np.abs(np.spacing(reference))

    np.testing.assert_array_equal(
        tweedie_numerics._clamp_ulp_negative(-one_ulp, reference),
        np.zeros_like(reference),
    )
    with pytest.raises(TweedieNumericalError, match="materially negative"):
        tweedie_numerics._clamp_ulp_negative(-2.0 * one_ulp, reference)


def test_negative_certification_uses_finite_ulp_at_float_maximum():
    reference = np.array([np.finfo(np.float64).max])
    one_ulp = reference - np.nextafter(reference, 0.0)

    np.testing.assert_array_equal(
        tweedie_numerics._clamp_ulp_negative(-one_ulp, reference),
        np.zeros_like(reference),
    )
    with pytest.raises(TweedieNumericalError, match="materially negative"):
        tweedie_numerics._clamp_ulp_negative(-2.0 * one_ulp, reference)


def test_materially_negative_close_ratio_raises(monkeypatch):
    monkeypatch.setattr(
        tweedie_numerics,
        "_close_deviance_ratio",
        lambda log_ratio, q: np.full_like(log_ratio, -1e-15),
    )
    mu = np.array([1e-300])
    y = mu * (1.0 + 1e-4)

    with pytest.raises(TweedieNumericalError, match="materially negative"):
        tweedie_unit_deviance(y, mu, 1.5)


def test_close_ratio_branch_transition_matches_closed_form_continuously(monkeypatch):
    close_calls = []
    ordinary_calls = []
    original_close = tweedie_numerics._close_deviance_ratio
    original_ordinary = tweedie_numerics._ordinary_log_deviance_ratio

    def recording_close(log_ratio, q):
        close_calls.extend(np.asarray(log_ratio).tolist())
        return original_close(log_ratio, q)

    def recording_ordinary(log_ratio, r, q):
        ordinary_calls.extend(np.asarray(log_ratio).tolist())
        return original_ordinary(log_ratio, r, q)

    monkeypatch.setattr(tweedie_numerics, "_close_deviance_ratio", recording_close)
    monkeypatch.setattr(tweedie_numerics, "_ordinary_log_deviance_ratio", recording_ordinary)
    y = np.array(
        [
            float.fromhex("0x1.0041919b7ee33p+0"),
            float.fromhex("0x1.0041919b7ee34p+0"),
            float.fromhex("0x1.ff7cfe56f1a9ep-1"),
            float.fromhex("0x1.ff7cfe56f1a9dp-1"),
        ]
    )
    delta = y - 1.0
    root_difference = delta / (np.sqrt(y) + 1.0)
    expected = 4.0 * root_difference**2

    actual = tweedie_unit_deviance(y, np.ones_like(y), 1.5)

    assert len(close_calls) == 2
    assert len(ordinary_calls) == 2
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=0.0)
    for inside, outside in ((0, 1), (2, 3)):
        np.testing.assert_allclose(
            actual[outside] - actual[inside],
            expected[outside] - expected[inside],
            rtol=0.0,
            atol=16.0 * np.spacing(max(expected[inside], expected[outside])),
        )


def test_lower_power_branch_transition_matches_high_precision_continuously(monkeypatch):
    lower_calls = []
    ordinary_calls = []
    original_lower = tweedie_numerics._lower_power_log_deviance_ratio
    original_ordinary = tweedie_numerics._ordinary_log_deviance_ratio

    def recording_lower(log_ratio, r, q):
        lower_calls.extend(np.asarray(log_ratio).tolist())
        return original_lower(log_ratio, r, q)

    def recording_ordinary(log_ratio, r, q):
        ordinary_calls.extend(np.asarray(log_ratio).tolist())
        return original_ordinary(log_ratio, r, q)

    monkeypatch.setattr(tweedie_numerics, "_lower_power_log_deviance_ratio", recording_lower)
    monkeypatch.setattr(tweedie_numerics, "_ordinary_log_deviance_ratio", recording_ordinary)
    powers = np.array(
        [
            float.fromhex("0x1.004189374bc69p+0"),
            float.fromhex("0x1.004189374bc6bp+0"),
        ]
    )
    expected = np.array(
        [
            0.77240043862854091,
            0.7724004386285408,
        ]
    )

    actual = np.array([tweedie_unit_deviance(2.0, 1.0, p) for p in powers])

    assert len(lower_calls) == 1
    assert len(ordinary_calls) == 1
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=0.0)
    assert abs(actual[1] - actual[0]) <= 8.0 * np.spacing(expected[0])


def test_series_factor_transition_matches_closed_form_continuously():
    y = np.array(
        [
            float.fromhex("0x1.5bf0a8b145768p+1"),
            float.fromhex("0x1.5bf0a8b14576bp+1"),
            float.fromhex("0x1.78b56362cef39p-2"),
            float.fromhex("0x1.78b56362cef37p-2"),
        ]
    )
    delta = y - 1.0
    root_difference = delta / (np.sqrt(y) + 1.0)
    expected = 4.0 * root_difference**2

    actual = tweedie_unit_deviance(y, np.ones_like(y), 1.5)

    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=0.0)
    for inside, outside in ((0, 1), (2, 3)):
        np.testing.assert_allclose(
            actual[outside] - actual[inside],
            expected[outside] - expected[inside],
            rtol=0.0,
            atol=16.0 * np.spacing(max(expected[inside], expected[outside])),
        )


def test_compound_parameters_recover_weighted_tweedie_moments():
    parameters = compound_poisson_gamma_parameters(
        np.array([4.0]), 0.5, 1.5, weights=np.array([2.0])
    )
    expected_variance = 0.5 * 4.0**1.5 / 2.0
    np.testing.assert_allclose(
        parameters.rate * parameters.shape * parameters.scale,
        [4.0],
    )
    np.testing.assert_allclose(
        parameters.rate * parameters.shape * (1.0 + parameters.shape) * parameters.scale**2,
        [expected_variance],
    )


def test_compound_parameters_accept_per_observation_dispersion():
    mu = np.array([0.25, 4.0, 16.0])
    phi = np.array([0.5, 1.0, 2.0])
    power = 1.4
    q = 2.0 - power
    r = power - 1.0

    parameters = compound_poisson_gamma_parameters(mu, phi, power)

    np.testing.assert_array_equal(parameters.rate, mu**q / (phi * q))
    assert parameters.shape == q / r
    np.testing.assert_array_equal(parameters.scale, phi * r * mu**r)


def test_weighted_compound_parameters_preserve_representable_extremes():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        parameters = compound_poisson_gamma_parameters(
            np.array([0.01]),
            np.finfo(np.float64).max,
            1.5,
            weights=np.array([0.5]),
        )

    np.testing.assert_array_equal(
        parameters.rate,
        np.array([float.fromhex("0x0.0666666666666p-1022")]),
    )
    np.testing.assert_array_equal(
        parameters.scale,
        np.array([float.fromhex("0x1.9999999999999p+1020")]),
    )


def test_vector_dispersion_and_weights_preserve_representable_extreme_grid():
    mu = np.array([0.01, 1.0])
    phi = np.array([np.finfo(np.float64).max, np.finfo(np.float64).tiny])
    weights = np.array([0.5, np.nextafter(2.0, 0.0)])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        parameters = compound_poisson_gamma_parameters(mu, phi, 1.5, weights=weights)

    np.testing.assert_array_equal(
        parameters.rate,
        np.array(
            [
                float.fromhex("0x0.0666666666666p-1022"),
                float.fromhex("0x1.fffffffffffffp+1023"),
            ]
        ),
    )
    np.testing.assert_array_equal(
        parameters.scale,
        np.array(
            [
                float.fromhex("0x1.9999999999999p+1020"),
                float.fromhex("0x0.4000000000000p-1022"),
            ]
        ),
    )


@pytest.mark.parametrize(
    ("mu", "phi", "message"),
    [
        (1.0, np.nextafter(0.0, 1.0), "Poisson rate"),
        (np.finfo(np.float64).max, np.finfo(np.float64).max, "Gamma scale"),
    ],
)
def test_compound_parameters_reject_truly_unrepresentable_outputs(mu, phi, message):
    with pytest.raises(TweedieNumericalError, match=message):
        compound_poisson_gamma_parameters(np.array([mu]), phi, 1.5)
