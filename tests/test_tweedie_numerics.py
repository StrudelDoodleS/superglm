import math
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


def _decimal_pearson_dispersion(
    y: float, mu: float, p: float, weight: float, df_resid: float
) -> float:
    """Return an independent high-precision log-domain Pearson reference."""
    with localcontext() as context:
        context.prec = 250
        residual = abs(Decimal.from_float(y) - Decimal.from_float(mu))
        log_phi = (
            Decimal.from_float(weight).ln()
            + Decimal(2) * residual.ln()
            - Decimal.from_float(p) * Decimal.from_float(mu).ln()
            - Decimal.from_float(df_resid).ln()
        )
        result = log_phi.exp()
    return float(result)


def _decimal_pearson_sum(
    y: np.ndarray,
    mu: np.ndarray,
    p: float,
    weights: np.ndarray,
    df_resid: float,
    *,
    precision: int = 1000,
) -> Decimal:
    """Return an independent arbitrary-precision Pearson sum."""
    with localcontext() as context:
        context.prec = precision
        decimal_power = Decimal.from_float(p)
        numerator = Decimal(0)
        for y_value, mu_value, weight_value in zip(y, mu, weights, strict=True):
            decimal_y = Decimal.from_float(float(y_value))
            decimal_mu = Decimal.from_float(float(mu_value))
            decimal_weight = Decimal.from_float(float(weight_value))
            numerator += (
                decimal_weight * abs(decimal_y - decimal_mu) ** 2 / decimal_mu**decimal_power
            )
        return numerator / Decimal.from_float(df_resid)


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


@pytest.mark.parametrize("df_resid", [0, -1, np.nan, np.inf, True, "2", [2]])
def test_pearson_dispersion_rejects_invalid_residual_df(df_resid):
    from superglm.profiling.tweedie import estimate_phi

    exception = TypeError if isinstance(df_resid, bool | str | list) else ValueError
    with pytest.raises(exception, match="df_resid"):
        estimate_phi(np.array([1.0]), np.array([1.0]), 1.5, df_resid=df_resid)


def test_pearson_dispersion_preserves_valid_tiny_means():
    from superglm.profiling.tweedie import estimate_phi

    y = np.array([1e-12, 2e-12])
    mu = np.array([1e-20, 2e-20])
    expected = np.sum((y - mu) ** 2 / mu**1.5) / 2.0
    assert estimate_phi(y, mu, 1.5, df_resid=2.0) == pytest.approx(expected, rel=2e-13)


def test_estimate_phi_rejects_scalar_arrays_with_shape_error():
    from superglm.profiling.tweedie import estimate_phi

    with pytest.raises(ValueError, match="one-dimensional"):
        estimate_phi(np.array(1.0), np.array(1.0), 1.5)


@pytest.mark.parametrize("name", ["y", "mu", "weights"])
@pytest.mark.parametrize(
    "invalid",
    [
        pytest.param(np.array(["1.0"]), id="string"),
        pytest.param(np.array([True]), id="boolean"),
        pytest.param(np.array([1.0], dtype=object), id="object"),
        pytest.param(np.array([1.0 + 0.0j]), id="complex"),
    ],
)
def test_estimate_phi_rejects_non_real_numeric_public_arrays_without_warnings(name, invalid):
    from superglm.profiling.tweedie import estimate_phi

    arguments = {
        "y": np.array([2.0]),
        "mu": np.array([1.0]),
        "weights": np.array([1.0]),
    }
    arguments[name] = invalid

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(TypeError, match=name):
            estimate_phi(
                arguments["y"],
                arguments["mu"],
                1.5,
                weights=arguments["weights"],
                df_resid=1.0,
            )


@pytest.mark.parametrize(
    "invalid_p",
    [
        pytest.param("1.5", id="string"),
        pytest.param(True, id="boolean"),
        pytest.param(np.bool_(True), id="numpy-boolean"),
        pytest.param(1.5 + 0.0j, id="complex"),
        pytest.param(object(), id="object-scalar"),
        pytest.param(np.array("1.5", dtype=object), id="object-string-0d"),
        pytest.param(np.array(1.5, dtype=object), id="object-numeric-0d"),
        pytest.param(np.array(1.5), id="numeric-0d"),
        pytest.param([1.5], id="list"),
        pytest.param(np.array([1.5]), id="one-dimensional"),
    ],
)
def test_estimate_phi_rejects_non_scalar_real_public_power_without_warnings(invalid_p):
    from superglm.profiling.tweedie import estimate_phi

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(TypeError, match="p"):
            estimate_phi(
                np.array([2.0]),
                np.array([1.0]),
                invalid_p,
                df_resid=1.0,
            )


@pytest.mark.parametrize("p", [1.5, np.float32(1.5), np.float64(1.5)])
def test_estimate_phi_forwards_valid_power_as_builtin_float(monkeypatch, p):
    from superglm.profiling import tweedie as tweedie_profile

    captured = {}

    def record_pearson(y, mu, power, weights, denominator):
        captured["power"] = power
        return 0.25

    monkeypatch.setattr(tweedie_profile, "pearson_dispersion", record_pearson)

    result = tweedie_profile.estimate_phi(
        np.array([2.0]),
        np.array([1.0]),
        p,
        df_resid=1.0,
    )

    assert result == 0.25
    assert type(captured["power"]) is float
    assert captured["power"] == 1.5


def test_prepared_pearson_dispersion_preserves_valid_tiny_means():
    from types import SimpleNamespace

    from superglm.profiling.tweedie import _pearson_phi_from_prepared

    y = np.array([1e-12, 2e-12])
    mu = np.array([1e-20, 2e-20])
    prepared = SimpleNamespace(y=y, mu=mu, p=1.5, weights=np.ones_like(y))
    expected = np.sum((y - mu) ** 2 / mu**1.5) / 2.0

    assert _pearson_phi_from_prepared(prepared, 2.0) == pytest.approx(expected, rel=2e-13)


def test_pearson_dispersion_returns_lower_bound_for_exact_zero_numerator():
    from superglm._tweedie_numerics import PHI_LOWER_BOUND, pearson_dispersion

    result = pearson_dispersion(
        np.array([1.0, 2.0]),
        np.array([1.0, 2.0]),
        1.5,
        None,
        2.0,
    )

    assert result == PHI_LOWER_BOUND


@pytest.mark.parametrize("response_kind", ["equal", "nextafter"])
def test_profile_pearson_preserves_shared_lower_bound_without_uncertifiable_density(
    monkeypatch, response_kind
):
    import superglm._tweedie_density as density_module
    from superglm._tweedie_numerics import PHI_LOWER_BOUND
    from superglm.profiling.tweedie import _profile_phi_detailed

    def unexpected_density(*args, **kwargs):
        raise AssertionError("degenerate lower-bound density must not be evaluated")

    monkeypatch.setattr(density_module, "evaluate_tweedie_density", unexpected_density)

    values = np.array([1.0, 2.0])
    response = (
        values.copy()
        if response_kind == "equal"
        else np.nextafter(values, np.full_like(values, np.inf))
    )
    result = _profile_phi_detailed(
        response,
        values,
        1.5,
        phi_method="pearson",
    )

    assert result.phi == PHI_LOWER_BOUND
    assert result.optimizer == "pearson"
    assert np.isinf(result.nll)
    assert not result.objective_finite
    assert not result.converged
    assert result.n_evaluations == 0


def test_mle_phi_seeds_do_not_eagerly_probe_floored_boundary_values():
    from superglm._tweedie_numerics import PHI_LOWER_BOUND
    from superglm.profiling.tweedie import (
        _LOG_PHI_LOWER_BOUND,
        _LOG_PHI_UPPER_BOUND,
        _pearson_phi_from_prepared,
        _phi_profile_seeds,
        _prepare_tweedie_density,
    )

    mu = np.array([1.0, 2.0])
    y = np.nextafter(mu, np.full_like(mu, np.inf))
    prepared = _prepare_tweedie_density(y, mu, 1.5)
    pearson_phi = _pearson_phi_from_prepared(prepared, 2.0)
    seeds = _phi_profile_seeds(
        prepared,
        2.0,
        pearson_phi,
        PHI_LOWER_BOUND,
    )

    assert pearson_phi == PHI_LOWER_BOUND
    assert seeds
    assert all(_LOG_PHI_LOWER_BOUND < u < _LOG_PHI_UPPER_BOUND for u, _ in seeds)


def test_pearson_dispersion_applies_prior_weights_once():
    from superglm._tweedie_numerics import pearson_dispersion

    result = pearson_dispersion(
        np.array([3.0]),
        np.array([1.0]),
        1.5,
        np.array([4.0]),
        2.0,
    )

    assert result == pytest.approx(8.0)


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(np.array([2.0 + 7.0j]), id="complex"),
        pytest.param(np.array(["2.0"]), id="string"),
    ],
)
@pytest.mark.parametrize("name", ["y", "mu", "weights"])
def test_pearson_dispersion_rejects_non_real_arrays_without_cast_warning(name, bad):
    from superglm._tweedie_numerics import pearson_dispersion

    arguments = {
        "y": np.array([2.0]),
        "mu": np.array([1.0]),
        "p": 1.5,
        "weights": np.array([1.0]),
    }
    arguments[name] = bad

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(TypeError, match=name):
            pearson_dispersion(**arguments, df_resid=1.0)


@pytest.mark.parametrize(
    ("y", "mu", "weights"),
    [
        pytest.param(np.array(1.0), np.array(1.0), None, id="scalar"),
        pytest.param(np.ones((2, 1)), np.ones((2, 1)), None, id="two-dimensional"),
        pytest.param(np.ones(2), np.ones(1), None, id="mismatched-mu"),
        pytest.param(np.ones(2), np.ones(2), np.ones(1), id="mismatched-weights"),
    ],
)
def test_pearson_dispersion_rejects_non_matching_one_dimensional_arrays(y, mu, weights):
    from superglm._tweedie_numerics import pearson_dispersion

    with pytest.raises(ValueError, match="matching one-dimensional"):
        pearson_dispersion(y, mu, 1.5, weights, 1.0)


@pytest.mark.parametrize("invalid_y", [-1.0, np.nan, np.inf, -np.inf])
def test_pearson_dispersion_rejects_invalid_responses(invalid_y):
    from superglm._tweedie_numerics import pearson_dispersion

    with pytest.raises(ValueError, match="y must be finite and nonnegative"):
        pearson_dispersion(np.array([invalid_y]), np.array([1.0]), 1.5, None, 1.0)


@pytest.mark.parametrize("invalid_mu", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_pearson_dispersion_rejects_invalid_means(invalid_mu):
    from superglm._tweedie_numerics import pearson_dispersion

    with pytest.raises(ValueError, match="mu must be finite and strictly positive"):
        pearson_dispersion(np.array([1.0]), np.array([invalid_mu]), 1.5, None, 1.0)


@pytest.mark.parametrize("invalid_weight", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_pearson_dispersion_rejects_invalid_weights(invalid_weight):
    from superglm._tweedie_numerics import pearson_dispersion

    with pytest.raises(ValueError, match="weights must be finite and strictly positive"):
        pearson_dispersion(
            np.array([2.0]),
            np.array([1.0]),
            1.5,
            np.array([invalid_weight]),
            1.0,
        )


def test_pearson_dispersion_preserves_representable_log_domain_extreme():
    from superglm._tweedie_numerics import pearson_dispersion

    y = 1e-100
    mu = 1e-300
    p = 1.5
    weight = 0.75
    df_resid = 3.0
    expected = _decimal_pearson_dispersion(y, mu, p, weight, df_resid)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = pearson_dispersion(
            np.array([y]),
            np.array([mu]),
            p,
            np.array([weight]),
            df_resid,
        )

    assert result == pytest.approx(expected, rel=3e-13)


def test_pearson_dispersion_rejects_unrepresentable_result_with_typed_error():
    from superglm._tweedie_numerics import pearson_dispersion

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(TweedieNumericalError, match="Pearson dispersion exceeds"):
            pearson_dispersion(
                np.array([1.0]),
                np.array([1e-300]),
                1.5,
                None,
                1.0,
            )


def test_pearson_dispersion_returns_float_maximum_at_representable_boundary():
    from superglm._tweedie_numerics import pearson_dispersion

    float_max = np.finfo(np.float64).max
    result = pearson_dispersion(
        np.array([2.0]),
        np.array([1.0]),
        1.5,
        np.array([float_max]),
        1.0,
    )

    assert result == float_max


def test_pearson_dispersion_returns_exact_nonunit_power_boundary():
    from superglm._tweedie_numerics import pearson_dispersion

    float_max = np.finfo(np.float64).max
    y = np.array([6.0])
    mu = np.array([4.0])
    weights = np.array([float_max])
    reference = _decimal_pearson_sum(y, mu, 1.5, weights, 0.5)
    assert reference == Decimal.from_float(float(float_max))

    result = pearson_dispersion(y, mu, 1.5, weights, 0.5)

    assert result == float_max


def test_exact_nonunit_power_boundary_path_does_not_mask_overflow():
    from superglm._tweedie_numerics import pearson_dispersion

    float_max = np.finfo(np.float64).max
    y = np.array([np.nextafter(6.0, np.inf)])
    mu = np.array([4.0])
    weights = np.array([float_max])
    reference = _decimal_pearson_sum(y, mu, 1.5, weights, 0.5)
    assert reference > Decimal.from_float(float(float_max))

    with pytest.raises(TweedieNumericalError, match="Pearson dispersion exceeds"):
        pearson_dispersion(y, mu, 1.5, weights, 0.5)


def test_pearson_dispersion_rejects_result_just_above_float_maximum():
    from superglm._tweedie_numerics import pearson_dispersion

    with pytest.raises(TweedieNumericalError, match="Pearson dispersion exceeds"):
        pearson_dispersion(
            np.array([np.nextafter(2.0, np.inf)]),
            np.array([1.0]),
            1.5,
            np.array([np.finfo(np.float64).max]),
            1.0,
        )


def test_pearson_dispersion_detects_multiterm_overflow_beyond_fixed_decimal_precision():
    from superglm._tweedie_numerics import pearson_dispersion

    y = np.array([2.0, 2.0])
    mu = np.array([1.0, 1.0])
    weights = np.array(
        [
            np.finfo(np.float64).max,
            np.nextafter(0.0, 1.0),
        ]
    )
    reference = _decimal_pearson_sum(y, mu, 1.5, weights, 1.0)
    assert reference > Decimal.from_float(float(np.finfo(np.float64).max))

    with pytest.raises(TweedieNumericalError, match="Pearson dispersion exceeds"):
        pearson_dispersion(y, mu, 1.5, weights, 1.0)


def test_pearson_dispersion_accepts_multiterm_result_below_maximum_across_full_span():
    from superglm._tweedie_numerics import pearson_dispersion

    y = np.array([2.0, 2.0])
    mu = np.array([1.0, 1.0])
    weights = np.array(
        [
            np.nextafter(np.finfo(np.float64).max, 0.0),
            np.nextafter(0.0, 1.0),
        ]
    )
    reference = _decimal_pearson_sum(y, mu, 1.5, weights, 1.0)
    assert reference < Decimal.from_float(float(np.finfo(np.float64).max))

    result = pearson_dispersion(y, mu, 1.5, weights, 1.0)

    assert result == float(reference)


@pytest.mark.parametrize(
    ("n_terms", "y_value", "mu_value", "p", "df_resid"),
    [
        (
            3,
            5.19297880292482e286,
            1.0416594552757042e287,
            1.2004952851200859,
            2.236851632394152e229,
        ),
        (
            4,
            5.351651201995712e277,
            3.3922785646915843e279,
            1.499413664453422,
            3.2914884105058744e140,
        ),
    ],
)
def test_pearson_dispatch_certifies_multiterm_results_that_round_to_float_maximum(
    n_terms, y_value, mu_value, p, df_resid
):
    from superglm._tweedie_numerics import pearson_dispersion

    y = np.full(n_terms, y_value)
    mu = np.full(n_terms, mu_value)
    weights = np.full(n_terms, np.finfo(np.float64).max)
    reference = _decimal_pearson_sum(y, mu, p, weights, df_resid, precision=5000)
    float_max = Decimal.from_float(float(np.finfo(np.float64).max))
    assert reference < float_max
    assert float(reference) == float(float_max)

    result = pearson_dispersion(y, mu, p, weights, df_resid)

    assert result == float(float_max)


def test_pearson_dispatch_certifies_multiterm_overflow_outside_four_log_ulps():
    from superglm._tweedie_numerics import pearson_dispersion

    n_terms = 5
    y = np.full(n_terms, 5.351651201995712e277)
    mu = np.full(n_terms, 3.3922785646915843e279)
    p = 1.499413664453422
    weights = np.full(n_terms, np.finfo(np.float64).max)
    df_resid = 4.114360513132343e140
    reference = _decimal_pearson_sum(y, mu, p, weights, df_resid, precision=5000)
    assert reference > Decimal.from_float(float(np.finfo(np.float64).max))

    with pytest.raises(TweedieNumericalError, match="Pearson dispersion exceeds"):
        pearson_dispersion(y, mu, p, weights, df_resid)


def test_pearson_dispatch_routes_possible_top_binade_to_certifier(monkeypatch):
    sentinel = float(np.finfo(np.float64).max)
    calls = []

    def record_certification(*arguments):
        calls.append(arguments)
        return sentinel

    monkeypatch.setattr(tweedie_numerics, "_pearson_boundary_result", record_certification)
    n_terms = 4
    result = tweedie_numerics.pearson_dispersion(
        np.full(n_terms, 5.351651201995712e277),
        np.full(n_terms, 3.3922785646915843e279),
        1.499413664453422,
        np.full(n_terms, np.finfo(np.float64).max),
        3.2914884105058744e140,
    )

    assert result == sentinel
    assert len(calls) == 1


def test_pearson_boundary_aggregates_all_equal_rows_before_decimal_evaluation(monkeypatch):
    calls = []
    float_max = float(np.finfo(np.float64).max)
    expected = Decimal.from_float(float_max) / Decimal(16)

    def record_exact_candidate(*arguments, **keywords):
        calls.append((len(arguments[0]), keywords.get("multiplicity", 1)))
        return expected, True

    monkeypatch.setattr(
        tweedie_numerics,
        "_exact_dyadic_power_pearson_candidate",
        record_exact_candidate,
    )
    n_terms = 100_000

    result = tweedie_numerics.pearson_dispersion(
        np.full(n_terms, 2.0),
        np.ones(n_terms),
        1.5,
        np.full(n_terms, float_max),
        float(16 * n_terms),
    )

    assert result == float(expected)
    assert calls == [(1, n_terms)]


def test_pearson_dispatch_bypasses_certifier_for_large_ordinary_problem(monkeypatch):
    def unexpected_certification(*_arguments):
        raise AssertionError("ordinary Pearson result must stay on the float64 path")

    monkeypatch.setattr(
        tweedie_numerics,
        "_pearson_boundary_result",
        unexpected_certification,
    )
    n_terms = 10_000
    result = tweedie_numerics.pearson_dispersion(
        np.full(n_terms, 2.0),
        np.ones(n_terms),
        1.5,
        np.ones(n_terms),
        float(n_terms),
    )

    assert result == pytest.approx(1.0)


@pytest.mark.parametrize("weighted", [False, True], ids=["unit-weights", "explicit-weights"])
def test_pearson_large_ordinary_path_bypasses_log_and_exponent_routers(monkeypatch, weighted):
    def unexpected_slow_path(*_arguments):
        raise AssertionError("normal ordinary inputs must stay on the direct path")

    monkeypatch.setattr(
        tweedie_numerics,
        "_pearson_float64_range_route",
        unexpected_slow_path,
    )
    monkeypatch.setattr(
        tweedie_numerics,
        "_pearson_boundary_result",
        unexpected_slow_path,
    )
    monkeypatch.setattr(tweedie_numerics, "logsumexp", unexpected_slow_path)
    n_terms = 1_000_000
    weights = np.ones(n_terms) if weighted else None

    result = tweedie_numerics.pearson_dispersion(
        np.full(n_terms, 2.0),
        np.ones(n_terms),
        1.5,
        weights,
        float(n_terms),
    )

    assert result == 1.0


@pytest.mark.parametrize("weighted", [False, True], ids=["unit-weights", "explicit-weights"])
def test_pearson_dense_direct_path_avoids_repeated_nonzero_reductions(monkeypatch, weighted):
    original_count_nonzero = np.count_nonzero
    reduction_sizes = []

    def record_count_nonzero(values):
        reduction_sizes.append(np.asarray(values).size)
        return original_count_nonzero(values)

    monkeypatch.setattr(tweedie_numerics.np, "count_nonzero", record_count_nonzero)
    n_terms = 10_000
    weights = np.ones(n_terms) if weighted else None

    result = tweedie_numerics.pearson_dispersion(
        np.full(n_terms, 2.0),
        np.ones(n_terms),
        1.5,
        weights,
        float(n_terms),
    )

    assert result == 1.0
    assert reduction_sizes == [n_terms]


def test_pearson_direct_path_accepts_exact_subnormal_intermediate(monkeypatch):
    """An exact subnormal square loses no information and need not use the log fallback."""

    def unexpected_slow_path(*_arguments):
        raise AssertionError("an exact subnormal intermediate must stay on the direct path")

    monkeypatch.setattr(
        tweedie_numerics,
        "_pearson_float64_range_route",
        unexpected_slow_path,
    )
    monkeypatch.setattr(
        tweedie_numerics,
        "_pearson_boundary_result",
        unexpected_slow_path,
    )
    monkeypatch.setattr(tweedie_numerics, "logsumexp", unexpected_slow_path)
    mu_value = math.ldexp(1.0, -537)

    result = tweedie_numerics.pearson_dispersion(
        np.array([2.0 * mu_value]),
        np.array([mu_value]),
        1.5,
        None,
        1.0,
    )

    assert result == tweedie_numerics.PHI_LOWER_BOUND


def test_pearson_scalar_upper_exponent_rounds_each_operation_outward():
    power = 1.3789432830800634
    weight_exponent = -735
    residual_exponent = 859
    mean_exponent = -17
    term_count_exponent = 55
    denominator_exponent = 1016
    with localcontext() as context:
        context.prec = 100
        exact = (
            Decimal(weight_exponent)
            + Decimal(2 * residual_exponent)
            - Decimal.from_float(power) * Decimal(mean_exponent - 1)
            + Decimal(term_count_exponent)
            - Decimal(denominator_exponent - 1)
        )

    upper = tweedie_numerics._pearson_scalar_upper_exponent(
        weight_exponent=weight_exponent,
        residual_exponent=residual_exponent,
        mean_exponent=mean_exponent,
        power=power,
        term_count_exponent=term_count_exponent,
        denominator_exponent=denominator_exponent,
    )

    assert Decimal.from_float(upper) >= exact


def test_pearson_large_ordinary_path_stays_within_broad_direct_timing_guard():
    import time

    n_terms = 1_000_000
    y = np.full(n_terms, 2.0)
    mu = np.ones(n_terms)
    weights = np.ones(n_terms)
    denominator = float(n_terms)
    scratch = np.empty_like(y)
    variance = np.empty_like(y)

    def direct_reference():
        np.subtract(y, mu, out=scratch)
        np.square(scratch, out=scratch)
        np.multiply(scratch, weights, out=scratch)
        np.power(mu, 1.5, out=variance)
        np.divide(scratch, variance, out=scratch)
        return float(np.sum(scratch) / denominator)

    def elapsed(function):
        start = time.perf_counter()
        result = function()
        return time.perf_counter() - start, result

    direct_reference()
    tweedie_numerics.pearson_dispersion(y, mu, 1.5, weights, denominator)
    direct_samples = []
    actual_samples = []
    for _ in range(7):
        direct_samples.append(elapsed(direct_reference))
        actual_samples.append(
            elapsed(
                lambda: tweedie_numerics.pearson_dispersion(
                    y,
                    mu,
                    1.5,
                    weights,
                    denominator,
                )
            )
        )
    direct_median = float(np.median([sample[0] for sample in direct_samples]))
    actual_median = float(np.median([sample[0] for sample in actual_samples]))

    assert all(sample[1] == 1.0 for sample in direct_samples + actual_samples)
    assert actual_median <= 3.0 * direct_median + 0.02


@pytest.mark.parametrize(
    ("n_rows", "df_scale"),
    [
        *[
            pytest.param(2**exponent, 16, id=f"n={2**exponent}-df-scale=16")
            for exponent in range(1, 9)
        ],
        pytest.param(2, 32, id="n=2-df-scale=32"),
    ],
)
def test_pearson_repeated_reduction_overflow_fallback_is_exact(n_rows, df_scale):
    from superglm._tweedie_numerics import pearson_dispersion

    y = np.full(n_rows, 2.0)
    mu = np.ones(n_rows)
    weights = np.full(n_rows, np.finfo(np.float64).max)
    df_resid = float(df_scale * n_rows)
    reference = _decimal_pearson_sum(y, mu, 1.5, weights, df_resid)
    assert reference < Decimal.from_float(float(np.finfo(np.float64).max))
    expected = float(reference)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = pearson_dispersion(y, mu, 1.5, weights, df_resid)

    result_bits = int(np.float64(result).view(np.uint64))
    expected_bits = int(np.float64(expected).view(np.uint64))
    assert (result.hex(), abs(result_bits - expected_bits)) == (expected.hex(), 0)


@pytest.mark.parametrize("n_rows", [1, 2, 3])
def test_pearson_identical_subnormal_terms_return_floor_without_warning(n_rows):
    minimum_subnormal = np.nextafter(0.0, 1.0)
    tiny = np.finfo(np.float64).tiny
    mu_value = np.nextafter(tiny, np.inf)
    y_value = np.nextafter(mu_value, np.inf)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = tweedie_numerics.pearson_dispersion(
            np.full(n_rows, y_value),
            np.full(n_rows, mu_value),
            1.5,
            np.full(n_rows, minimum_subnormal),
            float(n_rows),
        )

    assert result == tweedie_numerics.PHI_LOWER_BOUND


def test_pearson_large_identical_subnormal_terms_floor_within_broad_timing_guard():
    import time

    n_rows = 100_000
    minimum_subnormal = np.nextafter(0.0, 1.0)
    tiny = np.finfo(np.float64).tiny
    mu_value = np.nextafter(tiny, np.inf)
    y_value = np.nextafter(mu_value, np.inf)
    started = time.perf_counter()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = tweedie_numerics.pearson_dispersion(
            np.full(n_rows, y_value),
            np.full(n_rows, mu_value),
            1.5,
            np.full(n_rows, minimum_subnormal),
            float(n_rows),
        )

    elapsed = time.perf_counter() - started
    assert result == tweedie_numerics.PHI_LOWER_BOUND
    assert elapsed < 1.0


def test_pearson_nonrepeated_reduction_overflow_keeps_log_fallback(monkeypatch):
    def unexpected_certification(*_arguments):
        raise AssertionError("nonrepeated ordinary fallback must stay on the log path")

    logsumexp = tweedie_numerics.logsumexp
    logsumexp_calls = []

    def record_logsumexp(log_terms):
        logsumexp_calls.append(log_terms.copy())
        return logsumexp(log_terms)

    monkeypatch.setattr(
        tweedie_numerics,
        "_pearson_boundary_result",
        unexpected_certification,
    )
    monkeypatch.setattr(tweedie_numerics, "logsumexp", record_logsumexp)
    float_max = float(np.finfo(np.float64).max)
    y = np.array([2.0, 2.0])
    mu = np.ones(2)
    weights = np.array([float_max, np.nextafter(float_max, 0.0)])
    reference = _decimal_pearson_sum(y, mu, 1.5, weights, 32.0)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = tweedie_numerics.pearson_dispersion(y, mu, 1.5, weights, 32.0)

    assert len(logsumexp_calls) == 1
    assert result == pytest.approx(float(reference), rel=3e-13)


@pytest.mark.parametrize(
    ("denominator", "force_ordinary_certificate"),
    [
        (np.nextafter(0.0, 1.0), True),
        (np.finfo(np.float64).max, False),
    ],
    ids=["division-overflow", "division-underflow"],
)
def test_direct_pearson_final_division_extremes_fall_back_without_warning(
    monkeypatch, denominator, force_ordinary_certificate
):
    if force_ordinary_certificate:
        monkeypatch.setattr(
            tweedie_numerics,
            "_pearson_scalar_range_is_ordinary",
            lambda *_arguments: True,
        )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = tweedie_numerics._direct_pearson_dispersion_if_safe(
            np.array([1.0]),
            np.array([1.0]),
            1.5,
            None,
            denominator,
            1,
        )

    assert result is None


@pytest.mark.parametrize(
    ("y", "mu", "weight"),
    [
        (1.0, 1.9e-216, np.nextafter(0.0, 1.0)),
        (1e-162, 3e-216, 1.0),
    ],
)
def test_pearson_direct_path_falls_back_for_representable_subnormal_intermediates(y, mu, weight):
    from superglm._tweedie_numerics import pearson_dispersion

    expected = _decimal_pearson_dispersion(y, mu, 1.5, weight, 1.0)

    result = pearson_dispersion(
        np.array([y]),
        np.array([mu]),
        1.5,
        np.array([weight]),
        1.0,
    )

    assert result == pytest.approx(expected, rel=3e-13)


def test_pearson_direct_path_does_not_hide_overflow_behind_subnormal_rounding():
    from superglm._tweedie_numerics import pearson_dispersion

    y = 2.7132228295948296e-162
    mu = 1.851696853758617e-216
    weight = 6.741349255733685e307
    reference = _decimal_pearson_sum(
        np.array([y]),
        np.array([mu]),
        1.5,
        np.array([weight]),
        1.0,
    )
    assert reference > Decimal.from_float(float(np.finfo(np.float64).max))

    with pytest.raises(TweedieNumericalError, match="Pearson dispersion exceeds"):
        pearson_dispersion(
            np.array([y]),
            np.array([mu]),
            1.5,
            np.array([weight]),
            1.0,
        )


def test_pearson_dispatch_rejects_large_proven_overflow_without_decimal(monkeypatch):
    def unexpected_certification(*_arguments):
        raise AssertionError("exponent lower bound must reject obvious overflow")

    monkeypatch.setattr(
        tweedie_numerics,
        "_pearson_boundary_result",
        unexpected_certification,
    )
    n_terms = 10_000
    with pytest.raises(TweedieNumericalError, match="Pearson dispersion exceeds"):
        tweedie_numerics.pearson_dispersion(
            np.ones(n_terms),
            np.full(n_terms, 1e-300),
            1.5,
            np.ones(n_terms),
            1.0,
        )


def test_pearson_dispersion_uses_exact_input_difference_to_detect_boundary_overflow():
    from superglm._tweedie_numerics import pearson_dispersion

    with pytest.raises(TweedieNumericalError, match="Pearson dispersion exceeds"):
        pearson_dispersion(
            np.array([2.575240705957428e-28]),
            np.array([1.4000185730028892e-29]),
            1.5,
            np.array([np.finfo(np.float64).max]),
            1.1320938493158106e-12,
        )


def test_pearson_dispersion_uses_exact_input_difference_to_accept_boundary_result():
    from superglm._tweedie_numerics import pearson_dispersion

    y = 1.0393692355722397e-247
    mu = 1.6809581325166083e-249
    p = 1.5
    weight = float(np.finfo(np.float64).max)
    df_resid = 1.5171969527427854e-121
    expected = _decimal_pearson_dispersion(y, mu, p, weight, df_resid)

    result = pearson_dispersion(
        np.array([y]),
        np.array([mu]),
        p,
        np.array([weight]),
        df_resid,
    )

    assert result == expected


@pytest.mark.parametrize(
    ("p", "expected_deviance"),
    [
        (np.nextafter(1.0, 2.0), 1414.79283706464),
        (1.000001, 1415.2947790683888),
    ],
)
def test_explicit_saddlepoint_uses_shared_representable_extreme_deviance(p, expected_deviance):
    from superglm._tweedie_density import approximate_tweedie_logpdf

    y = np.array([1.0])
    mu = np.array([np.finfo(np.float64).tiny])
    expected = -0.5 * (np.log(2.0 * np.pi) + expected_deviance)

    actual = approximate_tweedie_logpdf(y, mu, 1.0, p).logpdf

    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=0.0)


def test_explicit_saddlepoint_fails_closed_for_unrepresentable_deviance_row():
    from superglm._tweedie_density import TweedieDensityError, approximate_tweedie_logpdf

    y = np.array([1.0, 1e308])
    mu = np.array([np.finfo(np.float64).tiny, 1e-320])
    p = 1.000001
    expected_first = -0.5 * (np.log(2.0 * np.pi) + 1415.2947790683888)

    representable = approximate_tweedie_logpdf(y[:1], mu[:1], 1.0, p).logpdf

    assert np.isfinite(representable[0])
    np.testing.assert_allclose(representable[0], expected_first, rtol=2e-15, atol=0.0)
    with pytest.raises(TweedieDensityError, match="saddlepoint arithmetic"):
        approximate_tweedie_logpdf(y[1:], mu[1:], 1.0, p)


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
