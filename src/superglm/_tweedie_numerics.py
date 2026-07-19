from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import (
    ROUND_CEILING,
    ROUND_FLOOR,
    ROUND_HALF_EVEN,
    Context,
    Decimal,
    DecimalException,
    Inexact,
    localcontext,
)
from numbers import Real
from typing import Literal, SupportsFloat

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

PHI_LOWER_BOUND = 1e-12
_DEVIANCE_DTYPE = np.dtype(np.longdouble)
_LOWER_POWER_BOUNDARY = 1e-3
_SERIES_LOG_RATIO_BOUNDARY = 1.0
_PEARSON_MIN_CERTIFICATION_PRECISION = 80
# Positive float64 Pearson terms span fewer than 3,200 decimal orders over the
# full input range. This cap leaves room for outward-rounding guard digits.
_PEARSON_MAX_CERTIFICATION_PRECISION = 4096
_PEARSON_CERTIFICATION_GUARD_DIGITS = 32
_DECIMAL_CERTIFICATION_EMAX = 999_999
_DECIMAL_CERTIFICATION_EMIN = -999_999


class TweedieNumericalError(RuntimeError):
    """Exact Tweedie arithmetic could not be represented or certified."""


@dataclass(frozen=True)
class CompoundPoissonGammaParameters:
    rate: NDArray[np.float64]
    shape: float
    scale: NDArray[np.float64]


@dataclass(frozen=True)
class _DecimalInterval:
    lower: Decimal
    upper: Decimal


def _contains_masked_array(value: object) -> bool:
    """Return whether a supported nested container contains a masked array.

    ``np.asarray`` deliberately discards ``MaskedArray`` metadata.  Checking
    only the outer object is insufficient because a list or object array can
    contain a masked array whose hidden payload would then be consumed.  Walk
    the container shapes accepted by the numerical APIs before coercion, with
    cycle protection for adversarial object containers.
    """
    pending = [value]
    seen: set[int] = set()
    while pending:
        item = pending.pop()
        if np.ma.isMaskedArray(item):
            return True
        if isinstance(item, np.ndarray):
            if not item.dtype.hasobject:
                continue
            identity = id(item)
            if identity in seen:
                continue
            seen.add(identity)
            pending.extend(item.flat)
        elif isinstance(item, list | tuple):
            identity = id(item)
            if identity in seen:
                continue
            seen.add(identity)
            pending.extend(item)
    return False


def _configure_decimal_certification_context(
    context: Context,
    precision: int,
    rounding: str = ROUND_HALF_EVEN,
) -> None:
    context.prec = precision
    context.Emax = _DECIMAL_CERTIFICATION_EMAX
    context.Emin = _DECIMAL_CERTIFICATION_EMIN
    context.clamp = 0
    context.rounding = rounding


def normalize_real_scalar(name: str, value: object) -> float:
    if (
        isinstance(value, bool | np.bool_)
        or _contains_masked_array(value)
        or not isinstance(value, Real)
    ):
        raise TypeError(f"{name} must be one finite real scalar")
    try:
        result = float(value)
    except Exception as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def normalize_tweedie_power(value: object) -> float:
    p = normalize_real_scalar("p", value)
    if not 1.0 < p < 2.0:
        raise ValueError(f"Tweedie p must be in (1, 2), got {p}")
    return p


def normalize_positive_scalar(name: str, value: object) -> float:
    result = normalize_real_scalar(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be strictly positive")
    return result


def normalize_numeric_vector(
    value: object,
    *,
    name: str,
    length: int | None = None,
    positive: bool = False,
    nonnegative: bool = False,
) -> NDArray[np.float64]:
    """Return an owning float64 copy of one strict real numeric vector."""
    if _contains_masked_array(value):
        raise TypeError(f"{name} must be a one-dimensional real numeric array without a mask")
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a one-dimensional real numeric array") from exc
    if raw.ndim != 1 or raw.dtype.kind not in "fiu":
        raise TypeError(f"{name} must be a one-dimensional real numeric array")
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            result = np.array(raw, dtype=np.float64, copy=True)
    except Exception as exc:
        raise TypeError(f"{name} must be a one-dimensional real numeric array") from exc
    if length is not None and result.size != length:
        raise ValueError(f"{name} must have length {length}, got {result.size}")
    if np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    if positive and np.any(result <= 0.0):
        raise ValueError(f"{name} must be strictly positive")
    if nonnegative and np.any(result < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return result


def normalize_positive_int(
    value: object,
    *,
    name: str,
    minimum: int = 1,
    maximum: int | None = None,
) -> int:
    """Return an integer control at or above ``minimum``, excluding booleans."""
    if isinstance(value, bool | np.bool_) or not isinstance(value, int | np.integer):
        raise TypeError(f"{name} must be an integer")
    try:
        result = int(value)
    except Exception as exc:
        raise ValueError(f"{name} must be a representable integer") from exc
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return result


def normalize_boolean(value: object, *, name: str) -> bool:
    """Return a strict Python boolean without accepting integer coercions."""
    if not isinstance(value, bool | np.bool_):
        raise TypeError(f"{name} must be a boolean")
    return bool(value)


def normalize_optional_callable(value: object, *, name: str):
    """Return ``None`` or a callable, rejecting deferred raw call failures."""
    if value is not None and not callable(value):
        raise TypeError(f"{name} must be callable or None")
    return value


def normalize_tweedie_bounds(
    value: object,
    *,
    name: str = "p_bounds",
) -> tuple[float, float]:
    """Validate two ordered Tweedie power-search bounds inside ``(1, 2)``."""
    bounds = normalize_numeric_vector(value, name=name)
    if bounds.size != 2:
        raise ValueError(f"{name} must contain exactly two bounds")
    lower, upper = float(bounds[0]), float(bounds[1])
    if not 1.0 < lower < upper < 2.0:
        raise ValueError(f"{name} must satisfy 1 < lower < upper < 2")
    return lower, upper


def normalize_tweedie_grid(
    value: object,
    *,
    name: str = "grid",
    maximum: int | None = None,
) -> NDArray[np.float64]:
    """Validate an explicit nonempty one-dimensional Tweedie power grid."""
    grid = normalize_numeric_vector(value, name=name)
    if grid.size == 0:
        raise ValueError(f"{name} must contain at least one point")
    if maximum is not None and grid.size > maximum:
        raise ValueError(f"{name} must contain at most {maximum} points")
    if np.any((grid <= 1.0) | (grid >= 2.0)):
        raise ValueError(f"{name} values must be strictly inside (1, 2)")
    return grid


def _as_real_float64_array(name: str, value: object) -> NDArray[np.float64]:
    """Convert a real numeric array without silently coercing non-real values."""
    if _contains_masked_array(value):
        raise TypeError(f"{name} must be a real numeric array without a mask")
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a real numeric array") from exc
    if raw.dtype.kind not in "iuf":
        raise TypeError(f"{name} must be a real numeric array")
    with np.errstate(over="ignore", invalid="ignore"):
        result: NDArray[np.float64] = np.asarray(raw, dtype=np.float64)
    return result


def _required_exact_decimal_precision(left: Decimal, right: Decimal) -> int:
    lowest_exponent = min(
        int(left.as_tuple().exponent),
        int(right.as_tuple().exponent),
    )
    highest_adjusted = max(left.adjusted(), right.adjusted())
    return max(1, highest_adjusted - lowest_exponent + 2)


def _exact_decimal_add(left: Decimal, right: Decimal) -> Decimal:
    precision = _required_exact_decimal_precision(left, right)
    with localcontext() as context:
        _configure_decimal_certification_context(context, precision)
        context.clear_flags()
        result = context.add(left, right)
        if context.flags[Inexact]:
            raise TweedieNumericalError("Pearson boundary arithmetic could not be exact")
    return result


def _exact_decimal_difference(left: float, right: float) -> Decimal:
    """Subtract two finite float64 values exactly in Decimal arithmetic."""
    decimal_left = Decimal.from_float(left)
    decimal_right = Decimal.from_float(right)
    precision = _required_exact_decimal_precision(decimal_left, decimal_right)
    with localcontext() as context:
        _configure_decimal_certification_context(context, precision)
        context.clear_flags()
        result = abs(context.subtract(decimal_left, decimal_right))
        if context.flags[Inexact]:
            raise TweedieNumericalError("Pearson residual could not be represented exactly")
    return result


def _exact_decimal_midpoint(left: Decimal, right: Decimal) -> Decimal:
    total = _exact_decimal_add(left, right)
    with localcontext() as context:
        _configure_decimal_certification_context(context, len(total.as_tuple().digits) + 2)
        context.clear_flags()
        result = context.divide(total, Decimal(2))
        if context.flags[Inexact]:
            raise TweedieNumericalError("Pearson rounding boundary could not be exact")
    return result


def _float64_rounding_bin_contains(interval: _DecimalInterval, candidate: float) -> bool:
    """Return whether an interval is strictly inside one binary64 rounding bin."""
    if not math.isfinite(candidate) or candidate <= 0.0:
        return False
    candidate_decimal = Decimal.from_float(candidate)
    previous = float(np.nextafter(candidate, -np.inf))
    lower_boundary = _exact_decimal_midpoint(Decimal.from_float(previous), candidate_decimal)
    if candidate == float(np.finfo(np.float64).max):
        spacing = _exact_decimal_difference(candidate, previous)
        upper_boundary = _exact_decimal_midpoint(
            candidate_decimal,
            _exact_decimal_add(candidate_decimal, spacing),
        )
    else:
        following = float(np.nextafter(candidate, np.inf))
        upper_boundary = _exact_decimal_midpoint(candidate_decimal, Decimal.from_float(following))
    return interval.lower > lower_boundary and interval.upper < upper_boundary


def _add_decimal_intervals(
    left: _DecimalInterval,
    right: _DecimalInterval,
    precision: int,
) -> _DecimalInterval:
    with localcontext() as context:
        _configure_decimal_certification_context(context, precision, ROUND_FLOOR)
        lower = context.add(left.lower, right.lower)
    with localcontext() as context:
        _configure_decimal_certification_context(context, precision, ROUND_CEILING)
        upper = context.add(left.upper, right.upper)
    return _DecimalInterval(lower, upper)


def _subtract_decimal_intervals(
    left: _DecimalInterval,
    right: _DecimalInterval,
    precision: int,
) -> _DecimalInterval:
    with localcontext() as context:
        _configure_decimal_certification_context(context, precision, ROUND_FLOOR)
        lower = context.subtract(left.lower, right.upper)
    with localcontext() as context:
        _configure_decimal_certification_context(context, precision, ROUND_CEILING)
        upper = context.subtract(left.upper, right.lower)
    return _DecimalInterval(lower, upper)


def _scale_decimal_interval(
    interval: _DecimalInterval,
    positive_scale: Decimal,
    precision: int,
) -> _DecimalInterval:
    with localcontext() as context:
        _configure_decimal_certification_context(context, precision, ROUND_FLOOR)
        lower = context.multiply(interval.lower, positive_scale)
    with localcontext() as context:
        _configure_decimal_certification_context(context, precision, ROUND_CEILING)
        upper = context.multiply(interval.upper, positive_scale)
    return _DecimalInterval(lower, upper)


def _divide_decimal_interval(
    interval: _DecimalInterval,
    positive_divisor: Decimal,
    precision: int,
) -> _DecimalInterval:
    with localcontext() as context:
        _configure_decimal_certification_context(context, precision, ROUND_FLOOR)
        lower = context.divide(interval.lower, positive_divisor)
    with localcontext() as context:
        _configure_decimal_certification_context(context, precision, ROUND_CEILING)
        upper = context.divide(interval.upper, positive_divisor)
    return _DecimalInterval(lower, upper)


def _decimal_log_interval(value: Decimal, precision: int) -> _DecimalInterval:
    """Enclose a positive Decimal logarithm using its correctly rounded value."""
    with localcontext() as context:
        _configure_decimal_certification_context(context, precision)
        context.clear_flags()
        midpoint = context.ln(value)
        if not context.flags[Inexact]:
            return _DecimalInterval(midpoint, midpoint)
        return _DecimalInterval(
            context.next_minus(midpoint),
            context.next_plus(midpoint),
        )


def _decimal_exp_interval(interval: _DecimalInterval, precision: int) -> _DecimalInterval:
    """Exponentiate an interval and round both endpoints outwards."""
    with localcontext() as context:
        _configure_decimal_certification_context(context, precision)
        context.clear_flags()
        lower_midpoint = context.exp(interval.lower)
        lower = context.next_minus(lower_midpoint) if context.flags[Inexact] else lower_midpoint
        context.clear_flags()
        upper_midpoint = context.exp(interval.upper)
        upper = context.next_plus(upper_midpoint) if context.flags[Inexact] else upper_midpoint
    return _DecimalInterval(lower, upper)


def _decimal_pearson_interval(
    y: NDArray[np.float64],
    mus: NDArray[np.float64],
    power: Decimal,
    weights: NDArray[np.float64],
    denominator: Decimal,
    precision: int,
    *,
    multiplicity: int = 1,
) -> _DecimalInterval:
    numerator = _DecimalInterval(Decimal(0), Decimal(0))
    two = Decimal(2)
    decimal_multiplicity = Decimal(multiplicity)
    for y_value, mu_value, weight_value in zip(y, mus, weights, strict=True):
        residual = _exact_decimal_difference(float(y_value), float(mu_value))
        mu = Decimal.from_float(float(mu_value))
        weight = Decimal.from_float(float(weight_value))
        log_term = _add_decimal_intervals(
            _decimal_log_interval(weight, precision),
            _scale_decimal_interval(_decimal_log_interval(residual, precision), two, precision),
            precision,
        )
        log_term = _subtract_decimal_intervals(
            log_term,
            _scale_decimal_interval(_decimal_log_interval(mu, precision), power, precision),
            precision,
        )
        term = _decimal_exp_interval(log_term, precision)
        if multiplicity != 1:
            term = _scale_decimal_interval(term, decimal_multiplicity, precision)
        numerator = _add_decimal_intervals(numerator, term, precision)
    return _divide_decimal_interval(numerator, denominator, precision)


def _exact_dyadic_power_pearson_candidate(
    y: NDArray[np.float64],
    mus: NDArray[np.float64],
    power_numerator: int,
    power_denominator: int,
    weights: NDArray[np.float64],
    denominator: Decimal,
    precision: int,
    *,
    multiplicity: int = 1,
) -> tuple[Decimal, bool]:
    """Return an exact candidate when the dyadic powers and arithmetic terminate."""
    root_count = power_denominator.bit_length() - 1
    if power_denominator != 1 << root_count:
        return Decimal(0), False
    with localcontext() as context:
        _configure_decimal_certification_context(context, precision)
        context.clear_flags()
        numerator = Decimal(0)
        decimal_multiplicity = Decimal(multiplicity)
        for y_value, mu_value, weight_value in zip(y, mus, weights, strict=True):
            residual = _exact_decimal_difference(float(y_value), float(mu_value))
            mu = Decimal.from_float(float(mu_value))
            weight = Decimal.from_float(float(weight_value))
            mu_root = mu
            for _ in range(root_count):
                mu_root = context.sqrt(mu_root)
                if context.flags[Inexact]:
                    return Decimal(0), False
            mu_power = context.power(mu_root, Decimal(power_numerator))
            if context.flags[Inexact]:
                return Decimal(0), False
            squared_residual = context.multiply(residual, residual)
            weighted_residual = context.multiply(weight, squared_residual)
            term = context.divide(weighted_residual, mu_power)
            if multiplicity != 1:
                term = context.multiply(term, decimal_multiplicity)
            numerator = context.add(numerator, term)
        result = context.divide(numerator, denominator)
        return result, not context.flags[Inexact]


def _pearson_float64_range_route(
    residuals: NDArray[np.float64],
    mus: NDArray[np.float64],
    power: float,
    weights: NDArray[np.float64],
    denominator: float,
) -> Literal["ordinary", "certify", "overflow"]:
    """Route an exact Pearson result using conservative binary exponent bounds.

    This is a conservative exponent-only router, independent of the approximate
    logarithms used by the ordinary path. For ``x = m * 2**e`` returned by
    ``frexp``, ``2**(e - 1) <= x < 2**e``. The next float above each rounded
    residual strictly bounds the exact float-input subtraction. Combining the
    weight and residual upper bounds with the mean and denominator lower bounds,
    then multiplying the largest term bound by the next power of two above the
    term count, gives a strict upper bound for the exact dispersion.

    The matching lower bound uses one rounded residual binade of slack, the
    weight's lower binade edge, the mean's upper edge, and the denominator's
    upper edge. It can reject an individual term proved above ``2**1024``
    without invoking expensive Decimal transcendental arithmetic.

    ``"ordinary"`` proves that the exact result is below ``2**1023``. Every
    result that could occupy the top binary64 binade is sent to the certified
    Decimal path, unless ``"overflow"`` already proves it exceeds float64.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        residual_upper = np.nextafter(residuals, np.inf)
    finite_residual_upper = np.isfinite(residual_upper)
    residual_exponents = np.full(
        residuals.shape,
        np.finfo(np.float64).maxexp,
        dtype=np.int64,
    )
    if np.any(finite_residual_upper):
        _, finite_exponents = np.frexp(residual_upper[finite_residual_upper])
        residual_exponents[finite_residual_upper] = finite_exponents

    _, weight_exponents = np.frexp(weights)
    _, mean_exponents = np.frexp(mus)
    mean_power_lower_exponents = np.nextafter(
        power * (mean_exponents.astype(np.float64) - 1.0),
        -np.inf,
    )
    term_upper_exponents = np.nextafter(
        weight_exponents.astype(np.float64)
        + 2.0 * residual_exponents.astype(np.float64)
        - mean_power_lower_exponents,
        np.inf,
    )

    term_count_exponent = (len(residuals) - 1).bit_length()
    denominator_exponent = math.frexp(denominator)[1]
    dispersion_upper_exponent = math.nextafter(
        float(np.max(term_upper_exponents)) + term_count_exponent - (denominator_exponent - 1),
        math.inf,
    )

    _, rounded_residual_exponents = np.frexp(residuals)
    mean_power_upper_exponents = np.nextafter(
        power * mean_exponents.astype(np.float64),
        np.inf,
    )
    term_lower_exponents = np.nextafter(
        weight_exponents.astype(np.float64)
        - 1.0
        + 2.0 * (rounded_residual_exponents.astype(np.float64) - 2.0)
        - mean_power_upper_exponents,
        -np.inf,
    )
    dispersion_lower_exponent = math.nextafter(
        float(np.max(term_lower_exponents)) - denominator_exponent,
        -math.inf,
    )
    float64_limit_exponent = np.finfo(np.float64).maxexp
    if dispersion_lower_exponent >= float64_limit_exponent:
        return "overflow"
    if dispersion_upper_exponent > float64_limit_exponent - 1:
        return "certify"
    return "ordinary"


def _pearson_rows_are_identical(
    y: NDArray[np.float64],
    mu: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> bool:
    """Return whether two or more Pearson rows have identical inputs."""
    return bool(
        len(y) > 1 and np.all(y == y[0]) and np.all(mu == mu[0]) and np.all(weights == weights[0])
    )


def _pearson_boundary_result(
    y: NDArray[np.float64],
    mu: NDArray[np.float64],
    power: float,
    weights: NDArray[np.float64],
    denominator: float,
    log_terms: NDArray[np.float64],
) -> float:
    """Certify a Pearson result with exact or interval Decimal arithmetic."""
    multiplicity = 1
    if _pearson_rows_are_identical(y, mu, weights):
        multiplicity = len(y)
        y = y[:1]
        mu = mu[:1]
        weights = weights[:1]
        log_terms = log_terms[:1]

    decimal_power = Decimal.from_float(power)
    power_numerator, power_denominator = power.as_integer_ratio()
    decimal_denominator = Decimal.from_float(denominator)
    decimal_floor = Decimal.from_float(PHI_LOWER_BOUND)
    float_max = Decimal.from_float(float(np.finfo(np.float64).max))

    term_span_digits = math.ceil(float(np.ptp(log_terms)) / math.log(10.0))
    term_count_digits = math.ceil(math.log10(max(1, len(log_terms) * multiplicity)))
    precision = min(
        _PEARSON_MAX_CERTIFICATION_PRECISION,
        max(
            _PEARSON_MIN_CERTIFICATION_PRECISION,
            term_span_digits + term_count_digits + _PEARSON_CERTIFICATION_GUARD_DIGITS,
        ),
    )
    try:
        while True:
            exact_result, is_exact = _exact_dyadic_power_pearson_candidate(
                y,
                mu,
                power_numerator,
                power_denominator,
                weights,
                decimal_denominator,
                precision,
                multiplicity=multiplicity,
            )
            if is_exact:
                if Decimal(0) <= exact_result <= decimal_floor:
                    return PHI_LOWER_BOUND
                if exact_result > float_max:
                    raise TweedieNumericalError("Pearson dispersion exceeds float64 range")
                return max(float(exact_result), PHI_LOWER_BOUND)

            result_interval = _decimal_pearson_interval(
                y,
                mu,
                decimal_power,
                weights,
                decimal_denominator,
                precision,
                multiplicity=multiplicity,
            )
            if Decimal(0) <= result_interval.lower and result_interval.upper <= decimal_floor:
                return PHI_LOWER_BOUND
            if result_interval.lower > float_max:
                raise TweedieNumericalError("Pearson dispersion exceeds float64 range")
            if result_interval.upper <= float_max:
                lower_float = float(result_interval.lower)
                upper_float = float(result_interval.upper)
                if lower_float == upper_float and _float64_rounding_bin_contains(
                    result_interval, lower_float
                ):
                    return max(lower_float, PHI_LOWER_BOUND)
            if precision == _PEARSON_MAX_CERTIFICATION_PRECISION:
                break
            precision = min(_PEARSON_MAX_CERTIFICATION_PRECISION, 2 * precision)
    except DecimalException as exc:
        raise TweedieNumericalError(
            "Pearson dispersion could not be certified near float64 range"
        ) from exc
    raise TweedieNumericalError("Pearson dispersion could not be certified near float64 range")


def _pearson_scalar_upper_exponent(
    *,
    weight_exponent: int,
    residual_exponent: int,
    mean_exponent: int,
    power: float,
    term_count_exponent: int,
    denominator_exponent: int,
) -> float:
    """Return an outward-rounded upper exponent for the exact Pearson result."""
    mean_power_lower_exponent = math.nextafter(
        power * float(mean_exponent - 1),
        -math.inf,
    )
    integer_term_exponent = weight_exponent + 2 * residual_exponent
    term_upper_exponent = math.nextafter(
        float(integer_term_exponent) - mean_power_lower_exponent,
        math.inf,
    )
    numerator_upper_exponent = math.nextafter(
        term_upper_exponent + float(term_count_exponent),
        math.inf,
    )
    return math.nextafter(
        numerator_upper_exponent - float(denominator_exponent - 1),
        math.inf,
    )


def _pearson_scalar_range_is_ordinary(
    residuals: NDArray[np.float64],
    mus: NDArray[np.float64],
    power: float,
    weights: NDArray[np.float64] | None,
    denominator: float,
    nonzero_count: int,
) -> bool:
    """Prove from scalar extrema that the exact result is below ``2**1023``."""
    minimum_residual = float(np.min(residuals))
    maximum_residual = float(np.max(residuals))
    maximum_absolute_residual = max(abs(minimum_residual), abs(maximum_residual))
    residual_upper = math.nextafter(maximum_absolute_residual, math.inf)
    if math.isfinite(residual_upper):
        residual_exponent = math.frexp(residual_upper)[1]
    else:
        residual_exponent = np.finfo(np.float64).maxexp

    maximum_weight = 1.0 if weights is None else float(np.max(weights))
    weight_exponent = math.frexp(maximum_weight)[1]
    minimum_mean = float(np.min(mus))
    mean_exponent = math.frexp(minimum_mean)[1]

    term_count_exponent = (nonzero_count - 1).bit_length()
    denominator_exponent = math.frexp(denominator)[1]
    dispersion_upper_exponent = _pearson_scalar_upper_exponent(
        weight_exponent=weight_exponent,
        residual_exponent=residual_exponent,
        mean_exponent=mean_exponent,
        power=power,
        term_count_exponent=term_count_exponent,
        denominator_exponent=denominator_exponent,
    )
    return dispersion_upper_exponent <= np.finfo(np.float64).maxexp - 1


def _direct_pearson_dispersion_if_safe(
    residuals: NDArray[np.float64],
    mus: NDArray[np.float64],
    power: float,
    weights: NDArray[np.float64] | None,
    denominator: float,
    nonzero_count: int,
) -> float | None:
    """Return a direct Pearson result only when range and accuracy are guarded."""
    if 8 * nonzero_count < len(residuals):
        return None
    if not _pearson_scalar_range_is_ordinary(
        residuals,
        mus,
        power,
        weights,
        denominator,
        nonzero_count,
    ):
        return None

    terms = np.empty_like(residuals)
    try:
        # Float exceptions detect every lossy underflow plus overflow/invalid arithmetic
        # while the ufuncs already own the data, avoiding separate full-array safety scans.
        # An exact subnormal intermediate loses no information and can remain on this path.
        with np.errstate(over="raise", under="raise", divide="raise", invalid="raise"):
            np.square(residuals, out=terms)
            if weights is not None:
                np.multiply(terms, weights, out=terms)
            np.power(mus, power, out=residuals)
            np.divide(terms, residuals, out=terms)
            numerator = float(np.sum(terms))
            result = float(np.divide(numerator, denominator))
    except FloatingPointError:
        return None

    if not math.isfinite(numerator) or numerator < np.finfo(np.float64).tiny:
        return None
    top_binade = math.ldexp(1.0, np.finfo(np.float64).maxexp - 1)
    if not math.isfinite(result) or result < np.finfo(np.float64).tiny or result >= top_binade:
        return None
    return max(result, PHI_LOWER_BOUND)


def pearson_dispersion(
    y: object,
    mu: object,
    p: object,
    weights: object | None,
    df_resid: object,
) -> float:
    """Return the weighted Pearson estimate of Tweedie dispersion."""
    power = normalize_tweedie_power(p)
    denominator = normalize_positive_scalar("df_resid", df_resid)
    y_arr = _as_real_float64_array("y", y)
    mu_arr = _as_real_float64_array("mu", mu)
    weight_arr = None if weights is None else _as_real_float64_array("weights", weights)
    if (
        y_arr.ndim != 1
        or mu_arr.shape != y_arr.shape
        or (weight_arr is not None and weight_arr.shape != y_arr.shape)
    ):
        raise ValueError("y, mu, and weights must be matching one-dimensional arrays")
    if np.any(~np.isfinite(y_arr)) or np.any(y_arr < 0.0):
        raise ValueError("y must be finite and nonnegative")
    if np.any(~np.isfinite(mu_arr)) or np.any(mu_arr <= 0.0):
        raise ValueError("mu must be finite and strictly positive")
    if weight_arr is not None and (np.any(~np.isfinite(weight_arr)) or np.any(weight_arr <= 0.0)):
        raise ValueError("weights must be finite and strictly positive")
    residual = np.subtract(y_arr, mu_arr)
    nonzero_count = int(np.count_nonzero(residual))
    if nonzero_count == 0:
        return PHI_LOWER_BOUND

    direct_result = _direct_pearson_dispersion_if_safe(
        residual,
        mu_arr,
        power,
        weight_arr,
        denominator,
        nonzero_count,
    )
    if direct_result is not None:
        return direct_result

    np.subtract(y_arr, mu_arr, out=residual)
    np.abs(residual, out=residual)
    nonzero = residual > 0.0
    nonzero_weights = (
        np.ones(nonzero_count, dtype=np.float64) if weight_arr is None else weight_arr[nonzero]
    )
    range_route = _pearson_float64_range_route(
        residual[nonzero],
        mu_arr[nonzero],
        power,
        nonzero_weights,
        denominator,
    )
    if range_route == "overflow":
        raise TweedieNumericalError("Pearson dispersion exceeds float64 range")
    log_terms = (
        np.log(nonzero_weights) + 2.0 * np.log(residual[nonzero]) - power * np.log(mu_arr[nonzero])
    )
    if range_route == "certify" or _pearson_rows_are_identical(
        y_arr[nonzero], mu_arr[nonzero], nonzero_weights
    ):
        return _pearson_boundary_result(
            y_arr[nonzero],
            mu_arr[nonzero],
            power,
            nonzero_weights,
            denominator,
            log_terms,
        )
    log_phi = float(logsumexp(log_terms) - math.log(denominator))
    log_float_max = math.log(float(np.finfo(np.float64).max))
    if log_phi > log_float_max:
        raise TweedieNumericalError("Pearson dispersion float64 path exceeded its proven range")
    return max(float(math.exp(log_phi)), PHI_LOWER_BOUND)


def _scaled_positive_ratio(
    shape: tuple[int, ...],
    numerators: tuple[object, ...],
    denominators: tuple[object, ...],
    *,
    binary_exponent: object = 0,
) -> NDArray[np.float64]:
    """Multiply and divide positive factors without overflowing intermediates."""
    mantissa = np.ones(shape, dtype=np.float64)
    exponent = np.broadcast_to(np.asarray(binary_exponent, dtype=np.int64), shape).copy()
    for value in numerators:
        factor_mantissa, factor_exponent = np.frexp(np.asarray(value, dtype=np.float64))
        mantissa *= factor_mantissa
        exponent += factor_exponent
    for value in denominators:
        factor_mantissa, factor_exponent = np.frexp(np.asarray(value, dtype=np.float64))
        mantissa /= factor_mantissa
        exponent -= factor_exponent
    mantissa, adjustment = np.frexp(mantissa)
    exponent += adjustment
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result: NDArray[np.float64] = np.ldexp(mantissa, exponent)
    return result


def _scaled_power_deviance(
    base: object,
    power: SupportsFloat,
    numerators: tuple[object, ...],
    denominators: tuple[object, ...] = (),
) -> NDArray[np.float64]:
    """Compose a positive deviance power product in authoritative float64.

    Keeping the power of ``base`` as a compensated binary exponent avoids both
    an unrepresentable intermediate power and the precision loss of a final
    large logarithm/exponential round trip.
    """
    base_arr = np.asarray(base, dtype=np.float64)
    float_power = float(power)

    base_mantissa, base_exponent = np.frexp(base_arr)
    normalized_mantissa = 2.0 * base_mantissa
    normalized_exponent = base_exponent.astype(np.int64) - 1

    exponent_factor = normalized_exponent.astype(np.float64)
    exponent_product = float_power * exponent_factor
    splitter = 134_217_729.0  # 2**27 + 1, for an error-free float64 product split.
    split_power = splitter * float_power
    power_high = split_power - (split_power - float_power)
    power_low = float_power - power_high
    exponent_error = (power_high * exponent_factor - exponent_product) + (
        power_low * exponent_factor
    )

    binary_exponent = np.floor(exponent_product).astype(np.int64)
    fractional_exponent = exponent_product - binary_exponent + exponent_error
    below_zero = fractional_exponent < 0.0
    if np.any(below_zero):
        fractional_exponent[below_zero] += 1.0
        binary_exponent[below_zero] -= 1
    at_least_one = fractional_exponent >= 1.0
    if np.any(at_least_one):
        fractional_exponent[at_least_one] -= 1.0
        binary_exponent[at_least_one] += 1

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        fractional_power = np.exp2(fractional_exponent + float_power * np.log2(normalized_mantissa))
    result = _scaled_positive_ratio(
        base_arr.shape,
        (*numerators, fractional_power),
        denominators,
        binary_exponent=binary_exponent,
    )
    if np.any(~np.isfinite(result)) or np.any(result <= 0.0):
        raise TweedieNumericalError("unit deviance could not be represented as finite")
    return result


def _scaled_response_factored_deviance(
    y: object,
    mu: object,
    power_minus_one: SupportsFloat,
    log_correction: object,
) -> NDArray[np.float64]:
    """Compose ``2 * y * mu**(1 - p) * correction`` in float64."""
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        correction = np.exp(np.asarray(log_correction, dtype=np.float64))
    return _scaled_power_deviance(
        mu,
        -float(power_minus_one),
        (2.0, y, correction),
    )


def compound_poisson_gamma_parameters(
    mu: object,
    phi: object,
    p: object,
    *,
    weights: object | None = None,
) -> CompoundPoissonGammaParameters:
    power = normalize_tweedie_power(p)
    mu_arr = _as_real_float64_array("mu", mu)
    weight_arr = (
        np.ones_like(mu_arr) if weights is None else _as_real_float64_array("weights", weights)
    )
    if _contains_masked_array(phi):
        raise TypeError("phi must be a real numeric scalar or array without a mask")
    phi_raw = np.asarray(phi)
    if phi_raw.ndim == 0:
        dispersion: float | NDArray[np.float64] = normalize_positive_scalar("phi", phi)
    elif phi_raw.ndim == 1 and phi_raw.shape == mu_arr.shape:
        dispersion = _as_real_float64_array("phi", phi)
    else:
        raise ValueError("phi must be one positive real scalar or match the one-dimensional mu")
    if mu_arr.ndim != 1 or weight_arr.shape != mu_arr.shape:
        raise ValueError("mu and weights must be matching one-dimensional arrays")
    if np.any(~np.isfinite(mu_arr)) or np.any(mu_arr <= 0.0):
        raise ValueError("mu must be finite and strictly positive")
    if np.any(~np.isfinite(dispersion)) or np.any(dispersion <= 0.0):
        raise ValueError("phi must be finite and strictly positive")
    if np.any(~np.isfinite(weight_arr)) or np.any(weight_arr <= 0.0):
        raise ValueError("weights must be finite and strictly positive")
    q = 2.0 - power
    r = power - 1.0
    mu_to_q = np.power(mu_arr, q)
    mu_to_r = np.power(mu_arr, r)
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        effective_phi = dispersion / weight_arr
        rate = mu_to_q / (effective_phi * q)
        scale = effective_phi * r * mu_to_r

    invalid_rate = ~np.isfinite(rate) | (rate <= 0.0)
    if np.any(invalid_rate):
        stable_rate = _scaled_positive_ratio(
            mu_arr.shape,
            (mu_to_q, weight_arr),
            (dispersion, q),
        )
        rate[invalid_rate] = stable_rate[invalid_rate]
    invalid_scale = ~np.isfinite(scale) | (scale <= 0.0)
    if np.any(invalid_scale):
        stable_scale = _scaled_positive_ratio(
            mu_arr.shape,
            (dispersion, r, mu_to_r),
            (weight_arr,),
        )
        scale[invalid_scale] = stable_scale[invalid_scale]

    if np.any(~np.isfinite(rate)) or np.any(rate <= 0.0):
        raise TweedieNumericalError("Poisson rate could not be represented as finite and positive")
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
        raise TweedieNumericalError("Gamma scale could not be represented as finite and positive")
    shape = q / r
    return CompoundPoissonGammaParameters(rate=rate, shape=shape, scale=scale)


def _clamp_ulp_negative(
    values: NDArray[np.float64], reference: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Clamp a negative no larger than one local ULP; reject anything material."""
    value_arr, reference_arr = np.broadcast_arrays(
        np.asarray(values),
        np.asarray(reference),
    )
    absolute_reference = np.abs(reference_arr)
    if np.any(~np.isfinite(absolute_reference)):
        raise TweedieNumericalError("unit deviance ratio reference was not finite")
    with np.errstate(over="ignore", invalid="ignore"):
        next_larger = np.nextafter(
            absolute_reference,
            np.full_like(absolute_reference, np.inf),
        )
        upward_ulp = next_larger - absolute_reference
    downward_ulp = absolute_reference - np.nextafter(
        absolute_reference,
        np.zeros_like(absolute_reference),
    )
    tolerance: NDArray[np.float64] = np.where(
        np.isfinite(upward_ulp),
        upward_ulp,
        downward_ulp,
    )
    if np.any(value_arr < -tolerance):
        raise TweedieNumericalError("unit deviance ratio became materially negative")
    result: NDArray[np.float64] = np.maximum(value_arr, 0.0)
    return result


def _certified_log_positive(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Take a logarithm only after certifying a dimensionless positive factor."""
    certified = _clamp_ulp_negative(values, np.ones_like(values))
    if np.any(certified <= 0.0) or np.any(~np.isfinite(certified)):
        raise TweedieNumericalError("unit deviance ratio was not certified positive")
    result: NDArray[np.float64] = np.log(certified)
    return result


def _series_deviance_ratio(log_ratio: NDArray[np.float64], q: float) -> NDArray[np.float64]:
    """Return half-deviance divided by ``mu**q`` as a ratio series.

    Dividing each coefficient by ``(1 - p) * q`` analytically turns it into
    ``1 + q + ... + q**(k - 2)``. That geometric sum stays well-conditioned
    at both power boundaries.
    """
    term: NDArray[np.float64] = log_ratio * log_ratio / 2.0
    series: NDArray[np.float64] = term.copy()
    geometric_sum = 1.0
    q_power = 1.0
    for order in range(3, 32):
        term *= log_ratio / order
        q_power *= q
        geometric_sum += q_power
        series += geometric_sum * term
    return series


def _close_deviance_ratio(log_ratio: NDArray[np.float64], q: float) -> NDArray[np.float64]:
    """Return the ratio series used by the explicit close-ratio branch."""
    return _series_deviance_ratio(log_ratio, q)


def _lower_boundary_composition(
    log_ratio: NDArray[np.float64], r: float, q: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compose after factoring ``r = p - 1`` around the lower boundary.

    The second result contains the correction after a leading
    ``exp(log_ratio)``. The caller combines that response power with
    ``mu**q`` before rounding whenever it is present.
    """
    result = np.empty_like(log_ratio)
    above_correction = np.full_like(log_ratio, np.nan)
    series = np.abs(log_ratio) <= _SERIES_LOG_RATIO_BOUNDARY
    if np.any(series):
        ratio_deviance = _series_deviance_ratio(log_ratio[series], q)
        result[series] = _certified_log_positive(ratio_deviance)

    above = ~series & (log_ratio > 0.0)
    if np.any(above):
        z = log_ratio[above]
        factor = -np.expm1(-r * z) / r + np.expm1(-z)
        correction = _certified_log_positive(factor) - np.log(q)
        above_correction[above] = correction
        result[above] = z + correction

    below = ~series & ~above
    if np.any(below):
        z = log_ratio[below]
        scaled_term = np.empty_like(z)
        exponent = -r * z
        direct = exponent <= 0.5
        if np.any(direct):
            scaled_term[direct] = np.exp(z[direct]) * np.expm1(exponent[direct]) / r
        if np.any(~direct):
            scaled_term[~direct] = (np.exp(q * z[~direct]) - np.exp(z[~direct])) / r
        factor = -np.expm1(z) - scaled_term
        result[below] = _certified_log_positive(factor) - np.log(q)
    return result, above_correction


def _lower_power_log_deviance_ratio(
    log_ratio: NDArray[np.float64], r: float, q: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the lower-power log ratio and response-power correction."""
    return _lower_boundary_composition(log_ratio, r, q)


def _ordinary_log_deviance_ratio(
    log_ratio: NDArray[np.float64], r: float, q: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the ordinary log ratio and response-power correction."""
    if r <= q:
        return _lower_boundary_composition(log_ratio, r, q)

    result = np.empty_like(log_ratio)
    above_correction = np.full_like(log_ratio, np.nan)
    series = np.abs(log_ratio) <= _SERIES_LOG_RATIO_BOUNDARY
    if np.any(series):
        ratio_deviance = _series_deviance_ratio(log_ratio[series], q)
        result[series] = _certified_log_positive(ratio_deviance)

    above = ~series & (log_ratio > 0.0)
    if np.any(above):
        z = log_ratio[above]
        factor = -np.expm1(-z) - np.exp(-r * z) * (-np.expm1(-q * z) / q)
        correction = _certified_log_positive(factor) - np.log(r)
        above_correction[above] = correction
        result[above] = z + correction

    below = ~series & ~above
    if np.any(below):
        z = log_ratio[below]
        factor = -np.expm1(q * z) / q + np.expm1(z)
        result[below] = _certified_log_positive(factor) - np.log(r)
    return result, above_correction


def _representable_unit_deviance(log_values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Exponentiate a final log deviance only when float64 can represent it."""
    value_dtype = np.asarray(log_values).dtype
    log_float_max = np.log(np.asarray(np.finfo(np.float64).max, dtype=value_dtype))
    if np.any(~np.isfinite(log_values)) or np.any(log_values > log_float_max):
        raise TweedieNumericalError("unit deviance could not be represented as finite")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        extended_result = np.exp(log_values)
        result: NDArray[np.float64] = np.asarray(extended_result, dtype=np.float64)
    if np.any(~np.isfinite(result)) or np.any(result <= 0.0):
        raise TweedieNumericalError("unit deviance could not be represented as finite")
    return result


def tweedie_unit_deviance(y: object, mu: object, p: object) -> NDArray[np.float64]:
    power = normalize_tweedie_power(p)
    y_arr, mu_arr = np.broadcast_arrays(
        _as_real_float64_array("y", y),
        _as_real_float64_array("mu", mu),
    )
    if np.any(~np.isfinite(y_arr)) or np.any(y_arr < 0.0):
        raise ValueError("y must be finite and nonnegative")
    if np.any(~np.isfinite(mu_arr)) or np.any(mu_arr <= 0.0):
        raise ValueError("mu must be finite and strictly positive")

    work_type = _DEVIANCE_DTYPE.type
    work_power = work_type(power)
    two = work_type(2.0)
    q = two - work_power
    result = np.empty(y_arr.shape, dtype=np.float64)
    equal = y_arr == mu_arr
    zero = (y_arr == 0.0) & ~equal
    result[equal] = 0.0
    if np.any(zero):
        result[zero] = _scaled_power_deviance(
            mu_arr[zero],
            q,
            (2.0,),
            (q,),
        )

    positive = ~(equal | zero)
    if np.any(positive):
        positive_y = np.asarray(y_arr[positive], dtype=_DEVIANCE_DTYPE)
        positive_mu = np.asarray(mu_arr[positive], dtype=_DEVIANCE_DTYPE)
        log_y = np.log(positive_y)
        log_mu = np.log(positive_mu)
        log_ratio = log_y - log_mu
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            ratio = positive_y / positive_mu
        work_tiny = work_type(np.finfo(_DEVIANCE_DTYPE).tiny)
        usable_ratio = np.isfinite(ratio) & (ratio >= work_tiny)
        if np.any(usable_ratio):
            log_ratio[usable_ratio] = np.log(ratio[usable_ratio])

        close = np.abs(log_ratio) <= 1e-3
        if np.any(close):
            # Subtracting nearby logarithms loses the precision needed by the remainder series.
            relative_difference = (positive_y[close] - positive_mu[close]) / positive_mu[close]
            log_ratio[close] = np.log1p(relative_difference)

        r = work_power - work_type(1.0)
        lower_power = ~close & (r <= _LOWER_POWER_BOUNDARY)
        ordinary = ~(close | lower_power)
        log_ratio_deviance = np.empty_like(log_ratio)
        above_correction = np.full_like(log_ratio, np.nan)

        if np.any(close):
            ratio_deviance = _close_deviance_ratio(log_ratio[close], q)
            log_ratio_deviance[close] = _certified_log_positive(ratio_deviance)
        if np.any(lower_power):
            branch_log_ratio, branch_correction = _lower_power_log_deviance_ratio(
                log_ratio[lower_power], r, q
            )
            log_ratio_deviance[lower_power] = branch_log_ratio
            above_correction[lower_power] = branch_correction
        if np.any(ordinary):
            branch_log_ratio, branch_correction = _ordinary_log_deviance_ratio(
                log_ratio[ordinary], r, q
            )
            log_ratio_deviance[ordinary] = branch_log_ratio
            above_correction[ordinary] = branch_correction

        log_half_deviance = q * log_mu + log_ratio_deviance
        response_factored = np.isfinite(above_correction)
        if np.any(response_factored):
            log_half_deviance[response_factored] = (
                log_y[response_factored]
                - r * log_mu[response_factored]
                + above_correction[response_factored]
            )
        log_deviance = np.log(two) + log_half_deviance

        values = np.empty_like(log_ratio, dtype=np.float64)
        direct_usable = np.zeros(log_ratio.shape, dtype=np.bool_)
        if np.any(response_factored):
            values[response_factored] = _scaled_response_factored_deviance(
                positive_y[response_factored],
                positive_mu[response_factored],
                r,
                above_correction[response_factored],
            )
            direct_usable[response_factored] = True
        series_region = np.abs(log_ratio) <= _SERIES_LOG_RATIO_BOUNDARY
        mean_factored = ~(series_region | response_factored)
        if np.any(mean_factored):
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                mean_correction = np.exp(
                    np.asarray(log_ratio_deviance[mean_factored], dtype=np.float64)
                )
            values[mean_factored] = _scaled_power_deviance(
                positive_mu[mean_factored],
                q,
                (2.0, mean_correction),
            )
            direct_usable[mean_factored] = True
        if np.any(series_region):
            ratio_deviance = _series_deviance_ratio(log_ratio[series_region], q)
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                scale = np.power(positive_mu[series_region], q)
                direct_extended = scale * (two * ratio_deviance)
                direct_float = np.asarray(direct_extended, dtype=np.float64)
            usable = (
                (scale >= work_tiny)
                & np.isfinite(direct_extended)
                & np.isfinite(direct_float)
                & (direct_float > 0.0)
            )
            series_indices = np.flatnonzero(series_region)
            direct_indices = series_indices[usable]
            direct_usable[direct_indices] = True
            values[direct_indices] = direct_float[usable]

        unresolved = ~direct_usable
        if np.any(unresolved):
            values[unresolved] = _representable_unit_deviance(log_deviance[unresolved])
        result[positive] = values
    return result
