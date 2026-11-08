"""Float64 Dirichlet density and signed natural jets for row-first compositions.

The response is the mathematical normalization of the supplied positive row,
not a repeatedly rounded division. Close log ratios use compensated cross
products. Gamma normalizers and scaled digamma residuals remove cancellation
at high concentration. No parameter-dependent cache or shape wall is used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations_with_replacement

import numpy as np

from superglm.distributional.kernels._common import readonly, validated_derivative_order
from superglm.distributional.kernels._severity_primitives import (
    _EULER,
    _SRES,
    _ZETA,
    gnorm,
    sres_0123,
)

_U = 2.0**-53
NORMALIZATION_POLICY = "positive-shares/compensated-normalization-8gamma/v1"


def _sum(values):
    return math.fsum(map(float, values))


def prepare_log_shares(response):
    """Validate the simplex and prepare normalized logs once per bound plan.

    For K positive inputs the allowed sum error is 8 gamma_(K+1) times
    the row magnitude, with unit roundoff u=2^-53. Retain the supplied bytes.
    The low sum part matters when one component rounds to one.
    """
    x = np.asarray(response, dtype=np.float64)
    if x.ndim != 2 or not len(x) or x.shape[1] < 2:
        raise ValueError("Dirichlet response must have shape (n, K), K >= 2")
    if not np.isfinite(x).all() or np.any(x <= 0):
        raise ValueError("Dirichlet shares must be finite and strictly positive")
    k = x.shape[1]
    allowance = 8 * ((k + 1) * _U / (1 - (k + 1) * _U))
    logs = np.empty_like(x)
    for i, row in enumerate(x):
        try:
            total = _sum(row)
            tail = _sum([*row, -total])
        except OverflowError as exc:
            raise ValueError("Dirichlet shares must sum to one") from exc
        if abs(total - 1.0) > allowance * total:
            raise ValueError("Dirichlet shares must sum to one within 8 gamma_(K+1)")
        logtotal = math.log(total) + math.log1p(tail / total)
        logs[i] = [math.log(float(v)) - logtotal for v in row]
    return logs


def _product_parts(a, b):
    """Exact product of frexp mantissas; every split operation stays normal."""
    product = a * b
    split_a = 134217729.0 * a
    hi_a = split_a - (split_a - a)
    lo_a = a - hi_a
    split_b = 134217729.0 * b
    hi_b = split_b - (split_b - b)
    lo_b = b - hi_b
    error = ((hi_a * hi_b - product) + hi_a * lo_b + lo_a * hi_b) + lo_a * lo_b
    return product, error


def _close_ratio(x, alpha, j):
    """(y_j / mu_j - 1) from sum_k (a_k x_j - a_j x_k).

    Products are split before scaling by the exponent of a_j. Their sum
    keeps the ulp residual of both row totals instead of subtracting two
    rounded totals. The caller only enters when the ratio is near one, so
    the scaled positive/negative sums cannot overflow. Underflow of remote
    summands costs at most half a subnormal in these scaled coordinates.
    """
    aj, exponent = math.frexp(float(alpha[j]))
    xj, xje = math.frexp(float(x[j]))
    parts = []
    for k in range(len(x)):
        if k == j:
            continue
        ak, ake = math.frexp(float(alpha[k]))
        xk, xke = math.frexp(float(x[k]))
        p, e = _product_parts(ak, xj)
        q, f = _product_parts(aj, xk)
        parts.extend(
            (
                math.ldexp(p, ake + xje - exponent),
                math.ldexp(e, ake + xje - exponent),
                -math.ldexp(q, xke),
                -math.ldexp(f, xke),
            )
        )
    return _sum(parts) / aj / _sum(x)


def _r_minus_log1p(r):
    if abs(r) > 0.125:
        return r - math.log1p(r)
    # Alternating Taylor series, |r| <= 1/8; 20 terms leave < u of r^2/2.
    term = r * r
    terms = []
    for degree in range(2, 22):
        terms.append(term / degree)
        term *= -r
    return _sum(terms)


def _scaled_product(*values):
    mantissa, exponent = 1.0, 0
    for value in values:
        m, e = math.frexp(float(value))
        mantissa *= m
        exponent += e
    return math.ldexp(mantissa, exponent)


def _residual_difference(a, total, other, order):
    """Scalar counterpart of the directional recurrence below."""
    if a >= 10:
        mu, rho = a / total, other / total
        inverse = 1 / a
        power = inverse
        terms = []

        def delta(p):
            return -rho * _sum(mu**i for i in range(p))

        terms.append((-0.5 if order == 0 else 0.5) * delta(order + 1))
        for i, coefficient in enumerate(_SRES[order]):
            terms.append(coefficient * power * delta(order + 2 * (i + 1)))
            power *= inverse * inverse
        return _sum(terms)
    shift = math.ceil(10 - a)
    left, right = a + shift, total + shift
    shifted = _residual_difference(left, right, other, order)
    if order == 0:
        terms = [(a / left) * shifted, a * (math.log1p(other / left) - math.log1p(other / a))]
        terms.extend((a / (a + i)) * (other / (total + i)) for i in range(shift))
    else:
        terms = [(a / left) ** 2 * shifted, a * (other / total), -(a / left) * (a / right) * other]
        terms.extend(
            -(a / (a + i)) * (a / (total + i)) * (other / (a + i) + other / (total + i))
            for i in range(shift)
        )
    return _sum(terms)


def _row_state(x, logs, alpha):
    total = _sum(alpha)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("nonrepresentable concentration sum")
    mu = alpha / total
    if np.any(mu == 0):
        raise ValueError("nonrepresentable positive mean share")
    residual = np.array([sres_0123(float(a)) for a in alpha])
    total_residual = np.array(sres_0123(total))
    ratios = np.empty(len(alpha))
    divergence = []
    for j, a in enumerate(alpha):
        # The logs only select the close branch; they never decide a close ratio.
        rough = logs[j] + math.log(total) - math.log(float(a))
        if abs(rough) < 0.5:
            r = _close_ratio(x, alpha, j)
            ratios[j] = math.log1p(r)
            divergence.append(a * _r_minus_log1p(r))
        else:
            ratios[j] = rough
            # a*(r-log1p(r)), without forming a possibly overflowing y/mu.
            divergence.append(_sum((total * math.exp(logs[j]), -a, -a * rough)))
    value = _sum((-_sum(divergence), -gnorm(total), *[gnorm(float(a)) for a in alpha], *(-logs)))
    scaled_score = np.empty(len(alpha))
    k = len(alpha)
    scaled_hessian = np.empty((k, k))
    for j in range(k):
        other = _sum(np.delete(alpha, j))
        if mu[j] > 0.5:
            # A dominant component can have a residual smaller than one ulp
            # of either sres term, even though its scaled score is representable.
            score_residual = _residual_difference(float(alpha[j]), total, other, 0)
            hessian_residual = _residual_difference(float(alpha[j]), total, other, 1)
        else:
            score_residual = mu[j] * total_residual[0] - residual[j, 0]
            hessian_residual = mu[j] ** 2 * total_residual[1] - residual[j, 1]
        scaled_score[j] = alpha[j] * ratios[j] + score_residual
        scaled_hessian[j, j] = -alpha[j] * (other / total) + hessian_residual
        for column in range(j + 1, k):
            scaled_hessian[j, column] = scaled_hessian[column, j] = (
                alpha[j] * mu[column] + mu[j] * mu[column] * total_residual[1]
            )
    return value, scaled_score, scaled_hessian, total, mu, residual, total_residual, ratios


@dataclass(frozen=True)
class DirichletRows:
    optimizing_log_likelihood: np.ndarray
    parameter_independent_carrier: np.ndarray
    score: np.ndarray | None
    hessian_packed: np.ndarray | None
    valid: np.ndarray
    failure_reasons: tuple[str | None, ...]

    @property
    def reported_log_likelihood(self):
        return self.optimizing_log_likelihood

    @property
    def derivative_order(self):
        return 0 if self.score is None else (1 if self.hessian_packed is None else 2)


def evaluate_dirichlet_rows(response, theta, multiplier, *, derivative_order=2, log_shares=None):
    """Primitive core result; the family adapter supplies NaturalLikelihoodEvaluation.

    Invalid requested numerical channels have finite zero placeholders and a
    false row-validity flag. The scaled log-link reconstruction is an admission
    check, not a replacement Hessian or a rank decision. Order zero uses exactly
    the same normalized value as the derivative orders.
    """
    order = validated_derivative_order(derivative_order)
    x, alpha, mass = (np.asarray(v, dtype=np.float64) for v in (response, theta, multiplier))
    if x.ndim != 2 or alpha.shape != x.shape or mass.shape != (len(x),):
        raise ValueError("Dirichlet response, parameters and row multiplier must agree")
    if np.any(~np.isfinite(mass)) or np.any(mass <= 0):
        raise ValueError("row multiplier must be finite and positive")
    logs = prepare_log_shares(x) if log_shares is None else np.asarray(log_shares)
    if logs.shape != x.shape or not np.isfinite(logs).all():
        raise ValueError("prepared log shares must match the response")
    n, k = x.shape
    pairs = tuple(combinations_with_replacement(range(k), 2))
    values = np.zeros(n)
    score = np.zeros((n, k)) if order else None
    hessian = np.zeros((n, len(pairs))) if order == 2 else None
    valid = np.zeros(n, dtype=bool)
    failures = []
    for i in range(n):
        try:
            a = alpha[i]
            if np.any(a <= 0) or not np.isfinite(a).all():
                raise ValueError("alphas must be finite and positive")
            with np.errstate(over="raise", divide="raise", invalid="raise", under="ignore"):
                value, scaled_s, scaled_h, *_ = _row_state(x[i], logs[i], a)
                value *= mass[i]
                if not math.isfinite(value):
                    raise ValueError("nonrepresentable normalized log likelihood")
                if order:
                    natural_s = (scaled_s / a) * mass[i]
                    target = scaled_s * mass[i]
                    reconstructed = natural_s * a
                    if not np.isfinite(natural_s).all() or np.any(
                        np.abs(reconstructed - target) > 128 * _U * np.abs(target)
                    ):
                        raise ValueError("nonrepresentable natural score / scaled log-link score")
                if order == 2:
                    natural_h = np.array(
                        [
                            _scaled_product(scaled_h[j, column], mass[i], 1 / a[j], 1 / a[column])
                            for j, column in pairs
                        ]
                    )
                    target_h = np.array([scaled_h[j, column] * mass[i] for j, column in pairs])
                    reconstruction = np.array(
                        [
                            _scaled_product(h, a[j], a[column])
                            for h, (j, column) in zip(natural_h, pairs, strict=True)
                        ]
                    )
                    if not np.isfinite(natural_h).all() or np.any(
                        np.abs(reconstruction - target_h) > 256 * _U * np.abs(target_h)
                    ):
                        raise ValueError(
                            "nonrepresentable natural Hessian / scaled log-link curvature"
                        )
                values[i] = value
                if order:
                    score[i] = natural_s
                if order == 2:
                    hessian[i] = natural_h
            valid[i] = True
            failures.append(None)
        except (ValueError, OverflowError, FloatingPointError, ZeroDivisionError) as exc:
            failures.append(str(exc))
    return DirichletRows(
        readonly(values),
        readonly(np.zeros(n)),
        None if score is None else readonly(score),
        None if hessian is None else readonly(hessian),
        readonly(valid, dtype=bool),
        tuple(failures),
    )


def normalized_mean(theta):
    alpha = np.asarray(theta, dtype=np.float64)
    out = np.empty_like(alpha)
    for i, row in enumerate(alpha):
        try:
            total = _sum(row)
        except OverflowError as exc:
            raise ValueError("Dirichlet concentration sum is outside float64 range") from exc
        out[i] = row / total
    if np.any(out <= 0) or not np.isfinite(out).all():
        raise ValueError("Dirichlet positive mean share is outside float64 range")
    return readonly(out)


# A two-direction jet (value, a, b, ab). Each entry is a scalar, so the
# directional evaluator stores only O(K^2), never a third/fourth tensor.
def _mul(x, y):
    return np.array(
        [
            x[0] * y[0],
            x[1] * y[0] + x[0] * y[1],
            x[2] * y[0] + x[0] * y[2],
            _sum((x[3] * y[0], x[1] * y[2], x[2] * y[1], x[0] * y[3])),
        ]
    )


def _div(x, y):
    value = x[0] / y[0]
    first = (x[1] - value * y[1]) / y[0]
    second = (x[2] - value * y[2]) / y[0]
    mixed = _sum((x[3], -value * y[3], -first * y[2], -second * y[1])) / y[0]
    return np.array([value, first, second, mixed])


def _residual_jet(alpha, residual, order):
    m = order
    value = residual[m]
    a = alpha[0]
    if a < 0.25:
        # Differentiate the origin series itself: sres0+sres1 would
        # cancel -1+1 and erase the derivative of a tiny shape.
        slope = -a * (_EULER + math.log(a) + 1) if m == 0 else -a
        curvature = -a if m == 0 else 0.0
        power = a
        for i, coefficient in enumerate(_ZETA):
            power *= a
            degree = i + 2
            term = coefficient * power * (1 if i % 2 == 0 else -1)
            factor = 1 if m == 0 else degree - 1
            slope += degree * factor * term
            curvature += degree * (degree - 1) * factor * term
    elif a >= 10:
        slope_terms, curvature_terms = [], []
        inverse = 1 / a
        power = inverse
        for i, coefficient in enumerate(_SRES[m]):
            degree = 2 * i + 1
            term = coefficient * power
            slope_terms.append(-degree * term)
            curvature_terms.append(degree * (degree + 1) * term)
            power *= inverse * inverse
        slope, curvature = _sum(slope_terms), _sum(curvature_terms)
    else:
        slope = (m + 1) * value + residual[m + 1]
        curvature = m * (m + 1) * value + 2 * (m + 1) * residual[m + 1] + residual[m + 2]
    da, db, dab = alpha[1:] / alpha[0]
    return np.array([value, slope * da, slope * db, curvature * da * db + slope * dab])


def _log1p_jet(x):
    denominator = 1 + x[0]
    return np.array(
        [
            math.log1p(x[0]),
            x[1] / denominator,
            x[2] / denominator,
            x[3] / denominator - (x[1] / denominator) * (x[2] / denominator),
        ]
    )


def _residual_difference_jet(a, total, other, order):
    """a^(m+1) [psires_m(A)-psires_m(a)], preserving A-a separately.

    The asymptotic difference factors mu^p-1 as -rho sum(mu^i).
    Below ten, shift both arguments together and difference the recurrence's
    reciprocal terms before summation. These are float64 formulas even when
    a+other rounds back to a. This same expression supplies optional directions.
    """
    one = np.array([1.0, 0.0, 0.0, 0.0])
    if a[0] >= 10:
        mu = _div(a, total)
        rho = _div(other, total)
        mu[1:] = -rho[1:]

        def power_difference(p):
            series, term = one.copy(), one.copy()
            for _ in range(1, p):
                term = _mul(term, mu)
                series += term
            return -_mul(rho, series)

        constant = -0.5 if order == 0 else 0.5
        result = constant * power_difference(order + 1)
        inverse = _div(one, a)
        inverse2 = _mul(inverse, inverse)
        power = inverse.copy()
        for i, coefficient in enumerate(_SRES[order]):
            result += coefficient * _mul(power, power_difference(order + 2 * (i + 1)))
            power = _mul(power, inverse2)
        return result
    shift = math.ceil(10 - a[0])
    shifted_a, shifted_total = a + shift * one, total + shift * one
    ratio = _div(a, shifted_a)
    shifted = _residual_difference_jet(shifted_a, shifted_total, other, order)
    if order == 0:
        result = _mul(ratio, shifted)
        result += _mul(a, _log1p_jet(_div(other, shifted_a)) - _log1p_jet(_div(other, a)))
    else:
        result = _mul(_mul(ratio, ratio), shifted)
        result += _mul(a, _div(other, total)) - _mul(_mul(ratio, _div(a, shifted_total)), other)
    for i in range(shift):
        left, right = a + i * one, total + i * one
        if order == 0:
            result += _mul(_div(a, left), _div(other, right))
        else:
            result -= _mul(
                _mul(_div(a, left), _div(a, right)), _div(other, left) + _div(other, right)
            )
    return result


def dirichlet_curvature_direction(
    response, theta, multiplier, link_derivatives, first, second=None, *, log_shares
):
    """Observed predictor-curvature directions, using separable links to order four."""
    n, k = theta.shape
    result = np.empty((n, k * (k + 1) // 2))
    double = second is not None
    second = np.zeros_like(first) if second is None else second
    pairs = tuple(combinations_with_replacement(range(k), 2))
    for row in range(n):
        alpha = theta[row]
        _, _, _, _, _, residual, total_residual, ratios = _row_state(
            response[row], log_shares[row], alpha
        )
        d1, d2, d3, d4 = [d[row] for d in link_derivatives]
        v, w = first[row], second[row]
        ajet = np.column_stack((alpha, d1 * v, d1 * w, d2 * v * w))
        totaljet = np.array([_sum(ajet[:, c]) for c in range(4)])
        mujet, otherjet = [], []
        complementjets = []
        for j in range(k):
            complement = np.array([_sum(np.delete(ajet[:, c], j)) for c in range(4)])
            mu, rho = _div(ajet[j], totaljet), _div(complement, totaljet)
            if mu[0] > 0.5:
                mu[1:] = -rho[1:]
            else:
                rho[1:] = -mu[1:]
            mujet.append(mu)
            otherjet.append(rho)
            complementjets.append(complement)
        rtotal = _residual_jet(totaljet, total_residual, 1)
        bjet = np.empty((k, k, 4))
        for j, column in pairs:
            if j == column:
                remainder = (
                    _residual_difference_jet(ajet[j], totaljet, complementjets[j], 1)
                    if mujet[j][0] > 0.5
                    else _mul(_mul(mujet[j], mujet[j]), rtotal)
                    - _residual_jet(ajet[j], residual[j], 1)
                )
                b = -_mul(ajet[j], otherjet[j]) + remainder
            else:
                b = _mul(ajet[j], mujet[column]) + _mul(_mul(mujet[j], mujet[column]), rtotal)
            bjet[j, column] = bjet[column, j] = b
        # Differentiate a*log(y/mu) and the regular residual separately.
        # In dS=h*S+B*h the origin's +1 and -1 erase a tiny direction.
        r0total = _residual_jet(totaljet, total_residual, 0)
        sjet = np.empty((k, 4))
        for j in range(k):
            mu = mujet[j]
            ra, rb = -mu[1] / mu[0], -mu[2] / mu[0]
            ratiojet = np.array([ratios[j], ra, rb, -mu[3] / mu[0] + ra * rb])
            remainder = (
                _residual_difference_jet(ajet[j], totaljet, complementjets[j], 0)
                if mu[0] > 0.5
                else _mul(mu, r0total) - _residual_jet(ajet[j], residual[j], 0)
            )
            sjet[j] = _mul(ajet[j], ratiojet) + remainder
        qjet, zjet = [], []
        for j in range(k):
            qjet.append(
                _div(np.array([d1[j], d2[j] * v[j], d2[j] * w[j], d3[j] * v[j] * w[j]]), ajet[j])
            )
            zjet.append(
                _div(np.array([d2[j], d3[j] * v[j], d3[j] * w[j], d4[j] * v[j] * w[j]]), ajet[j])
            )
        for p, (j, column) in enumerate(pairs):
            h = _mul(_mul(qjet[j], qjet[column]), bjet[j, column])
            if j == column:
                h += _mul(zjet[j], sjet[j])
            result[row, p] = -multiplier[row] * h[3 if double else 1]
    if not np.isfinite(result).all():
        raise ValueError("Dirichlet optional curvature direction is outside float64 range")
    return readonly(result)
