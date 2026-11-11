"""Compiled NB2 unit deviance in two vectorised passes around NumPy's SIMD logarithms.

The half deviance ``y log(y / mu) - (y + theta) log((y + theta) / (mu + theta))``
takes one of three algorithms per row.

* ``y = 0`` (or any ``y`` that is not positive): ``theta log1p(mu / theta)``,
  with ``log(mu) - log(theta)`` where ``mu / theta`` overflows.
* Near ``y = mu`` (``|t| <= 1/8``, ``t = (y - mu) / mu``), with ``r = y - mu``,
  ``s = mu / (mu + theta)`` and ``1 - s = theta / (mu + theta)``, the series
  ``(r t) (1 - s) sum_k (-t)^k (1 + s + ... + s^k) / ((k + 1)(k + 2))``, from
  ``(1 + t) log1p(t) - t = sum_n (-t)^n / (n (n - 1))`` at ``(mu, t)`` and at
  ``(mu + theta, r / (mu + theta))``. Each term is under ``|t| <= 1/8`` of the one
  before, the first is 1/2 and the sum is at least 0.45, so it subtracts nothing.
  Eighteen terms leave a tail of at most ``(8 / 7) |t|^18 / 20`` against that
  sum, under ``u / 15`` at ``|t| = 1/8``. ``r`` is exact there (Sterbenz), and
  ``s`` and ``1 - s`` come from ``min(mu, theta) / max(mu, theta)``, so neither
  overflows.
* Otherwise ``y log(y (mu + theta) / (mu (y + theta))) - theta log((y + theta) /
  (mu + theta))``, which leaves no theta-sized factor on a logarithm near zero:
  the two terms' summed size is at most 35 times the result (at ``t = 1/8`` in
  the Poisson limit). Each logarithm takes ``log1p`` of its difference where its
  argument lies in (1/2, 2) (Goldberg 1991, Theorem 4), and the ``log`` of the
  quotient elsewhere.

The first pass is branch-free so that it vectorises: every row evaluates the
series (at ``t = 0`` off its window) and writes four arguments, two for ``log1p``
and two for ``log``, setting each one its row does not use to ``0`` or ``1``,
whose logarithm is an exact zero. NumPy then takes the four logarithms in SIMD,
about a tenth of the cost of a compiled scalar call, and the second pass adds
each pair and combines them. A quotient that is not a finite normal number, from
a tiny ``y`` or a trial ``mu`` of 0, inf, NaN or below zero, is repaired after
both passes by the scalar formula, which takes a difference of logarithms so a
tiny ``y`` keeps a finite deviance. A result below zero in the last bits is set
to zero. The windows select the algorithm, not the value.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, types  # type: ignore[import-untyped]
from numpy.typing import ArrayLike, NDArray

from superglm._numba_compile import compile_closed

_MIN_NORMAL = float(np.finfo(np.float64).tiny)
_CENTRED_TERMS = 18
# Signed series coefficients (-1)^k / ((k + 1)(k + 2)).
_SERIES = np.array([(-1.0) ** k / ((k + 1) * (k + 2)) for k in range(_CENTRED_TERMS)])
# Rows per pass: the four logarithm buffers (1 MiB) stay in cache.
_CHUNK = 1 << 15


@njit(cache=True, error_model="numpy", inline="always")
def _split(count: float, mean: float, theta: float):
    """``(positive, r, t, centred, delta, cross_direct)`` for one row."""
    positive = count > 0.0
    r = count - mean
    t = r / mean
    centred = positive and abs(t) <= 0.125
    delta = (theta / (count + theta)) * t
    return positive, r, t, centred, delta, -0.5 < delta < 1.0


@njit(cache=True, error_model="numpy", inline="always")
def _is_normal(value: float) -> bool:
    return _MIN_NORMAL <= value < math.inf


@njit(cache=True, error_model="numpy")
def _plan_rows(count, mean, theta, out, cross_x, cross_q, shifted_x, shifted_q) -> int:
    """Write each row's centred deviance and its four logarithm arguments.

    Return the rows the scalar repair must evaluate.
    """
    exceptional = 0
    for row in range(count.size):
        y, mu = count[row], mean[row]
        positive, r, t, centred, delta, cross_direct = _split(y, mu, theta)
        mean_is_smaller = mu <= theta
        ratio = mu / theta if mean_is_smaller else theta / mu
        share = 1.0 / (1.0 + ratio)
        s = ratio * share if mean_is_smaller else share
        one_minus_s = share if mean_is_smaller else ratio * share
        tc = t if centred else 0.0
        series, power, partial = 0.0, 1.0, 1.0
        for k in range(_CENTRED_TERMS):
            series += power * partial * _SERIES[k]
            power *= tc
            partial = 1.0 + s * partial
        out[row] = 2.0 * ((r * tc) * one_minus_s * series)
        far = positive and not centred
        relative = r / (mu + theta)
        shifted_direct = -0.5 < relative < 1.0
        shifted_quotient = (y + theta) / (mu + theta)
        cross_quotient = y / mu
        zero_ratio = mu / theta
        cross_x[row] = zero_ratio if not positive else (delta if far and cross_direct else 0.0)
        cross_q[row] = cross_quotient if far and not cross_direct else 1.0
        shifted_x[row] = relative if far and shifted_direct else 0.0
        shifted_q[row] = shifted_quotient if far and not shifted_direct else 1.0
        exceptional += (
            far
            and (
                (not cross_direct and not _is_normal(cross_quotient))
                or (not shifted_direct and not _is_normal(shifted_quotient))
            )
        ) or (not positive and zero_ratio == math.inf and mu < math.inf)
    return exceptional


@njit(cache=True, error_model="numpy")
def _combine_rows(count, mean, theta, out, cross_x, cross_q, shifted_x, shifted_q) -> None:
    """Form every row the series did not from its logarithms, each one ``0`` or a pair's sum."""
    for row in range(count.size):
        y, mu = count[row], mean[row]
        positive, _, _, centred, _, cross_direct = _split(y, mu, theta)
        shifted_log = shifted_x[row] + shifted_q[row]
        cross_log = cross_x[row] + cross_q[row]
        cross_log = cross_log if cross_direct else cross_log - shifted_log
        half = y * cross_log - theta * shifted_log
        half = 0.0 if half < 0.0 else half
        value = 2.0 * half if positive else 2.0 * theta * cross_log
        out[row] = out[row] if centred else value


@njit(cache=True, error_model="numpy")
def _log_of_quotient(numerator: float, denominator: float) -> float:
    quotient = numerator / denominator
    if _is_normal(quotient):
        return math.log(quotient)
    return math.log(numerator) - math.log(denominator)


@njit(cache=True, error_model="numpy")
def _repair_rows(count, mean, theta, out) -> None:
    """Evaluate by the scalar formula each row whose quotient is not a finite normal number."""
    for row in range(count.size):
        y, mu = count[row], mean[row]
        positive, r, t, centred, delta, cross_direct = _split(y, mu, theta)
        if not positive:
            if mu / theta == math.inf and mu < math.inf:
                out[row] = 2.0 * theta * (math.log(mu) - math.log(theta))
            continue
        relative = r / (mu + theta)
        shifted_direct = -0.5 < relative < 1.0
        if centred or (
            (cross_direct or _is_normal(y / mu))
            and (shifted_direct or _is_normal((y + theta) / (mu + theta)))
        ):
            continue
        if shifted_direct:
            shifted_log = math.log1p(relative)
        else:
            shifted_log = _log_of_quotient(y + theta, mu + theta)
        if cross_direct:
            cross_log = math.log1p(delta)
        else:
            cross_log = _log_of_quotient(y, mu) - shifted_log
        half = y * cross_log - theta * shifted_log
        out[row] = 2.0 * (0.0 if half < 0.0 else half)


# One signature per kernel: closed to it, numba passes a read-only operand as it is.
_ROWS = types.Array(types.float64, 1, "C", readonly=True)
_BUFFER = types.Array(types.float64, 1, "C")
_PASS = (_ROWS, _ROWS, types.float64, _BUFFER, _BUFFER, _BUFFER, _BUFFER, _BUFFER)
_KERNELS = (
    (_plan_rows, _PASS),
    (_combine_rows, _PASS),
    (_repair_rows, (_ROWS, _ROWS, types.float64, _BUFFER)),
)


def unit_deviance(y: ArrayLike, mu: ArrayLike, theta: float) -> NDArray:
    """The NB2 unit deviance of ``y`` at ``mu``, broadcast.

    The module docstring gives the algorithms and their bounds. Beyond the output
    it allocates four buffers of at most ``_CHUNK`` rows.
    """
    compile_closed(*_KERNELS)
    count, mean = np.broadcast_arrays(
        np.asarray(y, dtype=np.float64), np.asarray(mu, dtype=np.float64)
    )
    shape = count.shape
    count = np.ascontiguousarray(count.reshape(-1))
    mean = np.ascontiguousarray(mean.reshape(-1))
    out = np.empty(count.size, dtype=np.float64)
    theta = float(theta)
    width = min(count.size, _CHUNK)
    buffers = [np.empty(width, dtype=np.float64) for _ in range(4)]
    for start in range(0, count.size, _CHUNK):
        stop = min(count.size, start + _CHUNK)
        y, mu, part = count[start:stop], mean[start:stop], out[start:stop]
        logs = [buffer[: stop - start] for buffer in buffers]
        exceptional = _plan_rows(y, mu, theta, part, *logs)
        # A trial mu of -theta or below has an infinite or NaN log1p, silently.
        with np.errstate(invalid="ignore", divide="ignore"):
            for buffer, function in zip(logs, (np.log1p, np.log, np.log1p, np.log), strict=True):
                function(buffer, out=buffer)
        _combine_rows(y, mu, theta, part, *logs)
        if exceptional:
            _repair_rows(y, mu, theta, part)
    return out.reshape(shape)


def _warmup_nb2_deviance() -> None:
    """Compile the three row kernels (called by superglm.warmup)."""
    compile_closed(*_KERNELS)
