"""Check that a rebound likelihood plan is the one a saved fit recorded.

A built-in plan identifier hashes the plan's exact inputs: family and row-law
tags, the resolved weight digest and, except for ``GaussianLS`` (whose plan
stores no response), the response digest (``NegativeBinomialLS`` also the
count digest), so it reproduces bit for bit. The parameter-independent carrier
(``-log y``, ``0.5 log w``, ``-log(count!)``, ...) is transcendental and may
move by an ulp per row across CPUs and SIMD loops, so it is checked against
the stored carrier sum within the bound below rather than hashed. Legacy
``/v1`` identifiers also hashed the carrier and are never recomputed: they are
accepted on a matching family tag plus the carrier-sum bound (the weights are
still checked exactly against the stored weight root digest). Refits record
``/v2``.

Bound (Higham, *Accuracy and Stability*, 2nd ed., 2002, §§2.2, 3.1, 4.2).
Row ``i`` is ``c_i = -m_i (F_i + h)`` with ``F_i`` one transcendental
evaluation, ``m_i`` an exact multiplier and ``h`` a library constant. If every
platform's ``F_i`` and ``h`` lie within ``kappa u`` of exact, two platforms'
rows differ by at most ``gamma_{2 kappa + 8} a_i``, ``a_i = |c_i| + beta |m_i|``
(``2 kappa`` for the two evaluations, 4 for each platform's sum and product,
4 for the magnitude conversion), with ``beta`` ``2 |h|`` for a constant term,
1 for a function bounded in absolute error below one, else 0. Summation adds
``gamma_{n-1} sum |c_i|`` per platform, so

    |S_saved - S_live| <= gamma_k A + gamma_{n-1} (2 C + gamma_k A),

with ``A >= sum a_i`` and ``C >= sum |c_i|`` from the live rows, accumulated
upward. ``kappa`` is 8 for ``log`` (two ulps per implementation against
NumPy's one-ulp validation and NEP 38's 1-3 ulp allowance; Gladman et al.
2025) and 16 for ``gammaln`` (three times Cephes ``lgam``'s ``4.9 u`` peak on
``[0, 3]``, relative above magnitude one, absolute below; Moshier).
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from superglm._roundoff import UNIT_ROUNDOFF as _UNIT_ROUNDOFF

LOG_ERROR_UNITS = 8
"""``kappa`` for one ``log`` evaluation: four ulps, each at most ``2 u`` relative."""

LGAMMA_ERROR_UNITS = 16
"""``kappa`` for one ``gammaln`` evaluation, absolute below magnitude one."""


class CarrierRounding(NamedTuple):
    """What a built-in plan certifies about its platform-dependent carrier.

    ``legacy_identifier_tag`` names the plan's ``Family/v1`` tag (``None`` or
    absent if it never wrote one); identifiers under it are accepted on tag
    and carrier sum alone, so it must differ from the plan's own tags.
    """

    carrier: NDArray[np.float64]
    """The live per-row carrier."""

    scale: NDArray[np.float64]
    """Per-row ``a_i`` of the module docstring, rounded upward."""

    error_units: int
    """``kappa``: the transcendental's error in units of ``u``."""


def carrier_rounding_scale(
    carrier: NDArray,
    multiplier: NDArray | None = None,
    beta: float = 0.0,
) -> NDArray[np.float64]:
    """``a_i = |c_i| + beta |m_i|``, each operation rounded upward."""
    scale = np.abs(np.asarray(carrier, dtype=np.float64))
    if beta == 0.0:
        return scale
    if multiplier is None:
        raise ValueError("a carrier with a constant term needs its row multiplier")
    term = np.nextafter(np.abs(np.asarray(multiplier, dtype=np.float64)) * beta, np.inf)
    return np.nextafter(scale + term, np.inf)


def _up(value: float) -> float:
    return float(np.nextafter(value, math.inf))


def _gamma_upper(count: int) -> float:
    """Higham's ``gamma_count``, rounded upward."""
    product = count * _UNIT_ROUNDOFF
    if product >= 0.5:
        return math.inf
    denominator = float(np.nextafter(1.0 - product, 0.0))
    return _up(product / denominator)


def _sum_upper(values: NDArray[np.float64]) -> float:
    """An upper bound on the exact sum of nonnegative ``values``.

    Any order of ``n - 1`` additions returns ``S`` with
    ``|S - sum| <= gamma_{n-1} sum``, so ``sum <= S / (1 - gamma_{n-1})``.
    """
    total = float(np.sum(values, dtype=np.float64))
    denominator = float(np.nextafter(1.0 - _gamma_upper(max(len(values) - 1, 0)), 0.0))
    return _up(total / denominator)


def carrier_sum_bound(rounding: CarrierRounding) -> float:
    """Largest ``|S_saved - S_live|`` two faithful platforms can produce."""
    carrier = np.asarray(rounding.carrier, dtype=np.float64)
    n_rows = len(carrier)
    row_gamma = _gamma_upper(2 * rounding.error_units + 8)
    sum_gamma = _gamma_upper(max(n_rows - 1, 0))
    scale_sum = _sum_upper(np.asarray(rounding.scale, dtype=np.float64))
    magnitude_sum = _sum_upper(np.abs(carrier))
    row_term = _up(row_gamma * scale_sum)
    saved_magnitude = _up(_up(2.0 * magnitude_sum) + row_term)
    return _up(row_term + _up(sum_gamma * saved_magnitude))


def rebound_plan_mismatch(
    plan: object,
    stored_identifier: str,
    stored_carrier_sum: float,
) -> str | None:
    """Why ``plan`` is not the saved fit's likelihood plan, or ``None`` if it is."""
    live_identifier = plan.plan_identifier  # type: ignore[attr-defined]
    carrier_rounding = getattr(plan, "carrier_rounding", None)
    if carrier_rounding is None:
        # A plan without a certified carrier identifies itself exactly.
        return None if live_identifier == stored_identifier else "identifier"
    rounding = carrier_rounding()
    if not isinstance(rounding, CarrierRounding) or legacy_identifier_problem(plan) is not None:
        return "carrier certificate"
    tag = getattr(plan, "legacy_identifier_tag", None)
    if live_identifier != stored_identifier and (
        tag is None or not _is_legacy_identifier(stored_identifier, tag)
    ):
        return "identifier"
    if not isinstance(stored_carrier_sum, float) or not math.isfinite(stored_carrier_sum):
        return "carrier"
    live_sum = float(np.sum(np.asarray(rounding.carrier, dtype=np.float64), dtype=np.float64))
    bound = carrier_sum_bound(rounding)
    difference = abs(live_sum - stored_carrier_sum)
    # One step up covers the rounding of the difference itself.
    if not math.isfinite(difference) or not difference <= _up(bound):
        return "carrier"
    return None


def legacy_identifier_problem(plan: object) -> str | None:
    """Why ``plan``'s ``legacy_identifier_tag`` cannot be trusted, or ``None``.

    It reads only the tag and the plan identifier, both strings, so a fit can
    check it without building the carrier.
    """
    tag = getattr(plan, "legacy_identifier_tag", None)
    if tag is None:
        return None
    live_identifier = plan.plan_identifier  # type: ignore[attr-defined]
    if not isinstance(tag, str) or not tag or live_identifier.startswith(f"{tag}:"):
        return (
            "family likelihood plan legacy_identifier_tag must be None or a nonempty tag "
            "that its own identifiers do not carry; use None for a family with no legacy "
            "identifiers"
        )
    return None


def _is_legacy_identifier(identifier: str, tag: str) -> bool:
    prefix = f"{tag}:"
    if not isinstance(identifier, str) or not identifier.startswith(prefix):
        return False
    digest = identifier[len(prefix) :]
    return len(digest) == 64 and all(char in "0123456789abcdef" for char in digest)


__all__ = [
    "LGAMMA_ERROR_UNITS",
    "LOG_ERROR_UNITS",
    "CarrierRounding",
    "carrier_rounding_scale",
    "carrier_sum_bound",
    "legacy_identifier_problem",
    "rebound_plan_mismatch",
]
