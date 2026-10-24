"""Check that a rebound likelihood plan is the one a saved fit recorded.

A built-in plan identifier hashes the plan's inputs as stored: its family
and row-law tags, the resolved weight digest and, for every family except
``GaussianLS``, the response digest (``NegativeBinomialLS`` also hashes the
count digest). ``GaussianLS``'s plan stores no response, so its identifier
binds the weights and not the response. Those inputs are exact on every
platform, so a rebound plan reproduces the identifier bit for bit. The
parameter-independent carrier (``-log y``, ``0.5 log w``,
``-log(count!)`` and similar) is a transcendental function of those inputs.
IEEE 754 recommends but does not require a correctly rounded ``log``, and
NumPy dispatches different SIMD loops on different CPUs, so the carrier can
move by an ulp per row between machines. It is checked against the fit's
stored carrier sum within the bound below rather than hashed.

Identifiers written before this rule (tag ``/v1``) also hashed the carrier.
They cannot be recomputed portably, so they are never recomputed: such an
identifier is accepted when its family tag matches the rebound plan's and the
carrier sum passes the bound. The loader still checks the retained weights
exactly against the stored weight root digest. The retained response is tied
to such a fit only through the carrier sum: the artifact's response digest
binds the stored response to the manifest, not to the fit. The sum does not
change when response rows are reordered, does not depend on the response
where the carrier depends only on weights or constants (``GaussianLS``'s
``0.5 log w``, ``TwoPieceNormalLSS``'s ``-m 0.5 log(2 pi)``), and admits any
change within the bound. Refitting records a ``/v2`` identifier, which binds
the weights exactly and, for every family except ``GaussianLS``, the response
exactly too; a ``GaussianLS`` identifier of either version never binds the
response.

Bound. Write ``u = 2**-53`` and ``gamma_k = k u / (1 - k u)`` (Higham,
*Accuracy and Stability of Numerical Algorithms*, 2nd ed., 2002, sections 2.2
and 3.1). Row ``i`` of a carrier is ``c_i = -m_i (F_i + h)``, where ``F_i`` is
one transcendental evaluation of exact inputs, ``m_i`` an exact stored
multiplier and ``h`` a library constant such as ``0.5 log(2 pi)``. If every
platform's ``F_i`` and ``h`` lie within ``kappa u`` of the exact value
(relative, or absolute below magnitude one where the plan says so), then two
platforms' rows differ by at most ``gamma_{2 kappa + 8} a_i`` with
``a_i = |c_i| + beta |m_i|``. The ``2 kappa`` covers the two evaluations, 4 the
sum and product rounding on each platform, and 4 the conversion of
``|m_i| (|F_i| + |h|)`` to the live row magnitude and second-order terms.
``beta`` is ``2 |h|`` for a constant term, 1 for a function bounded in
absolute error below one, and 0 otherwise. Any order of ``n - 1`` additions
sums the rows with error at most ``gamma_{n-1} sum |c_i|`` (Higham, section
4.2), on either platform. Hence

    |S_saved - S_live| <= gamma_k A + gamma_{n-1} (2 C + gamma_k A),

with ``A >= sum a_i`` and ``C >= sum |c_i|`` taken from the live rows. Every
quantity is accumulated upward.

``kappa`` for ``log`` is 8: four ulps per implementation, at most ``2 u``
relative each. NumPy's own validation set requires float64 ``log`` within one
ulp of the correctly rounded value, NEP 38 lets a new SIMD loop lose at most
1-3 ulps against its predecessor, and the largest known binary64 ``log``
errors across thirteen libraries in Gladman, Innocente, Mather and
Zimmermann, *Accuracy of Mathematical Functions in Single, Double, Double
Extended, and Quadruple Precision* (2025), are below one ulp. ``kappa`` for
``gammaln`` is 16, three times the Cephes ``lgam`` peak error of
``5.4e-16`` (about ``4.9 u``) on ``[0, 3]``, relative above magnitude one and
absolute below it (Moshier, *Cephes Mathematical Library* documentation).
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

    The plan names the ``Family/v1`` tag whose stored identifiers hashed the
    carrier in its own ``legacy_identifier_tag`` attribute, so the tag can be
    checked without building the carrier. A plan without the attribute, or
    with ``None``, never wrote such identifiers. A stored identifier under
    that tag is accepted on its tag and carrier sum alone, so the tag must not
    be one the plan's own identifiers carry.
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
