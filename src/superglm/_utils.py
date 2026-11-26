"""Shared private array helpers used across superglm submodules."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def declared(obj: object, name: str, default: object = None) -> object:
    """``obj.name`` where ``obj``'s class declares ``name``, else ``default``.

    A family or link declares a fact on its class, so a subclass inherits it.
    The class is the one ``isinstance`` reads (``obj.__class__``): a proxy that
    presents the class it wraps keeps that class's facts, and an object that
    only forwards attribute reads (``__getattr__``) declares none.
    """
    return getattr(obj, name) if hasattr(obj.__class__, name) else default


def _validate_strict_prior_weights(weights, n: int) -> NDArray:
    """Return one-dimensional, finite, strictly positive float64 prior weights."""
    message = f"weights must be finite and strictly positive, one-dimensional, and have length {n}"
    try:
        raw = np.asarray(weights)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if (
        raw.ndim != 1
        or raw.shape[0] != n
        or np.iscomplexobj(raw)
        or getattr(raw.dtype, "kind", None) in {"M", "m"}
    ):
        raise ValueError(message)
    try:
        validated = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.all(np.isfinite(validated)) or np.any(validated <= 0.0):
        raise ValueError(message)
    return validated


def contract_weights(
    sample_weight,
    n: int,
    *,
    family=None,
    weight_semantics=None,
    name: str = "sample_weight",
    finite_sum: bool = False,
) -> NDArray:
    """Return ``sample_weight`` as float64 row weights under the weight contract.

    ``None`` is unit weights. Strict positivity is a Tweedie density requirement
    under the prior contract only: the compound-Poisson normalizer carries
    ``log w``, so a zero prior weight is an unevaluable density. Read as a
    replication count the weight never enters that normalizer, and zero means
    what it means for every other family, a row that appears no times. Every
    other weight is finite and nonnegative, with at least one positive.
    ``finite_sum`` also requires the total to be finite.
    """
    from superglm.distributions import prior_weights_in_normalizer
    from superglm.model.input_validation import _finite_vector
    from superglm.solvers.dispersion import PRIOR_WEIGHTS

    if sample_weight is None:
        weights = np.ones(n, dtype=np.float64)
    elif prior_weights_in_normalizer(family) and weight_semantics == PRIOR_WEIGHTS:
        weights = _validate_strict_prior_weights(sample_weight, n)
    else:
        weights = _finite_vector(name, sample_weight, n)
        if np.any(weights < 0.0):
            raise ValueError(f"{name} must be nonnegative")
        if not np.any(weights > 0.0):
            raise ValueError(f"{name} must not be all zero")
    if finite_sum and not np.isfinite(np.sum(weights, dtype=np.float64)):
        raise ValueError(f"{name} must have a finite sum")
    return weights


def weighted_row_sum(weights, row_term: Callable[..., NDArray], *rows) -> float:
    """Return ``sum(w * row_term(*rows))`` with every zero-weight row contributing an exact 0.

    Zero-weight rows are excluded before calling ``row_term`` so an infinite
    unit deviance on an unused row (for example, a Gaussian identity mean or
    response past ``sqrt(DBL_MAX)``) cannot produce ``0 * inf = NaN`` and poison
    the sum. Nonzero rows keep their positions and exact products while zero
    rows hold ``+0.0``.
    """
    mass = np.asarray(weights, dtype=np.float64)
    carried = mass != 0.0
    if carried.all():
        return float(np.sum(weights * row_term(*rows)))
    terms = np.zeros_like(mass)
    terms[carried] = mass[carried] * row_term(
        *(np.asarray(row, dtype=np.float64)[carried] for row in rows)
    )
    return float(np.sum(terms))


def occurrence_rows(counts, cap: int, generator: np.random.Generator) -> NDArray[np.intp]:
    """Return the row of every literal occurrence, or of ``cap`` of them drawn without replacement.

    ``counts`` are exact non-negative integer occurrence counts per row.  A
    subset drawn without replacement from independent occurrences is itself
    independent, so a check reading it keeps the reference law of ``cap``
    independent trials; a draw with replacement would read some occurrences
    twice.  Up to ``cap`` the rows come back in literal order; above it, in
    the draw's random order, so copies of one row do not sit together and a
    check that splits tied values by position splits them at random.
    """
    # Partial sums of non-negative integers are exact below 2**53 and round
    # monotonically, so the float total reaches 2**53 exactly when the true one does.
    if float(np.sum(counts, dtype=np.float64)) >= 2.0**53:
        raise ValueError(
            "the frequency weights total 2**53 or more occurrences, which cannot be "
            "counted exactly; rescale them or declare prior weights"
        )
    occurrences = np.asarray(counts, dtype=np.int64)
    ends = np.cumsum(occurrences)
    total = int(ends[-1]) if len(ends) else 0
    if total <= cap:
        return np.repeat(np.arange(len(occurrences), dtype=np.intp), occurrences)
    chosen = generator.choice(total, size=cap, replace=False)
    return np.searchsorted(ends, chosen, side="right").astype(np.intp, copy=False)


def score_bins(scores, mass, n_bins: int) -> NDArray[np.intp]:
    """Return each row's equal-mass bin of ``scores``, with every tied score in one bin.

    Rows sharing a score form a tier, placed whole by the middle of its mass: in
    bin ``ceil(n_bins * F) - 1``, clipped to ``[0, n_bins - 1]``, where
    ``F = (mass below the tier + half the tier's mass) / total mass`` is the
    sample mid-distribution function (Parzen, "Quantile probability and
    statistical data modeling", Statistical Science 19, 2004). With unit mass
    ``F = (mean rank - 1/2) / n``, so ties take their mean rank, as under the
    default ``TIES=MEAN`` of SAS PROC RANK, whose ``GROUPS=`` "always assigns
    observations with the same value to the same group". A tier holding more
    than one bin's mass leaves bins empty and the bins uneven, as Stata's
    ``estat gof, group()`` also does. Distinct unit-mass scores fill bins whose
    sizes differ by at most one. Every table and plot that cuts rows by score
    bins through here.

    A tier whose mass ends on an edge has half its mass on the inner side, so
    rounding in the sums moves only a tier whose middle lies on an edge to
    within rounding, which fractional masses can reach. Integer masses with
    ``n_bins * total < 2**52`` are summed and compared exactly, so a frequency
    weight bins as its literal copies. A zero-mass tier sits at the mass below
    it: in the lower bin at an edge, and in bin 0 at the bottom. Masses are
    scaled by a power of two so the largest is below 1, which is exact, so
    large finite masses bin as their unit-scale equivalents; a non-finite
    mass is refused.
    """
    values = np.asarray(scores, dtype=np.float64)
    weights = np.asarray(mass, dtype=np.float64)
    # Mass is the second sort key, so a tier's sums are the same whatever
    # order its rows arrive in.
    order = np.lexsort((weights, values))
    ordered = values[order]
    # Dividing by 2**exponent of the largest mass leaves every mass below 1,
    # so the total stays below the row count and n_bins * (below + through)
    # cannot overflow; a power of two changes no rounding in the normal range.
    _, exponent = math.frexp(float(np.max(np.abs(weights), initial=0.0)))
    cumulative = np.concatenate(([0.0], np.cumsum(np.ldexp(weights[order], -exponent))))
    total = cumulative[-1]
    if not math.isfinite(total):
        raise ValueError(
            "score bins need finite masses, but a weight or exposure is infinite or NaN; "
            "pass finite weights and exposures"
        )
    codes = np.zeros(len(values), dtype=np.intp)
    if not total > 0.0:
        return codes
    below = cumulative[np.searchsorted(ordered, ordered, side="left")]
    through = cumulative[np.searchsorted(ordered, ordered, side="right")]
    # n_bins * F = n_bins * (below + through) / (2 * total), in one rounding.
    position = np.ceil(n_bins * (below + through) / (2.0 * total))
    codes[order] = np.clip(position - 1.0, 0, n_bins - 1).astype(np.intp)
    return codes


def bin_rows(codes: NDArray[np.intp]) -> tuple[NDArray[np.intp], list[NDArray[np.intp]]]:
    """Return the filled bins of ``codes`` in order, and the rows in each."""
    sizes = np.bincount(codes)
    filled = np.flatnonzero(sizes)
    order = np.argsort(codes, kind="stable")
    ends = np.cumsum(sizes[filled])
    return filled, [order[end - size : end] for end, size in zip(ends, sizes[filled], strict=True)]


def _ulp_size(values: NDArray) -> NDArray:
    """Return the local float64 spacing without overflowing at max-float."""
    magnitude = np.abs(np.asarray(values, dtype=np.float64))
    with np.errstate(over="ignore", invalid="ignore"):
        upward = np.nextafter(magnitude, np.inf) - magnitude
        downward = magnitude - np.nextafter(magnitude, 0.0)
    return np.where(np.isfinite(upward), np.maximum(upward, downward), downward)


def _explained_deviance(
    deviance: float,
    null_deviance: float,
    y: NDArray,
    null_mu: NDArray,
    weights: NDArray,
) -> float:
    """Return explained deviance with an ulp-scale exact-null convention.

    A mathematically constant response can acquire a positive null deviance because
    its floating-point weighted mean differs by a few ulps.  The null model is
    treated as exact only when the active response values are exactly constant and
    each agrees with its null prediction within eight local ulps.
    """
    if null_deviance <= 0.0:
        return 0.0

    observed = np.asarray(y, dtype=np.float64)
    baseline = np.asarray(null_mu, dtype=np.float64)
    sample_weight = np.asarray(weights, dtype=np.float64)
    if observed.shape != baseline.shape or observed.shape != sample_weight.shape:
        raise ValueError("explained-deviance arrays must have identical shapes")

    active = sample_weight > 0.0
    if not np.any(active):
        return 0.0
    active_observed = observed[active]
    if not np.all(active_observed == active_observed[0]):
        return float(1.0 - deviance / null_deviance)
    with np.errstate(over="ignore", invalid="ignore"):
        difference = np.abs(observed - baseline)
    tolerance = 8.0 * np.maximum(_ulp_size(observed), _ulp_size(baseline))
    numerically_exact = bool(
        np.all(np.isfinite(difference[active])) and np.all(difference[active] <= tolerance[active])
    )
    if numerically_exact:
        return 0.0
    return float(1.0 - deviance / null_deviance)
