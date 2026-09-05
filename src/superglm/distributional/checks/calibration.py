"""Calibration tables for a fitted distributional model.

Four questions a pricing or forecasting review asks of a predictive
distribution, and one table each.

* **Does the interval hold?**  The realised coverage of the central predictive
  interval at each level, overall and by decile of each predicted parameter.
  Calibration in the sense of Gneiting, Balabdaoui and Raftery (2007), *Journal
  of the Royal Statistical Society: Series B* 69(2), 243-268.
* **Does the tail hold?**  Expected ``sum P(Y > t)`` against the realised count
  of exceedances, with the Poisson-binomial standard error and the log score of
  the binary event, overall and by decile of the predicted exceedance
  probability.
* **Do the quantiles hold?**  The realised exceedance rate of each predicted
  ``p``-quantile against ``1 - p``.
* **Does the total hold?**  :func:`actual_expected_check` reports, per bin or
  level of a covariate, the realised total ``sum w y`` against the predicted
  total ``sum w mu_hat`` and their ratio.  This is a **ratio of sums**, computed
  through :func:`~superglm.distributional.checks._aggregate.grouped_ratio`: for a
  burn-cost target with exposure weights it is total cost over total expected
  cost, never the mean of per-row ratios.

Where a threshold is given, :func:`reliability_curve` adds the CORP reliability
diagram of Dimitriadis, Gneiting and Jordan (2021), *Proceedings of the National
Academy of Sciences* 118(8), e2016191118: the exceedance forecast is
(re)calibrated by isotonic regression through the pool-adjacent-violators
algorithm (Ayer, Brunk, Ewing, Reid and Silverman 1955, *Annals of Mathematical
Statistics* 26(4), 641-647; de Leeuw, Hornik and Mair 2009, *Journal of
Statistical Software* 32(5)), which chooses the binning rather than taking one
from the analyst.  The bands are the paper's **consistency** bands: each
resample draws fresh events from the original forecast probabilities, so the
band shows how far a diagram can stray while the forecast is in fact calibrated,
and is positioned about the diagonal.

**Interval coverage and quantile calibration read the randomised PIT whenever
the family has an atom.**  On the response, a point mass makes both tables
answer the wrong question: every central interval whose lower quantile has
landed on the atom contains the atom, so a zero response counts as covered, and
every predicted ``p``-quantile below the mass is zero, so the realised
exceedance is flat at ``1 - P(Y = 0)`` for each such ``p`` however large ``p``
is.  On an 83 per cent zero book that reads 0.85 at the nominal 50 per cent
level.  The randomised probability integral transform
``u ~ U(F(y-), F(y))`` of Dunn and Smyth (1996) is uniform under the model
whatever the atom does, so "covered at level ``l``" becomes
``(1 - l) / 2 <= u <= 1 - (1 - l) / 2`` and "exceeds the ``p``-quantile"
becomes ``u > p``, both exact.  ``CalibrationPayload.calibration_law`` names
which of the two a table was built on; tail exceedance and actual-versus-
expected compare totals rather than positions and are unaffected.

Weights are read under the model's declared contract exactly as the fit reads
them.  A prior weight is part of the row's own law, so every distribution
function and quantile here is the prior-weighted one; a family that cannot
express its weighted law refuses rather than quietly inverting the unit-weight
distribution.  A frequency weight is replication, so it multiplies each row's
contribution to a total and its variance.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from superglm._frame import EagerFrame, FrameLike, as_eager_frame
from superglm.distributional.checks._aggregate import grouped_ratio

# One binning rule for the whole suite: an actual-versus-expected table and a
# binned residual check must cut a covariate the same way or they cannot be read
# side by side.
from superglm.distributional.checks.binned import _covariate_bins
from superglm.distributional.family import (
    AtomFamily,
    DefaultPredictionFamily,
    DistributionFunctionFamily,
    PriorWeightedDistributionFunctionFamily,
    PriorWeightedVarianceFamily,
    VarianceFamily,
)

# The response and offset shapes are checked before zero-weight rows are
# selected away, exactly as the fit and the residuals check them.
from superglm.distributional.model import (
    _take_unvalidated_offsets,
    _unvalidated_offset_shapes,
    _unvalidated_response_shape,
)
from superglm.distributional.posterior import posterior_predictive
from superglm.distributional.residuals import ResidualSet, compute_residuals, replication_sample

# ``_equal_count_bins`` splits rows by rank rather than by value, which is what
# keeps a decile table ten rows wide even for a parameter the model holds
# constant; ``SCHEMA_VERSION`` and the JSON helpers are the suite's payload
# conventions.
from superglm.distributional.surfaces import (
    SCHEMA_VERSION,
    _equal_count_bins,
    _json_frame,
    _json_vector,
    _readonly,
)
from superglm.distributional.weights import resolve_likelihood_weights

#: Groups a decile table cuts rows into.
_DECILES = 10
#: Percentiles of the resampling distribution reported as a band.
_BAND_PERCENTILES = (0.025, 0.975)
#: A probability is pulled inside this closed interval before a log score.
_LOG_SCORE_FLOOR = 1.0e-15
#: Default grid for quantile calibration.
_QUANTILE_GRID = (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
#: Default interval levels for coverage.
_LEVELS = (0.5, 0.8, 0.9, 0.95, 0.99)

#: Which law interval coverage and quantile calibration were read on.
CalibrationLaw = Literal["response", "randomised_pit"]


# --------------------------------------------------------------------------- #
# Payloads
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ActualExpected:
    """Realised and predicted totals per bin or level of one covariate.

    ``actual`` is ``sum w y``, ``expected`` is ``sum w mu_hat`` and ``ratio`` is
    the first over the second -- a ratio of sums, so a bin's number is total
    cost over total expected cost and a one-day policy does not get the say of a
    one-year one.  ``ratio_se`` divides the standard deviation of the realised
    total by the predicted total; ``variance_law`` names where each row's
    variance came from:

    ``"family"``
        the family's own ``variance(theta)``, and the row law is the unit law.
    ``"family_prior_weighted"``
        the family's own ``variance_prior_weighted(theta, weights)``, evaluated
        on each row's prior-weighted law.
    ``"family_unit_law"``
        the family's ``variance(theta)`` while the rows carry non-unit prior
        weights, so the number is the unit-weight variance.
    ``"prior_weighted_draws"``
        plug-in predictive draws on the row's own prior-weighted law.
    ``"draws"``
        plug-in predictive draws on the unit law, which is the row's own law.
    ``"unit_law_draws"``
        plug-in predictive draws on the unit law while the rows carry non-unit
        prior weights, because the family cannot express its weighted law.
    """

    covariate: str
    edges: NDArray[np.float64] | None
    levels: tuple[str, ...] | None
    centers: NDArray[np.float64]
    n: NDArray[np.int64]
    weight: NDArray[np.float64]
    actual: NDArray[np.float64]
    expected: NDArray[np.float64]
    ratio: NDArray[np.float64]
    ratio_se: NDArray[np.float64]
    variance_law: str
    weight_semantics: str
    kind: str = "actual_expected"
    schema_version: int = SCHEMA_VERSION

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "schema_version": int(self.schema_version),
            "covariate": str(self.covariate),
            "edges": None if self.edges is None else _json_vector(self.edges),
            "levels": None if self.levels is None else [str(level) for level in self.levels],
            "centers": _json_vector(self.centers),
            "n": _json_vector(self.n),
            "weight": _json_vector(self.weight),
            "actual": _json_vector(self.actual),
            "expected": _json_vector(self.expected),
            "ratio": _json_vector(self.ratio),
            "ratio_se": _json_vector(self.ratio_se),
            "variance_law": str(self.variance_law),
            "weight_semantics": str(self.weight_semantics),
        }


@dataclass(frozen=True)
class ReliabilityCurve:
    """The CORP reliability diagram of one binary forecast.

    ``x`` holds the unique forecast probabilities in increasing order and
    ``calibrated`` the isotonic (PAV) estimate of the conditional event
    probability at each of them, with ``count`` rows behind each value.
    ``lower`` and ``upper`` are the 95 % pointwise consistency band: the
    resampled diagram under the assumption that the original forecast is
    calibrated, so a curve leaving the band is evidence of miscalibration.
    """

    x: NDArray[np.float64]
    calibrated: NDArray[np.float64]
    count: NDArray[np.int64]
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    n_boot: int
    seed: int
    kind: str = "reliability"
    schema_version: int = SCHEMA_VERSION

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "schema_version": int(self.schema_version),
            "x": _json_vector(self.x),
            "calibrated": _json_vector(self.calibrated),
            "count": _json_vector(self.count),
            "lower": _json_vector(self.lower),
            "upper": _json_vector(self.upper),
            "n_boot": int(self.n_boot),
            "seed": int(self.seed),
        }


@dataclass(frozen=True)
class CalibrationPayload:
    """Interval coverage, tail exceedance, quantile calibration and reliability.

    ``calibration_law`` names the law the ``coverage`` and ``quantiles`` tables
    were read on:

    ``"response"``
        the response against the predicted quantiles, which is exact for a
        family with no point mass.
    ``"randomised_pit"``
        the randomised probability integral transform of ``residuals``, which
        is what a family with an atom needs -- see the module docstring.

    ``tails`` and :class:`ActualExpected` compare totals and are the same under
    either law.
    """

    coverage: pd.DataFrame
    tails: pd.DataFrame
    quantiles: pd.DataFrame
    reliability: Mapping[float, ReliabilityCurve]
    levels: tuple[float, ...]
    thresholds: tuple[float, ...]
    quantile_grid: tuple[float, ...]
    n_rows: int
    weight_semantics: str
    calibration_law: CalibrationLaw
    seed: int
    kind: str = "calibration"
    schema_version: int = SCHEMA_VERSION

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "schema_version": int(self.schema_version),
            "coverage": _json_frame(self.coverage),
            "tails": _json_frame(self.tails),
            "quantiles": _json_frame(self.quantiles),
            "reliability": {
                str(float(threshold)): curve.to_json()
                for threshold, curve in self.reliability.items()
            },
            "levels": [float(value) for value in self.levels],
            "thresholds": [float(value) for value in self.thresholds],
            "quantile_grid": [float(value) for value in self.quantile_grid],
            "n_rows": int(self.n_rows),
            "weight_semantics": str(self.weight_semantics),
            "calibration_law": str(self.calibration_law),
            "seed": int(self.seed),
        }


# --------------------------------------------------------------------------- #
# Isotonic regression and the CORP reliability diagram
# --------------------------------------------------------------------------- #


def _pava(values: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the weighted isotonic (non-decreasing) least-squares fit of ``values``.

    The pool-adjacent-violators algorithm: walk left to right, and whenever the
    new block falls below its left neighbour merge the two into their weighted
    mean and repeat.  Ayer et al. (1955); de Leeuw, Hornik and Mair (2009).
    """
    size = len(values)
    level = np.empty(size, dtype=np.float64)
    mass = np.empty(size, dtype=np.float64)
    width = np.empty(size, dtype=np.intp)
    blocks = 0
    for index in range(size):
        level[blocks] = values[index]
        mass[blocks] = weights[index]
        width[blocks] = 1
        blocks += 1
        while blocks > 1 and level[blocks - 1] < level[blocks - 2]:
            total = mass[blocks - 2] + mass[blocks - 1]
            level[blocks - 2] = (
                mass[blocks - 2] * level[blocks - 2] + mass[blocks - 1] * level[blocks - 1]
            ) / total
            mass[blocks - 2] = total
            width[blocks - 2] += width[blocks - 1]
            blocks -= 1
    return np.repeat(level[:blocks], width[:blocks])


def _validated_events(probability: NDArray, event: NDArray) -> tuple[NDArray, NDArray]:
    forecast = np.asarray(probability, dtype=np.float64)
    if forecast.ndim != 1 or len(forecast) < 1:
        raise ValueError("a reliability curve needs at least one row")
    outcome = np.asarray(event, dtype=np.float64)
    if outcome.shape != forecast.shape:
        raise ValueError("a reliability curve needs one event per forecast probability")
    if not np.all(np.isfinite(forecast)) or np.any(forecast < 0.0) or np.any(forecast > 1.0):
        raise ValueError("forecast probabilities must be finite and inside [0, 1]")
    if not np.all((outcome == 0.0) | (outcome == 1.0)):
        raise ValueError("events must be zero or one")
    return forecast, outcome


def reliability_curve(
    probability: NDArray,
    event: NDArray,
    *,
    n_boot: int = 200,
    seed: int = 42,
) -> ReliabilityCurve:
    """Return the CORP reliability diagram of a binary forecast.

    The conditional event probability is estimated by isotonic regression of the
    events on the forecast values, which pools tied forecasts and then chooses
    its own bins through the PAV algorithm (Dimitriadis, Gneiting and Jordan
    2021).  The consistency band resamples ``n_boot`` sets of events drawn from
    the original forecast probabilities and reads the 2.5th and 97.5th
    percentiles of the resampled diagrams, so it answers "how far would a
    calibrated forecast's diagram wander here".
    """
    boot = int(n_boot)
    if boot < 1:
        raise ValueError("n_boot must be at least one resample")
    forecast, outcome = _validated_events(probability, event)

    order = np.argsort(forecast, kind="stable")
    sorted_forecast = forecast[order]
    unique, starts, counts = np.unique(sorted_forecast, return_index=True, return_counts=True)
    mass = counts.astype(np.float64)
    observed = np.add.reduceat(outcome[order], starts) / mass
    calibrated = _pava(observed, mass)

    generator = np.random.default_rng(seed)
    resampled = np.empty((boot, len(unique)), dtype=np.float64)
    draws = (generator.uniform(size=(boot, len(forecast))) < sorted_forecast).astype(np.float64)
    pooled = np.add.reduceat(draws, starts, axis=1) / mass
    for index in range(boot):
        resampled[index] = _pava(pooled[index], mass)
    lower, upper = np.quantile(resampled, _BAND_PERCENTILES, axis=0)

    return ReliabilityCurve(
        x=_readonly(unique.astype(np.float64)),
        calibrated=_readonly(calibrated),
        count=_readonly(counts.astype(np.int64)),
        lower=_readonly(lower),
        upper=_readonly(upper),
        n_boot=boot,
        seed=int(seed),
    )


# --------------------------------------------------------------------------- #
# Rows, weights and the row law
# --------------------------------------------------------------------------- #


class _Rows(NamedTuple):
    """The retained rows of a call, with the two kinds of weight kept apart."""

    frame: EagerFrame
    response: NDArray[np.float64]
    offsets: Mapping[str, NDArray] | None
    weights: NDArray[np.float64]
    prior_law: NDArray[np.float64] | None
    semantics: str
    positions: NDArray[np.intp]


def _retained_rows(
    fitted: Any,
    X: FrameLike | EagerFrame,
    y: NDArray,
    *,
    sample_weight: NDArray | None,
    offsets: Mapping[str, NDArray] | None,
) -> _Rows:
    """Resolve weights and drop the zero-weight rows, as the fit itself does."""
    frame = as_eager_frame(X)
    n_observations = len(frame)
    response = _unvalidated_response_shape(y, n_observations)
    contract = fitted.fit_state.weight_contract
    resolved = resolve_likelihood_weights(
        sample_weight, n_observations=n_observations, contract=contract
    )
    positions = np.asarray(resolved.input_positions, dtype=np.intp)
    resolved_offsets = offsets
    if len(positions) != n_observations:
        frame = as_eager_frame(frame.take_rows(positions))
        response = response[positions]
        resolved_offsets = _take_unvalidated_offsets(
            _unvalidated_offset_shapes(offsets, n_observations), positions
        )
    weights = np.asarray(resolved.values, dtype=np.float64)
    prior = contract.semantics == "prior"
    return _Rows(
        frame=frame,
        response=np.asarray(response, dtype=np.float64),
        offsets=resolved_offsets,
        weights=weights,
        prior_law=None if not prior or resolved.provenance.all_unit else weights,
        semantics=contract.semantics,
        positions=positions,
    )


def _row_law(family: Any, prior_law: NDArray | None) -> tuple[Any, Any]:
    """Return the ``(cdf, quantile)`` pair of the rows' own law.

    Under non-unit prior weights the weight is part of the distribution, so the
    prior-weighted pair is the only correct one; a family without it refuses
    rather than reading the unit-weight functions.
    """
    if not isinstance(family, DistributionFunctionFamily):
        raise NotImplementedError(
            "calibration needs a family with a distribution function; this one has none"
        )
    if prior_law is None:
        return family.cdf, family.quantile
    if not isinstance(family, PriorWeightedDistributionFunctionFamily):
        raise NotImplementedError(
            f"{type(family).__name__} has no prior-weighted distribution function, so the row "
            "law under non-unit prior weights is unavailable; fit with unit weights, declare "
            "frequency semantics, or implement PriorWeightedDistributionFunctionFamily"
        )

    def weighted_cdf(y: NDArray, theta: NDArray) -> NDArray:
        return family.cdf_prior_weighted(y, theta, prior_law)

    def weighted_quantile(p: NDArray, theta: NDArray) -> NDArray:
        return family.quantile_prior_weighted(p, theta, prior_law)

    return weighted_cdf, weighted_quantile


def _validated_probability(name: str, value: float) -> float:
    number = float(value)
    if not 0.0 < number < 1.0:
        raise ValueError(f"{name} must lie strictly inside (0, 1)")
    return number


def _binomial_error(rate: NDArray[np.float64], exposure: NDArray[np.float64]) -> NDArray:
    error = np.full(len(rate), np.nan, dtype=np.float64)
    np.divide(rate * (1.0 - rate), exposure, out=error, where=exposure > 0.0)
    return np.sqrt(error)


# --------------------------------------------------------------------------- #
# Actual versus expected
# --------------------------------------------------------------------------- #


def _row_variance(
    fitted: Any,
    rows: _Rows,
    theta: NDArray[np.float64],
    *,
    n_draws: int,
    seed: int,
) -> tuple[NDArray[np.float64], str]:
    """Return each row's predictive variance and the law it was read from.

    Under non-unit prior weights the weight is inside the row's law, so the
    order of preference is the family's own weighted variance, then simulation
    on the weighted law, and only then the unit-weight variance -- which
    ignores the weight altogether and is reported as such.
    """
    family = fitted.family
    prior_law = rows.prior_law
    if prior_law is not None:
        if isinstance(family, PriorWeightedVarianceFamily):
            weighted = family.variance_prior_weighted(theta, prior_law)
            return np.asarray(weighted, dtype=np.float64), "family_prior_weighted"
        if isinstance(family, PriorWeightedDistributionFunctionFamily):
            law = "prior_weighted_draws"
        elif isinstance(family, VarianceFamily):
            return np.asarray(family.variance(theta), dtype=np.float64), "family_unit_law"
        else:
            law = "unit_law_draws"
            prior_law = None
    elif isinstance(family, VarianceFamily):
        return np.asarray(family.variance(theta), dtype=np.float64), "family"
    else:
        law = "draws"
    draws = posterior_predictive(
        fitted,
        rows.frame,
        n_draws,
        parameter_uncertainty=False,
        offsets=rows.offsets,
        seed=seed,
        weights=prior_law,
    )
    return np.var(draws, axis=0, ddof=1), law


def actual_expected_check(
    fitted: Any,
    X: FrameLike | EagerFrame,
    y: NDArray,
    covariate: NDArray,
    *,
    name: str,
    sample_weight: NDArray | None = None,
    n_bins: int = 20,
    offsets: Mapping[str, NDArray] | None = None,
    n_draws: int = 200,
    seed: int = 42,
) -> ActualExpected:
    """Return realised against predicted totals per bin or level of ``covariate``.

    ``covariate`` gives one value per row of ``X``.  Every group's number is a
    ratio of weighted sums: ``sum w y`` over ``sum w mu_hat``, so with exposure
    weights on a rate target the table reads as total cost over total expected
    cost.  The standard error is that of the realised total under the fitted
    law -- ``sum w^2 Var(Y)`` under prior weights, where the weight sits inside
    each row's law, and ``sum w Var(Y)`` under frequency weights, where a weight
    of ``w`` is ``w`` independent rows -- over the magnitude of the predicted
    total, so ``|ratio - 1| <= 3 ratio_se`` is exactly ``|actual - expected|``
    within three standard deviations whatever sign the total carries.
    """
    family = fitted.family
    if not isinstance(family, DefaultPredictionFamily):
        raise NotImplementedError(
            "an actual-versus-expected table needs a family that names a default prediction; "
            "this one has none"
        )
    grouped = np.asarray(covariate)
    if grouped.ndim != 1 or len(grouped) != len(as_eager_frame(X)):
        raise ValueError("covariate must give one value per row of X")

    rows = _retained_rows(fitted, X, y, sample_weight=sample_weight, offsets=offsets)
    grouped = grouped[rows.positions]
    binning = _covariate_bins(grouped, n_bins)
    width = binning.n_groups

    theta = np.asarray(
        fitted.predict_parameters(rows.frame, offsets=rows.offsets), dtype=np.float64
    )
    mean = np.asarray(family.default_prediction(theta), dtype=np.float64)
    weights = rows.weights

    actual, exposure, _ = grouped_ratio(
        weights * rows.response, weights, binning.codes, n_groups=width
    )
    _, expected, ratio = grouped_ratio(
        weights * rows.response, weights * mean, binning.codes, n_groups=width
    )

    variance, variance_law = _row_variance(
        fitted, rows, theta, n_draws=int(n_draws), seed=int(seed)
    )
    scale = weights * weights if rows.semantics == "prior" else weights
    total_variance = np.bincount(binning.codes, weights=scale * variance, minlength=width)
    error = np.full(width, np.nan, dtype=np.float64)
    magnitude = np.abs(expected)
    np.divide(np.sqrt(total_variance), magnitude, out=error, where=magnitude > 0.0)

    return ActualExpected(
        covariate=str(name),
        edges=None if binning.edges is None else _readonly(binning.edges),
        levels=binning.levels,
        centers=_readonly(binning.centers),
        n=_readonly(np.bincount(binning.codes, minlength=width).astype(np.int64)),
        weight=_readonly(exposure),
        actual=_readonly(actual),
        expected=_readonly(expected),
        ratio=_readonly(ratio),
        ratio_se=_readonly(error),
        variance_law=variance_law,
        weight_semantics=rows.semantics,
    )


# --------------------------------------------------------------------------- #
# Calibration tables
# --------------------------------------------------------------------------- #


class _Grouping(NamedTuple):
    """One group code per row beside the label of every group."""

    labels: tuple[str, ...]
    codes: NDArray[np.intp]


def _overall(rows: int) -> _Grouping:
    return _Grouping(("all",), np.zeros(rows, dtype=np.intp))


def _decile_grouping(values: NDArray, name: str) -> _Grouping:
    codes, _ = _equal_count_bins(np.asarray(values, dtype=np.float64), _DECILES)
    return _Grouping(
        tuple(f"{name}:decile {index}" for index in range(1, _DECILES + 1)),
        np.asarray(codes, dtype=np.intp),
    )


def calibration_payload(
    fitted: Any,
    X: FrameLike | EagerFrame,
    y: NDArray,
    *,
    residuals: ResidualSet | None = None,
    levels: Sequence[float] = _LEVELS,
    thresholds: Sequence[float] = (),
    quantile_grid: Sequence[float] = _QUANTILE_GRID,
    by_parameter_deciles: bool = True,
    sample_weight: NDArray | None = None,
    offsets: Mapping[str, NDArray] | None = None,
    seed: int = 42,
) -> CalibrationPayload:
    """Return the coverage, tail, quantile-calibration and reliability tables.

    ``by_parameter_deciles`` adds, beside the overall row, one row per decile of
    each predicted parameter to the coverage table and one per decile of the
    predicted exceedance probability to the tail table.  Passing ``residuals``
    reuses an already-computed residual payload and spares a second parameter
    prediction; it is checked against the rows this call resolved, because
    residuals from another model or another sample would silently make every
    table below a statement about the wrong fit.

    Coverage and quantile calibration are read on the randomised PIT of
    ``residuals`` when the family declares an atom (or when the residuals
    carry randomised rows), and on the response otherwise; the payload's
    ``calibration_law`` says which, and the module docstring says why.
    """
    asked_levels = tuple(_validated_probability("level", value) for value in levels)
    asked_grid = tuple(_validated_probability("p", value) for value in quantile_grid)
    asked_thresholds = tuple(float(value) for value in thresholds)

    rows = _retained_rows(fitted, X, y, sample_weight=sample_weight, offsets=offsets)
    if residuals is None:
        residuals = compute_residuals(
            fitted, X, y, sample_weight=sample_weight, offsets=offsets, seed=seed
        )
    elif not isinstance(residuals, ResidualSet):
        raise TypeError("residuals must be a ResidualSet")
    if residuals.n_rows != len(rows.positions) or not np.array_equal(residuals.y, rows.response):
        raise ValueError("residuals must come from the same rows as X and y")

    family = fitted.family
    cdf, quantile = _row_law(family, rows.prior_law)
    theta = residuals.theta
    response = rows.response
    # A family with a point mass answers "is the response inside this interval"
    # and "does it exceed this quantile" on the randomised PIT, which is uniform
    # under the model whatever the atom does; anything else reads the atom as
    # covered by every interval that touches it.  The declared atom decides,
    # not the sample, so a book that happens to carry no atom row still reads
    # the same law -- there the two agree row for row anyway.
    randomised = isinstance(family, AtomFamily) or residuals.randomised_rows > 0
    law: CalibrationLaw = "randomised_pit" if randomised else "response"
    transform = residuals.pit
    # Coverage, exceedance and quantile indicators are per-row facts rather than
    # rates, so they are weighted by replication only: ones under the prior
    # contract, where the declared weight is already inside the row's own law,
    # and the declared counts under the frequency contract.
    weights = residuals.weights
    n_rows = len(response)

    groupings = [_overall(n_rows)]
    if by_parameter_deciles:
        groupings.extend(
            _decile_grouping(theta[:, index], spec.name)
            for index, spec in enumerate(family.parameters)
        )

    coverage: dict[str, list[Any]] = {
        key: [] for key in ("level", "group", "n", "weight", "realised", "se")
    }
    for level in asked_levels:
        margin = 0.5 * (1.0 - level)
        if randomised:
            covered = ((transform >= margin) & (transform <= 1.0 - margin)).astype(np.float64)
        else:
            lower = np.asarray(quantile(np.full(n_rows, margin), theta), dtype=np.float64)
            upper = np.asarray(quantile(np.full(n_rows, 1.0 - margin), theta), dtype=np.float64)
            covered = ((response >= lower) & (response <= upper)).astype(np.float64)
        for grouping in groupings:
            width = len(grouping.labels)
            _, exposure, realised = grouped_ratio(
                weights * covered, weights, grouping.codes, n_groups=width
            )
            counts = np.bincount(grouping.codes, minlength=width).astype(np.int64)
            error = _binomial_error(realised, exposure)
            coverage["level"].extend([level] * width)
            coverage["group"].extend(grouping.labels)
            coverage["n"].extend(int(value) for value in counts)
            coverage["weight"].extend(float(value) for value in exposure)
            coverage["realised"].extend(float(value) for value in realised)
            coverage["se"].extend(float(value) for value in error)

    tails: dict[str, list[Any]] = {
        key: []
        for key in ("threshold", "group", "n", "weight", "expected", "realised", "se", "log_score")
    }
    reliability: dict[float, ReliabilityCurve] = {}
    replication = replication_sample(residuals, seed=seed)
    for threshold in asked_thresholds:
        exceedance = 1.0 - np.asarray(cdf(np.full(n_rows, threshold), theta), dtype=np.float64)
        event = (response > threshold).astype(np.float64)
        clipped = np.clip(exceedance, _LOG_SCORE_FLOOR, 1.0 - _LOG_SCORE_FLOOR)
        score = -(event * np.log(clipped) + (1.0 - event) * np.log1p(-clipped))
        tail_groupings = list(groupings[:1])
        if by_parameter_deciles:
            tail_groupings.append(_decile_grouping(exceedance, "exceedance"))
        for grouping in tail_groupings:
            width = len(grouping.labels)
            realised, expected, _ = grouped_ratio(
                weights * event, weights * exceedance, grouping.codes, n_groups=width
            )
            _, exposure, mean_score = grouped_ratio(
                weights * score, weights, grouping.codes, n_groups=width
            )
            counts = np.bincount(grouping.codes, minlength=width).astype(np.int64)
            error = np.sqrt(
                np.bincount(
                    grouping.codes,
                    weights=weights * exceedance * (1.0 - exceedance),
                    minlength=width,
                )
            )
            tails["threshold"].extend([threshold] * width)
            tails["group"].extend(grouping.labels)
            tails["n"].extend(int(value) for value in counts)
            tails["weight"].extend(float(value) for value in exposure)
            tails["expected"].extend(float(value) for value in expected)
            tails["realised"].extend(float(value) for value in realised)
            tails["se"].extend(float(value) for value in error)
            tails["log_score"].extend(float(value) for value in mean_score)
        reliability[threshold] = reliability_curve(
            exceedance[replication], event[replication], seed=seed
        )

    quantiles: dict[str, list[Any]] = {
        key: [] for key in ("p", "n", "weight", "expected", "realised_exceedance", "se")
    }
    codes = np.zeros(n_rows, dtype=np.intp)
    for probability in asked_grid:
        if randomised:
            exceeded = (transform > probability).astype(np.float64)
        else:
            predicted = np.asarray(quantile(np.full(n_rows, probability), theta), dtype=np.float64)
            exceeded = (response > predicted).astype(np.float64)
        _, exposure, realised = grouped_ratio(weights * exceeded, weights, codes, n_groups=1)
        quantiles["p"].append(probability)
        quantiles["n"].append(n_rows)
        quantiles["weight"].append(float(exposure[0]))
        quantiles["expected"].append(1.0 - probability)
        quantiles["realised_exceedance"].append(float(realised[0]))
        quantiles["se"].append(float(_binomial_error(realised, exposure)[0]))

    return CalibrationPayload(
        coverage=pd.DataFrame(coverage),
        tails=pd.DataFrame(tails),
        quantiles=pd.DataFrame(quantiles),
        reliability=MappingProxyType(reliability),
        levels=asked_levels,
        thresholds=asked_thresholds,
        quantile_grid=asked_grid,
        n_rows=n_rows,
        weight_semantics=rows.semantics,
        calibration_law=law,
        seed=int(seed),
    )


__all__ = [
    "ActualExpected",
    "CalibrationLaw",
    "CalibrationPayload",
    "ReliabilityCurve",
    "VarianceFamily",
    "actual_expected_check",
    "calibration_payload",
    "reliability_curve",
]
