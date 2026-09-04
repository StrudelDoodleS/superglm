"""Story-telling surfaces for a fitted distributional model.

Four payloads, all built on the posterior primitive of
:mod:`superglm.distributional.posterior`:

* :func:`risk_curves` sweeps one covariate and reports predicted quantiles with
  posterior bands -- the centile curve of Rigby and Stasinopoulos (2005),
  *Journal of the Royal Statistical Society: Series C* 54(3), 507-554, drawn in
  pricing clothing.
* :func:`density_fan` puts the whole conditional density on the same sweep,
  differencing the family's distribution function on a response grid.
* :func:`parameter_spread` reports the sharpness of the fitted parameters
  (Gneiting, Balabdaoui and Raftery 2007, *JRSS-B* 69(2), 243-268) and the
  spread of tail probability among rows the model prices identically.
* :func:`portfolio` summarises simulated total loss, overall and per segment.

Grouped means here are ratios of weighted sums through
:func:`superglm.distributional.checks._aggregate.grouped_ratio`, never means of
per-row ratios.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, NamedTuple, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from superglm._frame import EagerFrame, FrameLike, as_eager_frame
from superglm.distributional.checks._aggregate import grouped_ratio
from superglm.distributional.family import (
    DefaultPredictionFamily,
    DistributionFunctionFamily,
)
from superglm.distributional.posterior import (
    CovarianceKind,
    posterior_bounds,
    posterior_draws,
    posterior_predictive,
    resolve_quantity,
)

# ``_required_columns`` is the one reader of which columns a compiled predictor
# needs; a sweep frame that restated the rule would drift from the design it is
# built to feed.
from superglm.distributional.prediction_design import _required_columns

#: Payload schema version shared by the surface serializers.
SCHEMA_VERSION = 1
#: Bins in every sharpness histogram.
_HISTOGRAM_BINS = 30
#: Tail probability bounding the response grid of a density fan.
_DENSITY_TAIL = 0.001
#: Percentiles reported within an identically-priced band.
_SPREAD_PERCENTILES = (0.05, 0.95)
#: A differenced density below this multiple of its own scale is round-off.
_DENSITY_NEGATIVE_TOLERANCE = 1.0e-9


# --------------------------------------------------------------------------- #
# Plain-value helpers
# --------------------------------------------------------------------------- #


def _readonly(values: NDArray) -> NDArray:
    array = np.array(values, copy=True)
    array.setflags(write=False)
    return array


def _json_scalar(value: Any) -> float | int | bool | str | None:
    """Return one payload value as a JSON leaf; non-finite numbers become null."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, str):
        return str(value)
    if isinstance(value, int | np.integer):
        return int(value)
    number = float(value)
    return number if np.isfinite(number) else None


def _json_vector(values: Any) -> list[Any]:
    return [_json_scalar(value) for value in np.asarray(values).tolist()]


def _json_matrix(values: Any) -> list[list[Any]]:
    return [_json_vector(row) for row in np.asarray(values)]


def _json_frame(frame: pd.DataFrame) -> dict[str, list[Any]]:
    return {str(name): _json_vector(frame[name].to_numpy()) for name in frame.columns}


# --------------------------------------------------------------------------- #
# Payloads
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Histogram:
    """Counts over ``len(edges) - 1`` contiguous bins."""

    edges: NDArray[np.float64]
    counts: NDArray[np.int64]

    def to_json(self) -> dict[str, Any]:
        return {"edges": _json_vector(self.edges), "counts": _json_vector(self.counts)}


@dataclass(frozen=True)
class RiskCurves:
    """Predicted quantiles along one covariate with posterior bands.

    ``x`` is the plotting coordinate: the swept numeric values, or ``0 .. L-1``
    for the ``L`` levels named in ``levels``.  ``values`` is the plug-in
    quantile at the fitted coefficients and ``lower``/``upper`` the equal-tailed
    posterior band at ``level``, all shaped ``(len(quantiles), len(x))``.
    """

    covariate: str
    x: NDArray[np.float64]
    levels: tuple[str, ...] | None
    quantiles: tuple[float, ...]
    values: NDArray[np.float64]
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    reference: Mapping[str, Any]
    level: float
    n_draws: int
    seed: int
    covariance: str
    kind: str = "risk_curves"
    schema_version: int = SCHEMA_VERSION

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "schema_version": int(self.schema_version),
            "covariate": str(self.covariate),
            "x": _json_vector(self.x),
            "levels": None if self.levels is None else [str(level) for level in self.levels],
            "quantiles": [float(value) for value in self.quantiles],
            "values": _json_matrix(self.values),
            "lower": _json_matrix(self.lower),
            "upper": _json_matrix(self.upper),
            "reference": {str(key): _json_scalar(value) for key, value in self.reference.items()},
            "level": float(self.level),
            "n_draws": int(self.n_draws),
            "seed": int(self.seed),
            "covariance": str(self.covariance),
        }


@dataclass(frozen=True)
class DensityFan:
    """The conditional density on a response grid along one covariate sweep."""

    covariate: str
    x: NDArray[np.float64]
    levels: tuple[str, ...] | None
    y_grid: NDArray[np.float64]
    density: NDArray[np.float64]
    reference: Mapping[str, Any]
    kind: str = "density_fan"
    schema_version: int = SCHEMA_VERSION
    quantile_levels: tuple[float, ...] | None = None
    quantiles: NDArray[np.float64] | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "schema_version": int(self.schema_version),
            "covariate": str(self.covariate),
            "x": _json_vector(self.x),
            "levels": None if self.levels is None else [str(level) for level in self.levels],
            "y_grid": _json_vector(self.y_grid),
            "density": _json_matrix(self.density),
            "reference": {str(key): _json_scalar(value) for key, value in self.reference.items()},
            "quantile_levels": None
            if self.quantile_levels is None
            else [float(level) for level in self.quantile_levels],
            "quantiles": None if self.quantiles is None else _json_matrix(self.quantiles),
        }


@dataclass(frozen=True)
class Spread:
    """Sharpness of the fitted parameters and spread among identical prices."""

    parameters: Mapping[str, Histogram]
    tail_quantile: Histogram
    tail_p: float
    identically_priced: pd.DataFrame
    threshold: float
    by: str
    n_bins: int
    kind: str = "spread"
    schema_version: int = SCHEMA_VERSION

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "schema_version": int(self.schema_version),
            "parameters": {
                str(name): histogram.to_json() for name, histogram in self.parameters.items()
            },
            "tail_quantile": self.tail_quantile.to_json(),
            "tail_p": float(self.tail_p),
            "identically_priced": _json_frame(self.identically_priced),
            "threshold": float(self.threshold),
            "by": str(self.by),
            "n_bins": int(self.n_bins),
        }


@dataclass(frozen=True)
class Portfolio:
    """Simulated total loss for a book of rows, overall and per segment."""

    quantiles: tuple[float, ...]
    total_quantiles: Mapping[float, float]
    total_mean: float
    total_sd: float
    total_draws: NDArray[np.float64] | None
    by_segment: pd.DataFrame | None
    by: str | None
    n_draws: int
    seed: int
    parameter_uncertainty: bool
    kind: str = "portfolio"
    schema_version: int = SCHEMA_VERSION

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "schema_version": int(self.schema_version),
            "quantiles": [float(value) for value in self.quantiles],
            "total_quantiles": [float(self.total_quantiles[value]) for value in self.quantiles],
            "total_mean": float(self.total_mean),
            "total_sd": float(self.total_sd),
            "total_draws": None if self.total_draws is None else _json_vector(self.total_draws),
            "by_segment": None if self.by_segment is None else _json_frame(self.by_segment),
            "by": None if self.by is None else str(self.by),
            "n_draws": int(self.n_draws),
            "seed": int(self.seed),
            "parameter_uncertainty": bool(self.parameter_uncertainty),
        }


# --------------------------------------------------------------------------- #
# The covariate sweep
# --------------------------------------------------------------------------- #


def _model_columns(fitted: Any) -> tuple[str, ...]:
    return _required_columns(tuple(fitted.compiled_predictors))


def _covariate_grid(
    frame: EagerFrame, covariate: str, n_points: int
) -> tuple[NDArray, tuple[str, ...] | None, NDArray[np.float64]]:
    """Return the swept values, the level labels if any, and the plot coordinate."""
    if frame.column_kind(covariate) == "numeric":
        points = int(n_points)
        if points < 2:
            raise ValueError("a swept covariate needs at least two grid points")
        values = frame.column_array(covariate, dtype=float)
        grid = np.linspace(float(values.min()), float(values.max()), points)
        return grid, None, grid
    declared = frame.column_declared_categories(covariate)
    observed = declared if declared is not None else np.unique(frame.column_array(covariate))
    labels = list(observed)
    return (
        np.array(labels, dtype=object),
        tuple(str(label) for label in labels),
        np.arange(len(labels), dtype=float),
    )


def _default_value(frame: EagerFrame, name: str) -> Any:
    """The training median of a numeric column, else its modal value."""
    values = frame.column_array(name)
    if frame.column_kind(name) == "numeric":
        return float(np.median(np.asarray(values, dtype=float)))
    labels, counts = np.unique(values, return_counts=True)
    return labels[int(np.argmax(counts))]


def _reference_values(
    frame: EagerFrame,
    reference: Mapping[str, Any] | pd.Series,
    covariate: str,
    columns: tuple[str, ...],
) -> dict[str, Any]:
    if not isinstance(reference, Mapping | pd.Series):
        raise TypeError("reference must be a mapping or a pandas Series of column values")
    supplied = {str(key): value for key, value in reference.items()}

    known = {str(name) for name in frame.columns}
    unknown = sorted(name for name in supplied if name not in known)
    if unknown:
        raise ValueError(f"reference names {unknown}, which is not a column of the training frame")
    # A reference taken from a training row carries the swept covariate as well;
    # the sweep overrides it rather than refusing the row.
    return {
        name: supplied[name] if name in supplied else _default_value(frame, name)
        for name in columns
        if name != covariate
    }


def _sweep_frame(
    frame: EagerFrame,
    columns: tuple[str, ...],
    covariate: str,
    sweep: NDArray,
    reference: Mapping[str, Any],
) -> pd.DataFrame:
    width = len(sweep)
    data: dict[str, NDArray] = {}
    for name in columns:
        if name == covariate:
            data[name] = np.asarray(sweep)
        elif frame.column_kind(name) == "numeric":
            data[name] = np.full(width, float(reference[name]), dtype=float)
        else:
            data[name] = np.array([reference[name]] * width, dtype=object)
    return pd.DataFrame(data)


def _sweep(
    fitted: Any,
    X_train: FrameLike | EagerFrame,
    reference: Mapping[str, Any] | pd.Series,
    covariate: str,
    n_points: int,
) -> tuple[pd.DataFrame, tuple[str, ...] | None, NDArray[np.float64], dict[str, Any]]:
    """Build the one-covariate sweep frame every surface is evaluated on."""
    frame = as_eager_frame(X_train)
    columns = _model_columns(fitted)
    if covariate not in columns:
        raise ValueError(f"the model does not use column {covariate!r}; it reads {list(columns)}")
    frame.require_columns(columns)
    values, levels, positions = _covariate_grid(frame, covariate, n_points)
    resolved = _reference_values(frame, reference, covariate, columns)
    return _sweep_frame(frame, columns, covariate, values, resolved), levels, positions, resolved


# --------------------------------------------------------------------------- #
# Risk curves and the density fan
# --------------------------------------------------------------------------- #


def _asked_quantiles(quantiles: Any, *, what: str) -> tuple[float, ...]:
    asked = tuple(float(value) for value in quantiles)
    if not asked:
        raise ValueError(f"{what} need at least one quantile")
    if len(set(asked)) != len(asked):
        raise ValueError(f"{what} need distinct quantiles")
    return asked


def risk_curves(
    fitted: Any,
    X_train: FrameLike | EagerFrame,
    reference: Mapping[str, Any] | pd.Series,
    covariate: str,
    *,
    quantiles: tuple[float, ...] = (0.5, 0.9, 0.99),
    n_points: int = 100,
    level: float = 0.9,
    n_draws: int = 1000,
    covariance: CovarianceKind = "fixed",
    seed: int = 42,
    offsets: Mapping[str, NDArray] | None = None,
    weights: NDArray | None = None,
) -> RiskCurves:
    """Predicted quantiles along one covariate, with posterior bands.

    The covariate sweeps its training range (or its levels) while every other
    column the model reads is held at ``reference``; a column the reference does
    not name is held at its training median (numeric) or modal value.  One
    posterior draw set serves every requested quantile, so the bands of the
    curves are coherent with one another rather than independently simulated.

    ``offsets`` are offsets of the *sweep*, so each one carries ``n_points``
    rows (one per level for a non-numeric covariate), not one per training row.
    ``weights`` are the same shape: the curve is a reference policy at full
    exposure unless the caller states an exposure per swept point, and then
    each point's quantile is the one of its own prior-weighted law.
    """
    asked = _asked_quantiles(quantiles, what="risk curves")
    sweep, levels, positions, resolved = _sweep(fitted, X_train, reference, covariate, n_points)

    draws = posterior_draws(fitted, n_draws, covariance=covariance, seed=seed)
    estimates, lower, upper = [], [], []
    for probability in asked:
        bounds = cast(
            "pd.DataFrame",
            posterior_bounds(
                fitted,
                sweep,
                ("quantile", probability),
                level=level,
                draws=draws,
                offsets=offsets,
                weights=weights,
            ),
        )
        estimates.append(bounds["estimate"].to_numpy())
        lower.append(bounds["lower"].to_numpy())
        upper.append(bounds["upper"].to_numpy())

    return RiskCurves(
        covariate=str(covariate),
        x=_readonly(positions),
        levels=levels,
        quantiles=asked,
        values=_readonly(np.vstack(estimates)),
        lower=_readonly(np.vstack(lower)),
        upper=_readonly(np.vstack(upper)),
        reference=MappingProxyType(dict(resolved)),
        level=float(level),
        n_draws=int(draws.n_draws),
        seed=int(seed),
        covariance=str(covariance),
    )


def _clipped_density(values: NDArray) -> NDArray[np.float64]:
    """Clip differencing round-off to zero and refuse a real negative density."""
    density = np.asarray(values, dtype=np.float64)
    scale = float(np.max(np.abs(density))) if density.size else 0.0
    tolerance = _DENSITY_NEGATIVE_TOLERANCE * max(scale, 1.0)
    if np.any(density < -tolerance):
        raise ValueError(
            "differencing the family's cdf produced a materially negative density; "
            "the distribution function is not monotone on this response grid"
        )
    return np.where(density < 0.0, 0.0, density)


def density_fan(
    fitted: Any,
    X_train: FrameLike | EagerFrame,
    reference: Mapping[str, Any] | pd.Series,
    covariate: str,
    *,
    n_points: int = 60,
    n_y: int = 200,
    offsets: Mapping[str, NDArray] | None = None,
    quantiles: tuple[float, ...] | None = (0.5, 0.9, 0.99),
) -> DensityFan:
    """The conditional density along one covariate sweep.

    The response grid spans the union of the plug-in 0.001 and 0.999 quantiles
    over the sweep, and the density is the central difference of the family's
    distribution function on that grid.  ``offsets`` are offsets of the sweep,
    carrying one row per swept point exactly as :func:`risk_curves` reads them.
    """
    family = fitted.family
    if not isinstance(family, DistributionFunctionFamily):
        raise NotImplementedError(
            "a density fan needs a family with a cdf and a quantile; this one has neither"
        )
    grid_points = int(n_y)
    if grid_points < 3:
        raise ValueError("a density fan needs at least three response grid points")

    sweep, levels, positions, resolved = _sweep(fitted, X_train, reference, covariate, n_points)
    theta = np.asarray(fitted.predict_parameters(sweep, offsets=offsets), dtype=np.float64)
    rows = len(theta)
    lower = np.asarray(family.quantile(np.full(rows, _DENSITY_TAIL), theta), dtype=np.float64)
    upper = np.asarray(family.quantile(np.full(rows, 1.0 - _DENSITY_TAIL), theta), dtype=np.float64)
    y_grid = np.linspace(float(lower.min()), float(upper.max()), grid_points)

    cdf = np.column_stack(
        [
            np.asarray(family.cdf(np.full(rows, float(value)), theta), dtype=np.float64)
            for value in y_grid
        ]
    )
    density = _clipped_density(np.gradient(cdf, y_grid, axis=1))

    quantile_levels = None if quantiles is None else tuple(float(level) for level in quantiles)
    quantile_curves = None
    if quantile_levels:
        quantile_curves = np.vstack(
            [
                np.asarray(family.quantile(np.full(rows, level), theta), dtype=np.float64)
                for level in quantile_levels
            ]
        )
    return DensityFan(
        covariate=str(covariate),
        x=_readonly(positions),
        levels=levels,
        y_grid=_readonly(y_grid),
        density=_readonly(density),
        quantile_levels=quantile_levels,
        quantiles=None if quantile_curves is None else _readonly(quantile_curves),
        reference=MappingProxyType(dict(resolved)),
    )


# --------------------------------------------------------------------------- #
# Parameter spread and the identically priced
# --------------------------------------------------------------------------- #


def _histogram(values: NDArray) -> Histogram:
    counts, edges = np.histogram(np.asarray(values, dtype=np.float64), bins=_HISTOGRAM_BINS)
    return Histogram(edges=_readonly(edges), counts=_readonly(counts.astype(np.int64)))


def _resolved_weights(
    sample_weight: NDArray | None, rows: int, *, name: str = "sample_weight"
) -> NDArray[np.float64]:
    if sample_weight is None:
        return np.ones(rows, dtype=np.float64)
    weights = np.asarray(sample_weight, dtype=np.float64)
    if weights.shape != (rows,):
        raise ValueError(f"{name} must give one weight per row of X")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError(f"{name} must be finite and non-negative")
    return weights


def _equal_count_bins(values: NDArray, n_bins: int) -> tuple[NDArray[np.intp], list[NDArray]]:
    """Split rows into ``n_bins`` groups of equal count, ordered by ``values``."""
    order = np.argsort(values, kind="stable")
    groups = list(np.array_split(order, n_bins))
    codes = np.empty(len(values), dtype=np.intp)
    for index, rows in enumerate(groups):
        codes[rows] = index
    return codes, groups


def parameter_spread(
    fitted: Any,
    X: FrameLike | EagerFrame,
    *,
    threshold: float,
    tail_p: float = 0.99,
    n_bins: int = 20,
    by: str = "mean",
    sample_weight: NDArray | None = None,
    weights: NDArray | None = None,
) -> Spread:
    """Sharpness of the fitted parameters and spread among identical prices.

    The histograms show how far the fitted parameters and the predicted
    ``tail_p`` quantile spread across the rows.  The identically-priced table
    bins rows into ``n_bins`` equal-count bins of the predicted mean and reports,
    per bin, the weighted mean price as a ratio of sums and the 5th and 95th
    percentiles of the exceedance probability ``P(Y > threshold)`` with their
    ratio: rows a model prices alike can still differ many-fold in tail risk.

    ``sample_weight`` weighs the ratio of sums the table reports; ``weights``
    are the rows' *prior* weights, which are part of each row's own law, so the
    tail quantile and the exceedance probability are read from the
    prior-weighted law of the row rather than from the unit-weight one.  The
    prices the table bins by do not move with it: a prior weight leaves the
    mean of the row law alone.
    """
    if by != "mean":
        raise NotImplementedError(
            f"parameter_spread bins by the predicted mean; pass by='mean', not {by!r}"
        )
    family = fitted.family
    if not isinstance(family, DefaultPredictionFamily):
        raise NotImplementedError(
            "this family names no default prediction, so it has no predicted mean to bin by"
        )
    if not isinstance(family, DistributionFunctionFamily):
        raise NotImplementedError(
            "an exceedance probability needs a family with a cdf; this one has none"
        )
    bins = int(n_bins)
    if bins < 1:
        raise ValueError("n_bins must be at least one bin")
    tail = float(tail_p)
    if not 0.0 < tail < 1.0:
        raise ValueError("tail_p must lie strictly inside (0, 1)")

    theta = np.asarray(fitted.predict_parameters(X), dtype=np.float64)
    rows = len(theta)
    if rows < bins:
        raise ValueError(f"X has fewer rows than bins: {rows} rows for {bins} equal-count bins")
    # The two weight vectors mean different things and must not share a name:
    # ``aggregation`` weighs the ratio of sums, ``prior`` is part of the law.
    aggregation = _resolved_weights(sample_weight, rows)
    prior = None if weights is None else _resolved_weights(weights, rows, name="weights")
    mean = np.asarray(family.default_prediction(theta), dtype=np.float64)
    tail_values = resolve_quantity(family, ("quantile", tail), weights=prior)(theta)
    exceedance = resolve_quantity(family, ("exceedance", float(threshold)), weights=prior)(theta)

    codes, groups = _equal_count_bins(mean, bins)
    _, exposure, weighted_mean = grouped_ratio(
        aggregation * mean, aggregation, codes, n_groups=bins
    )
    low = np.array([float(mean[rows_in_bin].min()) for rows_in_bin in groups])
    high = np.array([float(mean[rows_in_bin].max()) for rows_in_bin in groups])
    percentiles = np.array(
        [np.quantile(exceedance[rows_in_bin], _SPREAD_PERCENTILES) for rows_in_bin in groups]
    )
    spread_ratio = np.full(bins, np.nan)
    np.divide(percentiles[:, 1], percentiles[:, 0], out=spread_ratio, where=percentiles[:, 0] > 0.0)

    table = pd.DataFrame(
        {
            "bin": np.arange(bins, dtype=np.int64),
            "n": np.bincount(codes, minlength=bins).astype(np.int64),
            "weight": exposure,
            "mean_lo": low,
            "mean_hi": high,
            "mean": weighted_mean,
            "p_lo": percentiles[:, 0],
            "p_hi": percentiles[:, 1],
            "ratio": spread_ratio,
        }
    )
    parameters = {
        spec.name: _histogram(theta[:, index]) for index, spec in enumerate(family.parameters)
    }
    return Spread(
        parameters=MappingProxyType(parameters),
        tail_quantile=_histogram(tail_values),
        tail_p=tail,
        identically_priced=table,
        threshold=float(threshold),
        by=str(by),
        n_bins=bins,
    )


# --------------------------------------------------------------------------- #
# Portfolio
# --------------------------------------------------------------------------- #


class _Segmentation(NamedTuple):
    """One segment code per row, the segment labels, and the column asked for."""

    name: str | None
    codes: NDArray[np.intp]
    labels: tuple[Any, ...]


def _paid(block: NDArray, weights: NDArray[np.float64] | None, cursor: int) -> NDArray[np.float64]:
    """Return what the chunk's simulated rows cost: a rate times its own weight."""
    if weights is None:
        return np.asarray(block, dtype=np.float64)
    return np.asarray(block, dtype=np.float64) * weights[cursor : cursor + block.shape[1]]


class _WeightedTotal:
    """A predictive reduce that pays each simulated rate by its own row's weight.

    A burn cost is a rate per unit of exposure, so the book's loss is
    ``sum_i w_i y_i`` and not the sum of the rates.  Chunks arrive once each,
    in row order, which is what the cursor consumes.
    """

    def __init__(self, weights: NDArray[np.float64]) -> None:
        self.weights = weights
        self.cursor = 0

    def __call__(self, block: NDArray) -> NDArray[np.float64]:
        width = block.shape[1]
        if self.cursor + width > len(self.weights):
            raise RuntimeError("the predictive reduce was handed more rows than X has")
        paid = _paid(block, self.weights, self.cursor)
        self.cursor += width
        return paid.sum(axis=1)


class _SegmentTotals:
    """A predictive reduce that totals each segment while returning the book total.

    ``posterior_predictive`` sums a ``(draws,)`` reduce across row chunks and
    concatenates a ``(draws, columns)`` one, so a per-segment total -- additive,
    not concatenable -- is accumulated here and the chunk's overall total is what
    the primitive sees.  Chunks arrive once each, in row order, which is what
    the cursor consumes.
    """

    def __init__(
        self,
        segmentation: _Segmentation,
        n_draws: int,
        weights: NDArray[np.float64] | None = None,
    ) -> None:
        self.segmentation = segmentation
        self.weights = weights
        self.totals = np.zeros((int(n_draws), len(segmentation.labels)), dtype=np.float64)
        self.cursor = 0

    def __call__(self, block: NDArray) -> NDArray[np.float64]:
        width = block.shape[1]
        codes = self.segmentation.codes[self.cursor : self.cursor + width]
        if len(codes) != width:
            raise RuntimeError("the predictive reduce was handed more rows than X has")
        paid = _paid(block, self.weights, self.cursor)
        self.cursor += width
        for index in range(block.shape[0]):
            self.totals[index] += np.bincount(
                codes, weights=paid[index], minlength=self.totals.shape[1]
            )
        return paid.sum(axis=1)

    def table(self, columns: list[str], quantiles: tuple[float, ...], *, rows: int) -> pd.DataFrame:
        """Return one row per segment: its size, mean total and total quantiles."""
        if self.cursor != int(rows):
            raise RuntimeError("the predictive reduce did not see every row of X")
        labels = self.segmentation.labels
        data: dict[str, Any] = {
            "segment": list(labels),
            "n": np.bincount(self.segmentation.codes, minlength=len(labels)).astype(np.int64),
            "mean_total": self.totals.mean(axis=0),
        }
        for column, value in zip(columns, quantiles, strict=True):
            data[column] = np.quantile(self.totals, value, axis=0)
        return pd.DataFrame(data)


def _resolved_segments(
    frame: EagerFrame, by: str | NDArray | None, rows: int
) -> _Segmentation | None:
    if by is None:
        return None
    if isinstance(by, str):
        if by not in {str(name) for name in frame.columns}:
            raise ValueError(f"by={by!r} is not a column of X")
        labels = frame.column_array(by)
        name: str | None = by
    else:
        labels = np.asarray(by)
        if labels.shape != (rows,):
            raise ValueError("by must give one segment label per row of X")
        name = None
    codes, uniques = pd.factorize(labels, sort=True)
    if np.any(codes < 0):
        raise ValueError("by has missing segment labels; every row needs a segment")
    return _Segmentation(name, np.asarray(codes, dtype=np.intp), tuple(uniques))


def portfolio(
    fitted: Any,
    X: FrameLike | EagerFrame,
    *,
    n_draws: int = 500,
    by: str | NDArray | None = None,
    seed: int = 42,
    parameter_uncertainty: bool = True,
    quantiles: tuple[float, ...] = (0.5, 0.9, 0.99),
    return_draws: bool = False,
    chunk_rows: int | None = None,
    weights: NDArray | None = None,
) -> Portfolio:
    """Summarise the simulated total loss of a book of rows.

    The total is ``posterior_predictive(..., reduce="sum")``: one simulated
    response per row per draw, summed over rows.  ``by`` -- a column of ``X`` or
    an array of labels -- also reports each segment's own total, whose means sum
    to the book mean because every row lands in exactly one segment.
    ``chunk_rows`` is the memory knob of the primitive; a predictive total is not
    chunk-invariant, since the uniforms are drawn per chunk.

    ``weights`` are the rows' prior weights -- an exposure on a burn-cost
    model.  They enter twice, because they mean the same thing in both places:
    each row is simulated on its own prior-weighted law, and what the book pays
    is ``sum_i w_i y_i``, the rate times the exposure that bought it, overall
    and within every segment.
    """
    asked = _asked_quantiles(quantiles, what="a portfolio payload")
    columns = [f"q{value:g}" for value in asked]
    if len(set(columns)) != len(columns):
        raise ValueError("a portfolio payload needs distinct quantiles")
    draws = int(n_draws)
    if draws < 2:
        raise ValueError("n_draws must be at least two to summarise a total")

    frame = as_eager_frame(X)
    segmentation = _resolved_segments(frame, by, len(frame))
    exposure = None if weights is None else _resolved_weights(weights, len(frame), name="weights")
    segment_totals = None if segmentation is None else _SegmentTotals(segmentation, draws, exposure)
    accumulator: Any = segment_totals
    if segment_totals is None and exposure is not None:
        accumulator = _WeightedTotal(exposure)
    totals = np.asarray(
        posterior_predictive(
            fitted,
            frame,
            draws,
            parameter_uncertainty=parameter_uncertainty,
            reduce="sum" if accumulator is None else accumulator,
            seed=seed,
            chunk_rows=chunk_rows,
            weights=exposure,
        ),
        dtype=np.float64,
    )

    table = (
        None if segment_totals is None else segment_totals.table(columns, asked, rows=len(frame))
    )

    return Portfolio(
        quantiles=asked,
        total_quantiles=MappingProxyType(
            {value: float(np.quantile(totals, value)) for value in asked}
        ),
        total_mean=float(totals.mean()),
        total_sd=float(totals.std(ddof=1)),
        total_draws=_readonly(totals) if return_draws else None,
        by_segment=table,
        by=None if segmentation is None else segmentation.name,
        n_draws=draws,
        seed=int(seed),
        parameter_uncertainty=bool(parameter_uncertainty),
    )


__all__ = [
    "SCHEMA_VERSION",
    "DensityFan",
    "Histogram",
    "Portfolio",
    "RiskCurves",
    "Spread",
    "density_fan",
    "parameter_spread",
    "portfolio",
    "risk_curves",
]
