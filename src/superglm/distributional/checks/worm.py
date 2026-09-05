"""Worm plot and its Q-statistics table.

The worm plot of van Buuren and Fredriks (2001), *Statistics in Medicine* 20(8),
1259-1277, is the Q-Q plot with the line taken out: against the theoretical
quantile ``z`` of each order statistic it plots the deviation
``observed - z``, so a departure that a Q-Q plot hides inside the diagonal
becomes a shape.  A worm that sits above the axis says the residuals are
shifted, one that runs at a slope says they are over- or under-dispersed, a
U says they are skewed and an S says the tails are wrong -- which, in a
distributional model, names the predictor at fault rather than merely the fit.
The pointwise 95 per cent band is theirs too,

    +/- 1.96 sqrt(p (1 - p) / n) / phi(z),   p = Phi(z),

the standard error of the empirical quantile at ``z`` mapped onto the deviation
scale.  It is pointwise, not simultaneous: it says where one point is
surprising, not where the whole worm is.

With a covariate the rows are cut into equal-count intervals -- one worm per
interval, which is where a misfit's location shows -- or one worm per level of a
categorical covariate.  The companion table is the Q-statistics of Royston and
Wright (2000), *Statistics in Medicine* 19(21), 2943-2962: per group the
standardised mean, variance, skewness and kurtosis of the residuals.  The
constants are the asymptotic null standard deviations of those four sample
moments for standard normal residuals -- ``sqrt(1/n)``, ``sqrt(2/n)``,
``sqrt(6/n)`` and ``sqrt(24/n)`` -- so each statistic is standard normal under a
correct fit and ``|z| > 2`` is the flag.  Reading them names the predictor: a
mean ``z`` flags the location model, a variance ``z`` the scale model, and a
skewness or kurtosis ``z`` the shape model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import special, stats

from superglm.distributional.checks.qq import order_statistic_grid
from superglm.distributional.residuals import ResidualSet, replication_sample

#: Pointwise band multiplier: the two-sided 95 % normal critical value.
_BAND_CRITICAL_VALUE = 1.96
#: The band grid spans this many standard deviations either side of zero.
_BAND_LIMIT = 3.0
#: A Q-statistic is flagged beyond this many standard deviations.
_FLAG_THRESHOLD = 2.0
#: Row set for a check without a seed of its own: the replication default.
_REPLICATION_SEED = 42
#: The overall row of a Q-statistics table.
_OVERALL_LABEL = "all"
_Q_COLUMNS = ("group", "n", "mean_z", "variance_z", "skewness_z", "kurtosis_z", "flagged")


def _readonly(values: NDArray) -> NDArray[np.float64]:
    array = np.array(values, dtype=np.float64, copy=True)
    array.setflags(write=False)
    return array


def _json_numbers(values: NDArray) -> list[float | None]:
    """Emit a float list with every non-finite entry as ``null``."""
    return [None if not np.isfinite(value) else float(value) for value in np.asarray(values)]


def _json_number(value: float) -> float | None:
    return None if not np.isfinite(value) else float(value)


def _standard_normal_pdf(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.exp(-0.5 * values * values) / np.sqrt(2.0 * np.pi)


@dataclass(frozen=True)
class WormPanel:
    """One worm: the deviation cloud on its order-statistic grid, plus its band.

    ``z`` and ``deviation`` are the point cloud, one point per row of the panel.
    ``band`` is the half-width of the pointwise 95 per cent band on the separate
    grid ``band_z``, so a renderer draws ``+band`` and ``-band`` around zero and
    reads a point as surprising when its deviation leaves the interpolated band.
    """

    label: str
    z: NDArray[np.float64]
    deviation: NDArray[np.float64]
    band_z: NDArray[np.float64]
    band: NDArray[np.float64]
    n: int
    interval: tuple[float, float] | None

    def __post_init__(self) -> None:
        positions = _readonly(self.z)
        deviation = _readonly(self.deviation)
        if deviation.shape != positions.shape:
            raise ValueError("deviation must carry one value per order statistic")
        grid = _readonly(self.band_z)
        band = _readonly(self.band)
        if band.shape != grid.shape:
            raise ValueError("band must carry one half-width per band grid point")
        object.__setattr__(self, "z", positions)
        object.__setattr__(self, "deviation", deviation)
        object.__setattr__(self, "band_z", grid)
        object.__setattr__(self, "band", band)
        object.__setattr__(self, "label", str(self.label))
        object.__setattr__(self, "n", int(self.n))
        if self.interval is not None:
            low, high = self.interval
            object.__setattr__(self, "interval", (float(low), float(high)))

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-safe payload of lists, numbers, strings and ``None``."""
        return {
            "label": self.label,
            "n": self.n,
            "interval": None if self.interval is None else [self.interval[0], self.interval[1]],
            "z": _json_numbers(self.z),
            "deviation": _json_numbers(self.deviation),
            "band_z": _json_numbers(self.band_z),
            "band": _json_numbers(self.band),
        }


@dataclass(frozen=True)
class WormPayload:
    """Every worm of one check, and the Q-statistics that go beside them."""

    panels: tuple[WormPanel, ...]
    covariate: str | None
    q_statistics: pd.DataFrame | None
    kind: str = "worm"
    schema_version: int = 1

    def __post_init__(self) -> None:
        panels = tuple(self.panels)
        if not panels or not all(isinstance(panel, WormPanel) for panel in panels):
            raise ValueError("a worm payload carries at least one WormPanel")
        object.__setattr__(self, "panels", panels)
        if self.covariate is not None:
            object.__setattr__(self, "covariate", str(self.covariate))

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-safe payload of lists, numbers, strings and ``None``."""
        table = None
        if self.q_statistics is not None:
            table = [
                {
                    "group": str(row["group"]),
                    "n": int(row["n"]),
                    "mean_z": _json_number(row["mean_z"]),
                    "variance_z": _json_number(row["variance_z"]),
                    "skewness_z": _json_number(row["skewness_z"]),
                    "kurtosis_z": _json_number(row["kurtosis_z"]),
                    "flagged": bool(row["flagged"]),
                }
                for _, row in self.q_statistics.iterrows()
            ]
        return {
            "kind": self.kind,
            "schema_version": self.schema_version,
            "covariate": self.covariate,
            "panels": [panel.to_json() for panel in self.panels],
            "q_statistics": table,
        }


def _moment_row(label: Any, values: NDArray[np.float64]) -> tuple[Any, ...]:
    """Return one Q-statistics record: the four standardised moments of ``values``."""
    n = len(values)
    mean_z = float(np.sqrt(n) * np.mean(values))
    variance_z = np.nan
    skewness_z = np.nan
    kurtosis_z = np.nan
    if n > 1:
        variance = float(np.var(values, ddof=1))
        variance_z = float(np.sqrt(n / 2.0) * (variance - 1.0))
        if variance > 0.0:
            skewness_z = float(np.sqrt(n / 6.0) * stats.skew(values, bias=True))
            kurtosis_z = float(np.sqrt(n / 24.0) * stats.kurtosis(values, fisher=True, bias=True))
    moments = np.array([mean_z, variance_z, skewness_z, kurtosis_z], dtype=np.float64)
    flagged = bool(np.any(np.abs(moments) > _FLAG_THRESHOLD))
    return (label, n, mean_z, variance_z, skewness_z, kurtosis_z, flagged)


def q_statistics(residual_quantiles: NDArray, groups: NDArray) -> pd.DataFrame:
    """Return the Q-statistics of Royston and Wright (2000) per group.

    Each column is one of the four sample moments of the residuals, divided by
    its asymptotic null standard deviation for standard normal residuals, so
    every entry is standard normal under a correct fit:

    ``mean_z = sqrt(n) mean(r)``, ``variance_z = sqrt(n / 2) (var(r) - 1)``,
    ``skewness_z = sqrt(n / 6) skew(r)`` and
    ``kurtosis_z = sqrt(n / 24) (kurtosis(r) - 3)``,

    with the sample skewness and excess kurtosis in their plain moment-ratio
    form and ``var`` the unbiased estimate.  ``flagged`` is ``True`` where any
    of the four exceeds two in absolute value.  Groups appear in order of first
    appearance; an overall ``"all"`` row is appended whenever there is more than
    one, where a single group would only repeat itself.  A statistic a group is
    too thin to support -- a variance from one row, a skewness or kurtosis from
    a group with no spread -- is ``NaN`` rather than a fabricated number.
    """
    values = np.asarray(residual_quantiles, dtype=np.float64)
    if values.ndim != 1 or len(values) < 1:
        raise ValueError("residual_quantiles must be a non-empty one-dimensional array")
    labels = np.asarray(groups)
    if labels.shape != values.shape:
        raise ValueError("q_statistics needs one group label per residual")

    codes, uniques = pd.factorize(labels, sort=False)
    records = [_moment_row(label, values[codes == index]) for index, label in enumerate(uniques)]
    if len(uniques) > 1:
        records.append(_moment_row(_OVERALL_LABEL, values))
    frame = pd.DataFrame.from_records(records, columns=list(_Q_COLUMNS))
    frame["n"] = frame["n"].astype(np.int64)
    frame["flagged"] = frame["flagged"].astype(bool)
    return frame


def _interval_codes(
    column: NDArray[np.float64], n_intervals: int
) -> tuple[NDArray[np.intp], list[str], list[tuple[float, float]]]:
    """Return equal-count interval codes, labels and edges for a numeric covariate."""
    intervals = int(n_intervals)
    if intervals < 1:
        raise ValueError("n_intervals must cut at least one interval")
    if not np.all(np.isfinite(column)):
        raise ValueError("a numeric covariate must be finite to cut into equal-count intervals")
    edges = np.quantile(column, np.linspace(0.0, 1.0, intervals + 1))
    codes = np.asarray(np.searchsorted(edges[1:-1], column, side="right"), dtype=np.intp)
    if np.any(np.bincount(codes, minlength=intervals) == 0):
        raise ValueError(
            f"{intervals} equal-count intervals leave one of them empty; this covariate has too "
            "many ties for that many intervals"
        )
    bounds = [(float(edges[index]), float(edges[index + 1])) for index in range(intervals)]
    labels = [f"[{low:.4g}, {high:.4g})" for low, high in bounds]
    return codes, labels, bounds


def _covariate_groups(
    covariate: Any, rows: NDArray[np.intp], n_rows: int, n_intervals: int
) -> tuple[NDArray[np.intp], list[str], list[tuple[float, float] | None]]:
    """Return the panel code of each row, the panel labels and their intervals."""
    series = covariate if isinstance(covariate, pd.Series) else pd.Series(covariate)
    if len(series) != n_rows:
        raise ValueError("covariate must give one value per row of the residuals")
    selected = series.iloc[rows]
    if pd.api.types.is_numeric_dtype(selected) and not pd.api.types.is_bool_dtype(selected):
        codes, labels, bounds = _interval_codes(selected.to_numpy(dtype=np.float64), n_intervals)
        return codes, labels, list(bounds)
    codes, uniques = pd.factorize(selected, sort=True)
    return (
        np.asarray(codes, dtype=np.intp),
        [str(level) for level in uniques],
        [None] * len(uniques),
    )


def worm_payload(
    residuals: ResidualSet,
    *,
    covariate: NDArray | None = None,
    covariate_name: str | None = None,
    n_intervals: int = 4,
    n_points: int = 200,
) -> WormPayload:
    """Return the worm payload of ``residuals``, one panel per covariate group.

    Without a covariate there is one worm over every row.  A numeric covariate
    is cut into ``n_intervals`` equal-count intervals by its quantiles and each
    panel records its own ``interval``; a categorical, boolean or object
    covariate gets one panel per level instead.  Rows are the ones
    :func:`superglm.distributional.residuals.replication_sample` gives, so
    frequency weights are literal replication here as they are in the fit.
    """
    if not isinstance(residuals, ResidualSet):
        raise TypeError("residuals must be a ResidualSet")
    points = int(n_points)
    if points < 2:
        raise ValueError("n_points must place at least two points on the band grid")

    rows = replication_sample(residuals, seed=_REPLICATION_SEED)
    values = residuals.quantile[rows]
    if covariate is None:
        codes = np.zeros(len(rows), dtype=np.intp)
        labels: list[str] = [_OVERALL_LABEL]
        bounds: list[tuple[float, float] | None] = [None]
    else:
        codes, labels, bounds = _covariate_groups(covariate, rows, residuals.n_rows, n_intervals)

    band_z = np.linspace(-_BAND_LIMIT, _BAND_LIMIT, points)
    probability = special.ndtr(band_z)
    panels: list[WormPanel] = []
    order: list[NDArray[np.intp]] = []
    for index, label in enumerate(labels):
        selected = np.flatnonzero(codes == index)
        order.append(selected)
        count = len(selected)
        theoretical = order_statistic_grid(count)
        panels.append(
            WormPanel(
                label=label,
                z=theoretical,
                deviation=np.sort(values[selected]) - theoretical,
                band_z=band_z,
                band=_BAND_CRITICAL_VALUE
                * np.sqrt(probability * (1.0 - probability) / count)
                / _standard_normal_pdf(band_z),
                n=count,
                interval=bounds[index],
            )
        )

    # Ordering the rows by panel makes the table's first-appearance order the
    # panel order, so the worms and their statistics read down the page together.
    ordered = np.concatenate(order)
    table = q_statistics(
        values[ordered], np.array([labels[code] for code in codes[ordered]], dtype=object)
    )
    return WormPayload(
        panels=tuple(panels),
        covariate=None if covariate_name is None else str(covariate_name),
        q_statistics=table,
    )


__all__ = ["WormPanel", "WormPayload", "q_statistics", "worm_payload"]
