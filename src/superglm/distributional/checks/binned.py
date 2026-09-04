"""Binned residual checks against a covariate.

A Q-Q plot says whether the residuals are standard normal overall; it cannot say
*where* they are not.  Binning the quantile residuals by a covariate and
reporting the moments per bin is the scalable checking construction of Fasiolo,
Nedellec, Goude and Wood (2020), *Journal of Computational and Graphical
Statistics* 29(1), 78-86, which replaces a scatter of n points by a handful of
summaries with bootstrap bands.

We report three moments rather than one.  A distributional model has a predictor
for every parameter, so the question is not only "is the fit wrong here" but "in
which moment": a bin whose **mean** band excludes zero indicts the location
predictor, one whose **standard deviation** band excludes one indicts the scale
predictor, and one whose **skewness** band excludes zero indicts the shape
predictor.  The bands are seeded percentile bootstrap intervals over the rows of
the bin, which treats rows within a bin as exchangeable.

Residuals are not rates, so a binned mean here is the plain mean over rows,
weighted only by replication: under the frequency contract the rows are expanded
by :func:`~superglm.distributional.residuals.replication_sample` before any
moment is taken, and under the prior contract the weight is already inside each
row's own distribution function.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

from superglm.distributional.residuals import ResidualSet, replication_sample

# One JSON convention and one schema version for every payload in the suite;
# restating them here is how two payloads drift into two encodings.
from superglm.distributional.surfaces import (
    SCHEMA_VERSION,
    _json_matrix,
    _json_vector,
    _readonly,
)

#: Percentiles of the bootstrap distribution reported as a band.
_BAND_PERCENTILES = (0.025, 0.975)
#: A bin below this many rows has no second or third moment worth reporting.
_MINIMUM_BIN_ROWS = 3


class _Binning(NamedTuple):
    """One group code per row, with the edges or levels that produced them."""

    codes: NDArray[np.intp]
    n_groups: int
    edges: NDArray[np.float64] | None
    levels: tuple[str, ...] | None
    centers: NDArray[np.float64]


def _covariate_bins(values: Any, n_bins: int, *, name: str = "covariate") -> _Binning:
    """Split rows into equal-count bins of a numeric covariate, or one bin per level.

    A numeric covariate is cut at its ``n_bins`` quantile edges, deduplicated so
    a covariate with ties or few distinct values yields fewer, non-empty bins
    rather than empty ones.  Anything else -- strings, categories, booleans -- is
    grouped by level in sorted order, and a missing level is refused rather than
    silently dropped into a bin of its own.
    """
    bins = int(n_bins)
    if bins < 1:
        raise ValueError("n_bins must be at least one bin")
    array = np.asarray(values)
    if array.ndim != 1 or len(array) < 1:
        raise ValueError(f"{name} must be a one-dimensional array with at least one row")

    if not np.issubdtype(array.dtype, np.number) or array.dtype.kind == "b":
        codes, uniques = pd.factorize(array, sort=True)
        if np.any(codes < 0):
            raise ValueError(f"{name} must not contain missing values")
        levels = tuple(str(level) for level in uniques)
        return _Binning(
            codes=np.asarray(codes, dtype=np.intp),
            n_groups=len(levels),
            edges=None,
            levels=levels,
            centers=np.arange(len(levels), dtype=np.float64),
        )

    numeric = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{name} must be finite")
    edges = np.unique(np.quantile(numeric, np.linspace(0.0, 1.0, bins + 1)))
    if edges.size < 2:
        # A constant covariate is one bin; a degenerate edge pair keeps the
        # payload's shape contract (one more edge than centre) intact.
        value = float(edges[0])
        return _Binning(
            codes=np.zeros(len(numeric), dtype=np.intp),
            n_groups=1,
            edges=np.array([value, value], dtype=np.float64),
            levels=None,
            centers=np.array([value], dtype=np.float64),
        )
    codes = np.searchsorted(edges[1:-1], numeric, side="right").astype(np.intp)
    return _Binning(
        codes=codes,
        n_groups=len(edges) - 1,
        edges=edges,
        levels=None,
        centers=0.5 * (edges[:-1] + edges[1:]),
    )


@dataclass(frozen=True)
class BinnedCheck:
    """Mean, standard deviation and skewness of the residuals per bin, with bands.

    ``edges`` is set for a numeric covariate and ``levels`` for a categorical
    one; exactly one of the two is ``None``.  ``centers`` is the plotting
    coordinate either way -- the bin midpoint, or ``0 .. L-1`` for ``L`` levels.
    A bin holding fewer than three rows reports ``nan`` moments and ``nan``
    bands: it has no third moment to speak of, and a number there would read as
    evidence.
    """

    covariate: str
    edges: NDArray[np.float64] | None
    levels: tuple[str, ...] | None
    centers: NDArray[np.float64]
    n: NDArray[np.int64]
    mean: NDArray[np.float64]
    mean_lower: NDArray[np.float64]
    mean_upper: NDArray[np.float64]
    sd: NDArray[np.float64]
    sd_lower: NDArray[np.float64]
    sd_upper: NDArray[np.float64]
    skew: NDArray[np.float64]
    skew_lower: NDArray[np.float64]
    skew_upper: NDArray[np.float64]
    n_boot: int
    seed: int
    kind: str = "binned"
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
            "mean": _json_vector(self.mean),
            "mean_lower": _json_vector(self.mean_lower),
            "mean_upper": _json_vector(self.mean_upper),
            "sd": _json_vector(self.sd),
            "sd_lower": _json_vector(self.sd_lower),
            "sd_upper": _json_vector(self.sd_upper),
            "skew": _json_vector(self.skew),
            "skew_lower": _json_vector(self.skew_lower),
            "skew_upper": _json_vector(self.skew_upper),
            "n_boot": int(self.n_boot),
            "seed": int(self.seed),
        }


@dataclass(frozen=True)
class BinnedCheck2D:
    """The binned mean residual over a grid of two covariates, with cell counts."""

    covariates: tuple[str, str]
    x_edges: NDArray[np.float64]
    y_edges: NDArray[np.float64]
    mean: NDArray[np.float64]
    count: NDArray[np.int64]
    kind: str = "binned2d"
    schema_version: int = SCHEMA_VERSION

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "schema_version": int(self.schema_version),
            "covariates": [str(name) for name in self.covariates],
            "x_edges": _json_vector(self.x_edges),
            "y_edges": _json_vector(self.y_edges),
            "mean": _json_matrix(self.mean),
            "count": _json_matrix(self.count),
        }


def _replicated(residuals: Any, covariates: tuple[NDArray, ...]) -> tuple[NDArray, ...]:
    """Return the residual values and covariates expanded by the replication weights."""
    if not isinstance(residuals, ResidualSet):
        raise TypeError("residuals must be a ResidualSet")
    rows = residuals.n_rows
    for values in covariates:
        array = np.asarray(values)
        if array.ndim != 1 or len(array) != rows:
            raise ValueError("a binned check needs one value per residual row")
    index = replication_sample(residuals)
    return (residuals.quantile[index], *(np.asarray(v)[index] for v in covariates))


def binned_check(
    residuals: ResidualSet,
    covariate: NDArray,
    *,
    name: str,
    n_bins: int = 20,
    n_boot: int = 200,
    seed: int = 42,
) -> BinnedCheck:
    """Return the first three residual moments per bin of ``covariate`` with bands.

    ``covariate`` gives one value per row of ``residuals``.  The bands are
    ``n_boot`` seeded resamples of the rows within each bin, read at the 2.5th
    and 97.5th percentiles (Fasiolo et al. 2020, extended from the mean to the
    second and third moments).
    """
    boot = int(n_boot)
    if boot < 1:
        raise ValueError("n_boot must be at least one resample")
    values, grouped = _replicated(residuals, (covariate,))
    binning = _covariate_bins(grouped, n_bins, name="covariate")

    width = binning.n_groups
    counts = np.zeros(width, dtype=np.int64)
    moments = {key: np.full(width, np.nan) for key in ("mean", "sd", "skew")}
    bands = {key: np.full((width, 2), np.nan) for key in ("mean", "sd", "skew")}
    generator = np.random.default_rng(seed)
    for index in range(width):
        rows = np.flatnonzero(binning.codes == index)
        counts[index] = len(rows)
        if len(rows) < _MINIMUM_BIN_ROWS:
            continue
        sample = values[rows]
        resampled = sample[generator.integers(0, len(rows), size=(boot, len(rows)))]
        with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
            # In a very small bin a resample can draw one row throughout, and a
            # constant sample has no skewness; scipy reports that as a warning
            # and a nan.  Such replicates leave the percentile band instead of
            # voiding it -- the bin's own ``n`` says how much the band is worth
            # -- and a bin whose every replicate is degenerate has no band at
            # all, which is the honest answer there.
            warnings.filterwarnings("ignore", "Precision loss occurred", RuntimeWarning)
            warnings.filterwarnings("ignore", "All-NaN slice encountered", RuntimeWarning)
            statistics = {
                "mean": (float(sample.mean()), resampled.mean(axis=1)),
                "sd": (float(sample.std(ddof=1)), resampled.std(axis=1, ddof=1)),
                "skew": (float(stats.skew(sample)), np.asarray(stats.skew(resampled, axis=1))),
            }
            for key, (point, replicates) in statistics.items():
                moments[key][index] = point
                bands[key][index] = np.nanquantile(replicates, _BAND_PERCENTILES)

    return BinnedCheck(
        covariate=str(name),
        edges=None if binning.edges is None else _readonly(binning.edges),
        levels=binning.levels,
        centers=_readonly(binning.centers),
        n=_readonly(counts),
        mean=_readonly(moments["mean"]),
        mean_lower=_readonly(bands["mean"][:, 0]),
        mean_upper=_readonly(bands["mean"][:, 1]),
        sd=_readonly(moments["sd"]),
        sd_lower=_readonly(bands["sd"][:, 0]),
        sd_upper=_readonly(bands["sd"][:, 1]),
        skew=_readonly(moments["skew"]),
        skew_lower=_readonly(bands["skew"][:, 0]),
        skew_upper=_readonly(bands["skew"][:, 1]),
        n_boot=boot,
        seed=int(seed),
    )


def binned_check_2d(
    residuals: ResidualSet,
    x: NDArray,
    y_cov: NDArray,
    *,
    names: tuple[str, str],
    n_bins: tuple[int, int] = (12, 12),
) -> BinnedCheck2D:
    """Return the mean residual and the row count over a grid of two numeric covariates.

    Both axes are cut into equal-count bins, so the grid is dense where the data
    are.  A cell no row falls into reports a ``nan`` mean beside a zero count
    rather than a zero, which on a heatmap would read as a well-fitting cell.
    """
    x_count, y_count = (int(value) for value in n_bins)
    values, x_values, y_values = _replicated(residuals, (x, y_cov))
    x_bins = _covariate_bins(x_values, x_count, name="x")
    y_bins = _covariate_bins(y_values, y_count, name="y_cov")
    if x_bins.edges is None:
        raise ValueError("a two-dimensional binned check needs a numeric x covariate")
    if y_bins.edges is None:
        raise ValueError("a two-dimensional binned check needs a numeric y_cov covariate")

    cells = x_bins.codes * y_bins.n_groups + y_bins.codes
    size = x_bins.n_groups * y_bins.n_groups
    count = np.bincount(cells, minlength=size).astype(np.int64)
    total = np.bincount(cells, weights=values, minlength=size)
    mean = np.full(size, np.nan)
    np.divide(total, count, out=mean, where=count > 0)

    shape = (x_bins.n_groups, y_bins.n_groups)
    return BinnedCheck2D(
        covariates=(str(names[0]), str(names[1])),
        x_edges=_readonly(x_bins.edges),
        y_edges=_readonly(y_bins.edges),
        mean=_readonly(mean.reshape(shape)),
        count=_readonly(count.reshape(shape)),
    )


__all__ = [
    "BinnedCheck",
    "BinnedCheck2D",
    "binned_check",
    "binned_check_2d",
]
