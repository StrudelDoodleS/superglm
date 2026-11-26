"""The two plots that guide the choice of a large-loss threshold, and the numbers behind them.

:func:`mean_excess` is the empirical mean excess of the claims with a pointwise band
(the mean residual life plot). :func:`threshold_stability` fits a generalised Pareto tail
to the excesses over each candidate threshold and reports its shape and its threshold-free
scale with their bands (the parameter stability plot). Both are read before a
body-plus-tail model is fitted: the threshold is the lowest above which the mean excess
runs straight and the shape stays level within its band (Belzile, Dutang, Northrop and
Opitz, arXiv:2205.07714, section 2.6.1; Belzile and Davison, arXiv:2606.28540, section 3).

Estimates at neighbouring thresholds share most of their claims, so every band is for its
own point, not for the curve. Amounts are unit-free.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from statistics import NormalDist
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from superglm._experimental import warn_experimental
from superglm._frame import EagerFrame, FrameLike, as_eager_frame
from superglm._warn import warn_caller
from superglm.diagnostics.convergence import ConvergenceWarning
from superglm.distributional.api import SuperLSS, _renderer
from superglm.distributional.families.generalized_pareto import GeneralizedParetoLSS
from superglm.distributional.offsets import prediction_offsets, shape_fit_offsets, sources_of
from superglm.distributional.prediction_design import map_design_blocks
from superglm.distributional.rebuild import unfitted_clone
from superglm.distributional.splice import _aligned, _split_offsets
from superglm.distributional.surfaces import SCHEMA_VERSION, _json_frame, _json_scalar
from superglm.distributional.weights import WeightContract, _weight_values

__all__ = [
    "MeanExcessFigure",
    "MeanExcessResult",
    "TailLine",
    "ThresholdStabilityResult",
    "mean_excess",
    "plot_mean_excess",
    "plot_threshold_stability",
    "threshold_stability",
]

_FREQUENCY = WeightContract("frequency")
#: Levels of the default candidate thresholds, in hundredths: the 0.80 to 0.99 quantiles of the
#: complete losses (the grid of Belzile and Davison, arXiv:2606.28540, Table 6).
_DEFAULT_LEVELS = np.arange(80, 100)
_STABILITY_COLUMNS = (
    "threshold",
    "n_above",
    "n_censored",
    "shape",
    "shape_lower",
    "shape_upper",
    "scale",
    "modified_scale",
    "modified_scale_lower",
    "modified_scale_upper",
    "converged",
    "wall",
    "notes",
)
_ONE_TAIL = "one scale, one shape"
_SUPPLIED_BY_THE_FIT = ("truncation", "censored", "sample_weight")


# --------------------------------------------------------------------------- #
# Reading the claims
# --------------------------------------------------------------------------- #


class _Claims(NamedTuple):
    """The rows with a positive count: losses (or limits), counts, censoring flags, positions."""

    loss: NDArray[np.float64]
    counts: NDArray[np.float64]
    censored: NDArray[np.bool_]
    rows: NDArray[np.intp]


def _normal_quantile(level: float) -> float:
    """The two-sided standard normal quantile of a band at ``level``."""
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must lie strictly between 0 and 1, not {level!r}.")
    return NormalDist().inv_cdf(0.5 + 0.5 * level)


def _read_claims(
    y: ArrayLike, sample_weight: ArrayLike | None, censored: Any, n: int | None = None
) -> _Claims:
    """Check the claims row by row; rows with a zero count are dropped before any is read."""
    values = np.asarray(y)
    if values.ndim != 1 or n not in (None, len(values)):
        of_x = "" if n is None else f" of X ({n} rows)"
        raise ValueError(f"y must hold one loss per row{of_x}, not shape {values.shape}.")
    n = len(values)
    counts = _weight_values(
        None if sample_weight is None else np.asarray(sample_weight), n, _FREQUENCY
    )
    flags = np.asarray(
        False if censored is None else censored,
        dtype=object if isinstance(censored, list | tuple) else None,
    )
    if flags.dtype.kind == "O" and all(isinstance(flag, bool | np.bool_) for flag in flags.flat):
        flags = flags.astype(bool)
    if flags.dtype.kind != "b" or flags.shape not in ((), (n,)):
        raise ValueError("censored must be one boolean flag per row, or one for every row.")
    rows = np.flatnonzero(counts > 0)
    try:
        loss = np.asarray(values[rows], dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("Losses must be positive amounts, one number per row.") from error
    bad = ~(np.isfinite(loss) & (loss > 0.0))
    if np.any(bad):
        raise ValueError(
            f"Losses must be positive amounts; {np.count_nonzero(bad)} are missing or not "
            f"positive (first at row {rows[np.argmax(bad)]}). Drop or correct those rows."
        )
    return _Claims(loss, counts[rows], np.broadcast_to(flags, (n,))[rows], rows)


def _explicit_thresholds(thresholds: ArrayLike) -> NDArray[np.float64]:
    """The caller's candidates, sorted and distinct."""
    try:
        values = np.unique(np.asarray(thresholds, dtype=np.float64).ravel())
    except (TypeError, ValueError) as error:
        raise ValueError("thresholds must be finite, positive amounts.") from error
    if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("thresholds must be finite, positive amounts.")
    return values


def _one_threshold(threshold: float) -> float:
    return float(_explicit_thresholds([threshold])[0])


# --------------------------------------------------------------------------- #
# Empirical mean excess
# --------------------------------------------------------------------------- #


def _default_candidates(
    values: NDArray, above: NDArray, min_above: int, max_points: int
) -> NDArray[np.float64]:
    """The distinct losses with ``min_above`` claims above, thinned evenly in ``log n_above``."""
    keep = np.flatnonzero(above >= min_above)
    if len(keep) > max_points:
        targets = np.exp(np.linspace(np.log(above[0]), np.log(min_above), max_points))
        # the candidate with the fewest claims above that still has at least the target
        fewest = np.searchsorted(above[keep][::-1], targets, side="left")
        keep = np.unique(np.clip(len(keep) - 1 - fewest, 0, len(keep) - 1))
    return values[keep]


def _mean_excess_table(
    values: NDArray, counts: NDArray, candidates: NDArray, z: float
) -> pd.DataFrame:
    """Mean excess, standard error and band at each candidate, from one sweep down the losses.

    ``values`` are the distinct losses in increasing order. The claims between two consecutive
    candidates form a block. A block's moments are taken about its own smallest loss, and the
    sweep merges blocks from the top by the pooled-variance identity
    ``M2 = M2_a + M2_b + n_a n_b d^2 / (n_a + n_b)``, with ``d`` the distance between the two
    means, and the two means by their weighted average. Every term is non-negative and local,
    so a threshold far from the losses costs no digits. An empty block only shifts the running
    mean by the gap between its candidates.
    """
    start = np.searchsorted(values, candidates, side="right")
    stop = np.append(start[1:], len(values))
    filled = np.flatnonzero(stop > start)
    first = start[filled]
    block = np.repeat(np.arange(len(filled)), (stop - start)[filled])
    anchor = values[first]
    distance = values[first[0] :] - anchor[block]
    weight = counts[first[0] :]
    edges = first - first[0]
    n_block = np.add.reduceat(weight, edges)
    mean_block = np.add.reduceat(weight * distance, edges) / n_block
    m2_block = np.add.reduceat(weight * (distance - mean_block[block]) ** 2, edges)
    slot = np.full(len(candidates), -1)
    slot[filled] = np.arange(len(filled))

    n = np.empty(len(candidates))
    mean = np.empty(len(candidates))
    m2 = np.empty(len(candidates))
    n_run = mean_run = m2_run = 0.0
    upper = candidates[-1]
    for j in range(len(candidates) - 1, -1, -1):
        # the claims already swept, measured from this candidate
        shifted = mean_run + (upper - candidates[j])
        b = slot[j]
        if b >= 0:
            total = n_run + n_block[b]
            # how far the swept claims' mean lies above the block's
            apart = mean_run + ((upper - anchor[b]) - mean_block[b])
            m2_run += m2_block[b] + n_run * n_block[b] * apart * apart / total
            own = (anchor[b] - candidates[j]) + mean_block[b]
            shifted = (n_run * shifted + n_block[b] * own) / total
            n_run = total
        mean_run, upper = shifted, candidates[j]
        n[j], mean[j], m2[j] = n_run, mean_run, m2_run
    se = np.sqrt(m2 / (n * (n - 1.0)))
    return pd.DataFrame(
        {
            "threshold": candidates,
            "mean_excess": mean,
            "se": se,
            "lower": mean - z * se,
            "upper": mean + z * se,
            "n_above": n.astype(np.int64),
        }
    )


@dataclass(frozen=True)
class TailLine:
    """The mean excess of a generalised Pareto tail (one scale, one shape) fitted above a threshold.

    ``x`` is the threshold and the largest candidate of the table; ``y`` is
    ``(scale + shape (x - threshold)) / (1 - shape)`` at both. ``converged`` and ``wall`` are
    the fit's own flags, as in :class:`ThresholdStabilityResult`.
    """

    threshold: float
    scale: float
    shape: float
    n_above: int
    converged: bool
    wall: str | None
    x: tuple[float, float]
    y: tuple[float, float]

    def to_json(self) -> dict[str, Any]:
        """Return the line as JSON values."""
        return {
            name: [_json_scalar(item) for item in value]
            if isinstance(value, tuple)
            else _json_scalar(value)
            for name, value in asdict(self).items()
        }


@dataclass(frozen=True, eq=False)
class MeanExcessFigure:
    """What :func:`plot_mean_excess` draws: the table, the fitted line and where to shade.

    ``shaded_from`` is the smallest candidate with fewer than ``few_claims`` claims above
    it, or ``None`` when every candidate has that many.
    """

    table: pd.DataFrame
    level: float
    threshold: float | None
    line: TailLine | None
    shaded_from: float | None
    few_claims: int
    kind: str = "mean_excess"
    schema_version: int = SCHEMA_VERSION

    def to_json(self) -> dict[str, Any]:
        """Return the figure's numbers as JSON values."""
        return {
            "kind": self.kind,
            "schema_version": self.schema_version,
            "level": self.level,
            "table": _json_frame(self.table),
            "threshold": self.threshold,
            "line": None if self.line is None else self.line.to_json(),
            "shaded_from": self.shaded_from,
            "few_claims": self.few_claims,
        }


@dataclass(frozen=True, eq=False)
class MeanExcessResult:
    """Empirical mean excess table and settings returned by :func:`mean_excess`.

    Keeps the distinct positive losses and their frequency counts sorted
    internally so :meth:`tail_line` and :func:`plot_mean_excess` can fit a tail
    above any chosen threshold; those stored losses are omitted from
    :meth:`to_json`.

    Attributes
    ----------
    table : DataFrame
        One row per retained candidate threshold with columns ``threshold``,
        ``mean_excess``, ``se``, ``lower``, ``upper``, and ``n_above`` (claims
        strictly above ``threshold``). The band endpoints are unclipped
        pointwise normal-theory intervals and can dip below zero where few
        claims remain.
    dropped : DataFrame
        Explicitly requested candidates left out because fewer than
        ``min_above`` claims lie strictly above them, with columns
        ``threshold`` and ``n_above``.
    level : float
        Nominal pointwise coverage of ``lower`` and ``upper``.
    n_claims : int
        Total frequency of the positive-weight claims read.
    min_above : int
        Minimum number of claims strictly above a retained candidate.
    kind : str
        Payload identifier (``"mean_excess"``).
    schema_version : int
        Schema version of :meth:`to_json`.
    """

    table: pd.DataFrame
    dropped: pd.DataFrame
    level: float
    n_claims: int
    min_above: int
    _losses: NDArray[np.float64] = field(repr=False)
    _counts: NDArray[np.float64] = field(repr=False)
    kind: str = "mean_excess"
    schema_version: int = SCHEMA_VERSION

    def tail_line(self, threshold: float) -> TailLine:
        """Fit a one-scale, one-shape generalized Pareto tail above ``threshold``.

        Uses the same engine and default walls as :func:`threshold_stability`
        on the sorted losses stored in this result, returning the straight mean
        excess line from ``threshold`` to the largest candidate. Refuses with
        ``ValueError`` if ``threshold`` is not finite and positive or leaves
        fewer than ``min_above`` claims strictly above it. Emits a
        ``ConvergenceWarning`` and flags ``converged`` / ``wall`` on the
        returned ``TailLine`` if the fit did not converge or sits at a shape
        wall.
        """
        u = _one_threshold(threshold)
        keep = self._losses > u
        counts = self._counts[keep]
        n_above = int(counts.sum())
        if n_above < self.min_above:
            raise ValueError(
                f"The tail line needs {self.min_above} or more claims above its threshold "
                f"(min_above), and {n_above} lie above {u:g}. Pass a lower threshold."
            )
        model = _classical_tail()
        rows = np.arange(np.count_nonzero(keep))
        _, foreign = _fit_quietly(
            model, "fit", pd.DataFrame(index=rows), self._losses[keep] - u, counts, None, rows, {}
        )
        _reemit(foreign)
        fit = _estimates(model, u, _normal_quantile(self.level), None)
        if not fit["converged"] or fit["wall"] is not None:
            warn_caller(
                f"The tail fitted above {u:g} for the mean excess line did not converge or sits "
                "at a wall of the shape, so the line is set by where the fit stopped, not by "
                "the claims alone. Its converged and wall fields say which. Try another "
                "threshold.",
                ConvergenceWarning,
            )
        scale, shape = fit["scale"], fit["shape"]
        x = (u, max(u, float(self.table["threshold"].iloc[-1])))
        return TailLine(
            threshold=u,
            scale=scale,
            shape=shape,
            n_above=n_above,
            converged=fit["converged"],
            wall=fit["wall"],
            x=x,
            y=(scale / (1.0 - shape), (scale + shape * (x[1] - u)) / (1.0 - shape)),
        )

    def figure(self, threshold: float | None = None, *, few_claims: int = 50) -> MeanExcessFigure:
        """Return what :func:`plot_mean_excess` draws; ``threshold`` adds the fitted line."""
        line = None if threshold is None else self.tail_line(threshold)
        few = self.table.loc[self.table["n_above"] < few_claims, "threshold"]
        return MeanExcessFigure(
            table=self.table,
            level=self.level,
            threshold=None if line is None else line.threshold,
            line=line,
            shaded_from=None if few.empty else float(few.iloc[0]),
            few_claims=few_claims,
        )

    def to_json(self) -> dict[str, Any]:
        """Return the table and its settings as JSON values."""
        return {
            "kind": self.kind,
            "schema_version": self.schema_version,
            "level": self.level,
            "n_claims": self.n_claims,
            "min_above": self.min_above,
            "table": _json_frame(self.table),
            "dropped": _json_frame(self.dropped),
        }


def mean_excess(
    y: ArrayLike,
    sample_weight: ArrayLike | None = None,
    *,
    thresholds: ArrayLike | None = None,
    censored: ArrayLike | None = None,
    level: float = 0.95,
    max_points: int = 200,
    min_above: int = 5,
) -> MeanExcessResult:
    """Return the empirical mean excess of the claims at each candidate threshold.

    At each candidate threshold ``u``, computes the mean of ``y - u`` over
    claims strictly above ``u``, its standard error, and a pointwise
    normal-theory band at ``level``. The band assumes finite excess variance
    (a tail shape below 1/2), the plot has no finite limit at shape 1 or above,
    and estimates at neighbouring thresholds share most of their claims.
    Experimental: its name, arguments and return layout may change in a minor
    release without a deprecation period; its numbers are tested like the rest
    of superglm.

    Parameters
    ----------
    y : array-like of shape (n_claims,)
        Positive unit-free loss amounts.
    sample_weight : array-like of shape (n_claims,), optional
        Nonnegative integer frequency counts: a count of ``k`` reads as ``k``
        identical claims. Rows with a zero count are dropped before losses are
        validated.
    thresholds : array-like, optional
        Explicit candidate thresholds (sorted and deduplicated). By default,
        the distinct losses with at least ``min_above`` claims strictly above,
        thinned to at most ``max_points`` evenly in ``log(n_above)``.
    censored : array-like of bool, optional
        Accepted only to refuse flagged rows with a pointer to
        :func:`threshold_stability`: this function reads complete losses only.
    level : float, default=0.95
        Nominal pointwise coverage of the band, strictly between 0 and 1.
    max_points : int, default=200
        Maximum number of candidates on the default grid (at least 2).
    min_above : int, default=5
        Minimum number of claims strictly above a candidate (at least 2);
        trimming the top five claims is a convention. Explicit candidates with
        fewer claims above go to ``result.dropped``.

    Returns
    -------
    MeanExcessResult
        Table of mean excess estimates, dropped candidates, and settings.

    Raises
    ------
    ValueError
        If ``level`` is not in ``(0, 1)``, ``min_above < 2`` or
        ``max_points < 2``, any positive-weight row is flagged censored, any
        loss is missing or not positive, explicit ``thresholds`` are empty or
        not finite and positive, or no candidate leaves at least ``min_above``
        claims strictly above it.

    Notes
    -----
    Exceedance is strict (``y > u``), so ties at ``u`` are not counted as zero
    excesses. Standard errors use the sample variance with divisor
    ``n_above - 1``, computed in one ``O(n log n)`` downward sweep with locally
    centred block moments so large common offsets lose no digits. Band
    endpoints are unclipped normal-theory intervals and can dip below zero when
    few claims remain.
    """
    warn_experimental("mean_excess")
    z = _normal_quantile(level)
    if min_above < 2 or max_points < 2:
        raise ValueError(
            "min_above and max_points must each be 2 or more: a standard error needs two "
            "claims above a threshold."
        )
    claims = _read_claims(y, sample_weight, censored)
    if np.any(claims.censored):
        raise ValueError(
            f"mean_excess reads complete losses, and {np.count_nonzero(claims.censored)} rows "
            f"are flagged censored (first at row {claims.rows[np.argmax(claims.censored)]}). A "
            "censored loss is at least its limit, so how far it exceeds a threshold is not "
            "known. Use threshold_stability(..., censored=...), which reads censored rows, or "
            "pass the complete rows only."
        )
    values, inverse = np.unique(claims.loss, return_inverse=True)
    counts = np.bincount(inverse, weights=claims.counts, minlength=len(values))
    # claims at or above each distinct loss, then none: an exact count at every position
    at_or_above = np.append(np.cumsum(counts[::-1])[::-1], 0.0)
    dropped = pd.DataFrame({"threshold": np.empty(0), "n_above": np.empty(0, dtype=np.int64)})
    if thresholds is None:
        candidates = _default_candidates(values, at_or_above[1:], min_above, max_points)
    else:
        asked = _explicit_thresholds(thresholds)
        n_asked = at_or_above[np.searchsorted(values, asked, side="right")].astype(np.int64)
        few = n_asked < min_above
        dropped = pd.DataFrame({"threshold": asked[few], "n_above": n_asked[few]})
        candidates = asked[~few]
    if candidates.size == 0:
        raise ValueError(
            f"No candidate threshold leaves {min_above} or more claims above it. Pass lower "
            "thresholds or a smaller min_above."
        )
    return MeanExcessResult(
        table=_mean_excess_table(values, counts, candidates, z),
        dropped=dropped,
        level=float(level),
        n_claims=int(at_or_above[0]),
        min_above=int(min_above),
        _losses=values,
        _counts=counts,
    )


def plot_mean_excess(
    result: MeanExcessResult,
    threshold: float | None = None,
    *,
    few_claims: int = 50,
    engine: str = "matplotlib",
) -> Any:
    """Draw the empirical mean excess points and pointwise band from :func:`mean_excess`.

    Plots the observed mean excess and its pointwise normal-theory band across
    candidate thresholds, shading the right-hand stretch where fewer than
    ``few_claims`` claims remain. Passing ``threshold`` marks it and overlays
    the straight mean excess line of a classical generalized Pareto tail (one
    scale, one shape) fitted to the excesses above it on the sorted losses
    stored in ``result``; if that fit did not converge or sits at a wall, the
    line is drawn dashed and a ``ConvergenceWarning`` is emitted. Experimental:
    its name, arguments and return layout may change in a minor release without
    a deprecation period; its numbers are tested like the rest of superglm.

    Parameters
    ----------
    result : MeanExcessResult
        Output of :func:`mean_excess`.
    threshold : float, optional
        Chosen candidate threshold to mark with a vertical line and fitted
        generalized Pareto mean excess line.
    few_claims : int, default=50
        Shade the plot from the first candidate with fewer than ``few_claims``
        claims strictly above it.
    engine : {"matplotlib", "plotly"}, default="matplotlib"
        Plotting backend.

    Returns
    -------
    Figure
        Matplotlib or Plotly figure.

    Raises
    ------
    ValueError
        If ``threshold`` is not finite and positive, leaves fewer than
        ``result.min_above`` claims strictly above it, or ``engine`` is unknown.
    """
    warn_experimental("plot_mean_excess")
    return _renderer(engine, "mean_excess")(result.figure(threshold, few_claims=few_claims))


# --------------------------------------------------------------------------- #
# Threshold stability
# --------------------------------------------------------------------------- #


class _Book(NamedTuple):
    """The rows a row-varying shape is averaged over: every claim read, with its count."""

    frame: EagerFrame
    weights: NDArray[np.float64]
    offsets: Mapping[str, NDArray[np.float64]] | None


def _classical_tail() -> SuperLSS:
    """An unfitted tail with one scale and one shape, read with frequency counts."""
    family = GeneralizedParetoLSS()
    return SuperLSS(family, family.scale(), family.shape(), weight_semantics="frequency")


def _is_one_number(predictor: Any) -> bool:
    """Whether a parameter is the same on every row: an intercept, no term and no offset."""
    return bool(
        predictor.intercept
        and not predictor.features
        and not predictor.interaction_specs
        and not sources_of(predictor)
    )


def _describe(tail: SuperLSS) -> str:
    """The supplied tail's predictors in one line, for the plot's caption."""
    parts = []
    for predictor in tail._predictors:
        terms = [
            *(["1"] if predictor.intercept else []),
            *map(str, (*predictor.features, *predictor.interaction_specs)),
            *(["offset"] if sources_of(predictor) else []),
        ]
        parts.append(f"{predictor.name} ~ {' + '.join(terms) or '0'}")
    return f"refit of the supplied tail at each threshold ({'; '.join(parts)})"


def _fit_plan(
    tail: SuperLSS | None, reml: bool | None, fit_options: Mapping[str, Any] | None
) -> tuple[str, dict[str, Any]]:
    """The fit method and its keywords: the caller's, over a fitted tail's recorded settings."""
    options = dict(fit_options or {})
    for key in _SUPPLIED_BY_THE_FIT:
        if key in options:
            raise ValueError(
                f"threshold_stability supplies {key} to each fit; remove it from fit_options."
            )
    recorded = None if tail is None else tail._fit_settings
    recorded_method = None if recorded is None else recorded["method"]
    method = (
        recorded_method if reml is None and recorded is not None else "fit_reml" if reml else "fit"
    )
    if recorded is not None and method == recorded_method:
        options = {**{k: v for k, v in recorded.items() if k != "method"}, **options}
    return method, options


def _array_offsets(
    tail: SuperLSS, frame: EagerFrame, rows: NDArray[np.intp]
) -> dict[str, NDArray[np.float64]] | None:
    """Each predictor's offset arrays summed on ``rows``: what a prediction takes in their place."""
    shaped = shape_fit_offsets(tail._predictors, frame)
    replacements = {}
    for predictor in tail._predictors:
        arrays = [
            np.asarray(values[rows], dtype=np.float64)
            for source, values in zip(sources_of(predictor), shaped[predictor.name], strict=True)
            if not isinstance(source, str)
        ]
        if arrays:
            replacements[predictor.name] = np.sum(arrays, axis=0)
    return replacements or None


def _fit_quietly(
    model: SuperLSS,
    method: str,
    X: FrameLike,
    excess: NDArray,
    counts: NDArray | None,
    censored: NDArray | None,
    rows: NDArray[np.intp],
    options: Mapping[str, Any],
) -> tuple[list[str], list[warnings.WarningMessage]]:
    """Fit ``model`` on one candidate's rows with every warning recorded instead of shown.

    A local ``always`` filter keeps an ambient ``error`` filter from aborting the fit. The
    fit's own warnings (a class superglm defines) come back as texts, merged with
    ``result_.warnings``; any other warning comes back whole, for the caller to emit again.
    """
    with warnings.catch_warnings(record=True) as caught, _split_offsets(model, rows):
        warnings.simplefilter("always")
        getattr(model, method)(X, excess, counts, censored=censored, **options)
    own = [str(w.message) for w in caught if _is_fit_warning(w)]
    notes = list(dict.fromkeys((*model.result_.warnings, *own)))
    return notes, [w for w in caught if not _is_fit_warning(w)]


def _is_fit_warning(warning: warnings.WarningMessage) -> bool:
    return warning.category.__module__.partition(".")[0] == "superglm"


def _reemit(caught: list[warnings.WarningMessage]) -> None:
    for warning in caught:
        warnings.warn_explicit(warning.message, warning.category, warning.filename, warning.lineno)


def _book_mean_shape(fitted: Any, state: Any, book: _Book) -> tuple[float, NDArray[np.float64]]:
    """The book's count-weighted mean fitted shape, and its gradient in the shape's coefficients.

    The gradient of ``sum(w_i h(eta_i)) / sum(w_i)`` is ``sum(w_i h'(eta_i) x_i) / sum(w_i)``,
    with ``x_i`` the row of the shape's design and ``h`` the inverse link.
    """
    offset = prediction_offsets(fitted.fit_state.predictor_templates, book.frame, book.offsets)[
        state.name
    ]
    beta = fitted.coefficients[state.coefficient_slice]
    share = book.weights / book.weights.sum()

    def block(rows: slice, local: Mapping[str, NDArray]) -> tuple[Any, ...]:
        design = local[state.name]
        eta = design @ beta + offset[rows]
        values = state.link.inverse(eta)
        slope = share[rows] * state.link.deriv_inverse(eta)
        return share[rows] @ values, slope @ design, values.min(), values.max()

    parts = map_design_blocks(book.frame, fitted.compiled_predictors, fitted.layout, block)
    mean = math.fsum(part[0] for part in parts)
    # a mean lies between its smallest and largest term, which rounding must not undo at a wall
    mean = min(max(mean, min(part[2] for part in parts)), max(part[3] for part in parts))
    return float(mean), np.sum([part[1] for part in parts], axis=0)


def _estimates(model: SuperLSS, threshold: float, z: float, book: _Book | None) -> dict[str, Any]:
    """The shape, the scale and the threshold-free scale of one fitted tail, with bands and flags.

    The shape's band is a Wald interval on its link's scale, mapped back, so it stays inside
    the walls. A shape that varies by row is reported as its mean over ``book``, with the
    delta-method standard error of that mean carried to the link's scale. The threshold-free
    scale ``scale - shape * threshold`` is reported when both are one number, with a
    delta-method band from the two links' derivatives at the fitted predictors.
    """
    fitted = model._require_fitted()
    result = fitted.fitted_result
    beta, covariance = result.coefficients, result.covariance
    states = {state.name: state for state in fitted.layout.predictors}
    one_number = {predictor.name: _is_one_number(predictor) for predictor in model._predictors}
    scale, shape = states["scale"], states["shape"]
    i, j = scale.intercept_index, shape.intercept_index
    if one_number["shape"]:
        eta = float(beta[j])
        xi = float(shape.link.inverse(eta))
        se_eta = math.sqrt(covariance[j, j])
    else:
        assert book is not None
        xi, gradient = _book_mean_shape(fitted, shape, book)
        block = covariance[shape.coefficient_slice, shape.coefficient_slice]
        eta = float(shape.link.link(xi))
        se_eta = math.sqrt(max(gradient @ block @ gradient, 0.0)) * float(shape.link.deriv(xi))
    ends = np.asarray(shape.link.inverse(np.array([eta - z * se_eta, eta + z * se_eta])))
    nan = float("nan")
    row = {
        "shape": xi,
        "shape_lower": float(ends.min()),
        "shape_upper": float(ends.max()),
        "scale": float(scale.link.inverse(beta[i])) if one_number["scale"] else nan,
        "modified_scale": nan,
        "modified_scale_lower": nan,
        "modified_scale_upper": nan,
        "converged": bool(result.converged),
        "wall": None if result.wall is None else result.wall.side,
    }
    if one_number["scale"] and one_number["shape"]:
        gradient = np.array(
            [scale.link.deriv_inverse(beta[i]), -threshold * shape.link.deriv_inverse(beta[j])],
            dtype=np.float64,
        )
        se = math.sqrt(max(gradient @ covariance[np.ix_([i, j], [i, j])] @ gradient, 0.0))
        modified = row["scale"] - xi * threshold
        row.update(
            modified_scale=modified,
            modified_scale_lower=modified - z * se,
            modified_scale_upper=modified + z * se,
        )
    return row


@dataclass(frozen=True, eq=False)
class ThresholdStabilityResult:
    """Generalized Pareto tail fits across candidate thresholds from :func:`threshold_stability`.

    Attributes
    ----------
    table : DataFrame
        One row per fitted candidate threshold with columns:

        - ``threshold``: candidate threshold ``u``.
        - ``n_above``: total frequency of kept claims (complete with ``y > u``
          plus censored with limit ``c >= u``).
        - ``n_censored``: kept censored claims among ``n_above``.
        - ``shape``, ``shape_lower``, ``shape_upper``: fitted tail shape (or
          frequency-weighted mean fitted shape over all read claims when shape
          varies across rows) and its pointwise link-scale Wald band.
        - ``scale``: fitted tail scale when scale is constant across rows, else
          ``NaN``.
        - ``modified_scale``, ``modified_scale_lower``,
          ``modified_scale_upper``: threshold-free scale
          ``scale - shape * threshold`` and its pointwise delta-method band
          when both scale and shape are constant across rows, else ``NaN``.
        - ``converged``: whether that candidate's fit converged.
        - ``wall``: ``"lower"``, ``"upper"``, ``"both"``, or empty
          (``None`` / ``NaN``; test with ``pandas.notna``) where the fit is not
          at a wall of the shape.
        - ``notes``: fit warnings and any count of censored claims left out
          below ``threshold``, joined with ``" | "``.
    dropped : DataFrame
        Candidates not fitted (fewer than ``min_claims`` complete claims above,
        or a higher candidate whose fit raised ``ValueError``), with columns
        ``threshold``, ``n_above``, ``n_censored``, and ``reason``.
    level : float
        Nominal pointwise coverage of the bands.
    min_claims : int
        Minimum number of complete claims strictly above a fitted candidate.
    tail : str
        Description of the fitted tail model (``"one scale, one shape"`` or the
        supplied tail's predictors).
    shape_label : str
        Axis label for the shape panel: ``"tail shape"`` when constant, or
        ``"mean fitted shape over the book"`` when shape varies across rows.
    shape_walls : tuple of (float, float)
        Lower and upper walls of the family's bounded shape link.
    kind : str
        Payload identifier (``"threshold_stability"``).
    schema_version : int
        Schema version of :meth:`to_json`.
    """

    table: pd.DataFrame
    dropped: pd.DataFrame
    level: float
    min_claims: int
    tail: str
    shape_label: str
    shape_walls: tuple[float, float]
    kind: str = "threshold_stability"
    schema_version: int = SCHEMA_VERSION

    def to_json(self) -> dict[str, Any]:
        """Return the tables and their settings as JSON values."""
        return {
            "kind": self.kind,
            "schema_version": self.schema_version,
            "level": self.level,
            "min_claims": self.min_claims,
            "tail": self.tail,
            "shape_label": self.shape_label,
            "shape_walls": list(self.shape_walls),
            "table": _json_frame(self.table),
            "dropped": _json_frame(self.dropped),
        }


def _default_stability_thresholds(claims: _Claims) -> NDArray[np.float64]:
    """The 0.80 to 0.99 quantiles of the complete losses, by count.

    The quantile at ``k / 100`` is the smallest loss whose cumulative count reaches that
    share of the total (the inverted distribution function). The comparison is between
    exact integers, so counts and repeated rows give the same thresholds.
    """
    complete = ~claims.censored
    values, inverse = np.unique(claims.loss[complete], return_inverse=True)
    if values.size == 0:
        return values
    reached = np.cumsum(np.bincount(inverse, weights=claims.counts[complete]))
    at = np.searchsorted(100.0 * reached, _DEFAULT_LEVELS * reached[-1], side="left")
    return np.unique(values[at])


def threshold_stability(
    X: FrameLike | None,
    y: ArrayLike,
    thresholds: ArrayLike | None = None,
    *,
    tail: SuperLSS | None = None,
    sample_weight: ArrayLike | None = None,
    censored: ArrayLike = False,
    level: float = 0.95,
    min_claims: int = 20,
    reml: bool | None = None,
    fit_options: Mapping[str, Any] | None = None,
) -> ThresholdStabilityResult:
    """Fit a generalized Pareto tail to excesses over each candidate threshold.

    By default fits one scale and one shape for all claims (the classical
    parameter stability plot) and reports both the tail shape and the
    threshold-free modified scale ``scale - shape * threshold`` with pointwise
    bands. Passing ``tail`` refits a fresh copy of that specification at every
    candidate without modifying ``tail``. Experimental: its name, arguments and
    return layout may change in a minor release without a deprecation period;
    its numbers are tested like the rest of superglm.

    Parameters
    ----------
    X : DataFrame or None
        Columns read by ``tail``'s terms or column offsets. May be ``None``
        when ``tail`` is ``None``.
    y : array-like of shape (n_claims,)
        Positive unit-free loss amounts, or the policy limit for a censored
        claim.
    thresholds : array-like, optional
        Explicit candidate thresholds (sorted and deduplicated). By default,
        the 0.80, 0.81, ..., 0.99 inverted-ECDF quantiles of the complete
        losses (weighted by ``sample_weight``).
    tail : SuperLSS, optional
        A ``SuperLSS`` using ``GeneralizedParetoLSS``. A fitted tail is
        refitted with its recorded method (``fit`` or ``fit_reml``) and
        settings unless overridden by ``reml`` or ``fit_options``; an unfitted
        tail uses ``reml``. ``tail`` itself is never fitted or modified.
    sample_weight : array-like of shape (n_claims,), optional
        Nonnegative integer frequency counts. Rows with a zero count are
        dropped before losses are validated. Non-unit counts require
        ``weight_semantics="frequency"`` on ``tail``.
    censored : bool or array-like of bool, default=False
        Claims settled at a limit, whose ``y`` is that limit. At candidate
        ``u``, censored claims with limit at least ``u`` are kept as
        right-censored excesses ``limit - u``; censored claims with limit below
        ``u`` are left out at ``u`` and counted in ``notes``, which assumes
        limits do not depend on the loss.
    level : float, default=0.95
        Nominal pointwise coverage of the bands, strictly between 0 and 1.
    min_claims : int, default=20
        Minimum number of complete claims strictly above a candidate for it to
        be fitted; candidates with fewer go to ``result.dropped``.
    reml : bool, optional
        ``True`` fits with ``fit_reml`` and ``False`` with ``fit``. When
        ``None`` (default), a fitted ``tail`` keeps its recorded method and
        anything else uses ``fit``.
    fit_options : mapping, optional
        Extra keyword arguments passed to each fit. Must not contain
        ``"truncation"``, ``"censored"``, or ``"sample_weight"``, which
        ``threshold_stability`` supplies.

    Returns
    -------
    ThresholdStabilityResult
        One row per fitted candidate in ``table``, plus any unattempted or
        dropped candidates in ``dropped``.

    Raises
    ------
    TypeError
        If ``tail`` is not a ``SuperLSS`` with ``GeneralizedParetoLSS``, or
        ``tail`` is provided with ``X=None``.
    ValueError
        If ``level`` is not in ``(0, 1)``, ``fit_options`` includes a managed
        keyword, inputs are misaligned or invalid, non-unit ``sample_weight``
        is passed with a prior-weight ``tail``, no candidate leaves at least
        ``min_claims`` complete claims above it, or the lowest candidate's fit
        raises ``ValueError``.

    Notes
    -----
    Candidates whose fit did not converge or sits at a wall of the shape
    (default walls are 0 and 1, so a tail with no positive shape sits at the
    lower wall) remain in ``table``, flagged by ``converged`` and ``wall``;
    per-fit superglm warnings are collected into ``notes`` and summarised by a
    single ``ConvergenceWarning`` after the loop. Non-superglm warnings are
    re-emitted. At candidates above the lowest, a ``ValueError`` from the fit
    (such as a categorical level disappearing) moves that candidate to
    ``dropped`` with the engine's message.

    When scale has terms or an offset, no threshold-free scale exists and
    ``scale`` / ``modified_scale`` are ``NaN``; a log-link scale model does not
    keep its functional form as the threshold changes, so drift in the shape
    can reflect the scale's form as well as the tail. When shape has terms or
    an offset, ``shape`` is the frequency-weighted mean fitted shape over all
    positive-weight rows of ``X`` (the same reference book at every candidate),
    with a link-scale delta-method band.
    """
    warn_experimental("threshold_stability")
    z = _normal_quantile(level)
    if tail is not None and not (
        isinstance(tail, SuperLSS) and isinstance(tail.family, GeneralizedParetoLSS)
    ):
        raise TypeError("tail must be a SuperLSS with GeneralizedParetoLSS")
    if tail is not None and X is None:
        raise TypeError(
            "threshold_stability refits the supplied tail on the rows of X, and X is None. "
            "Pass the frame the tail's terms read."
        )
    method, options = _fit_plan(tail, reml, fit_options)
    frame = None if X is None else as_eager_frame(X)
    if frame is not None:
        for name, values in (("y", y), ("censored", censored), ("sample_weight", sample_weight)):
            _aligned(frame, values, name)
    claims = _read_claims(y, sample_weight, censored, None if frame is None else len(frame))
    if tail is not None and tail.weight_semantics != "frequency" and np.any(claims.counts != 1):
        raise ValueError(
            "sample_weight holds counts other than one, and the supplied tail reads weights "
            "as 'prior'. Build the tail with SuperLSS(..., weight_semantics='frequency')."
        )
    family = GeneralizedParetoLSS() if tail is None else tail.family
    book = None
    if tail is not None and not _is_one_number(
        next(predictor for predictor in tail._predictors if predictor.name == "shape")
    ):
        whole = len(claims.rows) == len(frame)
        book = _Book(
            frame if whole else as_eager_frame(frame.take_rows(claims.rows)),
            claims.counts,
            _array_offsets(tail, frame, claims.rows),
        )
    candidates = (
        _default_stability_thresholds(claims)
        if thresholds is None
        else _explicit_thresholds(thresholds)
    )

    records: list[dict[str, Any]] = []
    dropped: list[tuple[float, int, int, str]] = []
    foreign: list[warnings.WarningMessage] = []
    lowest = True
    for u in candidates:
        # complete losses above the candidate, and censored ones whose limit reaches it
        keep = np.flatnonzero(np.where(claims.censored, claims.loss >= u, claims.loss > u))
        limited = claims.censored[keep]
        n_above = int(claims.counts[keep].sum())
        n_censored = int(claims.counts[keep][limited].sum())
        if n_above - n_censored < min_claims:
            dropped.append(
                (float(u), n_above, n_censored, f"fewer than {min_claims} complete claims above")
            )
            continue
        model = _classical_tail() if tail is None else unfitted_clone(tail)
        rows = claims.rows[keep]
        try:
            notes, other = _fit_quietly(
                model,
                method,
                pd.DataFrame(index=np.arange(len(keep))) if tail is None else frame.take_rows(rows),
                claims.loss[keep] - u,
                None if sample_weight is None else claims.counts[keep],
                limited if np.any(limited) else None,
                rows,
                options,
            )
        except ValueError as error:
            if lowest:
                raise
            dropped.append((float(u), n_above, n_censored, str(error)))
            continue
        finally:
            lowest = False
        foreign.extend(other)
        left_out = int(claims.counts[claims.censored & (claims.loss < u)].sum())
        if left_out:
            notes.append(
                f"{left_out} censored claims with a limit below this threshold were left out"
            )
        records.append(
            {
                "threshold": float(u),
                "n_above": n_above,
                "n_censored": n_censored,
                **_estimates(model, float(u), z, book),
                "notes": " | ".join(notes),
            }
        )
    _reemit(foreign)
    if not records:
        most = max(dropped, key=lambda item: item[1] - item[2], default=None)
        raise ValueError(
            f"No candidate threshold leaves {min_claims} or more complete claims above it"
            + ("" if most is None else f"; the most is {most[1] - most[2]}, above {most[0]:g}")
            + ". Pass lower thresholds or a smaller min_claims."
        )
    table = pd.DataFrame(records, columns=list(_STABILITY_COLUMNS))
    flagged = int((~table["converged"] | table["wall"].notna()).sum())
    if flagged:
        warn_caller(
            f"threshold_stability: {flagged} of {len(table)} candidate thresholds have a tail "
            "fit that did not converge or sits at a wall of the shape. The table's converged "
            "and wall columns name them and its notes column says why; their estimates and "
            "bands are set by where the fit stopped, not by the claims alone.",
            ConvergenceWarning,
        )
    return ThresholdStabilityResult(
        table=table,
        dropped=pd.DataFrame(dropped, columns=["threshold", "n_above", "n_censored", "reason"]),
        level=float(level),
        min_claims=int(min_claims),
        tail=_ONE_TAIL if tail is None else _describe(tail),
        shape_label="tail shape" if book is None else "mean fitted shape over the book",
        shape_walls=(float(family.shape_lower), float(family.shape_upper)),
    )


def plot_threshold_stability(
    result: ThresholdStabilityResult, *, engine: str = "matplotlib"
) -> Any:
    """Draw the tail shape and threshold-free scale from :func:`threshold_stability`.

    Plots each candidate threshold's fitted shape (or mean fitted shape over
    the book when shape varies across rows) with its pointwise band and shape
    walls, alongside the threshold-free modified scale panel whenever scale and
    shape are both constant. Candidates whose fit did not converge or sits at a
    wall are drawn hollow. Experimental: its name, arguments and return layout
    may change in a minor release without a deprecation period; its numbers are
    tested like the rest of superglm.

    Parameters
    ----------
    result : ThresholdStabilityResult
        Output of :func:`threshold_stability`.
    engine : {"matplotlib", "plotly"}, default="matplotlib"
        Plotting backend.

    Returns
    -------
    Figure
        Matplotlib or Plotly figure.

    Raises
    ------
    ValueError
        If ``engine`` is unknown.
    """
    warn_experimental("plot_threshold_stability")
    return _renderer(engine, "threshold_stability")(result)
