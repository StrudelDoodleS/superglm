"""Paired comparison of two fitted distributional models, and the Murphy diagram.

Two models scored on the same rows are compared by their per-row score
differences: the rows are conditionally independent given the covariates, so the
paired mean, standard error and ``t`` are the whole test -- with literal row
replication under frequency weights, and no autocorrelation correction, which
would be a forecasting-over-time construct this design does not have.

A single mean says *which* model wins.  The Murphy diagram of Ehm, Gneiting,
Jordan and Krueger (2016), *Journal of the Royal Statistical Society: Series B*
78(3), 505-562, says *where*.  Every consistent scoring function for the
``alpha``-quantile is a mixture of the elementary scores

    S_alpha^eta(x, y) = (1 - alpha) 1{y <= eta < x} + alpha 1{x <= eta < y},

so plotting the mean elementary score of both models against ``eta`` decomposes
the quantile (pinball) loss over the outcome scale: one curve below the other on
a stretch of ``eta`` is dominance on that stretch, and the curves integrate back
to the pinball losses being compared.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from superglm._frame import EagerFrame, FrameLike, as_eager_frame
from superglm.distributional.checks.scores import (
    _retained_crps,
    _retained_log_score,
    _row_law,
    _scoring_rows,
    _ScoringRows,
)
from superglm.distributional.family import DistributionFunctionFamily
from superglm.distributional.weights import (
    LikelihoodWeightError,
    UnsupportedLikelihoodContractError,
    WeightContract,
    resolve_likelihood_weights,
)

ScoreName = Literal["log", "crps"]

#: Threshold grid size when a Murphy diagram is asked for without one.
_DEFAULT_MURPHY_POINTS = 101
#: Elementary scores are evaluated in blocks of about this many (threshold, row)
#: cells, so a fine grid over a long book never materialises one dense matrix.
_MURPHY_BLOCK_CELLS = 4_000_000


def _json_numbers(values: NDArray) -> list[float | None]:
    """Emit a float list with every non-finite entry as ``null``."""
    return [None if not np.isfinite(value) else float(value) for value in np.asarray(values)]


def _json_number(value: float) -> float | None:
    return None if not np.isfinite(value) else float(value)


def _readonly(values: NDArray) -> NDArray[np.float64]:
    array = np.array(values, dtype=np.float64, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class MurphyPayload:
    """Mean elementary quantile scores of two models over a threshold grid."""

    level: float
    thresholds: NDArray[np.float64]
    a: NDArray[np.float64]
    b: NDArray[np.float64]
    difference: NDArray[np.float64]
    difference_se: NDArray[np.float64]
    n_observations: int
    kind: str = "murphy"
    schema_version: int = 1

    def __post_init__(self) -> None:
        grid = _readonly(self.thresholds)
        for name in ("a", "b", "difference", "difference_se"):
            curve = _readonly(getattr(self, name))
            if curve.shape != grid.shape:
                raise ValueError(f"{name} must carry one value per threshold")
            object.__setattr__(self, name, curve)
        object.__setattr__(self, "thresholds", grid)
        object.__setattr__(self, "level", float(self.level))
        object.__setattr__(self, "n_observations", int(self.n_observations))

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "schema_version": self.schema_version,
            "level": self.level,
            "n_observations": self.n_observations,
            "thresholds": _json_numbers(self.thresholds),
            "a": _json_numbers(self.a),
            "b": _json_numbers(self.b),
            "difference": _json_numbers(self.difference),
            "difference_se": _json_numbers(self.difference_se),
        }


@dataclass(frozen=True)
class Comparison:
    """A paired score comparison, overall and optionally by segment."""

    score: str
    overall: Mapping[str, float]
    by_segment: pd.DataFrame | None
    murphy: MurphyPayload | None
    kind: str = "comparison"
    schema_version: int = 1

    def to_json(self) -> dict[str, Any]:
        segments = None
        if self.by_segment is not None:
            segments = [
                {
                    "segment": str(label),
                    "n": int(row["n"]),
                    "mean_diff": _json_number(row["mean_diff"]),
                    "se": _json_number(row["se"]),
                    "t": _json_number(row["t"]),
                }
                for label, row in self.by_segment.iterrows()
            ]
        return {
            "kind": self.kind,
            "schema_version": self.schema_version,
            "score": self.score,
            "overall": {
                "mean_diff": _json_number(self.overall["mean_diff"]),
                "se": _json_number(self.overall["se"]),
                "t": _json_number(self.overall["t"]),
                "n": int(self.overall["n"]),
            },
            "by_segment": segments,
            "murphy": None if self.murphy is None else self.murphy.to_json(),
        }


def _paired_summary(
    differences: NDArray[np.float64], aggregation_mass: NDArray[np.float64] | None = None
) -> dict[str, float]:
    """Replication-correct mean, paired standard error and ``t``."""
    values = np.asarray(differences, dtype=np.float64)
    mass = (
        np.ones(values.size, dtype=np.float64)
        if aggregation_mass is None
        else np.asarray(aggregation_mass, dtype=np.float64)
    )
    if mass.shape != values.shape:
        raise ValueError("aggregation mass must give one value per score difference")
    count = int(np.sum(mass, dtype=np.float64))
    mean = float(np.dot(mass, values) / count) if count else float("nan")
    if count < 2:
        # One row carries no within-sample spread, so it certifies no difference.
        return {"mean_diff": mean, "se": float("nan"), "t": float("nan"), "n": count}
    squared_error = float(np.dot(mass, (values - mean) ** 2) / (count * (count - 1)))
    error = float(np.sqrt(squared_error))
    ratio = mean / error if error > 0.0 else float("nan")
    return {"mean_diff": mean, "se": error, "t": float(ratio), "n": count}


def _elementary_quantile_scores(
    forecast: NDArray[np.float64],
    below_response: NDArray[np.bool_],
    grid: NDArray[np.float64],
    level: float,
) -> NDArray[np.float64]:
    """``S_alpha^eta(x, y)`` over a ``(thresholds, rows)`` block."""
    below_forecast = forecast[None, :] <= grid[:, None]
    return np.where(
        below_response != below_forecast, np.where(below_response, 1.0 - level, level), 0.0
    )


def murphy_diagram(
    a_quantiles: NDArray,
    b_quantiles: NDArray,
    y: NDArray,
    *,
    level: float,
    thresholds: NDArray,
) -> MurphyPayload:
    """Mean elementary quantile scores of two quantile forecasts over ``thresholds``.

    ``a_quantiles`` and ``b_quantiles`` are each model's ``level``-quantile
    forecast per row.  The elementary scores are non-negative and integrate over
    ``eta`` to the pinball loss ``(1{y <= x} - level)(x - y)``, which is how the
    tests certify the construction rather than restating it.
    """
    return _murphy_diagram(
        a_quantiles,
        b_quantiles,
        y,
        level=level,
        thresholds=thresholds,
        aggregation_mass=None,
    )


def _murphy_diagram(
    a_quantiles: NDArray,
    b_quantiles: NDArray,
    y: NDArray,
    *,
    level: float,
    thresholds: NDArray,
    aggregation_mass: NDArray[np.float64] | None,
) -> MurphyPayload:
    """Build Murphy curves, treating aggregation mass as literal replication."""
    alpha = float(level)
    if not 0.0 < alpha < 1.0:
        raise ValueError("level must be a quantile level strictly inside (0, 1)")
    response = np.asarray(y, dtype=np.float64)
    if response.ndim != 1:
        raise ValueError("the Murphy diagram needs a one-dimensional response of at least two rows")
    mass = (
        np.ones(len(response), dtype=np.float64)
        if aggregation_mass is None
        else np.asarray(aggregation_mass, dtype=np.float64)
    )
    if mass.shape != response.shape:
        raise ValueError("aggregation mass must give one value per Murphy-diagram row")
    count = int(np.sum(mass, dtype=np.float64))
    if count < 2:
        raise ValueError("the Murphy diagram needs a one-dimensional response of at least two rows")
    forecasts = []
    for values in (a_quantiles, b_quantiles):
        array = np.asarray(values, dtype=np.float64)
        if array.shape != response.shape:
            raise ValueError("the Murphy diagram needs one forecast per row for both models")
        forecasts.append(array)
    grid = np.asarray(thresholds, dtype=np.float64)
    if grid.ndim != 1 or len(grid) < 1:
        raise ValueError("the Murphy diagram needs at least one threshold")

    curves = [np.empty(len(grid), dtype=np.float64) for _ in range(2)]
    difference = np.empty(len(grid), dtype=np.float64)
    difference_se = np.empty(len(grid), dtype=np.float64)
    block = max(1, _MURPHY_BLOCK_CELLS // len(response))
    for start in range(0, len(grid), block):
        panel = grid[start : start + block]
        below_response = response[None, :] <= panel[:, None]
        scores = [
            _elementary_quantile_scores(forecast, below_response, panel, alpha)
            for forecast in forecasts
        ]
        for curve, block_scores in zip(curves, scores, strict=True):
            curve[start : start + len(panel)] = (block_scores @ mass) / count
        gap = scores[0] - scores[1]
        gap_mean = (gap @ mass) / count
        difference[start : start + len(panel)] = gap_mean
        centred = gap - gap_mean[:, None]
        difference_se[start : start + len(panel)] = np.sqrt(
            ((centred * centred) @ mass) / (count * (count - 1))
        )

    return MurphyPayload(
        level=alpha,
        thresholds=grid,
        a=curves[0],
        b=curves[1],
        difference=difference,
        difference_se=difference_se,
        n_observations=count,
    )


def _segment_labels(
    by: str | Sequence[Any] | NDArray, frame: EagerFrame, n_observations: int
) -> NDArray:
    if isinstance(by, str):
        if by not in frame.columns:
            raise ValueError(f"unknown segment column {by!r}")
        labels = np.asarray(frame.column_array(by))
    else:
        labels = np.asarray(by)
    if labels.ndim != 1 or len(labels) != n_observations:
        raise ValueError("the comparison needs one segment label per row")
    return labels


def _segment_table(
    labels: NDArray,
    differences: NDArray[np.float64],
    aggregation_mass: NDArray[np.float64],
) -> pd.DataFrame:
    grouped = pd.DataFrame(
        {"segment": labels, "difference": differences, "mass": aggregation_mass}
    ).groupby("segment", sort=True)
    rows = [
        {
            "segment": name,
            **_paired_summary(group["difference"].to_numpy(), group["mass"].to_numpy()),
        }
        for name, group in grouped
    ]
    table = pd.DataFrame(rows).set_index("segment")
    return table[["n", "mean_diff", "se", "t"]]


def _row_scores(
    fitted: Any,
    rows: _ScoringRows,
    which: str,
    n_nodes: int,
) -> NDArray[np.float64]:
    if which == "log":
        # Family likelihoods already compress frequency rows.  Divide back to
        # one replicated observation before the paired summary applies mass.
        return _retained_log_score(fitted, rows) / rows.aggregation_mass
    return _retained_crps(fitted, rows, method="auto", n_nodes=n_nodes)


def _quantile_family(fitted: Any) -> Any:
    """Return the family, refusing one that cannot answer a quantile."""
    family = fitted.family
    if not isinstance(family, DistributionFunctionFamily):
        raise NotImplementedError(
            "a Murphy diagram needs a family with a quantile function; this one has none"
        )
    return family


def _level_quantiles(fitted: Any, rows: _ScoringRows, level: float) -> NDArray[np.float64]:
    family = _row_law(fitted, rows)
    theta = np.asarray(
        fitted.predict_parameters(rows.frame, offsets=rows.offsets), dtype=np.float64
    )
    levels = np.full(len(theta), float(level), dtype=np.float64)
    return np.asarray(family.quantile(levels, theta), dtype=np.float64)


def _replicated_quantiles(
    values: NDArray[np.float64],
    aggregation_mass: NDArray[np.float64],
    probabilities: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Match ``np.quantile(np.repeat(values, mass), probabilities)`` without expansion."""
    data = np.asarray(values, dtype=np.float64)
    mass = np.asarray(aggregation_mass, dtype=np.float64)
    if data.shape != mass.shape:
        raise ValueError("aggregation mass must give one value per default-grid value")
    if np.all(mass == 1.0):
        return np.asarray(np.quantile(data, probabilities), dtype=np.float64)

    order = np.argsort(data)
    ordered = data[order]
    cumulative = np.cumsum(mass[order].astype(np.int64), dtype=np.int64)
    count = int(cumulative[-1])
    ranks = (count - 1) * np.asarray(probabilities, dtype=np.float64)
    lower_ranks = np.floor(ranks).astype(np.int64)
    upper_ranks = np.ceil(ranks).astype(np.int64)
    lower = ordered[np.searchsorted(cumulative, lower_ranks, side="right")]
    upper = ordered[np.searchsorted(cumulative, upper_ranks, side="right")]
    fraction = ranks - lower_ranks
    difference = upper - lower
    return np.where(
        fraction >= 0.5,
        upper - difference * (1.0 - fraction),
        lower + difference * fraction,
    )


def _default_thresholds(
    *columns: NDArray[np.float64],
    aggregation_mass: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """A grid over the central 98 % of the replication-expanded pooled rows."""
    pooled = np.concatenate(columns)
    with np.errstate(over="ignore", invalid="ignore"):
        if aggregation_mass is None:
            bounds = np.quantile(pooled, [0.01, 0.99])
        else:
            mass = np.asarray(aggregation_mass, dtype=np.float64)
            bounds = _replicated_quantiles(
                pooled,
                np.tile(mass, len(columns)),
                np.array([0.01, 0.99]),
            )
    if not np.all(np.isfinite(bounds)):
        raise ValueError(
            "default Murphy grid bounds must be finite and representable; "
            "pass thresholds explicitly"
        )
    lower, upper = (float(bound) for bound in bounds)
    if not upper > lower:  # pragma: no cover - a fitted continuous response is never this flat
        raise ValueError(
            "the pooled outcomes and forecasts span no range, so there is no default Murphy "
            "grid to build; pass thresholds explicitly"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        grid = np.linspace(lower, upper, _DEFAULT_MURPHY_POINTS)
    if not np.all(np.isfinite(grid)):
        raise ValueError(
            "default Murphy grid must be finite and representable; pass thresholds explicitly"
        )
    return grid


def compare_models(
    a_fitted: Any,
    b_fitted: Any,
    X: FrameLike | EagerFrame,
    y: NDArray,
    *,
    which: ScoreName = "log",
    by: str | Sequence[Any] | NDArray | None = None,
    murphy_quantile: float | None = None,
    thresholds: NDArray | None = None,
    sample_weight: NDArray | None = None,
    offsets: Mapping[str, NDArray] | None = None,
    n_nodes: int = 64,
) -> Comparison:
    """Compare two fits on the same rows by their per-row score differences.

    The difference is ``score(a) - score(b)``, so a negative mean says ``a`` is
    the better model.  The two fits must share the row set and, if ``offsets``
    are given, the predictor names they are keyed by; they need not share a
    family.  Non-unit ``sample_weight`` requires the candidates to declare the
    same likelihood-weight semantics.  Prior semantics changes each row's law;
    frequency semantics gives it literal replication mass.  Zero-weight rows
    are omitted from the comparison.  ``by`` segments the difference, by a
    column name of ``X`` or by an array of labels.  ``murphy_quantile`` adds the
    Murphy diagram at that quantile level over ``thresholds``, defaulting to a
    grid over the replication-expanded pooled outcomes and forecasts.
    """
    if which not in ("log", "crps"):
        raise ValueError(f"unknown score {which!r}; compare_models scores 'log' or 'crps'")
    frame = as_eager_frame(X)
    n_observations = len(frame)
    response = np.asarray(y, dtype=np.float64)
    if murphy_quantile is not None:
        # Refuse a Murphy diagram before scoring rather than after: the scores
        # are the expensive half and a missing quantile function is known now.
        for fitted in (a_fitted, b_fitted):
            _quantile_family(fitted)

    declared_semantics = tuple(
        fitted.fit_state.weight_contract.semantics for fitted in (a_fitted, b_fitted)
    )
    if declared_semantics[0] != declared_semantics[1] and sample_weight is not None:
        # Inspect against the permissive prior contract before either candidate
        # can reject a fractional value merely because it declares frequency
        # semantics.  Invalid vectors still reach the ordinary weight validator.
        try:
            supplied = resolve_likelihood_weights(
                sample_weight,
                n_observations=n_observations,
                contract=WeightContract("prior"),
            )
        except LikelihoodWeightError:
            pass
        else:
            if not supplied.provenance.all_unit:
                raise UnsupportedLikelihoodContractError(
                    "non-unit comparison weights require candidates with the same weight semantics"
                )

    rows = [
        _scoring_rows(
            fitted,
            frame,
            response,
            sample_weight=sample_weight,
            offsets=offsets,
        )
        for fitted in (a_fitted, b_fitted)
    ]
    if declared_semantics[0] != declared_semantics[1] and not all(
        row.resolved.provenance.all_unit for row in rows
    ):
        raise UnsupportedLikelihoodContractError(
            "non-unit comparison weights require candidates with the same weight semantics"
        )
    if not np.array_equal(rows[0].positions, rows[1].positions):  # pragma: no cover
        raise RuntimeError("candidate weight resolution retained different comparison rows")
    aggregation_mass = rows[0].aggregation_mass
    if not np.array_equal(aggregation_mass, rows[1].aggregation_mass):  # pragma: no cover
        raise RuntimeError("candidate weight resolution assigned different aggregation mass")

    scores = [
        _row_scores(fitted, fitted_rows, which, n_nodes)
        for fitted, fitted_rows in zip((a_fitted, b_fitted), rows, strict=True)
    ]
    for label, values in zip(("a", "b"), scores, strict=True):
        unusable = int(np.count_nonzero(~np.isfinite(values)))
        if unusable:
            raise ValueError(
                f"model {label} scored {unusable} of {len(rows[0].positions)} rows non-finite under "
                f"the {which!r} score; a paired mean over those rows would not be a comparison"
            )
    differences = scores[0] - scores[1]

    segments = None
    if by is not None:
        labels = _segment_labels(by, frame, n_observations)[rows[0].positions]
        segments = _segment_table(labels, differences, aggregation_mass)
    murphy = None
    if murphy_quantile is not None:
        level = float(murphy_quantile)
        a_quantiles = _level_quantiles(a_fitted, rows[0], level)
        b_quantiles = _level_quantiles(b_fitted, rows[1], level)
        retained_response = rows[0].response
        grid = (
            _default_thresholds(
                retained_response,
                a_quantiles,
                b_quantiles,
                aggregation_mass=aggregation_mass,
            )
            if thresholds is None
            else np.asarray(thresholds, dtype=np.float64)
        )
        murphy = _murphy_diagram(
            a_quantiles,
            b_quantiles,
            retained_response,
            level=level,
            thresholds=grid,
            aggregation_mass=aggregation_mass,
        )

    return Comparison(
        score=str(which),
        overall=_paired_summary(differences, aggregation_mass),
        by_segment=segments,
        murphy=murphy,
    )


__all__ = [
    "Comparison",
    "MurphyPayload",
    "compare_models",
    "murphy_diagram",
]
