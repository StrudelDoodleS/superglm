"""The editor's four checks of a SuperLSS's predicted distribution on one split.

Each check reads the split through the library's own builders and holds its
flag to one band for the whole check, so its chance does not grow with the
points or cells read:

* ``check_qq``: the sorted randomised quantile residuals (Dunn and Smyth 1996)
  against the equal local levels band, mapped to the normal scale;
* ``check_pit``: the PIT histogram, each bin against its binomial band;
* ``check_calibration``: the realised exceedance of each predicted quantile
  against its binomial band, over the grid;
* ``check_ae``: actual over expected by band of the predicted Mean and by each
  input, each interval ``ratio -/+ z ratio_se``.

The bands' levels depend on the rows. On the rows the model was fitted on they
are the known-parameter levels: the equal local levels band
(:func:`superglm.stats.equal_local_levels_band`), Bonferroni's binomial band at
``CHECK_ALPHA / (2 cells)`` in each tail, and ``z = Phi^{-1}(1 - CHECK_ALPHA /
(2 cells))``. The fit absorbs part of the noise there, so these are
conservative (Durbin 1973; Weine, McPeek and Abney 2023).

On any other split the fit's own noise is shared by every row, so the
known-parameter bands flag a correct model more often than ``CHECK_ALPHA``, by
an amount set by the ratio of the split's rows to the fit's (West 1996). There
each level comes from a holdout predictive check (Moran, Blei and Ranganath
2024; Li and Huggins 2024): :func:`null_replicates` draws ``CHECK_DRAWS``
coefficient sets from the fit's posterior, simulates the split's responses
under each, scores every replicate with the fitted model exactly as the split
is scored, and each check's level is Barnard's Monte Carlo level of its
replicates' statistic (Barnard 1963, as Myllymaki et al. 2017 describe it;
:func:`superglm.stats.monte_carlo_level`). The band is drawn at that level and
the flag is read from the drawn band. A flag on a held-out split means the
model's predictions miss new rows by more than the fit's uncertainty explains:
either the model's form is wrong, or its estimates are too noisy for the data
(many coefficients for the fitted rows, or a very small fit). Noisy estimates
fall back toward the average on new rows, which replicates drawn around the
fitted coefficients do not show. The level holds about ``CHECK_ALPHA`` for a
correctly specified model only when its coefficients are few for its rows.
Where the fit's uncertainty cannot be sampled, or the rows cannot be simulated,
the band keeps the known-parameter level and the body says which.

A body is JSON-clean: a value with no finite number is ``None``. A check the
model or split cannot answer is the unavailable body, whose ``reason`` is a
sentence the page shows.
"""

from __future__ import annotations

import contextlib
import functools
import math
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, NamedTuple

import numpy as np
import scipy.linalg
import scipy.spatial
from numpy.typing import NDArray
from scipy import special, stats

from superglm._frame import EagerFrame, as_eager_frame
from superglm.distributional.api import SuperLSS
from superglm.distributional.checks.binned import _covariate_bins
from superglm.distributional.checks.calibration import (
    _VARIANCE_DRAWS,
    ActualExpected,
    _actual_expected,
    _retained_rows,
    _row_law,
    _row_variance,
    calibration_payload,
)
from superglm.distributional.checks.pit import pit_payload
from superglm.distributional.family import (
    AtomFamily,
    DefaultPredictionFamily,
    DistributionFunctionFamily,
    VarianceFamily,
)
from superglm.distributional.posterior import PosteriorDraws, posterior_draws, posterior_parameters
from superglm.distributional.residuals import (
    _PROBABILITY_CEILING,
    _PROBABILITY_FLOOR,
    ResidualSet,
    _residual_rng,
    _sample_residuals,
    compute_residuals,
    replication_sample,
)
from superglm.distributional.response import response_spec
from superglm.distributional.weights import resolve_likelihood_weights
from superglm.editor.errors import no_quote_message
from superglm.editor.evaluation import EvaluationDataset
from superglm.editor.lss.validation import lss_split_reason
from superglm.editor.lss.view import LSSModelView, column_of, is_lss, lss_key
from superglm.features.categorical import Categorical
from superglm.features.ordered_categorical import OrderedCategorical
from superglm.features.random_effect import RandomEffect
from superglm.model.base import NoPredictedMeanError
from superglm.stats.bands import (
    _LOG_BELOW,
    _binomial_cell_levels,
    _frozen,
    _log_band_lower,
    _monte_carlo_order_statistic_level,
    _stored_median,
    band_at_level,
    bonferroni_binomial_band,
    equal_local_levels_band,
    monte_carlo_level,
)

#: The checks, cheapest first.
CHECK_REPORTS = ("check_pit", "check_ae", "check_calibration", "check_qq")
#: Global level of each check: a calibrated model is flagged with chance at most this.
CHECK_ALPHA = 0.05
#: Seed of the randomised residuals, any simulated variance and the replicates, for every model.
CHECK_SEED = 42
#: Replicates behind a held-out split's levels: at ``CHECK_ALPHA`` Barnard's rule reads the 5th.
CHECK_DRAWS = 99
#: Bins of the PIT histogram.
PIT_BINS = 20
#: Bins of the predicted Mean and of each numeric input in actual over expected.
AE_BINS = 10
#: Probabilities whose predicted quantiles quantile calibration reads.
QUANTILE_GRID = (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99, 0.995)
#: Order statistics a Q-Q body sends evenly spaced, beside its ends; it sends at most as many
#: of those outside the band, beside the farthest beyond each edge.
QQ_DISPLAY_POINTS = 800
#: Lowest and highest order statistics a Q-Q body always sends.
_QQ_ENDS = 10
#: The ``name`` of the split that holds the rows the model was fitted on, supplied or retained.
_FIT_ROWS = "train"
#: Stream of the replicates' uniforms; the residuals' own streams are domains 0 to 5.
_RESPONSE_STREAM = 6
#: Rows simulated at once: the posterior pushforward and the responses hold this many rows of
#: ``(draws, rows, parameters)`` floats at a time; the transforms kept are one ``(draws, read)``
#: array over the occurrences Q-Q and PIT read, at most the residual sampler's cap.
_SIMULATION_ROWS = 1024
#: Simulated uniforms lie strictly inside (0, 1) by this margin, as ``posterior_predictive``'s do.
_PROBABILITY_MARGIN = 1.0e-12

_LABELS = {
    "check_qq": "Q-Q plot",
    "check_pit": "PIT histogram",
    "check_ae": "actual over expected",
    "check_calibration": "quantile calibration",
}
_NOT_LSS = "This check is available for SuperLSS models."
_NO_MEAN = "This model's family names no predicted Mean, so it has no actual over expected."
_NO_VARIANCE_LAW = (
    "This model's family has no predicted variance, so actual over expected has no interval."
)
_TOTAL_TOO_LARGE = (
    "A group's observed or predicted total is too large to represent, so actual over expected "
    "has no ratio. Rescale the response and refit."
)
#: What the library's grouped ratio says when a row's weighted sum term is not finite.
_OVERFLOWED_SUMS = frozenset({"numerator must be finite", "denominator must be finite"})
_NO_FINITE_MEAN = (
    "The predicted distribution has no finite Mean at these parameters, "
    "so actual over expected has no expected total."
)
_NO_VARIANCE = (
    "The predicted distribution has no finite variance at these parameters, "
    "so actual over expected has no interval."
)
_UNSAMPLED_FIT = (
    "The band treats the fitted parameters as exact, because the fit's uncertainty could not "
    "be sampled."
)
_UNSIMULATED_ROWS = (
    "The band treats the fitted parameters as exact, because these rows could not be simulated."
)
#: The pushforward's refusal of a draw that puts a parameter outside its support on some row.
_OUTSIDE_SUPPORT = "outside its supported range"


class CheckUnavailable(Exception):  # noqa: N818 - a reason the page shows, not a fault
    """A check this model or split cannot answer; ``reason`` is the sentence the page shows."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _lss(model: Any) -> SuperLSS | None:
    """The SuperLSS ``model`` is or views; ``None`` for any other model."""
    if not is_lss(model):
        return None
    return model.lss if isinstance(model, LSSModelView) else model


def _unsupported(report: str, family: Any) -> str | None:
    """Why ``family`` cannot give ``report``; ``None`` when it can."""
    if report == "check_ae":
        if response_spec(family).shape or not isinstance(family, DefaultPredictionFamily):
            return _NO_MEAN
        # Its interval reads each row's predicted variance: the family's own, or simulated
        # through its quantile function.
        if not isinstance(family, VarianceFamily | DistributionFunctionFamily):
            return _NO_VARIANCE_LAW
        return None
    if response_spec(family).shape or not isinstance(family, DistributionFunctionFamily):
        return f"This model's family has no distribution function, so it has no {_LABELS[report]}."
    return None


def offered_checks(model: Any) -> list[str]:
    """The checks ``model``'s family supports, in ``CHECK_REPORTS`` order; none for a SuperGLM."""
    lss = _lss(model)
    if lss is None:
        return []
    return [report for report in CHECK_REPORTS if _unsupported(report, lss.family) is None]


def check_inputs(model: Any, terms: Iterable[str]) -> list[str]:
    """The distinct data columns of ``terms`` that read one column each, in term order.

    An interaction reads several columns and is left out.
    """
    lss = _lss(model)
    if lss is None:
        single = set(getattr(model, "_feature_order", ()))
    else:
        single = {
            lss_key(predictor.name, feature)
            for predictor in lss._require_fitted().compiled_predictors
            for feature in predictor.compiled.feature_order
        }
    return list(dict.fromkeys(column_of(model, term) for term in terms if term in single))


def _json_clean(value: Any) -> Any:
    """``value`` with every number a Python one, and ``None`` where it is not finite."""
    if isinstance(value, dict):
        return {key: _json_clean(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_clean(item) for item in value.tolist()]
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def _unavailable(reason: str) -> dict[str, Any]:
    return {"available": False, "reason": reason, "flagged": False, "outside": 0, "cells": 0}


def _answered(body: dict[str, Any], *, outside: int, cells: int) -> dict[str, Any]:
    return _json_clean(
        {
            **body,
            "available": True,
            "reason": None,
            "flagged": outside > 0,
            "outside": int(outside),
            "cells": int(cells),
        }
    )


def qq_from_residuals(
    quantile: NDArray[np.float64], *, local_level: float | None = None, log: bool = False
) -> dict[str, Any]:
    """The Q-Q body of the quantile residuals ``quantile``, in any order.

    The band holds for all ``n`` order statistics at once, and ``outside``
    counts every one that leaves it. Its local level is ``local_level``, as
    :func:`null_replicates` calibrates it, or by default the known-parameter
    level of :func:`superglm.stats.equal_local_levels_band`. ``index`` names
    the order statistics the body sends: ``QQ_DISPLAY_POINTS`` evenly spaced,
    the ``10`` lowest and highest, and those outside the band: every one up to
    ``QQ_DISPLAY_POINTS`` of them, else that many evenly by rank, the lowest and
    highest included, with the farthest beyond each edge on the normal scale.

    With ``log``, ``local_level`` is the level's natural log. Below twice the
    least normal float each edge is found in logs and mapped to the normal
    scale from its log (``ndtri_exp``), so an edge float64 cannot place as a
    probability is still drawn where it is. The body then sends that log as
    ``log_local_level`` and ``local_level`` as ``None``.
    """
    z = np.sort(np.asarray(quantile, dtype=np.float64).ravel())
    n = len(z)
    if log and local_level is not None and local_level < math.log(_LOG_BELOW):
        lower = special.ndtri_exp(_log_band_lower(n, local_level))
        upper, median = -lower[::-1], _frozen(_stored_median(n))
        stated = _level_fields(local_level, log=True)
    else:
        band = (
            equal_local_levels_band(n, alpha=CHECK_ALPHA)
            if local_level is None
            else band_at_level(n, math.exp(local_level) if log else local_level, alpha=CHECK_ALPHA)
        )
        lower, upper, median = special.ndtri(band.lower), special.ndtri(band.upper), band.median
        stated = _level_fields(float(band.local_level), log=False)
    outside = (z < lower) | (z > upper)
    spaced = _evenly(np.arange(n))
    ends = np.r_[np.arange(min(_QQ_ENDS, n)), np.arange(max(n - _QQ_ENDS, 0), n)]
    # The farthest beyond each edge, on the sides some point leaves.
    farthest = [np.argmax(gap) for gap in (lower - z, z - upper) if np.any(gap > 0.0)]
    shown = np.r_[_evenly(np.flatnonzero(outside)), np.asarray(farthest, dtype=np.intp)]
    index = np.union1d(np.union1d(spaced, ends), shown).astype(np.intp)
    body = {
        "n": n,
        "alpha": CHECK_ALPHA,
        **stated,
        "index": index,
        "theoretical": special.ndtri(median[index]),
        "observed": z[index],
        "band_lower": lower[index],
        "band_upper": upper[index],
    }
    return _answered(body, outside=int(np.count_nonzero(outside)), cells=n)


def _evenly(positions: NDArray[np.intp]) -> NDArray[np.intp]:
    """``positions``, or ``QQ_DISPLAY_POINTS`` of them evenly by rank, the first and last kept."""
    if len(positions) <= QQ_DISPLAY_POINTS:
        return positions
    return positions[np.rint(np.linspace(0, len(positions) - 1, QQ_DISPLAY_POINTS)).astype(np.intp)]


def _level_fields(level: float, *, log: bool) -> dict[str, Any]:
    """A body's level: ``local_level``, or below float64's range its natural log as
    ``log_local_level``, with ``local_level`` ``None``."""
    return {"local_level": None, "log_local_level": level} if log else {"local_level": level}


def _pit_body(residuals: Any, level: _Level) -> dict[str, Any]:
    payload = pit_payload(residuals, n_bins=PIT_BINS, alpha=CHECK_ALPHA / PIT_BINS)
    body = {**payload.to_json(), "alpha": CHECK_ALPHA, "local_level": CHECK_ALPHA / (2 * PIT_BINS)}
    cut, log = level.band_level()
    if cut is not None:
        lower, upper = bonferroni_binomial_band(payload.n_rows, 1.0 / PIT_BINS, cut=cut, log=log)
        body.update(band_lower=float(lower), band_upper=float(upper), **_level_fields(cut, log=log))
    counts = payload.counts
    outside = int(np.count_nonzero((counts < body["band_lower"]) | (counts > body["band_upper"])))
    return _answered({**body, **level.fields()}, outside=outside, cells=PIT_BINS)


def _calibration_body(
    fitted: Any, dataset: EvaluationDataset, y, residuals, level: _Level
) -> dict[str, Any]:
    payload = calibration_payload(
        fitted,
        dataset.X,
        y,
        residuals=residuals,
        thresholds=(),
        quantile_grid=QUANTILE_GRID,
        by_parameter_deciles=False,
        sample_weight=dataset.sample_weight,
        offsets=dataset.offset,
        seed=CHECK_SEED,
    )
    table = payload.quantiles
    p = table["p"].to_numpy(dtype=np.float64)
    trials = np.rint(table["weight"].to_numpy(dtype=np.float64))
    cut, log = level.band_level()
    if cut is None:
        low, high = bonferroni_binomial_band(trials, 1.0 - p, alpha=CHECK_ALPHA)
        cut = CHECK_ALPHA / (2 * len(QUANTILE_GRID))
    else:
        low, high = bonferroni_binomial_band(trials, 1.0 - p, cut=cut, log=log)
    with np.errstate(divide="ignore", invalid="ignore"):
        band_lower, band_upper = low / trials, high / trials
    shares = table["realised_exceedance"].to_numpy(dtype=np.float64)
    outside = int(np.count_nonzero((shares < band_lower) | (shares > band_upper)))
    body = {
        "quantiles": {
            name: table[name].to_numpy()
            for name in ("p", "n", "weight", "expected", "realised_exceedance")
        },
        "band_lower": band_lower,
        "band_upper": band_upper,
        **_level_fields(cut, log=log),
        "calibration_law": payload.calibration_law,
        **level.fields(),
    }
    return _answered(body, outside=outside, cells=len(QUANTILE_GRID))


def _ae_covariate(column: NDArray, levelled: bool) -> NDArray:
    """``column``'s retained values as actual over expected groups them, read as the fit reads
    them: by level, as objects, so integer labels sort as numbers and are never cut into
    ranges; otherwise as numbers, so text-backed numbers are cut into ranges and never read as
    levels."""
    return np.asarray(column, dtype=object if levelled else np.float64)


def _ae_bins(values: NDArray) -> int:
    """``AE_BINS``, or fewer for a numeric covariate with fewer distinct values.

    Equal-count edges interpolated between fewer values than bins would leave bins empty.
    """
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.number) or array.dtype.kind == "b":
        return AE_BINS
    return max(1, min(AE_BINS, np.unique(array).size))


class _AETable(NamedTuple):
    """One actual-over-expected table, with the cell of each retained row."""

    check: ActualExpected
    codes: NDArray[np.intp]
    by: str


def _ae_table(fitted, rows, theta, covariate, name: str, by: str, variance) -> _AETable:
    """The table of the retained ``rows`` by ``covariate``, one value per retained row."""
    binning = _covariate_bins(covariate, _ae_bins(covariate))
    try:
        table = _actual_expected(fitted, rows, theta, binning, name=name, variance=variance)
    except NoPredictedMeanError:
        raise
    except ValueError as exc:
        # A law without finite variance is a reason, and so is a row's weighted response or
        # Mean past float64's range: both are finite, so only their product can overflow.
        # Any other error is a fault.
        if "finite" in str(exc) and "variance" in str(exc):
            raise CheckUnavailable(_NO_VARIANCE) from None
        if str(exc) in _OVERFLOWED_SUMS:
            raise CheckUnavailable(_TOTAL_TOO_LARGE) from None
        raise
    return _AETable(table, binning.codes, by)


def _ae_tables(fitted, dataset: EvaluationDataset, y, inputs: Sequence[str]) -> list[_AETable]:
    """The band table of the predicted Mean, then a table for each input in the split."""
    rows = _retained_rows(
        fitted, dataset.X, y, sample_weight=dataset.sample_weight, offsets=dataset.offset
    )
    theta = np.asarray(
        fitted.parameters_from_eta(fitted.predict_eta(rows.frame, offsets=rows.offsets)),
        dtype=np.float64,
    )
    band = np.asarray(fitted.family.default_prediction(theta), dtype=np.float64)
    if not np.all(np.isfinite(band)):
        raise CheckUnavailable(_NO_FINITE_MEAN)
    # Each input is grouped as the fit reads it, by level or as a number, whatever its dtype.
    levelled = {
        name
        for predictor in fitted.compiled_predictors
        for name, spec in predictor.compiled.specs.items()
        if isinstance(spec, Categorical | OrderedCategorical | RandomEffect)
    }
    # Only the retained rows are read, so a value in a row the split ignores is never parsed.
    covariates = [(band, "predicted Mean", "band")] + [
        (_ae_covariate(rows.frame.column_array(name), name in levelled), name, "input")
        for name in inputs
        if name in rows.frame.columns
    ]
    # Every table reads the same rows at the same seed, so each row's variance is the same for
    # all of them: computed once, by the first table once its sums are formed, and kept for
    # this call alone.
    variance = functools.cache(
        functools.partial(
            _row_variance, fitted, rows, theta, n_draws=_VARIANCE_DRAWS, seed=CHECK_SEED
        )
    )
    return [
        _ae_table(fitted, rows, theta, covariate, name, by, variance)
        for covariate, name, by in covariates
    ]


def _ae_read(se: NDArray[np.float64]) -> NDArray[np.bool_]:
    """The cells actual over expected reads: those whose ratio has a finite, positive SE."""
    return np.isfinite(se) & (se > 0.0)


def _ae_body(
    fitted, dataset: EvaluationDataset, y, inputs: Sequence[str], level: Callable[[], _Level]
) -> dict[str, Any]:
    built = _ae_tables(fitted, dataset, y, inputs)
    with np.errstate(over="ignore", invalid="ignore"):
        actual = float(np.sum(built[0].check.actual))
        expected = float(np.sum(built[0].check.expected))
    overflowed = any(
        np.isinf(t.check.actual).any() or np.isinf(t.check.expected).any() for t in built
    )
    # Finite responses can still sum past float64's range, and such a total has no ratio.
    if overflowed or not (math.isfinite(actual) and math.isfinite(expected)):
        return _unavailable(_TOTAL_TOO_LARGE)
    tables = [{**table.check.to_json(), "by": table.by} for table in built]
    ratios = [np.asarray(t["ratio"], dtype=np.float64) for t in tables]
    errors = [np.asarray(t["ratio_se"], dtype=np.float64) for t in tables]
    # A cell is read when its SE is finite and positive, as the replicates' maximum reads it.
    cells = int(sum(np.count_nonzero(_ae_read(se)) for se in errors))
    found = level()
    z = found.value
    if z is None and cells:
        z = float(stats.norm.isf(CHECK_ALPHA / (2.0 * cells)))
    outside = 0
    for table, ratio, se in zip(tables, ratios, errors, strict=True):
        # A cell that is not read has no interval, so the page draws nothing it does not count.
        with np.errstate(invalid="ignore"):
            half = np.where(_ae_read(se), np.nan if z is None else z * se, np.nan)
        table["lower"], table["upper"] = ratio - half, ratio + half
        outside += int(np.count_nonzero((ratio - half > 1.0) | (ratio + half < 1.0)))
    # A zero expected total has no ratio: NaN, as the library's grouped ratios, sent as null.
    ratio = actual / expected if expected != 0.0 else math.nan
    body = {
        "tables": tables,
        "z": z,
        "overall": {"actual": actual, "expected": expected, "ratio": ratio},
        **found.fields(),
    }
    return _answered(body, outside=outside, cells=cells)


def _no_quote(fitted, dataset: EvaluationDataset, y, refused: NoPredictedMeanError) -> str:
    """The page's sentence for ``refused``, its rows named among the split's own."""
    positions = resolve_likelihood_weights(
        dataset.sample_weight,
        n_observations=y.size,
        contract=fitted.fit_state.weight_contract,
    ).input_positions
    return no_quote_message(refused._restated(np.asarray(positions)[refused.positions], y.size))


class _Level(NamedTuple):
    """Where a body's band level comes from: replicates of a held-out split, or known parameters.

    ``value`` is the level the check's replicates give, or ``None`` for the known-parameter
    level; ``note`` is the page's sentence when a held-out split keeps that level.
    ``log_value`` is the level's natural log where it lies below twice the least normal float,
    and ``None`` elsewhere.
    """

    value: float | None
    n_draws: int
    note: str | None
    log_value: float | None = None

    def band_level(self) -> tuple[float | None, bool]:
        """The level a band is drawn at, and whether it is a natural log."""
        if self.log_value is not None:
            return self.log_value, True
        return self.value, False

    def fields(self) -> dict[str, Any]:
        simulated = self.value is not None
        return {
            "level_source": "simulated" if simulated else "known",
            "n_draws": self.n_draws if simulated else 0,
            "level_note": None if simulated else self.note,
        }


def _level(
    report: str,
    model: Any,
    dataset: EvaluationDataset,
    inputs: Sequence[str],
    replicates: NullReplicates | None,
    fitted: Any,
) -> _Level:
    """The level of ``report``'s band on ``dataset``: known on the fitted rows, else simulated."""
    if dataset.name == _FIT_ROWS:
        return _Level(None, 0, None)
    if replicates is None:
        replicates = null_replicates(model, dataset, inputs=inputs, fitted=fitted)
    return _Level(
        replicates.levels.get(report),
        replicates.n_draws,
        replicates.note,
        replicates.log_levels.get(report),
    )


def _body(
    report: str,
    model: Any,
    dataset: EvaluationDataset,
    inputs: Sequence[str],
    replicates: NullReplicates | None,
    fitted: Any,
):
    lss = _lss(model)
    if lss is None:
        raise CheckUnavailable(_NOT_LSS)
    fitted = lss._require_fitted() if fitted is None else fitted
    reason = _unsupported(report, fitted.family) or lss_split_reason(lss, dataset)
    if reason is not None:
        raise CheckUnavailable(reason)
    y = np.asarray(dataset.y, dtype=np.float64).ravel()
    # The level is read once the body's own rows are known to be scorable.
    level = functools.partial(_level, report, model, dataset, inputs, replicates, fitted)
    try:
        if report == "check_ae":
            return _ae_body(fitted, dataset, y, inputs, level)
        residuals = compute_residuals(
            fitted,
            dataset.X,
            y,
            sample_weight=dataset.sample_weight,
            offsets=dataset.offset,
            seed=CHECK_SEED,
        )
        if report == "check_qq":
            found = level()
            quantile = _sample_residuals(residuals, seed=CHECK_SEED).quantile
            cut, log = found.band_level()
            return {**qq_from_residuals(quantile, local_level=cut, log=log), **found.fields()}
        if report == "check_pit":
            return _pit_body(residuals, level())
        return _calibration_body(fitted, dataset, y, residuals, level())
    except NoPredictedMeanError as refused:
        raise CheckUnavailable(_no_quote(fitted, dataset, y, refused)) from None


def check_body(
    report: str,
    model: Any,
    dataset: EvaluationDataset,
    *,
    inputs: Sequence[str],
    replicates: NullReplicates | None = None,
    fitted: Any = None,
) -> dict[str, Any]:
    """One model's body of ``report`` on ``dataset``; ``inputs`` are the columns A/E groups by.

    On the rows the model was fitted on (the split named ``"train"``) the band
    has its known-parameter level. On any other split the level is read from ``replicates``,
    :func:`null_replicates` of this model, split and inputs, which one simulation gives every
    check; with none given the body simulates its own. Each available body says which in
    ``level_source`` (``"simulated"`` or ``"known"``), ``n_draws`` (0 when known) and
    ``level_note`` (the page's sentence when a held-out split keeps the known level, else
    ``None``).

    ``fitted``, the fitted model ``model`` held when the caller captured it (``model``'s own
    when omitted), is the fit scored, as :func:`null_replicates` simulates it: a refit in place
    replaces ``model``'s, so a body read against a captured fit's levels reads that fit too.
    """
    if report not in CHECK_REPORTS:
        raise ValueError(f"report must be one of {CHECK_REPORTS}; got {report!r}")
    try:
        return _body(report, model, dataset, inputs, replicates, fitted)
    except CheckUnavailable as unavailable:
        return _unavailable(unavailable.reason)


class _NotSampled(Exception):  # noqa: N818 - the level falls back, not a fault
    """The split's replicates cannot be simulated, so its bands keep the known-parameter level.

    ``note`` is the page's sentence for why: the fit's uncertainty, or the rows.
    """

    def __init__(self, note: str) -> None:
        super().__init__(note)
        self.note = note


class SimulationCancelled(Exception):  # noqa: N818 - the caller stopped it, not a fault
    """A simulation stopped at a block boundary because its caller asked it to; it keeps
    nothing."""


def _unless_cancelled(
    blocks: Iterator[tuple[slice, NDArray[np.float64]]], cancelled: Callable[[], bool]
) -> Iterator[tuple[slice, NDArray[np.float64]]]:
    """``blocks``, asking ``cancelled`` before each is drawn and once more after the last, so
    nothing is scored once it answers true: the stream raises :class:`SimulationCancelled`
    instead."""
    while not cancelled():
        block = next(blocks, None)
        if block is None:
            if cancelled():  # asked again once the blocks end, before the final scoring
                break
            return
        yield block
    raise SimulationCancelled


# Compared and hashed by identity: a cache holds it as a value, keyed by what it was drawn from.
@dataclass(frozen=True, eq=False)
class NullReplicates:
    """One split's simulated null: each check's level, read from ``n_draws`` replicates.

    ``levels`` maps each of ``CHECK_REPORTS`` to the level its band is drawn at, Barnard's rule
    at ``CHECK_ALPHA`` over its replicate statistics: Q-Q's local level, the PIT and quantile
    per-cell cuts, and actual over expected's ``z``. Each is one float, exactly the statistic of
    the replicate at that rank. A check maps to ``None`` when it has no level: it is
    unavailable on its own (actual over expected without a finite variance, or with tables the
    rows refuse), its replicate statistics give no finite level, or nothing was simulated.

    ``note`` is the page's sentence for each check that keeps the known-parameter level on this
    split for want of a replicate: the fit's uncertainty could not be sampled, or these rows
    could not be simulated (for every check, for the checks a capped split's sample does not
    reach, or for a check whose level could not be formed). It is ``None`` when every check
    that answers was simulated, when the known level is the right one (the rows the model was
    fitted on), or when no check can answer. Nothing simulated leaves ``n_draws`` 0.

    ``log_levels`` maps Q-Q, PIT and quantile calibration to the natural log of their level
    where it lies below twice the least normal float, read by Barnard's rule from the
    statistics' logs. ``levels`` then holds float64's rounding of it, zero or a subnormal, and
    the band is drawn at the log. Every other check maps to ``None``.
    """

    levels: Mapping[str, float | None]
    n_draws: int
    seed: int
    note: str | None = None
    log_levels: Mapping[str, float | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("levels", "log_levels"):
            given = getattr(self, name)
            floats = {
                report: None if (level := given.get(report)) is None else float(level)
                for report in CHECK_REPORTS
            }
            object.__setattr__(self, name, MappingProxyType(floats))


class _Split(NamedTuple):
    """A split's retained rows as every check scores them and as each replicate draws them.

    A replicate draws one response per simulated occurrence. Under prior or unit weights these
    are the retained rows, of which Q-Q and PIT read ``read``, the rows ``replication_sample``
    selects (``None`` for all). Under other frequency weights they are the occurrences
    ``replication_sample`` selects, all read; ``per_occurrence`` marks them, each occurrence's
    atom randomised on its own. ``complete`` says they are every occurrence that quantile
    calibration of the response and actual over expected read. ``rows`` is each occurrence's
    retained row, ``prior_law`` its prior weight where the row law carries one, and
    ``total_weights`` each retained row's weight on one occurrence's response in an actual
    total: its prior weight, or one under frequency weights.
    """

    fitted: Any
    residuals: ResidualSet
    frame: EagerFrame
    offsets: Mapping[str, NDArray] | None
    rows: NDArray[np.intp]
    read: NDArray[np.intp] | None
    prior_law: NDArray[np.float64] | None
    total_weights: NDArray[np.float64]
    per_occurrence: bool
    complete: bool


def _split(fitted: Any, dataset: EvaluationDataset) -> _Split:
    """``dataset``'s rows as the checks score them; ``_NotSampled`` when no replicate can.

    A replicate can be drawn for every row under prior weights, and under frequency weights
    for the occurrences ``replication_sample`` selects, which it never expands in full. A
    fractional weight has no literal occurrence, and a truncated or censored row no law a
    replicate draws, so neither is drawn.
    """
    y = np.asarray(dataset.y, dtype=np.float64).ravel()
    residuals = compute_residuals(
        fitted,
        dataset.X,
        y,
        sample_weight=dataset.sample_weight,
        offsets=dataset.offset,
        seed=CHECK_SEED,
    )
    if residuals.bounds is not None:
        raise _NotSampled(_UNSIMULATED_ROWS)
    retained = _retained_rows(
        fitted, dataset.X, y, sample_weight=dataset.sample_weight, offsets=dataset.offset
    )
    sample = replication_sample(residuals, seed=CHECK_SEED)
    counts = residuals.weights
    # The rule ``_sample_residuals`` follows: such weights read each occurrence on its own.
    per_occurrence = residuals.weight_semantics == "frequency" and bool(np.any(counts != 1.0))
    if per_occurrence:
        if not np.array_equal(counts, np.rint(counts)):
            raise _NotSampled(_UNSIMULATED_ROWS)
        rows, read, total_weights = sample, None, np.ones(residuals.n_rows)
        frame = as_eager_frame(retained.frame.take_rows(rows))
        complete = len(sample) == float(np.sum(counts))
    else:
        rows = np.arange(residuals.n_rows, dtype=np.intp)
        read = None if np.array_equal(sample, rows) else sample
        total_weights, frame, complete = retained.weights, retained.frame, True
    offsets = retained.offsets
    if offsets is not None:
        offsets = {name: np.asarray(values)[rows] for name, values in offsets.items()}
    law = retained.prior_law
    return _Split(
        fitted=fitted,
        residuals=residuals,
        frame=frame,
        offsets=offsets,
        rows=rows,
        read=read,
        prior_law=None if law is None else law[rows],
        total_weights=total_weights,
        per_occurrence=per_occurrence,
        complete=complete,
    )


def _draws(fitted: Any, n_draws: int, seed: int) -> PosteriorDraws:
    """The replicates' coefficients, ``N(beta_hat, V)``; ``_NotSampled`` when ``V`` refuses."""
    try:
        return posterior_draws(fitted, n_draws, covariance="fixed", seed=seed)
    except ValueError as refused:
        raise _NotSampled(_UNSAMPLED_FIT) from refused


def _simulated_responses(
    split: _Split, draws: PosteriorDraws, seed: int
) -> Iterator[tuple[slice, NDArray[np.float64]]]:
    """Yield ``(occurrences, responses)`` blocks, one response row per draw.

    Each draw is pushed to the occurrences' parameters and each response drawn through the
    family's quantile function at a uniform, as ``posterior_predictive`` draws them. A draw
    that leaves a parameter without a value or outside its support on some row, or a response
    that is not finite, is ``_NotSampled``.
    """
    family = split.fitted.family
    uniforms = _residual_rng(seed, _RESPONSE_STREAM)
    blocks = posterior_parameters(
        split.fitted,
        split.frame,
        draws,
        offsets=split.offsets,
        chunk_rows=_SIMULATION_ROWS,
        name_levels=False,
    )
    for span, theta in _within_support(blocks):
        if not np.all(np.isfinite(theta)):
            raise _NotSampled(_UNSIMULATED_ROWS)
        count, width, k = theta.shape
        levels = uniforms.uniform(_PROBABILITY_MARGIN, 1.0 - _PROBABILITY_MARGIN, (count, width))
        flat = theta.reshape(-1, k)
        drawn = (
            family.quantile(levels.ravel(), flat)
            if split.prior_law is None
            else family.quantile_prior_weighted(
                levels.ravel(), flat, np.tile(split.prior_law[span], count)
            )
        )
        responses = np.asarray(drawn, dtype=np.float64).reshape(count, width)
        if not np.all(np.isfinite(responses)):
            raise _NotSampled(_UNSIMULATED_ROWS)
        yield span, responses


def _within_support(blocks: Iterator[tuple[slice, NDArray]]) -> Iterator[tuple[slice, NDArray]]:
    """``posterior_parameters``' blocks; a draw it refuses as outside a support is
    ``_NotSampled``, and any other refusal a fault."""
    try:
        yield from blocks
    except ValueError as refused:
        if _OUTSIDE_SUPPORT not in str(refused):
            raise
        raise _NotSampled(_UNSIMULATED_ROWS) from refused


def _transforms(
    split: _Split,
    span: slice,
    responses: NDArray,
    randomise: Callable[[NDArray, NDArray], NDArray] | None,
) -> NDArray:
    """Each response's transform under the fitted law, randomised and clipped as the split's."""
    family = split.fitted.family
    count, width = responses.shape
    theta = np.tile(split.residuals.theta[split.rows[span]], (count, 1))
    values = responses.ravel()
    law = None if split.prior_law is None else np.tile(split.prior_law[span], count)
    upper = np.asarray(
        family.cdf(values, theta) if law is None else family.cdf_prior_weighted(values, theta, law),
        dtype=np.float64,
    ).reshape(count, width)
    if randomise is not None:
        lower = np.asarray(
            family.cdf_left_limit(values, theta, weights=law), dtype=np.float64
        ).reshape(count, width)
        upper = randomise(lower, upper)
    return np.clip(upper, _PROBABILITY_FLOOR, _PROBABILITY_CEILING)


def _atom_randomiser(split: _Split, count: int) -> Callable[[NDArray, NDArray], NDArray]:
    """Each replicate's transforms, each atom ``lower < upper`` drawn as the split's own are.

    ``compute_residuals`` draws ``uniform(lower, upper)`` for every row, atom or not, from
    stream 0 of ``CHECK_SEED``, and ``_sample_residuals`` keeps those under prior or unit
    weights. Under other frequency weights it draws each selected occurrence at an atom anew,
    in order, from stream 1, and nothing for the others. Each replicate continues its own copy
    of that stream block by block, so it is randomised as a split with its responses would be.
    """
    streams = [_residual_rng(CHECK_SEED, 1 if split.per_occurrence else 0) for _ in range(count)]

    def randomise(lower: NDArray, upper: NDArray) -> NDArray:
        drawn = np.array(upper, dtype=np.float64)
        for stream, row, low, high in zip(streams, drawn, lower, upper, strict=True):
            atoms = low < high
            if split.per_occurrence:
                row[atoms] = stream.uniform(low[atoms], high[atoms])
            else:
                row[atoms] = stream.uniform(low, high)[atoms]
        return drawn

    return randomise


def _cell_totals(codes: NDArray[np.intp], values: NDArray, width: int) -> NDArray:
    """Per draw, the sum of ``values`` over each cell: ``codes`` is ``(rows, tables)``."""
    count = len(values)
    cells = codes[None, :, :] + (width * np.arange(count))[:, None, None]
    weights = np.broadcast_to(values[:, :, None], cells.shape)
    return np.bincount(cells.ravel(), weights=weights.ravel(), minlength=count * width).reshape(
        count, width
    )


def _quantile_thresholds(split: _Split) -> NDArray[np.float64]:
    """Each retained row's predicted quantile at each level of ``QUANTILE_GRID``."""
    law = None if split.prior_law is None else split.residuals.prior_weights
    _, quantile = _row_law(split.fitted.family, law)
    theta = split.residuals.theta
    return np.column_stack(
        [
            np.asarray(quantile(np.full(len(theta), p), theta), dtype=np.float64)
            for p in QUANTILE_GRID
        ]
    )


def _ae_statistics(tables: list[_AETable], totals: NDArray) -> NDArray[np.float64] | None:
    """Per replicate, the largest ``|ratio - 1| / ratio_se`` over the cells with a positive SE."""
    expected = np.concatenate([table.check.expected for table in tables])
    se = np.concatenate([table.check.ratio_se for table in tables])
    cells = _ae_read(se)
    if not np.any(cells):
        return None
    ratio = totals[:, cells] / expected[cells]
    return np.max(np.abs(ratio - 1.0) / se[cells], axis=1)


class _Scores(NamedTuple):
    """What the replicates give: each replicate's sorted transforms the Q-Q plot reads, a
    ``(replicates, n)`` block; each other check's statistic per replicate, or ``None`` where it
    has none; each replicate's PIT histogram; each actual-over-expected table's
    ``(replicates, cells)`` actual totals by name; and the natural logs of the PIT and quantile
    calibration statistics."""

    transforms: NDArray[np.float64]
    statistics: dict[str, NDArray[np.float64] | None]
    counts: NDArray[np.int64]
    totals: dict[str, NDArray[np.float64]]
    log_statistics: dict[str, NDArray[np.float64] | None]

    def levels(self) -> tuple[dict[str, float | None], dict[str, float], bool]:
        """Each check's level by Barnard's rule at ``CHECK_ALPHA``, Q-Q's read exactly; the
        natural log of each level below twice the least normal float, read by the same rule
        from the statistics' logs; and whether some check's level could not be formed.

        Each level is selected on its own: a check whose replicate statistics give no finite
        level (one undefined, too few replicates) has none, and the others keep theirs.
        """
        selections = {
            "check_qq": functools.partial(
                _monte_carlo_order_statistic_level, self.transforms, alpha=CHECK_ALPHA
            )
        }
        for report, values in self.statistics.items():
            if values is not None:
                # Actual over expected's statistic is large when extreme, the others small.
                selections[report] = functools.partial(
                    monte_carlo_level,
                    values,
                    alpha=CHECK_ALPHA,
                    larger_is_extreme=report == "check_ae",
                )
        in_logs = {
            "check_qq": functools.partial(
                _monte_carlo_order_statistic_level, self.transforms, alpha=CHECK_ALPHA, log=True
            )
        }
        for report, values in self.log_statistics.items():
            if values is not None:
                in_logs[report] = functools.partial(monte_carlo_level, values, alpha=CHECK_ALPHA)
        levels: dict[str, float | None] = dict.fromkeys(self.statistics)
        log_levels: dict[str, float] = {}
        for report, select in selections.items():
            with contextlib.suppress(ValueError):
                level = select()
                levels[report] = level if math.isfinite(level) else None
                if report in in_logs and level < _LOG_BELOW:
                    log_levels[report] = in_logs[report]()
        refused = any(levels.get(report) is None for report in selections)
        return levels, log_levels, refused


def _scores(
    split: _Split,
    blocks: Iterable[tuple[slice, NDArray[np.float64]]],
    count: int,
    tables: list[_AETable],
) -> _Scores:
    """Each check's statistic on every replicate, each scored as the split's rows are.

    ``blocks`` are ``(occurrences, responses)`` with one response row per replicate. One pass
    keeps each replicate's transforms, its exceedance counts of the predicted quantiles and its
    actual-over-expected cell totals; Q-Q and PIT then read the transforms of the occurrences
    they read. Quantile calibration of the response and actual over expected read every
    occurrence, so they have no statistic unless the split's are all simulated, and ``tables``
    are empty unless they are.
    """
    occurrences = len(split.rows)
    randomised = isinstance(split.fitted.family, AtomFamily)
    randomise = _atom_randomiser(split, count) if randomised else None
    grid = np.asarray(QUANTILE_GRID)
    thresholds = _quantile_thresholds(split) if split.complete and not randomised else None
    starts = np.cumsum([0] + [table.check.expected.size for table in tables])
    codes = np.column_stack(
        [table.codes + start for table, start in zip(tables, starts[:-1], strict=True)]
        or [np.zeros(split.residuals.n_rows, dtype=np.intp)]
    )
    width = max(int(starts[-1]), 1)
    # Only the occurrences Q-Q and PIT read are kept, in ``split.read``'s order: ``order`` sorts
    # them by occurrence, so each block finds its own by a search.
    chosen = np.arange(occurrences, dtype=np.intp) if split.read is None else split.read
    order = np.argsort(chosen, kind="stable")
    by_occurrence = chosen[order]
    read = np.empty((count, len(chosen)))
    exceeded = np.zeros((count, grid.size), dtype=np.int64)
    totals = np.zeros((count, width))
    for span, responses in blocks:
        rows = split.rows[span]
        lo, hi = np.searchsorted(by_occurrence, (span.start, span.stop))
        transforms = _transforms(split, span, responses, randomise)
        read[:, order[lo:hi]] = transforms[:, by_occurrence[lo:hi] - span.start]
        if thresholds is not None:
            exceeded += np.count_nonzero(responses[:, :, None] > thresholds[rows], axis=1)
        if tables:
            totals += _cell_totals(codes[rows], split.total_weights[rows] * responses, width)

    # Binned as ``pit_payload`` bins the split, one replicate at a time.
    pit = np.stack([np.histogram(row, bins=PIT_BINS, range=(0.0, 1.0))[0] for row in read])
    calibration = log_calibration = None
    if randomised:
        # A family with an atom reads quantile calibration on the randomised transform.
        exceeded = np.column_stack([np.count_nonzero(read > p, axis=1) for p in grid])
        calibration, log_calibration = _cell_statistics(exceeded, read.shape[1], 1.0 - grid)
    elif thresholds is not None:
        calibration, log_calibration = _cell_statistics(exceeded, occurrences, 1.0 - grid)
    read.sort(axis=1)
    pit_levels, log_pit_levels = _cell_statistics(pit, read.shape[1], 1.0 / PIT_BINS)
    statistics = {
        "check_pit": pit_levels,
        "check_calibration": calibration,
        "check_ae": _ae_statistics(tables, totals) if tables else None,
    }
    by_table = {
        table.check.covariate: totals[:, start:end]
        for table, start, end in zip(tables, starts[:-1], starts[1:], strict=True)
    }
    log_statistics = {"check_pit": log_pit_levels, "check_calibration": log_calibration}
    return _Scores(read, statistics, pit, by_table, log_statistics)


def _cell_statistics(
    counts: NDArray, trials: Any, probability: Any
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Each replicate's binomial cell level and its natural log."""
    return (
        _binomial_cell_levels(counts, trials, probability),
        _binomial_cell_levels(counts, trials, probability, log=True),
    )


def _replicates(
    fitted: Any,
    dataset: EvaluationDataset,
    inputs: Sequence[str],
    *,
    n_draws: int,
    seed: int,
    responses: NDArray[np.float64] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> tuple[_Split, _Scores, int]:
    """The split, what its replicates give and how many there are; ``_NotSampled`` when none.

    ``responses``, one row per replicate over the occurrences the replicates draw, stand in for
    simulated ones. ``cancelled`` is asked before each block is drawn (see ``null_replicates``).
    """
    y = np.asarray(dataset.y, dtype=np.float64).ravel()
    split = _split(fitted, dataset)
    tables: list[_AETable] = []
    # Actual over expected reads every occurrence, so its tables are built only when every one
    # is simulated. Its preparation stands alone: tables the rows refuse, or a law without a
    # finite variance, leave it without a statistic and the other checks simulated, and its
    # own body meets the same refusal on the same rows.
    if split.complete and _unsupported("check_ae", fitted.family) is None:
        with contextlib.suppress(CheckUnavailable, ValueError):
            tables = _ae_tables(fitted, dataset, y, inputs)
    if responses is None:
        draws = _draws(fitted, n_draws, seed)
        count, blocks = draws.n_draws, _simulated_responses(split, draws, seed)
        if cancelled is not None:
            blocks = _unless_cancelled(blocks, cancelled)
    else:
        given = np.atleast_2d(np.asarray(responses, dtype=np.float64))
        spans = (
            slice(start, min(start + _SIMULATION_ROWS, given.shape[1]))
            for start in range(0, given.shape[1], _SIMULATION_ROWS)
        )
        count, blocks = len(given), ((span, given[:, span]) for span in spans)
    return split, _scores(split, blocks, count, tables), count


def null_replicates(
    model: Any,
    dataset: EvaluationDataset,
    *,
    inputs: Sequence[str],
    n_draws: int = CHECK_DRAWS,
    seed: int = CHECK_SEED,
    fitted: Any = None,
    cancelled: Callable[[], bool] | None = None,
) -> NullReplicates:
    """Simulate the null of ``dataset``'s checks once and read each check's level from it.

    ``n_draws`` coefficient sets are drawn from ``N(beta_hat, V)``, the fit's posterior
    (:func:`superglm.distributional.posterior.posterior_draws`, fixed smoothing), and each is
    pushed to the split's rows. Each replicate draws one response per occurrence the checks
    read through the family's quantile function (prior weights and offsets as the split's rows
    carry them; each frequency occurrence its own response), and is scored with the fitted
    model exactly as the split is: the same transform, the same randomisation of atoms, the
    same cells. Each check's level is Barnard's rule over its replicates' statistics, and only
    the levels are kept. ``inputs`` are the columns actual over expected groups by.

    The rows the model was fitted on (the split named ``"train"``) keep the known-parameter
    bands, so nothing is simulated for them, nor for a model or split no check can answer.
    Where the posterior refuses its draws, or the rows have no law a replicate can draw
    (truncated or censored rows, a draw that leaves a parameter outside its link's domain, a
    family with no distribution function), nothing is simulated and ``note`` says which.
    Beyond the residual sampler's cap a frequency-weighted split draws the occurrences that
    sampler selects, the ones Q-Q, PIT and a randomised quantile calibration read; quantile
    calibration of the response and actual over expected read every occurrence, so they keep
    the known level and ``note`` says why.

    ``fitted``, the fitted model ``model`` held when the caller captured it (``model``'s own
    when omitted), is the fit simulated: a refit in place replaces ``model``'s, so a caller
    that keys the result on a captured fit simulates that fit. ``cancelled`` is asked before
    each block of rows is drawn, and once it answers true the simulation stops there with
    :class:`SimulationCancelled`.
    """
    nothing = NullReplicates(levels={}, n_draws=0, seed=int(seed))
    lss = _lss(model)
    if lss is None or dataset.name == _FIT_ROWS or lss_split_reason(lss, dataset) is not None:
        return nothing
    fitted = lss._require_fitted() if fitted is None else fitted
    family = fitted.family
    if not isinstance(family, DistributionFunctionFamily) or response_spec(family).shape:
        return NullReplicates(levels={}, n_draws=0, seed=int(seed), note=_UNSIMULATED_ROWS)
    try:
        split, scores, count = _replicates(
            fitted, dataset, inputs, n_draws=int(n_draws), seed=int(seed), cancelled=cancelled
        )
    except NoPredictedMeanError:
        return nothing
    except _NotSampled as unsampled:
        return NullReplicates(levels={}, n_draws=0, seed=int(seed), note=unsampled.note)
    levels, log_levels, refused = scores.levels()
    return NullReplicates(
        levels=levels,
        n_draws=count,
        seed=int(seed),
        note=_UNSIMULATED_ROWS if refused or not split.complete else None,
        log_levels=log_levels,
    )


def _replicate_responses(
    model: Any, dataset: EvaluationDataset, *, n_draws: int = CHECK_DRAWS, seed: int = CHECK_SEED
) -> NDArray[np.float64]:
    """The responses :func:`null_replicates` scores, one row per replicate, for tests to read."""
    fitted = _lss(model)._require_fitted()
    split = _split(fitted, dataset)
    blocks = _simulated_responses(split, _draws(fitted, n_draws, seed), seed)
    return np.concatenate([responses for _, responses in blocks], axis=1)


def _scored_replicates(
    model: Any,
    dataset: EvaluationDataset,
    *,
    inputs: Sequence[str],
    responses: NDArray[np.float64] | None = None,
    n_draws: int = CHECK_DRAWS,
    seed: int = CHECK_SEED,
) -> _Scores:
    """What :func:`null_replicates` reads its levels from, PIT histograms and cell totals too,
    for tests.

    ``responses``, one row per replicate over the occurrences the replicates draw, are scored
    in place of simulated ones.
    """
    fitted = _lss(model)._require_fitted()
    return _replicates(fitted, dataset, inputs, n_draws=n_draws, seed=seed, responses=responses)[1]


def check_report_payload(
    report: str,
    *,
    dataset: EvaluationDataset | None,
    compares_edit: bool,
    edited: dict[str, Any],
    original: dict[str, Any] | None,
) -> dict[str, Any]:
    """The ``/report`` payload of ``report``: the edited model's body and the fitted one's.

    ``split`` is ``None`` for a session with no rows to read, whose bodies are unavailable; its
    ``rows_id`` names the rows (``EvaluationDataset.rows_id``), which the page's zoom is kept on.
    """
    split = (
        None
        if dataset is None
        else {
            "name": dataset.name,
            "label": dataset.label,
            "n_obs": dataset.n_obs,
            "rows_id": dataset.rows_id,
        }
    )
    return {
        "report": report,
        "available": edited["available"],
        "reason": edited["reason"],
        "split": split,
        "compares_edit": bool(compares_edit),
        "models": {"edited": edited, "original": original},
    }


__all__ = [
    "AE_BINS",
    "CHECK_ALPHA",
    "CHECK_DRAWS",
    "CHECK_REPORTS",
    "CHECK_SEED",
    "PIT_BINS",
    "QQ_DISPLAY_POINTS",
    "QUANTILE_GRID",
    "CheckUnavailable",
    "NullReplicates",
    "SimulationCancelled",
    "check_body",
    "check_inputs",
    "check_report_payload",
    "null_replicates",
    "offered_checks",
    "qq_from_residuals",
]
