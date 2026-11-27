"""The editor's four checks of a SuperLSS's predicted distribution on one split.

Each check reads the split through the library's own builders and holds its
flag to one band for the whole check, so a calibrated model is flagged with
chance at most ``CHECK_ALPHA`` however many points or cells the check reads:

* ``check_qq``: the sorted randomised quantile residuals (Dunn and Smyth 1996)
  against the equal local levels band (:func:`superglm.stats.equal_local_levels_band`),
  mapped to the normal scale;
* ``check_pit``: the PIT histogram, each bin against its binomial band at
  ``CHECK_ALPHA / PIT_BINS`` (Bonferroni over the bins);
* ``check_calibration``: the realised exceedance of each predicted quantile
  against :func:`superglm.stats.bonferroni_binomial_band` over the grid;
* ``check_ae``: actual over expected by band of the predicted Mean and by each
  input, each interval at ``z = Phi^{-1}(1 - CHECK_ALPHA / (2 cells))``.

A body is JSON-clean: a value with no finite number is ``None``. A check the
model or split cannot answer is the unavailable body, whose ``reason`` is a
sentence the page shows.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import special, stats

from superglm._frame import as_eager_frame
from superglm.distributional.api import SuperLSS
from superglm.distributional.checks.calibration import (
    _retained_rows,
    actual_expected_check,
    calibration_payload,
)
from superglm.distributional.checks.pit import pit_payload
from superglm.distributional.family import DefaultPredictionFamily, DistributionFunctionFamily
from superglm.distributional.residuals import _sample_residuals, compute_residuals
from superglm.distributional.response import response_spec
from superglm.distributional.weights import resolve_likelihood_weights
from superglm.editor.errors import no_quote_message
from superglm.editor.evaluation import EvaluationDataset
from superglm.editor.lss.validation import lss_split_reason
from superglm.editor.lss.view import LSSModelView, column_of, is_lss, lss_key
from superglm.model.base import NoPredictedMeanError
from superglm.stats.bands import bonferroni_binomial_band, equal_local_levels_band

#: The checks, cheapest first.
CHECK_REPORTS = ("check_pit", "check_ae", "check_calibration", "check_qq")
#: Global level of each check: a calibrated model is flagged with chance at most this.
CHECK_ALPHA = 0.05
#: Seed of the randomised residuals and of any simulated variance, the same for every model.
CHECK_SEED = 42
#: Bins of the PIT histogram.
PIT_BINS = 20
#: Bins of the predicted Mean and of each numeric input in actual over expected.
AE_BINS = 10
#: Probabilities whose predicted quantiles quantile calibration reads.
QUANTILE_GRID = (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99, 0.995)
#: Order statistics a Q-Q body sends evenly spaced, beside its ends and every point outside.
QQ_DISPLAY_POINTS = 800
#: Lowest and highest order statistics a Q-Q body always sends.
_QQ_ENDS = 10

_LABELS = {
    "check_qq": "Q-Q plot",
    "check_pit": "PIT histogram",
    "check_ae": "actual over expected",
    "check_calibration": "quantile calibration",
}
_NOT_LSS = "This check is available for SuperLSS models."
_NO_MEAN = "This model's family names no predicted Mean, so it has no actual over expected."
_NO_VARIANCE = (
    "The predicted distribution has no finite variance at these parameters, "
    "so actual over expected has no interval."
)


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


def qq_from_residuals(quantile: NDArray[np.float64]) -> dict[str, Any]:
    """The Q-Q body of the quantile residuals ``quantile``, in any order.

    The band holds for all ``n`` order statistics at once, and ``outside``
    counts every one that leaves it. ``index`` names the order statistics the
    body sends: ``QQ_DISPLAY_POINTS`` evenly spaced, the ``10`` lowest and
    highest and every one outside the band.
    """
    z = np.sort(np.asarray(quantile, dtype=np.float64).ravel())
    n = len(z)
    band = equal_local_levels_band(n, alpha=CHECK_ALPHA)
    lower, upper = special.ndtri(band.lower), special.ndtri(band.upper)
    outside = (z < lower) | (z > upper)
    spaced = np.rint(np.linspace(0, n - 1, min(n, QQ_DISPLAY_POINTS))).astype(np.intp)
    ends = np.r_[np.arange(min(_QQ_ENDS, n)), np.arange(max(n - _QQ_ENDS, 0), n)]
    index = np.union1d(np.union1d(spaced, ends), np.flatnonzero(outside)).astype(np.intp)
    body = {
        "n": n,
        "alpha": CHECK_ALPHA,
        "local_level": float(band.local_level),
        "index": index,
        "theoretical": special.ndtri(band.median[index]),
        "observed": z[index],
        "band_lower": lower[index],
        "band_upper": upper[index],
    }
    return _answered(body, outside=int(np.count_nonzero(outside)), cells=n)


def _pit_body(residuals: Any) -> dict[str, Any]:
    payload = pit_payload(residuals, n_bins=PIT_BINS, alpha=CHECK_ALPHA / PIT_BINS)
    counts = payload.counts
    outside = int(np.count_nonzero((counts < payload.band_lower) | (counts > payload.band_upper)))
    return _answered({**payload.to_json(), "alpha": CHECK_ALPHA}, outside=outside, cells=PIT_BINS)


def _calibration_body(fitted: Any, dataset: EvaluationDataset, y, residuals) -> dict[str, Any]:
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
    low, high = bonferroni_binomial_band(trials, 1.0 - p, alpha=CHECK_ALPHA)
    with np.errstate(divide="ignore", invalid="ignore"):
        band_lower, band_upper = low / trials, high / trials
    shares = table["realised_exceedance"].to_numpy(dtype=np.float64)
    outside = int(np.count_nonzero((shares < band_lower) | (shares > band_upper)))
    coverage = payload.coverage[payload.coverage["group"] == "all"]
    body = {
        "quantiles": {
            name: table[name].to_numpy()
            for name in ("p", "n", "weight", "expected", "realised_exceedance")
        },
        "band_lower": band_lower,
        "band_upper": band_upper,
        "coverage": {
            "level": coverage["level"].to_numpy(),
            "realised": coverage["realised"].to_numpy(),
        },
        "calibration_law": payload.calibration_law,
    }
    return _answered(body, outside=outside, cells=len(QUANTILE_GRID))


def _ae_bins(values: NDArray, positions: NDArray[np.intp]) -> int:
    """``AE_BINS``, or fewer for a numeric covariate with fewer distinct retained values.

    Equal-count edges interpolated between fewer values than bins would leave bins empty.
    """
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.number) or array.dtype.kind == "b":
        return AE_BINS
    return max(1, min(AE_BINS, np.unique(array[positions]).size))


def _ae_table(fitted, dataset, y, covariate, positions, name: str, by: str) -> dict[str, Any]:
    try:
        table = actual_expected_check(
            fitted,
            dataset.X,
            y,
            covariate,
            name=name,
            sample_weight=dataset.sample_weight,
            n_bins=_ae_bins(covariate, positions),
            offsets=dataset.offset,
            seed=CHECK_SEED,
        )
    except NoPredictedMeanError:
        raise
    except ValueError as exc:
        # Only a law without finite variance is a reason; any other error is a fault.
        if "finite" in str(exc) and "variance" in str(exc):
            raise CheckUnavailable(_NO_VARIANCE) from None
        raise
    return {**table.to_json(), "by": by}


def _ae_body(fitted, dataset: EvaluationDataset, y, inputs: Sequence[str]) -> dict[str, Any]:
    rows = _retained_rows(
        fitted, dataset.X, y, sample_weight=dataset.sample_weight, offsets=dataset.offset
    )
    theta = fitted.parameters_from_eta(fitted.predict_eta(rows.frame, offsets=rows.offsets))
    band = np.full(len(y), np.nan)
    band[rows.positions] = fitted.family.default_prediction(theta)
    tables = [_ae_table(fitted, dataset, y, band, rows.positions, "predicted Mean", "band")]
    frame = as_eager_frame(dataset.X)
    for name in inputs:
        if name in frame.columns:
            column = frame.column_array(name)
            tables.append(_ae_table(fitted, dataset, y, column, rows.positions, name, "input"))

    ratios = [np.asarray(t["ratio"], dtype=np.float64) for t in tables]
    errors = [np.asarray(t["ratio_se"], dtype=np.float64) for t in tables]
    cells = int(sum(np.count_nonzero(np.isfinite(se)) for se in errors))
    z = float(stats.norm.isf(CHECK_ALPHA / (2.0 * cells))) if cells else None
    outside = 0
    for table, ratio, se in zip(tables, ratios, errors, strict=True):
        with np.errstate(invalid="ignore"):
            half = np.full(len(se), np.nan) if z is None else z * se
        table["lower"], table["upper"] = ratio - half, ratio + half
        outside += int(np.count_nonzero((ratio - half > 1.0) | (ratio + half < 1.0)))
    actual = float(np.sum(tables[0]["actual"]))
    expected = float(np.sum(tables[0]["expected"]))
    body = {
        "tables": tables,
        "z": z,
        "overall": {"actual": actual, "expected": expected, "ratio": actual / expected},
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


def _body(report: str, model: Any, dataset: EvaluationDataset, inputs: Sequence[str]):
    lss = _lss(model)
    if lss is None:
        raise CheckUnavailable(_NOT_LSS)
    fitted = lss._require_fitted()
    reason = _unsupported(report, fitted.family) or lss_split_reason(lss, dataset)
    if reason is not None:
        raise CheckUnavailable(reason)
    y = np.asarray(dataset.y, dtype=np.float64).ravel()
    try:
        if report == "check_ae":
            return _ae_body(fitted, dataset, y, inputs)
        residuals = compute_residuals(
            fitted,
            dataset.X,
            y,
            sample_weight=dataset.sample_weight,
            offsets=dataset.offset,
            seed=CHECK_SEED,
        )
        if report == "check_qq":
            return qq_from_residuals(_sample_residuals(residuals, seed=CHECK_SEED).quantile)
        if report == "check_pit":
            return _pit_body(residuals)
        return _calibration_body(fitted, dataset, y, residuals)
    except NoPredictedMeanError as refused:
        raise CheckUnavailable(_no_quote(fitted, dataset, y, refused)) from None


def check_body(
    report: str, model: Any, dataset: EvaluationDataset, *, inputs: Sequence[str]
) -> dict[str, Any]:
    """One model's body of ``report`` on ``dataset``; ``inputs`` are the columns A/E groups by."""
    if report not in CHECK_REPORTS:
        raise ValueError(f"report must be one of {CHECK_REPORTS}; got {report!r}")
    try:
        return _body(report, model, dataset, inputs)
    except CheckUnavailable as unavailable:
        return _unavailable(unavailable.reason)


def check_report_payload(
    report: str,
    *,
    dataset: EvaluationDataset,
    compares_edit: bool,
    edited: dict[str, Any],
    original: dict[str, Any] | None,
) -> dict[str, Any]:
    """The ``/report`` payload of ``report``: the edited model's body and the fitted one's."""
    return {
        "report": report,
        "available": edited["available"],
        "reason": edited["reason"],
        "split": {"name": dataset.name, "label": dataset.label, "n_obs": dataset.n_obs},
        "compares_edit": bool(compares_edit),
        "models": {"edited": edited, "original": original},
    }


__all__ = [
    "AE_BINS",
    "CHECK_ALPHA",
    "CHECK_REPORTS",
    "CHECK_SEED",
    "PIT_BINS",
    "QQ_DISPLAY_POINTS",
    "QUANTILE_GRID",
    "CheckUnavailable",
    "check_body",
    "check_inputs",
    "check_report_payload",
    "offered_checks",
    "qq_from_residuals",
]
