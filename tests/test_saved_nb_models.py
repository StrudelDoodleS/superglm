"""Negative-binomial results saved by superglm v0.35.0 load and report (T8).

The fixtures in ``fixtures/saved_v0_35_0`` were written by the v0.35.0 release
(commit ea837196) with ``scripts/make_saved_nb_fixtures.py`` on small synthetic
NB2 counts.  v0.35.0's ``NBProfileResult`` kept each theta interval as a
``(lower, upper)`` tuple and none of the current result's interval state, so
until ``NBProfileResult.__setstate__`` restated it, ``summary()`` on such a
model raised AttributeError, and a summary v0.35.0 had cached raised KeyError
on the interval status it never recorded.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np
import pytest
from matplotlib.figure import Figure

from superglm.profiling.nb import NBProfileResult
from superglm.solvers.dispersion import model_weight_semantics

from .test_saved_fs_models import _se_tolerance, assert_predicts_as_saved

FIXTURES = Path(__file__).parent / "fixtures" / "saved_v0_35_0"


def _load(name: str) -> dict:
    with open(FIXTURES / f"{name}.pkl", "rb") as handle:
        return pickle.load(handle)


def _bounds(interval) -> tuple[float, float]:
    return interval.lower, interval.upper


def _assert_reports_saved_intervals(result: NBProfileResult, saved: dict) -> None:
    """The estimate and each interval as v0.35.0 computed them; a side is censored
    where v0.35.0's likelihood-ratio excess had no crossing before its range's end."""
    assert (result.theta_hat, result.nll, result.converged) == (
        saved["theta_hat"],
        saved["nll"],
        saved["converged"],
    )
    assert list(result.evaluations["theta"]) == list(saved["iterates"])
    assert list(result.evaluations["nll"]) == list(saved["iterates"].values())
    for alpha, bounds in saved["ci"].items():
        interval = result._interval(alpha)
        assert (interval.lower, interval.upper) == bounds
        assert (interval.lower_censored, interval.upper_censored) == saved["no_crossing"][alpha]


@pytest.mark.parametrize(
    "name", ["nb_estimate_theta_fit", "nb_estimate_theta_reml", "nb_near_poisson"]
)
def test_an_estimate_theta_model_saved_by_v0_35_0_reports_its_estimate_and_interval(name) -> None:
    """Predictions are v0.35.0's within two evaluations' rounding, standard errors are v0.35.0's (to ``n u kappa_s``
    where REML published them; an ML fit's pattern), and the summary reports
    v0.35.0's estimate and interval with its censoring.  An interval v0.35.0
    did not compute (the REML fixture has none) is inverted on the profile at
    the model's own published mean, as a current result there inverts it."""
    record = _load(name)
    assert record["version"] == "0.35.0"
    saved = record["profile"]
    model, frame, y = record["model"], record["frame"], record["y"]
    assert_predicts_as_saved(model, frame, record["prediction"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        se = model.metrics(frame, y).coefficient_se
    tolerance = None if model._reml_lambdas is None else _se_tolerance(model)
    for term, expected in record["se"].items():
        got = np.asarray(se[term], dtype=float)
        np.testing.assert_array_equal(np.isfinite(got), np.isfinite(expected))
        if tolerance is not None:
            finite = np.isfinite(got)
            relative = np.abs(got[finite] - expected[finite]) / np.abs(expected[finite])
            assert np.all(relative <= tolerance), (term, float(np.max(relative)), tolerance)

    current = NBProfileResult(
        theta_hat=saved["theta_hat"],
        nll=saved["nll"],
        converged=saved["converged"],
        _y=np.asarray(y, dtype=np.float64),
        _mu=model._fit_mu,
        _weights=model._fit_weights,
        _weight_semantics=model_weight_semantics(model),
    )
    info = model.summary()._info
    assert info["nb_theta"] == saved["theta_hat"]
    if 0.05 in saved["ci"]:
        assert tuple(info["nb_theta_ci"]) == saved["ci"][0.05]
        censored = any(saved["no_crossing"][0.05])
        assert info["nb_theta_ci_status"] == ("censored" if censored else "available")
    else:
        assert tuple(info["nb_theta_ci"]) == _bounds(current._interval(0.05))
        assert info["nb_theta_ci_status"] == "available"
    assert str(model.summary())
    assert model.summary(alpha=0.10)._info["nb_theta_ci"] == _bounds(current._interval(0.10))
    _assert_reports_saved_intervals(model._nb_profile_result, saved)

    again = pickle.loads(pickle.dumps(model))
    assert_predicts_as_saved(again, frame, record["prediction"])
    _assert_reports_saved_intervals(again._nb_profile_result, saved)


def test_a_censored_v0_35_0_interval_warns_where_it_stopped() -> None:
    """The near-Poisson fixture's upper side had no crossing: the restated
    interval warns about it on every call and records it in ``warnings``."""
    profile = _load("nb_near_poisson")["model"]._nb_profile_result
    with pytest.warns(UserWarning, match="censored at its upper end"):
        profile.interval(0.05)
    assert any("censored at its upper end" in message for message in profile.warnings)


def test_an_estimate_theta_result_saved_on_its_own_by_v0_35_0_loads() -> None:
    """The value ``estimate_theta`` returns, pickled without its model, restates
    itself on load as a result inside a model does.  A new level's interval is
    inverted on the saved fixed-mean profile from its optimum, so it lies
    inside v0.35.0's wider one, which was inverted from ``theta_hat``."""
    record = _load("nb_estimate_theta_result")
    assert record["version"] == "0.35.0"
    saved = record["profile"]
    result = record["result"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert result.ci(0.05) == saved["ci"][0.05]
        lower, upper = result.ci(0.10)
    assert type(result) is NBProfileResult
    _assert_reports_saved_intervals(result, saved)
    saved_lower, saved_upper = saved["ci"][0.05]
    assert saved_lower < lower < upper < saved_upper
    ax = Figure().subplots()
    assert result.profile_plot(ax=ax) is ax

    again = pickle.loads(pickle.dumps(result))
    assert again.ci(0.05) == saved["ci"][0.05]
    assert again.ci(0.10) == (lower, upper)
