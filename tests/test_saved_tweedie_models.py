"""Tweedie models saved by superglm v0.35.0 load, predict and report (design §3.12, T8).

The fixtures in ``fixtures/saved_v0_35_0`` were written by the v0.35.0 release
(commit ea837196) with ``scripts/make_saved_tweedie_fixtures.py`` on small
synthetic compound Poisson-Gamma data at p = 1.5.  A v0.35.0 REML fit of a
Tweedie model pickles its saturated-density memo
(``superglm.reml.scale.TweedieScaleProfileData`` holding a
``superglm.profiling.tweedie._PreparedTweedieDensity``), and ``estimate_p``
pickles its search, density and interval classes; #422 deleted all of them,
so without the module ``__getattr__`` of ``reml.scale`` and
``profiling.tweedie`` none of these models loads.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np
import pytest
from matplotlib.figure import Figure

from superglm.profiling.tweedie import SavedTweedieProfileResult
from superglm.reml.penalty_algebra import build_penalty_matrix
from superglm.solvers._structured.block_leaves import FactorSmoothLeafFactor
from superglm.solvers._structured.nested import NestedSchurFactor

from .test_saved_fs_models import _se_tolerance, assert_predicts_as_saved

FIXTURES = Path(__file__).parent / "fixtures" / "saved_v0_35_0"
NOTICE = "rebuilt with the current solver"


def _load(name: str) -> dict:
    with open(FIXTURES / f"{name}.pkl", "rb") as handle:
        return pickle.load(handle)


def _metrics_se(model, frame, y):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        se = model.metrics(frame, y).coefficient_se
    return se, sum(NOTICE in str(item.message) for item in caught)


def _dense_se(model) -> np.ndarray:
    """``sqrt(phi diag(H^-1))`` of the model's own design at its saved coefficients."""
    from superglm.model.state_ops import _solver_space_working_weights

    dm = model._dm
    X = np.hstack([np.ones((dm.n, 1)), dm.toarray()])
    weights = _solver_space_working_weights(model)
    H = X.T @ (weights[:, None] * X)
    H[1:, 1:] += build_penalty_matrix(
        dm.group_matrices, model._groups, model._reml_lambdas, dm.p, model._reml_penalties
    )
    return np.sqrt(np.diag(np.linalg.inv(H))[1:] * model._result.phi)


def _assert_se_close(rebuilt: dict, reference: dict, tolerance: float, pattern=True) -> None:
    for term, expected in reference.items():
        got = np.asarray(rebuilt[term], dtype=float)
        if pattern:
            np.testing.assert_array_equal(np.isfinite(got), np.isfinite(expected))
        finite = np.isfinite(got)
        relative = np.abs(got[finite] - expected[finite]) / np.abs(expected[finite])
        assert np.all(relative <= tolerance), (term, float(np.max(relative)), tolerance)


@pytest.mark.parametrize(
    ("name", "retired", "factor"),
    [
        ("tweedie_re_exact", "ScalarSchurFactor", NestedSchurFactor),
        ("tweedie_fs_discrete", "BlockSchurFactor", FactorSmoothLeafFactor),
    ],
)
def test_a_tweedie_model_saved_by_v0_35_0_loads_predicts_and_rebuilds(
    name, retired, factor
) -> None:
    """Predictions are v0.35.0's within two evaluations' rounding; the first inference call rebuilds the retired
    structured state once, with the notice, at the saved coefficients and
    smoothing parameters, and the REML memo is gone.

    The rebuilt standard errors are those of the model's own penalized Hessian
    at the saved coefficients, within the variances' ``n u kappa_s``
    (``_se_tolerance``), with v0.35.0's pattern of reported values.  On the fs
    discrete fixture v0.35.0's own reported errors sit 3.5e-5 from that
    Hessian (measured on v0.35.0 itself: its factor was built at other working
    weights), so only the exact random-effect fixture is also held to them.
    """
    record = _load(name)
    assert record["version"] == "0.35.0"
    assert record["state_types"]["augmented_factor"] == retired
    assert record["scale_data_type"] == "TweedieScaleProfileData"
    model = record["model"]
    frame, y = record["frame"], record["y"]
    assert model._reml_result.tweedie_scale_data is None

    assert_predicts_as_saved(model, frame, record["prediction"])
    se, notices = _metrics_se(model, frame, y)
    assert notices == 1
    assert isinstance(model._linear_system_state.augmented_factor, factor)

    tolerance = _se_tolerance(model)
    dense = _dense_se(model)
    assert dense.size == sum(np.asarray(value).size for value in se.values())
    offsets = np.cumsum([0] + [np.asarray(value).size for value in se.values()])
    reference = {term: dense[offsets[i] : offsets[i + 1]] for i, term in enumerate(se)}
    # the reported entries (a random-effect level's is NaN by the package's
    # convention, and its pattern is v0.35.0's)
    _assert_se_close(se, reference, tolerance, pattern=False)
    for term, saved in record["se"].items():
        np.testing.assert_array_equal(np.isfinite(se[term]), np.isfinite(saved))
    if name == "tweedie_re_exact":
        _assert_se_close(se, record["se"], tolerance)

    assert str(model.summary())
    _, notices = _metrics_se(model, frame, y)
    assert notices == 0


def test_a_tweedie_model_saved_by_v0_35_0_on_gram_loads_and_reports_as_saved() -> None:
    """Without a structured state nothing is rebuilt: no notice, and the
    standard errors are the ones v0.35.0 reported."""
    record = _load("tweedie_no_group")
    assert record["version"] == "0.35.0"
    assert record["state_types"] == {}
    model = record["model"]
    frame, y = record["frame"], record["y"]
    assert model._reml_result.tweedie_scale_data is None
    assert_predicts_as_saved(model, frame, record["prediction"])
    se, notices = _metrics_se(model, frame, y)
    assert notices == 0
    _assert_se_close(se, record["se"], _se_tolerance(model))
    assert str(model.summary())


@pytest.mark.parametrize("name", ["tweedie_estimate_p_reml", "tweedie_estimate_p_fit"])
def test_an_estimate_p_model_saved_by_v0_35_0_keeps_its_estimate_and_interval(name) -> None:
    """``estimate_p`` pickled v0.35.0's ``TweedieProfileResult`` (its bound
    search methods and interval details included).  The model reports the
    estimate and the 95% interval v0.35.0 computed; an interval at another
    level needs the retired search and says so; the model saves again."""
    record = _load(name)
    assert record["version"] == "0.35.0"
    saved = record["profile"]
    model = record["model"]
    frame, y = record["frame"], record["y"]
    assert_predicts_as_saved(model, frame, record["prediction"])
    assert model._distribution.p == saved["p_hat"]

    profile = model._tweedie_profile_result
    assert (profile.p_hat, profile.phi_hat, profile.nll) == (
        saved["p_hat"],
        saved["phi_hat"],
        saved["nll"],
    )
    assert saved["p_hat"] in set(profile.evaluations["p"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert profile.ci(0.05) == saved["ci"][0.05]
    info = model.summary()._info
    assert info["tweedie_p"] == saved["p_hat"]
    assert tuple(info["tweedie_p_ci"]) == saved["ci"][0.05]
    assert info["tweedie_p_ci_status"] == "available"
    with pytest.raises(RuntimeError, match="call estimate_p again"):
        profile.ci(0.10)

    se, notices = _metrics_se(model, frame, y)
    assert notices == 0
    if model._reml_lambdas is None:  # an ML search publishes an ML fit: its pattern
        for term, saved_se in record["se"].items():
            np.testing.assert_array_equal(np.isfinite(se[term]), np.isfinite(saved_se))
    else:
        _assert_se_close(se, record["se"], _se_tolerance(model))

    again = pickle.loads(pickle.dumps(model))
    assert_predicts_as_saved(again, frame, record["prediction"])
    assert again._tweedie_profile_result.ci(0.05) == saved["ci"][0.05]


def test_an_estimate_p_result_saved_on_its_own_by_v0_35_0_keeps_its_estimate_and_interval() -> None:
    """The value ``estimate_p`` returns, pickled without its model, restates
    itself on load as a result inside a model does (one ``__setstate__``): it
    reports v0.35.0's estimate and interval, plots them, asks for a new
    ``estimate_p`` at another level, and saves again."""
    record = _load("tweedie_estimate_p_result")
    assert record["version"] == "0.35.0"
    saved = record["profile"]
    result = record["result"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert result.ci(0.05) == saved["ci"][0.05]
        assert (result.interval(0.05).lower, result.interval(0.05).upper) == saved["ci"][0.05]
    assert type(result) is SavedTweedieProfileResult
    assert (result.p_hat, result.phi_hat, result.nll) == (
        saved["p_hat"],
        saved["phi_hat"],
        saved["nll"],
    )
    assert saved["p_hat"] in set(result.evaluations["p"])
    with pytest.raises(RuntimeError, match="call estimate_p again"):
        result.ci(0.10)
    ax = Figure().subplots()
    assert result.profile_plot(ax=ax) is ax

    again = pickle.loads(pickle.dumps(result))
    assert type(again) is SavedTweedieProfileResult
    assert again.ci(0.05) == saved["ci"][0.05]
