"""Every reader of a row with no predicted mean uses one rule.

``predict`` gives such a row NaN with a warning; the rows are those outside the
link's domain (``links.outside_eta_domain``). A reader that scores, tests or ranks rows
refuses a row of positive weight with no mean with the named
``NoPredictedMeanError`` (``base.require_priced_rows``), and leaves a zero-weight
one out; a comparison leaves out the rows its full model cannot price and reports
a comparison that loses further rows as unavailable. None reads the stabilizer's
clipped mean, ``1e12`` under the inverse link. Each test fails against the code
before the rule reached its reader.
"""

import pickle
import re
import warnings
from dataclasses import replace

import matplotlib
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

matplotlib.use("Agg")

from matplotlib.figure import Figure  # noqa: E402

from superglm import (  # noqa: E402
    Categorical,
    Constraint,
    Numeric,
    PSpline,
    Spline,
    SuperGLM,
    dispersion_test,
    score_test_zi,
    vuong_test,
)
from superglm._frame import as_eager_frame  # noqa: E402
from superglm.distributions import Gamma  # noqa: E402
from superglm.model.base import NoPredictedMeanError  # noqa: E402
from superglm.model_selection import cross_validate  # noqa: E402
from superglm.sklearn import SuperGLMRegressor  # noqa: E402
from tests._no_mean_fixture import (  # noqa: E402
    CELLS,
    CELLS_Y,
    TRAIN,
    TRAIN_Y,
    UpperBoundedIdentity,
    cv_frame,
    editor_fit,
    negative_power_lss,
    three_cell_fit,
)

_IGNORE_LAST = np.array([1.0, 1.0, 1.0, 0.0])
_NO_MEAN = r"1 of 4 rows have no predicted mean under the InverseLink \(row positions 3\)"


def test_an_overflowing_predictor_is_outside_the_inverse_links_domain():
    """The domain is ``0 < eta < inf``: a finite covariate whose predictor overflows predicts NaN.

    The named warning is the only one, and under warnings as errors ``metrics``
    raises the named error. Mutation: the stabilizer clipped ``eta = inf`` to
    ``1e12``, so the row was priced at ``1e-12``; then NumPy's overflow warning
    came too, and under ``-W error`` was raised in place of the named error.
    """
    x = np.linspace(0.0, 1.0, 40)
    model = SuperGLM(
        family="gamma", link="inverse", selection_penalty=0.0, features={"x": Numeric()}
    ).fit(pd.DataFrame({"x": x}), np.tile([0.9, 1.1], 20) / (1.0 + 5.0 * x))
    far = pd.DataFrame({"x": [0.5, 1e308]})
    assert np.isinf(model._predict_eta_raw_exact(far)[1])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        predicted = model.predict(far)
    assert [type(w.message) for w in caught] == [UserWarning]
    assert re.search(r"1 of 2 rows .*not finite", str(caught[0].message))
    assert np.isfinite(predicted[0]) and np.isnan(predicted[1])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(NoPredictedMeanError, match=r"positions 1\)"):
            model.metrics(far, np.ones(2))


def test_the_named_error_keeps_its_rows_through_a_pickle_round_trip():
    """A fold's refusal crosses a process boundary with its row details.

    Mutation: ``BaseException`` rebuilt the error from its message alone and
    raised ``TypeError`` for the missing keyword arguments.
    """
    model, *_ = three_cell_fit("inverse", [2.0, 6.0])
    with pytest.raises(NoPredictedMeanError) as caught:
        model.metrics(CELLS, CELLS_Y)
    copy = pickle.loads(pickle.dumps(caught.value))
    assert type(copy) is NoPredictedMeanError
    assert str(copy) == str(caught.value)
    assert copy.positions.tolist() == [3] and copy.n_rows == 4 and copy.link == "InverseLink"


def test_a_zero_weight_row_with_no_mean_reads_nan_in_residuals_and_its_raw_eta(monkeypatch):
    """Poisson quantile residuals and the diagnostic plot leave the row out instead of crashing.

    The other rows keep their seeded draws, exactly, and so their residuals, within what two
    evaluations of their mean can move them: the mean's rounding moves the PIT interval
    ``[F(y - 1), F(y)]`` at the slope ``dF(k) / dmu = -pmf(k)``, and ``ndtri`` carries that at
    ``1 / phi`` of the farther residual. ``eta`` reports the row's raw linear predictor, and
    the other rows' within two evaluations' rounding. Mutation: the NaN mean reached
    ``rng.uniform`` and raised ``OverflowError``, and ``eta`` read the clipped ``1e-12``.
    """
    from scipy import special, stats

    from superglm.model.base import _prediction_plan
    from tests.test_saved_fs_models import _gamma, _prediction_rounding, _term_magnitude

    counts = np.array([0.0, 2.0, 2.0, 6.0, 2.0, 6.0] * 5)
    model = editor_fit("poisson", counts)
    real_rng = np.random.default_rng
    streams: list[np.ndarray] = []

    class Recording(np.random.Generator):
        def uniform(self, low=0.0, high=1.0, size=None):
            twin = np.random.Generator(type(self.bit_generator)())
            twin.bit_generator.state = self.bit_generator.state
            out = super().uniform(low, high, size)
            streams.append(twin.random(np.shape(out)))  # the standard draws it consumed
            return out

    monkeypatch.setattr(
        np.random, "default_rng", lambda seed=None: Recording(real_rng(seed).bit_generator)
    )
    metrics = model.metrics(CELLS, CELLS_Y, sample_weight=_IGNORE_LAST)
    seen = model.metrics(CELLS.iloc[:3], CELLS_Y[:3])

    residuals = metrics.residuals("quantile")
    reference = seen.residuals("quantile")
    assert np.isnan(residuals[3])
    assert [len(stream) for stream in streams] == [4, 3]
    np.testing.assert_array_equal(streams[0][:3], streams[1])
    u = 0.5 * np.finfo(np.float64).eps
    mu, d_mu = model.predict(CELLS.iloc[:3]), _prediction_rounding(model, CELLS.iloc[:3])
    below, at = stats.poisson.pmf(CELLS_Y[:3] - 1, mu), stats.poisson.pmf(CELLS_Y[:3], mu)
    # Each CDF and the jittered PIT round within a few ulps of their unit scale.
    d_pit = (below + streams[1] * np.abs(at - below)) * d_mu + 64.0 * u
    pit = stats.norm.cdf(reference)
    farther = np.maximum(np.abs(reference), np.abs(special.ndtri(np.clip(pit + d_pit, 0, 1))))
    assert np.all(np.abs(residuals[:3] - reference) <= d_pit / stats.norm.pdf(farther))

    raw = model._predict_eta_raw_exact(CELLS)
    beta, plan = np.asarray(model.result.beta, dtype=np.float64), _prediction_plan(model)
    magnitude = abs(float(model.result.intercept)) + sum(
        _term_magnitude(term, as_eager_frame(CELLS), beta)
        for term in plan["features"] + plan["interactions"]
    )
    # eta_i is a dot product of p + 1 terms: two evaluations differ by 2 gamma_{p+1} a_i.
    d_eta = 2.0 * _gamma(beta.size + 1) * magnitude
    assert raw[3] < 0.0 and abs(metrics.eta[3] - raw[3]) <= d_eta[3]
    assert np.all(np.abs(metrics.eta[:3] - seen.eta) <= d_eta[:3])
    assert np.isnan(metrics.residuals("deviance")[3])
    assert isinstance(
        model.plot_diagnostics(CELLS, CELLS_Y, sample_weight=_IGNORE_LAST, n_sim=2), Figure
    )


def test_holdout_term_drop_leaves_out_a_row_the_full_model_cannot_price():
    """Every term's delta is read without the row, named in a warning, as at weight zero.

    Mutation: the whole table was refused (``NoPredictedMeanError``); before that the
    full model's deviance read the row at ``1e12``.
    """
    model, X, y, _ = three_cell_fit("inverse", [2.0, 6.0])
    left_out = r"1 of 4 rows of positive weight have no predicted mean .*positions 3\).*term drop"
    with pytest.warns(UserWarning, match=left_out):
        table = model.term_drop_diagnostics(X, y, mode="holdout", X_val=CELLS, y_val=CELLS_Y)
    ignored = model.term_drop_diagnostics(
        X, y, mode="holdout", X_val=CELLS, y_val=CELLS_Y, sample_weight_val=_IGNORE_LAST
    )
    seen = model.term_drop_diagnostics(
        X, y, mode="holdout", X_val=CELLS.iloc[:3], y_val=CELLS_Y[:3]
    )
    pd.testing.assert_frame_equal(table, seen, rtol=1e-12)
    pd.testing.assert_frame_equal(ignored, seen, rtol=1e-12)


def test_a_term_whose_removal_leaves_rows_with_no_mean_reports_its_delta_unavailable():
    """Dropping ``x`` from ``eta = -1 + 2x + 0.1 [g = b]`` leaves every row at ``eta <= 0``.

    Its delta is NaN with the count of those rows, not read without them, and the other term
    is computed. Mutation: the whole table was refused, with advice to zero-weight the rows,
    which would leave out the rows the reduced model prices worst.
    """
    x = np.linspace(1.0, 2.0, 40)
    X = pd.DataFrame({"x": x, "g": np.tile(["a", "b"], 20)})
    mean = 1.0 / (-1.0 + 2.0 * x + 0.1 * (X["g"] == "b").to_numpy())
    y = mean * np.tile([0.9, 1.1], 20)
    model = SuperGLM(
        family="gamma",
        link="inverse",
        selection_penalty=0.0,
        features={"x": Numeric(), "g": Categorical()},
    ).fit(X, y)
    with pytest.warns(UserWarning, match=r"Without 'x' \(6\) the model has no predicted mean"):
        table = model.term_drop_diagnostics(X, y, mode="holdout", X_val=X.iloc[:6], y_val=y[:6])
    by_term = table.set_index("feature")
    assert (
        np.isnan(by_term.loc["x", "delta_deviance"]) and by_term.loc["x", "rows_without_mean"] == 6
    )
    assert by_term.loc["g", "rows_without_mean"] == 0
    # Without g every row is read at its base level's predictor.
    full = model.predict(X.iloc[:6])
    without_g = model.predict(X.iloc[:6].assign(g="a"))
    unit = model._distribution.deviance_unit
    expected = np.sum(unit(y[:6], without_g)) - np.sum(unit(y[:6], full))
    assert by_term.loc["g", "delta_deviance"] == pytest.approx(expected, rel=1e-9)


def _spline_fit():
    """A Gamma/inverse spline fit whose predictor is ``1 + 3x`` up to noise."""
    x = np.linspace(0.0, 1.0, 60)
    return SuperGLM(
        family="gamma", link="inverse", selection_penalty=0.0, features={"x": Spline(n_knots=6)}
    ).fit(pd.DataFrame({"x": x}), (1.0 + 3.0 * x) * np.tile([0.9, 1.0, 1.1], 20))


def test_discretization_impact_leaves_out_a_row_the_fitted_terms_cannot_price():
    """A held-out offset takes one row past ``eta = 0``: the comparison is read without it.

    Mutation: the whole comparison was refused; before that both prediction arrays
    read ``1e12`` there and the deviances were built on it.
    """
    model = _spline_fit()
    holdout = pd.DataFrame({"x": [0.1, 0.4, 0.8]})
    y = np.array([1.0, 2.0, 3.0])
    offset = np.array([0.0, -0.9, 0.0])
    assert model._predict_eta_raw_exact(holdout, offset)[1] < 0.0
    with pytest.warns(UserWarning, match=r"1 of 3 rows of positive weight .*positions 1\)"):
        result = model.discretization_impact(holdout, y, offset=offset, n_bins=4)
    ignored = model.discretization_impact(
        holdout, y, sample_weight=np.array([1.0, 0.0, 1.0]), offset=offset, n_bins=4
    )
    pd.testing.assert_series_equal(pd.Series(result.metrics), pd.Series(ignored.metrics))
    for name, table in result.tables.items():
        pd.testing.assert_frame_equal(table, ignored.tables[name])
    assert np.isfinite(result.metrics["deviance_change"])
    assert np.isnan(result.original_predictions[1])
    assert np.all(np.isfinite(result.original_predictions[[0, 2]]))
    assert np.all(np.isfinite(result.predictions[[0, 2]]))


def test_discretization_reads_each_predictors_domain_on_its_raw_value():
    """One uniform bin replaces the spline by its mean, moving a row's predictor by ``delta``.

    An offset puts a row at ``eta = -delta / 2``, so binning carries it across zero.
    Positive before and tiny (inside the stabilizer's ``1e-12`` clip), it crosses to
    negative: the binned row has no mean, so the comparison is unavailable (NaN) with the
    count. Negative before (a zero-weight row), it crosses to positive: the binned row is
    priced. Mutation: the binned mask was read after the clip, so the first row was priced
    at ``1e12`` and scored; and the original mask was forced onto the binned predictor, so
    the second row read NaN.
    """
    model = _spline_fit()
    held = pd.DataFrame({"x": [0.5, 0.5 + 4e-12]})
    raw = model._predict_eta_raw_exact(held)
    delta = np.mean(raw) - raw
    assert -2e-12 < delta[0] < 0.0
    offset = np.array([-delta[0] / 2.0, 0.5]) - raw
    assert 0.0 < model._predict_eta_raw_exact(held, offset)[0] < 1e-12
    with pytest.warns(UserWarning, match=r"leaves 1 rows .*positions 0\).*unavailable"):
        result = model.discretization_impact(
            held, np.ones(2), offset=offset, n_bins=1, bin_strategy="uniform"
        )
    assert np.isnan(result.predictions[0]) and np.isfinite(result.original_predictions[0])
    assert result.metrics["rows_without_binned_mean"] == 1.0
    assert np.isnan(result.metrics["deviance_change"])
    assert np.isnan(result.metrics["prediction_correlation"])

    held = pd.DataFrame({"x": [0.0, 0.1, 0.5, 0.99, 1.0]})
    weights = np.array([1.0, 1.0, 1.0, 1.0, 0.0])
    plain = model.discretization_impact(held, np.ones(5), sample_weight=weights, n_bins=1)
    eta_fit, eta_bin = 1.0 / plain.original_predictions[-1], 1.0 / plain.predictions[-1]
    offset = np.r_[np.zeros(4), -(eta_fit + eta_bin) / 2.0]
    restored = model.discretization_impact(
        held, np.ones(5), sample_weight=weights, offset=offset, n_bins=1
    )
    assert np.isnan(restored.original_predictions[-1])
    assert restored.predictions[-1] == pytest.approx(2.0 / (eta_bin - eta_fit), rel=1e-9)
    pd.testing.assert_series_equal(pd.Series(restored.metrics), pd.Series(plain.metrics))


def test_statistical_tests_refuse_rows_with_no_mean():
    """Mutation: the Vuong test labelled a NaN statistic "indistinguishable", and the dispersion
    test gave statistic 0 and p-value 0.5 from an undefined mean."""
    inverse = editor_fit()
    # A main-effect-only rival, so the two models differ on every cell.
    log = SuperGLM(
        family="gamma", link="log", selection_penalty=0.0, features={"a": Categorical()}
    ).fit(TRAIN, TRAIN_Y)
    with pytest.raises(NoPredictedMeanError, match=_NO_MEAN + ".*Vuong test"):
        vuong_test(inverse, log, CELLS, CELLS_Y)
    ignored = vuong_test(inverse, log, CELLS, CELLS_Y, sample_weight=_IGNORE_LAST)
    seen = vuong_test(inverse, log, CELLS.iloc[:3], CELLS_Y[:3])
    assert np.isfinite(seen.statistic) and seen.omega > 0.0
    assert_allclose(
        [ignored.statistic, ignored.p_value, ignored.mean_diff, ignored.omega],
        [seen.statistic, seen.p_value, seen.mean_diff, seen.omega],
        rtol=1e-12,
    )
    assert ignored.preferred == seen.preferred

    poisson = editor_fit("poisson", np.array([0.0, 2.0, 2.0, 6.0, 2.0, 6.0] * 5))
    with pytest.raises(NoPredictedMeanError, match=_NO_MEAN + ".*dispersion test"):
        dispersion_test(poisson, CELLS, CELLS_Y)
    with pytest.raises(NoPredictedMeanError, match=_NO_MEAN + ".*zero-inflation"):
        score_test_zi(poisson, CELLS, CELLS_Y)


def test_interaction_screening_refuses_a_weighted_holdout_row_with_no_mean():
    """Mutation: the screening scores read the row at ``1e12`` and came out NaN."""
    model = editor_fit()
    with pytest.raises(NoPredictedMeanError, match=_NO_MEAN + ".*screening"):
        model.screen_interactions(CELLS, CELLS_Y, candidates=[("a", "b")])


class _InverseGamma(Gamma):
    @property
    def default_link(self):
        return "inverse"


def test_sklearn_score_refuses_a_weighted_row_with_no_mean():
    """Mutation: scikit-learn's R² refused the NaN with its own "Input contains NaN"."""
    est = SuperGLMRegressor(
        family=_InverseGamma(),
        selection_penalty=0.0,
        features={"a": Categorical(), "b": Categorical()},
    ).fit(TRAIN, TRAIN_Y)
    with pytest.raises(NoPredictedMeanError, match=_NO_MEAN + ".*R²"):
        est.score(CELLS, CELLS_Y)
    assert_allclose(
        est.score(CELLS, CELLS_Y, sample_weight=_IGNORE_LAST),
        est.score(CELLS.iloc[:3], CELLS_Y[:3]),
        rtol=1e-12,
    )


def test_post_fit_shape_repair_refuses_to_publish_a_weighted_row_with_no_mean():
    """A decreasing repair moves a low-weight, strongly offset row past ``eta = 0``.

    The retained frame adapter takes the fitted-design branch. Mutation: that
    branch stored ``1e12`` as the row's fitted mean and published statistics on it.
    """
    x = np.linspace(0.0, 1.0, 81)
    X = as_eager_frame(pd.DataFrame({"x": x}))
    offset = np.zeros_like(x)
    offset[-1] = -5.0
    weights = np.ones_like(x)
    weights[-1] = 1e-4
    y = (1.0 / (0.5 + 6.0 * x + offset)) * (1.0 + 0.1 * np.sin(np.arange(len(x))))
    model = SuperGLM(
        family="gamma",
        link="inverse",
        selection_penalty=0.0,
        features={"x": PSpline(n_knots=7, constraint=Constraint.postfit.decreasing)},
    ).fit(X, y, offset=offset, sample_weight=weights)
    assert X is model._fit_X_ref
    with pytest.raises(
        NoPredictedMeanError, match=r"1 of 81 rows .*positions 80\).*fit statistics"
    ):
        model.apply_shape_postfit(X, sample_weight=weights, offset=offset)


def test_a_fold_whose_score_is_refused_keeps_its_fit_estimator_and_predictions():
    """The refusal is the score's, not the fold's, and the summary counts the folds scored.

    Mutation: the fold was handled as a failed fit: its out-of-fold predictions
    were all NaN, its estimator None, and its record ``converged=False``,
    ``n_iter=0``; then its refusal named the row by its fold-local position 3.
    """
    X, y, cv = cv_frame()
    model = SuperGLM(
        family="gamma",
        link="inverse",
        selection_penalty=0.0,
        features={"a": Categorical(), "b": Categorical()},
    )
    with pytest.warns(UserWarning) as caught:
        result = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=("deviance", "gini"),
            return_oof=True,
            return_estimators=True,
        )
    messages = [str(warning.message) for warning in caught]
    assert any("1 held-out rows have no predicted mean" in m and "33" in m for m in messages)
    assert any("scored 1 of 2 folds" in m and "Fold 0 refused" in m for m in messages)

    refused, scored = result.fold_scores.to_dict("records")
    estimator = result.estimators[0]
    assert estimator is not None
    assert refused["converged"] and refused["n_iter"] == estimator.result.n_iter
    assert refused["effective_df"] == pytest.approx(3.0)
    assert np.isnan(refused["deviance"]) and np.isnan(refused["gini"])
    # Named by their position in the whole frame, as the out-of-fold warning names them.
    assert re.match(
        r"1 of fold 0's 4 held-out rows have no predicted mean .*positions 33 in the frame\)",
        refused["score refused"],
    )
    assert not isinstance(scored.get("score refused"), str)
    assert np.isfinite(scored["deviance"])
    oof = result.oof_predictions[len(TRAIN_Y) : len(TRAIN_Y) + 4]
    assert np.isnan(oof[3])
    assert_allclose(oof[:3], [1.0, 4.0, 4.0], rtol=1e-6)
    # The scores average and pool the scored fold alone.
    assert result.mean_scores["deviance"] == scored["deviance"]
    assert result.pooled_scores["deviance"] == pytest.approx(scored["deviance"], rel=1e-12)


def test_under_warnings_as_errors_a_built_in_score_raises_the_named_error():
    """The built-in scorers do not warn: a refused row raises the named error, and a zero-weight
    one is scored quietly.

    Mutation: the scorers predicted with predict's warning, which ``-W error``
    raised before the named error, and which failed a valid weighted score.
    """
    X, y, cv = cv_frame()
    model = SuperGLM(
        family="gamma",
        link="inverse",
        selection_penalty=0.0,
        features={"a": Categorical(), "b": Categorical()},
    )
    weights = np.ones_like(y)
    weights[len(TRAIN_Y) + 3] = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with pytest.raises(NoPredictedMeanError, match=r"33 in the frame\).*the deviance") as e:
            cross_validate(model, X, y, cv=cv, error_score="raise")
        assert e.value.positions.tolist() == [33]
        result = cross_validate(
            model,
            X,
            y,
            cv=cv,
            sample_weight=weights,
            scoring=("deviance", "nll", "gini"),
            error_score="raise",
        )
    assert np.all(np.isfinite(result.fold_scores[["deviance", "nll", "gini"]].to_numpy()))


@pytest.mark.pins_behaviour
def test_a_nan_from_a_link_with_no_declared_domain_is_not_a_missing_mean():
    """A control: the rows with no mean are the link's domain's, so a custom link's NaN keeps
    master's reading.

    ``metrics`` scores it (a NaN deviance, as before) and ``discretization_impact``
    raises its named ``OverflowError``. Mutation: every NaN was a missing mean, so
    ``metrics`` refused it as one under an identity link, and the binned NaN went
    unchecked.
    """
    x = np.arange(8.0)
    model = SuperGLM(
        family="gaussian",
        link=UpperBoundedIdentity(),
        selection_penalty=0.0,
        features={"x": Numeric()},
    ).fit(pd.DataFrame({"x": x}), x + np.tile([0.1, -0.1], 4))
    beyond = pd.DataFrame({"x": [2.0, 14.0]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.isnan(model.predict(beyond)[1])
        assert np.isnan(model.metrics(beyond, [2.0, 14.0]).deviance)
    with pytest.raises(OverflowError, match="non-finite or zero predictions on 1 rows"):
        model.discretization_impact(beyond, [2.0, 14.0])


@pytest.mark.parametrize(("power", "level"), [(-1.0, 4.0), (-2.0, 4.0), (-0.5, 9.0)])
def test_superlss_gives_a_row_outside_its_links_domain_no_value_also_once_saved(power, level):
    """A negative power has no mean at ``eta <= 0``, where its inverse's floor quoted 1e15 or 1e30.

    SuperLSS builds and fits with the link. ``predict``, ``predict_quantile`` and
    ``predict_cdf`` give the unseen cell NaN with one warning naming it and its
    link's domain, on the fitted model and on a pickled and a copied one, and the
    other cells their values; a score of the cell is refused with the named
    ``NoPredictedMeanError``. ``level`` is the two seen cells' mean beside the
    first's 1, which puts the unseen cell's predictor ``2 level**p - 1`` clear of
    zero. Mutation: the cell was priced at the floor (``1e15`` under
    ``PowerLink(-1)``) with no warning by every copy, and scored there; the
    quantile and the CDF read the stand-in law's values (those of cell (0, 0)).
    """
    import copy

    from superglm import GammaLS, SuperLSS, cat
    from superglm.links import PowerLink

    family = GammaLS()
    model = SuperLSS(family, family.mean(cat("a"), cat("b"), link=PowerLink(power)), family.scale())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(TRAIN, np.where(TRAIN_Y > 1.5, TRAIN_Y * level / 4.0, TRAIN_Y))
    assert model.predict_link(CELLS).to_numpy()[3, 0] < -0.1
    no_value = (
        r"1 of 4 rows have no predicted value of 'mean' under the PowerLink, which has one only "
        r"at a linear predictor above zero.*positions 3\)"
    )
    for restored in (model, pickle.loads(pickle.dumps(model)), copy.deepcopy(model)):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            predicted = restored.predict(CELLS)
            parameters = restored.predict_parameters(CELLS).to_numpy()
            laws = restored.predict_quantile(CELLS, 0.5), restored.predict_cdf(CELLS, 1.0)
        assert [re.match(no_value, str(w.message)) is not None for w in caught] == [True] * 4
        assert np.isnan(predicted[3]) and np.isnan(parameters[3, 0])
        assert np.isfinite(parameters[3, 1])
        assert_allclose(predicted[:3], [1.0, level, level], rtol=1e-6)
        for values in laws:
            assert np.isnan(values[3]) and np.all(np.isfinite(values[:3]))
    with pytest.raises(NoPredictedMeanError, match="'mean' has no value on 1 of the 4 rows read"):
        model.scores(CELLS, CELLS_Y)


def _frequency_fit(family, link, y, **options):
    return SuperGLM(
        family=family,
        link=link,
        selection_penalty=0.0,
        features={"a": Categorical(), "b": Categorical()},
        **options,
    ).fit(TRAIN, y)


@pytest.mark.parametrize(
    "case",
    ["tweedie-frequency", "binomial-frequency", "nb2-prior"],
)
def test_a_zero_weight_row_with_no_mean_adds_nothing_to_any_score(case):
    """The unseen cell, moved past ``eta = 0`` by an offset, at weight zero.

    Every score equals the score of the three priced rows. Mutation: a Tweedie
    likelihood refused the NaN mean ("mu must be finite"), a Bernoulli one summed
    ``0 * NaN``, and NB2's prior-weight quantile residuals drew from a NaN interval
    (``OverflowError``).
    """
    from superglm.distributions import NegativeBinomial, Tweedie
    from superglm.links import PowerLink
    from superglm.model_selection import _score_nll

    counts = np.array([0.0, 2.0, 2.0, 6.0, 2.0, 6.0] * 5)
    family, link, y, y_cells, options = {
        "tweedie-frequency": (
            Tweedie(p=1.5),
            PowerLink(-0.5),
            TRAIN_Y,
            CELLS_Y,
            {"weight_semantics": "frequency"},
        ),
        "binomial-frequency": (
            "binomial",
            "inverse",
            np.tile([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], 5),
            np.array([0.0, 1.0, 0.0, 1.0]),
            {"weight_semantics": "frequency"},
        ),
        "nb2-prior": (
            NegativeBinomial(2.0),
            "inverse",
            counts,
            CELLS_Y,
            {"weight_semantics": "prior"},
        ),
    }[case]
    model = _frequency_fit(family, link, y, **options)
    rival = _frequency_fit(family, "log", y, **options)
    offset = np.zeros(4)
    offset[3] = -model._predict_eta_raw_exact(CELLS)[3] - 0.5
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ignored = model.metrics(CELLS, y_cells, sample_weight=_IGNORE_LAST, offset=offset)
        seen = model.metrics(CELLS.iloc[:3], y_cells[:3], offset=offset[:3])
        for name in ("log_likelihood", "aic", "bic", "deviance", "pearson_chi2"):
            assert getattr(ignored, name) == pytest.approx(getattr(seen, name), rel=1e-12), name
        quantile = ignored.residuals("quantile")
        assert np.isnan(quantile[3])
        assert_allclose(quantile[:3], seen.residuals("quantile"), rtol=1e-12)
        assert _score_nll(
            model, CELLS, y_cells, sample_weight=_IGNORE_LAST, offset=offset
        ) == pytest.approx(_score_nll(model, CELLS.iloc[:3], y_cells[:3], offset=offset[:3]))
        ignored_test = vuong_test(model, rival, CELLS, y_cells, _IGNORE_LAST, offset)
        seen_test = vuong_test(model, rival, CELLS.iloc[:3], y_cells[:3], None, offset[:3])
    assert ignored_test.statistic == pytest.approx(seen_test.statistic, rel=1e-12)


class _RepeatedHoldout:
    """Holds the four cells out twice: first beside the training rows, then with more copies."""

    def split(self, X, y, groups=None):
        yield np.arange(30), np.arange(30, 34)
        yield np.r_[0:30, 34:38], np.arange(30, 34)


@pytest.mark.pins_behaviour
def test_a_repeated_split_does_not_call_a_finite_out_of_fold_prediction_nan():
    """Row 33 has no mean under the first fold's fit and a price under the second's, which stands.

    A control: master had no such warning and a finite prediction there; it pins
    both for the new warning about out-of-fold NaNs. Mutation: the mask of NaN
    predictions kept the first fold's reading, so the warning named row 33 as
    NaN beside its finite prediction.
    """
    X = pd.concat([TRAIN, CELLS, CELLS], ignore_index=True)
    y = np.r_[TRAIN_Y, CELLS_Y, CELLS_Y]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = cross_validate(editor_fit(), X, y, cv=_RepeatedHoldout(), return_oof=True)
    assert np.isfinite(result.oof_predictions[33])
    assert not [w for w in caught if "out-of-fold predictions are NaN" in str(w.message)]


def test_an_extreme_negative_power_warns_only_of_the_missing_mean():
    """``PowerLink(-0.025)``: the stand-in for a row with no mean is not inverted at the clip.

    Mutation: ``1e-12 ** -40`` overflowed, NumPy warned beside the named warning,
    and under ``-W error`` ``metrics`` raised that ``RuntimeWarning`` first.
    """
    from superglm.links import PowerLink

    X = pd.DataFrame({"x": np.tile([-1.0, 0.0, 1.0], 12)})
    y = np.tile([0.8, 1.0, 1.2, 1.0, 1.2, 1.4], 6)
    model = SuperGLM(
        family="gamma", link=PowerLink(-0.025), selection_penalty=0.0, features={"x": Numeric()}
    ).fit(X, y)
    offset = np.array([-2.0, 0.0])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        predicted = model.predict(X.iloc[:2], offset=offset)
    assert [type(w.message) for w in caught] == [UserWarning]
    assert np.isnan(predicted[0]) and np.isfinite(predicted[1])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(NoPredictedMeanError, match=r"positions 0\)"):
            model.metrics(X.iloc[:2], y[:2], offset=offset)


def test_a_zero_weight_rows_working_residual_reads_nan_without_a_warning():
    """The inverse and its derivative are read at a predictor inside the domain; the residual is NaN.

    Under ``PowerLink(-0.025)`` the derivative at the clip, ``1e-12 ** -41``,
    overflows. Mutation: the inverse-squared derivative ``eta ** -1.5`` was
    evaluated at the row's raw ``eta = -0.5`` and warned "invalid value"; the
    power link's at the clip warned "overflow", raised under ``-W error``.
    """
    from superglm.links import PowerLink

    model, *_ = three_cell_fit("inverse_squared", [1.0, 3.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        metrics = model.metrics(CELLS, CELLS_Y, sample_weight=_IGNORE_LAST)
        working = metrics.residuals("working")
    assert np.isnan(working[3]) and np.all(np.isfinite(working[:3]))
    assert metrics.eta[3] == pytest.approx(-0.5, rel=1e-6)

    X = pd.DataFrame({"x": np.tile([-1.0, 0.0, 1.0], 12)})
    y = np.tile([0.8, 1.0, 1.2, 1.0, 1.2, 1.4], 6)
    power = SuperGLM(
        family="gamma", link=PowerLink(-0.025), selection_penalty=0.0, features={"x": Numeric()}
    ).fit(X, y)
    offset = np.array([-2.0, 0.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        metrics = power.metrics(X.iloc[:2], y[:2], sample_weight=[0.0, 1.0], offset=offset)
        working = metrics.residuals("working")
        eta = metrics.eta
    seen = power.metrics(X.iloc[1:2], y[1:2], offset=offset[1:])
    assert np.isnan(working[0]) and working[1] == pytest.approx(seen.residuals("working")[0])
    assert eta[0] == pytest.approx(power._predict_eta_raw_exact(X.iloc[:1], offset[:1])[0])


def test_the_check_names_ten_rows_and_leaves_the_callers_mask_alone():
    """Mutation: the message named ten rows with no pointer to the rest, the error was not
    exported from ``superglm``, and the check wrote the weight mask into the caller's array."""
    import superglm
    from superglm.model.base import require_priced_rows

    assert superglm.NoPredictedMeanError is NoPredictedMeanError
    assert "NoPredictedMeanError" in superglm.__all__
    model = editor_fit()
    rows = CELLS.iloc[[0, *[3] * 12]]
    with pytest.raises(NoPredictedMeanError, match=r"the error's \.positions holds all 12") as e:
        model.metrics(rows, np.ones(13))
    assert e.value.positions.tolist() == list(range(1, 13))

    missing = np.array([True, True])
    with pytest.raises(NoPredictedMeanError, match=r"1 of 2 rows"):
        require_priced_rows(model, np.full(2, np.nan), [1.0, 0.0], "the score", missing=missing)
    assert missing.tolist() == [True, True]


@pytest.mark.pins_behaviour
def test_the_sklearn_score_is_the_r2_of_the_estimators_own_predict():
    """A control: scikit-learn's own ``score`` already read ``self.predict`` and checked the fit.

    It pins both for the override that refuses rows with no mean. Mutation: the
    override read the wrapped model's predictions, not a subclass's, and read
    the wrapped model before the fit was checked (``AttributeError`` for
    ``NotFittedError``).
    """
    from sklearn.exceptions import NotFittedError
    from sklearn.metrics import r2_score

    with pytest.raises(NotFittedError):
        SuperGLMRegressor().score(TRAIN, TRAIN_Y)

    class Doubled(SuperGLMRegressor):
        def predict(self, X):
            return 2.0 * super().predict(X)

    est = Doubled(
        family="gamma", selection_penalty=0.0, features={"a": Categorical(), "b": Categorical()}
    ).fit(TRAIN, TRAIN_Y)
    assert est.score(TRAIN, TRAIN_Y) == pytest.approx(r2_score(TRAIN_Y, est.predict(TRAIN)))


def test_a_custom_scorer_that_reorders_or_reprices_its_rows_has_them_named_among_its_own():
    """A refusal names frame rows only where the scorer read the held-out rows as given.

    One scorer reverses its held-out rows, another repeats two before the unseen
    cell, and a third reads them in order with offsets ``[-2, 0, 0, 2]``, which
    takes away cell 0's mean and gives cell 3 one. Each refusal names its rows
    among those its scorer read (positions 0, 4 and 0), and the result says it
    scored 1 of 2 folds, with the mean and pooled scores covering only that fold.
    Mutation: a refusal of as many rows as the frame has without a mean was named
    as those frame rows, so the repricing scorer named row 33, which it priced.
    """
    X, y, splitter = cv_frame()

    def reversed_rows(model, X, y, *, sample_weight=None, offset=None):
        return model.metrics(X.iloc[::-1], y[::-1]).deviance

    def repeated_rows(model, X, y, *, sample_weight=None, offset=None):
        rows = [0, 1, 2, 0, len(y) - 1]
        return model.metrics(X.iloc[rows], y[rows]).deviance

    def repriced_rows(model, X, y, *, sample_weight=None, offset=None):
        # Fold 0 holds out the four cells; fold 1's three rows are read unshifted.
        shift = np.array([-2.0, 0.0, 0.0, 2.0]) if len(y) == 4 else np.zeros(len(y))
        return model.metrics(X, y, offset=shift).deviance

    for scorer, position, n_read in [
        (reversed_rows, 0, 4),
        (repeated_rows, 4, 5),
        (repriced_rows, 0, 4),
    ]:
        with pytest.warns(UserWarning) as caught:
            result = cross_validate(editor_fit(), X, y, cv=splitter, scoring=scorer)
        refused, scored = result.fold_scores.to_dict("records")
        assert f"of the {n_read} rows fold 0's scorer read" in refused["score refused"]
        assert re.search(rf"positions {position} among them\)", refused["score refused"])
        assert "in the frame" not in refused["score refused"]
        name = scorer.__name__
        assert refused["converged"] and np.isfinite(scored[name])
        [summary] = [str(w.message) for w in caught if "folds" in str(w.message)]
        assert "scored 1 of 2 folds" in summary
        assert "left out of mean_scores, std_scores and pooled_scores" in summary
        assert result.mean_scores[name] == scored[name]


def test_a_zero_weight_row_never_enters_the_vuong_moments():
    """A zero-weight row with no mean and a response of ``1e80`` adds nothing.

    Mutation: its density, read at a stand-in mean, entered the variance as
    ``0 * inf``: the statistic and p-value were NaN, with NumPy's overflow and
    invalid-value warnings.
    """
    model = SuperGLM(
        family="gaussian",
        link="inverse",
        selection_penalty=0.0,
        weight_semantics="frequency",
        features={"a": Categorical(), "b": Categorical()},
    ).fit(TRAIN, TRAIN_Y)
    rival = SuperGLM(
        family="gaussian",
        link="identity",
        selection_penalty=0.0,
        weight_semantics="frequency",
        features={"a": Categorical()},
    ).fit(TRAIN, TRAIN_Y)
    y = np.r_[CELLS_Y[:3], 1e80]
    seen = vuong_test(model, rival, CELLS.iloc[:3], y[:3])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ignored = vuong_test(model, rival, CELLS, y, _IGNORE_LAST)
    assert ignored.statistic == pytest.approx(seen.statistic, rel=1e-12)
    assert ignored.p_value == pytest.approx(seen.p_value, rel=1e-12)


_NO_VALUE = (
    r"1 of {} rows have no predicted value of 'mean' under the PowerLink.*, so their {} are NaN "
    r"\(row positions {}\)"
)


def _warned_once(call, pattern):
    """``call()``'s result; it gives one warning, matching ``pattern``, and no other."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = call()
    assert [re.match(pattern, str(w.message)) is not None for w in caught] == [True]
    return result


def test_superlss_posterior_readers_give_a_row_with_no_value_nan_and_refuse_its_total():
    """The unseen cell, at ``eta = -0.5`` under ``PowerLink(-1)``, has no posterior value.

    Its predictive draws, with or without parameter uncertainty, and its posterior
    summaries read NaN, with one warning, beside the other cells' finite ones; a
    portfolio total, which cannot leave it out, is refused by name. Mutation: the
    draws read the inverse's floor, 1e15, with no warning, and the portfolio's total
    mean was about 9e14; the plug-in path warned through ``predict_parameters`` and
    then refused the NaN with the family's support error.
    """
    model = negative_power_lss()
    for uncertainty in (True, False):
        draws = _warned_once(
            lambda: model.posterior_predictive(
                CELLS, n_draws=20, parameter_uncertainty=uncertainty, seed=0
            ),
            _NO_VALUE.format(4, "predictive draws", 3),
        )
        assert np.all(np.isnan(draws[:, 3])) and np.all(np.isfinite(draws[:, :3]))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(NoPredictedMeanError, match=r"positions 3\), so the portfolio") as e:
                model.portfolio(CELLS, n_draws=20, parameter_uncertainty=uncertainty, seed=0)
        assert e.value.positions.tolist() == [3]
    bounds = _warned_once(
        lambda: model.posterior_bounds(CELLS, quantity="mean", n_draws=20, seed=0),
        _NO_VALUE.format(4, "posterior summaries", 3),
    )
    assert bounds.iloc[3].isna().all() and np.all(np.isfinite(bounds.iloc[:3].to_numpy()))


def test_a_row_whose_posterior_draws_cross_the_domain_has_no_posterior_summary(monkeypatch):
    """At ``x = 1`` the fit's mean predictor is 0.135; four posterior draws are put at -0.01.

    Every draw is the fit's coefficients except those four, whose slope moves the mean
    predictor at ``x = 1`` alone across zero (it is 0.02 at the next training row, 28/29).
    There the mean is infinite on part of the posterior, so the row's posterior mean,
    sd and band read NaN, with one warning, and its plug-in estimate stands; the risk
    curves read NaN at that point alone, with one warning for the three quantiles; the
    Q-Q envelope of the fitted rows, which cannot leave one out, refuses it by name.
    Mutation: the crossing draws read 1e15, the posterior mean was 2e13 and the 99%
    curve's band reached 3.7e15, and the envelope was drawn from them, with no warning.
    """
    from superglm import GammaLS, SuperLSS, term
    from superglm.distributional import posterior
    from superglm.links import PowerLink

    x = np.linspace(0.0, 1.0, 30)
    family = GammaLS()
    model = SuperLSS(
        family, family.mean(term("x", Numeric()), link=PowerLink(-1.0)), family.scale()
    )
    frame, y = pd.DataFrame({"x": x}), np.where(x < 0.5, 1.0, 4.0) * np.tile([0.4, 1.6], 15)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(frame, y)
    fitted = model._require_fitted()
    names = fitted.layout.coefficient_names
    intercept, slope = names.index("mean:(intercept)"), names.index("mean:x")
    beta = np.asarray(fitted.prediction_coefficients, dtype=np.float64)
    crossing = -0.01 - beta[intercept]  # the slope that puts the predictor at x = 1 on -0.01
    assert beta[intercept] + beta[slope] > 0.1 and beta[intercept] + crossing * 28 / 29 > 0.0
    real_draws = posterior._posterior_draws_from_covariance

    def injected(fitted, matrix, count, **options):
        draws = real_draws(fitted, matrix, count, **options)
        coefficients = np.tile(beta, (count, 1))
        coefficients[:4, slope] = crossing
        return replace(draws, coefficients=coefficients)

    monkeypatch.setattr(posterior, "_posterior_draws_from_covariance", injected)
    bounds = _warned_once(
        lambda: model.posterior_bounds(
            pd.DataFrame({"x": [0.0, 1.0]}), quantity="mean", n_draws=200, seed=0
        ),
        _NO_VALUE.format(2, "posterior summaries", 1),
    )
    assert np.all(np.isfinite(bounds.iloc[0])) and np.isfinite(bounds["estimate"].iloc[1])
    assert bounds.iloc[1].drop("estimate").isna().all()
    curves = _warned_once(
        lambda: model.risk_curves({}, "x", n_points=5, n_draws=200, level=0.98, seed=0),
        _NO_VALUE.format(5, "posterior summaries", 4),
    )
    assert np.flatnonzero(np.isnan(curves.upper).any(axis=0)).tolist() == [4]
    assert np.all(np.isnan(curves.upper[:, 4]))
    with pytest.raises(NoPredictedMeanError, match="so the Q-Q envelope cannot be computed") as e:
        model.plot_data("qq", X=frame, y=y, parameter_uncertainty=True, n_sim=200)
    assert e.value.positions.tolist() == [29]


def test_a_row_with_no_value_at_the_fit_has_no_predictive_draws_whatever_its_draws_give():
    """An offset puts the row's fitted predictor at ``-1e-8`` under ``PowerLink(-1)``.

    The seed is chosen so both posterior draws of that predictor land inside the
    domain, so only the fit lacks the value: the row's predictive draws read NaN with
    one warning, and ``reduce="sum"`` refuses it by name. Mutation: the draws alone
    decided, so the row was drawn and summed with no warning.
    """
    from superglm import GammaLS, SuperLSS, term
    from superglm.distributional.posterior import posterior_draws, posterior_parameters
    from superglm.links import PowerLink

    x = np.linspace(0.0, 1.0, 30)
    family = GammaLS()
    model = SuperLSS(
        family, family.mean(term("x", Numeric()), link=PowerLink(-1.0)), family.scale()
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(pd.DataFrame({"x": x}), np.tile([0.9, 1.1], 15) / (1.0 + x))
    fitted = model._require_fitted()
    row = pd.DataFrame({"x": [0.5]})
    eta_hat = float(fitted.predict_eta(row)[0, 0])
    offsets = {"mean": np.array([-eta_hat - 1e-8]), "scale": np.zeros(1)}
    seed = next(
        seed
        for seed in range(200)
        for [(_, block)] in [
            list(
                posterior_parameters(
                    fitted, row, posterior_draws(fitted, 2, seed=seed), offsets=offsets
                )
            )
        ]
        if np.all(np.isfinite(block))
    )
    drawn = _warned_once(
        lambda: model.posterior_predictive(row, n_draws=2, offsets=offsets, seed=seed),
        r"1 of 1 rows have no predicted value of 'mean'",
    )
    assert drawn.shape == (2, 1) and np.all(np.isnan(drawn))
    with pytest.raises(NoPredictedMeanError, match="so the reduced draws cannot be computed"):
        model.posterior_predictive(row, n_draws=2, offsets=offsets, seed=seed, reduce="sum")


def test_superlss_readers_of_a_row_with_no_value_refuse_it_by_name():
    """``actual_expected``, ``parameter_spread`` and ``density_fan`` read every row they are given.

    Each refuses the unseen cell with the named ``NoPredictedMeanError``, and does
    not warn. Mutation: each warned through ``predict_parameters`` and then refused
    its NaN with the family's support error ("outside its finite support"), which
    names no row; under warnings as errors the warning was raised instead.
    """
    model = negative_power_lss()
    readers = (
        lambda: model.actual_expected(CELLS, CELLS_Y, covariate="a"),
        lambda: model.parameter_spread(CELLS, threshold=2.0, n_bins=2),
        lambda: model.density_fan({"a": "1", "b": "1"}, "a"),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for reader in readers:
            with pytest.raises(NoPredictedMeanError, match="'mean' has no value on 1 of the"):
                reader()


@pytest.mark.parametrize("pooled", [False, True], ids=["inline", "pooled"])
def test_one_rows_overflowing_superlss_predictor_leaves_the_other_rows_their_quotes(pooled):
    """A covariate of 1e308 overflows the mean's predictor, and so does a finite offset of 1e308.

    That row has no value, with the one named warning, and every other row keeps its
    quote, on the inline and the pooled assembly; a covariate that is not finite is
    still refused. Mutation: the overflow raised ("prediction contribution for 'x' is
    invalid", or "non-finite values") with NumPy's overflow warning, and no row was
    quoted.
    """
    from superglm import GammaLS, SuperLSS
    from superglm.distributional.model import POOLED_PREDICTION_ROWS
    from superglm.links import PowerLink

    x = np.linspace(0.0, 1.0, 40)
    family = GammaLS()
    model = SuperLSS(family, family.mean("x", link=PowerLink(-1.0)), family.scale())
    model.fit(pd.DataFrame({"x": x}), np.tile([0.9, 1.1], 20) / (1.0 + 5.0 * x))
    n = POOLED_PREDICTION_ROWS if pooled else 2
    [quote] = model.predict(pd.DataFrame({"x": [0.5]}))
    for large, offset in [(1e308, 0.0), (2e307, 1e308)]:
        rows = pd.DataFrame({"x": np.r_[np.full(n - 1, 0.5), large]})
        predicted = _warned_once(
            lambda: model.predict(rows, offsets={"mean": np.r_[np.zeros(n - 1), offset]}),
            rf"1 of {n} rows have no predicted value of 'mean'",
        )
        assert np.isnan(predicted[-1])
        assert_allclose(predicted[:-1], quote, rtol=1e-15)
    with pytest.raises(ValueError, match="prediction contribution for 'x' is invalid"):
        model.predict(pd.DataFrame({"x": [0.5, np.nan]}))


def test_interaction_screening_reads_a_zero_weight_row_with_no_mean_at_a_stand_in():
    """Holdout row 0 has offset -2, so no mean under ``PowerLink(-0.025)``, and weight zero.

    It adds nothing: the screen equals the screen without it, with no NumPy warning.
    Mutation: its score and Fisher weights were read at the clipped predictor, where
    the inverse overflows: under warnings as errors the inverse raised, otherwise the
    QR refused the infinities.
    """
    from superglm.links import PowerLink

    X = pd.DataFrame({"x": np.tile([-1.0, 0.0, 1.0], 12), "a": np.tile(["0", "1", "0", "1"], 9)})
    y = np.tile([0.8, 1.0, 1.2, 1.0, 1.2, 1.4], 6)
    model = SuperGLM(
        family="gamma",
        link=PowerLink(-0.025),
        selection_penalty=0.0,
        features={"x": Numeric(), "a": Categorical()},
    ).fit(X, y)
    weights, offset = np.ones(len(y)), np.zeros(len(y))
    weights[0], offset[0] = 0.0, -2.0
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        kept = model.screen_interactions(
            X, y, sample_weight=weights, offset=offset, candidates=[("x", "a")]
        )
        without = model.screen_interactions(
            X.iloc[1:], y[1:], sample_weight=weights[1:], offset=offset[1:], candidates=[("x", "a")]
        )
    assert kept["statistic"].iloc[0] == pytest.approx(without["statistic"].iloc[0], rel=1e-10)


def test_a_zero_weight_tweedie_row_is_not_evaluated():
    """A zero-weight row with no mean and a response of 1e308 adds an exact 0 to the likelihood.

    Mutation: its density was read at a stand-in mean, where 1e308 gives ``-inf``,
    and ``0 * -inf`` made the log-likelihood NaN with NumPy's invalid-value warning.
    """
    from superglm.distributions import Tweedie
    from superglm.links import PowerLink

    model = _frequency_fit(Tweedie(p=1.5), PowerLink(-0.5), TRAIN_Y, weight_semantics="frequency")
    offset = np.zeros(4)
    offset[3] = -model._predict_eta_raw_exact(CELLS)[3] - 0.5
    y = np.array([1.0, 4.0, 4.0, 1e308])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ignored = model.metrics(CELLS, y, sample_weight=_IGNORE_LAST, offset=offset)
        seen = model.metrics(CELLS.iloc[:3], y[:3], offset=offset[:3])
        assert ignored.log_likelihood == pytest.approx(seen.log_likelihood, rel=1e-12)


def test_a_numeric_error_score_stays_out_of_the_mean_and_spread():
    """A refused fold's score reads ``error_score``, and leaves the mean, spread and pooled score.

    Mutation: ``error_score=-999`` entered the mean (-499.43) and the standard
    deviation (499.57), while the warning said the means covered the scored folds.
    """
    X, y, splitter = cv_frame()
    with pytest.warns(UserWarning, match="left out of mean_scores, std_scores and pooled_scores"):
        result = cross_validate(editor_fit(), X, y, cv=splitter, error_score=-999.0)
    refused, scored = result.fold_scores["deviance"]
    assert refused == -999.0
    assert result.mean_scores["deviance"] == scored and result.std_scores["deviance"] == 0.0
    assert result.pooled_scores["deviance"] == pytest.approx(scored, rel=1e-12)


def test_a_custom_scorer_that_reads_fewer_rows_has_them_named_among_its_own():
    """Held-out rows 32 and 33 have no mean (32 by an offset); the scorer reads row 33 alone.

    The refusal names the one row the scorer read, among its rows and with its fold,
    not both held-out rows. Mutation: CV named rows 32 and 33 and told the caller to
    drop or zero-weight both.
    """
    from sklearn.model_selection import PredefinedSplit

    X, y, _ = cv_frame()
    X, y = X.iloc[:34], y[:34]
    splitter = PredefinedSplit(np.r_[np.full(len(TRAIN_Y), -1), np.zeros(4)])
    offset = np.zeros(len(y))
    offset[32] = -1.0

    def last_row(model, rows, target, *, sample_weight=None, offset=None):
        return model.metrics(rows.iloc[[-1]], target[-1:], offset=offset[-1:]).deviance

    with pytest.raises(NoPredictedMeanError, match=r"of the 1 rows fold 0's scorer read") as e:
        cross_validate(
            editor_fit(), X, y, cv=splitter, offset=offset, scoring=last_row, error_score="raise"
        )
    assert e.value.positions.tolist() == [0]


def test_a_score_may_be_named_as_the_old_refusal_column():
    """A score's name, a function's or a returned key, is the caller's: the refusal column is not one.

    Mutation: a scorer named ``score_refused`` had its folds counted as refused ("scored
    0 of 2 folds"), and a returned ``score_refused`` key was refused as reserved.
    """

    def score_refused(model, rows, target, *, sample_weight=None, offset=None):
        return 0.25

    def returned(model, rows, target, *, sample_weight=None, offset=None):
        return {"score_refused": 0.25}

    X, y, splitter = cv_frame()
    model = SuperGLM(
        family="gamma", selection_penalty=0.0, features={"a": Categorical(), "b": Categorical()}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for scorer in (score_refused, returned):
            result = cross_validate(model, X, y, cv=splitter, scoring=scorer, error_score="raise")
            assert result.fold_scores["score_refused"].tolist() == [0.25, 0.25]
            assert result.mean_scores["score_refused"] == 0.25


def test_a_superlss_fold_with_a_row_with_no_value_keeps_its_fit_and_names_it_in_the_frame():
    """Fold 0 holds out the unseen cell, row 33 of the frame, which has no value of the mean.

    Its scores are refused, naming row 33 in the frame; the fold keeps its fit and the
    other rows' out-of-fold predictions. At weight zero the row is scored by no fold, and
    the one out-of-fold warning names it as row 33. Mutation: the fold failed whole
    (``converged`` False, every out-of-fold prediction NaN), and at weight zero
    ``predict``'s own warning named the fold-local position 3.
    """
    X, y, splitter = cv_frame()
    with pytest.warns(UserWarning) as caught:
        result = cross_validate(
            negative_power_lss(fitted=False), X, y, cv=splitter, return_oof=True
        )
    assert any("scored 1 of 2 folds" in str(w.message) for w in caught)
    refused = result.fold_scores.iloc[0]
    assert refused["converged"] and np.isnan(refused["nll"])
    assert re.search(r"positions 33 in the frame\)", refused["score refused"])
    oof = result.oof_predictions[len(TRAIN_Y) : len(TRAIN_Y) + 4]
    assert np.isnan(oof[3]) and np.all(np.isfinite(oof[:3]))

    weights = np.ones_like(y)
    weights[len(TRAIN_Y) + 3] = 0.0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cross_validate(
            negative_power_lss(fitted=False),
            X,
            y,
            cv=splitter,
            sample_weight=weights,
            return_oof=True,
        )
    [message] = [str(w.message) for w in caught if "no predicted" in str(w.message)]
    assert "out-of-fold predictions are NaN (row positions 33)" in message
