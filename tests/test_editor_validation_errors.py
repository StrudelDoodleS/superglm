"""Intentional editor validation is distinguishable from backend failures."""

import re
import warnings

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, Numeric, SuperGLM
from superglm.editor import EditableTerm, EditorSession, edit
from superglm.editor.collapse import collapsed_feature_spec
from superglm.editor.controls import control_curve_after_move
from superglm.editor.diagnose import worst_rows
from superglm.editor.evaluation import EvaluationDataset
from superglm.editor.level_order import level_order_for_direction
from superglm.editor.persistence import serialize_validated_model
from superglm.editor.refit import fit_refit_model
from superglm.editor.summaries import summary_payload
from superglm.editor.terms import term_offset_values
from superglm.links import InverseLink
from superglm.model_selection import fold_adapter
from tests._no_mean_fixture import (
    CELLS,
    CELLS_Y,
    TRAIN,
    TRAIN_Y,
    cv_frame,
    editor_fit,
    negative_power_lss,
)


@pytest.fixture
def session():
    curve = EditableTerm(
        name="curve",
        kind="spline",
        original_log_effect=np.zeros(3),
        edited_log_effect=np.zeros(3),
        x=np.arange(3, dtype=float),
    )
    category = EditableTerm(
        name="category",
        kind="categorical",
        original_log_effect=np.zeros(2),
        edited_log_effect=np.zeros(2),
        levels=["A", "B"],
    )
    return EditorSession(None, {"curve": curve, "category": category})


@pytest.mark.parametrize(
    ("operation", "error_type", "message"),
    [
        (
            lambda session: session.select_indices("curve", [3]),
            IndexError,
            "Selection indices out of range for term 'curve'.",
        ),
        (
            lambda session: session.selection("missing"),
            KeyError,
            "Unknown editable term: 'missing'",
        ),
        (
            lambda session: session.select_levels("curve", ["A"]),
            TypeError,
            "Term 'curve' does not have levels.",
        ),
        (
            lambda session: session.select_levels("category", ["C"]),
            KeyError,
            "Unknown level(s) for term 'category': ['C']",
        ),
        (
            lambda session: session.smooth("curve", strength=2),
            ValueError,
            "strength must be between 0 and 1, got 2",
        ),
        (
            lambda session: control_curve_after_move(None, session.terms["curve"], 99, 0.0),
            IndexError,
            "Control handle index out of range for term 'curve'.",
        ),
        (
            lambda session: collapsed_feature_spec(
                None, session.terms["category"], np.array([0]), X=None
            ),
            ValueError,
            "Select at least two levels to collapse term 'category'.",
        ),
        (
            lambda session: level_order_for_direction(2, np.array([0]), "up"),
            ValueError,
            "direction must be 'left' or 'right', got 'up'",
        ),
        (
            lambda session: fit_refit_model(None, None, method="invalid", X=None, y=None),
            ValueError,
            "method must be 'auto', 'fit', or 'fit_reml'.",
        ),
        (
            lambda session: term_offset_values(session.terms["category"], ["C"]),
            KeyError,
            "Offset data contains unseen level(s) for 'category': ['C']",
        ),
    ],
)
def test_intentional_validation_has_public_message(session, operation, error_type, message):
    with pytest.raises(error_type) as caught:
        operation(session)
    assert getattr(caught.value, "public_message", None) == message


@pytest.mark.parametrize("error_type", [KeyError, ValueError, TypeError, IndexError])
def test_arbitrary_builtin_errors_are_not_public(error_type):
    assert getattr(error_type("backend failure"), "public_message", None) is None


def test_array_conversion_failure_is_not_promoted_to_public_validation(session):
    with pytest.raises(ValueError) as caught:
        session.select_indices("curve", ["invalid index"])
    assert getattr(caught.value, "public_message", None) is None


_NO_QUOTE = r"{} of {} rows have no quote under this model \(row positions {}\)"


def test_rows_with_no_quote_are_shown_as_such_and_an_edit_past_zero_stands():
    """A Gamma/inverse model has no mean where its linear predictor is at or below zero.

    Three of four cells of two two-level factors make the fit saturated, so the
    unseen fourth cell (validation rows 3 and 4) has ``eta = -0.5``. The editor
    shows that split as having no quote and keeps scoring the training rows;
    the rows lead the worst-rows list with no prediction. An edit that takes
    training rows past zero is not refused: it stands without fit statistics,
    and the summary says why. Mutation: the rows read 1e12 and the validation
    deviance was a finite number dominated by them; with NaN predictions and no
    handling, the edit's refreshed deviance was NaN and its publication raised.
    """
    model = editor_fit()
    X_val = pd.DataFrame({"a": ["0", "1", "0", "1", "1"], "b": ["0", "0", "1", "1", "1"]})
    y_val = np.array([1.0, 4.0, 4.0, 3.0, 5.0])
    session = edit(model, train_data=(TRAIN, TRAIN_Y), validation_data=(X_val, y_val))
    no_quote = _NO_QUOTE.format(2, 5, "3, 4")
    widget = session.widget(mode="server")
    try:
        train, validation = widget._report("validation")["splits"]
        panel = widget._metrics("deviance", dataset="validation")
        assert np.isfinite(train["metrics"]["original"]["deviance"])
        assert validation["available"] is False and re.match(no_quote, validation["error"])
        assert panel["available"] is False and re.match(no_quote, panel["error"])

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            listed = worst_rows(
                fold_adapter(model), model, X_val, y_val, None, None, np.arange(5), ["a", "b"]
            )
        # The two rows tie (no cost), so their order is not part of the contract.
        assert {row["row"] for row in listed["rows"][:2]} == {3, 4}
        assert all(row["predicted"] is None and row["cost"] is None for row in listed["rows"][:2])

        # Level 1 of `a` carries the cell (1, 0), whose eta is 1/4: -0.3 takes it past zero.
        session.select_indices("a", [1])
        session.shift("a", -0.3)
        edited = session.to_model()
        summary = summary_payload(widget, "in_force", model_override=edited)
    finally:
        widget.close()
    assert edited._fit_stats is None
    with pytest.warns(UserWarning, match=r"10 of 30 rows have no predicted mean"):
        assert np.count_nonzero(np.isnan(edited.predict(TRAIN))) == 10
    assert summary["available"] is False and "no fit statistics" in summary["error"]


@pytest.mark.pins_behaviour
def test_a_zero_weight_row_with_no_quote_adds_nothing_to_an_edits_statistics():
    """A control: a zero-weight row added nothing to the statistics before, priced at 1e12.

    It pins that for a row with no quote, under Tweedie frequency weights. The
    edit is materialized on the validation fallback, whose unseen cell has weight
    zero. Mutation: the NaN reached the Tweedie density, which refused it ("mu
    must be finite and strictly positive"), and the edit failed.
    """
    from superglm.distributions import Tweedie
    from superglm.editor.apply import materialize_edit_request

    model = SuperGLM(
        family=Tweedie(p=1.5),
        link="inverse",
        selection_penalty=0.0,
        weight_semantics="frequency",
        retain_fit_state=False,
        features={"a": Categorical(), "b": Categorical()},
    ).fit(TRAIN, TRAIN_Y)
    weights = np.array([1.0, 1.0, 1.0, 0.0])
    session = edit(model, validation_data=(CELLS, CELLS_Y, weights))
    session.select_indices("a", [1])
    session.shift("a", 0.01)
    edited = materialize_edit_request(session.capture_materialization_request())
    seen = edited.metrics(CELLS.iloc[:3], CELLS_Y[:3])
    assert edited.result.deviance == pytest.approx(seen.deviance, rel=1e-12)
    assert edited._fit_stats.log_likelihood == pytest.approx(seen.log_likelihood, rel=1e-12)


def test_a_validation_split_with_no_quote_anywhere_leaves_an_edit_standing(tmp_path):
    """``retain_fit_state=False`` and no ``train_data``: the validation split scores the edit.

    Every row of it is the unseen cell. The edit stands without statistics; the
    split, the metric panel and the summary say why, and the model saves.
    Mutation: the rows read 1e12 and the split's scores were finite; with NaN
    predictions and no handling, the edit's deviance was NaN and saving raised.
    """
    model = SuperGLM(
        family="gamma",
        link="inverse",
        selection_penalty=0.0,
        retain_fit_state=False,
        features={"a": Categorical(), "b": Categorical()},
    ).fit(TRAIN, TRAIN_Y)
    session = edit(model, validation_data=(CELLS.iloc[[3, 3]], CELLS_Y[[3, 3]]))
    session.select_indices("a", [1])
    session.shift("a", 0.01)
    widget = session.widget(mode="server")
    try:
        [validation] = widget._report("validation")["splits"]
        panel = widget._metrics("deviance", dataset="validation")
        summary = summary_payload(widget, "in_force")
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            exported = widget._export_bytes("joblib")
    finally:
        widget.close()
    assert re.match(_NO_QUOTE.format(2, 2, "0, 1"), validation["error"])
    assert panel["available"] is False and panel["error"] == validation["error"]
    assert summary["available"] is False and "no fit statistics" in summary["error"]
    assert exported.validation_scope == "artifact+predictions"
    assert session.save_model(tmp_path / "edited.joblib").exists()


class _DeclaredTooShortInverse(InverseLink):
    """Declares the inverse link's domain, but its inverse has no value at ``eta >= 4`` inside it."""

    eta_domain = (0.0, np.inf)

    def inverse(self, eta):
        return np.where(np.asarray(eta) < 4.0, 1.0 / np.asarray(eta), np.nan)


def test_an_export_accepts_a_missing_quote_only_outside_the_links_domain():
    """The joblib check reads the rows with no quote from the raw predictor, not the NaNs.

    The unseen cell, outside the domain, exports with no quote and loads with
    none. A NaN at ``eta = 6``, inside the domain the link declares, is refused.
    Mutation: before the fix the unseen cell exported at 1e12; then any NaN under
    a link with a bounded domain validated, including the one at ``eta = 6``.
    """
    from superglm.editor.persistence import joblib_load_bytes

    unseen = EvaluationDataset("val", "Validation", CELLS, CELLS_Y)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        data, validation = serialize_validated_model(editor_fit(), dataset=unseen)
    with pytest.warns(UserWarning, match=r"1 of 4 rows have no predicted mean"):
        loaded = joblib_load_bytes(data).predict(CELLS)
    assert validation.scope == "artifact+predictions" and validation.prediction_rows == 4
    assert np.flatnonzero(np.isnan(loaded)).tolist() == [3]

    x = pd.DataFrame({"x": np.arange(8.0)})
    short = SuperGLM(
        family="gamma",
        link=_DeclaredTooShortInverse(),
        selection_penalty=0.0,
        features={"x": Numeric()},
    ).fit(x, np.tile([0.9, 1.1], 4))
    offset = np.array([0.0, 5.0])
    assert 4.0 < short._predict_eta_raw_exact(x.iloc[:2], offset)[1] < np.inf
    inside = EvaluationDataset("val", "Validation", x.iloc[:2], [1.0, 1.0], offset=offset)
    with pytest.raises(ValueError, match="prediction validation failed"):
        serialize_validated_model(short, dataset=inside)


class _Context:
    """A Run CV job context that never cancels."""

    def check(self):
        return None

    def progress(self, phase, **details):
        return None


def test_run_cv_keeps_every_usable_fold_and_says_why_one_has_no_score():
    """Fold 1 holds out the unseen cell, which its fit gives no quote; fold 2 scores.

    The replay keeps both folds, their fits and traces, and the reason; so does the
    supplied result's payload. Mutation: the rows read 1e12 and fold 1 a finite
    score dominated by them; with the named refusal and no handling, it stopped
    Run CV at fold 1 ("could not be fitted or scored"), and the tab dropped
    ``score_refused``.
    """
    from superglm.editor.cv import capture_cv_run, cv_report_payload, run_cv
    from superglm.model_selection import cross_validate

    X, y, splitter = cv_frame()
    model = editor_fit()
    with pytest.warns(UserWarning, match="scored 1 of 2 folds"):
        supplied = cross_validate(model, X, y, cv=splitter)
    session = edit(model, cv=supplied, cv_data=(X, y), train_data=(TRAIN, TRAIN_Y))
    run = run_cv(capture_cv_run(session), _Context())
    refused, scored = run.result.fold_scores.to_dict("records")
    assert np.isnan(refused["deviance"]) and refused["converged"]
    assert re.match(
        r"1 of fold 1's 4 held-out rows have no quote .*positions 33\)", refused["score refused"]
    )
    assert scored["deviance"] == pytest.approx(supplied.fold_scores["deviance"][1], rel=1e-12)
    assert run.result.pooled_scores["deviance"] == pytest.approx(scored["deviance"], rel=1e-12)
    assert [fit["id"] for fit in run.fold_fits] == ["fold-1", "fold-2"]
    widget = session.widget(mode="server")
    try:
        [result] = cv_report_payload(widget)["results"]
    finally:
        widget.close()
    assert result["folds"][0]["score_refused"] == supplied.fold_scores["score refused"][0]


@pytest.mark.pins_behaviour
def test_a_zero_weight_row_with_no_quote_adds_nothing_to_a_splits_scores():
    """A control: a zero-weight row added nothing to a split's deviance before, priced at 1e12.

    The validation split's unseen cell has weight zero, and the metric panel and the
    report score the split as its three priced rows. Mutation: the deviance summed
    ``0 * NaN`` for that row and both read NaN.
    """
    model = editor_fit()
    y_val = np.array([1.5, 3.0, 5.0, 3.0])
    session = edit(
        model,
        train_data=(TRAIN, TRAIN_Y),
        validation_data=(CELLS, y_val, np.array([1.0, 1.0, 1.0, 0.0])),
    )
    seen = model.metrics(CELLS.iloc[:3], y_val[:3]).deviance
    widget = session.widget(mode="server")
    try:
        panel = widget._metrics("deviance", dataset="validation")
        _, validation = widget._report("validation")["splits"]
    finally:
        widget.close()
    assert panel["original"] == pytest.approx(seen, rel=1e-12)
    assert validation["metrics"]["edited"]["deviance"] == pytest.approx(seen, rel=1e-12)


def _says_no_value(error, count, total, positions):
    """What the user acts on: how many rows, which, the parameter and the link lacking it."""
    assert re.search(rf"\b{count} of {total} rows\b", error), error
    assert re.search(rf"\(row positions {re.escape(positions)}\)", error), error
    assert "'mean'" in error and "PowerLink" in error, error


def test_a_superlss_split_or_edit_with_rows_with_no_value_is_shown_as_such():
    """A SuperLSS whose mean has ``PowerLink(-1)``: the unseen cell has no value of the mean.

    The validation split holding it is unavailable with the reason, and the training
    split is still scored. An edit that takes training rows past zero stands:
    ``to_model()`` returns it, and the report says which training rows have no quote.
    Mutation: the report raised the library's refusal ("'mean' has no value on 1 of the
    4 rows read"), so no split was shown, and ``to_model()`` raised it too.
    """
    model = negative_power_lss()
    session = edit(model, train_data=(TRAIN, TRAIN_Y), validation_data=(CELLS, CELLS_Y))
    widget = session.widget(mode="server")
    try:
        train, validation = widget._report("validation")["splits"]
        assert np.isfinite(train["metrics"]["original"]["log_score"])
        assert validation["available"] is False
        _says_no_value(validation["error"], 1, 4, "3")

        # Level 1 of `a` carries the cell (1, 0), whose predictor is 1/4: -0.3 takes it past zero.
        session.select_indices("mean:a", [1])
        session.shift("mean:a", -0.3)
        edited = session.to_model()
        train, _ = widget._report("validation")["splits"]
    finally:
        widget.close()
    assert edited.predict_link(CELLS).to_numpy()[1, 0] == pytest.approx(-0.05, rel=1e-6)
    assert train["available"] is False
    _says_no_value(train["error"], 10, 30, "2, 3, 8, 9, 14, 15, 20, 21, 26, 27")


def test_run_cv_keeps_a_superlss_fold_whose_held_out_rows_have_no_value():
    """Fold 1 holds out the SuperLSS's unseen cell, row 33; fold 2 scores.

    The replay keeps both folds and their fits, and names the row in the frame.
    Mutation: the fold scorer's refusal stopped Run CV at fold 1 with no result.
    """
    from superglm.editor.cv import capture_cv_run, run_cv
    from superglm.model_selection import cross_validate

    X, y, splitter = cv_frame()
    model = negative_power_lss()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        supplied = cross_validate(model, X, y, cv=splitter)
    session = edit(model, cv=supplied, cv_data=(X, y), train_data=(TRAIN, TRAIN_Y))
    run = run_cv(capture_cv_run(session), _Context())
    refused, scored = run.result.fold_scores.to_dict("records")
    assert refused["converged"] and np.isnan(refused["nll"]) and np.isfinite(scored["nll"])
    assert re.match(
        r"1 of fold 1's 4 held-out rows have no quote .*positions 33\)", refused["score refused"]
    )


def test_the_cv_tab_leaves_a_refused_folds_error_score_out():
    """A supplied result run with a numeric ``error_score``: the refused fold has no score.

    Its cell reads none, with the reason, and the mean is the scored fold's. Mutation:
    the tab showed fold 1's score as -999, counted it as scored, and its mean was -499.4.
    """
    from superglm.editor.cv import cv_report_payload
    from superglm.model_selection import cross_validate

    X, y, splitter = cv_frame()
    model = editor_fit()
    with pytest.warns(UserWarning, match="scored 1 of 2 folds"):
        supplied = cross_validate(model, X, y, cv=splitter, error_score=-999.0)
    session = edit(model, cv=supplied, cv_data=(X, y), train_data=(TRAIN, TRAIN_Y))
    widget = session.widget(mode="server")
    try:
        [result] = cv_report_payload(widget)["results"]
    finally:
        widget.close()
    refused, scored = result["folds"]
    assert refused["scores"]["deviance"] is None and refused["score_refused"]
    assert result["mean"]["deviance"] == scored["scores"]["deviance"]


def test_the_cv_tab_keeps_a_score_its_refused_fold_did_compute():
    """A supplied result's custom ``gini`` scores the priced rows; the deviance is refused.

    In the refused fold the deviance reads none, with the reason, and the Gini keeps its
    value, which its mean includes. Mutation: every score of a fold with a refusal was
    shown as none, so the tab hid a Gini that its mean counted.
    """
    from superglm.editor.cv import cv_report_payload
    from superglm.model_selection import cross_validate

    def gini(model, rows, target, *, sample_weight=None, offset=None):
        return 0.25 + 0.5 * len(target) / 10.0

    X, y, splitter = cv_frame()
    model = editor_fit()
    with pytest.warns(UserWarning, match="scored 1 of 2 folds"):
        supplied = cross_validate(model, X, y, cv=splitter, scoring=("deviance", gini))
    assert supplied.refused_scores == ((0, "deviance"),)
    session = edit(model, cv=supplied, cv_data=(X, y), train_data=(TRAIN, TRAIN_Y))
    widget = session.widget(mode="server")
    try:
        [result] = cv_report_payload(widget)["results"]
    finally:
        widget.close()
    refused, scored = result["folds"]
    assert refused["scores"]["deviance"] is None and refused["score_refused"]
    assert refused["scores"]["gini"] == pytest.approx(0.45)
    assert scored["scores"]["gini"] == pytest.approx(0.4)
    assert result["mean"]["gini"] == pytest.approx(0.425)
