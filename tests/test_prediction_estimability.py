"""Predictions that lean on what the training data do not determine are named once.

R's ``predict.lm(rankdeficient = "warnif")``: a row ``r`` is off the training
design's row space when ``||r N|| > 1e-6 ||r||`` (Lenth, R Journal 7(1),
2015), with ``N`` the null space the fit's data rank recorded, read in the
record's equilibrated scale (``model.estimability_ops``).  ``predict`` returns
the same price as before and gives one ``NonEstimablePredictionWarning`` whose
``rows`` names them; the rating-table export marks the entries that lean.

Each fixture forces its null direction by construction: a level seen at one
value of its by-level slope, or a column equal to another on every training
row.  Each test that checks a warning or a mark fails on master, which has
neither (the class is matched by name so that it fails on the assertion);
the tests that check silence are marked as controls.  Every fixture's shares
are asserted to sit at least three orders of magnitude on the right side of
the tolerance, so the tolerance does not pick the answer.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import superglm
from superglm import Categorical, Numeric, Spline, SuperGLM
from superglm._frame import as_eager_frame
from superglm.model import base

TOL = 1e-6  # R's predict.lm tol, the module's PREDICTION_TOLERANCE
MARGIN = 1e3  # each side of TOL by at least this factor

rng = np.random.default_rng(7)
N = 2000
X_SPREAD = rng.uniform(0, 10, N)
LEVEL = rng.choice(list("ABCDE"), N)
Y = rng.gamma(2.0, np.exp(0.1 * np.sin(X_SPREAD)) / 2.0)
# Row J: level E sits at x = 5 alone, so its by-level slope (or spline) is
# aliased with its level on every training row.
ROW_J = pd.DataFrame({"x": np.where(LEVEL == "E", 5.0, X_SPREAD), "f": LEVEL})
# E at x - 1 and x + 1 is off the span; E at 5 and the other levels are on it.
ROW_J_NEW = pd.DataFrame({"x": [5.0, 4.0, 6.0, 3.0, 5.0], "f": ["E", "E", "E", "A", "B"]})


def _fit(X, y, features, interactions=(), *, method="fit", family=None):
    model = SuperGLM(
        family=family or superglm.families.gamma(),
        link="log",
        features=features,
        interactions=list(interactions),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        getattr(model, method)(X, y)
    return model


def _row_j(x_spec, method):
    return _fit(ROW_J, Y, {"x": x_spec(), "f": Categorical()}, [("x", "f")], method=method)


def _disclosures(call):
    """``call``'s value and its off-span warnings, matched by class name."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = call()
    found = [
        c.message for c in caught if type(c.message).__name__ == "NonEstimablePredictionWarning"
    ]
    return value, found


def _shares(model, X):
    from superglm.model.estimability_ops import prediction_shares

    return prediction_shares(model, as_eager_frame(X), base._prediction_plan(model), "conditional")


def _assert_gap(model, on_span, off_span):
    """The tolerance sits at least ``MARGIN`` from every share on either side."""
    assert np.max(_shares(model, on_span)) < TOL / MARGIN
    assert np.min(_shares(model, off_span)) > TOL * MARGIN


X_SPECS = [Numeric, lambda: Spline(n_knots=6)]
X_IDS = ["numeric", "spline"]


@pytest.mark.parametrize("method", ["fit", "fit_reml"])
@pytest.mark.parametrize("x_spec", X_SPECS, ids=X_IDS)
def test_row_j_off_its_level_value_is_warned_at_the_unchanged_price(x_spec, method):
    """Level E quoted at x = 4 and 6 leans on the alias; the price is the unwarned path's.

    ``rows`` names 1 and 2 only.  The price is bit for bit what ``predict_exact``
    returns, the path master's ``predict`` ran, and ``b0 + x beta`` at the
    published coefficients to ``kappa u``.
    """
    model = _row_j(x_spec, method)
    mu, found = _disclosures(lambda: model.predict(ROW_J_NEW))
    assert len(found) == 1
    assert found[0].rows.tolist() == [1, 2]
    assert "rows 1, 2" in str(found[0])
    np.testing.assert_array_equal(mu, base.predict_exact(model, ROW_J_NEW, warn=False))
    frame = as_eager_frame(ROW_J_NEW)
    plan = base._prediction_plan(model)
    p = model.result.beta.size
    design = np.column_stack(
        [
            sum(
                base._score_prediction_term_exact(term, frame, unit)
                for term in (*plan["features"], *plan["interactions"])
            )
            for unit in np.eye(p)
        ]
    )
    eta = model.result.intercept + design @ model.result.beta
    np.testing.assert_allclose(np.log(mu), eta, rtol=0, atol=1e-10 * np.max(np.abs(eta)))
    _assert_gap(model, ROW_J_NEW.iloc[[0, 3, 4]], ROW_J_NEW.iloc[[1, 2]])


@pytest.mark.pins_behaviour
@pytest.mark.parametrize("x_spec", X_SPECS, ids=X_IDS)
def test_row_j_training_rows_and_their_copy_are_never_warned(x_spec):
    """Control (passes on master): the fit's own rows, and a copy, read only rounding."""
    model = _row_j(x_spec, "fit")
    for frame in (ROW_J, ROW_J.copy(), ROW_J.iloc[::-1].reset_index(drop=True)):
        _, found = _disclosures(lambda frame=frame: model.predict(frame))
        assert not found
    assert np.max(_shares(model, ROW_J)) < TOL / MARGIN


def _aliased():
    """``x2 = 2 x1 + 3`` on every training row: one null direction across two terms."""
    x1 = rng.uniform(0, 4, N)
    X = pd.DataFrame({"x1": x1, "x2": 2.0 * x1 + 3.0, "f": LEVEL})
    y = rng.poisson(np.exp(0.2 * x1 - 0.5)).astype(float)
    features = {"x1": Numeric(), "x2": Numeric(), "f": Categorical()}
    return X, y, features


@pytest.mark.parametrize("method", ["fit", "fit_reml"])
def test_an_aliased_column_quoted_off_its_span_is_warned(method):
    """Rows with ``x2 != 2 x1 + 3`` are named; rows on the relation are not."""
    X, y, features = _aliased()
    model = _fit(X, y, features, method=method, family=superglm.families.poisson())
    new = pd.DataFrame(
        {"x1": [1.0, 1.0, 2.0, 0.5], "x2": [5.0, 6.0, 7.0, 2.0], "f": ["A", "B", "C", "D"]}
    )
    mu, found = _disclosures(lambda: model.predict(new))
    assert [w.rows.tolist() for w in found] == [[1, 3]]
    np.testing.assert_array_equal(mu, base.predict_exact(model, new, warn=False))
    _assert_gap(model, new.iloc[[0, 2]], new.iloc[[1, 3]])
    _, found = _disclosures(lambda: model.predict(X.copy()))
    assert not found


@pytest.mark.pins_behaviour
def test_a_full_rank_fit_is_never_tested_or_warned():
    """Control (passes on master): no null direction, no pass, no warning, wherever it is quoted."""
    from superglm.model.estimability_ops import data_null_directions

    X = pd.DataFrame({"x": X_SPREAD, "f": LEVEL})
    model = _fit(X, Y, {"x": Spline(n_knots=6), "f": Categorical()}, [("x", "f")])
    assert data_null_directions(model) is None
    far = pd.DataFrame({"x": [-50.0, 0.0, 10.0, 60.0], "f": ["E", "A", "B", "C"]})
    for frame in (X, X.copy(), far):
        _, found = _disclosures(lambda frame=frame: model.predict(frame))
        assert not found


@pytest.mark.pins_behaviour
def test_predicts_readers_do_not_repeat_the_warning():
    """Control (passes on master): ``predict_priced`` reads ``predict``'s means without its test."""
    model = _row_j(Numeric, "fit")
    mu, found = _disclosures(lambda: base.predict_priced(model, ROW_J_NEW, None))
    assert not found
    np.testing.assert_array_equal(mu, base.predict_exact(model, ROW_J_NEW, warn=False))


def test_the_rating_table_marks_the_entries_the_data_do_not_determine(tmp_path):
    """Row J (spline): f's level E and the E column of ``x:f`` are marked; nothing else is.

    Mean-centred, every level of ``f`` reads the block's mean, which E moves,
    so every level is marked.  In the workbook a mark is a shaded cell and a
    note.
    """
    model = _row_j(lambda: Spline(n_knots=6), "fit")
    for centering, levels in (("native", [4]), ("mean", [0, 1, 2, 3, 4])):
        payload = model.rating_table_payload(ROW_J, Y, centering=centering, impact_bins=())
        blocks = {block.name: block for block in payload.main_effects}
        assert np.flatnonzero(blocks["f"].non_estimable).tolist() == levels
        assert not np.any(blocks["x"].non_estimable)
        (cells,) = payload.interactions
        columns = list(cells.table.columns[1:])
        marked = np.flatnonzero(np.any(cells.non_estimable, axis=0)).tolist()
        assert [columns[j] for j in marked] == ["E"]
        # every E cell away from x = 5 is marked
        keys = cells.table.iloc[:, 0].tolist()
        e = cells.non_estimable[:, columns.index("E")]
        assert e.sum() >= len(keys) - 1

    import openpyxl

    path = tmp_path / "rating.xlsx"
    model.export_rating_tables(path, ROW_J, Y, impact_bins=())
    notes = [
        cell.value
        for sheet in openpyxl.load_workbook(path).worksheets
        for row in sheet.iter_rows()
        for cell in row
        if isinstance(cell.value, str) and cell.value.startswith("Not determined")
    ]
    assert len(notes) == 2


def test_the_aliased_columns_per_unit_blocks_are_marked():
    """``x1`` and ``x2`` trade one for the other: each per-unit factor leans; ``f`` does not."""
    X, y, features = _aliased()
    model = _fit(X, y, features, family=superglm.families.poisson())
    payload = model.rating_table_payload(X, y, impact_bins=())
    blocks = {block.name: block for block in payload.main_effects}
    assert blocks["x1"].non_estimable.tolist() == [True]
    assert blocks["x2"].non_estimable.tolist() == [True]
    assert not np.any(blocks["f"].non_estimable)
