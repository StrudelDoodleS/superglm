"""The one-risk payload: completed risks, features and link-scale predictors."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, GammaLS, Numeric, SuperLSS, s, term
from superglm.editor import EditorSession, edit, notebook
from superglm.editor.errors import EditorKeyError, EditorValueError
from superglm.editor.lss.apply import lss_model_with_edits
from superglm.editor.lss.panel import NOT_SUPERLSS
from superglm.editor.lss.risk import risk_payload, risk_row
from superglm.editor.server import create_editor_app
from tests._lss_editor_fixtures import (
    EPS,
    gamma_case,
    lognormal_case,
    nb_case,
    open_case,
    tweedie_case,
)
from tests.test_editor import editor_frame, editor_model  # noqa: F401

CASES = [lognormal_case, gamma_case]
RISKS = [{"age": 30.0, "region": "R3"}, {"power": 6.0}]


def _shift(session, key, delta):
    session.select_indices(key, list(range(session.terms[key].size)))
    session.shift(key, delta)


def _risks(session, risks):
    return risk_payload(session, risks, edited=lss_model_with_edits(session.model, session.terms))


def _predict_link(model, session, values):
    """The reference: the model's public ``predict_link`` on a one-row frame, offsets at zero."""
    frame = session.model.reference_row()
    for column, value in values.items():
        frame[column] = value
    return model.predict_link(frame).to_numpy()[0]


def _tolerance(session, eta):
    """Row-wise dot products differ between batch shapes by gamma_n times their size."""
    n_coefficients = session.model._fitted.layout.n_coefficients
    return 4 * n_coefficients * EPS * max(1.0, float(np.max(np.abs(eta))))


@pytest.mark.parametrize("case", CASES)
def test_each_risk_is_the_models_own_prediction_fitted_and_edited(case):
    case = case()
    session = open_case(case)
    unedited = _risks(session, RISKS)
    assert [r["eta"]["edited"] for r in unedited["risks"]] == [
        r["eta"]["fitted"] for r in unedited["risks"]
    ]

    _shift(session, "scale:age", 0.3)
    edited_model = lss_model_with_edits(session.model, session.terms)
    payload = _risks(session, RISKS)
    for risk, asked in zip(payload["risks"], RISKS, strict=True):
        fitted = _predict_link(case.model, session, asked)
        revised = _predict_link(edited_model, session, asked)
        tol = _tolerance(session, fitted)
        np.testing.assert_allclose(risk["eta"]["fitted"], fitted, rtol=0, atol=tol)
        np.testing.assert_allclose(risk["eta"]["edited"], revised, rtol=0, atol=tol)
        # Only the scale predictor moved, and by the shift.
        np.testing.assert_allclose(risk["eta"]["edited"][0], risk["eta"]["fitted"][0], atol=tol)
        np.testing.assert_allclose(
            risk["eta"]["edited"][1] - risk["eta"]["fitted"][1], 0.3, rtol=0, atol=tol
        )
    assert payload["amount_axis"] is not None and payload["kind"] == "risk"


@pytest.mark.parametrize("case", CASES)
def test_each_feature_moved_alone_from_the_reference_risk_explains_the_risk(case):
    case = case()
    session = open_case(case)
    # Raise the young end of the curve alone, so the edit moves what age 30 adds to the reference.
    session.select_indices("scale:age", list(range(session.terms["scale:age"].size // 3)))
    session.shift("scale:age", 0.3)
    edited_model = lss_model_with_edits(session.model, session.terms)
    payload = _risks(session, RISKS)
    columns = [feature["column"] for feature in payload["features"]]
    young = payload["risks"][0]["effects"]
    age = columns.index("age")
    assert young["edited"][age][1] - young["fitted"][age][1] > 0.1
    assert young["edited"][age][0] == young["fitted"][age][0]

    for which, model in (("fitted", case.model), ("edited", edited_model)):
        base = _predict_link(model, session, {})
        tol = _tolerance(session, base)
        np.testing.assert_allclose(payload["reference"][which], base, rtol=0, atol=tol)
        for risk in payload["risks"]:
            moved = np.asarray(risk["effects"][which])
            assert moved.shape == (len(columns), len(base))
            for column, effect in zip(columns, moved, strict=True):
                alone = _predict_link(model, session, {column: risk["values"][column]})
                np.testing.assert_allclose(effect, alone - base, rtol=0, atol=2 * tol)
            # These fixtures hold main effects only, so the moves add up to the risk's own.
            np.testing.assert_allclose(
                base + moved.sum(axis=0), risk["eta"][which], rtol=0, atol=4 * tol
            )
    # The first risk leaves ``power`` at the reference, and ``power`` feeds the mean only.
    first, second = payload["risks"]
    assert first["effects"]["fitted"][columns.index("power")] == [0.0, 0.0]
    assert second["effects"]["fitted"][columns.index("power")][1] == 0.0
    assert second["effects"]["fitted"][columns.index("power")][0] != 0.0
    json.dumps(payload)


def test_a_risk_names_only_what_it_changes():
    session = open_case(lognormal_case())
    reference = session.model.reference_row().iloc[0]
    payload = _risks(session, [{}, {"power": 9.0}])
    bare, powered = payload["risks"]

    assert bare["values"] == {
        "age": reference["age"],
        "region": reference["region"],
        "power": reference["power"],
    }
    assert powered["values"] == {**bare["values"], "power": 9.0}
    fitted = _predict_link(session.model.lss, session, {})
    np.testing.assert_allclose(
        bare["eta"]["fitted"], fitted, rtol=0, atol=_tolerance(session, fitted)
    )
    # ``power`` feeds the mean only.
    assert powered["eta"]["fitted"][1] == bare["eta"]["fitted"][1]
    assert powered["eta"]["fitted"][0] != bare["eta"]["fitted"][0]
    json.dumps(payload)


def test_features_describe_the_training_columns():
    case = lognormal_case()
    session = open_case(case)
    reference = session.model.reference_row().iloc[0]
    features = {f["column"]: f for f in _risks(session, [{}])["features"]}

    assert list(features) == list(session.model.data_columns())
    for column, terms in (
        ("age", {"mean": "mean:age", "scale": "scale:age"}),
        ("power", {"mean": "mean:power"}),
    ):
        assert features[column] == {
            "column": column,
            "kind": "numeric",
            "min": float(case.X[column].min()),
            "max": float(case.X[column].max()),
            "reference": float(reference[column]),
            "levels": None,
            "terms": terms,
        }
    region = features["region"]
    assert region["kind"] == "levels" and region["min"] is None and region["max"] is None
    assert region["reference"] == reference["region"]
    assert region["terms"] == {"mean": "mean:region", "scale": "scale:region"}
    assert region["levels"] == session.terms["mean:region"].levels

    session.terms["mean:region"].levels = list(reversed(region["levels"]))
    reordered = {f["column"]: f for f in _risks(session, [{}])["features"]}
    assert reordered["region"]["levels"] == list(reversed(region["levels"]))


def _grade_session(categorical):
    """A gamma model reading integer grades 1, 2 as levels 1 to 3; level 3 is declared only."""
    case = gamma_case()
    grades = np.resize([1, 2], len(case.y))
    X = case.X.assign(grade=pd.Categorical(grades, categories=[1, 2, 3]) if categorical else grades)
    family = GammaLS()
    model = SuperLSS(
        family, family.mean(term("grade", Categorical(levels=[1, 2, 3]))), family.scale()
    ).fit_reml(X, case.y)
    return model, X, edit(model, train_data=(X, case.y))


def test_a_feature_names_every_fitted_term_it_moves_when_only_some_are_editable():
    """A term not opened for editing still moves the risk, so the walk keeps its parameter."""
    case = lognormal_case()
    session = open_case(case, terms=["mean:age"])
    assert "scale:age" not in session.terms
    features = {f["column"]: f for f in _risks(session, [{}])["features"]}
    assert features["age"]["terms"] == {"mean": "mean:age", "scale": "scale:age"}


def test_a_numeric_column_read_as_levels_takes_its_declared_level_as_a_number():
    model, X, session = _grade_session(False)
    feature = _risks(session, [{}])["features"][-1]
    assert feature["kind"] == "numeric" and (feature["min"], feature["max"]) == (1.0, 2.0)
    for risk, level in zip(
        _risks(session, [{"grade": 3}, {"grade": 1}])["risks"], (3, 1), strict=True
    ):
        own = model.predict_link(X.iloc[:1].assign(grade=level)).to_numpy()[0]
        np.testing.assert_allclose(risk["eta"]["fitted"], own, rtol=1e-12)


def test_a_level_label_is_read_back_as_the_training_value():
    # Integer categories show as text and are predicted from the integer, declared-only ones too.
    model, X, session = _grade_session(True)
    feature = _risks(session, [{}])["features"][-1]
    assert feature["kind"] == "levels" and feature["levels"] == ["1", "2", "3"]
    for risk, level in zip(
        _risks(session, [{"grade": "3"}, {"grade": 1}])["risks"], (3, 1), strict=True
    ):
        own = model.predict_link(X.iloc[:1].assign(grade=level)).to_numpy()[0]
        np.testing.assert_allclose(risk["eta"]["fitted"], own, rtol=1e-12)


def test_a_count_family_is_read_as_counts_with_no_amount_axis():
    session = open_case(nb_case())
    payload = _risks(session, [{}])
    assert payload["family"] == "negative_binomial"
    assert (payload["kind"], payload["amount_axis"]) == ("counts", None)
    assert [parameter["name"] for parameter in payload["parameters"]] == ["mean", "theta"]
    # Its risks are explained as any other's: the reference risk plus each feature's own move.
    moved = _risks(session, [{"age": 30.0, "region": "R3"}])
    risk = moved["risks"][0]
    base = np.asarray(moved["reference"]["fitted"])
    np.testing.assert_allclose(
        base + np.asarray(risk["effects"]["fitted"]).sum(axis=0),
        risk["eta"]["fitted"],
        rtol=0,
        atol=4 * _tolerance(session, base),
    )
    assert risk["eta"]["fitted"] != moved["reference"]["fitted"]


def test_a_tweedie_risk_has_three_predictors_and_an_axis_for_its_positive_part():
    case = tweedie_case()
    session = open_case(case)
    payload = _risks(session, [{}, {"age": 30.0, "region": "R3"}])

    assert (payload["family"], payload["kind"]) == ("tweedie", "risk")
    assert [parameter["name"] for parameter in payload["parameters"]] == [
        "mean",
        "dispersion",
        "power",
    ]
    assert payload["amount_axis"]["min_log10"] < payload["amount_axis"]["max_log10"]
    base = np.asarray(payload["reference"]["fitted"])
    for risk, asked in zip(payload["risks"], ({}, {"age": 30.0, "region": "R3"}), strict=True):
        fitted = _predict_link(case.model, session, asked)
        tol = _tolerance(session, fitted)
        np.testing.assert_allclose(risk["eta"]["fitted"], fitted, rtol=0, atol=tol)
        moved = np.asarray(risk["effects"]["fitted"])
        np.testing.assert_allclose(
            base + moved.sum(axis=0), risk["eta"]["fitted"], rtol=0, atol=4 * tol
        )
        # The power is a constant: no feature moves it.
        assert not moved[:, 2].any()
    assert payload["risks"][1]["eta"]["fitted"][:2] != payload["risks"][0]["eta"]["fitted"][:2]


def test_refusals_say_what_to_change():
    session = open_case(lognormal_case())
    with pytest.raises(EditorKeyError) as unknown:
        _risks(session, [{"colour": 1.0}])
    assert unknown.value.public_message == (
        "No feature of this model reads 'colour'; choose one of: age, region, power."
    )
    for bad in (float("nan"), float("inf"), "5", True, None):
        with pytest.raises(EditorValueError) as number:
            _risks(session, [{"age": bad}])
        assert number.value.public_message == (
            f"'age' takes a finite number; got {bad!r}. Enter a number."
        )
    with pytest.raises(EditorValueError) as level:
        _risks(session, [{"region": "R9"}])
    assert level.value.public_message == (
        "'R9' is not a level of 'region', which holds R0, R1, R2, R3, R4 and others. "
        "Choose one of its levels."
    )
    for risks in ([], [{}] * 9):
        with pytest.raises(EditorValueError) as count:
            _risks(session, risks)
        assert count.value.public_message == "Give between 1 and 8 risks to compare."
    with pytest.raises(EditorValueError, match="mapping"):
        _risks(session, [3.0])
    # A value outside the training range is extrapolated, not clamped.
    assert _risks(session, [{"age": 500.0}])["risks"][0]["values"]["age"] == 500.0


def test_a_superglm_session_has_no_risk_view(editor_model):  # noqa: F811
    session = EditorSession.from_model(editor_model)
    with pytest.raises(EditorValueError) as refusal:
        risk_payload(session, [{}], edited=session.model)
    assert refusal.value.public_message == NOT_SUPERLSS
    with pytest.raises(EditorValueError, match="SuperLSS"):
        risk_row(session)


def test_risk_row_is_a_training_row():
    case = lognormal_case()
    session = open_case(case)
    row = risk_row(session, 17)
    expected = case.X.iloc[17]
    assert row == {
        "values": {
            "age": expected["age"],
            "region": expected["region"],
            "power": expected["power"],
        },
        "index": 17,
    }
    drawn = risk_row(session)
    train = case.X.iloc[drawn["index"]]
    assert drawn["values"] == {c: train[c] for c in ("age", "region", "power")}
    assert 0 <= drawn["index"] < len(case.X)
    for bad in (-1, len(case.X), True, 1.5):
        with pytest.raises(EditorValueError, match="not a training row"):
            risk_row(session, bad)


def _post(widget, path, body):
    request = urllib.request.Request(
        f"{widget.url}{path}",
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "X-SuperGLM-Editor-Token": widget._token},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as reply:
            return reply.status, json.loads(reply.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_the_routes_are_read_routes_and_the_widget_serves_them():
    routes = {
        (route.path, frozenset(route.methods or ())) for route in create_editor_app(object()).routes
    }
    for path in ("/risk", "/risk_row"):
        assert path in notebook.READ_ROUTES and path not in notebook.CHANGE_ROUTES
        assert (path, frozenset({"POST"})) in routes

    session = open_case(lognormal_case())
    widget = session.widget()
    try:
        status, unedited = _post(widget, "/risk", {"risks": RISKS})
        assert status == 200
        assert [r["eta"]["edited"] for r in unedited["risks"]] == [
            r["eta"]["fitted"] for r in unedited["risks"]
        ]
        _shift(session, "scale:age", 0.3)
        status, edited = _post(widget, "/risk", {"risks": RISKS})
        assert status == 200 and edited == json.loads(json.dumps(widget._risk(RISKS)))
        assert edited["risks"][0]["eta"]["fitted"] == unedited["risks"][0]["eta"]["fitted"]
        assert edited["risks"][0]["eta"]["edited"] != edited["risks"][0]["eta"]["fitted"]

        assert _post(widget, "/risk", {}) == (400, {"error": "Missing required field: risks."})
        assert _post(widget, "/risk", {"risks": []})[0] == 400
        status, row = _post(widget, "/risk_row", {"index": 3})
        assert status == 200 and row == json.loads(json.dumps(risk_row(session, 3)))
        status, drawn = _post(widget, "/risk_row", {})
        assert status == 200 and set(drawn["values"]) == {"age", "region", "power"}
        assert _post(widget, "/risk_row", {"index": 10**6})[0] == 400
    finally:
        widget.close()


def test_a_model_that_reads_no_column_explains_its_one_risk():
    """An intercept-only fit opens in the editor; its Explain tab has risks and no features."""
    from superglm import GammaLS, SuperLSS
    from superglm.editor import edit
    from superglm.editor.lss.risk import risk_payload, risk_row

    X = pd.DataFrame({"unused": np.arange(50.0)})
    y = np.random.default_rng(3).gamma(4.0, 10.0, 50)
    family = GammaLS()
    model = SuperLSS(family, family.mean(), family.scale()).fit(X, y)
    session = edit(model, train_data=(X, y))
    reply = risk_payload(session, [{}], edited=session.model)
    assert reply["features"] == []
    [only] = reply["risks"]
    np.testing.assert_allclose(
        only["eta"]["fitted"], model.predict_link(X.iloc[:1]).to_numpy()[0], rtol=1e-12
    )
    assert risk_row(session, 0)["values"] == {}


def test_a_column_read_only_by_an_interaction_keeps_its_parameter_in_the_walk():
    """A mean holding only a factor smooth reads x and g through the interaction alone."""
    from superglm import FactorSmooth, interaction
    from superglm.editor import edit

    rng = np.random.default_rng(5)
    X = pd.DataFrame({"x": rng.uniform(0.0, 1.0, 120), "g": rng.choice(["a", "b"], 120)})
    y = rng.gamma(4.0, np.exp(2.0 + X.x + 0.8 * (X.g == "b")) / 4.0)
    family = GammaLS()
    model = SuperLSS(
        family, family.mean(interaction(FactorSmooth("x", group="g", k=5))), family.scale()
    ).fit_reml(X, y)
    session = edit(model, train_data=(X, y))
    features = {f["column"]: f for f in _risks(session, [{}])["features"]}
    assert set(features) == {"x", "g"}
    for column in ("x", "g"):
        assert list(features[column]["terms"]) == ["mean"]


def _linear_session(storage):
    """A ``Numeric`` term on ``x`` stored as ``storage``; booleans split exactly in half."""
    rng = np.random.default_rng(617)
    if storage == "bool":
        x = pd.Series(np.arange(300) % 2 == 0)
    else:
        x = pd.Series(10.0 + rng.uniform(0.0, 1.0, 300)).astype(storage)
    X = pd.DataFrame({"x": x, "z": rng.uniform(0.0, 1.0, 300)})
    numbers = np.asarray(x, dtype=np.float64)
    y = rng.gamma(10.0, np.exp(1.0 + 0.4 * numbers + X["z"].to_numpy()) / 10.0)
    family = GammaLS()
    model = SuperLSS(
        family, family.mean(term("x", Numeric()), s("z", k=5)), family.scale()
    ).fit_reml(X, y)
    return edit(model, train_data=(X, y)), model, numbers


@pytest.mark.parametrize("storage", ["float64", "object", "str", "bool"])
def test_a_linear_column_round_trips_through_the_explain_tab_whatever_its_storage(storage):
    session, model, numbers = _linear_session(storage)
    reply = risk_payload(session, [{"z": 0.3}], edited=model)
    (feature,) = [f for f in reply["features"] if f["column"] == "x"]
    if storage == "bool":
        # A boolean column keeps its levels, and is held at one it takes: a tied mode
        # goes to the first, False, so the reference risk is a real one.
        assert feature["kind"] == "levels"
        assert feature["reference"] == "False" and feature["reference"] in feature["levels"]
        held = 0.0
    else:
        assert feature["kind"] == "numeric"
        assert feature["reference"] == np.median(numbers)
        assert (feature["min"], feature["max"]) == (numbers.min(), numbers.max())
        held = feature["reference"]
    # The reference risk is priced at the held value, which the Editor curve is read from.
    expected = model.predict_link(pd.DataFrame({"x": [held], "z": [0.3]})).to_numpy()[0]
    np.testing.assert_allclose(
        reply["risks"][0]["eta"]["fitted"], expected, rtol=0.0, atol=_tolerance(session, expected)
    )
    effect = model.term_inference("mean", "x", n_points=3, simultaneous=False)
    np.testing.assert_allclose(
        effect.effect, model.coef_["mean:x"] * (effect.x - held), rtol=1e-12, atol=1e-15
    )

    # A reply's own values, and a training row's, are accepted when sent back.
    again = risk_payload(session, [reply["risks"][0]["values"]], edited=model)
    assert again["risks"][0]["eta"] == reply["risks"][0]["eta"]
    row = risk_row(session, 5)["values"]
    (priced,) = risk_payload(session, [row], edited=model)["risks"]
    training = model.predict_link(session.model.X_train.iloc[[5]]).to_numpy()[0]
    np.testing.assert_allclose(
        priced["eta"]["fitted"], training, rtol=0.0, atol=_tolerance(session, training)
    )


def test_a_text_numeric_column_with_a_missing_zero_weight_row_reads_after_a_reload():
    # Sol, PR 617 round 3: a missing value in an object or string column read by
    # ``Numeric`` must be dropped before conversion, as master does.
    import pickle

    rng = np.random.default_rng(617)
    n = 240
    x = np.tile([10.0, 11.0, 12.0], n // 3)
    g = np.tile(list("abcd"), n // 4)
    y = rng.gamma(20.0, np.exp(1.0 + 0.1 * (x - 11.0)) / 20.0)
    weights = np.ones(n)
    for storage in ("object", "string"):
        X = pd.DataFrame({"x": pd.Series(x).astype(storage), "g": g})
        X.loc[0, "x"] = pd.NA
        weights[0] = 0.0
        family = GammaLS()
        model = SuperLSS(
            family, family.mean(term("x", Numeric()), term("g", Categorical())), family.scale()
        ).fit(X, y, sample_weight=weights)
        restored = pickle.loads(pickle.dumps(model))
        inference = restored.term_inference("mean", "g", X_train=X, simultaneous=False)
        assert np.all(np.isfinite(inference.multiplier))
