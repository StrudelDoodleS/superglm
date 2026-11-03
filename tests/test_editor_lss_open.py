"""Opening a SuperLSS model in the editor: composite keys, display mapping, refusals."""

from __future__ import annotations

import functools
import re

import joblib
import numpy as np
import pandas as pd
import pytest

import superglm
from superglm import GammaLS, LogNormalLS, Numeric, SuperLSS, TweedieLSS, s, ti
from superglm.editor import edit
from superglm.editor.errors import EditorKeyError, EditorValueError
from superglm.editor.lss import (
    RELATIVITY,
    DisplayTransform,
    LSSModelView,
    column_of,
    display_for,
    is_lss,
    lss_key,
    parameter_display,
    predictor_of,
    split_key,
)
from tests._lss_editor_fixtures import EPS, gamma_case, lognormal_case, nb_case, open_case
from tests.test_editor import editor_frame, editor_model  # noqa: F401

CASES = [lognormal_case, gamma_case, nb_case]


@functools.cache
def _interaction_model():
    case = lognormal_case()
    family = LogNormalLS()
    return SuperLSS(
        family,
        family.mean(s("age", k=6), s("power", k=5), ti("age", "power")),
        family.scale(s("age", k=5)),
    ).fit_reml(case.X, case.y, outer="efs+newton")


@functools.cache
def _truncated_model():
    rng = np.random.default_rng(20261106)
    X = pd.DataFrame({"age": rng.uniform(18.0, 90.0, 120)})
    y = rng.gamma(4.0, 100.0, 120)
    family = GammaLS()
    censored = np.zeros(120, dtype=bool)
    censored[:5] = True
    model = SuperLSS(family, family.mean(s("age", k=5)), family.scale())
    model.fit_reml(X, y, truncation=(1.0, None), censored=censored)
    return model, X, y


@functools.cache
def _linked_model(scale_link: str | None, mean_link: str | None = None):
    """A small gamma fit with the scale (and optionally the mean) on a stated link."""
    rng = np.random.default_rng(20261103)
    X = pd.DataFrame({"age": rng.uniform(18.0, 90.0, 400)})
    y = rng.gamma(4.0, 100.0 * (1.0 + X["age"].to_numpy() / 90.0))
    family = GammaLS()
    model = SuperLSS(
        family,
        family.mean(s("age", k=5), link=mean_link),
        family.scale(link=scale_link),
    )
    return model.fit_reml(X, y), X, y


@functools.cache
def _integer_column_model():
    rng = np.random.default_rng(20261103)
    X = pd.DataFrame({0: rng.uniform(18.0, 90.0, 300)})
    y = rng.gamma(4.0, 100.0, 300)
    family = GammaLS()
    model = SuperLSS(family, family.mean(superglm.term(0, Numeric())), family.scale())
    return model.fit_reml(X, y), X, y


def _unfitted_tweedie():
    family = TweedieLSS()
    return SuperLSS(family, family.mu(), family.phi(), family.p())


def _unfitted_location_form():
    family = LogNormalLS(parametrisation="location")
    return SuperLSS(family, family.location(), family.scale())


def _unfitted_mean_form():
    family = LogNormalLS()
    return SuperLSS(family, family.mean(), family.scale())


@pytest.mark.parametrize("make", CASES)
def test_opens_each_supported_family(make):
    case = make()
    session = edit(case.model, train_data=(case.X, case.y, case.sample_weight, case.offsets))
    second = case.model.parameter_names_[1]
    assert list(session.terms) == [
        "mean:age",
        "mean:region",
        "mean:power",
        f"{second}:age",
        f"{second}:region",
    ]
    assert isinstance(session.model, LSSModelView)
    assert session.model.lss is case.model
    assert isinstance(session.reference_model, LSSModelView)
    assert session.reference_model is session.model
    assert session.reference_model.lss is case.model
    assert session.centering == "native"
    for key, term in session.terms.items():
        parameter, name = split_key(key)
        reference = case.model.term_inference(parameter, name, X_train=case.X)
        np.testing.assert_array_equal(term.original_log_effect, reference.effect)
        np.testing.assert_array_equal(term.edited_log_effect, reference.effect)
        assert term.name == key
        # The interval is stored as exp(link) and read back as a link-scale effect.
        slack = 4.0 * EPS * (1.0 + np.max(np.abs(reference.lower)))
        np.testing.assert_allclose(term.ci_lower_log_effect, reference.lower, rtol=0, atol=slack)
        np.testing.assert_allclose(term.ci_upper_log_effect, reference.upper, rtol=0, atol=slack)
        assert term.weights.shape == term.original_log_effect.shape
        assert term.weights.sum() == len(case.X)
        assert term.metadata["centering_mode"] == "native"
        assert term.metadata["edf"] == reference.edf
        assert term.metadata["shape_support"] is None


def test_term_types_and_spline_metadata_come_from_the_fitted_spec():
    case = lognormal_case()
    session = open_case(case)
    age = session.terms["mean:age"]
    spec = session.model._specs["mean:age"]
    assert age.kind == "spline"
    assert age.metadata["term_type"] == "spline"
    assert age.metadata["spline"]["n_basis"] == spec._n_basis
    assert age.metadata["spline"]["degree"] == spec.degree
    assert age.metadata["spline"]["boundary"] == tuple(spec.fitted_boundary)
    assert "smooth_curve" not in age.metadata
    region = session.terms["scale:region"]
    assert region.kind == "categorical"
    assert region.metadata["term_type"] == "categorical"
    assert region.levels == list(
        case.model.term_inference("scale", "region", X_train=case.X).levels
    )


def test_bare_feature_name_opens_it_in_every_parameter():
    case = lognormal_case()
    assert list(open_case(case, terms=["age"]).terms) == ["mean:age", "scale:age"]
    assert list(open_case(case, terms=["power"]).terms) == ["mean:power"]
    assert list(open_case(case, terms=["scale:age", "region"]).terms) == [
        "scale:age",
        "mean:region",
        "scale:region",
    ]
    assert list(open_case(case, terms=["age", "mean:age"]).terms) == ["mean:age", "scale:age"]


def test_unknown_terms_are_refused_by_name():
    case = lognormal_case()
    for terms in (["bogus"], ["mean:bogus"], ["spread:age"]):
        with pytest.raises(EditorKeyError, match="Term not found"):
            open_case(case, terms=terms)


@pytest.mark.parametrize("terms", [["mean:age:power"], ["age:power"]])
def test_interactions_are_not_editable(terms):
    model = _interaction_model()
    case = lognormal_case()
    message = "Interactions are not editable in v1: " + repr(terms[0])
    with pytest.raises(EditorValueError, match=re.escape(message)):
        edit(model, terms=terms, train_data=(case.X, case.y))
    # Opening everything else is unaffected by the interaction.
    session = edit(model, train_data=(case.X, case.y))
    assert list(session.terms) == ["mean:age", "mean:power", "scale:age"]


def _refusal(case_id):
    """The model, the keywords to open it with, the error and its exact message."""
    case = lognormal_case()
    names = "mean, scale"
    rows = (case.X, case.y, None, None)
    if case_id == "family":
        message = (
            "The editor opens LogNormalLS (mean form), GammaLS and NegativeBinomialLS models; "
            "this model is TweedieLSS."
        )
        return _unfitted_tweedie(), {}, EditorValueError, message
    if case_id == "location-form":
        message = (
            "The editor opens LogNormalLS models in mean form; this one uses "
            "parametrisation='location'."
        )
        return _unfitted_location_form(), {}, EditorValueError, message
    if case_id == "unfitted":
        message = "Model must be fitted before creating an editor session."
        return _unfitted_mean_form(), {"train_data": rows}, RuntimeError, message
    if case_id == "truncated":
        model, X, y = _truncated_model()
        message = (
            "The editor opens models fitted without truncation or censoring; this model has "
            "120 truncated and 5 censored rows."
        )
        return model, {"train_data": (X, y)}, EditorValueError, message
    if case_id == "column-name":
        model, X, y = _integer_column_model()
        message = (
            "The editor names each SuperLSS term '<parameter>:<column>', so it opens models "
            "whose columns are named by strings; this model reads column 0. Rename the column "
            "to a string and refit to open it in the editor."
        )
        return model, {"train_data": (X, y)}, EditorValueError, message
    if case_id == "link":
        model, X, y = _linked_model("logit")
        message = (
            "The editor draws every parameter through its family's default link; this model "
            "fits 'scale' on another link. Refit it without link= on 'scale' to open it in "
            "the editor."
        )
        return model, {"train_data": (X, y)}, EditorValueError, message
    if case_id == "no-train-data":
        message = (
            "Opening a SuperLSS model needs its training rows: pass "
            "train_data=(X, y, sample_weight, offsets), with offsets a dict keyed by parameter "
            "name, or None."
        )
        return case.model, {}, EditorValueError, message
    if case_id == "centering":
        message = "SuperLSS terms open on their native scale; pass centering='native'."
        return case.model, {"train_data": rows, "centering": "mean"}, EditorValueError, message
    if case_id == "offsets-array":
        message = (
            f"Offsets for a SuperLSS model are a dict keyed by parameter name ({names}); "
            "got an array."
        )
        array = (case.X, case.y, None, np.zeros(len(case.y)))
        return case.model, {"train_data": array}, EditorValueError, message
    if case_id == "offsets-unknown-key":
        message = f"Unknown offset parameter 'location'; this model's parameters are {names}."
        unknown = (case.X, case.y, None, {"location": np.zeros(len(case.y))})
        return case.model, {"train_data": unknown}, EditorValueError, message
    if case_id == "train-rows":
        message = (
            "The editor needs the rows the model was fitted on: train_data has 400 rows, but "
            "the fit was given 600 and kept 600. Pass the frame the model was fitted on."
        )
        return case.model, {"train_data": (case.X[:400], case.y[:400])}, EditorValueError, message
    assert case_id == "amount-unit-glm"
    return None, {"amount_unit": "GBP"}, EditorValueError, "amount_unit applies to SuperLSS models."


@pytest.mark.parametrize(
    "case_id",
    [
        "family",
        "location-form",
        "unfitted",
        "truncated",
        "column-name",
        "link",
        "no-train-data",
        "centering",
        "offsets-array",
        "offsets-unknown-key",
        "train-rows",
        "amount-unit-glm",
    ],
)
def test_refusals(case_id, request):
    model, keywords, error, message = _refusal(case_id)
    if model is None:
        model = request.getfixturevalue("editor_model")
    with pytest.raises(error, match=f"^{re.escape(message)}$"):
        edit(model, **keywords)


def test_a_link_named_but_equal_to_the_default_still_opens():
    model, X, y = _linked_model(None, mean_link="log")
    session = edit(model, train_data=(X, y))
    assert "mean:age" in session.terms


def test_refusals_follow_the_documented_order():
    case = lognormal_case()
    model, X, y = _truncated_model()
    # Truncation is refused before the missing rows are.
    with pytest.raises(EditorValueError, match="truncated and 5 censored"):
        edit(model)
    # Missing rows before the scale, and the scale before the offsets.
    with pytest.raises(EditorValueError, match="needs its training rows"):
        edit(case.model, centering="mean")
    with pytest.raises(EditorValueError, match="native scale"):
        edit(case.model, centering="mean", train_data=(case.X, case.y, None, np.zeros(len(case.y))))
    # An unsupported family is named before the model is found unfitted.
    with pytest.raises(EditorValueError, match="this model is TweedieLSS"):
        edit(_unfitted_tweedie(), centering="mean")


def test_offsets_are_stored_by_parameter_name():
    nb = nb_case()
    session = open_case(nb)
    train = session._evaluation_data["train"]
    assert list(train.offset) == ["mean"]
    np.testing.assert_array_equal(train.offset["mean"], nb.offsets["mean"])
    lognormal = lognormal_case()
    assert open_case(lognormal)._evaluation_data["train"].offset is None
    held_out = (nb.X.iloc[:50], nb.y[:50], None, {"mean": nb.offsets["mean"][:50]})
    session = open_case(nb, validation_data=held_out)
    assert list(session._evaluation_data["validation"].offset) == ["mean"]
    short = (nb.X.iloc[:50], nb.y[:50], None, {"mean": nb.offsets["mean"][:10]})
    with pytest.raises(ValueError, match="offset"):
        open_case(nb, test_data=short)


def test_amount_unit_is_kept_on_the_view():
    case = lognormal_case()
    view = open_case(case, amount_unit="GBP").model
    assert view.amount_unit == "GBP"
    view = open_case(case).model
    assert view.amount_unit is None
    assert view.with_model(case.model).amount_unit is None


def test_restored_model_opens_with_train_data(tmp_path):
    case = lognormal_case()
    path = tmp_path / "model.joblib"
    joblib.dump(case.model, path)
    restored = joblib.load(path)
    assert restored._training_frame is None
    fresh = open_case(case)
    opened = edit(restored, train_data=(case.X, case.y, case.sample_weight, case.offsets))
    assert list(opened.terms) == list(fresh.terms)
    for key, term in opened.terms.items():
        np.testing.assert_array_equal(
            term.original_log_effect, fresh.terms[key].original_log_effect
        )
        np.testing.assert_array_equal(term.weights, fresh.terms[key].weights)
    np.testing.assert_array_equal(opened.model.reference_row(), fresh.model.reference_row())


def test_glm_open_is_unchanged(editor_model):  # noqa: F811
    session = edit(editor_model)
    assert session.model is editor_model
    assert session.reference_model is editor_model
    assert list(session.terms) == list(editor_model._feature_order)
    assert not is_lss(session.model)
    assert predictor_of(session.model, "x_spline") is None
    assert predictor_of(session.model, None) is None
    assert column_of(session.model, "x_spline") == "x_spline"
    assert display_for(session.model, "x_spline") is RELATIVITY


def test_keys_split_and_join():
    assert lss_key("mean", "age") == "mean:age"
    assert split_key("mean:age") == ("mean", "age")
    # A term is named by everything after the first colon; an interaction keeps its own.
    assert split_key("mean:age:power") == ("mean", "age:power")
    for bad in ("age", ":age", "mean:", ""):
        with pytest.raises(EditorKeyError):
            split_key(bad)


def test_view_answers_the_reads_the_editor_makes():
    case = lognormal_case()
    view = open_case(case).model
    assert is_lss(view) and is_lss(case.model)
    assert list(view._specs) == list(view._feature_order)
    assert view._feature_order == [
        "mean:age",
        "mean:region",
        "mean:power",
        "scale:age",
        "scale:region",
    ]
    assert dict(view._interaction_specs) == {}
    assert view._result is case.model.result_
    assert view.parameter_of("scale:age") == "scale"
    assert view.column_of("scale:age") == "age"
    assert predictor_of(view, "scale:age") == "scale"
    assert column_of(view, "scale:age") == "age"
    assert predictor_of(view, None) is None
    twin = view.with_model(case.model)
    assert twin is not view and twin.X_train is view.X_train
    with_interaction = LSSModelView(_interaction_model(), case.X)
    assert list(with_interaction._interaction_specs) == ["mean:age:power"]
    assert with_interaction._feature_order == ["mean:age", "mean:power", "scale:age"]


def test_reference_row_is_the_training_centre_and_survives_mutation():
    case = lognormal_case()
    view = open_case(case).model
    row = view.reference_row()
    assert list(row.columns) == ["age", "region", "power"]
    assert len(row) == 1
    assert row["age"].iloc[0] == float(np.median(case.X["age"]))
    assert row["power"].iloc[0] == float(np.median(case.X["power"]))
    assert row["region"].iloc[0] == "R0"
    row["age"] = -1.0
    assert view.reference_row()["age"].iloc[0] == float(np.median(case.X["age"]))


@pytest.mark.parametrize(
    ("parameter", "label", "role", "colour"),
    [
        ("mean", "Average", "average", "blue"),
        ("scale", "Spread", "spread", "orange"),
        ("theta", "Spread", "spread", "orange"),
    ],
)
def test_parameter_display_names_a_parameter_by_its_role(parameter, label, role, colour):
    display = parameter_display(parameter)
    assert (display.name, display.label, display.role, display.colour) == (
        parameter,
        label,
        role,
        colour,
    )
    assert display.chip == label


def test_parameter_display_refuses_what_it_does_not_name():
    with pytest.raises(EditorValueError, match="'power'"):
        parameter_display("power")


def _spread(kind, parameter_values):
    """The displayed spread a family's natural parameter stands for."""
    if kind == "lognormal_cv":
        return np.sqrt(np.expm1(parameter_values**2))
    if kind == "gamma_cv":
        return parameter_values
    return 1.0 / parameter_values


@pytest.mark.parametrize(
    ("make", "kind"),
    [(lognormal_case, "lognormal_cv"), (gamma_case, "gamma_cv"), (nb_case, "inverse_theta")],
)
def test_display_transform_reads_the_spread_at_the_reference_row(make, kind):
    case = make()
    session = open_case(case)
    second = case.model.parameter_names_[1]
    key = f"{second}:age"
    transform = display_for(session.model, key)
    assert transform.kind == kind
    assert display_for(session.model, "mean:age") is RELATIVITY
    term = session.terms[key]
    picks = np.arange(0, term.x.size, 40)
    frame = pd.concat([session.model.reference_row()] * picks.size, ignore_index=True)
    frame["age"] = term.x[picks]
    natural = case.model.predict_parameters(frame)[second].to_numpy()
    shown = transform.to_display(term.original_log_effect[picks])
    # eta is a sum of a handful of terms of size up to ~10; exp keeps that relative error.
    np.testing.assert_allclose(shown, _spread(kind, natural), rtol=64 * EPS, atol=0)


def test_lognormal_transform_carries_the_family_floor():
    case = lognormal_case()
    transform = display_for(open_case(case).model, "scale:age")
    assert transform.floor == LogNormalLS().scale_floor
    assert transform.to_payload() == {
        "kind": "lognormal_cv",
        "reference_eta": transform.reference_eta,
        "floor": transform.floor,
    }
    assert (transform.y_label, transform.y_format) == ("coefficient of variation", "times")
    assert (RELATIVITY.y_label, RELATIVITY.y_format) == ("relativity", "relativity")
    assert RELATIVITY.floor == 0.0 and RELATIVITY.reference_eta == 0.0
    inverse = DisplayTransform("inverse_theta")
    assert (inverse.y_label, inverse.y_format) == ("1/theta", "plain")


@pytest.mark.parametrize("kind", ["relativity", "lognormal_cv", "gamma_cv", "inverse_theta"])
def test_display_round_trips_and_clamps(kind):
    transform = DisplayTransform(kind, reference_eta=-0.7, floor=0.01)
    effect = np.linspace(-1.5, 1.5, 31)
    back = transform.from_display(transform.to_display(effect))
    np.testing.assert_allclose(back, effect, rtol=0, atol=64 * EPS * (1.0 + np.max(np.abs(effect))))
    # Values past the domain clamp to the nearest reachable effect and never give NaN.
    extreme = transform.from_display(np.array([-5.0, 0.0, 1e-300]))
    assert np.all(np.isfinite(extreme))
    assert (
        transform.from_display(np.array([0.0]))[0] == transform.from_display(np.array([1e-300]))[0]
    )


def test_display_matches_the_documented_formulas():
    e = np.array([-0.4, 0.0, 0.9])
    c, floor = -0.7, 0.01
    np.testing.assert_array_equal(DisplayTransform("relativity").to_display(e), np.exp(e))
    s = floor + np.exp(c + e)
    np.testing.assert_array_equal(
        DisplayTransform("lognormal_cv", c, floor).to_display(e), np.sqrt(np.expm1(s**2))
    )
    np.testing.assert_array_equal(DisplayTransform("gamma_cv", c).to_display(e), np.exp(c + e))
    np.testing.assert_array_equal(
        DisplayTransform("inverse_theta", c).to_display(e), np.exp(-(c + e))
    )
    v = np.array([0.5, 2.2, 7.0])
    np.testing.assert_array_equal(
        DisplayTransform("lognormal_cv", c, floor).from_display(v),
        np.log(np.sqrt(np.log1p(v**2)) - floor) - c,
    )
