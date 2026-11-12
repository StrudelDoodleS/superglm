"""Opening a SuperLSS model of any scalar family in the editor: keys, display, refusals."""

from __future__ import annotations

import dataclasses
import functools
import re

import joblib
import numpy as np
import pandas as pd
import pytest

import superglm
from superglm import GammaLS, LogNormalLS, Numeric, SuperLSS, TweedieLSS, s, ti
from superglm.distributional.families import LomaxLSS, NegativeBinomialLS
from superglm.editor import edit
from superglm.editor.display import (
    ADDITIVE,
    average_parameter,
    parameter_display_for,
    parameter_spec,
    term_y_label,
)
from superglm.editor.errors import EditorKeyError, EditorValueError
from superglm.editor.lss import (
    INVERSE_THETA,
    RELATIVITY,
    LSSModelView,
    column_of,
    display_for,
    is_lss,
    lss_key,
    parameter_display,
    predictor_of,
    split_key,
)
from superglm.editor.lss.apply import lss_model_with_edits
from superglm.editor.lss.display import y_label_for
from superglm.editor.lss.panel import NOT_DRAWN_YET, distribution_panel_payload
from superglm.editor.lss.risk import risk_payload
from superglm.editor.lss.validation import lss_dataset_metrics
from superglm.editor.payloads import session_payload
from tests._lss_editor_fixtures import (
    EPS,
    SCALAR_FAMILIES,
    gamma_case,
    lognormal_case,
    nb_case,
    open_case,
    scalar_family_case,
    tweedie_case,
)
from tests.test_editor import editor_frame, editor_model  # noqa: F401

CASES = [lognormal_case, gamma_case, nb_case, tweedie_case]


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
def _unnamed_column_model(label):
    rng = np.random.default_rng(20261103)
    X = pd.DataFrame({label: rng.uniform(18.0, 90.0, 300)})
    y = rng.gamma(4.0, 100.0, 300)
    family = GammaLS()
    model = SuperLSS(family, family.mean(superglm.term(label, Numeric())), family.scale())
    return model.fit_reml(X, y), X, y


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
        # A spline term carries the values a shaped range is counted on.
        assert (term.metadata["shape_support"] is None) == (term.x is None)


def test_a_row_the_fit_left_out_sets_no_grid_or_reference_value():
    rng = np.random.default_rng(20261103)
    X = pd.DataFrame({"x": np.linspace(0.0, 1.0, 180)})
    X.loc[179, "x"] = np.inf
    y = rng.gamma(4.0, 100.0, 180)
    weights = np.ones(180)
    weights[-1] = 0.0
    family = GammaLS()
    model = SuperLSS(family, family.mean(s("x", k=5)), family.scale()).fit_reml(
        X, y, sample_weight=weights
    )

    session = edit(model, train_data=(X, y, weights))

    grid = np.asarray(session.terms["mean:x"].x)
    assert grid.min() == 0.0
    assert grid.max() == X["x"].iloc[178]
    assert session.model.reference_row()["x"].iloc[0] == np.median(X["x"].iloc[:179])


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
    if case_id in ("column-name", "column-name-none"):
        label = 0 if case_id == "column-name" else None
        model, X, y = _unnamed_column_model(label)
        message = (
            "The editor names each SuperLSS term '<parameter>:<column>', so it opens models "
            f"whose columns are named by strings; this model reads column {label!r}. Rename the "
            "column to a string and refit to open it in the editor."
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
        "unfitted",
        "truncated",
        "column-name",
        "column-name-none",
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


@pytest.mark.parametrize("name", SCALAR_FAMILIES)
def test_every_scalar_family_opens(name):
    case = scalar_family_case(name)
    session = open_case(case)
    first, second, *_ = case.model.parameter_names_
    assert list(session.terms) == [f"{first}:age", f"{second}:age"]
    payload = session_payload(session)
    for key in session.terms:
        assert payload[key]["y_label"] == term_y_label(session.model, key)
        assert np.all(np.isfinite(payload[key]["y"]))
    reason = NOT_DRAWN_YET.format(family=type(case.model.family).__name__)
    panel = distribution_panel_payload(session, "age", edited=session.model)
    assert (panel["kind"], panel["family"], panel["empty_reason"]) == ("empty", None, reason)
    assert [p["name"] for p in panel["parameters"]] == list(case.model.parameter_names_)
    risk = risk_payload(session, [{"age": 30.0}], edited=session.model)
    assert (risk["kind"], risk["reason"], risk["amount_axis"]) == ("unavailable", reason, None)
    assert [feature["column"] for feature in risk["features"]] == ["age"]
    assert [p["name"] for p in risk["parameters"]] == list(case.model.parameter_names_)
    # The validation tab scores any response, a real-line one included.
    scores = lss_dataset_metrics(session.model, session._evaluation_data["train"])
    assert np.isfinite(scores["log_score"])
    assert np.isfinite(scores["actual_expected"])


def test_overridden_link_draws_by_the_fitted_link():
    gaussian = open_case(scalar_family_case("GaussianLS", ("log",)))
    # A log link on a location that defaults to identity draws a relativity.
    assert display_for(gaussian.model, "location:age") == RELATIVITY
    assert term_y_label(gaussian.model, "location:age") == "relativity of the location"
    model, X, y = _linked_model("logit")
    gamma = edit(model, train_data=(X, y))
    # A CV fitted on a logit link draws the relativity of its odds, and the browser's gamma
    # maths, which reads a log CV, is not used for it.
    shown = display_for(gamma.model, "mean:age"), display_for(gamma.model, "scale:age")
    assert [transform.to_payload() for transform in shown] == [
        {"kind": "relativity", "link": "log"},
        {"kind": "relativity", "link": "logit"},
    ]
    panel = distribution_panel_payload(gamma, "age", edited=gamma.model)
    assert (panel["kind"], panel["family"]) == ("empty", None)
    assert panel["parameters"][1]["link"] == {"kind": "logit"}


def test_tweedie_power_terms_open_and_draw():
    case = tweedie_case()
    family = TweedieLSS()
    model = SuperLSS(
        family, family.mu(s("age", k=5)), family.phi(s("age", k=4)), family.p(s("age", k=4))
    ).fit_reml(case.X, case.y, outer="efs+newton")
    session = edit(model, train_data=(case.X, case.y))
    assert list(session.terms) == ["mean:age", "dispersion:age", "power:age"]
    power = display_for(session.model, "power:age")
    assert power.to_payload() == {"kind": "additive", "link": "bounded_power"}
    assert (
        term_y_label(session.model, "power:age") == "effect on the bounded power scale of the power"
    )
    panel = distribution_panel_payload(session, "age", edited=session.model)
    assert (panel["kind"], panel["family"]) == ("numeric", "tweedie")
    # The browser's Tweedie law reads the power of each row from its own predictor.
    assert panel["term_keys"]["power"] == "power:age"
    powers = np.asarray(panel["eta"]["fitted"]["rows"])[:, 2]
    assert np.ptp(powers) > 0.0
    assert panel["amount_axis"]["min_log10"] < panel["amount_axis"]["max_log10"]


def test_gaussian_location_is_blue_and_additive():
    session = open_case(scalar_family_case("GaussianLS"))
    shown = parameter_display_for(session.model, "location")
    assert (shown.role, shown.colour) == ("average", "blue")
    assert display_for(session.model, "location:age") is ADDITIVE
    assert term_y_label(session.model, "location:age") == "effect on the location"
    assert parameter_display_for(session.model, "scale").role == "spread"


def test_lomax_scale_is_the_average_parameter():
    session = open_case(scalar_family_case("LomaxLSS"))
    assert average_parameter(session.model.family) == "scale"
    assert parameter_display_for(session.model, "scale").colour == "blue"
    assert parameter_display_for(session.model, "shape").colour == "orange"
    assert display_for(session.model, "scale:age") is RELATIVITY


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
    # No family, form or link is refused: an unfitted model of any family is found unfitted.
    family = LomaxLSS()
    with pytest.raises(RuntimeError, match="must be fitted"):
        edit(SuperLSS(family, family.scale(), family.shape()), centering="mean")


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
    # editor_model is gaussian/identity: its curves are added amounts.
    assert display_for(session.model, "x_spline") is ADDITIVE


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
    ("family", "parameter", "label", "chip", "role", "colour"),
    [
        (GammaLS(), "mean", "Mean", "Mean", "average", "blue"),
        (GammaLS(), "scale", "CV", "CV", "spread", "orange"),
        (NegativeBinomialLS(), "theta", "Overdispersion", "1/θ", "spread", "orange"),
        (TweedieLSS(), "dispersion", "Dispersion", "Dispersion", "spread", "orange"),
        (TweedieLSS(), "power", "Power", "Power", "spread", "orange"),
    ],
    ids=lambda value: value if isinstance(value, str) else type(value).__name__,
)
def test_parameter_display_names_a_parameter_by_its_family_s_label(
    family, parameter, label, chip, role, colour
):
    spec = parameter_spec(family, parameter)
    display = parameter_display(spec, average=parameter == average_parameter(family))
    assert (display.name, display.label, display.chip, display.role, display.colour) == (
        parameter,
        label,
        chip,
        role,
        colour,
    )
    assert display.full_name == spec.description


def test_an_unlabelled_parameter_shows_its_name_capitalised():
    spec = dataclasses.replace(parameter_spec(GammaLS(), "scale"), label=None, description=None)
    shown = parameter_display(spec, average=False)
    assert (shown.label, shown.chip, shown.full_name) == ("Scale", "Scale", "Scale")


@pytest.mark.parametrize(
    ("make", "transform"),
    [
        (lognormal_case, RELATIVITY),
        (gamma_case, RELATIVITY),
        (nb_case, INVERSE_THETA),
    ],
)
def test_a_term_is_the_relativity_its_parameter_moves_by(make, transform):
    case = make()
    session = open_case(case)
    second = case.model.parameter_names_[1]
    key = f"{second}:age"
    assert display_for(session.model, key) == transform
    assert display_for(session.model, "mean:age") is RELATIVITY
    term = session.terms[key]
    picks = np.arange(0, term.x.size, 40)
    frame = pd.concat([session.model.reference_row()] * picks.size, ignore_index=True)
    frame["age"] = term.x[picks]
    # The editor's reference has zero array offsets; a new frame needs these
    # replacements explicitly, even when only its spread is read here.
    offsets = None if case.offsets is None else dict.fromkeys(case.offsets, np.zeros(len(frame)))
    natural = case.model.predict_parameters(frame, offsets=offsets)[second].to_numpy()
    if isinstance(case.model.family, LogNormalLS):
        natural = natural - case.model.family.scale_floor
    # A relativity of the parameter between two points of the curve (the overdispersion's
    # is of 1/theta): the curve's own value, whatever the other terms hold the parameter at.
    shown = transform.to_display(term.original_log_effect[picks])
    expected = natural if transform.increasing else 1.0 / natural
    np.testing.assert_allclose(shown / shown[0], expected / expected[0], rtol=64 * EPS, atol=0)


def test_the_display_transforms_name_their_direction_and_payload():
    assert RELATIVITY.to_payload() == {"kind": "relativity", "link": "log"}
    assert RELATIVITY.increasing
    assert INVERSE_THETA.to_payload() == {"kind": "inverse_theta", "link": "log"}
    assert not INVERSE_THETA.increasing
    assert y_label_for(RELATIVITY, "mean") == "relativity"
    assert y_label_for(RELATIVITY, "CV") == "relativity of the CV"
    assert y_label_for(RELATIVITY, "σ") == "relativity of σ"
    assert y_label_for(INVERSE_THETA, "overdispersion") == "relativity of the overdispersion"
    assert y_label_for(RELATIVITY, "dispersion") == "relativity of the dispersion"


@pytest.mark.parametrize("transform", [RELATIVITY, INVERSE_THETA], ids=lambda t: t.kind)
def test_display_round_trips_and_clamps(transform):
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
    np.testing.assert_array_equal(RELATIVITY.to_display(e), np.exp(e))
    np.testing.assert_array_equal(INVERSE_THETA.to_display(e), np.exp(-e))
    v = np.array([0.5, 2.2, 7.0])
    np.testing.assert_array_equal(RELATIVITY.from_display(v), np.log(v))
    np.testing.assert_array_equal(INVERSE_THETA.from_display(v), -np.log(v))


def test_a_tweedie_dispersion_edit_in_display_units_moves_the_model_as_it_says():
    case = tweedie_case()
    session = open_case(case)
    key = "dispersion:age"
    transform = display_for(session.model, key)
    # φ carries the amount's units, so it is shown as relativities, as the mean is.
    assert transform is RELATIVITY
    term = session.terms[key]
    shown = transform.to_display(term.original_log_effect)
    # Dragging every point to twice its shown relativity doubles the model's dispersion.
    session.set_values(key, range(term.size), transform.from_display(2.0 * shown))
    revised = lss_model_with_edits(session.model, session.terms)
    for age in (25.0, 54.0, 80.0):
        frame = session.model.reference_row()
        frame["age"] = age
        before, after = (m.predict_parameters(frame).iloc[0] for m in (case.model, revised))
        np.testing.assert_allclose(after["dispersion"], 2.0 * before["dispersion"], rtol=1e-8)
        np.testing.assert_allclose(after["mean"], before["mean"], rtol=1e-8)
        assert after["power"] == before["power"]
