"""Turning edited SuperLSS curves into a revised model."""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd
import pytest

from superglm import LogNormalLS, Spline, SuperLSS, cat, s, term
from superglm.distributional.terms import _resolve_term, _term_design
from superglm.editor import edit
from superglm.editor.apply import _term_weights, materialize_edit_request
from superglm.editor.errors import EditorValueError
from superglm.editor.lss import split_key
from superglm.editor.lss.apply import lss_model_with_edits
from superglm.features.ordered_categorical import OrderedCategorical
from tests._lss_editor_fixtures import (
    EPS,
    REGIONS,
    FitCase,
    gamma_case,
    lognormal_case,
    nb_case,
    open_case,
)


@functools.cache
def select_case():
    """The log-normal case with a select-centred smooth of ``age`` in the mean."""
    base = lognormal_case()
    family = LogNormalLS()
    model = SuperLSS(
        family,
        family.mean(s("age", k=6, select=True), cat("region")),
        family.scale(s("age", k=5)),
    ).fit_reml(base.X, base.y, outer="efs+newton")
    return FitCase(model, base.X, base.y, None, None)


@functools.cache
def ordered_case():
    """The log-normal case with the scale's region read as an ordered smooth of its levels."""
    base = lognormal_case()
    family = LogNormalLS()
    ordered = OrderedCategorical(order=list(REGIONS), basis=Spline(kind="cr", k=4))
    model = SuperLSS(
        family, family.mean(s("age", k=6)), family.scale(term("region", ordered))
    ).fit_reml(base.X, base.y, outer="efs+newton")
    return FitCase(model, base.X, base.y, None, None)


def _rows_with(view, column, values):
    """The reference row repeated, with ``column`` set to ``values``."""
    frame = pd.concat([view.reference_row()] * len(values), ignore_index=True)
    frame[column] = list(values)
    return frame


def _basis_edit(view, key, term):
    """An edit the term's basis carries exactly: its rows, design and the curve's shift."""
    parameter, column = split_key(key)
    _, state, term_slice = _resolve_term(view._fitted, parameter, column)
    rows = _rows_with(view, column, term.levels if term.x is None else term.x)
    B = _term_design(view._fitted, rows, state, term_slice)
    delta = np.random.default_rng(20261103).normal(0.0, 0.05, B.shape[1])
    return rows, B, B @ delta


def _coefficients(model):
    return np.asarray(model._require_fitted().coefficients)


def _edited_positions(view, key):
    """The coefficient indices an edit of ``key`` may move: its block and its intercept."""
    parameter, column = split_key(key)
    _, state, term_slice = _resolve_term(view._fitted, parameter, column)
    return np.r_[np.arange(term_slice.start, term_slice.stop), state.intercept_index]


def test_no_edits_returns_the_fitted_model():
    case = lognormal_case()
    session = open_case(case)
    assert lss_model_with_edits(session.model, session.terms) is case.model


@pytest.mark.parametrize(
    ("make", "key"),
    [
        (lognormal_case, "scale:age"),
        (lognormal_case, "mean:power"),
        (gamma_case, "mean:age"),
        (select_case, "mean:age"),
        (ordered_case, "scale:region"),
    ],
)
def test_edit_inside_the_basis_is_reproduced(make, key):
    case = make()
    session = open_case(case)
    view = session.model
    term = session.terms[key]
    parameter, _ = split_key(key)
    rows, B, shift = _basis_edit(view, key, term)
    term.edited_log_effect = term.original_log_effect + shift
    target = term.edited_log_effect

    revised = lss_model_with_edits(view, session.terms)

    moved = (
        revised.predict_link(rows)[parameter].to_numpy()
        - case.model.predict_link(rows)[parameter].to_numpy()
    )
    sqrt_w = np.sqrt(_term_weights(term))
    bound = 1e3 * EPS * np.linalg.cond(sqrt_w[:, None] * B) * np.abs(target).max()
    assert bound < 0.1 * np.abs(shift).max()
    np.testing.assert_allclose(moved, shift, rtol=0.0, atol=bound)
    untouched = np.setdiff1d(
        np.arange(_coefficients(case.model).size), _edited_positions(view, key)
    )
    np.testing.assert_array_equal(
        _coefficients(revised)[untouched], _coefficients(case.model)[untouched]
    )


def test_the_level_of_a_curve_goes_to_the_intercept():
    case = lognormal_case()
    session = open_case(case)
    view = session.model
    key = "scale:age"
    term = session.terms[key]
    rows, B, _ = _basis_edit(view, key, term)
    term.edited_log_effect = term.original_log_effect + 0.3

    revised = lss_model_with_edits(view, session.terms)

    sqrt_w = np.sqrt(_term_weights(term))
    bound = 1e3 * EPS * np.linalg.cond(sqrt_w[:, None] * B) * np.abs(term.edited_log_effect).max()
    moved = revised.predict_link(rows)["scale"] - case.model.predict_link(rows)["scale"]
    np.testing.assert_allclose(moved, 0.3, rtol=0.0, atol=bound)
    intercept = view._fitted.layout.predictor("scale").intercept_index
    gain = _coefficients(revised)[intercept] - _coefficients(case.model)[intercept]
    np.testing.assert_allclose(gain, 0.3, rtol=0.0, atol=bound)


def test_categorical_edit_is_exact():
    case = lognormal_case()
    session = open_case(case)
    view = session.model
    term = session.terms["scale:region"]
    shift = np.array([0.3, -0.2, 0.1, 0.0, 0.25, -0.4, 0.15])  # the base level moves too
    term.edited_log_effect = term.original_log_effect + shift

    revised = lss_model_with_edits(view, session.terms)

    rows = _rows_with(view, "region", term.levels)
    after = revised.predict_link(rows)
    before = case.model.predict_link(rows)
    bound = 64 * EPS * np.abs(after["scale"]).max()
    np.testing.assert_allclose(after["scale"] - before["scale"], shift, rtol=0.0, atol=bound)
    np.testing.assert_array_equal(after["mean"], before["mean"])
    _, _, term_slice = _resolve_term(view._fitted, "scale", "region")
    np.testing.assert_allclose(
        _coefficients(revised)[term_slice],
        term.edited_log_effect[1:] - term.edited_log_effect[0],
        rtol=0.0,
        atol=bound,
    )


def test_spread_edit_leaves_the_mean_unchanged():
    case = lognormal_case()
    session = open_case(case)
    term = session.terms["scale:age"]
    unit = (term.x - term.x.min()) / np.ptp(term.x)
    term.edited_log_effect = term.original_log_effect + 0.3 * unit

    revised = lss_model_with_edits(session.model, session.terms)

    before = case.model.predict_parameters(case.X)
    after = revised.predict_parameters(case.X)
    np.testing.assert_array_equal(after["mean"], before["mean"])
    assert np.abs(after["scale"] - before["scale"]).max() > 0.01


@pytest.mark.parametrize("key", ["scale:region", "scale:age"])
def test_zero_exposure_level_projection_is_finite(key):
    case = lognormal_case()
    unseen = (case.X["region"] == "R6") | (case.X["age"] > 80.0)
    weights = np.where(unseen, 0.0, 1.0)
    session = edit(case.model, train_data=(case.X, case.y, weights, None))
    view = session.model
    term = session.terms[key]
    assert term.weights.min() == 0.0
    if term.levels is not None:
        rows = _rows_with(view, "region", term.levels)
        shift = np.linspace(0.1, -0.1, term.size)
    else:
        rows, B, shift = _basis_edit(view, key, term)
    term.edited_log_effect = term.original_log_effect + shift

    revised = lss_model_with_edits(view, session.terms)

    assert np.isfinite(_coefficients(revised)).all()
    after = revised.predict_link(rows)["scale"]
    moved = after - case.model.predict_link(rows)["scale"]
    if term.levels is not None:
        bound = 64 * EPS * np.abs(after).max()
    else:
        sqrt_w = np.sqrt(_term_weights(term))
        bound = (
            1e3 * EPS * np.linalg.cond(sqrt_w[:, None] * B) * np.abs(term.edited_log_effect).max()
        )
    np.testing.assert_allclose(moved, shift, rtol=0.0, atol=bound)


def test_projection_is_weighted_by_training_exposure():
    case = lognormal_case()
    weights = np.where(case.X["age"] > 80.0, 0.0, 1.0)
    session = edit(case.model, train_data=(case.X, case.y, weights, None))
    view = session.model
    term = session.terms["scale:age"]
    unrated = term.weights == 0.0
    assert unrated.any()
    term.edited_log_effect = term.original_log_effect + 0.5 * unrated

    revised = lss_model_with_edits(view, session.terms)

    rows, B, _ = _basis_edit(view, "scale:age", term)
    moved = (
        revised.predict_link(rows[~unrated])["scale"]
        - case.model.predict_link(rows[~unrated])["scale"]
    )
    # An edit on the unrated points reaches the rated ones only through the floor
    # `_term_weights` puts under their zero weight: with A = [1, B], the weighted
    # least-squares step is (A'WA)^-1 floor A_Z' e, and A'WA >= w_min sigma_min(A_R)^2.
    floor = _term_weights(term)[unrated].max()
    A = np.column_stack([np.ones(len(B)), B])
    rated_part, unrated_part = A[~unrated], A[unrated]
    sigma_min = np.linalg.svd(rated_part, compute_uv=False).min()
    bound = (
        floor
        * np.linalg.norm(rated_part, 2)
        * np.linalg.norm(unrated_part, 2)
        * np.linalg.norm(0.5 * np.ones(unrated.sum()))
        / (term.weights[~unrated].min() * sigma_min**2)
    )
    sqrt_w = np.sqrt(_term_weights(term))
    roundoff = (
        1e3 * EPS * np.linalg.cond(sqrt_w[:, None] * B) * np.abs(term.edited_log_effect).max()
    )
    assert bound < 1e-3
    assert np.abs(moved).max() <= bound + roundoff


def test_pinned_level_is_refused_unless_carried():
    case = lognormal_case()
    family = LogNormalLS()
    declared = cat("region", levels=[*REGIONS, "R9"])
    with pytest.warns(UserWarning, match="pinned to base"):
        model = SuperLSS(family, family.mean(s("age", k=6)), family.scale(declared)).fit_reml(
            case.X, case.y, outer="efs+newton"
        )
    session = edit(model, train_data=(case.X, case.y, None, None))
    view = session.model
    term = session.terms["scale:region"]
    assert list(view._specs["scale:region"]._pinned_levels) == ["R9"]
    term.edited_log_effect = term.original_log_effect + 0.2

    with pytest.raises(EditorValueError, match=r"R9"):
        lss_model_with_edits(view, session.terms)

    kept = lss_model_with_edits(view, session.terms, keep_pinned=True)
    rows = _rows_with(view, "region", term.levels)
    after = kept.predict_link(rows)["scale"]
    moved = after - model.predict_link(rows)["scale"]
    np.testing.assert_allclose(moved[:-1], 0.2, rtol=0.0, atol=64 * EPS * np.abs(after).max())
    assert moved.iloc[-1] == moved.iloc[0]  # the pinned level still reads as the base


def test_a_parameter_without_an_intercept_refuses_its_edits():
    case = lognormal_case()
    family = LogNormalLS()
    model = SuperLSS(
        family, family.mean(s("age", k=6)), family.scale(s("age", k=5), intercept=False)
    ).fit_reml(case.X, case.y, outer="efs+newton")
    session = edit(model, train_data=(case.X, case.y, None, None))
    term = session.terms["scale:age"]
    term.edited_log_effect = term.original_log_effect + 0.1

    with pytest.raises(EditorValueError, match="without an intercept"):
        lss_model_with_edits(session.model, session.terms)


def test_session_to_model_uses_offsets_by_parameter():
    case = nb_case()
    session = open_case(case)
    parameter = case.model.parameter_names_[0]
    term = session.terms[f"{parameter}:age"]
    unit = (term.x - term.x.min()) / np.ptp(term.x)
    term.edited_log_effect = term.original_log_effect + 0.25 * unit

    model = session.to_model()

    expected = lss_model_with_edits(session.model, session.terms)
    assert isinstance(model, SuperLSS)
    np.testing.assert_array_equal(
        model.predict_link(case.X, offsets=case.offsets),
        expected.predict_link(case.X, offsets=case.offsets),
    )
    fitted = model._model.fit_state.retained_rows
    np.testing.assert_array_equal(
        fitted.fitted_eta, model.predict_link(case.X, offsets=case.offsets).to_numpy()
    )
    explicit = session.to_model(X=case.X, y=case.y, offset=case.offsets)
    np.testing.assert_array_equal(
        explicit.predict_link(case.X, offsets=case.offsets),
        expected.predict_link(case.X, offsets=case.offsets),
    )
    request = session.capture_materialization_request()
    assert isinstance(materialize_edit_request(request), SuperLSS)
    assert request.dataset.offset.keys() == case.offsets.keys()
    refit_offsets = session._resolve_refit_data(None, None, None, None)[3]
    assert refit_offsets.keys() == case.offsets.keys()
    np.testing.assert_array_equal(refit_offsets["mean"], case.offsets["mean"])
