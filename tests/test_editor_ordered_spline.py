"""Handles, Contrib and Build for an OrderedCategorical with a spline basis.

The handles are the fitted spline's own coefficients, and moving one sets the
smooth levels to ``B(level positions) @ c``.  Every tolerance below is written
in the unit roundoff ``u = 2**-53`` and ``gamma_k = k u / (1 - k u)`` (Higham,
*Accuracy and Stability of Numerical Algorithms*, 2nd ed., 2002, sections 2.2,
3.1 and 3.5), times the magnitudes the computation actually combines.
"""

from __future__ import annotations

import json
import urllib.request

import numpy as np
import pandas as pd
import pytest

from superglm import OrderedCategorical, Piecewise, Spline, SuperGLM
from superglm.editor import EditorSession
from superglm.editor.controls import (
    ORDERED_SPLINE_GRID_STEPS,
    ORDERED_SPLINE_GROUPED,
    ORDERED_SPLINE_SHAPED,
    ORDERED_SPLINE_UNAVAILABLE,
)
from superglm.editor.payloads import session_payload

U = 2.0**-53
SMOOTH = ["1", "2", "3", "4", "5", "6"]
EFFECT = {"1": -0.30, "2": -0.18, "3": -0.05, "4": 0.06, "5": 0.15, "6": 0.20, "MISSING": 0.55}


def _gamma(count: int) -> float:
    return count * U / (1.0 - count * U)


def _fit(basis, *, specials=("MISSING",), seed=20261003):
    """A gaussian fit of one ordered band, with a special level when asked."""
    rng = np.random.default_rng(seed)
    labels = rng.choice(SMOOTH + list(specials), 900)
    X = pd.DataFrame({"band": labels})
    y = np.array([EFFECT[label] for label in labels]) + rng.normal(0.0, 0.15, 900)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={
            "band": OrderedCategorical(order=SMOOTH, specials=list(specials) or None, basis=basis)
        },
    )
    model.fit(X, y)
    return model, X


@pytest.fixture
def wide():
    """Eight basis columns over six levels: the levels do not fix the coefficients."""
    return _fit(Spline(kind="ps", k=8))


@pytest.fixture
def narrow():
    """Five basis columns over six levels, like the browser fixture's ``age_band``."""
    return _fit(Spline(kind="ps", k=5), specials=())


def _spline_parts(model):
    """The pieces an independent evaluation of the fitted curve needs."""
    spec = model._specs["band"]
    inner = spec._basis_spline
    beta = np.concatenate(
        [model.result.beta[g.sl] for g in model._groups if g.feature_name == "band"]
    )
    spline_beta, _ = spec._split_beta(beta)
    positions = np.array([spec._level_to_value[level] for level in spec._smooth_levels])
    basis = inner._basis_matrix(positions).toarray()
    base_row = inner._basis_matrix(np.array([spec._level_to_value[spec._base_level]])).toarray()[0]
    assert inner._R_inv is not None, "precondition: an SSP spline without a SCOP map"
    raw = inner._R_inv @ spline_beta
    fitted = raw - base_row @ raw
    weights = np.abs(inner._R_inv) @ np.abs(spline_beta)
    return basis, base_row, fitted, weights, spline_beta.size


def _curve_bound(basis, base_row, weights, p):
    """How far two float64 evaluations of the base-relative curve at a level can differ.

    ``fl(fl(b_i R) beta) - fl(fl(b_0 R) beta)`` against
    ``fl(b_i fl(R beta)) - fl(b_0 fl(R beta))``: each is within
    ``gamma_{K+p+3} (1 + |b_i|_1) (a_i + a_0)`` of the exact value, with
    ``a_i = |b_i| |R| |beta|``.
    """
    k = basis.shape[1]
    level = np.abs(basis) @ weights
    base = float(np.abs(base_row) @ weights)
    row_norm = np.maximum(1.0, np.sum(np.abs(basis), axis=1))
    return 4.0 * _gamma(2 * (k + p) + 6) * row_norm * (level + base)


def _least_squares_bound(basis, start, coefficients, effects):
    """Residual of the least-change solve when the edited levels are a spline of the basis.

    numpy's ``lstsq`` (LAPACK ``gelsd``) works by orthogonal transformations and
    is backward stable like Householder QR (Higham 2002, Thm 20.3), so on a
    consistent system its residual is within ``gamma`` of
    ``|B|_F |delta| + |r_0|``; the second term covers forming ``B @ start``,
    ``B @ c`` and the differences.
    """
    s, k = basis.shape
    correction = np.linalg.norm(coefficients - start)
    start_residual = np.linalg.norm(effects - basis @ start)
    solve = 4.0 * _gamma(4 * s * k) * (np.linalg.norm(basis) * correction + start_residual)
    forming = np.abs(basis) @ np.abs(coefficients) + np.abs(effects)
    return solve + 4.0 * _gamma(k + 1) * np.linalg.norm(forming)


def _smallest_singular_value(basis):
    """The smallest singular value ``lstsq`` keeps (numpy's default ``rcond``)."""
    values = np.linalg.svd(basis, compute_uv=False)
    return float(values[values > values[0] * max(basis.shape) * np.finfo(np.float64).eps][-1])


@pytest.mark.parametrize("fixture", ["wide", "narrow"])
def test_ordered_spline_handles_start_at_the_fitted_coefficients(fixture, request):
    # Mutation check: recovering the coefficients by least squares on the six
    # level points, as a numeric spline does on its grid, returns the
    # minimum-norm vector instead. On `wide` (eight columns) that misses the
    # fitted coefficients by O(0.1); on master `control_points` refuses the term.
    model, _ = request.getfixturevalue(fixture)
    session = EditorSession.from_model(model, terms=["band"])
    basis, base_row, fitted, weights, p = _spline_parts(model)
    geometry = session.ordered_spline("band")

    controls = session.control_points("band")

    # The session starts from the fit and corrects it by the minimum-norm
    # solution of `B delta = e - B c_fit`, whose right-hand side is the curve's
    # rounding, at most the curve bound in each entry.
    bound = _curve_bound(basis, base_row, weights, p)
    drift = np.linalg.norm(bound) / _smallest_singular_value(basis)
    evaluation = 4.0 * _gamma(2 * (basis.shape[1] + p) + 6) * (weights + float(base_row @ weights))
    live = geometry.live
    np.testing.assert_array_less(
        np.abs(np.asarray(controls["build_log_effect"]) - fitted[live]),
        drift + evaluation[live] + U,
    )
    assert np.all(np.diff(controls["x"]) >= 0.0)
    assert controls["x"][0] >= 0.0 and controls["x"][-1] <= len(SMOOTH) - 1.0


@pytest.mark.parametrize("fixture", ["wide", "narrow"])
def test_spline_view_reproduces_the_fitted_level_effects(fixture, request):
    model, _ = request.getfixturevalue(fixture)
    session = EditorSession.from_model(model, terms=["band"])
    basis, base_row, _, weights, p = _spline_parts(model)
    effects = session.terms["band"].original_log_effect[: len(SMOOTH)]

    view = session_payload(session)["band"]["spline_view"]

    steps = ORDERED_SPLINE_GRID_STEPS
    assert view["available"] is True
    assert view["fits_levels"] is True
    assert len(view["x"]) == steps * (len(SMOOTH) - 1) + 1
    assert view["x"][::steps] == [float(i) for i in range(len(SMOOTH))]
    assert view["level_indices"] == list(range(len(SMOOTH)))
    # exp(curve) at each level against exp(level effect): the curve is within
    # the curve bound of the effect, and each exp rounds within 2u (numpy's exp
    # is within one ulp), so by the mean value theorem the relativities differ
    # by at most exp(e) (2 bound + 8u) while the bound is far below 1.
    expected = np.exp(effects)
    tolerance = expected * (2.0 * _curve_bound(basis, base_row, weights, p) + 8.0 * U)
    for curve in (view["y"], view["original_y"]):
        at_levels = np.asarray(curve)[::steps]
        np.testing.assert_array_less(np.abs(at_levels - expected), tolerance + U)


def test_moving_a_handle_sets_every_smooth_level_to_the_spline(wide):
    model, _ = wide
    session = EditorSession.from_model(model, terms=["band"])
    term = session.terms["band"]
    basis, *_ = _spline_parts(model)
    before = term.edited_log_effect.copy()
    controls = session.control_points("band")
    handle = controls["x"].size // 2
    target = float(controls["log_effect"][handle] + 0.3)

    session.move_control_point("band", handle, target)

    record = session.history[-1]
    moved = np.asarray(record.params["coefficients"])
    assert moved[record.params["basis_index"]] == target
    # Two float64 evaluations of the same dot products b_i . c: each within
    # gamma_K |b_i| |c| of the exact value.
    smooth = term.edited_log_effect[: len(SMOOTH)]
    bound = 2.0 * _gamma(basis.shape[1]) * (np.abs(basis) @ np.abs(moved))
    np.testing.assert_array_less(np.abs(smooth - basis @ moved), bound + U)
    # The special level has no place on the spline: its value is not written.
    np.testing.assert_array_equal(term.edited_log_effect[len(SMOOTH) :], before[len(SMOOTH) :])
    # The next request starts from the moved coefficients, so the handle stays
    # where it was dropped. Mutation check: least change from the FIT instead
    # moves it by 0.3 times the row-space projection's diagonal, O(0.1) here.
    after = session.control_points("band")
    reproduce = np.linalg.norm(bound) / _smallest_singular_value(basis)
    assert abs(after["log_effect"][handle] - target) <= reproduce + 2.0 * U * abs(target)


def test_a_level_edit_keeps_the_spline_through_the_levels_and_a_handle_keeps_the_edit(wide):
    # Mutation check: restarting from the fitted coefficients after a level
    # edit draws a curve that misses level "3" by the 0.2 shift, and the handle
    # move below would then wipe the shift out.
    model, _ = wide
    session = EditorSession.from_model(model, terms=["band"])
    term = session.terms["band"]
    basis, *_ = _spline_parts(model)
    start = session.ordered_spline("band").fitted
    session.select_levels("band", ["3"])
    session.shift("band", 0.2)
    effects = term.edited_log_effect[: len(SMOOTH)].copy()

    view = session_payload(session)["band"]["spline_view"]
    geometry = session.ordered_spline("band")
    current = session.ordered_spline_coefficients("band", geometry)

    assert view["fits_levels"] is True
    bound = _least_squares_bound(basis, start, current, effects)
    assert np.linalg.norm(basis @ current - effects) <= bound

    controls = session.control_points("band")
    handle = controls["x"].size // 2
    column = int(geometry.live[controls["basis_index"][handle]])
    outside = np.flatnonzero(basis[:, column] == 0.0)
    assert outside.size, "precondition: the moved basis function misses some level"
    session.move_control_point("band", handle, float(controls["log_effect"][handle] + 0.3))
    # Off the moved column's support each level is b_i . c again: the edited
    # value up to the least-change residual and one more evaluation.
    moved_bound = bound + 2.0 * _gamma(basis.shape[1]) * float(
        np.max(np.abs(basis) @ np.abs(current))
    )
    np.testing.assert_array_less(
        np.abs(term.edited_log_effect[outside] - effects[outside]), moved_bound + U
    )


def test_a_level_edit_the_basis_cannot_follow_joins_the_levels_until_a_handle_move(narrow):
    # Mutation check: `spline_fits_levels` answering True whatever the edit
    # has the chart draw the least-change spline, which misses the shifted
    # level; answering False after a handle move joins levels that lie on it.
    model, _ = narrow
    session = EditorSession.from_model(model, terms=["band"])
    term = session.terms["band"]
    basis, *_ = _spline_parts(model)
    assert np.linalg.matrix_rank(basis) < len(SMOOTH), "precondition: K < S at the levels"
    start = session.ordered_spline("band").fitted
    session.select_levels("band", ["3"])
    session.shift("band", 0.2)
    effects = term.edited_log_effect[: len(SMOOTH)].copy()
    geometry = session.ordered_spline("band")
    current = session.ordered_spline_coefficients("band", geometry)

    # Were the edited levels a spline of the basis, the least-change residual
    # would be within this bound; it is not, so no spline passes through them.
    residual = np.linalg.norm(basis @ current - effects)
    assert residual > _least_squares_bound(basis, start, current, effects)
    assert session_payload(session)["band"]["spline_view"]["fits_levels"] is False

    # A handle move writes every smooth level as B(level positions) @ c.
    controls = session.control_points("band")
    session.move_control_point("band", 0, float(controls["log_effect"][0] + 0.1))
    assert session_payload(session)["band"]["spline_view"]["fits_levels"] is True

    # Undo takes the move back: the shift is again the latest level edit.
    session.undo()
    assert session_payload(session)["band"]["spline_view"]["fits_levels"] is False
    # Undoing the shift restores the fitted levels, which lie on the fit.
    session.undo()
    np.testing.assert_array_equal(term.edited_log_effect, term.original_log_effect)
    assert session_payload(session)["band"]["spline_view"]["fits_levels"] is True


def test_handles_are_off_with_a_reason_once_levels_are_grouped(wide):
    model, _ = wide
    session = EditorSession.from_model(model, terms=["band"])
    session.select_levels("band", ["2", "3"])
    session.replace_with_collapsed_levels("band", method="fit")

    payload = session_payload(session)["band"]

    assert payload["controls"] is None
    assert payload["spline_view"]["available"] is False
    assert payload["spline_view"]["reason"] == ORDERED_SPLINE_GROUPED
    with pytest.raises(TypeError, match="grouped"):
        session.control_points("band")


def test_handles_are_off_with_a_reason_once_a_band_is_shaped(wide):
    model, _ = wide
    session = EditorSession.from_model(model, terms=["band"])
    session.replace_with_shaped_range("band", lo="2", hi="4", degree=1, method="fit")

    payload = session_payload(session)["band"]

    assert payload["controls"] is None
    assert payload["spline_view"]["reason"] == ORDERED_SPLINE_SHAPED
    with pytest.raises(TypeError, match="shaped"):
        session.move_control_point("band", 0, 0.0)


def test_handles_are_off_when_the_certification_bound_overflows(wide, monkeypatch):
    """The bound |b| |M| |beta| overflows to inf, as on an ill-scaled fit where M beta cancels.

    Every discrepancy between the spline view and the fitted effects passed
    ``<= inf``, so handles were offered for a curve nothing certified. Here
    the view is also moved off the effects; a bound that is not finite
    certifies nothing.
    """
    import superglm.editor.controls as controls

    model, _ = wide
    session = EditorSession.from_model(model, terms=["band"])
    raw_map = controls._raw_coefficient_map
    monkeypatch.setattr(
        controls, "_raw_coefficient_map", lambda inner, width: 2.0 * raw_map(inner, width)
    )
    monkeypatch.setattr(
        controls,
        "_certification_bound",
        lambda level_basis, *args: np.full(level_basis.shape[0], np.inf),
    )

    assert controls.ordered_spline_geometry(model, session.terms["band"]) == (
        ORDERED_SPLINE_UNAVAILABLE
    )


def test_handles_survive_effects_of_subnormal_size():
    """Responses near 1e-310: the effects are subnormal, and so are their rounding errors.

    Under gradual underflow a product also carries an absolute error, up to
    half the subnormal spacing (Demmel 1984). The certificate was purely
    relative, so it rounded to zero at every level, the view and the effects
    differed by a subnormal spacing or two, and handles were refused for a
    fit that ordinary-scale responses give handles.
    """
    levels = ["0", "1", "2", "3", "4", "5"]
    X = pd.DataFrame({"band": np.tile(levels, 30)})
    y = 1e-310 * np.tile([-0.3, -0.18, -0.05, 0.06, 0.15, 0.2], 30)
    band = OrderedCategorical(order=levels, basis=Spline(kind="ps", k=8))
    model = SuperGLM(family="gaussian", selection_penalty=0.0, features={"band": band})
    model.fit(X, y)
    session = EditorSession.from_model(model, terms=["band"])

    assert session.ordered_spline("band") is not None
    assert not isinstance(session.ordered_spline("band"), str)
    assert len(session.control_points("band")["x"]) >= 3


def test_handles_need_no_underflow_allowance_from_numpy(wide):
    """The certificate's subnormal allowance underflows on purpose.

    Under np.errstate(under="raise") that multiplication raised
    FloatingPointError, so an ordinary-scale fit lost its handles to a NumPy
    setting the old bound never tripped.
    """
    model, _ = wide
    session = EditorSession.from_model(model, terms=["band"])

    with np.errstate(under="raise"):
        points = session.control_points("band")

    assert len(points["x"]) >= 3


def test_an_ordered_term_without_a_spline_basis_gets_no_spline_view():
    model, _ = _fit(Piecewise(breaks=["3"]), specials=())
    session = EditorSession.from_model(model, terms=["band"])

    payload = session_payload(session)["band"]

    assert payload["spline_view"] is None
    assert payload["controls"] is None
    with pytest.raises(TypeError, match="control handles"):
        session.control_points("band")


def test_to_model_moves_predictions_by_exactly_the_handle_edit(wide):
    # The edited levels are what `to_model` consumes (#453): the export moves
    # each smooth level's rows by its level's change, and the special's by none.
    model, X = wide
    session = EditorSession.from_model(model, terms=["band"])
    term = session.terms["band"]
    before = term.edited_log_effect.copy()
    controls = session.control_points("band")
    handle = controls["x"].size // 2
    session.move_control_point("band", handle, float(controls["log_effect"][handle] + 0.3))
    change = term.edited_log_effect - before

    edited = session.to_model()

    delta = np.asarray(edited._predict_eta_exact(X)) - np.asarray(model._predict_eta_exact(X))
    level_of_row = {label: i for i, label in enumerate(term.levels)}
    wanted = change[[level_of_row[str(label)] for label in X["band"]]]
    np.testing.assert_array_less(np.abs(delta - wanted), _projection_bound(model, edited, term, X))


def _projection_bound(model, edited, term, X):
    """Per row, how far the exported prediction change can sit from the edit.

    `_apply_ordered_spline_term` solves the weighted least-squares problem
    ``W^1/2 [1, T] x = W^1/2 t`` on the smooth levels; the targets are a spline
    of the basis, so the system is consistent and the backward-stable solve
    leaves a residual within ``gamma_{4 S n} (|A|_F |x| + |b|)`` (Higham 2002,
    Thm 20.3), divided by the smallest root weight to bound one level. Each
    row's linear predictor then rounds within ``gamma_{p+2}`` of
    ``|a| + |T_row| |beta|``, once per model.
    """
    spec = model._specs["band"]
    groups = [g for g in model._groups if g.feature_name == "band"]
    old_beta = np.concatenate([model.result.beta[g.sl] for g in groups])
    new_beta = np.concatenate([edited.result.beta[g.sl] for g in groups])
    design = np.asarray(spec.transform(X["band"].to_numpy()))
    rows = np.abs(design) @ np.abs(old_beta) + np.abs(design) @ np.abs(new_beta)
    evaluation = _gamma(design.shape[1] + 2) * (
        abs(model.result.intercept) + abs(edited.result.intercept) + rows
    )
    smooth = np.asarray(spec.transform(np.array(spec._smooth_levels, dtype=object)))
    width = spec._split_beta(np.zeros(design.shape[1]))[0].size
    weights = np.maximum(term.weights[: len(SMOOTH)], 1e-12)
    root = np.sqrt(weights)
    A = root[:, None] * np.column_stack([np.ones(len(SMOOTH)), smooth[:, :width]])
    # The solve's intercept is the change less the base shift put back after it.
    shift = spec._base_log_effect(old_beta)
    x = np.concatenate(
        [[edited.result.intercept - model.result.intercept - shift], new_beta[:width]]
    )
    b = root * term.edited_log_effect[: len(SMOOTH)]
    solve = 4.0 * _gamma(4 * A.size) * (np.linalg.norm(A) * np.linalg.norm(x) + np.linalg.norm(b))
    return solve / float(np.min(root)) + 2.0 * evaluation + U


def _post_json(widget, path: str, payload: dict) -> dict:
    request = urllib.request.Request(
        f"{widget.url}{path}",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "X-SuperGLM-Editor-Token": widget._token},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def test_widget_moves_an_ordered_spline_handle_over_http(wide):
    model, _ = wide
    session = EditorSession.from_model(model, terms=["band"])
    widget = session.widget()
    try:
        state = _post_json(widget, "/control_count", {"term": "band", "count": 4})
        band = state["terms"]["band"]
        controls = band["controls"]
        assert controls["count"] == 4
        assert len(controls["grid_x"]) == len(band["spline_view"]["x"])
        assert all(len(row) == len(controls["grid_x"]) for row in controls["build_basis"])
        assert all(len(row) == band["n_points"] for row in controls["basis"])
        assert len(controls["build_log_effect"]) == len(controls["build_basis"])

        before = session.terms["band"].edited_log_effect.copy()
        target = float(np.exp(controls["log_effect"][1] + 0.25))
        state = _post_json(widget, "/control", {"term": "band", "handle_index": 1, "value": target})
    finally:
        widget.close()

    after = session.terms["band"].edited_log_effect
    assert np.max(np.abs(after - before)) > 0.0
    assert session.history[-1].params["basis"] == "ordered_spline"
    assert state["terms"]["band"]["controls"]["count"] == 4
