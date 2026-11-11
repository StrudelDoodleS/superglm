"""SuperLSS shape rules in the editor: named on each term, and every edit kept inside them."""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize_scalar

import superglm.distributional.terms as distributional_terms
import superglm.editor.lss.view as lss_view
import superglm.solvers.irls_direct._qp as qp_retry
from superglm import (
    Constraint,
    GammaLS,
    NegativeBinomialLS,
    Numeric,
    OrderedCategorical,
    Poisson,
    Spline,
    SuperGLM,
    SuperLSS,
    cat,
    s,
    term,
)
from superglm.editor import EditorSession, edit
from superglm.editor.errors import EditorValueError
from superglm.editor.lss.apply import lss_model_with_edits
from superglm.editor.payloads import session_payload
from superglm.features._spline_runtime import fit_support
from tests._lss_editor_fixtures import (
    _SEEDS,
    N_ROWS,
    FitCase,
    _gamma_rows,
    _nb_rows,
    open_case,
    ruled_case,
)

# A grid curve meets its rule to the rounding of the projection's solve on unit-scale effects.
_TOL = 1e-9
_CAP = (np.log(0.8), np.log(1.25))


def _increasing(values) -> bool:
    return bool(np.min(np.diff(values)) >= -_TOL)


def _convex(values, x) -> bool:
    slopes = np.diff(values) / np.diff(x)
    return bool(np.min(np.diff(slopes)) >= -_TOL * np.max(np.abs(slopes), initial=1.0))


def _capped(session, key="scale:region") -> bool:
    term = session.terms[key]
    base = term.levels.index(str(session.model._specs[key]._base_level))
    relative = term.edited_log_effect - term.edited_log_effect[base]
    return bool(np.all(relative >= _CAP[0] - _TOL) and np.all(relative <= _CAP[1] + _TOL))


def _model_curve(model, column, parameter, x):
    case = ruled_case()
    rows = pd.concat([case.X.iloc[[0]]] * len(x), ignore_index=True)
    rows[column] = x
    return model.predict_link(rows)[parameter].to_numpy()


def test_each_term_names_its_rules_and_that_edits_keep_to_them():
    payload = session_payload(open_case(ruled_case()))

    assert payload["mean:power"]["rules"] == {"text": "increasing", "fit": True, "kept": True}
    assert payload["scale:age"]["rules"] == {"text": "convex", "fit": True, "kept": True}
    assert payload["scale:region"]["rules"] == {
        "text": "cap(0.8, 1.25)",
        "fit": True,
        "kept": True,
    }
    assert payload["mean:age"]["rules"] is None
    assert payload["mean:region"]["rules"] is None


def test_a_superglm_term_names_its_rules_without_keeping_edits_to_them():
    rng = np.random.default_rng(3)
    X = pd.DataFrame({"x": rng.uniform(0.0, 10.0, 400), "z": rng.uniform(0.0, 10.0, 400)})
    y = rng.poisson(np.exp(0.1 * X["x"])).astype(float)
    model = SuperGLM(
        family=Poisson(),
        features={
            "x": Spline(kind="cr", k=6, constraint=Constraint.fit.increasing),
            "z": Spline(kind="cr", k=6, constraint=Constraint.postfit.decreasing),
        },
    ).fit(X, y)
    payload = session_payload(EditorSession.from_model(model, terms=["x", "z"]))

    assert payload["x"]["rules"] == {"text": "increasing", "fit": True, "kept": False}
    assert payload["z"]["rules"] == {"text": "decreasing", "fit": False, "kept": False}


def test_a_drag_against_an_increasing_rule_keeps_the_curve_increasing():
    session = open_case(ruled_case())
    term = session.terms["mean:power"]
    tail = np.flatnonzero(term.x > 10.0)
    before = term.edited_log_effect.copy()

    session.set_values("mean:power", tail, term.edited_log_effect[tail] - 0.4)

    curve = session.terms["mean:power"].edited_log_effect
    assert _increasing(curve)
    # The edit moved the curve: the tail came down as far as the rule lets it.
    assert curve[-1] < before[-1] - 0.05
    # The model the editor writes is the curve it draws, inside the rule between grid points too.
    model = session.to_model()
    shown = _model_curve(model, "power", "mean", term.x)
    np.testing.assert_allclose(shown - shown[0], curve - curve[0], rtol=0.0, atol=1e-9)
    fine = _model_curve(model, "power", "mean", np.linspace(term.x[0], term.x[-1], 801))
    assert _increasing(fine)


def test_a_rule_kept_edit_is_one_step_undo_takes_back():
    session = open_case(ruled_case())
    term = session.terms["mean:power"]
    before = term.edited_log_effect.copy()
    session.select_indices("mean:power", np.flatnonzero(term.x > 10.0))

    session.shift("mean:power", -0.4)

    record = session.history[-1]
    assert record.operation == "shift" and len(session.history) == 1
    np.testing.assert_array_equal(record.indices, np.arange(term.size))
    # History still names the stretch the user edited.
    assert record.params["lo"] > 10.0
    session.undo()
    np.testing.assert_array_equal(session.terms["mean:power"].edited_log_effect, before)
    session.redo()
    assert _increasing(session.terms["mean:power"].edited_log_effect)


def test_an_edit_the_rule_allows_is_kept_as_drawn():
    session = open_case(ruled_case())
    term = session.terms["mean:power"]
    before = term.edited_log_effect.copy()
    session.select_indices("mean:power", np.arange(term.size))

    session.shift("mean:power", 0.1)

    np.testing.assert_allclose(
        session.terms["mean:power"].edited_log_effect, before + 0.1, rtol=0.0, atol=1e-9
    )


def test_an_edit_of_a_spread_curve_keeps_its_convex_rule():
    session = open_case(ruled_case())
    term = session.terms["scale:age"]
    middle = np.flatnonzero((term.x > 45.0) & (term.x < 60.0))

    session.set_values("scale:age", middle, term.edited_log_effect[middle] + 0.6)

    curve = session.terms["scale:age"].edited_log_effect
    assert _convex(curve, term.x)
    fine = np.linspace(term.x[0], term.x[-1], 801)
    assert _convex(_model_curve(session.to_model(), "age", "scale", fine), fine)


def test_a_level_set_past_its_cap_is_kept_at_the_cap():
    session = open_case(ruled_case())
    term = session.terms["scale:region"]
    base = term.levels.index(str(session.model._specs["scale:region"]._base_level))
    level = (base + 1) % term.size

    session.set_values("scale:region", [level], [term.edited_log_effect[level] + 2.0])

    assert _capped(session)
    curve = session.terms["scale:region"].edited_log_effect
    assert curve[level] - curve[base] == pytest.approx(_CAP[1], abs=1e-9)
    # The edit stops at the cap and moves nothing else: not the base, not another level.
    untouched = np.arange(term.size) != level
    np.testing.assert_allclose(
        curve[untouched], term.original_log_effect[untouched], rtol=0.0, atol=1e-12
    )
    # Nor does the exported model re-rate a row of an untouched region.
    case = ruled_case()
    fitted = case.model.predict_link(case.X)["scale"].to_numpy()
    exported = session.to_model().predict_link(case.X)["scale"].to_numpy()
    others = (case.X["region"].astype(str) != term.levels[level]).to_numpy()
    np.testing.assert_allclose(exported[others], fitted[others], rtol=0.0, atol=1e-9)


def test_a_partial_reset_keeps_the_rule():
    session = open_case(ruled_case())
    term = session.terms["mean:power"]
    session.select_indices("mean:power", np.flatnonzero(term.x < 6.0))
    session.shift("mean:power", 0.5)
    # Putting back part of the raised start leaves a dip the rule does not allow.
    session.select_indices("mean:power", np.flatnonzero(term.x < 5.0))

    session.reset("mean:power")

    assert _increasing(session.terms["mean:power"].edited_log_effect)
    assert session.history[-1].operation == "reset"


def test_writing_a_curve_against_the_rule_projects_it_inside():
    session = open_case(ruled_case())
    term = session.terms["mean:power"].copy()
    term.edited_log_effect = term.original_log_effect.copy()
    term.edited_log_effect[term.x > 10.0] -= 0.4

    model = lss_model_with_edits(session.model, {term.name: term})

    fine = np.linspace(term.x[0], term.x[-1], 801)
    assert _increasing(_model_curve(model, "power", "mean", fine))


@pytest.mark.parametrize("key", ["mean:age", "scale:age"])
def test_shaped_ranges_are_offered_on_a_superlss_spline_without_a_rule(key):
    session = open_case(ruled_case())
    shape = session_payload(session)[key]["shape"]

    if key == "mean:age":
        assert shape["available"] is True and shape["reason"] is None
        assert shape["support"] is not None
        assert len(shape["support"]["below"]) == session.terms[key].size
    else:
        assert shape["available"] is False
        assert shape["reason"] == "Remove the term's shape constraint to add shaped ranges."


def test_a_shaped_range_on_a_superlss_term_refits_into_force():
    session = open_case(ruled_case())

    step = session.stage_structural("shape", "mean:age", {"lo": 70.0, "hi": 90.0, "degree": 1})
    session.refit_pending()

    assert step.label.startswith("Line 70")
    ranges = session_payload(session)["mean:age"]["shape"]["ranges"]
    assert [(r["degree"], r["label"]) for r in ranges] == [(1, "Line")]
    # The other terms keep their rules through the refit.
    assert _increasing(session.terms["mean:power"].edited_log_effect)


def _flat_power_session():
    """The ruled case with its increasing mean curve levelled flat, where every row ``A b`` is 0."""
    session = open_case(ruled_case())
    term = session.terms["mean:power"]
    session.select_indices("mean:power", np.arange(term.size))
    session.weighted_average("mean:power")
    assert np.ptp(session.terms["mean:power"].edited_log_effect) < 1e-12
    return session


@pytest.mark.parametrize("delta", [0.001, 0.1, 0.3, -0.001, -0.1, -0.3])
def test_a_whole_curve_shift_of_a_flat_ruled_curve_is_kept_as_drawn(delta):
    session = _flat_power_session()
    before = session.terms["mean:power"].edited_log_effect.copy()

    session.shift("mean:power", delta)

    term = session.terms["mean:power"]
    np.testing.assert_allclose(term.edited_log_effect, before + delta, rtol=0.0, atol=1e-9)
    # The exported model takes the drawn curve, flat and moved by ``delta``.
    fitted = _model_curve(session.model.lss, "power", "mean", term.x)
    shown = _model_curve(session.to_model(), "power", "mean", term.x)
    np.testing.assert_allclose(
        shown - fitted, term.edited_log_effect - term.original_log_effect, rtol=0.0, atol=1e-9
    )


def test_an_edit_whose_nearest_curve_is_flat_is_kept_and_exported():
    session = open_case(ruled_case())
    term = session.terms["mean:power"]
    session.select_indices("mean:power", np.arange(term.size))

    # "Make decreasing" on an increasing rule: the nearest curve the rule allows is flat.
    session.isotonic("mean:power", "decreasing")

    curve = session.terms["mean:power"].edited_log_effect
    assert np.ptp(curve) < 1e-9
    np.testing.assert_allclose(
        _model_curve(session.to_model(), "power", "mean", term.x)
        - _model_curve(session.model.lss, "power", "mean", term.x),
        curve - term.original_log_effect,
        rtol=0.0,
        atol=1e-9,
    )


def test_an_edit_the_rules_take_back_entirely_records_nothing_and_says_why():
    session = open_case(ruled_case())
    term = session.terms["scale:region"]
    base = term.levels.index(str(session.model._specs["scale:region"]._base_level))
    relative = term.edited_log_effect - term.edited_log_effect[base]
    at_cap = int(np.argmax(relative))
    assert relative[at_cap] == pytest.approx(_CAP[1], abs=1e-9)
    revision = session.model_revision

    # Raising a level that already sits at its cap: the rule puts it straight back, and says so.
    with pytest.raises(EditorValueError, match=r"changed nothing: its rules \(cap\(0.8, 1.25\)\)"):
        session.set_values("scale:region", [at_cap], [term.edited_log_effect[at_cap] + 0.5])

    assert session.history == []
    assert session.edited_terms() == []
    assert session.model_revision == revision
    np.testing.assert_array_equal(
        session.terms["scale:region"].edited_log_effect, term.original_log_effect
    )


@functools.cache
def _many_levels_case() -> FitCase:
    """A gamma mean over 250 equally weighted levels, capped at 0.8-1.25 against the first."""
    levels = [f"L{i:03d}" for i in range(250)]
    X = pd.DataFrame({"g": pd.Categorical(np.repeat(levels, 8), categories=levels)})
    y = np.random.default_rng(0).gamma(5.0, 0.2, len(X))
    family = GammaLS()
    model = SuperLSS(
        family,
        family.mean(cat("g", base=levels[0], constraint=Constraint.fit.cap(0.8, 1.25))),
        family.scale(),
    ).fit(X, y)
    return FitCase(model, X, y, None, None)


def test_a_projection_over_many_capped_levels_reaches_the_nearest_allowed_curve():
    case = _many_levels_case()
    session = open_case(case)
    target = np.random.default_rng(621).uniform(-0.2, 0.2, 250)
    target[0] = 0.05
    target[1:81] = 2.0
    term = session.terms["mean:g"].copy()
    term.edited_log_effect = target.copy()

    # Every level edited, the base too: the nearest curve is the base's level ``a`` plus each
    # level's target clipped to the cap, for the ``a`` nearest the targets (equal weights).
    lo, hi = _CAP

    def allowed(a):
        return a + np.r_[0.0, np.clip(target[1:] - a, lo, hi)]

    a = minimize_scalar(
        lambda a: np.sum((allowed(a) - target) ** 2),
        bounds=(-3.0, 3.0),
        method="bounded",
        options={"xatol": 1e-12},
    ).x
    rows = pd.DataFrame({"g": pd.Categorical(term.levels, categories=term.levels)})
    fitted = case.model.predict_link(rows)["mean"].to_numpy()
    exported = lss_model_with_edits(session.model, {term.name: term}).predict_link(rows)["mean"]
    np.testing.assert_allclose(
        exported.to_numpy() - fitted, allowed(a) - term.original_log_effect, rtol=0.0, atol=1e-6
    )
    # The session keeps the same edit on the same curve.
    session.set_values("mean:g", np.arange(250), target)
    np.testing.assert_allclose(
        session.terms["mean:g"].edited_log_effect, allowed(a), rtol=0.0, atol=1e-6
    )


def test_a_projection_the_qp_does_not_certify_is_refused(monkeypatch):
    session = open_case(ruled_case())
    term = session.terms["mean:power"]
    before = term.edited_log_effect.copy()
    solve = qp_retry.solve_constrained_qp

    def stops_short(*args, **kwargs):
        result = solve(*args, **kwargs)
        return type(result)(**{**vars(result), "converged": False})

    monkeypatch.setattr(qp_retry, "solve_constrained_qp", stops_short)
    tail = np.flatnonzero(term.x > 10.0)

    with pytest.raises(EditorValueError, match="could not be confirmed"):
        session.set_values("mean:power", tail, term.edited_log_effect[tail] - 0.4)

    assert session.history == []
    np.testing.assert_array_equal(session.terms["mean:power"].edited_log_effect, before)


def test_opening_reads_no_reference_and_a_shown_term_edits_without_reading_rows(monkeypatch):
    reads = []
    original = lss_view._reference_value
    monkeypatch.setattr(
        lss_view, "_reference_value", lambda *a, **k: reads.append(a[-1]) or original(*a, **k)
    )
    session = open_case(ruled_case())
    # Opening builds no reference row; only a term the page shows builds what its edit reads.
    assert reads == []
    monkeypatch.setattr(
        distributional_terms, "_sweep", lambda *a, **k: pytest.fail("a level sweep for one row")
    )
    session._prepare_edit("mean:power")
    session._prepare_edit("scale:region")
    built = len(reads)
    assert built > 0
    monkeypatch.setattr(
        lss_view, "_reference_design", lambda *a, **k: pytest.fail("built on the first edit")
    )
    term = session.terms["mean:power"]
    tail = np.flatnonzero(term.x > 10.0)

    session.set_values("mean:power", tail, term.edited_log_effect[tail] - 0.4)
    session.set_values("scale:region", [1], [session.terms["scale:region"].edited_log_effect[1]])

    assert len(reads) == built


def test_opening_a_model_with_a_linear_term_reads_no_reference(monkeypatch):
    # A Numeric term's anchor is read at opening; it reads its own column, not the row.
    X, y, _ = _gamma_rows(_SEEDS[GammaLS], N_ROWS)
    family = GammaLS()
    model = SuperLSS(
        family,
        family.mean(term("age", Numeric()), s("power", k=6, constraint=Constraint.fit.increasing)),
        family.scale(cat("region")),
    ).fit_reml(X, y)
    reads = []
    original = lss_view._reference_value
    monkeypatch.setattr(
        lss_view, "_reference_value", lambda *a, **k: reads.append(a[-1]) or original(*a, **k)
    )
    session = open_case(FitCase(model, X, y, None, None))

    assert reads == []
    assert np.isfinite(session.terms["mean:age"].edited_log_effect).all()


@pytest.mark.parametrize(
    ("case", "key"),
    [("ruled", "scale:region"), ("ruled", "mean:region"), ("ordered", "mean:band")],
)
def test_a_reference_design_from_the_reference_row_is_the_sweep_s(case, key):
    view = open_case(ruled_case() if case == "ruled" else _ordered_case()).model
    parameter, column = key.split(":")
    _, state, term_slice = distributional_terms._resolve_term(view._fitted, parameter, column)
    frame = distributional_terms.as_eager_frame(view.X_train)
    swept = distributional_terms._reference_design(
        view._fitted, frame, parameter, column, state, term_slice
    )
    np.testing.assert_allclose(view.reference_design(key), swept, rtol=0.0, atol=1e-14)


@functools.cache
def _ruled_theta_session_case() -> FitCase:
    X, y, offsets = _nb_rows(_SEEDS[NegativeBinomialLS], N_ROWS)
    family = NegativeBinomialLS()
    model = SuperLSS(
        family,
        family.mean(s("age", k=6), offset=offsets["mean"]),
        family.theta(
            s("age", kind="cr", k=6, constraint=Constraint.fit.decreasing),
            cat("region", constraint=Constraint.fit.cap(0.7, 1.5)),
            s("power", kind="cr", k=6, constraint=Constraint.fit.convex),
        ),
    ).fit_reml(X, y, outer="efs+newton")
    return FitCase(model, X, y, None, offsets)


def test_a_theta_term_names_its_rule_on_the_overdispersion_it_draws():
    payload = session_payload(open_case(_ruled_theta_session_case()))

    # The chart draws 1/θ, which rises where θ falls and is capped at 1/1.5 to 1/0.7.
    assert payload["theta:age"]["rules"]["text"] == "increasing"
    assert np.min(np.diff(payload["theta:age"]["y"])) >= -1e-9
    assert payload["theta:region"]["rules"]["text"] == "cap(0.667, 1.43)"
    # A rule holds on log θ, and log(1/θ) = -log θ: convex in θ reads concave on the log of the
    # 1/θ the chart draws, as every rule reads on the log scale.
    assert payload["theta:power"]["rules"]["text"] == "concave"
    assert np.max(np.diff(np.log(payload["theta:power"]["y"]), 2)) <= _TOL


def test_shape_support_counts_the_bins_a_per_column_bin_count_gives():
    rng = np.random.default_rng(5)
    X = pd.DataFrame({"x": np.linspace(0.0, 1.0, 1001)})
    y = rng.gamma(5.0, np.exp(0.3 * X["x"].to_numpy()) / 5.0)
    family = GammaLS()
    model = SuperLSS(
        family, family.mean(s("x", k=8)), family.scale(), discrete=True, n_bins={"x": 512}
    ).fit_reml(X, y)
    session = open_case(FitCase(model, X, y, None, None))

    support = session_payload(session)["mean:x"]["shape"]["support"]

    # The refit bins ``x`` into 512, as its own ``n_bins`` entry says, not the default 256.
    assert support["through"][-1] == fit_support(X["x"].to_numpy(), 512).size


def _level_curve(model, key, levels):
    """The exported model's link-scale effect of ``key`` at each level, from the first level."""
    parameter, column = key.split(":")
    case = ruled_case()
    rows = pd.concat([case.X.iloc[[0]]] * len(levels), ignore_index=True)
    rows[column] = list(levels)
    values = model.predict_link(rows)[parameter].to_numpy()
    return values - values[0]


def test_a_reordered_capped_term_still_edits_resets_and_exports():
    session = open_case(ruled_case())
    key = "scale:region"
    term = session.terms[key]
    r2 = term.levels.index("R2")
    session.set_values(key, [r2], [term.edited_log_effect[r2] + 0.1])
    # The chart's order drag: R2 to the end of the axis.
    session.select_indices(key, [r2])
    order = list(term.levels)
    session.reorder_levels(key, target_index=term.size - 1)
    moved = session.terms[key]
    assert moved.levels != order and moved.levels.index("R2") != r2

    r3 = moved.levels.index("R3")
    session.select_indices(key, [r3])
    session.shift(key, 0.05)
    session.select_indices(key, [moved.levels.index("R2")])
    session.reset(key)

    assert _capped(session)
    curve = session.terms[key].edited_log_effect
    assert curve[r3] > moved.original_log_effect[r3] + 0.01
    # The exported model carries the curve the session draws, level by level.
    shown = _level_curve(session.to_model(), key, moved.levels)
    np.testing.assert_allclose(shown, curve - curve[0], rtol=0.0, atol=1e-9)


def test_an_edit_of_the_base_moves_the_levels_its_cap_ties_to_it_and_no_others():
    session = open_case(ruled_case())
    key = "scale:region"
    term = session.terms[key]
    base = term.levels.index(str(session.model._specs[key]._base_level))
    before = term.edited_log_effect.copy()
    relative = before - before[base]
    at_cap = np.abs(relative - _CAP[1]) <= 1e-9
    assert at_cap.any()

    # Lowering the base lifts every level read from it, which those at the cap cannot take.
    session.set_values(key, [base], [before[base] - 0.05])

    assert len(session.history) == 1
    assert _capped(session)
    curve = session.terms[key].edited_log_effect
    assert curve[base] < before[base] - 0.01
    # The levels at the cap come down with the base, held at the cap; the rest stay.
    np.testing.assert_allclose(curve[at_cap] - curve[base], _CAP[1], rtol=0.0, atol=1e-9)
    free = ~at_cap
    free[base] = False
    np.testing.assert_allclose(curve[free], before[free], rtol=0.0, atol=1e-12)
    shown = _level_curve(session.to_model(), key, term.levels)
    np.testing.assert_allclose(shown, curve - curve[0], rtol=0.0, atol=1e-9)


@functools.cache
def _ordered_case() -> FitCase:
    """A gamma mean over ten ordered bands, 100 rows each, increasing through a cr k=5 basis."""
    bands = [f"B{i}" for i in range(10)]
    rng = np.random.default_rng(621)
    X = pd.DataFrame({"band": np.repeat(bands, 100), "age": rng.uniform(18.0, 90.0, 1000)})
    mean = np.exp(1.0 + 0.08 * np.repeat(np.arange(10), 100))
    y = rng.gamma(4.0, mean / 4.0)
    family = GammaLS()
    basis = Spline(kind="cr", k=5, constraint=Constraint.fit.increasing)
    model = SuperLSS(
        family,
        family.mean(term("band", OrderedCategorical(order=bands, basis=basis))),
        family.scale(),
    ).fit_reml(X, y, outer="efs+newton")
    return FitCase(model, X, y, None, None)


def test_an_edit_of_one_ordered_band_holds_the_base_and_moves_the_curve():
    case = _ordered_case()
    session = open_case(case)
    key = "mean:band"
    term = session.terms[key]
    base = term.levels.index(str(session.model._specs[key]._base_level))
    before = term.edited_log_effect.copy()
    last = term.size - 1
    assert base != last

    session.set_values(key, [last], [before[last] - 0.5])

    curve = session.terms[key].edited_log_effect
    # A smooth basis cannot move one band alone: the nearest curve moves, the base does not.
    assert curve[last] < before[last] - 0.05
    assert _increasing(curve)
    assert curve[base] == pytest.approx(before[base], abs=1e-12)
    # Nor do the exported predictions of the base band's rows.
    fitted = case.model.predict_link(case.X)["mean"].to_numpy()
    exported = session.to_model().predict_link(case.X)["mean"].to_numpy()
    rows = (case.X["band"] == term.levels[base]).to_numpy()
    np.testing.assert_allclose(exported[rows], fitted[rows], rtol=0.0, atol=1e-12)


def test_opening_skips_rows_without_weight_whatever_they_hold():
    X, y, _ = _gamma_rows(_SEEDS[GammaLS], 180)
    X = X.astype({"power": object})
    X.loc[X.index[-1], "power"] = "ignored"
    weights = np.ones(len(X))
    weights[-1] = 0.0
    family = GammaLS()
    model = SuperLSS(family, family.mean(s("age", k=5), s("power", k=5)), family.scale()).fit_reml(
        X, y, sample_weight=weights
    )

    session = edit(model, train_data=(X, y, weights))

    assert session.terms["mean:power"].metadata["shape_support"] is not None


def test_a_bulk_edit_of_many_capped_levels_takes_a_few_qp_steps(monkeypatch):
    session = open_case(_many_levels_case())
    term = session.terms["mean:g"]
    steps = []
    solve = qp_retry.solve_constrained_qp

    def counted(*args, **kwargs):
        result = solve(*args, **kwargs)
        steps.append(result.n_iter)
        return result

    monkeypatch.setattr(qp_retry, "solve_constrained_qp", counted)
    others = np.arange(1, term.size)

    # Raising every level but the base pushes each past its cap, where it stops.
    session.set_values("mean:g", others, term.edited_log_effect[others] + 0.3)

    curve = session.terms["mean:g"].edited_log_effect
    np.testing.assert_allclose(
        curve[others] - curve[0],
        np.minimum(term.original_log_effect[others] - curve[0] + 0.3, _CAP[1]),
        rtol=0.0,
        atol=1e-9,
    )
    # The rows the unconstrained fit breaks are the ones the answer binds: one solve, few steps.
    assert len(steps) == 1
    assert steps[0] <= 3


@functools.cache
def _convex_nb_case() -> FitCase:
    """An NB mean held convex in ``age`` (cr k=12) on 3,000 rows drawn with a concave hump."""
    X, y, offsets = _nb_rows(_SEEDS[NegativeBinomialLS], 3000)
    family = NegativeBinomialLS()
    model = SuperLSS(
        family,
        family.mean(
            s("age", kind="cr", k=12, constraint=Constraint.fit.convex), offset=offsets["mean"]
        ),
        family.theta(),
    ).fit_reml(X, y)
    return FitCase(model, X, y, None, offsets)


@pytest.mark.parametrize("delta", [0.3, 0.1])
def test_lowering_the_tail_of_a_convex_curve_is_kept_inside_the_rule(delta):
    case = _convex_nb_case()
    session = open_case(case)
    term = session.terms["mean:age"]
    tail = np.flatnonzero(term.x > np.quantile(term.x, 0.8))
    before = term.edited_log_effect.copy()

    # The QP starts from the current curve, which meets the rule, so the edit is made.
    session.set_values("mean:age", tail, before[tail] - delta)

    curve = session.terms["mean:age"].edited_log_effect
    assert len(session.history) == 1
    assert curve[-1] < before[-1] - 0.01
    assert _convex(curve, term.x)
    rows = pd.concat([case.X.iloc[[0]]] * term.size, ignore_index=True)
    rows["age"] = term.x
    zero = {"mean": np.zeros(term.size)}
    shown = session.to_model().predict_link(rows, offsets=zero)["mean"].to_numpy()
    np.testing.assert_allclose(shown - shown[0], curve - curve[0], rtol=0.0, atol=1e-9)


def test_the_convex_tail_edit_needs_only_the_feasible_start(monkeypatch):
    # Without its first working rows the QP still reaches the answer from the current curve.
    solve = qp_retry.solve_constrained_qp
    monkeypatch.setattr(
        qp_retry,
        "solve_constrained_qp",
        lambda *a, **k: solve(*a, **{**k, "active_set_init": None}),
    )
    session = open_case(_convex_nb_case())
    term = session.terms["mean:age"]
    tail = np.flatnonzero(term.x > np.quantile(term.x, 0.8))

    session.set_values("mean:age", tail, term.edited_log_effect[tail] - 0.3)

    assert len(session.history) == 1
    assert _convex(session.terms["mean:age"].edited_log_effect, term.x)


_BANDS = [f"B{i}" for i in range(10)]

_BAND_RULES = {
    "decreasing & max_step(0.1)": Constraint.fit.decreasing & Constraint.fit.max_step(0.1),
    "increasing & cap & max_step": (
        Constraint.fit.increasing & Constraint.fit.cap(0.8, 1.25) & Constraint.fit.max_step(0.15)
    ),
    "max_step & cap": Constraint.fit.max_step(0.15) & Constraint.fit.cap(0.8, 1.25),
    "cap & max_step": Constraint.fit.cap(0.8, 1.25) & Constraint.fit.max_step(0.15),
}


@functools.cache
def _banded_case(rules: str) -> FitCase:
    """A gamma mean over ten ordered bands, 100 rows each, through a cr k=5 basis under ``rules``."""
    rng = np.random.default_rng(621)
    X = pd.DataFrame({"band": np.repeat(_BANDS, 100)})
    y = rng.gamma(4.0, np.exp(1.0 + 0.05 * np.repeat(np.arange(10), 100)) / 4.0)
    family = GammaLS()
    basis = Spline(kind="cr", k=5, constraint=_BAND_RULES[rules])
    model = SuperLSS(
        family,
        family.mean(term("band", OrderedCategorical(order=_BANDS, basis=basis))),
        family.scale(),
    ).fit_reml(X, y, outer="efs+newton")
    return FitCase(model, X, y, None, None)


@pytest.mark.parametrize(
    ("rules", "band", "delta"),
    [("decreasing & max_step(0.1)", 9, -0.6), ("increasing & cap & max_step", 0, 0.6)],
)
def test_an_ordered_edit_whose_broken_rows_are_dependent_is_kept(monkeypatch, rules, band, delta):
    # An ordered term's cap and step rows are dependent (a cap row less the next is a step
    # row), so seeding the QP with every row the edit breaks can keep one that does not bind.
    case = _banded_case(rules)
    session = open_case(case)
    key = "mean:band"
    before = session.terms[key].edited_log_effect.copy()
    seeds = []
    solve = qp_retry.solve_constrained_qp

    def counted(*args, **kwargs):
        seeds.append(list(kwargs["active_set_init"] or ()))
        return solve(*args, **kwargs)

    monkeypatch.setattr(qp_retry, "solve_constrained_qp", counted)

    session.set_values(key, [band], [before[band] + delta])

    assert len(session.history) == 1
    # Dependent rows are not seeded, so one unseeded solve certifies the answer.
    assert seeds == [[]]
    curve = session.terms[key].edited_log_effect
    assert np.max(np.abs(curve - before)) > 0.05
    shown = session.to_model().predict_link(pd.DataFrame({"band": _BANDS}))["mean"].to_numpy()
    np.testing.assert_allclose(shown - shown[0], curve - curve[0], rtol=0.0, atol=1e-9)


@pytest.mark.parametrize("rules", ["max_step & cap", "cap & max_step"])
def test_an_edit_the_rules_absorb_says_so_whatever_order_they_are_written_in(rules):
    session = open_case(_banded_case(rules))
    key = "mean:band"
    before = session.terms[key].edited_log_effect.copy()

    with pytest.raises(EditorValueError, match="changed nothing: its rules"):
        session.set_values(key, [9], [before[9] + 0.5])

    assert session.history == []


def test_a_rule_the_basis_holds_keeps_the_base_where_it_is():
    # ``flat.above`` is drawn by the basis and gives no rows ``A b >= c``; the base is held anyway.
    X = pd.DataFrame({"band": np.repeat(_BANDS, 50)})
    y = np.random.default_rng(5).gamma(4.0, np.exp(np.repeat(np.arange(10), 50) * 0.04) / 4.0)
    family = GammaLS()
    basis = Spline(kind="cr", k=6, constraint=Constraint.fit.flat.above("B7"))
    with pytest.warns(UserWarning, match="put a knot at 'B7'"):
        model = SuperLSS(
            family,
            family.mean(term("band", OrderedCategorical(order=_BANDS, basis=basis))),
            family.scale(),
        ).fit_reml(X, y)
    session = edit(model, train_data=(X, y, None, None))
    before = session.terms["mean:band"].edited_log_effect.copy()

    session.set_values("mean:band", [1], [before[1] + 0.5])

    assert session.terms["mean:band"].edited_log_effect[1] > before[1] + 0.05
    fitted = np.asarray(model.predict(X))
    exported = np.asarray(session.to_model().predict(X))
    rows = (X["band"] == "B0").to_numpy()
    np.testing.assert_allclose(exported[rows], fitted[rows], rtol=1e-12, atol=0.0)


def test_an_empty_string_base_level_is_held_like_any_other():
    X = pd.DataFrame({"g": np.repeat(["", "A", "B"], 100)})
    y = np.random.default_rng(621).gamma(5.0, 0.2, 300)
    family = GammaLS()
    model = SuperLSS(
        family,
        family.mean(cat("g", base="", constraint=Constraint.fit.cap(0.8, 1.25))),
        family.scale(),
    ).fit_reml(X, y)
    session = edit(model, train_data=(X, y, None, None))
    term = session.terms["mean:g"]
    a = term.levels.index("A")

    session.set_values("mean:g", [a], [term.edited_log_effect[a] + 0.5])

    fitted = np.asarray(model.predict(X))
    exported = np.asarray(session.to_model().predict(X))
    for level in ("", "B"):
        rows = (X["g"] == level).to_numpy()
        np.testing.assert_allclose(exported[rows], fitted[rows], rtol=1e-12, atol=0.0)
    rows = (X["g"] == "A").to_numpy()
    assert np.all(exported[rows] > fitted[rows] * 1.05)


def test_make_increasing_on_a_curve_its_rule_holds_flat_in_places_changes_nothing_quietly():
    # Where the rule binds the curve is flat, and its grid values dip by rounding: isotonic
    # asks for a change of that size, which is no change at all.
    rng = np.random.default_rng(3)
    X = pd.DataFrame({"x": rng.uniform(0.0, 10.0, 3000)})
    y = rng.gamma(4.0, np.exp(0.5 - 0.05 * np.abs(X["x"].to_numpy() - 5.0)) / 4.0)
    family = GammaLS()
    model = SuperLSS(
        family,
        family.mean(s("x", kind="cr", k=10, constraint=Constraint.fit.increasing)),
        family.scale(),
    ).fit_reml(X, y)
    session = edit(model, train_data=(X, y, None, None))
    curve = session.terms["mean:x"].edited_log_effect.copy()
    assert np.min(np.diff(curve)) < 0.0
    revision = session.model_revision

    session.isotonic("mean:x", "increasing")

    assert session.history == []
    assert session.model_revision == revision
    np.testing.assert_array_equal(session.terms["mean:x"].edited_log_effect, curve)


def test_an_edit_at_values_without_training_rows_says_so_rather_than_blame_the_rule():
    rng = np.random.default_rng(8)
    X = pd.DataFrame({"age": rng.integers(18, 91, 4000)})
    y = rng.gamma(4.0, np.exp(0.3 + 0.0004 * (X["age"].to_numpy() - 50.0) ** 2) / 4.0)
    family = GammaLS()
    model = SuperLSS(
        family,
        family.mean(s("age", kind="cr", k=8, constraint=Constraint.fit.convex)),
        family.scale(),
    ).fit_reml(X, y)
    session = edit(model, train_data=(X, y, None, None))
    term = session.terms["mean:age"]
    weights = np.asarray(term.weights)
    empty = int(np.flatnonzero(weights == 0.0)[3])

    with pytest.raises(EditorValueError) as refused:
        session.set_values("mean:age", [empty], [term.edited_log_effect[empty] - 0.3])

    message = str(refused.value)
    assert "no training rows sit at the values it moves" in message
    assert "shape constraint" not in message
    assert session.history == []
    # At a value with rows, the same edit is made.
    held = int(np.flatnonzero(weights > 0.0)[5])
    session.set_values("mean:age", [held], [term.edited_log_effect[held] - 0.3])
    assert len(session.history) == 1


@pytest.mark.parametrize("step", [-0.5, 0.3, 0.5])
def test_an_edit_of_a_whole_stretch_without_training_rows_is_refused_not_recorded(step):
    # The projection floors an empty value's weight at 1e-12, so on a poorly conditioned
    # P-spline such an edit moved the kept curve by a few billionths of the change asked:
    # past the no-move bar, so it was recorded as an edit that changed nothing a chart shows.
    rng = np.random.default_rng(5)
    x = np.concatenate([rng.uniform(0.0, 4.5, 1500), rng.uniform(9.5, 14.0, 1500)])
    X = pd.DataFrame({"x": x})
    y = rng.gamma(4.0, np.exp(0.2 + 0.05 * x) / 4.0)
    family = GammaLS()
    model = SuperLSS(
        family,
        family.mean(s("x", kind="ps", k=14, constraint=Constraint.fit.increasing)),
        family.scale(),
    ).fit_reml(X, y)
    session = edit(model, train_data=(X, y, None, None))
    term = session.terms["mean:x"]
    empty = np.flatnonzero(np.asarray(term.weights) == 0.0)
    assert empty.size > 20
    curve = term.edited_log_effect.copy()
    revision = session.model_revision

    with pytest.raises(EditorValueError, match="no training rows sit at the values it moves"):
        session.set_values("mean:x", empty, curve[empty] + step)

    assert session.history == []
    assert session.model_revision == revision
    np.testing.assert_array_equal(session.terms["mean:x"].edited_log_effect, curve)


def test_a_seeded_solve_that_does_not_certify_is_retried_from_the_current_curve(monkeypatch):
    session = open_case(_many_levels_case())
    term = session.terms["mean:g"]
    solve = qp_retry.solve_constrained_qp

    def seeded_stops_short(*args, **kwargs):
        result = solve(*args, **kwargs)
        if kwargs["active_set_init"]:
            return type(result)(**{**vars(result), "converged": False})
        return result

    monkeypatch.setattr(qp_retry, "solve_constrained_qp", seeded_stops_short)
    others = np.arange(1, term.size)

    session.set_values("mean:g", others, term.edited_log_effect[others] + 0.3)

    curve = session.terms["mean:g"].edited_log_effect
    assert len(session.history) == 1
    np.testing.assert_allclose(
        curve[others] - curve[0],
        np.minimum(term.original_log_effect[others] - curve[0] + 0.3, _CAP[1]),
        rtol=0.0,
        atol=1e-9,
    )
