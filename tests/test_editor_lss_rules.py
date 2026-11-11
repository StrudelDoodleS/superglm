"""SuperLSS shape rules in the editor: named on each term, and every edit kept inside them."""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize_scalar

import superglm.editor.lss.apply as lss_apply
import superglm.editor.lss.view as lss_view
from superglm import (
    Constraint,
    GammaLS,
    NegativeBinomialLS,
    Poisson,
    Spline,
    SuperGLM,
    SuperLSS,
    cat,
    s,
)
from superglm.editor import EditorSession
from superglm.editor.errors import EditorValueError
from superglm.editor.lss.apply import lss_model_with_edits
from superglm.editor.payloads import session_payload
from superglm.features._spline_runtime import fit_support
from tests._lss_editor_fixtures import (
    _SEEDS,
    N_ROWS,
    FitCase,
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


def test_an_edit_the_rules_take_back_entirely_records_nothing():
    session = open_case(ruled_case())
    term = session.terms["scale:region"]
    base = term.levels.index(str(session.model._specs["scale:region"]._base_level))
    relative = term.edited_log_effect - term.edited_log_effect[base]
    at_cap = int(np.argmax(relative))
    assert relative[at_cap] == pytest.approx(_CAP[1], abs=1e-9)
    revision = session.model_revision

    # Raising a level that already sits at its cap: the rule puts it straight back.
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
    solve = lss_apply.solve_constrained_qp

    def stops_short(*args, **kwargs):
        result = solve(*args, **kwargs)
        return type(result)(**{**vars(result), "converged": False})

    monkeypatch.setattr(lss_apply, "solve_constrained_qp", stops_short)
    tail = np.flatnonzero(term.x > 10.0)

    with pytest.raises(EditorValueError, match="could not be confirmed"):
        session.set_values("mean:power", tail, term.edited_log_effect[tail] - 0.4)

    assert session.history == []
    np.testing.assert_array_equal(session.terms["mean:power"].edited_log_effect, before)


def test_the_first_ruled_edit_reads_no_training_rows(monkeypatch):
    session = open_case(ruled_case())
    term = session.terms["mean:power"]
    reads = []
    original = lss_view._reference_value
    monkeypatch.setattr(
        lss_view, "_reference_value", lambda *a, **k: reads.append(a[-1]) or original(*a, **k)
    )
    monkeypatch.setattr(
        lss_view, "_reference_design", lambda *a, **k: pytest.fail("built on the first edit")
    )
    tail = np.flatnonzero(term.x > 10.0)

    session.set_values("mean:power", tail, term.edited_log_effect[tail] - 0.4)
    session.set_values("scale:region", [1], [session.terms["scale:region"].edited_log_effect[1]])

    assert reads == []


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
    # Curvature in θ has no image in 1/θ, so the rule names θ.
    assert payload["theta:power"]["rules"]["text"] == "convex in θ"


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
