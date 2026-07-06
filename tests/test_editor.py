from __future__ import annotations

import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from superglm import (
    Categorical,
    Numeric,
    OrderedCategorical,
    Polynomial,
    Spline,
    SuperGLM,
    families,
    generate_tweedie_cpg,
)
from superglm.editor import EditableTerm, EditorSession
from superglm.inference.summary import ModelSummary, _BasisDetailRow, _CoefRow


def test_editor_demo_notebook_includes_k_adequacy_sweep():
    notebook_path = Path(__file__).resolve().parents[1] / "docs/notebooks/editor_demo.ipynb"
    notebook = json.loads(notebook_path.read_text())
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    assert "## Choose Spline Capacity With A k Sweep" in source
    assert "n = 10_000" in source
    assert "## In-Sample k Adequacy" in source
    assert "## Cross-Validated k Check" in source
    assert "from sklearn.model_selection import KFold" in source
    assert "from sklearn.model_selection import KFold, train_test_split" in source
    assert "families" in source
    assert "generate_tweedie_cpg" in source
    assert "X_train_val, X_test" in source
    assert "X_train, X_val" in source
    assert "TRUE_TWEEDIE_P" in source
    assert "TRUE_TWEEDIE_P_INIT" in source
    assert "TRUE_TWEEDIE_PHI" in source
    assert "mu = np.exp(eta)" in source
    assert "y = generate_tweedie_cpg(" in source
    assert "TERRITORY_LEVELS" in source
    assert "territory_effect_map" in source
    assert "territory_probs" in source
    assert "territory = rng.choice(TERRITORY_LEVELS" in source
    assert '"T01"' in source
    assert '"T10"' in source
    assert "territory_effect" in source
    assert "family=families.tweedie(p=TRUE_TWEEDIE_P_INIT)" in source
    assert "model.estimate_p(" in source
    assert "tweedie_profile.search_trace" in source
    assert 'method="brent"' in source
    assert 'phi_method="mle"' in source
    assert '"territory": Categorical(base="most_exposed")' in source
    assert 'terms=["age", "mileage", "region", "age_band", "territory"]' in source
    assert "## Collapse Sparse Categorical Levels" in source
    assert 'collapse_session.select_levels("territory", ["T01", "T03"])' in source
    assert 'collapse_session.select_levels("territory", ["T08", "T10"])' in source
    assert "grouped_preview = session_payload(collapse_session)" in source
    assert 'collapse_session.replace_with_collapsed_levels("territory", method="fit")' in source
    assert "cross_validate(" in source
    assert "cv = KFold(" in source
    assert "cv=cv" in source
    assert "k_grid" in source
    assert "cv_sweep" in source
    assert "cv_report = {" in source
    assert '"method": "KFold"' in source
    assert "mean_val_deviance" in source
    assert "recommend_k_from_cv" in source
    assert "cv_one_se_k" in source
    assert "cv_recommended_k" in source
    assert "median_age_band_min_p" in source
    assert "chosen_k = cv_recommended_k" in source
    assert "age_edf" in source
    assert "total_edf" in source
    assert "edf_share" in source
    assert "chosen_k" in source
    assert 'Spline(kind="bs", k=age_k' in source
    assert "model = fit_age_model(chosen_k, X_fit=X_train, y_fit=y_train, w_fit=w_train)" in source
    assert "cv_loss_records = []" in source
    assert "fold_train_loss" in source
    assert "fold_validation_loss" in source
    assert "cv_loss_summary" in source
    assert "split_loss" in source
    assert '"metric": "weighted mean unit deviance"' in source
    assert '"rows": cv_loss.round(6).to_dict("records")' in source
    assert '"summary": cv_loss_summary.round(6).to_dict("records")' in source
    assert '"split_loss": split_loss.round(6).to_dict("records")' in source
    assert "train_data=(X_train, y_train, w_train)" in source
    assert "validation_data=(X_val, y_val, w_val)" in source
    assert "test_data=(X_test, y_test, w_test)" in source
    assert "cv_report=cv_report" in source


@pytest.fixture
def editor_frame():
    rng = np.random.default_rng(20260703)
    n = 450
    x_spline = rng.uniform(0.0, 10.0, n)
    x_poly = rng.uniform(-2.0, 3.0, n)
    x_num = rng.normal(0.0, 1.0, n)
    region = rng.choice(["A", "B", "C"], n, p=[0.4, 0.35, 0.25])
    band = rng.choice(["low", "medium", "high"], n, p=[0.45, 0.35, 0.20])

    eta = (
        0.6
        + 0.18 * np.sin(x_spline / 1.8)
        + 0.08 * x_poly**2
        + 0.12 * x_num
        + 0.25 * (region == "B")
        - 0.18 * (region == "C")
        + 0.15 * (band == "medium")
        + 0.32 * (band == "high")
    )
    y = eta + rng.normal(0.0, 0.08, n)
    X = pd.DataFrame(
        {
            "x_spline": x_spline,
            "x_poly": x_poly,
            "x_num": x_num,
            "region": region,
            "band": band,
        }
    )
    return X, y


@pytest.fixture
def editor_model(editor_frame):
    X, y = editor_frame
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        spline_penalty=0.1,
        features={
            "x_spline": Spline(n_knots=8),
            "x_poly": Polynomial(degree=2),
            "x_num": Numeric(),
            "region": Categorical(base="first"),
            "band": OrderedCategorical(order=["low", "medium", "high"], basis="step", base="first"),
        },
    )
    model.fit(X, y)
    return model


def test_session_extracts_1d_main_effects(editor_model):
    session = EditorSession.from_model(editor_model, terms=["x_spline", "region"])

    assert list(session.terms) == ["x_spline", "region"]
    assert session.terms["x_spline"].x is not None
    assert session.terms["region"].levels == ["A", "B", "C"]
    assert session.terms["x_spline"].metadata["term_type"] == "spline"
    assert session.terms["region"].metadata["term_type"] == "categorical"


def test_select_x_and_levels(editor_model):
    session = EditorSession.from_model(editor_model, terms=["x_spline", "region"])

    session.select_x("x_spline", 2.0, 5.0)
    assert session.selection("x_spline").size > 0

    session.select_levels("region", ["B", "C"])
    assert session.selection("region").tolist() == [1, 2]


def test_shift_set_interpolate_isotonic_smooth_and_history(editor_model):
    session = EditorSession.from_model(editor_model, terms=["x_spline"])
    term = session.terms["x_spline"]
    session.select_indices("x_spline", [10, 11, 12, 13])

    before = term.edited_log_effect.copy()
    session.shift("x_spline", 0.25)
    np.testing.assert_allclose(term.edited_log_effect[10:14], before[10:14] + 0.25)

    session.set_value("x_spline", -0.1)
    np.testing.assert_allclose(term.edited_log_effect[10:14], -0.1)

    term.edited_log_effect[10:14] = [0.4, -0.2, 0.3, -0.1]
    session.linear_interpolate("x_spline")
    np.testing.assert_allclose(term.edited_log_effect[[10, 13]], [0.4, -0.1])
    assert np.all(np.diff(term.edited_log_effect[10:14]) < 0)

    term.edited_log_effect[10:14] = [0.3, -0.2, 0.2, -0.1]
    session.isotonic("x_spline", direction="increasing")
    assert np.all(np.diff(term.edited_log_effect[10:14]) >= -1e-12)

    session.undo()
    np.testing.assert_allclose(term.edited_log_effect[10:14], [0.3, -0.2, 0.2, -0.1])

    session.redo()
    assert np.all(np.diff(term.edited_log_effect[10:14]) >= -1e-12)

    term.edited_log_effect[10:14] = [0.0, 1.0, -0.5, 2.0]
    session.linear_interpolate("x_spline", strength=0.5)
    full_target = np.interp(
        term.x[10:14],
        [term.x[10], term.x[13]],
        [0.0, 2.0],
    )
    expected_blend = 0.5 * np.array([0.0, 1.0, -0.5, 2.0]) + 0.5 * full_target
    np.testing.assert_allclose(term.edited_log_effect[10:14], expected_blend)

    term.edited_log_effect[10:14] = [0.0, 1.0, 0.0, 1.0]
    session.smooth("x_spline", strength=1.0)
    assert np.max(term.edited_log_effect[10:14]) < 1.0


def test_level_and_snap_selected_values(editor_model):
    session = EditorSession.from_model(editor_model, terms=["x_spline"])
    term = session.terms["x_spline"]
    session.select_indices("x_spline", [10, 11, 12, 13])

    term.edited_log_effect[10:14] = [0.2, 1.0, -0.4, 0.8]
    session.level_left("x_spline")
    np.testing.assert_allclose(term.edited_log_effect[10:14], 0.2)

    term.edited_log_effect[10:14] = [0.2, 1.0, -0.4, 0.8]
    session.level_right("x_spline")
    np.testing.assert_allclose(term.edited_log_effect[10:14], 0.8)

    term.edited_log_effect[10:14] = [0.2, 1.0, -0.4, 0.8]
    session.snap_highest("x_spline")
    np.testing.assert_allclose(term.edited_log_effect[10:14], 1.0)

    term.edited_log_effect[10:14] = [0.2, 1.0, -0.4, 0.8]
    session.snap_lowest("x_spline")
    np.testing.assert_allclose(term.edited_log_effect[10:14], -0.4)


def test_isotonic_anchors_selected_region_to_neighbors(editor_model):
    session = EditorSession.from_model(editor_model, terms=["x_spline"])
    term = session.terms["x_spline"]
    term.edited_log_effect[:] = np.linspace(0.0, 1.0, term.size)
    term.edited_log_effect[10:15] = [0.8, -0.2, 0.6, 0.1, 0.5]

    left_neighbor = term.edited_log_effect[9]
    right_neighbor = term.edited_log_effect[15]
    session.select_indices("x_spline", range(10, 15))
    session.isotonic("x_spline", direction="increasing")

    after = term.edited_log_effect
    assert after[10] >= left_neighbor
    assert after[14] <= right_neighbor
    assert np.all(np.diff(after[10:15]) >= -1e-12)


def test_isotonic_neighbor_anchors_constrain_without_pinning_to_both_sides():
    values = np.array([-0.007, 0.043, 0.027, 0.052, 0.113], dtype=np.float64)
    session = EditorSession(
        model=None,
        terms={
            "age_band": EditableTerm(
                name="age_band",
                kind="categorical",
                levels=["18-24", "25-34", "35-49", "50-64", "65-80"],
                original_log_effect=values.copy(),
                edited_log_effect=values.copy(),
                weights=np.ones_like(values),
            )
        },
    )

    session.select_indices("age_band", [2, 3])
    session.isotonic("age_band", direction="increasing")

    after = session.terms["age_band"].edited_log_effect
    assert after[2] == pytest.approx(values[1])
    assert after[3] == pytest.approx(values[3])
    assert after[3] < values[4]
    assert np.all(np.diff(after) >= -1e-12)


def test_categorical_monotone_uses_directional_clamp_instead_of_pooling():
    values = np.array([-0.007, 0.039, 0.023, 0.047, 0.113], dtype=np.float64)
    session = EditorSession(
        model=None,
        terms={
            "age_band": EditableTerm(
                name="age_band",
                kind="categorical",
                levels=["18-24", "25-34", "35-49", "50-64", "65-80"],
                original_log_effect=values.copy(),
                edited_log_effect=values.copy(),
                weights=np.array([10.0, 10.0, 30.0, 20.0, 30.0]),
            )
        },
    )

    session.select_indices("age_band", [0, 1, 2, 3, 4])
    session.isotonic("age_band", direction="increasing")
    np.testing.assert_allclose(
        session.terms["age_band"].edited_log_effect,
        [-0.007, 0.039, 0.039, 0.047, 0.113],
    )

    session.reset("age_band")
    session.select_indices("age_band", [1, 2, 3, 4])
    session.isotonic("age_band", direction="decreasing")
    np.testing.assert_allclose(
        session.terms["age_band"].edited_log_effect,
        [-0.007, -0.007, -0.007, -0.007, -0.007],
    )


def test_smooth_anchors_selected_region_to_neighbors(editor_model):
    session = EditorSession.from_model(editor_model, terms=["x_spline"])
    term = session.terms["x_spline"]
    term.edited_log_effect[:] = np.linspace(0.0, 1.0, term.size)
    term.edited_log_effect[10:15] = [0.8, -0.2, 0.6, 0.1, 0.5]

    before = term.edited_log_effect.copy()
    session.select_indices("x_spline", range(10, 15))
    session.smooth("x_spline", strength=1.0)

    after = term.edited_log_effect
    assert abs(after[10] - before[9]) <= abs(before[10] - before[9]) + 1e-12
    assert abs(after[14] - before[15]) <= abs(before[14] - before[15]) + 1e-12
    assert after[11] != pytest.approx(before[11])
    assert after[12] != pytest.approx(before[12])


def test_smooth_does_not_create_boundary_discontinuities(editor_model):
    session = EditorSession.from_model(editor_model, terms=["x_spline"])
    term = session.terms["x_spline"]
    term.edited_log_effect[:] = np.sin(np.linspace(0.0, 16.0, term.size))
    before = term.edited_log_effect.copy()
    start, stop = 40, 95

    session.select_indices("x_spline", range(start, stop))
    session.smooth("x_spline", strength=1.0)

    after = term.edited_log_effect
    assert abs(after[start] - after[start - 1]) <= abs(before[start] - before[start - 1]) + 1e-12
    assert abs(after[stop - 1] - after[stop]) <= abs(before[stop - 1] - before[stop]) + 1e-12
    assert not np.allclose(after[start:stop], before[start:stop])


def test_smooth_without_selection_applies_to_whole_term(editor_model):
    session = EditorSession.from_model(editor_model, terms=["x_spline"])
    term = session.terms["x_spline"]
    term.edited_log_effect[:] = np.sin(np.linspace(0.0, 20.0, term.size))
    before = term.edited_log_effect.copy()

    session.smooth("x_spline", strength=1.0)

    assert not np.allclose(term.edited_log_effect, before)
    assert session.history[-1].indices.size == term.size


def test_smooth_is_aggressive_for_narrow_spikes(editor_model):
    session = EditorSession.from_model(editor_model, terms=["x_spline"])
    term = session.terms["x_spline"]
    term.edited_log_effect[:] = 0.0
    term.edited_log_effect[term.size // 2] = 1.0

    session.smooth("x_spline", strength=1.0)

    assert np.max(term.edited_log_effect) < 0.2


def test_spline_control_points_are_fixed_x_vertical_handles(editor_model):
    session = EditorSession.from_model(editor_model, terms=["x_spline", "region"])
    term = session.terms["x_spline"]
    controls = session.control_points("x_spline")

    assert 6 <= controls["x"].size <= 12
    assert np.all(np.diff(controls["x"]) > 0)
    assert controls["x"][0] >= float(term.x.min()) - 1e-12
    assert controls["x"][-1] <= float(term.x.max()) + 1e-12

    before_curve = term.edited_log_effect.copy()
    before_controls_x = controls["x"].copy()
    handle_index = controls["x"].size // 2
    session.move_control_point(
        "x_spline", handle_index, float(controls["log_effect"][handle_index] + 0.35)
    )

    after_controls = session.control_points("x_spline")
    np.testing.assert_allclose(after_controls["x"], before_controls_x)
    assert np.max(np.abs(term.edited_log_effect - before_curve)) > 0.05
    assert after_controls["log_effect"][handle_index] == pytest.approx(
        controls["log_effect"][handle_index] + 0.35,
        abs=0.08,
    )

    session.undo("x_spline")
    np.testing.assert_allclose(term.edited_log_effect, before_curve)

    with pytest.raises(TypeError, match="control handles"):
        session.control_points("region")


def test_spline_control_point_edit_is_local_to_cubic_basis_support(editor_model):
    session = EditorSession.from_model(editor_model, terms=["x_spline"])
    term = session.terms["x_spline"]
    controls = session.control_points("x_spline")
    handle_index = controls["x"].size // 2
    basis_index = int(controls["basis_index"][handle_index])
    basis = editor_model._specs["x_spline"]._raw_basis_matrix(term.x)
    support = np.abs(basis[:, basis_index]) > 1e-12
    assert np.any(support)
    assert np.any(~support)

    before = term.edited_log_effect.copy()
    session.move_control_point(
        "x_spline",
        handle_index,
        float(controls["log_effect"][handle_index] + 0.35),
    )
    delta = term.edited_log_effect - before

    assert np.max(np.abs(delta[support])) > 0.05
    assert np.max(np.abs(delta[~support])) < 1e-8


def test_spline_control_point_count_selects_visible_basis_handles(editor_model):
    session = EditorSession.from_model(editor_model, terms=["x_spline"])
    term = session.terms["x_spline"]
    controls = session.control_points("x_spline", n_handles=6)

    assert controls["x"].size == 6
    assert controls["basis_index"].size == 6
    assert np.all(np.diff(controls["basis_index"]) > 0)
    assert np.all(np.diff(controls["x"]) > 0)

    handle_index = 2
    basis_index = int(controls["basis_index"][handle_index])
    basis = editor_model._specs["x_spline"]._raw_basis_matrix(term.x)
    support = np.abs(basis[:, basis_index]) > 1e-12
    before = term.edited_log_effect.copy()

    session.move_control_point(
        "x_spline",
        handle_index,
        float(controls["log_effect"][handle_index] + 0.35),
        n_handles=6,
    )
    delta = term.edited_log_effect - before

    assert np.max(np.abs(delta[support])) > 0.05
    assert np.max(np.abs(delta[~support])) < 1e-8


def test_reset_restores_selection_or_current_term(editor_model):
    session = EditorSession.from_model(editor_model, terms=["x_spline"])
    term = session.terms["x_spline"]
    original = term.original_log_effect.copy()

    session.select_indices("x_spline", [2, 3])
    session.shift("x_spline", 0.2)
    session.reset("x_spline")
    np.testing.assert_allclose(term.edited_log_effect[[2, 3]], original[[2, 3]])

    session.clear_selection("x_spline")
    term.edited_log_effect[:] = original + 0.3
    session.reset("x_spline")
    np.testing.assert_allclose(term.edited_log_effect, original)


def test_to_model_returns_copy_and_leaves_source_unchanged(editor_model):
    session = EditorSession.from_model(editor_model, terms=["x_spline"])
    original_beta = editor_model.result.beta.copy()
    original_intercept = editor_model.result.intercept

    session.select_x("x_spline", 2.0, 5.0)
    session.shift("x_spline", 0.35)
    edited = session.to_model()

    assert edited is not editor_model
    np.testing.assert_allclose(editor_model.result.beta, original_beta)
    assert editor_model.result.intercept == original_intercept
    assert not np.allclose(edited.result.beta, original_beta)


def test_to_model_refreshes_fit_statistics_after_manual_edit(editor_model):
    session = EditorSession.from_model(editor_model, terms=["x_spline"])
    original_deviance = editor_model.result.deviance

    session.select_x("x_spline", 2.0, 5.0)
    session.shift("x_spline", 1.25)
    edited = session.to_model()

    y = np.asarray(edited._fit_y_ref, dtype=np.float64)
    mu = edited.predict(edited._fit_X_ref, offset=edited._fit_offset_ref)
    expected_deviance = float(
        np.sum(edited._fit_weights * edited._distribution.deviance_unit(y, mu))
    )

    assert edited.result.deviance == pytest.approx(expected_deviance)
    assert edited.summary()["deviance"]["deviance"] == pytest.approx(expected_deviance)
    assert edited.result.deviance != pytest.approx(original_deviance)


def test_to_model_can_refresh_fit_statistics_from_explicit_data():
    rng = np.random.default_rng(20260710)
    n = 180
    X = pd.DataFrame({"x": rng.normal(size=n)})
    sample_weight = rng.uniform(0.5, 2.0, size=n)
    offset = rng.normal(0.0, 0.05, size=n)
    y = 0.4 + 0.3 * X["x"].to_numpy() + offset + rng.normal(0.0, 0.04, size=n)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x": Numeric()},
    )
    model.fit(X, y, sample_weight=sample_weight, offset=offset)
    session = EditorSession.from_model(model, terms=["x"])
    session.select_indices("x", [0])
    session.shift("x", 0.2)

    eval_offset = offset + 0.01
    edited = session.to_model(X=X, y=y, sample_weight=sample_weight, offset=eval_offset)
    mu = edited.predict(X, offset=eval_offset)
    expected_deviance = float(np.sum(sample_weight * edited._distribution.deviance_unit(y, mu)))

    assert edited.result.deviance == pytest.approx(expected_deviance)
    assert edited._fit_stats.pearson_chi2 == pytest.approx(expected_deviance)


def test_to_model_mean_centering_applies_equivalent_native_curve_edit(editor_model):
    native_session = EditorSession.from_model(
        editor_model,
        terms=["x_spline"],
        n_points=80,
        centering="native",
        with_se=False,
    )
    mean_session = EditorSession.from_model(
        editor_model,
        terms=["x_spline"],
        n_points=80,
        centering="mean",
        with_se=False,
    )
    indices = [22, 23, 24, 25]
    native_session.select_indices("x_spline", indices)
    mean_session.select_indices("x_spline", indices)
    native_session.shift("x_spline", 0.35)
    mean_session.shift("x_spline", 0.35)

    native_model = native_session.to_model()
    mean_model = mean_session.to_model()

    X_ref = editor_model._fit_X_ref
    np.testing.assert_allclose(
        mean_model.predict(X_ref),
        native_model.predict(X_ref),
        rtol=1e-10,
        atol=1e-10,
    )


def test_edited_offset_mean_centering_applies_equivalent_native_curve_edit(editor_model):
    native_session = EditorSession.from_model(
        editor_model,
        terms=["x_spline"],
        n_points=80,
        centering="native",
        with_se=False,
    )
    mean_session = EditorSession.from_model(
        editor_model,
        terms=["x_spline"],
        n_points=80,
        centering="mean",
        with_se=False,
    )
    indices = [22, 23, 24, 25]
    native_session.select_indices("x_spline", indices)
    mean_session.select_indices("x_spline", indices)
    native_session.shift("x_spline", 0.35)
    mean_session.shift("x_spline", 0.35)

    X_ref = editor_model._fit_X_ref
    np.testing.assert_allclose(
        mean_session.edited_offset("x_spline", X=X_ref),
        native_session.edited_offset("x_spline", X=X_ref),
        rtol=1e-10,
        atol=1e-10,
    )


def test_to_model_marks_edited_copy_inference_stale(editor_model):
    session = EditorSession.from_model(editor_model, terms=["x_spline"])
    session.select_x("x_spline", 2.0, 5.0)
    session.shift("x_spline", 0.35)

    edited = session.to_model()

    assert edited._editor_inference_stale is True
    assert edited._editor_edits["terms"] == ["x_spline"]
    summary = edited.summary()
    assert summary["standard_errors"] == {"inference_stale": True}
    assert "Editor edits applied" in str(summary)
    spline_row = next(row for row in summary._coef_rows if row.name == "x_spline")
    assert spline_row.wald_p is None
    assert summary._basis_detail == {}
    with pytest.warns(UserWarning, match="Editor coefficient edits"):
        inference = edited.term_inference("x_spline", with_se=True)
    assert inference.se_log_relativity is None
    assert inference.ci_lower is None


def test_edited_offset_factor_and_refit_with_fixed_offset(editor_model, editor_frame):
    X, _ = editor_frame
    session = EditorSession.from_model(editor_model, terms=["region", "x_spline"])
    session.select_levels("region", ["B"])
    session.shift("region", 0.4)

    offset = session.edited_offset("region", X=X)
    factor = session.edited_offset_factor("region", X=X)
    assert np.all(factor > 0)
    assert np.mean(offset[X["region"] == "B"]) > np.mean(offset[X["region"] == "A"])

    refit = session.refit_with_edited_offset(terms=["region"], method="fit")

    assert "region" not in refit.features
    assert "x_spline" in refit.features
    np.testing.assert_allclose(refit._fit_offset, offset)
    assert refit._editor_offset["terms"] == ["region"]
    refit_summary = refit.summary()
    assert "Editor offset refit" in str(refit_summary)
    inference = refit.term_inference("x_spline", with_se=True)
    assert inference.se_log_relativity is not None


def test_collapse_selected_categorical_levels_refits_copied_model(editor_model, editor_frame):
    X, _ = editor_frame
    session = EditorSession.from_model(editor_model, terms=["region", "x_spline"])
    session.select_levels("region", ["B", "C"])

    refit = session.refit_with_collapsed_levels("region", method="fit")

    assert refit is not editor_model
    assert set(refit.features) == set(editor_model.features)
    assert getattr(editor_model.features["region"], "_grouping", None) is None
    grouping = refit.features["region"]._grouping
    assert grouping.original_to_group["A"] == "A"
    assert grouping.original_to_group["B"] == "B+C"
    assert grouping.original_to_group["C"] == "B+C"
    assert refit._editor_level_collapse == {
        "format": "superglm.editor.level_collapse.v1",
        "term": "region",
        "group_label": "B+C",
        "levels": ["B", "C"],
        "method": "fit",
        "message": "Selected categorical levels were collapsed and the full model was refit.",
    }
    assert refit._predict_eta_exact(X).shape == editor_model._predict_eta_exact(X).shape
    inference = refit.term_inference("region", with_se=True)
    assert inference.se_log_relativity is not None


def test_collapse_selected_ordered_categorical_levels_requires_contiguous_selection(editor_model):
    session = EditorSession.from_model(editor_model, terms=["band"])
    session.select_levels("band", ["low", "high"])

    with pytest.raises(ValueError, match="contiguous"):
        session.refit_with_collapsed_levels("band", method="fit")

    session.select_levels("band", ["medium", "high"])
    refit = session.refit_with_collapsed_levels("band", method="fit")

    grouping = refit.features["band"]._grouping
    assert grouping.original_to_group["low"] == "low"
    assert grouping.original_to_group["medium"] == "medium+high"
    assert grouping.original_to_group["high"] == "medium+high"


def test_auto_level_refits_use_reml_when_source_model_was_reml_fit():
    rng = np.random.default_rng(20260711)
    n = 700
    age = rng.uniform(18.0, 80.0, n)
    territory = rng.choice(["T01", "T02", "T03", "T04"], n, p=[0.10, 0.45, 0.10, 0.35])
    territory_effect = {"T01": 0.08, "T02": 0.0, "T03": 0.10, "T04": -0.06}
    X = pd.DataFrame({"age": age, "territory": territory})
    y = (
        0.8
        + 0.18 * np.sin(age / 8.0)
        + np.array([territory_effect[level] for level in territory])
        + rng.normal(0.0, 0.05, n)
    )
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={
            "age": Spline(k=8),
            "territory": Categorical(base="most_exposed"),
        },
    )
    model.fit_reml(X, y)
    session = EditorSession.from_model(model, terms=["age", "territory"], train_data=(X, y, None))

    session.select_levels("territory", ["T01", "T03"])
    collapsed = session.replace_with_collapsed_levels("territory", method="auto")

    assert collapsed._last_fit_meta["method"] == "fit_reml"
    assert collapsed._editor_level_collapse["method"] == "fit_reml"
    assert collapsed._reml_lambdas
    assert "age" in collapsed._reml_lambdas


def test_collapse_levels_replaces_in_force_model_and_clears_manual_edits(editor_model):
    session = EditorSession.from_model(editor_model, terms=["x_spline", "region"])
    session.select_indices("x_spline", [10, 11, 12])
    session.shift("x_spline", 0.4)
    assert session.edited_terms() == ["x_spline"]

    session.select_levels("region", ["B", "C"])
    refit = session.replace_with_collapsed_levels("region", method="fit")

    assert session.reference_model is editor_model
    assert session.model is refit
    assert session.edited_terms() == []
    assert session.history == []
    assert session.selection("x_spline").size == 0
    np.testing.assert_allclose(
        session.terms["x_spline"].edited_log_effect,
        session.terms["x_spline"].original_log_effect,
    )
    grouping = session.model.features["region"]._grouping
    assert grouping.original_to_group["B"] == "B+C"
    assert grouping.original_to_group["C"] == "B+C"


def test_to_model_applies_manual_edit_after_level_collapse(editor_model):
    session = EditorSession.from_model(editor_model, terms=["region"])
    session.select_levels("region", ["B", "C"])
    session.replace_with_collapsed_levels("region", method="fit")

    session.select_levels("region", ["B"])
    session.shift("region", 0.2)
    edited = session.to_model()

    mu = edited.predict(editor_model._fit_X_ref)
    assert np.isfinite(mu).all()


def test_uncollapse_levels_restores_previous_in_force_model(editor_model):
    session = EditorSession.from_model(editor_model, terms=["region", "band"])
    original_model = session.model
    session.select_levels("region", ["B", "C"])

    collapsed = session.replace_with_collapsed_levels("region", method="fit")

    assert session.model is collapsed
    assert session.can_uncollapse_levels() is True
    assert session.model.features["region"]._grouping.original_to_group["B"] == "B+C"

    restored = session.uncollapse_levels()

    assert restored is original_model
    assert session.model is original_model
    assert session.can_uncollapse_levels() is False
    assert getattr(session.model.features["region"], "_grouping", None) is None
    assert session.history == []
    assert session.redo_stack == []
    assert session.selection("region").size == 0


def test_uncollapse_levels_rolls_back_one_collapse_at_a_time(editor_model):
    session = EditorSession.from_model(editor_model, terms=["region", "band"])
    original_model = session.model

    session.select_levels("region", ["B", "C"])
    first_collapse = session.replace_with_collapsed_levels("region", method="fit")
    session.select_levels("band", ["medium", "high"])
    second_collapse = session.replace_with_collapsed_levels("band", method="fit")
    assert session.model is second_collapse

    restored_once = session.uncollapse_levels()

    assert restored_once is first_collapse
    assert session.model is first_collapse
    assert session.can_uncollapse_levels() is True
    assert session.model.features["region"]._grouping.original_to_group["B"] == "B+C"
    assert getattr(session.model.features["band"], "_grouping", None) is None

    restored_twice = session.uncollapse_levels()

    assert restored_twice is original_model
    assert session.model is original_model
    assert session.can_uncollapse_levels() is False


def test_repeated_categorical_collapses_create_distinct_level_groups():
    rng = np.random.default_rng(20260705)
    levels = [f"T{i:02d}" for i in range(1, 11)]
    probs = np.array([0.04, 0.26, 0.04, 0.16, 0.12, 0.10, 0.09, 0.03, 0.13, 0.03])
    territory = rng.choice(levels, 900, p=probs / probs.sum())
    effects = {
        "T01": 0.12,
        "T02": 0.10,
        "T03": 0.13,
        "T04": -0.08,
        "T05": 0.20,
        "T06": -0.12,
        "T07": 0.02,
        "T08": -0.18,
        "T09": -0.20,
        "T10": -0.17,
    }
    X = pd.DataFrame({"territory": territory})
    y = 0.5 + np.array([effects[value] for value in territory]) + rng.normal(0.0, 0.05, 900)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"territory": Categorical(base="most_exposed")},
    )
    model.fit(X, y)
    session = EditorSession.from_model(model, terms=["territory"], train_data=(X, y, None))

    session.select_levels("territory", ["T01", "T03"])
    session.replace_with_collapsed_levels("territory", method="fit")
    session.select_levels("territory", ["T08", "T10"])
    session.replace_with_collapsed_levels("territory", method="fit")

    grouping = session.model.features["territory"]._grouping
    assert grouping.original_to_group["T01"] == "T01+T03"
    assert grouping.original_to_group["T03"] == "T01+T03"
    assert grouping.original_to_group["T08"] == "T08+T10"
    assert grouping.original_to_group["T10"] == "T08+T10"
    from superglm.editor.payloads import session_payload

    groups = session_payload(session)["territory"]["level_groups"]
    assert groups == [
        {"label": "T01+T03", "indices": [0, 2], "levels": ["T01", "T03"]},
        {"label": "T08+T10", "indices": [7, 9], "levels": ["T08", "T10"]},
    ]


def test_compact_summary_shows_collapsed_reference_level_group():
    from superglm.editor.summaries import summary_payload

    rng = np.random.default_rng(20260709)
    levels = [f"T{i:02d}" for i in range(1, 11)]
    territory = np.repeat(levels, 40)
    X = pd.DataFrame({"territory": territory})
    effects = {
        "T01": 0.02,
        "T02": 0.03,
        "T03": 0.12,
        "T04": 0.13,
        "T05": 0.11,
        "T06": -0.04,
        "T07": -0.03,
        "T08": -0.05,
        "T09": -0.15,
        "T10": 0.05,
    }
    y = 0.5 + np.array([effects[value] for value in territory]) + rng.normal(0.0, 0.02, len(X))
    sample_weight = np.ones(len(X), dtype=np.float64)
    sample_weight[np.isin(territory, ["T03", "T04", "T05"])] = 3.0
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"territory": Categorical(base="most_exposed")},
    )
    model.fit(X, y, sample_weight=sample_weight)
    session = EditorSession.from_model(
        model,
        terms=["territory"],
        train_data=(X, y, sample_weight),
    )

    session.select_levels("territory", ["T03", "T04", "T05"])
    session.replace_with_collapsed_levels("territory", method="fit")
    session.select_levels("territory", ["T06", "T07", "T08"])
    session.replace_with_collapsed_levels("territory", method="fit")
    assert session.model.features["territory"]._base_level == "T03+T04+T05"

    widget = session.widget()
    try:
        payload = summary_payload(widget, "in_force")
    finally:
        widget.close()

    rows = payload["compact"]["rows"]
    names = [row["name"] for row in rows]
    assert "territory[T03+T04+T05]" in names
    assert "territory[T06+T07+T08]" in names
    reference_row = next(row for row in rows if row["name"] == "territory[T03+T04+T05]")
    assert reference_row["kind"] == "reference"
    assert reference_row["coef"] == 0.0
    assert reference_row["se_label"] == "ref"


def test_compact_summary_shows_regular_reference_level(editor_model):
    from superglm.editor.summaries import summary_payload

    session = EditorSession.from_model(editor_model, terms=["region"])
    widget = session.widget()
    try:
        payload = summary_payload(widget, "in_force")
    finally:
        widget.close()

    rows = payload["compact"]["rows"]
    reference_row = next(row for row in rows if row["name"] == "region[A]")
    assert reference_row["kind"] == "reference"
    assert reference_row["coef"] == 0.0
    assert reference_row["se_label"] == "ref"
    assert reference_row["sig_class"] == "sig-reference"


def test_collapse_selected_levels_across_existing_groups_preserves_remainders():
    rng = np.random.default_rng(20260707)
    levels = [f"T{i:02d}" for i in range(1, 7)]
    territory = rng.choice(levels, 720, p=[0.10, 0.25, 0.08, 0.09, 0.24, 0.24])
    effects = {"T01": 0.10, "T02": 0.11, "T03": 0.12, "T04": -0.18, "T05": -0.17, "T06": -0.16}
    X = pd.DataFrame({"territory": territory})
    y = 0.5 + np.array([effects[value] for value in territory]) + rng.normal(0.0, 0.05, 720)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"territory": Categorical(base="most_exposed")},
    )
    model.fit(X, y)
    session = EditorSession.from_model(model, terms=["territory"], train_data=(X, y, None))

    session.select_levels("territory", ["T01", "T02", "T03"])
    session.replace_with_collapsed_levels("territory", method="fit")
    session.select_levels("territory", ["T04", "T05", "T06"])
    session.replace_with_collapsed_levels("territory", method="fit")
    session.select_levels("territory", ["T03", "T04"])
    refit = session.replace_with_collapsed_levels("territory", method="fit")

    grouping = refit.features["territory"]._grouping
    assert grouping.original_to_group["T01"] == "T01+T02"
    assert grouping.original_to_group["T02"] == "T01+T02"
    assert grouping.original_to_group["T03"] == "T03+T04"
    assert grouping.original_to_group["T04"] == "T03+T04"
    assert grouping.original_to_group["T05"] == "T05+T06"
    assert grouping.original_to_group["T06"] == "T05+T06"


def test_ungroup_selected_levels_removes_subset_from_collapsed_group():
    rng = np.random.default_rng(20260706)
    levels = [f"T{i:02d}" for i in range(1, 7)]
    territory = rng.choice(levels, 600, p=[0.08, 0.35, 0.08, 0.22, 0.17, 0.10])
    effects = {"T01": 0.11, "T02": 0.10, "T03": 0.13, "T04": -0.08, "T05": -0.12, "T06": 0.02}
    X = pd.DataFrame({"territory": territory})
    y = 0.5 + np.array([effects[value] for value in territory]) + rng.normal(0.0, 0.05, 600)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"territory": Categorical(base="most_exposed")},
    )
    model.fit(X, y)
    session = EditorSession.from_model(model, terms=["territory"], train_data=(X, y, None))

    session.select_levels("territory", ["T01", "T02", "T03"])
    session.replace_with_collapsed_levels("territory", method="fit")
    session.select_levels("territory", ["T02"])
    refit = session.replace_with_ungrouped_levels("territory", method="fit")

    grouping = refit.features["territory"]._grouping
    assert grouping.original_to_group["T01"] == "T01+T03"
    assert grouping.original_to_group["T03"] == "T01+T03"
    assert grouping.original_to_group["T02"] == "T02"
    from superglm.editor.payloads import session_payload

    groups = session_payload(session)["territory"]["level_groups"]
    assert groups == [{"label": "T01+T03", "indices": [0, 2], "levels": ["T01", "T03"]}]


def test_ungroup_preserves_symbolic_base_policy_to_avoid_display_uplift():
    from superglm.editor.payloads import session_payload

    rng = np.random.default_rng(20260708)
    levels = [f"T{i:02d}" for i in range(1, 7)]
    territory = rng.choice(levels, 900, p=[0.05, 0.34, 0.07, 0.18, 0.31, 0.05])
    effects = {"T01": 0.10, "T02": 0.12, "T03": -0.16, "T04": 0.00, "T05": 0.22, "T06": -0.08}
    X = pd.DataFrame({"territory": territory})
    y = 0.8 + np.array([effects[value] for value in territory]) + rng.normal(0.0, 0.05, 900)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"territory": Categorical(base="most_exposed")},
    )
    model.fit(X, y)
    session = EditorSession.from_model(model, terms=["territory"], train_data=(X, y, None))

    session.select_levels("territory", ["T01", "T03"])
    session.replace_with_collapsed_levels("territory", method="fit")
    session.select_levels("territory", ["T03"])
    refit = session.replace_with_ungrouped_levels("territory", method="fit")

    assert refit.features["territory"].base == "most_exposed"
    assert refit.features["territory"]._base_level == model.features["territory"]._base_level
    payload = session_payload(session)["territory"]
    assert payload["impact"]["weighted_mean_relativity"] == pytest.approx(1.0, abs=0.03)
    assert payload["impact"]["max_abs_link_delta"] < 0.08


def test_ungroup_last_collapsed_levels_removes_identity_grouping():
    from superglm.editor.payloads import session_payload

    rng = np.random.default_rng(20260709)
    levels = [f"T{i:02d}" for i in range(1, 5)]
    territory = rng.choice(levels, 500, p=[0.20, 0.35, 0.25, 0.20])
    X = pd.DataFrame({"territory": territory})
    y = 0.6 + (territory == "T03") * 0.2 + rng.normal(0.0, 0.05, 500)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"territory": Categorical(base="most_exposed")},
    )
    model.fit(X, y)
    session = EditorSession.from_model(model, terms=["territory"], train_data=(X, y, None))

    session.select_levels("territory", ["T01", "T04"])
    session.replace_with_collapsed_levels("territory", method="fit")
    session.select_levels("territory", ["T01", "T04"])
    restored = session.replace_with_ungrouped_levels("territory", method="fit")

    assert restored is model
    assert session.model is model
    assert getattr(restored.features["territory"], "_grouping", None) is None
    payload = session_payload(session)["territory"]
    assert payload["level_groups"] == []
    np.testing.assert_allclose(payload["y"], payload["original_y"])
    assert payload["impact"]["weighted_mean_relativity"] == pytest.approx(1.0)


def test_reorder_categorical_levels_is_display_only(editor_model):
    session = EditorSession.from_model(editor_model, terms=["region"])
    term = session.terms["region"]
    original_values = dict(zip(term.levels, term.edited_log_effect, strict=False))

    session.select_levels("region", ["C"])
    session.reorder_levels("region", target_index=1)

    assert term.levels == ["A", "C", "B"]
    assert term.x.tolist() == [0.0, 1.0, 2.0]
    assert session.selection("region").tolist() == [1]
    assert dict(zip(term.levels, term.edited_log_effect, strict=False)) == pytest.approx(
        original_values
    )
    assert session.edited_terms() == []


def test_reorder_multiple_categorical_levels_moves_selection_as_contiguous_block():
    values = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    session = EditorSession(
        model=None,
        terms={
            "territory": EditableTerm(
                name="territory",
                kind="categorical",
                levels=["A", "B", "C", "D"],
                x=np.arange(4, dtype=np.float64),
                original_log_effect=values.copy(),
                edited_log_effect=values.copy(),
            )
        },
    )
    session.select_levels("territory", ["A", "C"])

    session.reorder_levels("territory", target_index=2)

    assert session.terms["territory"].levels == ["B", "A", "C", "D"]
    assert session.selection("territory").tolist() == [1, 2]


def test_reorder_multiple_categorical_levels_can_move_to_far_right():
    values = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
    session = EditorSession(
        model=None,
        terms={
            "territory": EditableTerm(
                name="territory",
                kind="categorical",
                levels=["A", "B", "C", "D", "E"],
                x=np.arange(5, dtype=np.float64),
                original_log_effect=values.copy(),
                edited_log_effect=values.copy(),
            )
        },
    )
    session.select_levels("territory", ["A", "C"])

    session.reorder_levels("territory", target_index=5)

    assert session.terms["territory"].levels == ["B", "D", "E", "A", "C"]
    assert session.selection("territory").tolist() == [3, 4]


def test_save_load_preserves_display_only_level_order(editor_model, tmp_path):
    session = EditorSession.from_model(editor_model, terms=["region"])
    session.select_levels("region", ["C"])
    session.reorder_levels("region", target_index=1)
    path = tmp_path / "editor-session.json"

    session.save(path)
    loaded = EditorSession.load(path, model=editor_model)

    assert loaded.terms["region"].levels == ["A", "C", "B"]
    assert loaded.selection("region").tolist() == [1]
    assert loaded.level_order_changed("region")


def test_reprofile_distribution_parameter_dispatches_to_model(
    editor_model, editor_frame, monkeypatch
):
    X, y = editor_frame
    session = EditorSession.from_model(
        editor_model,
        terms=["x_spline"],
        train_data=(X, y, None),
    )
    calls: list[dict[str, object]] = []

    replaced: list[object] = []

    def fake_estimate_p(self, X_arg, y_arg, sample_weight=None, offset=None, **kwargs):
        calls.append(
            {
                "parameter": "p",
                "model": self,
                "X": X_arg,
                "y": y_arg,
                "sample_weight": sample_weight,
                "offset": offset,
                "kwargs": kwargs,
            }
        )
        return "p-result"

    def fake_estimate_theta(self, X_arg, y_arg, sample_weight=None, offset=None, **kwargs):
        calls.append(
            {
                "parameter": "theta",
                "model": self,
                "X": X_arg,
                "y": y_arg,
                "sample_weight": sample_weight,
                "offset": offset,
                "kwargs": kwargs,
            }
        )
        return "theta-result"

    def fake_replace(model):
        replaced.append(model)
        return session

    monkeypatch.setattr(type(editor_model), "estimate_p", fake_estimate_p)
    monkeypatch.setattr(type(editor_model), "estimate_theta", fake_estimate_theta)
    monkeypatch.setattr(session, "replace_in_force_model", fake_replace)

    p_result = session.reprofile_distribution("tweedie_p", method="grid", n_grid=7)
    theta_result = session.reprofile_distribution("nb2_theta", theta_bounds=(0.2, 20.0))

    assert p_result == "p-result"
    assert theta_result == "theta-result"
    assert calls[0]["parameter"] == "p"
    assert calls[0]["model"] is not editor_model
    assert calls[0]["X"] is X
    assert calls[0]["y"] is y
    assert calls[0]["kwargs"] == {"method": "grid", "n_grid": 7, "fit_mode": "inherit"}
    assert calls[1]["parameter"] == "theta"
    assert calls[1]["model"] is not editor_model
    assert calls[1]["kwargs"] == {"theta_bounds": (0.2, 20.0)}
    assert replaced == [calls[0]["model"], calls[1]["model"]]


def test_reprofile_distribution_uses_cloned_in_force_model():
    rng = np.random.default_rng(20260711)
    n = 120
    x = rng.uniform(0.0, 1.0, n)
    X = pd.DataFrame({"x": x})
    mu = np.exp(0.2 + 0.4 * x)
    y = generate_tweedie_cpg(n, mu=mu, phi=0.5, p=1.45, rng=rng)
    model = SuperGLM(
        family=families.tweedie(p=1.3),
        selection_penalty=0.0,
        spline_penalty=0.1,
        features={"x": Spline(n_knots=5)},
    )
    model.fit(X, y)
    original_beta = model.result.beta.copy()
    original_intercept = model.result.intercept
    session = EditorSession.from_model(model, terms=["x"], train_data=(X, y, None))

    session.reprofile_distribution(
        "tweedie_p",
        fit_mode="fit",
        method="grid",
        grid=np.array([1.25, 1.45]),
    )

    assert session.model is not model
    assert session.reference_model is model
    assert model.family.p == 1.3
    np.testing.assert_allclose(model.result.beta, original_beta)
    assert model.result.intercept == original_intercept


def test_reprofile_distribution_rejects_pending_manual_edits(editor_model, editor_frame):
    X, y = editor_frame
    session = EditorSession.from_model(
        editor_model,
        terms=["x_spline"],
        train_data=(X, y, None),
    )
    session.select_indices("x_spline", [0, 1])
    session.shift("x_spline", 0.1)

    with pytest.raises(RuntimeError, match="manual coefficient edits"):
        session.reprofile_distribution("tweedie_p", method="grid", grid=np.array([1.25, 1.45]))


def test_tweedie_profile_trace_can_include_candidate_fit_curves():
    rng = np.random.default_rng(20260706)
    n = 120
    x = rng.uniform(0.0, 1.0, n)
    X = pd.DataFrame({"x": x})
    mu = np.exp(0.2 + 0.4 * np.sin(2 * np.pi * x))
    y = generate_tweedie_cpg(n, mu=mu, phi=0.5, p=1.45, rng=rng)
    model = SuperGLM(
        family=families.tweedie(p=1.3),
        selection_penalty=0.0,
        spline_penalty=0.1,
        features={"x": Spline(n_knots=5)},
    )

    result = model.estimate_p(
        X,
        y,
        fit_mode="fit",
        method="grid",
        grid=np.array([1.25, 1.45, 1.65]),
        trace_iterations=True,
    )

    assert "fit_trace" in result.search_trace.columns
    traces = result.search_trace["fit_trace"].tolist()
    assert all(isinstance(trace, list) for trace in traces)
    assert any(len(trace) >= 1 for trace in traces)
    first_trace = next(trace for trace in traces if trace)
    assert {"iteration", "loss"}.issubset(first_trace[0])


def test_tweedie_reml_profile_trace_can_include_candidate_objective_curves():
    rng = np.random.default_rng(20260707)
    n = 120
    x = rng.uniform(0.0, 1.0, n)
    X = pd.DataFrame({"x": x})
    mu = np.exp(0.2 + 0.4 * np.sin(2 * np.pi * x))
    y = generate_tweedie_cpg(n, mu=mu, phi=0.5, p=1.45, rng=rng)
    model = SuperGLM(
        family=families.tweedie(p=1.3),
        selection_penalty=0.0,
        spline_penalty=0.1,
        features={"x": Spline(n_knots=5)},
    )
    model.fit_reml(X, y, max_reml_iter=5)

    result = model.estimate_p(
        X,
        y,
        fit_mode="inherit",
        method="grid",
        grid=np.array([1.25, 1.45]),
        trace_iterations=True,
    )

    traces = result.search_trace["fit_trace"].tolist()
    assert all(isinstance(trace, list) for trace in traces)
    assert any(len(trace) >= 1 for trace in traces)
    assert set(result.search_trace["fit_trace_kind"]) == {"REML objective"}


def test_widget_profile_distribution_forwards_options_and_returns_trace(
    editor_model, editor_frame, monkeypatch
):
    X, y = editor_frame
    session = EditorSession.from_model(
        editor_model,
        terms=["x_spline"],
        train_data=(X, y, None),
    )
    widget = session.widget()
    calls: list[dict[str, object]] = []

    class FakeTrace:
        def to_dict(self, orient):
            assert orient == "records"
            return [{"step": 0, "p": 1.42, "phi": 0.3, "nll": 0.12, "source": "brent"}]

    class FakeResult:
        p_hat = 1.42
        phi_hat = 0.3
        nll = 0.12
        method = "brent"
        phi_method = "mle"
        search_trace = FakeTrace()

        def ci(self, alpha=0.05):
            assert alpha == 0.05
            return (1.31, 1.53)

    def fake_reprofile(parameter, **kwargs):
        calls.append({"parameter": parameter, "kwargs": kwargs})
        return FakeResult()

    monkeypatch.setattr(session, "reprofile_distribution", fake_reprofile)
    monkeypatch.setattr(
        "superglm.editor.widget.summary_payload",
        lambda _widget, source: {"available": True, "source": source, "compact": {"model": {}}},
    )

    try:
        payload = widget._profile_distribution(
            "tweedie_p",
            method="brent",
            phi_method="mle",
            xatol=0.002,
        )
    finally:
        widget.close()

    assert calls == [
        {
            "parameter": "tweedie_p",
            "kwargs": {"method": "brent", "phi_method": "mle", "xatol": 0.002},
        }
    ]
    assert payload["profile_trace"] == [
        {"step": 0, "p": 1.42, "phi": 0.3, "nll": 0.12, "source": "brent"}
    ]
    assert payload["profile_estimate"] == {
        "parameter": "p",
        "label": "p_hat",
        "value": 1.42,
        "ci_low": 1.31,
        "ci_high": 1.53,
        "objective": 0.12,
        "objective_label": "loss",
        "lower_is_better": True,
    }


def test_widget_profile_distribution_job_reports_live_trace(
    editor_model, editor_frame, monkeypatch
):
    X, y = editor_frame
    session = EditorSession.from_model(
        editor_model,
        terms=["x_spline"],
        train_data=(X, y, None),
    )
    widget = session.widget()

    class FakeTrace:
        def to_dict(self, orient):
            assert orient == "records"
            return [{"step": 1, "p": 1.5, "phi": 0.2, "nll": 0.1, "source": "final"}]

    class FakeResult:
        search_trace = FakeTrace()

    def fake_reprofile(parameter, **kwargs):
        trace_callback = kwargs["trace_callback"]
        trace_callback({"step": 0, "p": 1.3, "phi": 0.22, "nll": 0.14, "source": "brent"})
        return FakeResult()

    monkeypatch.setattr(session, "reprofile_distribution", fake_reprofile)
    monkeypatch.setattr(
        "superglm.editor.widget.summary_payload",
        lambda _widget, source: {"available": True, "source": source, "compact": {"model": {}}},
    )

    try:
        started = widget._start_profile_distribution_job(
            "tweedie_p",
            method="brent",
            phi_method="mle",
            xatol=0.001,
        )
        job_id = started["job_id"]
        status = widget._profile_distribution_status(job_id, wait=True)
    finally:
        widget.close()

    assert started["status"] in {"running", "complete"}
    assert status["status"] == "complete"
    assert status["options"]["xatol"] == 0.001
    assert status["trace"] == [
        {"step": 0, "p": 1.3, "phi": 0.22, "nll": 0.14, "source": "brent"},
        {"step": 1, "p": 1.5, "phi": 0.2, "nll": 0.1, "source": "final"},
    ]
    assert status["result"]["source"] == "in_force"


def test_widget_profile_distribution_job_reports_finalizing_phase(
    editor_model, editor_frame, monkeypatch
):
    X, y = editor_frame
    session = EditorSession.from_model(
        editor_model,
        terms=["x_spline"],
        train_data=(X, y, None),
    )
    widget = session.widget()
    summary_started = threading.Event()
    release_summary = threading.Event()

    class FakeTrace:
        def to_dict(self, orient):
            assert orient == "records"
            return [{"step": 1, "p": 1.5, "phi": 0.2, "nll": 0.1, "source": "final"}]

    class FakeResult:
        search_trace = FakeTrace()

    def fake_reprofile(parameter, **kwargs):
        trace_callback = kwargs["trace_callback"]
        trace_callback({"step": 0, "p": 1.3, "phi": 0.22, "nll": 0.14, "source": "brent"})
        return FakeResult()

    def blocking_summary(_widget, source):
        summary_started.set()
        assert release_summary.wait(timeout=2.0)
        return {"available": True, "source": source, "compact": {"model": {}}}

    monkeypatch.setattr(session, "reprofile_distribution", fake_reprofile)
    monkeypatch.setattr("superglm.editor.widget.summary_payload", blocking_summary)

    try:
        started = widget._start_profile_distribution_job(
            "tweedie_p",
            method="brent",
            phi_method="mle",
            xatol=0.001,
        )
        job_id = started["job_id"]
        assert summary_started.wait(timeout=2.0)
        status = widget._profile_distribution_status(job_id)
        release_summary.set()
        complete = widget._profile_distribution_status(job_id, wait=True)
    finally:
        release_summary.set()
        widget.close()

    assert status["status"] == "running"
    assert status["phase"] == "finalizing"
    assert status["trace"] == [{"step": 0, "p": 1.3, "phi": 0.22, "nll": 0.14, "source": "brent"}]
    assert complete["status"] == "complete"
    assert complete["phase"] == "complete"


def test_widget_profile_distribution_job_reports_best_parameter_and_refit_phase(
    editor_model, editor_frame, monkeypatch
):
    X, y = editor_frame
    session = EditorSession.from_model(
        editor_model,
        terms=["x_spline"],
        train_data=(X, y, None),
    )
    widget = session.widget()
    refit_started = threading.Event()
    release_refit = threading.Event()

    class FakeTrace:
        def to_dict(self, orient):
            assert orient == "records"
            return [{"step": 1, "p": 1.5, "phi": 0.2, "nll": 0.1, "source": "final"}]

    class FakeResult:
        p_hat = 1.5
        phi_hat = 0.2
        nll = 0.1
        search_trace = FakeTrace()

        def ci(self, alpha=0.05):
            assert alpha == 0.05
            return (1.4, 1.6)

    def fake_reprofile(parameter, **kwargs):
        kwargs["trace_callback"]({"step": 0, "p": 1.3, "phi": 0.22, "nll": 0.14, "source": "brent"})
        kwargs["progress_callback"](
            "best_found",
            {
                "profile_estimate": {
                    "parameter": "p",
                    "label": "p_hat",
                    "value": 1.5,
                    "objective": 0.1,
                    "objective_label": "loss",
                    "lower_is_better": True,
                }
            },
        )
        kwargs["progress_callback"]("final_refit")
        refit_started.set()
        assert release_refit.wait(timeout=2.0)
        return FakeResult()

    monkeypatch.setattr(session, "reprofile_distribution", fake_reprofile)
    monkeypatch.setattr(
        "superglm.editor.widget.summary_payload",
        lambda _widget, source: {"available": True, "source": source, "compact": {"model": {}}},
    )

    try:
        started = widget._start_profile_distribution_job(
            "tweedie_p",
            method="brent",
            phi_method="mle",
            xatol=0.001,
        )
        job_id = started["job_id"]
        assert refit_started.wait(timeout=2.0)
        status = widget._profile_distribution_status(job_id)
        release_refit.set()
        complete = widget._profile_distribution_status(job_id, wait=True)
    finally:
        release_refit.set()
        widget.close()

    assert status["status"] == "running"
    assert status["phase"] == "final_refit"
    assert status["profile_estimate"]["value"] == 1.5
    assert status["profile_estimate"]["ci_low"] is None
    assert complete["status"] == "complete"
    assert complete["phase"] == "complete"
    assert complete["result"]["profile_estimate"]["ci_low"] == 1.4
    assert complete["result"]["profile_estimate"]["ci_high"] == 1.6


def test_reprofile_distribution_parameter_rejects_unknown_parameter(editor_model, editor_frame):
    X, y = editor_frame
    session = EditorSession.from_model(editor_model, terms=["x_spline"], train_data=(X, y, None))

    with pytest.raises(ValueError, match="parameter must be"):
        session.reprofile_distribution("dispersion")


def test_reordered_categorical_levels_persist_across_collapse_refit(editor_model):
    session = EditorSession.from_model(editor_model, terms=["region"])

    session.select_levels("region", ["C"])
    session.reorder_levels("region", target_index=1)
    assert session.terms["region"].levels == ["A", "C", "B"]

    session.select_levels("region", ["C", "B"])
    session.replace_with_collapsed_levels("region", method="fit")

    assert session.terms["region"].levels == ["A", "C", "B"]


def test_reordered_categorical_levels_persist_after_ungroup_restores_reference_model(editor_model):
    session = EditorSession.from_model(editor_model, terms=["region"])
    session.select_levels("region", ["C"])
    session.reorder_levels("region", target_index=1)
    assert session.terms["region"].levels == ["A", "C", "B"]

    session.select_levels("region", ["C", "B"])
    session.replace_with_collapsed_levels("region", method="fit")
    session.select_levels("region", ["C", "B"])
    session.replace_with_ungrouped_levels("region", method="fit")

    assert session.model is session.reference_model
    assert session.terms["region"].levels == ["A", "C", "B"]


def test_reset_level_order_restores_model_order_after_visual_reorder(editor_model):
    session = EditorSession.from_model(editor_model, terms=["region"])
    session.select_levels("region", ["C"])
    session.reorder_levels("region", target_index=1)

    session.reset_level_order("region")

    assert session.terms["region"].levels == ["A", "B", "C"]
    assert session.terms["region"].x.tolist() == [0.0, 1.0, 2.0]
    assert session.selection("region").tolist() == [2]


def test_session_payload_compares_in_force_against_reference_model_after_collapse(editor_model):
    from superglm.editor.payloads import session_payload

    session = EditorSession.from_model(editor_model, terms=["region"])
    session.select_levels("region", ["B", "C"])
    session.replace_with_collapsed_levels("region", method="fit")

    payload = session_payload(session)
    region = payload["region"]
    reference = editor_model.term_inference("region", with_se=True)
    in_force = session.model.term_inference("region", with_se=True)
    export_df = session.model.plot_data("region")["terms"][0]["effect"]

    np.testing.assert_allclose(region["original_y"], reference.relativity)
    np.testing.assert_allclose(region["y"], in_force.relativity)
    expected_mean_relativity = np.average(
        np.asarray(region["y"]) / np.asarray(region["original_y"]),
        weights=np.asarray(region["weights"]),
    )
    assert region["impact"]["weighted_mean_relativity"] == pytest.approx(expected_mean_relativity)
    assert expected_mean_relativity != pytest.approx(1.0)
    assert region["levels"] == ["A", "B", "C"]
    assert region["y"][1] == pytest.approx(region["y"][2])
    assert region["level_groups"] == [{"label": "B+C", "indices": [1, 2], "levels": ["B", "C"]}]
    assert export_df["level"].tolist() == ["A", "B", "C"]
    assert export_df.loc[export_df["level"] == "B", "relativity"].item() == pytest.approx(
        export_df.loc[export_df["level"] == "C", "relativity"].item()
    )


def test_categorical_level_edit_changes_copied_predictions(editor_model, editor_frame):
    X, _ = editor_frame
    session = EditorSession.from_model(editor_model, terms=["region"])
    session.select_levels("region", ["B"])
    session.shift("region", 0.5)

    edited = session.to_model()

    mask = X["region"] == "B"
    eta_delta = edited._predict_eta_exact(X) - editor_model._predict_eta_exact(X)
    assert np.mean(eta_delta[mask]) > 0.3
    assert np.mean(np.abs(eta_delta[~mask])) < 0.1


def test_session_extracts_all_supported_1d_main_effects(editor_model):
    session = EditorSession.from_model(editor_model)

    assert list(session.terms) == ["x_spline", "x_poly", "x_num", "region", "band"]
    assert session.terms["x_poly"].x is not None
    assert session.terms["x_num"].edited_log_effect.shape == (1,)
    assert session.terms["band"].levels == ["low", "medium", "high"]


def test_session_uses_fit_row_counts_for_term_weights(editor_model, editor_frame):
    X, _ = editor_frame
    session = EditorSession.from_model(editor_model, terms=["region"])

    weights = session.terms["region"].weights
    expected = np.array([(X["region"] == level).sum() for level in ["A", "B", "C"]])

    np.testing.assert_allclose(weights, expected)


def test_session_uses_fit_sample_weight_for_term_weights():
    X = pd.DataFrame({"region": ["A", "A", "B", "B", "C", "C"], "x": np.arange(6)})
    sample_weight = np.array([1.0, 2.0, 0.5, 1.5, 3.0, 4.0])
    y = np.array([0.1, 0.0, 0.4, 0.3, -0.2, -0.1])
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"region": Categorical(base="first"), "x": Numeric()},
    )
    model.fit(X, y, sample_weight=sample_weight)

    session = EditorSession.from_model(model, terms=["region"])

    np.testing.assert_allclose(session.terms["region"].weights, [3.0, 2.0, 7.0])


def test_widget_state_exposes_ci_bands_and_single_point_centering(editor_model):
    session = EditorSession.from_model(editor_model, terms=["x_num", "x_spline"])
    widget = session.widget()
    try:
        state = _get_json(f"{widget.url}/state")
        numeric = state["terms"]["x_num"]
        spline = state["terms"]["x_spline"]

        assert numeric["x_domain"][0] < numeric["x"][0] < numeric["x_domain"][1]
        assert numeric["y_label"] == "relativity"
        np.testing.assert_allclose(numeric["y"], np.exp(session.terms["x_num"].edited_log_effect))
        assert spline["ci_lower_y"] is not None
        assert spline["ci_upper_y"] is not None
        assert len(spline["ci_lower_y"]) == spline["n_points"]
        np.testing.assert_allclose(spline["y"], np.exp(session.terms["x_spline"].edited_log_effect))
        np.testing.assert_allclose(
            spline["original_y"],
            np.exp(session.terms["x_spline"].original_log_effect),
        )
    finally:
        widget.close()


def test_widget_state_exposes_density_for_continuous_and_bars_for_levels(editor_model):
    session = EditorSession.from_model(editor_model, terms=["x_spline", "region"])
    widget = session.widget()
    try:
        state = _get_json(f"{widget.url}/state")
        spline_exposure = state["terms"]["x_spline"]["exposure"]
        region_exposure = state["terms"]["region"]["exposure"]

        assert spline_exposure["kind"] == "density"
        assert len(spline_exposure["x"]) == state["terms"]["x_spline"]["n_points"]
        assert len(spline_exposure["y"]) == state["terms"]["x_spline"]["n_points"]
        assert max(spline_exposure["y"]) > min(spline_exposure["y"])
        assert region_exposure["kind"] == "bars"
        assert len(region_exposure["y"]) == state["terms"]["region"]["n_points"]
    finally:
        widget.close()


def test_widget_state_caps_continuous_handles_for_large_grids(editor_model):
    session = EditorSession.from_model(editor_model, terms=["x_spline", "region"], n_points=1200)
    widget = session.widget()
    try:
        state = _get_json(f"{widget.url}/state")
        spline = state["terms"]["x_spline"]
        region = state["terms"]["region"]

        assert spline["n_points"] == 1200
        assert 100 < len(spline["handle_indices"]) < spline["n_points"]
        assert spline["handle_indices"][0] == 0
        assert spline["handle_indices"][-1] == spline["n_points"] - 1
        assert region["handle_indices"] == list(range(region["n_points"]))
    finally:
        widget.close()


def test_numeric_polynomial_and_ordered_step_edits_apply(editor_model, editor_frame):
    X, _ = editor_frame
    session = EditorSession.from_model(editor_model, terms=["x_num", "x_poly", "band"])

    session.select_indices("x_num", [0])
    session.shift("x_num", 0.1)
    session.select_x("x_poly", -1.0, 1.0)
    session.shift("x_poly", -0.15)
    session.select_levels("band", ["high"])
    session.shift("band", 0.25)

    edited = session.to_model()

    eta_delta = edited._predict_eta_exact(X) - editor_model._predict_eta_exact(X)
    assert np.max(np.abs(eta_delta)) > 0.1
    assert np.mean(eta_delta[X["band"] == "high"]) > np.mean(eta_delta[X["band"] == "low"])


def test_ordered_categorical_spline_level_edit_applies_to_model_copy():
    rng = np.random.default_rng(1234)
    levels = ["young", "mid", "senior", "older"]
    n = 360
    X = pd.DataFrame(
        {
            "age_band": rng.choice(levels, n, p=[0.25, 0.35, 0.25, 0.15]),
            "x": rng.normal(size=n),
        }
    )
    level_effect = {"young": -0.1, "mid": 0.0, "senior": 0.2, "older": 0.35}
    y = (
        np.array([level_effect[level] for level in X["age_band"]])
        + 0.1 * X["x"].to_numpy()
        + rng.normal(0.0, 0.05, n)
    )
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={
            "age_band": OrderedCategorical(order=levels, basis="spline", n_knots=3),
            "x": Numeric(),
        },
    )
    model.fit(X, y)

    session = EditorSession.from_model(model, terms=["age_band"])
    session.select_levels("age_band", ["older"])
    session.shift("age_band", 0.3)
    edited = session.to_model()

    eta_delta = edited._predict_eta_exact(X) - model._predict_eta_exact(X)
    assert np.mean(eta_delta[X["age_band"] == "older"]) > 0.1


def test_save_load_roundtrip(editor_model, tmp_path):
    session = EditorSession.from_model(editor_model, terms=["x_spline"])
    session.select_indices("x_spline", [1, 2])
    session.shift("x_spline", 0.2)
    path = tmp_path / "edits.json"

    session.save(path)
    loaded = EditorSession.load(path, model=editor_model)

    np.testing.assert_allclose(
        loaded.terms["x_spline"].edited_log_effect,
        session.terms["x_spline"].edited_log_effect,
    )
    assert loaded.history[-1].operation == "shift"


def test_widget_import_is_lazy():
    import sys

    import superglm.editor

    assert "anywidget" not in sys.modules
    assert "ipympl" not in sys.modules
    assert "ipywidgets" not in sys.modules
    assert hasattr(superglm.editor, "EditorSession")


def test_editor_asset_reader_rejects_path_traversal(monkeypatch):
    from superglm.editor import assets

    assert assets.app_asset_content_type("styles.css").startswith("text/css")

    def fail_resource_lookup(_package):
        raise AssertionError("invalid asset paths must be rejected before resource lookup")

    monkeypatch.setattr(assets, "files", fail_resource_lookup)
    for path in [
        "",
        ".",
        "..",
        "../widget.py",
        r"..\widget.py",
        r"nested\..\asset.js",
        "/styles.css",
        "styles.css/",
        "nested//asset.js",
    ]:
        with pytest.raises(FileNotFoundError):
            assets.read_app_asset(path)


def test_widget_renders_plain_iframe_app(editor_model):
    session = EditorSession.from_model(editor_model, terms=["x_spline"])

    widget = session.widget()
    try:
        html = widget._repr_html_()

        assert "<iframe" in html
        assert "127.0.0.1" in html
        assert "token=" in html
        assert not hasattr(widget, "_model_module")
        assert widget.selected_term == "x_spline"
        assert widget.terms["x_spline"]["n_points"] == session.terms["x_spline"].size

        payload = _get_json(f"{widget.url}/state")
        assert payload["selected_term"] == "x_spline"
        assert payload["terms"]["x_spline"]["n_points"] == session.terms["x_spline"].size
    finally:
        widget.close()


def test_widget_http_rejects_missing_editor_token(editor_model):
    session = EditorSession.from_model(editor_model, terms=["x_spline"])
    widget = session.widget()
    try:
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(f"{widget.url}/state", timeout=5)
        assert excinfo.value.code == 403
        assert _get_json(f"{widget.url}/state")["selected_term"] == "x_spline"
    finally:
        widget.close()


def test_widget_favicon_does_not_emit_browser_404(editor_model):
    session = EditorSession.from_model(editor_model, terms=["x_spline"])
    widget = session.widget()
    try:
        request = urllib.request.Request(f"{widget.url}/favicon.ico", method="GET")
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                assert response.status == 204
        except urllib.error.HTTPError as exc:  # pragma: no cover - assertion path
            pytest.fail(f"favicon request returned HTTP {exc.code}")
    finally:
        widget.close()


def test_widget_http_selection_and_average_updates_session(editor_model):
    session = EditorSession.from_model(editor_model, terms=["x_spline"])
    widget = session.widget()
    try:
        term = session.terms["x_spline"]
        indices = [8, 9, 10, 11]
        before = term.edited_log_effect[indices].copy()

        _post_json(f"{widget.url}/select", {"term": "x_spline", "indices": indices})
        assert session.selection("x_spline").tolist() == indices

        _post_json(f"{widget.url}/op", {"operation": "average"})
        expected = np.log(np.average(np.exp(before), weights=term.weights[indices]))
        np.testing.assert_allclose(term.edited_log_effect[indices], expected)

        term.edited_log_effect[indices] = np.array([0.0, 1.0, -0.5, 2.0])
        _post_json(f"{widget.url}/op", {"operation": "linearise"})
        linear_target = np.interp(
            term.x[indices],
            [term.x[indices[0]], term.x[indices[-1]]],
            [0.0, 2.0],
        )
        expected_linear = 0.5 * np.array([0.0, 1.0, -0.5, 2.0]) + 0.5 * linear_target
        np.testing.assert_allclose(term.edited_log_effect[indices], expected_linear)

        term.edited_log_effect[indices] = np.array([0.2, 1.0, -0.4, 0.8])
        _post_json(f"{widget.url}/op", {"operation": "level_left"})
        np.testing.assert_allclose(term.edited_log_effect[indices], 0.2)

        term.edited_log_effect[indices] = np.array([0.2, 1.0, -0.4, 0.8])
        _post_json(f"{widget.url}/op", {"operation": "level_right"})
        np.testing.assert_allclose(term.edited_log_effect[indices], 0.8)

        term.edited_log_effect[indices] = np.array([0.2, 1.0, -0.4, 0.8])
        _post_json(f"{widget.url}/op", {"operation": "snap_highest"})
        np.testing.assert_allclose(term.edited_log_effect[indices], 1.0)

        term.edited_log_effect[indices] = np.array([0.2, 1.0, -0.4, 0.8])
        _post_json(f"{widget.url}/op", {"operation": "snap_lowest"})
        np.testing.assert_allclose(term.edited_log_effect[indices], -0.4)
    finally:
        widget.close()


def test_widget_http_collapse_levels_refit_updates_summary_source(editor_model):
    session = EditorSession.from_model(editor_model, terms=["region"])
    widget = session.widget()
    try:
        _post_json(f"{widget.url}/select", {"term": "region", "indices": [1, 2]})
        payload = _post_json(
            f"{widget.url}/collapse_levels",
            {"term": "region", "method": "fit"},
        )

        assert payload["available"] is True
        assert payload["source"] == "in_force"
        assert payload["label"] == "In-force edit model"
        assert widget.session.model is not editor_model
        grouping = widget.session.model.features["region"]._grouping
        assert grouping.original_to_group["B"] == "B+C"
        assert grouping.original_to_group["C"] == "B+C"

        state = _get_json(f"{widget.url}/state")
        assert "model_source" not in state
        assert "model_sources" not in state
        assert state["terms"]["region"]["y"][1] == pytest.approx(state["terms"]["region"]["y"][2])

        summary = _post_json(f"{widget.url}/summary", {"source": "in_force"})
        assert summary["available"] is True
        assert summary["source"] == "in_force"
        assert "Current editable model" in summary["note"]
    finally:
        widget.close()


def test_widget_http_uncollapse_levels_restores_previous_model(editor_model):
    session = EditorSession.from_model(editor_model, terms=["region"])
    widget = session.widget()
    try:
        _post_json(f"{widget.url}/select", {"term": "region", "indices": [1, 2]})
        _post_json(f"{widget.url}/collapse_levels", {"term": "region", "method": "fit"})
        collapsed_state = _get_json(f"{widget.url}/state")
        assert collapsed_state["can_uncollapse_levels"] is True
        assert widget.session.model is not editor_model
        assert widget.session.model.features["region"]._grouping.original_to_group["B"] == "B+C"

        payload = _post_json(f"{widget.url}/uncollapse_levels", {})

        assert payload["available"] is True
        assert payload["source"] == "in_force"
        assert widget.session.model is editor_model
        assert getattr(widget.session.model.features["region"], "_grouping", None) is None
        state = _get_json(f"{widget.url}/state")
        assert state["can_uncollapse_levels"] is False
        assert state["last_collapse"] is None
        assert state["terms"]["region"]["y"][1] != pytest.approx(state["terms"]["region"]["y"][2])
    finally:
        widget.close()


def test_widget_http_selected_summary_uses_in_force_without_model_selector(editor_model):
    session = EditorSession.from_model(editor_model, terms=["region"])
    widget = session.widget()
    try:
        state = _get_json(f"{widget.url}/state")
        assert "model_source" not in state
        assert "model_sources" not in state

        summary = _post_json(f"{widget.url}/summary", {"source": "selected"})
        assert summary["source"] == "in_force"

        original = _post_json(f"{widget.url}/summary", {"source": "original"})
        assert original["source"] == "original"
    finally:
        widget.close()


def test_widget_http_drag_and_reset_updates_session(editor_model):
    session = EditorSession.from_model(editor_model, terms=["x_spline"])
    widget = session.widget()
    try:
        term = session.terms["x_spline"]
        original = term.original_log_effect.copy()

        _post_json(
            f"{widget.url}/drag",
            {"term": "x_spline", "indices": [5], "values": [float(np.exp(original[5]) + 0.25)]},
        )
        assert session.selection("x_spline").tolist() == [5]
        assert np.exp(term.edited_log_effect[5]) == pytest.approx(np.exp(original[5]) + 0.25)

        _post_json(f"{widget.url}/op", {"operation": "reset"})
        assert term.edited_log_effect[5] == pytest.approx(original[5])
    finally:
        widget.close()


def test_widget_http_control_handle_updates_session(editor_model):
    session = EditorSession.from_model(editor_model, terms=["x_spline"])
    widget = session.widget()
    try:
        term = session.terms["x_spline"]
        state = _post_json(f"{widget.url}/control_count", {"term": "x_spline", "count": 6})
        controls = session.control_points("x_spline", n_handles=6)
        assert len(state["terms"]["x_spline"]["controls"]["x"]) == 6
        assert state["terms"]["x_spline"]["controls"]["count"] == 6
        assert state["terms"]["x_spline"]["controls"]["basis_index"] == (
            controls["basis_index"].astype(int).tolist()
        )
        payload_controls = state["terms"]["x_spline"]["controls"]
        payload_basis = np.asarray(payload_controls["basis"], dtype=np.float64)
        raw_basis = editor_model._specs["x_spline"]._raw_basis_matrix(term.x)
        if hasattr(raw_basis, "toarray"):
            raw_basis = raw_basis.toarray()
        basis_index = int(payload_controls["basis_index"][2])
        assert payload_basis.shape == (6, term.size)
        np.testing.assert_allclose(payload_basis[2], np.asarray(raw_basis)[:, basis_index])

        before = term.edited_log_effect.copy()
        handle_index = int(controls["x"].size // 2)
        target_rel = float(np.exp(controls["log_effect"][handle_index] + 0.25))

        state = _post_json(
            f"{widget.url}/control",
            {"term": "x_spline", "handle_index": handle_index, "value": target_rel},
        )

        assert np.max(np.abs(term.edited_log_effect - before)) > 0.02
        assert len(state["terms"]["x_spline"]["controls"]["x"]) == 6
        assert state["terms"]["x_spline"]["controls"]["x"][handle_index] == pytest.approx(
            controls["x"][handle_index]
        )
        assert state["terms"]["x_spline"]["controls"]["y"][handle_index] == pytest.approx(
            target_rel,
            rel=0.08,
        )
    finally:
        widget.close()


def test_widget_http_select_all_and_impact_updates_session(editor_model):
    session = EditorSession.from_model(editor_model, terms=["x_spline"])
    widget = session.widget()
    try:
        term = session.terms["x_spline"]

        state = _post_json(f"{widget.url}/op", {"operation": "select_all"})
        assert session.selection("x_spline").tolist() == list(range(term.size))
        assert state["terms"]["x_spline"]["impact"]["weighted_mean_relativity"] == pytest.approx(
            1.0
        )

        state = _post_json(
            f"{widget.url}/drag",
            {
                "term": "x_spline",
                "indices": [0],
                "values": [float(np.exp(term.original_log_effect[0]) + 0.2)],
            },
        )
        impact = state["terms"]["x_spline"]["impact"]
        assert impact["selected_weight_share"] > 0.0
        assert impact["weighted_mean_relativity"] > 1.0
    finally:
        widget.close()


def test_widget_http_metrics_recomputes_fit_metric_for_edited_copy(editor_model):
    session = EditorSession.from_model(editor_model, terms=["x_spline"])
    widget = session.widget()
    try:
        baseline = _post_json(f"{widget.url}/metrics", {"metric": "deviance"})
        assert baseline["metric"] == "deviance"
        assert baseline["available"] is True
        assert baseline["original"] == pytest.approx(baseline["edited"])
        assert baseline["metrics"]["original"]["effective_df"] == pytest.approx(
            editor_model.result.effective_df
        )
        assert baseline["metrics"]["edited"]["effective_df"] == pytest.approx(
            editor_model.result.effective_df
        )

        _post_json(f"{widget.url}/select", {"term": "x_spline", "indices": [20, 21, 22]})
        _post_json(f"{widget.url}/op", {"operation": "shift_up"})
        changed = _post_json(f"{widget.url}/metrics", {"metric": "deviance"})

        assert changed["available"] is True
        assert changed["edited"] != pytest.approx(changed["original"])
        assert "metrics" in changed
        assert "aic" in changed["metrics"]["edited"]
        assert "effective_df" in changed["metrics"]["edited"]
    finally:
        widget.close()


def test_editor_session_accepts_plain_evaluation_tuples_and_cv_report(editor_model, editor_frame):
    from superglm.editor.metrics import metrics_payload
    from superglm.editor.reports import validation_report_payload

    X, y = editor_frame
    X_train, y_train = X.iloc[:300], y[:300]
    X_val, y_val = X.iloc[300:380], y[300:380]
    X_test, y_test = X.iloc[380:], y[380:]
    w_train = np.linspace(0.8, 1.2, len(X_train))
    w_val = np.linspace(1.0, 1.5, len(X_val))
    w_test = np.linspace(0.7, 1.1, len(X_test))
    cv_report = {
        "method": "KFold",
        "folds": 5,
        "rows": [{"candidate": "k=8", "mean_val_deviance": 0.0148, "se_val_deviance": 0.0002}],
    }

    session = EditorSession.from_model(
        editor_model,
        terms=["x_spline"],
        train_data=(X_train, y_train, w_train),
        validation_data=(X_val, y_val, w_val),
        test_data=(X_test, y_test, w_test),
        cv_report=cv_report,
    )

    metric_payload = metrics_payload(session, "deviance")

    assert metric_payload["available"] is True
    assert metric_payload["dataset"] == "validation"
    assert metric_payload["n_obs"] == len(X_val)

    report = validation_report_payload(session)
    assert report["available"] is True
    assert [split["name"] for split in report["splits"]] == ["train", "validation", "test"]
    assert report["splits"][0]["n_obs"] == len(X_train)
    assert report["splits"][1]["n_obs"] == len(X_val)
    assert report["splits"][2]["n_obs"] == len(X_test)
    assert report["cv_report"] == cv_report
    assert report["can_run_cv"] is False


def test_dataset_metrics_use_offset_aware_null_deviance():
    from superglm.editor.evaluation import EvaluationDataset
    from superglm.editor.metrics import compute_dataset_metrics
    from superglm.model.fit_ops import _compute_null_mu

    rng = np.random.default_rng(20260711)
    n = 160
    x = rng.normal(size=n)
    offset = np.linspace(-1.0, 1.0, n)
    sample_weight = rng.uniform(0.7, 1.6, size=n)
    X = pd.DataFrame({"x": x})
    y = 1.2 + 0.25 * x + offset + rng.normal(0.0, 0.04, n)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x": Numeric()},
    )
    model.fit(X, y, sample_weight=sample_weight, offset=offset)

    dataset = EvaluationDataset(
        name="validation",
        label="Validation",
        X=X,
        y=y,
        sample_weight=sample_weight,
        offset=offset,
    )
    metrics = compute_dataset_metrics(model, dataset)

    mu = model.predict(X, offset=offset)
    deviance = float(np.sum(sample_weight * model._distribution.deviance_unit(y, mu)))
    null_mu = _compute_null_mu(
        y,
        sample_weight,
        offset,
        model._distribution,
        model._link,
    )
    null_deviance = float(np.sum(sample_weight * model._distribution.deviance_unit(y, null_mu)))
    assert metrics["explained_deviance"] == pytest.approx(1.0 - deviance / null_deviance)


def test_dataset_metrics_flatten_column_vector_inputs():
    from superglm.editor.evaluation import EvaluationDataset
    from superglm.editor.metrics import compute_dataset_metrics

    rng = np.random.default_rng(20260712)
    n = 120
    x = rng.normal(size=n)
    offset = np.linspace(-0.5, 0.5, n)
    sample_weight = rng.uniform(0.8, 1.4, size=n)
    X = pd.DataFrame({"x": x})
    y = 0.7 + 0.2 * x + offset + rng.normal(0.0, 0.03, n)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x": Numeric()},
    )
    model.fit(X, y, sample_weight=sample_weight, offset=offset)

    flat = EvaluationDataset(
        name="validation",
        label="Validation",
        X=X,
        y=y,
        sample_weight=sample_weight,
        offset=offset,
    )
    column = EvaluationDataset(
        name="validation",
        label="Validation",
        X=X,
        y=y.reshape(-1, 1),
        sample_weight=sample_weight.reshape(-1, 1),
        offset=offset.reshape(-1, 1),
    )

    flat_metrics = compute_dataset_metrics(model, flat)
    column_metrics = compute_dataset_metrics(model, column)

    for key, value in flat_metrics.items():
        assert column_metrics[key] == pytest.approx(value)


def test_widget_http_summary_and_fixed_offset_refit(editor_model):
    session = EditorSession.from_model(editor_model, terms=["x_spline", "region"])
    widget = session.widget()
    try:
        original = _post_json(f"{widget.url}/summary", {"source": "original"})
        assert original["available"] is True
        assert original["source"] == "original"
        assert "SuperGLM Results" in original["html"]
        assert original["compact"]["source"] == "original"
        assert original["compact"]["model"]["family"].lower() == "gaussian"
        assert original["compact"]["model"]["link"].lower() == "identity"
        assert original["compact"]["model"]["method"] == "MLE"
        assert original["compact"]["rows"]
        assert all("sig_class" in row for row in original["compact"]["rows"])

        intercept = next(row for row in original["compact"]["rows"] if row["name"] == "Intercept")
        assert intercept["edf"] == pytest.approx(1.0)
        assert intercept["se"] is not None
        assert intercept["p_value"] is not None
        assert intercept["sig_code"] in {"***", "**", "*", ".", ""}
        assert intercept["sig_class"] in {
            "sig-strong",
            "sig-medium",
            "sig-standard",
            "sig-weak",
            "sig-none",
        }

        spline_row = next(row for row in original["compact"]["rows"] if row["name"] == "x_spline")
        assert spline_row["kind"] == "spline"
        assert spline_row["stat_label"] == "chi2"
        assert spline_row["se"] is None
        assert spline_row["se_label"] == "curve"
        assert spline_row["edf"] is not None
        assert spline_row["p_value"] is not None

        _post_json(f"{widget.url}/select", {"term": "region", "indices": [1]})
        _post_json(f"{widget.url}/op", {"operation": "shift_up"})

        edited = _post_json(f"{widget.url}/summary", {"source": "in_force"})
        assert edited["available"] is True
        assert "Editor edits applied" in edited["html"]
        assert edited["compact"]["source"] == "in_force"
        assert any(row["sig_class"] == "sig-unknown" for row in edited["compact"]["rows"])

        before_refit = _post_json(f"{widget.url}/summary", {"source": "refit"})
        assert before_refit["available"] is False
        assert "No fixed-offset refit" in before_refit["error"]

        refit = _post_json(f"{widget.url}/refit_offset", {})
        assert refit["available"] is True
        assert refit["source"] == "refit"
        assert refit["offset_terms"] == ["region"]
        assert "Editor offset refit" in refit["html"]
        assert refit["compact"]["source"] == "refit"
        assert refit["compact"]["offset_terms"] == ["region"]
        assert refit["compact"]["rows"]
        assert all("sig_class" in row for row in refit["compact"]["rows"])
        assert refit["offset_labels"][0]["term"] == "region"
        assert refit["offset_labels"][0]["scale"] == "log edited relativity"
        assert any(item["label"] == "B" for item in refit["offset_labels"][0]["values"])

        after_refit = _post_json(f"{widget.url}/summary", {"source": "refit"})
        assert after_refit["available"] is True
        assert after_refit["offset_terms"] == ["region"]
    finally:
        widget.close()


def test_widget_http_reports_are_display_only_evidence_surfaces(editor_model, editor_frame):
    X, y = editor_frame
    session = EditorSession.from_model(
        editor_model,
        terms=["x_spline", "region"],
        train_data=(X.iloc[:300], y[:300], None),
        validation_data=(X.iloc[300:380], y[300:380], None),
        test_data=(X.iloc[380:], y[380:], None),
        cv_report={
            "method": "KFold",
            "folds": 3,
            "metric": "mean unit deviance",
            "rows": [{"fold": 1, "train_loss": 0.012, "validation_loss": 0.015}],
            "summary": [{"loss": "validation_loss", "mean": 0.015, "std": 0.001}],
            "split_loss": [{"split": "test", "loss": 0.016, "n_obs": 70}],
        },
    )
    widget = session.widget()
    try:
        validation = _post_json(f"{widget.url}/report", {"report": "validation"})
        assert validation["available"] is True
        assert validation["report"] == "validation"
        assert validation["can_run_cv"] is False
        assert [split["name"] for split in validation["splits"]] == [
            "train",
            "validation",
            "test",
        ]
        assert validation["cv_report"]["method"] == "KFold"
        assert validation["cv_report"]["summary"][0]["loss"] == "validation_loss"
        assert validation["cv_report"]["split_loss"][0]["split"] == "test"

        final = _post_json(f"{widget.url}/report", {"report": "final"})
        assert final["available"] is True
        assert final["report"] == "final"
        assert final["summary"]["source"] == "in_force"
        assert [split["name"] for split in final["splits"]] == [
            "train",
            "validation",
            "test",
        ]
    finally:
        widget.close()


def test_widget_http_report_sanitizes_non_finite_cv_values(editor_model, editor_frame):
    X, y = editor_frame
    session = EditorSession.from_model(
        editor_model,
        terms=["x_spline"],
        train_data=(X.iloc[:300], y[:300], None),
        validation_data=(X.iloc[300:380], y[300:380], None),
        cv_report={
            "method": "KFold",
            "rows": [
                {
                    "k": 5,
                    "cv_improvement": float("nan"),
                    "se_val_deviance": np.float64(np.nan),
                }
            ],
        },
    )
    widget = session.widget()
    try:
        data = json.dumps({"report": "validation"}).encode("utf-8")
        request = urllib.request.Request(
            f"{widget.url}/report",
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                **_editor_token_header(f"{widget.url}/report"),
            },
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            body = response.read().decode("utf-8")

        assert "NaN" not in body
        payload = json.loads(body)
        row = payload["cv_report"]["rows"][0]
        assert row["cv_improvement"] is None
        assert row["se_val_deviance"] is None
    finally:
        widget.close()


def test_widget_http_save_model_writes_edited_joblib(editor_model, tmp_path):
    import joblib

    session = EditorSession.from_model(editor_model, terms=["x_spline"])
    original_beta = editor_model.result.beta.copy()
    session.select_indices("x_spline", [4, 5, 6])
    session.shift("x_spline", 0.2)
    widget = session.widget()
    try:
        payload = _post_json(
            f"{widget.url}/save_model",
            {"directory": str(tmp_path), "filename": "edited-model"},
        )
    finally:
        widget.close()

    path = Path(payload["path"])
    assert path == tmp_path / "edited-model.joblib"
    assert path.exists()
    saved_model = joblib.load(path)
    assert saved_model is not editor_model
    np.testing.assert_allclose(editor_model.result.beta, original_beta)
    assert not np.allclose(saved_model.result.beta, original_beta)


def test_widget_http_download_model_returns_joblib_attachment(editor_model):
    import io

    import joblib

    session = EditorSession.from_model(editor_model, terms=["x_spline"])
    original_beta = editor_model.result.beta.copy()
    session.select_indices("x_spline", [4, 5, 6])
    session.shift("x_spline", 0.2)
    widget = session.widget()
    try:
        request = urllib.request.Request(
            f"{widget.url}/download_model?filename=edited-model.joblib",
            headers=_editor_token_header(widget.url),
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = response.read()
            content_type = response.headers["content-type"]
            disposition = response.headers["content-disposition"]
    finally:
        widget.close()

    downloaded_model = joblib.load(io.BytesIO(payload))
    assert content_type == "application/octet-stream"
    assert 'filename="edited-model.joblib"' in disposition
    assert downloaded_model is not editor_model
    np.testing.assert_allclose(editor_model.result.beta, original_beta)
    assert not np.allclose(downloaded_model.result.beta, original_beta)


def test_widget_http_native_save_dialog_returns_selected_path(editor_model, tmp_path, monkeypatch):
    from superglm.editor import widget as widget_module

    selected = tmp_path / "picked-model.joblib"
    monkeypatch.setattr(widget_module, "choose_save_path", lambda **_kwargs: str(selected))
    session = EditorSession.from_model(editor_model, terms=["x_spline"])
    widget = session.widget()
    try:
        payload = _post_json(
            f"{widget.url}/native_save_dialog",
            {"directory": str(tmp_path), "filename": "default-name.joblib"},
        )
    finally:
        widget.close()

    assert payload == {
        "cancelled": False,
        "path": str(selected),
        "directory": str(tmp_path),
        "filename": "picked-model.joblib",
    }


def test_widget_http_native_save_dialog_handles_cancel(editor_model, monkeypatch):
    from superglm.editor import widget as widget_module

    monkeypatch.setattr(widget_module, "choose_save_path", lambda **_kwargs: None)
    session = EditorSession.from_model(editor_model, terms=["x_spline"])
    widget = session.widget()
    try:
        payload = _post_json(f"{widget.url}/native_save_dialog", {})
    finally:
        widget.close()

    assert payload == {"cancelled": True}


def test_widget_http_open_directory_launches_current_folder(editor_model, tmp_path, monkeypatch):
    from superglm.editor import widget as widget_module

    opened: list[str] = []

    def fake_open_directory(path):
        opened.append(str(path))
        return Path(path)

    monkeypatch.setattr(widget_module, "open_directory_path", fake_open_directory)
    session = EditorSession.from_model(editor_model, terms=["x_spline"])
    widget = session.widget()
    try:
        payload = _post_json(f"{widget.url}/open_directory", {"path": str(tmp_path)})
    finally:
        widget.close()

    assert payload == {"path": str(tmp_path.resolve())}
    assert opened == [str(tmp_path.resolve())]


def test_open_directory_prefers_dolphin_on_kde(tmp_path, monkeypatch):
    from superglm.editor import native_dialogs

    commands = []

    class FakeProcess:
        def __init__(self, command):
            self.command = command

        def poll(self):
            return None

        def communicate(self, timeout=None):
            return "", ""

    monkeypatch.setattr(native_dialogs.sys, "platform", "linux")
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "KDE")
    monkeypatch.setattr(native_dialogs.shutil, "which", lambda command: f"/usr/bin/{command}")
    monkeypatch.setattr(
        native_dialogs.subprocess,
        "Popen",
        lambda command, **_kwargs: commands.append(tuple(command)) or FakeProcess(command),
    )

    opened = native_dialogs.open_directory_path(tmp_path)

    assert opened == tmp_path.resolve()
    assert commands[0] == ("dolphin", "--new-window", str(tmp_path.resolve()))


def test_open_directory_raises_when_launcher_exits_immediately(tmp_path, monkeypatch):
    from superglm.editor import native_dialogs

    class FailedProcess:
        def poll(self):
            return 1

        def communicate(self, timeout=None):
            return "", "could not open"

    monkeypatch.setattr(native_dialogs.sys, "platform", "linux")
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "KDE")
    monkeypatch.setattr(native_dialogs.shutil, "which", lambda command: "/usr/bin/dolphin")
    monkeypatch.setattr(
        native_dialogs.subprocess, "Popen", lambda *_args, **_kwargs: FailedProcess()
    )

    with pytest.raises(RuntimeError, match="could not open"):
        native_dialogs.open_directory_path(tmp_path)


def test_widget_save_directory_defaults_to_cwd(editor_model, tmp_path, monkeypatch):
    child = tmp_path / "models"
    child.mkdir()
    file_path = tmp_path / "existing-model.joblib"
    file_path.write_text("placeholder")
    monkeypatch.chdir(tmp_path)
    session = EditorSession.from_model(editor_model, terms=["x_spline"])
    widget = session.widget()
    try:
        payload = widget._save_directory()
    finally:
        widget.close()

    assert payload["cwd"] == str(tmp_path.resolve())
    assert payload["path"] == str(tmp_path.resolve())
    assert {
        "kind": "directory",
        "name": "models",
        "path": str(child.resolve()),
    } in payload["entries"]
    assert {
        "kind": "file",
        "name": "existing-model.joblib",
        "path": str(file_path.resolve()),
    } in payload["entries"]
    assert [entry["kind"] for entry in payload["entries"]] == ["directory", "file"]


def test_widget_reset_and_undo_target_current_term_after_switching(editor_model):
    session = EditorSession.from_model(editor_model, terms=["x_spline", "region"])
    widget = session.widget()
    try:
        spline = session.terms["x_spline"]
        region = session.terms["region"]
        spline_original = spline.original_log_effect.copy()
        region_original = region.original_log_effect.copy()

        _post_json(f"{widget.url}/select", {"term": "x_spline", "indices": [4, 5]})
        _post_json(f"{widget.url}/op", {"operation": "shift_up"})
        _post_json(f"{widget.url}/select", {"term": "region", "indices": [1]})
        _post_json(f"{widget.url}/op", {"operation": "shift_down"})
        _post_json(f"{widget.url}/term", {"term": "x_spline"})

        _post_json(f"{widget.url}/op", {"operation": "undo"})
        np.testing.assert_allclose(spline.edited_log_effect, spline_original)
        assert region.edited_log_effect[1] == pytest.approx(region_original[1] + np.log(0.95))

        _post_json(f"{widget.url}/op", {"operation": "shift_up"})
        _post_json(f"{widget.url}/term", {"term": "region"})
        _post_json(f"{widget.url}/term", {"term": "x_spline"})
        _post_json(f"{widget.url}/op", {"operation": "reset"})
        np.testing.assert_allclose(spline.edited_log_effect, spline_original)
    finally:
        widget.close()


def test_widget_app_shell_contains_drag_editor(editor_model):
    session = EditorSession.from_model(editor_model, terms=["x_spline"])
    widget = session.widget()
    try:
        with urllib.request.urlopen(widget.url, timeout=5) as response:
            shell = response.read().decode("utf-8")
        js_request = urllib.request.Request(f"{widget.url}/assets/main.js", method="GET")
        with urllib.request.urlopen(js_request, timeout=5) as response:
            js = response.read().decode("utf-8")
        module_sources = []
        for asset in [
            "api.js",
            "format.js",
            "chart.js",
            "metrics.js",
            "summary.js",
            "reports.js",
            "interactions.js",
        ]:
            request = urllib.request.Request(f"{widget.url}/assets/{asset}", method="GET")
            with urllib.request.urlopen(request, timeout=5) as response:
                module_sources.append(response.read().decode("utf-8"))
        js = "\n".join([js, *module_sources])
        css_request = urllib.request.Request(f"{widget.url}/assets/styles.css", method="GET")
        with urllib.request.urlopen(css_request, timeout=5) as response:
            css = response.read().decode("utf-8")

        assert '<link rel="stylesheet" href="/assets/styles.css">' in shell
        assert '<script type="module" src="/assets/main.js"></script>' in shell
        assert "Editor" in shell
        assert "Validation Report" in shell
        assert "Final Fit Report" in shell
        assert "reportPanel" in shell
        assert "reportFrame" in shell
        assert "Run CV" not in shell
        assert 'id="modelSource"' not in shell
        assert 'value="original"' not in shell
        assert "Mode" in shell
        assert "Move" in shell
        assert "Handles" in shell
        assert "Select all" in shell
        assert "Undo collapse" in shell
        assert "Linearise" in shell
        assert "Level left" in shell
        assert "Level right" in shell
        assert "Ungroup selected levels" in shell
        assert "Snap highest" in shell
        assert "Snap lowest" in shell
        assert "Home" in shell
        assert "CI" in shell
        assert "chart-shell" in shell
        assert "selectionMenu" in shell
        assert "selection-item" in shell
        assert "selection-separator" in shell
        assert "selection-submenu" in shell
        assert "selection-icon" in shell
        assert 'id="uncollapseLevels" class="selection-item"' in shell
        assert ">Undo collapse</button>" not in shell
        assert 'aria-label="Smooth"' in shell
        assert 'aria-label="Level selected values"' in shell
        assert 'aria-label="Snap selected values"' in shell
        assert ">Smooth</button>" not in shell
        assert ">Average</button>" not in shell
        assert ">Linearise</button>" not in shell
        assert ">Increasing</button>" not in shell
        assert ">Decreasing</button>" not in shell
        assert ">Level left</button>" not in shell
        assert ">Level right</button>" not in shell
        assert ">Snap highest</button>" not in shell
        assert ">Snap lowest</button>" not in shell
        assert ">Undo</button>" not in shell
        assert ">Redo</button>" not in shell
        assert "Ref CI" in shell
        assert "Reset" in shell
        assert "saveModel" in shell
        assert "saveDialog" in shell
        assert "saveDirectory" in shell
        assert "saveBrowse" in shell
        assert "saveFilename" in shell
        assert "Save edited model" in shell
        assert "app-shell" in shell
        assert "app-shell" in css
        assert "justify-content: center" in css
        assert "openSaveDialog" in js
        assert 'aria-haspopup="dialog"' in shell
        assert "Reset order" in shell
        assert "resetOrder" in shell
        assert "metricSelect" in shell
        assert "metricGrid" in shell
        assert "Recompute all" not in shell
        assert "summaryPanel" in shell
        assert "summaryFrame" in shell
        assert "summarySource" in shell
        assert "refitOffset" in shell
        assert "Fixed-offset refit" in shell
        assert "better" not in shell
        assert "worse" not in shell
        state = _get_json(f"{widget.url}/state")
        assert state["terms"]["x_spline"]["y_label"] == "relativity"
        assert state["terms"]["x_spline"]["effective_df"] == pytest.approx(
            session.terms["x_spline"].metadata["edf"]
        )
        controls = state["terms"]["x_spline"]["controls"]
        assert "build_basis" in controls
        assert "build_log_effect" in controls
        assert len(controls["build_basis"]) == len(controls["build_log_effect"])
        assert len(controls["build_basis"][0]) == state["terms"]["x_spline"]["n_points"]
        assert "Average" in shell
        assert "pointerdown" in js
        assert "pointermove" in js
        assert "pointerup" in js
        assert 'addEventListener("wheel"' in js
        assert "zoomState" in js
        assert "resetZoom" in js
        assert "panDrag" in js
        assert "panZoomView" in js
        assert "getScreenCTM" in js
        assert "matrixTransform" in js
        assert "pendingClickIndex" in js
        assert "nearestIndex" not in js
        assert "togglePointSelection" in js
        assert "event.ctrlKey || event.metaKey" in js
        assert "selectedPoints" in js
        assert "optgroup" in js
        assert "relativity" in js
        assert "X-SuperGLM-Editor-Token" in js
        assert "URLSearchParams(window.location.search)" in js
        assert "/drag" in js
        assert "/control" in js
        assert "/control_count" in js
        assert "handleCount" in js
        assert "basisToggle" in shell
        assert "contribPlay" in shell
        assert "buildDuration" in shell
        assert "buildDurationValue" in shell
        assert "Contrib" in shell
        assert "Build" in shell
        assert "showContrib" in js
        assert "graphMode" in js
        assert "visualMode" in js
        assert 'modeSelect.value !== "zoom"' in js
        assert "requestAnimationFrame" in js
        assert "buildDurationMs" in js
        assert "advanceContributionBuild" in js
        assert "contribPlay.disabled = buildFrame !== null" in js
        assert "buildAccumulationCurve" in js
        assert "drawActiveBasis" in js
        assert "activeBasisIndex" in js
        assert "basisColor" in js
        assert "mixBuildColor" in js
        assert "data-progress" in js
        assert "data-active-basis" in js
        assert "total - j" not in js
        assert "buildContributionCurves" not in js
        assert "basis-contribution" in js
        assert "basis-contribution" in css
        assert "basis-build" in js
        assert "basis-build" in css
        assert "basis-active" in js
        assert "basis-active" in css
        assert "basis-sweep" not in js
        assert "basis-scanline" not in js
        assert "basis-scan-dot" not in js
        assert "basis-build-halo" in js
        assert "basis-build-halo" in css
        assert "if (!buildActive) path(svg, x, original" in js
        assert "controlDrag" in js
        assert "data-control-index" in js
        assert "keydown" in js
        assert "isEditableTarget" in js
        assert "metaKey" in js
        assert "errorBars" in js
        assert "refreshMetrics" in js
        assert "refreshReport" in js
        assert "/report" in js
        assert "cv-report" in js
        assert "CV Summary" in js
        assert "Split Loss" in js
        assert "Fold Loss" in js
        assert "Run CV" not in js
        assert "modelSource" not in js
        assert '<option value="zoom">Zoom</option>' in shell
        assert "zoomBox" in js
        assert "beginBoxZoom" in js
        assert "applyBoxZoom" in js
        assert "box-zoom" in js
        assert 'metricGrid.textContent = "Computing metrics..."' not in js
        assert 'summaryFrame.innerHTML = ""' not in js
        assert "/metrics" in js
        assert "/summary" in js
        assert "/profile_distribution" in js
        assert "/save_model" in js
        assert "/download_model" in js
        assert "/native_save_dialog" in js
        assert "/open_directory" in js
        assert "saveEditedModel" in js
        assert "downloadEditedModel" in js
        assert "openNativeSaveDialog" in js
        assert "openDirectoryInFileManager" in js
        assert "formatSaveRouteError" in js
        assert "Rerun session.widget()" in js
        assert "Opening file dialog" in js
        assert "Opening folder" in js
        assert "saveBlobToFile" in js
        assert "showSaveFilePicker" in js
        assert "URL.createObjectURL" in js
        assert "openSaveDialog" in js
        assert "directoryDialog" not in shell
        assert "Choose Save Location" not in shell
        assert 'id="saveDownload"' in shell
        assert 'id="saveOpenDirectory"' in shell
        assert "Open Folder" in shell
        assert "Download Edited Model" in shell
        assert "Saved " in js
        assert "/profile_distribution/start" in js
        assert "/profile_distribution/status/" in js
        assert "runDistributionProfile" in js
        assert "reprofileTweedie" in shell
        assert "reprofileNb2" in shell
        assert "profileDialog" in shell
        assert "profileDialogClose" in shell
        assert "profileRun" in shell
        assert "profileOptions" in shell
        assert "profileTolerance" in shell
        assert '<option value="mle" selected>MLE</option>' in shell
        assert "profileTracePlot" in shell
        assert "profile-dialog" in css
        assert "profile-trace-line" in css
        assert "profile-trace-best" in css
        assert "profile-learning-curve" in css
        assert "profile-learning-point" in css
        assert "profileLearningCurvesSVG" in js
        assert "profileFitTraceRows" in js
        assert "profileFitIterTicks" in js
        assert "fitTraceKind" in js
        assert "profileStatusLabel" in js
        assert "profile-running" in css
        assert "profile-spin" in css
        assert "profileTraceLegend" in shell
        assert "profile-trace-figure" in css
        assert "profile-trace-legend" in css
        assert "profileEstimate" in js
        assert "outerProfileObjective" in js
        assert "profileObjectiveSVG" in js
        assert "return profileObjectiveSVG" in js
        assert "profile loss" in js
        assert "inner fit trace" in js
        assert "profile_ci" in js
        assert "final refit" in js
        assert "p_hat" in js
        assert "CI" in js
        assert "start -> final" in js
        assert "showModal" in js
        assert "openProfileDialog" in js
        assert "showDistributionProfileDialog" in js
        assert "profileRun.addEventListener" in js
        assert "trace_iterations" in js
        assert "Profile trace" in shell
        assert "profileOptionsPayload" in js
        assert "renderProfileTrace" in js
        assert "updateDistributionProfileActions" in js
        assert "renderSummaryRows" in js
        assert "summary-group-row" in js
        assert "summary-group-row" in css
        assert 'colspan="6"' in js
        assert "/refit_offset" in js
        assert "/collapse_levels" in js
        assert "/uncollapse_levels" in js
        assert "can_uncollapse_levels" in js
        assert "last_collapse" in js
        assert "selectionTouchesCollapsedGroup" in js
        assert "updateResetOrderAction" in js
        assert "reset_order" in js
        assert "orderDrag" in js
        assert "beginOrderDrag" in js
        assert "drawOrderDropPreview" in js
        assert "clearOrderDropPreview" in js
        assert "order-drop-preview" in js
        assert "order-drop-ghost" in js
        assert "targetIndexFromPoint" in js
        assert "Math.min(levels.length," in js
        assert "pixelStep * count * 0.82" not in js
        assert "/reorder_levels" in js
        assert "isDisplayOnlyOperation" in js
        assert "exposureDensity" in js
        assert "level_groups" in js
        assert "drawLevelGroups" in js
        assert "drawLevelGroupMarker" in js
        assert "levelGroupColor" in js
        assert "level-group-link" in js
        assert "level-group-link" in css
        assert "level-group-marker" in css
        assert "level-group-label" in css
        assert "order-drop-preview" in css
        assert "order-drop-ghost" in css
        assert "plotClip" in js
        assert "applyPlotClip" in js
        assert "positionSelectionMenu" in js
        assert "compact-summary" in js
        assert "summary-table" in js
        assert "row.edf" in js
        assert "formatCompactNumber" in js
        assert "summary-number" in css
        assert "se-cell" in js
        assert "raw-summary" in js
        assert "payloadNumber" in js
        assert "rotate(-90" in js
        assert "toExponential" not in js
        assert "100dvh" in css
        assert "overflow: hidden" in css
        assert "minmax(0, 1fr)" in css
        assert "user-select: none" in css
        assert "-webkit-user-select: none" in css
        assert "pointer-events: all" in css
        assert "overflow-x: hidden" in css
        assert "sig-strong" in css
        assert "sig-none" in css
        assert "sig-unknown" in css
        assert "control-handle" in css
        assert "ci-whisker" in css
        assert "metric-item" in css
        assert "metric-divider" in css
        assert "grid-template-rows: 28px 20px auto" in css
        assert "metric-card" not in css
        assert "app-tabs" in css
        assert "report-table" in css
        assert "cv-report" in css
        assert "effective_df" in js
        assert "Log Likelihood" in shell
        assert "Explained Deviance" in shell
        assert "Pearson Chi2" in shell
        assert "exposure-axis" in css
        assert "selection-bounds" in css
        assert "#F4D35E" in css
        assert "#D8A10F" in css
    finally:
        widget.close()


def test_widget_serves_editor_app_assets(editor_model):
    session = EditorSession.from_model(editor_model, terms=["x_spline"])
    widget = session.widget()
    try:
        with urllib.request.urlopen(widget.url, timeout=5) as response:
            shell = response.read().decode("utf-8")
        assert '<link rel="stylesheet" href="/assets/styles.css">' in shell
        assert '<script type="module" src="/assets/main.js"></script>' in shell
        assert "<style>" not in shell
        assert "<script>\nconst svg" not in shell

        css_request = urllib.request.Request(f"{widget.url}/assets/styles.css", method="GET")
        with urllib.request.urlopen(css_request, timeout=5) as response:
            assert response.headers["Content-Type"].startswith("text/css")
            css = response.read().decode("utf-8")
        assert ".toolbar" in css
        assert ".summary-frame" in css
        assert ".metric-grid" in css

        js_request = urllib.request.Request(f"{widget.url}/assets/main.js", method="GET")
        with urllib.request.urlopen(js_request, timeout=5) as response:
            assert response.headers["Content-Type"].startswith("application/javascript")
            js = response.read().decode("utf-8")
        assert "loadState" in js
        assert "drawChart" in js
        assert "runOffsetRefit" in js

        for asset in [
            "api.js",
            "format.js",
            "chart.js",
            "metrics.js",
            "summary.js",
            "reports.js",
            "interactions.js",
        ]:
            request = urllib.request.Request(f"{widget.url}/assets/{asset}", method="GET")
            with urllib.request.urlopen(request, timeout=5) as response:
                assert response.headers["Content-Type"].startswith("application/javascript")
                assert response.read()

        assert 'from "./chart.js"' in js
        assert 'from "./metrics.js"' in js
        assert 'from "./summary.js"' in js
        assert 'from "./interactions.js"' in js
    finally:
        widget.close()


def test_widget_unknown_route_uses_editor_error_shape(editor_model):
    session = EditorSession.from_model(editor_model, terms=["x_spline"])
    widget = session.widget()
    try:
        request = urllib.request.Request(
            f"{widget.url}/missing-route",
            headers=_editor_token_header(f"{widget.url}/missing-route"),
        )
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request, timeout=5)

        assert error.value.code == 404
        payload = json.loads(error.value.read().decode("utf-8"))
        assert payload == {"error": "not found"}
    finally:
        widget.close()


def test_editor_server_declares_fastapi_routes():
    from superglm.editor.server import create_editor_app

    class DummyWidget:
        pass

    app = create_editor_app(DummyWidget())
    routes = {
        (route.path, frozenset(route.methods or set()))
        for route in app.routes
        if getattr(route, "include_in_schema", True)
    }

    assert ("/state", frozenset({"GET"})) in routes
    assert ("/health", frozenset({"GET"})) in routes
    assert ("/term", frozenset({"POST"})) in routes
    assert ("/select", frozenset({"POST"})) in routes
    assert ("/op", frozenset({"POST"})) in routes
    assert ("/drag", frozenset({"POST"})) in routes
    assert ("/control", frozenset({"POST"})) in routes
    assert ("/control_count", frozenset({"POST"})) in routes
    assert ("/metrics", frozenset({"POST"})) in routes
    assert ("/summary", frozenset({"POST"})) in routes
    assert ("/report", frozenset({"POST"})) in routes
    assert ("/save_model", frozenset({"POST"})) in routes
    assert ("/download_model", frozenset({"GET"})) in routes
    assert ("/native_save_dialog", frozenset({"POST"})) in routes
    assert ("/open_directory", frozenset({"POST"})) in routes
    assert ("/save_directory", frozenset({"POST"})) in routes
    assert ("/refit_offset", frozenset({"POST"})) in routes
    assert ("/profile_distribution", frozenset({"POST"})) in routes
    assert ("/profile_distribution/start", frozenset({"POST"})) in routes
    assert ("/profile_distribution/status/{job_id}", frozenset({"GET"})) in routes
    assert ("/collapse_levels", frozenset({"POST"})) in routes
    assert ("/ungroup_levels", frozenset({"POST"})) in routes
    assert ("/reorder_levels", frozenset({"POST"})) in routes
    assert ("/uncollapse_levels", frozenset({"POST"})) in routes
    assert ("/model_source", frozenset({"POST"})) not in routes


def test_summary_html_keeps_spline_significance_out_of_qs_column():
    summary = ModelSummary(
        data={"fit": {}},
        model_info={
            "family": "gaussian",
            "link": "identity",
            "penalty": "ridge",
            "n_obs": 100,
            "effective_df": 4.2,
            "phi": 1.0,
            "log_likelihood": -10.0,
            "deviance": 20.0,
            "null_deviance": 30.0,
            "pseudo_r2": 0.3,
            "aic": 40.0,
            "aicc": 41.0,
            "bic": 45.0,
            "ebic": 46.0,
            "converged": True,
            "n_iter": 3,
        },
        coef_rows=[
            _CoefRow(name="Intercept", coef=0.0, se=1.0, z=0.0, p=1.0, ci_low=-1.0, ci_high=1.0),
            _CoefRow(
                name="age",
                group="age",
                is_spline=True,
                n_params=5,
                active=True,
                wald_chi2=25.0,
                wald_p=1e-4,
                ref_df=3.0,
                edf=3.2,
            ),
        ],
        basis_detail={
            "age": [
                _BasisDetailRow(
                    parent_name="age",
                    basis_index=0,
                    coef=0.2,
                    se=0.05,
                    z=4.0,
                    p=1e-4,
                    ci_low=0.1,
                    ci_high=0.3,
                )
            ]
        },
    )

    html = summary._repr_html_()

    assert '<td style="padding:3px 4px;text-align:left;border:none;">***</td><td' in html
    assert "<td style='padding:1px 6px;'>***</td><td style='padding:1px 6px;'></td>" in html


def _get_json(url: str):
    request = urllib.request.Request(url, headers=_editor_token_header(url))
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def _post_json(url: str, payload: dict):
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    headers.update(_editor_token_header(url))
    request = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def _editor_token_header(url: str) -> dict[str, str]:
    from superglm.editor.widget import _LIVE_WIDGETS

    parsed = urllib.parse.urlparse(url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    for widget in list(_LIVE_WIDGETS):
        if getattr(widget, "url", "") == origin:
            return {"X-SuperGLM-Editor-Token": widget._token}
    return {}
