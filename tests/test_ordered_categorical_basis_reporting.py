"""Reporting and export for OrderedCategorical Piecewise/Polynomial bases.

Pins the inner-basis reporting vocabulary: segmented terms report STRUCTURAL
CONTRASTS -- one slope-change Wald row per stated break and one curvature row
per degree>=2 segment, never per-segment per-power z rows (under C0 seams that
geometry does not exist; the rows are ordinary fixed-knot inference, Smith
1979) -- while ``basis=Polynomial(powers=[...])`` keeps one clean-z row per
stated power (the main-effect property of the exposure-orthonormal ordinal
contrasts). Both whole-term rows are plain Wald chi-squares on an unpenalized
block, counted in the parametric df bucket, and never trigger the Wood smooth
footnote on their own. Rating-table export stays one row per band for every
inner basis, exact against the fitted model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from superglm import OrderedCategorical, Piecewise, Polynomial, Spline, SuperGLM

LEVELS = [f"Mi{i:03d}" for i in range(10)]


def _frame(n: int = 4000, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"band": rng.choice(LEVELS, n)})
    position = {level: index for index, level in enumerate(LEVELS)}
    signal = np.array(
        [0.02 * min(position[b], 4) ** 2 + (0.05 if position[b] > 6 else 0.0) for b in X["band"]]
    )
    y = signal + rng.normal(0.0, 0.05, n)
    return X, y


def _fit(spec: OrderedCategorical) -> SuperGLM:
    X, y = _frame()
    model = SuperGLM(family="gaussian", selection_penalty=0.0, features={"band": spec})
    model.fit(X, y)
    return model


@pytest.fixture(scope="module")
def segmented_model() -> SuperGLM:
    return _fit(
        OrderedCategorical(
            order=LEVELS, basis=Piecewise(breaks=["Mi004", "Mi007"], degrees=[2, 1, 0])
        )
    )


@pytest.fixture(scope="module")
def polynomial_model() -> SuperGLM:
    return _fit(OrderedCategorical(order=LEVELS, basis=Polynomial(powers=[1, 3])))


# ── summary rows ─────────────────────────────────────────────────────


def test_segmented_summary_reports_structural_contrasts(segmented_model) -> None:
    text = str(segmented_model.summary())
    # Knots [0, 4, 7, 9] with degrees [2, 1, 0]: the flat tail merges knots 7
    # and 9 into one value column -> 3 knot-value groups minus the base, plus
    # one curvature column = 3 parameters.
    assert "[ordered piecewise, 3 params" in text
    assert "band[slope-change @ Mi004]" in text
    assert "band[slope-change @ Mi007]" in text
    assert "band[curvature Mi000..Mi004]" in text
    # One curvature row only: the degree-1 and degree-0 segments state none.
    assert text.count("curvature") == 1


def test_segmented_summary_has_no_per_power_rows(segmented_model) -> None:
    text = str(segmented_model.summary())
    assert "band[P1]" not in text
    assert "band[P2]" not in text


def test_segmented_slope_change_rows_are_real_wald_rows(segmented_model) -> None:
    for line in str(segmented_model.summary()).splitlines():
        if "slope-change @ Mi004" in line:
            # coef, se, z, p and CI all populated (no "---" placeholders in
            # the numeric fields).
            assert line.count("---") == 1  # the QS column only
            return
    raise AssertionError("slope-change row missing")


def test_segmented_df_lands_in_the_parametric_bucket(segmented_model) -> None:
    text = str(segmented_model.summary())
    assert "(0 smooth)" in text
    assert "Wood (2013)" not in text


def test_segmented_level_rows_one_per_band(segmented_model) -> None:
    text = str(segmented_model.summary())
    for level in LEVELS:
        assert f"band[{level}]" in text


def test_polynomial_summary_reports_stated_powers_with_clean_z(polynomial_model) -> None:
    text = str(polynomial_model.summary())
    assert "[ordered polynomial, 2 params" in text
    assert "band[P1]" in text
    assert "band[P3]" in text
    assert "band[P2]" not in text
    for line in text.splitlines():
        if "band[P1]" in line:
            assert line.count("---") == 1  # z/p/CI populated; QS column only
            return
    raise AssertionError("per-power row missing")


def test_polynomial_summary_has_no_structural_rows(polynomial_model) -> None:
    text = str(polynomial_model.summary())
    assert "slope-change" not in text
    assert "curvature" not in text


def test_curvature_family_of_two_freedoms_reports_a_joint_test() -> None:
    model = _fit(
        OrderedCategorical(order=LEVELS, basis=Piecewise(breaks=["Mi005"], degrees=[3, 1]))
    )
    text = str(model.summary())
    for line in text.splitlines():
        if "curvature Mi000..Mi005" in line:
            assert "[curvature, 2 params, chi2(2.0)=" in line
            return
    raise AssertionError("curvature family row missing")


def test_editor_stale_summary_keeps_the_basis_vocabulary(segmented_model) -> None:
    from superglm.editor.apply import apply_edits_to_model_copy
    from superglm.editor.session import EditorSession

    session = EditorSession.from_model(segmented_model)
    term = session._require_term("band")
    term.edited_log_effect = term.edited_log_effect + 0.01
    edited = apply_edits_to_model_copy(segmented_model, {"band": term})
    text = str(edited.summary())
    assert "[ordered piecewise" in text
    assert "band[Mi000]" in text


# ── export ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "basis",
    [
        Piecewise(breaks=["Mi004", "Mi007"], degrees=[2, 1, 0]),
        Polynomial(powers=[1, 2]),
        Spline(kind="cr", n_knots=4),
    ],
    ids=["piecewise", "polynomial", "spline"],
)
def test_rating_table_is_one_row_per_band_for_every_basis(basis, tmp_path) -> None:
    X, y = _frame()
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"band": OrderedCategorical(order=LEVELS, basis=basis)},
    )
    model.fit(X, y)
    path = model.export_rating_tables(tmp_path / "tables.xlsx", X, y)
    sheet = pd.read_excel(path, sheet_name="Rating Tables", header=None)
    column = sheet[0].astype(str)
    for level in LEVELS:
        assert (column == level).any(), f"missing band row {level!r}"


def test_workbook_reconstruction_is_exact_for_a_segmented_term() -> None:
    """Per-band table ratios equal per-band prediction ratios exactly.

    Under the log link the exported per-band relativity IS the model: any
    degree is table-exact on the level axis, because the table carries one row
    per band and scoring a band reads that row.
    """
    rng = np.random.default_rng(11)
    X = pd.DataFrame({"band": rng.choice(LEVELS, 4000)})
    position = {level: index for index, level in enumerate(LEVELS)}
    mu = np.exp(-1.0 + 0.05 * np.array([min(position[b], 6) for b in X["band"]]))
    y = rng.poisson(mu).astype(float)
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={
            "band": OrderedCategorical(
                order=LEVELS, basis=Piecewise(breaks=["Mi004", "Mi007"], degrees=[2, 1, 0])
            )
        },
    )
    model.fit(X, y)
    ti = model.term_inference("band")
    table = dict(zip(ti.levels, ti.relativity))
    grid = pd.DataFrame({"band": LEVELS})
    predicted = model.predict(grid)
    ratios = predicted / predicted[0]
    table_ratios = np.array([table[b] for b in LEVELS]) / table[LEVELS[0]]
    assert np.allclose(ratios, table_ratios, rtol=1e-12, atol=0.0)


def test_grouped_display_curve_drops_bands_it_cannot_align() -> None:
    """A grouped term's rebuilt display curve never carries foreign band arrays.

    The pre-expansion curve of a hosted Piecewise is knot-grid-length, so
    copying its se/ci onto the 200-point PCHIP rebuild produced a SmoothCurve
    whose bands could not zip against x (observed: len(x)=200 against
    len(se)=26). The rebuilt curve is display interpolation only: bands
    survive exactly when the rebuilt grid equals the pre-expansion grid (the
    common order= spline case), and the per-level SEs -- the rated quantities
    -- stay expanded either way.
    """
    from superglm.features.grouping import collapse_levels

    X, y = _frame()
    data = X["band"].to_numpy(dtype=object)
    groups = {"Mi001+Mi002": ["Mi001", "Mi002"]}
    covered = {member for members in groups.values() for member in members}
    for level in LEVELS:
        if level not in covered:
            groups[level] = [level]
    grouping = collapse_levels(data, groups=groups, order=LEVELS)

    def fitted_curve(basis):
        spec = OrderedCategorical(order=LEVELS, basis=basis, grouping=grouping)
        model = SuperGLM(family="gaussian", selection_penalty=0.0, features={"band": spec})
        model.fit(X, y)
        inference = model.term_inference("band", with_se=True)
        assert inference.se_log_relativity is not None
        assert len(inference.se_log_relativity) == len(inference.levels)
        return inference.smooth_curve

    piecewise_curve = fitted_curve(Piecewise(breaks=["Mi004"], degrees=[2, 1]))
    assert piecewise_curve.se_log_relativity is None
    assert piecewise_curve.ci_lower is None
    assert piecewise_curve.ci_upper is None

    spline_curve = fitted_curve(Spline(kind="cr", n_knots=4))
    assert spline_curve.se_log_relativity is not None
    assert len(spline_curve.se_log_relativity) == len(spline_curve.x)


def test_plateau_bands_share_one_table_row_value(segmented_model) -> None:
    relativities = segmented_model.relativities()["band"]
    tail = relativities.set_index("level").loc[["Mi007", "Mi008", "Mi009"], "relativity"]
    assert float(tail.max()) == pytest.approx(float(tail.min()), abs=0.0)


def test_export_summary_group_kind_says_group_not_smooth() -> None:
    from superglm.export.summary import _group_test_kind

    X, y = _frame()
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"band": OrderedCategorical(order=LEVELS, basis=Piecewise(breaks=["Mi004"]))},
    )
    model.fit(X, y)
    groups = tuple(group for group in model._groups if group.feature_name == "band")

    class _Row:
        name = "band"
        subgroup_type = None

    assert _group_test_kind(model, _Row(), groups) == "group"


def test_specials_ride_alongside_each_basis_in_the_summary() -> None:
    X, y = _frame()
    rng = np.random.default_rng(9)
    X = X.copy()
    missing = rng.random(len(X)) < 0.08
    X.loc[missing, "band"] = "MISSING"
    for basis in (Piecewise(breaks=["Mi004"]), Polynomial(powers=[1, 2])):
        spec = OrderedCategorical(order=LEVELS, basis=basis, specials=["MISSING"])
        model = SuperGLM(family="gaussian", selection_penalty=0.0, features={"band": spec})
        model.fit(X, y)
        text = str(model.summary())
        assert "band[MISSING]" in text
        assert "free" in text
