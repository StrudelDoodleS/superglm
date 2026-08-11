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


def _log_link_frame(n: int = 4000, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    """``_frame``'s band signal as a Poisson count, for the export tests.

    The rating-table export takes log-link models only -- its table is a
    product of factors -- so the tests that go through it need a response the
    log link can carry.  Same levels and same monotone-then-flat shape as
    ``_frame``, so the level axis under test is unchanged.
    """
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"band": rng.choice(LEVELS, n)})
    position = {level: index for index, level in enumerate(LEVELS)}
    mu = np.exp(
        -1.0
        + np.array(
            [
                0.02 * min(position[b], 4) ** 2 + (0.05 if position[b] > 6 else 0.0)
                for b in X["band"]
            ]
        )
    )
    return X, rng.poisson(mu).astype(float)


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
    """One row per band whatever the basis, on the link a rating table is for.

    Poisson rather than this module's shared gaussian ``_fit``: the exported
    table is multiplicative, so the export accepts log-link models only, the
    same restriction ``test_workbook_reconstruction_is_exact_for_a_segmented_
    term`` below already relies on.  What is under test here is the level axis
    -- every band gets a row regardless of the basis fitted through them --
    which the family does not touch.
    """
    X, y = _log_link_frame()
    model = SuperGLM(
        family="poisson",
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


def _grouping(X, groups: dict[str, list[str]]):
    """A LevelGrouping over LEVELS with everything unnamed left as a singleton."""
    from superglm.features.grouping import collapse_levels

    covered = {member for members in groups.values() for member in members}
    full = dict(groups)
    for level in LEVELS:
        if level not in covered:
            full[level] = [level]
    return collapse_levels(X["band"].to_numpy(dtype=object), groups=full, order=LEVELS)


def _grouped_curve(X, y, basis, groups: dict[str, list[str]]):
    spec = OrderedCategorical(order=LEVELS, basis=basis, grouping=_grouping(X, groups))
    model = SuperGLM(family="gaussian", selection_penalty=0.0, features={"band": spec})
    model.fit(X, y)
    inference = model.term_inference("band", with_se=True)
    assert inference.se_log_relativity is not None
    assert len(inference.se_log_relativity) == len(inference.levels)
    return inference


def test_grouped_display_curve_drops_bands_it_cannot_align() -> None:
    """A grouped term's REBUILT display curve never carries foreign band arrays.

    A collapse that moves the display axis leaves the fitted curve undrawable
    against the expanded markers, so that expansion interpolates a fresh curve
    -- and the pre-expansion bands belong to the pre-expansion grid. They
    survive exactly when the rebuilt grid IS that grid: merging an INTERIOR
    pair leaves the axis endpoints alone and both grids are the same 200-point
    linspace, while merging the FIRST pair moves the grouped boundary off the
    original one and the two disagree. Per-level SEs -- the rated quantities --
    stay expanded either way.

    A hosted Piecewise no longer appears here: its level axis is positions
    0..L-1 and a collapse does not move it, so nothing is rebuilt and the
    fitted bands stay on their own fitted grid. That case is pinned by
    ``test_grouped_display_curve_keeps_the_stated_c0_corner`` below.
    """
    X, y = _frame()

    moved = _grouped_curve(X, y, Spline(kind="cr", n_knots=4), {"Mi000+Mi001": ["Mi000", "Mi001"]})
    assert moved.smooth_curve.se_log_relativity is None
    assert moved.smooth_curve.ci_lower is None
    assert moved.smooth_curve.ci_upper is None

    kept = _grouped_curve(X, y, Spline(kind="cr", n_knots=4), {"Mi001+Mi002": ["Mi001", "Mi002"]})
    assert kept.smooth_curve.se_log_relativity is not None
    assert len(kept.smooth_curve.se_log_relativity) == len(kept.smooth_curve.x)


def test_grouped_display_curve_keeps_the_stated_c0_corner() -> None:
    """A grouped Piecewise term draws its stated corner, not a smoothed bend.

    The whole contract of ``Piecewise`` is STATED breaks with C0 joins, so a
    display path that rounds the join shows a different function from the one
    that was fitted. The grouped expansion used to replace the fitted curve
    with a 200-point ``PchipInterpolator`` through the expanded level markers.
    PCHIP is C1 by construction -- one derivative per node -- so the drawn
    curve had no corner anywhere, and the stated break was not even a point on
    its grid. Measured on this fit: the drawn secants straddling the break
    differed by -0.008956 against the fitted slope change of -0.057576 (6.4x
    too shallow, and what little difference remained was the interpolant's own
    curvature rather than a join), and the drawn curve ran 0.006570 in
    log-relativity -- 0.66% in relativity -- away from the fitted shape. The
    same rebuild on a merge that also shifts the break gave 0.012590 (1.27%)
    and 5.9x; on a sharper stated kink it reached 0.0393 (4.0%) and 6.3x.

    Grouping a pair AFTER the break leaves the break at position 4, so the
    fitted shape is exactly the polyline through the three knot bands
    (Mi000, Mi004, Mi009) and this test needs no model internals to state it.
    """
    X, y = _frame()
    ti = _grouped_curve(X, y, Piecewise(breaks=["Mi004"]), {"Mi007+Mi008": ["Mi007", "Mi008"]})
    curve = ti.smooth_curve

    curve_x = np.asarray(curve.x, dtype=np.float64)
    curve_y = np.asarray(curve.log_relativity, dtype=np.float64)
    level_x = np.asarray(curve.level_x, dtype=np.float64)
    by_position = dict(zip(level_x.tolist(), np.asarray(ti.log_relativity, dtype=np.float64)))
    break_x = 4.0

    # The corner is a point ON the drawn curve, not something a renderer has to
    # interpolate its way through.
    assert np.any(curve_x == break_x), (
        f"the stated break sits at x={break_x} but the drawn grid is {curve_x!r}"
    )
    corner = int(np.flatnonzero(curve_x == break_x)[0])
    # ...and the curve meets the break band at exactly the value that band rates.
    assert curve_y[corner] == by_position[break_x]

    # The drawn curve IS the fitted piecewise-linear function: exactly the
    # polyline through the three knot bands, to the last bit.
    knot_positions = np.array([0.0, break_x, 8.0])
    knot_values = np.array([by_position[position] for position in knot_positions])
    assert np.abs(curve_y - np.interp(curve_x, knot_positions, knot_values)).max() == 0.0

    # And the join is a genuine corner: the two drawn segments meeting at the
    # break have different slopes, by the amount the fit states.
    left = (curve_y[corner] - curve_y[corner - 1]) / (curve_x[corner] - curve_x[corner - 1])
    right = (curve_y[corner + 1] - curve_y[corner]) / (curve_x[corner + 1] - curve_x[corner])
    fitted_left = (knot_values[1] - knot_values[0]) / (knot_positions[1] - knot_positions[0])
    fitted_right = (knot_values[2] - knot_values[1]) / (knot_positions[2] - knot_positions[1])
    assert right - left == pytest.approx(fitted_right - fitted_left, abs=1e-15)
    # Guards the line above against vacuity: a corner that small would be a
    # straight line either way. Measured -0.0576 here, against the -0.0090 the
    # PCHIP's own curvature produced.
    assert abs(right - left) > 0.04

    # The bands now belong to the curve they are drawn around, so they zip.
    assert curve.se_log_relativity is not None
    assert len(curve.se_log_relativity) == len(curve_x)


def test_grouped_piecewise_panel_draws_the_corner_on_the_canvas() -> None:
    """The corner survives all the way to the rendered line, not just the term.

    ``_collapsed_smooth_curve`` keeps whatever curve the term inference hands
    it, so a rounded curve upstream is a rounded curve on the canvas.
    """
    import matplotlib

    matplotlib.use("Agg")

    X, y = _frame()
    spec = OrderedCategorical(
        order=LEVELS,
        basis=Piecewise(breaks=["Mi004"]),
        grouping=_grouping(X, {"Mi007+Mi008": ["Mi007", "Mi008"]}),
    )
    model = SuperGLM(family="gaussian", selection_penalty=0.0, features={"band": spec})
    model.fit(X, y)
    curve = model.term_inference("band").smooth_curve
    expected_x = np.asarray(curve.x, dtype=np.float64)
    expected_y = np.asarray(curve.relativity, dtype=np.float64)

    ax = model.plot("band", X=X).axes[0]
    drawn = [line for line in ax.lines if len(line.get_xdata()) == len(expected_x)]
    assert len(drawn) == 1, f"expected one curve line of {len(expected_x)} vertices, got {drawn}"
    np.testing.assert_array_equal(np.asarray(drawn[0].get_xdata(), dtype=np.float64), expected_x)
    np.testing.assert_array_equal(np.asarray(drawn[0].get_ydata(), dtype=np.float64), expected_y)
    # Three vertices: the two boundary knots and the stated break. A polyline
    # through them is the fitted shape exactly; anything denser is a rebuild.
    np.testing.assert_array_equal(expected_x, [0.0, 4.0, 8.0])


def test_grouped_polynomial_display_curve_is_the_fitted_polynomial() -> None:
    """The same rebuild also replaced a hosted Polynomial's fitted curve.

    A Polynomial inner basis shares the Piecewise level-position axis, so a
    collapse does not move it either -- and there the rebuilt 200-point grid
    happened to MATCH the fitted one, so the pre-expansion bands were carried
    onto a curve that was not the one they were computed for. There is no
    corner to lose here; the shape itself was wrong.

    ``powers=[1, 2]`` fits a parabola, and on the uniform display grid a
    parabola has a constant second difference. Measured: 2.8e-16 spread on a
    mean of -3.08e-05 once the fitted curve is kept, against 9.59e-05 through
    the PCHIP -- a spread 3.1x the mean itself, i.e. not a parabola at all.
    """
    X, y = _frame()
    ti = _grouped_curve(X, y, Polynomial(powers=[1, 2]), {"Mi001+Mi002": ["Mi001", "Mi002"]})
    curve_x = np.asarray(ti.smooth_curve.x, dtype=np.float64)
    curve_y = np.asarray(ti.smooth_curve.log_relativity, dtype=np.float64)

    assert np.ptp(np.diff(curve_x)) < 1e-12  # uniform grid, so second differences compare
    second = np.diff(curve_y, 2)
    assert np.ptp(second) < 1e-14 * abs(float(second.mean())) + 1e-15


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
    """The Fit column's word matches the whole-term classification.

    A level carried by an unpenalized parametric block reported "smooth"
    before the fix, while the whole-term row and the exported workbook said
    the block is a plain group -- the word must match the model.
    """
    X, y = _frame()
    rng = np.random.default_rng(9)
    X = X.copy()
    missing = rng.random(len(X)) < 0.08
    X.loc[missing, "band"] = "MISSING"
    for basis, fit_word in (
        (Piecewise(breaks=["Mi004"]), "piecewise"),
        (Polynomial(powers=[1, 2]), "polynomial"),
    ):
        spec = OrderedCategorical(order=LEVELS, basis=basis, specials=["MISSING"])
        model = SuperGLM(family="gaussian", selection_penalty=0.0, features={"band": spec})
        model.fit(X, y)
        text = str(model.summary())
        assert "band[MISSING]" in text
        assert "free" in text
        level_line = next(line for line in text.splitlines() if "band[Mi000]" in line)
        assert fit_word in level_line
        assert "smooth" not in level_line

        # The editor-stale row builder carries the same vocabulary.
        from superglm.editor.apply import apply_edits_to_model_copy
        from superglm.editor.session import EditorSession

        session = EditorSession.from_model(model)
        term = session._require_term("band")
        term.edited_log_effect = term.edited_log_effect + 0.01
        edited = apply_edits_to_model_copy(model, {"band": term})
        stale_line = next(
            line for line in str(edited.summary()).splitlines() if "band[Mi000]" in line
        )
        assert fit_word in stale_line
        assert "smooth" not in stale_line
