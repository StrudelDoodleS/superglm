"""Summary reporting of free (special) levels on ordered categorical terms."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from superglm import OrderedCategorical, Spline, SuperGLM
from superglm.inference.summary import ModelSummary, _CoefRow

ORDERED = [str(i) for i in range(1, 11)]


def _specials_model():
    """Fit a 10-band ordered spline with a free MISSING level."""
    rng = np.random.default_rng(20260805)
    codes = np.repeat(np.arange(len(ORDERED)), 220)
    band_ordered = np.asarray(ORDERED, dtype=object)[codes]
    x = codes / (len(ORDERED) - 1.0)
    eta_ordered = -1.2 + 1.1 * x - 0.4 * x**2
    band_missing = np.full(600, "MISSING", dtype=object)
    eta_missing = np.full(600, -1.2 - 0.55)
    band = np.concatenate([band_ordered, band_missing])
    eta = np.concatenate([eta_ordered, eta_missing])
    weights = rng.uniform(0.7, 1.5, band.size)
    y = rng.poisson(np.exp(eta) * weights).astype(float)
    X = pd.DataFrame({"band": band})
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={
            "band": OrderedCategorical(
                order=ORDERED,
                specials=["MISSING"],
                base="first",
                basis=Spline(kind="ps", k=6),
            )
        },
    )
    model.fit(X, y, sample_weight=weights)
    return model, X, y, weights


def _level_rows(summary, feature="band"):
    return [row for row in summary._coef_rows if row.group == feature and not row.is_spline]


def test_specials_level_rows_carry_fit_marker_and_real_standard_errors():
    # False today: _CoefRow has no level_fit attribute at all, so both the
    # marker and the "free level keeps a std err" guard are unasserted.
    model, _, _, _ = _specials_model()
    rows = _level_rows(model.summary())

    assert [row.name for row in rows] == [f"band[{level}]" for level in [*ORDERED, "MISSING"]]
    assert [row.level_fit for row in rows] == ["smooth"] * len(ORDERED) + ["free"]

    free_row = rows[-1]
    # coef_tables.py:445 blanks the std err via `i < len(se_levels)` without
    # ever failing; assert the content, not that the call returned.
    assert free_row.se is not None
    assert np.isfinite(free_row.se) and free_row.se > 0.0
    assert free_row.ci_low is not None and np.isfinite(free_row.ci_low)
    assert free_row.ci_high is not None and np.isfinite(free_row.ci_high)
    assert free_row.ci_low < free_row.coef < free_row.ci_high


def test_ordered_term_without_specials_leaves_fit_marker_unset():
    # False today: level_fit does not exist, so nothing pins "no specials ->
    # no marker", which is what keeps existing OC output the same width.
    rng = np.random.default_rng(20260806)
    codes = np.repeat(np.arange(len(ORDERED)), 200)
    X = pd.DataFrame({"band": np.asarray(ORDERED, dtype=object)[codes]})
    weights = rng.uniform(0.7, 1.5, codes.size)
    y = rng.poisson(np.exp(-1.0 + 0.9 * codes / 9.0) * weights).astype(float)
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={
            "band": OrderedCategorical(order=ORDERED, base="first", basis=Spline(kind="ps", k=6))
        },
    )
    model.fit(X, y, sample_weight=weights)

    assert [row.level_fit for row in _level_rows(model.summary())] == [None] * len(ORDERED)


def test_editor_stale_row_builder_also_marks_free_levels():
    # False today: report_ops._build_editor_stale_coef_rows builds its own OC
    # level rows and would leave the marker unset on every edited model.
    from superglm.model.report_ops import _build_editor_stale_coef_rows

    model, _, _, _ = _specials_model()
    rows = _build_editor_stale_coef_rows(model)
    level_rows = [row for row in rows if row.group == "band" and not row.is_spline]

    assert [row.name for row in level_rows] == [f"band[{level}]" for level in [*ORDERED, "MISSING"]]
    assert [row.level_fit for row in level_rows] == ["smooth"] * len(ORDERED) + ["free"]


def test_level_display_relayout_preserves_the_fit_marker():
    # False today: nothing asserts that the summary_levels re-layout carries
    # level_fit onto the display rows it emits (summary_levels.py:222-232 is
    # the documented silent-degradation site).
    from superglm.inference.summary_levels import build_summary_level_display

    model, _, _, _ = _specials_model()
    summary = model.summary()
    display = build_summary_level_display(
        summary._coef_rows,
        specs=model._specs,
        groups=model._groups,
        level_display="expanded",
    )
    level_rows = [row for row in display.rows if row.group == "band" and not row.is_spline]

    assert [row.name for row in level_rows] == [f"band[{level}]" for level in [*ORDERED, "MISSING"]]
    assert [row.level_fit for row in level_rows] == ["smooth"] * len(ORDERED) + ["free"]
