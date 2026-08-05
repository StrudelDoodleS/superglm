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


def _model_info() -> dict[str, object]:
    return {
        "family": "Poisson",
        "link": "Log",
        "penalty": "None",
        "method": "ML",
        "n_obs": 2800,
        "effective_df": 6.0,
        "phi": 1.0,
        "pearson_chi2": 2790.0,
        "deviance": 2750.0,
        "log_likelihood": -1400.0,
        "aic": 2812.0,
        "aicc": 2812.1,
        "bic": 2850.0,
        "ebic": 2855.0,
        "converged": True,
        "n_iter": 5,
    }


def _synthetic_rows(*, with_specials: bool) -> list[_CoefRow]:
    """One ordered-spline group row plus three level rows."""
    fits = ["smooth", "smooth", "free"] if with_specials else [None, None, None]
    return [
        _CoefRow(name="Intercept", coef=-1.2, se=0.02, z=-60.0, p=0.0, ci_low=-1.24, ci_high=-1.16),
        _CoefRow(
            name="band",
            group="band",
            is_spline=True,
            n_params=5,
            active=True,
            group_norm=1.0,
            wald_chi2=42.0,
            wald_p=0.0004,
            ref_df=4.0,
            subgroup_type="ordered_spline",
            edf=3.5,
            smoothing_lambda=2.0,
        ),
        _CoefRow(name="band[1]", group="band", coef=0.0, se=None, level_fit=fits[0]),
        _CoefRow(
            name="band[10]",
            group="band",
            coef=0.42,
            se=0.05,
            ci_low=0.32,
            ci_high=0.52,
            level_fit=fits[1],
        ),
        _CoefRow(
            name="band[MISSING]",
            group="band",
            coef=-0.55,
            se=0.07,
            ci_low=-0.69,
            ci_high=-0.41,
            level_fit=fits[2],
        ),
    ]


def _summary(*, with_specials: bool) -> ModelSummary:
    rows = _synthetic_rows(with_specials=with_specials)
    return ModelSummary({}, _model_info(), rows)


def test_ascii_summary_renders_a_fit_column_only_when_levels_are_marked():
    # False today: the ASCII renderer has no Fit column, so "smooth"/"free"
    # never appear and the header never carries "Fit".
    text = str(_summary(with_specials=True))
    lines = text.splitlines()
    header = next(line for line in lines if "std err" in line)

    assert "Fit" in header
    assert re.search(r"band\[10\]\s+smooth\s", text)
    assert re.search(r"band\[MISSING\]\s+free\s", text)
    # The raw lookup key is undecorated: no asterisk, no suffix on the label.
    assert "band[MISSING]*" not in text
    assert "band[MISSING] (free)" not in text

    plain = str(_summary(with_specials=False))
    plain_header = next(line for line in plain.splitlines() if "std err" in line)
    assert "Fit" not in plain_header
    # The header block always says "(3.500 smooth)" for the edf breakdown, so
    # look for the marker only where the Fit column would render it.
    plain_levels = [line for line in plain.splitlines() if "band[" in line]
    assert len(plain_levels) == 3
    assert not [line for line in plain_levels if "smooth" in line or "free" in line]


def test_ascii_fit_column_is_included_in_the_box_width():
    # False today: there is no Fit column, so nothing pins that adding it
    # widens coef_W. Without the width fix the marked level lines overflow
    # W and the box loses its single line length.
    text = str(_summary(with_specials=True))
    boxed = [line for line in text.splitlines() if line.startswith(("║", "╠", "╟"))]

    assert len({len(line) for line in boxed}) == 1


def _row_widths(html: str) -> list[int]:
    """Effective column count of every <tr>, honouring colspan."""
    widths = []
    for chunk in re.findall(r"<tr>(.*?)</tr>", html, flags=re.S):
        total = 0
        for cell in re.findall(r"<t[dh][^>]*>", chunk):
            match = re.search(r'colspan="(\d+)"', cell)
            total += int(match.group(1)) if match else 1
        widths.append(total)
    return widths


def test_html_summary_renders_a_fit_column_only_when_levels_are_marked():
    # False today: the HTML renderer emits 9 columns and no Fit cell, so
    # ">free<" never appears in the output.
    html = _summary(with_specials=True)._repr_html_()

    assert ">Fit</td>" in html
    assert html.count(">free</td>") == 1
    assert html.count(">smooth</td>") == 2

    plain = _summary(with_specials=False)._repr_html_()
    assert ">Fit</td>" not in plain
    assert ">free</td>" not in plain


def test_html_summary_column_grid_absorbs_the_fit_column():
    # False today: every row is 9 wide. With the Fit column the whole grid
    # must be 10 wide, including the colspan'd spline and header rows.
    marked = _summary(with_specials=True)._repr_html_()
    plain = _summary(with_specials=False)._repr_html_()

    assert set(_row_widths(marked)) == {10}
    assert set(_row_widths(plain)) == {9}
