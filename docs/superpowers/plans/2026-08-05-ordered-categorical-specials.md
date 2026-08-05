# OrderedCategorical `specials=` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `specials=[...]` argument to `OrderedCategorical` that holds named levels out of the spline and fits them as free, unpenalized categorical level effects.

**Architecture:** `OrderedCategorical.build()` returns two `GroupInfo` objects instead of one — a penalized spline block built on the ordered rows only and row-expanded with zeros, and an unpenalized block of one indicator per special. The coefficient split lives entirely inside the term, so every downstream consumer keeps passing a full-width vector. Specials are appended to the level channel so the rating export, plotting and editor keep working, with one parallel boolean marking which levels were fitted free.

**Tech Stack:** Python 3.10+, NumPy, SciPy sparse, pandas, pytest, ruff, uv.

**Spec:** `docs/superpowers/specs/2026-08-05-ordered-categorical-specials-design.md` — read it before starting. It records *why* each rule exists.

## Global Constraints

- Branch `oc-specials`, worktree `.worktrees/oc-specials`, off `origin/master` @ `7109e7f`.
- Checks: `uv run pytest tests/ -q`, `uv run ruff check src/ tests/`, `uv run ruff format --check src/ tests/`, `uv run python run_test.py`.
- Target Python 3.10+. Public APIs are exported through `src/superglm/__init__.py`.
- Never run `git stash` — the stash stack is shared across worktrees in this repo.
- Every test must assert what the change *produces*. A test that passes against unmodified code is worthless; for each, the comment states what is false today.
- New solver, REML, family, input-boundary or feature behaviour requires focused regression tests.
- The PR body declares exactly one advisory impact. This feature is `release:minor`.

### Frozen interface

Every task uses these exact names.

```python
# On OrderedCategorical
self._specials: list[str]        # special labels, declaration order, [] when none
self._smooth_levels: list[str]   # ordered levels only; drives spline positions,
                                 # base selection, knot clamp, penalty
self._ordered_levels: list[str]  # _smooth_levels + _specials (display order, specials last)
self._level_to_value: dict       # SMOOTH LEVELS ONLY — specials deliberately absent
self.has_specials -> bool        # property
self._split_beta(beta) -> tuple[NDArray, NDArray]   # (spline_beta, special_beta)
```

Design matrix column order is a **contract**: spline block first, special block second. Special indicator column `j` corresponds to `self._specials[j]`.

```python
# GroupInfo for the special block
GroupInfo(columns=..., n_cols=len(self._specials), subgroup_name="special",
          penalized=False, penalty_matrix=None, projection=None, reparametrize=False)
# The spline block keeps subgroup_name=None.
# GroupSlice names become "<feature>" and "<feature>:special".

# TermInference gains
level_is_special: NDArray[np.bool_] | None = None   # parallel to .levels, None when no specials

# Summary coefficient rows gain
level_fit: str | None    # "smooth" | "free" | None  (rendered as a 'fit' column)

# Screening deferral
table.attrs["deferred_features"]: dict[str, str]     # feature name -> reason
```

`SmoothCurve.level_x` stays **smooth-levels-only**. Plots place specials at integer positions after the last ordered level, separated by a visible gap, derived from `level_is_special`.

---

### Task 1: Prerequisite: both plot backends draw the fitted curve


On `origin/master` the two backends draw **different curves for the same fitted term**.
`_plot_ordered_spline_panel` (`src/superglm/plotting/main_effects.py:524-635`) never references
`ti.smooth_curve`: it builds `x_pos = np.arange(n_levels)` (`543`) and a
`PchipInterpolator(x_pos, level_rel)` over `np.linspace(x_pos[0], x_pos[-1], 200)` (`585-596`).
The plotly panel draws the genuine fit from `curve.x` / `curve.relativity`
(`src/superglm/plotting/main_effects_plotly.py:1160-1175`) with markers at `curve.level_x`
(`1104-1108`). `_collapsed_smooth_curve` (`src/superglm/plotting/group_display.py:162-184`)
pushes the same integer-position PCHIP into plotly, and `resolve_grouped_level_display`
(`group_display.py:80-92`) returns `"collapsed"` as the **auto default** for OrderedCategorical,
so the fabricated curve reaches plotly through the default path too.

This task is committed **alone**, before any `specials=` work, because it deliberately changes
every existing OrderedCategorical matplotlib figure.

**Files:**
- Create: `tests/test_ordered_categorical_plot_backends.py`
- Modify: `src/superglm/plotting/common.py:66-67` (insert helper before `_exposure_kde`)
- Modify: `src/superglm/plotting/main_effects.py:16-35` (import list), `:524-635` (the panel)
- Modify: `src/superglm/plotting/group_display.py:61-71` (call site), `:162-184` (the function)
- Modify: `src/superglm/plotting/main_effects_plotly.py:14-34` (import list), `:1687-1696` (`_ordered_bar_width`)
- Test: `tests/test_ordered_categorical_plot_backends.py`

**Interfaces:**
- Consumes: nothing from earlier tasks. Existing types only —
  `SmoothCurve(x, log_relativity, relativity, level_x, se_log_relativity, ci_lower, ci_upper)`,
  a frozen dataclass at `src/superglm/inference/_term_types.py:38-52`;
  `TermInference.smooth_curve: SmoothCurve | None` (`_term_types.py:55-…`);
  `GroupedTermDisplay(term, source_levels, source_indices, collapsed)` (`group_display.py:17-24`);
  `grouped_level_exposure(display, X, sample_weight)` (`group_display.py:95-122`).
- Produces:
  - `superglm.plotting.common._ordered_level_spacing(x: NDArray) -> float`
  - `_plot_ordered_spline_panel(ax, ti, interval, *, X=None, sample_weight=None, weight_label="Weight", display=None)` — signature unchanged, now draws `ti.smooth_curve` and positions everything at `ti.smooth_curve.level_x`. Task 8 (specials rendering) replaces this function's body and depends on `x_pos` being derived from `level_x`.
  - `superglm.plotting.group_display._collapsed_smooth_curve(ti: TermInference, groups: list[list[int]]) -> SmoothCurve | None` — **signature change** from `(ti, log_rel, n_levels)`.
  - `_ordered_bar_width(x)` in `main_effects_plotly.py` delegates to `_ordered_level_spacing`.
  - `resolve_grouped_level_display` is **not** changed: `"collapsed"` stays the auto default for OC, and is now safe because the collapsed projection no longer rebuilds the curve.

---

- [ ] **Step 1: Write the failing test**

Create `tests/test_ordered_categorical_plot_backends.py`. Note `plotly` lives in the optional
`plotting` extra, so the cross-backend test uses `pytest.importorskip`; the other three run in
the plain dev environment.

```python
"""The two plot backends must draw the same fitted curve for an OC term."""

import numpy as np
import pandas as pd
import pytest

from superglm import Numeric, OrderedCategorical, Spline, SuperGLM
from superglm.editor import EditorSession
from superglm.plotting.group_display import project_grouped_term_for_display

AGE_VALUES = {
    "18-24": 21.0,
    "25-34": 30.0,
    "35-49": 42.0,
    "50-64": 57.0,
    "65-80": 72.0,
}


def _age_band_frame(seed: int, n: int):
    rng = np.random.default_rng(seed)
    levels = list(AGE_VALUES)
    band = rng.choice(levels, n, p=[0.15, 0.25, 0.28, 0.20, 0.12])
    mileage = rng.normal(0.0, 1.0, n)
    sample_weight = rng.uniform(0.5, 1.5, n)
    age = np.array([AGE_VALUES[value] for value in band], dtype=np.float64)
    y = 0.8 + 0.25 * np.sin(age / 22.0) + 0.04 * mileage + rng.normal(0.0, 0.05, n)
    return pd.DataFrame({"age_band": band, "mileage": mileage}), y, sample_weight


@pytest.fixture
def ordered_spline_model():
    X, y, sample_weight = _age_band_frame(20260805, 800)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={
            "age_band": OrderedCategorical(values=AGE_VALUES, basis=Spline(kind="ps", k=6)),
            "mileage": Numeric(),
        },
    )
    model.fit(X, y, sample_weight=sample_weight)
    return X, sample_weight, model


@pytest.fixture
def collapsed_ordered_spline_model():
    X, y, sample_weight = _age_band_frame(20260806, 700)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={
            "age_band": OrderedCategorical(values=AGE_VALUES, basis=Spline(kind="ps", k=5)),
            "mileage": Numeric(),
        },
    )
    model.fit(X, y, sample_weight=sample_weight)
    session = EditorSession.from_model(
        model,
        terms=["age_band"],
        train_data=(X, y, sample_weight),
    )
    session.select_levels("age_band", ["18-24", "25-34", "35-49"])
    collapsed = session.replace_with_collapsed_levels("age_band", method="fit")
    return X, sample_weight, collapsed


def _matplotlib_curve(ax, n_points: int):
    """The one drawn line with as many vertices as the fitted curve."""
    lines = [line for line in ax.lines if len(line.get_xdata()) == n_points]
    assert len(lines) == 1, f"expected exactly one curve line, got {len(lines)}"
    return (
        np.asarray(lines[0].get_xdata(), dtype=np.float64),
        np.asarray(lines[0].get_ydata(), dtype=np.float64),
    )


def test_matplotlib_ordered_panel_draws_the_fitted_curve(ordered_spline_model):
    # False today: the panel draws PchipInterpolator(arange(K), relativity) over
    # linspace(0, K-1, 200), so curve_x is [0 .. 4] while the fitted curve
    # spans the level values [21 .. 72].
    import matplotlib

    matplotlib.use("Agg")

    X, sample_weight, model = ordered_spline_model
    ti = model.term_inference("age_band")
    fig = model.plot("age_band", X=X, sample_weight=sample_weight)

    curve_x, curve_y = _matplotlib_curve(fig.axes[0], len(ti.smooth_curve.x))
    np.testing.assert_allclose(curve_x, np.asarray(ti.smooth_curve.x, dtype=np.float64))
    np.testing.assert_allclose(curve_y, np.asarray(ti.smooth_curve.relativity, dtype=np.float64))


def test_matplotlib_ordered_panel_places_levels_at_fitted_positions(ordered_spline_model):
    # False today: x_pos is arange(K), so the ticks are [0, 1, 2, 3, 4] and the
    # exposure bars are centred there instead of at level_x = [21 .. 72].
    import matplotlib

    matplotlib.use("Agg")

    X, sample_weight, model = ordered_spline_model
    ti = model.term_inference("age_band")
    level_x = np.asarray(ti.smooth_curve.level_x, dtype=np.float64)
    fig = model.plot("age_band", X=X, sample_weight=sample_weight)

    ax = fig.axes[0]
    np.testing.assert_allclose(np.asarray(ax.get_xticks(), dtype=np.float64), level_x)
    assert [tick.get_text() for tick in ax.get_xticklabels()] == list(ti.levels)

    bars = fig.axes[1].patches
    centres = np.asarray(
        [patch.get_x() + patch.get_width() / 2.0 for patch in bars], dtype=np.float64
    )
    np.testing.assert_allclose(centres, level_x)


def test_both_backends_draw_the_same_ordered_curve(ordered_spline_model):
    # False today: matplotlib draws a PCHIP over [0, 4] and plotly draws the
    # fitted spline over [21, 72]; the x arrays do not even overlap.
    go = pytest.importorskip("plotly.graph_objects")
    import matplotlib

    matplotlib.use("Agg")

    X, sample_weight, model = ordered_spline_model
    ti = model.term_inference("age_band")
    mpl_fig = model.plot("age_band", X=X, sample_weight=sample_weight)
    mpl_x, mpl_y = _matplotlib_curve(mpl_fig.axes[0], len(ti.smooth_curve.x))

    plotly_fig = model.plot(
        ["age_band", "mileage"],
        engine="plotly",
        X=X,
        sample_weight=sample_weight,
    )
    curve = next(
        trace
        for trace in plotly_fig.data
        if isinstance(trace, go.Scatter) and trace.name == "Smooth curve"
    )
    np.testing.assert_allclose(mpl_x, np.asarray(curve.x, dtype=np.float64))
    np.testing.assert_allclose(mpl_y, np.asarray(curve.y, dtype=np.float64))


def test_collapsed_display_keeps_the_fitted_curve(collapsed_ordered_spline_model):
    # False today: _collapsed_smooth_curve replaces the curve with a PCHIP
    # through the collapsed relativities at arange(3), so curve.x becomes
    # [0 .. 2] and level_x becomes [0, 1, 2] rather than the group-mean
    # positions on the fitted axis.
    X, sample_weight, model = collapsed_ordered_spline_model
    ti = model.term_inference("age_band")
    display = project_grouped_term_for_display(model, ti, "auto")

    assert display.collapsed is True
    assert display.term.levels == ["18-24+25-34+35-49", "50-64", "65-80"]

    np.testing.assert_allclose(
        np.asarray(display.term.smooth_curve.x, dtype=np.float64),
        np.asarray(ti.smooth_curve.x, dtype=np.float64),
    )
    np.testing.assert_allclose(
        np.asarray(display.term.smooth_curve.relativity, dtype=np.float64),
        np.asarray(ti.smooth_curve.relativity, dtype=np.float64),
    )
    np.testing.assert_allclose(
        np.asarray(display.term.smooth_curve.level_x, dtype=np.float64),
        [np.mean([21.0, 30.0, 42.0]), 57.0, 72.0],
    )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --extra plotting pytest tests/test_ordered_categorical_plot_backends.py -v`

Expected: **4 failed**. Every failure is an `np.testing.assert_allclose` mismatch of 100% of
elements. The first one reads:

```
    Mismatched elements: 200 / 200 (100%)
    First 5 mismatches are at indices:
     [0]: 0.0 (ACTUAL), 21.0 (DESIRED)
     [1]: 0.020100502512562814 (ACTUAL), 21.256281407035175 (DESIRED)
    Max absolute difference among violations: 68.
tests/test_ordered_categorical_plot_backends.py:91: AssertionError
```

`test_..._places_levels_at_fitted_positions` reports `Mismatched elements: 5 / 5 (100%)`
(ticks `[0, 1, 2, 3, 4]` vs `[21, 30, 42, 57, 72]`), `test_both_backends_...` reports
`Mismatched elements: 200 / 200 (100%)`, and `test_collapsed_display_...` reports
`Max absolute difference among violations: 70.` (collapsed curve x `[0 .. 2]` vs `[21 .. 72]`).

- [ ] **Step 3: Add the shared level-spacing helper**

In `src/superglm/plotting/common.py`, insert immediately **before** `_exposure_kde`
(currently line 68, right after the `_PLOTLY_DENSITY_SCALE` list that ends at line 65):

```python
def _ordered_level_spacing(x: NDArray) -> float:
    """Smallest positive gap between ordered-category level positions.

    Used by both backends to size exposure bars and axis padding when
    ``SmoothCurve.level_x`` places levels at their fitted x-positions
    rather than at consecutive integers.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size <= 1:
        return 1.0
    diffs = np.diff(np.sort(x))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 1.0
    return float(diffs.min())
```

`numpy as np` and `NDArray` are already imported at `common.py:5-6`.

- [ ] **Step 4: Draw the fitted curve in the matplotlib panel**

First extend the import block in `src/superglm/plotting/main_effects.py` (lines 16-35). Insert
after `_make_continuous_figure,` on line 34 — ruff's isort order puts `_o…` after `_m…`:

```python
from superglm.plotting.common import (
    _EXP_EDGE,
    _EXP_EDGE_LW,
    _EXP_FILL,
    _KNOT_COLOR,
    _LINE_COLOR,
    _LINE_WIDTH,
    _PW_ALPHA,
    _PW_EDGE_ALPHA,
    _PW_EDGE_LW,
    _PW_FILL,
    _REF_COLOR,
    _REF_LW,
    _SIM_ALPHA,
    _SIM_EDGE_ALPHA,
    _SIM_EDGE_LW,
    _SIM_FILL,
    _exposure_kde,
    _make_continuous_figure,
    _ordered_level_spacing,
)
```

Then, in `_plot_ordered_spline_panel`, replace the docstring and the four lines that follow it
(`main_effects.py:534-543`):

```python
    """Render an OrderedCategorical(spline) panel.

    Levels sit at their fitted x-positions (``smooth_curve.level_x``) with
    the fitted smooth curve drawn through them, plus per-level error bars
    and sample_weight bars.  This is the same curve the plotly backend
    draws in ``_add_categorical_term_trace``.
    """
    levels = list(ti.levels)
    level_rel = np.asarray(ti.relativity)
    n_levels = len(levels)
    curve = ti.smooth_curve
    if curve is not None and curve.level_x is not None:
        x_pos = np.asarray(curve.level_x, dtype=np.float64)
    else:
        x_pos = np.arange(n_levels, dtype=np.float64)
    spacing = _ordered_level_spacing(x_pos)
```

Change the exposure bar width (`main_effects.py:564`) from `width=0.6` to:

```python
            width=spacing * 0.6,
```

Replace the PCHIP block (`main_effects.py:583-596`) with the fitted curve:

```python
    # Fitted smooth curve (never an interpolation through the markers)
    if curve is not None:
        ax.plot(
            np.asarray(curve.x, dtype=np.float64),
            np.asarray(curve.relativity, dtype=np.float64),
            color=_LINE_COLOR,
            linewidth=_LINE_WIDTH,
            alpha=0.6,
            zorder=4,
        )
```

Replace the fixed x-limit (`main_effects.py:629`):

```python
    ax.set_xlim(float(x_pos.min()) - spacing / 2.0, float(x_pos.max()) + spacing / 2.0)
```

The `ax.set_xticks(x_pos)` / `set_xticklabels(levels)` pair at `625-628`, the errorbar and
scatter branches at `599-623`, and `rot = 45 if n_levels > 8 else 0` all stay as they are — they
already consume `x_pos` and `n_levels`.

- [ ] **Step 5: Run the matplotlib tests**

Run: `uv run --extra plotting pytest tests/test_ordered_categorical_plot_backends.py -v`

Expected: 3 passed, 1 failed. `test_matplotlib_ordered_panel_draws_the_fitted_curve`,
`test_matplotlib_ordered_panel_places_levels_at_fitted_positions` and
`test_both_backends_draw_the_same_ordered_curve` now pass;
`test_collapsed_display_keeps_the_fitted_curve` still fails with
`Max absolute difference among violations: 70.` because the collapsed projection is untouched.

- [ ] **Step 6: Stop the collapsed display from rebuilding the curve**

In `src/superglm/plotting/group_display.py`, change the call site at line 70 inside
`project_grouped_term_for_display`:

```python
    display_term = replace(
        ti,
        levels=display_levels,
        log_relativity=log_rel,
        relativity=np.exp(log_rel),
        se_log_relativity=_collapse_array(ti.se_log_relativity, group_indices),
        ci_lower=_collapse_array(ti.ci_lower, group_indices),
        ci_upper=_collapse_array(ti.ci_upper, group_indices),
        spline=None,
        smooth_curve=_collapsed_smooth_curve(ti, group_indices),
    )
```

`spline=None` stays — `tests/test_relativities.py:602-655` depends on it to suppress stale knot
diagnostics. Then replace the whole of `_collapsed_smooth_curve` (`group_display.py:162-184`):

```python
def _collapsed_smooth_curve(
    ti: TermInference,
    groups: list[list[int]],
) -> SmoothCurve | None:
    """Keep the fitted curve and move each marker to its group's mean position.

    The curve itself is never rebuilt: collapsing levels is a display
    operation, and re-interpolating through the collapsed markers would
    draw a shape the model never fitted.
    """
    curve = ti.smooth_curve
    if curve is None or curve.level_x is None:
        return curve
    level_x = np.asarray(curve.level_x, dtype=np.float64)
    collapsed_x = np.asarray(
        [float(np.mean(level_x[indices])) for indices in groups], dtype=np.float64
    )
    return replace(curve, level_x=collapsed_x)
```

`replace` is already imported at `group_display.py:5`; the local
`from scipy.interpolate import PchipInterpolator` goes away with the old body. `SmoothCurve` is
still used in the return annotation and `NDArray` is still used by `_collapse_array`, so the
module imports are unchanged.

Why the mean position is right rather than a fabrication: in a collapsed model every member of a
group carries the same log-relativity, so the expanded display curve
(`_expand_grouped_term`, `src/superglm/inference/_term_helpers.py:186-232`) is flat across the
group and passes through the group's value at *every* member position, the mean included. The
collapsed marker therefore lands exactly on the curve it is plotted against.

- [ ] **Step 7: Run the whole new test file**

Run: `uv run --extra plotting pytest tests/test_ordered_categorical_plot_backends.py -v`

Expected: PASS (4 passed).

- [ ] **Step 8: Fold the duplicated bar-width logic into the shared helper**

In `src/superglm/plotting/main_effects_plotly.py`, add `_ordered_level_spacing,` to the
`superglm.plotting.common` import block after `_hex_to_rgba,` (line 33), then replace
`_ordered_bar_width` (lines 1687-1696):

```python
def _ordered_bar_width(x: NDArray) -> float:
    """Reasonable bar width for ordered-category numeric positions."""
    return _ordered_level_spacing(x) * 0.72
```

This is behaviour-identical for every fitted OC term. It differs only in the degenerate branches
the old body special-cased (fewer than two positions, or all positions equal), where the width
becomes `0.72` instead of `0.6`; a fitted OC spline always has at least two distinct level
values, and no test pins the value.

- [ ] **Step 9: Run the regression set and the linters**

Run:

```bash
uv run --extra plotting pytest tests/test_relativities.py tests/test_plot_api.py \
  tests/test_categorical_ux.py tests/test_ordered_categorical.py \
  tests/test_plot_diagnostics.py tests/test_ordered_categorical_plot_backends.py -q
uv run --extra plotting pytest tests/test_editor.py tests/editor -q -m "not browser"
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
```

Expected: PASS throughout, `All checks passed!`, and `N files already formatted`. In particular
these four keep passing unchanged, and are worth reading before you assume otherwise:

- `tests/test_relativities.py:530-558` `test_ordered_categorical_plot_defaults_to_collapsed_group_display` — its fixture (`tests/test_relativities.py:86-97`) uses `basis="step"`, so `smooth_curve` is `None`, the term renders through `_plot_categorical_panel_vertical` (`main_effects.py:637`), and `_collapsed_smooth_curve` returns `None` as before.
- `tests/test_relativities.py:602-655` `test_plotly_collapsed_ordered_categorical_suppresses_stale_knot_diagnostics` — spline mode *with* grouping, so it does exercise the new `_collapsed_smooth_curve`; it asserts no `"Interior knots"` traces, which comes from `spline=None` at `group_display.py:69`, left untouched.
- `tests/test_plot_api.py:763-806` `test_plotly_ordered_categorical_spline_uses_numeric_axis` — asserts plotly markers at `midpoints.values()` and a 200-point numeric curve. This is exactly the behaviour matplotlib is being brought into line with.
- `tests/test_dataframe_boundary_diagnostics.py:200-213` — compares polars-built against pandas-built figures; both sides move together.

- [ ] **Step 10: Commit**

```bash
git add tests/test_ordered_categorical_plot_backends.py \
        src/superglm/plotting/common.py \
        src/superglm/plotting/main_effects.py \
        src/superglm/plotting/main_effects_plotly.py \
        src/superglm/plotting/group_display.py
git commit -m "fix: draw the fitted curve in the matplotlib OrderedCategorical panel

The matplotlib panel invented a PCHIP interpolation through the level
relativities at integer positions and never looked at ti.smooth_curve,
while the plotly panel drew the genuine fit at the fitted level
positions. The same integer-position PCHIP reached plotly as well,
through _collapsed_smooth_curve, which the auto display default selects
for every grouped OrderedCategorical.

Both backends now draw ti.smooth_curve. Collapsing levels for display
moves each marker to its group's mean position and leaves the curve
alone.

This deliberately changes every existing OrderedCategorical matplotlib
figure: levels, ticks and exposure bars move from 0..K-1 to their
fitted x-positions, and the line drawn is the fitted spline rather than
an interpolation through the markers.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `specials=` constructor, normalisation and validation

**Files:**
- Modify: `src/superglm/features/ordered_categorical.py:26-47` (module helpers), `:131-296` (constructor), `:343-362` (`_choose_base`)
- Test: `tests/test_ordered_categorical_specials.py` (create)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `OrderedCategorical(specials=[...])`; `self._specials: list[str]`, `self._smooth_levels: list[str]`, `self._ordered_levels: list[str]` (smooth then specials), `self._level_to_value` over smooth levels only, `self._n_levels == len(self._smooth_levels)`, `self.has_specials -> bool`; module helpers `_require_two_smooth_levels(smooth_levels, special_set)` and `_require_no_grouped_specials(grouping, special_set)`. `_choose_base` never selects a special.

- [ ] **Step 1: Write the failing test for the basic split**

Create `tests/test_ordered_categorical_specials.py`:

```python
"""Specials: levels held out of the smooth and fitted as free level effects."""

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, OrderedCategorical, Spline, SuperGLM

ORDERED = [str(i) for i in range(1, 11)]
SPECIAL = "MISSING"


def _oc(**kwargs):
    params = dict(order=list(ORDERED), specials=[SPECIAL], basis=Spline(kind="ps", k=8))
    params.update(kwargs)
    return OrderedCategorical(**params)


# Today `specials` is not a parameter at all, so construction raises TypeError.
def test_specials_are_held_out_of_the_smooth_levels():
    spec = _oc()
    assert spec._specials == [SPECIAL]
    assert spec._smooth_levels == ORDERED
    assert spec._ordered_levels == ORDERED + [SPECIAL]
    assert SPECIAL not in spec._level_to_value
    assert set(spec._level_to_value) == set(ORDERED)
    assert spec._n_levels == len(ORDERED)
    assert spec.has_specials is True


def test_no_specials_leaves_everything_unchanged():
    spec = OrderedCategorical(order=list(ORDERED), basis=Spline(kind="ps", k=8))
    assert spec._specials == []
    assert spec._smooth_levels == ORDERED
    assert spec._ordered_levels == ORDERED
    assert spec.has_specials is False
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_ordered_categorical_specials.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'specials'`

- [ ] **Step 3: Add the parameter and normalise the level lists**

In `src/superglm/features/ordered_categorical.py`, add `specials: list[str] | None = None` to the signature after `grouping: Any = None` (line 142), and add to the docstring Parameters block:

```
    specials : list[str] or None
        Level labels held out of the smooth and fitted as free, unpenalized
        level effects — one indicator column and one coefficient each. Use for
        levels that are structurally different rather than merely sparse (a
        ``MISSING`` band, a structural zero); the penalty already handles
        sparse bands better than free levels do. A label listed here is
        removed from ``order``/``values`` if also present there, and never
        receives a numeric position on the smooth's axis.
```

Immediately after the `values`/`order` XOR checks (after line 149), normalise:

```python
        self._specials: list[str] = []
        if specials is not None:
            for lev in specials:
                if lev in self._specials:
                    raise ValueError(f"Duplicate special level {lev!r} in 'specials'.")
                self._specials.append(lev)
        special_set = set(self._specials)

        if special_set:
            if values is not None:
                values = {k: v for k, v in values.items() if k not in special_set}
            else:
                order = [lev for lev in order if lev not in special_set]
```

Then replace the level-derivation block at lines 232-243 with:

```python
        self._smooth_levels: list[str] = []
        self._ordered_levels: list[str] = []

        # Derive smooth levels and numeric values
        if values is not None:
            sorted_items = sorted(values.items(), key=lambda kv: kv[1])
            self._smooth_levels = [k for k, _ in sorted_items]
            self._level_to_value = dict(values)
        else:
            self._smooth_levels = list(order)
            n = len(order)
            vals = np.linspace(0.0, 1.0, n) if n > 1 else np.array([0.0])
            self._level_to_value = dict(zip(order, vals.tolist()))
        self._ordered_levels = list(self._smooth_levels)
```

At the end of the grouping branch (after line 269, `self._ordered_levels = list(grouping.grouped_levels)`), the grouped path must set `_smooth_levels` too, and both paths then append the specials. Replace lines 268-272 with:

```python
            self._level_to_value = grouped_ltv
            self._smooth_levels = [
                lev for lev in grouping.grouped_levels if lev not in special_set
            ]
            self._known_levels = set(grouping.all_original_levels) | special_set
        else:
            self._known_levels = set(self._smooth_levels) | special_set
        self._ordered_levels = list(self._smooth_levels) + list(self._specials)
        self._n_levels = len(self._smooth_levels)
```

Note `self._known_levels` is now set in both branches here; delete the original assignment at line 250 inside the grouping branch and at line 271.

Add the property after `__repr__` (line 296):

```python
    @property
    def has_specials(self) -> bool:
        """True when one or more levels are fitted as free effects."""
        return bool(self._specials)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_ordered_categorical_specials.py -v`
Expected: PASS (both tests)

- [ ] **Step 5: Write the failing validation tests**

Append to `tests/test_ordered_categorical_specials.py`:

```python
# Each of these raises nothing today — `specials` does not exist, and once it
# does, the naive implementation accepts all of them.
def test_label_in_both_order_and_specials_is_popped_from_order():
    spec = OrderedCategorical(
        order=[SPECIAL] + list(ORDERED), specials=[SPECIAL], basis=Spline(kind="ps", k=8)
    )
    assert spec._smooth_levels == ORDERED
    assert SPECIAL not in spec._level_to_value
    # Positions are computed over the survivors, so band 1 is at 0.0 and band 10 at 1.0.
    assert spec._level_to_value["1"] == pytest.approx(0.0)
    assert spec._level_to_value["10"] == pytest.approx(1.0)


def test_label_in_both_values_and_specials_is_popped_from_values():
    spec = OrderedCategorical(
        values={SPECIAL: -1.0, "a": 1.0, "b": 2.0, "c": 3.0},
        specials=[SPECIAL],
        basis=Spline(kind="ps", k=3),
    )
    assert spec._smooth_levels == ["a", "b", "c"]
    assert SPECIAL not in spec._level_to_value


def test_duplicate_special_is_rejected():
    with pytest.raises(ValueError, match="Duplicate special level"):
        _oc(specials=[SPECIAL, SPECIAL])


def test_fewer_than_two_smooth_levels_is_rejected():
    with pytest.raises(ValueError, match="at least two"):
        OrderedCategorical(order=["a", SPECIAL], specials=[SPECIAL], basis=Spline(kind="ps", k=2))


def test_specials_with_step_basis_is_rejected():
    with pytest.raises(ValueError, match="basis='step'"):
        OrderedCategorical(order=list(ORDERED), specials=[SPECIAL], basis="step")


def test_explicit_special_base_is_rejected():
    with pytest.raises(ValueError, match="reporting base"):
        _oc(base=SPECIAL)


def test_grouping_that_merges_a_special_is_rejected():
    # The spec's validation table forbids mixing a special with ordered levels
    # in one group, but only the editor's collapse path enforces it. Built
    # directly, the special is silently smoothed inside group "6+MISSING"
    # while `_specials` still lists it as free — an inconsistent spec state
    # with no error anywhere.
    from superglm.features.grouping import collapse_levels

    grouping = collapse_levels(
        np.array(ORDERED + [SPECIAL], dtype=object),
        groups={"6+MISSING": ["6", SPECIAL]},
        order=ORDERED + [SPECIAL],
    )
    with pytest.raises(ValueError, match="free level"):
        _oc(grouping=grouping)


def test_grouping_that_collapses_every_ordered_level_is_rejected():
    # The at-least-two-smooth-levels check runs on the pre-grouping level list,
    # so a grouping that leaves one smooth level reaches the spline build with
    # a single distinct position.
    from superglm.features.grouping import collapse_levels

    grouping = collapse_levels(
        np.array(ORDERED + [SPECIAL], dtype=object),
        groups={"all": list(ORDERED)},
        order=ORDERED + [SPECIAL],
    )
    with pytest.raises(ValueError, match="at least two"):
        _oc(grouping=grouping)
```

- [ ] **Step 6: Run them to verify they fail**

Run: `uv run pytest tests/test_ordered_categorical_specials.py -v -k "popped or duplicate or fewer or step_basis or special_base or grouping_that"`
Expected: FAIL — `test_duplicate_special_is_rejected`, `test_fewer_than_two_smooth_levels_is_rejected`, `test_specials_with_step_basis_is_rejected`, `test_explicit_special_base_is_rejected`, `test_grouping_that_merges_a_special_is_rejected` and `test_grouping_that_collapses_every_ordered_level_is_rejected` all fail with `Failed: DID NOT RAISE`. (The two `popped` tests pass already from Step 3 — keep them, they pin the normalisation.)

- [ ] **Step 7: Add the remaining validation**

Add two module-level helpers after `_spline_kind_name` (which ends at line 47), so the
pre-grouping and post-grouping paths raise the same message rather than two drifting copies:

```python
def _require_two_smooth_levels(smooth_levels: list[str], special_set: set[str]) -> None:
    """Both level-derivation paths must leave at least two levels to smooth."""
    if special_set and len(smooth_levels) < 2:
        raise ValueError(
            "OrderedCategorical needs at least two non-special levels to fit a "
            f"smooth; got {smooth_levels!r} after removing {sorted(special_set)!r}. "
            "Use Categorical(...) for independent level effects."
        )


def _require_no_grouped_specials(grouping: Any, special_set: set[str]) -> None:
    """Refuse a grouping that merges a special into any other level.

    Merging a special into an ordered group would smooth it after all, while
    ``_specials`` still reports it free — an inconsistent spec with no error.
    Merging two specials is refused for the same reason the editor refuses it
    (``_require_no_special_members``): the group label would have to replace
    both members in ``specials=``.
    """
    if not special_set or grouping is None:
        return
    for label, originals in grouping.group_to_originals.items():
        members = [str(member) for member in originals]
        if len(members) < 2:
            continue
        merged = [member for member in members if member in special_set]
        if merged:
            joined = ", ".join(repr(member) for member in merged)
            raise ValueError(
                f"OrderedCategorical grouping merges free level(s) {joined} into group "
                f"{label!r}. Specials are fitted outside the smooth and may not be "
                "grouped; group the ordered levels only."
            )
```

After the `special_set` normalisation added in Step 3, add the smooth-level-count check. Place it
after the level derivation, immediately before `self._grouping = grouping` (line 246):

```python
        _require_two_smooth_levels(self._smooth_levels, special_set)
```

In the step-mode guard block, extend the existing `select` check at lines 223-224:

```python
        if self.basis == "step" and resolved_select:
            raise ValueError("select=True is not supported with basis='step'.")
        if self.basis == "step" and special_set:
            raise ValueError(
                "specials= is not supported with basis='step', which is deprecated. "
                "Use basis=Spline(...) for a smoothed ordinal term with free levels."
            )
```

Add the base check and the two grouping-aware checks immediately after `self._ordered_levels` /
`self._n_levels` are set (the grouping branch has just recomputed `_smooth_levels`, so the
level-count check has to run again here — the pre-grouping copy sees the ungrouped list and a
grouping that collapses every ordered level into one would otherwise reach the spline build with
a single distinct position):

```python
        _require_no_grouped_specials(grouping, special_set)
        _require_two_smooth_levels(self._smooth_levels, special_set)
        if base in special_set:
            raise ValueError(
                f"OrderedCategorical reporting base {base!r} is a special level. The base "
                "anchors every reported relativity and must lie on the smooth; choose one "
                f"of {self._smooth_levels!r}."
            )
```

- [ ] **Step 8: Run the tests**

Run: `uv run pytest tests/test_ordered_categorical_specials.py -v`
Expected: PASS (all ten)

- [ ] **Step 9: Write the failing test for base selection**

Append:

```python
# Today `_choose_base` iterates `_ordered_levels`, so once specials are appended
# there, `most_exposed` picks MISSING whenever it dominates exposure — and it
# usually does on a real book.
def test_most_exposed_base_never_selects_a_special():
    spec = _oc()
    x = np.array(["1"] * 10 + [SPECIAL] * 1000, dtype=object)
    weight = np.ones(len(x))
    spec._choose_base(x, weight)
    assert spec._base_level != SPECIAL
    assert spec._base_level in ORDERED
```

- [ ] **Step 10: Run it to verify it fails**

Run: `uv run pytest tests/test_ordered_categorical_specials.py::test_most_exposed_base_never_selects_a_special -v`
Expected: FAIL — `assert 'MISSING' != 'MISSING'`

- [ ] **Step 11: Restrict base selection to the smooth levels**

In `_choose_base` (lines 343-362), replace every reference to `self._ordered_levels` with `self._smooth_levels`:

```python
    def _choose_base(self, x: NDArray, sample_weight: NDArray | None) -> None:
        """Choose the base level for relativities.

        Specials are excluded: the base anchors every reported relativity and
        must lie on the smooth. On a real book a MISSING band is often the most
        exposed level, so ``most_exposed`` would otherwise select it.
        """
        if self._base_level and self._base_level in self._smooth_levels:
            return

        if self.base == "most_exposed" and sample_weight is not None:
            exp_by_level = {
                lev: float(sample_weight[x == lev].sum()) for lev in self._smooth_levels
            }
            self._base_level = max(exp_by_level, key=exp_by_level.get)
        elif self.base == "most_exposed" and sample_weight is None:
            self._base_level = self._smooth_levels[0]
        elif self.base == "first":
            self._base_level = self._smooth_levels[0]
        elif self.base in self._smooth_levels:
            self._base_level = self.base
        else:
            raise ValueError(f"Base '{self.base}' not found in levels: {self._smooth_levels}")

        self._non_base = [lev for lev in self._smooth_levels if lev != self._base_level]
```

- [ ] **Step 12: Run the full OC suite to check for regressions**

Run: `uv run pytest tests/test_ordered_categorical.py tests/test_ordered_categorical_api.py tests/test_ordered_categorical_inference.py tests/test_ordered_categorical_specials.py -q`
Expected: PASS — the existing suites are unaffected because with no specials `_smooth_levels == _ordered_levels`.

- [ ] **Step 13: Commit**

```bash
git add src/superglm/features/ordered_categorical.py tests/test_ordered_categorical_specials.py
git commit -m "feat: accept specials= on OrderedCategorical and hold them out of the smooth

Specials are normalised out of order=/values=, never receive a numeric
position, and are excluded from reporting-base selection — most_exposed
would otherwise anchor the term on a MISSING band, which is often the
most exposed level on a real book."
```

---

### Task 3: two-block `build()` with an ordered-row spline

**Files:**
- Modify: `src/superglm/features/ordered_categorical.py:366-392` (`build`, `_build_spline`)
- Modify: `src/superglm/types.py:180` (`GroupInfo.basis_rows`)
- Modify: `src/superglm/dm_builder.py:89-136` (both SSP helpers), `:139` (new `_basis_weight_sum`), `:605`, `:645` (call sites)
- Test: `tests/test_ordered_categorical_specials.py`

**Interfaces:**
- Consumes: `self._specials`, `self._smooth_levels`, `self._level_to_value`, `self.has_specials` (Task 2).
- Produces: `build()` returns `[spline_info, special_info]` when `has_specials`, in that order; `special_info.subgroup_name == "special"`, `penalized=False`, `penalty_matrix=None`, `projection=None`, `reparametrize=False`; a declared-but-unobserved special raises `ValueError` at fit time; `GroupInfo.basis_rows: NDArray[np.bool_] | None` set on the spline block, and `compute_R_inv`/`compute_projected_R_inv` accepting `weight_sum=`.

- [ ] **Step 1: Write the failing test for the two-block shape**

```python
def _fit_frame(n=4000, seed=11):
    rng = np.random.default_rng(seed)
    band = rng.choice(ORDERED + [SPECIAL], size=n)
    exposure = rng.gamma(shape=4.0, scale=0.25, size=n)
    t = {lv: i / (len(ORDERED) - 1) for i, lv in enumerate(ORDERED)}
    log_rel = np.array([0.6 * (1 - np.exp(-3 * t[b])) if b != SPECIAL else -0.55 for b in band])
    claims = rng.poisson(exposure * np.exp(np.log(0.08) + log_rel))
    return pd.DataFrame(
        {"band": band, "exposure": exposure, "freq": claims / exposure}
    )


# build() has never returned more than one GroupInfo for an OC term.
def test_build_returns_spline_block_then_special_block():
    frame = _fit_frame()
    spec = _oc()
    infos = spec.build(frame["band"].to_numpy(), frame["exposure"].to_numpy())
    assert isinstance(infos, list) and len(infos) == 2
    spline_info, special_info = infos
    assert spline_info.subgroup_name is None
    assert special_info.subgroup_name == "special"
    assert special_info.n_cols == 1
    assert special_info.penalized is False
    assert special_info.penalty_matrix is None
    assert special_info.projection is None
    assert special_info.reparametrize is False


def test_spline_block_is_zero_on_special_rows():
    frame = _fit_frame()
    spec = _oc()
    spline_info, special_info = spec.build(
        frame["band"].to_numpy(), frame["exposure"].to_numpy()
    )
    is_special = (frame["band"] == SPECIAL).to_numpy()
    spline_cols = np.asarray(spline_info.columns.todense())
    assert np.allclose(spline_cols[is_special], 0.0)
    assert not np.allclose(spline_cols[~is_special], 0.0)
    indicator = np.asarray(special_info.columns.todense()).ravel()
    assert np.array_equal(indicator == 1.0, is_special)


def test_declared_special_absent_from_training_data_is_rejected():
    frame = _fit_frame()
    ordered_only = frame[frame["band"] != SPECIAL]
    spec = _oc()
    with pytest.raises(ValueError, match="never observed"):
        spec.build(ordered_only["band"].to_numpy(), ordered_only["exposure"].to_numpy())
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_ordered_categorical_specials.py -v -k "build or spline_block or absent"`
Expected: FAIL — `test_build_returns_spline_block_then_special_block` fails with `assert isinstance(infos, list)` because `build` returns a single `GroupInfo`; the other two fail on the same unpacking.

- [ ] **Step 3: Build the spline on ordered rows only and add the special block**

Replace `_build_spline` (lines 386-392):

```python
    def _build_spline(
        self, x: NDArray, sample_weight: NDArray | None
    ) -> GroupInfo | list[GroupInfo]:
        """Spline mode: map to numeric, delegate to internal Spline."""
        self._choose_base(x, sample_weight)
        if not self.has_specials:
            return self._spline.build(self._map_to_numeric(x))

        special_mask = self._special_mask(x)
        ordered_mask = ~special_mask.any(axis=1)
        missing = [
            lev for j, lev in enumerate(self._specials) if not special_mask[:, j].any()
        ]
        if missing:
            raise ValueError(
                f"Special level(s) {missing!r} were never observed in the training data. "
                "A special with no rows has an all-zero indicator column and an "
                "unidentifiable coefficient; remove it from specials= or supply data "
                "containing it."
            )

        # The identifiability constraint is a column sum over the rows present, so
        # the spline must be built on exactly the rows its block is nonzero on.
        # Building over all rows would break 1'(B@Z) = 0 once the special rows are
        # zeroed, and would let a fabricated coordinate reach knot placement.
        ordered_numeric = self._map_to_numeric(x[ordered_mask])
        spline_info = self._spline.build(ordered_numeric)
        spline_info = self._expand_rows(spline_info, ordered_mask)

        indicators = sp.csr_matrix(special_mask.astype(np.float64))
        special_info = GroupInfo(
            columns=indicators,
            n_cols=len(self._specials),
            penalty_matrix=None,
            reparametrize=False,
            penalized=False,
            subgroup_name="special",
            projection=None,
        )
        return [spline_info, special_info]

    def _special_mask(self, x: NDArray) -> NDArray[np.bool_]:
        """(n, n_specials) boolean membership matrix, column j == self._specials[j]."""
        return np.column_stack([np.asarray(x == lev) for lev in self._specials])

    @staticmethod
    def _expand_rows(info: GroupInfo, ordered_mask: NDArray[np.bool_]) -> GroupInfo:
        """Re-embed an ordered-row basis into full-length rows, zero elsewhere."""
        import dataclasses

        n = len(ordered_mask)
        compact = info.columns
        expanded = sp.lil_matrix((n, compact.shape[1]), dtype=np.float64)
        expanded[np.flatnonzero(ordered_mask)] = compact
        return dataclasses.replace(info, columns=expanded.tocsr())
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_ordered_categorical_specials.py -v -k "build or spline_block or absent"`
Expected: PASS

- [ ] **Step 5: Write the failing test for full rank of the assembled design**

```python
# The construction is only legitimate if [1 | centered spline | indicators] is
# full rank. A centered basis cannot reproduce a constant, so no indicator is
# recoverable from the other columns — this pins that argument.
def test_assembled_design_with_intercept_is_full_rank():
    frame = _fit_frame()
    spec = _oc()
    spline_info, special_info = spec.build(
        frame["band"].to_numpy(), frame["exposure"].to_numpy()
    )
    design = np.column_stack(
        [
            np.ones(len(frame)),
            np.asarray(spline_info.columns.todense()),
            np.asarray(special_info.columns.todense()),
        ]
    )
    assert np.linalg.matrix_rank(design) == design.shape[1]
```

- [ ] **Step 6: Run it**

Run: `uv run pytest tests/test_ordered_categorical_specials.py::test_assembled_design_with_intercept_is_full_rank -v`
Expected: PASS — no code change needed; this pins the construction's central claim so a later change to the identifiability path cannot silently break it.

- [ ] **Step 7: Write the failing SSP conditioning test for a high-exposure special**

Spec §Risks: `compute_projected_R_inv` (`src/superglm/dm_builder.py:114-136`) normalises the Gram
by `float(np.sum(sample_weight))` over all `n` rows, while a zero-filled spline block is nonzero
on the ordered rows alone. The Gram therefore comes out scaled by
`ordered_exposure / total_exposure`, and the fixed `1e-8·I` ridge grows relative to it exactly
when a special carries material exposure. The SSP contract (`compute_R_inv`'s docstring) is
`X'WX / weight_sum ≈ I` regardless of λ, so a weight-share mis-normalisation is directly
assertable.

Append to `tests/test_ordered_categorical_specials.py`:

```python
def test_ssp_gram_is_normalised_by_the_ordered_row_weight_sum():
    # False today: GroupInfo has no basis_rows field, and the SSP Gram for the
    # spline block is divided by the ALL-ROW weight sum while only the ordered
    # rows contribute. With a special carrying ~90% of the exposure the
    # normalised Gram comes out ~6x the identity instead of ~I, which is the
    # fixed 1e-8 ridge becoming relatively large on the same term.
    from superglm.dm_builder import _process_info

    frame = _fit_frame(n=6000, seed=17)
    band = frame["band"].to_numpy()
    is_special = band == SPECIAL
    weight = np.where(is_special, 50.0, 1.0)  # the special dominates exposure

    spec = _oc()
    spline_info, _ = spec.build(band, weight)
    assert spline_info.basis_rows is not None
    np.testing.assert_array_equal(spline_info.basis_rows, ~is_special)

    gm, _, _ = _process_info(spline_info, sample_weight=weight, lambda2=0.0)
    X = np.asarray(gm.toarray(), dtype=np.float64)
    assert np.allclose(X[is_special], 0.0)

    w_ordered = weight[~is_special]
    X_ordered = X[~is_special]
    gram = X_ordered.T @ (w_ordered[:, None] * X_ordered) / w_ordered.sum()
    # atol is loose against the 1e-8 ridge's imprint (it lifts the identity by
    # 1e-8 / smallest Gram eigenvalue) and tight against the defect, which is a
    # pure scale error of total/ordered ~ 6x on every diagonal entry.
    np.testing.assert_allclose(gram, np.eye(gram.shape[0]), atol=1e-3)
    assert 0.99 < float(np.mean(np.diag(gram))) < 1.01
```

- [ ] **Step 8: Run it to verify it fails**

Run: `uv run pytest tests/test_ordered_categorical_specials.py::test_ssp_gram_is_normalised_by_the_ordered_row_weight_sum -v`
Expected: FAIL — `AttributeError: 'GroupInfo' object has no attribute 'basis_rows'`. Deleting that
assertion locally and re-running shows the substantive failure: `gram` is roughly
`(total_weight / ordered_weight) · I ≈ 6·I` (the special is 1 row in 11 at 50x weight), not the
identity.

- [ ] **Step 9: Normalise the SSP Gram by the rows the basis was built on**

Add the field to `GroupInfo` in `src/superglm/types.py`, immediately after
`repeated_penalty_components` (line 180) and before `__post_init__`:

```python
    # Rows the basis was actually built on.  A block built on a subset of rows
    # and row-expanded with zeros (the OrderedCategorical spline beside free
    # levels) sets this so the SSP normalisation uses the weight sum over those
    # rows rather than over every row.  None means "every row contributes".
    basis_rows: NDArray[np.bool_] | None = None
```

In `_expand_rows` (added in Step 3), record it:

```python
        return dataclasses.replace(
            info, columns=expanded.tocsr(), basis_rows=np.asarray(ordered_mask, dtype=bool)
        )
```

In `src/superglm/dm_builder.py`, give both SSP helpers an optional override. `compute_R_inv`
(lines 89-111):

```python
def compute_R_inv(
    B: sp.spmatrix | NDArray,
    omega: NDArray,
    sample_weight: NDArray,
    lambda2: float | dict,
    weight_sum: float | None = None,
) -> NDArray:
    """Compute SSP reparametrisation matrix R_inv without forming B @ R_inv.

    Wood (2011) Section 3.1 / Section 5: absorb penalty into parameterization.
    R = chol(B'WB/n + λΩ + εI)^T, then R_inv = R^{-1} so that the SSP basis
    X_ssp = B @ R_inv has near-identity X'WX regardless of λ.

    ``weight_sum`` overrides the normalising total. It defaults to the sum over
    every row, which is right whenever every row carries basis mass; a block
    built on a subset of rows and zero-filled elsewhere passes that subset's
    weight sum, so the fixed ``1e-8`` ridge does not grow relative to the Gram.
    """
    lam2 = _resolve_lambda2(lambda2)
    total = float(np.sum(sample_weight)) if weight_sum is None else float(weight_sum)
    if total <= 0.0:
        G = np.zeros((omega.shape[0], omega.shape[0]), dtype=np.float64)
    elif sp.issparse(B):
        G = np.asarray((B.multiply(sample_weight[:, None]).T @ B).todense()) / total
    else:
        G = (B * sample_weight[:, None]).T @ B / total
    M = G + lam2 * omega + np.eye(omega.shape[0]) * 1e-8
    R = np.linalg.cholesky(M).T
    return np.linalg.inv(R)
```

`compute_projected_R_inv` (lines 114-136) takes the same parameter; note its sparse branch
currently recomputes `np.sum(sample_weight)` inline instead of reusing the local, so both
branches must move to `total`:

```python
def compute_projected_R_inv(
    B: sp.spmatrix | NDArray,
    projection: NDArray,
    penalty_sub: NDArray,
    sample_weight: NDArray,
    lambda2: float | dict,
    weight_sum: float | None = None,
) -> NDArray:
    """Compute SSP R_inv within a projected subspace (linear-split range space).

    ``weight_sum`` overrides the normalising total; see ``compute_R_inv``.
    """
    lam2 = _resolve_lambda2(lambda2)
    total = float(np.sum(sample_weight)) if weight_sum is None else float(weight_sum)
    if total <= 0.0:
        G_full = np.zeros((projection.shape[0], projection.shape[0]), dtype=np.float64)
    elif sp.issparse(B):
        G_full = np.asarray((B.multiply(sample_weight[:, None]).T @ B).todense()) / total
    else:
        G_full = (B * sample_weight[:, None]).T @ B / total
    G_sub = projection.T @ G_full @ projection
    n_sub = penalty_sub.shape[0]
    M_sub = G_sub + lam2 * penalty_sub + np.eye(n_sub) * 1e-8
    R_sub = np.linalg.cholesky(M_sub).T
    return np.linalg.inv(R_sub)
```

Add the resolver beside them (after `compute_projected_R_inv`, before `should_discretize` at
line 139):

```python
def _basis_weight_sum(info: GroupInfo, weights: NDArray, use_discrete: bool) -> float | None:
    """Weight total over the rows a block's basis was built on, else None.

    Returns None under discretisation, where ``weights`` is the per-bin
    aggregate rather than one entry per row; no restricted-row block takes
    that path (``should_discretize`` requires a ``_SplineBase`` spec).
    """
    if info.basis_rows is None or use_discrete:
        return None
    return float(np.sum(np.asarray(weights)[np.asarray(info.basis_rows, dtype=bool)]))
```

Then pass it at the two `_process_info` call sites. Line 605:

```python
            R_inv_local = compute_projected_R_inv(
                B_for,
                P,
                info.penalty_matrix,
                exp_for,
                lambda2,
                weight_sum=_basis_weight_sum(info, exp_for, use_discrete),
            )
```

and line 645:

```python
        R_inv = compute_R_inv(
            B_for,
            info.penalty_matrix,
            exp_for,
            lambda2,
            weight_sum=_basis_weight_sum(info, exp_for, use_discrete),
        )
```

Every existing block leaves `basis_rows` at `None`, so `_basis_weight_sum` returns `None` and both
helpers keep their current normalisation byte-for-byte.

- [ ] **Step 10: Run the conditioning test and the design-matrix suites**

Run: `uv run pytest tests/test_ordered_categorical_specials.py -v && uv run pytest tests/test_ssp_audit.py tests/test_spline_weight_geometry.py tests/test_ordered_categorical.py -q`
Expected: PASS — the new test passes and nothing else moves, because no other `GroupInfo` sets
`basis_rows`.

- [ ] **Step 11: Commit**

```bash
git add src/superglm/features/ordered_categorical.py src/superglm/types.py \
        src/superglm/dm_builder.py tests/test_ordered_categorical_specials.py
git commit -m "feat: build the OrderedCategorical smooth on ordered rows beside a free-level block

The spline is built on the ordered rows only and row-expanded with zeros,
because build_identifiability_projection forms its constraint as a column
sum over the rows present. Building over all rows and zeroing afterwards
breaks 1'(B@Z)=0 and lets a fabricated coordinate reach knot placement.

GroupInfo.basis_rows records which rows the basis was built on, so the SSP
normalisation divides by the ordered-row weight sum rather than the total.
Without it a high-exposure special shrinks the block's Gram by its exposure
share and the fixed 1e-8 ridge grows relative to it."
```

---

### Task 4: `transform` widening and the in-term coefficient split

This task also owns the read-back seam in `inference/_term_ops.py`. From Step 7 onward
`reconstruct()` returns specials in `raw["levels"]`, while `_term_ops.py:226` and `:252` build
`level_x` by looking **every** level up in `raw["level_values"]` — which is deliberately
smooth-only. Any `term_inference` call on a specials model therefore raises
`KeyError('MISSING')`, and behind it the curve-SE call at `:228-240` raises `IndexError` inside
`_spline_se`. Both are fixed here, in Steps 9-13, before this task's own invariance test runs.
The plotting task consumes `TermInference.level_is_special` but no longer introduces it.

**Files:**
- Modify: `src/superglm/features/ordered_categorical.py:448-470` (`transform`), `:472-494` (`score`), `:498-503` (`_base_log_effect`), `:512-535` (`_reconstruct_spline`)
- Modify: `src/superglm/inference/_term_types.py:49` (`level_x` comment), `:96-97` (`level_is_special`)
- Modify: `src/superglm/inference/_term_ops.py:202-206`, `:225-226`, `:228-240`, `:251-252`, `:262-279`
- Verified no change needed: `src/superglm/features/ordered_categorical.py:559-563` (`set_reparametrisation`) — `dm_builder.py:1032-1033`/`:1055` collects `r_inv` from reparametrized groups only, and the special block is not one, so the spline's `R_inv` arrives unchanged.
- Create: `tests/test_ordered_categorical_specials_plots.py`
- Test: `tests/test_ordered_categorical_specials.py`

**Interfaces:**
- Consumes: `self._specials`, `self._smooth_levels`, `self.has_specials`, `self._special_mask` (Tasks 2-3).
- Produces: `self._split_beta(beta) -> (spline_beta, special_beta)`; `transform(x)` returns `[spline columns | special indicators]`; `score`, `_base_log_effect` and `reconstruct` handle the full-width vector; `reconstruct()` output gains `special_levels: list[str]` and its `levels`/`level_log_relativities`/`level_relativities` cover specials; `TermInference.level_is_special: NDArray[np.bool_] | None`; `SmoothCurve.level_x` guaranteed smooth-levels-only.

- [ ] **Step 1: Write the failing test for the split and transform width**

```python
# transform() has always returned only the spline's columns, so its width is
# n_spline_cols today and the assertions below are off by len(specials).
def test_transform_emits_spline_then_special_columns():
    frame = _fit_frame()
    spec = _oc()
    spline_info, special_info = spec.build(
        frame["band"].to_numpy(), frame["exposure"].to_numpy()
    )
    probe = np.array(ORDERED + [SPECIAL], dtype=object)
    out = spec.transform(probe)
    assert out.shape == (len(probe), spline_info.n_cols + special_info.n_cols)
    # Special rows are zero across the spline block, ordered rows zero across the indicators.
    assert np.allclose(out[-1, : spline_info.n_cols], 0.0)
    assert np.allclose(out[:-1, spline_info.n_cols :], 0.0)
    assert out[-1, spline_info.n_cols] == 1.0


def test_split_beta_partitions_by_block_width():
    frame = _fit_frame()
    spec = _oc()
    spline_info, special_info = spec.build(
        frame["band"].to_numpy(), frame["exposure"].to_numpy()
    )
    beta = np.arange(spline_info.n_cols + special_info.n_cols, dtype=np.float64)
    spline_beta, special_beta = spec._split_beta(beta)
    assert len(spline_beta) == spline_info.n_cols
    assert len(special_beta) == special_info.n_cols
    assert special_beta[0] == beta[-1]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_ordered_categorical_specials.py -v -k "transform_emits or split_beta"`
Expected: FAIL — `test_transform_emits_spline_then_special_columns` fails on the shape assertion; `test_split_beta_partitions_by_block_width` fails with `AttributeError: 'OrderedCategorical' object has no attribute '_split_beta'`.

- [ ] **Step 3: Add the split and widen `transform`**

Add after `_special_mask`:

```python
    @property
    def _n_special_cols(self) -> int:
        return len(self._specials)

    def _split_beta(
        self, beta: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Split a full-width feature coefficient vector into its two blocks.

        Callers throughout inference concatenate every GroupSlice of a feature
        and hand the result here; the block order is the documented build()
        contract — spline first, specials second.
        """
        beta = np.asarray(beta, dtype=np.float64).ravel()
        if not self.has_specials:
            return beta, np.empty(0, dtype=np.float64)
        n_special = self._n_special_cols
        if len(beta) < n_special:
            raise ValueError(
                f"OrderedCategorical received {len(beta)} coefficients but has "
                f"{n_special} special level(s); the feature's blocks are out of order "
                "or a caller passed only the spline block."
            )
        return beta[: len(beta) - n_special], beta[len(beta) - n_special :]
```

In `transform` (lines 462-464), replace the spline branch:

```python
        if self.basis == "spline":
            if not self.has_specials:
                return self._spline.transform(self._map_to_numeric(x))
            special_mask = self._special_mask(x)
            ordered_mask = ~special_mask.any(axis=1)
            spline_cols = np.zeros((len(x), self._spline_n_cols()), dtype=np.float64)
            if ordered_mask.any():
                spline_cols[ordered_mask] = self._spline.transform(
                    self._map_to_numeric(x[ordered_mask])
                )
            return np.column_stack([spline_cols, special_mask.astype(np.float64)])
```

and add the helper beside `_split_beta`:

```python
    def _spline_n_cols(self) -> int:
        """Fitted width of the spline block, for zero-filling special rows."""
        probe = np.array([self._level_to_value[self._smooth_levels[0]]], dtype=np.float64)
        return int(np.asarray(self._spline.transform(probe)).shape[1])
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_ordered_categorical_specials.py -v -k "transform_emits or split_beta"`
Expected: PASS

- [ ] **Step 5: Write the failing test for scoring and reconstruction**

```python
# score() and reconstruct() forward the whole vector to the inner spline today,
# so with specials present they read special coefficients as spline ones.
def test_score_uses_the_free_coefficient_on_special_rows():
    frame = _fit_frame()
    spec = _oc()
    spline_info, special_info = spec.build(
        frame["band"].to_numpy(), frame["exposure"].to_numpy()
    )
    beta = np.zeros(spline_info.n_cols + special_info.n_cols)
    beta[-1] = -0.55
    scored = spec.score(np.array(["1", SPECIAL], dtype=object), beta)
    assert scored[1] == pytest.approx(-0.55)
    # The special's coefficient must not leak into an ordered row.
    assert scored[0] == pytest.approx(0.0)


def test_reconstruct_reports_every_level_and_flags_the_specials():
    frame = _fit_frame()
    spec = _oc()
    spline_info, special_info = spec.build(
        frame["band"].to_numpy(), frame["exposure"].to_numpy()
    )
    beta = np.zeros(spline_info.n_cols + special_info.n_cols)
    beta[-1] = -0.55
    raw = spec.reconstruct(beta)
    assert raw["levels"] == ORDERED + [SPECIAL]
    assert raw["special_levels"] == [SPECIAL]
    assert set(raw["level_relativities"]) == set(ORDERED + [SPECIAL])
    assert raw["level_log_relativities"][SPECIAL] == pytest.approx(-0.55)
```

- [ ] **Step 6: Run to verify failure**

Run: `uv run pytest tests/test_ordered_categorical_specials.py -v -k "score_uses or reconstruct_reports"`
Expected: FAIL — both raise inside the inner spline, which receives a vector one column wider than its basis.

- [ ] **Step 7: Route every read-back through the split**

In `score` (lines 486-488), replace the spline branch:

```python
        if self.basis == "spline":
            spline_beta, special_beta = self._split_beta(beta)
            if not self.has_specials:
                return self._spline.score(self._map_to_numeric(x), spline_beta)
            special_mask = self._special_mask(x)
            ordered_mask = ~special_mask.any(axis=1)
            out = special_mask.astype(np.float64) @ special_beta
            if ordered_mask.any():
                out[ordered_mask] = self._spline.score(
                    self._map_to_numeric(x[ordered_mask]), spline_beta
                )
            return out
```

In `_base_log_effect` (lines 498-503):

```python
    def _base_log_effect(self, beta: NDArray[np.floating]) -> float:
        """Return the fitted term effect at the reporting reference level."""
        if self.basis != "spline":
            return 0.0
        spline_beta, _ = self._split_beta(beta)
        base_value = np.array([self._level_to_value[self._base_level]], dtype=np.float64)
        return float(self._spline.score(base_value, spline_beta)[0])
```

In `_reconstruct_spline` (lines 512-535), replace the body up to the `raw[...]` assignments:

```python
    def _reconstruct_spline(self, beta: NDArray) -> dict[str, Any]:
        """Spline mode: delegate to internal spline, add per-level annotations.

        Shifts the curve so that the base level has log_relativity=0 (relativity=1),
        giving proper categorical-style relativities. Specials are reported on the
        same scale — beta_special minus the curve at the base — so the rating table
        can reconstruct predictions from one level table.
        """
        spline_beta, special_beta = self._split_beta(beta)
        raw = self._spline.reconstruct(spline_beta)

        # Per-level values on the fitted curve
        level_values = np.array([self._level_to_value[lev] for lev in self._smooth_levels])
        level_log_rels = np.asarray(
            self._spline.score(level_values, spline_beta), dtype=np.float64
        )

        # Shift so base level = 0 (relativity = 1)
        base_shift = self._base_log_effect(beta)
        level_log_rels = level_log_rels - base_shift
        raw["log_relativity"] = raw["log_relativity"] - base_shift
        raw["relativity"] = np.exp(raw["log_relativity"])

        all_levels = list(self._smooth_levels) + list(self._specials)
        all_log_rels = np.concatenate(
            [level_log_rels, np.asarray(special_beta, dtype=np.float64) - base_shift]
        )

        raw["base_level"] = self._base_level
        raw["levels"] = all_levels
        raw["special_levels"] = list(self._specials)
        raw["level_values"] = dict(zip(self._smooth_levels, level_values.tolist()))
        raw["level_log_relativities"] = dict(zip(all_levels, all_log_rels.tolist()))
        raw["level_relativities"] = dict(zip(all_levels, np.exp(all_log_rels).tolist()))
        return raw
```

Note `level_values` stays keyed on the smooth levels only — a special never receives a spline-axis coordinate.

- [ ] **Step 8: Run the tests**

Run: `uv run pytest tests/test_ordered_categorical_specials.py -v`
Expected: PASS (all)

- [ ] **Step 9: Write the failing inference-plumbing test**

Create `tests/test_ordered_categorical_specials_plots.py` — the plotting task appends to this
file, but the two tests that pin the read-back seam belong here, because Step 7 is what breaks it.

```python
"""Plot rendering of OrderedCategorical special (free) levels, both backends."""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from superglm import Numeric, OrderedCategorical, Spline, SuperGLM

PLOTLY_AVAILABLE = importlib.util.find_spec("plotly") is not None

BANDS = [str(i) for i in range(1, 11)]


@pytest.fixture
def specials_model():
    """Ten ordered bands on a saturating curve plus an 18% free MISSING level."""
    rng = np.random.default_rng(20260805)
    n = 4000
    band = rng.choice([*BANDS, "MISSING"], n, p=[0.082] * 10 + [0.18])
    idx = np.array([BANDS.index(b) if b in BANDS else -1 for b in band], dtype=np.float64)
    log_effect = np.where(
        idx >= 0, 0.6 * np.sqrt(np.maximum(idx, 0.0) / 9.0), np.log(0.577)
    )
    x = rng.normal(size=n)
    sample_weight = rng.uniform(0.5, 1.0, n)
    mu = np.exp(-2.0 + log_effect + 0.1 * x)
    y = rng.poisson(mu * sample_weight).astype(float)
    frame = pd.DataFrame({"band": band, "x": x})
    model = SuperGLM(
        features={
            "band": OrderedCategorical(
                order=BANDS, specials=["MISSING"], basis=Spline(kind="ps", k=6)
            ),
            "x": Numeric(),
        }
    )
    model.fit(frame, y, sample_weight=sample_weight)
    return model, frame, sample_weight


def test_term_inference_marks_specials_and_keeps_level_x_smooth_only(specials_model):
    # False today: TermInference has no level_is_special at all, and _term_ops.py:226
    # builds level_x by looking every level up in raw["level_values"], which has no
    # entry for MISSING — so today this raises KeyError before any figure exists.
    model, _, _ = specials_model
    ti = model.term_inference("band")

    assert list(ti.levels) == [*BANDS, "MISSING"]
    assert ti.level_is_special is not None
    np.testing.assert_array_equal(ti.level_is_special, [False] * 10 + [True])
    assert len(ti.relativity) == 11
    assert len(ti.smooth_curve.level_x) == 10
    # with_se defaults to True (api.py:1017) and both plot backends need the band:
    # the curve SE must exist and be finite, not silently vanish or crash.
    assert ti.smooth_curve.se_log_relativity is not None
    assert np.all(np.isfinite(ti.smooth_curve.se_log_relativity))
    assert len(ti.smooth_curve.se_log_relativity) == len(ti.smooth_curve.x)


def test_term_inference_level_is_special_is_none_without_specials():
    # False today: the attribute does not exist, so this is an AttributeError.
    rng = np.random.default_rng(7)
    n = 1500
    band = rng.choice(BANDS, n)
    idx = np.array([BANDS.index(b) for b in band], dtype=np.float64)
    sample_weight = rng.uniform(0.5, 1.0, n)
    y = rng.poisson(np.exp(-2.0 + 0.4 * idx / 9.0) * sample_weight).astype(float)
    frame = pd.DataFrame({"band": band})
    model = SuperGLM(
        features={"band": OrderedCategorical(order=BANDS, basis=Spline(kind="ps", k=5))}
    )
    model.fit(frame, y, sample_weight=sample_weight)

    ti = model.term_inference("band")
    assert ti.level_is_special is None
    assert len(ti.smooth_curve.level_x) == 10
```

- [ ] **Step 10: Run the test to verify it fails**

Run: `uv run pytest tests/test_ordered_categorical_specials_plots.py -v`
Expected: FAIL — `KeyError: 'MISSING'` raised from `superglm/inference/_term_ops.py:226`
(`level_x = np.array([raw["level_values"][lv] for lv in levels])`) for the first test, and
`AttributeError: 'TermInference' object has no attribute 'level_is_special'` for the second.
The `KeyError` hides a second crash one line later: with both blocks' groups passed to
`_spline_se`, `active_cols` spans `p + s` indices against a `p`-column `M` and raises
`IndexError` at `_term_helpers.py:127`. Step 12 fixes both.

- [ ] **Step 11: Add the `level_is_special` field to `TermInference`**

In `src/superglm/inference/_term_types.py:96-97`, replace:

```python
    # Smooth curve for plotting (OrderedCategorical spline mode)
    smooth_curve: SmoothCurve | None = None
```

with:

```python
    # Smooth curve for plotting (OrderedCategorical spline mode)
    smooth_curve: SmoothCurve | None = None

    # Free (unpenalised) levels held out of the smooth: parallel to ``levels``.
    # None when the term has no specials, so existing terms are unchanged.
    level_is_special: NDArray[np.bool_] | None = None
```

Also extend the `SmoothCurve.level_x` comment at `_term_types.py:49`:

```python
    level_x: NDArray | None = None  # numeric x positions of the K *smooth* levels
```

- [ ] **Step 12: Populate the mask, keep `level_x` smooth-only, and scope the curve SE**

In `src/superglm/inference/_term_ops.py:202-206`, replace:

```python
            inner = spec._spline
            raw = spec.reconstruct(beta_combined)
            levels = raw["levels"]
            level_log_rels = np.array([raw["level_log_relativities"][lv] for lv in levels])
            level_rels = np.array([raw["level_relativities"][lv] for lv in levels])
```

with:

```python
            inner = spec._spline
            raw = spec.reconstruct(beta_combined)
            levels = raw["levels"]
            level_log_rels = np.array([raw["level_log_relativities"][lv] for lv in levels])
            level_rels = np.array([raw["level_relativities"][lv] for lv in levels])

            # Specials are free levels with no position on the spline axis: they
            # stay out of level_x and are flagged for the renderers instead.
            special_labels = set(spec._specials) if spec.has_specials else set()
            level_is_special = (
                np.array([lv in special_labels for lv in levels], dtype=bool)
                if special_labels
                else None
            )
            smooth_levels = [lv for lv in levels if lv not in special_labels]
```

Then at `_term_ops.py:225-226`, replace:

```python
                # Continuous curve for plotting
                level_x = np.array([raw["level_values"][lv] for lv in levels])
```

with:

```python
                # Continuous curve for plotting (ordered levels only)
                level_x = np.array([raw["level_values"][lv] for lv in smooth_levels])
```

Next, the curve-SE call at `_term_ops.py:227-240`. `_spline_se` (`_term_helpers.py:88-135`)
computes `active_cols` as `arange(g.start, g.end) - feature_groups[0].start` over the feature's
groups and slices `M = inner.transform(raw["x"])`, which has the spline block's `p` columns only.
Passing the special block leaves `active_cols` running to `p + s` and `M[:, active_cols]` raises
`IndexError`; the covariance block `Cov_g` comes out `(p+s, p+s)` against a `p`-column `M` as
well. Both are fixed by handing `_spline_se` the smooth blocks alone — `covariance.py:524-536`
copies `subgroup_type` onto the re-indexed active groups, so the same filter works on either
list. Replace:

```python
                assert active_groups_cov is not None
                curve_se = _spline_se(
                    inner,
                    name,
                    result.beta,
                    feature_groups,
                    active_groups_cov,
                    Cov_active,
```

with:

```python
                assert active_groups_cov is not None
                # The curve is a statement about the spline block alone; its SE
                # must be too.  feature_se_from_cov's level SEs are unaffected —
                # that path transforms through the OC spec at full p+s width.
                smooth_feature_groups = [
                    fg for fg in feature_groups if fg.subgroup_type != "special"
                ]
                smooth_active_cov = [
                    ag
                    for ag in active_groups_cov
                    if not (ag.feature_name == name and ag.subgroup_type == "special")
                ]
                curve_se = _spline_se(
                    inner,
                    name,
                    result.beta,
                    smooth_feature_groups,
                    smooth_active_cov,
                    Cov_active,
```

(the `x_eval` / `reference_x` keywords that close the call are unchanged.)

At `_term_ops.py:251-252`, replace:

```python
                # No SEs requested but still provide the curve shape
                level_x = np.array([raw["level_values"][lv] for lv in levels])
```

with:

```python
                # No SEs requested but still provide the curve shape
                level_x = np.array([raw["level_values"][lv] for lv in smooth_levels])
```

Finally, in the `TermInference(...)` call at `_term_ops.py:262-279`, add the field after
`smooth_curve=curve,`:

```python
                smooth_curve=curve,
                level_is_special=level_is_special,
                spline=spline_meta,
```

- [ ] **Step 13: Run the inference-plumbing tests**

Run: `uv run pytest tests/test_ordered_categorical_specials_plots.py -v`
Expected: PASS (both tests). A remaining `IndexError` inside `_spline_se` means one of the two
filters in Step 12 was dropped; a `ValueError` about `level_x` width means the smooth-only
`level_x` edits were applied to only one of the two branches.

- [ ] **Step 14: Write the failing end-to-end invariance test**

This is the spike's finding turned into a regression test — the central claim of the feature.

```python
# Nothing enforces this today because the feature does not exist. It is the
# claim the design rests on: a free level effect makes those rows uninformative
# for every other coefficient, so the fitted curve must not move.
def test_adding_a_special_does_not_move_the_fitted_curve():
    frame = _fit_frame(n=8000, seed=3)
    ordered_only = frame[frame["band"] != SPECIAL].reset_index(drop=True)

    without = SuperGLM(
        family="poisson",
        link="log",
        features={"band": OrderedCategorical(order=list(ORDERED), basis=Spline(kind="ps", k=8))},
    )
    without.fit_reml(
        ordered_only[["band"]],
        ordered_only["freq"].to_numpy(),
        sample_weight=ordered_only["exposure"].to_numpy(),
    )

    with_special = SuperGLM(
        family="poisson", link="log", features={"band": _oc()}
    )
    with_special.fit_reml(
        frame[["band"]], frame["freq"].to_numpy(), sample_weight=frame["exposure"].to_numpy()
    )

    a = without.term_inference("band", with_se=False)
    b = with_special.term_inference("band", with_se=False)
    rel_a = dict(zip([str(v) for v in a.levels], np.asarray(a.relativity, dtype=float)))
    rel_b = dict(zip([str(v) for v in b.levels], np.asarray(b.relativity, dtype=float)))
    for lev in ORDERED:
        assert rel_b[lev] == pytest.approx(rel_a[lev], rel=2e-2)
```

- [ ] **Step 15: Run it**

Run: `uv run pytest tests/test_ordered_categorical_specials.py::test_adding_a_special_does_not_move_the_fitted_curve -v`
Expected: PASS. If it fails, the spline is not being built on the ordered rows alone — go back to Task 3 Step 3 rather than loosening the tolerance.

- [ ] **Step 16: Run the full suite**

Run: `uv run pytest tests/ -q -m "not slow"` then `uv run ruff check src/ tests/` and `uv run ruff format --check src/ tests/`
Expected: PASS

- [ ] **Step 17: Commit**

```bash
git add src/superglm/features/ordered_categorical.py \
        src/superglm/inference/_term_types.py src/superglm/inference/_term_ops.py \
        tests/test_ordered_categorical_specials.py \
        tests/test_ordered_categorical_specials_plots.py
git commit -m "feat: split OrderedCategorical coefficients inside the term

Roughly fifteen call sites concatenate a feature's GroupSlices and hand the
full-width vector to reconstruct/score/_base_log_effect, and model/base.py
enforces that width. So the split lives here rather than at the call sites,
and transform widens to match the documented block order.

term_inference lands in the same commit because reconstruct() returning
specials breaks it: level_x is built from the smooth-only level_values map,
and the curve SE re-indexes the feature's groups against the inner spline's
columns. SmoothCurve.level_x now covers the ordered levels only, and
TermInference carries a mask parallel to .levels."
```

---

### Task 5: block-order contract on `build()`

The two positional metadata readers the spec names (§Block order is a contract) are **not**
converted here. `coef_tables.py:412-415` and `report_ops.py:405` are converted exactly once, in
Task 6 Steps 6 and 11, as part of the same restructure that filters the smooth row to the spline
block — converting them twice would leave Task 6's quoted anchors unmatchable. Note the spec's
`report_ops.py:~563` citation is wrong for the tree at `7109e7f`: line ~563 is `knot_summary`, a
`_SplineBase`-only builder. The ordered-categorical positional read is `report_ops.py:405`.

What is left here is the half that has no other home: the contract itself, written where an
implementer will read it.

**Files:**
- Modify: `src/superglm/features/ordered_categorical.py:371` (`build` docstring)

**Interfaces:**
- Consumes: `build()` block order and `subgroup_name == "special"` (Task 3).
- Produces: the documented block-order contract on `build()`. λ and knot metadata selection by `subgroup_type` is Task 6's.

- [ ] **Step 1: Locate the positional reads**

Run: `rg -n "feature_groups\[0\]" src/superglm/`
Expected: two hits — `inference/coef_tables.py:413` and `model/report_ops.py:405`. Read fifteen lines around each to see which metadata is taken from the group (λ, knot count) and that it is reported as *absent* with no error when the group has none. That silence is why the contract is documented and why Task 6 converts both sites rather than one.

- [ ] **Step 2: Document the contract on `build()`**

Add to the `build` docstring in `src/superglm/features/ordered_categorical.py:371`:

```
        With ``specials=``, returns two GroupInfos in a fixed order: the
        penalized spline block first, the unpenalized special-indicator block
        second. Downstream metadata readers select by ``subgroup_type``, but
        the order is part of the contract — ``_split_beta`` and ``transform``
        both assume it.
```

- [ ] **Step 3: Run the suite and commit**

```bash
uv run pytest tests/ -q -m "not slow"
git add src/superglm/features/ordered_categorical.py
git commit -m "docs: document the OrderedCategorical block-order contract on build()

The spline block comes first and the special-indicator block second.
_split_beta and transform both assume it, and the metadata readers that
take feature_groups[0] report absent metadata without erroring, so a
reversed order would degrade silently."
```

---

### Task 6: Restrict the whole-smooth test and edf to the spline block


`coef_tables.py:340` selects an OC feature's groups with `[fg for fg in groups if fg.feature_name == g.feature_name]` and no subgroup filter. Once `build()` emits a second `GroupSlice` (`band:special`) under the same `feature_name`, lines 386-397 concatenate both blocks into `active_indices`, so `X_j`, `V_b_j`, `beta_active` and `edf1_j` all carry the special column: the reported p-value silently becomes a joint test of "curve is flat **and** every special offset is zero". `feature_edf` (343-347), `ref_df` (374) and `n_params` (421) inflate by the free level's ~1.0 edf / 1 column. `report_ops.py:397-419` is the editor's stale-summary twin of the same code and inflates `edf`, `n_params` and `group_norm` identically.

The `_SplineBase` branch at `coef_tables.py:492-533` already shows the shape the codebase uses for this: it selects on `g.subgroup_type` (line 493) and takes its Wood inputs from a single `ag.sl` (523-524). Per the design spec (§Reporting → Summary) the OC term keeps **one** group row restricted to the spline block, and **no** sibling row is emitted for the specials block.

This task is also the **sole** conversion of the two positional `feature_groups[0]` metadata readers the spec names (§Block order is a contract): `coef_tables.py:412-415` in Step 6 and `report_ops.py:405` in Step 11. Task 5 documents the contract and changes no reader, so both anchors below still match the tree as fitted.

**Files:**
- Modify: `src/superglm/inference/coef_tables.py:340-347`
- Modify: `src/superglm/inference/coef_tables.py:364`
- Modify: `src/superglm/inference/coef_tables.py:374`
- Modify: `src/superglm/inference/coef_tables.py:412-415`
- Modify: `src/superglm/inference/coef_tables.py:421`
- Modify: `src/superglm/model/report_ops.py:397-419`
- Test: `tests/test_ordered_categorical_inference.py` (append after line 303)

**Interfaces:**
- Consumes: `OrderedCategorical(order=..., specials=["MISSING"], base="first", basis=Spline(kind="ps", k=7))`; the two-block `build()` producing `GroupSlice(name="band", subgroup_type=None)` then `GroupSlice(name="band:special", subgroup_type="special")`, both with `feature_name="band"`; `spec.reconstruct(full_width_beta)`; `wood_test_smooth(beta_j, X_j, V_b_j, edf1_j, res_df)` (`stats/wood_pvalue.py:186`); `_build_editor_stale_coef_rows(model)` (`report_ops.py:363`); `model._group_edf` (`model/api.py:755`).
- Produces: the invariant that the single `subgroup_type="ordered_spline"` row's `wald_chi2`/`wald_p`/`ref_df`/`edf`/`n_params`/`group_norm` are spline-block-only; λ/knot metadata read from `smooth_groups[0].name` rather than `feature_groups[0].name`; test helpers `_special_band_data()` and `_fit_special_band_model()`.

---

- [ ] **Step 1: Add the specials fixture helpers to the inference test module**

Append to `tests/test_ordered_categorical_inference.py` (after line 303). This mirrors `_ordered_trend_data` (lines 12-21) and `_fit_ordered_and_direct_spline` (lines 24-45), adding a `MISSING` population whose rate is nothing like the ordered trend. The special carries only ~6.5% of the weight, which keeps the with/without comparison in Step 3 tight while still costing a full degree of freedom.

```python
def _special_band_data():
    """Ordered bands on a smooth trend plus a structurally different MISSING band."""
    rng = np.random.default_rng(20260805)
    levels = [f"L{i}" for i in range(7)]
    codes = np.tile(np.arange(len(levels)), 180)
    rng.shuffle(codes)
    x_ordered = np.asarray(levels, dtype=object)[codes]
    x_numeric = codes / (len(levels) - 1)
    w_ordered = rng.uniform(0.6, 1.8, len(codes))
    eta_ordered = -0.8 + 0.9 * x_numeric + 0.15 * np.sin(2.0 * np.pi * x_numeric)

    n_special = 90
    w_special = rng.uniform(0.6, 1.8, n_special)
    eta_special = np.full(n_special, 0.9)

    band = np.concatenate([x_ordered, np.full(n_special, "MISSING", dtype=object)])
    weights = np.concatenate([w_ordered, w_special])
    eta = np.concatenate([eta_ordered, eta_special])
    y = rng.poisson(np.exp(eta) * weights).astype(float)
    ordered_mask = np.asarray(band != "MISSING", dtype=bool)
    return pd.DataFrame({"band": band}), y, weights, ordered_mask, levels


def _fit_special_band_model():
    """Fit the specials term. Poisson keeps the scale known, so the Wood test
    uses res_df = -1 and no dispersion estimate enters the comparison."""
    frame, y, weights, _, levels = _special_band_data()
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        tol=1e-10,
        max_iter=200,
        features={
            "band": OrderedCategorical(
                order=levels,
                specials=["MISSING"],
                base="first",
                basis=Spline(kind="ps", k=7),
            )
        },
    )
    model.fit(frame, y, sample_weight=weights)
    return model, levels
```

- [ ] **Step 2: Write the failing spy test on the Wood test inputs**

Append after the helpers. This is the exact, tolerance-free statement of the defect: it inspects the arguments `build_coef_rows` hands to `wood_test_smooth`. The monkeypatch target follows the existing precedent at lines 74-85 — `coef_tables.py:384` imports `wood_test_smooth` inside the loop, so patching the module attribute takes effect. `model.summary()` is called only after patching because `report_ops` caches the summary object (`report_ops.py:354`).

```python
def test_whole_smooth_test_sees_only_the_spline_block(monkeypatch):
    model, _ = _fit_special_band_model()
    spline_group = next(g for g in model._groups if g.name == "band")
    special_group = next(g for g in model._groups if g.name == "band:special")
    assert special_group.size == 1
    assert special_group.subgroup_type == "special"

    from superglm.stats import wood_pvalue

    real_wood_test = wood_pvalue.wood_test_smooth
    calls = []

    def recording_wood_test(beta_j, X_j, V_b_j, edf1_j, res_df=-1.0):
        calls.append((np.shape(beta_j), np.shape(X_j), np.shape(V_b_j)))
        return real_wood_test(beta_j, X_j, V_b_j, edf1_j, res_df)

    monkeypatch.setattr(wood_pvalue, "wood_test_smooth", recording_wood_test)
    summary = model.summary()

    # FAILS TODAY: coef_tables.py:340 selects every GroupSlice whose
    # feature_name is "band", so active_indices (386-388) spans the spline
    # block AND the special column.  beta_j / X_j / V_b_j therefore come in
    # one column too wide and the p-value tests "curve is flat AND the
    # MISSING offset is zero".
    assert len(calls) == 1
    beta_shape, x_shape, v_shape = calls[0]
    assert beta_shape == (spline_group.size,)
    assert x_shape[1] == spline_group.size
    assert v_shape == (spline_group.size, spline_group.size)

    # The specials block gets no sibling smooth row: one group row per term.
    smooth_rows = [row for row in summary._coef_rows if row.is_spline and row.group == "band"]
    assert [row.name for row in smooth_rows] == ["band"]
    # FAILS TODAY: n_params is len(beta_combined) at coef_tables.py:421.
    assert smooth_rows[0].n_params == spline_group.size
```

- [ ] **Step 3: Write the failing invariance test against a specials-free refit**

A free, unpenalized level makes the special rows uninformative for every other coefficient: the score equation for the special coefficient forces the special rows' weighted residuals to zero, so the intercept and spline solve exactly the estimating equations of the ordered rows alone, and the Schur complement of the special block reproduces the ordered-only centered Gram. With Poisson (`scale_known=True`, `distributions.py:62`) and `fit()`'s fixed `spline_penalty`, the spline block's edf, `V_b_j` and `X_j'X_j` are mathematically identical to the ordered-only refit. Tolerances are wide relative to that agreement and narrow relative to the defect (edf inflates by ~1.0 out of ~4; the joint chi2 differs by orders of magnitude because the MISSING offset is large).

```python
def _fit_ordered_only_reference():
    """The same model with the special rows removed instead of held out."""
    frame, y, weights, ordered_mask, levels = _special_band_data()
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        tol=1e-10,
        max_iter=200,
        features={
            "band": OrderedCategorical(
                order=levels,
                base="first",
                basis=Spline(kind="ps", k=7),
            )
        },
    )
    model.fit(
        frame.loc[ordered_mask].reset_index(drop=True),
        y[ordered_mask],
        sample_weight=weights[ordered_mask],
    )
    return model


def test_special_level_does_not_change_the_reported_smooth_statistics():
    with_special, _ = _fit_special_band_model()
    reference = _fit_ordered_only_reference()

    row = _smooth_row(with_special.summary())
    ref_row = _smooth_row(reference.summary())

    assert row.subgroup_type == "ordered_spline"
    assert row.active

    # FAILS TODAY, all four:
    #   n_params  -> spline size + 1   (coef_tables.py:421)
    #   edf       -> spline edf + ~1.0 (coef_tables.py:343-347)
    #   wald_chi2 -> joint test including the large MISSING offset
    #   ref_df    -> driven by edf1 summed over the special column too
    assert row.n_params == ref_row.n_params
    assert row.edf == pytest.approx(ref_row.edf, rel=1e-3)
    assert row.wald_chi2 == pytest.approx(ref_row.wald_chi2, rel=1e-2)
    assert row.ref_df == pytest.approx(ref_row.ref_df, rel=1e-2)
    assert row.wald_p == pytest.approx(ref_row.wald_p, rel=1e-2, abs=1e-6)

    # Lambda / knot metadata must come from the spline block, not from
    # whichever block happens to sit first (coef_tables.py:412-415).
    assert row.spline_kind == ref_row.spline_kind
    assert row.smoothing_lambda == pytest.approx(ref_row.smoothing_lambda, rel=1e-12)
```

- [ ] **Step 4: Run both tests to verify they fail**

Run: `uv run pytest tests/test_ordered_categorical_inference.py -k "spline_block or reported_smooth_statistics" -v`

Expected: FAIL. `test_whole_smooth_test_sees_only_the_spline_block` fails at `assert beta_shape == (spline_group.size,)` with `AssertionError: assert (7,) == (6,)`-shaped output (the recorded width is one larger than the spline group). `test_special_level_does_not_change_the_reported_smooth_statistics` fails at `assert row.n_params == ref_row.n_params` with the with-specials value one larger.

- [ ] **Step 5: Restrict the OC branch head to the spline block**

In `src/superglm/inference/coef_tables.py`, replace lines 340-347:

```python
            feature_groups = [fg for fg in groups if fg.feature_name == g.feature_name]
            beta_combined = np.concatenate([beta[fg.sl] for fg in feature_groups])
            feature_active = any(fg.name in selected_names for fg in feature_groups)
            feature_edf = (
                sum(_get_group_edf_map().get(fg.name, 0.0) for fg in feature_groups)
                if feature_active
                else 0.0
            )
```

with:

```python
            feature_groups = [fg for fg in groups if fg.feature_name == g.feature_name]
            # A specials term owns a second, unpenalized GroupSlice under the
            # same feature_name.  ``reconstruct`` needs the full-width vector,
            # but every statistic reported on the smooth row — edf, the Wood
            # test, ref_df, n_params — is a statement about the spline block.
            smooth_groups = [fg for fg in feature_groups if fg.subgroup_type != "special"]
            beta_combined = np.concatenate([beta[fg.sl] for fg in feature_groups])
            feature_active = any(fg.name in selected_names for fg in feature_groups)
            feature_edf = (
                sum(_get_group_edf_map().get(fg.name, 0.0) for fg in smooth_groups)
                if feature_active
                else 0.0
            )
```

- [ ] **Step 6: Point the Wood test, ref_df, metadata and n_params at the spline block**

Four single-line edits in the `if spec.basis == "spline":` branch of the same function.

Line 364 — `active_pairs` drives `active_indices`, `X_j`, `V_b_j`, `edf1_j` and `beta_active`:

```python
                for feature_group in smooth_groups:
```

Line 374 — the fallback reference df used when `wood_test_smooth` raises:

```python
                ref_df = float(sum(fg.size for fg in smooth_groups))
```

Lines 412-415 — select the λ / knot / boundary source by subgroup, not by position (spec §Block order is a contract):

```python
                _, s_lam, s_kind, s_knot_strat, s_bnd = _spline_enrichment(
                    smooth_groups[0].name,
                    spec._spline,
                )
```

Line 421 — the reported parameter count:

```python
                        n_params=sum(fg.size for fg in smooth_groups),
```

- [ ] **Step 7: Run the tests**

Run: `uv run pytest tests/test_ordered_categorical_inference.py tests/test_ordered_categorical.py -v`

Expected: PASS. In particular `test_ordered_spline_uses_same_global_wood_test_as_direct_spline` (line 57) still passes at `rel=1e-10` — with no specials, `smooth_groups == feature_groups` and nothing changes.

- [ ] **Step 8: Commit**

```bash
git add src/superglm/inference/coef_tables.py tests/test_ordered_categorical_inference.py
git commit -m "fix(summary): restrict the ordered-spline whole-smooth test to the spline block

A specials term owns a second GroupSlice under the same feature_name.
Selecting a feature's groups without a subgroup filter turned the
reported whole-smooth p-value into a joint test of the curve and the
free level offsets, and inflated edf, ref_df and n_params by the
specials block."
```

- [ ] **Step 9: Write the failing test for the editor stale-summary twin**

`report_ops._build_editor_stale_coef_rows` (line 363) is a second, p-value-free copy of the same OC branch and inflates `edf` (399-401), `n_params` (417) and `group_norm` (419) the same way. It is callable directly on a fitted model — it reads only `model.result`, `model._groups`, `model._specs`, `model._interaction_specs`, `model._feature_order` and `model._group_edf`. Append to `tests/test_ordered_categorical_inference.py`:

```python
def test_editor_stale_rows_report_the_spline_block_only():
    from superglm.model.report_ops import _build_editor_stale_coef_rows

    model, _ = _fit_special_band_model()
    spline_group = next(g for g in model._groups if g.name == "band")
    # The free level really does cost a whole degree of freedom.
    assert model._group_edf["band:special"] == pytest.approx(1.0, abs=0.05)

    rows = _build_editor_stale_coef_rows(model)
    smooth_row = next(row for row in rows if row.is_spline and row.name == "band")

    # FAILS TODAY: report_ops.py:399-401 sums group_edf over both blocks,
    # :417 counts len(beta_combined), :419 norms the full-width vector —
    # so the stale summary shows the free level's edf and its coefficient
    # inside the smooth row.
    assert smooth_row.n_params == spline_group.size
    assert smooth_row.edf == pytest.approx(model._group_edf["band"], rel=1e-12)
    assert smooth_row.group_norm == pytest.approx(
        float(np.linalg.norm(model.result.beta[spline_group.sl])), rel=1e-12
    )
```

- [ ] **Step 10: Run it to verify it fails**

Run: `uv run pytest tests/test_ordered_categorical_inference.py -k editor_stale -v`

Expected: FAIL at `assert smooth_row.n_params == spline_group.size`, the reported value being one larger than the spline group size.

- [ ] **Step 11: Apply the same restriction in report_ops**

In `src/superglm/model/report_ops.py`, replace lines 397-401:

```python
            feature_groups = [fg for fg in model._groups if fg.feature_name == g.feature_name]
            beta_combined = np.concatenate([model.result.beta[fg.sl] for fg in feature_groups])
            feature_edf = (
                sum(group_edf.get(fg.name, 0.0) for fg in feature_groups) if group_edf else None
            )
```

with:

```python
            feature_groups = [fg for fg in model._groups if fg.feature_name == g.feature_name]
            # Mirrors coef_tables: the smooth row describes the spline block,
            # while reconstruct() still needs the full-width coefficients.
            smooth_groups = [fg for fg in feature_groups if fg.subgroup_type != "special"]
            beta_combined = np.concatenate([model.result.beta[fg.sl] for fg in feature_groups])
            beta_smooth = np.concatenate([model.result.beta[fg.sl] for fg in smooth_groups])
            feature_edf = (
                sum(group_edf.get(fg.name, 0.0) for fg in smooth_groups) if group_edf else None
            )
```

Then replace line 405 `feature_groups[0].name,` with:

```python
                    smooth_groups[0].name,
```

and lines 417-419:

```python
                        n_params=len(beta_smooth),
                        active=any(fg.name in selected_names for fg in feature_groups),
                        group_norm=float(np.linalg.norm(beta_smooth)),
```

- [ ] **Step 12: Run the tests**

Run: `uv run pytest tests/test_ordered_categorical_inference.py tests/test_editor.py -q`

Expected: PASS. `tests/test_editor.py:4335-4338` reads the stale `age_band` smooth row and is unaffected — that term has no specials, so `smooth_groups == feature_groups`.

- [ ] **Step 13: Commit**

```bash
git add src/superglm/model/report_ops.py tests/test_ordered_categorical_inference.py
git commit -m "fix(editor): report ordered-spline stale rows over the spline block only

_build_editor_stale_coef_rows is a second copy of the ordered-spline
summary branch; it inflated edf, n_params and group_norm with the
specials block in exactly the same way."
```

- [ ] **Step 14: Full verification**

Run: `uv run pytest tests/ -q && uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/`

Expected: PASS on all three.

---

### Task 7: Per-level fit marker through summary, HTML and the Excel Summary sheet


Nothing in `coef_tables.py`, `summary.py` or `summary_levels.py` consumes `TermInference`. Summary level rows are rebuilt independently from `spec.reconstruct()` + `feature_se_from_cov()` (`coef_tables.py:441-467`) and re-laid-out by `build_summary_level_display` (`summary_levels.py:48-235`) keyed on `spec._ordered_levels`. The marker is therefore threaded through the coefficient-row layer as a `_CoefRow` field, **not** through `TermInference`.

The level label column stays the raw lookup key — no asterisk, no decoration. The `fit` column is present only on terms that have specials, so no existing OC output changes width.

**Excel scope note:** the Summary sheet gets provenance at **both** levels of granularity. The term's group row is marked through the existing `SummaryTermRow.kind` machinery (`"smooth+free"`), and each special's *level* row is marked `"free level"` instead of `"level"`.

Per-level provenance costs nothing here, contrary to the design doc's original justification: `export/summary.py:301` iterates `summary._coef_rows`, so every OC level row is *already* emitted as its own `SummaryTermRow` (`_canonical_level_row_names`, `export/summary.py:228-242`, reads `spec._ordered_levels` and therefore picks specials up unchanged), and `tests/test_rating_table_export.py:1325-1348` already pins one Summary row per level. No new column, no new sheet, no rating-sheet change — only the `kind` string chosen at `export/summary.py:306-310`. The spec's §Export section is corrected to match.

The Excel **rating** sheet still must not change — that decision stands: `excel.py:176` hard-codes `start_col = 1 + idx * 3` with number formats keyed on `cell.column % 3` (`excel.py:186,188`), and `tests/test_rating_table_export.py:1309-1319` pins block 2 to columns 4-6. A fourth column would overwrite the next block's name column and desync formatting for every later block.

**Files:**
- Create: `tests/test_ordered_categorical_specials_summary.py`
- Modify: `src/superglm/inference/summary.py:56-63` (add `level_fit` to `_CoefRow`)
- Modify: `src/superglm/inference/coef_tables.py:441-467` (set `level_fit` on ordered-spline level rows)
- Modify: `src/superglm/model/report_ops.py:423-430` (same, editor-stale row builder)
- Modify: `src/superglm/inference/summary.py:285-286,341-357,411-415,455-459,552,557` (ASCII `Fit` column)
- Modify: `src/superglm/inference/summary.py:737-756,840-849,881-888,943-973,1018-1084` (HTML `Fit` column)
- Modify: `src/superglm/export/summary.py:266-274` (Summary-sheet term marker), `:290-310` (per-level `kind`), `:359-364` (Wood note guard)
- Modify: `docs/guide/features.md:137-166` (OrderedCategorical section)
- Test: `tests/test_ordered_categorical_specials_summary.py`
- Test: `tests/test_rating_table_export.py`

**Interfaces:**
- Consumes: `OrderedCategorical(order=..., specials=[...], base=..., basis=Spline(...))`; `spec._specials: list[str]`; `spec._ordered_levels: list[str]`; `spec.has_specials -> bool`; `spec.reconstruct(beta)` with `raw["levels"] == spec._ordered_levels`; `feature_se_from_cov(...)` returning `len(spec._ordered_levels)` entries (`_term_covariance.py:96-105` passes `x_eval=np.array(spec._ordered_levels, dtype=object)`); `TermInference` for a specials term already being buildable (Task 4), which `export_rating_tables` needs at `export/rating_tables.py:152-171`.
- Produces: `_CoefRow.level_fit: str | None = None` with values `"smooth"` / `"free"` / `None`; ASCII and HTML `Fit` columns keyed off it; `_group_test_kind` returning `"smooth+free"` for a spline OC with specials; `export/summary._level_row_kind` returning `"free level"` for a special's level row.

---

- [ ] **Step 1: Write the failing test for the `_CoefRow` field on fitted level rows**

Create `tests/test_ordered_categorical_specials_summary.py`:

```python
"""Summary reporting of free (special) levels on ordered categorical terms."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from superglm import OrderedCategorical, Spline, SuperGLM
from superglm.inference.summary import ModelSummary, _CoefRow
from superglm.types import GroupSlice

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
        features={"band": OrderedCategorical(order=ORDERED, base="first", basis=Spline(kind="ps", k=6))},
    )
    model.fit(X, y, sample_weight=weights)

    assert [row.level_fit for row in _level_rows(model.summary())] == [None] * len(ORDERED)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_ordered_categorical_specials_summary.py -v`
Expected: FAIL — both tests error with `AttributeError: '_CoefRow' object has no attribute 'level_fit'`.

- [ ] **Step 3: Add the `level_fit` field to `_CoefRow`**

In `src/superglm/inference/summary.py`, the block currently reads (lines 56-63):

```python
    # Quasi-separation warning
    quasi_separated: bool = False
    level_n_obs: int | None = None
    level_exposure_share: float | None = None
    # Summary presentation only. Canonical coefficient builders leave these
    # fields at their defaults.
    level_group: str = ""
    is_reference: bool = False
```

Replace with:

```python
    # Quasi-separation warning
    quasi_separated: bool = False
    level_n_obs: int | None = None
    level_exposure_share: float | None = None
    # Per-level fit provenance: "smooth" for a level carried by the spline,
    # "free" for an OrderedCategorical special, None when the term has no
    # specials. Drives the optional `fit` column in both renderers.
    level_fit: str | None = None
    # Summary presentation only. Canonical coefficient builders leave these
    # fields at their defaults.
    level_group: str = ""
    is_reference: bool = False
```

- [ ] **Step 4: Set the marker in `build_coef_rows`**

In `src/superglm/inference/coef_tables.py`, the ordered-spline level loop at lines 441-467 currently starts `levels = raw["levels"]` and appends a `_CoefRow` with `name/group/coef/se/ci_low/ci_high`. Replace lines 441-467 with:

```python
                levels = raw["levels"]
                special_labels = set(spec._specials) if spec.has_specials else None
                for i, level in enumerate(levels):
                    coef_val = float(raw["level_log_relativities"][level])
                    se_val: float | None = (
                        float(se_levels[i]) if feature_active and i < len(se_levels) else None
                    )
                    level_ci_lo: float | None
                    level_ci_hi: float | None
                    if se_val is not None and np.isfinite(se_val) and se_val > 0.0:
                        _, _, level_ci_lo, level_ci_hi = _compute_coef_stats(
                            coef_val, se_val, alpha
                        )
                    elif se_val is not None and np.isfinite(se_val) and level == spec._base_level:
                        level_ci_lo = level_ci_hi = coef_val
                    else:
                        se_val = None
                        level_ci_lo = level_ci_hi = None
                    rows.append(
                        _CoefRow(
                            name=f"{g.feature_name}[{level}]",
                            group=feature_label,
                            coef=coef_val,
                            se=se_val,
                            ci_low=level_ci_lo,
                            ci_high=level_ci_hi,
                            level_fit=(
                                None
                                if special_labels is None
                                else ("free" if level in special_labels else "smooth")
                            ),
                        )
                    )
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_ordered_categorical_specials_summary.py -v`
Expected: PASS (both tests).

- [ ] **Step 6: Write the failing test for the editor-stale row builder and the `summary_levels` pass-through**

`report_ops._build_editor_stale_coef_rows` (`report_ops.py:363-436`) is a *second*, independent builder of OC level rows, used whenever `editor_inference_stale` is set (`report_ops.py:243-262`). `build_summary_level_display` rebuilds every level row via `dataclasses.replace` (`summary_levels.py:155,170`), which is where an unthreaded field is silently dropped.

Append to `tests/test_ordered_categorical_specials_summary.py`:

```python
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
```

- [ ] **Step 7: Run the tests to verify the editor one fails**

Run: `uv run pytest tests/test_ordered_categorical_specials_summary.py -v`
Expected: `test_editor_stale_row_builder_also_marks_free_levels` FAILS with `AssertionError: assert [None, None, ...] == ['smooth', ..., 'free']`. `test_level_display_relayout_preserves_the_fit_marker` should already PASS (`dataclasses.replace` copies the field) — if it fails, the re-layout is dropping the field and that must be fixed before continuing.

- [ ] **Step 8: Set the marker in `_build_editor_stale_coef_rows`**

In `src/superglm/model/report_ops.py`, lines 423-430 currently read:

```python
                for level in raw["levels"]:
                    rows.append(
                        _CoefRow(
                            name=f"{g.feature_name}[{level}]",
                            group=g.feature_name,
                            coef=float(raw["level_log_relativities"][level]),
                        )
                    )
```

Replace with:

```python
                special_labels = set(spec._specials) if spec.has_specials else None
                for level in raw["levels"]:
                    rows.append(
                        _CoefRow(
                            name=f"{g.feature_name}[{level}]",
                            group=g.feature_name,
                            coef=float(raw["level_log_relativities"][level]),
                            level_fit=(
                                None
                                if special_labels is None
                                else ("free" if level in special_labels else "smooth")
                            ),
                        )
                    )
```

- [ ] **Step 9: Run the tests**

Run: `uv run pytest tests/test_ordered_categorical_specials_summary.py tests/test_summary_level_display.py -v`
Expected: PASS.

- [ ] **Step 10: Commit the row field**

```bash
git add src/superglm/inference/summary.py src/superglm/inference/coef_tables.py \
        src/superglm/model/report_ops.py tests/test_ordered_categorical_specials_summary.py
git commit -m "feat(summary): record smooth/free provenance on ordered-categorical level rows"
```

- [ ] **Step 11: Write the failing ASCII renderer test**

The ASCII box pads every bordered line to the same width `W` via `_row` (`summary.py:405-406`), and `:<{W}s` never truncates. So if the `Fit` column is rendered but not included in `coef_W` (`summary.py:357`), the level lines come out *longer* than the header lines. Asserting one distinct line length is therefore a real check on the width arithmetic, not a smoke test.

Append to `tests/test_ordered_categorical_specials_summary.py`:

```python
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
    assert "smooth" not in plain and "free" not in plain


def test_ascii_fit_column_is_included_in_the_box_width():
    # False today: there is no Fit column, so nothing pins that adding it
    # widens coef_W. Without the width fix the marked level lines overflow
    # W and the box loses its single line length.
    text = str(_summary(with_specials=True))
    boxed = [line for line in text.splitlines() if line.startswith(("║", "╠", "╟"))]

    assert len({len(line) for line in boxed}) == 1
```

- [ ] **Step 12: Run the test to verify it fails**

Run: `uv run pytest tests/test_ordered_categorical_specials_summary.py -k ascii -v`
Expected: FAIL — `test_ascii_summary_renders_a_fit_column_only_when_levels_are_marked` fails at `assert "Fit" in header` (the header is `Term ... coef std err z P>|z| ...`).

- [ ] **Step 13: Implement the ASCII `Fit` column**

All four edits are in `ModelSummary.__str__` in `src/superglm/inference/summary.py`.

(a) At line 286, `has_level_groups = bool(self._level_groups)` — add the sibling flag right after:

```python
        has_level_groups = bool(self._level_groups)
        has_level_fit = any(row.level_fit is not None for row in display_rows)
```

(b) Lines 352-357 currently read:

```python
        level_group_w = (
            max(len("Level group"), *(len(row.level_group) for row in display_rows))
            if has_level_groups
            else 0
        )
        coef_W = name_w + level_group_w + sum(coef_field_widths) + len(coef_field_widths)
```

Replace with:

```python
        level_group_w = (
            max(len("Level group"), *(len(row.level_group) for row in display_rows))
            if has_level_groups
            else 0
        )
        level_fit_w = (
            max(len("Fit"), *(len(row.level_fit or "") for row in display_rows)) + 2
            if has_level_fit
            else 0
        )
        coef_W = (
            name_w
            + level_group_w
            + level_fit_w
            + sum(coef_field_widths)
            + len(coef_field_widths)
        )
```

(c) `_coef_prefix` at lines 411-415 currently reads:

```python
        def _coef_prefix(row: _CoefRow, *, name: str | None = None) -> str:
            prefix = f"{row.name if name is None else name:<{name_w}s}"
            if has_level_groups:
                prefix += f"{row.level_group if name is None else '':>{level_group_w}s}"
            return prefix
```

Replace with:

```python
        def _coef_prefix(row: _CoefRow, *, name: str | None = None) -> str:
            prefix = f"{row.name if name is None else name:<{name_w}s}"
            if has_level_groups:
                prefix += f"{row.level_group if name is None else '':>{level_group_w}s}"
            if has_level_fit:
                fit = (row.level_fit or "") if name is None else ""
                prefix += f"{fit:>{level_fit_w}s}"
            return prefix
```

(d) The table header at lines 455-459 currently reads:

```python
        hdr_prefix = (
            f"{'Term':<{name_w}s}{'Level group':>{level_group_w}s}"
            if has_level_groups
            else f"{'Term':<{name_w}s}"
        )
```

Replace with:

```python
        hdr_prefix = f"{'Term':<{name_w}s}"
        if has_level_groups:
            hdr_prefix += f"{'Level group':>{level_group_w}s}"
        if has_level_fit:
            hdr_prefix += f"{'Fit':>{level_fit_w}s}"
```

(e) The two spline detail-line indents at lines 552 and 557 both read
`lines.append(_row(f"{'':<{name_w + level_group_w}s}    {detail_str}"))`. Replace **both** with:

```python
                        lines.append(
                            _row(f"{'':<{name_w + level_group_w + level_fit_w}s}    {detail_str}")
                        )
```

- [ ] **Step 14: Run the ASCII tests**

Run: `uv run pytest tests/test_ordered_categorical_specials_summary.py -k ascii -v`
Expected: PASS.

- [ ] **Step 15: Write the failing HTML renderer test**

The HTML table is a fixed `ncols` grid: every `<tr>` must sum to `ncols` counting `colspan`. Summing effective width per row catches a missed `_level_fit_cell` call *and* a missed `colspan` adjustment, both of which render as a visually broken table with no exception.

Append to `tests/test_ordered_categorical_specials_summary.py`:

```python
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
```

- [ ] **Step 16: Run the test to verify it fails**

Run: `uv run pytest tests/test_ordered_categorical_specials_summary.py -k html -v`
Expected: FAIL — `assert ">Fit</td>" in html` fails, and `set(_row_widths(marked)) == {10}` fails with `{9} == {10}`.

- [ ] **Step 17: Implement the HTML `Fit` column**

All edits are in `ModelSummary._repr_html_` in `src/superglm/inference/summary.py`.

(a) Lines 738-739 currently read:

```python
        has_level_groups = bool(self._level_groups)
        ncols = 10 if has_level_groups else 9
```

Replace with:

```python
        has_level_groups = bool(self._level_groups)
        has_level_fit = any(row.level_fit is not None for row in display_rows)
        extra_cols = int(has_level_groups) + int(has_level_fit)
        ncols = 9 + extra_cols
```

(b) After `_level_group_cell` (lines 750-753) add its sibling:

```python
        def _level_fit_cell(row: _CoefRow) -> str:
            if not has_level_fit:
                return ""
            return f'<td style="{cell_l}">{html_escape(row.level_fit or "")}</td>'
```

(c) The header column block at lines 825-849 currently reads:

```python
        col_names = [""]
        if has_level_groups:
            col_names.append("Level group")
        col_names.extend(
            [
                "coef",
                ...
            ]
        )
        parts.append("<tr>")
        parts.append(f'<td style="{hdr_cell_l}">{col_names[0]}</td>')
        first_numeric = 1
        if has_level_groups:
            parts.append(f'<td style="{hdr_cell_l}">{col_names[1]}</td>')
            first_numeric = 2
        for cn in col_names[first_numeric:-1]:
```

Replace the `col_names` prelude and the emission block with:

```python
        col_names = [""]
        if has_level_groups:
            col_names.append("Level group")
        if has_level_fit:
            col_names.append("Fit")
        col_names.extend(
            [
                "coef",
                "std err",
                "z",
                "P>|z|",
                f"[{half:.3f}",
                f"{1 - half:.3f}]",
                "Sig",
                "QS",
            ]
        )
        parts.append("<tr>")
        parts.append(f'<td style="{hdr_cell_l}">{col_names[0]}</td>')
        first_numeric = 1 + extra_cols
        for cn in col_names[1:first_numeric]:
            parts.append(f'<td style="{hdr_cell_l}">{cn}</td>')
        for cn in col_names[first_numeric:-1]:
            parts.append(f'<td style="{hdr_cell}">{cn}</td>')
        parts.append(f'<td style="{hdr_cell_l}">{col_names[-1]}</td>')
        parts.append("</tr>")
```

(d) Insert `f"{_level_fit_cell(row)}"` immediately after **every** `f"{_level_group_cell(row)}"` — there are seven, at lines 884, 946, 959, 969, 1022, 1058 and 1074. Each becomes, for example (structured-kind row, line 881-888):

```python
                parts.append(
                    f"<tr>"
                    f'<td style="{cell_l}">{html_escape(row.name)}</td>'
                    f"{_level_group_cell(row)}"
                    f"{_level_fit_cell(row)}"
                    f'<td colspan="{ncols - 1 - extra_cols}" '
                    f'style="{cell_l};color:#666;font-style:italic;">{text}</td>'
                    f"</tr>"
                )
```

(e) Replace `int(has_level_groups)` with `extra_cols` in the four colspan expressions, at lines 885 (`ncols - 1 - int(has_level_groups)`), 947 (`ncols - 3 - int(has_level_groups)`), 960 and 970 (`ncols - 1 - int(has_level_groups)`). No other `ncols` arithmetic changes: the header value cell at line 817 (`ncols - 4`) and every full-width `colspan="{ncols}"` follow `ncols` automatically.

- [ ] **Step 18: Run the HTML tests and the full renderer suites**

Run: `uv run pytest tests/test_ordered_categorical_specials_summary.py tests/test_summary_level_display.py tests/test_design_summary.py tests/test_ordered_categorical_inference.py -q`
Expected: PASS.

- [ ] **Step 19: Check formatting and lint**

Run: `uv run ruff format --check src/ tests/ && uv run ruff check src/ tests/`
Expected: PASS (run `uv run ruff format src/ tests/` first if the check reports reformatting).

- [ ] **Step 20: Commit the renderers**

```bash
git add src/superglm/inference/summary.py tests/test_ordered_categorical_specials_summary.py
git commit -m "feat(summary): render a fit column marking free levels in ASCII and HTML"
```

- [ ] **Step 21: Write the failing Excel Summary-sheet test**

Add to `tests/test_rating_table_export.py`, after `test_ordered_spline_workbook_keeps_only_global_inference` (which ends at line ~1349). It reuses the module's existing `_table_records`, `EXPECTED_SUMMARY_TERM_HEADERS` and `_write_workbook` helpers.

```python
def _fit_ordered_specials_export_model():
    rng = np.random.default_rng(20260805)
    ordered = [f"L{i}" for i in range(7)]
    codes = np.repeat(np.arange(len(ordered)), 90)
    band_ordered = np.asarray(ordered, dtype=object)[codes]
    x = codes / (len(ordered) - 1.0)
    eta_ordered = -0.8 + 0.9 * x
    band_missing = np.full(240, "MISSING", dtype=object)
    eta_missing = np.full(240, -0.8 - 0.5)
    band = np.concatenate([band_ordered, band_missing])
    eta = np.concatenate([eta_ordered, eta_missing])
    region = np.resize(np.array(["N", "S", "E"], dtype=object), band.size)
    weights = rng.uniform(0.6, 1.8, band.size)
    y = rng.poisson(np.exp(eta) * weights).astype(float)
    X = pd.DataFrame({"band": band, "region": region})
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={
            "band": OrderedCategorical(
                order=ordered,
                specials=["MISSING"],
                base="first",
                basis=Spline(kind="ps", k=5),
            ),
            "region": Categorical(base="N"),
        },
    )
    model.fit(X, y, sample_weight=weights)
    return model, X, y, weights, ordered


def test_summary_sheet_marks_a_term_that_contains_free_levels():
    # False today: a specials term's group row is Kind="smooth", identical to
    # a term with no specials, so the workbook records nothing about free
    # levels. Also pins that the Wood note survives the new kind value.
    model, _, _, _, ordered = _fit_ordered_specials_export_model()
    payload = build_summary_export_payload(model)
    band_rows = [row for row in payload.terms if row.group == "band"]

    marked = [row for row in band_rows if row.kind == "smooth+free"]
    assert len(marked) == 1
    assert marked[0].term == "band"
    assert isinstance(marked[0].p_value, float)
    assert not [row for row in band_rows if row.kind == "smooth"]
    assert "Smooth p-values use Wood (2013) Bayesian tests." in payload.notes


def test_summary_sheet_marks_the_free_level_row_itself():
    # False today: every level row is Kind="level", so the workbook cannot say
    # WHICH level was fitted free. export/summary.py:301 already emits one row
    # per level (test_rating_table_export.py:1325-1348 pins that), so this is
    # the kind string alone — no new column and no rating-sheet change.
    model, _, _, _, ordered = _fit_ordered_specials_export_model()
    payload = build_summary_export_payload(model)
    level_rows = [
        row for row in payload.terms if row.group == "band" and row.kind in {"level", "free level"}
    ]

    assert [row.term for row in level_rows] == [
        f"band[{level}]" for level in [*ordered, "MISSING"]
    ]
    assert [row.kind for row in level_rows] == ["level"] * len(ordered) + ["free level"]
    free_row = level_rows[-1]
    assert free_row.estimate is not None
    assert free_row.std_error is not None
    assert free_row.p_value is None


def test_summary_sheet_level_kinds_are_unchanged_without_specials():
    # Guards the width/format contract the other direction: a term with no
    # specials must keep every level row at Kind="level".
    model, levels = _fit_ordered_export_model()
    payload = build_summary_export_payload(model)
    level_rows = [
        row for row in payload.terms if row.group == "band" and row.kind in {"level", "free level"}
    ]

    assert [row.kind for row in level_rows] == ["level"] * len(levels)


def test_specials_workbook_keeps_summary_columns_and_rating_block_layout(tmp_path):
    # False today: nothing exercises a specials model through the workbook, so
    # neither the fixed Summary header set nor the 3-column rating blocks are
    # pinned against a marker column creeping onto the rating sheet.
    model, X, y, weights, _ = _fit_ordered_specials_export_model()
    output = tmp_path / "specials.xlsx"

    model.export_rating_tables(output, X, y, sample_weight=weights, n_bins=20)

    wb = load_workbook(output, data_only=True)
    summary_ws = wb["Model Summary"]
    term_min_col, term_min_row, term_max_col, _ = range_boundaries(
        summary_ws.tables["TermInference"].ref
    )
    term_headers = [
        summary_ws.cell(row=term_min_row, column=column).value
        for column in range(term_min_col, term_max_col + 1)
    ]
    assert term_headers == EXPECTED_SUMMARY_TERM_HEADERS
    kinds = {row["Kind"] for row in _table_records(summary_ws, "TermInference")}
    assert "smooth+free" in kinds
    assert "free level" in kinds

    rating_ws = wb["Rating Tables"]
    assert rating_ws["A5"].value == "band"
    assert [rating_ws.cell(row=7, column=column).value for column in range(1, 4)] == [
        "band",
        "Relativity",
        "Weight",
    ]
    # Block 2 must still start at column 4: excel.py:176 keys start_col and
    # excel.py:186/188 key number formats on a 3-column stride.
    assert rating_ws["D5"].value == "region"
    assert [rating_ws.cell(row=7, column=column).value for column in range(4, 7)] == [
        "region",
        "Relativity",
        "Weight",
    ]
```

- [ ] **Step 22: Run the tests to verify they fail**

Run: `uv run pytest tests/test_rating_table_export.py -k "specials or level_kinds" -v`
Expected: FAIL — `test_summary_sheet_marks_a_term_that_contains_free_levels` fails at `assert len(marked) == 1` with `0 == 1` (the row is still `kind="smooth"`), and `test_summary_sheet_marks_the_free_level_row_itself` fails at the kind list with `['level', ..., 'level'] != ['level', ..., 'free level']`. `test_summary_sheet_level_kinds_are_unchanged_without_specials` passes already — it is the guard on the other direction.

- [ ] **Step 23: Emit the term and level markers through the existing `kind` machinery**

In `src/superglm/export/summary.py`, `_group_test_kind` at lines 266-274 currently reads:

```python
def _group_test_kind(model: SuperGLM, row, groups: tuple[Any, ...]) -> str:
    spec = _source_spec(model, groups)
    if isinstance(spec, OrderedCategorical) and spec.basis == "spline":
        return "smooth"
```

Replace the OC branch with:

```python
def _group_test_kind(model: SuperGLM, row, groups: tuple[Any, ...]) -> str:
    spec = _source_spec(model, groups)
    if isinstance(spec, OrderedCategorical) and spec.basis == "spline":
        # Per-term marker: the term's whole-smooth row records that the term
        # contains free levels.  Which ones is recorded on the level rows.
        return "smooth+free" if spec.has_specials else "smooth"
```

Add the per-level classifier immediately after `_group_test_kind`:

```python
def _level_row_kind(row) -> str:
    """Per-level provenance for a level row.

    The Summary sheet emits one row per coefficient row, so an
    OrderedCategorical special already has its own row here; it only needs a
    kind that says so.  ``level_fit`` is set by both level-row builders
    (``coef_tables.build_coef_rows`` and
    ``report_ops._build_editor_stale_coef_rows``), so the edited-model path
    carries the marker too.
    """
    return "free level" if getattr(row, "level_fit", None) == "free" else "level"
```

and use it in `_term_rows` (lines 290-310), replacing the `kind = (...)` expression:

```python
        kind = (
            _group_test_kind(model, row, source_groups)
            if is_group_row
            else (_level_row_kind(row) if row.name in level_names else "coefficient")
        )
```

Then `_summary_notes` at lines 359-364 currently reads:

```python
    if not inference_stale:
        if any(row.kind == "smooth" for row in terms):
            notes.append(_SMOOTH_WOOD_NOTE)
```

The exact-equality test would silently drop the Wood note for a specials-only model. Replace with:

```python
    if not inference_stale:
        if any(row.kind.startswith("smooth") for row in terms):
            notes.append(_SMOOTH_WOOD_NOTE)
```

- [ ] **Step 24: Run the export tests**

Run: `uv run pytest tests/test_rating_table_export.py -q`
Expected: PASS — including `test_ordered_spline_workbook_keeps_only_global_inference` and the two header assertions at lines 1239 and 1305, which are untouched because no column was added.

- [ ] **Step 25: Run the full suite, lint and format**

Run: `uv run pytest tests/ -q && uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/`
Expected: PASS.

- [ ] **Step 26: Commit the Excel marker**

```bash
git add src/superglm/export/summary.py tests/test_rating_table_export.py
git commit -m "feat(export): mark ordered-categorical free levels on the summary sheet

The term's whole-smooth row becomes kind='smooth+free' and each special's
level row becomes kind='free level'. The Summary sheet already emits one
row per level, so per-level provenance needs no new column and the rating
sheet's 3-column block stride is untouched."
```

- [ ] **Step 27: Document `specials=` and the intercept shift in the feature guide**

The reported intercept **moves** when a special is added to or removed from an existing model,
because the identifiability constraint is a column sum over the rows present
(`_spline_identifiability.py:23-29`) and the spline is now built on the ordered rows alone. Spec
§Risks requires this to be "documented rather than hidden", and it is editor-visible: an analyst
who adds `specials=["MISSING"]` to a fitted model sees a different `Intercept` row in the summary
even though nothing about the ordered levels changed. Nothing else in this plan says so anywhere
a user will look.

In `docs/guide/features.md`, after the paragraph ending "...not the whole-smooth p-value." (line
161) and before "This interpretation depends on the numeric positions..." (line 163), insert:

````markdown
### Free levels (`specials=`)

Some levels do not belong on the ordering at all — a `MISSING` band, a
structural zero. Listing them in `specials=` holds them out of the smooth and
fits each one as a free, unpenalized level effect:

```python
OrderedCategorical(
    order=["1", "2", "3", "4", "5", "6"],
    specials=["MISSING"],
    basis=Spline(kind="ps", k=6),
)
```

The smooth then spans the ordered levels only, and the special reports its own
base-relative relativity beside them. Use this for levels that are
*structurally* different, never for merely sparse ones: the penalty already
handles a sparse band better than a free level does.

The summary marks each level in a `Fit` column reading `smooth` or `free`, and
the exported workbook records the term as `smooth+free` with the special's own
row as a `free level`. Plots draw the fitted curve across the ordered levels
and place free levels as detached points past its end.

**The reported intercept changes when you add or remove a special.** The
smooth's identifiability constraint is taken over the rows it is built on, so
with `specials=` the intercept is the baseline of the *ordered* rows alone;
without it, the special's rows are inside that baseline. Level relativities are
reported against the base level and are unaffected, but do not compare
intercepts across two models that differ in `specials=`.

A special must be present in the training data (an all-zero indicator column
has no identifiable coefficient), may not be the reporting `base=`, and may not
be merged into a level group. `specials=` requires `basis=Spline(...)`;
interactions and PSST screening on a term with specials are not supported yet
and are reported as deferred rather than silently skipped.
````

- [ ] **Step 28: Check the docs build inputs and commit**

Run: `rg -n "specials" docs/guide/features.md` and confirm the fenced Python block inside the new
subsection renders (the surrounding section already uses the same ```` ```python ```` fences).

```bash
git add docs/guide/features.md
git commit -m "docs: document OrderedCategorical specials= and the intercept shift

Restricting the identifiability constraint to the ordered rows moves the
reported intercept when a special is added or removed. The spec requires
that to be documented rather than hidden."
```

---

### Task 8: Render special levels in both plotting backends

The inference half of this work landed in Task 4: `TermInference.level_is_special` exists,
`SmoothCurve.level_x` is already smooth-levels-only, and `term_inference` on a specials model
already returns without raising. What is left is display — turning that mask plus the smooth-only
`level_x` into one x-position per displayed level, and drawing free levels as detached points in
both backends rather than letting plotly truncate them away at `min(len(x), len(y))`.

Grouped expansion (`_term_helpers._expand_grouped_term`) is **not** touched here: nothing in this
task's tests exercises it, and a term that has both a grouping and specials cannot exist until
Task 10's collapse work. Task 10 Step 14 rewrites that function once, against the unmodified file.

**Files:**
- Modify: `src/superglm/plotting/common.py:24-26`, `src/superglm/plotting/common.py:66-67`
- Modify: `src/superglm/plotting/data.py:11-14`, `src/superglm/plotting/data.py:125-131`
- Modify: `src/superglm/plotting/group_display.py:54-71`, `src/superglm/plotting/group_display.py:155-184`
- Modify: `src/superglm/plotting/main_effects.py:16-35`, `src/superglm/plotting/main_effects.py:524-634`
- Modify: `src/superglm/plotting/main_effects_plotly.py:14-34`, `:57-69`, `:739-755`, `:1102-1158`, `:1356-1387`
- Test: `tests/test_ordered_categorical_specials_plots.py` (created in Task 4, appended to here)

**Interfaces:**
- Consumes: `OrderedCategorical._specials: list[str]`, `OrderedCategorical.has_specials -> bool`, `spec._ordered_levels == _smooth_levels + _specials`; `TermInference.level_is_special: NDArray[np.bool_] | None` and the smooth-only `SmoothCurve.level_x`, both landed in Task 4; the prerequisite commit's `_plot_ordered_spline_panel`, which draws `ti.smooth_curve` instead of a PCHIP through level relativities, and its `_collapsed_smooth_curve(ti, groups)` signature.
- Produces: `superglm.plotting.common._ordered_level_step`, `._special_level_positions`, `._level_positions_with_specials`, `._SPECIAL_COLOR`, `._PLOTLY_SPECIAL_COLOR`; `superglm.plotting.group_display._collapse_special_mask` and a `_collapsed_smooth_curve` that drops all-special display groups; plotly style key `"special_color"`; the matplotlib container label and plotly trace name `"Free levels"`; `plot_data` payload column `x_position` covering all K+S rows.

---

- [ ] **Step 1: Write the failing test for the shared position helpers**

Append to `tests/test_ordered_categorical_specials_plots.py`:

```python
def test_special_level_positions_leave_a_gap_after_the_last_ordered_level():
    # False today: superglm.plotting.common has no position helper at all.
    from superglm.plotting.common import _level_positions_with_specials

    level_x = np.arange(10, dtype=np.float64)
    mask = np.array([False] * 10 + [True])
    pos = _level_positions_with_specials(level_x, mask, 11)

    np.testing.assert_allclose(pos[:10], level_x)
    assert pos[10] == 11.0  # one empty slot past the last ordered level at x=9


def test_special_level_positions_scale_with_level_spacing():
    # False today: no helper; the plotly panel would place a special at an
    # arange() index while the ordered levels sit on midpoint values.
    from superglm.plotting.common import _level_positions_with_specials

    pos = _level_positions_with_specials(
        np.array([20.0, 30.0, 40.0]), np.array([False, False, False, True]), 4
    )
    np.testing.assert_allclose(pos, [20.0, 30.0, 40.0, 60.0])


def test_level_positions_with_specials_rejects_a_full_width_level_x():
    # False today: no helper. The point of the raise is that a level_x left at
    # K+S is the silent-drop bug; it must fail loudly instead.
    from superglm.plotting.common import _level_positions_with_specials

    with pytest.raises(ValueError, match="ordered levels only"):
        _level_positions_with_specials(
            np.arange(11, dtype=np.float64), np.array([False] * 10 + [True]), 11
        )


def test_level_positions_without_specials_is_the_identity():
    # False today: no helper. Guards the no-specials path both panels take.
    from superglm.plotting.common import _level_positions_with_specials

    level_x = np.array([21.5, 30.5, 40.5])
    np.testing.assert_allclose(_level_positions_with_specials(level_x, None, 3), level_x)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_ordered_categorical_specials_plots.py -k position -v`
Expected: FAIL — `ImportError: cannot import name '_level_positions_with_specials' from 'superglm.plotting.common'`.

- [ ] **Step 3: Add the position helpers to `plotting/common.py`**

In `src/superglm/plotting/common.py`, add the two colour constants — after `_CAT_BAR_COLOR = "#006FDD"` at line 25:

```python
_SPECIAL_COLOR = "#7A3EA1"
```

and after `_PLOTLY_CAT_BAR_COLOR = "#E10600"` at line 34:

```python
_PLOTLY_SPECIAL_COLOR = "#1E63D7"
```

Then insert the helpers before `_exposure_kde` at line 68:

```python
def _ordered_level_step(x: NDArray) -> float:
    """Median positive spacing of ordered level positions (1.0 when unknown)."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.size < 2:
        return 1.0
    diffs = np.diff(np.sort(arr))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 1.0
    return float(np.median(diffs))


def _special_level_positions(level_x: NDArray, n_specials: int, *, gap_steps: float = 2.0) -> NDArray:
    """X positions for free (special) levels, set off after the ordered levels.

    Each special takes one level-step, starting ``gap_steps`` steps past the
    last ordered level so the detached points read as a separate block.  With
    the default ``level_x = 0..K-1`` spacing this puts them on the integer
    positions ``K+1, K+2, ...``, one empty slot clear of the curve.
    """
    if n_specials <= 0:
        return np.empty(0, dtype=np.float64)
    arr = np.asarray(level_x, dtype=np.float64)
    step = _ordered_level_step(arr)
    start = (float(arr.max()) if arr.size else 0.0) + gap_steps * step
    return start + step * np.arange(n_specials, dtype=np.float64)


def _level_positions_with_specials(
    level_x: NDArray, level_is_special: NDArray | None, n_levels: int
) -> NDArray:
    """Positions for every displayed level, ordered levels first.

    ``level_x`` is ``SmoothCurve.level_x`` — the ordered levels only — and
    ``level_is_special`` is the mask parallel to ``TermInference.levels``.
    The result is ``n_levels`` long so callers can zip it against ``levels``
    without plotly truncating to ``min(len(x), len(y))`` and dropping the
    specials from markers, bars and tick labels.
    """
    arr = np.asarray(level_x, dtype=np.float64)
    mask = (
        np.zeros(n_levels, dtype=bool)
        if level_is_special is None
        else np.asarray(level_is_special, dtype=bool)
    )
    if mask.size != n_levels:
        raise ValueError(f"level_is_special has {mask.size} entries for {n_levels} levels.")
    n_ordered = int((~mask).sum())
    if arr.size != n_ordered:
        raise ValueError(
            f"level_x carries {arr.size} positions for {n_ordered} ordered levels; "
            "it must cover the ordered levels only."
        )
    positions = np.empty(n_levels, dtype=np.float64)
    positions[~mask] = arr
    positions[mask] = _special_level_positions(arr, int(mask.sum()))
    return positions
```

- [ ] **Step 4: Run the helper tests**

Run: `uv run pytest tests/test_ordered_categorical_specials_plots.py -k position -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Write the failing tests for the plot-data payload and the display mask**

Append to `tests/test_ordered_categorical_specials_plots.py`:

```python
def test_plot_data_keeps_x_position_for_the_special_level(specials_model):
    # False today: data.py:126-130 only attaches x_position when
    # len(effect) == len(level_x); with one special those differ by one, so the
    # column is dropped for the WHOLE term, not just for MISSING.
    model, _, _ = specials_model
    payload = model.plot_data("band")
    effect = payload["terms"][0]["effect"]

    assert list(effect["level"]) == [*BANDS, "MISSING"]
    assert "x_position" in effect.columns
    pos = effect["x_position"].to_numpy(dtype=np.float64)
    ordered_step = pos[9] - pos[8]
    assert pos[10] - pos[9] > 1.5 * ordered_step


def test_collapse_special_mask_marks_only_all_special_groups():
    # False today: group_display has no mask collapse, so replace(ti, ...) at
    # group_display.py:61-71 would carry a K+S mask onto a shorter display term.
    from superglm.plotting.group_display import _collapse_special_mask

    mask = np.array([False, False, False, True])
    np.testing.assert_array_equal(
        _collapse_special_mask(mask, [[0, 1], [2], [3]]), [False, False, True]
    )
    assert _collapse_special_mask(None, [[0], [1]]) is None
```

- [ ] **Step 6: Run the tests to verify they fail**

Run: `uv run pytest tests/test_ordered_categorical_specials_plots.py -k "plot_data or collapse_special" -v`
Expected: FAIL — `KeyError: 'x_position'` from `effect["x_position"]` (the column is absent) and `ImportError: cannot import name '_collapse_special_mask' from 'superglm.plotting.group_display'`.

- [ ] **Step 7: Fix the plot-data payload and the display projection**

In `src/superglm/plotting/data.py:13`, extend the import:

```python
from superglm.plotting.common import (
    _exposure_kde,
    _kde_2d,
    _level_positions_with_specials,
)
```

Replace `data.py:125-131`:

```python
    effect = ti.to_dataframe()
    if ti.smooth_curve is not None:
        level_x = ti.smooth_curve.level_x
        if level_x is not None and len(effect) == len(level_x):
            effect = effect.copy()
            effect["x_position"] = np.asarray(level_x, dtype=np.float64)
```

with:

```python
    effect = ti.to_dataframe()
    if ti.smooth_curve is not None:
        level_x = ti.smooth_curve.level_x
        if level_x is not None:
            n_special = (
                int(np.asarray(ti.level_is_special, dtype=bool).sum())
                if ti.level_is_special is not None
                else 0
            )
            if len(effect) == len(level_x) + n_special:
                effect = effect.copy()
                effect["x_position"] = _level_positions_with_specials(
                    level_x, ti.level_is_special, len(effect)
                )
```

In `src/superglm/plotting/group_display.py`, add after `_collapse_array` (`group_display.py:155-159`):

```python
def _collapse_special_mask(mask: NDArray | None, groups: list[list[int]]) -> NDArray | None:
    """Collapse the free-level mask onto display groups.

    A group is free only when every member is; a grouping may not mix a special
    with ordered levels, so this is an all-or-nothing test.
    """
    if mask is None:
        return None
    arr = np.asarray(mask, dtype=bool)
    return np.asarray([bool(arr[indices].all()) for indices in groups], dtype=bool)
```

and thread it into the display term at `group_display.py:61-71`, adding one keyword to the `replace(...)` call after `spline=None,`. The `smooth_curve=` line is quoted as the **prerequisite commit** left it (Task 1 Step 6 changed the signature to `(ti, group_indices)`); add only the `level_is_special=` keyword:

```python
        spline=None,
        level_is_special=_collapse_special_mask(ti.level_is_special, group_indices),
        smooth_curve=_collapsed_smooth_curve(ti, group_indices),
```

Then `_collapsed_smooth_curve` itself (`group_display.py:162-184` after Task 1) must stop indexing
`level_x` with display-level indices. Post-Task-4 `level_x` covers the **smooth** levels only,
while `groups` indexes all K+S display levels, so a term with both a grouping and specials — the
state Task 10's collapse produces, and `"collapsed"` is the OC auto default
(`group_display.py:92`) — raises `IndexError` on the last group. Replace the body:

```python
def _collapsed_smooth_curve(
    ti: TermInference,
    groups: list[list[int]],
) -> SmoothCurve | None:
    """Keep the fitted curve and move each marker to its group's mean position.

    The curve itself is never rebuilt: collapsing levels is a display
    operation, and re-interpolating through the collapsed markers would
    draw a shape the model never fitted.

    ``level_x`` covers the smoothed levels only, so an all-special group has no
    position on the curve's axis and is dropped here rather than indexed into
    ``level_x``.  The renderers place those markers from ``level_is_special``,
    which ``_collapse_special_mask`` keeps parallel to the display levels.
    """
    curve = ti.smooth_curve
    if curve is None or curve.level_x is None:
        return curve
    level_x = np.asarray(curve.level_x, dtype=np.float64)
    n_levels = len(ti.levels or [])
    special = (
        np.asarray(ti.level_is_special, dtype=bool)
        if ti.level_is_special is not None
        else np.zeros(n_levels, dtype=bool)
    )
    # Position of each smooth display level within level_x.
    smooth_pos = np.cumsum(~special) - 1
    collapsed: list[float] = []
    for indices in groups:
        idx = np.asarray(indices, dtype=np.intp)
        smooth_idx = idx[~special[idx]]
        if smooth_idx.size == 0:
            continue  # an all-free group has no place on the fitted curve
        collapsed.append(float(np.mean(level_x[smooth_pos[smooth_idx]])))
    return replace(curve, level_x=np.asarray(collapsed, dtype=np.float64))
```

With no specials `special` is all-False, `smooth_pos == arange(n_levels)`, no group is dropped and
the result is the previous mean-position array unchanged.

- [ ] **Step 8: Run the tests**

Run: `uv run pytest tests/test_ordered_categorical_specials_plots.py -v`
Expected: PASS (8 tests — the two inference-plumbing tests from Task 4 plus the six added here).

- [ ] **Step 9: Commit the payload and display plumbing**

```bash
git add src/superglm/plotting/common.py src/superglm/plotting/data.py \
        src/superglm/plotting/group_display.py \
        tests/test_ordered_categorical_specials_plots.py
git commit -m "feat(plotting): position free levels off the end of the fitted curve

_level_positions_with_specials turns the smooth-only level_x plus the
level_is_special mask into one position per displayed level, so plot_data
keeps x_position for the whole term instead of dropping the column, and the
collapsed display projection stops indexing level_x with display indices."
```

- [ ] **Step 10: Write the failing matplotlib test**

Append to `tests/test_ordered_categorical_specials_plots.py`:

```python
def test_matplotlib_panel_detaches_the_special_level(specials_model):
    # False today: the panel puts every level on one K-long position grid, so
    # either errorbar() raises on the 10-vs-11 mismatch or MISSING is drawn
    # adjacent to band 10 with no gap. There is no "Free levels" container, no
    # tick past the last ordered level, and the curve is drawn through all 11.
    import matplotlib

    matplotlib.use("Agg")
    model, frame, sample_weight = specials_model
    fig = model.plot("band", engine="matplotlib", X=frame, sample_weight=sample_weight)
    ax = fig.axes[0]

    ticks = np.asarray(ax.get_xticks(), dtype=np.float64)
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == [*BANDS, "MISSING"]
    assert len(ticks) == 11
    assert ticks[10] - ticks[9] > 1.5 * (ticks[9] - ticks[8])

    containers = {c.get_label(): c for c in ax.containers}
    assert "Free levels" in containers
    free_x = np.asarray(containers["Free levels"][0].get_xdata(), dtype=np.float64)
    np.testing.assert_allclose(free_x, [ticks[10]])
    ordered_x = np.asarray(containers["Relativity"][0].get_xdata(), dtype=np.float64)
    assert len(ordered_x) == 10

    curve_line = max(ax.lines, key=lambda line: len(line.get_xdata()))
    assert len(curve_line.get_xdata()) == 200
    assert float(np.max(curve_line.get_xdata())) < float(free_x[0])

    bar_axes = [a for a in fig.axes if a.patches]
    assert bar_axes, "no exposure bars were drawn"
    centers = sorted(p.get_x() + p.get_width() / 2 for p in bar_axes[0].patches)
    assert len(centers) == 11
    assert centers[-1] == pytest.approx(float(ticks[10]))
```

- [ ] **Step 11: Run the test to verify it fails**

Run: `uv run pytest tests/test_ordered_categorical_specials_plots.py -k matplotlib -v`
Expected: FAIL — `ValueError: 'x' and 'y' must have the same first dimension, but have shapes (10,) and (11,)` raised from the `ax.errorbar(...)` call in `_plot_ordered_spline_panel`; if the prerequisite left the panel on `np.arange(n_levels)` positions instead, `AssertionError: assert 'Free levels' in {'Relativity': ...}`.

- [ ] **Step 12: Rewrite `_plot_ordered_spline_panel`**

Extend the common import block in `src/superglm/plotting/main_effects.py:16-35` with `_SPECIAL_COLOR`, `_level_positions_with_specials` and `_ordered_level_step` (alphabetical: `_SPECIAL_COLOR` after `_SIM_FILL`; `_level_positions_with_specials` after `_exposure_kde`; `_ordered_level_step` after `_make_continuous_figure`).

**Also delete `_ordered_level_spacing` from that same import block.** Task 1 added it for the one
call in `_plot_ordered_spline_panel`, and the replacement body below uses `_ordered_level_step`
instead; leaving the import in place fails `ruff` F401 at Step 20's gate. `_ordered_level_spacing`
stays in `plotting/common.py` — `main_effects_plotly._ordered_bar_width` still delegates to it
(Task 1 Step 8), so it is not dead code, only unused *here*.

Two near-duplicate spacing helpers therefore ship: `_ordered_level_spacing` (minimum positive gap,
sizing bars) and `_ordered_level_step` (median positive gap, laying out the special block). That
is deliberate — the median is what places specials evenly when the ordered levels are unequally
spaced, and the minimum is what keeps bars from overlapping — but it is a real duplication and is
called out here rather than discovered later.

Replace the body of `_plot_ordered_spline_panel` (`main_effects.py:524-634`) with:

```python
def _plot_ordered_spline_panel(
    ax,
    ti: TermInference,
    interval: str | None,
    *,
    X: EagerFrame | None = None,
    sample_weight: NDArray | None = None,
    weight_label: str = "Weight",
    display: GroupedTermDisplay | None = None,
):
    """Render an OrderedCategorical(spline) panel.

    Ordered levels sit at their spline x-positions under the fitted curve.
    Free (special) levels are detached points past the end of the curve,
    separated by a visible gap, with their own ticks and exposure bars.
    """
    levels = list(ti.levels)
    level_rel = np.asarray(ti.relativity)
    n_levels = len(levels)
    curve = ti.smooth_curve
    level_x = (
        np.asarray(curve.level_x, dtype=np.float64)
        if curve is not None and curve.level_x is not None
        else np.arange(n_levels, dtype=float)
    )
    is_special = (
        np.asarray(ti.level_is_special, dtype=bool)
        if ti.level_is_special is not None
        else np.zeros(n_levels, dtype=bool)
    )
    x_pos = _level_positions_with_specials(level_x, ti.level_is_special, n_levels)
    step = _ordered_level_step(x_pos)

    # Exposure bars in background
    if sample_weight is not None and X is not None and ti.name in X.columns:
        exp_vals = grouped_level_exposure(display, X, sample_weight)
        if exp_vals is None:
            level_exp = (
                pd.DataFrame(
                    {
                        "level": X.column_array(ti.name),
                        "sample_weight": sample_weight,
                    }
                )
                .groupby("level", sort=False)["sample_weight"]
                .sum()
            )
            exp_vals = np.array([level_exp.get(lv, 0.0) for lv in levels])
        ax2 = ax.twinx()
        ax2.bar(
            x_pos,
            exp_vals,
            width=0.6 * step,
            color=_EXP_FILL,
            edgecolor=_EXP_EDGE,
            linewidth=_EXP_EDGE_LW,
            alpha=1.0,
            zorder=0,
            label=weight_label,
        )
        ymax = float(exp_vals.max()) if exp_vals.size else 0.0
        ax2.set_ylim(0.0, ymax * 1.12 if ymax > 0 else 1.0)
        ax2.set_ylabel(weight_label, color=_EXP_EDGE)
        ax2.tick_params(axis="y", colors=_EXP_EDGE, labelsize=9)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_color(_EXP_EDGE)
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)

    ax.axhline(1.0, linestyle="--", linewidth=_REF_LW, color=_REF_COLOR, zorder=0)

    # Fitted curve — spans the ordered levels only
    if curve is not None:
        ax.plot(
            np.asarray(curve.x, dtype=np.float64),
            np.asarray(curve.relativity, dtype=np.float64),
            color=_LINE_COLOR,
            linewidth=_LINE_WIDTH,
            alpha=0.6,
            zorder=4,
        )

    # Per-level dots with error bars — ordered and free levels drawn separately
    marker_specs = (
        ("Relativity", ~is_special, "o", _LINE_COLOR),
        ("Free levels", is_special, "D", _SPECIAL_COLOR),
    )
    if interval is not None and ti.ci_lower is not None:
        ci_lo = np.asarray(ti.ci_lower)
        ci_hi = np.asarray(ti.ci_upper)
        yerr = np.vstack([level_rel - ci_lo, ci_hi - level_rel])
        for label, mask, marker, color in marker_specs:
            if not mask.any():
                continue
            ax.errorbar(
                x_pos[mask],
                level_rel[mask],
                yerr=yerr[:, mask],
                fmt=marker,
                color=color,
                markersize=7,
                ecolor="#333333",
                elinewidth=1.2,
                capsize=4,
                label=label,
                zorder=5,
            )
    else:
        for label, mask, marker, color in marker_specs:
            if not mask.any():
                continue
            ax.scatter(
                x_pos[mask],
                level_rel[mask],
                color=color,
                s=50,
                marker=marker,
                zorder=5,
                label=label,
            )

    if is_special.any():
        divider = 0.5 * (float(x_pos[~is_special].max()) + float(x_pos[is_special].min()))
        ax.axvline(divider, linestyle=":", linewidth=_REF_LW, color=_REF_COLOR, zorder=1)

    ax.set_xticks(x_pos)
    rot = 45 if n_levels > 8 else 0
    ha = "right" if rot else "center"
    ax.set_xticklabels(levels, rotation=rot, ha=ha, fontsize=8)
    ax.set_xlim(float(x_pos.min()) - 0.5 * step, float(x_pos.max()) + 0.5 * step)
    ax.set_ylabel("Relativity")
    ax.set_title(ti.name, fontweight="bold")
    ax.grid(alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
```

- [ ] **Step 13: Run the matplotlib tests**

Run: `uv run pytest tests/test_ordered_categorical_specials_plots.py -k matplotlib -v && uv run pytest tests/test_plot_api.py -q`
Expected: PASS — the new test passes and `tests/test_plot_api.py` stays green (it pins the plotly OC panel, not the matplotlib one).

- [ ] **Step 14: Commit the matplotlib panel**

```bash
git add src/superglm/plotting/main_effects.py tests/test_ordered_categorical_specials_plots.py
git commit -m "feat(plotting): draw OC free levels as detached points in the matplotlib panel

The fitted curve spans the ordered levels only; specials get their own marker,
tick, exposure bar and a divider, two level-steps clear of the curve."
```

- [ ] **Step 15: Write the failing plotly test**

Append to `tests/test_ordered_categorical_specials_plots.py`:

```python
@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="plotly not installed")
def test_plotly_panel_renders_the_special_in_markers_bars_and_ticks(specials_model):
    # False today: level_x is 10 long against 11 levels, so plotly renders
    # min(len(x), len(y)) and MISSING is silently dropped from the marker trace
    # (main_effects_plotly.py:1130-1149) and the exposure bars (:1360-1378),
    # while the tick config (:739-751) zips 10 tickvals against 11 ticktext.
    model, frame, sample_weight = specials_model
    fig = model.plot(engine="plotly", X=frame, sample_weight=sample_weight)

    ticktext = list(fig.layout.xaxis.ticktext)
    tickvals = np.asarray(fig.layout.xaxis.tickvals, dtype=np.float64)
    assert ticktext == [*BANDS, "MISSING"]
    assert len(tickvals) == 11

    markers = next(t for t in fig.data if t.type == "scatter" and t.name == "Relativity")
    free = next(t for t in fig.data if t.type == "scatter" and t.name == "Free levels")
    assert len(markers.x) == 10
    assert list(free.hovertext) == ["MISSING"]
    assert float(free.x[0]) == pytest.approx(float(tickvals[-1]))
    assert float(free.x[0]) > max(float(v) for v in markers.x)
    assert len(free.error_y.array) == 1
    # customdata must travel with the special, not stay behind by the offset
    assert float(free.customdata[0][0]) == pytest.approx(float(free.y[0]))

    curve = next(t for t in fig.data if t.type == "scatter" and t.name == "Smooth curve")
    assert max(float(v) for v in curve.x) < float(free.x[0])

    bars = next(t for t in fig.data if t.type == "bar" and t.name == "Exposure")
    assert len(bars.x) == 11
    assert float(bars.x[-1]) == pytest.approx(float(tickvals[-1]))
    assert str(bars.customdata[-1]) == "MISSING"
    assert float(bars.y[-1]) > 0.0
```

- [ ] **Step 16: Run the test to verify it fails**

Run: `uv run pytest tests/test_ordered_categorical_specials_plots.py -k plotly -v`
Expected: FAIL — `StopIteration` from `next(t for t in fig.data if t.type == "scatter" and t.name == "Free levels")`; the marker trace is 10 long with MISSING silently absent.

- [ ] **Step 17: Split the plotly marker trace and add the style key**

Extend the common import block at `main_effects_plotly.py:14-34` with `_PLOTLY_SPECIAL_COLOR` (after `_PLOTLY_SIM_FILL`) and `_level_positions_with_specials` (after `_hex_to_rgba`), and add the module alias beside `_KNOT_COLOR = _PLOTLY_KNOT_COLOR` at line 45:

```python
_SPECIAL_COLOR = _PLOTLY_SPECIAL_COLOR
```

Add the style default in `_resolve_plotly_style` (`main_effects_plotly.py:57-69`), after `"line_color": _LINE_COLOR,`:

```python
        "special_color": _SPECIAL_COLOR,
```

Then in `_add_categorical_term_trace`, replace the position/marker block at `main_effects_plotly.py:1102-1109` and the single marker trace at `:1129-1158`. The head becomes:

```python
    if is_ordered:
        # OrderedCategorical: markers with error bars + smooth curve overlay.
        # Ordered levels sit under the curve; free levels are detached points
        # past its end — level_x covers the ordered levels only.
        level_x = (
            np.asarray(curve.level_x, dtype=np.float64)
            if curve is not None and curve.level_x is not None
            else np.arange(len(ti.levels), dtype=np.float64)
        )
        is_special = (
            np.asarray(ti.level_is_special, dtype=bool)
            if ti.level_is_special is not None
            else np.zeros(len(ti.levels), dtype=bool)
        )
        x_all = _level_positions_with_specials(level_x, ti.level_is_special, len(ti.levels))
        level_labels = [str(level) for level in ti.levels]
```

(the `customdata` / `resp_hover_marker` / `link_hover` block at `:1110-1127` is unchanged), and the marker trace becomes:

```python
        # Marker traces with error bars — ordered levels, then free levels
        marker_specs = (
            ("Relativity", ~is_special, "circle", style_cfg["line_color"]),
            ("Free levels", is_special, "diamond", style_cfg["special_color"]),
        )
        for trace_name, mask, symbol, marker_color in marker_specs:
            if not mask.any():
                continue
            trace_error_y = None
            if resp_error_y is not None:
                trace_error_y = dict(
                    resp_error_y,
                    array=resp_error_y["array"][mask],
                    arrayminus=resp_error_y["arrayminus"][mask],
                )
            fig.add_trace(
                go.Scatter(
                    x=x_all[mask].tolist(),
                    y=resp_y[mask],
                    mode="markers",
                    name=trace_name,
                    marker=dict(
                        size=9,
                        symbol=symbol,
                        color=marker_color,
                        line=dict(color=style_cfg["text_outline_color"], width=0.8),
                    ),
                    error_y=trace_error_y,
                    customdata=customdata[mask],
                    hovertext=[lab for lab, keep in zip(level_labels, mask) if keep],
                    legendgroup=f"{ti.name}:markers",
                    hovertemplate=resp_hover_marker,
                ),
                row=1,
                col=1,
            )
            entries.append(_TraceEntry(term_idx=term_idx, default_visibility=True))
            link_variants.append(
                _LinkVariant(
                    y=link_y[mask].tolist(),
                    hovertemplate=link_hover,
                    error_y_array=(
                        np.asarray(link_err_up)[mask].tolist()
                        if link_err_up is not None
                        else None
                    ),
                    error_y_arrayminus=(
                        np.asarray(link_err_down)[mask].tolist()
                        if link_err_down is not None
                        else None
                    ),
                )
            )
```

The "Smooth curve" overlay at `:1160-1177` is unchanged — it already draws `curve.x`, which spans the ordered levels only.

- [ ] **Step 18: Widen the plotly exposure bars and tick config**

In `_add_categorical_density_trace`, replace `main_effects_plotly.py:1358-1364`:

```python
    if ti.smooth_curve is not None and ti.smooth_curve.level_x is not None:
        level_x = np.asarray(ti.smooth_curve.level_x, dtype=np.float64)
        fig.add_trace(
            go.Bar(
                x=level_x.tolist(),
                y=weights,
                width=_ordered_bar_width(level_x),
```

with:

```python
    if ti.smooth_curve is not None and ti.smooth_curve.level_x is not None:
        level_x = np.asarray(ti.smooth_curve.level_x, dtype=np.float64)
        x_all = _level_positions_with_specials(level_x, ti.level_is_special, len(levels))
        fig.add_trace(
            go.Bar(
                x=x_all.tolist(),
                y=weights,
                width=_ordered_bar_width(x_all),
```

In `_add_term_traces`, replace the tick config at `main_effects_plotly.py:739-741`:

```python
    if ti.smooth_curve is not None and ti.smooth_curve.level_x is not None:
        tickvals = np.asarray(ti.smooth_curve.level_x, dtype=np.float64).tolist()
        ticktext = [str(level) for level in ti.levels]
```

with:

```python
    if ti.smooth_curve is not None and ti.smooth_curve.level_x is not None:
        tickvals = _level_positions_with_specials(
            np.asarray(ti.smooth_curve.level_x, dtype=np.float64),
            ti.level_is_special,
            len(ti.levels),
        ).tolist()
        ticktext = [str(level) for level in ti.levels]
```

- [ ] **Step 19: Run the plotly tests**

Run: `uv run pytest tests/test_ordered_categorical_specials_plots.py -v`
Expected: PASS (10 tests).

- [ ] **Step 20: Run the surrounding suites and the linters**

Run: `uv run pytest tests/test_plot_api.py tests/test_plot_comparison.py tests/test_interaction_plots.py tests/test_ordered_categorical_inference.py -q && uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/`
Expected: PASS — no failures, `All checks passed!`, and no formatting diff. `test_plotly_ordered_categorical_spline_uses_numeric_axis` still finds the first `name="Relativity"` scatter at the midpoint positions, because a term with no specials produces exactly one marker trace.

- [ ] **Step 21: Commit the plotly panel**

```bash
git add src/superglm/plotting/main_effects_plotly.py tests/test_ordered_categorical_specials_plots.py
git commit -m "feat(plotting): render OC free levels in the plotly panel

Markers split into 'Relativity' and 'Free levels' traces, exposure bars and
tick labels run over the full level list, and hover customdata travels with
its own trace instead of being truncated to min(len(x), len(y))."
```

---

### Task 9: Screening deferral and interaction refusal


**Files:**
- Create: `tests/test_ordered_categorical_specials_screening.py`
- Modify: `src/superglm/model/screening_ops.py:96` (import), `:245-263` (`_margin_kind`), `:266` (new `_deferral_reason`), `:282-310` (`_validated_pairs`), `:459-468` (docstring), `:602-615` (`deferred_features`), `:1352-1355` (`attrs`)
- Modify: `src/superglm/features/ordered_categorical.py:566-583`
- Modify: `src/superglm/dm_builder.py:439-455`
- Modify: `docs/guide/screening.md:82-89`
- Test: `tests/test_ordered_categorical_specials_screening.py`, `tests/test_ordered_categorical_interactions.py`

**Interfaces:**
- Consumes: `OrderedCategorical(order=..., specials=[...], basis=Spline(...))`, `spec._specials: list[str]`, `spec.has_specials -> bool`; `resolve_interaction_parent(spec, x)` (`features/ordered_categorical.py:566`); `_margin_kind(spec) -> str | None` (`model/screening_ops.py:245`); `_validated_pairs(candidates, margin_kinds, fitted_pairs, fitted_names)` (`:282`, sole caller `:615`); `add_interaction(...)` (`dm_builder.py:419`).
- Produces: `screening_ops._deferral_reason(spec) -> str`; `_margin_kind` returning `None` for a specials OC; `_validated_pairs(candidates, margin_kinds, fitted_pairs, deferred_features)`; `table.attrs["deferred_features"]: dict[str, str]`; `NotImplementedError` from `resolve_interaction_parent` and from `add_interaction` for a specials parent.

**ORDER IS LOAD-BEARING.** Steps 3-6 change `_margin_kind` and the deferral report *first*; the resolver guard only arrives at Step 9. Adding the guard before `_margin_kind` returns `None` would abort the entire automatic sweep: `_margin_source` (`screening_ops.py:661-682`) calls `resolve_interaction_parent` unguarded at `:673`, inside the eager pre-read loop at `:697-705`, before a single statistic is computed. Do not reorder.

- [ ] **Step 1: Write the failing screening tests**

Create `tests/test_ordered_categorical_specials_screening.py`:

```python
"""A specials OrderedCategorical is deferred by screening, and says so."""

import warnings

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, SuperGLM
from superglm.features.ordered_categorical import OrderedCategorical
from superglm.features.polynomial import Polynomial
from superglm.features.spline import Spline

BANDS = [str(i) for i in range(1, 11)]


def _specials_frame(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    band = rng.choice(BANDS, n)
    band = np.where(rng.random(n) < 0.18, "MISSING", band)
    df = pd.DataFrame(
        {
            "band": band,
            "region": rng.choice(list("ABCD"), n),
            "age": rng.uniform(18.0, 80.0, n),
        }
    )
    y = rng.poisson(np.exp(-1.5 + 0.004 * df["age"])).astype(np.float64)
    return df, y


def _specials_oc():
    return OrderedCategorical(
        order=BANDS,
        specials=["MISSING"],
        basis=Spline(kind="ps", n_knots=6),
    )


def _fit_with_specials(df, y):
    model = SuperGLM(
        family="poisson",
        features={
            "band": _specials_oc(),
            "region": Categorical(),
            "age": Spline(kind="ps", n_knots=6),
        },
    )
    model.fit_reml(df, y)
    return model


def test_a_specials_term_is_excluded_without_aborting_the_sweep():
    # FALSE TODAY: _margin_kind reads a specials OC as "spline", so the eager
    # pre-read resolves it to level scores, MISSING maps to NaN (it is a known
    # level with no entry in _level_to_value) and the WHOLE sweep dies with
    # "screen_interactions requires finite covariates; 'band' maps to
    # non-finite scores" before one statistic is computed.
    df, y = _specials_frame()
    model = _fit_with_specials(df, y)

    table = model.screen_interactions(df, y)

    pairs = {frozenset((a, b)) for a, b in zip(table["feature_a"], table["feature_b"])}
    assert pairs == {frozenset(("region", "age"))}
    assert "band" not in set(table["feature_a"]) | set(table["feature_b"])
    assert np.isfinite(table["z"]).all()


def test_the_deferred_term_and_its_reason_are_reported_on_the_table():
    # FALSE TODAY: attrs carries "phi" alone, so a term that was never screened
    # is indistinguishable from one that screened badly.
    df, y = _specials_frame()
    model = _fit_with_specials(df, y)

    table = model.screen_interactions(df, y)

    deferred = table.attrs["deferred_features"]
    assert set(deferred) == {"band"}
    assert "specials" in deferred["band"]
    assert "deferred" in deferred["band"]


def test_naming_a_specials_term_in_candidates_raises_with_the_reason():
    # FALSE TODAY: the pair is accepted and dies in the pre-read with the
    # generic "maps to non-finite scores" -- also a ValueError, so this test
    # only passes once the deferral is what refuses it.
    df, y = _specials_frame()
    model = _fit_with_specials(df, y)

    with pytest.raises(ValueError, match="no screenable margin") as excinfo:
        model.screen_interactions(df, y, candidates=[("band", "age")])
    assert "specials" in str(excinfo.value)


def test_polynomial_and_step_mode_oc_are_reported_deferred_too():
    # FALSE TODAY: both are dropped silently -- the sweep returns exactly one
    # row (region x age) and no record that two fitted mains were skipped.
    df, y = _specials_frame()
    df = df.assign(dens=np.linspace(0.0, 1.0, len(df)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        step = OrderedCategorical(order=BANDS + ["MISSING"], basis="step")
        model = SuperGLM(
            family="poisson",
            features={
                "band": step,
                "region": Categorical(),
                "age": Spline(kind="ps", n_knots=6),
                "dens": Polynomial(degree=2),
            },
        )
        model.fit_reml(df, y)

    table = model.screen_interactions(df, y)

    deferred = table.attrs["deferred_features"]
    assert set(deferred) == {"band", "dens"}
    assert "step" in deferred["band"]
    assert "Polynomial" in deferred["dens"]
    assert "age" not in deferred and "region" not in deferred


def test_a_fully_screenable_model_reports_an_empty_mapping():
    # FALSE TODAY: KeyError -- the key does not exist at all.
    df, y = _specials_frame(n=1500)
    model = SuperGLM(
        family="poisson",
        features={"region": Categorical(), "age": Spline(kind="ps", n_knots=6)},
    )
    model.fit_reml(df, y)

    table = model.screen_interactions(df, y)

    assert table.attrs["deferred_features"] == {}
```

- [ ] **Step 2: Run the screening tests to verify they fail**

Run: `uv run pytest tests/test_ordered_categorical_specials_screening.py -v`

Expected: FAIL. `test_a_specials_term_is_excluded_without_aborting_the_sweep` and `test_naming_a_specials_term_in_candidates_raises_with_the_reason` fail with `ValueError: screen_interactions requires finite covariates; 'band' maps to non-finite scores` (raised at `screening_ops.py:675`); the other three fail with `KeyError: 'deferred_features'`.

- [ ] **Step 3: Make `_margin_kind` refuse a specials term (this must land before the resolver guard)**

In `src/superglm/model/screening_ops.py:245-263`, replace the `OrderedCategorical` branch:

```python
def _margin_kind(spec) -> str | None:
    """Classify a fitted spec for screening; None means not screenable."""
    if isinstance(spec, _SplineBase):
        return "spline"
    if isinstance(spec, OrderedCategorical):
        # A term with specials= is refused HERE, ahead of the basis test and
        # before any column is read.  A special is a free level with no
        # position on the spline axis, so ``resolve_interaction_parent``
        # refuses it -- and that resolver runs inside the eager pre-read
        # below, which would abort the WHOLE sweep on the first specials term
        # rather than skipping it.  Screening one needs composite margins.
        if spec.has_specials:
            return None
        # A spline-mode OC is a spline through the level values, so it screens
        # (and refits) exactly like one; step mode has no interaction target.
        if spec.basis == "spline" and spec._spline is not None:
            return "spline"
        return None
    if isinstance(spec, Categorical):
```

Leave the rest of the function (the `Categorical`, `Numeric` and fallback branches, `:255-263`) untouched.

- [ ] **Step 4: Run the screening tests again**

Run: `uv run pytest tests/test_ordered_categorical_specials_screening.py -v`

Expected: `test_a_specials_term_is_excluded_without_aborting_the_sweep` PASSES (the sweep now runs and reports region x age alone). The other four still FAIL — three on `KeyError: 'deferred_features'`, and the candidates test on the `match="no screenable margin"` assertion now reaching the *old* generic message (`... that have no screenable margin — spline x numeric screening is deferred ...`), whose `str(excinfo.value)` does not contain "specials".

- [ ] **Step 5: Add `_deferral_reason` and report the mapping**

In `src/superglm/model/screening_ops.py`, add the import after the `ordered_categorical` block (before `from superglm.features.spline import _SplineBase` at `:96`):

```python
from superglm.features.polynomial import Polynomial
```

Then add this function immediately after `_margin_kind`, above the `_PAIR_KINDS` table at `:269`:

```python
def _deferral_reason(spec) -> str:
    """Why a fitted main effect has no screenable margin.

    Called only for names ``_margin_kind`` refused, so every branch is a
    deferral rather than an error, and the reason is REPORTED --- on
    ``table.attrs["deferred_features"]`` and in the candidates error --- rather
    than dropped on the floor.  Polynomial and step-mode OrderedCategorical
    were silently skipped before this existed, which is the same defect.
    """
    if isinstance(spec, OrderedCategorical):
        if spec.has_specials:
            return (
                "OrderedCategorical with specials= is deferred: a special is a free "
                "level with no position on the spline axis, so the margin has no "
                "score to grid on; screening the pair needs composite margins"
            )
        return (
            "step-mode OrderedCategorical is deferred: the deprecated one-hot "
            "geometry has no marginal smooth to cross with"
        )
    if isinstance(spec, Polynomial):
        return (
            "Polynomial margins are deferred: the basis is not a penalized marginal "
            "smooth, so no interaction class refits the pair"
        )
    return f"{type(spec).__name__} margins are deferred: no screenable margin"
```

Now build the mapping where `margin_kinds` is built, `:602-606`:

```python
    margin_kinds = {
        name: kind
        for name in model._feature_order
        if (kind := _margin_kind(model._specs.get(name))) is not None
    }
    # Every fitted main is either screenable or REPORTED as deferred.  The two
    # sets partition ``_feature_order``, so a caller can tell a feature that
    # screened badly from one that was never screened at all.
    deferred_features = {
        name: _deferral_reason(model._specs.get(name))
        for name in model._feature_order
        if name not in margin_kinds
    }
```

Change the call at `:615` to pass it instead of `set(model._specs)`:

```python
    pairs = _validated_pairs(candidates, margin_kinds, fitted_pairs, deferred_features)
```

And attach it beside `phi` at `:1352-1355`:

```python
    table = pd.DataFrame(rows, columns=_RESULT_COLUMNS)
    table = table.sort_values("z", ascending=False, ignore_index=True)
    table.attrs["phi"] = phi_hat
    table.attrs["deferred_features"] = deferred_features
    return table
```

- [ ] **Step 6: Give the candidates error the per-feature reason**

In `_validated_pairs` (`src/superglm/model/screening_ops.py:282-310`), rename the fourth parameter and use the mapping. `_DEFERRED_KIND_HINT` stays in use at `:321` for the pair-kind branch, which is unchanged.

```python
def _validated_pairs(candidates, margin_kinds, fitted_pairs, deferred_features):
```

```python
        if len(pair) == 2 and pair[0] != pair[1]:
            # A name the model DID fit but cannot screen (Polynomial, step-mode
            # or specials OrderedCategorical) is deferred, not a typo; listing
            # the screenable features would send the caller hunting for a
            # misspelling that isn't there.  The reason quoted here is the same
            # string the result table reports.
            deferred_names = sorted(name for name in pair if name in deferred_features)
            if deferred_names:
                detail = "; ".join(f"{n} — {deferred_features[n]}" for n in deferred_names)
                raise ValueError(
                    f"candidates entry {raw!r} names fitted feature(s) "
                    f"{deferred_names} that have no screenable margin: {detail}"
                )
```

Document the mapping in the `screen_interactions` docstring, after the `attrs["phi"]` sentence that ends at `:468`:

```python
    Fitted mains with no screenable margin — ``Polynomial``, ``RandomEffect``,
    step-mode ``OrderedCategorical`` and any ``OrderedCategorical`` carrying
    ``specials=`` — are excluded from the sweep and reported in
    ``table.attrs["deferred_features"]``, a ``{feature: reason}`` mapping that
    is empty when everything fitted was screened.  Naming one of them in
    ``candidates`` raises with the same reason.
```

- [ ] **Step 7: Run the screening tests and the existing screening suites**

Run: `uv run pytest tests/test_ordered_categorical_specials_screening.py tests/test_mixed_interaction_screening.py -q`

Expected: PASS. `tests/test_mixed_interaction_screening.py:82` (`test_candidates_rejects_deferred_and_ineligible_kinds`) still matches on `"deferred"` — the Polynomial reason contains it — and its `"screenable features"` case is untouched.

- [ ] **Step 8: Write the failing interaction-refusal tests**

Append to `tests/test_ordered_categorical_interactions.py` (its existing header imports `Categorical`, `SuperGLM`, `SplineCategorical`, `TensorInteraction`, `OrderedCategorical`, `resolve_interaction_parent`, `Spline` at lines 7-17; add the FactorSmooth import beside them):

```python
from superglm.features.factor_smooth import FactorSmooth
```

```python
def _specials_frame(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    band = rng.choice(BANDS, n)
    band = np.where(rng.random(n) < 0.18, "MISSING", band)
    df = pd.DataFrame(
        {
            "age_band": band,
            "region": rng.choice(list("ABCD"), n),
            "power": rng.uniform(20.0, 200.0, n),
        }
    )
    y = rng.poisson(np.exp(-1.5 + 0.002 * df["power"])).astype(np.float64)
    return df, y


def _oc_specials():
    return OrderedCategorical(
        order=BANDS,
        specials=["MISSING"],
        basis=Spline(kind="ps", n_knots=4),
    )


def test_resolver_rejects_a_specials_parent():
    """FALSE TODAY: the resolver has no specials rule, so it validates MISSING
    as a known level and returns NaN scores from _map_to_numeric instead."""
    spec = _oc_specials()
    with pytest.raises(NotImplementedError, match="specials"):
        resolve_interaction_parent(spec, np.array(["18-25", "MISSING"], dtype=object))


def test_specials_parent_is_rejected_where_the_interaction_is_declared():
    """FALSE TODAY: the declaration guard only knows about basis='step', so a
    specials pair registers and fails much later, mid design-matrix build."""
    df, y = _specials_frame()

    with pytest.raises(NotImplementedError, match="age_band.*specials"):
        SuperGLM(
            family="poisson",
            features={"age_band": _oc_specials(), "region": Categorical()},
            interactions=[("age_band", "region")],
        ).fit_reml(df, y)

    # ... and through the incremental API, with the OC in either position
    model = SuperGLM(
        family="poisson",
        features={"age_band": _oc_specials(), "power": Spline(kind="ps", n_knots=4)},
    )
    with pytest.raises(NotImplementedError, match=r"\('power', 'age_band'\)"):
        model._add_interaction("power", "age_band")
    assert model._interaction_specs == {}  # nothing was half-registered


def test_an_explicit_interaction_spec_cannot_smuggle_a_specials_parent_past_it():
    """FALSE TODAY: base.py:805-811 deep-copies anything carrying .parent_names
    and .name straight into _interaction_specs without calling add_interaction,
    so the declaration guard never runs for an explicit spec object.  Nothing
    then refuses the pair -- it reaches the build with the special's score NaN.
    The resolver guard at dm_builder.py:1071 is what stops this form."""
    df, y = _specials_frame()
    ti = TensorInteraction("age_band", "power")
    ti.name = "age_band:power"
    model = SuperGLM(
        family="poisson",
        features={"age_band": _oc_specials(), "power": Spline(kind="ps", n_knots=4)},
        interactions=[ti],
    )
    with pytest.raises(NotImplementedError, match="specials"):
        model.fit_reml(df, y)


def test_a_specials_term_may_still_be_a_factor_smooth_group():
    """Pins the EXEMPTION, so the refusal cannot be widened by accident: a
    FactorSmooth reads its group column as labels, never as scores, and
    resolve_interaction_parent_of hands it both columns untouched.  This fails
    if the guard is put in resolve_interaction_parent_of instead."""
    df, y = _specials_frame()
    model = SuperGLM(
        family="poisson",
        features={"age_band": _oc_specials(), "power": Spline(kind="ps", n_knots=4)},
        interactions=[FactorSmooth(variable="power", group="age_band")],
    )
    model.fit_reml(df, y)
    assert np.isfinite(model._result.effective_df)
```

- [ ] **Step 9: Run the interaction tests to verify they fail**

Run: `uv run pytest tests/test_ordered_categorical_interactions.py -v -k specials`

Expected: FAIL. `test_resolver_rejects_a_specials_parent` fails with `DID NOT RAISE NotImplementedError`; the two declaration/explicit-spec tests fail with `DID NOT RAISE` or with an unrelated numpy/ValueError from the NaN scores. `test_a_specials_term_may_still_be_a_factor_smooth_group` passes already — it is the exemption pin.

- [ ] **Step 10: Add the refusal to the resolver (only now that `_margin_kind` returns None)**

In `src/superglm/features/ordered_categorical.py:566-583`, extend the docstring and add the guard after the existing step-mode raise:

```python
def resolve_interaction_parent(spec: Any, x: NDArray) -> tuple[Any, NDArray]:
    """Resolve one interaction parent (spec, column) for assembly.

    Identity for every spec — including ``None``, which FactorSmooth group
    columns carry — except spline-mode OrderedCategorical, which
    contributes its inner Spline on the mapped numeric scores, applying
    the same grouping, level validation, and score mapping its own
    ``build``/``transform`` apply.  Step-mode OC cannot parent an
    interaction: the deprecated one-hot geometry has no marginal smooth.
    Neither can a term carrying ``specials=``: a special is a free level with
    no position on the spline axis, so there is no single marginal smooth to
    cross with.
    """
    if not isinstance(spec, OrderedCategorical):
        return spec, x
    if spec.basis != "spline" or spec._spline is None:
        raise NotImplementedError(
            "OrderedCategorical with basis='step' is deprecated and cannot parent "
            "an interaction; use basis=Spline(...) for a smoothed ordinal parent "
            "or a Categorical feature for unsmoothed level effects."
        )
    if spec.has_specials:
        raise NotImplementedError(
            f"OrderedCategorical with specials={spec._specials!r} cannot parent an "
            "interaction: a special is a free level with no position on the spline "
            "axis, so the term has no single marginal smooth to cross with; drop "
            "specials= to interact the smoothed ordinal parent, or use a Categorical "
            "feature for unsmoothed level effects."
        )
    x = np.asarray(x).ravel()
```

- [ ] **Step 11: Refuse the pair where it is declared**

In `src/superglm/dm_builder.py:439-455`, extend the pre-existing parent loop (do not replace the step-mode raise; both refusals live in the same loop):

```python
    # _spec_kind reads a step-mode OrderedCategorical as "categorical", which
    # is right for its MAIN effect but wrong for an interaction parent: the
    # deprecated one-hot geometry has no marginal smooth to cross with, and
    # resolve_interaction_parent refuses it.  A specials term is refused for
    # the same reason -- its free levels have no spline-axis position.  Without
    # this the pair registers here and only fails much later, mid design-matrix
    # build, after the caller has already committed a fit.
    for parent in (feat1, feat2):
        spec = specs[parent]
        if isinstance(spec, OrderedCategorical) and (
            spec.basis != "spline" or spec._spline is None
        ):
            raise NotImplementedError(
                f"cannot add the interaction ({feat1!r}, {feat2!r}): {parent!r} is an "
                "OrderedCategorical with basis='step', which is deprecated and cannot "
                "parent an interaction; use basis=Spline(...) for a smoothed ordinal "
                "parent or a Categorical feature for unsmoothed level effects."
            )
        if isinstance(spec, OrderedCategorical) and spec.has_specials:
            raise NotImplementedError(
                f"cannot add the interaction ({feat1!r}, {feat2!r}): {parent!r} is an "
                f"OrderedCategorical with specials={spec._specials!r}, whose free "
                "levels have no position on the spline axis and so no marginal smooth "
                "to cross with; drop specials= to interact the smoothed ordinal "
                "parent, or use a Categorical feature for unsmoothed level effects."
            )
```

- [ ] **Step 12: Run both test files**

Run: `uv run pytest tests/test_ordered_categorical_interactions.py tests/test_ordered_categorical_specials_screening.py -v`

Expected: PASS, including the pre-existing step-mode tests at `tests/test_ordered_categorical_interactions.py:58-97`, whose messages and behaviour are unchanged.

- [ ] **Step 13: Update the screening guide**

In `docs/guide/screening.md:82-89`, replace:

```markdown
**What gets swept.** `candidates=None` pairs every eligible fitted feature:
splines, spline-mode `OrderedCategorical`, `Categorical` and `Numeric`.
`Polynomial`, `RandomEffect` and
step-mode `OrderedCategorical` have no screenable
margin. A `Categorical` carrying a `grouping=` is eligible too:
```

with:

```markdown
**What gets swept.** `candidates=None` pairs every eligible fitted feature:
splines, spline-mode `OrderedCategorical`, `Categorical` and `Numeric`.
`Polynomial`, `RandomEffect`, step-mode `OrderedCategorical` and any
`OrderedCategorical` carrying `specials=` have no screenable margin — a
special is a free level with no position on the spline axis, so the pair
would need a composite margin. Each of them is reported in
`table.attrs["deferred_features"]`, a `{feature: reason}` mapping, and naming
one in `candidates` raises with that same reason. A `Categorical` carrying a
`grouping=` is eligible too:
```

- [ ] **Step 14: Full verification**

Run: `uv run pytest tests/ -q && uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/`

Expected: PASS on all three.

- [ ] **Step 15: Commit (both halves together — the screening change and the refusal must not be split)**

```bash
git add src/superglm/model/screening_ops.py src/superglm/features/ordered_categorical.py \
        src/superglm/dm_builder.py docs/guide/screening.md \
        tests/test_ordered_categorical_specials_screening.py \
        tests/test_ordered_categorical_interactions.py
git commit -m "feat: defer specials terms from screening and refuse them as interaction parents

_margin_kind returns None for an OrderedCategorical carrying specials=, which
excludes it from the automatic sweep and gives the purpose-built candidates
error.  That change lands with the resolver guard, never after it: the sweep
resolves every margin in an eager pre-read, so a NotImplementedError in
resolve_interaction_parent alone would abort the whole screen.

Deferral is now reported rather than silent -- table.attrs[\"deferred_features\"]
maps feature name to reason -- which also covers Polynomial and step-mode
OrderedCategorical, dropped without a word until now.  The pair is refused
where it is declared as well as in the resolver, because an explicit
interaction spec bypasses add_interaction entirely."
```

---

### Task 10: Editor: exact-assignment refit, collapse rules, spec clone


**Files:**
- Create: `tests/test_ordered_categorical_specials_editor.py`
- Modify: `src/superglm/editor/apply.py:150-193`
- Modify: `src/superglm/editor/collapse.py:51-64`
- Modify: `src/superglm/editor/collapse.py:347-388`
- Modify: `src/superglm/inference/_term_helpers.py:158-259`
- Test: `tests/test_ordered_categorical_specials_editor.py`

**Interfaces:**
- Consumes: `OrderedCategorical(..., specials=[...])` with `_specials: list[str]`, `_smooth_levels: list[str]`, `_ordered_levels: list[str]`, `has_specials -> bool`, `_split_beta(beta) -> (spline_beta, special_beta)`; `spec.transform(labels)` emitting `[centered spline | special indicators]` with zero spline columns on special rows; GroupSlices named `"<feature>"` and `"<feature>:special"` in that column order; `TermInference.level_is_special: NDArray[np.bool_] | None`; `SmoothCurve.level_x` smooth-levels-only.
- Produces: `_apply_ordered_spline_term(model, spec, groups, term) -> None` (editor/apply.py); `_require_no_special_members(spec, term_name, members) -> None` (editor/collapse.py); `_ordered_spec_with_grouping` forwarding `specials=`; `_expand_grouped_term` returning an expanded `level_is_special` and a smooth-only `level_x`.

---

- [ ] **Step 1: Write the failing editor-refit tests**

Create `tests/test_ordered_categorical_specials_editor.py`:

```python
"""Editor behaviour for OrderedCategorical terms that declare specials."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from superglm import OrderedCategorical, Spline, SuperGLM
from superglm.editor import EditorSession

SMOOTH_LEVELS = ["1", "2", "3", "4", "5", "6"]
SMOOTH_EFFECT = {"1": -0.30, "2": -0.18, "3": -0.05, "4": 0.06, "5": 0.15, "6": 0.20}


def _fit(specials, probabilities, effects):
    rng = np.random.default_rng(20260805)
    labels = rng.choice(SMOOTH_LEVELS + specials, 900, p=probabilities)
    X = pd.DataFrame({"band": labels})
    y = np.array([effects[label] for label in labels]) + rng.normal(0.0, 0.15, 900)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={
            "band": OrderedCategorical(
                order=SMOOTH_LEVELS,
                specials=specials,
                basis=Spline(kind="ps", k=8),
            )
        },
    )
    model.fit(X, y)
    return model, X, y


@pytest.fixture
def specials_model():
    # k=8 over 6 smoothed levels gives a 7-column spline block, so the ordered
    # system is 6 rows x 8 unknowns: underdetermined, which is exactly where a
    # joint least-squares solve over the special columns splits the edit between
    # the intercept and the spline block instead of assigning it outright.
    return _fit(["MISSING"], [0.14] * 6 + [0.16], {**SMOOTH_EFFECT, "MISSING": 0.55})


@pytest.fixture
def two_specials_model():
    return _fit(
        ["MISSING", "UNKNOWN"],
        [0.14] * 6 + [0.10, 0.06],
        {**SMOOTH_EFFECT, "MISSING": 0.55, "UNKNOWN": -0.40},
    )


def _band_blocks(model):
    groups = {str(group.name): group for group in model._groups if group.feature_name == "band"}
    return groups["band"], groups["band:special"]


def _edit_special(model, level, delta):
    session = EditorSession.from_model(model, terms=["band"])
    session.select_levels("band", [level])
    session.shift("band", delta)
    edited = session.to_model()
    spline_group, special_group = _band_blocks(edited)
    return (
        float(edited.result.intercept),
        np.asarray(edited.result.beta[spline_group.sl], dtype=np.float64).copy(),
        np.asarray(edited.result.beta[special_group.sl], dtype=np.float64).copy(),
    )


def test_special_edit_leaves_the_ordered_projection_untouched(specials_model):
    model, _, _ = specials_model

    small = _edit_special(model, "MISSING", 0.5)
    large = _edit_special(model, "MISSING", 1.5)

    # Only the ordered levels go through the least-squares projection, so the
    # SIZE of a special's edit may not move the intercept or the spline block.
    # Today `_apply_projected_term` (apply.py:185-193) solves one joint system
    # over all 7 rows and 9 columns; its min-norm solution trades the intercept
    # off against the special's coefficient, so both of these move with delta.
    assert small[0] == pytest.approx(large[0], abs=1e-12)
    np.testing.assert_allclose(small[1], large[1], atol=1e-12)
    # ...and the special's own coefficient carries the whole difference.
    np.testing.assert_allclose(large[2] - small[2], 1.0, atol=1e-12)


def test_editing_one_special_leaves_the_other_special_coefficient_alone(two_specials_model):
    model, _, _ = two_specials_model

    small = _edit_special(model, "UNKNOWN", 0.5)
    large = _edit_special(model, "UNKNOWN", 1.5)

    assert small[0] == pytest.approx(large[0], abs=1e-12)
    np.testing.assert_allclose(small[1], large[1], atol=1e-12)
    # Special indicator column j belongs to _specials[j] = ["MISSING", "UNKNOWN"].
    # MISSING was not edited and its coefficient is set from its own effect, so
    # it is identical across the two runs. Under the joint min-norm solve the
    # shared intercept column drags it with UNKNOWN's edit, so this fails today.
    np.testing.assert_allclose(small[2][0], large[2][0], atol=1e-12)
    np.testing.assert_allclose(large[2][1] - small[2][1], 1.0, atol=1e-12)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_ordered_categorical_specials_editor.py -v`
Expected: FAIL — both tests fail on the first assertion, e.g.
`assert 0.0512... == 0.1128... ± 1.0e-12` (the intercept moves with the size of the special's edit because the special rows enter `_solve_with_intercept`).

- [ ] **Step 3: Split the OC spline refit into projection + exact assignment**

In `src/superglm/editor/apply.py`, change the dispatch in `_apply_term_edit` (currently lines 154-159):

```python
    if isinstance(spec, OrderedCategorical):
        if spec.basis == "spline":
            _apply_ordered_spline_term(model, spec, groups, term)
        else:
            _apply_ordered_step_term(model, spec, groups, term)
        return
```

and insert this function immediately after `_apply_projected_term` (which ends at line 193, before `_apply_categorical_term`):

```python
def _apply_ordered_spline_term(
    model,
    spec: OrderedCategorical,
    groups: list[GroupSlice],
    term: EditableTerm,
) -> None:
    """Project ordered levels onto the spline and assign special levels exactly.

    A special contributes exactly one design row whose spline columns are zero,
    so its edited effect determines its coefficient outright.  Feeding that row
    to the least-squares solve instead leaves the intercept and the special
    coefficient jointly under-determined, and the min-norm solution splits the
    edit between them — moving the reported intercept and the fitted curve.
    """
    x_values = _ordered_spline_x(term)
    if not spec.has_specials:
        _apply_projected_term(model, spec, groups, term, x_values)
        return

    B = _as_dense(spec.transform(x_values))
    targets = native_log_effect_values(term)
    weights = _term_weights(term)
    labels = [str(level) for level in x_values]
    specials = [str(level) for level in spec._specials]
    missing = [level for level in specials if level not in labels]
    if missing:
        raise ValueError(
            f"Editable term {term.name!r} has no row for special level(s) {missing}."
        )
    row_of = {label: index for index, label in enumerate(labels)}
    special_rows = np.array([row_of[level] for level in specials], dtype=np.intp)
    smooth_rows = np.setdiff1d(np.arange(len(labels), dtype=np.intp), special_rows)
    n_spline = spec._split_beta(np.zeros(B.shape[1], dtype=np.float64))[0].size

    intercept_delta, spline_beta = _solve_with_intercept(
        B[smooth_rows, :n_spline],
        targets[smooth_rows],
        weights[smooth_rows],
    )
    special_beta = targets[special_rows] - intercept_delta
    _adjust_intercept(model, intercept_delta)
    _patch_beta_block(model, groups, np.concatenate([spline_beta, special_beta]))
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_ordered_categorical_specials_editor.py tests/test_editor.py -q`
Expected: PASS (the new tests plus the existing OC editor tests at `tests/test_editor.py:4240-4341`, which take the `not spec.has_specials` branch and are unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/superglm/editor/apply.py tests/test_ordered_categorical_specials_editor.py
git commit -m "fix(editor): assign special coefficients exactly instead of projecting them"
```

- [ ] **Step 6: Write the failing collapse tests**

Append to `tests/test_ordered_categorical_specials_editor.py`:

```python
def _term_and_indices(model, levels):
    session = EditorSession.from_model(model, terms=["band"])
    term = session.terms["band"]
    return term, np.array([term.levels.index(level) for level in levels], dtype=np.intp)


def test_collapse_refuses_a_group_that_mixes_a_special_with_ordered_levels(specials_model):
    from superglm.editor.collapse import collapsed_feature_spec

    model, X, _ = specials_model
    term, idx = _term_and_indices(model, ["6", "MISSING"])

    # "MISSING" sits last in _ordered_levels, so it is *adjacent* to "6" and the
    # contiguity check at collapse.py:58-64 waves this selection through today.
    with pytest.raises(ValueError, match="free level"):
        collapsed_feature_spec(model, term, idx, X=X)


def test_collapsing_ordered_levels_keeps_the_special_free(specials_model):
    from superglm.editor.collapse import collapsed_feature_spec

    model, X, _ = specials_model
    term, idx = _term_and_indices(model, ["2", "3"])

    replacement, metadata = collapsed_feature_spec(model, term, idx, X=X)

    assert metadata["group_label"] == "2+3"
    # _ordered_spec_with_grouping rebuilds the spec from an explicit argument
    # list (collapse.py:367-372); without specials= the free level is silently
    # smoothed back into the curve.
    assert replacement._specials == ["MISSING"]
    assert replacement._smooth_levels == ["1", "2+3", "4", "5", "6"]
    assert replacement._ordered_levels == ["1", "2+3", "4", "5", "6", "MISSING"]
    assert "MISSING" not in replacement._level_to_value
```

- [ ] **Step 7: Run the tests to verify they fail**

Run: `uv run pytest tests/test_ordered_categorical_specials_editor.py -v -k collaps`
Expected: FAIL — `Failed: DID NOT RAISE <class 'ValueError'>` for the mixed group, and `AssertionError: assert [] == ['MISSING']` for the clone.

- [ ] **Step 8: Refuse collapse groups that touch a special**

In `src/superglm/editor/collapse.py`, extend the `OrderedCategorical` branch of `collapsed_feature_spec` (lines 51-64) to call the new guard before the contiguity check:

```python
    existing = getattr(spec, "_grouping", None)
    if isinstance(spec, OrderedCategorical):
        selected_originals = _selected_original_members(
            selected_levels,
            existing,
            _displayed_members(term, existing),
        )
        _require_no_special_members(spec, term.name, selected_originals)
        if not _members_are_contiguous(
            selected_originals,
            _original_level_order(spec, term, existing),
        ):
            raise ValueError(
                f"Ordered categorical collapse for {term.name!r} must be contiguous "
                "in fitted order."
            )
```

and add the guard next to `_members_are_contiguous` (after line 480):

```python
def _require_no_special_members(spec, term_name: str, members: list[str]) -> None:
    """Refuse a collapse selection that contains a free (special) level."""
    specials = {str(level) for level in getattr(spec, "_specials", ())}
    if not specials:
        return
    selected = [member for member in members if member in specials]
    if not selected:
        return
    joined = ", ".join(repr(member) for member in selected)
    raise ValueError(
        f"Ordered categorical collapse for {term_name!r} cannot include free level(s) "
        f"{joined}: specials are fitted outside the smooth and cannot be grouped."
    )
```

- [ ] **Step 9: Carry `specials=` through the spec clone**

In `src/superglm/editor/collapse.py`, change the spline branch of `_ordered_spec_with_grouping` (lines 354-372) so the free levels survive the rebuild:

```python
    values, native_base = _ordered_original_values(spec, grouping, data, base)
    specials = list(getattr(spec, "_specials", ()) or ())
    if spec.basis == "spline":
        basis = (
            copy.deepcopy(spec._spline_obj)
            if spec._spline_obj is not None
            else Spline(
                kind=spec.kind,
                n_knots=spec.n_knots,
                degree=spec.degree,
                select=spec.select,
                penalty=spec.penalty,
            )
        )
        return OrderedCategorical(
            values=values,
            basis=basis,
            base=native_base,
            grouping=grouping,
            specials=specials or None,
        )
```

The step branch (lines 376-387) is left alone: `specials` with `basis="step"` is a construction-time `ValueError`, so a step spec never carries any.

- [ ] **Step 10: Run the tests**

Run: `uv run pytest tests/test_ordered_categorical_specials_editor.py tests/test_ordered_categorical_api.py -q`
Expected: PASS (including the three existing `_ordered_spec_with_grouping` clone tests at `tests/test_ordered_categorical_api.py:154-204`, which pass specs with no specials).

- [ ] **Step 11: Commit**

```bash
git add src/superglm/editor/collapse.py tests/test_ordered_categorical_specials_editor.py
git commit -m "fix(editor): keep specials out of collapse groups and through the spec clone"
```

- [ ] **Step 12: Write the failing grouped-expansion tests**

Append to `tests/test_ordered_categorical_specials_editor.py`:

```python
def _grouped_term_inference(curve):
    from superglm.inference._term_types import TermInference

    return TermInference(
        name="band",
        kind="categorical",
        active=True,
        levels=["1+2", "3", "MISSING"],
        log_relativity=np.array([0.0, 0.2, 0.6]),
        relativity=np.exp(np.array([0.0, 0.2, 0.6])),
        absorbs_intercept=False,
        centering_mode="base_level",
        smooth_curve=curve,
        level_is_special=np.array([False, False, True]),
    )


def _identity_plus_pair_grouping():
    from superglm.features.grouping import collapse_levels

    return collapse_levels(
        np.array(["1", "2", "3", "MISSING"], dtype=object),
        groups={"1+2": ["1", "2"]},
        order=["1", "2", "3", "MISSING"],
    )


def test_grouped_expansion_keeps_the_special_marker():
    from superglm.inference._term_helpers import _expand_grouped_term

    expanded = _expand_grouped_term(_grouped_term_inference(None), _identity_plus_pair_grouping())

    assert expanded.levels == ["1", "2", "3", "MISSING"]
    # _expand_grouped_term rebuilds TermInference by hand-listing its fields
    # (_term_helpers.py:236-258), so level_is_special is None today and every
    # grouped specials term loses its marker with no error.
    assert expanded.level_is_special is not None
    np.testing.assert_array_equal(expanded.level_is_special, [False, False, False, True])


def test_grouped_expansion_keeps_level_x_on_smoothed_levels_only():
    from superglm.inference._term_helpers import _expand_grouped_term
    from superglm.inference._term_types import SmoothCurve

    grid = np.linspace(0.0, 1.0, 5)
    curve = SmoothCurve(
        x=grid,
        log_relativity=grid * 0.2,
        relativity=np.exp(grid * 0.2),
        level_x=np.array([0.25, 1.0]),
    )

    expanded = _expand_grouped_term(
        _grouped_term_inference(curve),
        _identity_plus_pair_grouping(),
        {"1": 0.0, "2": 0.5, "3": 1.0},
    )

    # Today this raises KeyError('MISSING') at _term_helpers.py:196: the special
    # has no entry in _level_to_value and therefore none in original_level_values.
    np.testing.assert_allclose(expanded.smooth_curve.level_x, [0.0, 0.5, 1.0])
    np.testing.assert_array_equal(expanded.level_is_special, [False, False, False, True])


def test_specials_term_round_trips_through_a_level_collapse(specials_model):
    model, _, _ = specials_model
    session = EditorSession.from_model(model, terms=["band"])
    session.select_levels("band", ["2", "3"])

    refit = session.replace_with_collapsed_levels("band")

    assert refit._specs["band"]._specials == ["MISSING"]
    assert refit._specs["band"]._smooth_levels == ["1", "2+3", "4", "5", "6"]
    ti = refit.term_inference("band")
    assert ti.levels == ["1", "2", "3", "4", "5", "6", "MISSING"]
    np.testing.assert_array_equal(
        ti.level_is_special, [False, False, False, False, False, False, True]
    )


def test_collapsed_display_of_a_grouped_specials_term_keeps_level_x_smooth_only(specials_model):
    # This is the only place grouping and specials coexist, and "collapsed" is
    # the auto display default for OrderedCategorical (group_display.py:92), so
    # this is what the default plot path does. level_x covers the smoothed
    # levels only while the display groups span all K+S levels, so a naive
    # mean-position calculation indexes past the end of level_x.
    from superglm.plotting.group_display import project_grouped_term_for_display

    model, _, _ = specials_model
    session = EditorSession.from_model(model, terms=["band"])
    session.select_levels("band", ["2", "3"])
    refit = session.replace_with_collapsed_levels("band")

    display = project_grouped_term_for_display(refit, refit.term_inference("band"), "collapsed")

    assert display.collapsed
    assert display.term.levels == ["1", "2+3", "4", "5", "6", "MISSING"]
    np.testing.assert_array_equal(
        display.term.level_is_special, [False, False, False, False, False, True]
    )
    # One position per SMOOTH display group: the free level has none.
    smooth_count = int((~np.asarray(display.term.level_is_special)).sum())
    assert len(display.term.smooth_curve.level_x) == smooth_count == 5
```

- [ ] **Step 13: Run the tests to verify they fail**

Run: `uv run pytest tests/test_ordered_categorical_specials_editor.py -v -k "grouped or round_trips or collapsed_display"`
Expected: FAIL — `assert expanded.level_is_special is not None` fails for the first, and `KeyError: 'MISSING'` from `_term_helpers.py:196` for the other three (the round-trip and the display projection both reach it through `replace_in_force_model` → `term_inference`). `test_collapsed_display_of_a_grouped_specials_term_keeps_level_x_smooth_only` is the regression the plotting task's `_collapsed_smooth_curve` rewrite (Task 8 Step 7) exists for: with that rewrite reverted it fails instead with `IndexError` out of `level_x`.

- [ ] **Step 14: Rebuild the expanded term with `replace` and expand the marker**

This is the **only** task that edits `_expand_grouped_term`: the plotting task's grouped-expansion
step was dropped (nothing in it exercised grouped expansion, and a grouped specials term cannot
exist before this task's collapse work). So lines 188-259 still read exactly as they do on
`origin/master` — no insertions to reconcile, and no leftover locals for `ruff` F841 to catch at
Step 16.

In `src/superglm/inference/_term_helpers.py`, replace the curve block and the hand-listed constructor (lines 188-259) with:

```python
    # Expand smooth_curve: give each original level its own x-position and
    # rebuild the display curve via PCHIP interpolation through the expanded
    # (level_x, relativity) pairs so it passes through every marker.  level_x
    # covers the SMOOTHED levels only, so specials are held out of the rebuild
    # and keep their detached marker rows.
    expanded_special = (
        None
        if ti.level_is_special is None
        else np.asarray(ti.level_is_special, dtype=bool)[indices]
    )
    curve = ti.smooth_curve
    if curve is not None and curve.level_x is not None:
        from scipy.interpolate import PchipInterpolator

        smooth_mask = (
            np.ones(len(expanded_levels), dtype=bool)
            if expanded_special is None
            else ~expanded_special
        )
        smooth_levels = [lev for lev, keep in zip(expanded_levels, smooth_mask) if keep]
        smooth_log_rel = log_rel[smooth_mask]

        if original_level_values is not None:
            expanded_level_x = np.array([original_level_values[lev] for lev in smooth_levels])
        else:
            grouped_lx = np.asarray(curve.level_x)
            n_expanded = len(smooth_levels)
            expanded_level_x = (
                np.linspace(float(grouped_lx.min()), float(grouped_lx.max()), n_expanded)
                if n_expanded > 1
                else grouped_lx[np.asarray(indices, dtype=np.intp)[smooth_mask]]
            )

        # Rebuild display curve through expanded level positions
        # Deduplicate x-positions (grouped levels share the same relativity
        # but may have different x — keep first occurrence for interpolation)
        seen_x = {}
        for xi, yi in zip(expanded_level_x, smooth_log_rel):
            if xi not in seen_x:
                seen_x[xi] = yi
        uniq_x = np.array(sorted(seen_x.keys()))
        uniq_log_y = np.array([seen_x[x] for x in uniq_x])

        if len(uniq_x) >= 2:
            pchip = PchipInterpolator(uniq_x, uniq_log_y)
            new_x = np.linspace(float(uniq_x[0]), float(uniq_x[-1]), 200)
            new_log_rel = pchip(new_x)
            new_rel = np.exp(new_log_rel)
        else:
            new_x = curve.x
            new_log_rel = curve.log_relativity
            new_rel = curve.relativity

        curve = SmoothCurve(
            x=new_x,
            log_relativity=new_log_rel,
            relativity=new_rel,
            level_x=expanded_level_x,
            se_log_relativity=curve.se_log_relativity,
            ci_lower=curve.ci_lower,
            ci_upper=curve.ci_upper,
        )

    # dataclasses.replace, not a hand-listed rebuild: every field this function
    # does not touch — including level_is_special's siblings — survives by
    # construction rather than by remembering to list it.
    return replace(
        ti,
        levels=expanded_levels,
        log_relativity=log_rel,
        relativity=rel,
        se_log_relativity=se,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        smooth_curve=curve,
        level_is_special=expanded_special,
    )
```

`replace` is already imported at `_term_helpers.py:6` and used by `_recenter_term` (line 72).

- [ ] **Step 15: Run the tests**

Run: `uv run pytest tests/test_ordered_categorical_specials_editor.py tests/test_term_inference.py tests/test_summary_level_display.py tests/test_plot_api.py tests/test_relativities.py -q`
Expected: PASS (the grouped-expansion path for OC terms without specials keeps `expanded_special is None`, so `smooth_mask` is all-True and the curve rebuild is byte-for-byte the previous one). `tests/test_relativities.py` is in the list because `test_collapsed_display_of_a_grouped_specials_term_keeps_level_x_smooth_only` exercises the same collapsed-display path as `test_plotly_collapsed_ordered_categorical_suppresses_stale_knot_diagnostics` (`tests/test_relativities.py:602-655`), on a term without specials.

- [ ] **Step 16: Run the full gate and commit**

Run: `uv run pytest tests/ -q && uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/`
Expected: PASS

```bash
git add src/superglm/inference/_term_helpers.py tests/test_ordered_categorical_specials_editor.py
git commit -m "fix(inference): keep the special marker and smooth-only level_x through grouped expansion"
```

---

## Appendix: drafter concerns

> **Adjudicated 2026-08-05.** Every concern below has been reviewed against the
> real source and either applied to the task bodies above or rejected. The task
> bodies are authoritative wherever they disagree with this appendix — several
> entries describe a sequencing or a file state that the applied edits have
> since changed.

Each task section above was drafted against the real code by an agent that also
recorded where the code contradicted the spec. These are unresolved inputs to
review, not decisions. Three of them change the plan and are called out in the
review brief: the Excel Summary sheet is per-LEVEL not per-TERM; the positional
reader is report_ops.py:405 not ~563; and Task 4 leaves _term_ops raising
KeyError until the TermInference work lands.

## Prerequisite: both plot backends draw the fitted curve

- The spec says this commit "breaks tests asserting PCHIP-through-levels" (design doc line 214). I searched tests/ and found none. `grep -rn "PchipInterpolator|pchip|PCHIP" tests/` returns nothing; no test reads `ax.lines` or `ax.get_xticks()` for an OrderedCategorical matplotlib panel. The four near-misses are named in Step 9 with file:line and each survives for a stated reason. Plan for zero test edits, not for a sweep.
- I verified the whole task against a real (copied) checkout rather than by inspection: the four new tests fail on origin/master with the messages quoted in Step 2 and pass after the edits, and `tests/test_relativities.py tests/test_plot_api.py tests/test_categorical_ux.py tests/test_ordered_categorical.py tests/test_plot_diagnostics.py` (231 passed) plus `tests/test_editor.py tests/editor -m "not browser"` (250 passed) stay green. `ruff check` and `ruff format --check` pass on the edited files. A full-suite before/after failure diff was still running when I finished; treat Step 9 as the gate, not my word.
- `_expand_grouped_term` (src/superglm/inference/_term_helpers.py:186-232) still rebuilds a grouped term's `smooth_curve` by PCHIP through the expanded level positions. So for a *grouped* model, `ti.smooth_curve` is itself an interpolation, not the fitted spline — this task only guarantees the two backends draw the same thing, and that display-time collapsing does not fabricate a second one. Fixing the inference-level rebuild is a separate, larger change (it needs the grouped spec's own basis re-evaluated on the original level values) and the spec does not ask for it.
- The x-axis semantics of the matplotlib OC panel change from ordinal to metric. With `values={...}` giving unequal gaps (21, 30, 42, 57, 72) the level markers are no longer equispaced and long tick labels can now collide, since `rot = 45 if n_levels > 8 else 0` (main_effects.py:626) keys rotation off level count, not off label width at the new positions. Plotly has had this behaviour all along, so the risk is bounded, but it is a real visual change to review on a wide term.
- Task 7 (specials rendering) must revisit `_ordered_level_spacing`: once specials are parked past the last ordered level with a deliberate gap, the minimum-gap rule will still size bars from the ordered block, which is what you want, but `ax.set_xlim` will need the special positions folded in or the last special falls outside the axes.
- `_collapsed_smooth_curve`'s signature changes from `(ti, log_rel, n_levels)` to `(ti, groups)`. It is module-private with exactly one caller (group_display.py:70) — I grepped src/ and tests/ for it — but note it in review since the argument list changes shape rather than growing.

## Restrict the whole-smooth test and edf to the spline block

- The invariance test's tolerances are derived analytically, not measured — `specials=` does not exist on this branch yet, so I could not run it. The derivation: with an unpenalized indicator block, the Schur complement over the special coefficient turns the with-specials centered Gram into exactly the ordered-only centered Gram, so `V_b_j`, the spline edf and `X_j'X_j` are mathematically identical to the reference refit; Poisson keeps `known_scale=True` (`distributions.py:62`), so `res_df = -1` and no dispersion estimate can differ. The one term that is NOT exactly invariant is `edf1_j = 2*diag(F) - diag(F**2)` (`coef_tables.py:239`), because `F` uses the intercept-profiled centered Gram (`_metrics_design.py:290-294`) rather than a special-profiled one, leaving a cross term proportional to `w_special / (w_ordered * w_total)` — of order 1e-4 at the fixture's 6.5% special weight share. If Step 4 shows the post-fix values agreeing less tightly than the stated tolerances, widen `wald_chi2`/`ref_df` to `rel=5e-2` and `edf` to `rel=1e-2` before suspecting the fix: the defect being caught is ~25% on edf and orders of magnitude on chi2, so there is a four-order-of-magnitude safety window. Do NOT widen `n_params`, which is exact integer equality.
- `feature_active` (`coef_tables.py:342`) is deliberately left spanning ALL blocks, not just the spline block. It gates the level rows' standard errors at line 445 and the `wald_chi2`/`wald_p`/`ref_df` fields at 424-426. Restricting it would blank every level SE in the corner case where selection drops the spline group but keeps the (unpenalized, hence never-penalized-away) specials group. In that corner the smooth row would report `active=True` with `active_pairs` empty, so `wald_chi2` is NaN and `group_norm` 0.0 — the same shape the code already has for a deselected multi-group feature. The design spec does not say what selection should do to a specials term; if a later task gives an answer, this line is where it lands.
- The spec (§Block order is a contract) names `coef_tables.py:413` and `report_ops.py:~563` as the two positional `feature_groups[0]` readers to convert to subgroup selection. This task converts `coef_tables.py:412-415` and `report_ops.py:405` — note that 405, not 563, is the ordered-categorical positional read in the file as it stands at 7109e7f (line ~563 is in a different builder). If the two-block `build()` task has already converted `coef_tables.py:413`, Step 6's fourth edit is a no-op; check before applying rather than assuming a conflict.
- This task cannot start before the `specials=` constructor, the two-block `build()` and the `transform`/`reconstruct` widening have landed — the tests fit a real specials model and read `spec.reconstruct(beta_combined)` through the unchanged line 360. That matches the spec's implementation order (this is item 5, after 2-4).
- Not fixed here, and deliberately out of scope: `coef_tables.py:445` (`i < len(se_levels)`) still blanks the standard error on free levels, because `feature_se_from_cov` (line 351) returns one SE per smooth level until the read-back task widens it. The spec's risk table lists it as its own required test; a specials level will show a blank SE in the summary until then, and the tests in this task deliberately assert nothing about level-row SEs.

## Per-level fit marker through summary, HTML and the Excel Summary sheet

- The spec (line 186-191) and the task brief both state "the Summary sheet carries one row per *term*". That is not what the code does: `export/summary.py:301` iterates `summary._coef_rows`, so every OC *level* row is already exported as its own `SummaryTermRow` with `kind="level"` (classified by `_canonical_level_row_names`, `export/summary.py:228-242`, which reads `spec._ordered_levels` and therefore already picks up specials). `tests/test_rating_table_export.py:1341-1349` asserts exactly that. Per-level provenance on the Excel Summary sheet is therefore available for free by giving the special's level row `kind="free level"` instead of `"level"` — no new column, no rating-sheet change. I followed the brief and put the marker only on the term's group row, but the stated justification for that limit is factually wrong and the decision is worth revisiting.
- `_summary_notes` (`export/summary.py:360`) tests `row.kind == "smooth"` exactly. Any new kind string for a specials term silently removes the Wood (2013) note from the workbook. Step 23 widens it to `startswith("smooth")` and Step 21 asserts the note survives, but the same exact-equality pattern would bite any future kind value.
- `test_specials_workbook_keeps_summary_columns_and_rating_block_layout` calls `model.export_rating_tables(...)` end to end, so it depends on `TermInference` for a specials term already being buildable (`export/rating_tables.py:152-171` reads `ti.levels`/`ti.relativity`). If the plotting/`TermInference` task lands after this one, that single test will fail on a missing `x_position` or a `level_x` width mismatch (spec risk table, `plotting/data.py:126-130`), not on anything this task changed. Sequence this task after the `TermInference`/`level_is_special` work, or split that one test into the plotting task.
- `summary_levels.py` needs no code change: `dataclasses.replace` at lines 155 and 170 carries `level_fit` onto every display row automatically. The one gap is the synthesized reference row at `summary_levels.py:149-154`, which constructs a bare `_CoefRow` and would leave `level_fit=None`. That path only fires for a reference-only feature with no matched coefficient rows, which cannot happen for a spline OC (`coef_tables.py:442` emits a row for every level including the base), so it is safe today but is an unguarded assumption.
- The ASCII `Fit` column width is computed as `max(len("Fit"), max value length) + 2` so there is always a visible gap after a maximal-length term name. The pre-existing `Level group` column (`summary.py:352-356`) has no such padding and can abut the longest name; I deliberately did not change that behaviour, but the two columns will look inconsistent when both are present.

## Render special levels in both plotting backends

- The prerequisite commit rewrites `_plot_ordered_spline_panel` (main_effects.py:524-634) before this task lands. Step 18 is a full replacement of the function written from master's body plus the one stated prerequisite change (curve taken from `ti.smooth_curve` rather than a PCHIP through level relativities at integer positions, main_effects.py:583-596). If the prerequisite settled on different curve/marker styling or a different xlim rule, the replacement must be reconciled with it rather than pasted over it.
- `_term_ops.py:226` and `:252` build `level_x` from `raw["level_values"]` over every level, so they raise `KeyError: 'MISSING'` from the moment `reconstruct` starts returning specials in `raw["levels"]` — i.e. from the read-back task onward. The Summary-sheet/rating-table task also consumes `TermInference`, so it may have to make that same edit first. If it already did, merge Step 4 into the existing code; do not end up with two `smooth_levels` definitions or a re-widened `level_x`.
- Grouping combined with specials is fixed in `_expand_grouped_term` (Step 5) but is not covered by a test in this task: constructing a grouped OC that also has specials needs the collapse work from the editor task. That task must add the regression test (a collapse of ordered levels on a specials term, then `term_inference`, asserting `level_is_special` lines up with the expanded levels and `level_x` stays smooth-only).
- `_collapsed_smooth_curve` (group_display.py:162-184) still builds `level_x = np.arange(n_levels)` over every display level, which violates the new smooth-only invariant and would make `_level_positions_with_specials` raise. It is left untouched because the prerequisite removes the collapsed display as the OC auto default; whoever keeps collapsed display alive for OC must restrict that array to the ordered levels.
- `main_effects_plotly.py:1687-1696` keeps a private `_ordered_bar_width` that duplicates the spacing logic now in `plotting/common._ordered_level_step`. It is left alone to keep this diff tight; its min-positive-diff rule still gives the right width once the gap is present, since the gap is a whole number of steps.
- The spec's risk table lists `plotting/data.py:266-268` (basis-decomposition panel vanishing on a width mismatch) as a specials risk. It cannot fire for an OC term: `_main_effect_basis_dataframe` returns at data.py:249 whenever `ti.kind != "spline"`, and OC reports `kind="categorical"`. No change is made there. The concatenated-beta width bug at data.py:264 would only bite if OC were ever routed into that function.
- `model.plot(engine="plotly")` refuses fewer than two main effects (plot_ops.py:159-164), so the plotly test necessarily fits a second feature and reads `fig.layout.xaxis`, which carries the first term's axis config. This mirrors the existing convention in tests/test_plot_api.py:796-806.
- None of the new tests could be executed here: they depend on `specials=` existing on `OrderedCategorical` and on `reconstruct` returning special levels, neither of which is on the branch yet. The expected failure messages in Steps 2, 8, 12, 17 and 22 are derived from reading the code paths, not from observed runs.

## Screening deferral and interaction refusal

- Spec citation drift (mechanism verified, line numbers off): the design says the "purpose-built ValueError when the term is explicitly named as a candidate" lives at screening_ops.py:656-671, but 656-671 is `_raw_object`/the head of `_margin_source`. The error is actually raised in `_validated_pairs` at :302-310. Likewise the eager pre-read loop is at :697-705, not :703-712 (:707-712 is offset handling). The claims themselves check out: I confirmed empirically that a known level with no `_level_to_value` entry makes the whole sweep die with `ValueError: screen_interactions requires finite covariates; 'band' maps to non-finite scores`.
- The explicit-spec bypass is real but narrower than it reads: NO interaction class in features/interaction.py defines `.name`, and base.py:783-800 requires both `.parent_names` (a 2-tuple of str) and a non-empty str `.name`, so only `FactorSmooth` ships qualifying — and FactorSmooth is the one form that is deliberately exempt. I verified the bypass by setting `.name` manually on a `TensorInteraction` and on a `SplineCategorical`: both deep-copy into `_interaction_specs` and fit today with an OC parent, never touching `add_interaction`. The test uses that construction. If a reviewer objects to a test that assigns an attribute the class does not declare, the alternative is a small duck-typed stub, but the assigned-attribute form is what the probe found.
- `test_a_specials_term_may_still_be_a_factor_smooth_group` does NOT fail against unmodified code — it pins the exemption (the guard must go in `resolve_interaction_parent`, not `resolve_interaction_parent_of`). It is labelled as such in its docstring. I verified a FactorSmooth with a (non-specials) OC group main fits today, so the shape of the test is sound; whether it fits with a SPECIALS group main depends entirely on Tasks 2-4. If it fails, the defect is in the build, not here.
- Every screening test in this task fits a real model with `specials=`, so this task cannot be executed or verified before Tasks 2-4 land. There is no way to test the `_margin_kind` branch in isolation without them short of constructing a spec by hand.
- `_validated_pairs`'s fourth parameter changes meaning (`fitted_names: set[str]` -> `deferred_features: dict[str, str]`). I grepped src/ and tests/: the only caller is screening_ops.py:615 and no test calls it directly. The old predicate `name not in margin_kinds and name in set(model._specs)` and the new `name in deferred_features` differ only for a name in `_specs` but absent from `_feature_order`, which base.py:806-807 makes impossible (both are appended together).
- `_DEFERRED_KIND_HINT` (screening_ops.py:237-242) still says "Polynomial margins are likewise deferred" and remains in use for the pair-KIND branch at :321. After this change the per-NAME branch quotes `_deferral_reason` instead, so the Polynomial sentence in the shared hint is redundant on that path. Left alone deliberately — trimming it would change the spline x numeric message this task has no business touching.

## Editor: exact-assignment refit, collapse rules, spec clone

- Constructor grouping branch: `_ordered_spec_with_grouping` always clones with `grouping=` set, so the collapse round-trip only works if task 2's normalisation is applied in the `grouping is not None` branch of `OrderedCategorical.__init__` (ordered_categorical.py:246-272), which rebuilds `_ordered_levels` from `grouping.grouped_levels` and `_level_to_value` from `orig_ltv`. If task 2 only normalises the `order=`/`values=` path, `test_collapsing_ordered_levels_keeps_the_special_free` and the round-trip test fail in ordered_categorical.py, not in the editor. Needed there: `_smooth_levels = [lev for lev in grouping.grouped_levels if lev not in specials]`, `_ordered_levels = _smooth_levels + specials`, `_n_levels = len(_smooth_levels)` (so the knot clamp stays smooth-only), and no `_level_to_value` entry for a special (which falls out naturally, since `grouped_ltv` skips a group whose originals have no values).
- The exact-assignment convention and today's joint least-squares solve agree to machine precision whenever the ordered projection is OVERDETERMINED (more smoothed levels than spline columns + 1): the special rows are exactly fittable, contribute zero residual, and do not move the minimiser. They diverge only when the ordered system is underdetermined, which is why the fixture uses 6 smoothed levels against a 7-column `Spline(kind='ps', k=8)` block (verified on this worktree: 6 levels + k=8 builds a 7-column group). If the implementer changes `k` or the level count to something overdetermined, the two apply tests stop discriminating and become worthless — the shape is load-bearing, not decorative.
- The spec's editor section says the failure today is all-NaN coefficients from `_map_to_numeric`; that is true of `origin/master` but not of the state this task starts in. Once task 4 widens `transform`, the NaN path is already gone and the remaining defect is the min-norm split of the intercept against the special coefficients. The tests here therefore assert the split, not finiteness; a finiteness-only test would pass against the unfixed code.
- Grouping two specials *with each other* is not addressed by the spec's rule table (which only forbids mixing a special with ordered levels). `_require_no_special_members` refuses any group containing a special at all, because merging two specials would need the clone to replace both labels with the group label in `specials=` and would leave `_specials` inconsistent with the grouping. If a later reviewer wants special+special merges, that is a deliberate extension, not an oversight.
- `_expand_grouped_term`'s no-`original_level_values` fallback (`grouped_lx[indices]`, line 203) indexes `level_x` with grouped-level positions. That stays in range only because specials sit LAST in `_ordered_levels`, so every smoothed grouped level has a position below `len(level_x)`. If the display order ever puts specials anywhere but last, that branch silently mis-indexes. It is only reachable when fewer than two smoothed levels survive expansion.
- Editing individual original levels inside a collapsed group: `_apply_ordered_spline_term` (like `_apply_projected_term` before it) feeds one least-squares row per ORIGINAL level, while `_level_target_map` (apply.py:245-273) does the exposure-weighted group average used by the categorical and step paths. The two paths disagree when members of one group are edited differently. Pre-existing asymmetry on `origin/master`; unchanged here and untested by this task.
