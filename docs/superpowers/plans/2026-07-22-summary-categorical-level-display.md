# Categorical Level Display in Summaries Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make categorical levels expanded by default in every model-summary surface, provide an explicit compact grouped view with per-feature `G1`, `G2`, … indicators and membership legends, and leave the existing Excel export byte-contract unchanged.

**Architecture:** Canonical coefficient rows remain parameter-oriented and continue to feed inference and export. A new summary-only presentation adapter combines those rows with fitted categorical specs and `LevelGrouping` metadata, synthesizes reference rows, and returns display rows plus legends for `ModelSummary` and the editor. `level_display` is validated at the public boundary and participates in model-summary caching; editor state carries the same preference through every endpoint that returns summary evidence without changing model state.

**Tech Stack:** Python 3.10–3.14, dataclasses, NumPy, pytest, Ruff, FastAPI editor routes, browser-native JavaScript with `// @ts-check`, Node's test runner, TypeScript checking through `jsconfig.json`, CSS, Playwright, openpyxl, uv.

**Design:** [`docs/superpowers/specs/2026-07-22-summary-categorical-level-display-design.md`](../specs/2026-07-22-summary-categorical-level-display-design.md)

---

## File responsibility map

- `src/superglm/inference/summary_levels.py`: validate the display mode and adapt canonical categorical coefficient rows into summary-only rows and legends.
- `src/superglm/inference/summary.py`: retain canonical rows, render display rows in ASCII/HTML, add the conditional `Level group` column, reference formatting, wrapping, and escaping.
- `src/superglm/model/report_ops.py`: validate `level_display`, include it in the cache key, and build a presentation for normal and editor-stale summaries.
- `src/superglm/inference/metrics.py`: expose the same option and use the same adapter as `model.summary()`.
- `src/superglm/model/api.py`, `src/superglm/sklearn.py`: expose and forward the public option without changing existing defaults other than expanded categorical presentation.
- `src/superglm/editor/summaries.py`: produce typed compact rows and legends from `ModelSummary` display rows; stop independently reconstructing reference levels.
- `src/superglm/editor/widget.py`, `src/superglm/editor/server.py`: carry the requested presentation through summary, refit, profile, and structural-response paths.
- `src/superglm/editor/app/api/contracts.js`, `state/store.js`, `state/selectors.js`: type and store the view-only preference.
- `src/superglm/editor/app/index.html`, `main.js`: add an accessible Expanded/Grouped control and include the preference in summary-producing requests.
- `src/superglm/editor/app/summary.js`, `styles.css`: render the optional indicator column and wrapped membership legend in the compact editor summary.
- `tests/test_summary_level_display.py`: focused adapter, renderer, API, cache, ordered-categorical, and wrapper coverage.
- `tests/test_editor.py`, `tests/editor_frontend/summary.test.js`, `tests/editor_frontend/store.test.js`, `tests/editor/test_editor_workspace_browser.py`: editor payload, state, rendering, sequencing, keyboard, and no-mutation coverage.
- `tests/test_rating_table_export.py`: prove both summary modes leave the existing workbook contract and original-level rating rows unchanged.
- `docs/guide/results.md`: document expanded/grouped summary presentation and the Excel boundary.

## Invariants to preserve throughout

- `ModelSummary._coef_rows` remains the canonical list. Export code and existing internal consumers must never receive presentation-expanded rows.
- Original level strings are never rewritten. `G1` is carried in a separate field and column.
- `G1` numbering restarts per feature and follows the first member's position in `LevelGrouping.all_original_levels`.
- Only groups with at least two original members receive IDs or trigger the column.
- Reference rows have estimate `0`, standard error label `ref`, and no test or interval.
- Expanded duplicates share inference values but carry a feature's EDF only once.
- Unmatched canonical rows are preserved rather than discarded.
- The editor control updates view/evidence state only. It does not call collapse, ungroup, refit, history, or export actions.
- `src/superglm/export/` remains unmodified.

### Task 1: Build the categorical summary presentation adapter

**Files:**
- Create: `src/superglm/inference/summary_levels.py`
- Modify: `src/superglm/inference/summary.py:12-44`
- Create: `tests/test_summary_level_display.py`

- [ ] **Step 1: Add failing validation and deterministic-group tests**

Create `tests/test_summary_level_display.py` with a concrete categorical fixture and canonical rows. Use the real grouping/spec types so the test exercises the fitted metadata contract:

```python
import numpy as np
import pytest

from superglm.features import Categorical, OrderedCategorical
from superglm.features.grouping import collapse_levels
from superglm.inference.summary import _CoefRow
from superglm.inference.summary_levels import (
    build_summary_level_display,
    validate_level_display,
)
from superglm.types import GroupSlice


def _categorical_case():
    levels = ["A", "B", "C", "D", "E", "F"]
    grouping = collapse_levels(
        levels,
        groups={"B+C fitted label": ["B", "C"], "D+E fitted label": ["D", "E"]},
        order=levels,
    )
    spec = Categorical(base="A", grouping=grouping)
    spec.build(np.asarray(levels))
    rows = [
        _CoefRow(name="Intercept", coef=0.5),
        _CoefRow(
            name="territory[B+C fitted label]",
            group="territory",
            coef=0.2,
            se=0.05,
            z=4.0,
            p=0.001,
            ci_low=0.1,
            ci_high=0.3,
            edf=1.0,
        ),
        _CoefRow(
            name="territory[D+E fitted label]",
            group="territory",
            coef=-0.1,
            se=0.04,
            z=-2.5,
            p=0.012,
            ci_low=-0.18,
            ci_high=-0.02,
        ),
        _CoefRow(name="territory[F]", group="territory", coef=0.05, se=0.03),
    ]
    return spec, rows, [GroupSlice("territory", 0, 3)]


@pytest.mark.parametrize("value", ["", "expand", "ungrouped", "GROUPED", None])
def test_validate_level_display_rejects_unknown_values(value):
    with pytest.raises(ValueError, match=r"expanded.*grouped"):
        validate_level_display(value)


def test_expanded_display_uses_original_levels_and_deterministic_groups():
    spec, rows, groups = _categorical_case()
    display = build_summary_level_display(
        rows, specs={"territory": spec}, groups=groups, level_display="expanded"
    )

    territory = [row for row in display.rows if row.group == "territory"]
    assert [row.name for row in territory] == [
        "territory[A]",
        "territory[B]",
        "territory[C]",
        "territory[D]",
        "territory[E]",
        "territory[F]",
    ]
    assert [row.level_group for row in territory] == ["", "G1", "G1", "G2", "G2", ""]
    assert [row.edf for row in territory].count(1.0) == 1
    assert territory[0].is_reference is True
    assert (territory[0].coef, territory[0].se, territory[0].p) == (0.0, None, None)
    assert territory[1].coef == territory[2].coef == 0.2
```

- [ ] **Step 2: Run the focused file and confirm import failures**

```bash
rtk test uv run pytest tests/test_summary_level_display.py -q
```

Expected: collection fails because `superglm.inference.summary_levels` and the presentation-only row fields do not exist.

- [ ] **Step 3: Add presentation metadata fields without changing canonical construction**

Append defaults to `_CoefRow` in `inference/summary.py`:

```python
    # Summary presentation only; canonical coefficient builders leave these defaults untouched.
    level_group: str = ""
    is_reference: bool = False
```

Do not edit `inference/coef_tables.py`; canonical row creation must continue to populate the existing fields only.

- [ ] **Step 4: Define the adapter's typed contract and strict validator**

Create `inference/summary_levels.py` with these public-internal types:

```python
from dataclasses import dataclass, replace
from typing import Any, Literal, Mapping, Sequence

from superglm.inference.summary import _CoefRow
from superglm.types import GroupSlice

LevelDisplay = Literal["expanded", "grouped"]
_VALID_LEVEL_DISPLAYS = frozenset({"expanded", "grouped"})


@dataclass(frozen=True)
class LevelGroupLegend:
    feature: str
    group_id: str
    members: tuple[str, ...]


@dataclass(frozen=True)
class SummaryLevelDisplay:
    level_display: LevelDisplay
    rows: tuple[_CoefRow, ...]
    level_groups: tuple[LevelGroupLegend, ...]

    @property
    def has_level_groups(self) -> bool:
        return bool(self.level_groups)


def validate_level_display(value: object) -> LevelDisplay:
    if not isinstance(value, str) or value not in _VALID_LEVEL_DISPLAYS:
        raise ValueError(
            f"level_display={value!r} is not valid. "
            f"Expected one of {sorted(_VALID_LEVEL_DISPLAYS)}."
        )
    return value  # type: ignore[return-value]
```

If Ruff rejects the narrow return, replace the ignore with `typing.cast(LevelDisplay, value)`.

- [ ] **Step 5: Implement per-feature grouping metadata**

Add a private helper that returns original levels, fitted levels, the reference fitted level, and IDs. The key loop must be order-driven, not dict-order-driven:

```python
grouped_members: dict[str, list[str]] = {}
for original in grouping.all_original_levels:
    fitted = grouping.original_to_group[original]
    grouped_members.setdefault(fitted, []).append(original)

group_ids = {
    fitted: f"G{index}"
    for index, (fitted, members) in enumerate(
        ((name, members) for name, members in grouped_members.items() if len(members) > 1),
        start=1,
    )
}
```

For a spec with no `_grouping`, derive an identity mapping from `Categorical._levels` or `OrderedCategorical._ordered_levels`. Use the relevant `GroupSlice.name` as the categorical term prefix and the feature name for ordered categorical rows.

- [ ] **Step 6: Implement expanded replacement and reference synthesis**

Build a canonical lookup with exact expected names (`f"{term_prefix}[{fitted_level}]"`) rather than parsing bracket contents. For each original level, clone its fitted source with `dataclasses.replace`:

```python
display_row = replace(
    source,
    name=f"{term_prefix}[{original_level}]",
    level_group=group_ids.get(fitted_level, ""),
    is_reference=fitted_level == base_level,
    coef=0.0 if fitted_level == base_level else source.coef,
    se=None if fitted_level == base_level else source.se,
    z=None if fitted_level == base_level else source.z,
    p=None if fitted_level == base_level else source.p,
    ci_low=None if fitted_level == base_level else source.ci_low,
    ci_high=None if fitted_level == base_level else source.ci_high,
    edf=source.edf if id(source) not in edf_emitted else None,
)
```

Use a synthesized `_CoefRow(name=f"{term_prefix}[{original_level}]", group=term_prefix, coef=0.0, active=True, is_reference=True)` when treatment coding omitted the fitted reference. Track EDF by the source row's object identity so only the first expanded clone retains it. Preserve every canonical row that was not consumed by an exact fitted-level match.

Immediately after appending a clone whose source has non-`None` EDF, add `id(source)` to `edf_emitted`; reference rows and clones of rows with `edf=None` do not alter the set.

- [ ] **Step 7: Implement grouped replacement and legends**

For grouped mode, emit one row per fitted level in fitted order. Multi-member rows use the feature/term name plus the separate ID; singletons keep their exact level name:

```python
row_name = (
    term_prefix
    if fitted_level in group_ids
    else f"{term_prefix}[{members[0]}]"
)
```

Create one `LevelGroupLegend(feature, group_id, tuple(members))` for each multi-member group. A collapsed reference group uses the same compact name and ID but reference inference cells. Leave ordered-spline whole-feature rows in place before their adapted level rows.

- [ ] **Step 8: Add ordered, identity, fallback, and per-feature-reset tests**

Extend the focused test file to cover:

```python
def test_grouped_display_uses_one_row_per_fitted_group_and_legends():
    spec, rows, groups = _categorical_case()
    display = build_summary_level_display(
        rows, specs={"territory": spec}, groups=groups, level_display="grouped"
    )
    territory = [row for row in display.rows if row.group == "territory"]
    assert [(row.name, row.level_group) for row in territory] == [
        ("territory[A]", ""),
        ("territory", "G1"),
        ("territory", "G2"),
        ("territory[F]", ""),
    ]
    assert [(item.feature, item.group_id, item.members) for item in display.level_groups] == [
        ("territory", "G1", ("B", "C")),
        ("territory", "G2", ("D", "E")),
    ]
```

Add a second categorical feature and assert it starts again at `G1`; add an identity grouping and assert `has_level_groups is False`; add an unmatched canonical row and assert object content survives. Build `OrderedCategorical(order=["A", "B", "C"], basis="step", base="A", grouping=grouping)` for step coverage, then use a spline-backed ordered spec with a whole-feature row followed by level rows.

- [ ] **Step 9: Run adapter validation**

```bash
rtk test uv run pytest tests/test_summary_level_display.py -q
rtk ruff check src/superglm/inference/summary.py src/superglm/inference/summary_levels.py tests/test_summary_level_display.py
```

Expected: all focused tests pass and Ruff reports no errors.

- [ ] **Step 10: Commit the adapter**

```bash
rtk git add src/superglm/inference/summary.py src/superglm/inference/summary_levels.py tests/test_summary_level_display.py
rtk git commit -m "Add categorical summary display adapter"
```

### Task 2: Render display rows in ASCII and HTML

**Files:**
- Modify: `src/superglm/inference/summary.py:146-886`
- Modify: `tests/test_summary_level_display.py`

- [ ] **Step 1: Add failing renderer tests**

Construct `ModelSummary` with a presentation from Task 1 and assert both renderers:

```python
def test_ascii_and_html_render_expanded_groups_separately(summary_case):
    summary = summary_case(level_display="expanded")
    text = str(summary)
    html = summary._repr_html_()

    assert "Level group" in text
    assert "territory[B]" in text and "territory[C]" in text
    assert "B+C fitted label" not in text
    assert "Level group" in html
    assert "territory[B]" in html and "territory[C]" in html


def test_grouped_renderer_uses_short_rows_and_wrapped_membership_legend(summary_case):
    summary = summary_case(level_display="grouped")
    text = str(summary)
    assert "territory[B+C fitted label]" not in text
    assert "Level groups (territory):" in text
    assert "G1 = B, C" in text
```

Also assert reference rows include `0.0000` and `ref`, ordinary ungrouped summaries omit the new column, and a level such as `"<script>alert(1)</script>"` is escaped in HTML.

- [ ] **Step 2: Run the renderer tests and confirm failures**

```bash
rtk test uv run pytest tests/test_summary_level_display.py -q
```

Expected: display rows are not accepted or used yet; reference formatting, column, legends, and HTML escaping assertions fail.

- [ ] **Step 3: Extend `ModelSummary` without breaking canonical consumers**

Add an optional presentation argument at the end of the constructor:

```python
def __init__(
    self,
    data: dict[str, Any],
    model_info: dict[str, Any],
    coef_rows: list[_CoefRow],
    alpha: float = 0.05,
    detail: str = "compact",
    basis_detail: dict[str, list[_BasisDetailRow]] | None = None,
    level_presentation: SummaryLevelDisplay | None = None,
):
    self._coef_rows = coef_rows
    self._display_rows = (
        list(level_presentation.rows) if level_presentation is not None else list(coef_rows)
    )
    self._level_display = (
        level_presentation.level_display if level_presentation is not None else "expanded"
    )
    self._level_groups = (
        level_presentation.level_groups if level_presentation is not None else ()
    )
```

Use `TYPE_CHECKING` for the adapter type to avoid a module cycle. Keep EDF breakdown, smooth detection, basis disclosure, and general fit metadata based on `_coef_rows`; use `_display_rows` for coefficient rendering and row-specific warnings.

- [ ] **Step 4: Add the optional ASCII column and reference branch**

Compute `has_level_groups = bool(self._level_groups)` and a fixed-small ID width from display rows. Add that width only when true, so an ordinary model's table width is unchanged. Use one prefix helper for spline, basis, coefficient, and fallback rows:

```python
def _display_prefix(row: _CoefRow) -> str:
    prefix = f"{row.name:<{name_w}s}"
    if has_level_groups:
        prefix += f"{row.level_group:>{level_group_w}s}"
    return prefix
```

Place `Level group` immediately after the term name. Handle `row.is_reference` before the generic missing-SE branch and print estimate `0.0000`, SE `ref`, and blanks/`---` for statistic, p-value, and interval.

- [ ] **Step 5: Add grouped ASCII legends without affecting table width**

After the boxed coefficient table, group legend records by feature and wrap each line independently with `textwrap.wrap`:

```python
mapping = "; ".join(
    f"{item.group_id} = {', '.join(item.members)}" for item in feature_groups
)
legend = f"Level groups ({feature}): {mapping}"
lines.extend(textwrap.wrap(legend, width=max(60, min(W + 2, 100)), subsequent_indent="  "))
```

Render legends only for `level_display == "grouped"`. Never include legend member length in `name_w`, `coef_W`, or `W`.

- [ ] **Step 6: Add the optional HTML column and reference branch**

Set `ncols = 10 if has_level_groups else 9`. Insert a `Level group` header and corresponding cell in all row shapes, including spline rows, basis details, regular coefficient rows, reference rows, separators, and note colspans. Escape row names and group separator labels with `html.escape(row.name, quote=True)` and `html.escape(row.group, quote=True)`.

- [ ] **Step 7: Add safe, wrapping HTML legends**

Render grouped legends in a full-width row beneath the coefficient rows:

```python
parts.append(
    f'<tr><td colspan="{ncols}" style="padding:4px 8px;white-space:normal;'
    f'overflow-wrap:anywhere;">'
    f'<strong>Level groups ({escape(feature)}):</strong> {mapping_html}</td></tr>'
)
```

Escape feature names, group IDs, and every member separately. Preserve intentional internal markup such as `&lambda;`, `&chi;`, and `<br>` by escaping only user-derived strings.

- [ ] **Step 8: Verify canonical compatibility and width behavior**

Add assertions that:

- `summary._coef_rows` is the original canonical list;
- `summary._display_rows` contains exact expanded names;
- a long synthetic fitted label is absent from grouped `str(summary)` and does not determine the top-border width;
- the same long membership appears in wrapped legend lines;
- quasi-separation details use adapted original names;
- full spline basis disclosure still has the correct dynamic colspan.

- [ ] **Step 9: Run renderer and existing summary regression tests**

```bash
rtk test uv run pytest tests/test_summary_level_display.py tests/test_metrics.py -q
rtk ruff check src/superglm/inference/summary.py tests/test_summary_level_display.py
```

Expected: all tests pass; ordinary summary snapshots/shape assertions remain unchanged because the new column is conditional.

- [ ] **Step 10: Commit renderer integration**

```bash
rtk git add src/superglm/inference/summary.py tests/test_summary_level_display.py
rtk git commit -m "Render categorical level groups in summaries"
```

### Task 3: Expose `level_display` through model, metrics, cache, and sklearn APIs

**Files:**
- Modify: `src/superglm/model/report_ops.py:65-300`
- Modify: `src/superglm/inference/metrics.py:1214-1320`
- Modify: `src/superglm/model/api.py:711-714`
- Modify: `src/superglm/sklearn.py:695-698,844-847`
- Modify: `tests/test_summary_level_display.py`
- Modify: `tests/test_metrics.py:2217-2405`
- Modify: `tests/test_sklearn.py`

- [ ] **Step 1: Add failing public-API and cache tests**

Fit a small model with a collapsed categorical group and add:

```python
def test_model_summary_defaults_to_expanded_and_caches_modes_separately(grouped_model):
    expanded = grouped_model.summary()
    grouped = grouped_model.summary(level_display="grouped")

    assert expanded._level_display == "expanded"
    assert grouped._level_display == "grouped"
    assert grouped_model.summary() is expanded
    assert grouped_model.summary(level_display="grouped") is grouped
    assert expanded is not grouped


def test_model_and_metrics_summary_share_level_display(grouped_model, grouped_data):
    X, y, weight = grouped_data
    model_rows = grouped_model.summary(level_display="grouped")._display_rows
    metric_rows = grouped_model.metrics(X, y, sample_weight=weight).summary(
        level_display="grouped"
    )._display_rows
    assert [(row.name, row.level_group) for row in metric_rows] == [
        (row.name, row.level_group) for row in model_rows
    ]
```

Parametrize invalid values across `model.summary()` and `metrics.summary()`. For the model cache, seed or warm a valid cache first, then prove invalid values still raise rather than returning cached content.

- [ ] **Step 2: Add failing sklearn-forwarding tests**

For both fitted `SuperGLMRegressor` and `SuperGLMClassifier`, call:

```python
summary = estimator.summary(detail="full", level_display="grouped")
assert summary._detail == "full"
assert summary._level_display == "grouped"
```

This also locks in wrapper support for the core `detail` option instead of silently exposing a narrower summary API.

- [ ] **Step 3: Run the new API tests and confirm signature failures**

```bash
rtk test uv run pytest tests/test_summary_level_display.py tests/test_sklearn.py -q
```

Expected: unexpected-keyword failures for `level_display` and `detail` on current entry points.

- [ ] **Step 4: Validate before the model-summary cache and split cache entries**

Change `report_ops.summary` to:

```python
def summary(
    model,
    alpha: float = 0.05,
    detail: str = "compact",
    level_display: str = "expanded",
):
    from superglm.inference.summary_levels import (
        build_summary_level_display,
        validate_level_display,
    )

    level_display = validate_level_display(level_display)
    cache = model._summary_cache
    if cache is None:
        cache = {}
        model._summary_cache = cache
    tw_pr = getattr(model, "_tweedie_profile_result", None)
    tweedie_identity = (
        None if tw_pr is None else tweedie_profile_report_identity(tw_pr, alpha)
    )
    key = (float(alpha), detail, level_display, tweedie_identity)
```

Validation must happen before `_summary_cache` lookup. Build and pass `level_presentation` after both coefficient-row paths: normal inference and `_build_editor_stale_coef_rows` after `_suppress_editor_inference` has run.

- [ ] **Step 5: Adapt both model-summary construction sites**

Use exactly the same helper call in normal and stale branches:

```python
level_presentation = build_summary_level_display(
    coef_rows,
    specs=model._specs,
    groups=model._groups,
    level_display=level_display,
)
summary_obj = ModelSummary(
    data,
    model_info,
    coef_rows,
    alpha=alpha,
    detail=detail,
    basis_detail=basis_detail,
    level_presentation=level_presentation,
)
```

Do not pass `_interaction_specs`; categorical interaction rows remain canonical and pass through.

- [ ] **Step 6: Adapt `ModelMetrics.summary`**

Add the parameter and its accepted values to the docstring. Validate at method entry, build the presentation from `self._model._specs` and `self._groups`, and pass it to `ModelSummary`. Keep `_build_coef_rows` canonical.

- [ ] **Step 7: Forward through core and sklearn APIs**

Use these signatures:

```python
# model/api.py
def summary(
    self,
    alpha: float = 0.05,
    detail: str = "compact",
    level_display: str = "expanded",
):
    return report_ops.summary(self, alpha, detail=detail, level_display=level_display)

# both sklearn wrappers
def summary(
    self,
    alpha: float = 0.05,
    detail: str = "compact",
    level_display: str = "expanded",
):
    check_is_fitted(self)
    return self._model.summary(alpha=alpha, detail=detail, level_display=level_display)
```

Expand the docstrings enough that `expanded` default, grouped compact behavior, and strict accepted values are visible in generated API documentation.

- [ ] **Step 8: Add stale-inference and ordinary-model regressions**

Use the existing editor-stale fixture/path to assert expanded duplicates all keep `se`, `p`, and intervals suppressed, and grouped mode cannot restore them. Fit an ungrouped categorical model and assert it gains its reference row but no `Level group` column.

- [ ] **Step 9: Run the public API suite**

```bash
rtk test uv run pytest tests/test_summary_level_display.py tests/test_metrics.py tests/test_sklearn.py tests/test_ordered_categorical_inference.py -q
rtk ruff check src/superglm/model/report_ops.py src/superglm/inference/metrics.py src/superglm/model/api.py src/superglm/sklearn.py tests/test_summary_level_display.py tests/test_metrics.py tests/test_sklearn.py
```

Expected: all tests pass; the two modes use distinct cached objects and model/metrics rows agree.

- [ ] **Step 10: Commit the public API**

```bash
rtk git add src/superglm/model/report_ops.py src/superglm/inference/metrics.py src/superglm/model/api.py src/superglm/sklearn.py tests/test_summary_level_display.py tests/test_metrics.py tests/test_sklearn.py
rtk git commit -m "Expose categorical summary display modes"
```

### Task 4: Make editor summary payloads consume the shared presentation

**Files:**
- Modify: `src/superglm/editor/summaries.py:15-275`
- Modify: `src/superglm/editor/widget.py:476-527`
- Modify: `src/superglm/editor/server.py:132-143`
- Modify: `tests/test_editor.py:1837-1948,2565-2665,4670-4720`

- [ ] **Step 1: Rewrite the collapsed-reference test expectations first**

Update `test_compact_summary_shows_collapsed_reference_level_group` so the default compact payload expects original member rows with separate IDs:

```python
rows = {row["name"]: row for row in payload["compact"]["rows"]}
for level in ["T03", "T04", "T05"]:
    assert rows[f"territory[{level}]"]["kind"] == "reference"
    assert rows[f"territory[{level}]"]["level_group"] == "G1"
    assert rows[f"territory[{level}]"]["coef"] == 0.0
    assert rows[f"territory[{level}]"]["se_label"] == "ref"

assert payload["compact"]["has_level_groups"] is True
assert payload["compact"]["level_display"] == "expanded"
```

Call `summary_payload(widget, "in_force", level_display="grouped")` in the same fixture and assert one `G1` reference row plus the exact `T03`, `T04`, `T05` membership legend. Assert the non-reference collapsed group repeats inference in expanded mode.

- [ ] **Step 2: Add typed-row, legacy-fallback, and endpoint-forwarding failures**

Add assertions for `level_group`, `kind="reference"`, `se_label="ref"`, `level_groups`, and top-level `level_display`. Preserve the existing compact test built with `SimpleNamespace(_coef_rows=[_CoefRow(name="Intercept", coef=0.5)])` by asserting it still defaults cleanly when `_display_rows` is absent. Add a `/summary` route test that posts `{"source": "in_force", "level_display": "grouped"}` and captures the value received by `widget._summary`.

- [ ] **Step 3: Run the focused editor tests and confirm failures**

```bash
rtk test uv run pytest tests/test_editor.py -q -k "compact_summary or summary_endpoint"
```

Expected: payload shape and forwarding assertions fail because the editor still reads canonical rows and reconstructs only one reference label.

- [ ] **Step 4: Add `level_display` to `summary_payload` and use display rows**

Validate the value at the top of `summary_payload`, call `model.summary(level_display=level_display)`, then use:

```python
display_rows = getattr(summary, "_display_rows", getattr(summary, "_coef_rows", []))
rows = [_compact_summary_row(row) for row in display_rows]
```

Remove `_with_reference_rows` and `_compact_reference_row`; the shared adapter is now the single authority for ordinary and collapsed references. Remove the unused `model` argument from `_compact_summary_payload` after updating direct tests.

- [ ] **Step 5: Serialize presentation metadata explicitly**

Add this compact contract:

```python
"level_display": getattr(summary, "_level_display", "expanded"),
"has_level_groups": bool(getattr(summary, "_level_groups", ())),
"level_groups": [
    {
        "feature": item.feature,
        "group_id": item.group_id,
        "members": list(item.members),
    }
    for item in getattr(summary, "_level_groups", ())
],
```

Add the same `level_display` at the top-level response so frontend sequencing can identify the rendered variant without inspecting table rows.

Include that validated field in early `available=False` responses as well, so an unavailable refit and a successful summary have the same variant identity.

- [ ] **Step 6: Serialize reference and group cells from each row**

In `_compact_summary_row`, set:

```python
is_reference = bool(getattr(row, "is_reference", False))
kind = "reference" if is_reference else ("spline" if row.is_spline else "coef")
```

Return `level_group`, use `se_label="ref"` for references, and force reference statistic/p-value/EDF to `None`. Continue using the canonical stale values already suppressed by the model path; do not recompute inference in the editor.

- [ ] **Step 7: Thread the option through `/summary` only**

Add `level_display` to `widget._summary` and its `summary_payload` call. Parse it in the route with default `expanded`:

```python
level_display=str(payload.get("level_display", "expanded"))
```

Leave model revision and request sequence logic unchanged.

- [ ] **Step 8: Run editor backend validation**

```bash
rtk test uv run pytest tests/test_editor.py -q -k "summary"
rtk ruff check src/superglm/editor/summaries.py src/superglm/editor/widget.py src/superglm/editor/server.py tests/test_editor.py
```

Expected: compact payload tests pass, including grouped references and legacy fallback.

- [ ] **Step 9: Commit shared editor payloads**

```bash
rtk git add src/superglm/editor/summaries.py src/superglm/editor/widget.py src/superglm/editor/server.py tests/test_editor.py
rtk git commit -m "Use shared categorical rows in editor summaries"
```

### Task 5: Preserve the selected mode across every summary-producing editor response

**Files:**
- Modify: `src/superglm/editor/widget.py:751-900,909-990`
- Modify: `src/superglm/editor/server.py:208-266`
- Modify: `tests/test_editor.py:3280-3555,4670-4720`

- [ ] **Step 1: Add failing propagation tests for auxiliary responses**

Parametrize the summary-producing paths:

- fixed-offset refit;
- synchronous distribution profile;
- background distribution profile start/final result;
- collapse levels;
- ungroup levels;
- restore collapsed levels.

Monkeypatch `summary_payload` with a keyword-aware recorder and assert each path passes `level_display="grouped"`. For profile paths, separately assert `level_display` is not forwarded into the numerical `reprofile_distribution` options.

- [ ] **Step 2: Run the propagation subset and confirm failures**

```bash
rtk test uv run pytest tests/test_editor.py -q -k "level_display and (refit or profile or collapse or ungroup)"
```

Expected: current widget signatures reject or drop `level_display`.

- [ ] **Step 3: Add keyword defaults to widget response builders**

Use `level_display: str = "expanded"` in `_refit_offset`, `_profile_distribution`, `_start_profile_distribution_job`, `_collapse_levels`, `_ungroup_levels`, `_uncollapse_levels`, and `_transition_envelope`. Validate only at `summary_payload`; numerical code receives no presentation option.

- [ ] **Step 4: Carry the mode into transition envelopes**

Change `_transition_envelope` to call:

```python
summary = jsonable(summary_payload(self, "in_force", level_display=level_display))
```

Pass the keyword from each structural operation after its fit succeeds. This changes response formatting only; do not store the option on the model, session, edit history, or transition state.

- [ ] **Step 5: Parse the field separately from profile options**

For `/profile_distribution` and `/profile_distribution/start`, pass:

```python
level_display=str(payload.get("level_display", "expanded")),
**_profile_options(payload),
```

Do not add `level_display` to `_profile_options`. Parse the same field for `/refit_offset`, `/collapse_levels`, `/ungroup_levels`, and the currently bodyless `/uncollapse_levels`; change the last route to accept `Body(default_factory=dict)`.

- [ ] **Step 6: Update existing monkeypatch fakes for keyword compatibility**

Existing tests around widget lines 3282, 3402, 3460, 4676, and 4703 replace `summary_payload`. Update each fake to accept `**kwargs` or an explicit `level_display` and assert it where relevant. Do not weaken existing blocking/sequencing behavior.

- [ ] **Step 7: Run all editor backend tests**

```bash
rtk test uv run pytest tests/test_editor.py -q
rtk ruff check src/superglm/editor/widget.py src/superglm/editor/server.py tests/test_editor.py
```

Expected: the complete editor backend suite passes and every response embeds a matching summary variant.

- [ ] **Step 8: Commit response propagation**

```bash
rtk git add src/superglm/editor/widget.py src/superglm/editor/server.py tests/test_editor.py
rtk git commit -m "Propagate editor summary display preference"
```

### Task 6: Add view state and the accessible editor control

**Files:**
- Modify: `src/superglm/editor/app/api/contracts.js:1-115`
- Modify: `src/superglm/editor/app/state/store.js:12-40`
- Modify: `src/superglm/editor/app/state/selectors.js:1-70`
- Modify: `src/superglm/editor/app/index.html:298-313`
- Modify: `src/superglm/editor/app/main.js:1-40,107-155,343-510,515-565,1375-1465`
- Modify: `tests/editor_frontend/store.test.js`
- Modify: `tests/test_editor.py:6320-6460`

- [ ] **Step 1: Add failing state tests**

In `store.test.js`, assert:

```javascript
const state = createInitialEditorState(snapshot());
assert.equal(state.view.summaryLevelDisplay, "expanded");

const grouped = patchView(state, { summaryLevelDisplay: "grouped" });
assert.equal(grouped.view.summaryLevelDisplay, "grouped");
assert.equal(grouped.remote.snapshot.model_revision, state.remote.snapshot.model_revision);
assert.deepEqual(grouped.remote.snapshot.history, state.remote.snapshot.history);
```

Add a selector assertion for `selectSummaryLevelDisplay`.

- [ ] **Step 2: Add a failing static UI contract test**

In `tests/test_editor.py`, assert the editor HTML contains a control labelled `Categorical levels`, two native radio inputs named `summaryLevelDisplay`, and labels `Expanded`/`Grouped`; assert it does not call the structural ungroup action from the control's binding.

- [ ] **Step 3: Run state/UI tests and confirm failures**

```bash
rtk test node --test tests/editor_frontend/store.test.js
rtk test uv run pytest tests/test_editor.py -q -k "workspace_shell or categorical_levels_control"
```

Expected: the state property, selector, and markup are absent.

- [ ] **Step 4: Type and initialize the preference**

Add:

```javascript
/** @typedef {'expanded'|'grouped'} SummaryLevelDisplay */
```

to `api/contracts.js`, add `@property {SummaryLevelDisplay} summaryLevelDisplay` to `EditorViewState`, initialize it to `"expanded"`, and export:

```javascript
export const selectSummaryLevelDisplay = (state) => state.view.summaryLevelDisplay;
```

- [ ] **Step 5: Add a native-radio segmented control**

Inside `.summary-controls`, add a fieldset so keyboard and accessible-name behavior come from the platform:

```html
<fieldset id="summaryLevelDisplay" class="summary-level-display">
  <legend>Categorical levels</legend>
  <div class="summary-level-segments">
    <label><input type="radio" name="summaryLevelDisplay" value="expanded" checked>
      <span>Expanded</span></label>
    <label><input type="radio" name="summaryLevelDisplay" value="grouped">
      <span>Grouped</span></label>
  </div>
</fieldset>
```

Keep the structural action named `Ungroup levels` elsewhere; never use `Ungrouped` for this view control.

- [ ] **Step 6: Bind the control to view state only**

In `main.js`, collect the radio nodes, subscribe them to `selectSummaryLevelDisplay`, and on `change` call only:

```javascript
actions.patchView({ summaryLevelDisplay: input.value });
void refreshSummaryView();
```

Do not invoke `runStructuralRefit`, edit actions, chart grouping, history actions, or model initialization.

- [ ] **Step 7: Centralize the evidence request payload**

Add:

```javascript
function summaryRequestPayload() {
  return {
    source: summarySource.value,
    level_display: selectSummaryLevelDisplay(store.getState())
  };
}
```

Use it from `refreshSummaryView` and scheduled summary evidence. Add the display value to `summaryNodes()` so refit/profile helpers can include it. In `runStructuralRefit`, merge `level_display` into the descriptor payload immediately before `executeStructuralMutation`; the descriptor helpers themselves remain model-action descriptions and stay unchanged.

- [ ] **Step 8: Guard against a mismatched retained payload**

In `renderSummaryEvidence`, accept a payload only when its `level_display` (defaulting to `expanded` for legacy payloads) matches current view state. When a retained payload is for the other mode, clear it and show the existing updating state until the matching response arrives. Existing request sequence/revision checks remain authoritative.

Because `summary.js` memoizes rendered markup in `summaryMarkupByFrame`, make `updateSummaryMarkup` compare both the memoized string and `summaryFrame.innerHTML` before returning. That ensures a rapid Expanded → Grouped → Expanded sequence can restore identical earlier markup after main.js temporarily clears a mismatched view.

- [ ] **Step 9: Run state and static validation**

```bash
rtk test node --test tests/editor_frontend/store.test.js
rtk test uv run pytest tests/test_editor.py -q -k "workspace_shell or summary"
rtk npm run typecheck:frontend
```

Expected: state, markup, and type checks pass; no model revision field is touched by `patchView`.

- [ ] **Step 10: Commit the editor view control**

```bash
rtk git add src/superglm/editor/app/api/contracts.js src/superglm/editor/app/state/store.js src/superglm/editor/app/state/selectors.js src/superglm/editor/app/index.html src/superglm/editor/app/main.js tests/editor_frontend/store.test.js tests/test_editor.py
rtk git commit -m "Add editor summary level display control"
```

### Task 7: Render group indicators and legends in the compact editor summary

**Files:**
- Modify: `src/superglm/editor/app/summary.js:1-30,56-100,593-730`
- Modify: `src/superglm/editor/app/styles.css:293-470`
- Modify: `tests/editor_frontend/summary.test.js`

- [ ] **Step 1: Add failing compact-renderer tests**

Create a payload containing expanded rows and group metadata, render through `renderSummary`, and assert:

```javascript
assert.match(nodes.summaryFrame.innerHTML, /Level group/);
assert.match(nodes.summaryFrame.innerHTML, /territory\[B\]/);
assert.match(nodes.summaryFrame.innerHTML, />G1</);
assert.doesNotMatch(nodes.summaryFrame.innerHTML, /Level groups \(territory\)/);
```

For `level_display: "grouped"`, assert one group row and `Level groups (territory): G1 = B, C`. Add malicious members such as `<img src=x onerror=alert(1)>` and assert only escaped text appears. Add an ordinary payload and assert the table remains six columns with no empty group column.

- [ ] **Step 2: Add failing request-body tests for direct summary helpers**

Inject/capture `requestJSON` using the existing test seam and assert `refreshSummary`, `runOffsetRefit`, and `runDistributionProfile` send `level_display: "grouped"` when `nodes.summaryLevelDisplay` is grouped; missing node data defaults to expanded for backward-compatible tests.

- [ ] **Step 3: Run frontend summary tests and confirm failures**

```bash
rtk test node --test tests/editor_frontend/summary.test.js
```

Expected: no group column/legend exists and summary-producing request bodies omit the mode.

- [ ] **Step 4: Make the compact table column count data-driven**

Use `compact.has_level_groups === true` to calculate six or seven columns. Pass the value through `renderSummaryRows` and `renderSummaryRow`, insert this cell immediately after Term when enabled:

```javascript
const levelGroupCell = hasLevelGroups
  ? `<td class="summary-level-group">${escapeHTML(row.level_group || "")}</td>`
  : "";
```

Use the same dynamic count for `.summary-group-row` colspans.

- [ ] **Step 5: Render grouped membership as escaped semantic markup**

Group `compact.level_groups` by feature and render only when `compact.level_display === "grouped"`. Use a section or definition list with `aria-label="Level group membership"`, visible `Level groups (<feature>):`, and wrapping member text. Escape feature, ID, and each member separately.

- [ ] **Step 6: Keep compact and Full summary synchronized**

Continue sourcing the iframe `srcdoc` from the same response's `payload.html`. Do not add client-side transformation to the full summary; matching is guaranteed by the backend `level_display` and the main.js payload guard.

- [ ] **Step 7: Include the mode in direct helper requests**

Add a small helper:

```javascript
function requestedLevelDisplay(nodes) {
  return nodes.summaryLevelDisplay === "grouped" ? "grouped" : "expanded";
}
```

Use it in `/summary`, `/refit_offset`, and `/profile_distribution/start` request bodies. The profile polling URL needs no field because the started job retains the chosen response mode.

- [ ] **Step 8: Style the segmented control, column, and wrapping legend**

Add styles that keep the fieldset compact and focus-visible, allocate a narrow fixed width for `.summary-level-group`, and apply `overflow-wrap:anywhere` to legend members. Replace fragile `nth-child` width rules with semantic column classes or selectors that work for both six- and seven-column tables.

- [ ] **Step 9: Run all frontend checks**

```bash
rtk npm run check:frontend
```

Expected: TypeScript checking and every Node frontend test pass.

- [ ] **Step 10: Commit compact rendering**

```bash
rtk git add src/superglm/editor/app/summary.js src/superglm/editor/app/styles.css tests/editor_frontend/summary.test.js
rtk git commit -m "Render grouped levels in editor summaries"
```

### Task 8: Prove browser behavior and the Excel export boundary

**Files:**
- Modify: `tests/editor/test_editor_workspace_browser.py`
- Modify: `tests/test_rating_table_export.py:1028-1105`
- Modify: `docs/guide/results.md:1-25,100-140`

- [ ] **Step 1: Add a browser test for the view-only toggle**

Use the existing editor browser fixture. Create or retain a collapsed categorical state, then record model revision, history length, and outbound requests. Toggle with the keyboard through the native radio role:

```python
grouped = page.get_by_role("radio", name="Grouped")
grouped.focus()
grouped.press("Space")
expect(grouped).to_be_checked()
```

Wait for the grouped `/summary` response and assert the compact table and Full summary iframe both show `G1`/the grouped legend. Toggle back and assert exact expanded member rows return.

- [ ] **Step 2: Assert the toggle performs no structural work**

After both toggles, assert:

- model revision is unchanged;
- history length is unchanged;
- no request was made to `/collapse_levels`, `/ungroup_levels`, `/uncollapse_levels`, or `/refit_offset`;
- no chart grouping mode changed;
- summary request bodies contain the selected `level_display`;
- the control's accessible group name is `Categorical levels` and both radios are keyboard reachable.

- [ ] **Step 3: Run the focused browser test**

```bash
rtk test uv run pytest tests/editor/test_editor_workspace_browser.py -q -k "summary_level_display"
```

Expected: the browser test passes with the selected mode synchronized across compact and full views.

- [ ] **Step 4: Add an Excel non-regression test around a collapsed model**

Fit/export a model with at least one collapsed categorical group. Capture `build_summary_export_payload(model)` before calling either summary mode, call both, export, and assert:

```python
before = build_summary_export_payload(model)
model.summary(level_display="expanded")
model.summary(level_display="grouped")
after = build_summary_export_payload(model)
assert after == before

term_headers = [
    summary_ws.cell(row=term_min_row, column=column).value
    for column in range(term_min_col, term_max_col + 1)
]
assert term_headers == EXPECTED_EXISTING_SUMMARY_HEADERS
assert "Level group" not in term_headers
assert "G1" not in {cell.value for row in summary_ws.iter_rows() for cell in row}
```

Also assert the Rating Tables sheet still contains separate rows for every exact original categorical level with the existing relativity and weight values. Do not edit any file under `src/superglm/export/`.

- [ ] **Step 5: Run export tests**

```bash
rtk test uv run pytest tests/test_rating_table_export.py -q
```

Expected: all workbook layout, type, table-name, header, and new mode-isolation assertions pass.

- [ ] **Step 6: Document the summary option and export distinction**

Expand `docs/guide/results.md` with:

```python
print(model.summary())                           # exact original levels, expanded
print(model.summary(level_display="grouped"))   # one row per fitted group + legend
```

Explain that `Level group` IDs are summary-local, scoped per feature, and never rename levels. State explicitly that `export_rating_tables()` remains expanded over exact original levels and does not accept or emit `level_display`/`G1` metadata.

- [ ] **Step 7: Run all touched Python and frontend suites**

```bash
rtk test uv run pytest tests/test_summary_level_display.py tests/test_metrics.py tests/test_sklearn.py tests/test_editor.py tests/test_rating_table_export.py tests/test_ordered_categorical_inference.py -q
rtk npm run check:frontend
rtk ruff check src/superglm/inference/summary.py src/superglm/inference/summary_levels.py src/superglm/inference/metrics.py src/superglm/model/report_ops.py src/superglm/model/api.py src/superglm/sklearn.py src/superglm/editor/ tests/test_summary_level_display.py tests/test_metrics.py tests/test_sklearn.py tests/test_editor.py tests/test_rating_table_export.py
```

Expected: all focused Python tests, frontend tests/type checks, and Ruff pass.

- [ ] **Step 8: Run broad non-slow validation**

```bash
rtk test uv run pytest tests/ -q -m "not slow"
rtk mypy src/
```

Expected: the repository's non-slow suite passes and mypy reports no new errors. If a pre-existing unrelated mypy baseline exists, record the exact unchanged diagnostics in the handoff instead of editing unrelated files.

- [ ] **Step 9: Inspect the final diff for boundary violations**

```bash
rtk git status --short
rtk git diff --stat origin/master...HEAD
rtk git diff origin/master...HEAD -- src/superglm/export
rtk git diff --check
```

Expected: the export-source diff is empty, whitespace checks pass, and only the planned summary/editor/test/docs files changed.

- [ ] **Step 10: Commit documentation and end-to-end regressions**

```bash
rtk git add tests/editor/test_editor_workspace_browser.py tests/test_rating_table_export.py docs/guide/results.md
rtk git commit -m "Document and verify summary level displays"
```

## Completion checklist

- [ ] `model.summary()` defaults to exact expanded categorical rows, including references.
- [ ] `level_display="grouped"` uses compact IDs and wrapped exact-member legends.
- [ ] ASCII, notebook HTML, editor compact, and editor Full summary agree.
- [ ] Model, metrics, regressor, and classifier entry points accept the same option.
- [ ] Invalid modes fail before cache lookup with both accepted values in the message.
- [ ] Expanded duplicates do not duplicate feature EDF or restore stale editor inference.
- [ ] Ordinary summaries do not gain an empty `Level group` column.
- [ ] The editor preference is view-only and survives summary-producing workflows.
- [ ] HTML/editor strings are escaped and long legend membership wraps.
- [ ] Excel output is unchanged and contains neither `Level group` nor `G1` metadata.
- [ ] Focused, frontend, browser, export, Ruff, mypy, and non-slow verification evidence is recorded.
