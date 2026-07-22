# Categorical Level Display in Model Summaries

**Date:** 2026-07-22
**Status:** Approved

## Purpose

Collapsed categorical groups currently appear in `model.summary()` under a fitted label made by
joining every member with `+`. The summary sizes its coefficient table from the longest term name,
so a group containing many levels can make terminal and notebook summaries extremely wide. The
Excel rating-table export does not have this problem because it already presents original levels as
structured rows.

Model summaries should present original categorical levels without changing fitted level labels or
grouping metadata. Users should also be able to switch to a compact, one-row-per-fitted-group view.
The existing Excel export is explicitly out of scope and must remain unchanged.

## User-Facing Behavior

### Public summary API

Add a keyword argument to model, metrics, and sklearn summary entry points:

```python
model.summary(level_display="expanded")  # default
model.summary(level_display="grouped")
```

`level_display` accepts only `"expanded"` and `"grouped"`. Any other value raises `ValueError`
with the accepted values in the message. The setting is independent of `detail`; `detail` continues
to control spline basis disclosure.

### Expanded view

Expanded is the default. Each original categorical level gets its own row, including reference
levels. Original level strings remain exact; SuperGLM does not replace them with a generated group
label.

When two or more original levels share one fitted parameter, a `Level group` column identifies the
relationship with deterministic IDs scoped to the feature:

```text
Term                 Level group    Estimate    SE       p
territory[A]                        0.0000      ref      --
territory[B]         G1             0.1840      0.0520   0.001
territory[C]         G1             0.1840      0.0520   0.001
territory[D]         G2            -0.0920      0.0410   0.025
territory[E]         G2            -0.0920      0.0410   0.025
```

Only groups containing more than one original level receive an ID. Ungrouped levels leave the
column blank. IDs are assigned as `G1`, `G2`, and so on by the first member's position in
`LevelGrouping.all_original_levels`, and restart for each feature. Feature separators keep
identically named IDs unambiguous.

Expanded member rows repeat the shared coefficient, standard error, statistic, p-value,
confidence interval, significance marker, and warnings because those values describe the one
fitted parameter shared by those levels. Feature EDF appears only once for the feature so expanded
rows cannot imply an inflated total EDF. A reference row reports coefficient zero, `ref` for its
standard error, and no statistic, p-value, or interval.

### Grouped view

Grouped view emits one row per fitted categorical level, including the reference level. A collapsed
row is identified by its feature and compact `Level group` ID rather than by a synthetic
`A+B+C+...` level label. Exact members are listed in a wrapped level-group legend beneath the
feature/table, for example:

```text
Level groups (territory): G1 = B, C; G2 = D, E
```

HTML renders the same mapping with wrapping and accessible text. Ungrouped fitted levels retain
their exact level names. A fitted reference group is present and marked `ref`.

The legend is presentation metadata. It does not rename levels, alter predictions, or change the
serialized model.

### Where the behavior applies

The selected display applies consistently to:

- terminal/ASCII `str(summary)` output;
- notebook `_repr_html_()` output;
- the editor Summary inspector's compact table and Full summary disclosure.

The `Level group` column is shown only when the model contains at least one multi-level fitted
group. This avoids making ordinary, ungrouped summaries wider.

## Editor Interaction

The editor Summary inspector gains an accessible `Expanded` / `Grouped` segmented control and
defaults to Expanded. These terms describe presentation only; the control must not say “Ungrouped,”
which could be confused with the editor's structural ungroup-and-refit action.

The preference lives in editor view state. Changing it requests or renders the matching summary
variant but does not:

- refit or materialize a different model;
- increment the model revision or edit epoch;
- create history;
- change chart grouping mode;
- invalidate metrics or reports; or
- affect either export format.

The compact table and Full summary disclosure must always use the same selected variant. Existing
summary evidence sequencing still rejects stale responses. Both variants may reuse the same fitted
inference and maintain separate presentation cache entries.

## Architecture

Coefficient inference remains canonical and parameter-oriented. The existing coefficient rows are
not expanded and their fitted group labels are not rewritten. A presentation adapter combines:

1. canonical coefficient/inference rows;
2. categorical specs and their `LevelGrouping` mappings; and
3. the selected `level_display` mode.

It produces immutable display rows plus level-group legend metadata for the ASCII, HTML, and editor
renderers:

```text
canonical inference rows + grouping metadata
                    |
                    v
            summary display adapter
               /      |      \
            ASCII    HTML    editor typed payload
```

Reference rows are synthesized only in this display layer because treatment-coded categorical
features do not have a fitted coefficient row for their reference level. Non-categorical,
interaction, spline, and polynomial rows pass through unchanged.

The adapter must handle both ordinary and ordered categorical specs, including groups created by
the public `collapse_levels(...)` API and groups created through the editor. Existing stale-editor
inference suppression remains authoritative: expansion may repeat suppressed cells but must never
restore stale SEs or tests.

`ModelSummary` retains canonical rows for backward compatibility with current internal consumers
and receives the display metadata needed by its renderers. `level_display` is included in summary
presentation cache keys. Model and metrics summaries use the same adapter so their displayed rows
remain aligned.

## Export Boundary

Excel export remains on its existing renderer-independent rating-table and summary-export paths.
It must not gain a `Level group` column, generated `G1` IDs, editor view state, or a
`level_display` option. Existing sheet names, table names, columns, cell types, original-level
expansion, and values remain unchanged.

This boundary is enforced structurally: export code consumes canonical inference/rating-table data,
not presentation rows from `ModelSummary`.

## Validation and Failure Behavior

- Validate `level_display` before consulting the summary cache.
- Report the accepted values in invalid-argument errors.
- If grouping metadata is absent or is an identity mapping, both modes produce ordinary level rows
  and no `Level group` column.
- If a canonical row cannot be matched to grouping metadata, preserve the canonical row rather than
  dropping inference information.
- Exact level strings are escaped in HTML and editor markup using the existing escaping paths.
- Group legends wrap; they never determine the table width from the concatenated member text.

## Test Strategy

### Python summary tests

- Expanded is the default and uses exact original level rows.
- Grouped mode has one row per fitted group and a correct membership legend.
- Multiple groups receive deterministic per-feature `G1`, `G2`, ... IDs.
- Group IDs restart for another feature without ambiguity.
- Reference groups and ordinary reference levels render with zero/`ref` and blank tests.
- Shared estimates, SEs, tests, intervals, warnings, and significance markers repeat correctly.
- EDF is not duplicated across expanded members.
- Ungrouped categorical and non-categorical summaries remain stable.
- Ordered categorical grouping works in both display modes.
- Editor-stale inference remains suppressed in both modes.
- `ModelMetrics.summary()` and `model.summary()` agree.
- Invalid `level_display` values fail clearly.
- Summary cache entries do not cross display modes.

### Renderer and editor tests

- ASCII and HTML include `Level group` only when a multi-level group exists.
- Long collapsed memberships no longer set table width through a joined fitted label.
- HTML and editor output escape original level strings.
- The editor defaults to Expanded and switches both compact and Full summary output together.
- Switching modes changes only view/evidence state and never calls structural refit endpoints,
  changes model revision, or writes history.
- Browser keyboard and accessibility tests cover the segmented control.

### Export regression tests

- Workbook sheet/table names and headers remain unchanged.
- No `Level group` or `G1` display metadata appears in Excel.
- Categorical rating tables remain expanded over exact original levels with their existing values
  and weights.
- Structured Model Summary export continues to use its existing canonical contract.

## Non-Goals

- Renaming fitted groups or changing `LevelGrouping` serialization.
- Changing how collapsing, ungrouping, fitting, prediction, plotting, or inference works.
- Adding group IDs to Excel or other deployment artifacts.
- Truncating or otherwise mutating original level strings.
- Making the editor Summary control perform a structural grouping action.
