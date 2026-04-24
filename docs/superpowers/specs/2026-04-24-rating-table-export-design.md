# Rating Table Export Design

## Goal

Add a production-facing rating table export for fitted `SuperGLM` models. The first renderer is Excel. The design keeps a structured payload boundary so JSON can be added later without changing model-side extraction logic.

The export mirrors the notebook pattern from the provided screenshots:

- an intercept/base relativity near the top of the workbook,
- one-dimensional rating blocks arranged left to right in three-column groups,
- interaction tables placed two blank rows below the one-dimensional blocks,
- a plain model summary sheet.

Spline and polynomial terms must use the existing discretization machinery, not a plotting grid.

## Public API

Add a public model method:

```python
model.export_rating_tables(
    file_path,
    X,
    y,
    sample_weight=None,
    *,
    n_bins=150,
    impact_bins=(20, 50, 100, 200, 250),
    bin_strategy="exposure_quantile",
    format=None,
    sheet_name="Rating Tables",
    summary_sheet_name="Model Summary",
    impact_sheet_name="Discretization Impact",
    centering="native",
)
```

Add a functional equivalent:

```python
from superglm.export import export_rating_tables
```

`format=None` infers from `file_path`. Initial support accepts `.xlsx` / `.xlsm` and writes Excel. A future JSON renderer can consume the same structured payload.

## Discretization Behavior

The selected export fidelity is `n_bins`, default `150`.

For spline and polynomial main effects, call:

```python
model.discretization_impact(
    X,
    y,
    sample_weight=sample_weight,
    n_bins=n_bins,
    bin_strategy=bin_strategy,
    features=[...],
)
```

The rating table rows for continuous main effects come from `DiscretizationResult.tables[feature]`. Export columns are:

- level label, formatted from `bin_from` and `bin_to`,
- relativity,
- sample weight.

Also run a sweep over `impact_bins` and write a `Discretization Impact` sheet. The sweep does not change the exported rating tables. It reports one row per `(n_bins, feature)` plus model-level metrics copied onto each row:

- `n_bins`
- `feature`
- `actual_bins`
- `deviance_original`
- `deviance_discretized`
- `deviance_change`
- `deviance_change_pct`
- `mean_abs_prediction_change_pct`
- `max_abs_prediction_change_pct`
- `prediction_correlation`

Categorical and numeric terms are not discretized by this diagnostic.

## Rating Table Payload

Build a renderer-independent payload before writing Excel.

Model metadata:

- intercept log value,
- base relativity, `exp(intercept)` for log-link models,
- family, link, method, effective degrees of freedom, deviance, and convergence fields from the fitted model metadata.

Main effect blocks:

- categorical: use `term_inference(...).to_dataframe()` and observed sample weight by level from `X`,
- numeric: use `term_inference(...).to_dataframe()` with a `per_unit` label,
- spline/polynomial: use the selected `DiscretizationResult.tables` data,
- ordered categorical: follow its categorical table output; if spline-backed, still export levels rather than the smooth curve.

Interaction blocks:

- categorical x categorical: two-way matrix, rows are the first parent levels, columns are the second parent levels, values are interaction relativities,
- spline/numeric/polynomial x categorical: matrix with continuous bins or per-unit labels down rows and categorical levels across columns when the interaction reconstructs to a per-level curve or per-level slope,
- tensor or continuous x continuous interactions: matrix over selected-bin axes when the interaction reconstructs a rectangular surface,
- interaction shapes outside those cases raise a clear `NotImplementedError` rather than silently exporting a misleading table.

Interactions are written two blank rows below the tallest one-dimensional block in the main sheet.

## Excel Renderer

Excel rendering uses `openpyxl`.

`pyproject.toml` should add `openpyxl>=3.1` as a normal dependency because Excel export is first-class initial support.

Workbook layout:

- `Rating Tables` sheet:
  - `A2`: `Base`
  - `C2`: base relativity
  - row 5: main-effect block titles
  - row 7: block headers
  - row 8 onward: block values
  - interactions start at `max_main_effect_row + 3`
- `Discretization Impact` sheet:
  - tidy sweep table, one row per feature/bin-count combination
- `Model Summary` sheet:
  - `str(model.summary(detail="compact"))`, one line per row in column A

Formatting:

- bold title/header cells,
- relativity number format `0.000000`,
- weight number format `#,##0.00`,
- auto-width columns bounded to a reasonable maximum,
- freeze panes at the rating table data region.

## Error Handling

- Raise `RuntimeError` if the model is not fitted.
- Raise `ValueError` for unsupported `format` or unsupported file extension.
- Raise `ValueError` if `X`, `y`, or `sample_weight` lengths are inconsistent.
- Raise `NotImplementedError` for interaction structures that cannot be represented as a rating table without ambiguity.
- Propagate existing validation from `discretization_impact()` for invalid continuous feature/bin settings.

## Tests

Use test-first implementation.

Targeted tests:

- `export_rating_tables(..., n_bins omitted)` calls discretization with `150`.
- the impact sweep includes exactly `20, 50, 100, 200, 250` by default.
- continuous main-effect rows come from `discretization_impact(..., n_bins=selected)` and not from `term_inference(n_points=...)`.
- Excel workbook has the expected sheets and core cells.
- one-dimensional blocks use three-column layout.
- interaction blocks start two blank rows below the one-dimensional section.
- unsupported formats raise a clear `ValueError`.
- top-level and model-method APIs produce equivalent files.

## Out Of Scope

- JSON renderer implementation.
- Interactive UI.
- Refitting models at different discretization levels.
- Arbitrary custom workbook themes.
- Guaranteeing every possible interaction type has an Excel matrix in the first version.
