# Editor Export and Ordered Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the editor Save action with validated joblib and structured Excel exports, and make ordered-spline p-value presentation consistent across Python, Excel, and the editor.

**Architecture:** First merge the already-tested ordered-inference branch into the editor worktree so shared model summaries have the correct statistical rows. Then add a renderer-independent typed summary payload, use it in the existing rating workbook, add validated artifact serialization and revision-pinned editor endpoints, and finish with a focused frontend Export dialog. The public rating-table exporter and editor share the same workbook renderer; the frontend never owns statistical interpretation.

**Tech Stack:** Python 3.10+, NumPy, pandas, joblib, openpyxl, FastAPI/Starlette responses, vanilla ES modules, Node test runner, Playwright, pytest, Ruff.

---

## File map

- Create `src/superglm/export/summary.py`: typed, renderer-independent model-summary export payload.
- Modify `src/superglm/export/rating_tables.py`: attach the typed summary payload to `RatingTablePayload`.
- Modify `src/superglm/export/excel.py`: render typed summary cells/tables and accept paths or binary streams.
- Modify `src/superglm/editor/persistence.py`: serialize, round-trip load, and validate joblib artifacts.
- Modify `src/superglm/editor/evaluation.py`: resolve training data separately from validation-first metrics data.
- Modify `src/superglm/editor/widget.py`: pin model revision and coordinate both export formats.
- Modify `src/superglm/editor/server.py`: expose generic export download/save routes and preserve legacy model routes.
- Create `src/superglm/editor/app/views/export_dialog.js`: own format selection, progress, download, and kernel-path behavior.
- Modify `src/superglm/editor/app/index.html`: rename Save to Export and replace the dialog controls.
- Modify `src/superglm/editor/app/main.js`: bind the focused export view instead of owning export workflow details.
- Modify `src/superglm/editor/app/styles/dialogs.css`: format choice and validation-status styling.
- Modify `src/superglm/editor/app/views/help_content.js`: export help and popover wording.
- Modify `docs/notebooks/editor_demo.ipynb`: use canonical ordered spline construction and one global term p-value.
- Modify `docs/guide/editor.md` and `docs/guide/results.md`: document both editor exports and structured workbook layout.
- Modify `tests/test_ordered_categorical_inference.py`, `tests/test_ordered_reference_export.py`, and `tests/test_editor.py`: ordered-inference integration and backend exports.
- Modify `tests/test_rating_table_export.py`: structured workbook contracts.
- Create `tests/editor_frontend/export_dialog.test.js`: fast frontend export-state tests.
- Modify `tests/editor/test_editor_workspace_browser.py`: real-browser Export dialog and ordered-summary coverage.

### Task 1: Integrate the completed ordered-categorical inference foundation

**Files:**
- Merge from: `feat/ordered-categorical-inference`
- Modify on conflict: `src/superglm/editor/summaries.py`
- Modify on conflict: `tests/test_editor.py`
- Modify on conflict: `docs/notebooks/editor_demo.ipynb`
- Test: `tests/test_ordered_categorical_api.py`
- Test: `tests/test_ordered_categorical_inference.py`
- Test: `tests/test_ordered_reference_export.py`

- [ ] **Step 1: Merge the completed inference branch without committing automatically**

Run:

```bash
rtk git merge --no-ff --no-commit feat/ordered-categorical-inference
```

Expected: the merge stops with, at most, conflicts in the editor demo notebook,
`src/superglm/editor/summaries.py`, and `tests/test_editor.py`.

- [ ] **Step 2: Resolve the editor summary conflict by keeping both contracts**

The resulting `_compact_summary_row` must preserve the editor's typed fields while allowing
ordered level rows from shared inference to carry no statistic or p-value:

```python
def _compact_summary_row(row) -> dict[str, Any]:
    p_value = _finite_float(row.wald_p if row.is_spline else row.p)
    return {
        "name": str(row.name),
        "group": str(row.group or ""),
        "kind": "spline" if row.is_spline else "coef",
        "coef": _finite_float(row.coef),
        "se": None if row.is_spline else _finite_float(row.se),
        "se_label": "curve" if row.is_spline else "",
        "stat": _finite_float(row.wald_chi2 if row.is_spline else row.z),
        "stat_label": "chi2" if row.is_spline else "",
        "p_value": p_value,
        "sig_code": _summary_sig_code(p_value, bool(row.quasi_separated)),
        "sig_class": _summary_sig_class(p_value, bool(row.quasi_separated)),
        "quasi_separated": bool(row.quasi_separated),
        "active": bool(row.active),
        "n_params": int(row.n_params or 0),
        "ref_df": _finite_float(row.ref_df),
        "edf": _row_edf(row),
    }
```

Retain all other editor payload, grouped-reference, and stale-inference helpers from the current
branch. Resolve `tests/test_editor.py` by retaining all existing tests plus the ordered assertions
from the merged branch. Resolve the notebook structurally: retain the redesigned editor workflow,
but use `basis=Spline(...)` and the global ordered term p-value cells from the inference branch.

- [ ] **Step 3: Run the ordered-inference regression suite**

Run:

```bash
rtk pytest tests/test_ordered_categorical_api.py tests/test_ordered_categorical_inference.py tests/test_ordered_reference_export.py
```

Expected: all tests pass; the ordered spline has one Wood test and level rows have `p is None`.

- [ ] **Step 4: Run the editor and rating-table conflict surface**

Run:

```bash
rtk pytest tests/test_editor.py tests/test_rating_table_export.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit the merge**

```bash
rtk git add src tests docs README.md
rtk git commit -m "Integrate ordered categorical inference"
```

### Task 2: Define a typed model-summary export payload

**Files:**
- Create: `src/superglm/export/summary.py`
- Modify: `src/superglm/export/rating_tables.py:35-55, 590-600`
- Test: `tests/test_rating_table_export.py`

- [ ] **Step 1: Write failing payload tests**

Add tests that fit the existing rating-table fixture and assert typed overview and term rows:

```python
from superglm.export.summary import build_summary_export_payload


def test_summary_export_payload_keeps_typed_model_and_term_values():
    model, *_ = _fit_export_model()

    payload = build_summary_export_payload(model)

    overview = {(row.section, row.metric): row.value for row in payload.overview}
    assert overview[("Model", "Family")] == "Poisson"
    assert isinstance(overview[("Fit", "Observations")], int)
    assert isinstance(overview[("Information Criteria", "AIC")], float)
    intercept = next(row for row in payload.terms if row.term == "Intercept")
    assert isinstance(intercept.estimate, float)
    assert isinstance(intercept.p_value, float)


def test_summary_export_payload_leaves_noninferential_ordered_levels_blank():
    model = _fit_ordered_spline_export_model()

    payload = build_summary_export_payload(model)

    global_row = next(row for row in payload.terms if row.kind == "smooth")
    level_rows = [row for row in payload.terms if row.group == "band" and row.kind == "level"]
    assert global_row.p_value is not None
    assert level_rows
    assert all(row.statistic is None and row.p_value is None for row in level_rows)
```

- [ ] **Step 2: Run the tests to verify RED**

Run:

```bash
rtk pytest tests/test_rating_table_export.py -k "summary_export_payload"
```

Expected: collection fails because `superglm.export.summary` does not exist.

- [ ] **Step 3: Implement the typed payload**

Create immutable rows and a conversion function in `src/superglm/export/summary.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SummaryOverviewRow:
    section: str
    metric: str
    value: str | int | float | bool | None


@dataclass(frozen=True)
class SummaryTermRow:
    term: str
    group: str
    kind: str
    estimate: float | None
    std_error: float | None
    statistic: float | None
    statistic_type: str
    p_value: float | None
    ci_lower: float | None
    ci_upper: float | None
    edf: float | None
    smoothing_lambda: float | None
    active: bool
    significance: str
    warning: str


@dataclass(frozen=True)
class SummaryExportPayload:
    overview: tuple[SummaryOverviewRow, ...]
    terms: tuple[SummaryTermRow, ...]
    notes: tuple[str, ...]


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _stars(value: float | None) -> str:
    if value is None:
        return ""
    if value < 0.001:
        return "***"
    if value < 0.01:
        return "**"
    if value < 0.05:
        return "*"
    if value < 0.1:
        return "."
    return ""


def _summary_notes(info: dict[str, Any], rows: list[Any]) -> tuple[str, ...]:
    notes: list[str] = []
    if info.get("editor_inference_stale"):
        notes.append(
            "Editor edits applied: coefficient standard errors, confidence intervals, "
            "and p-values are suppressed because they belong to the original fitted model."
        )
    elif any(row.is_spline for row in rows):
        notes.append(
            "Smooth p-values use the Wood (2013) Bayesian test; parametric p-values "
            "are Wald approximations."
        )
    else:
        notes.append("Parametric p-values are Wald approximations.")
    if any(row.quasi_separated for row in rows):
        notes.append("QS marks a quasi-separated coefficient with unreliable finite inference.")
    return tuple(notes)


def build_summary_export_payload(model) -> SummaryExportPayload:
    summary = model.summary(detail="compact")
    info = summary._info
    raw = summary.to_dict()
    overview = [
        SummaryOverviewRow("Model", "Family", str(info["family"])),
        SummaryOverviewRow("Model", "Link", str(info["link"])),
        SummaryOverviewRow("Model", "Method", "MLE" if info.get("method") == "ML" else str(info.get("method", ""))),
        SummaryOverviewRow("Model", "Penalty", str(info["penalty"])),
        SummaryOverviewRow("Fit", "Observations", int(info["n_obs"])),
        SummaryOverviewRow("Fit", "Effective DF", _finite(info["effective_df"])),
        SummaryOverviewRow("Fit", "Scale (phi)", _finite(info["phi"])),
        SummaryOverviewRow("Fit", "Deviance", _finite(info["deviance"])),
        SummaryOverviewRow("Fit", "Null Deviance", _finite(raw["deviance"].get("null_deviance"))),
        SummaryOverviewRow("Fit", "Explained Deviance", _finite(raw["deviance"].get("explained_deviance"))),
        SummaryOverviewRow("Fit", "Converged", bool(info["converged"])),
        SummaryOverviewRow("Fit", "Iterations", int(info["n_iter"])),
        SummaryOverviewRow("Information Criteria", "Log-Likelihood", _finite(info["log_likelihood"])),
        SummaryOverviewRow("Information Criteria", "AIC", _finite(info["aic"])),
        SummaryOverviewRow("Information Criteria", "AICc", _finite(info["aicc"])),
        SummaryOverviewRow("Information Criteria", "BIC", _finite(info["bic"])),
        SummaryOverviewRow("Information Criteria", "EBIC", _finite(info["ebic"])),
    ]
    if "nb_theta" in info:
        lower, upper = info["nb_theta_ci"]
        overview.extend(
            [
                SummaryOverviewRow("Distribution Profile", "NB2 Theta", _finite(info["nb_theta"])),
                SummaryOverviewRow("Distribution Profile", "NB2 Theta CI Lower", _finite(lower)),
                SummaryOverviewRow("Distribution Profile", "NB2 Theta CI Upper", _finite(upper)),
                SummaryOverviewRow("Distribution Profile", "NB2 Theta Method", str(info["nb_theta_method"])),
            ]
        )
    if "tweedie_p" in info:
        lower, upper = info["tweedie_p_ci"]
        overview.extend(
            [
                SummaryOverviewRow("Distribution Profile", "Tweedie p", _finite(info["tweedie_p"])),
                SummaryOverviewRow("Distribution Profile", "Tweedie p CI Lower", _finite(lower)),
                SummaryOverviewRow("Distribution Profile", "Tweedie p CI Upper", _finite(upper)),
                SummaryOverviewRow("Distribution Profile", "Tweedie phi", _finite(info["tweedie_phi"])),
                SummaryOverviewRow("Distribution Profile", "Tweedie p Method", str(info["tweedie_p_method"])),
            ]
        )
    terms = []
    for row in summary._coef_rows:
        is_smooth = bool(row.is_spline)
        p_value = _finite(row.wald_p if is_smooth else row.p)
        terms.append(
            SummaryTermRow(
                term=str(row.name),
                group=str(row.group or ""),
                kind=(
                    "smooth"
                    if is_smooth
                    else ("level" if row.group and "[" in str(row.name) else "coefficient")
                ),
                estimate=None if is_smooth else _finite(row.coef),
                std_error=None if is_smooth else _finite(row.se),
                statistic=_finite(row.wald_chi2 if is_smooth else row.z),
                statistic_type="chi2" if is_smooth else ("z" if row.z is not None else ""),
                p_value=p_value,
                ci_lower=None if is_smooth else _finite(row.ci_low),
                ci_upper=None if is_smooth else _finite(row.ci_high),
                edf=_finite(row.edf),
                smoothing_lambda=_finite(row.smoothing_lambda),
                active=bool(row.active) if is_smooth else row.coef is not None,
                significance="QS" if row.quasi_separated else _stars(p_value),
                warning="Quasi-separated" if row.quasi_separated else "",
            )
        )
    return SummaryExportPayload(
        overview=tuple(overview),
        terms=tuple(terms),
        notes=_summary_notes(info, summary._coef_rows),
    )
```

- [ ] **Step 4: Attach the payload to rating-table construction**

Change `RatingTablePayload.summary_lines` to `summary: SummaryExportPayload`, import
`build_summary_export_payload`, and construct it with:

```python
return RatingTablePayload(
    base_relativity=float(np.exp(model.result.intercept)),
    selected_n_bins=int(n_bins),
    main_effects=main_effects,
    interactions=_interaction_blocks(model, n_bins),
    discretization_impact=impact,
    summary=build_summary_export_payload(model),
)
```

- [ ] **Step 5: Run payload tests to verify GREEN**

```bash
rtk pytest tests/test_rating_table_export.py -k "summary_export_payload"
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

```bash
rtk git add src/superglm/export/summary.py src/superglm/export/rating_tables.py tests/test_rating_table_export.py
rtk git commit -m "Add typed model summary export payload"
```

### Task 3: Render a structured Excel summary and support in-memory workbooks

**Files:**
- Modify: `src/superglm/export/excel.py`
- Test: `tests/test_rating_table_export.py`

- [ ] **Step 1: Replace the old layout assertion with failing structured-sheet assertions**

Extend `test_excel_workbook_layout`:

```python
summary_ws = wb["Model Summary"]
assert summary_ws["A1"].value == "Model Summary"
assert [summary_ws.cell(4, col).value for col in range(1, 4)] == ["Section", "Metric", "Value"]
assert "ModelOverview" in summary_ws.tables
assert "TermInference" in summary_ws.tables
from openpyxl.utils.cell import range_boundaries

min_col, min_row, max_col, _ = range_boundaries(summary_ws.tables["TermInference"].ref)
headers = [summary_ws.cell(min_row, col).value for col in range(min_col, max_col + 1)]
assert headers[:3] == ["Term", "Group", "Kind"]
assert summary_ws.freeze_panes == "A5"

metric_rows = {
    summary_ws.cell(row, 2).value: summary_ws.cell(row, 3).value
    for row in range(5, 30)
    if summary_ws.cell(row, 2).value
}
assert isinstance(metric_rows["Observations"], int)
assert isinstance(metric_rows["AIC"], float)
assert not any("SuperGLM Results" in str(cell.value) for row in summary_ws for cell in row)
```

Add a stream test:

```python
def test_excel_renderer_accepts_binary_stream():
    model, X, y, w = _fit_export_model()
    payload = build_rating_table_payload(model, X, y, sample_weight=w)
    target = io.BytesIO()

    write_rating_table_workbook(
        payload,
        target,
        sheet_name="Rating Tables",
        summary_sheet_name="Model Summary",
        impact_sheet_name="Discretization Impact",
    )

    target.seek(0)
    workbook = load_workbook(target, data_only=True)
    assert workbook.sheetnames == ["Rating Tables", "Discretization Impact", "Model Summary"]
```

- [ ] **Step 2: Run the tests to verify RED**

```bash
rtk pytest tests/test_rating_table_export.py -k "excel_workbook_layout or renderer_accepts_binary_stream"
```

Expected: the summary assertions fail because column A still contains ASCII lines; stream setup
fails while the renderer assumes a filesystem path.

- [ ] **Step 3: Add focused Excel table helpers**

In `src/superglm/export/excel.py`, add:

```python
from os import PathLike
from typing import BinaryIO


def _add_excel_table(ws, name: str, ref: str) -> None:
    from openpyxl.worksheet.table import Table, TableStyleInfo

    table = Table(displayName=name, ref=ref)
    table.tableStyleInfo = TableStyleInfo(
        name="TableStyleMedium2",
        showFirstColumn=False,
        showLastColumn=False,
        showRowStripes=True,
        showColumnStripes=False,
    )
    ws.add_table(table)


def _write_summary_sheet(ws, payload) -> None:
    from openpyxl.styles import Font
    from openpyxl.utils import get_column_letter

    ws["A1"] = "Model Summary"
    ws["A1"].font = Font(bold=True, size=14)
    ws.append([])
    ws.append(["Fit and model overview"])
    ws.append(["Section", "Metric", "Value"])
    for row in payload.overview:
        ws.append([row.section, row.metric, row.value])
    overview_end = ws.max_row
    _add_excel_table(ws, "ModelOverview", f"A4:C{overview_end}")

    term_start = overview_end + 3
    term_headers = [
        "Term", "Group", "Kind", "Estimate", "Std Error", "Statistic",
        "Statistic Type", "P Value", "CI Lower", "CI Upper", "EDF", "Lambda",
        "Active", "Significance", "Warning",
    ]
    for col, value in enumerate(term_headers, start=1):
        ws.cell(term_start, col, value)
    for out_row, row in enumerate(payload.terms, start=term_start + 1):
        values = [
            row.term, row.group, row.kind, row.estimate, row.std_error, row.statistic,
            row.statistic_type, row.p_value, row.ci_lower, row.ci_upper, row.edf,
            row.smoothing_lambda, row.active, row.significance, row.warning,
        ]
        for col, value in enumerate(values, start=1):
            ws.cell(out_row, col, value)
    if not payload.terms:
        ws.cell(term_start + 1, 1, "")
    term_end = max(term_start + 1, ws.max_row)
    _add_excel_table(ws, "TermInference", f"A{term_start}:O{term_end}")

    for row in ws.iter_rows(min_row=5, min_col=3, max_col=3, max_row=overview_end):
        if isinstance(row[0].value, float):
            row[0].number_format = "0.000000"
    for row in ws.iter_rows(min_row=term_start + 1, max_row=term_end, min_col=4, max_col=12):
        for cell in row:
            if isinstance(cell.value, float):
                cell.number_format = "0.000000"
    ws.freeze_panes = "A5"
    ws.auto_filter.ref = f"A{term_start}:O{term_end}"
    for col in range(1, 16):
        ws.column_dimensions[get_column_letter(col)].width = 16
    ws.column_dimensions["A"].width = 32
    ws.column_dimensions["B"].width = 24

    if payload.notes:
        notes_start = term_end + 3
        ws.cell(notes_start, 1, "Notes").font = Font(bold=True)
        for index, note in enumerate(payload.notes, start=notes_start + 1):
            ws.cell(index, 1, note)
```

- [ ] **Step 4: Make the workbook target path-or-stream aware**

Change the renderer signature to:

```python
def write_rating_table_workbook(
    payload,
    target: str | PathLike[str] | BinaryIO,
    *,
    sheet_name: str,
    summary_sheet_name: str,
    impact_sheet_name: str,
) -> None:
    if isinstance(target, str | PathLike):
        out = Path(target)
        out.parent.mkdir(parents=True, exist_ok=True)
    else:
        out = target
    # Build workbook exactly once, then:
    wb.save(out)
```

Replace the ASCII loop with `_write_summary_sheet(summary_ws, payload.summary)`. Keep the rating
and impact sheet construction unchanged.

- [ ] **Step 5: Run the rating exporter suite**

```bash
rtk pytest tests/test_rating_table_export.py
```

Expected: all tests pass, including typed cell and in-memory rendering assertions.

- [ ] **Step 6: Commit**

```bash
rtk git add src/superglm/export/excel.py tests/test_rating_table_export.py
rtk git commit -m "Structure Excel model summaries"
```

### Task 4: Validate serialized Python model artifacts

**Files:**
- Modify: `src/superglm/editor/persistence.py`
- Test: `tests/test_editor.py`

- [ ] **Step 1: Write failing serialization-validation tests**

Add:

```python
def test_serialize_validated_model_round_trips_predictions(editor_model, editor_frame):
    import io
    import joblib

    from superglm.editor.evaluation import coerce_dataset
    from superglm.editor.persistence import serialize_validated_model

    X, y = editor_frame
    dataset = coerce_dataset("validation", (X, y, None))
    data, validation = serialize_validated_model(editor_model, dataset=dataset, max_rows=17)
    loaded = joblib.load(io.BytesIO(data))

    assert validation.artifact_round_trip is True
    assert validation.prediction_rows == 17
    np.testing.assert_allclose(loaded.predict(X.iloc[:17]), editor_model.predict(X.iloc[:17]))


def test_serialize_validated_model_blocks_prediction_mismatch(editor_model, editor_frame, monkeypatch):
    from superglm.editor.evaluation import coerce_dataset
    from superglm.editor import persistence

    X, y = editor_frame
    dataset = coerce_dataset("validation", (X, y, None))
    original_load = persistence.joblib_load_bytes

    def load_with_bad_predict(data):
        loaded = original_load(data)
        loaded.predict = lambda X, offset=None: np.full(len(X), -999.0)
        return loaded

    monkeypatch.setattr(persistence, "joblib_load_bytes", load_with_bad_predict)
    with pytest.raises(ValueError, match="prediction validation failed"):
        persistence.serialize_validated_model(editor_model, dataset=dataset)


def test_serialize_validated_model_without_data_reports_contract_only(editor_model):
    from superglm.editor.persistence import serialize_validated_model

    _, validation = serialize_validated_model(editor_model, dataset=None)

    assert validation.artifact_round_trip is True
    assert validation.prediction_rows == 0
    assert validation.scope == "artifact"
```

- [ ] **Step 2: Run tests to verify RED**

```bash
rtk pytest tests/test_editor.py -k "serialize_validated_model"
```

Expected: import failure because `serialize_validated_model` is undefined.

- [ ] **Step 3: Implement deterministic bounded validation**

Add to `persistence.py`:

```python
import io
from dataclasses import dataclass
from typing import Any

import joblib


@dataclass(frozen=True)
class ModelArtifactValidation:
    artifact_round_trip: bool
    prediction_rows: int
    scope: str


def joblib_load_bytes(data: bytes):
    return joblib.load(io.BytesIO(data))


def _take_rows(value: Any, indices: np.ndarray):
    if value is None:
        return None
    if hasattr(value, "iloc"):
        return value.iloc[indices]
    return np.asarray(value)[indices]


def _validation_indices(n_rows: int, maximum: int) -> np.ndarray:
    count = min(n_rows, maximum)
    if count <= 0:
        return np.empty(0, dtype=np.intp)
    return np.linspace(0, n_rows - 1, count, dtype=np.intp)


def serialize_validated_model(model, *, dataset=None, max_rows: int = 512):
    if max_rows < 1:
        raise ValueError("max_rows must be at least 1")
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    data = buffer.getvalue()
    loaded = joblib_load_bytes(data)
    if not isinstance(loaded, type(model)):
        raise ValueError("artifact contract validation failed: unexpected model type")
    if getattr(loaded, "result", None) is None or not callable(getattr(loaded, "predict", None)):
        raise ValueError("artifact contract validation failed: model is not fitted")
    beta = np.asarray(loaded.result.beta, dtype=np.float64)
    if beta.shape != np.asarray(model.result.beta).shape or not np.all(np.isfinite(beta)):
        raise ValueError("artifact contract validation failed: invalid fitted coefficients")
    if not np.isfinite(float(loaded.result.intercept)):
        raise ValueError("artifact contract validation failed: invalid fitted intercept")
    if tuple(loaded._feature_order) != tuple(model._feature_order):
        raise ValueError("artifact contract validation failed: feature metadata differs")
    if tuple(loaded._specs) != tuple(model._specs):
        raise ValueError("artifact contract validation failed: feature specifications differ")

    prediction_rows = 0
    if dataset is not None and dataset.n_obs:
        indices = _validation_indices(dataset.n_obs, max_rows)
        X = _take_rows(dataset.X, indices)
        offset = _take_rows(dataset.offset, indices)
        expected = np.asarray(model.predict(X, offset=offset), dtype=np.float64)
        actual = np.asarray(loaded.predict(X, offset=offset), dtype=np.float64)
        if expected.shape != actual.shape or not np.allclose(expected, actual, rtol=1e-12, atol=1e-12):
            raise ValueError("prediction validation failed for serialized model")
        prediction_rows = int(indices.size)
    scope = "artifact+predictions" if prediction_rows else "artifact"
    return data, ModelArtifactValidation(True, prediction_rows, scope)
```

Change `save_model` in this task to call `serialize_validated_model(model, dataset=dataset)`, then
write the returned bytes with `Path.write_bytes` only after validation succeeds. Keep the current
`download_model` wrapper intact until Task 5 redirects it through the shared export coordinator.

- [ ] **Step 4: Run validation and existing persistence tests**

```bash
rtk pytest tests/test_editor.py -k "serialize_validated_model or save_model or download_model"
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
rtk git add src/superglm/editor/persistence.py tests/test_editor.py
rtk git commit -m "Validate exported Python models"
```

### Task 5: Add revision-pinned model and Excel export endpoints

**Files:**
- Modify: `src/superglm/editor/evaluation.py`
- Modify: `src/superglm/editor/widget.py:519-570`
- Modify: `src/superglm/editor/server.py:145-180`
- Test: `tests/test_editor.py`

- [ ] **Step 1: Write failing backend export tests**

Add tests for a generic download route, Excel MIME type, validation headers, kernel-path Excel,
and missing training data:

```python
def test_widget_http_download_export_returns_validated_model(editor_model, editor_frame):
    session = EditorSession.from_model(editor_model, train_data=(*editor_frame, None))
    widget = session.widget()
    try:
        request = urllib.request.Request(
            f"{widget.url}/download_export?format=joblib&filename=edited.joblib",
            headers=_editor_token_header(widget.url),
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = response.read()
            assert response.headers["x-superglm-validation"] == "artifact+predictions"
            assert response.headers["content-type"] == "application/octet-stream"
    finally:
        widget.close()
    assert joblib.load(io.BytesIO(payload)).result is not None


def test_widget_http_download_export_returns_excel(editor_model, editor_frame):
    X, y = editor_frame
    session = EditorSession.from_model(editor_model, train_data=(X, y, None))
    widget = session.widget()
    try:
        request = urllib.request.Request(
            f"{widget.url}/download_export?format=xlsx&filename=rating.xlsx",
            headers=_editor_token_header(widget.url),
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = response.read()
            assert response.headers["content-type"] == (
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    finally:
        widget.close()
    workbook = load_workbook(io.BytesIO(payload), data_only=True)
    assert "Model Summary" in workbook.sheetnames


def test_widget_excel_export_requires_training_data(editor_model):
    editor_model._fit_X_ref = None
    editor_model._fit_y_ref = None
    session = EditorSession.from_model(editor_model)
    widget = session.widget()
    try:
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(
                urllib.request.Request(
                    f"{widget.url}/download_export?format=xlsx&filename=rating.xlsx",
                    headers=_editor_token_header(widget.url),
                ),
                timeout=5,
            )
        assert error.value.code == 400
        assert "train_data" in error.value.read().decode()
    finally:
        widget.close()


@pytest.mark.parametrize(
    ("format_name", "filename"),
    [("joblib", "edited.joblib"), ("xlsx", "rating.xlsx")],
)
def test_widget_http_export_file_writes_each_format(
    editor_model, editor_frame, tmp_path, format_name, filename
):
    session = EditorSession.from_model(editor_model, train_data=(*editor_frame, None))
    widget = session.widget()
    try:
        result = _post_json(
            f"{widget.url}/export_file",
            {"format": format_name, "directory": str(tmp_path), "filename": filename},
        )
    finally:
        widget.close()
    path = Path(result["path"])
    assert path == tmp_path / filename
    assert path.stat().st_size > 0
    if format_name == "joblib":
        assert joblib.load(path).result is not None
    else:
        assert "Model Summary" in load_workbook(path, data_only=True).sheetnames


def test_export_file_does_not_write_superseded_result(editor_model, editor_frame, tmp_path, monkeypatch):
    session = EditorSession.from_model(editor_model, train_data=(*editor_frame, None))
    widget = session.widget()
    original = widget._export_bytes

    def supersede(*args, **kwargs):
        result = original(*args, **kwargs)
        session._model_revision += 1
        return result

    monkeypatch.setattr(widget, "_export_bytes", supersede)
    try:
        with pytest.raises(RuntimeError, match="superseded"):
            widget._export_file("joblib", directory=str(tmp_path), filename="stale.joblib")
    finally:
        widget.close()
    assert not (tmp_path / "stale.joblib").exists()
```

- [ ] **Step 2: Run tests to verify RED**

```bash
rtk pytest tests/test_editor.py -k "download_export or excel_export_requires or export_file"
```

Expected: 404 responses because the routes do not exist.

- [ ] **Step 3: Add an explicit training export resolver**

In `evaluation.py`:

```python
def training_export_dataset(session) -> EvaluationDataset | None:
    explicit = getattr(session, "_evaluation_data", {})
    if "train" in explicit:
        return explicit["train"]
    return retained_fit_dataset(session)
```

Export it in `__all__`. Keep `default_metrics_dataset` validation-first for metrics and model
artifact prediction checks.

- [ ] **Step 4: Implement widget export coordinators**

Add a normalized format helper and two renderers:

```python
def _normalise_export_format(value: str) -> Literal["joblib", "xlsx"]:
    key = str(value).lower().lstrip(".")
    if key in {"joblib", "model"}:
        return "joblib"
    if key in {"xlsx", "excel"}:
        return "xlsx"
    raise ValueError(f"Unsupported export format: {value!r}")
```

Within `EditorWidget`, implement `_export_bytes(format, filename)` by calling
`_current_model_for_evidence()` once. For joblib, call `serialize_validated_model` with
`default_metrics_dataset(self.session)`. For Excel, require `training_export_dataset`, build the
rating payload with its `X`, `y`, `sample_weight`, and `offset`, and render to `io.BytesIO`.
Return an immutable result containing `data`, safe filename, MIME type, captured revision, and
validation scope. Re-check `self.session.model_revision` before returning.

Implement `_export_file(format, directory, filename)` by calling the same byte builder and writing
the completed bytes only after revision and validation checks pass. Preserve `_save_model` and
`_download_model` as delegates for compatibility.

- [ ] **Step 5: Add generic HTTP routes and legacy delegation**

Add:

```python
@app.get("/download_export")
def download_export(format: str, filename: str = "") -> Response:
    try:
        result = widget._export_bytes(format, filename or None)
    except _CLIENT_ERROR_TYPES as exc:
        return _json_response({"error": _client_error_message(exc)}, status_code=400)
    except Exception:  # pragma: no cover - surfaced to browser/tests as JSON
        _LOGGER.exception("Unhandled SuperGLM editor export error.")
        return _json_response({"error": "internal editor error"}, status_code=500)
    headers = {
        **_no_store_headers(),
        "Content-Disposition": f'attachment; filename="{result.filename}"',
        "X-SuperGLM-Model-Revision": str(result.model_revision),
    }
    if result.validation_scope:
        headers["X-SuperGLM-Validation"] = result.validation_scope
    return Response(content=result.data, media_type=result.media_type, headers=headers)


@app.post("/export_file")
def export_file(payload: dict[str, Any] = Body(default_factory=dict)) -> Response:
    return _guarded_json(
        lambda: widget._export_file(
            str(payload.get("format", "joblib")),
            directory=None if "directory" not in payload else str(payload["directory"]),
            filename=None if "filename" not in payload else str(payload["filename"]),
        )
    )
```

Replace `/download_model`'s independent dump with `result = widget._export_bytes("joblib",
filename)` inside its existing `try`/`except` block, and build the legacy response from
`result.data`, `result.filename`, and `result.media_type` so its status and attachment contract
remain unchanged.

- [ ] **Step 6: Run backend export and existing route tests**

```bash
rtk pytest tests/test_editor.py -k "export or save_model or download_model or routes"
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit**

```bash
rtk git add src/superglm/editor/evaluation.py src/superglm/editor/widget.py src/superglm/editor/server.py tests/test_editor.py
rtk git commit -m "Expose validated editor exports"
```

### Task 6: Replace Save with the focused Export dialog

**Files:**
- Create: `src/superglm/editor/app/views/export_dialog.js`
- Modify: `src/superglm/editor/app/index.html`
- Modify: `src/superglm/editor/app/main.js`
- Modify: `src/superglm/editor/app/styles/dialogs.css`
- Modify: `src/superglm/editor/app/views/help_content.js`
- Create: `tests/editor_frontend/export_dialog.test.js`
- Modify: `tests/test_editor.py`

- [ ] **Step 1: Write failing frontend controller tests**

Test the controller with the repository's fake-element pattern:

```javascript
import assert from "node:assert/strict";
import test from "node:test";

import { bindExportDialog } from "../../src/superglm/editor/app/views/export_dialog.js";

class FakeElement {
  constructor(value = "") {
    this.value = value;
    this.checked = false;
    this.disabled = false;
    this.textContent = "";
    this.listeners = new Map();
  }

  addEventListener(name, listener) {
    const listeners = this.listeners.get(name) ?? new Set();
    listeners.add(listener);
    this.listeners.set(name, listeners);
  }

  removeEventListener(name, listener) {
    this.listeners.get(name)?.delete(listener);
  }

  async emit(name) {
    const event = { target: this, preventDefault() {} };
    await Promise.all([...(this.listeners.get(name) ?? [])].map(listener => listener(event)));
  }
}

function blobResponse({ validation = "" } = {}) {
  return new Response(new Blob(["artifact"]), {
    headers: validation ? { "X-SuperGLM-Validation": validation } : {}
  });
}

function exportFixture({ pending = false } = {}) {
  const formats = { joblib: new FakeElement("joblib"), xlsx: new FakeElement("xlsx") };
  formats.joblib.checked = true;
  const filename = new FakeElement("superglm_edited_model.joblib");
  const directory = new FakeElement(".");
  const download = new FakeElement();
  const saveToKernel = new FakeElement();
  const status = new FakeElement();
  const saved = [];
  const posts = [];
  let requests = 0;
  let rejectPending = () => {};
  const pendingRequest = new Promise((_, reject) => { rejectPending = reject; });
  const client = {
    requestBlob: async () => {
      requests += 1;
      return pending ? pendingRequest : blobResponse();
    },
    postJSON: async (path, payload) => {
      posts.push({ path, payload });
      return { path: `${payload.directory}/${payload.filename}` };
    }
  };
  return {
    formats, filename, directory, download, saveToKernel, status, saved, posts,
    get requests() { return requests; },
    reject: rejectPending,
    context: {
      client, formats, filename, directory, download, saveToKernel, status,
      saveBlobToFile: async (_response, name) => { saved.push({ filename: name }); }
    }
  };
}

test("model export downloads a validated joblib artifact", async () => {
  const fixture = exportFixture();
  fixture.filename.value = "edited-model.joblib";
  fixture.client.requestBlob = async path => {
    assert.equal(path, "/download_export?format=joblib&filename=edited-model.joblib");
    return blobResponse({ validation: "artifact+predictions" });
  };

  const binding = bindExportDialog(fixture.context);
  await fixture.download.emit("click");

  assert.deepEqual(fixture.saved, [{ filename: "edited-model.joblib" }]);
  assert.match(fixture.status.textContent, /validated.*predictions/i);
  binding.destroy();
});

test("excel selection normalizes the filename and kernel payload", async () => {
  const fixture = exportFixture();
  fixture.formats.joblib.checked = false;
  fixture.formats.xlsx.checked = true;
  await fixture.formats.xlsx.emit("change");
  assert.equal(fixture.filename.value, "superglm_rating_tables.xlsx");

  await fixture.saveToKernel.emit("click");

  assert.deepEqual(fixture.posts, [{
    path: "/export_file",
    payload: { format: "xlsx", directory: ".", filename: "superglm_rating_tables.xlsx" }
  }]);
});

test("export suppresses duplicate clicks and restores controls after failure", async () => {
  const fixture = exportFixture({ pending: true });
  bindExportDialog(fixture.context);
  const first = fixture.download.emit("click");
  await fixture.download.emit("click");
  assert.equal(fixture.requests, 1);
  fixture.reject(new Error("training data required"));
  await first;
  assert.equal(fixture.download.disabled, false);
  assert.equal(fixture.status.textContent, "training data required");
});
```

- [ ] **Step 2: Run tests to verify RED**

```bash
rtk proxy node --test tests/editor_frontend/export_dialog.test.js
```

Expected: import failure because `export_dialog.js` does not exist.

- [ ] **Step 3: Implement the export view controller**

Create `bindExportDialog` with injected `client`, `saveBlobToFile`, and nodes. It must:

```javascript
const EXPORTS = Object.freeze({
  joblib: {
    filename: "superglm_edited_model.joblib",
    description: "Validated Python model",
    accept: { "application/octet-stream": [".joblib"] }
  },
  xlsx: {
    filename: "superglm_rating_tables.xlsx",
    description: "Excel rating workbook",
    accept: {
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [".xlsx"]
    }
  }
});
```

On format change, replace the default filename only when the prior value was empty or equal to the
other format's default. On download, call `/download_export`, pass the format-specific picker type,
and report `X-SuperGLM-Validation` when present. On kernel save, POST `/export_file`. Maintain one
local `pending` flag, disable both actions during work, escape no server strings into HTML, and
return a `destroy()` method that removes listeners.

- [ ] **Step 4: Replace the HTML dialog and header semantics**

Rename the header button and tooltip:

```html
<button id="exportAction" class="icon-button" type="button" aria-label="Export model or workbook"
  data-popover-title="Export"
  data-popover-body="Download a validated Python model or Excel rating workbook.">
  <svg class="toolbar-icon" viewBox="0 0 24 24" aria-hidden="true">
    <path d="M12 3v12"></path>
    <path d="m7 10 5 5 5-5"></path>
    <path d="M5 20h14"></path>
  </svg>
</button>
```

Replace `saveDialog` with `exportDialog`, a labelled radio group for Python model versus Excel
workbook, filename and directory fields, Download and Save to Kernel Path buttons, Close, and one
`role="status" aria-live="polite"` status region. Keep native `<dialog>` focus behavior.

- [ ] **Step 5: Bind the module from `main.js`**

Remove `saveEditedModel`, `downloadEditedModel`, and format-specific picker assumptions from
`main.js`. Import and bind the controller once with the existing client and a generalized
`saveBlobToFile(blob, filename, fileType)` helper. The helper must treat picker cancellation as a
normal `null` result and fall back to an object-URL download when the picker is unavailable.

- [ ] **Step 6: Style and document the two format choices**

Add the compact format-card layout in `styles/dialogs.css`:

```css
.export-format-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: var(--space-3);
}

.export-format-card {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: var(--space-2);
  align-items: start;
  padding: var(--space-3);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  cursor: pointer;
}

.export-format-card:has(input:checked) {
  border-color: var(--accent);
  background: var(--accent-soft);
}

@media (max-width: 620px) {
  .export-format-grid { grid-template-columns: 1fr; }
}
```

Extend `CONTROL_HELP`/`HELP_SECTIONS` with: "Python model exports are round-trip validated and
prediction-checked when evaluation rows are available. Excel rating workbooks require training or
retained fit data." Update static asset contract tests in `tests/test_editor.py` to assert
`exportAction`, `exportDialog`, `/download_export`, `/export_file`, that exact help copy, and the
absence of `id="saveAction"`.

- [ ] **Step 7: Run frontend and static contract tests**

```bash
rtk npm run check:frontend
rtk pytest tests/test_editor.py -k "frontend or export or application_shell"
```

Expected: all selected checks pass.

- [ ] **Step 8: Commit**

```bash
rtk git add src/superglm/editor/app tests/editor_frontend/export_dialog.test.js tests/test_editor.py
rtk git commit -m "Add editor Export dialog"
```

### Task 7: Prove the real browser and demo show the intended behavior

**Files:**
- Modify: `tests/editor/test_editor_workspace_browser.py`
- Modify: `docs/notebooks/editor_demo.ipynb`
- Modify: `docs/guide/editor.md`
- Modify: `docs/guide/results.md`

- [ ] **Step 1: Write failing browser assertions for both downloads**

Add one browser test that opens Export, selects each format, and captures its response:

```python
def test_export_dialog_downloads_validated_model_and_structured_excel(open_editor_page):
    with open_editor_page(selected_term="territory") as (page, _session):
        page.get_by_role("button", name="Export model or workbook").click()
        dialog = page.get_by_role("dialog", name="Export")
        dialog.get_by_role("radio", name="Python model").check()
        with page.expect_response(lambda r: "/download_export?" in r.url) as model_response:
            dialog.get_by_role("button", name="Download").click()
        assert model_response.value.status == 200
        assert model_response.value.headers["x-superglm-validation"].startswith("artifact")

        dialog.get_by_role("radio", name="Excel rating workbook").check()
        with page.expect_response(lambda r: "/download_export?" in r.url) as excel_response:
            dialog.get_by_role("button", name="Download").click()
        assert excel_response.value.status == 200
        assert excel_response.value.headers["content-type"].startswith(
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
```

Use the test-mode blob-save interception already established by browser fixtures so no native
picker opens. Record chart, summary, term options, and report nodes before export and assert they
are the same DOM nodes afterward.

- [ ] **Step 2: Write a browser assertion for ordered summary semantics**

Open a fixture with an explicit spline-backed ordered term. In the Summary panel assert one
`kind=spline`/chi-square row has a p-value and each displayed ordered level row has a blank p-value
and significance cell. If the shared browser model does not contain such a term, add `age_band`
with `basis=Spline(kind="ps", k=5)` to `tests/editor/conftest.py`.

- [ ] **Step 3: Run browser tests to verify RED, then complete any missing wiring**

```bash
rtk proxy uv run pytest tests/editor/test_editor_workspace_browser.py -k "export_dialog or ordered_summary" --run-browser -q
```

Expected before final wiring: a focused role/response/summary assertion fails. Make only the
frontend or fixture correction identified by that failure, then rerun until both pass.

- [ ] **Step 4: Update user documentation and demo semantics**

Document:

```markdown
- Export > Python model performs a joblib round trip and validates predictions when evaluation
  rows are available.
- Export > Excel rating workbook requires training or retained fit data and contains typed model
  summary and term-inference tables.
- Spline-backed ordered categoricals have one whole-smooth p-value; level rows are effect
  estimates with intervals, not separate tests.
```

In the demo, replace the deprecated step basis with `basis=Spline(kind="ps", k=5)` and replace
`age_band_min_p` calculations with a helper that selects the single smooth row's `wald_p`. Rerun
the notebook's source-contract tests; do not commit regenerated binary outputs unrelated to the
changed cells.

- [ ] **Step 5: Run the focused end-to-end checks**

```bash
rtk pytest tests/test_ordered_categorical_inference.py tests/test_ordered_reference_export.py tests/test_rating_table_export.py tests/test_editor.py
rtk proxy uv run pytest tests/editor/ --run-browser -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
rtk git add tests/editor docs/notebooks/editor_demo.ipynb docs/guide/editor.md docs/guide/results.md
rtk git commit -m "Verify editor export and ordered inference workflows"
```

### Task 8: Full verification and packaging audit

**Files:**
- Modify only if a verification failure identifies an in-scope defect.

- [ ] **Step 1: Run frontend checks**

```bash
rtk npm run check:frontend
```

Expected: TypeScript checking and all Node tests pass.

- [ ] **Step 2: Run Python lint and formatting checks**

```bash
rtk ruff check src/ tests/
rtk proxy uv run ruff format --check src/ tests/
```

Expected: no lint or formatting failures.

- [ ] **Step 3: Run non-browser Python tests**

```bash
rtk proxy uv run pytest tests/ -m "not browser" -q
```

Expected: all selected tests pass.

- [ ] **Step 4: Run the full editor browser suite**

```bash
rtk proxy uv run pytest tests/editor/ tests/test_editor_browser.py --run-browser -q
```

Expected: all browser tests pass.

- [ ] **Step 5: Build and inspect the wheel**

```bash
rtk proxy uv build
rtk proxy sh -c 'wheel=$(ls -t dist/superglm-*.whl | head -n 1); unzip -l "$wheel"' | rtk proxy rg "editor/app|spline_diagnostics|\.superglm-debug"
```

Expected: packaged editor assets include the new export view; no `spline_diagnostics` or
`.superglm-debug` paths appear.

- [ ] **Step 6: Review the final diff and commit any verification-only correction**

```bash
rtk git diff --check
rtk git status --short
rtk git log --oneline -10
```

Expected: clean worktree after intentional commits and no unrelated changes.
