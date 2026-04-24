# Rating Table Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `SuperGLM.export_rating_tables()` and `superglm.export.export_rating_tables()` with Excel output, selected-bin spline rating tables, and a discretization impact sweep.

**Architecture:** Add a focused `superglm.export` package that builds a renderer-independent payload, then renders that payload to Excel with `openpyxl`. Model methods remain thin facades in `model/api.py`, matching existing model operation patterns. Continuous main-effect tables are sourced from `discretization_impact(..., n_bins=selected)`, while the impact sweep is diagnostic-only.

**Tech Stack:** Python 3.10+, pandas, numpy, openpyxl, pytest, existing `SuperGLM.term_inference()` and `SuperGLM.discretization_impact()`.

---

## File Structure

- Create `src/superglm/export/__init__.py`
  - Public export package surface.
- Create `src/superglm/export/rating_tables.py`
  - Payload dataclasses, validation, main-effect extraction, interaction extraction, impact sweep, and `export_rating_tables()`.
- Create `src/superglm/export/excel.py`
  - Excel workbook renderer only.
- Modify `src/superglm/model/api.py`
  - Add `SuperGLM.export_rating_tables()` thin facade.
- Modify `src/superglm/__init__.py`
  - Re-export `export_rating_tables`.
- Modify `pyproject.toml`
  - Add `openpyxl>=3.1`.
- Create `tests/test_rating_table_export.py`
  - Unit and workbook tests for the new public behavior.

---

### Task 1: Public API Skeleton And Dependency

**Files:**
- Create: `tests/test_rating_table_export.py`
- Create: `src/superglm/export/__init__.py`
- Create: `src/superglm/export/rating_tables.py`
- Create: `src/superglm/export/excel.py`
- Modify: `src/superglm/model/api.py`
- Modify: `src/superglm/__init__.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Write the failing public API test**

Add this to `tests/test_rating_table_export.py`:

```python
import numpy as np
import pandas as pd

from superglm import Categorical, Spline, SuperGLM, export_rating_tables


def _fit_export_model():
    rng = np.random.default_rng(123)
    n = 300
    X = pd.DataFrame(
        {
            "age": rng.uniform(18, 80, n),
            "region": rng.choice(["A", "B", "C"], n),
        }
    )
    eta = -1.0 + 0.15 * np.sin(X["age"].to_numpy() / 8.0) + 0.2 * (X["region"] == "B")
    y = rng.poisson(np.exp(eta)).astype(float)
    w = rng.uniform(0.5, 2.0, n)
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={"age": Spline(n_knots=8), "region": Categorical(base="first")},
    )
    model.fit(X, y, sample_weight=w)
    return model, X, y, w


def test_public_export_api_exists(tmp_path):
    model, X, y, w = _fit_export_model()
    output = tmp_path / "rating_tables.xlsx"

    result_path = export_rating_tables(model, output, X, y, sample_weight=w)
    method_path = model.export_rating_tables(tmp_path / "rating_tables_method.xlsx", X, y, sample_weight=w)

    assert result_path == output
    assert output.exists()
    assert method_path.exists()
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
rtk uv run pytest tests/test_rating_table_export.py::test_public_export_api_exists -q
```

Expected: FAIL with `ImportError` for `export_rating_tables` or `AttributeError` on `model.export_rating_tables`.

- [ ] **Step 3: Add minimal dependency and API shell**

Modify `pyproject.toml` dependencies:

```toml
    "openpyxl>=3.1",
```

Create `src/superglm/export/__init__.py`:

```python
"""Rating table export helpers."""

from superglm.export.rating_tables import export_rating_tables

__all__ = ["export_rating_tables"]
```

Create `src/superglm/export/rating_tables.py`:

```python
"""Structured rating-table export for fitted SuperGLM models."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from numpy.typing import NDArray

if TYPE_CHECKING:
    from superglm.model import SuperGLM


def _resolve_format(file_path: str | Path, format: str | None) -> str:
    if format is not None:
        fmt = format.lower().lstrip(".")
    else:
        suffix = Path(file_path).suffix.lower()
        fmt = suffix.lstrip(".")
    if fmt in {"xlsx", "xlsm", "excel"}:
        return "excel"
    raise ValueError(f"Unsupported rating table export format: {format or Path(file_path).suffix!r}")


def export_rating_tables(
    model: SuperGLM,
    file_path: str | Path,
    X: pd.DataFrame,
    y: NDArray,
    sample_weight: NDArray | None = None,
    *,
    n_bins: int = 150,
    impact_bins: tuple[int, ...] = (20, 50, 100, 200, 250),
    bin_strategy: str = "exposure_quantile",
    format: str | None = None,
    sheet_name: str = "Rating Tables",
    summary_sheet_name: str = "Model Summary",
    impact_sheet_name: str = "Discretization Impact",
    centering: str = "native",
) -> Path:
    del X, y, sample_weight, n_bins, impact_bins, bin_strategy, centering
    if model._result is None:
        raise RuntimeError("Model must be fitted before exporting rating tables.")
    out = Path(file_path)
    fmt = _resolve_format(out, format)
    if fmt != "excel":
        raise ValueError(f"Unsupported rating table export format: {fmt!r}")
    from superglm.export.excel import write_rating_table_workbook

    payload = {
        "base_relativity": float(__import__("numpy").exp(model.result.intercept)),
        "main_effects": [],
        "interactions": [],
        "impact": pd.DataFrame(),
        "summary_lines": str(model.summary(detail="compact")).splitlines(),
    }
    write_rating_table_workbook(
        payload,
        out,
        sheet_name=sheet_name,
        summary_sheet_name=summary_sheet_name,
        impact_sheet_name=impact_sheet_name,
    )
    return out
```

Create `src/superglm/export/excel.py`:

```python
"""Excel renderer for rating-table export payloads."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def write_rating_table_workbook(
    payload: dict[str, Any],
    file_path: str | Path,
    *,
    sheet_name: str,
    summary_sheet_name: str,
    impact_sheet_name: str,
) -> None:
    from openpyxl import Workbook

    out = Path(file_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    ws = wb.active
    ws.title = sheet_name
    ws["A2"] = "Base"
    ws["C2"] = payload["base_relativity"]
    impact_ws = wb.create_sheet(impact_sheet_name)
    impact_ws["A1"] = "n_bins"
    summary_ws = wb.create_sheet(summary_sheet_name)
    for row, line in enumerate(payload["summary_lines"], start=1):
        summary_ws.cell(row=row, column=1, value=line)
    wb.save(out)
```

Modify `src/superglm/model/api.py` near `discretization_impact()`:

```python
    def export_rating_tables(
        self,
        file_path,
        X: pd.DataFrame,
        y: NDArray,
        sample_weight: NDArray | None = None,
        **kwargs,
    ):
        """Export deployment rating tables for the fitted model."""
        from superglm.export import export_rating_tables

        return export_rating_tables(self, file_path, X, y, sample_weight=sample_weight, **kwargs)
```

Modify `src/superglm/__init__.py`:

```python
from superglm.export import export_rating_tables
```

and add `"export_rating_tables"` to `__all__`.

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
rtk uv sync --extra dev
rtk uv run pytest tests/test_rating_table_export.py::test_public_export_api_exists -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
rtk git add pyproject.toml uv.lock src/superglm/export src/superglm/model/api.py src/superglm/__init__.py tests/test_rating_table_export.py
rtk git commit -m "feat: add rating table export API shell"
```

---

### Task 2: Selected-Bin Continuous Tables And Impact Sweep

**Files:**
- Modify: `tests/test_rating_table_export.py`
- Modify: `src/superglm/export/rating_tables.py`

- [ ] **Step 1: Write failing tests for selected bins and impact bins**

Append:

```python
from superglm.export.rating_tables import build_rating_table_payload


def test_default_selected_bins_are_150():
    model, X, y, w = _fit_export_model()
    payload = build_rating_table_payload(model, X, y, sample_weight=w)

    age_block = next(block for block in payload.main_effects if block.name == "age")

    assert payload.selected_n_bins == 150
    assert len(age_block.table) <= 150
    assert {"Level", "Relativity", "Weight"} <= set(age_block.table.columns)


def test_default_impact_sweep_bins():
    model, X, y, w = _fit_export_model()
    payload = build_rating_table_payload(model, X, y, sample_weight=w)

    assert sorted(payload.discretization_impact["n_bins"].unique().tolist()) == [
        20,
        50,
        100,
        200,
        250,
    ]
    assert set(payload.discretization_impact["feature"]) == {"age"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
rtk uv run pytest tests/test_rating_table_export.py::test_default_selected_bins_are_150 tests/test_rating_table_export.py::test_default_impact_sweep_bins -q
```

Expected: FAIL because `build_rating_table_payload` does not exist.

- [ ] **Step 3: Implement payload dataclasses and continuous extraction**

Replace `src/superglm/export/rating_tables.py` with a structured implementation containing:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from superglm.features.polynomial import Polynomial
from superglm.features.spline import _SplineBase

if TYPE_CHECKING:
    from superglm.model import SuperGLM


@dataclass(frozen=True)
class RatingTableBlock:
    name: str
    kind: str
    table: pd.DataFrame


@dataclass(frozen=True)
class InteractionTableBlock:
    name: str
    table: pd.DataFrame


@dataclass(frozen=True)
class RatingTablePayload:
    base_relativity: float
    selected_n_bins: int
    main_effects: list[RatingTableBlock]
    interactions: list[InteractionTableBlock]
    discretization_impact: pd.DataFrame
    summary_lines: list[str]


def _continuous_features(model: SuperGLM) -> list[str]:
    return [
        name
        for name in model._feature_order
        if isinstance(model._specs.get(name), _SplineBase | Polynomial)
    ]


def _format_interval(left: float, right: float) -> str:
    return f"[{left:.10g}, {right:.10g})"


def _continuous_block(name: str, table: pd.DataFrame) -> RatingTableBlock:
    out = pd.DataFrame(
        {
            "Level": [
                _format_interval(float(row.bin_from), float(row.bin_to))
                for row in table.itertuples(index=False)
            ],
            "Relativity": table["relativity"].astype(float).to_numpy(),
            "Weight": table["sample_weight"].astype(float).to_numpy(),
        }
    )
    return RatingTableBlock(name=name, kind="continuous", table=out)


def _impact_sweep(
    model: SuperGLM,
    X: pd.DataFrame,
    y: NDArray,
    sample_weight: NDArray | None,
    *,
    impact_bins: tuple[int, ...],
    bin_strategy: str,
    features: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    if not features:
        return pd.DataFrame(
            columns=[
                "n_bins",
                "feature",
                "actual_bins",
                "deviance_original",
                "deviance_discretized",
                "deviance_change",
                "deviance_change_pct",
                "mean_abs_prediction_change_pct",
                "max_abs_prediction_change_pct",
                "prediction_correlation",
            ]
        )
    for n_bins in impact_bins:
        result = model.discretization_impact(
            X,
            y,
            sample_weight=sample_weight,
            n_bins=int(n_bins),
            bin_strategy=bin_strategy,
            features=features,
        )
        for feature, table in result.tables.items():
            row = {
                "n_bins": int(n_bins),
                "feature": feature,
                "actual_bins": int(len(table)),
            }
            row.update(result.metrics)
            rows.append(row)
    return pd.DataFrame(rows)


def build_rating_table_payload(
    model: SuperGLM,
    X: pd.DataFrame,
    y: NDArray,
    sample_weight: NDArray | None = None,
    *,
    n_bins: int = 150,
    impact_bins: tuple[int, ...] = (20, 50, 100, 200, 250),
    bin_strategy: str = "exposure_quantile",
    centering: str = "native",
) -> RatingTablePayload:
    del centering
    if model._result is None:
        raise RuntimeError("Model must be fitted before exporting rating tables.")
    y_arr = np.asarray(y, dtype=np.float64)
    if len(X) != len(y_arr):
        raise ValueError("X and y must have the same length.")
    if sample_weight is not None and len(sample_weight) != len(X):
        raise ValueError("sample_weight must have the same length as X.")

    continuous = _continuous_features(model)
    selected = (
        model.discretization_impact(
            X,
            y_arr,
            sample_weight=sample_weight,
            n_bins=n_bins,
            bin_strategy=bin_strategy,
            features=continuous,
        )
        if continuous
        else None
    )

    main_effects: list[RatingTableBlock] = []
    if selected is not None:
        for name in model._feature_order:
            if name in selected.tables:
                main_effects.append(_continuous_block(name, selected.tables[name]))

    impact = _impact_sweep(
        model,
        X,
        y_arr,
        sample_weight,
        impact_bins=impact_bins,
        bin_strategy=bin_strategy,
        features=continuous,
    )
    return RatingTablePayload(
        base_relativity=float(np.exp(model.result.intercept)),
        selected_n_bins=int(n_bins),
        main_effects=main_effects,
        interactions=[],
        discretization_impact=impact,
        summary_lines=str(model.summary(detail="compact")).splitlines(),
    )
```

Update `export_rating_tables()` to call `build_rating_table_payload(...)` and pass the dataclass to the Excel renderer.

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
rtk uv run pytest tests/test_rating_table_export.py::test_default_selected_bins_are_150 tests/test_rating_table_export.py::test_default_impact_sweep_bins -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
rtk git add src/superglm/export/rating_tables.py tests/test_rating_table_export.py
rtk git commit -m "feat: build selected-bin rating table payload"
```

---

### Task 3: Categorical And Numeric Main-Effect Blocks

**Files:**
- Modify: `tests/test_rating_table_export.py`
- Modify: `src/superglm/export/rating_tables.py`

- [ ] **Step 1: Write failing tests for categorical and numeric blocks**

Extend `_fit_export_model()` to include a numeric feature:

```python
X["score"] = rng.normal(0.0, 1.0, n)
eta = eta + 0.05 * X["score"].to_numpy()
features={"age": Spline(n_knots=8), "region": Categorical(base="first"), "score": Numeric()}
```

Add `Numeric` to imports and append:

```python
from superglm import Numeric


def test_categorical_and_numeric_blocks_are_exported():
    model, X, y, w = _fit_export_model()
    payload = build_rating_table_payload(model, X, y, sample_weight=w)

    names = [block.name for block in payload.main_effects]
    assert names == ["age", "region", "score"]

    region = next(block for block in payload.main_effects if block.name == "region")
    score = next(block for block in payload.main_effects if block.name == "score")

    assert set(region.table["Level"]) == {"A", "B", "C"}
    assert np.isclose(region.table["Weight"].sum(), w.sum())
    assert score.table["Level"].tolist() == ["per_unit"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
rtk uv run pytest tests/test_rating_table_export.py::test_categorical_and_numeric_blocks_are_exported -q
```

Expected: FAIL because only continuous blocks are exported.

- [ ] **Step 3: Implement categorical and numeric extraction**

Add helpers:

```python
from superglm.features.categorical import Categorical
from superglm.features.numeric import Numeric
from superglm.features.ordered_categorical import OrderedCategorical


def _weights_by_level(X: pd.DataFrame, name: str, levels: list[str], sample_weight: NDArray | None) -> np.ndarray:
    weights = np.ones(len(X), dtype=np.float64) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    grouped = (
        pd.DataFrame({"level": X[name].astype(str), "weight": weights})
        .groupby("level", sort=False)["weight"]
        .sum()
    )
    return np.array([float(grouped.get(level, 0.0)) for level in levels], dtype=np.float64)


def _categorical_block(model: SuperGLM, X: pd.DataFrame, name: str, sample_weight: NDArray | None, centering: str) -> RatingTableBlock:
    ti = model.term_inference(name, with_se=False, centering=centering)
    levels = list(ti.levels or [])
    return RatingTableBlock(
        name=name,
        kind="categorical",
        table=pd.DataFrame(
            {
                "Level": levels,
                "Relativity": np.asarray(ti.relativity, dtype=np.float64),
                "Weight": _weights_by_level(X, name, levels, sample_weight),
            }
        ),
    )


def _numeric_block(model: SuperGLM, name: str, centering: str) -> RatingTableBlock:
    ti = model.term_inference(name, with_se=False, centering=centering)
    return RatingTableBlock(
        name=name,
        kind="numeric",
        table=pd.DataFrame(
            {
                "Level": ["per_unit"],
                "Relativity": np.asarray(ti.relativity, dtype=np.float64),
                "Weight": [0.0],
            }
        ),
    )
```

Update `build_rating_table_payload()` loop over `model._feature_order` so each feature appends exactly one block in model order:

```python
for name in model._feature_order:
    spec = model._specs[name]
    if selected is not None and name in selected.tables:
        main_effects.append(_continuous_block(name, selected.tables[name]))
    elif isinstance(spec, Categorical | OrderedCategorical):
        main_effects.append(_categorical_block(model, X, name, sample_weight, centering))
    elif isinstance(spec, Numeric):
        main_effects.append(_numeric_block(model, name, centering))
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
rtk uv run pytest tests/test_rating_table_export.py::test_categorical_and_numeric_blocks_are_exported -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
rtk git add src/superglm/export/rating_tables.py tests/test_rating_table_export.py
rtk git commit -m "feat: export categorical and numeric rating blocks"
```

---

### Task 4: Excel Workbook Layout

**Files:**
- Modify: `tests/test_rating_table_export.py`
- Modify: `src/superglm/export/excel.py`
- Modify: `src/superglm/export/rating_tables.py`

- [ ] **Step 1: Write failing workbook layout test**

Append:

```python
from openpyxl import load_workbook


def test_excel_workbook_layout(tmp_path):
    model, X, y, w = _fit_export_model()
    output = tmp_path / "tables.xlsx"

    model.export_rating_tables(output, X, y, sample_weight=w, n_bins=20)

    wb = load_workbook(output, data_only=True)
    assert wb.sheetnames == ["Rating Tables", "Discretization Impact", "Model Summary"]
    ws = wb["Rating Tables"]
    assert ws["A2"].value == "Base"
    assert isinstance(ws["C2"].value, float)
    assert ws["A5"].value == "age"
    assert ws["A7"].value == "Level"
    assert ws["B7"].value == "Relativity"
    assert ws["C7"].value == "Weight"
    assert ws["D5"].value == "region"
    assert ws["G5"].value == "score"

    impact_ws = wb["Discretization Impact"]
    headers = [impact_ws.cell(row=1, column=i).value for i in range(1, 11)]
    assert headers[:3] == ["n_bins", "feature", "actual_bins"]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
rtk uv run pytest tests/test_rating_table_export.py::test_excel_workbook_layout -q
```

Expected: FAIL because the Excel renderer only writes the base and empty impact header.

- [ ] **Step 3: Implement Excel layout**

Replace `src/superglm/export/excel.py` with:

```python
from __future__ import annotations

from pathlib import Path

import pandas as pd


def _write_dataframe(ws, df: pd.DataFrame, start_row: int, start_col: int) -> tuple[int, int]:
    for c, column in enumerate(df.columns, start=start_col):
        ws.cell(row=start_row, column=c, value=column)
        ws.cell(row=start_row, column=c).font = __import__("openpyxl").styles.Font(bold=True)
    for r, row in enumerate(df.itertuples(index=False), start=start_row + 1):
        for c, value in enumerate(row, start=start_col):
            ws.cell(row=r, column=c, value=value)
    return start_row + len(df), start_col + len(df.columns) - 1


def _autosize(ws) -> None:
    from openpyxl.utils import get_column_letter

    for column_cells in ws.columns:
        letter = get_column_letter(column_cells[0].column)
        max_length = max(len(str(cell.value)) if cell.value is not None else 0 for cell in column_cells)
        ws.column_dimensions[letter].width = min(max(max_length + 2, 12), 36)


def write_rating_table_workbook(
    payload,
    file_path: str | Path,
    *,
    sheet_name: str,
    summary_sheet_name: str,
    impact_sheet_name: str,
) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Font

    out = Path(file_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    ws = wb.active
    ws.title = sheet_name
    ws.freeze_panes = "A8"

    ws["A2"] = "Base"
    ws["A2"].font = Font(bold=True)
    ws["C2"] = float(payload.base_relativity)
    ws["C2"].number_format = "0.000000"

    max_main_row = 8
    for idx, block in enumerate(payload.main_effects):
        start_col = 1 + idx * 3
        ws.cell(row=5, column=start_col, value=block.name)
        ws.cell(row=5, column=start_col).font = Font(bold=True)
        end_row, _ = _write_dataframe(ws, block.table, 7, start_col)
        max_main_row = max(max_main_row, end_row)

    for row in ws.iter_rows():
        for cell in row:
            if cell.value is None:
                continue
            if cell.column % 3 == 2:
                cell.number_format = "0.000000"
            if cell.column % 3 == 0:
                cell.number_format = "#,##0.00"

    impact_ws = wb.create_sheet(impact_sheet_name)
    _write_dataframe(impact_ws, payload.discretization_impact, 1, 1)

    summary_ws = wb.create_sheet(summary_sheet_name)
    for row, line in enumerate(payload.summary_lines, start=1):
        summary_ws.cell(row=row, column=1, value=line)
        summary_ws.cell(row=row, column=1).font = Font(name="Consolas")

    for sheet in wb.worksheets:
        _autosize(sheet)
    summary_ws.column_dimensions["A"].width = 140
    wb.save(out)
```

Update `export_rating_tables()` to pass the dataclass payload rather than a dict.

- [ ] **Step 4: Run workbook test**

Run:

```bash
rtk uv run pytest tests/test_rating_table_export.py::test_excel_workbook_layout -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
rtk git add src/superglm/export/excel.py src/superglm/export/rating_tables.py tests/test_rating_table_export.py
rtk git commit -m "feat: render rating tables to Excel"
```

---

### Task 5: Interaction Table Placement

**Files:**
- Modify: `tests/test_rating_table_export.py`
- Modify: `src/superglm/export/rating_tables.py`
- Modify: `src/superglm/export/excel.py`

- [ ] **Step 1: Write failing interaction placement test**

Add a categorical x categorical interaction fixture inside the test:

```python
def test_interactions_start_two_blank_rows_below_main_effects(tmp_path):
    rng = np.random.default_rng(321)
    n = 400
    X = pd.DataFrame(
        {
            "region": rng.choice(["A", "B", "C"], n),
            "type": rng.choice(["X", "Y"], n),
        }
    )
    y = rng.poisson(1.0 + 0.2 * (X["region"] == "B")).astype(float)
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={"region": Categorical(base="first"), "type": Categorical(base="first")},
        interactions=[("region", "type")],
    )
    model.fit(X, y)
    output = tmp_path / "interaction.xlsx"

    model.export_rating_tables(output, X, y)

    ws = load_workbook(output, data_only=True)["Rating Tables"]
    main_last_row = 8 + max(len(block.table) for block in build_rating_table_payload(model, X, y).main_effects) - 1
    interaction_title_row = main_last_row + 3
    assert ws.cell(row=interaction_title_row - 1, column=1).value is None
    assert ws.cell(row=interaction_title_row, column=1).value == "region:type"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
rtk uv run pytest tests/test_rating_table_export.py::test_interactions_start_two_blank_rows_below_main_effects -q
```

Expected: FAIL because interactions are not included.

- [ ] **Step 3: Implement categorical interaction blocks and placement**

Add to `rating_tables.py`:

```python
def _interaction_blocks(model: SuperGLM) -> list[InteractionTableBlock]:
    blocks: list[InteractionTableBlock] = []
    for name in model._interaction_order:
        raw = model.reconstruct_feature(name)
        if "pairs" not in raw:
            raise NotImplementedError(f"Interaction {name!r} is not yet exportable as a rating table.")
        levels1 = raw["levels1"]
        levels2 = raw["levels2"]
        rows = []
        for level1 in levels1:
            row = {raw.get("parent1", "Level"): level1}
            for level2 in levels2:
                key = f"{level1}:{level2}"
                row[level2] = float(raw["relativities"].get(key, 1.0))
            rows.append(row)
        blocks.append(InteractionTableBlock(name=name, table=pd.DataFrame(rows)))
    return blocks
```

Set `interactions=_interaction_blocks(model)` in `RatingTablePayload`.

Update `excel.py` after main effects:

```python
interaction_row = max_main_row + 3
for block in payload.interactions:
    ws.cell(row=interaction_row, column=1, value=block.name)
    ws.cell(row=interaction_row, column=1).font = Font(bold=True)
    end_row, _ = _write_dataframe(ws, block.table, interaction_row + 2, 1)
    interaction_row = end_row + 3
```

- [ ] **Step 4: Run interaction test**

Run:

```bash
rtk uv run pytest tests/test_rating_table_export.py::test_interactions_start_two_blank_rows_below_main_effects -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
rtk git add src/superglm/export/rating_tables.py src/superglm/export/excel.py tests/test_rating_table_export.py
rtk git commit -m "feat: export categorical interaction rating tables"
```

---

### Task 6: Error Handling And Format Validation

**Files:**
- Modify: `tests/test_rating_table_export.py`
- Modify: `src/superglm/export/rating_tables.py`

- [ ] **Step 1: Write failing validation tests**

Append:

```python
import pytest


def test_export_rejects_unsupported_format(tmp_path):
    model, X, y, w = _fit_export_model()
    with pytest.raises(ValueError, match="Unsupported rating table export format"):
        model.export_rating_tables(tmp_path / "tables.csv", X, y, sample_weight=w)


def test_export_validates_lengths(tmp_path):
    model, X, y, w = _fit_export_model()
    with pytest.raises(ValueError, match="same length"):
        model.export_rating_tables(tmp_path / "tables.xlsx", X.iloc[:-1], y, sample_weight=w)
```

- [ ] **Step 2: Run tests to verify failures**

Run:

```bash
rtk uv run pytest tests/test_rating_table_export.py::test_export_rejects_unsupported_format tests/test_rating_table_export.py::test_export_validates_lengths -q
```

Expected: format test may pass from Task 1; length test should fail if validation has gaps.

- [ ] **Step 3: Tighten validation**

In `build_rating_table_payload()`, keep explicit length checks:

```python
if len(X) != len(y_arr):
    raise ValueError("X and y must have the same length.")
if sample_weight is not None and len(sample_weight) != len(X):
    raise ValueError("sample_weight must have the same length as X.")
```

In `export_rating_tables()`, call `_resolve_format()` before payload construction:

```python
fmt = _resolve_format(file_path, format)
payload = build_rating_table_payload(...)
if fmt == "excel":
    ...
```

- [ ] **Step 4: Run validation tests**

Run:

```bash
rtk uv run pytest tests/test_rating_table_export.py::test_export_rejects_unsupported_format tests/test_rating_table_export.py::test_export_validates_lengths -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
rtk git add src/superglm/export/rating_tables.py tests/test_rating_table_export.py
rtk git commit -m "test: cover rating table export validation"
```

---

### Task 7: Final Verification And Documentation Touch

**Files:**
- Modify: `docs/guide/results.md`
- Modify: `docs/api/model.md` if needed by existing docs style.

- [ ] **Step 1: Add a short docs example**

In `docs/guide/results.md`, add:

```markdown
## Rating table export

Use `export_rating_tables()` to create deployment-oriented tables with binned spline effects:

```python
model.export_rating_tables(
    "rating_tables.xlsx",
    X_train,
    y_train,
    sample_weight=exposure_train,
    n_bins=150,
)
```

The workbook includes selected-bin rating tables, a discretization impact sweep for `20, 50, 100, 200, 250` bins, and the model summary.
```

- [ ] **Step 2: Run targeted tests**

Run:

```bash
rtk uv run pytest tests/test_rating_table_export.py -q
```

Expected: PASS.

- [ ] **Step 3: Run lint**

Run:

```bash
rtk uv run ruff check src/superglm/export src/superglm/model/api.py src/superglm/__init__.py tests/test_rating_table_export.py
```

Expected: PASS.

- [ ] **Step 4: Run formatting check/fix**

Run:

```bash
rtk uv run ruff format src/superglm/export src/superglm/model/api.py src/superglm/__init__.py tests/test_rating_table_export.py
rtk uv run ruff check src/superglm/export src/superglm/model/api.py src/superglm/__init__.py tests/test_rating_table_export.py
```

Expected: formatter completes and lint passes.

- [ ] **Step 5: Run full test suite**

Run:

```bash
rtk uv run pytest tests/ -q
```

Expected: full suite passes. Baseline before implementation was `2005 passed, 99 skipped`.

- [ ] **Step 6: Commit docs and final cleanup**

Run:

```bash
rtk git add docs/guide/results.md
rtk git commit -m "docs: add rating table export example"
```

If formatting changed files after the prior commits, include those files in the closest relevant commit or make a final cleanup commit:

```bash
rtk git add src/superglm/export src/superglm/model/api.py src/superglm/__init__.py tests/test_rating_table_export.py pyproject.toml uv.lock
rtk git commit -m "chore: format rating table export"
```

---

## Self-Review

Spec coverage:

- Public API: covered by Tasks 1 and 6.
- Selected `n_bins=150`: covered by Task 2.
- Impact sweep `20, 50, 100, 200, 250`: covered by Task 2.
- Continuous tables from `discretization_impact()`: covered by Task 2.
- Categorical/numeric blocks: covered by Task 3.
- Excel sheets and layout: covered by Task 4.
- Interaction placement two blank rows below one-dimensional blocks: covered by Task 5.
- Validation and unsupported formats: covered by Task 6.
- Documentation and verification: covered by Task 7.

Placeholder scan: no deferred-work markers remain in the plan. Unsupported interaction types deliberately raise `NotImplementedError`, matching the design.

Type consistency: the plan consistently uses `RatingTablePayload`, `RatingTableBlock`, `InteractionTableBlock`, `build_rating_table_payload()`, and `export_rating_tables()`.
