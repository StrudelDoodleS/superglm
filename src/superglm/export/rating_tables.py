"""Structured rating-table export for fitted SuperGLM models."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
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
    raise ValueError(
        f"Unsupported rating table export format: {format or Path(file_path).suffix!r}"
    )


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
        "base_relativity": float(np.exp(model.result.intercept)),
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
