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
