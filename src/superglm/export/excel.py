"""Excel renderer for rating-table export payloads."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _plain_summary_lines(summary) -> list[str]:
    """Temporarily flatten a typed summary for the legacy one-cell renderer."""
    lines = ["SuperGLM Results"]
    current_section = None
    for row in summary.overview:
        if row.section != current_section:
            lines.extend(["", row.section])
            current_section = row.section
        value = "" if row.value is None else str(row.value)
        lines.append(f"{row.metric}: {value}")

    lines.extend(["", "Terms", "term | kind | estimate | statistic | p-value"])
    for row in summary.terms:
        values = (row.term, row.kind, row.estimate, row.statistic, row.p_value)
        lines.append(" | ".join("" if value is None else str(value) for value in values))

    if summary.notes:
        lines.extend(["", "Notes", *summary.notes])
    return lines


def _write_dataframe(ws, df: pd.DataFrame, start_row: int, start_col: int) -> tuple[int, int]:
    from openpyxl.styles import Font

    for c, column in enumerate(df.columns, start=start_col):
        cell = ws.cell(row=start_row, column=c, value=column)
        cell.font = Font(bold=True)
    for r, row in enumerate(df.itertuples(index=False), start=start_row + 1):
        for c, value in enumerate(row, start=start_col):
            ws.cell(row=r, column=c, value=value)
    return start_row + len(df), start_col + len(df.columns) - 1


def _autosize(ws) -> None:
    from openpyxl.utils import get_column_letter

    for column_cells in ws.columns:
        letter = get_column_letter(column_cells[0].column)
        max_length = max(
            len(str(cell.value)) if cell.value is not None else 0 for cell in column_cells
        )
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
        title_cell = ws.cell(row=5, column=start_col, value=block.name)
        title_cell.font = Font(bold=True)
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

    interaction_row = max_main_row + 3
    for block in payload.interactions:
        title_cell = ws.cell(row=interaction_row, column=1, value=block.name)
        title_cell.font = Font(bold=True)
        end_row, _ = _write_dataframe(ws, block.table, interaction_row + 2, 1)
        for row in ws.iter_rows(
            min_row=interaction_row + 3,
            max_row=end_row,
            min_col=2,
            max_col=block.table.shape[1],
        ):
            for cell in row:
                cell.number_format = "0.000000"
        interaction_row = end_row + 3

    impact_ws = wb.create_sheet(impact_sheet_name)
    _write_dataframe(impact_ws, payload.discretization_impact, 1, 1)

    summary_ws = wb.create_sheet(summary_sheet_name)
    for row, line in enumerate(_plain_summary_lines(payload.summary), start=1):
        cell = summary_ws.cell(row=row, column=1, value=line)
        cell.font = Font(name="Consolas")

    for sheet in wb.worksheets:
        _autosize(sheet)
    summary_ws.column_dimensions["A"].width = 140
    wb.save(out)
