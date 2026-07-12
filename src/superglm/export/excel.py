"""Excel renderer for rating-table export payloads."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import BinaryIO

import pandas as pd

_TERM_COLUMNS = (
    ("Term", "term"),
    ("Group", "group"),
    ("Kind", "kind"),
    ("Estimate", "estimate"),
    ("Std Error", "std_error"),
    ("Statistic", "statistic"),
    ("Statistic Type", "statistic_type"),
    ("P Value", "p_value"),
    ("CI Lower", "ci_lower"),
    ("CI Upper", "ci_upper"),
    ("EDF", "edf"),
    ("Lambda", "smoothing_lambda"),
    ("Active", "active"),
    ("Significance", "significance"),
    ("Warning", "warning"),
)
_TERM_NUMERIC_COLUMNS = frozenset(
    {"Estimate", "Std Error", "Statistic", "P Value", "CI Lower", "CI Upper", "EDF", "Lambda"}
)


def _resolve_workbook_target(
    target: str | PathLike[str] | BinaryIO,
) -> Path | BinaryIO:
    if isinstance(target, str | PathLike):
        out = Path(target)
        out.parent.mkdir(parents=True, exist_ok=True)
        return out
    return target


def _add_excel_table(
    ws,
    *,
    display_name: str,
    min_row: int,
    max_row: int,
    min_col: int,
    max_col: int,
) -> None:
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.table import Table, TableStyleInfo

    reference = f"{get_column_letter(min_col)}{min_row}:{get_column_letter(max_col)}{max_row}"
    table = Table(displayName=display_name, ref=reference)
    table.tableStyleInfo = TableStyleInfo(
        name="TableStyleMedium2",
        showFirstColumn=False,
        showLastColumn=False,
        showRowStripes=True,
        showColumnStripes=False,
    )
    ws.add_table(table)


def _write_summary_sheet(ws, summary) -> None:
    from openpyxl.styles import Alignment, Font

    ws["A1"] = "Model Summary"
    ws["A1"].font = Font(bold=True, size=14)
    ws["A3"] = "Fit and model overview"
    ws["A3"].font = Font(bold=True)

    overview_header_row = 4
    overview_headers = ("Section", "Metric", "Value")
    for column, header in enumerate(overview_headers, start=1):
        ws.cell(row=overview_header_row, column=column, value=header).font = Font(bold=True)
    for row_number, overview_row in enumerate(summary.overview, start=overview_header_row + 1):
        ws.cell(row=row_number, column=1, value=overview_row.section)
        ws.cell(row=row_number, column=2, value=overview_row.metric)
        value_cell = ws.cell(row=row_number, column=3, value=overview_row.value)
        if isinstance(overview_row.value, int) and not isinstance(overview_row.value, bool):
            value_cell.number_format = "0"
        elif isinstance(overview_row.value, float):
            value_cell.number_format = "0.000000"

    overview_end_row = overview_header_row + len(summary.overview)
    _add_excel_table(
        ws,
        display_name="ModelOverview",
        min_row=overview_header_row,
        max_row=overview_end_row,
        min_col=1,
        max_col=len(overview_headers),
    )

    term_header_row = overview_end_row + 3
    for column, (header, _) in enumerate(_TERM_COLUMNS, start=1):
        ws.cell(row=term_header_row, column=column, value=header).font = Font(bold=True)

    term_rows = summary.terms or (None,)
    for row_number, term_row in enumerate(term_rows, start=term_header_row + 1):
        for column, (header, attribute) in enumerate(_TERM_COLUMNS, start=1):
            value = None if term_row is None else getattr(term_row, attribute)
            cell = ws.cell(row=row_number, column=column, value=value)
            if value is not None and header in _TERM_NUMERIC_COLUMNS:
                cell.number_format = "0.000000E+00" if header == "P Value" else "0.000000"

    term_end_row = term_header_row + len(term_rows)
    _add_excel_table(
        ws,
        display_name="TermInference",
        min_row=term_header_row,
        max_row=term_end_row,
        min_col=1,
        max_col=len(_TERM_COLUMNS),
    )

    notes_header_row = term_end_row + 3
    ws.cell(row=notes_header_row, column=1, value="Notes").font = Font(bold=True)
    for row_number, note in enumerate(summary.notes, start=notes_header_row + 1):
        cell = ws.cell(row=row_number, column=1, value=note)
        cell.alignment = Alignment(wrap_text=True, vertical="top")

    ws.freeze_panes = "A5"


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
    target: str | PathLike[str] | BinaryIO,
    *,
    sheet_name: str,
    summary_sheet_name: str,
    impact_sheet_name: str,
) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Font

    out = _resolve_workbook_target(target)
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
    _write_summary_sheet(summary_ws, payload.summary)

    for sheet in wb.worksheets:
        _autosize(sheet)
    summary_ws.column_dimensions["A"].width = 32
    summary_ws.column_dimensions["B"].width = 24
    for column in "CDEFGHIJKLMNO":
        summary_ws.column_dimensions[column].width = min(
            max(summary_ws.column_dimensions[column].width, 14),
            18,
        )
    wb.save(out)
