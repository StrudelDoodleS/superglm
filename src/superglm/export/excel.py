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

# Main-effect blocks sit on a fixed three-column stride and the number-format
# loop below keys on ``cell.column % 3``.  Both are named here rather than
# repeated as bare 3s so a future widening has one place to change.
_MAIN_EFFECT_BLOCK_STRIDE = 3
_MAIN_EFFECT_TITLE_ROW = 5
_MAIN_EFFECT_NOTE_ROW = 6
_MAIN_EFFECT_HEADER_ROW = 7

# The piecewise block's two numeric columns are re-formatted after the global
# loop.  ``Log relativity`` lands on ``column % 3 == 0`` and would otherwise
# render at two decimal places: the value stored in the cell stays exact, but a
# human reading or copy-pasting the sheet would see ``0.00``, which defeats the
# entire purpose of publishing that column.
_PIECEWISE_NUMBER_FORMAT = "0.000000000000"

# The base relativity cell, for the same reason and re-applied at the same
# point.  ``C2`` is column 3, so the global loop's ``column % 3 == 0`` arm
# claims it and renders the base at two decimal places; the value stored stays
# exact, so no reader of the file object notices, but a human reading the sheet
# does.  It is the one cell that multiplies EVERY row of the tariff, and under
# ``centering="mean"`` it additionally carries the whole transferred centering
# constant, so a reader rating off the displayed number is uniformly wrong.
# Measured on a two-term Poisson fit: base 0.3719954211385351 displays as 0.37,
# a 5.4e-03 relative error on every risk; 0.4027922135365106 -> 0.40 (6.9e-03)
# in the mean-centered export of the same model.
#
# ``General`` rather than a fixed-decimal format, which is where this differs
# from the piecewise columns.  Those hold log relativities, which are bounded
# by the tariff's own design, so twelve decimals is always enough.  The base is
# ``exp(intercept)`` and its magnitude is not bounded by anything: a low-
# frequency Poisson fit puts it near 1e-6, where twelve decimals leaves seven
# significant digits.  ``General`` carries eleven significant digits at every
# magnitude, which is more than the ten this module already prints bin edges
# and axis values at -- the base should not be the least precisely stated
# number on a sheet whose other public strings are ``.10g``.
_BASE_RELATIVITY_CELL = "C2"
_BASE_RELATIVITY_NUMBER_FORMAT = "General"
# Significant digits Excel's ``General`` renders, given a column wide enough to
# hold them -- ``_autosize`` sizes from ``str(value)``, so it always is.
_GENERAL_SIGNIFICANT_DIGITS = 11


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


def _piecewise_interpolation_note(
    table: pd.DataFrame, extrapolation: str, centering_shift: float = 0.0
) -> str:
    """The interpolation and extrapolation rule for one piecewise block.

    Derived from the block's own printed columns, so a reader can check the
    stated slopes against the two end rows rather than take them on trust.

    The slopes are written at full round-trip precision on purpose.  The whole
    claim of a piecewise rating table is that the workbook reproduces the model
    exactly; a boundary slope rounded for readability would put a small,
    silent discrepancy back into the tariff at exactly the rows -- the ones
    outside the rated range -- where nobody would look for it.

    That exactness claim holds in both centerings.  ``centering="mean"``
    shifts this block by a constant so its relativities have geometric mean 1,
    and the exported base relativity carries the same constant back, so the
    base-times-blocks product is the same number either way.  The note states
    the constant it was shifted by, because a reader holding the two workbooks
    side by side can otherwise only see that the columns disagree.

    The out-of-range rule is the term's ``extrapolation`` parameter, and it is
    stated here rather than assumed because both directions exist in practice:
    holding flat beyond the outermost knots is the default (the library's
    splines hold flat too), extending the boundary segments is the stated
    alternative.  What ``lower``/``upper`` buy is that the knots sit where the
    tariff's rated range says they should.
    """
    knots = [float(value) for value in table.iloc[:, 0]]
    interpolate = (
        "Interpolate linearly on Log relativity (equivalently, geometrically on Relativity). "
    )
    # The exactness claim, stated in the sheet itself, with the centering
    # constant named so a reader can check the two exports against each other.
    # Printed at full round-trip precision for the same reason the boundary
    # slopes are: it is the number that reconciles this block with the base.
    scope = (
        " Exact reproduction of the fitted model holds in either centering."
        if centering_shift == 0.0
        else (
            " Exact reproduction of the fitted model holds in either centering: "
            f"these Log relativity values are the fitted ones less {centering_shift}, "
            "and the base relativity carries that constant back."
        )
    )
    if extrapolation == "extend":
        log_relativity = [float(value) for value in table["Log relativity"]]
        slope_low = (log_relativity[1] - log_relativity[0]) / (knots[1] - knots[0])
        slope_high = (log_relativity[-1] - log_relativity[-2]) / (knots[-1] - knots[-2])
        return (
            interpolate
            + (
                "Beyond the tabulated range the boundary segments continue: "
                f"below {knots[0]} use slope {slope_low}; "
                f"above {knots[-1]} use slope {slope_high}."
            )
            + scope
        )
    if extrapolation == "error":
        return (
            interpolate
            + (
                f"Values outside [{knots[0]}, {knots[-1]}] are not rated: the model "
                "refuses them (extrapolation='error')."
            )
            + scope
        )
    return (
        interpolate
        + (
            "Beyond the tabulated range hold the end rows flat: "
            f"below {knots[0]} use the {knots[0]} row; "
            f"above {knots[-1]} use the {knots[-1]} row."
        )
        + scope
    )


def _annotate_piecewise_blocks(ws, main_effects) -> None:
    """Re-format the piecewise cells and write each block's interpolation note.

    Runs *after* the global ``column % 3`` format loop and touches nothing
    else: the block placement and the format loop stay exactly as they were, so
    every other block keeps its coordinates and its formats.
    """
    for idx, block in enumerate(main_effects):
        if block.kind != "piecewise":
            continue
        start_col = 1 + idx * _MAIN_EFFECT_BLOCK_STRIDE
        first_row = _MAIN_EFFECT_HEADER_ROW + 1
        for row in range(first_row, first_row + len(block.table)):
            for offset in (1, 2):
                cell = ws.cell(row=row, column=start_col + offset)
                cell.number_format = _PIECEWISE_NUMBER_FORMAT
        # Row 6 sits between the block title and the dataframe header and is
        # otherwise unused, so the note needs no layout change to land in.
        ws.cell(
            row=_MAIN_EFFECT_NOTE_ROW,
            column=start_col,
            value=_piecewise_interpolation_note(
                block.table,
                block.extrapolation or "clip",
                block.centering_shift,
            ),
        )


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
    ws[_BASE_RELATIVITY_CELL] = float(payload.base_relativity)

    max_main_row = 8
    for idx, block in enumerate(payload.main_effects):
        start_col = 1 + idx * _MAIN_EFFECT_BLOCK_STRIDE
        title_cell = ws.cell(row=_MAIN_EFFECT_TITLE_ROW, column=start_col, value=block.name)
        title_cell.font = Font(bold=True)
        end_row, _ = _write_dataframe(ws, block.table, _MAIN_EFFECT_HEADER_ROW, start_col)
        max_main_row = max(max_main_row, end_row)

    for row in ws.iter_rows():
        for cell in row:
            if cell.value is None:
                continue
            if cell.column % 3 == 2:
                cell.number_format = "0.000000"
            if cell.column % 3 == 0:
                cell.number_format = "#,##0.00"

    ws[_BASE_RELATIVITY_CELL].number_format = _BASE_RELATIVITY_NUMBER_FORMAT
    _annotate_piecewise_blocks(ws, payload.main_effects)

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
