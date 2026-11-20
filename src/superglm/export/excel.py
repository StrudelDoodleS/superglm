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

# Width of an ordinary main-effect block: key, relativity, weight. Blocks are placed by
# their own widths (``_main_effect_start_columns``; a ppform block is seven wide). The
# downstream loader finds blocks by a header signature with ``relativity`` at +1 and
# ``weight`` at +2, so anything a block adds goes to the right of these three.
_MAIN_EFFECT_BLOCK_STRIDE = 3

# Blocks sit end to end with no blank column, so an all-three-wide layout lands on
# columns 1, 4, 7.
_MAIN_EFFECT_BLOCK_GAP = 0


def _main_effect_start_columns(blocks) -> list[int]:
    """The first column of each main-effect block, by cumulative width.

    Every caller that places or re-reaches a block goes through this, so placement, number
    formats, piecewise notes and key-column widths agree about where a block is.
    """
    starts: list[int] = []
    column = 1
    for block in blocks:
        starts.append(column)
        column += len(block.table.columns) + _MAIN_EFFECT_BLOCK_GAP
    return starts


_MAIN_EFFECT_TITLE_ROW = 5
_MAIN_EFFECT_NOTE_ROW = 6
_MAIN_EFFECT_HEADER_ROW = 7

# The piecewise block's two numeric columns are re-formatted after the per-block pass;
# ``Log relativity`` is not a role that pass knows, and two decimals would defeat
# publishing it.
_PIECEWISE_NUMBER_FORMAT = "0.000000000000"

# The ppform exact-form columns: coefficients are signed log-space numbers of unbounded
# scale, so they get significant digits (scientific), not decimal places.
_PPFORM_NUMBER_FORMAT = "0.00000000000000E+00"

# Formats keyed by a column's role, read from the header the block declares, because
# blocks differ in width and a position rule like ``column % 3`` misfires on a wider block.
_MAIN_EFFECT_ROLE_FORMATS = {
    "Relativity": "0.000000",
    "Weight": "#,##0.00",
}

# The base relativity cell (``C2``, above the first block, so the per-block pass never
# reaches it). It multiplies every row of the tariff, and under ``centering="mean"``
# carries the transferred centering constant, so a displayed 0.37 for 0.3719954 is a
# 5.4e-03 error on every risk. It is ``exp(intercept)`` with unbounded magnitude, so
# the format is scientific, not fixed-decimal. ``General`` is rejected because ECMA-376
# s18.8.30 budgets it in display characters (leading zeros count) and then switches to
# exponential with no pinned precision; ``0.0000000000E+00`` gives eleven significant
# digits at every magnitude.
_BASE_RELATIVITY_CELL = "C2"
_BASE_RELATIVITY_NUMBER_FORMAT = "0.0000000000E+00"

# A format Excel cannot fit renders as ``########``, and ``_autosize`` measures the raw
# float. Width derived from the format: 1 digit, separator, 10 decimals, ``E``, sign and 3
# exponent digits = 17 characters, plus the two columns of padding ``_autosize`` adds.
_BASE_RELATIVITY_MIN_WIDTH = 17 + 2

# A block's key column is not prose, so it is exempt from ``_autosize``'s 36-character
# cap: Excel clips an interval string, leaving a plausible but wrong right edge. A float64
# ``repr`` is at most 24 characters, so the widest interval is
# ``[`` + 24 + ``, `` + 24 + ``)`` = 52, plus the two columns of padding.
_KEY_COLUMN_MAX_WIDTH = 1 + 24 + 2 + 24 + 1 + 2


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

    Derived from the block's own printed columns, so a reader can check the stated slopes
    against the two end rows. Slopes are written at full round-trip precision: a rounded
    boundary slope would silently put a discrepancy back into the tariff outside the rated
    range. The claim holds in both centerings; under ``centering="mean"`` the note states the
    constant the block was shifted by, which the exported base relativity carries back. The
    out-of-range rule is the term's ``extrapolation`` parameter: flat beyond the outermost
    knots (default) or extending the boundary segments.
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


def _main_effect_number_format(block, column_name: str, offset: int) -> str | None:
    """The format one column of one block should carry, or ``None`` for plain.

    ``Relativity`` and ``Weight`` are recognised by name wherever they appear, so the lookup
    half of a ppform block renders like the binned block it stands in for. Everything past the
    mandatory three-column prefix is the exact form and gets significant digits. Keyed on the
    block's ``kind``, so a column renamed in ``rating_tables`` cannot fall back to the default.
    """
    role = _MAIN_EFFECT_ROLE_FORMATS.get(column_name)
    if role is not None:
        return role
    if block.kind == "continuous_ppform" and offset >= _MAIN_EFFECT_BLOCK_STRIDE:
        return _PPFORM_NUMBER_FORMAT
    return None


def _format_main_effect_blocks(ws, main_effects) -> None:
    """Apply each block's number formats within its own columns.

    Per block rather than over the whole sheet, because a sheet-wide rule can
    only be expressed in column arithmetic and column arithmetic stopped being
    able to say which column is which the moment blocks could differ in width.
    """
    for block, start_col in zip(
        main_effects, _main_effect_start_columns(main_effects), strict=True
    ):
        first_row = _MAIN_EFFECT_HEADER_ROW + 1
        for offset, column_name in enumerate(block.table.columns):
            number_format = _main_effect_number_format(block, str(column_name), offset)
            if number_format is None:
                continue
            for row in range(first_row, first_row + len(block.table)):
                ws.cell(row=row, column=start_col + offset).number_format = number_format


def _annotate_piecewise_blocks(ws, main_effects) -> None:
    """Re-format the piecewise cells and write each block's interpolation note.

    Runs *after* the per-block format pass and touches nothing else: the block
    placement stays exactly as it was, so every other block keeps its
    coordinates and its formats.
    """
    starts = _main_effect_start_columns(main_effects)
    for idx, block in enumerate(main_effects):
        if block.kind != "piecewise":
            continue
        start_col = starts[idx]
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


def _per_unit_offset_note(name: str) -> str:
    """The multiply rule for one per-unit offset block, stated on the sheet.

    A ``Numeric`` block and a per-unit offset block have the same shape (one row keyed
    ``per_unit`` with a ``Relativity`` and a ``Weight``) but different arithmetic:
    ``Relativity ** x`` for the estimated numeric coefficient, ``Relativity * x`` for the
    offset, whose coefficient is 1 by construction. ``RatingTableBlock.kind`` is not written to
    the workbook, so this note is the consumer's routing information. It holds for both
    declared (keyed on the named source column) and undeclared (the offset multiplier the
    consumer computes; ``Relativity`` is exactly 1.0) forms. It states that no rated range
    exists, since nothing is estimated and there is no extrapolation rule.
    """
    return (
        f"Multiply this risk's {name} by the Relativity below: "
        f"factor = {name} x Relativity. "
        f"NOT Relativity to the power of {name} -- a Numeric term exports an "
        "identically shaped per_unit row that does take the power rule, and the "
        "block's kind is not written to this sheet, so this note is what "
        "separates them. "
        "An offset's coefficient is 1 by construction, so nothing here is "
        "estimated: the relation is exact, it carries no rated range and no "
        "extrapolation rule, and it holds above the largest value the fit saw."
    )


def _ppform_evaluation_note(name: str, extrapolation: str | None) -> str:
    """How to score one ppform block, stated on the sheet the consumer holds.

    A bounded row is evaluated; an unbounded row (the leading and trailing constant pieces)
    is read, its factor being the ``Relativity`` beside it, because
    ``u = (x - lower) / (upper - lower)`` with an infinite bound is ``nan`` and ``0 * nan`` is
    ``nan``. Under ``extrapolation="error"`` there are no unbounded rows and the second
    sentence is not written. Goes in the note row the piecewise and per-unit blocks use.
    """
    formula = (
        f"Evaluate this block: u = ({name} - lower) / (upper - lower) from the "
        "interval key, then factor = EXP(a + u*(b + u*(c + u*d)))."
    )
    if extrapolation == "error":
        return formula + (
            " Every row is bounded -- this term declines to price outside its "
            f"fitted range, so a {name} matching no row has no factor here. "
            "Read the closing bracket: rows are [lower, upper) except the last, "
            "which is [lower, upper] so that the boundary itself has a row."
        )
    return formula + (
        " EXCEPT on the first and last rows, whose interval is unbounded (-inf "
        "or inf): those are CONSTANT pieces, so read Relativity and stop. Do "
        "not apply the formula to them -- u is not a number on an infinite "
        "width, and the zero coefficients do not make it one."
    )


def _annotate_ppform_blocks(ws, main_effects) -> None:
    """Write each ppform block's evaluation rule into its note row.

    Beside ``_annotate_per_unit_offset_blocks`` and for the same reason: the
    sheet does not carry ``RatingTableBlock.kind``, so a consumer holding only
    the workbook has to be told which arithmetic a block takes. Here the
    consequence of guessing is sharper than a wrong factor -- applying the
    polynomial to an unbounded row returns ``nan``.
    """
    starts = _main_effect_start_columns(main_effects)
    for idx, block in enumerate(main_effects):
        if block.kind != "continuous_ppform":
            continue
        ws.cell(
            row=_MAIN_EFFECT_NOTE_ROW,
            column=starts[idx],
            value=_ppform_evaluation_note(block.name, block.extrapolation),
        )


def _annotate_per_unit_offset_blocks(ws, main_effects) -> None:
    """Write each per-unit offset block's multiply rule into its note row.

    Separate from ``_annotate_piecewise_blocks`` and run beside it: that
    function also re-formats the piecewise cells, and this block kind needs no
    re-formatting at all -- its ``Relativity`` and ``Weight`` already carry the
    formats their names earn from ``_MAIN_EFFECT_ROLE_FORMATS``.  All this adds
    is the note, in the same otherwise-unused row 6.
    """
    starts = _main_effect_start_columns(main_effects)
    for idx, block in enumerate(main_effects):
        if block.kind != "offset_per_unit":
            continue
        ws.cell(
            row=_MAIN_EFFECT_NOTE_ROW,
            column=starts[idx],
            value=_per_unit_offset_note(block.name),
        )


def _autosize(ws) -> None:
    from openpyxl.utils import get_column_letter

    for column_cells in ws.columns:
        letter = get_column_letter(column_cells[0].column)
        max_length = max(
            len(str(cell.value)) if cell.value is not None else 0 for cell in column_cells
        )
        ws.column_dimensions[letter].width = min(max(max_length + 2, 12), 36)


def _widen_block_key_columns(ws, main_effects) -> None:
    """Let each block's key column fit its own keys, past the prose cap.

    Only the FIRST column of each main-effect block, because that is the one
    holding the keys a reader looks a risk up by; the ``Relativity`` and
    ``Weight`` columns beside it are formatted numbers and are already short.
    The cap stays, at the width the printer can actually produce, so the rule
    is still a bound rather than "as wide as it takes".
    """
    from openpyxl.utils import get_column_letter

    for start_col in _main_effect_start_columns(main_effects):
        letter = get_column_letter(start_col)
        longest = max(
            (len(str(cell.value)) for cell in ws[letter] if cell.value is not None),
            default=0,
        )
        ws.column_dimensions[letter].width = min(
            max(ws.column_dimensions[letter].width or 0, longest + 2),
            _KEY_COLUMN_MAX_WIDTH,
        )


def _write_rating_blocks(ws, main_effects, interactions) -> None:
    """Lay the main-effect blocks side by side from row 5, then the interaction blocks below."""
    from openpyxl.styles import Font

    max_main_row = 8
    for block, start_col in zip(
        main_effects,
        _main_effect_start_columns(main_effects),
        strict=True,
    ):
        title_cell = ws.cell(row=_MAIN_EFFECT_TITLE_ROW, column=start_col, value=block.name)
        title_cell.font = Font(bold=True)
        end_row, _ = _write_dataframe(ws, block.table, _MAIN_EFFECT_HEADER_ROW, start_col)
        max_main_row = max(max_main_row, end_row)

    _format_main_effect_blocks(ws, main_effects)

    ws[_BASE_RELATIVITY_CELL].number_format = _BASE_RELATIVITY_NUMBER_FORMAT
    _annotate_piecewise_blocks(ws, main_effects)
    _annotate_per_unit_offset_blocks(ws, main_effects)
    _annotate_ppform_blocks(ws, main_effects)

    interaction_row = max_main_row + 3
    for block in interactions:
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


def write_rating_table_workbook(
    payload,
    target: str | PathLike[str] | BinaryIO,
    *,
    sheet_name: str,
    summary_sheet_name: str,
    impact_sheet_name: str,
) -> None:
    """Write an exported rating table to an Excel workbook."""

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

    _write_rating_blocks(ws, payload.main_effects, payload.interactions)

    impact_ws = wb.create_sheet(impact_sheet_name)
    _write_dataframe(impact_ws, payload.discretization_impact, 1, 1)

    summary_ws = wb.create_sheet(summary_sheet_name)
    _write_summary_sheet(summary_ws, payload.summary)

    for sheet in wb.worksheets:
        _autosize(sheet)

    # After ``_autosize``, and floors rather than assignments, so a column that
    # is already wider keeps its width.
    base_column = ws[_BASE_RELATIVITY_CELL].column_letter
    ws.column_dimensions[base_column].width = max(
        ws.column_dimensions[base_column].width,
        _BASE_RELATIVITY_MIN_WIDTH,
    )
    _widen_block_key_columns(ws, payload.main_effects)

    summary_ws.column_dimensions["A"].width = 32
    summary_ws.column_dimensions["B"].width = 24
    for column in "CDEFGHIJKLMNO":
        summary_ws.column_dimensions[column].width = min(
            max(summary_ws.column_dimensions[column].width, 14),
            18,
        )
    wb.save(out)


def write_lss_rating_table_workbook(
    payload,
    target: str | PathLike[str] | BinaryIO,
    *,
    summary_sheet_name: str = "Summary",
) -> None:
    """Write a SuperLSS rating-table payload: one sheet per parameter, then a summary.

    A parameter's sheet holds its base in ``C2``, its floor in ``C3`` where the parameter is a
    floor plus the product, and the sentence saying how the parameter follows from the product in
    ``A4``; the blocks follow, laid out as on a SuperGLM sheet.
    """
    from openpyxl import Workbook
    from openpyxl.styles import Font

    from superglm.export.lss_rating_tables import parameter_label

    out = _resolve_workbook_target(target)
    wb = Workbook()
    wb.remove(wb.active)
    for table in payload.parameters:
        ws = wb.create_sheet(table.quantity)
        ws.freeze_panes = "A8"
        ws["A1"] = f"{table.quantity} ({table.parameter} parameter)"
        ws["A1"].font = Font(bold=True)
        ws["A2"] = "Base"
        ws["A2"].font = Font(bold=True)
        ws[_BASE_RELATIVITY_CELL] = float(table.base_relativity)
        if table.rule == "floor_plus_product":
            ws["A3"] = "Floor"
            ws["A3"].font = Font(bold=True)
            ws["C3"] = float(table.floor)
        ws["A4"] = table.note
        _write_rating_blocks(ws, table.main_effects, table.interactions)
        _autosize(ws)
        ws.column_dimensions["A"].width = 32
        ws.column_dimensions[
            ws[_BASE_RELATIVITY_CELL].column_letter
        ].width = _BASE_RELATIVITY_MIN_WIDTH
        _widen_block_key_columns(ws, table.main_effects)

    # Only a floored log link has a floor to read: the column is left out when no parameter has one.
    has_floor = any(t.floor for t in payload.parameters)
    summary = pd.DataFrame(
        [
            {
                "Parameter": parameter_label(t.parameter),
                "Name in code": t.parameter,
                "Table": t.quantity,
                "Link": t.link,
                "Parameter follows from the product by": t.rule,
                **({"Floor": t.floor} if has_floor else {}),
                "Base": t.base_relativity,
                "How to read it": t.note,
            }
            for t in payload.parameters
        ]
    )
    ws = wb.create_sheet(summary_sheet_name)
    end_row, _ = _write_dataframe(ws, summary, 1, 1)
    row = end_row + 2
    for sentence in payload.omitted.values():
        ws.cell(row=row, column=1, value=sentence)
        row += 1
    if payload.notes:
        ws.cell(row=row + 1, column=1, value="Notes").font = Font(bold=True)
        for row, sentence in enumerate(payload.notes, start=row + 2):
            ws.cell(row=row, column=1, value=sentence)
    _autosize(ws)
    wb.save(out)
