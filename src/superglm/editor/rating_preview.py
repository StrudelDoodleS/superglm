"""The rating-table preview: one term's block of the Excel export, for the browser.

The widget builds the payload through the same call the workbook export makes
(``EditorWidget._rating_table_payload``); this module picks one term's block
out of it and renders it as JSON, with the cell formats and the note the
workbook gives that block.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from superglm.editor.io import jsonable
from superglm.editor.lss.view import split_key

# The preview asks for no discretisation-impact sweep: it fills only the
# workbook's impact sheet and is built after every block, so the blocks match.
PREVIEW_IMPACT_BINS: tuple[int, ...] = ()

# One fixed sentence per reason there is no table to show.
UNSUPPORTED_TERMS = (
    "Rating tables do not cover random-effect or factor-smooth terms, so this model has none."
)
EXPORT_REFUSED = "The Excel export refuses this model, so there is no rating table to preview."
NO_BLOCK = "The rating table has no block for this term."
SUPERSEDED = "The model changed while the table was built. Showing the new one next."


@dataclass(frozen=True)
class RatingTables:
    """One revision's rating tables as the editor reads them, whichever model built them.

    ``blocks`` holds each table by its term key; ``write`` writes the workbook
    to a path or buffer. For a SuperLSS, ``lines`` says which quantity a term's
    factors multiply and ``reasons`` gives the library's sentence for each term
    of a parameter it left out.
    """

    blocks: Mapping[str, Any]
    write: Callable[[Any], None]
    lines: Mapping[str, str] = field(default_factory=dict)
    reasons: Mapping[str, str] = field(default_factory=dict)


def glm_rating_tables(payload) -> RatingTables:
    """A SuperGLM payload's main effects, written to the workbook the export names."""
    from superglm.export.excel import write_rating_table_workbook

    return RatingTables(
        {block.name: block for block in payload.main_effects},
        lambda target: write_rating_table_workbook(
            payload,
            target,
            sheet_name="Rating Tables",
            summary_sheet_name="Model Summary",
            impact_sheet_name="Discretization Impact",
        ),
    )


def lss_rating_tables(payload, term_keys) -> RatingTables:
    """A SuperLSS payload: a sheet per parameter, and the library's sentence for each left out."""
    from superglm.export.excel import write_lss_rating_table_workbook

    return RatingTables(
        payload.blocks(),
        lambda target: write_lss_rating_table_workbook(payload, target),
        lines={
            f"{table.parameter}:{block.name}": table.note
            for table in payload.parameters
            for block in table.main_effects
        },
        reasons={
            key: payload.omitted[split_key(key)[0]]
            for key in term_keys
            if split_key(key)[0] in payload.omitted
        },
    )


@dataclass(frozen=True)
class RatingPreview:
    """One model revision's rating tables, or why it has none.

    Owned by the widget and kept until the session's ``model_revision``
    moves, which every edit, undo, redo and structural step does; switching
    terms reuses it.
    """

    model_revision: int
    payload: RatingTables | None
    reason: str | None


def refusal_reason(model) -> str:
    """The fixed sentence for an export the builder refused."""
    from superglm.export.rating_tables import _unsupported_structured_export_terms

    return UNSUPPORTED_TERMS if _unsupported_structured_export_terms(model) else EXPORT_REFUSED


def term_rating_table(preview: RatingPreview, term: str) -> dict[str, Any]:
    """``term``'s main-effect block as ``{term, available, reason, columns, rows, ...}``."""
    if preview.payload is None:
        return _unavailable(term, preview.reason or EXPORT_REFUSED, preview.model_revision)
    tables = preview.payload
    block = tables.blocks.get(term)
    if block is None:
        return _unavailable(term, tables.reasons.get(term, NO_BLOCK), preview.model_revision)
    table = block.table
    return {
        "term": term,
        "available": True,
        "reason": None,
        "columns": [str(column) for column in table.columns],
        "rows": [[_cell(value) for value in row] for row in table.itertuples(index=False)],
        "formats": _block_formats(block),
        "note": " ".join(filter(None, (tables.lines.get(term), _block_note(block)))) or None,
        "model_revision": preview.model_revision,
    }


def _unavailable(term: str, reason: str, model_revision: int) -> dict[str, Any]:
    return {
        "term": term,
        "available": False,
        "reason": reason,
        "columns": [],
        "rows": [],
        "formats": [],
        "note": None,
        "model_revision": model_revision,
    }


def _block_formats(block) -> list[str | None]:
    """The number format the workbook gives each column, in the order it applies them."""
    from superglm.export.excel import _PIECEWISE_NUMBER_FORMAT, _main_effect_number_format

    formats = [
        _main_effect_number_format(block, str(column), offset)
        for offset, column in enumerate(block.table.columns)
    ]
    if block.kind == "piecewise":
        for offset in (1, 2):
            if offset < len(formats):
                formats[offset] = _PIECEWISE_NUMBER_FORMAT
    return formats


def _block_note(block) -> str | None:
    """The note the workbook writes above the block, or None where it writes none."""
    from superglm.export.excel import _piecewise_interpolation_note, _ppform_evaluation_note

    if block.kind == "piecewise":
        return _piecewise_interpolation_note(
            block.table, block.extrapolation or "clip", block.centering_shift
        )
    if block.kind == "continuous_ppform":
        return _ppform_evaluation_note(block.name, block.extrapolation)
    return None


def _cell(value: Any) -> str | int | float | bool | None:
    """A table cell as JSON: numpy scalars as Python ones, a missing or non-finite value as None."""
    plain = jsonable(value)
    if plain is None or isinstance(plain, bool | int | float):
        return plain
    return None if pd.isna(plain) else str(plain)
