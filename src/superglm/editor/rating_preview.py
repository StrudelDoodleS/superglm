"""The rating-table preview: one term's block of the Excel export, for the browser.

The widget builds the payload through the same call the workbook export makes
(``EditorWidget._rating_table_payload``); this module picks one term's block
out of it and renders it as JSON, with the cell formats and the note the
workbook gives that block.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from superglm.editor.io import jsonable

# The preview asks for no discretisation-impact sweep. The builder makes every
# block before it sweeps, and the sweep fills only the workbook's impact sheet,
# so the blocks are the workbook's; the sweep was 5.5 s of the 7.0 s build on
# the 678,013-row freMTPL2 book with two splines.
PREVIEW_IMPACT_BINS: tuple[int, ...] = ()

# One fixed sentence per reason there is no table to show.
UNSUPPORTED_TERMS = (
    "Rating tables do not cover random-effect or factor-smooth terms, so this model has none."
)
EXPORT_REFUSED = "The Excel export refuses this model, so there is no rating table to preview."
NO_BLOCK = "The rating table has no block for this term."
SUPERSEDED = "The model changed while the table was built. Showing the new one next."


@dataclass(frozen=True)
class RatingPreview:
    """One model revision's rating-table payload, or why it has none.

    Owned by the widget and kept until the session's ``model_revision``
    moves, which every edit, undo, redo and structural step does; switching
    terms reuses it.
    """

    model_revision: int
    payload: Any | None
    reason: str | None


def refusal_reason(model) -> str:
    """The fixed sentence for an export the builder refused."""
    from superglm.export.rating_tables import _unsupported_structured_export_terms

    return UNSUPPORTED_TERMS if _unsupported_structured_export_terms(model) else EXPORT_REFUSED


def term_rating_table(preview: RatingPreview, term: str) -> dict[str, Any]:
    """``term``'s main-effect block as ``{term, available, reason, columns, rows, ...}``."""
    if preview.payload is None:
        return _unavailable(term, preview.reason or EXPORT_REFUSED, preview.model_revision)
    block = next((item for item in preview.payload.main_effects if item.name == term), None)
    if block is None:
        return _unavailable(term, NO_BLOCK, preview.model_revision)
    table = block.table
    return {
        "term": term,
        "available": True,
        "reason": None,
        "columns": [str(column) for column in table.columns],
        "rows": [[_cell(value) for value in row] for row in table.itertuples(index=False)],
        "formats": _block_formats(block),
        "note": _block_note(block),
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
