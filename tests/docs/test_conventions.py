"""Conventions for executed documentation pages.

Executed pages must use the public API only. The audit of the old site
found four private-attribute reads in notebooks; this keeps them out.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[2] / "docs"
CODE_CELL = re.compile(r"```\{code-cell\}[^\n]*\n(.*?)```", re.S)
PRIVATE_ACCESS = re.compile(r"[\w\)\]]\._(?!_)[A-Za-z]\w*")


def executed_pages() -> list[Path]:
    return [
        p
        for p in sorted(DOCS.rglob("*.md"))
        if "superpowers" not in p.parts and "_build" not in p.parts
    ]


def test_no_private_attribute_access_in_executed_pages() -> None:
    offenders: list[str] = []
    for path in executed_pages():
        for cell in CODE_CELL.findall(path.read_text(encoding="utf-8")):
            for match in PRIVATE_ACCESS.finditer(cell):
                offenders.append(f"{path.relative_to(DOCS)}: {match.group(0)}")
    assert offenders == [], "private attribute access in executed pages:\n" + "\n".join(offenders)


SETUP_FAMILIES = {
    "SuperGLM": [
        DOCS / "api" / "model" / f"{s}.md"
        for s in ("inference", "fit", "predict", "deploy", "plot")
    ],
    "SuperLSS": [
        DOCS / "api" / "distributional" / f"{s}.md"
        for s in ("inference", "price", "predict", "check-the-fit")
    ],
}


def first_visible_cell(path: Path) -> str:
    cells = CODE_CELL.findall(path.read_text(encoding="utf-8"))
    return next(cell for cell in cells if ":tags: [remove-cell]" not in cell)


@pytest.mark.parametrize("family", sorted(SETUP_FAMILIES))
def test_example_setup_cells_are_identical(family: str) -> None:
    """Each class's example pages share one simulation; the prose numbers depend on it.

    The setup cell is repeated per page so every page runs on its own. Editing one
    copy would silently invalidate the numbers cited on the others, so the copies
    must stay byte-identical.
    """
    pages = SETUP_FAMILIES[family]
    reference = first_visible_cell(pages[0])
    differing = sorted(p.name for p in pages[1:] if first_visible_cell(p) != reference)
    assert differing == [], f"{family} setup cells differ from {pages[0].name}: {differing}"
