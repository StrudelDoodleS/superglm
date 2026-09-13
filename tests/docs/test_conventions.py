"""Conventions for executed documentation pages.

Executed pages must use the public API only. The audit of the old site
found four private-attribute reads in notebooks; this keeps them out.
"""

from __future__ import annotations

import re
from pathlib import Path

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
