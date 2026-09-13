"""Execute every MyST Markdown notebook under ``docs/`` with a fresh kernel.

This is the authoritative check on executable documentation. The Sphinx
build caches renders by file content, so a library change that breaks a
tutorial without touching its text would otherwise go unnoticed.
"""

from __future__ import annotations

from pathlib import Path

import jupytext
import pytest
from nbclient import NotebookClient

DOCS = Path(__file__).resolve().parents[2] / "docs"
SKIP_DIRS = {"superpowers", "_build", "api"}


def myst_notebooks() -> list[Path]:
    found: list[Path] = []
    for path in sorted(DOCS.rglob("*.md")):
        if SKIP_DIRS & set(path.relative_to(DOCS).parts):
            continue
        head = path.read_text(encoding="utf-8")[:600]
        if head.startswith("---") and "format_name: myst" in head:
            found.append(path)
    return found


@pytest.mark.docs
@pytest.mark.parametrize("path", myst_notebooks(), ids=lambda p: str(p.relative_to(DOCS)))
def test_notebook_executes(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MPLBACKEND", "Agg")
    notebook = jupytext.read(path)
    client = NotebookClient(
        notebook,
        timeout=900,
        kernel_name="python3",
        skip_cells_with_tag="skip-execution",
        resources={"metadata": {"path": str(path.parent)}},
    )
    client.execute()
    executed = [c for c in notebook.cells if c.cell_type == "code" and c.get("execution_count")]
    assert executed, f"{path.name} has no executed code cells"
