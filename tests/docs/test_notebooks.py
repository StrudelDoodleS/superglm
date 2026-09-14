"""Execute every MyST Markdown notebook under ``docs/`` with a fresh kernel.

This is the authoritative check on executable documentation. The Sphinx
build caches renders by file content, so a library change that breaks a
tutorial without touching its text would otherwise go unnoticed.

Figures are not checked here: the kernel runs under the Agg backend, so a
cell whose only output is a figure is checked for not raising. The strict
Sphinx build is what asserts that every cell renders an output.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# The ``docs`` dependency group is absent from the test matrix environment, and
# a module-level import would error at collection time — before ``-m "not
# docs"`` deselects anything. Skipping keeps parametrisation (pathlib only).
jupytext = pytest.importorskip("jupytext")
nbclient = pytest.importorskip("nbclient")

from tests.docs.page_rules import docs_pages, is_myst_notebook  # noqa: E402

DOCS = Path(__file__).resolve().parents[2] / "docs"


def myst_notebooks() -> list[Path]:
    return [p for p in docs_pages(DOCS) if is_myst_notebook(p.read_text(encoding="utf-8"))]


@pytest.mark.docs
@pytest.mark.parametrize("path", myst_notebooks(), ids=lambda p: str(p.relative_to(DOCS)))
def test_notebook_executes(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MPLBACKEND", "Agg")
    notebook = jupytext.read(path)
    client = nbclient.NotebookClient(
        notebook,
        timeout=900,
        kernel_name="python3",
        skip_cells_with_tag="skip-execution",
        resources={"metadata": {"path": str(path.parent)}},
    )
    client.execute()
    executed = [c for c in notebook.cells if c.cell_type == "code" and c.get("execution_count")]
    assert executed, f"{path.name} has no executed code cells"
