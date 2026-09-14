"""Conventions for executed documentation pages.

Executed pages must use the public API only. The audit of the old site
found four private-attribute reads in notebooks; this keeps them out.

The reference group pages that carry an executed example repeat one setup
cell per class so that every page runs on its own; the prose numbers on all
of them depend on that one simulation, so the copies must stay identical.

Pages that paste glued figures must glue every key they paste. The
pull-request docs build runs without execution, where glue data is empty by
design, so ``docs/conf.py`` silences ``mystnb.glue`` there; this keeps a
mistyped key from surviving until the executed deploy build.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.docs.page_rules import SKIP_DIRS, docs_pages, is_myst_notebook

DOCS = Path(__file__).resolve().parents[2] / "docs"
CODE_CELL = re.compile(r"```\{code-cell\}[^\n]*\n(.*?)```", re.S)
PRIVATE_ACCESS = re.compile(r"[\w\)\]]\._(?!_)[A-Za-z]\w*")
GLUE_CALL = re.compile(r"\bglue\(\s*[\"']([^\"']+)[\"']")
# A {glue:figure} / {glue:any} directive on a backtick or colon fence of any
# width, indented or not, or a {glue:text}`key` role (with or without a
# :format suffix).
GLUE_PASTE = re.compile(
    r"^[ \t]*(?:`{3,}|:{3,})\{glue(?::\w+)?\}\s+(\S+)|\{glue(?::\w+)?\}`([^`:]+)", re.M
)
SKIP_EXECUTION = re.compile(r"^:tags:.*skip-execution", re.M)


def executed_pages() -> list[Path]:
    return docs_pages(DOCS)


def test_no_private_attribute_access_in_executed_pages() -> None:
    offenders: list[str] = []
    for path in executed_pages():
        for cell in CODE_CELL.findall(path.read_text(encoding="utf-8")):
            for match in PRIVATE_ACCESS.finditer(cell):
                offenders.append(f"{path.relative_to(DOCS)}: {match.group(0)}")
    assert offenders == [], "private attribute access in executed pages:\n" + "\n".join(offenders)


def test_glue_pastes_are_glued_on_the_same_page() -> None:
    """Every ``glue:`` paste names a key that an executed cell on the same page glues.

    The pull-request build cannot check this itself: it runs with execution off,
    so every glue lookup is empty there and ``docs/conf.py`` suppresses the
    warning in that mode. Without this test a mistyped key would first fail in
    the executed deploy build, after the merge. Only code cells that run count
    as gluing: a call quoted in prose or sitting in a ``skip-execution`` cell
    never populates the glue store.
    """
    missing: list[str] = []
    pasted = 0
    for path in executed_pages():
        text = path.read_text(encoding="utf-8")
        executed_cells = [c for c in CODE_CELL.findall(text) if not SKIP_EXECUTION.search(c)]
        glued = {key for cell in executed_cells for key in GLUE_CALL.findall(cell)}
        for match in GLUE_PASTE.finditer(text):
            pasted += 1
            key = match.group(1) or match.group(2)
            if key not in glued:
                missing.append(f"{path.relative_to(DOCS)}: {key}")
    assert pasted, "no glue pastes found; GLUE_PASTE no longer matches the pages"
    assert missing == [], "glue pastes without a glue call on their page:\n" + "\n".join(missing)


def test_pages_with_code_cells_are_myst_notebooks() -> None:
    """Sphinx and the notebook test must agree on which pages execute.

    The Sphinx build executes any page that holds a ``{code-cell}``; the
    notebook test executes the pages ``page_rules.is_myst_notebook`` accepts,
    the same rule it imports. A page with cells but no front matter would run
    on the deploy build and never in the tests, so the two must select the
    same files.
    """
    mismatched: list[str] = []
    for path in executed_pages():
        text = path.read_text(encoding="utf-8")
        has_cells = "```{code-cell}" in text
        if has_cells != is_myst_notebook(text):
            mismatched.append(
                f"{path.relative_to(DOCS)}: code cells={has_cells}, "
                f"notebook front matter={not has_cells}"
            )
    assert mismatched == [], "pages Sphinx and the notebook test disagree on:\n" + "\n".join(
        mismatched
    )


def test_notebook_files_outside_examples_are_tutorial_pairs() -> None:
    """An ``.ipynb`` under docs is a stored example or the pair of a tutorial page.

    Both selection rules read ``*.md`` only, so a bare notebook anywhere else
    would execute on the deploy build and be invisible to every test. Only
    ``docs/tutorials/`` is equipped for pairs: the jupytext sync and nbstripout
    hooks and the Sphinx exclusion of ``tutorials/*.ipynb`` are all scoped there.
    """
    tutorials = DOCS / "tutorials"
    stray = [
        str(p.relative_to(DOCS))
        for p in sorted(DOCS.rglob("*.ipynb"))
        if not SKIP_DIRS & set(p.relative_to(DOCS).parts)
        and "examples" not in p.relative_to(DOCS).parts
        and not (p.parent == tutorials and p.with_suffix(".md").exists())
    ]
    assert stray == [], f"notebooks that are neither examples nor tutorial pairs: {stray}"


EXAMPLE_DIRS = {"SuperGLM": DOCS / "api" / "model", "SuperLSS": DOCS / "api" / "distributional"}


def example_pages(directory: Path) -> list[Path]:
    """Every group page in ``directory`` that carries an executed example."""
    return [
        p
        for p in sorted(directory.glob("*.md"))
        if "\n## Example\n" in p.read_text(encoding="utf-8")
    ]


def first_visible_cell(path: Path) -> str:
    cells = CODE_CELL.findall(path.read_text(encoding="utf-8"))
    visible = [cell for cell in cells if ":tags: [remove-cell]" not in cell]
    assert visible, f"{path.relative_to(DOCS)} has an Example section but no visible code cell"
    return visible[0]


@pytest.mark.parametrize("family", sorted(EXAMPLE_DIRS))
def test_example_setup_cells_are_identical(family: str) -> None:
    """Each class's example pages share one simulation; the prose numbers depend on it.

    The setup cell is repeated per page so every page runs on its own. Editing one
    copy would silently invalidate the numbers cited on the others, so the copies
    must stay byte-identical.
    """
    pages = example_pages(EXAMPLE_DIRS[family])
    assert len(pages) >= 2, f"{family}: fewer than two example pages found"
    reference = first_visible_cell(pages[0])
    differing = sorted(p.name for p in pages[1:] if first_visible_cell(p) != reference)
    assert differing == [], f"{family} setup cells differ from {pages[0].name}: {differing}"


def first_hidden_cell(path: Path) -> str:
    cells = CODE_CELL.findall(path.read_text(encoding="utf-8"))
    hidden = [cell for cell in cells if ":tags: [remove-cell]" in cell]
    assert hidden, f"{path.relative_to(DOCS)} has an Example section but no hidden style cell"
    return hidden[0]


@pytest.mark.parametrize("family", sorted(EXAMPLE_DIRS))
def test_example_style_cells_are_identical(family: str) -> None:
    """The hidden style cell is repeated per page as well, and it has drifted once already.

    The tutorial and the REML page are not held to this copy: they sit one
    directory shallower, and the tutorial's cell differs by design so that its
    paired notebook still runs in Colab.
    """
    pages = example_pages(EXAMPLE_DIRS[family])
    reference = first_hidden_cell(pages[0])
    differing = sorted(p.name for p in pages[1:] if first_hidden_cell(p) != reference)
    assert differing == [], f"{family} style cells differ from {pages[0].name}: {differing}"
