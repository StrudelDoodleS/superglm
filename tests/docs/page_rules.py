"""The rules that pick documentation pages for the docs tests.

``test_notebooks.py`` executes the pages :func:`is_myst_notebook` accepts, and
``test_conventions.py`` pins that rule against the Sphinx one (any page with a
``{code-cell}`` executes). Both read the rule from here, so the two tests
cannot drift apart. Keep this module free of third-party imports: the
conventions test must import it in an environment without ``jupytext``.
"""

from __future__ import annotations

from pathlib import Path

SKIP_DIRS = {"superpowers", "_build"}


def docs_pages(docs: Path) -> list[Path]:
    """Every Markdown page under ``docs`` outside the skipped directories."""
    return [p for p in sorted(docs.rglob("*.md")) if not SKIP_DIRS & set(p.relative_to(docs).parts)]


def is_myst_notebook(text: str) -> bool:
    """True when a page's front matter names the MyST notebook format."""
    head = text[:600]
    return head.startswith("---") and "format_name: myst" in head
