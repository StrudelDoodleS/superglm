"""Sphinx configuration for the superglm documentation.

Environment switches used by CI and the spike:

- ``SUPERGLM_DOCS_EXECUTE``: MyST-NB execution mode, default ``cache``.
  Pull-request builds set ``off``.
- ``SUPERGLM_DOCS_THEME``: theme name, default ``pydata_sphinx_theme``.
"""

from __future__ import annotations

import os
from importlib.metadata import version as _dist_version
from pathlib import Path

HERE = Path(__file__).resolve().parent

project = "superglm"
author = "Max Hicks"
copyright = "2026, Max Hicks"  # noqa: A001
release = _dist_version("superglm")
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
    "numpydoc",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinxext.rediraffe",
]

templates_path = ["_templates"]

exclude_patterns = [
    "_build",
    "_templates",
    "superpowers/**",
    "tutorials/*.ipynb",
    "**/.ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
]

# TEMPORARY (spike, Tasks 2-5): the legacy MkDocs pages still live under
# docs/ until Task 10 moves and converts them. Excluding them keeps the
# strict build to the new tree. Delete this block in Task 10.
exclude_patterns += [
    "api/diagnostics.md",
    "api/distributional.md",
    "api/families.md",
    "api/features.md",
    "api/inference.md",
    "api/model_selection.md",
    "api/penalties.md",
    "api/plotting.md",
    "api/validation.md",
    "audit/**",
    "development/**",
    "getting-started/**",
    "governance/**",
    "guide/**",
    "models/**",
    "notebooks/**",
    "research/**",
    "distributional-family-development.md",
    "editor_frontend.md",
    "ROADMAP.md",
    "tabmat-integration-notes.md",
]

# MyST Markdown
myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "substitution",
]
myst_heading_anchors = 3

# MyST-NB execution
nb_execution_mode = os.environ.get("SUPERGLM_DOCS_EXECUTE", "cache")
nb_execution_cache_path = str(HERE / "_build" / ".jupyter_cache")
nb_execution_timeout = 900
nb_execution_raise_on_error = True
nb_execution_excludepatterns = ["examples/*.ipynb"]
nb_merge_streams = True

# API reference
autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "signature"
autodoc_default_options = {"show-inheritance": True}
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False
numpydoc_xref_param_type = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}
intersphinx_timeout = 30

nitpicky = True
# Docstring type shorthands that are prose, not importable objects, followed
# by the types that are real but have no reference page. Extend only with
# entries the build reports, and name every dotted path you add.
nitpick_ignore_regex = [
    ("py:class", r"array[-_]like"),
    ("py:class", r"ArrayLike"),
    ("py:class", r"callable"),
    ("py:class", r"optional"),
    ("py:class", r"default .*"),
    ("py:class", r"(list|dict|tuple|sequence|iterable|mapping) of .*"),
    ("py:class", r"DataFrame|Series|ndarray"),
    ("py:class", r"(numpy\._typing\._array_like\.)?NDArray"),
    ("py:class", r"FrameLike"),
    # Real classes that ``superglm.__all__`` does not export, so autosummary
    # never makes a page for them and a reference can never bind. Base classes
    # and internal record/alias types, each referenced from a public
    # docstring's annotations: ``Distribution`` (family base),
    # ``Link`` (link-function base), ``Penalty`` (penalty base),
    # ``PIRLSResult`` (solver return record), ``FeatureSpec`` (typing alias),
    # and ``DiagnosticFinding`` / ``FitWorkProfile`` / ``JsonValue``
    # (fit-report records and their JSON alias).
    (
        "py:class",
        r"superglm\.(diagnostics\.fit_report\.(DiagnosticFinding|FitWorkProfile"
        r"|JsonValue)|distributions\.Distribution|links\.Link"
        r"|penalties\.base\.Penalty|solvers\.pirls\.PIRLSResult"
        r"|types\.FeatureSpec)",
    ),
    # Private row records behind ``ModelSummary``; private by design, so they
    # get no page and no cross-reference target.
    ("py:class", r"_(CoefRow|BasisDetailRow)"),
    # Display helper for summary levels; not exported, so no page.
    ("py:class", r"SummaryLevelDisplay"),
    # TEMPORARY: these four ARE in ``superglm.__all__`` and get their pages
    # from the twelve API pages Task 9 writes; until those pages exist the
    # unqualified references in docstrings have nothing to bind to.
    # Delete these two lines in Task 9 and confirm the strict build stays
    # green; if an unqualified name still does not bind, qualify it in the
    # docstring rather than restoring the ignore.
    ("py:class", r"FactorSmoothResult|RandomEffectResult"),
    ("py:class", r"TermInference|InteractionInference"),
]

# Redirects from the MkDocs site's URLs
rediraffe_redirects = "redirects.txt"

# HTML
html_theme = os.environ.get("SUPERGLM_DOCS_THEME", "pydata_sphinx_theme")
html_title = "superglm"
html_static_path = ["_static", "images"]
html_favicon = "images/logo.png"
html_css_files = [
    "https://fonts.googleapis.com/css2?family=Bangers&family=Source+Sans+3:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap",
    "custom.css",
]
html_sidebars = {"index": []}
html_context = {
    "github_user": "StrudelDoodleS",
    "github_repo": "superglm",
    "github_version": "master",
    "doc_path": "docs",
}

if html_theme == "pydata_sphinx_theme":
    html_theme_options = {
        "logo": {
            "image_light": "_static/logo.png",
            "image_dark": "_static/logo.png",
            "text": "superglm",
        },
        "icon_links": [
            {
                "name": "GitHub",
                "url": "https://github.com/StrudelDoodleS/superglm",
                "icon": "fa-brands fa-github",
            },
            {
                "name": "PyPI",
                "url": "https://pypi.org/project/superglm/",
                "icon": "fa-brands fa-python",
            },
        ],
        "navbar_align": "left",
        "header_links_before_dropdown": 8,
        "show_toc_level": 2,
        "use_edit_page_button": False,
        "footer_start": ["copyright"],
        "footer_end": [],
    }
elif html_theme == "shibuya":
    html_theme_options = {
        "github_url": "https://github.com/StrudelDoodleS/superglm",
        "accent_color": "red",
        "nav_links": [
            {"title": "Get started", "url": "get-started/index"},
            {"title": "Tutorials", "url": "tutorials/index"},
            {"title": "How-to", "url": "how-to/index"},
            {"title": "Explanation", "url": "explanation/index"},
            {"title": "API", "url": "api/index"},
        ],
    }

copybutton_exclude = ".linenos, .gp, .go"
