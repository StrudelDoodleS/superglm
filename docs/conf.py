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
nb_execution_excludepatterns = ["examples/*.ipynb", "*/examples/*.ipynb", "**/examples/*.ipynb"]
nb_merge_streams = True

# docs/examples/mtpl2_frequency_walkthrough.ipynb carries two stored figures in
# ``application/vnd.plotly.v1+json``, a MIME type MyST-NB has no renderer for.
# The pages are excluded from execution, so the stored outputs are what they are;
# suppress the per-output notice rather than rewriting the committed notebook.
suppress_warnings = ["mystnb.unknown_mime_type"]

# API reference
autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "signature"
# Wrap long signatures one parameter per line; SuperGLM has 25 constructor
# arguments and a single-line signature is unreadable.
maximum_signature_line_length = 88
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
    # NumPy scalar types and the private array-like alias: numpy's inventory
    # carries no page for them, so an annotation can never bind.
    ("py:class", r"numpy\.(float64|int64)"),
    ("py:class", r"numpy\._typing\._array_like\.ArrayLike"),
    # Abbreviated module alias used in hand-written docstring type lines.
    ("py:class", r"pd\.(DataFrame|Series)"),
    # numpydoc splits a type line at the first comma, so a subscripted generic
    # arrives truncated (``dict[str``) and is not a name at all.
    ("py:class", r"(dict|tuple|list|set|frozenset|collections\.abc\.Mapping)\[.*"),
    ("py:obj", r"typing\.Literal\[.*"),
    # Third-party objects named without their module in a type line:
    # matplotlib's ``Figure`` and ``Axes``, and ``pathlib``'s ``Path``.
    ("py:class", r"^(Figure|Axes|Path)$"),
    # Bare spellings of the internal types dispositioned just below, as they
    # appear in hand-written type lines: ``Link`` (link base), ``TermInput``
    # (term alias), ``GroupSlice`` (design-matrix record), ``_SplineBase``.
    ("py:class", r"^(Link|TermInput|GroupSlice|_SplineBase)$"),
    # Internal implementation types that ``superglm.__all__`` does not export,
    # so autosummary makes no page and a reference can never bind: the
    # distributional engine's family plans, parameter predictors, fit results
    # and check records under ``superglm.distributional`` (including
    # ``SuperLSSTrainingTelemetry``); the design-matrix records
    # ``DiscreteTensorBuildResult`` / ``GroupInfo`` / ``GroupSlice`` /
    # ``TensorMarginalInfo`` under ``superglm.types``; ``EagerFrame``;
    # ``InteractionSpec`` and the ``TermInput`` alias under ``superglm.terms``;
    # the spline base classes and ``StructuralContrastRow`` under
    # ``superglm.features``; ``Flavor`` under ``superglm.penalties.base``;
    # ``_CPGRNG`` under ``superglm.profiling.tweedie``; and
    # ``EditMaterializationRequest`` under ``superglm.editor``.
    (
        "py:class",
        r"superglm\.(" + "distributional\\.(api\\.SuperLSSTrainingTelemetry|checks\\.(binned\\.BinnedCheck(2D)?|calibration\\.(ActualExpected|CalibrationPayload)|compare\\.Comparison)|families\\._predictors\\.(LocationPredictor|MeanPredictor|ScalePredictor|ShapePredictor|SkewPredictor|ThetaPredictor|TweediePredictors)|family\\.(DistributionalFamily|FamilyLikelihoodPlan)|model\\.DenseDistributionalModel|posterior\\.PosteriorDraws|residuals\\.ResidualSet|results\\.fit\\.DistributionalFitResult|surfaces\\.(DensityFan|Portfolio|RiskCurves|Spread)|terms\\.(ParameterTermEffect|TermTest)|timing\\.FitPhaseRecorder)"
        r"|types\.(DiscreteTensorBuildResult|GroupInfo|GroupSlice"
        r"|TensorMarginalInfo)"
        r"|_frame\.EagerFrame"
        r"|terms\.(InteractionSpec|TermInput)"
        r"|features\.(piecewise\.StructuralContrastRow|spline\._B?SplineBase)"
        r"|penalties\.base\.Flavor"
        r"|profiling\.tweedie\._CPGRNG"
        r"|editor\.evaluation_cache\.EditMaterializationRequest)",
    ),
    # The ``superglm.families`` module page summarises its factory functions;
    # they are module members rather than exported names, so they get no page.
    ("py:obj", r"superglm\.families\.(binomial|gamma|gaussian|nb2|poisson|tweedie)"),
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
# The sources are on GitHub; this drops 27 MB from the published site.
html_copy_source = False
html_show_sourcelink = False
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
        # Depth 1 keeps the autosummary-generated member pages out of the
        # reference sidebar: they are children of api/model's and
        # api/distributional's own toctrees, so any deeper setting lists all
        # 233 of them, and html_sidebars cannot prune entries from a sidebar.
        # The cost is site-wide: second-level pages under Development >
        # Internals and under Examples leave the sidebar too; both remain
        # listed on their own index pages.
        "navigation_depth": 1,
        "use_edit_page_button": False,
        "footer_start": ["copyright"],
        "footer_end": [],
    }

copybutton_exclude = ".linenos, .gp, .go"
