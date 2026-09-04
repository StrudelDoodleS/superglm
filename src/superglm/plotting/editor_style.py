"""The notebook editor's design tokens and chart grammar as a style module.

The editor draws its chart as hand-written SVG over the custom properties in
``superglm/editor/app/styles/tokens.css`` and the classes in
``superglm/editor/app/styles.css``.  This module restates both as Python
constants, a matplotlib ``rc_context`` and a plotly template, so a figure from
the distributional suite and a chart in the editor read as one product.

The CSS is the source of truth.  :func:`css_tokens` and
:func:`css_chart_classes` parse the two packaged files, and
``tests/test_lss_editor_style.py`` asserts the constants below still equal what
they say, so a change to the editor's blue turns the plotting suite red instead
of silently forking the look.
"""

from __future__ import annotations

import re
from contextlib import AbstractContextManager
from importlib.resources import files
from pathlib import Path
from typing import Any

import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap

__all__ = [
    "BODY_PT",
    "CHART",
    "CHART_SELECTORS",
    "COLORWAY",
    "DIVERGING",
    "FONT_CSS",
    "FONT_STACK",
    "LABEL_PT",
    "PANEL",
    "SEQUENTIAL",
    "TOKENS",
    "apply_panel_frame",
    "css_chart_classes",
    "css_tokens",
    "diverging_cmap",
    "matplotlib_context",
    "plotly_template",
    "register_plotly_template",
    "sequential_cmap",
]

# ── The design tokens: editor/app/styles/tokens.css ────────────────
TOKENS: dict[str, str] = {
    "text": "#24292f",
    "muted": "#57606a",
    "surface": "#ffffff",
    "surface_subtle": "#f6f8fa",
    "border": "#d0d7de",
    "border_strong": "#8c959f",
    "grid": "rgba(140, 149, 159, 0.22)",
    "blue": "#0969da",
    "blue_soft": "#dbeafe",
    "red": "#d1242f",
    "orange": "#bf6a02",
    "yellow": "#f4d35e",
    "yellow_border": "#d8a10f",
    "danger": "#b42318",
}

# ── The chart grammar: the SVG classes in editor/app/styles.css ────
# Colours given as an ``(r, g, b)`` triple carry a separate ``alpha``, matching
# the ``rgba(...)`` the CSS writes; hex strings are opaque.
CHART: dict[str, dict[str, Any]] = {
    "edited": dict(color="#0969da", width=2.3),
    "original": dict(color="#8c959f", width=1.7, dash=(7, 5)),
    "previous_edit": dict(color="#f59e0b", width=2.0),
    "ci": dict(color=(9, 105, 218), alpha=0.13),
    "ci_whisker": dict(color=(9, 105, 218), alpha=0.55, width=1.4),
    "exposure": dict(fill="#f4d35e", alpha=0.95, edge="#d8a10f", width=1.0),
    "point": dict(face="#ffffff", edge="#0969da", width=1.5),
    "point_selected": dict(face="#d1242f", edge="#d1242f"),
    "basis_contribution": dict(color=(87, 96, 106), alpha=0.28, width=1.25, dash=(4, 4)),
    "zero": dict(color="#d0d7de", width=1.0, dash=(4, 4)),
    "axis": dict(color="#8c959f", width=1.0),
    "grid": dict(color=(140, 149, 159), alpha=0.22, width=1.0),
}

# Matplotlib resolves the first family it can find; the editor's own stack is
# the CSS system stack, unavailable to a headless renderer.
FONT_STACK = ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"]
FONT_CSS = "-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"
BODY_PT = 13
LABEL_PT = 11
PANEL: dict[str, Any] = dict(width_in=9.4, height_in=5.2, frame="#d0d7de", radius_px=6)
SEQUENTIAL = ["#dbeafe", "#0969da"]
DIVERGING = ["#d1242f", "#ffffff", "#0969da"]
COLORWAY = [
    TOKENS["blue"],
    TOKENS["orange"],
    TOKENS["red"],
    TOKENS["border_strong"],
    CHART["previous_edit"]["color"],
]

# ── CSS parsing ────────────────────────────────────────────────────
_PACKAGE = "superglm"
_TOKENS_CSS = ("editor", "app", "styles", "tokens.css")
_STYLES_CSS = ("editor", "app", "styles.css")

#: The chart selectors :func:`css_chart_classes` reads, as written in the CSS.
CHART_SELECTORS: tuple[str, ...] = (
    ".axis",
    ".tick",
    ".grid",
    ".zero",
    ".original",
    ".previous-edit",
    ".edited",
    ".ci",
    ".ci-whisker",
    ".exposure",
    ".exposure-density",
    ".exposure-axis",
    ".basis-contribution",
    ".basis-active",
    ".point",
    ".point.selected",
    ".level-group-label",
)

_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_CUSTOM_PROPERTY_RE = re.compile(r"--([\w-]+)\s*:\s*([^;{}]+);")
_RULE_RE = re.compile(r"([^{}]+)\{([^{}]*)\}")
_VAR_RE = re.compile(r"var\(\s*--([\w-]+)\s*\)")


def _read_css(path: str | Path | None, parts: tuple[str, ...]) -> str:
    """Read a CSS file, defaulting to the one packaged with ``superglm``."""
    if path is not None:
        return Path(path).read_text(encoding="utf-8")
    return files(_PACKAGE).joinpath(*parts).read_text(encoding="utf-8")


def _key(name: str) -> str:
    """CSS name to Python name: ``--surface-subtle`` -> ``surface_subtle``."""
    return name.lstrip(".").replace(".", "_").replace("-", "_")


def css_tokens(path: str | Path | None = None) -> dict[str, str]:
    """Parse the ``--name: value`` declarations of the editor's ``tokens.css``.

    Keys are the CSS names with hyphens replaced by underscores; values are the
    declaration text verbatim, so ``grid`` is ``"rgba(140, 149, 159, 0.22)"``.
    """
    text = _COMMENT_RE.sub("", _read_css(path, _TOKENS_CSS))
    return {_key(name): value.strip() for name, value in _CUSTOM_PROPERTY_RE.findall(text)}


def css_chart_classes(path: str | Path | None = None) -> dict[str, dict[str, str]]:
    """Parse the editor's chart rules from ``styles.css``.

    Returns one dict of declarations per selector in :data:`CHART_SELECTORS`,
    keyed the Python way (``.ci-whisker`` -> ``ci_whisker``, ``.point.selected``
    -> ``point_selected``), with every ``var(--x)`` resolved through
    :func:`css_tokens`.
    """
    text = _COMMENT_RE.sub("", _read_css(path, _STYLES_CSS))
    tokens = css_tokens()
    rules: dict[str, dict[str, str]] = {}
    for selectors, body in _RULE_RE.findall(text):
        for selector in (part.strip() for part in selectors.split(",")):
            if selector not in CHART_SELECTORS:
                continue
            declarations: dict[str, str] = {}
            for declaration in body.split(";"):
                name, separator, value = declaration.partition(":")
                if separator:
                    declarations[name.strip()] = _resolve_vars(value, tokens)
            rules[_key(selector)] = declarations
    return rules


def _resolve_vars(value: str, tokens: dict[str, str]) -> str:
    """Substitute ``var(--name)`` with the token's value, if it is known."""
    return _VAR_RE.sub(lambda m: tokens.get(_key(m.group(1)), m.group(0)), value).strip()


def _hex(rgb: tuple[int, int, int]) -> str:
    """An ``(r, g, b)`` triple as ``#rrggbb``."""
    return "#{:02x}{:02x}{:02x}".format(*rgb)


# ── matplotlib ─────────────────────────────────────────────────────
def matplotlib_context() -> AbstractContextManager[None]:
    """An ``rc_context`` carrying the editor's type, colours and grid."""
    axis = CHART["axis"]
    grid = CHART["grid"]
    return mpl.rc_context(
        rc={
            "font.family": FONT_STACK,
            "font.size": LABEL_PT,
            "axes.titlesize": BODY_PT,
            "axes.labelsize": BODY_PT,
            "axes.edgecolor": axis["color"],
            "axes.linewidth": axis["width"],
            "axes.grid": True,
            "axes.facecolor": TOKENS["surface"],
            "grid.color": _hex(grid["color"]),
            "grid.alpha": grid["alpha"],
            "grid.linewidth": grid["width"],
            "figure.facecolor": TOKENS["surface"],
            "figure.edgecolor": TOKENS["border"],
            "figure.dpi": 100,
            "savefig.facecolor": TOKENS["surface"],
            "legend.frameon": False,
            "xtick.color": axis["color"],
            "ytick.color": axis["color"],
            "xtick.labelcolor": TOKENS["text"],
            "ytick.labelcolor": TOKENS["text"],
            "text.color": TOKENS["text"],
        }
    )


def apply_panel_frame(ax: Axes) -> None:
    """Give ``ax`` the editor's panel: two spines, its grid, its frame."""
    axis = CHART["axis"]
    grid = CHART["grid"]
    for side, spine in ax.spines.items():
        if side in ("top", "right"):
            spine.set_visible(False)
        else:
            spine.set_color(axis["color"])
            spine.set_linewidth(axis["width"])
    ax.grid(
        True,
        color=_hex(grid["color"]),
        alpha=grid["alpha"],
        linewidth=grid["width"],
    )
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=LABEL_PT, color=axis["color"], labelcolor=TOKENS["text"])
    figure = ax.get_figure()
    if figure is not None:
        figure.patch.set_edgecolor(PANEL["frame"])
        figure.patch.set_linewidth(1.0)


def sequential_cmap() -> LinearSegmentedColormap:
    """Single-hue soft-blue to blue, for counts and densities."""
    return LinearSegmentedColormap.from_list("superglm_editor_sequential", SEQUENTIAL)


def diverging_cmap() -> LinearSegmentedColormap:
    """Red to white to blue, for signed quantities centred at zero."""
    return LinearSegmentedColormap.from_list("superglm_editor_diverging", DIVERGING)


# ── plotly (imported lazily: it is a development dependency only) ──
def plotly_template() -> Any:
    """The editor's grammar as a ``plotly.graph_objects.layout.Template``."""
    import plotly.graph_objects as go

    axis = dict(
        gridcolor=TOKENS["grid"],
        linecolor=TOKENS["border_strong"],
        zerolinecolor=TOKENS["border"],
        showline=True,
        zeroline=True,
        ticks="outside",
        tickcolor=TOKENS["border_strong"],
        tickfont=dict(size=LABEL_PT, color=TOKENS["text"]),
    )
    return go.layout.Template(
        layout=go.Layout(
            paper_bgcolor=TOKENS["surface"],
            plot_bgcolor=TOKENS["surface"],
            font=dict(family=FONT_CSS, size=BODY_PT, color=TOKENS["text"]),
            colorway=COLORWAY,
            xaxis=axis,
            yaxis=axis,
            legend=dict(font=dict(size=LABEL_PT, color=TOKENS["text"])),
            hoverlabel=dict(
                bgcolor=TOKENS["surface"],
                bordercolor=TOKENS["border"],
                font=dict(family=FONT_CSS, size=LABEL_PT, color=TOKENS["text"]),
            ),
        )
    )


def register_plotly_template(name: str = "superglm_editor") -> str:
    """Register :func:`plotly_template` under ``name``; idempotent."""
    import plotly.io as pio

    if name not in pio.templates:
        pio.templates[name] = plotly_template()
    return name
