"""The editor's CSS is the source of truth for the plotting suite's style.

Every constant in ``superglm.plotting.editor_style`` is asserted against the
CSS shipped in ``superglm/editor/app``, so an edit to the editor's palette
turns this red instead of silently forking the two looks.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from matplotlib.colors import to_hex

import superglm
from superglm.plotting.editor_style import (
    BODY_PT,
    CHART,
    DIVERGING,
    LABEL_PT,
    PANEL,
    SEQUENTIAL,
    TOKENS,
    apply_panel_frame,
    css_chart_classes,
    css_tokens,
    diverging_cmap,
    matplotlib_context,
    plotly_template,
    register_plotly_template,
    sequential_cmap,
)

PLOTLY_AVAILABLE = importlib.util.find_spec("plotly") is not None
_APP = Path(superglm.__file__).parent / "editor" / "app"

# Which CSS property each CHART field restates, per chart class.
_COLOUR_SOURCE: dict[str, dict[str, str]] = {
    "edited": {"color": "stroke"},
    "original": {"color": "stroke"},
    "previous_edit": {"color": "stroke"},
    "ci": {"color": "fill"},
    "ci_whisker": {"color": "stroke"},
    "exposure": {"fill": "fill", "edge": "stroke"},
    "point": {"face": "fill", "edge": "stroke"},
    "point_selected": {"face": "fill", "edge": "stroke"},
    "basis_contribution": {"color": "stroke"},
    "zero": {"color": "stroke"},
    "axis": {"color": "stroke"},
    "grid": {"color": "stroke"},
}


def _colour(value: str) -> tuple[str | tuple[int, int, int], float | None]:
    """A CSS colour in the form CHART writes it: hex, or ``((r, g, b), alpha)``."""
    rgba = re.fullmatch(r"rgba\((\d+),\s*(\d+),\s*(\d+),\s*([\d.]+)\)", value)
    if rgba is not None:
        red, green, blue, alpha = rgba.groups()
        return (int(red), int(green), int(blue)), float(alpha)
    if re.fullmatch(r"#[0-9a-fA-F]{3}", value):
        value = "#" + "".join(channel * 2 for channel in value[1:])
    return value.lower(), None


def test_tokens_match_the_editor_css():
    parsed = css_tokens()
    assert {name: parsed[name] for name in TOKENS} == TOKENS
    assert css_tokens(_APP / "styles" / "tokens.css") == parsed


def test_chart_grammar_matches_the_editor_css():
    rules = css_chart_classes()
    assert set(CHART) == set(_COLOUR_SOURCE)
    assert css_chart_classes(_APP / "styles.css") == rules
    for name, fields in _COLOUR_SOURCE.items():
        rule, spec = rules[name], CHART[name]
        for field, prop in fields.items():
            colour, alpha = _colour(rule[prop])
            assert spec[field] == colour, name
            if alpha is not None:
                assert spec["alpha"] == pytest.approx(alpha), name
        if "fill-opacity" in rule:
            assert spec["alpha"] == pytest.approx(float(rule["fill-opacity"])), name
        elif "alpha" not in spec:
            assert "fill-opacity" not in rule, name
        if "width" in spec:
            assert spec["width"] == pytest.approx(float(rule["stroke-width"])), name
        else:
            assert "stroke-width" not in rule, name
        if "dash" in spec:
            assert spec["dash"] == tuple(int(run) for run in rule["stroke-dasharray"].split()), name
        else:
            assert "stroke-dasharray" not in rule, name


def test_matplotlib_context_and_panel_frame():
    with matplotlib_context():
        assert plt.rcParams["font.size"] == LABEL_PT
        assert plt.rcParams["axes.labelsize"] == BODY_PT
        assert plt.rcParams["axes.grid"]
        figure, ax = plt.subplots()
        assert to_hex(figure.get_facecolor()) == TOKENS["surface"]
        assert to_hex(ax.get_facecolor()) == TOKENS["surface"]
        assert to_hex(ax.spines["left"].get_edgecolor()) == CHART["axis"]["color"]
        apply_panel_frame(ax)
    assert ax.spines["top"].get_visible() is False
    assert ax.spines["right"].get_visible() is False
    assert to_hex(ax.spines["bottom"].get_edgecolor()) == CHART["axis"]["color"]
    assert to_hex(figure.get_edgecolor()) == PANEL["frame"]
    assert figure.patch.get_linewidth() == pytest.approx(1.0)
    assert ax.xaxis.get_major_ticks()[0].label1.get_fontsize() == pytest.approx(LABEL_PT)
    gridline = ax.xaxis.get_gridlines()[0]
    assert to_hex(gridline.get_color()) == "#8c959f"
    assert gridline.get_alpha() == pytest.approx(CHART["grid"]["alpha"])


def test_colormaps_run_between_the_token_endpoints():
    sequential, diverging = sequential_cmap(), diverging_cmap()
    assert to_hex(sequential(0.0)) == SEQUENTIAL[0] == TOKENS["blue_soft"]
    assert to_hex(sequential(1.0)) == SEQUENTIAL[1] == TOKENS["blue"]
    assert to_hex(diverging(0.0)) == DIVERGING[0] == TOKENS["red"]
    assert to_hex(diverging(1.0)) == DIVERGING[2] == TOKENS["blue"]
    assert DIVERGING[1] == TOKENS["surface"]
    assert min(diverging(0.5)[:3]) > 0.99


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="plotly is not installed")
def test_plotly_template_carries_the_editor_look():
    import plotly.io as pio

    template = plotly_template()
    assert template.layout.paper_bgcolor == TOKENS["surface"]
    assert template.layout.plot_bgcolor == TOKENS["surface"]
    assert template.layout.font.size == BODY_PT
    assert template.layout.font.color == TOKENS["text"]
    assert template.layout.xaxis.gridcolor == TOKENS["grid"]
    assert template.layout.yaxis.gridcolor == TOKENS["grid"]
    assert template.layout.xaxis.linecolor == TOKENS["border_strong"]
    assert template.layout.xaxis.tickfont.size == LABEL_PT
    assert register_plotly_template() == "superglm_editor"
    assert "superglm_editor" in pio.templates
    assert register_plotly_template() == "superglm_editor"
