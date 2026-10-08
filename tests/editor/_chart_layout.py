"""Waits on the editor chart's layout, shared by the browser suites."""

from __future__ import annotations

_CHART_FITS = """() => {
    const svg = document.querySelector('#chart');
    const viewBox = svg.viewBox.baseVal;
    return viewBox.width === svg.clientWidth && viewBox.height === svg.clientHeight;
}"""


def wait_for_chart_to_fit(page) -> None:
    """Wait until the chart is drawn at its panel's size.

    A change that moves the layout, such as a tool whose controls wrap the
    toolbar onto a second row, shrinks the chart; its resize observer then
    redraws it a frame later, replacing every point and handle. Measure or
    drag one only after that redraw: until then the chart's drawing and its
    box disagree, which is what this waits out.
    """
    page.wait_for_function(_CHART_FITS)
