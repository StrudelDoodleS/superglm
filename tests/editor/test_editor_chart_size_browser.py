from __future__ import annotations

from urllib.parse import urlsplit

import pytest

pytest.importorskip("playwright.sync_api")
pytestmark = pytest.mark.browser

VIEWPORTS = [(1180, 720), (1920, 1080), (2560, 1440)]

CHART_GEOMETRY = """() => {
    const svg = document.querySelector('#chart');
    const box = svg.getBoundingClientRect();
    const shell = svg.parentElement.getBoundingClientRect();
    const style = getComputedStyle(svg);
    const border = side => parseFloat(style.getPropertyValue(`border-${side}-width`));
    const ctm = svg.getScreenCTM();
    const viewBox = svg.viewBox.baseVal;
    return {
        viewBox: [viewBox.width, viewBox.height],
        viewport: [
            box.width - border('left') - border('right'),
            box.height - border('top') - border('bottom'),
        ],
        chart: [box.width, box.height],
        shell: [shell.width, shell.height],
        scale: [ctm.a, ctm.d],
        overflow: document.documentElement.scrollWidth > window.innerWidth,
    };
}"""


def _drawn_to_fit(page) -> None:
    """Wait for the redraw that follows a layout change to land."""
    page.wait_for_function(
        """() => {
            const svg = document.querySelector('#chart');
            const viewBox = svg.viewBox.baseVal;
            return viewBox.width === svg.clientWidth && viewBox.height === svg.clientHeight;
        }""",
        timeout=5000,
    )


def _box_select_x(page, lo: float, hi: float) -> None:
    """Drag a Select box over the whole plot height between two x values."""
    corners = page.evaluate(
        """([lo, hi]) => {
            const svg = document.querySelector('#chart');
            const scale = svg._scale;
            const client = (x, y) => {
                const point = svg.createSVGPoint();
                point.x = x;
                point.y = y;
                const mapped = point.matrixTransform(svg.getScreenCTM());
                return { x: mapped.x, y: mapped.y };
            };
            return [
                client(scale.sx(lo), scale.margin.top + 1),
                client(scale.sx(hi), scale.margin.top + scale.innerH - 1),
            ];
        }""",
        [lo, hi],
    )
    page.mouse.move(corners[0]["x"], corners[0]["y"])
    page.mouse.down()
    page.mouse.move(corners[1]["x"], corners[1]["y"], steps=4)
    with page.expect_response(
        lambda response: (
            response.request.method == "POST" and urlsplit(response.url).path == "/select"
        )
    ):
        page.mouse.up()


@pytest.mark.parametrize("term", ["curve", "territory"])
@pytest.mark.parametrize(("width", "height"), VIEWPORTS)
def test_chart_is_drawn_at_the_measured_size_of_its_panel(open_editor_page, width, height, term):
    with open_editor_page(viewport={"width": width, "height": height}, selected_term=term) as (
        page,
        _session,
    ):
        _drawn_to_fit(page)
        geometry = page.evaluate(CHART_GEOMETRY)

        # One SVG unit is one CSS pixel: no letterboxing, text at its nominal size.
        assert geometry["viewBox"] == pytest.approx(geometry["viewport"], abs=1)
        assert geometry["scale"] == pytest.approx([1.0, 1.0], abs=2e-3)

        # The chart is its whole panel, never shorter than the 360px floor.
        assert geometry["chart"] == pytest.approx(geometry["shell"], abs=1)
        assert geometry["chart"][1] >= 360
        # A full-screen browser gets the plot column's width, which the old
        # 1400px shell capped at 746px.
        if width >= 1920:
            assert geometry["chart"][0] > 1100
        assert not geometry["overflow"]


def test_a_brush_after_a_viewport_resize_selects_the_points_under_it(open_editor_page):
    with open_editor_page(viewport={"width": 1180, "height": 720}) as (page, session):
        _drawn_to_fit(page)
        before = page.evaluate(CHART_GEOMETRY)

        page.set_viewport_size({"width": 1920, "height": 1080})
        page.wait_for_function(
            "width => document.querySelector('#chart').viewBox.baseVal.width > width + 300",
            arg=before["viewBox"][0],
            timeout=5000,
        )
        _drawn_to_fit(page)
        after = page.evaluate(CHART_GEOMETRY)
        assert after["viewBox"] == pytest.approx(after["viewport"], abs=1)
        assert after["scale"] == pytest.approx([1.0, 1.0], abs=2e-3)

        # Brush edges halfway between grid points, so the expected set is unambiguous.
        x = session.terms["curve"].x
        lo, hi = (x[39] + x[40]) / 2, (x[80] + x[81]) / 2
        _box_select_x(page, lo, hi)
        expected = [i for i, value in enumerate(x) if lo <= value <= hi]
        assert expected == list(range(40, 81))
        assert session.selection("curve").tolist() == expected
