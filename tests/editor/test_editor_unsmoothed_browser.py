"""The Unsmoothed line, driven from the browser."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("playwright.sync_api")
pytestmark = pytest.mark.browser

_LINE = "#chart .unsmoothed-layer path.unsmoothed"
_UNDETERMINED = (
    "With its smoothing off, the curve of 'curve' is not determined: some of its basis "
    "functions have too few rows under them, as across a gap in the data. Fewer knots, or knots "
    "where the rows are, would determine it."
)


def test_the_unsmoothed_toggle_draws_the_line_and_takes_it_away(open_editor_page):
    with open_editor_page() as (page, _session):
        toggle = page.locator("#unsmoothedToggle")
        assert toggle.is_visible()
        assert toggle.get_attribute("aria-pressed") == "false"
        assert page.locator(_LINE).count() == 0

        toggle.click()
        line = page.locator(_LINE).first
        line.wait_for(state="attached")
        assert toggle.get_attribute("aria-pressed") == "true"
        assert toggle.get_attribute("aria-busy") is None
        assert toggle.get_attribute("aria-disabled") == "false"
        # Over the curve, kept to the plot, named in the legend and in
        # its hover text.
        assert line.get_attribute("clip-path") == "url(#plotClip)"
        assert page.locator("#chart .legend-layer line.unsmoothed").count() == 1
        assert page.locator("#chart .legend-layer text", has_text="unsmoothed").count() == 1
        assert (
            "curve fitted with its smoothing switched off" in line.locator("title").text_content()
        )

        toggle.click()
        page.wait_for_function("() => !document.querySelector('#chart .unsmoothed-layer')")
        assert toggle.get_attribute("aria-pressed") == "false"
        assert page.locator("#chart .legend-layer line.unsmoothed").count() == 0


def test_a_refused_line_keeps_the_toggle_on_and_can_be_turned_off(open_editor_page):
    refusals: list[str] = []

    def refuse(route) -> None:
        # The page loads its own unsmoothed.js, which the pattern also matches.
        if route.request.method != "POST":
            route.continue_()
            return
        refusals.append(route.request.url)
        route.fulfill(
            status=400, content_type="application/json", body=json.dumps({"error": _UNDETERMINED})
        )

    def refuse_unsmoothed(page) -> None:
        page.route("**/unsmoothed*", refuse)

    with open_editor_page(prepare=refuse_unsmoothed) as (page, _session):
        toggle = page.locator("#unsmoothedToggle")
        toggle.click()
        # The refusal's sentence is the hover text, and no line is drawn.
        page.wait_for_function(
            "text => document.querySelector('#unsmoothedToggle').dataset.popoverBody === text",
            arg=_UNDETERMINED,
        )
        assert toggle.get_attribute("aria-pressed") == "true"
        assert toggle.get_attribute("aria-disabled") == "false"
        assert page.locator(_LINE).count() == 0

        toggle.click()
        assert toggle.get_attribute("aria-pressed") == "false"
        toggle.click()
        assert toggle.get_attribute("aria-pressed") == "true"
        # The refusal stands for the fit in force, so turning the choice back on asks nothing new.
        assert len(refusals) == 1
