"""The Unsmoothed line, driven from the browser."""

from __future__ import annotations

import pytest

pytest.importorskip("playwright.sync_api")
pytestmark = pytest.mark.browser

_LINE = "#chart .unsmoothed-layer path.unsmoothed"


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
