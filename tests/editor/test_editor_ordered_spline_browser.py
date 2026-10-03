from __future__ import annotations

from urllib.parse import urlsplit

import pytest

from superglm.editor.controls import ORDERED_SPLINE_GRID_STEPS, ORDERED_SPLINE_SHAPED

pytest.importorskip("playwright.sync_api")
pytestmark = pytest.mark.browser

# The browser fixture's age_band: six levels, no specials.
AGE_BANDS = 6


def _posted(path: str):
    return lambda response: (
        response.request.method == "POST" and urlsplit(response.url).path == path
    )


def _path_points(page, selector: str) -> int:
    return page.evaluate(
        "selector => document.querySelector(selector).getAttribute('d').match(/[ML]/g).length",
        selector,
    )


def _handles_tool(page):
    return page.get_by_role("radiogroup", name="Chart tools").get_by_role(
        "radio", name="Handles", exact=True
    )


def test_an_ordered_spline_is_drawn_as_its_spline_with_handles_contrib_and_build(
    open_editor_page,
):
    with open_editor_page(selected_term="age_band") as (page, session):
        grid_points = ORDERED_SPLINE_GRID_STEPS * (AGE_BANDS - 1) + 1
        # The curve between the dots is the spline, not straight segments.
        assert _path_points(page, "#chart path.edited") == grid_points
        assert _path_points(page, "#chart path.original") == grid_points

        _handles_tool(page).click()
        handles = page.locator("#chart .control-handle")
        handles.first.wait_for()
        live = session.ordered_spline("age_band").live.size
        assert handles.count() == live
        # Handles turn Contrib on; each contribution runs over the same grid.
        assert page.locator("#basisToggle").is_visible()
        assert page.locator("#contribPlay").is_visible()
        contributions = page.locator("#chart .basis-contribution")
        assert contributions.count() == live
        assert _path_points(page, "#chart .basis-contribution") == grid_points

        before = session.terms["age_band"].edited_log_effect.copy()
        box = handles.nth(live // 2).bounding_box()
        assert box is not None
        page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        page.mouse.down()
        page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2 - 30, steps=4)
        with page.expect_response(_posted("/control")) as response_info:
            page.mouse.up()
        assert response_info.value.status == 200

        record = session.history[-1]
        assert record.operation == "control_point"
        assert record.params["basis"] == "ordered_spline"
        assert (session.terms["age_band"].edited_log_effect != before).any()
        # After the move the levels lie on the spline again, so it is still drawn.
        page.wait_for_function(
            "n => document.querySelector('#chart path.edited')"
            ".getAttribute('d').match(/[ML]/g).length === n",
            arg=grid_points,
        )


def test_a_shaped_band_turns_handles_off_and_says_why(open_editor_page):
    with open_editor_page(selected_term="age_band") as (page, session):
        session.replace_with_shaped_range(
            "age_band", lo="25-34", hi="45-54", degree=1, method="fit"
        )
        page.reload(wait_until="domcontentloaded")
        page.locator("#chart path.edited").first.wait_for()
        page.wait_for_function(
            "term => document.querySelector('#status')?.dataset.term === term", arg="age_band"
        )

        handles = _handles_tool(page)
        assert handles.get_attribute("aria-disabled") == "true"
        assert handles.get_attribute("data-popover-body") == ORDERED_SPLINE_SHAPED
        handles.hover()
        popover = page.locator("#uiPopover")
        popover.wait_for(state="visible")
        assert ORDERED_SPLINE_SHAPED in popover.inner_text()
        handles.click(force=True)
        assert handles.get_attribute("aria-checked") == "false"
        assert page.locator("#chart .control-handle").count() == 0
