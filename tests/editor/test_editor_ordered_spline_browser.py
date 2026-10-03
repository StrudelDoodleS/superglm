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


def test_a_level_edit_the_spline_cannot_follow_joins_the_level_dots(open_editor_page):
    # age_band has five basis columns over six levels, so one level's change
    # leaves the levels off every spline of the basis. Mutation check:
    # `spline_fits_levels` answering True draws the least-change spline on the
    # grid instead, which misses the edited dot.
    with open_editor_page(selected_term="age_band") as (page, session):
        grid_points = ORDERED_SPLINE_GRID_STEPS * (AGE_BANDS - 1) + 1
        assert _path_points(page, "#chart path.edited") == grid_points
        page.get_by_role("radiogroup", name="Chart tools").get_by_role(
            "radio", name="Select", exact=True
        ).click()
        with page.expect_response(_posted("/select")):
            page.locator('#chart circle.point[data-index="2"]').click()
        page.locator("#selectionMenu").wait_for(state="visible")
        before = page.locator("#chart path.edited").get_attribute("d")
        with page.expect_response(_posted("/op")) as response_info:
            page.get_by_role("button", name="Increase selection", exact=True).click()
        assert response_info.value.status == 200
        assert session.history[-1].indices.tolist() == [2]

        page.wait_for_function(
            "d => document.querySelector('#chart path.edited').getAttribute('d') !== d",
            arg=before,
        )
        assert _path_points(page, "#chart path.edited") == AGE_BANDS
        drawn = page.evaluate(
            """() => ({
                vertices: [...document.querySelector('#chart path.edited').getAttribute('d')
                    .matchAll(/[ML] (-?[\\d.]+) (-?[\\d.]+)/g)]
                    .map(match => [Number(match[1]), Number(match[2])]),
                dots: [...document.querySelectorAll(
                    '#chart circle.point[data-index]:not([data-selection-supplemental])'
                )].map(node => [
                    Number(node.dataset.index),
                    Number(node.getAttribute('cx')),
                    Number(node.getAttribute('cy')),
                ]).sort((a, b) => a[0] - b[0]),
            })"""
        )
        # The line joins the dots, the edited one included: each vertex is its
        # level dot's centre written to two decimals.
        assert [dot[0] for dot in drawn["dots"]] == list(range(AGE_BANDS))
        for (x, y), (_, cx, cy) in zip(drawn["vertices"], drawn["dots"], strict=True):
            assert abs(x - cx) <= 0.005 + 1e-9 and abs(y - cy) <= 0.005 + 1e-9
        # The fit it is compared against is still drawn as its spline.
        assert _path_points(page, "#chart path.original") == grid_points


# Each level dot and the spline's vertex at that level: the path writes its
# vertices to two decimals, and level k sits at grid index k * steps.
_DOTS_ON_CURVE = """steps => {
    const vertices = [...document.querySelector('#chart path.edited').getAttribute('d')
        .matchAll(/[ML] (-?[\\d.]+) (-?[\\d.]+)/g)]
        .map(match => [Number(match[1]), Number(match[2])]);
    return [...document.querySelectorAll('#chart .spline-level-dot')].map((dot, k) => [
        Number(dot.getAttribute('cx')), Number(dot.getAttribute('cy')), ...vertices[k * steps],
    ]);
}"""


def test_handles_draw_the_level_dots_on_the_spline_and_carry_them_through_a_drag(
    open_editor_page,
):
    # Mutation check: drawing no level dots in Handles mode fails the count;
    # drawing them at the fitted values rather than the drag preview's leaves
    # them 17 px off the dragged curve.
    with open_editor_page(selected_term="age_band") as (page, _session):
        _handles_tool(page).click()
        handles = page.locator("#chart .control-handle")
        handles.first.wait_for()

        def dots_on_curve():
            dots = page.evaluate(_DOTS_ON_CURVE, ORDERED_SPLINE_GRID_STEPS)
            assert len(dots) == AGE_BANDS
            for cx, cy, vx, vy in dots:
                assert abs(cx - vx) <= 0.005 + 1e-6 and abs(cy - vy) <= 0.005 + 1e-6
            return [cy for _cx, cy, _vx, _vy in dots]

        resting = dots_on_curve()
        box = handles.nth(1).bounding_box()
        assert box is not None
        page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        page.mouse.down()
        page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2 - 40, steps=4)
        # Mid-drag the preview carries the dots with the curve.
        dragged = dots_on_curve()
        assert max(abs(a - b) for a, b in zip(resting, dragged, strict=True)) > 1.0
        with page.expect_response(_posted("/control")) as response_info:
            page.mouse.up()
        assert response_info.value.status == 200
        page.wait_for_function(
            "n => document.querySelectorAll('#chart .spline-level-dot').length === n",
            arg=AGE_BANDS,
        )
        dots_on_curve()
