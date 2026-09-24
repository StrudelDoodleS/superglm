from __future__ import annotations

from urllib.parse import urlsplit

import pytest

from superglm import Piecewise

pytest.importorskip("playwright.sync_api")
pytestmark = pytest.mark.browser


def _posted(path: str):
    return lambda response: (
        response.request.method == "POST" and urlsplit(response.url).path == path
    )


def _reload_editor(page, term: str) -> None:
    page.reload(wait_until="domcontentloaded")
    page.locator("#chart path.edited").first.wait_for()
    page.wait_for_function(
        "term => document.querySelector('#status')?.dataset.term === term",
        arg=term,
    )


def _plot_point(page, display_index: int) -> dict[str, float]:
    """The client position of a display point's x, halfway down the plot."""
    return page.evaluate(
        """index => {
            const svg = document.querySelector('#chart');
            const scale = svg._scale;
            const point = svg.createSVGPoint();
            point.x = scale.sx(scale.x[index]);
            point.y = scale.margin.top + scale.innerH / 2;
            const client = point.matrixTransform(svg.getScreenCTM());
            return { x: client.x, y: client.y };
        }""",
        display_index,
    )


def test_breaks_mode_transforms_an_ordered_term(open_editor_page):
    with open_editor_page(selected_term="age_band") as (page, session):
        page.locator("#chart").focus()
        page.keyboard.press("b")
        page.locator("#breaksControls").wait_for(state="visible")
        for band in (2, 4):
            point = _plot_point(page, band)
            page.mouse.click(point["x"], point["y"])
        page.wait_for_function("() => document.querySelectorAll('#chart .break-line').length === 2")

        chip = page.locator("#chart .degree-chip").first
        before = chip.get_attribute("aria-label")
        chip.click()
        page.wait_for_function(
            "label => document.querySelector('#chart .degree-chip')"
            "?.getAttribute('aria-label') !== label",
            arg=before,
        )

        with page.expect_response(_posted("/transform_term")) as response_info:
            page.locator("#transformTerm").click()
        assert response_info.value.status == 200
        page.locator("#restoreStructure").wait_for(state="visible")
        assert isinstance(session.model._specs["age_band"]._spline_obj, Piecewise)


def test_set_reference_icon_needs_exactly_one_level(open_editor_page):
    with open_editor_page(selected_term="territory") as (page, session):
        set_reference = page.locator("#setReference")
        session.select_levels("territory", ["T03", "T04"])
        _reload_editor(page, "territory")
        page.locator("#selectionMenu").wait_for(state="visible")
        assert set_reference.is_hidden()

        session.select_levels("territory", ["T03"])
        _reload_editor(page, "territory")
        set_reference.wait_for(state="visible")
        with page.expect_response(_posted("/set_reference")) as response_info:
            set_reference.click()
        assert response_info.value.status == 200
        page.wait_for_function(
            "() => document.querySelector('#termReference')?.textContent"
            " === 'reference T03 · pinned'"
        )


def test_refresh_pulls_a_notebook_side_structural_change(open_editor_page):
    with open_editor_page(selected_term="territory") as (page, session):
        restore = page.locator("#restoreStructure")
        refresh = page.locator("#refreshAction")
        refresh.wait_for(state="visible")
        page.wait_for_function("() => !document.querySelector('#refreshAction').disabled")
        session.select_levels("territory", ["T01", "T02"])
        session.replace_with_collapsed_levels("territory", method="fit")
        assert restore.is_hidden()
        assert page.locator("#chart .level-group-marker").count() == 0

        refresh.click()
        restore.wait_for(state="visible")
        page.locator("#chart .level-group-marker").first.wait_for()
        assert page.locator("#chart .level-group-marker").count() == 2


def test_revert_confirms_and_returns_to_the_opened_model(open_editor_page):
    with open_editor_page(
        selected_term="territory", collapsed_levels=("territory", ("T01", "T02"))
    ) as (page, session):
        revert = page.locator("#revertAction")
        restore = page.locator("#restoreStructure")
        dialog = page.locator("#structuralConfirmDialog")
        page.wait_for_function("() => !document.querySelector('#revertAction').disabled")
        assert restore.is_visible()

        revert.click()
        dialog.wait_for(state="visible")
        assert dialog.locator("#structuralConfirmMessage").text_content() == (
            "Revert to the original model? This clears 0 manual edits and "
            "1 structural step, and can't be undone."
        )
        with page.expect_response(_posted("/revert_to_original")) as response_info:
            dialog.get_by_role("button", name="Continue and refit", exact=True).click()
        assert response_info.value.status == 200
        restore.wait_for(state="hidden")
        page.wait_for_function("() => document.querySelector('#revertAction').disabled")
        assert session.structure_history == []
