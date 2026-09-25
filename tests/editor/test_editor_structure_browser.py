from __future__ import annotations

from urllib.parse import urlsplit

import numpy as np
import pytest
from tests.test_editor_structure import EPS, _line_residual, _pinning_tolerance

from superglm.editor.payloads import session_payload
from superglm.editor.shapes import _numeric_edges

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


def _settled_after_refit(page) -> None:
    """Wait until a structural refit has handed the page back to the analyst."""
    page.wait_for_function(
        "() => document.querySelector('#appBusyOverlay')?.hidden"
        " && !document.querySelector('#editorView')?.hasAttribute('inert')"
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
    with page.expect_response(_posted("/select")):
        page.mouse.up()


def test_line_icon_pins_a_run_of_points_and_restore_removes_it(open_editor_page):
    with open_editor_page() as (page, session):
        before = session.model
        _box_select_x(page, 3.0, 5.0)
        selected = session.selection("curve")
        # Precondition: the box took one contiguous run of the drawn points.
        assert selected.size > 2
        np.testing.assert_array_equal(np.diff(selected), 1)

        line = page.locator("#shapeLine")
        line.wait_for(state="visible")
        assert line.get_attribute("aria-disabled") == "false"
        with page.expect_response(_posted("/shape_range")) as response_info:
            line.click()
        assert response_info.value.status == 200
        _settled_after_refit(page)

        spec = session.model._specs["curve"]
        [pinned] = spec.polynomial_ranges
        term = session.terms["curve"]
        assert pinned.degree == 1
        assert pinned.lo <= term.x[selected[0]] and term.x[selected[-1]] <= pinned.hi

        band = page.locator("#chart .shape-range")
        assert band.count() == 1
        assert band.locator(".shape-range-label").text_content() == "Line"
        assert band.get_attribute("data-popover-title") == "Line"
        assert band.get_attribute("data-popover-body") == (
            f"Pinned to a straight line from {pinned.lo:g} to {pinned.hi:g}."
        )

        # The curve the page draws on the range is the pinned line: log of the
        # plotted relativity against x, to the fit's round-off plus one exp and
        # one log per value.
        drawn = page.evaluate(
            "() => { const s = document.querySelector('#chart')._scale;"
            " return { x: s.x, y: s.y }; }"
        )
        x = np.asarray(drawn["x"])
        inside = (x >= pinned.lo) & (x <= pinned.hi)
        effect = np.log(np.asarray(drawn["y"])[inside])
        bound = _pinning_tolerance(session.model, "curve", spec, effect) + 4 * EPS * (
            1.0 + np.max(np.abs(effect))
        )
        assert _line_residual(x[inside], effect) <= bound

        page.locator("#restoreStructure").click()
        _settled_after_refit(page)
        page.wait_for_function("() => !document.querySelector('#chart .shape-range')")
        assert session.model is before
        assert session.model._specs["curve"].polynomial_ranges == ()


def test_a_numeric_run_holding_too_few_values_disables_the_higher_shapes(open_editor_page):
    with open_editor_page() as (page, session):
        support = session_payload(session)["curve"]["shape"]["support"]
        held = np.array(support["through"][1:]) - np.array(support["below"][:-1])
        # Precondition: some two-point run holds two or three training values.
        [candidates] = np.nonzero((held >= 2) & (held < 4))
        assert candidates.size
        k = int(candidates[0])
        session.select_indices("curve", [k, k + 1])
        _reload_editor(page, "curve")
        page.locator("#selectionMenu").wait_for(state="visible")

        cubic = page.locator("#shapeCubic")
        assert cubic.get_attribute("aria-disabled") == "true"
        assert cubic.get_attribute("data-popover-body") == (
            "Select at least 4 distinct values for a Cubic."
        )
        assert page.locator("#shapeLine").get_attribute("aria-disabled") == "false"


def test_quadratic_on_bands_spans_whole_bands_and_cubic_says_why_not(open_editor_page):
    with open_editor_page(selected_term="age_band") as (page, session):
        session.select_levels("age_band", ["25-34", "35-44", "45-54"])
        _reload_editor(page, "age_band")
        page.locator("#selectionMenu").wait_for(state="visible")

        cubic = page.locator("#shapeCubic")
        assert cubic.get_attribute("aria-disabled") == "true"
        assert cubic.get_attribute("data-popover-body") == "Select at least 4 bands for a Cubic."
        with page.expect_response(_posted("/shape_range")) as response_info:
            page.locator("#shapeQuadratic").click()
        assert response_info.value.status == 200
        _settled_after_refit(page)

        declared = session.model._specs["age_band"]._spline_obj.polynomial_ranges
        assert [(r.lo, r.hi, r.degree) for r in declared] == [("25-34", "45-54", 2)]
        # The shaded range runs between the edge bands' positions, where the
        # pinned piece ends, so ranges sharing an edge band meet there.
        extent = page.evaluate(
            """() => {
                const svg = document.querySelector('#chart');
                const { sx, x } = svg._scale;
                const rect = svg.querySelector('.shape-range rect');
                const left = Number(rect.getAttribute('x'));
                return {
                    left,
                    right: left + Number(rect.getAttribute('width')),
                    expected: [sx(x[1]), sx(x[3])],
                };
            }"""
        )
        assert [extent["left"], extent["right"]] == pytest.approx(extent["expected"], abs=1e-9)
        label = page.locator("#chart .shape-range-label")
        assert label.text_content() == "Quadratic"


def test_back_to_back_runs_give_ranges_that_meet(open_editor_page):
    with open_editor_page() as (page, session):
        for run, icon in (((60, 100), "#shapeLine"), ((101, 140), "#shapeFlat")):
            session.select_indices("curve", list(range(run[0], run[1] + 1)))
            _reload_editor(page, "curve")
            page.locator("#selectionMenu").wait_for(state="visible")
            with page.expect_response(_posted("/shape_range")) as response_info:
                page.locator(icon).click()
            assert response_info.value.status == 200
            _settled_after_refit(page)

        spec = session.model._specs["curve"]
        first, second = spec.polynomial_ranges
        assert (first.degree, second.degree) == (1, 0)
        assert first.hi == second.lo
        # Snapped outward on its own, the second run would start past the first
        # range and leave a free sliver, with a kink at each end, between them.
        grid = session.terms["curve"].x
        assert _numeric_edges(spec, grid[101], grid[140])[0] > first.hi


def _drawn_y(page) -> np.ndarray:
    return np.asarray(page.evaluate("() => document.querySelector('#chart')._scale.y"))


def test_hold_levels_a_tail_at_its_first_value_and_undo_restores_it(open_editor_page):
    with open_editor_page() as (page, session):
        n = session.terms["curve"].size
        start = n - n // 5
        session.select_indices("curve", list(range(start, n)))
        _reload_editor(page, "curve")
        page.locator("#selectionMenu").wait_for(state="visible")
        before = _drawn_y(page)
        # Precondition: the tail is not already flat, so holding it moves it.
        assert np.ptp(before[start:]) > 0

        hold = page.locator("#shapeHold")
        assert hold.get_attribute("aria-disabled") == "false"
        with page.expect_response(_posted("/op")) as response_info:
            hold.click()
        assert response_info.value.status == 200
        assert response_info.value.request.post_data_json == {"operation": "level_left"}
        page.wait_for_function("() => !document.querySelector('#undoAction').disabled")

        # The payload plots exp of the edited log effect, and a JSON float64
        # round-trips exactly, so the held tail repeats its first value bit for
        # bit and the free points are the very values drawn before.
        after = _drawn_y(page)
        np.testing.assert_array_equal(after[start:], np.full(n - start, before[start]))
        np.testing.assert_array_equal(after[:start], before[:start])
        assert session.history[-1].operation == "level_left"

        with page.expect_response(_posted("/op")):
            page.locator("#undoAction").click()
        page.wait_for_function("() => document.querySelector('#undoAction').disabled")
        np.testing.assert_array_equal(_drawn_y(page), before)


def test_feature_search_filters_the_list_and_opens_the_first_match(open_editor_page):
    with open_editor_page() as (page, _session):
        feature_list = page.get_by_role("navigation", name="Features")
        # At 1180px the list opens collapsed to a strip; the analyst opens it.
        assert feature_list.get_attribute("data-open") == "false"
        page.locator("#featureListToggle").click()
        assert feature_list.get_attribute("data-open") == "true"
        search = page.get_by_role("searchbox", name="Search features")
        rows = feature_list.locator("[data-term]")
        assert rows.count() == 4

        search.fill("TERR")
        assert [row.get_attribute("data-term") for row in rows.all()] == ["territory"]
        with page.expect_response(_posted("/term")):
            search.press("Enter")
        page.wait_for_function(
            "() => document.querySelector('#status')?.dataset.term === 'territory'"
        )
        current = feature_list.locator('[aria-current="true"]')
        assert current.get_attribute("data-term") == "territory"
        # The shape icons stay hidden on an unordered categorical.
        assert page.locator("#shapeLine").is_hidden()

        search.press("Escape")
        assert search.input_value() == ""
        assert rows.count() == 4


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
        reference = page.locator("#termReference")
        refresh.wait_for(state="visible")
        page.wait_for_function("() => !document.querySelector('#refreshAction').disabled")
        session.select_levels("territory", ["T01", "T02"])
        session.replace_with_collapsed_levels("territory", method="fit")
        assert restore.is_hidden()
        assert page.locator("#chart .level-group-marker").count() == 0
        assert reference.text_content() == "reference T01 · first"

        refresh.click()
        restore.wait_for(state="visible")
        page.locator("#chart .level-group-marker").first.wait_for()
        assert page.locator("#chart .level-group-marker").count() == 2
        # The first level now sits inside the new group, so the group is the reference.
        assert reference.text_content() == "reference T01+T02 · first"


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
