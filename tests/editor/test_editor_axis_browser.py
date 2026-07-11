from __future__ import annotations

import pytest

pytest.importorskip("playwright.sync_api")
pytestmark = pytest.mark.browser


@pytest.mark.parametrize("width", [1180, 1048, 900])
def test_categorical_ticks_stay_inside_svg_and_above_title(open_editor_page, width):
    with open_editor_page(selected_term="territory", viewport={"width": width, "height": 720}) as (
        page,
        _session,
    ):
        result = page.locator("#chart").evaluate(
            """svg => {
                const bounds = node => {
                    const box = node.getBoundingClientRect();
                    return {
                        left: box.left,
                        right: box.right,
                        top: box.top,
                        bottom: box.bottom,
                    };
                };
                const title = svg.querySelector('.x-axis-title');
                const ticks = [...svg.querySelectorAll('.x-tick-label')];
                return {
                    root: bounds(svg),
                    title: title ? bounds(title) : null,
                    ticks: ticks.map(bounds),
                    measurementCount: Number(svg.dataset.axisMeasurementCount),
                };
            }"""
        )

        assert result["title"] is not None
        assert result["ticks"]
        assert 0 < result["measurementCount"] <= 30
        for tick in result["ticks"]:
            assert tick["left"] >= result["root"]["left"] - 0.5
            assert tick["right"] <= result["root"]["right"] + 0.5
            assert tick["bottom"] < result["title"]["top"]
        assert result["title"]["left"] >= result["root"]["left"] - 0.5
        assert result["title"]["right"] <= result["root"]["right"] + 0.5
        assert result["title"]["bottom"] <= result["root"]["bottom"] + 0.5


def test_long_labels_truncate_only_on_screen_and_keep_exact_model_strings(open_editor_page):
    full = "MyReallyLongCategoryNameThatWouldNeverFit"
    unicode_full = "Family👨‍👩‍👧‍👦DriverCaféCategory"

    with open_editor_page(
        selected_term="long_category", viewport={"width": 1048, "height": 720}
    ) as (page, session):
        original_levels = list(session.terms["long_category"].levels)
        assert full in original_levels
        assert unicode_full in original_levels

        tick = page.locator(f'.x-tick-label[aria-label="{full}"]')
        assert tick.count() == 1
        assert tick.text_content().endswith("…")
        assert tick.get_attribute("data-full-label") == full
        tick.focus()
        popover = page.get_by_role("tooltip")
        assert popover.is_visible()
        assert full in popover.inner_text()

        unicode_tick = page.locator(f'.x-tick-label[aria-label="{unicode_full}"]')
        assert unicode_tick.count() == 1
        assert unicode_tick.get_attribute("data-full-label") == unicode_full
        assert "�" not in unicode_tick.text_content()

        level_index = original_levels.index(full)
        point = page.locator(f'#chart .point[data-index="{level_index}"]')
        assert point.count() == 1
        point.hover()
        point_label = page.locator("#chart .point-tooltip-label")
        assert point_label.text_content() == full

        assert list(session.terms["long_category"].levels) == original_levels
        assert all(not level.endswith("…") for level in session.terms["long_category"].levels)
        assert page.locator("#chart").get_attribute("data-axis-measurement-count") == "10"


def test_identical_categorical_redraw_reuses_text_measurements(open_editor_page):
    with open_editor_page(viewport={"width": 1048, "height": 720}) as (page, _session):
        page.evaluate(
            """() => {
                const original = SVGTextElement.prototype.getComputedTextLength;
                window.__axisMeasurementCalls = 0;
                SVGTextElement.prototype.getComputedTextLength = function () {
                    window.__axisMeasurementCalls += 1;
                    return original.call(this);
                };
            }"""
        )

        with page.expect_response(
            lambda response: (
                response.request.method == "POST"
                and response.url.split("?", maxsplit=1)[0].endswith("/term")
            )
        ):
            page.select_option("#term", "long_category")
        page.wait_for_function(
            "term => document.querySelector('#status')?.dataset.term === term",
            arg="long_category",
        )
        page.locator("#chart .x-tick-label").first.wait_for()

        initial_calls = page.evaluate("window.__axisMeasurementCalls")
        assert initial_calls > 0

        page.get_by_role("button", name="Reference CI").click()
        page.wait_for_function(
            "() => document.querySelector('#ciToggle')?.getAttribute('aria-pressed') === 'true'"
        )
        assert page.evaluate("window.__axisMeasurementCalls") == initial_calls


def test_zoom_between_categories_does_not_draw_an_out_of_domain_tick(open_editor_page):
    with open_editor_page(selected_term="territory") as (page, _session):
        zoom = page.locator("#chart").evaluate(
            """svg => {
                for (let index = 0; index < 24; index += 1) {
                    const scale = svg._scale;
                    const x = scale.margin.left
                        + ((0.5 - scale.xMin) / (scale.xMax - scale.xMin)) * scale.innerW;
                    const y = scale.margin.top + scale.innerH / 2;
                    const point = svg.createSVGPoint();
                    point.x = x;
                    point.y = y;
                    const client = point.matrixTransform(svg.getScreenCTM());
                    svg.dispatchEvent(new WheelEvent('wheel', {
                        bubbles: true,
                        cancelable: true,
                        clientX: client.x,
                        clientY: client.y,
                        deltaY: -1,
                    }));
                }
                return { xMin: svg._scale.xMin, xMax: svg._scale.xMax };
            }"""
        )

        assert 0 < zoom["xMin"] < zoom["xMax"] < 1
        assert page.locator("#chart .x-tick-label").count() == 0
