from __future__ import annotations

import re

import pytest

pytest.importorskip("playwright.sync_api")
pytestmark = pytest.mark.browser


def select_chart_tool(page, name: str) -> None:
    page.get_by_role("radiogroup", name="Chart tools").get_by_role(
        "radio", name=name, exact=True
    ).click()


def remember_selection_dom(page) -> None:
    page.evaluate(
        """() => {
            window.__selectionDom = {
                chart: document.querySelector('#chart'),
                chartDescendants: Array.from(document.querySelectorAll('#chart *')),
                editedPath: document.querySelector('#chart path.edited'),
                firstPoint: document.querySelector('#chart circle.point[data-index]'),
                firstPointIndex: document.querySelector(
                    '#chart circle.point[data-index]'
                )?.dataset.index,
                firstAxis: document.querySelector('#chart line.axis'),
                chartTitle: document.querySelector('#chart text.label:not(.x-axis-title)'),
                xAxisTitle: document.querySelector('#chart .x-axis-title'),
                termOptions: Array.from(document.querySelectorAll('#featureRows [data-term]')),
            };
        }"""
    )


def selection_dom_is_unchanged(page) -> bool:
    return page.evaluate(
        """() => {
            const before = window.__selectionDom;
            const options = Array.from(document.querySelectorAll('#featureRows [data-term]'));
            return before.editedPath === document.querySelector('#chart path.edited')
                && before.firstPoint === document.querySelector(
                    `#chart circle.point[data-index="${before.firstPointIndex}"]`
                )
                && before.firstAxis === document.querySelector('#chart line.axis')
                && before.chartTitle === document.querySelector(
                    '#chart text.label:not(.x-axis-title)'
                )
                && before.xAxisTitle === document.querySelector('#chart .x-axis-title')
                && before.termOptions.length === options.length
                && before.termOptions.every((option, index) => option === options[index])
                && before.chartDescendants.every(
                    node => node.isConnected && node.closest('#chart') === before.chart
                );
        }"""
    )


def is_select_request(request) -> bool:
    return request.method == "POST" and request.url.split("?", maxsplit=1)[0].endswith("/select")


def selection_visual_state(page) -> dict:
    return page.evaluate(
        """() => {
            const bounds = document.querySelector('#chart .selection-bounds');
            return {
                selectedIndices: Array.from(
                    document.querySelectorAll('#chart circle.point.selected[data-index]')
                ).map(point => Number(point.dataset.index)),
                bounds: bounds ? {
                    x: bounds.getAttribute('x'),
                    y: bounds.getAttribute('y'),
                    width: bounds.getAttribute('width'),
                    height: bounds.getAttribute('height'),
                } : null,
                status: document.querySelector('#status')?.textContent,
                menuHidden: document.querySelector('#selectionMenu')?.hidden,
            };
        }"""
    )


def test_real_editor_boots_and_draws_svg(open_editor_page):
    with open_editor_page() as (page, _session):
        edited_path = page.locator("#chart path.edited").first

        assert page.title() == "SuperGLM Editor"
        assert page.locator("#chart").get_attribute("role") == "img"
        assert (
            page.locator('#featureList [aria-current="true"]').get_attribute("data-term") == "curve"
        )
        geometry = edited_path.evaluate(
            """path => ({
                namespace: path.namespaceURI,
                command: path.getAttribute('d')?.slice(0, 1),
                length: path.getTotalLength(),
            })"""
        )
        assert geometry["namespace"] == "http://www.w3.org/2000/svg"
        assert geometry["command"] == "M"
        assert geometry["length"] > 0


def test_selection_noop_empty_box_preserves_chart_without_posting(open_editor_page):
    with open_editor_page() as (page, session):
        select_chart_tool(page, "Select")
        initial_revision = session.model_revision
        select_requests = []
        page.on(
            "request",
            lambda request: select_requests.append(request) if is_select_request(request) else None,
        )
        remember_selection_dom(page)
        drag = page.evaluate(
            """() => {
                const svg = document.querySelector('#chart');
                const scale = svg._scale;
                const clientPoint = (x, y) => {
                    const point = svg.createSVGPoint();
                    point.x = x;
                    point.y = y;
                    const client = point.matrixTransform(svg.getScreenCTM());
                    return { x: client.x, y: client.y };
                };
                window.__selectionMutations = [];
                window.__selectionObserver = new MutationObserver(records => {
                    for (const record of records) {
                        if (record.type === 'attributes') {
                            window.__selectionMutations.push({
                                type: 'attribute',
                                className: record.target.getAttribute('class'),
                                attribute: record.attributeName,
                            });
                            continue;
                        }
                        for (const node of record.addedNodes) {
                            if (node.nodeType === Node.ELEMENT_NODE) {
                                window.__selectionMutations.push({
                                    type: 'added',
                                    className: node.getAttribute('class'),
                                });
                            }
                        }
                        for (const node of record.removedNodes) {
                            if (node.nodeType === Node.ELEMENT_NODE) {
                                window.__selectionMutations.push({
                                    type: 'removed',
                                    className: node.getAttribute('class'),
                                });
                            }
                        }
                    }
                });
                window.__selectionObserver.observe(svg, {
                    attributes: true,
                    childList: true,
                    subtree: true,
                });
                return {
                    start: clientPoint(scale.margin.left + 4, scale.margin.top + 4),
                    end: clientPoint(scale.margin.left + 20, scale.margin.top + 20),
                };
            }"""
        )

        page.mouse.move(drag["start"]["x"], drag["start"]["y"])
        page.mouse.down()
        page.mouse.move(drag["end"]["x"], drag["end"]["y"], steps=3)
        page.mouse.up()
        page.evaluate(
            """() => new Promise(resolve => requestAnimationFrame(
                () => requestAnimationFrame(resolve)
            ))"""
        )
        mutations = page.evaluate(
            """() => {
                window.__selectionObserver.disconnect();
                return window.__selectionMutations;
            }"""
        )

        assert session.selection("curve").tolist() == []
        assert session.model_revision == initial_revision
        assert select_requests == []
        assert selection_dom_is_unchanged(page)
        assert {mutation["type"] for mutation in mutations} >= {"added", "removed"}
        # Only the brush comes and goes; the lens that shows the points near
        # the pointer toggles a class on circles that are already drawn.
        assert {mutation["className"] for mutation in mutations} <= {"brush", "point", "point near"}
        assert {
            mutation["className"] for mutation in mutations if mutation["type"] != "attribute"
        } == {"brush"}


def test_selection_incremental_feedback_precedes_delayed_backend_success(open_editor_page):
    with open_editor_page() as (page, session):
        select_chart_tool(page, "Select")
        initial_revision = session.model_revision
        select_requests = []
        held_routes = []
        page.on(
            "request",
            lambda request: select_requests.append(request) if is_select_request(request) else None,
        )
        page.route("**/select", lambda route: held_routes.append(route))
        remember_selection_dom(page)
        point = page.locator("#chart circle.point[data-index]").first
        selected_index = int(point.get_attribute("data-index"))

        with page.expect_request(is_select_request):
            point.click()

        assert len(held_routes) == 1
        assert session.selection("curve").tolist() == []
        page.wait_for_function(
            """index => {
                const point = document.querySelector(
                    `#chart circle.point[data-index="${index}"]`
                );
                return point?.classList.contains('selected')
                    && point.getAttribute('r') === '4.6'
                    && document.querySelector('#status')?.textContent.startsWith('1 of ');
            }""",
            arg=selected_index,
        )
        assert selection_dom_is_unchanged(page)

        with page.expect_response(
            lambda response: is_select_request(response.request)
        ) as response_info:
            backend_response = held_routes[0].fetch()
            payload = backend_response.json()
            payload["terms"]["curve"]["impact"] = {
                **payload["terms"]["curve"]["impact"],
                "weighted_mean_relativity": 1.23,
                "selected_weight_share": 0.45,
            }
            held_routes[0].fulfill(response=backend_response, json=payload)

        assert response_info.value.status == 200
        page.wait_for_function("() => document.querySelector('#appBusyOverlay')?.hidden")
        assert "average edit relativity 1.23x" in page.locator("#status").inner_text()
        assert "selected exposure 45%" in page.locator("#status").inner_text()
        assert session.selection("curve").tolist() == [selected_index]
        assert session.model_revision == initial_revision
        assert len(select_requests) == 1
        assert selection_dom_is_unchanged(page)


def test_failed_delayed_selection_recovers_without_rebuilding_chart(open_editor_page):
    with open_editor_page() as (page, session):
        # The error banner this test raises takes a row above the editor, and a
        # chart whose panel shrinks is redrawn at its new size. Pin the panel at
        # its current height so the only redraw in play is the one the
        # recovery path must not make.
        page.locator(".chart-shell").evaluate(
            "node => { node.style.height = `${node.getBoundingClientRect().height}px`; }"
        )
        select_chart_tool(page, "Select")
        term = "curve"
        total_points = session.terms[term].size
        authoritative_selection = list(range(total_points))

        with page.expect_response(
            lambda response: is_select_request(response.request)
        ) as initial_response:
            page.locator('button[data-op="select_all"]').click()
        assert initial_response.value.status == 200
        page.wait_for_function("() => document.querySelector('#appBusyOverlay')?.hidden")
        assert session.selection(term).tolist() == authoritative_selection

        remember_selection_dom(page)
        authoritative_visuals = selection_visual_state(page)
        held_routes = []
        page.route("**/select", lambda route: held_routes.append(route))
        candidate = page.locator("#chart circle.point[data-index]").nth(1)
        candidate_index = int(candidate.get_attribute("data-index"))

        with page.expect_request(is_select_request) as request_info:
            candidate.click()

        assert len(held_routes) == 1
        assert request_info.value.post_data_json == {
            "term": term,
            "indices": [candidate_index],
        }
        page.wait_for_function(
            """index => {
                const point = document.querySelector(
                    `#chart circle.point[data-index="${index}"]`
                );
                return point?.classList.contains('selected')
                    && point.getAttribute('r') === '4.6'
                    && document.querySelector('#status')?.textContent.startsWith('1 of ');
            }""",
            arg=candidate_index,
        )
        provisional_visuals = selection_visual_state(page)
        assert provisional_visuals["selectedIndices"] == [candidate_index]
        assert provisional_visuals["bounds"] != authoritative_visuals["bounds"]
        assert provisional_visuals["status"] != authoritative_visuals["status"]
        assert session.selection(term).tolist() == authoritative_selection
        assert selection_dom_is_unchanged(page)

        with page.expect_response(
            lambda response: (
                response.request.method == "GET"
                and response.url.split("?", maxsplit=1)[0].endswith("/state")
            )
        ) as recovery_response:
            held_routes[0].fulfill(
                status=500,
                content_type="application/json",
                body='{"error":"Selection rejected for browser test."}',
            )

        assert recovery_response.value.status == 200
        alert = page.locator("#appAlert")
        alert.wait_for(state="visible")
        page.wait_for_function("() => document.querySelector('#appBusyOverlay')?.hidden")
        page.wait_for_function(
            "expected => document.querySelector('#status')?.textContent === expected",
            arg=authoritative_visuals["status"],
        )

        assert "Selection rejected for browser test." in alert.inner_text()
        assert page.locator("#appAlertRetry").is_visible()
        assert selection_visual_state(page) == authoritative_visuals
        assert session.selection(term).tolist() == authoritative_selection
        assert selection_dom_is_unchanged(page)


def test_collapsed_categorical_selection_posts_exact_source_indices_without_redraw(
    open_editor_page,
):
    collapsed_levels = ("T02", "T03")
    with open_editor_page(
        selected_term="territory",
        collapsed_levels=("territory", collapsed_levels),
    ) as (page, session):
        select_chart_tool(page, "Select")
        assert page.locator("#groupDisplayMode").is_enabled()
        assert page.locator("#groupDisplayMode").input_value() == "expanded"
        page.select_option("#groupDisplayMode", "collapsed")
        page.wait_for_function("() => document.querySelector('#chart')?._scale?.displayIsCollapsed")

        mapping = page.evaluate(
            "() => document.querySelector('#chart')._scale.displayToSourceIndices"
        )
        display_index = next(i for i, source in enumerate(mapping) if len(source) > 1)
        source_indices = mapping[display_index]
        levels = session.terms["territory"].levels
        assert [levels[index] for index in source_indices] == list(collapsed_levels)
        assert session.selection("territory").tolist() == []

        held_routes = []
        page.route("**/select", lambda route: held_routes.append(route))
        remember_selection_dom(page)
        point = page.locator(f'#chart circle.point[data-index="{display_index}"]')

        with page.expect_request(is_select_request) as request_info:
            point.click()

        assert len(held_routes) == 1
        assert request_info.value.post_data_json == {
            "term": "territory",
            "indices": source_indices,
        }
        assert session.selection("territory").tolist() == []
        page.wait_for_function(
            """({displayIndex, sourceCount}) => {
                const point = document.querySelector(
                    `#chart circle.point[data-index="${displayIndex}"]`
                );
                return point?.classList.contains('selected')
                    && point.getAttribute('r') === '4.6'
                    && document.querySelector('#status')?.textContent.startsWith(
                        `${sourceCount} of `
                    );
            }""",
            arg={"displayIndex": display_index, "sourceCount": len(source_indices)},
        )
        assert page.locator("#chart circle.point.selected[data-index]").count() == 1
        assert page.locator("#selectionMenu").is_visible()
        assert page.locator("#ungroupLevels").is_visible()
        assert (
            "original line is grouped by exposure-weighted averaging"
            in page.locator("#status").inner_text()
        )
        assert selection_dom_is_unchanged(page)

        with page.expect_response(
            lambda response: is_select_request(response.request)
        ) as response_info:
            held_routes[0].continue_()

        assert response_info.value.status == 200
        page.wait_for_function("() => document.querySelector('#appBusyOverlay')?.hidden")
        assert session.selection("territory").tolist() == source_indices
        assert page.locator("#chart circle.point.selected[data-index]").count() == 1
        assert page.locator("#selectionMenu").is_visible()
        assert page.locator("#ungroupLevels").is_visible()
        assert (
            page.locator("#status")
            .inner_text()
            .startswith(f"{len(source_indices)} of {session.terms['territory'].size} selected")
        )
        assert selection_dom_is_unchanged(page)


def test_expanded_group_members_remain_individually_selectable_for_regrouping(
    open_editor_page,
):
    with open_editor_page(
        selected_term="territory",
        collapsed_levels=("territory", ("T02", "T03")),
    ) as (page, session):
        select_chart_tool(page, "Select")
        assert page.locator("#groupDisplayMode").input_value() == "expanded"
        source_index = session.terms["territory"].levels.index("T02")
        point = page.locator(f'#chart circle.point[data-index="{source_index}"]')

        with page.expect_response(
            lambda response: is_select_request(response.request)
        ) as response_info:
            point.click()

        assert response_info.value.status == 200
        assert response_info.value.request.post_data_json == {
            "term": "territory",
            "indices": [source_index],
        }
        page.wait_for_function("() => document.querySelector('#appBusyOverlay')?.hidden")
        assert session.selection("territory").tolist() == [source_index]
        assert page.locator("#chart circle.point.selected[data-index]").count() == 1


def test_expanded_structural_group_moves_without_preview_jump(open_editor_page):
    grouped_levels = ("T02", "T03")
    with open_editor_page(
        selected_term="territory",
        collapsed_levels=("territory", grouped_levels),
    ) as (page, session):
        levels = session.terms["territory"].levels
        source_indices = [levels.index(level) for level in grouped_levels]
        points = [
            page.locator(f'#chart circle.point[data-index="{index}"]') for index in source_indices
        ]
        select_chart_tool(page, "Move")
        box = points[0].bounding_box()
        assert box is not None
        x = box["x"] + box["width"] / 2
        y = box["y"] + box["height"] / 2
        original_cy = float(points[0].get_attribute("cy"))

        page.mouse.move(x, y)
        page.mouse.down()
        page.mouse.move(x, y - 30)
        page.wait_for_function(
            """({indices, originalCy}) => {
                const values = indices.map(index => Number(document.querySelector(
                    `#chart circle.point[data-index="${index}"]`
                )?.getAttribute('cy')));
                return values.every(Number.isFinite)
                    && Math.abs(values[0] - values[1]) < 1e-6
                    && Math.abs(values[0] - originalCy) > 1;
            }""",
            arg={"indices": source_indices, "originalCy": original_cy},
        )
        preview_cy = float(points[0].get_attribute("cy"))

        with page.expect_response(
            lambda response: (
                response.request.method == "POST"
                and response.url.split("?", maxsplit=1)[0].endswith("/drag")
            )
        ) as drag_info:
            page.mouse.up()

        request = drag_info.value.request.post_data_json
        assert request["indices"] == [source_indices[0]]
        page.wait_for_function(
            """({indices, previewCy}) => indices.every(index => {
                const value = Number(document.querySelector(
                    `#chart circle.point[data-index="${index}"]`
                )?.getAttribute('cy'));
                return Number.isFinite(value) && Math.abs(value - previewCy) < 1e-6;
            })""",
            arg={"indices": source_indices, "previewCy": preview_cy},
        )
        assert session.selection("territory").tolist() == [source_indices[0]]
        effects = session.terms["territory"].edited_log_effect[source_indices]
        assert effects[0] == pytest.approx(effects[1])


def test_selection_menu_does_not_block_adjacent_modifier_selection(open_editor_page):
    with open_editor_page(selected_term="territory") as (page, session):
        select_chart_tool(page, "Select")
        points = page.locator("#chart circle.point[data-index]")
        first = points.nth(1)
        adjacent = points.nth(2)
        first_index = int(first.get_attribute("data-index"))
        adjacent_index = int(adjacent.get_attribute("data-index"))
        adjacent = page.locator(f'#chart circle.point[data-index="{adjacent_index}"]')

        with page.expect_response(lambda response: is_select_request(response.request)):
            first.click()
        page.locator("#selectionMenu").wait_for(state="visible")

        hit = adjacent.evaluate(
            """point => {
                const box = point.getBoundingClientRect();
                const target = document.elementFromPoint(
                    box.left + box.width / 2,
                    box.top + box.height / 2
                );
                const menuBox = document.querySelector('#selectionMenu').getBoundingClientRect();
                return {
                    index: target?.closest('circle.point[data-index]')?.dataset.index ?? null,
                    target: target?.getAttribute('aria-label') || target?.tagName || null,
                    point: { left: box.left, top: box.top, width: box.width, height: box.height },
                    menu: {
                        left: menuBox.left,
                        top: menuBox.top,
                        width: menuBox.width,
                        height: menuBox.height,
                    },
                };
            }"""
        )
        assert hit["index"] == str(adjacent_index), hit

        with page.expect_response(lambda response: is_select_request(response.request)):
            adjacent.click(modifiers=["Control"])

        assert session.selection("territory").tolist() == [first_index, adjacent_index]
        assert page.locator("#chart circle.point.selected[data-index]").count() == 2
        assert page.locator("#selectionMenu").is_visible()


def _click_selects(page, point, modifiers=(), timeout=30000) -> None:
    """Click, wait for its /select, and let the page settle before the next click.

    A selection posts without the busy overlay, and a click that lands while
    one is still running is skipped, so the next click waits for it.
    """
    with page.expect_response(
        lambda response: is_select_request(response.request), timeout=timeout
    ):
        point.click(modifiers=list(modifiers))
    page.wait_for_function("() => window.__superglmTest?.mutationStatus?.() !== 'running'")


def test_click_shift_click_selects_a_span_and_ctrl_click_toggles_on_a_spline(open_editor_page):
    # Thirty points draw every marker, so the selection palette steps around them.
    with open_editor_page(n_points=30) as (page, session):
        select_chart_tool(page, "Select")

        def point(index: int):
            return page.locator(f'#chart circle.point[data-index="{index}"]')

        _click_selects(page, point(8))
        _click_selects(page, point(14), modifiers=["Shift"], timeout=5000)
        assert session.selection("curve").tolist() == list(range(8, 15))
        # The Shift press did not pan: the chart still shows its whole x range.
        assert page.evaluate(
            "() => { const s = document.querySelector('#chart')._scale;"
            " return s.xMin === s.baseXMin && s.xMax === s.baseXMax; }"
        )

        _click_selects(page, point(11), modifiers=["Control"])
        assert session.selection("curve").tolist() == [8, 9, 10, 12, 13, 14]
        _click_selects(page, point(20), modifiers=["Control"])
        assert session.selection("curve").tolist() == [8, 9, 10, 12, 13, 14, 20]
        # The last Ctrl-click is the anchor: Shift-click 17 spans 17 to 20.
        _click_selects(page, point(17), modifiers=["Shift"])
        assert session.selection("curve").tolist() == [17, 18, 19, 20]


_POINTS_UNDER_THE_MENU = """() => [...document.querySelectorAll('#chart circle.point[data-index]')]
  .filter((point) => {
    const box = point.getBoundingClientRect();
    const hit = document.elementFromPoint(box.left + box.width / 2, box.top + box.height / 2);
    return hit?.closest('#selectionMenu') != null;
  })
  .map((point) => Number(point.dataset.index))"""


@pytest.mark.parametrize("viewport", [(1180, 720), (1600, 1000)])
def test_the_selection_menu_leaves_every_point_of_a_dense_curve_clickable(
    open_editor_page, viewport
):
    """After a click, the next click or Shift-click can land on any point of the curve.

    A dense curve hides its points, but each is a click target, so the
    selection menu that opens beside the clicked point must not cover them.
    """
    width, height = viewport
    with open_editor_page(viewport={"width": width, "height": height}) as (page, session):
        select_chart_tool(page, "Select")

        def click_point(index: int, modifiers: tuple[str, ...] = ()) -> int:
            """Click where point ``index`` is drawn; return the point the click lands on.

            A hidden point is a target at its place; on a dense curve a
            neighbour drawn over it can take the click.
            """
            box = page.locator(f'#chart circle.point[data-index="{index}"]').bounding_box()
            x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
            landed = page.evaluate(
                "([x, y]) => Number(document.elementFromPoint(x, y)"
                "?.closest('circle.point[data-index]')?.dataset.index ?? -1)",
                [x, y],
            )
            with page.expect_response(lambda response: is_select_request(response.request)):
                for key in modifiers:
                    page.keyboard.down(key)
                page.mouse.click(x, y)
                for key in modifiers:
                    page.keyboard.up(key)
            page.wait_for_function("() => window.__superglmTest?.mutationStatus?.() !== 'running'")
            return landed

        for first in (20, 60, 100, 140, 180):
            anchor = click_point(first)
            page.locator("#selectionMenu").wait_for(state="visible")
            assert page.evaluate(_POINTS_UNDER_THE_MENU) == [], first

        # The click at 180 set the anchor: a Shift-click near 95 spans to it.
        landed = click_point(95, modifiers=("Shift",))
        assert abs(landed - 95) <= 1 and abs(anchor - 180) <= 1
        assert session.selection("curve").tolist() == list(range(landed, anchor + 1))


def _tagged_x(text: str, gesture: str) -> float:
    """The x a tag such as ``"click · 2.77"`` names."""
    prefix = f"{gesture} · "
    assert text.startswith(prefix), text
    return float(text[len(prefix) :])


def test_the_anchor_and_a_shift_click_end_are_ringed_and_tagged(open_editor_page):
    with open_editor_page(n_points=30) as (page, session):
        select_chart_tool(page, "Select")
        x = session.terms["curve"].x
        # Every x here lies in [1, 10), which the axis prints to two decimals.
        printed = 0.005

        def point(index: int):
            return page.locator(f'#chart circle.point[data-index="{index}"]')

        tags = page.locator("#chart .anchor-tag:visible")
        rings = page.locator("#chart circle.anchor-ring:visible")
        status = page.locator("#status")
        assert tags.count() == 0 and rings.count() == 0

        _click_selects(page, point(8))
        [anchor_tag] = tags.all_text_contents()
        assert abs(_tagged_x(anchor_tag, "click") - x[8]) <= printed
        assert rings.count() == 1
        assert rings.first.get_attribute("cx") == point(8).get_attribute("cx")
        assert status.text_content().startswith("1 of 30 selected · ")

        _click_selects(page, point(14), modifiers=["Shift"], timeout=5000)
        anchor_tag, end_tag = tags.all_text_contents()
        assert abs(_tagged_x(anchor_tag, "click") - x[8]) <= printed
        assert abs(_tagged_x(end_tag, "Shift-click") - x[14]) <= printed
        assert sorted(ring.get_attribute("cx") for ring in rings.all()) == sorted(
            point(i).get_attribute("cx") for i in (8, 14)
        )
        range_line = re.fullmatch(
            r"Range (\S+) – (\S+) · 7 of 30 points · selected exposure [\d.]+%",
            status.text_content(),
        )
        assert range_line, status.text_content()
        assert abs(float(range_line[1]) - x[8]) <= printed
        assert abs(float(range_line[2]) - x[14]) <= printed

        # A span the other way still reads low to high.
        _click_selects(page, point(3), modifiers=["Shift"], timeout=5000)
        range_line = re.fullmatch(
            r"Range (\S+) – (\S+) · 6 of 30 points · .*", status.text_content()
        )
        assert range_line, status.text_content()
        assert abs(float(range_line[1]) - x[3]) <= printed
        assert abs(float(range_line[2]) - x[8]) <= printed

        # While the pointer drags, the tags step aside; the rings stay.
        corner = page.evaluate(
            """() => {
                const svg = document.querySelector('#chart');
                const { margin } = svg._scale;
                const at = (dx, dy) => {
                    const p = svg.createSVGPoint();
                    p.x = margin.left + dx;
                    p.y = margin.top + dy;
                    const client = p.matrixTransform(svg.getScreenCTM());
                    return { x: client.x, y: client.y };
                };
                return { start: at(4, 4), end: at(24, 24) };
            }"""
        )
        page.mouse.move(corner["start"]["x"], corner["start"]["y"])
        page.mouse.down()
        page.mouse.move(corner["end"]["x"], corner["end"]["y"], steps=3)
        assert tags.count() == 0
        assert rings.count() == 2
        with page.expect_response(lambda response: is_select_request(response.request)):
            page.mouse.up()
        page.wait_for_function("() => window.__superglmTest?.mutationStatus?.() !== 'running'")

        # The box changed the selection, so it is no span any more: the anchor
        # keeps its tag, and the status line its selection sentence.
        [anchor_tag] = tags.all_text_contents()
        assert abs(_tagged_x(anchor_tag, "click") - x[8]) <= printed
        assert status.text_content().startswith("0 of 30 selected · ")


def test_select_all_is_incremental_bounded_and_keeps_bounds_behind_points(open_editor_page):
    with open_editor_page(n_points=500) as (page, session):
        select_chart_tool(page, "Select")
        initial_revision = session.model_revision
        total_points = session.terms["curve"].size
        initial_markers = page.locator("#chart circle.point[data-index]").count()
        select_requests = []
        select_all_ops = []
        held_routes = []

        def record_request(request) -> None:
            if is_select_request(request):
                select_requests.append(request)
            elif (
                request.method == "POST"
                and request.url.split("?", maxsplit=1)[0].endswith("/op")
                and request.post_data_json == {"operation": "select_all"}
            ):
                select_all_ops.append(request)

        page.on("request", record_request)
        page.route("**/select", lambda route: held_routes.append(route))
        remember_selection_dom(page)

        with page.expect_request(is_select_request, timeout=1500):
            page.locator('button[data-op="select_all"]').click()

        assert len(held_routes) == 1
        assert session.selection("curve").tolist() == []
        page.wait_for_function(
            "count => document.querySelector('#status')?.textContent.startsWith(`${count} of `)",
            arg=total_points,
        )
        assert selection_dom_is_unchanged(page)
        assert page.locator("#chart circle.point[data-index]").count() == initial_markers
        assert page.locator("#chart circle.point.selected[data-index]").count() == initial_markers
        layer_order = page.evaluate(
            """() => {
                const svg = document.querySelector('#chart');
                const children = Array.from(svg.children);
                return {
                    halo: children.indexOf(svg.querySelector('.selection-bounds-halo')),
                    bounds: children.indexOf(svg.querySelector('.selection-bounds')),
                    point: children.indexOf(svg.querySelector('.point-layer')),
                };
            }"""
        )
        assert 0 <= layer_order["halo"] < layer_order["bounds"] < layer_order["point"]

        with page.expect_response(
            lambda response: is_select_request(response.request)
        ) as response_info:
            held_routes[0].continue_()

        assert response_info.value.status == 200
        page.wait_for_function("() => document.querySelector('#appBusyOverlay')?.hidden")
        assert session.selection("curve").tolist() == list(range(total_points))
        assert session.model_revision == initial_revision
        assert len(select_requests) == 1
        assert select_all_ops == []
        assert page.locator("#chart circle.point[data-index]").count() == initial_markers
        assert selection_dom_is_unchanged(page)


def test_incremental_selection_raises_points_without_recreating_them(open_editor_page):
    with open_editor_page(n_points=500) as (page, _session):
        select_chart_tool(page, "Select")
        point = page.locator("#chart circle.point[data-index]").last
        selected_index = int(point.get_attribute("data-index"))
        point.evaluate("node => { window.__selectedPointIdentity = node; }")

        with page.expect_response(lambda response: is_select_request(response.request)):
            point.click()

        paint = page.evaluate(
            """selectedIndex => {
                const svg = document.querySelector('#chart');
                const layer = svg.querySelector('.point-layer');
                const legend = svg.querySelector('.legend-layer');
                const points = Array.from(layer.querySelectorAll(':scope > circle.point[data-index]'));
                const selected = layer.querySelector(
                    `:scope > circle.point.selected[data-index="${selectedIndex}"]`
                );
                const unselected = layer.querySelector(':scope > circle.point:not(.selected)');
                const selectedPositions = points
                    .map((node, index) => node.classList.contains('selected') ? index : -1)
                    .filter(index => index >= 0);
                const unselectedPositions = points
                    .map((node, index) => node.classList.contains('selected') ? -1 : index)
                    .filter(index => index >= 0);
                unselected.setAttribute('cx', selected.getAttribute('cx'));
                unselected.setAttribute('cy', selected.getAttribute('cy'));
                const box = selected.getBoundingClientRect();
                const hit = document.elementFromPoint(
                    box.left + box.width / 2,
                    box.top + box.height / 2
                );
                return {
                    identity: selected === window.__selectedPointIdentity,
                    order: Math.max(...unselectedPositions) < Math.min(...selectedPositions),
                    belowLegend: Array.from(svg.children).indexOf(layer)
                        < Array.from(svg.children).indexOf(legend),
                    hitIndex: hit?.closest('circle.point[data-index]')?.dataset.index ?? null,
                };
            }""",
            selected_index,
        )

        assert paint == {
            "identity": True,
            "order": True,
            "belowLegend": True,
            "hitIndex": str(selected_index),
        }


def test_sparse_box_selection_keeps_supplemental_points_below_legend(open_editor_page):
    with open_editor_page(n_points=500) as (page, _session):
        select_chart_tool(page, "Select")
        drag = page.evaluate(
            """() => {
                const svg = document.querySelector('#chart');
                const scale = svg._scale;
                const drawn = new Set(Array.from(
                    svg.querySelectorAll('circle.point[data-index]'),
                    node => Number(node.dataset.index)
                ));
                const target = scale.x.findIndex((_, index) => index > 10 && !drawn.has(index));
                const client = (x, y) => {
                    const point = svg.createSVGPoint();
                    point.x = x;
                    point.y = y;
                    const result = point.matrixTransform(svg.getScreenCTM());
                    return { x: result.x, y: result.y };
                };
                const cx = scale.sx(scale.x[target]);
                const cy = scale.sy(scale.y[target]);
                return { start: client(cx - 7, cy - 7), end: client(cx + 7, cy + 7) };
            }"""
        )

        with page.expect_response(lambda response: is_select_request(response.request)):
            page.mouse.move(drag["start"]["x"], drag["start"]["y"])
            page.mouse.down()
            page.mouse.move(drag["end"]["x"], drag["end"]["y"])
            page.mouse.up()

        layering = page.evaluate(
            """() => {
                const svg = document.querySelector('#chart');
                const layer = svg.querySelector('.point-layer');
                const legend = svg.querySelector('.legend-layer');
                const supplemental = Array.from(
                    layer.querySelectorAll('circle[data-selection-supplemental="true"]')
                );
                return {
                    count: supplemental.length,
                    allInLayer: supplemental.every(point => point.parentElement === layer),
                    belowLegend: Array.from(svg.children).indexOf(layer)
                        < Array.from(svg.children).indexOf(legend),
                };
            }"""
        )
        assert layering["count"] > 0
        assert layering["allInLayer"] is True
        assert layering["belowLegend"] is True


def test_selection_palette_wraps_inside_narrow_notebook_viewport(open_editor_page):
    with open_editor_page(viewport={"width": 360, "height": 560}, selected_term="territory") as (
        page,
        _session,
    ):
        select_chart_tool(page, "Select")
        baseline_document_width = page.evaluate("document.documentElement.scrollWidth")
        with page.expect_response(lambda response: is_select_request(response.request)):
            page.locator('button[data-op="select_all"]').click()
        page.locator("#selectionMenu").wait_for(state="visible")

        layout = page.evaluate(
            """() => {
                const menu = document.querySelector('#selectionMenu');
                const chart = document.querySelector('#chart');
                const menuBox = menu.getBoundingClientRect();
                const chartBox = chart.getBoundingClientRect();
                const buttons = Array.from(menu.children).flatMap(child => {
                    if (child.matches(':scope > button.selection-item')) return [child];
                    const button = child.querySelector(':scope > button.selection-item');
                    return button ? [button] : [];
                }).filter(button => button.getClientRects().length > 0);
                return {
                    scrollFits: menu.scrollWidth <= menu.clientWidth,
                    documentWidth: document.documentElement.scrollWidth,
                    viewportWidth: window.innerWidth,
                    menuLeft: menuBox.left,
                    menuRight: menuBox.right,
                    chartLeft: chartBox.left,
                    chartRight: chartBox.right,
                    buttons: buttons.map(button => {
                        const box = button.getBoundingClientRect();
                        const hit = document.elementFromPoint(
                            box.left + box.width / 2,
                            box.top + box.height / 2
                        );
                        return {
                            width: box.width,
                            height: box.height,
                            insideMenu: box.left >= menuBox.left && box.right <= menuBox.right,
                            insideChart: box.left >= chartBox.left && box.right <= chartBox.right,
                            hit: hit?.closest('button') === button,
                        };
                    }),
                };
            }"""
        )

        assert layout["scrollFits"] is True
        assert layout["documentWidth"] <= baseline_document_width, layout
        assert layout["buttons"]
        for button in layout["buttons"]:
            assert button["width"] == pytest.approx(34, abs=0.5)
            assert button["height"] == pytest.approx(34, abs=0.5)
            assert button["insideMenu"] is True
            assert button["insideChart"] is True
            assert button["hit"] is True


def test_selection_popovers_keep_focus_and_explain_parent_icons(open_editor_page):
    with open_editor_page() as (page, _session):
        select_chart_tool(page, "Select")
        page.locator('button[data-op="select_all"]').click()
        menu = page.locator("#selectionMenu")
        menu.wait_for(state="visible")

        trigger = menu.locator('button[data-op="linearise"]')
        popover = page.locator("#uiPopover")
        trigger.hover()
        popover.wait_for(state="visible")
        trigger.focus()
        page.mouse.move(4, 4)

        assert popover.is_visible()
        assert trigger.get_attribute("aria-describedby") == "uiPopover"

        page.locator("#ciToggle").focus()
        popover.wait_for(state="hidden")
        assert trigger.get_attribute("aria-describedby") is None

        trigger.focus()
        popover.wait_for(state="visible")
        assert popover.locator("[data-popover-heading]").inner_text() == "Straighten selection"
        assert popover.locator("[data-popover-description]").inner_text() == (
            "Interpolate the selected relativities between their first and last points."
        )
        page.keyboard.press("Escape")
        popover.wait_for(state="hidden")
        assert trigger.get_attribute("aria-describedby") is None

        for control, heading in [
            ("level", "Level selected values"),
            ("snap", "Snap selected values"),
        ]:
            parent = menu.locator(f'button[data-help-control="{control}"]')
            parent.hover()
            popover.wait_for(state="visible")
            assert popover.locator("[data-popover-heading]").inner_text() == heading
            page.mouse.move(4, 4)
            popover.wait_for(state="hidden")


def test_tool_rail_popover_closes_when_tool_is_activated(open_editor_page):
    with open_editor_page() as (page, _session):
        trigger = page.get_by_role("radiogroup", name="Chart tools").get_by_role(
            "radio", name="Move", exact=True
        )
        popover = page.locator("#uiPopover")

        trigger.hover()
        popover.wait_for(state="visible")
        trigger.click()

        popover.wait_for(state="hidden")
        assert trigger.get_attribute("aria-describedby") is None
        page.wait_for_timeout(450)
        assert popover.is_hidden()


def test_ordinary_mutations_never_show_the_global_busy_overlay(open_editor_page, choose_feature):
    with open_editor_page() as (page, _session):
        held_select = []
        page.route("**/select", lambda route: held_select.append(route))
        point = page.locator("#chart circle.point[data-index]").first
        with page.expect_request(is_select_request):
            point.click()

        assert len(held_select) == 1
        assert page.locator("#appBusyOverlay").is_hidden()
        assert page.locator("#editorView").get_attribute("inert") is None
        assert page.evaluate("document.activeElement?.id") != "appBusyAnnouncement"
        with page.expect_response(lambda response: is_select_request(response.request)):
            held_select[0].continue_()
        page.unroute("**/select")

        select_chart_tool(page, "Handles")
        held_control = []
        page.route("**/control", lambda route: held_control.append(route))
        handle = page.locator("#chart .control-handle").first
        box = handle.bounding_box()
        assert box is not None
        with page.expect_request(
            lambda request: request.method == "POST" and request.url.endswith("/control")
        ):
            page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
            page.mouse.down()
            page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2 - 12)
            page.mouse.up()
        page.wait_for_timeout(50)
        assert len(held_control) == 1
        assert page.locator("#appBusyOverlay").is_hidden()
        assert page.locator("#editorView").get_attribute("inert") is None
        assert page.evaluate("document.activeElement?.id") != "appBusyAnnouncement"
        with page.expect_response(
            lambda response: response.request.method == "POST" and response.url.endswith("/control")
        ):
            held_control[0].continue_()
        page.unroute("**/control")

        held_term = []
        page.route("**/term", lambda route: held_term.append(route))
        with page.expect_request(
            lambda request: request.method == "POST" and request.url.endswith("/term")
        ):
            choose_feature(page, "territory")

        page.wait_for_timeout(50)
        assert len(held_term) == 1
        assert page.locator("#appBusyOverlay").is_hidden()
        assert page.locator("#editorView").get_attribute("inert") is None
        assert page.evaluate("document.activeElement?.id") != "appBusyAnnouncement"
        with page.expect_response(
            lambda response: response.request.method == "POST" and response.url.endswith("/term")
        ):
            held_term[0].continue_()


def test_term_change_does_not_rewrite_unrelated_editor_panels(open_editor_page, choose_feature):
    with open_editor_page() as (page, _session):
        page.locator("#metricGrid > *").first.wait_for()
        page.locator("#summaryFrame > *").first.wait_for()
        page.locator("#historyFrame > *").first.wait_for(state="attached")
        page.evaluate(
            """() => {
                window.__panelBoundaryNodes = {
                    editedPath: document.querySelector('#chart path.edited'),
                    metric: document.querySelector('#metricGrid > *'),
                    summary: document.querySelector('#summaryFrame > *'),
                    history: document.querySelector('#historyFrame > *'),
                    featureRows: document.querySelector('#featureRows'),
                    features: Array.from(
                        document.querySelectorAll('#featureRows [data-term]'),
                        row => row.dataset.term,
                    ),
                };
            }"""
        )

        with page.expect_response(
            lambda response: response.request.method == "POST" and response.url.endswith("/term")
        ):
            choose_feature(page, "territory")
        page.wait_for_function(
            "() => document.querySelector('#status')?.dataset.term === 'territory'"
        )

        boundaries = page.evaluate(
            """() => {
                const before = window.__panelBoundaryNodes;
                const features = Array.from(
                    document.querySelectorAll('#featureRows [data-term]'),
                    row => row.dataset.term,
                );
                return {
                    chartChanged: before.editedPath !== document.querySelector('#chart path.edited'),
                    metricStable: before.metric === document.querySelector('#metricGrid > *'),
                    summaryStable: before.summary === document.querySelector('#summaryFrame > *'),
                    historyStable: before.history === document.querySelector('#historyFrame > *'),
                    // The list moves its current row, so it re-renders its own
                    // rows, but it names the same features in the same place.
                    featuresStable: before.featureRows === document.querySelector('#featureRows')
                        && before.features.join('|') === features.join('|'),
                    current: document.querySelector('#featureRows [aria-current="true"]')
                        ?.dataset.term,
                };
            }"""
        )
        assert boundaries == {
            "chartChanged": True,
            "metricStable": True,
            "summaryStable": True,
            "historyStable": True,
            "featuresStable": True,
            "current": "territory",
        }


def test_contribution_frames_update_only_the_chart_scene(open_editor_page):
    with open_editor_page() as (page, _session):
        page.locator("#metricGrid > *").first.wait_for()
        page.locator("#summaryFrame > *").first.wait_for()
        select_chart_tool(page, "Handles")
        play = page.locator("#contribPlay")
        play.wait_for(state="visible")
        page.evaluate(
            """() => {
                window.__contributionBoundaryNodes = {
                    chart: document.querySelector('#chart path.edited'),
                    tool: document.querySelector('#toolRail [data-tool]'),
                    statusText: document.querySelector('#status')?.firstChild,
                    handleCountText: document.querySelector('#handleCountValue')?.firstChild,
                    metric: document.querySelector('#metricGrid > *'),
                    summary: document.querySelector('#summaryFrame > *'),
                    history: document.querySelector('#historyFrame > *'),
                    termOptions: Array.from(document.querySelectorAll('#featureRows [data-term]')),
                };
            }"""
        )

        play.click()
        page.locator("#chart .basis-build").wait_for()
        page.evaluate(
            "() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))"
        )
        boundaries = page.evaluate(
            """() => {
                const before = window.__contributionBoundaryNodes;
                const options = Array.from(document.querySelectorAll('#featureRows [data-term]'));
                return {
                    chartChanged: !before.chart.isConnected,
                    toolStable: before.tool === document.querySelector('#toolRail [data-tool]'),
                    statusStable: before.statusText === document.querySelector('#status')?.firstChild,
                    handleCountStable: before.handleCountText
                        === document.querySelector('#handleCountValue')?.firstChild,
                    metricStable: before.metric === document.querySelector('#metricGrid > *'),
                    summaryStable: before.summary === document.querySelector('#summaryFrame > *'),
                    historyStable: before.history === document.querySelector('#historyFrame > *'),
                    optionsStable: before.termOptions.length === options.length
                        && before.termOptions.every((node, index) => node === options[index]),
                    playDisabled: document.querySelector('#contribPlay').disabled,
                };
            }"""
        )
        assert boundaries == {
            "chartChanged": True,
            "toolStable": True,
            "statusStable": True,
            "handleCountStable": True,
            "metricStable": True,
            "summaryStable": True,
            "historyStable": True,
            "optionsStable": True,
            "playDisabled": True,
        }
        select_chart_tool(page, "Select")


def test_summary_updating_status_preserves_confirmed_table_nodes(open_editor_page):
    with open_editor_page() as (page, _session):
        summary_child = page.locator("#summaryFrame > *").first
        summary_child.wait_for()
        summary_child.evaluate(
            """node => {
                window.__confirmedSummaryChild = node;
                window.__summaryBoundaryNodes = {
                    chart: document.querySelector('#chart path.edited'),
                    metric: document.querySelector('#metricGrid > *'),
                    history: document.querySelector('#historyFrame > *'),
                    termOptions: Array.from(document.querySelectorAll('#featureRows [data-term]')),
                };
            }"""
        )
        held_summary = []
        page.route("**/summary", lambda route: held_summary.append(route))

        page.locator("#summarySource").evaluate(
            """node => {
                node.value = 'in_force';
                node.dispatchEvent(new Event('change', { bubbles: true }));
            }"""
        )
        page.wait_for_timeout(50)
        assert len(held_summary) == 1
        page.wait_for_function(
            "() => document.querySelector('#summaryFrame')?.getAttribute('aria-busy') === 'true'"
        )
        assert page.evaluate(
            "window.__confirmedSummaryChild === document.querySelector('#summaryFrame > *')"
        )

        with page.expect_response(
            lambda response: response.request.method == "POST" and response.url.endswith("/summary")
        ):
            backend_response = held_summary[0].fetch()
            payload = backend_response.json()
            payload["available"] = False
            payload["label"] = "Replacement summary"
            payload["error"] = "Replacement payload rendered locally."
            payload["compact"] = None
            payload["html"] = ""
            held_summary[0].fulfill(response=backend_response, json=payload)

        page.wait_for_function(
            "() => document.querySelector('#summaryFrame')?.textContent.includes('Replacement')"
        )
        isolation = page.evaluate(
            """() => {
                const before = window.__summaryBoundaryNodes;
                const options = Array.from(document.querySelectorAll('#featureRows [data-term]'));
                return {
                    summaryChanged: window.__confirmedSummaryChild
                        !== document.querySelector('#summaryFrame > *'),
                    chartStable: before.chart === document.querySelector('#chart path.edited'),
                    metricStable: before.metric === document.querySelector('#metricGrid > *'),
                    historyStable: before.history === document.querySelector('#historyFrame > *'),
                    optionsStable: before.termOptions.length === options.length
                        && before.termOptions.every((node, index) => node === options[index]),
                };
            }"""
        )
        assert isolation == {
            "summaryChanged": True,
            "chartStable": True,
            "metricStable": True,
            "historyStable": True,
            "optionsStable": True,
        }


def test_busy_state_makes_all_editor_regions_inert_and_cleans_up(open_editor_page):
    with open_editor_page() as (page, _session):
        reference_ci = page.locator("#ciToggle")
        reference_ci.focus()
        assert reference_ci.evaluate("node => document.activeElement === node")

        page.evaluate("window.__superglmTest.setAppBusy(true, 'Testing busy state', 'Waiting')")

        for selector in ["#appBar", ".context-bar", "#editorView", "#reportPanel"]:
            assert page.locator(selector).get_attribute("inert") == ""
        assert page.locator(".app-shell").get_attribute("aria-busy") == "true"
        overlay = page.locator("#appBusyOverlay")
        assert overlay.get_attribute("aria-live") is None
        assert overlay.is_visible()
        announcement = page.locator("#appBusyAnnouncement")
        assert announcement.get_attribute("role") == "status"
        assert announcement.get_attribute("aria-live") == "polite"
        assert announcement.get_attribute("tabindex") == "-1"
        assert announcement.evaluate("node => document.activeElement === node")
        assert page.locator("#appBusyTitle").inner_text() == "Testing busy state"
        assert page.locator("#appBusyMessage").inner_text() == "Waiting"
        detail = page.locator("#appBusyDetail")
        assert detail.get_attribute("aria-hidden") == "true"

        page.evaluate(
            "window.__superglmTest.setAppBusy(true, 'Ignored repeat title', 'Ignored repeat detail')"
        )
        assert page.locator("#appBusyTitle").inner_text() == "Testing busy state"
        assert page.locator("#appBusyMessage").inner_text() == "Waiting"
        first_elapsed = detail.inner_text()

        page.evaluate(
            """() => {
                const target = document.querySelector('#appBusyAnnouncement');
                window.__busyLiveMutationCount = 0;
                window.__busyLiveObserver = new MutationObserver(records => {
                    window.__busyLiveMutationCount += records.length;
                });
                window.__busyLiveObserver.observe(target, {
                    subtree: true,
                    childList: true,
                    characterData: true,
                });
            }"""
        )
        page.wait_for_timeout(650)
        observed = page.evaluate(
            """() => {
                window.__busyLiveObserver.disconnect();
                return {
                    mutations: window.__busyLiveMutationCount,
                    elapsed: document.querySelector('#appBusyDetail').textContent,
                };
            }"""
        )
        assert observed["mutations"] == 0
        assert observed["elapsed"] != first_elapsed
        assert page.locator("#appAlert").get_attribute("role") == "alert"
        assert page.locator("#appAlert").get_attribute("aria-live") == "assertive"

        page.evaluate("window.__superglmTest.setAppBusy(false)")

        for selector in ["#appBar", ".context-bar", "#editorView", "#reportPanel"]:
            assert page.locator(selector).get_attribute("inert") is None
        assert page.locator(".app-shell").get_attribute("aria-busy") == "false"
        assert overlay.is_hidden()
        assert reference_ci.evaluate("node => document.activeElement === node")

        page.evaluate("window.__superglmTest.setAppBusy(true, 'Testing fallback', 'Waiting')")
        assert announcement.evaluate("node => document.activeElement === node")
        reference_ci.evaluate("node => { node.hidden = true; }")
        page.evaluate("window.__superglmTest.setAppBusy(false)")

        assert page.evaluate("document.activeElement?.id") in {"featureSearch", "inspectorToggle"}
        for selector in ["#appBar", ".context-bar", "#editorView", "#reportPanel"]:
            assert page.locator(selector).get_attribute("inert") is None


def test_text_selection_is_scoped_and_reduced_motion_stops_spinner(open_editor_page):
    with open_editor_page() as (page, _session):
        page.get_by_role("button", name="Help", exact=True).click()
        help_panel = page.get_by_role("tabpanel", name="Help")
        assert help_panel.is_visible()

        for selector in ["#helpPane", "#status", "#reportFrame", "#appAlert"]:
            user_select = page.locator(selector).evaluate(
                "node => getComputedStyle(node).userSelect"
            )
            assert user_select in {"auto", "text"}
        for selector in ["#chart", "#toolRail", "#selectionMenu"]:
            assert (
                page.locator(selector).evaluate("node => getComputedStyle(node).userSelect")
                == "none"
            )

        page.emulate_media(reduced_motion="reduce")
        motion = page.locator(".busy-spinner").evaluate(
            """node => {
                const style = getComputedStyle(node);
                const value = style.animationDuration.split(',')[0].trim();
                const seconds = value.endsWith('ms')
                    ? Number.parseFloat(value) / 1000
                    : Number.parseFloat(value);
                return { name: style.animationName, seconds };
            }"""
        )
        assert motion["name"] == "none" or motion["seconds"] <= 0.000001


def test_application_bar_exposes_views_undo_redo_and_export(open_editor_page):
    with open_editor_page() as (page, _session):
        tabs = page.get_by_role("tablist", name="Editor views")
        assert tabs.get_by_role("tab").all_inner_texts() == [
            "Editor",
            "Validation",
            "Cross-validation",
            "Final Fit",
        ]
        assert page.get_by_role("button", name="Undo edit").is_disabled()
        assert page.get_by_role("button", name="Redo edit").is_disabled()
        assert page.get_by_role("button", name="Export model or workbook").is_visible()

        select_chart_tool(page, "Select")
        page.locator('button[data-op="select_all"]').click()
        page.locator("#selectionMenu").wait_for(state="visible")
        page.get_by_role("button", name="Increase selection").click()
        undo = page.get_by_role("button", name="Undo edit")
        page.wait_for_function("() => !document.querySelector('#undoAction').disabled")
        undo.click()
        redo = page.get_by_role("button", name="Redo edit")
        page.wait_for_function("() => !document.querySelector('#redoAction').disabled")
        redo.click()
        page.wait_for_function("() => !document.querySelector('#undoAction').disabled")


def test_application_bar_undo_and_redo_follow_one_timeline_across_terms(
    open_editor_page, choose_feature
):
    with open_editor_page(selected_term="curve") as (page, session):
        select_chart_tool(page, "Select")
        page.locator('button[data-op="select_all"]').click()
        page.locator("#selectionMenu").wait_for(state="visible")
        page.get_by_role("button", name="Increase selection").click()
        page.wait_for_function("() => !document.querySelector('#undoAction').disabled")

        with page.expect_response(
            lambda response: (
                response.request.method == "POST"
                and response.url.split("?", maxsplit=1)[0].endswith("/term")
            )
        ):
            choose_feature(page, "territory")
        # Another term is shown, and Undo still takes the curve's edit, by name.
        undo = page.get_by_role("button", name="Undo edit")
        assert undo.is_enabled()
        assert undo.get_attribute("data-popover-body") == "Undo: shift curve"

        undo.click()
        page.wait_for_function("() => !document.querySelector('#redoAction').disabled")
        redo = page.get_by_role("button", name="Redo edit")
        assert redo.get_attribute("data-popover-body") == "Redo: shift curve"
        assert undo.is_disabled() and session.history == []


def test_analyst_can_discover_edit_undo_redo_help_and_export(open_editor_page):
    with open_editor_page() as (page, _session):
        select_chart_tool(page, "Select")
        with page.expect_response(
            lambda response: (
                response.request.method == "POST"
                and response.url.split("?", maxsplit=1)[0].endswith("/select")
            )
        ) as selection_response:
            page.locator("#chart .point").nth(2).click()
        assert selection_response.value.status == 200
        page.locator("#selectionMenu").wait_for(state="visible")

        increase = page.get_by_role("button", name="Increase selection", exact=True)
        increase.focus()
        tooltip = page.get_by_role("tooltip")
        tooltip.wait_for(state="visible")
        assert tooltip.locator("[data-popover-heading]").inner_text() == "Increase selection"
        assert tooltip.locator("[data-popover-description]").inner_text() == (
            "Increase selected relativities by 5%."
        )

        with page.expect_response(
            lambda response: (
                response.request.method == "POST"
                and response.url.split("?", maxsplit=1)[0].endswith("/op")
            )
        ) as edit_response:
            increase.click()
        assert edit_response.value.status == 200
        assert edit_response.value.request.post_data_json == {"operation": "shift_up"}

        undo = page.get_by_role("button", name="Undo edit")
        page.wait_for_function("() => !document.querySelector('#undoAction').disabled")
        assert undo.is_enabled()
        with page.expect_response(
            lambda response: (
                response.request.method == "POST"
                and response.url.split("?", maxsplit=1)[0].endswith("/op")
            )
        ) as undo_response:
            undo.click()
        assert undo_response.value.status == 200
        assert undo_response.value.request.post_data_json == {"operation": "undo"}

        redo = page.get_by_role("button", name="Redo edit")
        page.wait_for_function("() => !document.querySelector('#redoAction').disabled")
        assert redo.is_enabled()
        with page.expect_response(
            lambda response: (
                response.request.method == "POST"
                and response.url.split("?", maxsplit=1)[0].endswith("/op")
            )
        ) as redo_response:
            redo.click()
        assert redo_response.value.status == 200
        assert redo_response.value.request.post_data_json == {"operation": "redo"}
        page.wait_for_function(
            """() => !document.querySelector('#undoAction').disabled
                && document.querySelector('#redoAction').disabled"""
        )
        assert undo.is_enabled()
        assert redo.is_disabled()

        page.get_by_role("button", name="Help", exact=True).click()
        assert page.get_by_role("tabpanel", name="Help").is_visible()

        export = page.get_by_role("button", name="Export model or workbook")
        assert export.is_visible()
        export.click()
        page.get_by_role("dialog", name="Export").wait_for(state="visible")


def test_export_dialog_downloads_both_formats_without_redrawing_the_app(open_editor_page):
    with open_editor_page(selected_term="territory") as (page, _session):
        page.locator("#summaryFrame > *").first.wait_for()
        page.evaluate(
            """() => {
                window.__exportBoundaryNodes = {
                    chart: document.querySelector('#chart'),
                    summary: document.querySelector('#summaryFrame'),
                    report: document.querySelector('#reportFrame'),
                    features: document.querySelector('#featureRows [data-term]'),
                };
            }"""
        )

        page.get_by_role("button", name="Export model or workbook").click()
        dialog = page.get_by_role("dialog", name="Export")
        dialog.wait_for(state="visible")

        dialog.get_by_role("radio", name="Python model").check()
        with page.expect_response(
            lambda response: (
                response.request.method == "GET"
                and response.url.split("?", maxsplit=1)[0].endswith("/download_export")
                and "format=joblib" in response.url
            )
        ) as model_response:
            dialog.get_by_role("button", name="Download", exact=True).click()
        assert model_response.value.status == 200
        assert model_response.value.headers["x-superglm-validation"].startswith("artifact")

        dialog.get_by_role("radio", name="Excel rating workbook").check()
        with page.expect_response(
            lambda response: (
                response.request.method == "GET"
                and response.url.split("?", maxsplit=1)[0].endswith("/download_export")
                and "format=xlsx" in response.url
            )
        ) as excel_response:
            dialog.get_by_role("button", name="Download", exact=True).click()
        assert excel_response.value.status == 200
        assert excel_response.value.headers["content-type"].startswith(
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        assert page.evaluate(
            """() => {
                const before = window.__exportBoundaryNodes;
                return before.chart === document.querySelector('#chart')
                    && before.summary === document.querySelector('#summaryFrame')
                    && before.report === document.querySelector('#reportFrame')
                    && before.features === document.querySelector('#featureRows [data-term]');
            }"""
        )


def test_summary_level_display_toggle_is_view_only_and_synchronizes_full_summary(
    open_editor_page,
):
    collapsed_levels = ("T02", "T03")
    with open_editor_page(
        selected_term="territory",
        collapsed_levels=("territory", collapsed_levels),
    ) as (page, session):
        page.wait_for_function(
            """() => document.querySelector('#summaryFrame')?.getAttribute('aria-busy') === 'false'
                && document.querySelector('#summaryFrame')?.textContent.includes('territory[T02]')
                && document.querySelector('#summaryFrame')?.textContent.includes('territory[T03]')"""
        )
        initial_revision = session.model_revision
        initial_history = (
            len(session.history),
            len(session.redo_stack),
            len(session.structure_history),
        )
        initial_chart_group_mode = page.locator("#groupDisplayMode").input_value()
        initial_chart_is_collapsed = page.evaluate(
            "Boolean(document.querySelector('#chart')?._scale?.displayIsCollapsed)"
        )
        requests = []

        def record_request(request) -> None:
            if request.method != "POST":
                return
            path = request.url.split("?", maxsplit=1)[0].rsplit("/", maxsplit=1)[-1]
            requests.append((path, request.post_data_json))

        page.on("request", record_request)
        control = page.get_by_role("group", name="Categorical levels")
        expanded = control.get_by_role("radio", name="Expanded")
        grouped = control.get_by_role("radio", name="Grouped")

        assert control.is_visible()
        assert expanded.is_checked()
        assert grouped.is_checked() is False

        grouped.focus()
        assert grouped.evaluate("node => document.activeElement === node")
        with page.expect_response(
            lambda response: (
                response.request.method == "POST"
                and response.url.split("?", maxsplit=1)[0].endswith("/summary")
                and response.request.post_data_json.get("level_display") == "grouped"
            )
        ) as grouped_response:
            grouped.press("Space")

        assert grouped_response.value.status == 200
        assert grouped.is_checked()
        page.wait_for_function(
            """() => document.querySelector('#summaryFrame')?.getAttribute('aria-busy') === 'false'
                && document.querySelector('#summaryFrame .summary-level-groups')
                && !document.querySelector('#summaryFrame')?.textContent.includes('territory[T02]')"""
        )
        compact_grouped = page.locator("#summaryFrame").inner_text()
        assert "Level group" in compact_grouped
        assert "G1" in compact_grouped
        assert "Level groups (territory):" in compact_grouped
        assert "T02, T03" in compact_grouped

        page.locator("details.raw-summary > summary").click()
        grouped_full = page.frame_locator(".raw-summary-frame").locator("body")
        grouped_full.wait_for()
        grouped_full_text = grouped_full.inner_text()
        assert "G1" in grouped_full_text
        assert "Level groups (territory): G1 = T02, T03" in grouped_full_text

        expanded.focus()
        assert expanded.evaluate("node => document.activeElement === node")
        with page.expect_response(
            lambda response: (
                response.request.method == "POST"
                and response.url.split("?", maxsplit=1)[0].endswith("/summary")
                and response.request.post_data_json.get("level_display") == "expanded"
            )
        ) as expanded_response:
            expanded.press("Space")

        assert expanded_response.value.status == 200
        assert expanded.is_checked()
        page.wait_for_function(
            """() => document.querySelector('#summaryFrame')?.getAttribute('aria-busy') === 'false'
                && document.querySelector('#summaryFrame')?.textContent.includes('territory[T02]')
                && document.querySelector('#summaryFrame')?.textContent.includes('territory[T03]')"""
        )
        assert "Level groups (territory):" not in page.locator("#summaryFrame").inner_text()

        page.locator("details.raw-summary > summary").click()
        expanded_full = page.frame_locator(".raw-summary-frame").locator("body")
        expanded_full.wait_for()
        expanded_full_text = expanded_full.inner_text()
        assert "territory[T02]" in expanded_full_text
        assert "territory[T03]" in expanded_full_text
        assert "Level groups (territory):" not in expanded_full_text

        assert session.model_revision == initial_revision
        assert (
            len(session.history),
            len(session.redo_stack),
            len(session.structure_history),
        ) == initial_history
        assert page.locator("#groupDisplayMode").input_value() == initial_chart_group_mode
        assert (
            page.evaluate("Boolean(document.querySelector('#chart')?._scale?.displayIsCollapsed)")
            is initial_chart_is_collapsed
        )

        forbidden = {"collapse_levels", "ungroup_levels", "stage", "refit_pending", "refit_offset"}
        assert not any(path in forbidden for path, _payload in requests)
        summary_payloads = [payload for path, payload in requests if path == "summary"]
        assert [payload["level_display"] for payload in summary_payloads] == [
            "grouped",
            "expanded",
        ]


def test_summary_level_display_toggle_cancels_queued_prior_mode_refresh(open_editor_page):
    with open_editor_page(
        selected_term="territory",
        collapsed_levels=("territory", ("T02", "T03")),
    ) as (page, _session):
        page.wait_for_function(
            """() => document.querySelector('#summaryFrame')?.getAttribute('aria-busy') === 'false'
                && document.querySelector('#summaryFrame')?.textContent.includes('territory[T02]')"""
        )
        summary_payloads = []

        def record_summary_request(request) -> None:
            if request.method == "POST" and request.url.split("?", maxsplit=1)[0].endswith(
                "/summary"
            ):
                summary_payloads.append(request.post_data_json)

        page.on("request", record_summary_request)
        page.evaluate(
            """() => {
                const frame = document.querySelector('#summaryFrame');
                const initialRevision = frame.dataset.modelRevision;
                const observer = new MutationObserver(() => {
                    if (frame.dataset.modelRevision === initialRevision) return;
                    observer.disconnect();
                    document.querySelector(
                        'input[name="summaryLevelDisplay"][value="grouped"]'
                    ).click();
                });
                observer.observe(frame, {
                    attributes: true,
                    attributeFilter: ['data-model-revision'],
                });
            }"""
        )

        select_chart_tool(page, "Select")
        page.locator('button[data-op="select_all"]').click()
        page.locator("#selectionMenu").wait_for(state="visible")
        with page.expect_response(
            lambda response: (
                response.request.method == "POST"
                and response.url.split("?", maxsplit=1)[0].endswith("/op")
            )
        ) as operation_response:
            page.get_by_role("button", name="Increase selection").click()

        assert operation_response.value.status == 200
        grouped = page.get_by_role("group", name="Categorical levels").get_by_role(
            "radio", name="Grouped"
        )
        page.wait_for_function(
            """() => document.querySelector(
                'input[name="summaryLevelDisplay"][value="grouped"]'
            )?.checked"""
        )
        page.wait_for_timeout(350)

        assert [payload["level_display"] for payload in summary_payloads] == ["grouped"]
        assert grouped.is_checked()
        page.wait_for_function(
            """() => document.querySelector('#summaryFrame')?.getAttribute('aria-busy') === 'false'
                && document.querySelector('#summaryFrame .summary-level-groups')"""
        )


def test_failed_summary_level_display_refresh_remains_retryable(open_editor_page):
    with open_editor_page(
        selected_term="territory",
        collapsed_levels=("territory", ("T02", "T03")),
    ) as (page, _session):
        page.wait_for_function(
            """() => document.querySelector('#summaryFrame')?.getAttribute('aria-busy') === 'false'
                && document.querySelector('#summaryFrame')?.textContent.includes('territory[T02]')"""
        )

        def fail_grouped_summary(route) -> None:
            payload = route.request.post_data_json
            if payload.get("level_display") == "grouped":
                route.fulfill(status=503, json={"error": "Grouped summary unavailable."})
            else:
                route.continue_()

        page.route("**/summary", fail_grouped_summary)
        grouped = page.get_by_role("group", name="Categorical levels").get_by_role(
            "radio", name="Grouped"
        )
        with page.expect_response(
            lambda response: (
                response.request.method == "POST"
                and response.url.split("?", maxsplit=1)[0].endswith("/summary")
                and response.request.post_data_json.get("level_display") == "grouped"
            )
        ) as grouped_response:
            grouped.check()

        assert grouped_response.value.status == 503
        # The page handles the response on its own schedule; a fixed 50 ms was
        # not always enough on a loaded runner.
        page.wait_for_function(
            "() => document.querySelector('#summaryFrame')?.dataset.freshness === 'stale'"
        )
        assert page.locator("#summaryFrame").get_attribute("aria-busy") == "false"
        assert page.locator("#summaryRetry").is_visible()
        assert "Grouped summary unavailable." in page.locator("#summaryStatus").inner_text()


def test_ordered_summary_has_one_whole_smooth_test_and_no_level_tests(open_editor_page):
    with open_editor_page(selected_term="age_band") as (page, _session):
        page.wait_for_function(
            """() => document.querySelector('#summaryFrame')?.getAttribute('aria-busy') === 'false'
                && document.querySelectorAll('#summaryFrame .summary-row').length > 0"""
        )
        compact_rows = page.locator("#summaryFrame .summary-row").evaluate_all(
            """rows => rows.map(row => ({
                term: row.querySelector('.summary-term span')?.textContent.trim() || '',
                cells: Array.from(row.cells, cell => cell.textContent.trim()),
            }))"""
        )
        smooth_rows = [row for row in compact_rows if row["term"] == "age_band"]
        assert len(smooth_rows) == 1
        assert smooth_rows[0]["cells"][4] not in {"", "--"}

        level_rows = [row for row in compact_rows if row["term"].startswith("age_band[")]
        assert level_rows
        for row in level_rows:
            assert row["cells"][2] not in {"", "--"}
            assert row["cells"][3] not in {"", "--"}
            assert row["cells"][4] == "--"
            assert row["cells"][5] == ""

        page.locator("details.raw-summary > summary").click()
        full_summary = page.frame_locator(".raw-summary-frame")
        full_summary.locator("body").wait_for()
        full_rows = full_summary.locator("tr").evaluate_all(
            "rows => rows.map(row => Array.from(row.cells, cell => cell.textContent.trim()))"
        )
        full_smooth_rows = [
            cells
            for cells in full_rows
            if cells
            and cells[0] == "age_band"
            and any("ordered spline" in cell and "p=" in cell for cell in cells)
        ]
        assert len(full_smooth_rows) == 1
        assert any("p=" in cell for cell in full_smooth_rows[0])

        for row in level_rows:
            full_row = next(cells for cells in full_rows if cells and cells[0] == row["term"])
            assert full_row[1] != "---"  # estimate
            if row["cells"][3] == "ref":
                assert full_row[2] == "ref"
                assert full_row[3:7] == ["---"] * 4
            else:
                assert full_row[2] != "---"  # standard error
                assert full_row[4] == "---"  # no separate level p-value
                assert full_row[5] != "---"  # confidence interval lower bound
                assert full_row[6] != "---"  # confidence interval upper bound
            assert full_row[7] == ""  # no significance marker


def test_raw_summary_html_is_isolated_in_a_sandboxed_iframe(open_editor_page):
    malicious_html = (
        "<div id='sandbox-payload'>Injected summary payload</div>"
        "<script>parent.document.documentElement.dataset.summarySandboxEscape='executed'</script>"
    )
    payload = {
        "available": True,
        "label": "Security summary",
        "note": "",
        "html": malicious_html,
        "compact": {
            "model": {"family": "Gaussian", "link": "Identity", "method": "PIRLS"},
            "rows": [],
        },
    }

    with open_editor_page() as (page, _session):
        page.wait_for_function(
            """() => {
                const frame = document.querySelector('#summaryFrame');
                return frame?.getAttribute('aria-busy') === 'false'
                    && frame.textContent.trim().length > 0;
            }"""
        )

        def fulfill_summary(route) -> None:
            request_payload = route.request.post_data_json
            route.fulfill(
                status=200,
                json={
                    **payload,
                    "model_revision": request_payload["model_revision"],
                    "request_sequence": request_payload["request_sequence"],
                },
            )

        page.route("**/summary", fulfill_summary)
        page.evaluate("delete document.documentElement.dataset.summarySandboxEscape")

        with page.expect_response(
            lambda response: response.url.split("?", maxsplit=1)[0].endswith("/summary")
        ) as summary_response:
            page.locator("#summarySource").evaluate(
                """node => {
                    node.value = 'selected';
                    node.dispatchEvent(new Event('change', { bubbles: true }));
                }"""
            )
        assert summary_response.value.status == 200
        page.wait_for_function(
            """() => document.querySelector('.raw-summary-frame')
                ?.getAttribute('srcdoc')?.includes('summarySandboxEscape')"""
        )

        frame = page.locator(".raw-summary-frame")
        assert frame.get_attribute("sandbox") == ""
        assert frame.get_attribute("referrerpolicy") == "no-referrer"
        assert malicious_html in frame.get_attribute("srcdoc")
        page.wait_for_timeout(250)
        assert page.evaluate("document.documentElement.dataset.summarySandboxEscape") is None


def test_summary_search_filters_rows_survives_a_new_payload_and_escape_clears_it(
    open_editor_page,
):
    with open_editor_page(selected_term="territory") as (page, _session):
        page.wait_for_function(
            """() => document.querySelector('#summaryFrame')?.getAttribute('aria-busy') === 'false'
                && document.querySelectorAll('#summaryFrame .summary-row').length > 0"""
        )
        search = page.get_by_role("searchbox", name="Search terms and levels")

        def shown_rows() -> list[str]:
            return page.locator("#summaryFrame tr.summary-row:not([hidden])").evaluate_all(
                "rows => rows.map(row => row.querySelector('.summary-term span').textContent)"
            )

        # T01..T10 are text: "t1" is in T10 only, and in no term's name.
        search.fill("t1")
        assert shown_rows() == ["territory[T10]"]
        marks = page.locator("#summaryFrame tr.summary-row:not([hidden]) mark")
        assert marks.all_inner_texts() == ["T1"]
        assert page.locator("#summarySearchCount").inner_text() == "1 term · 1 row"

        # A new payload rebuilds the frame, and the search is applied to it.
        grouped = page.get_by_role("group", name="Categorical levels").get_by_role(
            "radio", name="Grouped"
        )
        with page.expect_response(
            lambda response: (
                response.request.method == "POST"
                and response.url.split("?", maxsplit=1)[0].endswith("/summary")
            )
        ):
            grouped.check()
        page.wait_for_function(
            "() => document.querySelector('#summaryFrame')?.getAttribute('aria-busy') === 'false'"
        )
        assert shown_rows() == ["territory[T10]"]

        search.press("Escape")
        assert search.input_value() == ""
        assert page.locator("#summarySearchCount").inner_text() == ""
        assert len([row for row in shown_rows() if row.startswith("territory[")]) == 10
        assert page.locator("#inspector").get_attribute("data-open") == "true"


def test_summary_follows_the_chart_and_filters_edited_and_waiting_terms(
    open_editor_page, choose_feature
):
    with open_editor_page(selected_term="territory") as (page, _session):
        page.wait_for_function(
            """() => document.querySelector('#summaryFrame')?.getAttribute('aria-busy') === 'false'
                && document.querySelector('#summaryFrame tr.summary-section')"""
        )

        def lines() -> list[str]:
            return page.locator("#summaryFrame tr.summary-section:not([hidden])").evaluate_all(
                "rows => rows.map(row => row.dataset.term)"
            )

        def open_lines() -> list[str]:
            return page.locator(
                '#summaryFrame tr.summary-section:not([hidden]) [aria-expanded="true"]'
            ).evaluate_all("buttons => buttons.map(button => button.dataset.summarySection)")

        # The chart's term is open; every other term is one line.
        assert lines() == ["curve", "territory", "age_band", "long_category"]
        assert open_lines() == ["territory"]
        current = page.locator('#summaryFrame tr.summary-section[data-current="true"]')
        assert current.get_attribute("data-term") == "territory"
        # A spline, ordered or not, carries its whole-term test's p on its line; a
        # categorical has no whole-term test, so its line carries no p chip.
        chips = page.locator("#summaryFrame tr.summary-section").evaluate_all(
            "rows => rows.map(row => [row.dataset.term, row.querySelectorAll('.summary-p-chip').length])"
        )
        assert chips == [["curve", 1], ["territory", 0], ["age_band", 1], ["long_category", 0]]

        # Another term opens from its line and stays open while the chart stays put.
        page.locator('#summaryFrame [data-summary-section="age_band"]').click()
        assert open_lines() == ["territory", "age_band"]

        # The chart moves on: the summary follows it, folds the rest again, and
        # rewrites only its table body.
        summary_child = page.locator("#summaryFrame > *").first.element_handle()
        choose_feature(page, "curve")
        page.wait_for_function("() => document.querySelector('#status')?.dataset.term === 'curve'")
        assert open_lines() == ["curve"]
        assert summary_child.evaluate(
            "node => node === document.querySelector('#summaryFrame > *')"
        )

        # A hand edit on curve: Edited keeps it alone, and nothing is waiting.
        select_chart_tool(page, "Select")
        _click_selects(page, page.locator('#chart circle.point[data-index="5"]'))
        with page.expect_response(
            lambda response: (
                response.request.method == "POST"
                and response.url.split("?", maxsplit=1)[0].endswith("/op")
            )
        ):
            page.locator('button[data-op="shift_up"]').click()
        edited = page.get_by_role("button", name="Edited", exact=True)
        edited.click()
        page.wait_for_function(
            """() => [...document.querySelectorAll('#summaryFrame tr.summary-section:not([hidden])')]
                .map(row => row.dataset.term).join() === 'curve'"""
        )
        assert edited.get_attribute("aria-pressed") == "true"
        page.get_by_role("button", name="Waiting", exact=True).click()
        assert lines() == []
        assert page.locator("#summaryFrame .summary-empty-row").inner_text() == "No terms match."
        page.get_by_role("button", name="All", exact=True).click()
        assert lines() == ["curve", "territory", "age_band", "long_category"]


def test_summary_scrolls_the_chart_term_into_view(open_editor_page, choose_feature):
    # A short window leaves the summary frame room for a few lines only.
    with open_editor_page(viewport={"width": 1180, "height": 560}) as (page, _session):
        page.wait_for_function(
            """() => document.querySelector('#summaryFrame')?.getAttribute('aria-busy') === 'false'
                && document.querySelector('#summaryFrame tr.summary-section')"""
        )

        # The frame scrolls its padding box, clipped by the window. The
        # scroll offset is a whole pixel and the layout is not, so an edge it
        # aligns may sit up to a pixel past.
        in_view = """term => {
                    const frame = document.querySelector('#summaryFrame');
                    const box = frame.getBoundingClientRect();
                    const top = Math.max(box.top + frame.clientTop, 0);
                    const bottom = Math.min(
                        box.top + frame.clientTop + frame.clientHeight, window.innerHeight
                    );
                    const line = frame
                        .querySelector(`tr.summary-section[data-term="${term}"]`)
                        .getBoundingClientRect();
                    return line.top >= top - 1 && line.bottom <= bottom + 1;
                }"""

        def line_in_view(term: str) -> bool:
            return page.evaluate(in_view, term)

        page.locator("#summaryFrame").evaluate("node => { node.scrollTop = 0; }")
        assert not line_in_view("long_category")

        choose_feature(page, "long_category")
        page.wait_for_function(
            """() => document.querySelector(
                '#summaryFrame tr.summary-section[data-term="long_category"]'
            )?.dataset.current === 'true'"""
        )
        # The scroll lands after the line is marked current; on a loaded
        # machine that can be a frame later, so wait for it.
        page.wait_for_function(in_view, arg="long_category", timeout=5000)


def test_context_bar_reports_term_kind_and_edf(open_editor_page):
    with open_editor_page(selected_term="curve") as (page, _session):
        context = page.get_by_role("region", name="Term context")
        current = page.get_by_role("navigation", name="Features").locator('[aria-current="true"]')
        assert current.get_attribute("data-term") == "curve"
        assert "spline" in context.locator("#termKind").inner_text().lower()
        assert "EDF" in context.locator("#termEdf").inner_text()
        inspector_toggle = context.get_by_role("button", name="Inspector")
        controlled_id = inspector_toggle.get_attribute("aria-controls")
        assert controlled_id
        assert page.locator(f"#{controlled_id}").count() == 1


def test_tool_rail_selects_one_mode_and_supports_roving_shortcuts(open_editor_page):
    with open_editor_page() as (page, _session):
        rail = page.get_by_role("radiogroup", name="Chart tools")
        select = rail.get_by_role("radio", name="Select", exact=True)
        move = rail.get_by_role("radio", name="Move", exact=True)
        zoom = rail.get_by_role("radio", name="Zoom", exact=True)
        handles = rail.get_by_role("radio", name="Handles", exact=True)

        assert select.get_attribute("aria-checked") == "true"
        assert select.get_attribute("tabindex") == "0"
        assert move.get_attribute("aria-checked") == "false"

        move.click()
        assert move.get_attribute("aria-checked") == "true"
        assert move.get_attribute("tabindex") == "0"
        assert select.get_attribute("aria-checked") == "false"
        assert select.get_attribute("tabindex") == "-1"

        move.focus()
        page.keyboard.press("ArrowDown")
        assert zoom.get_attribute("aria-checked") == "true"
        assert zoom.evaluate("node => document.activeElement === node")

        page.keyboard.press("End")
        assert handles.get_attribute("aria-checked") == "true"
        assert handles.evaluate("node => document.activeElement === node")
        assert rail.get_by_role("radio", name="Breaks").count() == 0

        page.locator("#chart").focus()
        page.keyboard.press("b")
        assert handles.get_attribute("aria-checked") == "true"

        page.locator("#chart").focus()
        page.keyboard.press("v")
        assert select.get_attribute("aria-checked") == "true"
        page.keyboard.press("?")
        assert page.get_by_role("button", name="Help", exact=True).is_visible()


def test_handles_tool_is_disabled_only_when_the_term_has_no_controls(
    open_editor_page, choose_feature
):
    with open_editor_page(selected_term="territory") as (page, _session):
        handles = page.get_by_role("radiogroup", name="Chart tools").get_by_role(
            "radio", name="Handles", exact=True
        )
        assert handles.is_disabled()

        with page.expect_response(
            lambda response: (
                response.request.method == "POST"
                and response.url.split("?", maxsplit=1)[0].endswith("/term")
            )
        ):
            choose_feature(page, "curve")
        page.wait_for_function(
            "term => document.querySelector('#status')?.dataset.term === term", arg="curve"
        )
        assert handles.is_enabled()
        assert (
            page.get_by_role("radiogroup", name="Chart tools")
            .get_by_role("radio", name="Select", exact=True)
            .get_attribute("aria-checked")
            == "true"
        )


def test_same_term_control_loss_falls_back_to_select_before_drawing(open_editor_page):
    with open_editor_page() as (page, _session):
        rail = page.get_by_role("radiogroup", name="Chart tools")
        select = rail.get_by_role("radio", name="Select", exact=True)
        handles = rail.get_by_role("radio", name="Handles", exact=True)
        select_chart_tool(page, "Handles")
        assert handles.get_attribute("aria-checked") == "true"

        def remove_controls(route):
            response = route.fetch()
            payload = response.json()
            payload["terms"]["curve"]["controls"] = None
            route.fulfill(response=response, json=payload)

        page.route("**/op", remove_controls)
        with page.expect_response(
            lambda response: (
                response.request.method == "POST"
                and response.url.split("?", maxsplit=1)[0].endswith("/op")
            )
        ):
            page.locator('button[data-op="reset"]').click()

        actual = {
            "select_checked": select.get_attribute("aria-checked"),
            "select_tabindex": select.get_attribute("tabindex"),
            "handles_disabled": handles.is_disabled(),
            "handles_checked": handles.get_attribute("aria-checked"),
            "handles_tabindex": handles.get_attribute("tabindex"),
            "control_count": page.locator("#chart .control-handle").count(),
        }
        assert actual == {
            "select_checked": "true",
            "select_tabindex": "0",
            "handles_disabled": True,
            "handles_checked": "false",
            "handles_tabindex": "-1",
            "control_count": 0,
        }


def test_existing_svg_selection_operation_posts_linearise_unchanged(open_editor_page):
    with open_editor_page() as (page, session):
        select_chart_tool(page, "Select")
        with page.expect_response(
            lambda response: (
                response.request.method == "POST"
                and response.url.split("?", maxsplit=1)[0].endswith("/select")
            )
        ):
            page.locator('button[data-op="select_all"]').click()
        page.locator("#selectionMenu").wait_for(state="visible")
        straighten = page.get_by_role("button", name="Straighten selection", exact=True)

        with page.expect_response(
            lambda response: (
                response.request.method == "POST"
                and response.url.split("?", maxsplit=1)[0].endswith("/op")
            )
        ) as operation_response:
            straighten.click()

        response = operation_response.value
        assert response.status == 200
        assert response.request.post_data_json == {"operation": "linearise"}
        assert session.history[-1].operation == "linear_interpolate"


def test_inspector_uses_one_slot_for_summary_history_settings_and_help(open_editor_page):
    with open_editor_page() as (page, _session):
        inspector = page.get_by_role("complementary", name="Model inspector")

        assert inspector.count() == 1
        assert inspector.get_by_role("tab").all_inner_texts() == [
            "Summary",
            "History",
            "Settings",
            "Help",
        ]

        inspector.get_by_role("tab", name="Settings").click()
        settings = inspector.get_by_role("tabpanel", name="Settings")
        assert settings.is_visible()
        assert settings.get_by_label("Build animation duration").is_visible()
        assert page.locator("#buildDurationWrap").count() == 1

        page.get_by_role("button", name="Help", exact=True).click()
        help_panel = inspector.get_by_role("tabpanel", name="Help")
        assert help_panel.is_visible()
        assert "Straighten selection" in help_panel.inner_text()


def test_inspector_tabs_use_roving_focus_and_arrow_key_selection(open_editor_page):
    with open_editor_page() as (page, _session):
        inspector = page.get_by_role("complementary", name="Model inspector")
        summary = inspector.get_by_role("tab", name="Summary")
        history = inspector.get_by_role("tab", name="History")
        help_tab = inspector.get_by_role("tab", name="Help")

        summary.focus()
        page.keyboard.press("ArrowRight")
        assert history.get_attribute("aria-selected") == "true"
        assert history.get_attribute("tabindex") == "0"
        assert history.evaluate("node => document.activeElement === node")
        assert summary.get_attribute("tabindex") == "-1"

        page.keyboard.press("End")
        assert help_tab.get_attribute("aria-selected") == "true"
        assert help_tab.evaluate("node => document.activeElement === node")

        page.keyboard.press("Home")
        assert summary.get_attribute("aria-selected") == "true"
        page.keyboard.press("ArrowLeft")
        assert help_tab.get_attribute("aria-selected") == "true"
        assert help_tab.evaluate("node => document.activeElement === node")


def test_inspector_toggle_close_scrim_and_escape_restore_the_opener(open_editor_page):
    with open_editor_page(viewport={"width": 900, "height": 720}) as (page, _session):
        inspector = page.locator("#inspector")
        toggle = page.get_by_role("button", name="Inspector", exact=True)
        close = inspector.get_by_role("button", name="Close inspector")
        help_action = page.get_by_role("button", name="Help", exact=True)
        scrim = page.locator("#inspectorScrim")

        assert inspector.get_attribute("data-open") == "false"
        assert toggle.get_attribute("aria-expanded") == "false"

        toggle.click()
        assert inspector.get_attribute("data-open") == "true"
        close.click()
        assert inspector.get_attribute("data-open") == "false"
        assert toggle.evaluate("node => document.activeElement === node")

        toggle.click()
        scrim.click(position={"x": 1, "y": 1})
        assert inspector.get_attribute("data-open") == "false"
        assert toggle.evaluate("node => document.activeElement === node")

        help_action.click()
        assert inspector.get_attribute("data-open") == "true"
        assert not scrim.is_hidden()
        scrim.click(position={"x": 1, "y": 1})
        assert inspector.get_attribute("data-open") == "false"
        assert help_action.evaluate("node => document.activeElement === node")

        help_action.click()
        page.keyboard.press("Escape")
        assert inspector.get_attribute("data-open") == "false"
        assert help_action.evaluate("node => document.activeElement === node")


def test_keyboard_help_restores_the_control_that_owned_focus(open_editor_page):
    with open_editor_page(viewport={"width": 900, "height": 720}) as (page, _session):
        inspector = page.locator("#inspector")
        reference_ci = page.get_by_role("button", name="Reference CI")

        reference_ci.focus()
        page.keyboard.press("?")
        assert inspector.get_by_role("tabpanel", name="Help").is_visible()

        page.keyboard.press("Escape")
        assert inspector.get_attribute("data-open") == "false"
        assert reference_ci.evaluate("node => document.activeElement === node")


def boxes_overlap(first: dict, second: dict) -> bool:
    return not (
        first["x"] + first["width"] <= second["x"]
        or second["x"] + second["width"] <= first["x"]
        or first["y"] + first["height"] <= second["y"]
        or second["y"] + second["height"] <= first["y"]
    )


def test_notebook_view_keeps_chart_and_inspector_side_by_side(open_editor_page):
    with open_editor_page(viewport={"width": 1180, "height": 720}) as (page, _session):
        chart = page.locator("#chart").bounding_box()
        metrics = page.locator(".metrics-strip").bounding_box()
        inspector = page.locator("#inspector").bounding_box()

        assert chart is not None
        assert metrics is not None
        assert inspector is not None
        assert chart["width"] >= 600
        assert metrics["y"] >= chart["y"] + chart["height"]
        assert inspector["x"] > chart["x"] + chart["width"]
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        assert page.evaluate("document.documentElement.scrollHeight <= window.innerHeight")


def test_narrow_view_syncs_the_dismissible_inspector_drawer(open_editor_page):
    with open_editor_page(viewport={"width": 900, "height": 720}) as (page, _session):
        inspector = page.locator("#inspector")
        chart = page.locator("#chart").bounding_box()

        assert chart is not None
        assert inspector.get_attribute("data-open") == "false"
        assert chart["width"] >= 700
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        page.get_by_role("button", name="Help", exact=True).click()
        assert inspector.get_attribute("data-open") == "true"
        page.keyboard.press("Escape")
        assert inspector.get_attribute("data-open") == "false"

        page.set_viewport_size({"width": 1100, "height": 720})
        page.wait_for_function(
            "() => document.querySelector('#inspector')?.dataset.open === 'true'"
        )
        page.set_viewport_size({"width": 900, "height": 720})
        page.wait_for_function(
            "() => document.querySelector('#inspector')?.dataset.open === 'false'"
        )


def test_open_narrow_drawer_clears_its_scrim_when_resized_wide(open_editor_page):
    with open_editor_page(viewport={"width": 900, "height": 720}) as (page, _session):
        inspector = page.locator("#inspector")
        scrim = page.locator("#inspectorScrim")
        reference_ci = page.get_by_role("button", name="Reference CI")

        page.get_by_role("button", name="Help", exact=True).click()
        assert inspector.get_attribute("data-open") == "true"
        assert not scrim.is_hidden()

        page.set_viewport_size({"width": 1100, "height": 720})
        page.wait_for_function("() => !matchMedia('(max-width: 1085px)').matches")
        page.wait_for_function("() => document.querySelector('#inspectorScrim')?.hidden")
        assert scrim.is_hidden()
        scrim.evaluate("node => { node.hidden = false; }")
        assert scrim.evaluate("node => getComputedStyle(node).display === 'none'")
        reference_ci.click()
        assert reference_ci.get_attribute("aria-pressed") == "true"


def test_resize_to_narrow_restores_focus_from_inspector_to_toggle(open_editor_page):
    with open_editor_page(viewport={"width": 1100, "height": 720}) as (page, _session):
        inspector = page.locator("#inspector")
        summary_tab = inspector.get_by_role("tab", name="Summary")
        toggle = page.get_by_role("button", name="Inspector", exact=True)

        summary_tab.focus()
        assert summary_tab.evaluate("node => document.activeElement === node")
        page.set_viewport_size({"width": 900, "height": 720})
        page.wait_for_function(
            "() => document.querySelector('#inspector')?.dataset.open === 'false'"
        )

        assert toggle.evaluate("node => document.activeElement === node")


@pytest.mark.parametrize(
    ("width", "expected_open"),
    [(1085, "false"), (1086, "true")],
)
def test_workspace_breakpoint_preserves_chart_width_without_overflow(
    open_editor_page, width, expected_open
):
    with open_editor_page(viewport={"width": width, "height": 720}) as (page, _session):
        chart = page.locator("#chart").bounding_box()

        assert chart is not None
        assert page.locator("#inspector").get_attribute("data-open") == expected_open
        assert chart["width"] >= 600
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")


def test_short_window_scrolls_without_chart_metric_overlap(open_editor_page):
    with open_editor_page(viewport={"width": 1180, "height": 540}) as (page, _session):
        page.wait_for_function(
            """() => {
                const frame = document.querySelector('#summaryFrame');
                return frame?.getAttribute('aria-busy') === 'false'
                    && frame.textContent.trim().length > 0;
            }"""
        )
        chart = page.locator("#chart").bounding_box()
        metrics = page.locator(".metrics-strip").bounding_box()
        workspace = page.locator(".editor-workspace").bounding_box()

        assert chart is not None
        assert metrics is not None
        assert workspace is not None
        assert chart["height"] >= 360
        assert metrics["height"] < 250
        assert workspace["height"] < 650
        assert metrics["y"] >= chart["y"] + chart["height"]
        assert not boxes_overlap(chart, metrics)
        scroll_height = page.evaluate("document.documentElement.scrollHeight")
        assert scroll_height > page.evaluate("window.innerHeight")
        assert scroll_height < 900
        assert metrics["y"] + metrics["height"] <= scroll_height

        page.locator(".metrics-strip").scroll_into_view_if_needed()
        visible_metrics = page.locator(".metrics-strip").bounding_box()
        assert visible_metrics is not None
        assert visible_metrics["y"] >= 0
        assert visible_metrics["y"] + visible_metrics["height"] <= (
            page.evaluate("window.innerHeight") + 1
        )


@pytest.mark.parametrize(("width", "expected_open"), [(1277, "false"), (1278, "true")])
def test_feature_list_opens_by_default_only_where_the_chart_keeps_its_width(
    open_editor_page, width, expected_open
):
    with open_editor_page(viewport={"width": width, "height": 720}) as (page, _session):
        feature_list = page.get_by_role("navigation", name="Features")
        chart = page.locator("#chart").bounding_box()

        assert chart is not None
        assert feature_list.get_attribute("data-open") == expected_open
        assert page.locator("#inspector").get_attribute("data-open") == "true"
        assert chart["width"] >= 600
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")

        # A choice the analyst made outranks the width default on the next load.
        page.locator("#featureListToggle").click()
        page.reload(wait_until="domcontentloaded")
        page.locator("#chart path.edited").first.wait_for()
        assert feature_list.get_attribute("data-open") == (
            "true" if expected_open == "false" else "false"
        )
