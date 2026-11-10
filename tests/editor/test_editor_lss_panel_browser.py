"""The distribution panel of a SuperLSS session, driven from the browser."""

from __future__ import annotations

from contextlib import contextmanager
from urllib.parse import urlsplit

import pytest
from tests.editor._chart_layout import wait_for_chart_to_fit
from tests.editor._editor_waits import hold_requests, wait_for_editor_idle, wait_for_interception

pytest.importorskip("playwright.sync_api")
pytestmark = pytest.mark.browser

_READOUTS = "#distributionPanel text.dist-readout"
_READOUT_TEXT = f"() => [...document.querySelectorAll({_READOUTS!r})].map(e => e.textContent)"


# Two animation frames: whatever the page scheduled when the condition was checked has run.
_TWO_FRAMES = "() => new Promise(done => requestAnimationFrame(() => requestAnimationFrame(done)))"
_READOUTS_DIFFER = f"before => ({_READOUT_TEXT})().join('|') !== before.join('|')"
_READOUTS_EQUAL = f"before => ({_READOUT_TEXT})().join('|') === before.join('|')"


def _is_distribution(request) -> bool:
    return request.method == "POST" and urlsplit(request.url).path == "/distribution"


class _DistributionRequests:
    """Every /distribution request the page sends, and those still waiting for an answer."""

    def __init__(self, page) -> None:
        self.sent = []
        self.open = set()
        page.on("request", self._sent)
        page.on("requestfinished", self._closed)
        page.on("requestfailed", self._closed)

    def _sent(self, request) -> None:
        if _is_distribution(request):
            self.sent.append(request)
            self.open.add(request)

    def _closed(self, request) -> None:
        self.open.discard(request)

    def settle(self) -> None:
        """Wait until every request sent so far has been answered or has failed."""
        for request in list(self.open):
            # None when the request failed instead of being answered.
            response = request.response()
            if response is not None:
                response.finished()
            self.open.discard(request)


@contextmanager
def _spread_page(open_lss_editor_page, in_view: bool = True):
    """The page on the spread of age in the Handles tool, its panel drawn and every request answered."""
    tracked = {}

    def prepare(page) -> None:
        tracked["requests"] = _DistributionRequests(page)

    with open_lss_editor_page(selected_term="scale:age", prepare=prepare) as (page, _session):
        requests = tracked["requests"]
        page.locator(_READOUTS).first.wait_for()
        page.get_by_role("radiogroup", name="Chart tools").get_by_role(
            "radio", name="Handles", exact=True
        ).click()
        wait_for_chart_to_fit(page)
        page.locator("#chart .control-handle").first.wait_for()
        wait_for_interception(page)
        _settle(page, requests)
        if in_view:
            _bring_panel_into_view(page)
            _settle(page, requests)
        yield page, requests


def _bring_panel_into_view(page) -> None:
    """Below the chart the panel waits; this shows it, with the chart's handles still on screen."""
    page.evaluate("document.querySelector('.plot-column').scrollTo(0, 200)")
    page.evaluate(_TWO_FRAMES)


def _settle(page, requests) -> None:
    requests.settle()
    page.evaluate(_TWO_FRAMES)


def _readouts(page) -> list[str]:
    return page.evaluate(_READOUT_TEXT)


def _handle_centre(page, index: int) -> tuple[float, float]:
    box = page.locator("#chart .control-handle").nth(index).bounding_box()
    assert box is not None
    return box["x"] + box["width"] / 2, box["y"] + box["height"] / 2


def _press_and_drag(page, index: int, dy: float) -> None:
    x, y = _handle_centre(page, index)
    page.mouse.move(x, y)
    page.mouse.down()
    page.mouse.move(x, y + dy, steps=6)


def _answer_with_the_fitted_model(route) -> None:
    """The server's reply with its edited predictors replaced by the fitted ones."""
    response = route.fetch()
    body = response.json()
    body["eta"]["edited"] = body["eta"]["fitted"]
    route.fulfill(response=response, json=body)


@pytest.mark.parametrize(
    "viewport",
    [{"width": 1280, "height": 800}, {"width": 1800, "height": 1000}],
    ids=["laptop", "wide"],
)
def test_panel_lies_below_a_chart_that_keeps_the_column(open_lss_editor_page, viewport):
    with open_lss_editor_page(viewport=viewport) as (page, _session):
        page.locator("#distributionPanel svg").first.wait_for()
        wait_for_chart_to_fit(page)
        column = page.locator(".plot-column").bounding_box()
        chart = page.locator("#chart").bounding_box()
        metrics = page.locator(".metrics-strip").bounding_box()
        panel = page.locator("#distributionPanel").bounding_box()

        # The chart spans the column, less its scrollbar, and the panel starts after everything
        # the chart shares the first screen with, its controls showing at the column's foot.
        assert chart["width"] >= column["width"] - 24
        assert panel["y"] >= metrics["y"] + metrics["height"]
        assert 0 < column["y"] + column["height"] - panel["y"] <= 40

        page.locator(".plot-column").evaluate("column => column.scrollTo(0, column.scrollHeight)")
        panel = page.locator("#distributionPanel").bounding_box()
        assert panel["y"] + panel["height"] <= column["y"] + column["height"] + 1


def test_panel_moves_during_a_drag_without_requests(open_lss_editor_page):
    with _spread_page(open_lss_editor_page) as (page, requests):
        before = _readouts(page)
        loaded = len(requests.sent)

        _press_and_drag(page, 2, -60)
        page.wait_for_function(_READOUTS_DIFFER, arg=before)
        assert len(requests.sent) == loaded

        # The reply draws the fitted model, which the preview cannot: the panel reads as before
        # the drag only once the reply has replaced the preview.
        page.route("**/distribution", _answer_with_the_fitted_model)
        assert len(requests.sent) == loaded
        with page.expect_response(lambda response: _is_distribution(response.request)) as reply:
            page.mouse.up()
        assert reply.value.status == 200
        wait_for_editor_idle(page)
        page.wait_for_function(_READOUTS_EQUAL, arg=before)
        _settle(page, requests)
        assert len(requests.sent) == loaded + 1


def test_a_cancelled_drag_puts_the_panel_back(open_lss_editor_page):
    with _spread_page(open_lss_editor_page) as (page, requests):
        before = _readouts(page)
        loaded = len(requests.sent)

        _press_and_drag(page, 2, -60)
        page.wait_for_function(_READOUTS_DIFFER, arg=before)
        page.dispatch_event("#chart", "pointercancel")
        page.wait_for_function(_READOUTS_EQUAL, arg=before)
        page.mouse.up()
        wait_for_editor_idle(page)
        _settle(page, requests)
        assert len(requests.sent) == loaded


def test_a_drag_during_the_refetch_keeps_the_last_drawing(open_lss_editor_page):
    with _spread_page(open_lss_editor_page) as (page, _requests):
        before = _readouts(page)
        held = hold_requests(page, "**/distribution")
        _press_and_drag(page, 2, -60)
        page.mouse.up()
        held.wait(1)
        wait_for_editor_idle(page)
        page.evaluate(_TWO_FRAMES)
        first = _readouts(page)
        assert first != before

        # The payload still answers the model before the first drag; a preview against it would
        # draw the second drag without the first.
        _press_and_drag(page, 4, 60)
        page.evaluate(_TWO_FRAMES)
        assert _readouts(page) == first
        page.dispatch_event("#chart", "pointercancel")
        page.mouse.up()
        for route in held:
            route.continue_()


def test_an_unreadable_entry_does_not_follow_the_analyst_to_another_feature(
    open_lss_editor_page, choose_feature
):
    with open_lss_editor_page(selected_term="scale:age") as (page, _session):
        page.locator(_READOUTS).first.wait_for()
        _bring_panel_into_view(page)
        box = page.get_by_role("textbox", name="Line value", exact=False)
        box.fill("abc")
        box.press("Enter")
        assert box.get_attribute("aria-invalid") == "true"
        before = page.locator("#distributionPanel svg").first.get_attribute("aria-label")
        choose_feature(page, "scale:region")
        _bring_panel_into_view(page)
        page.wait_for_function(
            "before => document.querySelector('#distributionPanel svg')?.getAttribute("
            "'aria-label') !== before",
            arg=before,
        )
        assert box.get_attribute("aria-invalid") is None


def _panel_is_about(page, words: str) -> None:
    page.wait_for_function(
        "words => document.querySelector('#distributionPanel svg')?.getAttribute("
        "'aria-label')?.includes(words)",
        arg=words,
    )


def _features_asked_for(requests) -> list[str]:
    return [request.post_data_json["feature"] for request in requests.sent]


def test_going_back_to_a_feature_asks_again_only_after_an_edit(
    open_lss_editor_page, choose_feature
):
    with _spread_page(open_lss_editor_page) as (page, requests):
        asked = len(requests.sent)
        choose_feature(page, "scale:region")
        _bring_panel_into_view(page)
        _panel_is_about(page, "region")

        choose_feature(page, "scale:age")
        _bring_panel_into_view(page)
        _panel_is_about(page, "age")
        assert _features_asked_for(requests)[asked:] == ["scale:region"]

        # A term with no handles took the chart to Select.
        page.get_by_role("radiogroup", name="Chart tools").get_by_role(
            "radio", name="Handles", exact=True
        ).click()
        page.locator("#chart .control-handle").first.wait_for()
        _press_and_drag(page, 2, -60)
        page.mouse.up()
        wait_for_editor_idle(page)
        _bring_panel_into_view(page)
        _settle(page, requests)
        assert _features_asked_for(requests)[asked:] == ["scale:region", "scale:age"]


def test_a_panel_below_the_fold_asks_for_nothing_until_it_is_scrolled_to(
    open_lss_editor_page, choose_feature
):
    with _spread_page(open_lss_editor_page, in_view=False) as (page, requests):
        asked = len(requests.sent)
        choose_feature(page, "scale:region")
        page.evaluate(_TWO_FRAMES)
        page.wait_for_timeout(100)
        assert _features_asked_for(requests)[asked:] == []

        _bring_panel_into_view(page)
        _panel_is_about(page, "region")
        assert _features_asked_for(requests)[asked:] == ["scale:region"]


def test_a_panel_below_the_fold_waits_until_it_is_scrolled_to(open_lss_editor_page):
    with _spread_page(open_lss_editor_page, in_view=False) as (page, requests):
        before = _readouts(page)
        _press_and_drag(page, 2, -60)
        page.evaluate(_TWO_FRAMES)
        assert _readouts(page) == before
        page.mouse.up()
        wait_for_editor_idle(page)
        _settle(page, requests)
        assert _readouts(page) == before

        _bring_panel_into_view(page)
        page.wait_for_function(_READOUTS_DIFFER, arg=before)


def test_a_line_that_moves_clips_the_ridges_and_the_fan_and_leaves_their_shapes(
    open_lss_editor_page,
):
    orange = """() => {
        const canvas = document.querySelector('.distribution-fan-shade');
        const { data } = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height);
        let count = 0;
        for (let i = 0; i < data.length; i += 4) count += data[i] > data[i + 2] ? 1 : 0;
        return count;
    }"""
    drawn = """() => ({
        node: document.querySelector('.ridge-fill') === window.__ridge,
        shape: document.querySelector('.ridge-fill').getAttribute('d'),
        cut: document.querySelector('#ridgeBelow0 rect').getAttribute('width'),
    })"""
    with _spread_page(open_lss_editor_page) as (page, requests):
        box = page.get_by_role("textbox", name="Line value", exact=False)
        box.fill("100k")
        box.press("Enter")
        page.evaluate(_TWO_FRAMES)
        page.evaluate("window.__ridge = document.querySelector('.ridge-fill')")
        high = page.evaluate(drawn)
        past_high = page.evaluate(orange)

        box.fill("1k")
        box.press("Enter")
        page.evaluate(_TWO_FRAMES)
        low = page.evaluate(drawn)

        assert low["node"] is True
        assert low["shape"] == high["shape"]
        assert float(low["cut"]) < float(high["cut"])
        assert page.evaluate(orange) > past_high
        _settle(page, requests)


def test_the_percentile_axis_labels_each_amount_once(open_lss_editor_page):
    with _spread_page(open_lss_editor_page) as (page, requests):
        page.get_by_role("button", name="At a percentile", exact=True).click()
        page.locator("#distributionPanel .ridge-cut").first.wait_for(state="attached")
        _settle(page, requests)
        labels = page.evaluate(
            """() => {
                const svg = document.querySelector('#distributionPanel svg.distribution-ridges');
                const row = svg.querySelector('.ridge-axis text').getAttribute('y');
                return [...svg.querySelectorAll('text')]
                    .filter(node => node.getAttribute('y') === row)
                    .map(node => node.textContent);
            }"""
        )
        assert len(labels) > 1
        assert len(labels) == len(set(labels))


def test_the_labels_on_the_amount_axis_row_do_not_touch(open_lss_editor_page):
    def end_on_a_long_decade_label(route) -> None:
        response = route.fetch()
        panel = response.json()
        panel["amount_axis"]["max_log10"] = 5
        panel["amount_unit"] = "EUR"
        route.fulfill(response=response, json=panel)

    def prepare(page) -> None:
        page.route("**/distribution", end_on_a_long_decade_label)

    with open_lss_editor_page(selected_term="scale:age", prepare=prepare) as (page, _session):
        page.locator(_READOUTS).first.wait_for()
        _bring_panel_into_view(page)
        boxes = page.evaluate(
            """() => {
                const svg = document.querySelector('#distributionPanel svg.distribution-ridges');
                const row = svg.querySelector('.ridge-axis text').getAttribute('y');
                return [...svg.querySelectorAll('text')]
                    .filter(node => node.getAttribute('y') === row)
                    .map(node => { const box = node.getBBox(); return [box.x, box.x + box.width]; })
                    .sort((a, b) => a[0] - b[0]);
            }"""
        )
        assert len(boxes) > 3
        # Labels closer than a word space read as one: "EUR100k0.01%".
        for (_, right), (left, _) in zip(boxes, boxes[1:]):
            assert left - right >= 4


_ROW_BOXES = """row => {
    const svg = document.querySelector(row.svg);
    const y = row.y ?? svg.querySelector(row.anchor).getAttribute('y');
    return [...svg.querySelectorAll('text')]
        .filter(node => node.getAttribute('y') === y)
        .map(node => { const box = node.getBBox(); return [box.x, box.x + box.width, node.textContent]; })
        .sort((a, b) => a[0] - b[0]);
}"""


@pytest.mark.parametrize("width", [700, 1100])
def test_the_headings_over_the_ridges_and_the_fan_keep_clear_of_each_other(
    open_lss_editor_page, width
):
    def with_a_long_feature_name(route) -> None:
        response = route.fetch()
        panel = response.json()
        panel["feature"] = "Age of the driver at the start of the policy"
        route.fulfill(response=response, json=panel)

    def prepare(page) -> None:
        page.route("**/distribution", with_a_long_feature_name)

    with open_lss_editor_page(
        selected_term="scale:age", viewport={"width": width, "height": 720}, prepare=prepare
    ) as (page, _session):
        page.locator(_READOUTS).first.wait_for()
        _bring_panel_into_view(page)
        for svg in ("svg.distribution-ridges", "svg.distribution-fan"):
            boxes = page.evaluate(_ROW_BOXES, {"svg": f"#distributionPanel {svg}", "y": "22"})
            assert boxes, svg
            # Two headings closer than a word space read as one; a heading with no room is left out.
            for (_, right, _), (left, _, text) in zip(boxes, boxes[1:]):
                assert left - right >= 8, (svg, text)


def test_the_amount_axis_keeps_clear_of_the_dot_column_and_both_write_chances_alike(
    open_lss_editor_page,
):
    def end_on_a_long_decade_label(route) -> None:
        response = route.fetch()
        panel = response.json()
        panel["amount_axis"]["max_log10"] = 4
        panel["amount_unit"] = "EUR"
        route.fulfill(response=response, json=panel)

    def prepare(page) -> None:
        page.route("**/distribution", end_on_a_long_decade_label)

    with open_lss_editor_page(selected_term="scale:age", prepare=prepare) as (page, _session):
        page.locator(_READOUTS).first.wait_for()
        _bring_panel_into_view(page)
        page.get_by_role("button", name="1 in N", exact=True).click()
        page.evaluate(_TWO_FRAMES)
        boxes = page.evaluate(
            _ROW_BOXES,
            {"svg": "#distributionPanel svg.distribution-ridges", "anchor": ".ridge-axis text"},
        )
        labels = [text for _, _, text in boxes]
        # The dot column's axis reads as the readouts do: 1 in 10k, not 1:10k.
        assert labels[-2:] == ["1 in 10k", "1 in 1"]
        assert ":" not in "".join(labels)
        for (_, right, _), (left, _, text) in zip(boxes, boxes[1:]):
            assert left - right >= 24, text
        page.get_by_role("button", name="%", exact=True).click()
        page.evaluate(_TWO_FRAMES)
        boxes = page.evaluate(
            _ROW_BOXES,
            {"svg": "#distributionPanel svg.distribution-ridges", "anchor": ".ridge-axis text"},
        )
        assert [text for _, _, text in boxes][-3:] == ["0.01%", "1%", "100%"]


def test_the_fan_draws_the_mean_and_the_median_and_no_most_common_line(open_lss_editor_page):
    with open_lss_editor_page(selected_term="scale:age") as (page, _session):
        page.locator(_READOUTS).first.wait_for()
        _bring_panel_into_view(page)
        fan = page.locator("#distributionPanel svg.distribution-fan")
        fan.locator("text.dist-readout").first.wait_for()
        labels = fan.locator("text.dist-readout").evaluate_all(
            "nodes => nodes.map(n => n.textContent)"
        )
        assert {"mean", "median"} <= set(labels)
        assert not any("common" in label for label in labels)
        assert fan.locator("path.fan-mode").count() == 0


_WHEEL_OVER_CHART = """([ctrlKey, metaKey]) => {
    const chart = document.querySelector('#chart');
    const box = chart.getBoundingClientRect();
    return !chart.dispatchEvent(new WheelEvent('wheel', {
        deltaY: -100, ctrlKey, metaKey, bubbles: true, cancelable: true,
        clientX: box.left + box.width / 2, clientY: box.top + box.height / 2,
    }));
}"""


def test_a_scroll_over_the_chart_moves_the_column_and_leaves_the_zoom(open_lss_editor_page):
    with open_lss_editor_page(selected_term="scale:age") as (page, _session):
        page.locator(_READOUTS).first.wait_for()
        box = page.locator("#chart").bounding_box()
        page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        drawn = page.evaluate(
            "() => document.querySelector('#chart path.edited').getAttribute('d')"
        )

        page.mouse.wheel(0, 200)

        page.wait_for_function("() => document.querySelector('.plot-column').scrollTop > 0")
        page.evaluate(_TWO_FRAMES)
        assert drawn == page.evaluate(
            "() => document.querySelector('#chart path.edited').getAttribute('d')"
        )


@pytest.mark.parametrize("keys", [[True, False], [False, True]], ids=["ctrl-or-pinch", "cmd"])
def test_ctrl_or_cmd_with_the_wheel_zooms_where_the_column_scrolls(open_lss_editor_page, keys):
    with open_lss_editor_page(selected_term="scale:age") as (page, _session):
        page.locator(_READOUTS).first.wait_for()
        assert page.evaluate(_WHEEL_OVER_CHART, [False, False]) is False
        assert page.evaluate(_WHEEL_OVER_CHART, keys) is True


def test_a_plain_wheel_zooms_in_the_zoom_tool_where_the_column_scrolls(open_lss_editor_page):
    with open_lss_editor_page(selected_term="scale:age") as (page, _session):
        page.locator(_READOUTS).first.wait_for()
        page.get_by_role("radiogroup", name="Chart tools").get_by_role(
            "radio", name="Zoom", exact=True
        ).click()
        assert page.evaluate(_WHEEL_OVER_CHART, [False, False]) is True


def test_a_percentile_shown_as_one_in_n_takes_only_n(open_lss_editor_page):
    with _spread_page(open_lss_editor_page) as (page, requests):
        box = page.get_by_role("textbox", name="Line value", exact=False)
        prefix = page.locator("#distributionPanel .distribution-line-prefix")
        assert prefix.is_hidden()

        page.get_by_role("button", name="At a percentile", exact=True).click()
        page.get_by_role("button", name="1 in N", exact=True).click()
        page.evaluate(_TWO_FRAMES)
        assert prefix.is_visible()
        assert prefix.inner_text() == "1 in"
        assert "1 in" not in box.input_value()

        box.fill("500")
        box.press("Enter")
        page.evaluate(_TWO_FRAMES)
        assert box.input_value() == "500"
        assert box.get_attribute("aria-invalid") is None

        box.fill("99.5th")
        box.press("Enter")
        assert box.get_attribute("aria-invalid") == "true"
        assert "Type N" in page.locator("#distributionPanel .distribution-hint").inner_text()

        box.fill("200")
        box.press("Enter")
        page.get_by_role("button", name="%", exact=True).click()
        page.evaluate(_TWO_FRAMES)
        assert prefix.is_hidden()
        assert box.input_value() == "99.5th"
        _settle(page, requests)


def test_a_feature_drawn_before_another_failed_is_drawn_again(open_lss_editor_page, choose_feature):
    """A failed request's message gives way to a feature whose reply the page still holds.

    The window is wide enough that the feature list stays open, so choosing a feature leaves the
    panel's width as it was and nothing but the reply itself can ask for the redraw.
    """
    viewport = {"width": 1400, "height": 800}
    with open_lss_editor_page(viewport=viewport, selected_term="scale:age") as (page, _session):
        page.locator(_READOUTS).first.wait_for()
        _bring_panel_into_view(page)
        _panel_is_about(page, "age")
        message = page.locator("#distributionPanel .distribution-message")
        wait_for_interception(page)
        page.route("**/distribution", lambda route: route.abort())
        choose_feature(page, "scale:region")
        _bring_panel_into_view(page)
        message.wait_for(state="visible")
        page.unroute("**/distribution")

        choose_feature(page, "scale:age")
        _bring_panel_into_view(page)
        message.wait_for(state="hidden", timeout=5000)
        assert page.locator(_READOUTS).first.is_visible()
