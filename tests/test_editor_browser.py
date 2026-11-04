from __future__ import annotations

import threading

import numpy as np
import pandas as pd
import pytest
from playwright.sync_api import sync_playwright

from superglm import Categorical, Spline, SuperGLM
from superglm.editor import EditorSession


def select_chart_tool(page, name: str) -> None:
    page.get_by_role("radiogroup", name="Chart tools").get_by_role(
        "radio", name=name, exact=True
    ).click()


def choose_feature(page, term: str) -> None:
    """Pick a term from the feature list, opening its collapsed strip first."""
    collapsed = page.locator("#featureList").get_attribute("data-open") == "false"
    if collapsed:
        page.locator("#featureListToggle").click()
    page.locator(f'#featureList [data-term="{term}"]').click()
    if collapsed:
        page.locator("#featureListToggle").click()


def current_feature(page) -> str | None:
    return page.locator('#featureList [aria-current="true"]').get_attribute("data-term")


@pytest.fixture
def browser_editor_widget():
    rng = np.random.default_rng(20260711)
    n = 120
    X = pd.DataFrame(
        {
            "age": rng.uniform(18.0, 80.0, n),
            "region": rng.choice(["A", "B", "C"], n),
        }
    )
    y = 0.3 + 0.01 * X["age"].to_numpy() + 0.2 * (X["region"] == "B")
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"age": Spline(kind="ps", n_knots=7), "region": Categorical(base="first")},
    )
    model.fit(X, y)
    widget = EditorSession.from_model(model, terms=["age", "region"]).widget()
    try:
        yield widget
    finally:
        widget.close()


@pytest.mark.browser
def test_editor_browser_loads_authoritative_state(browser_editor_widget):
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1180, "height": 720})
        page.goto(browser_editor_widget.app_url)
        page.locator("#chart .edited").first.wait_for()
        assert current_feature(page) == "age"
        assert page.locator("#status").get_attribute("style") in (None, "")
        layout = page.evaluate(
            """() => {
                const shell = document.querySelector('.app-shell').getBoundingClientRect();
                const view = document.querySelector('#editorView').getBoundingClientRect();
                const style = getComputedStyle(document.querySelector('#editorView'));
                return {
                    shellBottom: shell.bottom,
                    viewBottom: view.bottom,
                    viewHeight: view.height,
                    viewGridRow: style.gridRowStart,
                };
            }"""
        )
        assert layout["viewGridRow"] == "view"
        assert abs(layout["viewBottom"] - layout["shellBottom"]) < 1
        assert layout["viewHeight"] > 400
        browser.close()


@pytest.mark.browser
def test_editor_browser_suppresses_duplicate_action_while_pending(
    browser_editor_widget, monkeypatch
):
    original_operate = browser_editor_widget._operate
    first_shift_started = threading.Event()
    second_shift_started = threading.Event()
    release_shifts = threading.Event()
    shift_requests = 0
    request_lock = threading.Lock()

    def delayed_operate(operation, term=None):
        nonlocal shift_requests
        if operation == "shift_up":
            with request_lock:
                shift_requests += 1
                request_number = shift_requests
            first_shift_started.set()
            if request_number == 2:
                second_shift_started.set()
            if not release_shifts.wait(timeout=5):
                raise RuntimeError("Timed out waiting to release shift_up")
        return original_operate(operation, term)

    monkeypatch.setattr(browser_editor_widget, "_operate", delayed_operate)

    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        try:
            page = browser.new_page(viewport={"width": 1180, "height": 720})
            page.goto(browser_editor_widget.app_url)
            edited_path = page.locator("#chart path.edited").first
            edited_path.wait_for()
            select_chart_tool(page, "Select")
            page.locator('button[data-op="select_all"]').click()
            page.locator("#selectionMenu").wait_for(state="visible")
            original_path = edited_path.get_attribute("d")

            page.locator('button[data-op="shift_up"]').dblclick()
            assert first_shift_started.wait(timeout=2)
            second_shift_started.wait(timeout=0.5)
            release_shifts.set()

            page.wait_for_function(
                """originalPath =>
                    document.querySelector('#chart path.edited')?.getAttribute('d') !== originalPath
                """,
                arg=original_path,
            )
            assert shift_requests == 1
            assert browser_editor_widget.session.model_revision == 1
        finally:
            release_shifts.set()
            browser.close()


@pytest.mark.browser
def test_editor_browser_failed_drag_restores_confirmed_curve_and_allows_next_drag(
    browser_editor_widget, monkeypatch
):
    original_drag = browser_editor_widget._drag
    original_state = browser_editor_widget._state
    drag_requests = []
    fail_recovery_state = False

    def fail_first_drag(term, indices, delta=0.0, values=None):
        nonlocal fail_recovery_state
        drag_requests.append(
            {
                "term": term,
                "indices": list(indices),
                "delta": delta,
                "values": None if values is None else list(values),
            }
        )
        if len(drag_requests) == 1:
            fail_recovery_state = True
            raise RuntimeError("drag failed once")
        return original_drag(term, indices, delta, values)

    def fail_first_recovery_state():
        nonlocal fail_recovery_state
        if fail_recovery_state:
            fail_recovery_state = False
            raise RuntimeError("state recovery failed once")
        return original_state()

    monkeypatch.setattr(browser_editor_widget, "_drag", fail_first_drag)
    monkeypatch.setattr(browser_editor_widget, "_state", fail_first_recovery_state)

    def begin_selected_point_drag(page, delta_y):
        point = page.locator("#chart circle.point.selected[data-index]").first
        point.wait_for()
        box = point.bounding_box()
        assert box is not None
        x = box["x"] + box["width"] / 2
        y = box["y"] + box["height"] / 2
        page.mouse.move(x, y)
        page.mouse.down()
        page.mouse.move(x, y + delta_y)

    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        try:
            page = browser.new_page(viewport={"width": 1180, "height": 720})
            page.goto(browser_editor_widget.app_url)
            page.locator("#chart path.edited").first.wait_for()
            # The chart is drawn at its panel's size, and the error banner this
            # test raises takes a row above the editor. Pin the panel so the
            # path data compared below stays in one coordinate system.
            page.locator(".chart-shell").evaluate(
                "node => { node.style.height = `${node.getBoundingClientRect().height}px`; }"
            )
            page.wait_for_function(
                """() => {
                    const svg = document.querySelector('#chart');
                    const viewBox = svg.viewBox.baseVal;
                    return viewBox.width === svg.clientWidth
                        && viewBox.height === svg.clientHeight;
                }"""
            )

            select_chart_tool(page, "Select")
            page.locator("#chart circle.point[data-index]").last.click()
            page.locator("#chart circle.point.selected[data-index]").first.wait_for()
            select_chart_tool(page, "Move")

            edited_path = page.locator("#chart path.edited").first
            confirmed_path = edited_path.get_attribute("d")
            confirmed_revision = browser_editor_widget.session.model_revision
            assert confirmed_path is not None
            normal_layout = page.evaluate(
                """() => {
                    const shell = document.querySelector('.app-shell').getBoundingClientRect();
                    const view = document.querySelector('#editorView').getBoundingClientRect();
                    return {
                        shellBottom: shell.bottom,
                        viewBottom: view.bottom,
                        viewHeight: view.height,
                    };
                }"""
            )
            assert abs(normal_layout["viewBottom"] - normal_layout["shellBottom"]) < 1

            page.evaluate(
                """() => {
                    const stats = { editedPathWrites: 0, termMutations: 0 };
                    window.__previewRenderStats = stats;
                    new MutationObserver((records) => {
                        for (const record of records) {
                            if (record.target.matches('path.edited')) stats.editedPathWrites += 1;
                        }
                    }).observe(document.querySelector('#chart'), {
                        attributes: true, attributeFilter: ['d'], subtree: true
                    });
                    new MutationObserver((records) => {
                        for (const record of records) {
                            stats.termMutations +=
                                record.addedNodes.length + record.removedNodes.length;
                        }
                    }).observe(document.querySelector('#featureRows'), { childList: true, subtree: true });
                }"""
            )
            begin_selected_point_drag(page, -36)
            page.wait_for_function("() => window.__previewRenderStats.editedPathWrites > 0")
            preview_path = edited_path.get_attribute("d")
            assert preview_path is not None
            assert preview_path != confirmed_path
            preview_stats = page.evaluate("() => window.__previewRenderStats")
            assert preview_stats == {"editedPathWrites": 1, "termMutations": 0}
            with page.expect_response(
                lambda response: (
                    response.request.method == "POST"
                    and response.url.split("?", maxsplit=1)[0].endswith("/drag")
                )
            ) as failed_drag:
                page.mouse.up()
            assert failed_drag.value.status == 500
            page.wait_for_function(
                """() => {
                    const alert = document.querySelector('#appAlert');
                    const status = document.querySelector('#status');
                    return (alert && !alert.hidden) ||
                        (status && status.textContent.includes('internal editor error'));
                }"""
            )

            assert edited_path.get_attribute("d") == confirmed_path
            assert browser_editor_widget.session.model_revision == confirmed_revision
            assert len(drag_requests) == 1
            assert drag_requests[0]["indices"]
            assert drag_requests[0]["values"]

            alert = page.locator("#appAlert")
            assert alert.is_visible()
            assert "internal editor error" in page.locator("#appAlertMessage").inner_text()
            alert_layout = page.evaluate(
                """() => {
                    const shell = document.querySelector('.app-shell').getBoundingClientRect();
                    const view = document.querySelector('#editorView').getBoundingClientRect();
                    return {
                        shellBottom: shell.bottom,
                        viewBottom: view.bottom,
                        viewHeight: view.height,
                    };
                }"""
            )
            assert abs(alert_layout["viewBottom"] - alert_layout["shellBottom"]) < 1
            assert alert_layout["viewHeight"] < normal_layout["viewHeight"]
            page.locator("#appAlertDismiss").click()
            alert.wait_for(state="hidden")
            dismissed_layout = page.evaluate(
                """() => {
                    const shell = document.querySelector('.app-shell').getBoundingClientRect();
                    const view = document.querySelector('#editorView').getBoundingClientRect();
                    return {
                        shellBottom: shell.bottom,
                        viewBottom: view.bottom,
                        viewHeight: view.height,
                    };
                }"""
            )
            assert abs(dismissed_layout["viewBottom"] - dismissed_layout["shellBottom"]) < 1
            assert abs(dismissed_layout["viewHeight"] - normal_layout["viewHeight"]) < 1

            begin_selected_point_drag(page, 28)
            with page.expect_response(
                lambda response: (
                    response.request.method == "POST"
                    and response.url.split("?", maxsplit=1)[0].endswith("/drag")
                )
            ) as successful_drag:
                page.mouse.up()
            assert successful_drag.value.status == 200
            page.wait_for_function(
                """confirmedPath =>
                    document.querySelector('#chart path.edited')?.getAttribute('d') !== confirmedPath
                """,
                arg=confirmed_path,
            )

            assert len(drag_requests) == 2
            assert browser_editor_widget.session.model_revision == confirmed_revision + 1
            assert edited_path.get_attribute("d") != confirmed_path
        finally:
            browser.close()


@pytest.mark.browser
@pytest.mark.parametrize("cancel_event", ["pointercancel", "lostpointercapture"])
def test_editor_browser_cancelled_drag_restores_confirmed_preview_without_post(
    browser_editor_widget, monkeypatch, cancel_event
):
    original_drag = browser_editor_widget._drag
    drag_requests = []

    def record_drag(term, indices, delta=0.0, values=None):
        drag_requests.append((term, list(indices)))
        return original_drag(term, indices, delta, values)

    monkeypatch.setattr(browser_editor_widget, "_drag", record_drag)

    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        try:
            page = browser.new_page(viewport={"width": 1180, "height": 720})
            page.goto(browser_editor_widget.app_url)
            page.locator("#chart path.edited").first.wait_for()
            select_chart_tool(page, "Select")
            page.locator("#chart circle.point[data-index]").last.click()
            select_chart_tool(page, "Move")

            edited_path = page.locator("#chart path.edited").first
            confirmed_path = edited_path.get_attribute("d")
            confirmed_revision = browser_editor_widget.session.model_revision
            point = page.locator("#chart circle.point.selected[data-index]").first
            box = point.bounding_box()
            assert box is not None
            x = box["x"] + box["width"] / 2
            y = box["y"] + box["height"] / 2
            page.mouse.move(x, y)
            page.mouse.down()
            page.mouse.move(x, y - 30)
            assert edited_path.get_attribute("d") != confirmed_path

            page.locator("#chart").dispatch_event(
                cancel_event,
                {"pointerId": 1, "pointerType": "mouse", "isPrimary": True, "button": 0},
            )
            page.wait_for_function(
                "confirmed => document.querySelector('#chart path.edited')?.getAttribute('d') === confirmed",
                arg=confirmed_path,
            )
            page.mouse.up()
            page.wait_for_timeout(100)

            assert drag_requests == []
            assert browser_editor_widget.session.model_revision == confirmed_revision
        finally:
            browser.close()


@pytest.mark.browser
def test_editor_browser_cancel_before_first_move_restores_confirmed_selection(
    browser_editor_widget,
):
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        try:
            page = browser.new_page(viewport={"width": 1180, "height": 720})
            page.goto(browser_editor_widget.app_url)
            page.locator("#chart path.edited").first.wait_for()
            select_chart_tool(page, "Move")
            point = page.locator("#chart circle.point[data-index]").last
            point.wait_for()
            assert page.locator("#chart circle.point.selected[data-index]").count() == 0

            point.hover()
            page.mouse.down()
            assert page.locator("#chart circle.point.selected[data-index]").count() == 1

            page.locator("#chart").dispatch_event(
                "pointercancel",
                {"pointerId": 1, "pointerType": "mouse", "isPrimary": True, "button": 0},
            )
            assert page.locator("#chart circle.point.selected[data-index]").count() == 0
            page.mouse.up()
        finally:
            browser.close()


@pytest.mark.browser
def test_editor_browser_zoom_preserves_handle_visuals(browser_editor_widget):
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        try:
            page = browser.new_page(viewport={"width": 1180, "height": 720})
            page.goto(browser_editor_widget.app_url)
            select_chart_tool(page, "Handles")
            handles = page.locator("#chart .control-handle")
            handles.first.wait_for()
            initial_handle_count = handles.count()
            initial_contribution_count = page.locator("#chart .basis-contribution").count()
            handles_tool = page.get_by_role("radiogroup", name="Chart tools").get_by_role(
                "radio", name="Handles", exact=True
            )
            assert handles_tool.get_attribute("aria-checked") == "true"

            select_chart_tool(page, "Zoom")

            assert (
                page.get_by_role("radiogroup", name="Chart tools")
                .get_by_role("radio", name="Zoom", exact=True)
                .get_attribute("aria-checked")
                == "true"
            )
            assert handles.count() == initial_handle_count
            if initial_contribution_count:
                assert (
                    page.locator("#chart .basis-contribution").count() == initial_contribution_count
                )
        finally:
            browser.close()


@pytest.mark.browser
def test_editor_browser_report_error_clears_mismatched_report(browser_editor_widget, monkeypatch):
    original_report = browser_editor_widget._report

    def fail_final_report(
        report="validation",
        *,
        model_revision=None,
        request_sequence=None,
    ):
        if report == "final":
            raise ValueError("final unavailable")
        return original_report(
            report,
            model_revision=model_revision,
            request_sequence=request_sequence,
        )

    monkeypatch.setattr(browser_editor_widget, "_report", fail_final_report)

    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        try:
            page = browser.new_page(viewport={"width": 1180, "height": 720})
            page.goto(browser_editor_widget.app_url)
            page.locator("#chart .edited").first.wait_for()

            page.locator('.app-tab[data-view="validation"]').click()
            page.get_by_role("table", name="Split metrics").wait_for()

            page.locator('.app-tab[data-view="final"]').click()
            page.wait_for_function(
                "document.querySelector('#reportFreshness')?.textContent.includes("
                "'internal editor error'"
                ")"
            )

            final_html = page.locator("#reportFrame").inner_html()
            assert final_html == ""
            assert page.locator("#reportTitle").text_content() == "Final Fit Report"
            assert page.locator("#reportFreshness").get_attribute("data-freshness") == "stale"
            assert "final unavailable" not in page.locator("#reportFreshness").inner_text()
        finally:
            browser.close()
